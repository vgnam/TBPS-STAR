# Tent / SAR / READ / TCR on IRRA (MLLM4Text-ReID checkpoint), t2i, following
# the TCR code release (XLearning-SCU/2025-ICLR-TCR, models/tta_baselines and
# models/clip_reid_tta.py): online, shuffled caption batches of 64, 3 steps per
# batch, LayerNorm affine params of the text transformer, AdamW lr 3e-4 wd 1e-4,
# the batch's similarities taken from the last forward. Gallery features stay
# frozen. Runs inside IRRA_UATTA/. usage: python tta_baselines.py CFG OUT_JSON
import copy
import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset

from datasets.bases import ImageDataset, TextDataset
from datasets.build import build_transforms, __factory
from model import build_model
from utils.checkpoint import Checkpointer
from utils.metrics import rank

METHODS = ["tent", "sar", "read", "tcr"]
BS, STEPS, LR, WD = 64, 3, 3e-4, 1e-4
TEMP = {"tcr": 0.02}  # TCR's own default; the others use TCR's ReID value
TEMP_REID = 1e-4
CON_RATIO, UNI_T, N_UPDATE = 0.3, 0.1, 10
dev = "cuda"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Indexed(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return (i,) + tuple(self.ds[i])


def entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class SAM(torch.optim.Optimizer):
    """github.com/davda54/sam, as vendored by TCR."""

    def __init__(self, params, base_optimizer, rho=0.05, **kw):
        super().__init__(params, dict(rho=rho, **kw))
        self.base_optimizer = base_optimizer(self.param_groups, **kw)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        norm = torch.norm(torch.stack([p.grad.norm(2) for g in self.param_groups for p in g["params"]
                                       if p.grad is not None]), 2)
        for g in self.param_groups:
            scale = g["rho"] / (norm + 1e-12)
            for p in g["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        self.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        self.zero_grad()


# TCR losses (losses.py of the release)
def modality_gap(a, b):
    return torch.norm(F.normalize(a).mean(0) - F.normalize(b).mean(0), p=2)


@torch.no_grad()
def update_queue(q, g, queue, max_size, temp):
    n = int(CON_RATIO * q.size(0))
    ind = 2 * torch.norm(q - g, dim=1) - torch.norm(q - q.mean(0), dim=1) - torch.norm(g - g.mean(0), dim=1)
    ent = entropy(q @ g.t() / temp)
    for i in torch.argsort(ind)[:n]:
        queue.append((ind[i].item(), q[i].detach(), g[i].detach(), ent[i].detach()))
    if len(queue) >= max_size:
        queue = sorted(queue, key=lambda x: x[0])[:max_size]
    return queue


def tcr_loss(qf, gall, queue, update, step, temp):
    sim = qf @ gall.t()
    gf = gall[sim.argmax(1)]
    if step == 0 and update:
        queue = update_queue(qf, gf, queue, BS, temp)
    _, f1, f2, ents = zip(*queue)
    margin = modality_gap(torch.stack(f1), torch.stack(f2))
    ent_q = torch.stack(ents)
    e = entropy(qf @ gf.t() / temp)
    thr = ent_q.max()
    w = torch.clamp(1.0 - e.detach() / (thr + 1e-3), min=0)
    rem = e.mul(w)[e <= thr].mean(0)
    uni = torch.exp(-torch.norm(qf - qf.mean(0), dim=1) * UNI_T).mean()
    emg = (modality_gap(qf, gf) - margin) ** 2
    return rem + uni + emg, queue, sim


def main():
    cfg_path, out = sys.argv[1], sys.argv[2]
    args = edict(yaml.load(open(cfg_path), Loader=yaml.FullLoader))
    seed_all(42)
    ds = __factory[args.dataset_name](root=args.root_dir)
    te = ds.test
    tf = build_transforms(img_size=args.img_size, aug=False, is_train=False)
    img_loader = DataLoader(ImageDataset(te["image_pids"], te["img_paths"], tf), batch_size=256, num_workers=4)
    txt_set = Indexed(TextDataset(te["caption_pids"], te["captions"], text_length=args.text_length))

    model = build_model(args, num_classes=len(ds.train_id_container))
    Checkpointer(model).load(f=args.ckpt_dir)
    model.to(dev).eval()
    gfeat, gids = [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for pid, img in img_loader:
            gfeat.append(F.normalize(model.encode_image(img.to(dev)), dim=-1))
            gids.append(pid)
    gall, gids = torch.cat(gfeat).float(), torch.cat(gids)
    qids = torch.as_tensor(te["caption_pids"])
    init = copy.deepcopy(model.state_dict())

    def text_feats(m, tokens):
        with torch.cuda.amp.autocast():
            return F.normalize(m.encode_text(tokens.to(dev)), dim=-1).float()

    def evaluate(S):
        cmc, mAP, mINP, _ = rank(S.cpu(), qids, gids, max_rank=10, get_mAP=True)
        return dict(R1=cmc[0].item(), R5=cmc[4].item(), R10=cmc[9].item(), mAP=mAP.item(), mINP=mINP.item())

    @torch.no_grad()
    def offline(m):
        m.eval()
        S = torch.cat([text_feats(m, cap) @ gall.t() for _, _, cap in
                       DataLoader(txt_set, batch_size=512, num_workers=4)])
        return evaluate(S)

    results = {"zero-shot": offline(model)}
    print("zero-shot", results["zero-shot"], flush=True)
    for method in sys.argv[3].split(",") if len(sys.argv) > 3 else METHODS:
        seed_all(42)
        model.load_state_dict(init)
        model.train()
        model.requires_grad_(False)
        for m in model.base_model.transformer.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        if method == "sar":
            opt = SAM(params, torch.optim.AdamW, lr=LR, weight_decay=WD)
        else:
            opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
        opt_init = copy.deepcopy(opt.state_dict())
        temp = TEMP.get(method, TEMP_REID)
        S = torch.full((len(txt_set), len(gall)), -100.0, device=dev)
        loader = DataLoader(txt_set, batch_size=BS, shuffle=True, num_workers=4,
                            generator=torch.Generator().manual_seed(42))
        queue, ema, resets = [], None, 0
        t0 = time.time()
        for b, (idx, _, cap) in enumerate(loader):
            for step in range(STEPS):
                with torch.enable_grad():
                    qf = text_feats(model, cap)
                    sim = qf @ gall.t()
                    logits = sim / temp
                    if method == "tent":
                        loss = entropy(logits).mean(0)
                    elif method == "read":
                        p_sum = logits.softmax(-1).sum(-2)
                        bal = -(p_sum.softmax(0) * p_sum.log_softmax(0)).sum()
                        pmax = logits.softmax(-1).max(-1)[0]
                        loss = (pmax * (1 - pmax.log() + math.log(math.exp(-1)))).mean() - bal
                    elif method == "tcr":
                        loss, queue, sim = tcr_loss(qf, gall, queue, b < N_UPDATE, step, temp)
                    if method == "sar":
                        e = entropy(logits)
                        thr = torch.topk(e, max(1, int(len(e) * 0.40)), largest=False)[0][-1]
                        keep = e <= thr
                        loss = e[keep].mean(0)
                        loss.backward()
                        opt.first_step()
                        e2 = entropy(text_feats(model, cap) @ gall.t() / temp)[keep]
                        e2 = e2[e2 < thr]
                        loss2 = e2.mean(0) if len(e2) else (e2.sum() * 0)
                        if not torch.isnan(loss2):
                            ema = loss2.item() if ema is None else 0.9 * ema + 0.1 * loss2.item()
                        loss2.backward()
                        opt.second_step()
                        if ema is not None and ema < 0.2:
                            model.load_state_dict(init)
                            opt.load_state_dict(opt_init)
                            ema, resets = None, resets + 1
                    else:
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
            S[idx.to(dev)] = sim.detach()
        r = dict(online=evaluate(S), final_model=offline(model), seconds=time.time() - t0)
        if method == "sar":
            r["resets"] = resets
        results[method] = r
        print(method, r, flush=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("->", out, flush=True)


if __name__ == "__main__":
    main()
