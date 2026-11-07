# STAR-TBPS step 2 on IRRA (MLLM4Text-ReID checkpoint), t2i, Pretrain-then-
# Adapt: offline EM over the test captions with the gallery features frozen.
#   E-step  encode all captions, S = T G^T, balanced Sinkhorn plan P (uniform
#           marginals, eps) -> soft pseudo targets P[q] * Q, confidence max P[q] * Q
#   M-step  one shuffled pass of batches of 64 over the captions, cross-entropy
#           of softmax(q G^T / tau) against the targets (optionally only the
#           most confident `keep` fraction), plus lam * energy of the centred
#           batch outside the gallery's principal subspace
# Same parameters and optimiser as the TCR port (kernel_baselines): LayerNorm
# affine params of the text transformer, AdamW lr 3e-4 wd 1e-4. A fixed number
# of epochs is reported (no test-set epoch selection); every epoch is logged.
# Runs inside IRRA_UATTA/. usage: python star_adapt.py CFG OUT_JSON [config names] [seeds]
import copy
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BS, LR, WD, EPOCHS = 64, 3e-4, 1e-4, 10
EPS, ENERGY = 0.05, 0.95
SEEDS = (42, 43, 44)
# v1 (5 epochs, seed 42) picked hard_keep0.6_tau0.05; v2 checks it against the
# same recipe with greedy argmax labels (confidence = softmax(S/eps) max), over
# seeds, and the sensitivity to `keep` with one seed.
CONFIGS = {
    "hard_keep0.6_tau0.05": dict(target="ot_hard", keep=0.6, tau=0.05, lam=0.0, seeds=SEEDS),
    "greedy_hard_keep0.6_tau0.05": dict(target="greedy_hard", keep=0.6, tau=0.05, lam=0.0, seeds=SEEDS),
    "hard_keep0.4_tau0.05": dict(target="ot_hard", keep=0.4, tau=0.05, lam=0.0, seeds=SEEDS[:1]),
    "hard_keep0.8_tau0.05": dict(target="ot_hard", keep=0.8, tau=0.05, lam=0.0, seeds=SEEDS[:1]),
}
dev = "cuda"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def sinkhorn(S, eps, iters=200):
    """log-domain Sinkhorn, uniform marginals; returns P (Q, G) summing to 1."""
    Q, G = S.shape
    K = S / eps
    u, v = torch.zeros(Q, device=S.device), torch.zeros(G, device=S.device)
    for _ in range(iters):
        u = -np.log(Q) - torch.logsumexp(K + v[None], 1)
        v = -np.log(G) - torch.logsumexp(K + u[:, None], 0)
    return torch.exp(K + u[:, None] + v[None])


def gallery_basis(gall, energy):
    gc = gall - gall.mean(0)
    w, V = torch.linalg.eigh(gc.T @ gc / len(gc))
    w, V = w.flip(0), V.flip(1)
    r = int((w.cumsum(0) / w.sum() < energy).sum().item()) + 1
    return V[:, :r]


@torch.no_grad()
def e_step(S, cfg):
    """soft targets (Q, G) and a per-caption weight in {0, 1}."""
    Q = len(S)
    if cfg["target"].startswith("greedy"):
        tgt = (S / EPS).softmax(1)
    else:
        tgt = sinkhorn(S, EPS) * Q
    conf, label = tgt.max(1)
    if cfg["target"].endswith("hard"):
        tgt = F.one_hot(label, S.shape[1]).float()
    w = (conf >= conf.quantile(1 - cfg["keep"])).float() if cfg["keep"] < 1 else torch.ones_like(conf)
    return tgt, w, label


def m_loss(qf, gall, tgt, w, B, cfg):
    ce = -(tgt * (qf @ gall.t() / cfg["tau"]).log_softmax(1)).sum(1)
    loss = (ce * w).sum() / w.sum().clamp(min=1)
    if cfg["lam"] > 0:
        qc = qf - qf.mean(0)
        loss = loss + cfg["lam"] * (1 - (qc @ B).pow(2).sum() / qc.pow(2).sum())
    return loss


def main():
    from torch.utils.data import DataLoader, Dataset
    import yaml
    from easydict import EasyDict as edict
    from datasets.bases import ImageDataset, TextDataset
    from datasets.build import build_transforms, __factory
    from model import build_model
    from utils.checkpoint import Checkpointer

    class Indexed(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            return (i,) + tuple(self.ds[i])

    cfg_path, out = sys.argv[1], sys.argv[2]
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(CONFIGS)
    seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
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
    B = gallery_basis(gall, ENERGY)
    init = copy.deepcopy(model.state_dict())

    def text_feats(m, tokens):
        with torch.cuda.amp.autocast():
            return F.normalize(m.encode_text(tokens.to(dev)), dim=-1).float()

    qd, gd = qids.to(dev), gids.to(dev)

    def evaluate(S, chunk=1024):
        """IRRA's rank() (R1/R5/R10/mAP/mINP), chunked on the GPU so the
        19848 x 19848 ICFG matrix is not argsorted on the CPU in one go."""
        cmc, ap, inp = [], [], []
        for s in range(0, len(S), chunk):
            match = gd[S[s:s + chunk].argsort(1, descending=True)] == qd[s:s + chunk, None]
            hits = match.cumsum(1)
            cmc.append(hits[:, :10] > 0)
            pos = torch.arange(1, match.shape[1] + 1, device=dev)
            ap.append(((hits / pos) * match).sum(1) / match.sum(1).clamp(min=1))
            last = match.shape[1] - 1 - match.flip(1).float().argmax(1)
            inp.append(hits.gather(1, last[:, None])[:, 0] / (last + 1.0))
        cmc = torch.cat(cmc).float().mean(0) * 100
        return dict(R1=cmc[0].item(), R5=cmc[4].item(), R10=cmc[9].item(),
                    mAP=torch.cat(ap).mean().item() * 100, mINP=torch.cat(inp).mean().item() * 100)

    @torch.no_grad()
    def scores(m):
        m.eval()
        S = torch.cat([text_feats(m, cap) @ gall.t() for _, _, cap in
                       DataLoader(txt_set, batch_size=512, num_workers=4)])
        m.train()
        return S

    def report(S):
        P = sinkhorn(S, EPS)
        p = P.argmax(1).cpu()
        return dict(cos=evaluate(S), ot=evaluate(torch.log(P + 1e-30)),
                    ot_pair_precision=(gids[p] == qids).float().mean().item() * 100)

    S0 = scores(model)
    with torch.no_grad():
        tfeat = torch.cat([text_feats(model, cap) for _, _, cap in DataLoader(txt_set, batch_size=512, num_workers=4)])
    np.savez(out.replace(".json", "_zeroshot.npz"), txt_feat=tfeat.cpu().numpy(), img_feat=gall.cpu().numpy(),
             qid=qids.numpy(), gid=gids.numpy())
    results = {"zero-shot": report(S0), "rank_B": B.shape[1], "eps": EPS, "epochs": EPOCHS}
    print("zero-shot", results["zero-shot"], "rank_B", B.shape[1], flush=True)
    for name in names:
        cfg = CONFIGS[name]
        for seed in seeds or cfg["seeds"]:
            seed_all(seed)
            model.load_state_dict(init)
            model.train()
            model.requires_grad_(False)
            for m in model.base_model.transformer.modules():
                if isinstance(m, nn.LayerNorm):
                    m.requires_grad_(True)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
            loader = DataLoader(txt_set, batch_size=BS, shuffle=True, num_workers=4,
                                generator=torch.Generator().manual_seed(seed))
            S, hist, prev, t0 = S0, [], None, time.time()
            for ep in range(EPOCHS):
                tgt, w, label = e_step(S, cfg)
                # label-free stopping signals: how many pseudo labels changed since
                # the previous E-step; precision of the kept labels is diagnostic only
                kept = w.bool().cpu()
                lab = label.cpu()
                r_e = dict(kept_precision=(gids[lab[kept]] == qids[kept]).float().mean().item() * 100)
                if prev is not None:
                    r_e["label_change"] = (lab != prev).float().mean().item() * 100
                prev = lab
                tot = 0.0
                for idx, _, cap in loader:
                    idx = idx.to(dev)
                    loss = m_loss(text_feats(model, cap), gall, tgt[idx], w[idx], B, cfg)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += loss.item()
                S = scores(model)
                r = report(S)
                r.update(epoch=ep + 1, loss=tot / len(loader), **r_e)
                hist.append(r)
                print(name, seed, r, flush=True)
            results[f"{name}/seed{seed}"] = dict(cfg=cfg, final=hist[-1], history=hist, seconds=time.time() - t0)
            with open(out, "w") as f:
                json.dump(results, f, indent=1)
    print("->", out, flush=True)


if __name__ == "__main__":
    main()
