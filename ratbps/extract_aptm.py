"""Cache frozen-APTM features for one split.

For every caption q the ITC stage picks the top-N gallery images; the ITM
stage then fuses each (q, candidate) pair. We keep, per pair, the ITC cosine,
the ITM log-odds and the fused [CLS] state (768-d), which RA-TBPS uses both as
the retrieval input and as the value it grounds on.

usage: python -m ratbps.extract_aptm --fmt cuhk --root data/CUHK-PEDES --split test \
           --ckpt checkpoints/aptm_ft_cuhk/checkpoint_best.pth --aptm third_party/APTM \
           --out emb/aptm_cuhk_test.npz [--gt]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .data import NORM, load_split


class Images(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def build_model(aptm_dir, ckpt):
    sys.path.insert(0, aptm_dir)
    from models import bert
    from models.model_retrieval import APTM_Retrieval

    # The checkpoint holds every text-encoder weight, so skip APTM's
    # from_pretrained call (it would fetch bert-base-uncased only to overwrite it).
    def from_config(_path, config=None, output_loading_info=False, **_):
        m = bert.BertForMaskedLM(config)
        return (m, {"missing_keys": []}) if output_loading_info else m

    bert.BertForMaskedLM.from_pretrained = from_config

    with open(os.path.join(aptm_dir, "configs", "Retrieval_cuhk.yaml")) as f:
        config = yaml.safe_load(f)
    config["vision_config"] = os.path.join(aptm_dir, config["vision_config"])
    config["text_config"] = os.path.join(aptm_dir, config["text_config"])
    model = APTM_Retrieval(config)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("model", state)
        msg = model.load_state_dict(state, strict=False)
        # The MLM head is unused here; anything else missing means a bad checkpoint.
        missing = [k for k in msg.missing_keys if ".cls." not in k and "attn_mask" not in k
                   and "relative_position_index" not in k]
        if missing:
            raise RuntimeError(f"checkpoint is missing {len(missing)} keys, e.g. {missing[:5]}")
        print(f"loaded {ckpt} (unexpected keys: {len(msg.unexpected_keys)})", flush=True)
    else:
        print("WARNING: no checkpoint, random weights (smoke test only)", flush=True)
    return model, config


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", required=True, choices=list(NORM))
    p.add_argument("--root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--aptm", default="third_party/APTM")
    p.add_argument("--tokenizer", default="bert-base-uncased")
    p.add_argument("--norm", default="cuhk", choices=list(NORM),
                   help="image statistics; defaults to the source domain the model was tuned on")
    p.add_argument("--out", required=True)
    p.add_argument("--topn", type=int, default=64, help="ITC candidates fused by ITM per caption")
    p.add_argument("--topr", type=int, default=10, help="ITC captions kept per image (i2t direction)")
    p.add_argument("--gt", action="store_true", help="also fuse each caption with its own image (source only)")
    p.add_argument("--limit", type=int, default=0, help="first n images only (smoke test)")
    p.add_argument("--amp", action="store_true", help="fp16 autocast (faster; check it against fp32 first)")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--pair-bs", type=int, default=256)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda" and args.amp)
    model, config = build_model(os.path.abspath(args.aptm), args.ckpt)
    model.to(dev).eval()
    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(args.tokenizer)

    images, gid, captions, cap_img = load_split(args.root, args.fmt, args.split)
    if args.limit:
        keep = [t for t, i in enumerate(cap_img) if i < args.limit]
        images, gid = images[:args.limit], gid[:args.limit]
        captions, cap_img = [captions[t] for t in keep], [cap_img[t] for t in keep]
    gid, cap_img = np.array(gid), np.array(cap_img)
    qid = gid[cap_img]
    print(f"{args.fmt}/{args.split}: {len(images)} images, {len(captions)} captions", flush=True)
    t0 = time.time()

    tf = transforms.Compose([
        transforms.Resize((config["h"], config["w"]), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(*NORM[args.norm]),
    ])
    img_embeds, img_feat = [], []
    for x in DataLoader(Images(images, tf), batch_size=args.bs, num_workers=4):
        with amp:
            e, _ = model.get_vision_embeds(x.to(dev))
            f = F.normalize(model.vision_proj(e[:, 0, :]).float(), dim=-1)
        img_embeds.append((e.half() if args.amp else e.float()).cpu())
        img_feat.append(f)
    img_embeds = torch.cat(img_embeds)
    img_feat = torch.cat(img_feat)
    print(f"images done {time.time() - t0:.0f}s, tokens {tuple(img_embeds.shape)}", flush=True)

    def encode_text(batch):
        t = tok(batch, padding="max_length", truncation=True, max_length=config["max_tokens"],
                return_tensors="pt").to(dev)
        with amp:
            e = model.get_text_embeds(t.input_ids, t.attention_mask)
            f = F.normalize(model.text_proj(e[:, 0, :]).float(), dim=-1)
        return e, t.attention_mask, f

    def fuse(t_emb, t_att, img_idx):
        # t_emb/t_att: one row per pair; img_idx: gallery index per pair
        out_cross, out_itm = [], []
        for s in range(0, len(img_idx), args.pair_bs):
            ii = img_idx[s:s + args.pair_bs]
            ie = img_embeds[ii.cpu()].to(dev, torch.float32)  # autocast downcasts when --amp
            ia = torch.ones(ie.shape[:-1], dtype=torch.long, device=dev)
            with amp:
                h = model.get_cross_embeds(ie, ia, text_embeds=t_emb[s:s + args.pair_bs],
                                           text_atts=t_att[s:s + args.pair_bs])[:, 0, :]
                logit = model.itm_head(h).float()
            out_cross.append(h.half())
            out_itm.append(logit[:, 1] - logit[:, 0])
        return torch.cat(out_cross), torch.cat(out_itm)

    n = min(args.topn, len(images))
    Q = len(captions)
    cand = np.zeros((Q, n), np.int32)
    cand_itc = np.zeros((Q, n), np.float32)
    cand_itm = np.zeros((Q, n), np.float32)
    cand_cross = np.zeros((Q, n, 768), np.float16)
    gt_cross = np.zeros((Q, 768), np.float16) if args.gt else None
    gt_itm = np.zeros(Q, np.float32) if args.gt else None
    txt_feat = []
    chunk = 32
    for s in range(0, Q, chunk):
        e, a, f = encode_text(captions[s:s + chunk])
        txt_feat.append(f)
        sim = f @ img_feat.T
        top_sim, top_idx = sim.topk(n, dim=1)
        b = len(f)
        rep_e = e.repeat_interleave(n, 0)
        rep_a = a.repeat_interleave(n, 0)
        h, itm = fuse(rep_e, rep_a, top_idx.reshape(-1))
        cand[s:s + b] = top_idx.cpu().numpy()
        cand_itc[s:s + b] = top_sim.cpu().numpy()
        cand_itm[s:s + b] = itm.view(b, n).cpu().numpy()
        cand_cross[s:s + b] = h.view(b, n, -1).cpu().numpy()
        if args.gt:
            own = torch.as_tensor(cap_img[s:s + b], device=dev)
            h, itm = fuse(e, a, own)
            gt_cross[s:s + b] = h.cpu().numpy()
            gt_itm[s:s + b] = itm.cpu().numpy()
        if (s // chunk) % 200 == 0:
            print(f"  captions {s + b}/{Q}  {time.time() - t0:.0f}s", flush=True)
    txt_feat = torch.cat(txt_feat)

    # i2t direction for the bidirectional agreement test: the captions each
    # image ranks highest. Computed from ITC only, no labels.
    r = min(args.topr, Q)
    img_top_txt = np.zeros((len(images), r), np.int32)
    for s in range(0, len(images), 1024):
        img_top_txt[s:s + 1024] = (img_feat[s:s + 1024] @ txt_feat.T).topk(r, dim=1)[1].cpu().numpy()

    out = dict(img_feat=img_feat.half().cpu().numpy(), txt_feat=txt_feat.half().cpu().numpy(),
               gid=gid, qid=qid, q2img=cap_img, cand=cand, cand_itc=cand_itc, cand_itm=cand_itm,
               cand_cross=cand_cross, img_top_txt=img_top_txt)
    if args.gt:
        out.update(gt_cross=gt_cross, gt_itm=gt_itm)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
