"""IRRA embeddings for CUHK-PEDES val/test, using IRRA's own model, tokenizer
and test transform (384x128).

usage: python -m opentbps.extract_irra --ckpt checkpoints/irra_cuhk --out emb/irra_cuhk
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "IRRA"))
from datasets.bases import tokenize  # noqa: E402
from datasets.build import build_transforms  # noqa: E402
from model import build_model  # noqa: E402
from utils.iotools import load_train_configs  # noqa: E402
from utils.simple_tokenizer import SimpleTokenizer  # noqa: E402

from .extract_clip import load_split  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CUHK-PEDES")
    ap.add_argument("--ckpt", default="checkpoints/irra_cuhk")
    ap.add_argument("--out", default="emb/irra_cuhk")
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = load_train_configs(os.path.join(args.ckpt, "configs.yaml"))
    cfg.training = False
    model = build_model(cfg, num_classes=11003)
    sd = torch.load(os.path.join(args.ckpt, "best.pth"), map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing", len(missing), "unexpected", len(unexpected))
    # build_model converts to fp16; fp32 is ~5x faster on the T2000 + cu130 build.
    model = model.float().cuda().eval()

    tf = build_transforms(img_size=cfg.img_size, is_train=False)
    tokenizer = SimpleTokenizer()
    for split in ["test", "val"]:
        paths, gid, caps, qid = load_split(args.root, split)
        g = []
        for i in range(0, len(paths), args.bs):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + args.bs]]).cuda()
            g.append(model.encode_image(x).cpu())
            print(f"{split} img {i + len(x)}/{len(paths)}", flush=True)
        q = []
        for i in range(0, len(caps), args.bs):
            t = torch.stack([tokenize(c, tokenizer, cfg.text_length) for c in caps[i:i + args.bs]]).cuda()
            q.append(model.encode_text(t).cpu())
        g = torch.nn.functional.normalize(torch.cat(g), dim=1).numpy()
        q = torch.nn.functional.normalize(torch.cat(q), dim=1).numpy()
        np.savez(f"{args.out}_{split}.npz", q=q, g=g, qid=qid, gid=gid)
        print(split, q.shape, g.shape)


if __name__ == "__main__":
    main()
