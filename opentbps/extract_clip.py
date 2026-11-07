"""Zero-shot CLIP embeddings for CUHK-PEDES val/test.

usage: python -m opentbps.extract_clip --root data/CUHK-PEDES --out emb/clip_cuhk
"""
import argparse
import json
import os

import numpy as np
import open_clip
import torch
import torchvision.transforms as T
from PIL import Image


def load_split(root, split):
    annos = [a for a in json.load(open(os.path.join(root, "reid_raw.json"))) if a["split"] == split]
    paths = [os.path.join(root, "imgs", a["file_path"]) for a in annos]
    gid = np.array([a["id"] for a in annos])
    caps, qid = [], []
    for a in annos:
        for c in a["captions"]:
            caps.append(c)
            qid.append(a["id"])
    return paths, gid, caps, np.array(qid)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/CUHK-PEDES")
    ap.add_argument("--out", default="emb/clip_cuhk")
    ap.add_argument("--arch", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model, _, _ = open_clip.create_model_and_transforms(args.arch, pretrained=args.pretrained)
    tok = open_clip.get_tokenizer(args.arch)
    # fp32: on the T2000 + torch cu130 build, fp16 is ~5x slower than fp32.
    model = model.cuda().eval()
    # Squash instead of center-crop so tall person crops keep head and feet.
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    for split in ["test", "val"]:
        paths, gid, caps, qid = load_split(args.root, split)
        g = []
        for i in range(0, len(paths), args.bs):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + args.bs]]).cuda()
            g.append(model.encode_image(x).float().cpu())
            print(f"{split} img {i + len(x)}/{len(paths)}", flush=True)
        q = []
        for i in range(0, len(caps), args.bs):
            q.append(model.encode_text(tok(caps[i:i + args.bs]).cuda()).float().cpu())
        g = torch.nn.functional.normalize(torch.cat(g), dim=1).numpy()
        q = torch.nn.functional.normalize(torch.cat(q), dim=1).numpy()
        np.savez(f"{args.out}_{split}.npz", q=q, g=g, qid=qid, gid=gid)
        print(split, q.shape, g.shape)


if __name__ == "__main__":
    main()
