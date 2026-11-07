"""Closed-set numbers of caches, to compare with APTM's published CUHK-PEDES
result (R1 76.53 with ITC top-128 re-ranked by ITM).

usage: python -m ratbps.check_baseline --json out.json emb/a.npz [emb/b.npz ...]
"""
import argparse
import json

import numpy as np

from .run import gpu, metrics


def report(path):
    d = dict(np.load(path))
    c = gpu(d)
    N = c["cand"].shape[1]
    hit = c["gid"][c["cand"]] == c["qid"][:, None]
    out = {"ITC": metrics(c, c["itc"])}
    line = f"{path}: ITC R1 {out['ITC']['R1']:.2f}"
    for n in sorted({n for n in (16, 32, 64, 128) if n <= N} | {N}):
        sub = gpu(d, n=n)
        m = metrics(sub, sub["itm"])
        out[f"ITM@{n}"] = dict(m, recall=100 * hit[:, :n].any(1).float().mean().item())
        line += f" | top-{n}: recall {out[f'ITM@{n}']['recall']:.2f}, ITM R1 {m['R1']:.2f} mAP {m['mAP']:.2f}"
    print(line, flush=True)
    out["ITM"] = out[f"ITM@{N}"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    res = {p: report(p) for p in args.caches}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
