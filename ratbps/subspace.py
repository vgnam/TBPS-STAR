"""Training-free subspace probes on frozen CLIP TBPS features (SubTTA-style).

SubTTA (arXiv 2601.08139) attributes VLM failure under shift to a modality gap
plus visual nuisance outside the textual semantic span, and fixes it by
aligning the visual principal subspace to the textual one and projecting onto
the latter. Here both test-split sets are available offline (Pretrain-then-
Adapt), so the textual subspace is the PCA of all query captions and the
visual one the PCA of the gallery. No labels, no pseudo pairs.

usage: python -m ratbps.subspace --npz results/kaggle_uatta_v1/clip_zeroshot_rstp_test.npz
"""
import argparse

import torch
import torch.nn.functional as F

from .clip_tta import load, metrics


def basis(x, r):
    """Top-r principal directions of the rows of x, (r, d)."""
    return torch.linalg.eigh(x.T @ x).eigenvectors.flip(1)[:, :r].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ranks", default="32,64,128,256,384")
    args = ap.parse_args()
    c = load(args.npz)
    T, I = c["txt"], c["img"]

    def report(name, t, i):
        t, i = F.normalize(t, dim=-1), F.normalize(i, dim=-1)
        m = metrics(c, lambda q: t[q] @ i.T)
        print(f"{name:<36} R1 {m['R1']:6.2f}  R5 {m['R5']:6.2f}  mAP {m['mAP']:6.2f}  mINP {m['mINP']:6.2f}", flush=True)

    report("zero-shot", T, I)
    gap = (T.mean(0) - I.mean(0)).norm().item()
    print(f"modality gap ||mean(T) - mean(I)|| = {gap:.3f}")
    Tc, Ic = T - T.mean(0), I - I.mean(0)
    report("centre each modality", Tc, Ic)
    for r in (int(x) for x in args.ranks.split(",")):
        Bt, Bv = basis(Tc, r), basis(Ic, r)
        print(f"r={r}: chordal distance^2 = {r - (Bt @ Bv.T).pow(2).sum().item():.1f}")
        report(f"  project both on text span", Tc @ Bt.T @ Bt, Ic @ Bt.T @ Bt)
        report(f"  project both on image span", Tc @ Bv.T @ Bv, Ic @ Bv.T @ Bv)
        # Fernando et al. subspace alignment: image coords mapped onto the text basis
        report(f"  SA image->text", Tc @ Bt.T, Ic @ Bv.T @ (Bv @ Bt.T))


if __name__ == "__main__":
    main()
