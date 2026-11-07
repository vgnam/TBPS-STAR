"""STAR-TBPS, training-free part (steps 0, 1, 3) on frozen CLIP TBPS features.

step 0  moment alignment: centre each modality on the test split, project onto
        the joint principal subspace keeping `energy` of the variance.
step 1  balanced assignment: Sinkhorn on exp(S/eps) with uniform marginals
        (every gallery image is described by about the same number of
        captions), giving hub-free soft pseudo pairs P.
step 3  same-modality evidence: g is rewarded by how much q reads like the
        captions OT assigned to g (q itself excluded), only for confident
        assignments; images without evidence get the query's average bonus.

The go/no-go question is step 1: are OT pseudo pairs cleaner than the greedy
argmax / mutual top-K pairs UATTA relies on, at equal coverage?

usage: python -m ratbps.star --npz results/kaggle_uatta_v1/clip_zeroshot_rstp_test.npz
"""
import argparse
import json

import torch
import torch.nn.functional as F

from .clip_tta import dev, load, metrics


def align(T, I, energy):
    Tc, Ic = T - T.mean(0), I - I.mean(0)
    cov = (Tc.T @ Tc / len(Tc) + Ic.T @ Ic / len(Ic)) / 2
    w, V = torch.linalg.eigh(cov)
    w, V = w.flip(0), V.flip(1)
    r = int((w.cumsum(0) / w.sum() < energy).sum().item()) + 1
    B = V[:, :r]
    return F.normalize(Tc @ B, dim=-1), F.normalize(Ic @ B, dim=-1), r


def sinkhorn(S, eps, iters=200):
    """log-domain Sinkhorn, uniform marginals; returns P (Q, G) summing to 1."""
    Q, G = S.shape
    log_a, log_b = -torch.log(torch.tensor(float(Q), device=dev)), -torch.log(torch.tensor(float(G), device=dev))
    K = S / eps
    u = torch.zeros(Q, device=dev)
    v = torch.zeros(G, device=dev)
    for _ in range(iters):
        u = log_a - torch.logsumexp(K + v[None], 1)
        v = log_b - torch.logsumexp(K + u[:, None], 0)
    return torch.exp(K + u[:, None] + v[None])


def mutual(S, r):
    p = S.argmax(1)
    beat = (S[:, p].T > S[torch.arange(len(p), device=dev), p][:, None]).sum(1)
    return p, beat < r


def precision_at_coverage(c, p, conf, covs):
    order = conf.argsort(descending=True)
    ok = (c["gid"][p] == c["qid"]).float()[order]
    return {f"{int(x * 100)}%": ok[:max(1, int(x * len(ok)))].mean().item() * 100 for x in covs}


def evidence_bonus(Tq, p, conf, keep, G):
    """mean_{j: p_j = g, j != q, j kept} cos(t_q, t_j); neutral fill where empty."""
    Q = len(Tq)
    A = torch.zeros(Q, G, device=dev)
    idx = keep.nonzero()[:, 0]
    A[idx, p[idx]] = 1.0
    tt = Tq @ Tq.T
    tot, cnt = tt @ A, A.sum(0, keepdim=True).expand(Q, G).clone()
    tot[idx, p[idx]] -= tt[idx, idx]
    cnt[idx, p[idx]] -= 1.0
    has = cnt > 0
    mean = tot / cnt.clamp(min=1)
    neutral = (mean * has).sum(1, keepdim=True) / has.sum(1, keepdim=True).clamp(min=1)
    return torch.where(has, mean, neutral.expand_as(mean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--energy", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--eps-sweep", default="0.005,0.01,0.02,0.05")
    ap.add_argument("--keep", type=float, default=0.6, help="fraction of most confident assignments used as evidence")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--beta-sweep", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--tau", type=float, default=0.02, help="dual-softmax / inverted-softmax temperature")
    args = ap.parse_args()

    c = load(args.npz)
    T, I = c["txt"], c["img"]
    Q, G = len(T), len(I)
    res = dict(rows={}, precision={}, sweep={})

    def row(name, S):
        res["rows"][name] = m = metrics(c, lambda q: S[q])
        print(f"{name:<40} R1 {m['R1']:6.2f}  R5 {m['R5']:6.2f}  R10 {m['R10']:6.2f}  mAP {m['mAP']:6.2f}  "
              f"mINP {m['mINP']:6.2f}", flush=True)

    S0 = T @ I.T
    row("zero-shot", S0)
    Tc, Ic = F.normalize(T - T.mean(0), dim=-1), F.normalize(I - I.mean(0), dim=-1)
    row("centre", Tc @ Ic.T)
    Ta, Ia, r = align(T, I, args.energy)
    S = Ta @ Ia.T
    row(f"step 0: centre + joint subspace (r={r})", S)

    # training-free hubness baselines on the step-0 scores
    row("  + dual-softmax (DSL)", S * F.softmax(S / args.tau, dim=0) * Q)
    row("  + inverted softmax (QB-Norm, bank=test queries)", S / args.tau - torch.logsumexp(S / args.tau, 0, keepdim=True))

    # step 1: pseudo-pair quality at equal coverage
    covs = (0.2, 0.4, 0.6, 0.8, 1.0)
    for name, base in (("zero-shot", S0), ("step 0", S)):
        top2 = base.topk(2, dim=1).values
        p_g = base.argmax(1)
        res["precision"][f"greedy argmax ({name})"] = precision_at_coverage(c, p_g, top2[:, 0] - top2[:, 1], covs)
        p_m, mu = mutual(base, 5)
        res["precision"][f"mutual top-5 ({name})"] = {
            "coverage": mu.float().mean().item() * 100,
            "precision": (c["gid"][p_m[mu]] == c["qid"][mu]).float().mean().item() * 100}
        for eps in (float(x) for x in args.eps_sweep.split(",")):
            P = sinkhorn(base, eps)
            res["precision"][f"OT eps={eps} ({name})"] = precision_at_coverage(c, P.argmax(1), P.max(1).values * Q, covs)
    for k, v in res["precision"].items():
        print(f"precision {k:<28} " + "  ".join(f"{a}:{b:5.1f}" for a, b in v.items()), flush=True)

    # Sinkhorn as a reranker, and step 3
    P = sinkhorn(S, args.eps)
    row(f"  + Sinkhorn rerank (log P, eps={args.eps})", torch.log(P + 1e-30))
    p, conf = P.argmax(1), P.max(1).values
    keep = conf >= conf.quantile(1 - args.keep)
    bonus = evidence_bonus(Ta, p, conf, keep, G)
    row(f"STAR-free: step 0 + step 3 (b={args.beta})", S + args.beta * bonus)
    keep_g = torch.ones_like(keep)
    pg = S.argmax(1)
    row("  step 3 with greedy argmax assignment", S + args.beta * evidence_bonus(Ta, pg, conf, keep_g, G))
    row("  step 3 with OT, no confidence gate", S + args.beta * evidence_bonus(Ta, p, conf, keep_g, G))
    for b in (float(x) for x in args.beta_sweep.split(",")):
        for kf in (0.4, 0.6, 0.8, 1.0):
            kp = conf >= conf.quantile(1 - kf)
            M = S + b * evidence_bonus(Ta, p, conf, kp, G)
            res["sweep"][f"b{b}_keep{kf}"] = metrics(c, lambda q: M[q])
    best = max(res["sweep"].items(), key=lambda kv: kv[1]["R1"])
    print(f"oracle sweep best {best[0]}: R1 {best[1]['R1']:.2f} mAP {best[1]['mAP']:.2f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
