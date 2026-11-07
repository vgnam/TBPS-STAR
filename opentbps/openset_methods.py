"""Open-set rejection rules borrowed from other fields, on precomputed embeddings.

Cohorts come from the other split (disjoint identities): for test episodes the
val captions/images, for the val calibration episode the test ones.

  face / biometrics     ID-level max, ID ratio test (Lowe / open-set face ID),
                        ID-level mean (template aggregation)
  speaker verification  Z-norm (gallery side), T-norm (query side), S-norm,
                        AS-norm (top-N cohort)
  extreme value / FRVT  T-norm Gaussian tail raised to the gallery size N:
                        P(max of N impostors < s1), the same ranking as T-norm
                        within an episode, but N-aware when thresholds move
  conformal novelty     gallery-side conformal p-values (Bates et al., 2023),
                        min over gallery, plain and N-aware
  OOD detection         GEN (Liu et al., CVPR 2023)

usage: python -m opentbps.openset_methods --emb emb/irra_cuhk
"""
import argparse

import numpy as np
from scipy.special import log_ndtr, logsumexp

from .calibrate import _row
from .run_openset import calibrated_far, load, make_split

ASNORM_TOP = 200


def id_scores(S, gid_e):
    """Per-identity max and mean similarity: (nq x n_ids) each."""
    order = np.argsort(gid_e, kind="stable")
    g_sorted = gid_e[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    Ss = S[:, order]
    counts = np.diff(np.r_[starts, len(g_sorted)])
    return np.maximum.reduceat(Ss, starts, axis=1), np.add.reduceat(Ss, starts, axis=1) / counts


def conformal_pmin(S, S_cohort_g):
    """Smallest gallery-side conformal p-value per query.

    p(q, g) = (1 + #{cohort captions r : s(r, g) >= s(q, g)}) / (1 + R).
    """
    R = S_cohort_g.shape[0]
    srt = np.sort(S_cohort_g, axis=0)
    p = np.empty_like(S)
    for j in range(S.shape[1]):
        p[:, j] = (1 + R - np.searchsorted(srt[:, j], S[:, j], side="left")) / (1 + R)
    return p.min(1)


def gen(S, T=0.02, M=10, gamma=0.1):
    logp = S / T - logsumexp(S / T, axis=1, keepdims=True)
    p = np.exp(-np.sort(-logp, axis=1)[:, :M])
    return -(p ** gamma * (1 - p) ** gamma).sum(1)


def confidences(q, g_e, gid_e, q_coh, g_coh):
    """All rules for one episode. q_coh / g_coh: cohort captions / images."""
    S = q @ g_e.T
    S_q = q @ g_coh.T                      # query vs cohort images   (T-norm)
    S_g = q_coh @ g_e.T                    # cohort captions vs gallery (Z-norm)
    mu_q, sd_q = S_q.mean(1, keepdims=True), S_q.std(1, keepdims=True)
    mu_g, sd_g = S_g.mean(0, keepdims=True), S_g.std(0, keepdims=True)
    top_q = -np.sort(-S_q, axis=1)[:, :ASNORM_TOP]
    top_g = -np.sort(-S_g, axis=0)[:ASNORM_TOP]
    amu_q, asd_q = top_q.mean(1, keepdims=True), top_q.std(1, keepdims=True)
    amu_g, asd_g = top_g.mean(0, keepdims=True), top_g.std(0, keepdims=True)
    N = S.shape[1]

    id_max, id_mean = id_scores(S, gid_e)
    top2 = -np.sort(-id_max, axis=1)[:, :2]
    t_norm = (S.max(1, keepdims=True) - mu_q)[:, 0] / sd_q[:, 0]
    p_min = conformal_pmin(S, S_g)
    as_norm = (0.5 * ((S - amu_g) / asd_g + (S - amu_q) / asd_q)).max(1)
    # Gaussian tail of the gallery-side null: a smooth stand-in for the
    # conformal p-value, whose resolution (1/(R+1)) saturates at the top.
    logp_z = log_ndtr(-((S - mu_g) / sd_g).max(1))
    mu_all, sd_all = S_q.mean(), S_q.std()
    return S, {
        "max_cos": S.max(1),
        "ID ratio test": top2[:, 0] - top2[:, 1],
        "ID mean (template)": id_mean.max(1),
        "Z-norm": ((S - mu_g) / sd_g).max(1),
        "T-norm": t_norm,
        "S-norm": (0.5 * ((S - mu_g) / sd_g + (S - mu_q) / sd_q)).max(1),
        "AS-norm": as_norm,
        "conformal p (min)": -np.log(p_min),
        "GEN": gen(S),
        # N-aware versions: log P(max of N null scores < observed), Gaussian null
        "T-norm ^N": N * log_ndtr(t_norm),
        "max_cos ^N (global)": N * log_ndtr((S.max(1) - mu_all) / sd_all),
        "AS-norm ^N": N * log_ndtr(as_norm),
        "Z-norm tail ^N": N * np.log1p(-np.exp(logp_z)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    qt, gt, qidt, gidt = load(args.emb, "test")
    qv, gv, qidv, gidv = load(args.emb, "val")

    def episode(split, ratio, seed):
        q, g, qid, gid, q_coh, g_coh = ((qt, gt, qidt, gidt, qv, gv) if split == "test"
                                        else (qv, gv, qidv, gidv, qt, gt))
        g_mask, known = make_split(gid, qid, int(ratio * len(np.unique(gid))), np.random.default_rng(seed))
        S, conf = confidences(q, g[g_mask], gid[g_mask], q_coh, g_coh)
        return conf, known, gid[g_mask][S.argmax(1)] == qid

    test = [episode("test", 0.7, s) for s in range(args.seeds)]
    names = list(test[0][0])
    print(f"\n== {args.emb} | open-set rules from other fields | 70% enrolled, {args.seeds} seeds")
    print(f"{'rule':<22}" + "".join(f"{m:>15}" for m in ["AUROC", "FPR@95", "OSCR", "DIR@1%", "DIR@10%"]))
    for n in names:
        print(f"{n:<22}" + "".join(f"{v:>15}" for v in _row([e[0][n] for e in test], test)))

    sweep = [0.2, 0.4, 0.7, 0.9]
    v = episode("val", 0.7, 123)
    t = {r: episode("test", r, 0) for r in sweep}
    print("\n== threshold for FAR=5% set on val (70%), applied to test at other gallery sizes")
    print(f"{'rule':<22}" + "".join(f"{'enr ' + str(r):>18}" for r in sweep) + f"{'max |FAR-5|':>13}")
    for n in names:
        cells = [calibrated_far(v[0][n], v[1], t[r][0][n], t[r][1]) for r in sweep]
        dev = max(abs(100 * a - 5) for a, _ in cells)
        print(f"{n:<22}" + "".join(f"FAR {100*a:4.1f} TAR {100*b:4.1f}".rjust(18) for a, b in cells) + f"{dev:13.1f}")
    # Baseline that needs re-calibration per deployment: a val episode with the
    # same enrolled ratio (hence ~ the same gallery size) as each test episode.
    vr = {r: episode("val", r, 123) for r in sweep}
    cells = [calibrated_far(vr[r][0]["max_cos"], vr[r][1], t[r][0]["max_cos"], t[r][1]) for r in sweep]
    dev = max(abs(100 * a - 5) for a, _ in cells)
    print(f"{'max_cos, val at test N':<22}" + "".join(f"FAR {100*a:4.1f} TAR {100*b:4.1f}".rjust(18)
                                                     for a, b in cells) + f"{dev:13.1f}")


if __name__ == "__main__":
    main()
