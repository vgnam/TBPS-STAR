"""Learned confidence: can a small model on top-K scores close the open-set gap?

The calibrator is trained only on the val split (identities disjoint from
test), with open-set episodes built the same way as the test protocol, then
evaluated on the exact test episodes of run_openset.

usage: python -m opentbps.calibrate --emb emb/irra_cuhk
"""
import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc, dir_at_far, fpr_at_tpr, oscr
from .run_openset import calibrated_far, load, make_split

K = 50


def features(S):
    top = -np.sort(-S, axis=1)[:, :K]
    rest = top[:, 3:]
    return np.column_stack([top, top[:, 0] - rest.mean(1), rest.std(1)])


def episodes(q, g, qid, gid, ratios, seeds):
    """Yield (features, known, correct) for each open-set episode."""
    n_ids = len(np.unique(gid))
    for ratio in ratios:
        for seed in seeds:
            g_mask, known = make_split(gid, qid, int(ratio * n_ids), np.random.default_rng(seed))
            S = q @ g[g_mask].T
            correct = gid[g_mask][S.argmax(1)] == qid
            yield features(S), known, correct


def train_set(q, g, qid, gid, ratios, seeds):
    X, yk, yc = zip(*episodes(q, g, qid, gid, ratios, seeds))
    X, yk, yc = np.vstack(X), np.concatenate(yk), np.concatenate(yc)
    return X, yk, yk & yc


def models():
    return {
        "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "GBM": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--enroll_ratio", type=float, default=0.7)
    args = ap.parse_args()

    qt, gt, qidt, gidt = load(args.emb, "test")
    qv, gv, qidv, gidv = load(args.emb, "val")
    test_eps = list(episodes(qt, gt, qidt, gidt, [args.enroll_ratio], range(args.seeds)))
    sweep = [0.2, 0.4, 0.7, 0.9]
    sweep_eps = {r: next(episodes(qt, gt, qidt, gidt, [r], [0])) for r in sweep}

    # Train variants: gallery size fixed at 70% vs mixed sizes; target known vs known&correct.
    train_cfgs = {
        "fix70": ([0.7], range(100, 110)),
        "mix": (sweep, range(100, 105)),
    }
    print(f"\n== {args.emb} | calibrators trained on val, evaluated on test ({args.seeds} seeds, {args.enroll_ratio:.0%} enrolled)")
    print(f"{'model':<24}" + "".join(f"{m:>15}" for m in ["AUROC", "FPR@95", "OSCR", "DIR@1%", "DIR@10%"]))
    print(f"{'(ref) max_cos':<24}" + "".join(f"{v:>15}" for v in _row([e[0][:, 0] for e in test_eps], test_eps)))

    drift_rows = {}
    for cfg, (ratios, seeds) in train_cfgs.items():
        X, y_known, y_cc = train_set(qv, gv, qidv, gidv, ratios, seeds)
        for target, y in [("known", y_known), ("correct", y_cc)]:
            for mname, mk in models().items():
                clf = mk().fit(X, y)
                conf = [clf.predict_proba(e[0])[:, 1] for e in test_eps]
                name = f"{mname}-{cfg}-{target}"
                print(f"{name:<24}" + "".join(f"{v:>15}" for v in _row(conf, test_eps)))
                if target == "known":
                    # threshold for FAR=5% on a val episode at 70%, applied across test gallery sizes
                    Xv, kv, _ = next(episodes(qv, gv, qidv, gidv, [0.7], [123]))
                    cv = clf.predict_proba(Xv)[:, 1]
                    drift_rows[name] = [calibrated_far(cv, kv, clf.predict_proba(sweep_eps[r][0])[:, 1], sweep_eps[r][1])
                                        for r in sweep]

    Xv, kv, _ = next(episodes(qv, gv, qidv, gidv, [0.7], [123]))
    drift_rows = {"(ref) max_cos": [calibrated_far(Xv[:, 0], kv, sweep_eps[r][0][:, 0], sweep_eps[r][1]) for r in sweep],
                  **drift_rows}
    print("\n== threshold for FAR=5% set on val (70% enrolled), applied to test at other gallery sizes")
    print(f"{'model':<24}" + "".join(f"{'enr ' + str(r):>18}" for r in sweep))
    for name, vals in drift_rows.items():
        print(f"{name:<24}" + "".join(f"FAR {100*f:4.1f} TAR {100*t:4.1f}".rjust(18) for f, t in vals))


def _row(confs, eps):
    out = []
    for fn in [lambda c, k, r: auroc(c, k), lambda c, k, r: fpr_at_tpr(c, k), oscr,
               lambda c, k, r: dir_at_far(c, k, r, 0.01), lambda c, k, r: dir_at_far(c, k, r, 0.10)]:
        v = 100 * np.array([fn(c, e[1], e[2]) for c, e in zip(confs, eps)])
        out.append(f"{v.mean():.1f} +-{v.std():.1f}")
    return out


if __name__ == "__main__":
    main()
