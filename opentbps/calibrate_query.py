"""Query-aware confidence: does knowing *what the caption says* close the gap?

Known vs unknown is assigned per identity at random, so the query alone carries
no label information (q-only must be ~50 AUROC; anything else is a leak). What
the query can carry is how high a true match is expected to score: a specific
caption should score high when its person is present, a generic one not.

Variants (all trained on val only, evaluated on the run_openset test episodes):
  topK+q       LR on top-K scores + text embedding (+ caption length)
  q-only       LR on text embedding alone (leak check)
  s1-expected  ridge regression q -> own-identity score on the full val
               gallery, confidence = s1 - predicted (a per-query threshold)

usage: python -m opentbps.calibrate_query --emb emb/irra_cuhk --root data/CUHK-PEDES
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibrate import _row, features
from .run_openset import calibrated_far, load, make_split


def caption_lengths(root, split):
    annos = [a for a in json.load(open(os.path.join(root, "reid_raw.json"))) if a["split"] == split]
    return np.array([len(c.split()) for a in annos for c in a["captions"]], dtype=float)


def episodes(q, g, qid, gid, ratios, seeds):
    n_ids = len(np.unique(gid))
    for ratio in ratios:
        for seed in seeds:
            g_mask, known = make_split(gid, qid, int(ratio * n_ids), np.random.default_rng(seed))
            S = q @ g[g_mask].T
            correct = gid[g_mask][S.argmax(1)] == qid
            yield S, known, correct


def own_score(q, g, qid, gid):
    """Best similarity of each query to an image of its own identity."""
    S = q @ g.T
    S[gid[None, :] != qid[:, None]] = -np.inf
    return S.max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--root", default="data/CUHK-PEDES")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    qt, gt, qidt, gidt = load(args.emb, "test")
    qv, gv, qidv, gidv = load(args.emb, "val")
    lent, lenv = caption_lengths(args.root, "test"), caption_lengths(args.root, "val")
    assert len(lent) == len(qt) and len(lenv) == len(qv), "caption order mismatch"

    test_eps = list(episodes(qt, gt, qidt, gidt, [0.7], range(args.seeds)))
    val_eps = list(episodes(qv, gv, qidv, gidv, [0.7], range(100, 110)))
    sweep = [0.2, 0.4, 0.7, 0.9]
    sweep_eps = {r: next(episodes(qt, gt, qidt, gidt, [r], [0])) for r in sweep}
    v_cal = next(episodes(qv, gv, qidv, gidv, [0.7], [123]))

    def stack(eps, qe, ln, use_topk=True, use_q=True):
        X = []
        for S, _, _ in eps:
            parts = []
            if use_topk:
                parts.append(features(S))
            if use_q:
                parts += [qe, ln[:, None]]
            X.append(np.hstack(parts))
        return X

    yk = np.concatenate([e[1] for e in val_eps])
    results = {}
    print(f"\n== {args.emb} | query-aware calibrators (train: val, test: {args.seeds} seeds, 70% enrolled)")
    print(f"{'model':<24}" + "".join(f"{m:>15}" for m in ["AUROC", "FPR@95", "OSCR", "DIR@1%", "DIR@10%"]))
    print(f"{'(ref) max_cos':<24}" + "".join(f"{v:>15}" for v in _row([e[0].max(1) for e in test_eps], test_eps)))

    for name, use_topk, use_q in [("LR topK", True, False), ("LR q-only (leak chk)", False, True),
                                  ("LR topK+q", True, True)]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        clf.fit(np.vstack(stack(val_eps, qv, lenv, use_topk, use_q)), yk)
        score = lambda eps, qe, ln: [clf.predict_proba(x)[:, 1] for x in stack(eps, qe, ln, use_topk, use_q)]
        conf = score(test_eps, qt, lent)
        print(f"{name:<24}" + "".join(f"{v:>15}" for v in _row(conf, test_eps)))
        results[name] = score

    # Per-query expected true-match score, learned on val with the full val gallery.
    reg = Ridge(alpha=10.0).fit(np.hstack([qv, lenv[:, None]]), own_score(qv, gv, qidv, gidv))
    exp_t = reg.predict(np.hstack([qt, lent[:, None]]))
    exp_v = reg.predict(np.hstack([qv, lenv[:, None]]))
    r = np.corrcoef(exp_t, own_score(qt, gt, qidt, gidt))[0, 1]
    conf = [e[0].max(1) - exp_t for e in test_eps]
    print(f"{'s1-expected':<24}" + "".join(f"{v:>15}" for v in _row(conf, test_eps)))
    print(f"   (corr. predicted vs actual own-ID score on test: r = {r:.3f})")
    results["s1-expected"] = None

    print("\n== threshold for FAR=5% set on val (70% enrolled), applied to test at other gallery sizes")
    print(f"{'model':<24}" + "".join(f"{'enr ' + str(x):>18}" for x in sweep))
    rows = {"(ref) max_cos": [calibrated_far(v_cal[0].max(1), v_cal[1], sweep_eps[x][0].max(1), sweep_eps[x][1])
                              for x in sweep],
            "s1-expected": [calibrated_far(v_cal[0].max(1) - exp_v, v_cal[1], sweep_eps[x][0].max(1) - exp_t,
                                           sweep_eps[x][1]) for x in sweep]}
    sc = results["LR topK+q"]
    cv = sc([v_cal], qv, lenv)[0]
    rows["LR topK+q"] = [calibrated_far(cv, v_cal[1], sc([sweep_eps[x]], qt, lent)[0], sweep_eps[x][1]) for x in sweep]
    for name, vals in rows.items():
        print(f"{name:<24}" + "".join(f"FAR {100*f:4.1f} TAR {100*t:4.1f}".rjust(18) for f, t in vals))


if __name__ == "__main__":
    main()
