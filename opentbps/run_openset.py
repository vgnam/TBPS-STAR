"""Open-set TBPS protocol on precomputed embeddings.

Per seed, test identities are split into enrolled (images go to the gallery,
captions are known queries) and unenrolled (captions are unknown queries,
images are dropped). Validation identities are disjoint from test and are
used only for calibration (threshold, CSLS reference bank).

usage: python -m opentbps.run_openset --emb emb/irra_cuhk
"""
import argparse
import json

import numpy as np

from .metrics import auroc, dir_at_far, fpr_at_tpr, oscr, rank1_map
from .scores import SCORES


def load(prefix, split):
    d = np.load(f"{prefix}_{split}.npz")
    return d["q"], d["g"], d["qid"], d["gid"]


def make_split(gid, qid, n_enroll, rng):
    ids = np.unique(gid)
    enrolled = rng.choice(ids, size=n_enroll, replace=False)
    g_mask = np.isin(gid, enrolled)
    known = np.isin(qid, enrolled)
    return g_mask, known


def evaluate(q, g, qid, gid, g_mask, known, q_ref, score_names):
    S = q @ g[g_mask].T
    gid_e = gid[g_mask]
    correct = gid_e[S.argmax(1)] == qid
    S_ref = q_ref @ g[g_mask].T
    r1, mAP = rank1_map(S[known], qid[known], gid_e)
    row = {"R1_known": r1, "mAP_known": mAP, "n_known": int(known.sum()),
           "n_unknown": int((~known).sum()), "n_gallery": int(g_mask.sum())}
    confs = {}
    for name in score_names:
        c = SCORES[name](S, S_ref=S_ref)
        confs[name] = c
        row[name] = {
            "AUROC": auroc(c, known),
            "FPR@95": fpr_at_tpr(c, known),
            "OSCR": oscr(c, known, correct),
            "DIR@1%": dir_at_far(c, known, correct, 0.01),
            "DIR@10%": dir_at_far(c, known, correct, 0.10),
        }
    return row, confs, correct


def calibrated_far(conf_val, known_val, conf_test, known_test, target_far=0.05):
    """Threshold set on the val protocol for a target FAR, applied to test."""
    tau = np.quantile(conf_val[~known_val], 1 - target_far)
    return (conf_test[~known_test] >= tau).mean(), (conf_test[known_test] >= tau).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="prefix; reads <prefix>_test.npz and <prefix>_val.npz")
    ap.add_argument("--enroll_ratio", type=float, default=0.7)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--scores", default=",".join(SCORES))
    ap.add_argument("--gallery_sweep", default="0.2,0.4,0.7,0.9",
                    help="enrolled ratios on test for the calibration-drift experiment")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    names = args.scores.split(",")

    qt, gt, qidt, gidt = load(args.emb, "test")
    qv, gv, qidv, gidv = load(args.emb, "val")
    n_test_ids = len(np.unique(gidt))
    n_val_ids = len(np.unique(gidv))

    # 1) main table: fixed enroll ratio, several seeds
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        g_mask, known = make_split(gidt, qidt, int(args.enroll_ratio * n_test_ids), rng)
        row, _, _ = evaluate(qt, gt, qidt, gidt, g_mask, known, qv, names)
        rows.append(row)

    print(f"\n== {args.emb} | enrolled {args.enroll_ratio:.0%} of {n_test_ids} test IDs | {args.seeds} seeds")
    r = rows[0]
    print(f"gallery {r['n_gallery']} imgs | known q {r['n_known']} | unknown q {r['n_unknown']}")
    print("closed-set on known queries: R1 %.1f  mAP %.1f" % (
        100 * np.mean([x["R1_known"] for x in rows]), 100 * np.mean([x["mAP_known"] for x in rows])))
    print(f"{'score':<12}" + "".join(f"{m:>16}" for m in ["AUROC", "FPR@95", "OSCR", "DIR@1%", "DIR@10%"]))
    summary = {}
    for n in names:
        cells, summary[n] = [], {}
        for m in ["AUROC", "FPR@95", "OSCR", "DIR@1%", "DIR@10%"]:
            v = np.array([x[n][m] for x in rows]) * 100
            summary[n][m] = (v.mean(), v.std())
            cells.append(f"{v.mean():9.1f} +-{v.std():4.1f}")
        print(f"{n:<12}" + "".join(f"{c:>16}" for c in cells))

    # 2) calibration drift: tau from val at 70% enrolled, test at varying gallery sizes
    rng = np.random.default_rng(123)
    gv_mask, known_v = make_split(gidv, qidv, int(0.7 * n_val_ids), rng)
    _, conf_v, _ = evaluate(qv, gv, qidv, gidv, gv_mask, known_v, qt, names)
    print(f"\n== threshold set on val (70% enrolled) for FAR=5%, applied to test with other gallery sizes")
    print(f"{'score':<12}" + "".join(f"{'enr ' + s:>18}" for s in args.gallery_sweep.split(",")))
    drift = {}
    for n in names:
        cells, drift[n] = [], {}
        for ratio in map(float, args.gallery_sweep.split(",")):
            g_mask, known = make_split(gidt, qidt, int(ratio * n_test_ids), np.random.default_rng(0))
            _, conf_t, _ = evaluate(qt, gt, qidt, gidt, g_mask, known, qv, [n])
            far, tar = calibrated_far(conf_v[n], known_v, conf_t[n], known)
            drift[n][ratio] = (far, tar)
            cells.append(f"FAR {100*far:4.1f} TAR {100*tar:4.1f}")
        print(f"{n:<12}" + "".join(f"{c:>18}" for c in cells))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "summary": summary,
                       "drift": {n: {str(k): v for k, v in d.items()} for n, d in drift.items()}},
                      f, indent=1, default=float)


if __name__ == "__main__":
    main()
