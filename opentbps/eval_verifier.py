"""Open-set evaluation of a cross-encoder verifier (ITM) on top of IRRA.

ITM scores were computed on each query's top-K over the full split gallery
(kernel_itm). In an episode, only the enrolled candidates among those K are
used. Identity is IRRA's enrolled top-1 unless stated ("rerank": ITM argmax).

usage: python -m opentbps.eval_verifier --emb emb/irra_cuhk --itm results/kaggle_itm/itm_blipbase
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibrate import _row
from .run_openset import calibrated_far, load, make_split


def episode(q, g, qid, gid, idx, itm, ratio, seed):
    """Per-query features restricted to enrolled candidates of one episode."""
    g_mask, known = make_split(gid, qid, int(ratio * len(np.unique(gid))), np.random.default_rng(seed))
    S = q @ g[g_mask].T
    s1 = S.max(1)
    gid_e = gid[g_mask]
    pred_irra = gid_e[S.argmax(1)]
    enrolled = g_mask[idx]                      # nq x K: candidate is in this gallery
    lo = itm.min() - 1.0
    itm_e = np.where(enrolled, itm, lo)
    has = enrolled.any(1)
    first = enrolled.argmax(1)                  # IRRA's enrolled top-1 within the K list
    itm_top1 = np.where(has, itm[np.arange(len(itm)), first], lo)
    itm_max = itm_e.max(1)
    pred_rerank = np.where(has, gid[idx[np.arange(len(idx)), itm_e.argmax(1)]], pred_irra)
    return dict(s1=s1, itm_top1=itm_top1, itm_max=itm_max, known=known,
                correct=pred_irra == qid, correct_rerank=pred_rerank == qid, coverage=has.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--itm", required=True, help="prefix; reads <prefix>_test.npz / _val.npz")
    ap.add_argument("--key", default="itm_logit")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    data = {}
    for split in ["test", "val"]:
        q, g, qid, gid = load(args.emb, split)
        d = np.load(f"{args.itm}_{split}.npz")
        data[split] = (q, g, qid, gid, d["idx"], d[args.key])

    test = [episode(*data["test"], 0.7, s) for s in range(args.seeds)]
    val = [episode(*data["val"], 0.7, s) for s in range(100, 110)]
    print(f"\n== ITM verifier ({args.itm}, {args.key}) | 70% enrolled, {args.seeds} seeds | "
          f"enrolled candidate in top-K for {100*np.mean([e['coverage'] for e in test]):.1f}% of queries")
    r1 = np.mean([e["correct"][e["known"]].mean() for e in test])
    r1r = np.mean([e["correct_rerank"][e["known"]].mean() for e in test])
    print(f"closed-set R1 on known: IRRA {100*r1:.1f} | ITM re-rank {100*r1r:.1f}")
    print(f"{'confidence':<28}" + "".join(f"{m:>15}" for m in ["AUROC", "FPR@95", "OSCR", "DIR@1%", "DIR@10%"]))

    def show(name, confs, eps, corr_key="correct"):
        eps2 = [(None, e["known"], e[corr_key]) for e in eps]
        print(f"{name:<28}" + "".join(f"{v:>15}" for v in _row(confs, eps2)))

    show("max_cos (IRRA)", [e["s1"] for e in test], test)
    show("ITM of IRRA top-1", [e["itm_top1"] for e in test], test)
    show("max ITM over top-K", [e["itm_max"] for e in test], test)
    show("max ITM + ITM re-rank", [e["itm_max"] for e in test], test, "correct_rerank")

    feats = lambda e: np.column_stack([e["s1"], e["itm_top1"], e["itm_max"]])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(np.vstack([feats(e) for e in val]), np.concatenate([e["known"] for e in val]))
    fused = [clf.predict_proba(feats(e))[:, 1] for e in test]
    show("LR(s1, ITM) fused [val]", fused, test)
    show("LR fused + ITM re-rank", fused, test, "correct_rerank")

    print("\n== threshold for FAR=5% set on val (70%), applied to test at other gallery sizes")
    sweep = [0.2, 0.4, 0.7, 0.9]
    v = val[0]
    t = {r: episode(*data["test"], r, 0) for r in sweep}
    print(f"{'confidence':<28}" + "".join(f"{'enr ' + str(r):>18}" for r in sweep))
    for name, f in [("max_cos (IRRA)", lambda e: e["s1"]), ("max ITM over top-K", lambda e: e["itm_max"]),
                    ("LR fused", lambda e: clf.predict_proba(feats(e))[:, 1])]:
        cells = [calibrated_far(f(v), v["known"], f(t[r]), t[r]["known"]) for r in sweep]
        print(f"{name:<28}" + "".join(f"FAR {100*a:4.1f} TAR {100*b:4.1f}".rjust(18) for a, b in cells))


if __name__ == "__main__":
    main()
