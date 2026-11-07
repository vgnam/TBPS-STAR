"""Two independent models on the same open-set episodes: IRRA (dual encoder)
and APTM (ITC + TBPS-trained ITM). If the ~86 AUROC ceiling is an
identifiability limit, combining them should not break it either.

usage: python -m opentbps.eval_ensemble --irra emb/irra_cuhk --aptm emb/aptm_cuhk --itm results/kaggle_aptm/itm_aptm
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibrate import _row
from .eval_verifier import episode
from .run_openset import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--irra", required=True)
    ap.add_argument("--aptm", required=True)
    ap.add_argument("--itm", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    def eps(split, seeds):
        qi, gi, qid, gid = load(args.irra, split)
        qa, ga, qid_a, gid_a = load(args.aptm, split)
        assert (qid == qid_a).all() and (gid == gid_a).all(), "split order differs between models"
        d = np.load(f"{args.itm}_{split}.npz")
        out = []
        for s in seeds:
            e = episode(qa, ga, qid, gid, d["idx"], d["itm_logit"], 0.7, s)
            e["s1_irra"] = (qi @ gi[_mask(gid, qid, s)].T).max(1)
            out.append(e)
        return out

    test, val = eps("test", range(args.seeds)), eps("val", range(100, 110))
    print(f"\n== IRRA + APTM-ITM ensemble | 70% enrolled, {args.seeds} seeds (identity: APTM ITM re-rank)")
    print(f"{'confidence':<28}" + "".join(f"{m:>15}" for m in ["AUROC", "FPR@95", "OSCR", "DIR@1%", "DIR@10%"]))
    rows = {"IRRA max_cos": lambda e: e["s1_irra"], "APTM max ITM": lambda e: e["itm_max"]}
    feats = lambda e: np.column_stack([e["s1_irra"], e["itm_max"], e["s1"]])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(np.vstack([feats(e) for e in val]), np.concatenate([e["known"] for e in val]))
    rows["LR(IRRA s1, APTM ITM, ITC)"] = lambda e: clf.predict_proba(feats(e))[:, 1]
    for name, f in rows.items():
        print(f"{name:<28}" + "".join(f"{v:>15}" for v in
                                      _row([f(e) for e in test], [(None, e["known"], e["correct_rerank"]) for e in test])))


def _mask(gid, qid, seed):
    from .run_openset import make_split
    return make_split(gid, qid, int(0.7 * len(np.unique(gid))), np.random.default_rng(seed))[0]


if __name__ == "__main__":
    main()
