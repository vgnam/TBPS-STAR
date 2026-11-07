"""Open-set metrics. `known` is a boolean mask over queries; `correct` marks
known queries whose top-1 gallery image has the right identity."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(conf, known):
    return roc_auc_score(known.astype(int), conf)


def fpr_at_tpr(conf, known, tpr=0.95):
    tau = np.quantile(conf[known], 1 - tpr)
    return (conf[~known] >= tau).mean()


def oscr_curve(conf, known, correct):
    """CCR (correct & accepted known / all known) vs FPR (accepted unknown /
    all unknown), sweeping the threshold from high to low (Dhamija et al., 2018)."""
    order = np.argsort(-conf, kind="stable")
    is_unk = ~known[order]
    is_cc = (known & correct)[order]
    fpr = np.concatenate([[0.0], np.cumsum(is_unk) / max(is_unk.sum(), 1)])
    ccr = np.concatenate([[0.0], np.cumsum(is_cc) / max(known.sum(), 1)])
    return fpr, ccr


def oscr(conf, known, correct):
    fpr, ccr = oscr_curve(conf, known, correct)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed it
    return trapz(ccr, fpr)


def dir_at_far(conf, known, correct, far):
    fpr, ccr = oscr_curve(conf, known, correct)
    return ccr[np.searchsorted(fpr, far, side="right") - 1]


def rank1_map(S, qid, gid):
    order = np.argsort(-S, axis=1)
    match = gid[order] == qid[:, None]
    r1 = match[:, 0].mean()
    hits = np.cumsum(match, 1)
    prec = hits / np.arange(1, S.shape[1] + 1)
    ap = (prec * match).sum(1) / np.maximum(match.sum(1), 1)
    return r1, ap.mean()
