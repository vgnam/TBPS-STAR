"""Training-free confidence scores for open-set TBPS.

Every function takes a similarity matrix S (num_queries x num_gallery, cosine)
and returns one confidence per query: higher = more likely the target is in
the gallery.
"""
import numpy as np
from scipy.special import logsumexp


def max_cos(S, **_):
    return S.max(1)


def margin(S, k=2, **_):
    # top-1 minus top-k. With ~3 images per identity, k=2 often compares two
    # images of the same person, so a larger k is reported as well.
    top = -np.sort(-S, axis=1)
    return top[:, 0] - top[:, k - 1]


def energy(S, T=0.02, **_):
    # Grows with gallery size by construction (more terms in the sum).
    return T * logsumexp(S / T, axis=1)


def max_softmax(S, T=0.02, **_):
    return np.exp(S.max(1) / T - logsumexp(S / T, axis=1))


def zscore(S, k=50, skip=3, **_):
    # Top-1 standardised against the query's own ranks skip+1..k (NQC-style).
    top = -np.sort(-S, axis=1)[:, :k]
    rest = top[:, skip:]
    return (top[:, 0] - rest.mean(1)) / (rest.std(1) + 1e-6)


def csls(S, S_ref=None, k=10, **_):
    """Cross-domain similarity local scaling (Conneau et al., 2018).

    r_x: mean top-k similarity of each gallery image to a reference text bank
    (captions from disjoint identities). Hub images get penalised.
    r_q: mean top-k similarity of the query to the gallery.
    """
    r_q = -np.sort(-S, axis=1)[:, :k].mean(1)
    if S_ref is None:
        r_x = np.zeros(S.shape[1])
    else:
        r_x = -np.sort(-S_ref, axis=0)[:k].mean(0)
    return (2 * S - r_x[None, :]).max(1) - r_q


def evt(S, k=50, skip=3, **_):
    """Meta-recognition (Scheirer et al., 2011): fit a Weibull to the non-match
    tail (ranks skip+1..k) and return the CDF of the top-1 score."""
    # Vectorised Weibull fit: least squares on the Weibull plot
    # log(-log(1-F)) = c*log(x) - c*log(scale), with median-rank F.
    top = -np.sort(-S, axis=1)[:, :k]
    tail = top[:, skip:][:, ::-1]  # ascending
    lo = tail[:, :1] - 1e-3
    x = np.log(tail - lo)
    n = tail.shape[1]
    F = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    y = np.log(-np.log(1 - F))[None, :]
    xm, ym = x.mean(1, keepdims=True), y.mean()
    c = ((x - xm) * (y - ym)).sum(1) / (((x - xm) ** 2).sum(1) + 1e-12)
    log_scale = (xm[:, 0] - ym / c)
    z = c * (np.log(top[:, 0] - lo[:, 0]) - log_scale)
    # return log(-log(1-CDF)) = z: monotone in the CDF, no saturation at 1.0
    return z


SCORES = {
    "max_cos": max_cos,
    "margin@2": lambda S, **kw: margin(S, k=2),
    "margin@10": lambda S, **kw: margin(S, k=10),
    "energy": energy,
    "max_softmax": max_softmax,
    "zscore": zscore,
    "csls": csls,
    "evt": evt,
}
