"""Evidence memory B (RA-VLA's context buffer, rebuilt for person search).

Each entry is one image-caption pair with its cached APTM features:
  txt, img   ITC embeddings (256-d) of the caption and the image
  cross      fused [CLS] state of the pair (768-d), the value g grounds on
  pid        identity of the image; used ONLY as a training signal on the
             source domain (and for logging pseudo-label precision)
  owner      (domain, caption index): lets a query skip its own entry
  pseudo     1 if the pair was not annotated but agreed on by the model

Source entries are annotated pairs. Target entries are pseudo-pairs kept only
when both retrieval directions agree, as in UATTA: the caption's best ITM
candidate also ranks that caption among its top-r captions under ITC.
"""
import numpy as np

FIELDS = ("txt", "img", "cross", "pid", "owner", "pseudo")
DOMAIN_STRIDE = 10_000_000


def owner_ids(domain, idx):
    return domain * DOMAIN_STRIDE + np.asarray(idx, np.int64)


def labelled(d, domain, captions=None):
    """Annotated (caption, own image) pairs of a source cache."""
    t = np.arange(len(d["qid"])) if captions is None else np.asarray(captions)
    return dict(txt=d["txt_feat"][t], img=d["img_feat"][d["q2img"][t]], cross=d["gt_cross"][t],
                pid=d["qid"][t], owner=owner_ids(domain, t), pseudo=np.zeros(len(t), np.int8))


def pseudo(d, domain, captions=None, r=5, min_logodds=0.0):
    """Pseudo-pairs agreed on in both directions. Uses no labels."""
    t = np.arange(len(d["cand"])) if captions is None else np.asarray(captions)
    j = d["cand_itm"][t].argmax(1)
    img = d["cand"][t, j]
    agree = (d["img_top_txt"][img, :r] == t[:, None]).any(1)
    keep = agree & (d["cand_itm"][t, j] > min_logodds)
    t, j, img = t[keep], j[keep], img[keep]
    return dict(txt=d["txt_feat"][t], img=d["img_feat"][img], cross=d["cand_cross"][t, j],
                pid=d["gid"][img], owner=owner_ids(domain, t), pseudo=np.ones(len(t), np.int8))


def concat(*parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


def describe(mem, true_pid_of_owner=None):
    """One-line summary. Pseudo precision needs labels and is for logging only."""
    n_ps = int(mem["pseudo"].sum())
    s = f"memory: {len(mem['pid'])} entries ({len(mem['pid']) - n_ps} annotated, {n_ps} pseudo)"
    if true_pid_of_owner is not None and n_ps:
        ps = mem["pseudo"] == 1
        s += f", pseudo-pair precision {100 * np.mean(true_pid_of_owner(mem['owner'][ps]) == mem['pid'][ps]):.1f}%"
    return s
