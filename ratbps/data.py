"""Split readers for CUHK-PEDES, ICFG-PEDES and RSTPReid annotation files.

All three ship one JSON list whose entries carry an identity, an image path,
a list of captions and a split name; only the key for the path differs.
"""
import json
import os
import re

ANNOTATION = {
    "cuhk": "reid_raw.json",
    "icfg": "ICFG-PEDES.json",
    "rstp": "data_captions.json",
}

# Per-dataset image statistics used by APTM (dataset/__init__.py).
NORM = {
    "cuhk": ((0.38901278, 0.3651612, 0.34836376), (0.24344306, 0.23738699, 0.23368555)),
    "icfg": ((0.30941582, 0.28956893, 0.30347288), (0.25849792, 0.24547698, 0.2366199)),
    "rstp": ((0.27722597, 0.26065794, 0.3036557), (0.2609547, 0.2508087, 0.25293276)),
}


ALIASES = {"icfg": ("ICFG-PEDES.json", "ICFG_PEDES.json")}  # releases differ in the file name


def annotation_path(root, fmt):
    for name in ALIASES.get(fmt, (ANNOTATION[fmt],)):
        if os.path.exists(os.path.join(root, name)):
            return os.path.join(root, name)
    return os.path.join(root, ANNOTATION[fmt])


def clean_caption(caption, fmt, max_words=56):
    # Mirrors APTM's pre_caption: ICFG/RSTP are lower-cased and stripped of
    # punctuation, CUHK captions are kept as written.
    if fmt in ("icfg", "rstp"):
        caption = re.sub(r"[^0-9a-z]+", " ", caption.lower())
    words = caption.split()
    return " ".join(words[:max_words])


def load_split(root, fmt, split):
    """Returns (image_paths, image_pids, captions, caption_image_index).

    caption_image_index[t] is the gallery index of the image caption t was
    written for. It is a label: only source-domain code may use it.
    """
    with open(annotation_path(root, fmt), encoding="utf-8") as f:
        ann = json.load(f)
    images, pids, captions, cap_img = [], [], [], []
    for a in ann:
        if a["split"] != split:
            continue
        rel = a.get("file_path", a.get("img_path"))
        images.append(os.path.join(root, "imgs", rel))
        pids.append(int(a["id"]))
        for c in a["captions"]:
            captions.append(clean_caption(c, fmt))
            cap_img.append(len(images) - 1)
    if not images:
        raise ValueError(f"no entries with split={split!r} in {fmt} annotations under {root}")
    return images, pids, captions, cap_img
