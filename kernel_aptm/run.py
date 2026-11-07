"""Kaggle job: APTM (X-VLM-style Swin-B + BERT, fine-tuned on CUHK-PEDES) as a
second backbone *and* a TBPS-trained ITM verifier.

For each split it saves, in the same formats as the IRRA/BLIP kernels:
  aptm_cuhk_<split>.npz  q, g (L2-normalized 256-d ITC features), qid, gid
  itm_aptm_<split>.npz   idx (nq x K, APTM top-K over the full split gallery),
                         itm (match probability), itm_logit (logit1 - logit0)
Checkpoint: APTM release (Shuyu-XJTU/APTM), Google Drive ft_cuhk.zip.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

W = "/kaggle/working"
T = "/kaggle/temp/tbps"
CUHK_URL = "https://huggingface.co/datasets/cjc/CHUK-PEDES/resolve/main/CUHK-PEDES.tar.gz"
FT_CUHK_ID = "1dokRhjYL71VbzFfLfyyDmh7MXUOyBySf"
K = 64


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.makedirs(T, exist_ok=True)
os.chdir(T)
VENV_PY = os.path.join(T, "venv/bin/python")
if os.environ.get("APTM_STAGE") != "2":
    # Kaggle's Python (3.13) is too new for the transformers APTM was written
    # against, so the model runs in a Python 3.11 venv; this stage only sets up.
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh("pip install -q gdown uv")
    sh(f"uv venv -q -p 3.11 {T}/venv")
    sh(f"uv pip install -q -p {VENV_PY} torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121")
    sh(f"uv pip install -q -p {VENV_PY} transformers==4.30.2 timm==0.4.12 'numpy<2' scipy pyyaml pillow")
    sh(f"curl -sL --fail -o cuhk.tar.gz {CUHK_URL} && mkdir -p data && tar -xzf cuhk.tar.gz -C data && rm cuhk.tar.gz")
    sh("git clone -q --depth 1 https://github.com/Shuyu-XJTU/APTM aptm")
    sh(f"gdown -q {FT_CUHK_ID} -O ft_cuhk.zip && unzip -q -o ft_cuhk.zip -d ckpt && rm ft_cuhk.zip")
    subprocess.run([VENV_PY, os.path.abspath(__file__)], env={**os.environ, "APTM_STAGE": "2"}, check=True)
    sys.exit(0)
ckpts = sorted(glob.glob("ckpt/**/*.pth", recursive=True) + glob.glob("ckpt/**/*.th", recursive=True))
print("checkpoint files:", ckpts, flush=True)
ckpt = next((c for c in ckpts if "best" in os.path.basename(c)), ckpts[0])

# Patch APTM for current libraries: docstring decorators -> no-ops, and build
# BERT from config (weights come from the APTM checkpoint, not bert-base).
bert_py = "aptm/models/bert.py"
src = open(bert_py).read()
src = re.sub(r"from transformers\.file_utils import \([^)]*\)",
             "from transformers.utils import ModelOutput\n"
             "def _noop(*a, **k):\n    return lambda f: f\n"
             "add_code_sample_docstrings = add_start_docstrings = add_start_docstrings_to_model_forward = "
             "replace_return_docstrings = _noop", src)
open(bert_py, "w").write(src)
aptm_py = "aptm/models/aptm.py"
src = open(aptm_py).read()
src = src.replace("BertForMaskedLM.from_pretrained(config['text_encoder'], config=config_text,\n"
                  "                                                            output_loading_info=True)",
                  "BertForMaskedLM(config_text), {'missing_keys': []}")
assert "BertForMaskedLM(config_text)" in src, "aptm.py patch did not apply"
open(aptm_py, "w").write(src)

sys.path.insert(0, os.path.join(T, "aptm"))
os.chdir(os.path.join(T, "aptm"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402
from transformers import BertTokenizer  # noqa: E402

from models.model_retrieval import APTM_Retrieval  # noqa: E402

config = yaml.safe_load(open("configs/Retrieval_cuhk.yaml"))
model = APTM_Retrieval(config=config)
sd = torch.load(os.path.join(T, ckpt), map_location="cpu", weights_only=False)
sd = sd.get("model", sd)
msg = model.load_state_dict(sd, strict=False)
print(f"loaded {ckpt}: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected", flush=True)
print("  missing:", msg.missing_keys[:20], "\n  unexpected:", msg.unexpected_keys[:20], flush=True)
model = model.cuda().eval()
tok = BertTokenizer.from_pretrained("bert-base-uncased")
tf = transforms.Compose([
    transforms.Resize((config["h"], config["w"]), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.38901278, 0.3651612, 0.34836376), (0.24344306, 0.23738699, 0.23368555)),  # cuhk_norm
])


class Imgs(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return tf(Image.open(self.paths[i]).convert("RGB"))


def autocast():
    return torch.autocast("cuda", dtype=torch.float16)


@torch.no_grad()
def encode_images(paths):
    embeds, feats = [], []
    for x in torch.utils.data.DataLoader(Imgs(paths), batch_size=64, num_workers=4):
        with autocast():
            e, _ = model.get_vision_embeds(x.cuda())
            feats.append(F.normalize(model.vision_proj(e[:, 0, :]).float(), dim=-1))
        embeds.append(e.half())
    return torch.cat(embeds), torch.cat(feats)


@torch.no_grad()
def encode_texts(caps, bs=256):
    embeds, atts, feats = [], [], []
    for i in range(0, len(caps), bs):
        t = tok(caps[i:i + bs], padding="max_length", truncation=True, max_length=config["max_tokens"],
                return_tensors="pt").to("cuda")
        with autocast():
            e = model.get_text_embeds(t.input_ids, t.attention_mask)
            feats.append(F.normalize(model.text_proj(e[:, 0, :]).float(), dim=-1))
        embeds.append(e.half())
        atts.append(t.attention_mask)
    return torch.cat(embeds), torch.cat(atts), torch.cat(feats)


@torch.no_grad()
def itm_scores(idx, img_e, txt_e, txt_a, qb=4):
    nq, k = idx.shape
    logits = torch.empty(nq, k, 2)
    idx_t = torch.as_tensor(idx, device="cuda")
    t0 = time.time()
    for i in range(0, nq, qb):
        q = torch.arange(i, min(nq, i + qb), device="cuda")
        enc = img_e[idx_t[q].reshape(-1)]
        with autocast():
            out = model.get_cross_embeds(enc, torch.ones(enc.shape[:2], dtype=torch.long, device="cuda"),
                                         text_embeds=txt_e[q].repeat_interleave(k, 0),
                                         text_atts=txt_a[q].repeat_interleave(k, 0))[:, 0, :]
            logits[i:i + len(q)] = model.itm_head(out).float().reshape(len(q), k, 2).cpu()
        if (i // qb) % 250 == 0:
            print(f"  queries {i + len(q)}/{nq}  {time.time() - t0:.0f}s", flush=True)
    prob = torch.softmax(logits, -1)[..., 1].numpy()
    return prob, (logits[..., 1] - logits[..., 0]).numpy()


root = os.path.join(T, "data/CUHK-PEDES")
annos = json.load(open(os.path.join(root, "reid_raw.json")))
for split in ["test", "val"]:
    a = [x for x in annos if x["split"] == split]
    paths = [os.path.join(root, "imgs", x["file_path"]) for x in a]
    gid = np.array([x["id"] for x in a])
    caps = [" ".join(c.split()[:config["max_words"]]) for x in a for c in x["captions"]]
    qid = np.array([x["id"] for x in a for _ in x["captions"]])
    print(f"{split}: {len(paths)} images, {len(caps)} captions", flush=True)
    img_e, img_f = encode_images(paths)
    txt_e, txt_a, txt_f = encode_texts(caps)
    q, g = txt_f.cpu().numpy(), img_f.cpu().numpy()
    np.savez(f"{W}/aptm_cuhk_{split}.npz", q=q, g=g, qid=qid, gid=gid)
    idx = np.argsort(-(q @ g.T), axis=1)[:, :K]
    prob, logit = itm_scores(idx, img_e, txt_e, txt_a)
    np.savez(f"{W}/itm_aptm_{split}.npz", idx=idx, itm=prob, itm_logit=logit)
    r1_itc = (gid[idx[:, 0]] == qid).mean()
    r1_itm = (gid[idx[np.arange(len(idx)), logit.argmax(1)]] == qid).mean()
    print(f"{split}: R1 ITC {100*r1_itc:.2f} | R1 after ITM re-rank of top-{K} {100*r1_itm:.2f} "
          f"(paper, test: 76.53)", flush=True)
    del img_e, txt_e
    torch.cuda.empty_cache()
print("done", flush=True)
