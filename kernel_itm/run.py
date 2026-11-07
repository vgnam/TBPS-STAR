"""Kaggle job: re-score IRRA's top-K candidates with a BLIP ITM cross-encoder.

Inputs: IRRA embeddings from the open-set-tbps-irra kernel output (attached as
a kernel source) and CUHK-PEDES downloaded here. For every query, the K best
images over the *full* split gallery are scored, so any open-set episode can
later read the ITM score of its enrolled candidates without rerunning the GPU.

Output (/kaggle/working/itm_<model>_<split>.npz): idx (nq x K gallery indices,
IRRA order), itm (nq x K match probability), itm_logit (nq x K).
"""
import glob
import json
import os
import subprocess
import time

W = "/kaggle/working"
T = "/kaggle/temp/tbps"
CUHK_URL = "https://huggingface.co/datasets/cjc/CHUK-PEDES/resolve/main/CUHK-PEDES.tar.gz"
MODEL = "Salesforce/blip-itm-base-coco"
K = 20


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.makedirs(T, exist_ok=True)
os.chdir(T)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"curl -sL --fail -o cuhk.tar.gz {CUHK_URL} && mkdir -p data && tar -xzf cuhk.tar.gz -C data && rm cuhk.tar.gz")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import BlipForImageTextRetrieval, BlipProcessor  # noqa: E402

root = "data/CUHK-PEDES"
annos = json.load(open(os.path.join(root, "reid_raw.json")))
emb_dir = os.path.dirname(glob.glob("/kaggle/input/**/irra_cuhk_test.npz", recursive=True)[0])
print("IRRA embeddings from", emb_dir, flush=True)

proc = BlipProcessor.from_pretrained(MODEL)
model = BlipForImageTextRetrieval.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()


@torch.no_grad()
def encode_images(paths, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        px = proc(images=[Image.open(p).convert("RGB") for p in paths[i:i + bs]], return_tensors="pt")["pixel_values"]
        out.append(model.vision_model(pixel_values=px.cuda().half())[0])
    return torch.cat(out)  # n_img x tokens x dim, kept on GPU


@torch.no_grad()
def itm_scores(captions, idx, img_emb, bs=256):
    nq, k = idx.shape
    q_rep = np.repeat(np.arange(nq), k)
    g_flat = idx.reshape(-1)
    logits = np.empty((nq * k, 2), dtype=np.float32)
    t0 = time.time()
    for i in range(0, nq * k, bs):
        qs, gs = q_rep[i:i + bs], g_flat[i:i + bs]
        tok = proc(text=[captions[j] for j in qs], return_tensors="pt", padding=True, truncation=True, max_length=72)
        enc = img_emb[torch.as_tensor(gs, device="cuda")]
        h = model.text_encoder(input_ids=tok["input_ids"].cuda(), attention_mask=tok["attention_mask"].cuda(),
                               encoder_hidden_states=enc,
                               encoder_attention_mask=torch.ones(enc.shape[:2], dtype=torch.long, device="cuda"))[0]
        logits[i:i + bs] = model.itm_head(h[:, 0, :]).float().cpu().numpy()
        if (i // bs) % 100 == 0:
            print(f"  pairs {i + len(qs)}/{nq * k}  {time.time() - t0:.0f}s", flush=True)
    logit = (logits[:, 1] - logits[:, 0]).reshape(nq, k)
    prob = torch.softmax(torch.from_numpy(logits), 1)[:, 1].numpy().reshape(nq, k)
    return prob, logit


for split in ["test", "val"]:
    a = [x for x in annos if x["split"] == split]
    paths = [os.path.join(root, "imgs", x["file_path"]) for x in a]
    caps = [c for x in a for c in x["captions"]]
    e = np.load(os.path.join(emb_dir, f"irra_cuhk_{split}.npz"))
    assert len(caps) == len(e["q"]) and len(paths) == len(e["g"]), "order mismatch with IRRA embeddings"
    idx = np.argsort(-(e["q"] @ e["g"].T), axis=1)[:, :K]
    print(f"{split}: encoding {len(paths)} images", flush=True)
    img_emb = encode_images(paths)
    prob, logit = itm_scores(caps, idx, img_emb)
    np.savez(f"{W}/itm_blipbase_{split}.npz", idx=idx, itm=prob, itm_logit=logit)
    # quick sanity: ITM re-ranking R1 within top-K vs IRRA R1
    qid, gid = e["qid"], e["gid"]
    r1_irra = (gid[idx[:, 0]] == qid).mean()
    r1_itm = (gid[idx[np.arange(len(idx)), logit.argmax(1)]] == qid).mean()
    print(f"{split}: R1 IRRA {100*r1_irra:.2f} | R1 after ITM re-rank of top-{K} {100*r1_itm:.2f}", flush=True)
    del img_emb
    torch.cuda.empty_cache()
print("done", flush=True)
