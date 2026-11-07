# Kaggle job: UATTA's CLIP branch (IRRA_UATTA) on the ICFG-PEDES test split,
# from the MLLM4Text-ReID checkpoint, same setup as kernel_uatta (RSTP/CUHK)
# with the repo's own ham_icfg_tta config (10 epochs, eval at 0,1,2,4,9).
# Images come from the HF zip cjc/ICFG-PEDES, as in kernel_icfg.
import os
import subprocess

W = "/kaggle/working"
T = "/kaggle/temp/uatta"
UATTA_REPO = "https://github.com/nkuzjh/UATTA.git"
CKPT_URL = "https://huggingface.co/datasets/TwT-6/MLLM4Text-ReID/resolve/main/best0.pth"
CLIP_BPE = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
ICFG_URL = "https://huggingface.co/datasets/cjc/ICFG-PEDES/resolve/main/ICFG-PDES.zip"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.makedirs(T, exist_ok=True)
os.chdir(T)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("pip install -q easydict prettytable ftfy regex ruamel.yaml peft tensorboard")
sh(f"git clone -q --depth 1 {UATTA_REPO} uatta")
os.chdir("uatta/IRRA_UATTA")
sh(f"curl -sL --fail -o data/bpe_simple_vocab_16e6.txt.gz {CLIP_BPE}")
sh(f"mkdir -p checkpoints/mllm4text && curl -sL --fail -o checkpoints/mllm4text/best0.pth {CKPT_URL}")
sh("mkdir -p input/images")
sh(f"curl -sL --fail -o icfg.zip {ICFG_URL} && unzip -q icfg.zip -d input/images && rm icfg.zip")
sh("ls input/images/ICFG-PEDES")

# Memory patches, results unchanged. v1 was OOM-killed at the first TTA eval:
# Evaluator keeps every gallery image tensor (19848 x 3x384x128 fp32 = 11.7 GB)
# on top of the copy the TTA dataset already holds, and IRRA's rank() argsorts
# the 19848 x 19848 matrix on the CPU in one go. So (1) once the TTA dataset
# exists, evaluations stop collecting images (tta.py does not use them after),
# and (2) rank() is redefined to compute the same CMC/mAP/mINP in query chunks
# on the GPU (the later definition wins at call time).
RANK_CHUNKED = r'''

KEEP_IMGS = True


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True, chunk=1024):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q_pids, g_pids = q_pids.to(dev), g_pids.to(dev)
    cmc, ap, inp, top = [], [], [], []
    for s in range(0, len(similarity), chunk):
        idx = similarity[s:s + chunk].to(dev).argsort(1, descending=True)
        match = g_pids[idx] == q_pids[s:s + chunk, None]
        hits = match.cumsum(1)
        cmc.append(hits[:, :max_rank] > 0)
        pos = torch.arange(1, match.shape[1] + 1, device=dev)
        ap.append(((hits / pos) * match).sum(1) / match.sum(1))
        last = match.shape[1] - 1 - match.flip(1).float().argmax(1)
        inp.append(hits.gather(1, last[:, None])[:, 0] / (last + 1.0))
        top.append(idx[:, :max_rank].cpu())
    all_cmc = torch.cat(cmc).float().mean(0).cpu() * 100
    if not get_mAP:
        return all_cmc, torch.cat(top)
    return all_cmc, torch.cat(ap).mean().cpu() * 100, torch.cat(inp).mean().cpu() * 100, torch.cat(top)
'''
with open("utils/metrics.py") as f:
    m = f.read()
assert m.count("            imgs.append(img.cpu())") == 1
m = m.replace("            imgs.append(img.cpu())", "            if KEEP_IMGS:\n                imgs.append(img.cpu())")
with open("utils/metrics.py", "w") as f:
    f.write(m + RANK_CHUNKED)
with open("tta.py") as f:
    t = f.read()
anchor = '        print(f"     tta_dataset: {len(tta_dataset)}")\n'
assert t.count(anchor) == 1
t = t.replace(anchor, anchor + "        import utils.metrics as _metrics\n"
              "        _metrics.KEEP_IMGS = False\n        images = captions = None\n")
with open("tta.py", "w") as f:
    f.write(t)

import yaml  # noqa: E402

with open("tta_configs/ham_icfg_tta/uatta_config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg["ckpt_dir"] = "checkpoints/mllm4text/best0.pth"
cfg["output_dir"] = f"{W}/uatta_icfg"
cfg_path = "tta_configs/ham_icfg_tta/mllm4text_config.yaml"
with open(cfg_path, "w") as f:
    yaml.dump(cfg, f)
os.makedirs(cfg["output_dir"], exist_ok=True)
sh(f"python -u tta.py --config_file {cfg_path} --seed 42 --device cuda 2>&1 | tee {W}/uatta_icfg.log")
print("done", flush=True)
