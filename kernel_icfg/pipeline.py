# Kaggle job: ICFG-PEDES test split on the MLLM4Text-ReID checkpoint, same
# IRRA_UATTA environment as kernel_baselines / kernel_star: TCR (tta_baselines.py)
# and STAR-TBPS step 2 with OT vs greedy pseudo labels (star_adapt.py). Images
# come from the HF zip cjc/ICFG-PEDES (IRRA layout, all 54521 images; the
# Kaggle mirror meetrathi97/icfg-pedes lacks 79% of the test images).
import json
import os
import subprocess

W = "/kaggle/working"
T = "/kaggle/temp/uatta"
UATTA_REPO = "https://github.com/nkuzjh/UATTA.git"
CKPT_URL = "https://huggingface.co/datasets/TwT-6/MLLM4Text-ReID/resolve/main/best0.pth"
CLIP_BPE = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
ICFG_URL = "https://huggingface.co/datasets/cjc/ICFG-PEDES/resolve/main/ICFG-PDES.zip"
STAR_CONFIGS = "hard_keep0.6_tau0.05,greedy_hard_keep0.6_tau0.05"
STAR_SEEDS = "42"
STAR_SRC = None  # filled in by build.py
BASELINES_SRC = None  # filled in by build.py


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
sh("mkdir -p input/images icfg_raw")
sh(f"curl -sL --fail -o icfg.zip {ICFG_URL} && unzip -q icfg.zip -d icfg_raw && rm icfg.zip")
src_dir = os.path.abspath("icfg_raw/ICFG-PEDES")
sh("mkdir -p input/images/ICFG-PEDES")
sh(f"ln -sfn {src_dir}/imgs input/images/ICFG-PEDES/imgs")
# check every test image is there (drop and report any that are not; ICFG has
# one caption per image), and give up if more than a few are missing
with open(f"{src_dir}/ICFG-PEDES.json") as f:
    annos = json.load(f)
test = [a for a in annos if a["split"] == "test"]
missing = [a for a in test if not os.path.exists(f"{src_dir}/imgs/{a['file_path']}")]
kept = [a for a in annos if a["split"] != "test" or os.path.exists(f"{src_dir}/imgs/{a['file_path']}")]
info = dict(test_images=len(test), missing=len(missing), test_ids_before=len({a["id"] for a in test}),
            test_ids_after=len({a["id"] for a in kept if a["split"] == "test"}))
print("ICFG test split", info, flush=True)
with open(f"{W}/icfg_subset.json", "w") as f:
    json.dump(dict(info, missing_files=[a["file_path"] for a in missing]), f, indent=1)
assert len(missing) <= 0.1 * len(test), "too many ICFG test images missing"
with open("input/images/ICFG-PEDES/ICFG-PEDES.json", "w") as f:
    json.dump(kept, f)

for fn, src in (("star_adapt.py", STAR_SRC), ("tta_baselines.py", BASELINES_SRC)):
    with open(fn, "w") as f:
        f.write(json.loads(src))

import yaml  # noqa: E402

with open("tta_configs/ham_icfg_tta/uatta_config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg["ckpt_dir"] = "checkpoints/mllm4text/best0.pth"
cfg_path = "tta_configs/ham_icfg_tta/mllm4text_config.yaml"
with open(cfg_path, "w") as f:
    yaml.dump(cfg, f)
sh(f"python -u star_adapt.py {cfg_path} {W}/star_icfg.json {STAR_CONFIGS} {STAR_SEEDS} 2>&1 | tee {W}/star_icfg.log")
sh(f"python -u tta_baselines.py {cfg_path} {W}/baselines_icfg.json tcr 2>&1 | tee {W}/baselines_icfg.log")
print("done", flush=True)
