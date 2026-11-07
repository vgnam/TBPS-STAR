# Kaggle pipeline body; build.py prepends RATBPS_SRC (the ratbps package).
# Source: APTM fine-tuned on CUHK-PEDES. Targets: CUHK test (in-domain
# reference) plus every cross-domain dataset attached as a Kaggle input
# (ICFG-PEDES.json or RSTPReid's data_captions.json, with an imgs/ folder).
import glob
import os
import subprocess

W = "/kaggle/working"
T = "/kaggle/temp/tbps"
CUHK_URL = "https://huggingface.co/datasets/cjc/CHUK-PEDES/resolve/main/CUHK-PEDES.tar.gz"
APTM_REPO = "https://github.com/Shuyu-XJTU/APTM.git"
APTM_FT_CUHK = "1dokRhjYL71VbzFfLfyyDmh7MXUOyBySf"  # ft_cuhk.zip in the APTM release folder
RSTP_GDRIVE = "1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X"  # RSTPReid, from github.com/NjtechCVLab/RSTPReid-Dataset
TOPN = 128  # ITC candidates re-ranked by ITM, as in APTM's evaluation (k_test)


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.makedirs(T, exist_ok=True)
os.chdir(T)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
# APTM's bert.py imports helpers that transformers 5 removed; 4.55 is the
# version verified locally.
sh('pip install -q gdown "transformers==4.55.0"')

sh(f"curl -sL --fail -o cuhk.tar.gz {CUHK_URL} && mkdir -p data && tar -xzf cuhk.tar.gz -C data && rm cuhk.tar.gz")
sh(f"git clone -q --depth 1 {APTM_REPO} third_party/APTM")
sh(f'gdown -q "{APTM_FT_CUHK}" -O ft_cuhk.zip && mkdir -p checkpoints/aptm_ft_cuhk '
   f'&& unzip -q ft_cuhk.zip -d checkpoints/aptm_ft_cuhk && rm ft_cuhk.zip')
# Cross-domain target. ICFG-PEDES is only given out on request, so it is used
# when attached as a Kaggle input.
sh(f'gdown -q "{RSTP_GDRIVE}" -O rstp.zip && mkdir -p data/rstp && unzip -q rstp.zip -d data/rstp && rm rstp.zip')
ckpts = sorted(glob.glob("checkpoints/aptm_ft_cuhk/**/*.pth", recursive=True))
print("checkpoints found:", ckpts, flush=True)
ckpt = next((c for c in ckpts if "best" in os.path.basename(c)), ckpts[0])

os.makedirs("ratbps", exist_ok=True)
for name, code in RATBPS_SRC.items():  # noqa: F821
    with open(os.path.join("ratbps", name), "w") as f:
        f.write(code)
os.environ["USE_TF"] = "0"


def extract(fmt, root, split, out, topn, amp, gt=False):
    sh(f"python -u -m ratbps.extract_aptm --fmt {fmt} --root {root} --split {split} --ckpt {ckpt} "
       f"--aptm third_party/APTM --topn {topn} --out {out}{' --amp' if amp else ''}{' --gt' if gt else ''}")


# 1) Baseline check: does the cache reproduce APTM's CUHK number, and does
#    fp16 cost accuracy?
# Kernel v4 measured fp16 vs fp32 on CUHK test (top-128): ITM R1 71.73 vs
# 71.75, so fp16 is used throughout.
amp = True
extract("cuhk", "data/CUHK-PEDES", "test", "emb/cuhk_test_fp16.npz", TOPN, amp)
sh(f"python -u -m ratbps.check_baseline emb/cuhk_test_fp16.npz --json {W}/baseline.json")

# 2) Source cache (memory + training) and targets
extract("cuhk", "data/CUHK-PEDES", "train", "emb/cuhk_train.npz", 16, amp, gt=True)
targets = ["cuhk_test=emb/cuhk_test_fp16.npz"]
for fmt, anns in (("icfg", ("ICFG-PEDES.json", "ICFG_PEDES.json")), ("rstp", ("data_captions.json",))):
    found = [p for a in anns for base in ("/kaggle/input", "data")
             for p in glob.glob(f"{base}/**/{a}", recursive=True)]
    if not found:
        print(f"no {'/'.join(anns)} found: skipping {fmt}", flush=True)
        continue
    root = os.path.dirname(found[0])
    extract(fmt, root, "test", f"emb/{fmt}_test.npz", TOPN, amp)
    targets.append(f"{fmt}_test=emb/{fmt}_test.npz")

sh("python -u -m ratbps.check_baseline " + " ".join(t.split("=")[1] for t in targets[1:]) +
   f" --json {W}/baseline_targets.json" if len(targets) > 1 else "true")
sh("python -u -m ratbps.run --src emb/cuhk_train.npz " + " ".join(f"--target {t}" for t in targets) +
   f" --out {W}/ratbps")
print("done", flush=True)
