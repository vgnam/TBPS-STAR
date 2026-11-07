# Kaggle job: Tent / SAR / READ / TCR (TCR's code release, t2i online protocol)
# on the MLLM4Text-ReID checkpoint, RSTPReid and CUHK-PEDES test splits, in the
# same IRRA_UATTA environment as the UATTA reproduction (kernel_uatta).
import json
import os
import subprocess

W = "/kaggle/working"
T = "/kaggle/temp/uatta"
UATTA_REPO = "https://github.com/nkuzjh/UATTA.git"
CKPT_URL = "https://huggingface.co/datasets/TwT-6/MLLM4Text-ReID/resolve/main/best0.pth"
CLIP_BPE = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
CUHK_URL = "https://huggingface.co/datasets/cjc/CHUK-PEDES/resolve/main/CUHK-PEDES.tar.gz"
RSTP_GDRIVE = "1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X"  # github.com/NjtechCVLab/RSTPReid-Dataset
TARGETS = [("rstp", "ham_rstp_tta"), ("cuhk", "ham_cuhk_tta")]
BASELINES_SRC = None  # filled in by build.py


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.makedirs(T, exist_ok=True)
os.chdir(T)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("pip install -q gdown easydict prettytable ftfy regex ruamel.yaml peft tensorboard")
sh(f"git clone -q --depth 1 {UATTA_REPO} uatta")
os.chdir("uatta/IRRA_UATTA")
sh(f"curl -sL --fail -o data/bpe_simple_vocab_16e6.txt.gz {CLIP_BPE}")
sh(f"mkdir -p checkpoints/mllm4text && curl -sL --fail -o checkpoints/mllm4text/best0.pth {CKPT_URL}")
sh("mkdir -p input/images")
sh(f"curl -sL --fail -o cuhk.tar.gz {CUHK_URL} && tar -xzf cuhk.tar.gz -C input/images && rm cuhk.tar.gz")
sh(f'gdown -q "{RSTP_GDRIVE}" -O rstp.zip && mkdir -p rstp_raw && unzip -q rstp.zip -d rstp_raw && rm rstp.zip')
sh("ln -sfn \"$(dirname \"$(find $PWD/rstp_raw -name data_captions.json | head -1)\")\" input/images/RSTPReid")

with open("tta_baselines.py", "w") as f:
    f.write(json.loads(BASELINES_SRC))

import yaml  # noqa: E402

for name, cfg_dir in TARGETS:
    with open(f"tta_configs/{cfg_dir}/uatta_config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["ckpt_dir"] = "checkpoints/mllm4text/best0.pth"
    cfg_path = f"tta_configs/{cfg_dir}/mllm4text_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    sh(f"python -u tta_baselines.py {cfg_path} {W}/baselines_{name}.json 2>&1 | tee {W}/baselines_{name}.log")
print("done", flush=True)
