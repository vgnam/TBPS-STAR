# Kaggle pipeline body; build.py prepends OPENTBPS_SRC (the opentbps package).
import os
import shutil
import subprocess

W = "/kaggle/working"
T = "/kaggle/temp/tbps"
CUHK_URL = "https://huggingface.co/datasets/cjc/CHUK-PEDES/resolve/main/CUHK-PEDES.tar.gz"
IRRA_CKPT = "https://drive.google.com/uc?id=1OBhFhpZpltRMZ88K6ceNUv4vZgevsFCW"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.makedirs(T, exist_ok=True)
os.chdir(T)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("pip install -q gdown ftfy regex easydict prettytable open_clip_torch")

sh(f"curl -sL --fail -o cuhk.tar.gz {CUHK_URL} && mkdir -p data && tar -xzf cuhk.tar.gz -C data && rm cuhk.tar.gz")
sh(f'gdown -q "{IRRA_CKPT}" -O irra.zip && mkdir -p checkpoints && unzip -q irra.zip -d checkpoints/irra_cuhk && rm irra.zip')
sh("git clone -q --depth 1 https://github.com/anosorae/IRRA.git third_party/IRRA")

os.makedirs("opentbps", exist_ok=True)
for name, code in OPENTBPS_SRC.items():  # noqa: F821
    with open(os.path.join("opentbps", name), "w") as f:
        f.write(code)

sh("python -u -m opentbps.extract_irra --root data/CUHK-PEDES --ckpt checkpoints/irra_cuhk --out emb/irra_cuhk")
sh("python -c \"import numpy as np; from opentbps.metrics import rank1_map; d=np.load('emb/irra_cuhk_test.npz'); "
   "r,m=rank1_map(d['q']@d['g'].T,d['qid'],d['gid']); print('IRRA full test closed-set R1 %.2f mAP %.2f'%(100*r,100*m))\"")
sh(f"python -u -m opentbps.run_openset --emb emb/irra_cuhk --out {W}/irra_cuhk.json 2>&1 | tee {W}/irra_cuhk.txt")

for split in ["test", "val"]:
    shutil.copy(f"emb/irra_cuhk_{split}.npz", W)
print("done", flush=True)
