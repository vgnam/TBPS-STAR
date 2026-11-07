# Kaggle job: reproduce UATTA's CLIP branch (IRRA_UATTA) in the Pretrain-then-
# Adapt setting, starting from the public MLLM4Text-ReID checkpoint (pretrained
# on LUPerson with MLLM captions only) instead of HAM, whose weights are only on
# Baidu. Targets: RSTPReid and CUHK-PEDES test splits. Also dumps the zero-shot
# embeddings of each target for the RA-TBPS experiments.
import os
import subprocess

W = "/kaggle/working"
T = "/kaggle/temp/uatta"
UATTA_REPO = "https://github.com/nkuzjh/UATTA.git"
CKPT_URL = "https://huggingface.co/datasets/TwT-6/MLLM4Text-ReID/resolve/main/best0.pth"
CLIP_BPE = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
CUHK_URL = "https://huggingface.co/datasets/cjc/CHUK-PEDES/resolve/main/CUHK-PEDES.tar.gz"
RSTP_GDRIVE = "1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X"  # github.com/NjtechCVLab/RSTPReid-Dataset
TARGETS = [("rstp", "ham_rstp_tta", "RSTPReid"), ("cuhk", "ham_cuhk_tta", "CUHK-PEDES")]

DUMP = r'''
import sys, numpy as np, torch, yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from model import build_model
from datasets.bases import ImageDataset, TextDataset
from datasets.build import build_transforms, __factory
args = edict(yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader))
ds = __factory[args.dataset_name](root=args.root_dir)
te = ds.test
tf = build_transforms(img_size=args.img_size, aug=False, is_train=False)
img = DataLoader(ImageDataset(te["image_pids"], te["img_paths"], tf), batch_size=args.test_batch_size)
txt = DataLoader(TextDataset(te["caption_pids"], te["captions"], text_length=args.text_length), batch_size=args.test_batch_size)
model = build_model(args, num_classes=len(ds.train_id_container))
Checkpointer(model).load(f=args.ckpt_dir)
model.to(args.device)
res, r1, sim, tf_, if_, qid, gid, _, _ = do_inference(model, img, txt)
np.savez(sys.argv[2], txt_feat=tf_.cpu().numpy(), img_feat=if_.cpu().numpy(), qid=qid.cpu().numpy(), gid=gid.cpu().numpy())
print("zero-shot", {k: float(v) for k, v in res.items()}, "->", sys.argv[2], flush=True)
'''


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
sh("python -c \"import torch; c=torch.load('checkpoints/mllm4text/best0.pth', map_location='cpu'); "
   "print('checkpoint keys:', list(c)); m=c.get('model', c); print(len(m), 'tensors, e.g.', list(m)[:5])\"")

sh("mkdir -p input/images")
sh(f"curl -sL --fail -o cuhk.tar.gz {CUHK_URL} && tar -xzf cuhk.tar.gz -C input/images && rm cuhk.tar.gz")
sh(f'gdown -q "{RSTP_GDRIVE}" -O rstp.zip && mkdir -p rstp_raw && unzip -q rstp.zip -d rstp_raw && rm rstp.zip')
sh("ln -sfn \"$(dirname \"$(find $PWD/rstp_raw -name data_captions.json | head -1)\")\" input/images/RSTPReid")
sh("ls input/images input/images/RSTPReid | head")

with open("dump_feats.py", "w") as f:
    f.write(DUMP)

import yaml  # noqa: E402

for name, cfg_dir, _ in TARGETS:
    with open(f"tta_configs/{cfg_dir}/uatta_config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["ckpt_dir"] = "checkpoints/mllm4text/best0.pth"
    cfg["output_dir"] = f"{W}/uatta_{name}"
    cfg_path = f"tta_configs/{cfg_dir}/mllm4text_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    sh(f"python -u dump_feats.py {cfg_path} {W}/clip_zeroshot_{name}_test.npz")
    sh(f"python -u tta.py --config_file {cfg_path} --seed 42 --device cuda 2>&1 | tee {W}/uatta_{name}.log")
print("done", flush=True)
