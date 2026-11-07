"""Bundle opentbps/*.py into a single Kaggle script (kernel/out/run.py).

The Kaggle run downloads CUHK-PEDES and the IRRA checkpoint itself, so nothing
but code is uploaded from this machine.

usage: python kernel/build.py && kaggle kernels push -p kernel/out
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")

src = {os.path.basename(p): open(p, encoding="utf-8").read()
       for p in glob.glob(os.path.join(ROOT, "opentbps", "*.py"))}

body = open(os.path.join(HERE, "pipeline.py"), encoding="utf-8").read()
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "run.py"), "w", encoding="utf-8") as f:
    f.write(f"OPENTBPS_SRC = {json.dumps(src)}\n\n")
    f.write(body)

meta = {
    "id": "nammmmmm/open-set-tbps-irra",
    "title": "open-set-tbps-irra",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
with open(os.path.join(OUT, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("wrote", OUT, "modules:", sorted(src))
