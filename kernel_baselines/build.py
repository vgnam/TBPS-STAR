"""Write the TTA-baselines kernel (kernel_baselines/out).

usage: python kernel_baselines/build.py && kaggle kernels push -p kernel_baselines/out
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(HERE, "tta_baselines.py")) as f:
    src = json.dumps(f.read())
with open(os.path.join(HERE, "pipeline.py")) as f:
    run = f.read().replace("BASELINES_SRC = None  # filled in by build.py", f"BASELINES_SRC = {src!r}")
with open(os.path.join(OUT, "run.py"), "w") as f:
    f.write(run)

meta = {
    "id": "nammmmmm/tta-baselines-mllm4text",
    "title": "tta-baselines-mllm4text",
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
print("wrote", OUT)
