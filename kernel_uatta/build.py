"""Write the UATTA reproduction kernel (kernel_uatta/out).

usage: python kernel_uatta/build.py && kaggle kernels push -p kernel_uatta/out
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
shutil.copy(os.path.join(HERE, "pipeline.py"), os.path.join(OUT, "run.py"))

meta = {
    "id": "nammmmmm/uatta-mllm4text",
    "title": "uatta-mllm4text",
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
