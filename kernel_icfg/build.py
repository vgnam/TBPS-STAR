"""Write the ICFG-PEDES kernel (kernel_icfg/out) from kernel_star/star_adapt.py
and kernel_baselines/tta_baselines.py.

usage: python kernel_icfg/build.py && kaggle kernels push -p kernel_icfg/out
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def src(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.dumps(f.read())


with open(os.path.join(HERE, "pipeline.py")) as f:
    run = f.read()
run = run.replace("STAR_SRC = None  # filled in by build.py", f"STAR_SRC = {src('kernel_star/star_adapt.py')!r}")
run = run.replace("BASELINES_SRC = None  # filled in by build.py",
                  f"BASELINES_SRC = {src('kernel_baselines/tta_baselines.py')!r}")
with open(os.path.join(OUT, "run.py"), "w") as f:
    f.write(run)

meta = {
    "id": "nammmmmm/star-icfg-mllm4text",
    "title": "star-icfg-mllm4text",
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
