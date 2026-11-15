"""
Push / check / download the research experiments on Kaggle (account samiulseikh only).

    uv run python experiments/kaggle_exp.py dataset                       # upload data/4_features/wind_cpcb.npz
    uv run python experiments/kaggle_exp.py push wind [--smoke]           # teacher wind experiment (GPU)
    uv run python experiments/kaggle_exp.py push probe                    # model-free probes (CPU)
    uv run python experiments/kaggle_exp.py status wind
    uv run python experiments/kaggle_exp.py get wind                      # -> data/6_experiments/wind/

Every experiment is a notebook built by experiments/exp_<name>.py (it re-uses notebook 04's own cells).
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "samiulseikh"
STAGING = ROOT / "kaggle" / "_staging"


# the samiulseikh token (KAGGLE_API2 in .env; KAGGLE_API_TOKEN also accepted)
def load_token():
    env = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if "=" in line)
    os.environ["KAGGLE_API_TOKEN"] = (env.get("KAGGLE_API_TOKEN") or env["KAGGLE_API2"]).strip()


def kaggle(*args, check=True):
    print(">", "kaggle", *args)
    return subprocess.run(["kaggle", *args], check=check)


# small private dataset with the measured CPCB wind (the big grids are already in aqi-features)
def dataset():
    folder = STAGING / "aqi-wind-cpcb"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy2(ROOT / "data/4_features/wind_cpcb.npz", folder)
    meta = {"title": "AQI measured CPCB wind", "id": f"{USER}/aqi-wind-cpcb", "licenses": [{"name": "CC0-1.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{USER}/aqi-wind-cpcb"], capture_output=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(folder), "-m", "update")
    else:
        kaggle("datasets", "create", "-p", str(folder))


# build the experiment notebook and push it
def push(name, smoke=False):
    exp = importlib.import_module(f"exp_{name}")
    folder = STAGING / f"aqi-exp-{name}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    nb = exp.build(smoke)
    (folder / "exp.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{USER}/aqi-exp-{name}", "title": f"aqi-exp-{name}", "code_file": "exp.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": exp.GPU, "enable_internet": False,
            "dataset_sources": exp.DATASETS, "kernel_sources": getattr(exp, "KERNELS", [])}
    if exp.GPU:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    kaggle("kernels", "push", "-p", str(folder))


def get(name):
    out = ROOT / "data" / "6_experiments" / name
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{USER}/aqi-exp-{name}", "-p", str(out))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    load_token()
    cmd = sys.argv[1]
    if cmd == "dataset":
        dataset()
    elif cmd == "push":
        push(sys.argv[2], smoke="--smoke" in sys.argv)
    elif cmd == "status":
        kaggle("kernels", "status", f"{USER}/aqi-exp-{sys.argv[2]}")
    elif cmd == "get":
        get(sys.argv[2])
