"""
Run notebook 04 (teacher, then student) on a free Kaggle GPU (T4), from the laptop.

    uv run python kaggle/kaggle_run.py upload                 # 1. send data/4_features (grids + windows) as a private dataset
    uv run python kaggle/kaggle_run.py train --smoke          # 2a. tiny test run (a few minutes: checks the setup)
    uv run python kaggle/kaggle_run.py train                  # 2b. full run: teacher + student (one session, max 12 h)
    uv run python kaggle/kaggle_run.py train --variants T0,T3 #     only some teacher runs (if 12 h is not enough)
    uv run python kaggle/kaggle_run.py status                 #     running / complete / error
    uv run python kaggle/kaggle_run.py download               # 3. results -> data/5_models/kaggle/,
                                                              #    and firmware/aqi_node/student_model.h for the ESP32

You can also skip this script: upload notebooks/04_train.ipynb on kaggle.com, add the dataset
samiulseikh/aqi-features as input, choose GPU T4, and "Save Version" (= commit).

Needs KAGGLE_API_TOKEN in .env (kaggle.com -> Settings -> API -> Create New Token).
Why GPU and not TPU: the teacher is PyTorch; on a TPU PyTorch needs torch_xla and the graph code
(eigen-decomposition, masks) is slow or unsupported there. A T4 GPU runs it as it is.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "samiulseikh"
DATASET = "aqi-features"
KERNEL = "aqi-train"
NOTEBOOK = "04_train.ipynb"
STAGING = ROOT / "kaggle" / "_staging"                     # temporary upload folders (git-ignored)


# read the Kaggle token from .env
def load_token():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("KAGGLE_API_TOKEN="):
            os.environ["KAGGLE_API_TOKEN"] = line.split("=", 1)[1].strip()


# run one kaggle CLI command and show it
def kaggle(*args):
    print(">", "kaggle", *args)
    subprocess.run(["kaggle", *args], check=True)


# upload the grids + window lists as a private dataset (new version if it already exists)
def upload():
    folder = STAGING / DATASET
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    feat = ROOT / "data" / "4_features"
    for f in sorted(feat.glob("grid_u*.npz")) + sorted(feat.glob("*windows*.parquet")):
        shutil.copy2(f, folder / f.name)
    meta = {"title": "AQI sensor features and teacher grids", "id": f"{USER}/{DATASET}", "licenses": [{"name": "CC0-1.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{USER}/{DATASET}"], capture_output=True, check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(folder), "-m", "update", "--dir-mode", "skip")
    else:
        kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "skip")      # private by default


# push the notebook to Kaggle and start it on a T4 GPU
def train(smoke=False, variants=None):
    folder = STAGING / KERNEL
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    text = (ROOT / "notebooks" / NOTEBOOK).read_text(encoding="utf-8")
    if smoke:                     # change the settings of the uploaded copy only
        text = text.replace("SMOKE = not ON_KAGGLE", "SMOKE = True")
    if variants:
        text = text.replace("VARIANTS = list(RUNS)", "VARIANTS = " + json.dumps(variants).replace('"', '\\"'))
    (folder / NOTEBOOK).write_text(text, encoding="utf-8")
    meta = {"id": f"{USER}/{KERNEL}", "title": KERNEL, "code_file": NOTEBOOK, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{USER}/{DATASET}"]}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    kaggle("kernels", "push", "-p", str(folder))


# download the results; the student's C header goes right beside the ESP32 sketch
def download():
    out = ROOT / "data" / "5_models" / "kaggle"
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{USER}/{KERNEL}", "-p", str(out))
    header = next(out.rglob("student_model.h"), None)
    if header:
        shutil.copy2(header, ROOT / "firmware" / "aqi_node" / "student_model.h")
        print("copied student_model.h -> firmware/aqi_node/")


if __name__ == "__main__":
    load_token()
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "upload":
        upload()
    elif command == "train":
        variants = sys.argv[sys.argv.index("--variants") + 1].split(",") if "--variants" in sys.argv else None
        train(smoke="--smoke" in sys.argv, variants=variants)
    elif command == "status":
        kaggle("kernels", "status", f"{USER}/{KERNEL}")
    elif command == "download":
        download()
    else:
        print(__doc__)
