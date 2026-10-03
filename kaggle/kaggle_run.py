"""
Run notebooks 04 (teacher) and 05 (student) on a free Kaggle GPU (T4), from the laptop.

    uv run python kaggle/kaggle_run.py upload                   # 1. send data/4_features (grids + windows) as a private dataset
    uv run python kaggle/kaggle_run.py teacher --smoke          # 2a. tiny test run (checks the setup, shows the speed)
    uv run python kaggle/kaggle_run.py teacher                  # 2b. full teacher run (T0-T3)
    uv run python kaggle/kaggle_run.py teacher --variants T3    #     or only some variants (one session = max 12 h)
    uv run python kaggle/kaggle_run.py status teacher           #     running / complete / error
    uv run python kaggle/kaggle_run.py download teacher         #     results -> data/5_models/kaggle/teacher/
    uv run python kaggle/kaggle_run.py student                  # 3. student + distillation (reads the teacher's output)

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
STAGING = ROOT / "kaggle" / "_staging"                     # temporary upload folders (git-ignored)
NOTEBOOKS = {"teacher": ("04_teacher.ipynb", "aqi-teacher"), "student": ("05_student_kd.ipynb", "aqi-student-kd")}


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


# push a notebook to Kaggle and start it on a T4 GPU
def push(name, smoke=False, variants=None):
    notebook, slug = NOTEBOOKS[name]
    folder = STAGING / slug
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    text = (ROOT / "notebooks" / notebook).read_text(encoding="utf-8")
    if smoke:                     # change the settings cell of the uploaded copy only
        text = text.replace("SMOKE = not ON_KAGGLE", "SMOKE = True")
    if variants:
        text = text.replace('VARIANTS = [\\"T0\\", \\"T1\\", \\"T2\\", \\"T3\\"]', "VARIANTS = " + json.dumps(variants).replace('"', '\\"'))
    (folder / notebook).write_text(text, encoding="utf-8")
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": notebook, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{USER}/{DATASET}"],
            "kernel_sources": [f"{USER}/aqi-teacher"] if name == "student" else []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    kaggle("kernels", "push", "-p", str(folder))


if __name__ == "__main__":
    load_token()
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "upload":
        upload()
    elif command in NOTEBOOKS:
        variants = sys.argv[sys.argv.index("--variants") + 1].split(",") if "--variants" in sys.argv else None
        push(command, smoke="--smoke" in sys.argv, variants=variants)
    elif command == "status":
        kaggle("kernels", "status", f"{USER}/{NOTEBOOKS[sys.argv[2]][1]}")
    elif command == "download":
        out = ROOT / "data" / "5_models" / "kaggle" / sys.argv[2]
        out.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", f"{USER}/{NOTEBOOKS[sys.argv[2]][1]}", "-p", str(out))
    else:
        print(__doc__)
