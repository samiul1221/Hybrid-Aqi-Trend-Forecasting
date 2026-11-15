# tiny helpers to build an experiment notebook from notebook 04's own cells (so the teacher code is identical)
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nb04_cells():
    return json.loads((ROOT / "notebooks/04_train.ipynb").read_text(encoding="utf-8"))["cells"]


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


# replace text in a cell and fail loudly if the text is not there (notebook 04 changed)
def patch(cell, old, new):
    src = "".join(cell["source"])
    assert old in src, f"patch target not found: {old[:60]}"
    cell = dict(cell, source=src.replace(old, new), outputs=[], execution_count=None) if cell["cell_type"] == "code" \
        else dict(cell, source=src.replace(old, new))
    return cell


def notebook(cells):
    for c in cells:
        c["source"] = "".join(c["source"])
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}
