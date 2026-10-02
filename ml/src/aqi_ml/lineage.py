"""Run lineage for tracking ([C08.7] reproducibility, [C11] tracking): what every pipeline run logs.

snapshot() = schema hash, git sha, sim_params version/hash, calibration version and a content hash of
the data snapshot (base table + dataset manifests). Hashes of large trees are cached by (path, size,
mtime) so repeated pipeline runs stay cheap.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from aqi_ml.features.dataset import ROOT
from aqi_shared.pipeline import load_calibration
from aqi_shared.schema_loader import load as load_schema

DATA = ROOT / "data"
_CACHE = DATA / "lineage_cache.json"


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def tree_hash(path: Path, pattern: str = "*.parquet") -> str:
    """sha256 over (relative path, bytes) of every matching file; cached by path/size/mtime."""
    files = sorted(path.rglob(pattern))
    stamp = hashlib.sha256(
        json.dumps([[f.relative_to(path).as_posix(), f.stat().st_size, f.stat().st_mtime_ns] for f in files]).encode()
    ).hexdigest()  # fmt: skip
    cache: dict[str, Any] = json.loads(_CACHE.read_text()) if _CACHE.exists() else {}
    key = f"{path.as_posix()}|{pattern}"
    if cache.get(key, {}).get("stamp") == stamp:
        return str(cache[key]["hash"])
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(path).as_posix().encode())
        h.update(f.read_bytes())  # ponytail: whole-file read; shards are a few MB
    out = "sha256:" + h.hexdigest()
    cache[key] = {"stamp": stamp, "hash": out}
    _CACHE.write_text(json.dumps(cache, indent=1))
    return out


def snapshot(dataset: str = "feat-v1") -> dict[str, Any]:
    """Identity of the data a run trains on (snapshot hash = hash of the parts below)."""
    ds_manifest = json.loads((DATA / "datasets" / dataset / "manifest.json").read_text())
    sim = ds_manifest.get("sim_manifest", {})
    parts = {
        "base_table_hash": tree_hash(DATA / "processed" / "base_table"),
        "dataset": dataset,
        "dataset_manifest_hash": "sha256:" + hashlib.sha256(json.dumps(ds_manifest, sort_keys=True).encode()).hexdigest(),
        "sim_version": sim.get("sim_version"),
        "sim_params_hash": sim.get("sim_params_hash"),
        "schema_hash": load_schema().hash,
        "calibration_version": load_calibration()["version"],
    }  # fmt: skip
    parts["snapshot_hash"] = (
        "sha256:" + hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    )
    parts["git_sha"] = git_sha()
    return parts
