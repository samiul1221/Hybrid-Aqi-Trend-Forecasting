"""Teacher training: plain PyTorch loop ([R07.6c]). Same functions run on the laptop and on Kaggle.

    uv run python -m aqi_ml.models.teacher.train --config ml/configs/teacher/T3.yaml \
        --data data/graph/graph-v1 --out data/models/teacher/T3        # resumable: rerun to continue

One sample = the whole station graph at one base hour of one virtual unit. Training stations exclude the 5
spatial hold-out stations. Loss = masked Huber on ln-change targets (+ pinball for q10/q90). AdamW,
ReduceLROnPlateau, grad clipping, early stopping on val loss; state saved every epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from aqi_ml.data import graph_windows as gw
from aqi_ml.models.teacher import config as tc
from aqi_ml.models.teacher.graph import geometry
from aqi_ml.models.teacher.model import TISTGNN

Pairs = list[tuple[int, int]]  # (unit, base hour index)


# ------------------------------------------------------------------------------------------- data
class Data:
    """Grids of all units + normalisation statistics (TRAIN hours, training stations only)."""

    def __init__(self, root: Path, units: list[int]) -> None:
        self.grids = {u: gw.open_grid(root, u) for u in units}
        g = self.grids[units[0]]
        self.idx_train = np.flatnonzero(~g["holdout"])  # 25 training stations
        self.idx_all = np.arange(len(g["stations"]))  # 30 (test includes the unseen stations)
        st = gw.train_stats(g, self.idx_train)
        self.mean, self.std = st["mean"], st["std"]  # [30]
        s = g["static"]
        self.s_mean, self.s_std = s.mean(0), np.maximum(s.std(0), 1e-3)
        self.meta_hash = hashlib.sha256((root / "meta.json").read_bytes()).hexdigest()[:16]

    def pairs(self, split: str, idx: npt.NDArray[np.int64]) -> Pairs:
        return [(u, int(t)) for u, g in self.grids.items() for t in gw.sample_index(g, split, idx)]

    def geometry(self, idx: npt.NDArray[np.int64]) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.grids[next(iter(self.grids))]["static"][idx]
        return geometry(s[:, 0].astype(np.float64), s[:, 1].astype(np.float64))

    def batch(self, pairs: Pairs, idx: npt.NDArray[np.int64], device: str,
              drop_p: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, torch.Tensor]:  # fmt: skip
        ws = [gw.window(self.grids[u], t, idx) for u, t in pairs]
        x = np.stack([w["X"] for w in ws])  # [B, N, L, 30] raw
        m = np.stack([w["M"] for w in ws])
        x = np.where(m > 0, (x - self.mean) / self.std, 0.0)  # z-score valid values only
        fut = np.stack([w["X_fut"] for w in ws])  # [B, N, H, 9]: 7 Open-Meteo + hod sin/cos
        fut[..., :7] = (fut[..., :7] - self.mean[23:30]) / self.std[23:30]
        s = (np.stack([w["S"] for w in ws]) - self.s_mean) / self.s_std
        if drop_p > 0 and rng is not None:  # station dropout: hide whole nodes, keep their targets
            keep = (rng.random(m.shape[:2]) >= drop_p)[:, :, None, None]
            x, m = x * keep, m * keep
        arrays = {"X": x, "M": m, "S": s, "X_fut": fut, "wind_t": np.stack([w["wind_t"] for w in ws]),
                  "y": np.stack([w["y"] for w in ws]), "y_mask": np.stack([w["y_mask"] for w in ws])}  # fmt: skip
        return {
            k: torch.as_tensor(np.asarray(v, np.float32), device=device) for k, v in arrays.items()
        }


# ------------------------------------------------------------------------------------------- loss
def masked_huber(
    pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, delta: float
) -> torch.Tensor:
    err = torch.nn.functional.huber_loss(pred, y, reduction="none", delta=delta)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def pinball(q: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
    diff = y - q
    return (torch.maximum(tau * diff, (tau - 1) * diff) * mask).sum() / mask.sum().clamp_min(1.0)


def loss_fn(
    out: dict[str, torch.Tensor], b: dict[str, torch.Tensor], cfg: tc.TeacherConfig
) -> torch.Tensor:
    loss = masked_huber(out["y"], b["y"], b["y_mask"], cfg.huber_delta)
    if cfg.quantiles:
        q = pinball(out["q10"], b["y"], b["y_mask"], 0.1) + pinball(
            out["q90"], b["y"], b["y_mask"], 0.9
        )
        loss = loss + cfg.pinball_weight * q
    return loss


# ------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model: TISTGNN, data: Data, pairs: Pairs, idx: npt.NDArray[np.int64], device: str,
             batch: int = 32) -> dict[str, Any]:  # fmt: skip
    """Mean loss + masked RMSE per output (ln-change space), pollutant-major order."""
    model.eval()
    dist, theta = (t.to(device) for t in data.geometry(idx))
    se, cnt, losses = torch.zeros(12, device=device), torch.zeros(12, device=device), []
    for k in range(0, len(pairs), batch):
        b = data.batch(pairs[k : k + batch], idx, device)
        out = model(b, dist, theta)
        losses.append(float(loss_fn(out, b, model.cfg)))
        se += (((out["y"] - b["y"]) ** 2) * b["y_mask"]).sum(dim=(0, 1))
        cnt += b["y_mask"].sum(dim=(0, 1))
    names = [f"{p}_h{h}" for p in ("pm25", "pm10") for h in range(1, 7)]
    rmse = (se / cnt.clamp_min(1)).sqrt().tolist()
    return {
        "loss": float(np.mean(losses)),
        "rmse_log": dict(zip(names, rmse, strict=True)),
        "n": len(pairs),
    }


# --------------------------------------------------------------------------------------- training
def _git_sha() -> str:
    from aqi_ml.lineage import git_sha

    return (
        os.environ.get("AQI_GIT_SHA") or git_sha()
    )  # Kaggle has no .git: the notebook sets the env var


def train(
    cfg: tc.TeacherConfig, data_root: Path, out: Path, device: str | None = None
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    data = Data(data_root, cfg.units)
    tr, va = data.pairs("train", data.idx_train), data.pairs("val", data.idx_train)
    if cfg.max_train_samples:
        tr = [
            tr[i]
            for i in rng.choice(len(tr), size=min(cfg.max_train_samples, len(tr)), replace=False)
        ]
        va = va[: max(64, cfg.max_train_samples // 4)]
    dist, theta = (t.to(device) for t in data.geometry(data.idx_train))

    model = TISTGNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=cfg.lr_patience, factor=cfg.lr_factor
    )
    st: dict[str, Any] = {"epoch": 0, "best": float("inf"), "best_epoch": -1, "history": []}
    if (out / "state.pt").exists():  # resume after a stopped Kaggle session
        ck = torch.load(out / "state.pt", map_location=device, weights_only=False)
        (
            model.load_state_dict(ck["model"]),
            opt.load_state_dict(ck["opt"]),
            sched.load_state_dict(ck["sched"]),
        )
        st, rng = ck["st"], ck["rng"]

    while st["epoch"] < cfg.max_epochs and st["epoch"] - st["best_epoch"] <= cfg.patience:
        model.train()
        t0, run = time.time(), []
        order = rng.permutation(len(tr))
        for k in range(0, len(order), cfg.batch):
            b = data.batch(
                [tr[i] for i in order[k : k + cfg.batch]],
                data.idx_train,
                device,
                cfg.station_dropout,
                rng,
            )
            loss = loss_fn(model(b, dist, theta), b, cfg)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            run.append(loss.item())
        val = evaluate(model, data, va, data.idx_train, device)
        sched.step(val["loss"])
        st["history"].append({"epoch": st["epoch"], "train_loss": float(np.mean(run)), "val_loss": val["loss"],
                              "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})  # fmt: skip
        print(st["history"][-1], flush=True)
        if val["loss"] < st["best"]:
            st["best"], st["best_epoch"] = val["loss"], st["epoch"]
            torch.save(model.state_dict(), out / "best.pt")
        st["epoch"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "st": st, "rng": rng}, out / "state.pt")  # fmt: skip

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    metrics = {"config": cfg.__dict__, "val": evaluate(model, data, va, data.idx_train, device),
               "history": st["history"], "params": sum(p.numel() for p in model.parameters()),
               "git_sha": _git_sha(), "graph_meta_hash": data.meta_hash, "torch": torch.__version__,
               "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
               "n_train": len(tr), "n_val": len(va)}  # fmt: skip
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aqi_ml.models.teacher.train")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    m = train(tc.load(a.config), a.data, a.out)
    print(json.dumps(m["val"], indent=1))


if __name__ == "__main__":
    main()
