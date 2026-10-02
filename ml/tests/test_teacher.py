"""N3 TI-STGNN unit tests on tiny random graphs (CR v2 [R07.3]-[R07.6] unit_tests + overfit check)."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from aqi_ml.models.teacher import adr, graph
from aqi_ml.models.teacher.config import TeacherConfig
from aqi_ml.models.teacher.gat import WindGAT
from aqi_ml.models.teacher.model import TISTGNN

CFG = TeacherConfig(name="t", use_graph=True, use_wind=True, use_adr=True, quantiles=True, d_model=16, heads=2,
                    temporal_layers=1, dropout=0.0, d_min_km=1.0, calm_ms=0.3, sigma_km=10.0, gat_eps=1e-3,
                    beta_init=1.0, adr_steps=2, adr_h=0.1, graph_version="x", units=[0], batch=8, lr=1e-3,
                    weight_decay=0.0, grad_clip=1.0, max_epochs=1, patience=1, lr_patience=1, lr_factor=0.5,
                    station_dropout=0.0, huber_delta=1.0, pinball_weight=0.5, seed=0, max_train_samples=0)  # fmt: skip
VARIANTS = {
    "T0": (False, False, False),
    "T1": (True, False, False),
    "T2": (True, True, False),
    "T3": (True, True, True),
}


def _west_mid_east() -> tuple[torch.Tensor, torch.Tensor]:
    return graph.geometry(
        np.array([28.6, 28.6, 28.6]), np.array([76.9, 77.0, 77.1])
    )  # W, middle, E


def test_wind_prior_downwind_only() -> None:
    dist, theta = _west_mid_east()
    assert theta[1, 2] == pytest.approx(90.0, abs=0.1) and dist[1, 2] == pytest.approx(
        9.77, abs=0.1
    )
    wind_from_west = torch.full((1, 3), 270.0)  # air moves east
    w = graph.wind_prior(torch.full((1, 3), 3.0), wind_from_west, dist, theta, 1.0, 0.3)[0]
    assert w[1, 2] > 0 and w[1, 0] == 0  # middle -> east neighbour gets weight, west gets none
    calm = graph.wind_prior(torch.full((1, 3), 0.1), wind_from_west, dist, theta, 1.0, 0.3)
    assert calm.abs().sum() == 0


def test_static_prior_and_laplacian() -> None:
    dist, _ = graph.geometry(
        np.random.default_rng(0).uniform(28.4, 28.8, 6),
        np.random.default_rng(1).uniform(76.9, 77.3, 6),
    )
    s = graph.static_prior(dist, 1.0)
    assert torch.allclose(s, s.T) and torch.all(torch.diag(s) == 0)
    v, lam = graph.diffusion_basis(dist, torch.ones(2, 6), 10.0)
    assert lam.min() >= 0 and lam.max() <= 2 + 1e-5 and v.shape == (2, 6, 6)


def test_gat_bias_uses_incoming_edge() -> None:
    """I26: with the wind blowing east, the EAST station (receiver) must favour the middle station (sender)."""
    dist, theta = _west_mid_east()
    w = graph.wind_prior(torch.full((1, 3), 3.0), torch.full((1, 3), 270.0), dist, theta, 1.0, 0.3)
    gat = WindGAT(4, 1.0, 1e-3)
    lg = gat.logits(torch.zeros(1, 3, 4), w)[0]  # zero features: only the prior bias remains
    assert lg[2, 1] > lg[1, 2]  # receiver east <- sender middle  >  receiver middle <- sender east


def test_advection_conserves_mass_and_masks() -> None:
    torch.manual_seed(0)
    b, n, d = 2, 5, 3
    valid = torch.ones(b, n)
    valid[1, 3] = 0
    p = adr.advection_weights(torch.randn(b, n, n), valid)
    cols = p.sum(dim=1)  # total weight each source sends
    assert torch.allclose(cols[0], torch.ones(n)) and cols[1, 3] == 0 and p[1, 3].sum() == 0
    assert torch.all(torch.diagonal(p, dim1=1, dim2=2) == 0)
    u = torch.randn(b, n, d)
    u2 = adr.advect(u, p, 0.1)
    assert torch.allclose(u2.sum(dim=1), u.sum(dim=1), atol=1e-5)  # mass conservation
    assert torch.allclose(u2[1, 3], u[1, 3])  # invalid node untouched


def test_kappa_range_and_diffusion_bounded() -> None:
    m = adr.ADR(4, 7, 1, 0.1)
    with torch.no_grad():
        m.kappa_raw.copy_(torch.tensor([-3.0, 0.2, 0.9, 5.0]))
    k = m.kappa()
    assert k.min() >= 0 and k.max() <= 1
    dist, _ = graph.geometry(np.linspace(28.5, 28.7, 5), np.linspace(77.0, 77.2, 5))
    valid = torch.tensor([[1.0, 1, 1, 1, 0]])
    v, lam = graph.diffusion_basis(dist, valid, 10.0)
    u = torch.randn(1, 5, 4)
    u2 = adr.diffuse(u, v, lam, k, 0.5)
    # (I + c L_hat)^-1 has spectral norm <= 1: the L2 norm never grows (the max-norm version of the CR test
    # does not hold for the normalised Laplacian with unequal degrees, so we check the provable bound)
    assert torch.all(u2.norm(dim=1) <= u.norm(dim=1) + 1e-5)
    assert torch.allclose(u2[0, 4], u[0, 4], atol=1e-5)  # cut-out node unchanged


def _batch(b: int, n: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {"X": torch.randn(b, n, 24, 30, generator=g), "M": torch.ones(b, n, 24, 30),
            "S": torch.randn(b, n, 6, generator=g), "X_fut": torch.randn(b, n, 6, 9, generator=g),
            "wind_t": torch.stack([torch.rand(b, n, generator=g) * 5, torch.rand(b, n, generator=g) * 360], -1),
            "y": torch.randn(b, n, 12, generator=g) * 0.3, "y_mask": torch.ones(b, n, 12)}  # fmt: skip


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_variants_shapes_and_equivariance(variant: str) -> None:
    g, w, a = VARIANTS[variant]
    torch.manual_seed(0)
    model = TISTGNN(replace(CFG, use_graph=g, use_wind=w, use_adr=a)).eval()
    lat, lon = (
        np.random.default_rng(0).uniform(28.4, 28.8, 5),
        np.random.default_rng(1).uniform(76.9, 77.3, 5),
    )
    dist, theta = graph.geometry(lat, lon)
    b = _batch(2, 5)
    out = model(b, dist, theta)
    assert out["y"].shape == (2, 5, 12) and out["feat_seq"].shape == (2, 5, 24, 16)
    assert out["attn"].shape == (2, 5, 2, 24, 24) and out["h_adr"].shape == (2, 5, 16)
    assert torch.all(out["q10"] <= out["y"]) and torch.all(out["y"] <= out["q90"])
    perm = torch.tensor([3, 0, 4, 1, 2])
    bp = {k: v[:, perm] for k, v in b.items()}
    outp = model(bp, dist[perm][:, perm], theta[perm][:, perm])
    assert torch.allclose(
        outp["y"], out["y"][:, perm], atol=1e-4
    )  # permuting stations permutes outputs


def test_overfit_50_samples() -> None:
    """The full T3 model can memorise 50 graph samples (checks that gradients reach every block)."""
    torch.manual_seed(0)
    model = TISTGNN(replace(CFG, d_model=32))
    dist, theta = graph.geometry(np.linspace(28.5, 28.7, 4), np.linspace(77.0, 77.2, 4))
    b = _batch(50, 4, seed=1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(250):
        loss = torch.nn.functional.huber_loss(model(b, dist, theta)["y"], b["y"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.2 * losses[0]
    assert model.gat is not None and model.gat.beta.grad is not None and model.adr is not None
    assert model.adr.kappa_raw.grad is not None and model.adr.kappa_raw.grad.abs().sum() > 0


def test_config_rejects_missing_and_inconsistent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import yaml

    from aqi_ml.models.teacher import config as tc

    for name in ("T0", "T1", "T2", "T3"):
        tc.load(tc.Path(__file__).parents[1] / "configs" / "teacher" / f"{name}.yaml")
    raw = dict(CFG.__dict__)
    raw.pop("seed")
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="missing"):
        tc.load(tmp_path / "bad.yaml")
    with pytest.raises(ValueError):
        replace(CFG, use_graph=False)  # ADR without a graph
