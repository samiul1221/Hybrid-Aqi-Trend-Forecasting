# 0013 — v2 graph prior and ADR operators (CR v2 I4, I5, I6, I12, I20)

Status: accepted (2026-10-02)

- **Wind direction (I4):** TransNet (npj Clean Air 2:12, Eq. for w_ij) feeds the meteorological FROM
  direction (270 − atan2(V,U)) straight into cos(θ_ij − WD); taken literally that points edges to the
  **upwind** station. We use φ_i = (wd_from_i + 180) mod 360 so the downwind neighbour gets the weight.
  Unit test: wind blowing east → east neighbour > 0, west neighbour = 0.
- **Where space happens (I5):** GAT and ADR run once at base_ts on h_t = feat_seq[:, -1].
- **Advection weights (I6):** the GAT logit matrix, diagonal masked, softmax over receivers per source →
  mass-conserving and wind-asymmetric without TransNet's separate A1–A4 network (simplification; A1–A4 = ablation).
- **Step size (I12):** TransNet uses h = 0.01; our states are LayerNorm-scaled, so start h = 0.1, K = 2, test {0.01, 0.1}.
- **Diffusion graph (I20):** TransNet diffuses on the Laplacian of the wind graph; we use the static
  Gaussian graph exp(−d²/σ²), σ = 10 km, so L̂ = VΛVᵀ is computed once and the implicit step is exact.
