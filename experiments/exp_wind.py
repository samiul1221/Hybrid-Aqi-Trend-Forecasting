# Experiment 1 (GPU): is the "wind adds nothing" result real?
# Re-uses notebook 04 Part A (cells 0-43) unchanged except 2 small patches, then trains every variant with 3 seeds:
#   T1         graph, distance prior (no wind)
#   T2-era5    graph + ERA5 wind prior (as in the 9 Oct run)
#   T2-cpcb    graph + MEASURED station wind (CPCB WS/WD, ERA5 where missing)
#   T2-rot180  graph + ERA5 wind turned 180 deg (placebo: wrong direction). If it scores like T2-era5,
#              the model does not use the wind direction at all.
#   T0, T0-nofuture, T0-noweather   no graph: how much of the teacher's lead is weather, not neighbours?
# Scored on the test years, overall, by wind-speed bin and by season, with a week-block bootstrap.
from nb import nb04_cells, code, md, patch, notebook

GPU = True
DATASETS = ["samiulseikh/aqi-features", "samiulseikh/aqi-wind-cpcb"]


def runs_cell(smoke):
    graph = [("T1", (True, False, False, True), "era5", "all"),
             ("T2-era5", (True, True, False, True), "era5", "all"),
             ("T2-rot180", (True, True, False, True), "rot180", "all"),
             ("T2-cpcb", (True, True, False, True), "cpcb", "all")]
    no_graph = [("T0", (False, False, False, True), "era5", "all"),
                ("T0-nofuture", (False, False, False, False), "era5", "all"),
                ("T0-noweather", (False, False, False, False), "era5", "none")]
    runs = {}
    for seed in (0, 1, 2):                         # seed by seed: if time runs out, every variant has seeds 0, 1
        for name, sw, wind, weather in graph + (no_graph if seed < 2 else []):
            runs[f"{name}-s{seed}"] = dict(variant=name, switches=sw, seed=seed, wind=wind, weather=weather)
    if smoke:
        runs = {k: runs[k] for k in ["T1-s0", "T2-cpcb-s0", "T2-rot180-s0", "T0-noweather-s0"]}
    return code(f"""
# every run of this experiment (ordered so that if time runs out, every variant still has seed 0 and 1)
RUNS = {runs!r}
VARIANTS = list(RUNS)
""")


WIND_CELL = """
# experiment switches: which wind feeds the graph prior, which weather the teacher sees (set per run)
WIND_MODE, WEATHER_MODE = "era5", "all"
wc = np.load(next(Path("/kaggle/input").rglob("wind_cpcb.npz")) if ON_KAGGLE else Path("../data/4_features/wind_cpcb.npz"))
assert (wc["ts"] == g0["ts"]).all() and list(wc["stations"]) == stations
CPCB_U, CPCB_V = torch.tensor(wc["u"], device=DEVICE), torch.tensor(wc["v"], device=DEVICE)
CPCB_OK = torch.isfinite(CPCB_U) & torch.isfinite(CPCB_V)
print("measured wind available in", round(float(CPCB_OK.float().mean()) * 100, 1), "% of station-hours")


# wind at hour t for the graph prior: ERA5, measured (ERA5 where missing), or ERA5 turned 180 deg (placebo)
def wind_now(u, t, idx):
    eu, ev = torch.nan_to_num(GW[u, t][:, idx, 3]), torch.nan_to_num(GW[u, t][:, idx, 4])
    if WIND_MODE == "cpcb":
        ok = CPCB_OK[t][:, idx]
        eu = torch.where(ok, torch.nan_to_num(CPCB_U[t][:, idx]), eu)
        ev = torch.where(ok, torch.nan_to_num(CPCB_V[t][:, idx]), ev)
    if WIND_MODE == "rot180":
        return -eu, -ev
    return eu, ev
"""

EVAL_CELLS = [
    """
# test samples, ERA5 wind speed and season of every hour (for the breakdowns)
test_seen, test_all = sample_pairs("test", train_idx), sample_pairs("test", all_idx)
if SMOKE:
    test_seen, test_all = test_seen[:256], test_all[:256]
ERA_WS = np.hypot(g0["W"][:, :, 3], g0["W"][:, :, 4])
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer", 6: "monsoon", 7: "monsoon",
          8: "monsoon", 9: "monsoon", 10: "post-monsoon", 11: "post-monsoon"}
H13 = [0, 1, 2, 6, 7, 8]                  # PM2.5 and PM10 at +1..+3 h (the student's horizons)
""",
    """
# skill vs persistence (%) of a set of samples: 1 - RMSE(model) / RMSE("no change"), over the chosen outputs
def skill(p, keep, outs):
    m = p["y_mask"][..., outs] * keep[..., None]
    se = ((p["y_hat"][..., outs] - p["y"][..., outs]) ** 2 * m).sum()
    sp = (p["y"][..., outs] ** 2 * m).sum()
    return 100 * (1 - np.sqrt(se / max(sp, 1e-9))), float(np.sqrt(se / max(m.sum(), 1))), int(m.sum())


# all the scores of one trained run
def score_run(run, model, cfg):
    rows = []
    p = predict(model, test_seen, train_idx)
    t = test_seen[:, 1].cpu().numpy()
    ws = ERA_WS[t][:, train_idx]                                     # [sample, station]
    season = np.vectorize(SEASON.get)(MONTH[t])[:, None].repeat(len(train_idx), 1)
    scopes = {"all": np.ones_like(ws, bool), "calm<1": ws < 1, "ws1-3": (ws >= 1) & (ws < 3), "windy>3": ws >= 3}
    scopes.update({s: season == s for s in ("winter", "summer", "monsoon", "post-monsoon")})
    for scope, keep in scopes.items():
        for outs, name in ((list(range(12)), "1-6h"), (H13, "1-3h")):
            s, r, n = skill(p, keep.astype(np.float32), outs)
            rows.append(dict(run=run, variant=cfg["variant"], seed=cfg["seed"], stations="seen", scope=scope,
                             outputs=name, skill=s, rmse_ln=r, n=n))
    # week blocks for the paired bootstrap (sum of squared errors and counts, all 12 outputs)
    m = p["y_mask"]
    se = (((p["y_hat"] - p["y"]) ** 2) * m).sum((1, 2))
    week = (t - t.min()) // 168
    blocks = dict(se=np.bincount(week, se), n=np.bincount(week, m.sum((1, 2))), sp=np.bincount(week, (p["y"] ** 2 * m).sum((1, 2))))
    # the 5 never-seen stations
    pa = predict(model, test_all, all_idx)
    keep = np.zeros(pa["y"].shape[:2], np.float32)
    keep[:, HOLDOUT_COLS] = 1
    s, r, n = skill(pa, keep, list(range(12)))
    rows.append(dict(run=run, variant=cfg["variant"], seed=cfg["seed"], stations="unseen", scope="all",
                     outputs="1-6h", skill=s, rmse_ln=r, n=n))
    unit0 = test_seen[:, 0].cpu().numpy() == 0
    return rows, blocks, p["y_hat"][unit0].astype(np.float16), t[unit0], p["y"][unit0], p["y_mask"][unit0]
""",
    """
# train and score every run (stop starting new runs after 10.5 h: Kaggle ends the session at 12 h)
results, blocks, betas = [], {}, {}
for run, cfg in RUNS.items():
    if time.time() - NOTEBOOK_START > 10.5 * 3600:
        print("time budget reached, skipping", run)
        continue
    WIND_MODE, WEATHER_MODE = cfg["wind"], cfg["weather"]
    start = time.time()
    model, history, best = train_teacher(run, train_pairs)
    rows, blocks[run], y_hat0, t0, y0, m0 = score_run(run, model, cfg)
    betas[run] = float(model.gat.beta) if model.gat is not None else None
    for r in rows:
        r.update(val_rmse=best, beta=betas[run], minutes=round((time.time() - start) / 60, 1))
    results += rows
    np.savez_compressed(OUT / f"pred_{run}.npz", y_hat=y_hat0, t=t0, stations=train_idx)
    if not (OUT / "truth_unit0.npz").exists():
        np.savez_compressed(OUT / "truth_unit0.npz", y=y0.astype(np.float32), y_mask=m0.astype(np.uint8), t=t0)
    pd.DataFrame(results).to_csv(OUT / "wind_results.csv", index=False)       # saved after every run
    np.savez_compressed(OUT / "wind_blocks.npz", **{f"{k}|{q}": v[q] for k, v in blocks.items() for q in v})
    print(run, "beta", betas[run], [round(r["skill"], 2) for r in rows if r["scope"] == "all"])
    del model
    torch.cuda.empty_cache()
""",
    """
# summary: mean +- spread over seeds of every variant (seen stations)
res = pd.DataFrame(results)
summary = (res.groupby(["variant", "stations", "scope", "outputs"]).skill.agg(["mean", "std", "count"]).round(2)
           .reset_index())
summary.to_csv(OUT / "wind_summary.csv", index=False)
print(summary[(summary.outputs == "1-6h")].pivot_table(index="variant", columns=["stations", "scope"], values="mean").round(2).to_string())
""",
    """
# paired week-block bootstrap: RMSE difference (%) of variant A vs B, both averaged over their seeds
def boot(a, b, n_boot=4000, seed=0):
    def pooled(v):
        runs = [r for r in blocks if RUNS[r]["variant"] == v]
        return sum(blocks[r]["se"] for r in runs) / len(runs), sum(blocks[r]["n"] for r in runs) / len(runs)
    (sa, na), (sb, nb) = pooled(a), pooled(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sa), (n_boot, len(sa)))
    diff = 100 * (np.sqrt(sa[idx].sum(1) / na[idx].sum(1)) / np.sqrt(sb[idx].sum(1) / nb[idx].sum(1)) - 1)
    full = 100 * (np.sqrt(sa.sum() / na.sum()) / np.sqrt(sb.sum() / nb.sum()) - 1)
    return dict(A=a, B=b, rmse_change_pct=round(full, 3), ci_low=round(np.percentile(diff, 2.5), 3),
                ci_high=round(np.percentile(diff, 97.5), 3), weeks=len(sa))


variants = {RUNS[r]["variant"] for r in blocks}
pairs = [("T2-era5", "T1"), ("T2-cpcb", "T1"), ("T2-rot180", "T1"), ("T2-era5", "T2-rot180"), ("T2-cpcb", "T2-era5"),
         ("T1", "T0"), ("T0", "T0-nofuture"), ("T0-nofuture", "T0-noweather")]
tests = pd.DataFrame([boot(a, b) for a, b in pairs if a in variants and b in variants])
tests.to_csv(OUT / "wind_bootstrap.csv", index=False)
print("negative = A has the smaller error; a CI that contains 0 = no real difference")
print(tests.to_string())
print("learned wind weight beta:", betas)
""",
]


def build(smoke):
    cells = nb04_cells()[:44]
    if smoke:
        cells[4] = patch(cells[4], "SMOKE = not ON_KAGGLE", "SMOKE = True")
    cells[0] = md("# Experiment 1: does the wind help the graph teacher? (built from notebook 04 Part A)")
    cells[5] = md("**Runs:** T1 (no wind), T2 with ERA5 / measured / 180-deg-turned wind, 3 seeds; T0 weather ablation.")
    cells[6] = runs_cell(smoke)
    cells[14] = patch(cells[14], "    wind_u, wind_v = torch.nan_to_num(GW[u, t][:, idx, 3]), torch.nan_to_num(GW[u, t][:, idx, 4])",
                      "    wind_u, wind_v = wind_now(u, t, idx)              # experiment: ERA5 / measured / placebo")
    cells[14] = patch(cells[14], "    x = torch.where(m > 0, (x - MEAN_T) / STD_T, 0.0)\n",
                      "    x = torch.where(m > 0, (x - MEAN_T) / STD_T, 0.0)\n"
                      "    if WEATHER_MODE == \"none\":                                    # experiment: no weather inputs at all\n"
                      "        x = torch.cat([x[..., :23], torch.zeros_like(x[..., 23:])], dim=-1)\n"
                      "        m = torch.cat([m[..., :23], torch.zeros_like(m[..., 23:])], dim=-1)\n")
    cells = cells[:13] + [code(WIND_CELL)] + cells[13:]
    cells += [md("## Experiment runs and results")] + [code(c) for c in EVAL_CELLS]
    return notebook(cells)
