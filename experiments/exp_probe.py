# Experiment 2 (CPU): model-free probes, no deep network.
#  P1  Does a wind-directed neighbour signal predict a station's PM change? LightGBM arms with neighbour features
#      weighted by distance / upwind (ERA5) / upwind (measured CPCB wind) / DOWNwind (placebo).
#  P2  Evolved transport kernel: differential evolution (a genetic-type search) tunes
#      w_ij = d^-p * exp(kappa * cos(angle) * min(ws / ws0, 1)). kappa > 0 = upwind matters, kappa ~ 0 = direction useless.
#  P3  Which sensors are worth their power? All 2^7 = 128 subsets of the optional sensors (exhaustive: with 7 sensors
#      this is exact and cheaper than a genetic algorithm), LightGBM proxy, Pareto front skill vs power.
# Inputs follow the project rule: only simulated sensor features (unit 0) + ERA5 / measured wind; true PM = target only.
from nb import code, md, notebook

GPU = False
DATASETS = ["samiulseikh/aqi-features", "samiulseikh/aqi-wind-cpcb"]

CELLS = [
    md("# Experiment 2: model-free probes (wind, evolved transport kernel, sensor subsets)"),
    code("""
# imports, data (unit 0 sensors, true PM as target only, ERA5 weather, measured CPCB wind)
import time, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import differential_evolution

SMOKE = False
ON_KAGGLE = Path("/kaggle/input").exists()
FEAT = next(Path("/kaggle/input").rglob("grid_u0.npz")).parent if ON_KAGGLE else Path("../data/4_features")
OUT = Path("/kaggle/working") if ON_KAGGLE else Path("../data/6_experiments/probe")
OUT.mkdir(parents=True, exist_ok=True)
g = np.load(FEAT / "grid_u0.npz")
wc = np.load(next(Path("/kaggle/input").rglob("wind_cpcb.npz")) if ON_KAGGLE else FEAT / "wind_cpcb.npz")
assert (wc["ts"] == g["ts"]).all()
X, M, PM, W, split = g["X"], g["M"], g["pm"], g["W"], g["split"]
ST = np.flatnonzero(~g["holdout"])                 # the 27 training stations (receivers and neighbours)
X, M, PM, W = X[:, ST], M[:, ST], PM[:, ST], W[:, ST]
CU, CV = wc["u"][:, ST], wc["v"][:, ST]
T, N = X.shape[:2]
FEATURES = ["ln_pm25", "ln_pm10", "ln_pm1", "coarse_ratio", "t", "rh", "p", "wind_u", "wind_v", "rain_frac",
            "ln_lux", "mq7_n", "mq135_n", "voc_n", "nox_n", "hod_sin", "hod_cos", "dow_sin", "dow_cos",
            "doy_sin", "doy_cos", "aqi_now", "pm_mask"]
F = {n: np.where(M[:, :, k] > 0, X[:, :, k], np.nan) for k, n in enumerate(FEATURES)}
print(T, "hours x", N, "stations")
"""),
    code("""
# targets: change of ln(1 + true PM) after h hours (PM2.5 +1 h, +3 h; PM10 +3 h)
def target(k, h):
    lp = np.log1p(PM[:, :, k])
    y = np.full_like(lp, np.nan)
    y[:-h] = lp[h:] - lp[:-h]
    return y

TARGETS = {"pm25_h1": target(0, 1), "pm25_h3": target(0, 3), "pm10_h3": target(1, 3)}

# station geometry: distance (km) and bearing from station j to station i
lat, lon = np.deg2rad(g["static"][ST, 0]), np.deg2rad(g["static"][ST, 1])
la_i, la_j, dlo = lat[:, None], lat[None, :], lon[:, None] - lon[None, :]
DIST = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la_i - la_j) / 2) ** 2 + np.cos(la_i) * np.cos(la_j) * np.sin(dlo / 2) ** 2))
BEAR_JI = np.rad2deg(np.arctan2(np.sin(dlo) * np.cos(la_i), np.cos(la_j) * np.sin(la_i) - np.sin(la_j) * np.cos(la_i) * np.cos(dlo))) % 360
np.fill_diagonal(DIST, np.inf)
"""),
    code("""
# wind at each receiving station: ERA5, or measured CPCB (ERA5 where missing). 'toward' = where the air goes
def wind(source):
    u, v = W[:, :, 3].copy(), W[:, :, 4].copy()
    if source == "cpcb":
        ok = np.isfinite(CU) & np.isfinite(CV)
        u, v = np.where(ok, CU, u), np.where(ok, CV, v)
    return np.hypot(u, v), np.rad2deg(np.arctan2(u, v)) % 360

WIND = {s: wind(s) for s in ("era5", "cpcb")}

# neighbour weights w[t, i, j] = d^-p * exp(kappa * cos(angle) * min(ws / ws0, 1)); angle = air path j->i vs wind
def weights(hours, p, kappa, ws0, source="era5", flip=False):
    ws, toward = WIND[source]
    ws, toward = ws[hours], toward[hours] + (180 if flip else 0)
    cosang = np.cos(np.deg2rad(BEAR_JI[None] - toward[:, :, None]))
    near = np.where(np.isinf(DIST), 0, DIST ** -p)          # no self-weight, also for p = 0
    return near[None] * np.exp(kappa * cosang * np.minimum(ws / ws0, 1)[:, :, None])

# neighbour features: weighted mean of (neighbour - own) sensor ln PM2.5 now, and of the neighbours' last 1-h change
S = F["ln_pm25"]
D1 = np.vstack([np.full((1, N), np.nan), S[1:] - S[:-1]])
def neighbour_features(hours, w):
    s, d1 = S[hours], D1[hours]
    ok_s, ok_d = np.isfinite(s)[:, None, :], np.isfinite(d1)[:, None, :]
    gap = (w * ok_s * np.nan_to_num(s)[:, None, :]).sum(2) / np.maximum((w * ok_s).sum(2), 1e-9) - s
    trend = (w * ok_d * np.nan_to_num(d1)[:, None, :]).sum(2) / np.maximum((w * ok_d).sum(2), 1e-9)
    return gap, trend
"""),
    code("""
# own features of every (hour, station): the node's sensors now, its PM changes, 24-h mean, ERA5 wind, clock
def lag_diff(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[k:] - a[:-k]
    return out

S24 = pd.DataFrame(S).rolling(24, min_periods=12).mean().to_numpy()
OWN = {"s": S, "d1": D1, "d3": lag_diff(S, 3), "d6": lag_diff(S, 6), "s24": S - S24, "pm10": F["ln_pm10"],
       "rh": F["rh"], "t": F["t"], "mq7": F["mq7_n"], "hs": F["hod_sin"], "hc": F["hod_cos"], "ds": F["doy_sin"],
       "dc": F["doy_cos"], "era_ws": WIND["era5"][0], "era_u": W[:, :, 3], "era_v": W[:, :, 4]}

def hours_of(name, step=1):
    h = np.flatnonzero(split == name)
    return h[(h > 24) & (h < T - 6)][::step]

H_TRAIN, H_VAL, H_TEST = hours_of("train", 1 if not SMOKE else 50), hours_of("val", 1 if not SMOKE else 50), hours_of("test", 1 if not SMOKE else 50)
"""),
    md("## P1. LightGBM arms: does a wind-directed neighbour signal add skill?"),
    code("""
# design matrix of a set of hours: own features + neighbour features of each kernel in `kernels`
KERNELS = {"dist": dict(p=1, kappa=0, ws0=1), "up_era5": dict(p=1, kappa=3, ws0=2),
           "down_era5": dict(p=1, kappa=3, ws0=2, flip=True), "up_cpcb": dict(p=1, kappa=3, ws0=2, source="cpcb")}

def design(hours, kernels):
    cols = {k: v[hours] for k, v in OWN.items()}
    for name in kernels:
        cols[f"{name}_gap"], cols[f"{name}_trend"] = neighbour_features(hours, weights(hours, **KERNELS[name]))
    return pd.DataFrame({k: v.ravel() for k, v in cols.items()})

ARMS = {"own": [], "own+dist": ["dist"], "own+dist+up_era5": ["dist", "up_era5"],
        "own+dist+down_era5 (placebo)": ["dist", "down_era5"], "own+dist+up_cpcb": ["dist", "up_cpcb"]}
PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
"""),
    code("""
# train every arm x target (early stopping on 2024), score on 2025-26; keep week sums for the paired bootstrap
p1_rows, p1_blocks = [], {}
ws_test = WIND["era5"][0][H_TEST].ravel()
week = np.repeat((H_TEST - H_TEST.min()) // 168, N)
for arm, kernels in ARMS.items():
    Xtr, Xva, Xte = design(H_TRAIN, kernels), design(H_VAL, kernels), design(H_TEST, kernels)
    for tname, Y in TARGETS.items():
        ytr, yva, yte = Y[H_TRAIN].ravel(), Y[H_VAL].ravel(), Y[H_TEST].ravel()
        a, b = np.isfinite(ytr), np.isfinite(yva)
        start = time.time()
        model = lgb.train(PARAMS, lgb.Dataset(Xtr[a], ytr[a]), 2000 if not SMOKE else 20,
                          valid_sets=[lgb.Dataset(Xva[b], yva[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred, ok = model.predict(Xte), np.isfinite(yte)
        se, sp = np.where(ok, (pred - yte) ** 2, 0), np.where(ok, yte ** 2, 0)
        p1_blocks[(arm, tname)] = (np.bincount(week, se), np.bincount(week, ok.astype(float)))
        for scope, keep in {"all": ok, "calm<1": ok & (ws_test < 1), "windy>3": ok & (ws_test >= 3)}.items():
            p1_rows.append(dict(arm=arm, target=tname, scope=scope, n=int(keep.sum()),
                                skill=100 * (1 - np.sqrt(se[keep].sum() / sp[keep].sum()))))
        print(arm, tname, round(p1_rows[-3]["skill"], 2), f"{time.time() - start:.0f}s", flush=True)
p1 = pd.DataFrame(p1_rows)
p1.to_csv(OUT / "p1_lightgbm_arms.csv", index=False)
print(p1.pivot_table(index="arm", columns=["target", "scope"], values="skill").round(2).to_string())
"""),
    code("""
# paired week-block bootstrap of the RMSE change (%) of arm A vs arm B (negative = A better)
def boot(a, b, tname, n=4000):
    (sa, na), (sb, nb) = p1_blocks[(a, tname)], p1_blocks[(b, tname)]
    idx = np.random.default_rng(0).integers(0, len(sa), (n, len(sa)))
    d = 100 * (np.sqrt(sa[idx].sum(1) / na[idx].sum(1)) / np.sqrt(sb[idx].sum(1) / nb[idx].sum(1)) - 1)
    return dict(A=a, B=b, target=tname, change_pct=round(100 * (np.sqrt(sa.sum() / na.sum()) / np.sqrt(sb.sum() / nb.sum()) - 1), 3),
                ci_low=round(np.percentile(d, 2.5), 3), ci_high=round(np.percentile(d, 97.5), 3))

p1_tests = pd.DataFrame([boot(a, b, t) for t in TARGETS for a, b in [
    ("own+dist", "own"), ("own+dist+up_era5", "own+dist"), ("own+dist+down_era5 (placebo)", "own+dist"),
    ("own+dist+up_era5", "own+dist+down_era5 (placebo)"), ("own+dist+up_cpcb", "own+dist"), ("own+dist+up_cpcb", "own+dist+up_era5")]])
p1_tests.to_csv(OUT / "p1_bootstrap.csv", index=False)
print(p1_tests.to_string())
"""),
    md("## P2. Evolved transport kernel (differential evolution)"),
    code("""
# fitness = validation RMSE of a ridge model on own + neighbour features (PM2.5 +3 h), for kernel (p, kappa, ws0)
OWN_KEYS = list(OWN)
def ridge_xy(hours, w, tname="pm25_h3"):
    gap, trend = neighbour_features(hours, w)
    cols = [OWN[k][hours].ravel() for k in OWN_KEYS] + [gap.ravel(), trend.ravel()]
    x, y = np.column_stack(cols), TARGETS[tname][hours].ravel()
    ok = np.isfinite(x).all(1) & np.isfinite(y)
    return x[ok], y[ok]

H_FIT = H_TRAIN[::3]
def fit_ridge(x, y, lam=1.0):
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = np.column_stack([(x - mu) / sd, np.ones(len(x))])
    beta = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ y)
    return lambda x2: np.column_stack([(x2 - mu) / sd, np.ones(len(x2))]) @ beta

def rmse_of(params, source, eval_hours):
    p, kappa, ws0 = params
    f = fit_ridge(*ridge_xy(H_FIT, weights(H_FIT, p, kappa, ws0, source)))
    x, y = ridge_xy(eval_hours, weights(eval_hours, p, kappa, ws0, source))
    return float(np.sqrt(np.mean((f(x) - y) ** 2)))
"""),
    code("""
# evolve (p, kappa, ws0) with 3 different seeds per wind source; also the best kernel WITHOUT direction (kappa = 0)
BOUNDS = [(0, 3), (-6, 6), (0.5, 8)]
p2_rows = []
for source in ("era5", "cpcb"):
    for seed in range(3 if not SMOKE else 1):
        start = time.time()
        r = differential_evolution(rmse_of, BOUNDS, args=(source, H_VAL), seed=seed, popsize=8, maxiter=20 if not SMOKE else 1,
                                   tol=1e-6, polish=False, init="sobol")
        p2_rows.append(dict(source=source, seed=seed, p=r.x[0], kappa=r.x[1], ws0=r.x[2], val_rmse=r.fun,
                            test_rmse=rmse_of(r.x, source, H_TEST), evals=r.nfev, minutes=round((time.time() - start) / 60, 1)))
        print(p2_rows[-1], flush=True)
r0 = differential_evolution(lambda q: rmse_of((q[0], 0.0, 1.0), "era5", H_VAL), [(0, 3)], seed=0, popsize=8,
                            maxiter=20 if not SMOKE else 1, polish=False)
p2_rows.append(dict(source="no direction (kappa=0)", seed=0, p=r0.x[0], kappa=0.0, ws0=1.0, val_rmse=r0.fun,
                    test_rmse=rmse_of((r0.x[0], 0.0, 1.0), "era5", H_TEST)))
p2 = pd.DataFrame(p2_rows)
p2.to_csv(OUT / "p2_evolved_kernel.csv", index=False)
print(p2.round(4).to_string())
"""),
    code("""
# landscape: validation RMSE along kappa (p, ws0 fixed at the ERA5 optimum): is the optimum sharp or flat?
best = p2[p2.source == "era5"].sort_values("val_rmse").iloc[0]
kap = np.linspace(-6, 6, 25 if not SMOKE else 3)
p2_curve = pd.DataFrame(dict(kappa=kap, val_rmse=[rmse_of((best.p, k, best.ws0), "era5", H_VAL) for k in kap]))
p2_curve.to_csv(OUT / "p2_kappa_curve.csv", index=False)
print(p2_curve.round(5).to_string())
"""),
    md("## P3. Sensor subsets vs power (exhaustive)"),
    code("""
# optional sensors -> their feature channels, and an assumed average power (W). TODO_VERIFY every power number
SENSORS = {"BMP280": (["p"], 0.003), "fans": (["wind_u", "wind_v"], 0.01), "rain": (["rain_frac"], 0.01),
           "BH1750": (["ln_lux"], 0.001), "MQ-7": (["mq7_n"], 0.35), "MQ-135": (["mq135_n"], 0.80),
           "SGP41": (["voc_n", "nox_n"], 0.01)}
CORE = ["ln_pm25", "ln_pm10", "ln_pm1", "coarse_ratio", "t", "rh", "aqi_now", "pm_mask", "hod_sin", "hod_cos",
        "dow_sin", "dow_cos", "doy_sin", "doy_cos"]
CORE_W = 0.55          # PM sensor + HDC1080 + ESP32 (TODO_VERIFY)

# per channel: now, 1-h and 3-h change, deviation from its 24-h mean (the node's own history only)
def channel_feats(name, hours):
    a = F[name]
    m24 = pd.DataFrame(a).rolling(24, min_periods=12).mean().to_numpy()
    return {f"{name}": a[hours].ravel(), f"{name}_d1": lag_diff(a, 1)[hours].ravel(),
            f"{name}_d3": lag_diff(a, 3)[hours].ravel(), f"{name}_m24": (a - m24)[hours].ravel()}

H3_TR, H3_VA = H_TRAIN[::4], H_VAL[::2]
def table(hours):
    cols = {}
    for name in FEATURES:
        cols.update(channel_feats(name, hours))
    return pd.DataFrame(cols)

TAB = {k: table(h) for k, h in (("tr", H3_TR), ("va", H3_VA), ("te", H_TEST))}
Y3 = {k: TARGETS["pm25_h3"][h].ravel() for k, h in (("tr", H3_TR), ("va", H3_VA), ("te", H_TEST))}
"""),
    code("""
# all 128 subsets: LightGBM (fixed 400 trees) on core + chosen sensors, test skill PM2.5 +3 h
from itertools import product
names = list(SENSORS)
p3_rows = []
subsets = list(product([0, 1], repeat=len(names)))
for bits in subsets if not SMOKE else subsets[:3]:
    chosen = [n for n, b in zip(names, bits) if b]
    chans = CORE + [c for n in chosen for c in SENSORS[n][0]]
    cols = [c for c in TAB["tr"].columns if c.rsplit("_", 1)[0] in chans or c in chans]
    a = np.isfinite(Y3["tr"])
    m = lgb.train(dict(PARAMS, num_leaves=31), lgb.Dataset(TAB["tr"].loc[a, cols], Y3["tr"][a]), 400 if not SMOKE else 10)
    ok = np.isfinite(Y3["te"])
    pred = m.predict(TAB["te"].loc[ok, cols])
    skill = 100 * (1 - np.sqrt(np.mean((pred - Y3["te"][ok]) ** 2) / np.mean(Y3["te"][ok] ** 2)))
    p3_rows.append(dict(sensors="+".join(chosen) or "core only", n_sensors=len(chosen),
                        power_w=CORE_W + sum(SENSORS[n][1] for n in chosen), skill=skill))
p3 = pd.DataFrame(p3_rows).sort_values("power_w")
p3["pareto"] = p3.skill >= p3.skill.cummax()
p3.to_csv(OUT / "p3_sensor_subsets.csv", index=False)
print(p3[p3.pareto].round(3).to_string())
print("all sensors:", p3[p3.n_sensors == len(names)].round(3).to_string())
"""),
]


def build(smoke):
    cells = [dict(c) for c in CELLS]
    if smoke:
        cells[1] = dict(cells[1], source=cells[1]["source"].replace("SMOKE = False", "SMOKE = True"))
    return notebook(cells)
