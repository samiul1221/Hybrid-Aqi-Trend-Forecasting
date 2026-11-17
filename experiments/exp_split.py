# Experiment 6 (CPU): using the NETWORK of boxes to forecast better, and custom losses.
# Idea from experiments 2 and 5: the boxes mostly share one regional (Delhi-wide) signal; each cheap sensor is noisy.
#   cloud model   LightGBM forecasts the REGIONAL change (mean over the boxes of the true ln-PM change, +1..+3 h, PM2.5 and
#                 PM10) from regional means of the boxes' own sensors + current ERA5 (no future weather). Out-of-fold on
#                 the training years (3 folds of 2 years), so the student never learns from a forecast that saw the answer.
#   arms (student = GDC-CNN + regime context, the best of experiment 3):
#     ctx          baseline
#     denoise      + consensus filter (sensor-network idea): d_i = R(t) + EWMA of the box's past offset from R
#     split        + the cloud's 6 regional forecasts (6 bytes) + gap to the region, as inputs
#     split-res    split ARCHITECTURE: output = regional forecast + the box's local correction (common-mode rejection)
#     ctx-trend    custom loss: Huber + 0.5 x hinge on the wrong DIRECTION (AQI trend) when |change| > 0.05
#     ctx-aqi      custom loss: errors x (1 + 2 exp(-d^2 / 2 0.15^2)), d = ln-distance of the true PM to a CPCB breakpoint
# 3 seeds; metrics: skill, direction accuracy, CPCB category accuracy; week-block bootstrap vs ctx.
from nb import nb04_cells, code, md, notebook
import exp_student as es

GPU = False
DATASETS = ["samiulseikh/aqi-features"]

REGION = """
# regional means of the boxes' own sensors (training stations, each unit = its own network of boxes)
import lightgbm as lgb
g0 = np.load(FEAT / "grid_u0.npz")
TRAIN_ST, SPLIT = np.flatnonzero(~g0["holdout"]), g0["split"]
W = g0["W"]
T_ALL, NS = len(SPLIT), len(g0["stations"])
YEAR = pd.to_datetime(g0["ts"], unit="s", utc=True).tz_convert("Asia/Kolkata").year.to_numpy()

def sensor(u, name):
    X = grids[u]["X"]
    ok = X[:, :, FEATURES.index("pm_mask")] > 0 if name.startswith("ln_pm") else np.isfinite(X[:, :, FEATURES.index(name)])
    return np.where(ok, X[:, :, FEATURES.index(name)], np.nan)

S25 = {u: sensor(u, "ln_pm25") for u in range(3)}
S10 = {u: sensor(u, "ln_pm10") for u in range(3)}
R25 = {u: np.nanmean(S25[u][:, TRAIN_ST], 1) for u in range(3)}
R10 = {u: np.nanmean(S10[u][:, TRAIN_ST], 1) for u in range(3)}
LOGPM = np.log1p(grids[0]["pm"][:, TRAIN_ST])           # true PM (the same for every unit): regional TARGET only
OUTPUTS = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]      # (pollutant, hours ahead) the cloud forecasts
def lagged(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k] if k else a
    return out
"""

CLOUD = """
# the cloud's regional model: features of hour t (past only), targets = mean true change over the boxes, 6 outputs
def region_table(u):
    cols = {}
    for name, R in (("r25", R25[u]), ("r10", R10[u])):
        for k in (0, 1, 2, 3, 6, 12, 23):
            cols[f"{name}_l{k}"] = lagged(R, k)
        cols[f"{name}_d1"], cols[f"{name}_d3"] = R - lagged(R, 1), R - lagged(R, 3)
    for name in ("rh", "t", "mq7_n", "wind_u", "wind_v"):
        cols[f"reg_{name}"] = np.nanmean(sensor(u, name)[:, TRAIN_ST], 1)
    for k, name in enumerate(("e_t", "e_rh", "e_p", "e_u", "e_v", "e_rain", "e_sw")):
        cols[name] = np.nanmean(W[:, TRAIN_ST, k], 1)
    for name in ("hod_sin", "hod_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"):
        cols[name] = grids[u]["X"][:, 0, FEATURES.index(name)]
    return pd.DataFrame(cols)

TARGET = np.full((T_ALL, len(OUTPUTS)), np.nan)
for j, (p, h) in enumerate(OUTPUTS):
    TARGET[:-h, j] = np.nanmean(LOGPM[h:, :, p] - LOGPM[:-h, :, p], 1)

TAB = {u: region_table(u) for u in range(3)}
PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, verbose=-1, num_threads=4)
FOLDS = [(2018, 2019), (2020, 2021), (2022, 2023)]
RHAT = {u: np.full((T_ALL, len(OUTPUTS)), np.nan, np.float32) for u in range(3)}
is_train = SPLIT == "train"
cloud_rows = []
for j in range(len(OUTPUTS)):
    def fit(hours):
        x = pd.concat([TAB[u].iloc[hours] for u in range(3)])
        y = np.concatenate([TARGET[hours, j]] * 3)
        ok = np.isfinite(y)
        return lgb.train(PARAMS, lgb.Dataset(x[ok], y[ok]), 300 if not SMOKE else 10)
    for fold in FOLDS:                                     # out-of-fold forecasts for the training years
        held = is_train & np.isin(YEAR, fold)
        m = fit(np.flatnonzero(is_train & ~held))
        for u in range(3):
            RHAT[u][held, j] = m.predict(TAB[u][held])
    m = fit(np.flatnonzero(is_train))                      # val / test: model trained on all training years
    for u in range(3):
        RHAT[u][~is_train, j] = m.predict(TAB[u][~is_train])
    ok = (SPLIT == "test") & np.isfinite(TARGET[:, j])
    cloud_rows.append(dict(output=j, regional_skill=100 * (1 - np.sqrt(np.mean((RHAT[0][ok, j] - TARGET[ok, j]) ** 2) / np.mean(TARGET[ok, j] ** 2)))))
print(pd.DataFrame(cloud_rows).round(2).to_string())
"""

DENOISE = """
# consensus filter: the box's level = region now + its own usual offset from the region (EWMA of past hours, 24 h)
DEN = {}
for u in range(3):
    off = pd.DataFrame(S25[u] - R25[u][:, None]).ewm(halflife=24, ignore_na=True).mean().shift(1).to_numpy()
    DEN[u] = R25[u][:, None] + off                       # [hour, station]

# all side features of a window (row = hour t, station, unit), stacked by unit
def net_features(s):
    n = len(s["row"])
    rows, st, un = s["row"], s["station"], s["unit"]
    f = {"split": np.zeros((n, 8), np.float32), "den": np.zeros((n, 3), np.float32), "rhat": np.zeros((n, 6), np.float32)}
    for u in range(3):
        k = un == u
        r, i = rows[k], st[k]
        own25, own10 = S25[u][r, i], S10[u][r, i]
        f["rhat"][k] = np.nan_to_num(RHAT[u][r])
        f["split"][k] = np.nan_to_num(np.column_stack([RHAT[u][r] * 3, R25[u][r] - own25, R10[u][r] - own10]))
        d = DEN[u]
        f["den"][k] = np.nan_to_num(np.column_stack([d[r, i] - own25, (d[r, i] - d[r - 1, i]) * 3, (d[r, i] - d[r - 3, i]) * 3]))
    return f

# true PM at the base hour (for the AQI-aware loss weight and the category metric only)
def base_pm(s):
    return np.stack([grids[u]["pm"][s["row"], s["station"]] for u in range(3)])[s["unit"], np.arange(len(s["row"]))]

BP = {0: np.log1p([30, 60, 90, 120, 250]), 1: np.log1p([50, 100, 250, 350, 430])}     # CPCB 24-h breakpoints
def aqi_weight(s):
    w = np.ones_like(s["y"])
    for j in range(6):
        p = 0 if j < 3 else 1
        fut = np.log1p(np.nan_to_num(s["base"][:, p])) + s["y"][:, j]
        dist = np.min(np.abs(fut[:, None] - BP[p][None]), axis=1)
        w[:, j] = 1 + 2 * np.exp(-dist ** 2 / (2 * 0.15 ** 2))
    return w.astype(np.float32)

for s in (train, val, test, unseen):
    s.update(net_features(s))
    s["base"] = base_pm(s)
    s["w"] = aqi_weight(s)
"""

MODEL = """
# arms: which side inputs, split-residual output, which loss
ARMS = {"ctx": ([], False, "huber"), "denoise": (["den"], False, "huber"), "split": (["split"], False, "huber"),
        "split-res": (["split"], True, "huber"), "ctx-trend": ([], False, "trend"), "ctx-aqi": ([], False, "aqi")}

def build(arm, seed):
    sides, residual, _ = ARMS[arm]
    keras.utils.set_random_seed(seed)
    x_in, s_in, c_in = keras.Input((12, train["x"].shape[2])), keras.Input((6,)), keras.Input((train["ctx"].shape[1],))
    x = PositionCode()(layers.Dense(D)(x_in))
    for dilation in (1, 2, 4):
        x = gdc_block(x, D, dilation)
    parts, inputs = [layers.Flatten()(x), layers.Dense(D, activation="gelu")(c_in)], [x_in, s_in, c_in]
    for name in sides:
        z = keras.Input((train[name].shape[1],))
        parts.append(layers.Dense(D, activation="gelu")(z))
        inputs.append(z)
    y = layers.Dense(6)(layers.Concatenate()(parts)) * s_in
    if residual:                                        # regional forecast + local correction
        r_in = keras.Input((6,))
        inputs.append(r_in)
        y = y + r_in
    return keras.Model(inputs, y)

def inputs_of(arm, s):
    sides, residual, _ = ARMS[arm]
    return [s["x"], s["scale"], s["ctx"]] + [s[n] for n in sides] + ([s["rhat"]] if residual else [])

# losses take y_true = [y (6), weight (6)]
def make_loss(kind):
    def loss(yt, yp):
        y, w = yt[:, :6], yt[:, 6:]
        d = keras.ops.abs(yp - y)
        hub = keras.ops.where(d <= 1.0, 0.5 * d * d, d - 0.5)
        if kind == "aqi":
            return keras.ops.sum(w * hub) / keras.ops.sum(w)
        out = keras.ops.mean(hub)
        if kind == "trend":
            moving = keras.ops.cast(keras.ops.abs(y) > 0.05, "float32")
            out = out + 0.5 * keras.ops.mean(moving * keras.ops.relu(0.02 - yp * keras.ops.sign(y)))
        return out
    return loss

def rmse6(yt, yp):
    return keras.ops.sqrt(keras.ops.mean((yp - yt[:, :6]) ** 2))

def train_arm(arm, seed):
    model = build(arm, seed)
    model.compile(keras.optimizers.Adam(LR), loss=make_loss(ARMS[arm][2]), metrics=[rmse6])
    stop = keras.callbacks.EarlyStopping("val_rmse6", patience=PATIENCE, restore_best_weights=True, mode="min")
    yt = lambda s: np.concatenate([s["y"], s["w"]], 1)
    model.fit(inputs_of(arm, train), yt(train), validation_data=(inputs_of(arm, val), yt(val)),
              batch_size=BATCH, epochs=EPOCHS, callbacks=[stop], verbose=2)
    return model
"""

RUN = """
# train every arm x seed; skill, direction accuracy, CPCB category accuracy (forecast PM from the true base PM)
def category(pm, p):
    return np.searchsorted(np.expm1(BP[p]), pm)
def metrics(p, s):
    out = dict(skill=100 * (1 - np.sqrt(((p - s["y"]) ** 2).mean() / (s["y"] ** 2).mean())),
               skill_pm25_h1=100 * (1 - np.sqrt(((p[:, 0] - s["y"][:, 0]) ** 2).mean() / (s["y"][:, 0] ** 2).mean())))
    moving = np.abs(s["y"]) > 0.05
    out["direction_acc"] = 100 * (np.sign(p) == np.sign(s["y"]))[moving].mean()
    hits = []
    for j in range(6):
        pol = 0 if j < 3 else 1
        b = s["base"][:, pol]
        ok = np.isfinite(b)
        hits.append(category((1 + b[ok]) * np.exp(p[ok, j]) - 1, pol) == category((1 + b[ok]) * np.exp(s["y"][ok, j]) - 1, pol))
    out["category_acc"] = 100 * np.concatenate(hits).mean()
    return out

rows, preds = [], {}
for seed in SEEDS:
    for arm in ARMS:
        if time.time() - NOTEBOOK_START > 11 * 3600:
            continue
        start = time.time()
        model = train_arm(arm, seed)
        for name, s in (("seen", test), ("unseen", unseen)):
            p = model.predict(inputs_of(arm, s), batch_size=4096, verbose=0)
            preds[(arm, seed, name)] = p
            rows.append(dict(arm=arm, seed=seed, stations=name, minutes=round((time.time() - start) / 60, 1), **metrics(p, s)))
        print(rows[-2], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "split_results.csv", index=False)
        del model
        gc.collect()
# reference rows: the cloud's regional forecast used directly as every box's forecast, and persistence
for name, s in (("seen", test), ("unseen", unseen)):
    rows.append(dict(arm="cloud-regional-only", seed=0, stations=name, **metrics(s["rhat"], s)))
pd.DataFrame(rows).to_csv(OUT / "split_results.csv", index=False)
"""

SUMMARY = """
res = pd.DataFrame(rows)
print(res.groupby(["stations", "arm"])[["skill", "skill_pm25_h1", "direction_acc", "category_acc"]].agg(["mean", "std"]).round(2).to_string())
week = (test["row"] - test["row"].min()) // 168
def mean_pred(arm):
    return np.mean([preds[k] for k in preds if k[0] == arm and k[2] == "seen"], axis=0)
def boot(a, b="ctx", n=4000):
    sa = np.bincount(week, ((mean_pred(a) - test["y"]) ** 2).sum(1))
    sb = np.bincount(week, ((mean_pred(b) - test["y"]) ** 2).sum(1))
    idx = np.random.default_rng(0).integers(0, len(sa), (n, len(sa)))
    d = 100 * (np.sqrt(sa[idx].sum(1) / sb[idx].sum(1)) - 1)
    return dict(A=a, B=b, rmse_change_pct=round(100 * (np.sqrt(sa.sum() / sb.sum()) - 1), 3),
                ci_low=round(np.percentile(d, 2.5), 3), ci_high=round(np.percentile(d, 97.5), 3))
tests = pd.DataFrame([boot(a) for a in ARMS if a != "ctx"] + [boot("split-res", "split")])
tests.to_csv(OUT / "split_bootstrap.csv", index=False)
print("negative = smaller error than B; CI containing 0 = no real difference")
print(tests.to_string())
"""


def build(smoke):
    nb4 = nb04_cells()
    setup = es.SETUP.replace("SMOKE = False", "SMOKE = True") if smoke else es.SETUP
    setup = setup.replace('Path("../data/6_experiments/student")', 'Path("../data/6_experiments/split")')
    cells = [md("# Experiment 6: split architecture (regional + local), consensus filter, custom losses"),
             code(setup), nb4[76], nb4[79], code(es.DATA), nb4[82], code(es.SIDE), nb4[85], nb4[86], nb4[87],
             code(REGION), code(CLOUD), code(DENOISE), code(MODEL), code(RUN), code(SUMMARY)]
    return notebook([dict(c) for c in cells])
