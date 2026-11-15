# Experiment 3 (CPU, TensorFlow): what can make the edge student better? (same student, same data as notebook 04)
#   S0        the plain GDC-CNN student (baseline, re-trained here with the same loop as every arm)
#   S0+ctx    + the raw "regime" numbers that instance norm throws away (PM level, RH, wind speed, clock, season, CO)
#   FZ        fuzzy Takagi-Sugeno head (ANFIS-style): 4 rules with Gaussian memberships on the same regime numbers,
#             each rule has its own linear forecast; output = firing-weighted mean. Control = S0+ctx.
#   S0+wx     + weather the cloud (or Open-Meteo) could send: now + next 3 h (ERA5 actuals = oracle, a hindcast)
#   S0+wxnow  + weather now and 3 h ago only (no forecast)
#   FZ+wx     fuzzy head + weather forecast
# 3 seeds each; test = notebook 04's student test windows; week-block paired bootstrap vs S0.
from nb import nb04_cells, code, md, notebook

GPU = False
DATASETS = ["samiulseikh/aqi-features"]

SETUP = """
# imports, settings
import gc, time, json
from pathlib import Path
import numpy as np, pandas as pd
import tensorflow as tf, keras
from keras import layers

SMOKE = False
ON_KAGGLE = Path("/kaggle/input").exists()
FEAT = next(Path("/kaggle/input").rglob("grid_u0.npz")).parent if ON_KAGGLE else Path("../data/4_features")
OUT = Path("/kaggle/working") if ON_KAGGLE else Path("../data/6_experiments/student")
OUT.mkdir(parents=True, exist_ok=True)
MODEL, D, LR, BATCH, EPOCHS, PATIENCE = "A", 32, 1e-3, 256, (2 if SMOKE else 60), 8     # d, lr = the tuned best of 04
SEEDS = [0] if SMOKE else [0, 1, 2]
K_RULES = 4
H = 6
NOTEBOOK_START = time.time()
"""

DATA = """
# windows: the same 200 000 train / 40 000 val distillation windows and the same test windows as notebook 04
kd_windows = pd.read_parquet(FEAT / "kd_windows.parquet")
train_w = kd_windows[kd_windows.split == "train"].reset_index(drop=True)
val_w = kd_windows[kd_windows.split == "val"].reset_index(drop=True)
pool = pd.read_parquet(FEAT / "windows_A.parquet")
pool = pool[(pool.split == "test") & (pool.row < len(grids[0]["ts"]) - H)]
test_w = pool[~pool.holdout].sample(150_000, random_state=0).reset_index(drop=True)
unseen_w = pool[pool.holdout].sample(100_000, random_state=0).reset_index(drop=True)
if SMOKE:
    train_w, val_w, test_w, unseen_w = train_w.head(4000), val_w.head(2000), test_w.head(3000), unseen_w.head(3000)
WX = np.load(FEAT / "grid_u0.npz")["W"]          # ERA5 weather [hour, station, 7] (the same for every unit)
"""

SIDE = """
# side inputs of a window. ctx = raw regime numbers at the current hour (instance norm removes the level);
# wx = weather now + next 3 h; wxnow = weather now + 3 h ago
CTX = ["ln_pm25", "rh", "t", "mq7_n", "hod_sin", "hod_cos", "doy_sin", "doy_cos"]
def side(w):
    rows, st, u = w.row.to_numpy(), w.station.to_numpy(), w.unit.to_numpy()
    X = np.stack([grids[k]["X"][rows, st] for k in range(3)])[u, np.arange(len(rows))]       # [n, 23]
    wind = np.hypot(X[:, FEATURES.index("wind_u")], X[:, FEATURES.index("wind_v")])
    ctx = np.column_stack([X[:, FEATURES.index(f)] for f in CTX] + [wind])
    wx = np.concatenate([WX[rows + h, st] for h in (0, 1, 2, 3)], axis=1)
    wxnow = np.concatenate([WX[rows + h, st] for h in (0, -3)], axis=1)
    return {k: np.nan_to_num(v).astype(np.float32) for k, v in (("ctx", ctx), ("wx", wx), ("wxnow", wxnow))}

def make_set(w):
    d = get_xy(w)
    patches, scale = preprocess(d["x"])
    return {"x": patches, "y": d["y"].astype(np.float32), "scale": scale, "month": d["month"],
            "row": w.row.to_numpy(), "station": w.station.to_numpy(), "unit": w.unit.to_numpy(), **side(w)}

train, val, test, unseen = [make_set(w) for w in (train_w, val_w, test_w, unseen_w)]
NORM = {k: (train[k].mean(0), train[k].std(0) + 1e-6) for k in ("ctx", "wx", "wxnow")}   # training statistics
def normalise(s):
    for k, (mu, sd) in NORM.items():
        s[k] = (s[k] - mu) / sd
    return s
for s in (train, val, test, unseen):
    normalise(s)
print({k: v.shape for k, v in train.items() if hasattr(v, "shape")})
"""

FUZZY = """
# fuzzy Takagi-Sugeno gate: rule k fires with strength prod_i exp(-(c_i - centre_ki)^2 / (2 width_ki^2)),
# normalised over the rules (softmax of the log firing). Centres start spread over the data (random training rows).
class FuzzyGate(layers.Layer):
    def __init__(self, k, init_centres, **kw):
        super().__init__(**kw)
        self.k, self.init_centres = k, init_centres

    def build(self, shape):
        self.centre = self.add_weight(shape=(self.k, shape[-1]), initializer=keras.initializers.Constant(self.init_centres), name="centre")
        self.log_width = self.add_weight(shape=(self.k, shape[-1]), initializer="zeros", name="log_width")

    def call(self, c):
        z = (c[:, None, :] - self.centre[None]) / keras.ops.exp(self.log_width)[None]
        return keras.ops.softmax(-0.5 * keras.ops.sum(z * z, axis=-1), axis=-1)       # [batch, k]
"""

MODEL_CELL = """
# one student for every arm: GDC-CNN trunk (as in 04) + optional side input + plain or fuzzy output head
ARMS = {"S0": (None, False), "S0+ctx": ("ctx", False), "FZ": (None, True), "S0+wx": ("wx", False),
        "S0+wxnow": ("wxnow", False), "FZ+wx": ("wx", True)}

def build(arm, seed):
    side_name, fuzzy = ARMS[arm]
    keras.utils.set_random_seed(seed)
    x_in, s_in = keras.Input((12, train["x"].shape[2])), keras.Input((6,))
    c_in = keras.Input((train["ctx"].shape[1],))
    w_in = keras.Input((train[side_name].shape[1],)) if side_name else None
    x = PositionCode()(layers.Dense(D)(x_in))
    for dilation in (1, 2, 4):
        x = gdc_block(x, D, dilation)
    h = layers.Flatten()(x)
    if side_name:
        h = layers.Concatenate()([h, layers.Dense(D, activation="gelu")(w_in)])
    if fuzzy:
        rng = np.random.default_rng(seed)
        gate = FuzzyGate(K_RULES, train["ctx"][rng.choice(len(train["ctx"]), K_RULES, replace=False)])(c_in)
        experts = layers.Reshape((K_RULES, 6))(layers.Dense(6 * K_RULES)(h))
        y = keras.ops.sum(experts * gate[:, :, None], axis=1)
    else:
        y = layers.Dense(6)(h)
    inputs = [x_in, s_in] + ([c_in] if fuzzy else []) + ([w_in] if side_name else [])
    return keras.Model(inputs, y * s_in)

def inputs_of(arm, s):
    side_name = ARMS[arm][0]
    return [s["x"], s["scale"]] + ([s["ctx"]] if ARMS[arm][1] else []) + ([s[side_name]] if side_name else [])

def train_arm(arm, seed):
    model = build(arm, seed)
    model.compile(keras.optimizers.Adam(LR), loss=keras.losses.Huber(1.0), metrics=[keras.metrics.RootMeanSquaredError()])
    stop = keras.callbacks.EarlyStopping("val_root_mean_squared_error", patience=PATIENCE, restore_best_weights=True)
    model.fit(inputs_of(arm, train), train["y"], validation_data=(inputs_of(arm, val), val["y"]),
              batch_size=BATCH, epochs=EPOCHS, callbacks=[stop], verbose=2)
    return model
"""

RUN = """
# train every arm x seed; keep the test forecasts (seen + unseen)
H13 = slice(0, 6)
preds, rows = {}, []
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer", 6: "monsoon", 7: "monsoon",
          8: "monsoon", 9: "monsoon", 10: "post-monsoon", 11: "post-monsoon"}
def skill(p, s, keep=None):
    keep = np.ones(len(p), bool) if keep is None else keep
    return 100 * (1 - np.sqrt(((p - s["y"]) ** 2)[keep].mean() / (s["y"] ** 2)[keep].mean()))

for seed in SEEDS:
    for arm in ARMS:
        if time.time() - NOTEBOOK_START > 11 * 3600:
            continue
        start = time.time()
        model = train_arm(arm, seed)
        for name, s in (("seen", test), ("unseen", unseen)):
            p = model.predict(inputs_of(arm, s), batch_size=4096, verbose=0)
            preds[(arm, seed, name)] = p
            row = dict(arm=arm, seed=seed, stations=name, skill_all=skill(p, s),
                       params=model.count_params(), minutes=round((time.time() - start) / 60, 1))
            row.update({f"skill_{o}": skill(p[:, i:i + 1], dict(y=s["y"][:, i:i + 1])) for i, o in enumerate(
                ["pm25_h1", "pm25_h2", "pm25_h3", "pm10_h1", "pm10_h2", "pm10_h3"])})
            season = np.vectorize(SEASON.get)(s["month"])
            row.update({f"skill_{q}": skill(p, s, season == q) for q in ("winter", "summer", "monsoon", "post-monsoon")})
            rows.append(row)
        print(rows[-2], flush=True)
        if seed == 0:
            np.savez_compressed(OUT / f"pred_{arm}.npz", y_hat=preds[(arm, 0, "seen")], row=test["row"],
                                station=test["station"], unit=test["unit"])
            # every test hour of unit 0 at the 27 training stations (for the edge-cloud fusion experiment)
            if "full0" not in globals():
                fw = pool[(pool.unit == 0) & ~pool.holdout].reset_index(drop=True)
                full0 = normalise(make_set(fw.head(3000) if SMOKE else fw))
            pf = model.predict(inputs_of(arm, full0), batch_size=4096, verbose=0)
            np.savez_compressed(OUT / f"pred_full0_{arm}.npz", y_hat=pf.astype(np.float16), row=full0["row"],
                                station=full0["station"], y=full0["y"])
        if arm.startswith("FZ") and seed == 0:          # the learned rules, in z-units of CTX + wind speed
            gate = [l for l in model.layers if isinstance(l, FuzzyGate)][0]
            pd.DataFrame(np.asarray(gate.centre), columns=CTX + ["wind_speed"]).to_csv(OUT / f"rules_{arm}.csv")
        pd.DataFrame(rows).to_csv(OUT / "student_results.csv", index=False)
        del model
        gc.collect()
np.savez_compressed(OUT / "truth_test.npz", y=test["y"], row=test["row"], station=test["station"], unit=test["unit"])
"""

SUMMARY = """
# summary over seeds, and paired week-block bootstrap of every arm vs S0 (seed-averaged forecasts, seen stations)
res = pd.DataFrame(rows)
summary = res.groupby(["arm", "stations"]).agg(["mean", "std"]).drop(columns=["seed"], level=0).round(2)
summary.to_csv(OUT / "student_summary.csv")
print(res.groupby(["stations", "arm"]).skill_all.agg(["mean", "std", "count"]).round(2).to_string())

def mean_pred(arm):
    ps = [preds[(arm, s, "seen")] for s in SEEDS if (arm, s, "seen") in preds]
    return np.mean(ps, axis=0)

week = (test["row"] - test["row"].min()) // 168
def boot(a, b="S0", n=4000):
    sa = np.bincount(week, ((mean_pred(a) - test["y"]) ** 2).sum(1))
    sb = np.bincount(week, ((mean_pred(b) - test["y"]) ** 2).sum(1))
    idx = np.random.default_rng(0).integers(0, len(sa), (n, len(sa)))
    d = 100 * (np.sqrt(sa[idx].sum(1) / sb[idx].sum(1)) - 1)
    return dict(A=a, B=b, rmse_change_pct=round(100 * (np.sqrt(sa.sum() / sb.sum()) - 1), 3),
                ci_low=round(np.percentile(d, 2.5), 3), ci_high=round(np.percentile(d, 97.5), 3))

tests = pd.DataFrame([boot(a) for a in ARMS if a != "S0"] + [boot("FZ", "S0+ctx"), boot("FZ+wx", "S0+wx"), boot("S0+wx", "S0+wxnow")])
tests.to_csv(OUT / "student_bootstrap.csv", index=False)
print("negative = A has the smaller error; CI containing 0 = no real difference")
print(tests.to_string())
"""


def build(smoke):
    nb4 = nb04_cells()
    cells = [md("# Experiment 3: improving the edge student (fuzzy TSK head, regime context, weather inputs)"),
             code(SETUP.replace("SMOKE = False", "SMOKE = True") if smoke else SETUP),
             nb4[76], nb4[79], code(DATA), nb4[82], code(SIDE), nb4[85], nb4[86], nb4[87],
             code(FUZZY), code(MODEL_CELL), code(RUN), code(SUMMARY)]
    return notebook([dict(c) for c in cells])
