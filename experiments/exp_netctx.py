# Experiment 5 (CPU): network ideas for the box. Neighbours only help as a REGIONAL level (experiment 2), so:
#   N1  shared-bottleneck clustering (network tomography idea): which stations share one regional signal?
#   N2  push-sum GOSSIP (distributed averaging): boxes agree on the regional mean by radio, without the cloud.
#       Accuracy vs rounds and packet loss.
#   N3  students that get ONE number per hour: the regional mean of the boxes' own sensor ln PM2.5 (no true PM),
#       8-bit quantised (1 byte), from the cloud (Delhi-wide or per cluster) or from gossip. The box derives
#       gap = R - own, and trends of R from the values it has received. Trained with 20 % missing and 0-3 h old
#       context (with an age input), then tested fresh, old (1-3 h) and missing (no-regret check).
# Baseline = S0+ctx (best student of experiment 3).
from nb import nb04_cells, code, md, notebook
import exp_student as es

GPU = False
DATASETS = ["samiulseikh/aqi-features"]

NET = """
# sensor ln PM2.5 of every box (each unit), training stations, geometry
TRAIN_ST = np.flatnonzero(~np.load(FEAT / "grid_u0.npz")["holdout"])
static = np.load(FEAT / "grid_u0.npz")["static"]
SENS = {u: np.where(np.isfinite(grids[u]["X"][:, :, 0]) & (grids[u]["X"][:, :, FEATURES.index("pm_mask")] > 0),
                    grids[u]["X"][:, :, 0], np.nan) for u in range(3)}
la, lo = np.deg2rad(static[:, 0]), np.deg2rad(static[:, 1])
DIST = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2))
NS = DIST.shape[0]
train_hours = np.flatnonzero(np.load(FEAT / "grid_u0.npz")["split"] == "train")
"""

CLUSTER = """
# N1: cluster the training stations by how much their 3-h PM changes move together (1 - correlation), 4 clusters
from scipy.cluster.hierarchy import linkage, fcluster
s = SENS[0][train_hours][:, TRAIN_ST]
d3 = pd.DataFrame(s[3:] - s[:-3])
corr = d3.corr(min_periods=1000).to_numpy()
cond = (1 - corr)[np.triu_indices(len(TRAIN_ST), 1)]
lab = fcluster(linkage(cond, "average"), 4, "maxclust")
CLUSTER = np.zeros(NS, int)
CLUSTER[TRAIN_ST] = lab
for i in np.flatnonzero(CLUSTER == 0):                     # never-seen stations join the nearest training station's cluster
    CLUSTER[i] = CLUSTER[TRAIN_ST[np.argmin(DIST[i, TRAIN_ST])]]
stations_all = list(np.load(FEAT / "grid_u0.npz")["stations"])
clusters = pd.DataFrame(dict(station=stations_all, cluster=CLUSTER, lat=static[:, 0], lon=static[:, 1]))
clusters.to_csv(OUT / "N1_clusters.csv", index=False)
print(clusters.sort_values("cluster").to_string())
within = np.mean([corr[a, b] for a in range(len(lab)) for b in range(len(lab)) if a != b and lab[a] == lab[b]])
between = np.mean([corr[a, b] for a in range(len(lab)) for b in range(len(lab)) if lab[a] != lab[b]])
print(f"mean correlation of 3-h changes: within clusters {within:.3f}, between {between:.3f}")
"""

GOSSIP = """
# N2: push-sum gossip among the training-station boxes. Radio link if closer than RANGE_KM (LoRa in a city,
# TODO_VERIFY on real hardware). Each round every box keeps half of (sum, weight) and sends half to one random
# neighbour; a message is lost with probability `loss`. Estimate = sum / weight. Vectorised over all hours.
RANGE_KM = 10.0
sub = DIST[np.ix_(TRAIN_ST, TRAIN_ST)]
NEIGH = [np.flatnonzero((sub[i] < RANGE_KM) & (sub[i] > 0)) for i in range(len(TRAIN_ST))]
DEG = np.array([len(n) for n in NEIGH])
PAD = np.array([np.resize(n, max(DEG)) if len(n) else np.full(max(DEG), i) for i, n in enumerate(NEIGH)])
print("radio neighbours per box:", DEG.tolist())

def gossip(x, rounds, loss, seed=0):
    rng = np.random.default_rng(seed)
    T_, n = x.shape
    ok = np.isfinite(x)
    s, w = np.where(ok, x, 0.0), ok.astype(float)
    for _ in range(rounds):
        to = PAD[np.arange(n)[None, :], rng.integers(0, 1 << 30, (T_, n)) % np.maximum(DEG, 1)[None, :]]
        sent = (rng.random((T_, n)) >= loss) & (DEG[None, :] > 0)
        s_new, w_new = s / 2 + np.where(DEG[None, :] == 0, s / 2, 0), w / 2 + np.where(DEG[None, :] == 0, w / 2, 0)
        rows = np.repeat(np.arange(T_), n)[sent.ravel()]
        np.add.at(s_new, (rows, to.ravel()[sent.ravel()]), (s / 2).ravel()[sent.ravel()])
        np.add.at(w_new, (rows, to.ravel()[sent.ravel()]), (w / 2).ravel()[sent.ravel()])
        s, w = s_new, w_new
    return np.where(w > 1e-9, s / np.maximum(w, 1e-9), np.nan)

# accuracy of gossip vs the true mean of the box's radio component (and vs the Delhi-wide mean), test hours
test_h = np.flatnonzero(np.load(FEAT / "grid_u0.npz")["split"] == "test")[::5]
xs = SENS[0][test_h][:, TRAIN_ST]
delhi = np.nanmean(xs, 1, keepdims=True)
rows = []
for loss in (0.0, 0.2, 0.5):
    for rounds in (1, 3, 5, 10, 20, 30):
        est = gossip(xs, rounds, loss)
        rows.append(dict(loss=loss, rounds=rounds, rmse_vs_delhi_mean=float(np.sqrt(np.nanmean((est - delhi) ** 2))),
                         rmse_own_vs_delhi=float(np.sqrt(np.nanmean((xs - delhi) ** 2)))))
gacc = pd.DataFrame(rows)
gacc.to_csv(OUT / "N2_gossip_accuracy.csv", index=False)
print(gacc.round(4).to_string())
"""

REGIONAL = """
# N3: the regional number R[t, station] of every unit (ln PM2.5 units), for each source
GOSSIP_ROUNDS, GOSSIP_LOSS = 20, 0.2      # ~20 short LoRa messages per box per hour
def regional(u):
    x = SENS[u][:, TRAIN_ST]
    out = {"delhi": np.repeat(np.nanmean(x, 1, keepdims=True), NS, 1)}
    clus = np.full((x.shape[0], NS), np.nan)
    for c in np.unique(CLUSTER):
        members = np.isin(TRAIN_ST, np.flatnonzero(CLUSTER == c))
        clus[:, CLUSTER == c] = np.nanmean(x[:, members], 1, keepdims=True)
    out["clus"] = clus
    g = np.full((x.shape[0], NS), np.nan)
    g[:, TRAIN_ST] = gossip(x, GOSSIP_ROUNDS, GOSSIP_LOSS, seed=u)
    for i in np.flatnonzero(~np.isin(np.arange(NS), TRAIN_ST)):          # a new box listens to its nearest gossiping box
        g[:, i] = g[:, TRAIN_ST[np.argmin(DIST[i, TRAIN_ST])]]
    out["gossip"] = g
    return out

REG = {u: regional(u) for u in range(3)}
quant = lambda r: np.round(np.clip((r - 0.0) / 8.0, 0, 1) * 255) / 255 * 8.0      # 1 byte over ln PM 0..8
print("8-bit step in ln PM:", round(8 / 255, 4))
"""

SIDE_REG = """
# the box's regional features of a window: gap = R(t-age) - own(t), R(t-age) - R(t-age-1), R(t-age) - R(t-age-3),
# age / 3, missing flag. Training: age 0 (70 %), 1, 2, 3 (10 % each), missing 20 % -> the box survives a dead link
def reg_features(s, source, eight_bit, age=None, missing=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(s["row"])
    if age is None:
        age = rng.choice([0, 1, 2, 3], n, p=[0.7, 0.1, 0.1, 0.1])
    age = np.broadcast_to(age, (n,))
    if missing is None:
        missing = rng.random(n) < 0.2
    missing = np.broadcast_to(missing, (n,))
    R = np.full((n, 3), np.nan)
    for u in range(3):
        k = s["unit"] == u
        r = REG[u][source]
        for j, lag in enumerate((0, 1, 3)):
            v = r[s["row"][k] - age[k] - lag, s["station"][k]]
            R[k, j] = quant(v) if eight_bit else v
    own = np.stack([SENS[u][s["row"], s["station"]] for u in range(3)])[s["unit"], np.arange(n)]
    f = np.column_stack([R[:, 0] - own, R[:, 0] - R[:, 1], R[:, 0] - R[:, 2]])
    bad = missing | ~np.isfinite(f).all(1)
    f = np.where(bad[:, None], 0, np.nan_to_num(f))
    return np.column_stack([f * 3, age / 3, bad]).astype(np.float32)      # x3: changes of ~0.1-0.3 -> order 1
"""

MODEL = """
# students: GDC-CNN trunk + raw regime context (S0+ctx) [+ the regional byte]
ARMS = {"ctx": None, "reg-float": ("delhi", False), "reg-8bit": ("delhi", True), "clus-8bit": ("clus", True),
        "gossip-8bit": ("gossip", True)}

def build(arm, seed):
    keras.utils.set_random_seed(seed)
    x_in, s_in, c_in = keras.Input((12, train["x"].shape[2])), keras.Input((6,)), keras.Input((train["ctx"].shape[1],))
    x = PositionCode()(layers.Dense(D)(x_in))
    for dilation in (1, 2, 4):
        x = gdc_block(x, D, dilation)
    parts, inputs = [layers.Flatten()(x), layers.Dense(D, activation="gelu")(c_in)], [x_in, s_in, c_in]
    if ARMS[arm]:
        r_in = keras.Input((5,))
        parts.append(layers.Dense(D, activation="gelu")(r_in))
        inputs.append(r_in)
    y = layers.Dense(6)(layers.Concatenate()(parts))
    return keras.Model(inputs, y * s_in)

def inputs_of(arm, s, **kw):
    base = [s["x"], s["scale"], s["ctx"]]
    return base + ([reg_features(s, *ARMS[arm], **kw)] if ARMS[arm] else [])

def train_arm(arm, seed):
    model = build(arm, seed)
    model.compile(keras.optimizers.Adam(LR), loss=keras.losses.Huber(1.0), metrics=[keras.metrics.RootMeanSquaredError()])
    stop = keras.callbacks.EarlyStopping("val_root_mean_squared_error", patience=PATIENCE, restore_best_weights=True)
    model.fit(inputs_of(arm, train, seed=seed), train["y"], validation_data=(inputs_of(arm, val, age=0, missing=False), val["y"]),
              batch_size=BATCH, epochs=EPOCHS, callbacks=[stop], verbose=2)
    return model
"""

RUN = """
# train every arm x seed; test fresh (age 0), old (1-3 h) and missing context
CONDITIONS = {"fresh": dict(age=0, missing=False), "age1": dict(age=1, missing=False), "age2": dict(age=2, missing=False),
              "age3": dict(age=3, missing=False), "missing": dict(age=0, missing=True)}
def skill(p, s):
    return 100 * (1 - np.sqrt(((p - s["y"]) ** 2).mean() / (s["y"] ** 2).mean()))

rows, preds = [], {}
for seed in SEEDS:
    for arm in ARMS:
        if time.time() - NOTEBOOK_START > 11 * 3600:
            continue
        start = time.time()
        model = train_arm(arm, seed)
        for name, s in (("seen", test), ("unseen", unseen)):
            for cond, kw in (CONDITIONS if ARMS[arm] else {"fresh": {}}).items():
                p = model.predict(inputs_of(arm, s, **kw), batch_size=4096, verbose=0)
                preds[(arm, seed, name, cond)] = p
                rows.append(dict(arm=arm, seed=seed, stations=name, context=cond, skill=skill(p, s),
                                 skill_pm25_h1=skill(p[:, :1], dict(y=s["y"][:, :1])), minutes=round((time.time() - start) / 60, 1)))
        print([(r["arm"], r["stations"], r["context"], round(r["skill"], 2)) for r in rows if r["seed"] == seed and r["arm"] == arm], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "N3_results.csv", index=False)
        del model
        gc.collect()
"""

SUMMARY = """
# summary over seeds + paired week-block bootstrap vs ctx (seed-averaged forecasts, seen stations)
res = pd.DataFrame(rows)
print(res.groupby(["stations", "arm", "context"]).skill.agg(["mean", "std", "count"]).round(2).to_string())
week = (test["row"] - test["row"].min()) // 168
def mean_pred(arm, cond):
    ps = [preds[k] for k in preds if k[0] == arm and k[2] == "seen" and k[3] == cond]
    return np.mean(ps, axis=0)
def boot(a, cond, b="ctx", n=4000):
    sa = np.bincount(week, ((mean_pred(a, cond) - test["y"]) ** 2).sum(1))
    sb = np.bincount(week, ((mean_pred(b, "fresh") - test["y"]) ** 2).sum(1))
    idx = np.random.default_rng(0).integers(0, len(sa), (n, len(sa)))
    d = 100 * (np.sqrt(sa[idx].sum(1) / sb[idx].sum(1)) - 1)
    return dict(A=a, context=cond, B=b, rmse_change_pct=round(100 * (np.sqrt(sa.sum() / sb.sum()) - 1), 3),
                ci_low=round(np.percentile(d, 2.5), 3), ci_high=round(np.percentile(d, 97.5), 3))
tests = pd.DataFrame([boot(a, c) for a in ARMS if ARMS[a] for c in CONDITIONS])
tests.to_csv(OUT / "N3_bootstrap.csv", index=False)
print("negative = smaller error than the ctx student; CI containing 0 = no real difference")
print(tests.to_string())
"""


def build(smoke):
    nb4 = nb04_cells()
    setup = es.SETUP.replace("SMOKE = False", "SMOKE = True") if smoke else es.SETUP
    setup = setup.replace('Path("../data/6_experiments/student")', 'Path("../data/6_experiments/netctx")')
    cells = [md("# Experiment 5: network ideas (clustering, gossip, 1-byte regional context)"),
             code(setup), nb4[76], nb4[79], code(es.DATA), nb4[82], code(es.SIDE), nb4[85], nb4[86], nb4[87],
             code(NET), code(CLUSTER), code(GOSSIP), code(REGIONAL), code(SIDE_REG), code(MODEL), code(RUN), code(SUMMARY)]
    return notebook([dict(c) for c in cells])
