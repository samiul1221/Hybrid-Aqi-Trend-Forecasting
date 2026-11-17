# Experiment 10 (GPU): is the box's student simply short of training data?
# Every student so far was trained on the 200 000 distillation windows, but 2.6 M training windows exist. The cloud
# teacher (trained on every hour) is ~6 skill points better than the student while the graph explains only ~2 of them.
# Check: the SAME student (S0 + ctx, experiment 3), the same validation and test windows, only the training set changes.
#   200k-d32   the 200 000 windows used so far (reference)
#   1M-d32     1 000 000 random training windows (27 training stations, 2018-23)
#   1M-d64     the same with width 64 (does a bigger box model use the extra data?)
# No kill threshold: this is a diagnosis, not a new idea. It tells how much head-room the student has on its own.
from nb import nb04_cells, code, md, notebook
import exp_student as es

GPU = True
DATASETS = ["samiulseikh/aqi-features"]

BIG = """
# the bigger training set: random training windows of the 27 training stations (never the 5 hold-out stations)
allw = pd.read_parquet(FEAT / "windows_A.parquet")
big_w = allw[(allw.split == "train") & ~allw.holdout].sample(4000 if SMOKE else 1_000_000, random_state=0).reset_index(drop=True)
del allw
big = normalise(make_set(big_w))
print("big training set:", big["x"].shape, "| GPUs:", tf.config.list_physical_devices("GPU"))
"""

RUN = """
# the same student on each training set (2 seeds); test forecasts kept for the bootstrap
RUNS = {"200k-d32": (train, 32), "1M-d32": (big, 32), "1M-d64": (big, 64)}
def skill(p, s):
    return 100 * (1 - np.sqrt(((p - s["y"]) ** 2).mean() / (s["y"] ** 2).mean()))
rows, preds = [], {}
for seed in SEEDS[:2]:
    for run, (train, D) in RUNS.items():                # build / train_arm read the globals `train` and `D`
        start = time.time()
        model = train_arm("S0+ctx", seed)
        for name, s in (("seen", test), ("unseen", unseen)):
            p = model.predict(inputs_of("S0+ctx", s), batch_size=4096, verbose=0)
            preds[(run, seed, name)] = p
            rows.append(dict(run=run, seed=seed, stations=name, skill=skill(p, s), params=model.count_params(),
                             minutes=round((time.time() - start) / 60, 1)))
        print(rows[-2], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
        del model
        gc.collect()
"""

SUMMARY = """
# mean over seeds and week-block paired bootstrap vs the 200k student
res = pd.DataFrame(rows)
print(res.groupby(["stations", "run"]).skill.agg(["mean", "std"]).round(2).to_string())
def mean_pred(run, name):
    return np.mean([preds[k] for k in preds if k[0] == run and k[2] == name], axis=0)
def boot(a, b, name, n=4000):
    s = test if name == "seen" else unseen
    week = (s["row"] - s["row"].min()) // 168
    sa = np.bincount(week, ((mean_pred(a, name) - s["y"]) ** 2).sum(1))
    sb = np.bincount(week, ((mean_pred(b, name) - s["y"]) ** 2).sum(1))
    sa, sb = sa[sb > 0], sb[sb > 0]
    idx = np.random.default_rng(0).integers(0, len(sa), (n, len(sa)))
    d = 100 * (np.sqrt(sa[idx].sum(1) / sb[idx].sum(1)) - 1)
    return dict(A=a, B=b, stations=name, rmse_change_pct=round(100 * (np.sqrt(sa.sum() / sb.sum()) - 1), 3),
                ci_low=round(np.percentile(d, 2.5), 3), ci_high=round(np.percentile(d, 97.5), 3))
tests = pd.DataFrame([boot(a, b, name) for name in ("seen", "unseen") for a, b in (("1M-d32", "200k-d32"), ("1M-d64", "200k-d32"), ("1M-d64", "1M-d32"))])
tests.to_csv(OUT / "bootstrap.csv", index=False)
print("negative = A has the smaller error than B; CI containing 0 = no real difference")
print(tests.to_string())
"""


def build(smoke):
    nb4 = nb04_cells()
    setup = es.SETUP.replace("SMOKE = False", "SMOKE = True") if smoke else es.SETUP
    setup = setup.replace('Path("../data/6_experiments/student")', 'Path("../data/6_experiments/moredata")')
    cells = [md("# Experiment 10: does the student need more training windows?"),
             code(setup), nb4[76], nb4[79], code(es.DATA), nb4[82], code(es.SIDE), code(BIG), nb4[85], nb4[86], nb4[87],
             code(es.MODEL_CELL), code(RUN), code(SUMMARY)]
    return notebook([dict(c) for c in cells])
