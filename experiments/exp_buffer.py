# Experiment 9 (CPU): BUFFERED BROADCAST. "Send the plan, not the next step."
# Experiment 5 showed a 2-3 h old broadcast is worth nothing. Reason: a forecast for +1..+3 h made 3 hours ago does
# not cover any future hour any more. Fix to test: the cloud broadcasts the regional PATH for +1..+9 h. A box whose
# last broadcast is `a` hours old shifts along the stored path:
#   change from now to now+h  =  path[a + h] - path[a]        (path made at hour t - a; "path re-basing" of exp 4)
# Arms (student = split-res of experiment 6, every arm trained with random ages 0..6 h and 15 % missing):
#   ctx     no broadcast at all (reference)
#   short   today's broadcast: +1..+3 h. At age a only the horizons with a + h <= 3 still exist.
#   long    the buffered path: +1..+9 h. Up to age 6 every horizon still exists.
# Both broadcast arms also get the region-minus-own level gap of the hour the broadcast was made, the age and flags.
# Scored at fixed ages 0, 1, 2, 3, 4, 6 h and "missing", seen and unseen stations, week-block bootstrap.
# Kill threshold (fixed before the run): at age 3 h, long vs ctx keeps less than 40 % of long's fresh gain, or the CI of
# long vs short at age 3 h contains 0. Also: long at age 0 must not be worse than short at age 0 by more than 0.2 %.
from nb import nb04_cells, code, md, notebook
import exp_student as es
import exp_split as sp
from exp_memory import swap

GPU = False
DATASETS = ["samiulseikh/aqi-features"]

OUTPUTS9 = "OUTPUTS = [(p, h) for p in (0, 1) for h in range(1, 10)]      # the regional PATH: +1..+9 h, PM2.5 then PM10"

AGED = """
# what a box has when its last broadcast is `age` hours old (age -1 = it never got one)
SIX = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]             # the student's 6 outputs (pollutant, hours ahead)
HMAX = {"short": 3, "long": 9}
def aged(s, arm, age):
    if arm == "ctx":
        return s
    n = len(s["row"])
    fc, have, gap = np.zeros((n, 6), np.float32), np.zeros((n, 6), np.float32), np.zeros((n, 2), np.float32)
    for u in range(3):
        for a in range(7):
            k = np.flatnonzero((s["unit"] == u) & (age == a))
            r, i = s["row"][k] - a, s["station"][k]                  # the hour the broadcast was made
            path = RHAT[u][r]                                        # [n, 18] regional change from r to r + 1..9
            for j, (p, h) in enumerate(SIX):
                if a + h <= HMAX[arm]:
                    fc[k, j] = path[:, p * 9 + a + h - 1] - (path[:, p * 9 + a - 1] if a else 0)
                    have[k, j] = 1
            gap[k] = np.column_stack([R25[u][r] - S25[u][r, i], R10[u][r] - S10[u][r, i]])
    fc, miss = np.nan_to_num(fc), (age < 0).astype(np.float32)
    side = np.column_stack([fc * 3, have, np.nan_to_num(gap), np.clip(age, 0, None) / 6, miss]).astype(np.float32)
    return dict(s, split=side, rhat=fc)

rng = np.random.default_rng(0)
def random_age(n):                                                 # 15 % missing, otherwise 0..6 h old
    return np.where(rng.random(n) < 0.15, -1, rng.integers(0, 7, n))
for s in (train, val, test, unseen):
    s["w"] = np.ones_like(s["y"])                                  # plain Huber loss (the weights are not used)
    s["split"], s["rhat"] = np.zeros((len(s["y"]), 16), np.float32), np.zeros((len(s["y"]), 6), np.float32)
BASE = dict(train=train, val=val)
AGE_TRAIN, AGE_VAL = random_age(len(train["y"])), random_age(len(val["y"]))
"""

ARMS = """ARMS = {"ctx": ([], False, "huber"), "short": (["split"], True, "huber"), "long": (["split"], True, "huber")}"""

RUN = """
# train every arm x seed, then score it at every broadcast age
AGES = [0, 1, 2, 3, 4, 6, -1]
def skill(p, s):
    return 100 * (1 - np.sqrt(((p - s["y"]) ** 2).mean() / (s["y"] ** 2).mean()))
rows, preds = [], {}
for seed in SEEDS:
    for arm in ARMS:
        if time.time() - NOTEBOOK_START > 11 * 3600:
            continue
        start = time.time()
        train, val = aged(BASE["train"], arm, AGE_TRAIN), aged(BASE["val"], arm, AGE_VAL)      # build / train_arm read these
        model = train_arm(arm, seed)
        for name, s0 in (("seen", test), ("unseen", unseen)):
            for age in AGES:
                s = aged(s0, arm, np.full(len(s0["y"]), age))
                p = model.predict(inputs_of(arm, s), batch_size=4096, verbose=0)
                preds[(arm, seed, name, age)] = p
                rows.append(dict(arm=arm, seed=seed, stations=name, age=age, skill=skill(p, s0),
                                 minutes=round((time.time() - start) / 60, 1)))
        print(arm, seed, {r["age"]: round(r["skill"], 2) for r in rows[-2 * len(AGES):-len(AGES)]}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
        del model
        gc.collect()
"""

SUMMARY = """
# mean over seeds; week-block paired bootstrap at every age (age -1 = no broadcast received)
res = pd.DataFrame(rows)
print(res.groupby(["stations", "arm", "age"]).skill.mean().unstack("age").round(2).to_string())
def mean_pred(arm, name, age):
    return np.mean([preds[k] for k in preds if k[0] == arm and k[2] == name and k[3] == age], axis=0)
def boot(a, b, name, age, n=4000):
    s = test if name == "seen" else unseen
    week = (s["row"] - s["row"].min()) // 168
    sa = np.bincount(week, ((mean_pred(a, name, age) - s["y"]) ** 2).sum(1))
    sb = np.bincount(week, ((mean_pred(b, name, age) - s["y"]) ** 2).sum(1))
    sa, sb = sa[sb > 0], sb[sb > 0]
    idx = np.random.default_rng(0).integers(0, len(sa), (n, len(sa)))
    d = 100 * (np.sqrt(sa[idx].sum(1) / sb[idx].sum(1)) - 1)
    return dict(A=a, B=b, stations=name, age=age, rmse_change_pct=round(100 * (np.sqrt(sa.sum() / sb.sum()) - 1), 3),
                ci_low=round(np.percentile(d, 2.5), 3), ci_high=round(np.percentile(d, 97.5), 3))
done = {k[0] for k in preds}
tests = pd.DataFrame([boot(a, b, name, age) for name in ("seen", "unseen") for a, b in (("long", "ctx"), ("short", "ctx"), ("long", "short"))
                      for age in AGES if a in done and b in done])
tests.to_csv(OUT / "bootstrap.csv", index=False)
print("negative = A has the smaller error than B; CI containing 0 = no real difference")
print(tests.to_string())
"""


def build(smoke):
    nb4 = nb04_cells()
    setup = es.SETUP.replace("SMOKE = False", "SMOKE = True") if smoke else es.SETUP
    setup = setup.replace('Path("../data/6_experiments/student")', 'Path("../data/6_experiments/buffer")')
    region = swap(sp.REGION, "OUTPUTS = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]      # (pollutant, hours ahead) the cloud forecasts", OUTPUTS9)
    old_arms = sp.MODEL[sp.MODEL.index("ARMS = {"):sp.MODEL.index("def build")]
    model = swap(sp.MODEL, old_arms, ARMS + "\n\n")
    cells = [md("# Experiment 9: buffered broadcast (regional path +1..+9 h, used while the link is down)"),
             code(setup), nb4[76], nb4[79], code(es.DATA), nb4[82], code(es.SIDE), nb4[85], nb4[86], nb4[87],
             code(region), code(sp.CLOUD), code(AGED), code(model), code(RUN), code(SUMMARY)]
    return notebook([dict(c) for c in cells])
