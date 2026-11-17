# Experiment 8 (CPU): NODE MEMORY. The box remembers its own usual offset from the region for every clock hour.
# The consensus filter of experiment 6 keeps ONE average offset (24 h EWMA), blind to the hour of day. But local
# sources (traffic peaks, evening burning) repeat every day at the same hours, and they differ from site to site.
#   offset   o_i(t) = own ln PM_i(t) - regional mean R(t)              (R comes in the broadcast, no true PM)
#   memory   M_i(c, t) = EWMA over past DAYS (half-life 10 days) of o_i at clock hour c   (48 numbers in flash)
#   surprise o_i(t) - M_i(clock(t))                 "how unusual am I right now"         (2 numbers)
#   profile  M_i(clock(t+h)) - M_i(clock(t)), h = 1..3   "how my offset usually moves next"  (6 numbers)
# The table entry used for hour t+h was last updated at t+h-24, so nothing from the future is used.
# Arms (student = split-res of experiment 6):
#   split-res      the result of experiment 6
#   b0             split-res + consensus filter (the combined baseline)
#   mem            b0 + surprise + profile
#   mem-surprise   b0 + surprise only
#   mem-profile    b0 + profile only
#   mem-placebo    b0 + the memory table of ANOTHER node (control: is the gain really site-specific?)
# Kill threshold (fixed before the run): mem vs b0 weaker than -0.3 % RMSE, or CI contains 0, or mem does not beat
# mem-placebo with a CI that excludes 0.
from nb import nb04_cells, code, md, notebook
import exp_student as es
import exp_split as sp

GPU = False
DATASETS = ["samiulseikh/aqi-features"]

MEMORY = """
# the node's memory table and the features made from it
def usual(off):                                        # EWMA over past days of the same clock hour (rows are hourly)
    out = np.full_like(off, np.nan)
    for c in range(24):
        out[c::24] = pd.DataFrame(off[c::24]).ewm(halflife=10, ignore_na=True).mean().shift(1).to_numpy()
    return out

MEM = {}
for u in range(3):
    o25, o10 = S25[u] - R25[u][:, None], S10[u] - R10[u][:, None]
    MEM[u] = (o25, o10, usual(o25), usual(o10))
SWAP = np.roll(np.arange(NS), 7)                        # placebo: every node reads the table of another node

def mem_features(s, swap=False):
    f = np.zeros((len(s["row"]), 8), np.float32)
    for u in range(3):
        k = s["unit"] == u
        r, i = s["row"][k], s["station"][k]
        j = SWAP[i] if swap else i
        o25, o10, m25, m10 = MEM[u]
        f[k] = np.nan_to_num(np.column_stack([o25[r, i] - m25[r, j], o10[r, i] - m10[r, j]] +
                                             [(m[r + h, j] - m[r, j]) * 3 for m in (m25, m10) for h in (1, 2, 3)]))
    return f

for s in (train, val, test, unseen):
    s["mem"] = mem_features(s)
    s["memS"], s["memP"] = s["mem"][:, :2], s["mem"][:, 2:]
    s["memX"] = mem_features(s, swap=True)
print("memory features, mean |value|:", np.abs(train["mem"]).mean(0).round(3))
"""

ARMS = """ARMS = {"split-res": (["split"], True, "huber"), "b0": (["split", "den"], True, "huber"),
        "mem": (["split", "den", "mem"], True, "huber"), "mem-surprise": (["split", "den", "memS"], True, "huber"),
        "mem-profile": (["split", "den", "memP"], True, "huber"), "mem-placebo": (["split", "den", "memX"], True, "huber")}"""

SUMMARY = """
# mean over seeds, then a week-block paired bootstrap (seen and unseen stations)
res = pd.DataFrame(rows)
print(res.groupby(["stations", "arm"])[["skill", "skill_pm25_h1", "direction_acc", "category_acc"]].agg(["mean", "std"]).round(2).to_string())
def mean_pred(arm, name):
    return np.mean([preds[k] for k in preds if k[0] == arm and k[2] == name], axis=0)
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
tests = pd.DataFrame([boot(a, b, name) for name in ("seen", "unseen") for a, b in PAIRS if (a, 0, name) in preds and (b, 0, name) in preds])
tests.to_csv(OUT / "bootstrap.csv", index=False)
print("negative = A has the smaller error than B; CI containing 0 = no real difference")
print(tests.to_string())
"""

PAIRS = """PAIRS = [("b0", "split-res"), ("mem", "b0"), ("mem-surprise", "b0"), ("mem-profile", "b0"), ("mem-placebo", "b0"), ("mem", "mem-placebo")]"""


# swap one piece of a cell of an older experiment and fail loudly if that experiment changed
def swap(text, old, new):
    assert text.count(old) == 1, old[:60]
    return text.replace(old, new)


def build(smoke):
    nb4 = nb04_cells()
    setup = es.SETUP.replace("SMOKE = False", "SMOKE = True") if smoke else es.SETUP
    setup = setup.replace('Path("../data/6_experiments/student")', 'Path("../data/6_experiments/memory")')
    old_arms = sp.MODEL[sp.MODEL.index("ARMS = {"):sp.MODEL.index("def build")]
    model = swap(sp.MODEL, old_arms, ARMS + "\n\n")
    run = sp.RUN.replace("split_results.csv", "results.csv")
    cells = [md("# Experiment 8: node memory (the box's usual offset from the region, per clock hour)"),
             code(setup), nb4[76], nb4[79], code(es.DATA), nb4[82], code(es.SIDE), nb4[85], nb4[86], nb4[87],
             code(sp.REGION), code(sp.CLOUD), code(sp.DENOISE), code(MEMORY), code(model), code(run), code(PAIRS + SUMMARY)]
    return notebook([dict(c) for c in cells])
