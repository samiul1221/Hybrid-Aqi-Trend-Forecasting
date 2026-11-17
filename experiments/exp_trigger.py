# Experiment 7 (CPU, no network training): a LEARNED send rule for the personal cloud forecast.
# Experiment 4 sends when |cloud - box| is large. At 20 % of hours that gives skill 20.7; an oracle that knows the
# true errors gets 27.2. Question: how much of that gap can the cloud close with things it really knows at hour t?
#   target   benefit = (box error)^2 - (cloud error)^2 of that hour (true PM used as a TARGET only)
#   inputs   both forecasts, their gaps, the teacher's +4..+6 h, the city-mean teacher forecast, the box's uploaded
#            sensors (PM level and slopes, RH, CO, fan wind), clock
#   model    LightGBM, trained on the FIRST half of the test period, every rule scored on the SECOND half
#            (the saved forecasts exist only for the test period; no 2024 forecasts were saved)
#   rules    random, disagreement (exp 4), learned, oracle; also under a token bucket (messages per box per day)
# Kill threshold (fixed before the run): learned - disagreement < +0.5 skill at 20 % of hours, or the CI contains 0.
from nb import code, md, notebook
import exp_fusion as ef

GPU = False
DATASETS = ef.DATASETS
KERNELS = ef.KERNELS

FEATS = """
# what the cloud knows at hour t about one box (no true PM): forecasts, gaps, uploaded sensors, clock
import lightgbm as lgb
best_t = "best" if "best" in TEACHERS else list(TEACHERS)[-1]
T, S = teacher_at(best_t, 0), STUD["S0"]
yhat, t_rows = TEACHERS[best_t]
pos = np.full(g["X"].shape[0], -1)
pos[t_rows] = np.arange(len(t_rows))
I = pos[ROW]
OK = (I >= 0) & np.isfinite(T).all(1) & np.isfinite(Y).all(1)
far = yhat[I, COL][:, [3, 4, 5, 9, 10, 11]]                       # the teacher's +4..+6 h
city = np.nanmean(yhat[:, :, OUTS], 1)[I]                        # city-mean teacher forecast of that hour
F = ["ln_pm25", "ln_pm10", "ln_pm1", "coarse_ratio", "t", "rh", "p", "wind_u", "wind_v", "rain_frac", "ln_lux", "mq7_n",
     "mq135_n", "voc_n", "nox_n", "hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "aqi_now", "pm_mask"]
box = g["X"][ROW, STA][:, [F.index(f) for f in ("rh", "t", "mq7_n", "wind_u", "wind_v", "hod_sin", "hod_cos", "doy_sin", "doy_cos")]]
slope = np.column_stack([SENSOR[ROW, STA, p] - SENSOR[ROW - k, STA, p] for p in (0, 1) for k in (1, 3)])
GAP = np.abs(T - S).mean(1)                                      # the old rule
XF = np.column_stack([T, S, T - S, GAP, far, city, T - city, SENSOR[ROW, STA], slope, box]).astype(np.float32)
BENEFIT = ((S - Y) ** 2).mean(1) - ((T - Y) ** 2).mean(1)        # > 0: the cloud forecast was better
print(XF.shape, "usable windows:", OK.sum(), "| cloud better in", round(100 * (BENEFIT[OK] > 0).mean(), 1), "% of hours")
"""

FIT = """
# learn the benefit on the first half of the test period, score every rule on the second half
first = ROW < np.median(ROW)
fit, ev = OK & first, OK & ~first
SMOKE = False
model = lgb.train(dict(objective="huber", alpha=0.05, learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1,
                       num_threads=4), lgb.Dataset(XF[fit], BENEFIT[fit]), 20 if SMOKE else 400)
SCORE = model.predict(XF)
names = ([f"T{k}" for k in range(6)] + [f"S{k}" for k in range(6)] + [f"gap{k}" for k in range(6)] + ["gap_abs"] +
         [f"far{k}" for k in range(6)] + [f"city{k}" for k in range(6)] + [f"Tloc{k}" for k in range(6)] +
         ["pm25", "pm10", "d25_1", "d25_3", "d10_1", "d10_3", "rh", "t", "mq7", "wind_u", "wind_v", "hod_s", "hod_c", "doy_s", "doy_c"])
gain = pd.Series(model.feature_importance("gain"), names).sort_values(ascending=False)
print((100 * gain / gain.sum()).round(1).head(12).to_string())
"""

RULES = """
# send the top x % of hours by each rule; week-block paired bootstrap of learned vs disagreement
def sk(pred, keep):
    return 100 * (1 - np.sqrt(((pred - Y) ** 2)[keep].mean() / (Y ** 2)[keep].mean()))
def top(score, frac):
    return ev & (score >= np.quantile(score[ev], 1 - frac)) if frac < 1 else ev
week = (ROW - ROW.min()) // 168
def boot(pa, pb, n=4000):
    sa = np.bincount(week[ev], ((pa - Y) ** 2).sum(1)[ev])
    sb = np.bincount(week[ev], ((pb - Y) ** 2).sum(1)[ev])
    sa, sb = sa[sb > 0], sb[sb > 0]
    idx = np.random.default_rng(0).integers(0, len(sa), (n, len(sa)))
    d = 100 * (np.sqrt(sa[idx].sum(1) / sb[idx].sum(1)) - 1)
    return 100 * (np.sqrt(sa.sum() / sb.sum()) - 1), np.percentile(d, 2.5), np.percentile(d, 97.5)

rng = np.random.default_rng(0)
rows = []
for frac in (0.05, 0.1, 0.2, 0.3, 0.5):
    p = {how: np.where(top(score, frac)[:, None], T, S) for how, score in
         (("random", rng.random(len(GAP))), ("disagreement", GAP), ("learned", SCORE), ("oracle", BENEFIT))}
    d, lo, hi = boot(p["learned"], p["disagreement"])
    rows.append(dict(sent_pct=100 * frac, **{how: sk(v, ev) for how, v in p.items()}, rmse_change_pct=d, ci_low=lo, ci_high=hi))
res = pd.DataFrame(rows)
res["learned_minus_disagreement"] = res.learned - res.disagreement
res.to_csv(OUT / "trigger_results.csv", index=False)
print("second half of the test period | box alone", round(sk(S, ev), 2), "| cloud always", round(sk(np.where(ev[:, None], T, S), ev), 2))
print("rmse_change_pct = learned vs disagreement (negative = learned better), 95 % week-block CI")
print(res.round(2).to_string())
"""

BUCKET = """
# token bucket: at most B messages per box per day; send when the rule's score is above the level that would use
# the budget. Between messages the box keeps the last cloud forecast (re-based on its own path, up to 3 h old).
order = np.lexsort((ROW, STA))
order = order[ev[order]]
TA = {a: teacher_at(best_t, a) for a in range(4)}
def run_policy(budget, score):
    thr = np.quantile(score[ev], max(0.0, 1 - budget / 24))
    pred, sent, last_sta = S.copy(), 0, None
    for i in order:
        if STA[i] != last_sta:
            tokens, last_sent, prev, last_sta = float(budget), -10 ** 9, ROW[i], STA[i]
        tokens = min(budget, tokens + budget / 24 * (ROW[i] - prev))
        prev = ROW[i]
        if score[i] > thr and tokens >= 1:
            tokens, last_sent, sent = tokens - 1, ROW[i], sent + 1
        age = ROW[i] - last_sent
        if age <= 3 and np.isfinite(TA[age][i]).all():
            pred[i] = TA[age][i]
    return pred, sent
rows = []
for budget in (1, 2, 4, 6, 12):
    pd_, nd = run_policy(budget, GAP)
    pl, nl = run_policy(budget, SCORE)
    d, lo, hi = boot(pl, pd_)
    rows.append(dict(budget_per_day=budget, disagreement=sk(pd_, ev), learned=sk(pl, ev), sent_pct_disagreement=100 * nd / ev.sum(),
                     sent_pct_learned=100 * nl / ev.sum(), rmse_change_pct=d, ci_low=lo, ci_high=hi))
    print(rows[-1], flush=True)
bucket = pd.DataFrame(rows)
bucket.to_csv(OUT / "trigger_token_bucket.csv", index=False)
print(bucket.round(2).to_string())
"""


def build(smoke):
    fit = FIT.replace("SMOKE = False", "SMOKE = True") if smoke else FIT
    cells = [md("# Experiment 7: learned send rule (value of information) vs disagreement"), ef.CELLS[1], ef.CELLS[2],
             code(FEATS), code(fit), md("## Send the top x % of hours"), code(RULES), md("## Token bucket"), code(BUCKET)]
    return notebook([dict(c) for c in cells])
