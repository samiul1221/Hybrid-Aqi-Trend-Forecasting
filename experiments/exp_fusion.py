# Experiment 4 (CPU, no training): how should the box combine its own forecast with the cloud's?
# Uses the saved test forecasts of experiment 1 (teachers, unit 0, every test hour) and experiment 3 (students,
# unit 0, every test hour). All at the 27 training stations, +1..+3 h, PM2.5 and PM10.
#   A  stale cloud forecast: the link died `a` hours ago; the box re-bases the old forecast with its own sensor
#   B  event-triggered sending: the cloud runs a copy of the student and sends only when they disagree (vs random)
#   C  forecast combination (Bates-Granger, econometrics): weights from recent errors against the box's OWN sensor
from nb import code, md, notebook

GPU = False
DATASETS = ["samiulseikh/aqi-features"]
KERNELS = ["samiulseikh/aqi-exp-student", "samiulseikh/aqi-exp-teachpred"]      # + aqi-exp-wind when it has run

CELLS = [
    md("# Experiment 4: edge-cloud fusion (stale forecasts, event-triggered sending, forecast combination)"),
    code("""
# load: teacher forecasts (exp 1), student forecasts (exp 3), the unit-0 sensor PM (to re-base stale forecasts)
from pathlib import Path
import numpy as np, pandas as pd
IN = Path("/kaggle/input")
OUT = Path("/kaggle/working")
g = np.load(next(IN.rglob("grid_u0.npz")))
ST = np.flatnonzero(~g["holdout"])
SENSOR = np.where(g["M"][:, :, :2] > 0, g["X"][:, :, :2], np.nan)          # ln sensor PM2.5, PM10 [hour, station, 2]
TEACHERS = {}
for name in ("T0-s0", "T1-s0", "T2-era5-s0", "best"):
    f = next(IN.rglob(f"pred_{name}.npz"), None)
    if f is not None:
        z = np.load(f)
        TEACHERS[name] = (z["y_hat"].astype(np.float32), z["t"])
STUDENTS = {f.stem.replace("pred_full0_", ""): np.load(f) for f in IN.rglob("pred_full0_*.npz")}
print("teachers:", list(TEACHERS), "| students:", list(STUDENTS))
"""),
    code("""
# line up: for every student window (hour t, station s) the teacher forecast made at hour t - a (a = age)
OUTS = [0, 1, 2, 6, 7, 8]                       # teacher columns of pm25 h1..h3, pm10 h1..h3
base = STUDENTS["S0"]
ROW, STA, Y = base["row"], base["station"], base["y"].astype(np.float32)
col_of = {s: i for i, s in enumerate(ST)}
COL = np.array([col_of[s] for s in STA])

def teacher_at(name, age, rebase="path"):
    yhat, t = TEACHERS[name]
    pos = np.full(g["X"].shape[0], -1)
    pos[t] = np.arange(len(t))
    i = pos[ROW - age]
    out = np.full((len(ROW), 6), np.nan, np.float32)
    ok = i >= 0
    for k, (pol, h) in enumerate([(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]):
        if h + age > 6:
            continue
        old = yhat[i[ok], COL[ok], pol * 6 + h + age - 1]                 # change from t-a to t+h
        if age == 0:
            out[ok, k] = old
        elif rebase == "path":      # the old forecast's own path: (t-a -> t+h) - (t-a -> t). No sensor noise.
            out[ok, k] = old - yhat[i[ok], COL[ok], pol * 6 + age - 1]
        else:                       # "sensor": subtract the box's own (noisy) view of t-a -> t
            moved = SENSOR[ROW[ok], STA[ok], pol] - SENSOR[ROW[ok] - age, STA[ok], pol]
            out[ok, k] = old - np.nan_to_num(moved)
    return out

def skill(pred, keep=None):
    ok = np.isfinite(pred) & np.isfinite(Y)
    if keep is not None:
        ok &= keep[:, None]
    return 100 * (1 - np.sqrt(((pred - Y) ** 2)[ok].mean() / (Y ** 2)[ok].mean()))

STUD = {k: v["y_hat"].astype(np.float32) for k, v in STUDENTS.items() if (v["row"] == ROW).all()}
print({k: round(skill(v), 2) for k, v in STUD.items()})
"""),
    md("## A. Stale cloud forecasts: when should the box switch to its own model?"),
    code("""
# skill of one output column k (the other columns left out)
def skill_col(p, k):
    q = np.full_like(Y, np.nan)
    q[:, k] = p[:, k]
    return skill(q)

rows = []
for name in TEACHERS:
    for age in range(0, 6):
        tp = teacher_at(name, age)
        for k, o in enumerate(["pm25_h1", "pm25_h2", "pm25_h3", "pm10_h1", "pm10_h2", "pm10_h3"]):
            if np.isfinite(tp[:, k]).any():
                rows.append(dict(teacher=name, age_h=age, output=o, teacher_skill=skill_col(tp, k), student_S0=skill_col(STUD["S0"], k)))
        if name == list(TEACHERS)[-1]:                                    # same, re-based with the box's noisy sensor
            tps = teacher_at(name, age, "sensor")
            rows += [dict(teacher=name + " (sensor re-base)", age_h=age, output=o, teacher_skill=skill_col(tps, k))
                     for k, o in enumerate(["pm25_h1", "pm25_h2", "pm25_h3", "pm10_h1", "pm10_h2", "pm10_h3"]) if np.isfinite(tps[:, k]).any()]
stale = pd.DataFrame(rows)
stale.to_csv(OUT / "A_stale.csv", index=False)
print(stale.pivot_table(index=["teacher", "output"], columns="age_h", values="teacher_skill").round(2).to_string())
print(stale.groupby("output").student_S0.first().round(2))
"""),
    md("## B. Event-triggered sending (send only when the cloud and the box disagree)"),
    code("""
# fresh teacher T (age 0) vs the box's S0; the cloud can run S0 itself (same weights, inputs uploaded)
best_t = "best" if "best" in TEACHERS else list(TEACHERS)[-1]
T = teacher_at(best_t, 0)
S = STUD["S0"]
ok = np.isfinite(T).all(1)
gap = np.abs(T - S).mean(1)
rng = np.random.default_rng(0)
rows = []
for frac in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    thr = np.quantile(gap[ok], 1 - frac) if 0 < frac < 1 else (np.inf if frac == 0 else -1)
    send_ev = ok & (gap > thr)
    send_rand = ok & (rng.random(len(gap)) < frac)
    err_t, err_s = ((T - Y) ** 2).mean(1), ((S - Y) ** 2).mean(1)
    send_oracle = ok & (np.argsort(np.argsort(np.where(ok, err_s - err_t, -np.inf))) >= len(gap) - frac * ok.sum())
    for how, send in (("disagreement", send_ev), ("random", send_rand), ("oracle", send_oracle)):
        rows.append(dict(sent_pct=100 * frac, how=how, skill=skill(np.where(send[:, None], T, S))))
event = pd.DataFrame(rows)
event.to_csv(OUT / "B_event_triggered.csv", index=False)
print("teacher", best_t, "| box alone", round(skill(S), 2), "| cloud always", round(skill(np.where(ok[:, None], T, S)), 2))
print(event.pivot_table(index="sent_pct", columns="how", values="skill").round(2).to_string())
"""),
    md("## C. Forecast combination with feedback from the box's own sensor (Bates-Granger)"),
    code("""
# per station, in time order: weight w = 1/MSE_T / (1/MSE_T + 1/MSE_S), MSEs = exponentially weighted errors against the
# SENSOR change seen h hours later (the box never sees the true PM). Fixed blend: weight fitted on the first half of 2025.
order = np.lexsort((ROW, STA))
sensor_y = np.column_stack([SENSOR[np.minimum(ROW + h, len(SENSOR) - 1), STA, p] - SENSOR[ROW, STA, p]
                            for p, h in [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]])
# mode "bg": exponentially weighted MSE (Bates-Granger). mode "jacobson": the TCP round-trip-time estimator
# (RFC 6298): srtt += 1/8 (|e| - srtt), rttvar += 1/4 (|srtt - |e|| - rttvar); error scale = srtt + 4 rttvar.
def online_blend(T, S, mode="bg", halflife=72):
    a = np.log(2) / halflife
    w = np.full_like(T, 0.5)
    jt, js = [np.full(6, 0.3), np.full(6, 0.1)], [np.full(6, 0.3), np.full(6, 0.1)]     # [srtt, rttvar]
    mt = np.ones(6); ms = np.ones(6); last = None
    hz = np.array([1, 2, 3, 1, 2, 3])
    pending = []
    for n, i in enumerate(order):
        if STA[i] != last:
            mt, ms, last, pending = np.full(6, 0.1), np.full(6, 0.1), STA[i], []
            jt, js = [np.full(6, 0.3), np.full(6, 0.1)], [np.full(6, 0.3), np.full(6, 0.1)]
        # errors whose truth (the sensor h hours later) is known by hour ROW[i]
        while pending and pending[0][0] <= ROW[i]:
            _, j, k = pending.pop(0)
            if np.isfinite(sensor_y[j, k]) and np.isfinite(T[j, k]):
                mt[k] = (1 - a) * mt[k] + a * (T[j, k] - sensor_y[j, k]) ** 2
                ms[k] = (1 - a) * ms[k] + a * (S[j, k] - sensor_y[j, k]) ** 2
                for jj, e in ((jt, abs(T[j, k] - sensor_y[j, k])), (js, abs(S[j, k] - sensor_y[j, k]))):
                    jj[1][k] += 0.25 * (abs(jj[0][k] - e) - jj[1][k])
                    jj[0][k] += 0.125 * (e - jj[0][k])
        if mode == "jacobson":
            vt, vs = (jt[0] + 4 * jt[1]) ** 2, (js[0] + 4 * js[1]) ** 2
            w[i] = (1 / vt) / (1 / vt + 1 / vs)
        else:
            w[i] = (1 / mt) / (1 / mt + 1 / ms)
        pending += [(ROW[i] + hz[k], i, k) for k in range(6)]
        pending.sort(key=lambda q: q[0])
    return np.where(np.isfinite(T), w * T + (1 - w) * S, S), w

rows = []
first = ROW < np.median(ROW)
for age in (0, 1, 2, 3):
    T = teacher_at(best_t, age)
    blend_on, w = online_blend(T, S)
    blend_j, wj = online_blend(T, S, "jacobson")
    Tf = np.where(np.isfinite(T), T, S)
    fit = first[:, None] & np.isfinite(T)                       # least-squares weight of the cloud, first half only
    alpha = float(np.clip(np.sum(((T - S) * (Y - S))[fit & np.isfinite(Y)]) / np.sum(((T - S) ** 2)[fit & np.isfinite(Y)]), 0, 1))
    later = ~first
    rows.append(dict(age_h=age, box=skill(S, later), cloud=skill(Tf, later), fixed_blend=skill(alpha * Tf + (1 - alpha) * S, later),
                     alpha=alpha, online_bates_granger=skill(blend_on, later), online_jacobson=skill(blend_j, later),
                     weight_bg=float(np.nanmean(w)), weight_jacobson=float(np.nanmean(wj))))
comb = pd.DataFrame(rows)
comb.to_csv(OUT / "C_combination.csv", index=False)
print("scored on the second half of the test period")
print(comb.round(3).to_string())
"""),
    code("""
# can the box replace the cloud? box students with weather inputs vs the cloud teachers (same windows)
rows = [dict(model=f"box {k}", skill=skill(v)) for k, v in STUD.items()]
rows += [dict(model=f"cloud {k}", skill=skill(np.where(np.isfinite(teacher_at(k, 0)), teacher_at(k, 0), np.nan))) for k in TEACHERS]
who = pd.DataFrame(rows).sort_values("skill")
who.to_csv(OUT / "D_box_vs_cloud.csv", index=False)
print(who.round(2).to_string())
"""),
    md("## E. Token bucket: at most B cloud messages per box per day"),
    code("""
# the cloud may send at most B messages per box per day (token bucket, refilled B/24 per hour, full = B).
# 'disagree': send when |cloud - box copy| is above the level that would use the budget; 'periodic': every 24/B h.
# Between messages the box re-bases the last cloud forecast (up to 3 h old), otherwise uses its own student.
TA = {a: teacher_at(best_t, a) for a in range(4)}
GAP = np.abs(TA[0] - S).mean(1)
def run_policy(budget, policy):
    thr = np.nanquantile(GAP, max(0.0, 1 - budget / 24))
    period = max(1, round(24 / budget))
    pred, sent, last_sta = S.copy(), 0, None
    for i in order:
        if STA[i] != last_sta:
            tokens, last_sent, prev, last_sta = float(budget), -10 ** 9, ROW[i], STA[i]
        tokens = min(budget, tokens + budget / 24 * (ROW[i] - prev))
        prev = ROW[i]
        want = GAP[i] > thr if policy == "disagree" else ROW[i] % period == 0
        if want and tokens >= 1 and np.isfinite(TA[0][i]).all():
            tokens, last_sent, sent = tokens - 1, ROW[i], sent + 1
        age = ROW[i] - last_sent
        if age <= 3 and np.isfinite(TA[age][i]).all():
            pred[i] = TA[age][i]
    return pred, sent
rows = []
for budget in (1, 2, 3, 4, 6, 8, 12, 24):
    for policy in ("disagree", "periodic"):
        pred, sent = run_policy(budget, policy)
        rows.append(dict(budget_per_day=budget, policy=policy, sent_pct=100 * sent / len(S), skill=skill(pred)))
        print(rows[-1], flush=True)
bucket = pd.DataFrame(rows)
bucket.to_csv(OUT / "E_token_bucket.csv", index=False)
print("box alone", round(skill(S), 2), "| cloud every hour", round(skill(np.where(np.isfinite(TA[0]), TA[0], S)), 2))
print(bucket.pivot_table(index="budget_per_day", columns="policy", values="skill").round(2).to_string())
"""),
]


def build(smoke):
    return notebook([dict(c) for c in CELLS])
