# builds the LaTeX table bodies in tables/ from the raw result files of the experiments (nothing is typed by hand)
# usage: python make_tables.py      (run from anywhere)
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "6_experiments"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)


from decimal import Decimal, ROUND_HALF_UP


def f2(x, sign=True):                                 # 2 decimals, halves round up (the csv holds 3 decimals)
    q = Decimal(str(round(float(x), 3))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{q:+}" if sign else f"{q}"


def ms(d, arm, st, col="skill"):                      # mean +- std over seeds
    v = d[(d.arm == arm) & (d.stations == st)][col]
    return v.mean(), v.std()


def fmt_ms(m, s):
    return f"{m:.2f}$\\pm${s:.2f}"


def boot(df, a, b, st=None):                          # "-1.09 [-1.27, -0.91]"
    q = df[(df.A == a) & (df.B == b)]
    if st is not None:
        q = q[q.stations == st]
    r = q.iloc[0]
    return f"{f2(r.rmse_change_pct)} [{f2(r.ci_low)}, {f2(r.ci_high)}]"


HEAD = {   # column spec and header row of every table (whole tabular is written, so no row ends a file before a rule)
    "stack": ("|p{4.3cm}|c|c|c|c|c|", r"Model & Known & New & Dir. & $\Delta$RMSE known [CI] & $\Delta$RMSE new [CI]"),
    "memory": ("|p{5.4cm}|c|c|c|c|", r"Model & Known & New & $\Delta$RMSE known [CI] & $\Delta$RMSE new [CI]"),
    "cloud": ("|p{4.3cm}|c|c|", r"Cloud variant & Skill & $\Delta$RMSE [CI]"),
    "stale": ("|l|p{2.1cm}|c|c|c|c|c|c|", "Places & Model & 0 & 1 & 2 & 3 & 6 & Miss."),
    "other": ("|p{4.0cm}|c|c|", r"Comparison & $\Delta$RMSE [CI] & Result"),
}


def save(name, rows):
    key = name.split("_")[0]
    spec, head = HEAD[key]
    body = [chr(92) + "begin{tabular}{" + spec + "}", chr(92) + "hline", head + " " + chr(92) * 2, chr(92) + "hline"] + rows + [chr(92) + "hline", chr(92) + "end{tabular}"]
    (OUT / name).write_bytes(("\n".join(body) + "\n").encode("utf8"))


sp = pd.read_csv(ROOT / "split" / "split_results.csv")
spb = pd.read_csv(ROOT / "split" / "split_bootstrap.csv")
me = pd.read_csv(ROOT / "memory" / "results.csv")
meb = pd.read_csv(ROOT / "memory" / "bootstrap.csv")
allr = pd.concat([sp, me[~me.arm.isin(["split-res"])]])   # split-res is in both runs, same windows

# ---- Table: the box-model stack (E6 + E8) ----
rows = []
names = [("ctx", "Box alone (ctx)"), ("denoise", "+ consensus level"), ("split", "+ regional broadcast (input)"),
         ("split-res", "+ residual output (split-res)"), ("b0", "+ consensus level (b0)"), ("mem", "+ node memory (full)")]
for arm, label in names:
    s, ss = ms(allr, arm, "seen")
    u, us = ms(allr, arm, "unseen")
    d = ms(allr, arm, "seen", "direction_acc")[0]
    if arm == "ctx":
        c1 = c2 = "reference"
    elif arm in ("denoise", "split", "split-res"):
        c1 = boot(spb.assign(stations="seen"), arm, "ctx")
        c2 = "--"
    elif arm == "b0":
        c1, c2 = boot(meb, "b0", "split-res", "seen"), boot(meb, "b0", "split-res", "unseen")
    else:
        c1, c2 = boot(meb, "mem", "b0", "seen"), boot(meb, "mem", "b0", "unseen")
    rows.append(f"{label} & {fmt_ms(s, ss)} & {fmt_ms(u, us)} & {d:.1f} & {c1} & {c2} \\\\")
rr = ms(allr, "cloud-regional-only", "seen")[0], ms(allr, "cloud-regional-only", "unseen")[0]
rows.insert(0, f"Regional forecast only (no box model) & {rr[0]:.2f} & {rr[1]:.2f} & 67.3 & -- & -- \\\\")
save("stack_rows.tex", rows)

# ---- Table: node memory ablation with placebo (E8) ----
rows = []
for arm, label in [("b0", "Base (split-res + consensus)"), ("mem-surprise", "+ surprise only"), ("mem-profile", "+ profile only"),
                   ("mem", "+ surprise and profile (node memory)"), ("mem-placebo", "+ another node's table (placebo)")]:
    s, ss = ms(me, arm, "seen")
    u, us = ms(me, arm, "unseen")
    if arm == "b0":
        c1 = c2 = "reference"
    else:
        c1, c2 = boot(meb, arm, "b0", "seen"), boot(meb, arm, "b0", "unseen")
    rows.append(f"{label} & {fmt_ms(s, ss)} & {fmt_ms(u, us)} & {c1} & {c2} \\\\")
rows.append(f"Node memory vs. placebo & & & {boot(meb, 'mem', 'mem-placebo', 'seen')} & {boot(meb, 'mem', 'mem-placebo', 'unseen')} \\\\")
save("memory_rows.tex", rows)

# ---- Table: cloud graph model (E1) ----
w = pd.read_csv(ROOT / "wind" / "wind_results.csv")
wb = pd.read_csv(ROOT / "wind" / "wind_bootstrap.csv")
w = w[(w.stations == "seen") & (w.scope == "all") & (w.outputs == "1-6h")]
rows = []
for v, label, cmp_b in [("T0-noweather", "Own station, no weather", None), ("T0-nofuture", "Own station, past weather", ("T0-nofuture", "T0-noweather")),
                        ("T0", "Own station, + future weather (hindcast)", ("T0", "T0-nofuture")),
                        ("T1", "+ graph attention over stations", ("T1", "T0")), ("T2-era5", "+ ERA5 wind prior", ("T2-era5", "T1")),
                        ("T2-cpcb", "+ measured CPCB wind prior", ("T2-cpcb", "T1")), ("T2-rot180", "+ wind rotated by 180$^\\circ$ (placebo)", ("T2-rot180", "T1"))]:
    x = w[w.variant == v].skill
    c = "--"
    if cmp_b:
        r = wb[(wb.A == cmp_b[0]) & (wb.B == cmp_b[1])].iloc[0]
        c = f"{f2(r.rmse_change_pct)} [{f2(r.ci_low)}, {f2(r.ci_high)}]"
    rows.append(f"{label} & {x.mean():.2f}$\\pm${x.std():.2f} & {c} \\\\")
rows.append("ERA5 wind vs. placebo & -- & " + (lambda r: f"{f2(r.rmse_change_pct)} [{f2(r.ci_low)}, {f2(r.ci_high)}]")(wb[(wb.A == "T2-era5") & (wb.B == "T2-rot180")].iloc[0]) + " \\\\")
save("cloud_rows.tex", rows)

# ---- Table: stale / missing broadcast (E9) ----
b = pd.read_csv(ROOT / "buffer" / "results.csv")
g = b.groupby(["arm", "stations", "age"]).skill.mean()
rows = []
for st in ("seen", "unseen"):
    for arm, label in [("ctx", "Box alone"), ("short", "+1..+3 h broadcast"), ("long", "+1..+9 h path")]:
        rows.append(f"{'Known' if st == 'seen' else 'New'} & {label} & " + " & ".join(f"{g[(arm, st, a)]:.2f}" for a in (0, 1, 2, 3, 6, -1)) + " \\\\")
save("stale_rows.tex", rows)

# ---- Table: other tests (losses, data size, width, learned rule, fuzzy) ----
mo = pd.read_csv(ROOT / "moredata" / "bootstrap.csv")
tr = pd.read_csv(ROOT / "trigger" / "trigger_results.csv")
st_b = pd.read_csv(ROOT / "student" / "student_bootstrap.csv")
r20 = tr[tr.sent_pct == 20].iloc[0]
def mo_row(a, b_, s):
    r = mo[(mo.A == a) & (mo.B == b_) & (mo.stations == s)].iloc[0]
    return f"{f2(r.rmse_change_pct)} [{f2(r.ci_low)}, {f2(r.ci_high)}]"
def sp_row(a, b_):
    r = spb[(spb.A == a) & (spb.B == b_)].iloc[0]
    return f"{f2(r.rmse_change_pct)} [{f2(r.ci_low)}, {f2(r.ci_high)}]"
rows = [
    f"Trend loss (hinge on wrong direction) vs. Huber & {sp_row('ctx-trend', 'ctx')} & worse \\\\",
    f"AQI-breakpoint weighted loss vs. Huber & {sp_row('ctx-aqi', 'ctx')} & worse \\\\",
    f"1 M training windows vs. 200 k & {mo_row('1M-d32', '200k-d32', 'seen')} & better \\\\",
    f"Width 64 vs. width 32 (1 M windows) & {mo_row('1M-d64', '1M-d32', 'seen')} & no difference \\\\",
    f"Learned send rule vs. disagreement rule (20\\% sent) & {f2(r20.rmse_change_pct)} [{f2(r20.ci_low)}, {f2(r20.ci_high)}] & below threshold \\\\",
]
save("other_rows.tex", rows)

# ---- numbers quoted in the text, checked here once ----
s_ctx, u_ctx = ms(allr, "ctx", "seen")[0], ms(allr, "ctx", "unseen")[0]
s_mem, u_mem = ms(allr, "mem", "seen")[0], ms(allr, "mem", "unseen")[0]
d_seen = 100 * ((1 - s_mem / 100) / (1 - s_ctx / 100) - 1)
d_new = 100 * ((1 - u_mem / 100) / (1 - u_ctx / 100) - 1)
(OUT / "derived.txt").write_text(f"mem vs ctx RMSE change from skills: seen {d_seen:.2f} %, new {d_new:.2f} %\n")
print((OUT / "derived.txt").read_text())
for f in sorted(OUT.glob("*.tex")):
    print("==", f.name)
    print(f.read_text())
