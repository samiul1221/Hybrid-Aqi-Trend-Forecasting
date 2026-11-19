# draws the result plots (vector PDF) from the raw result files
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "data" / "6_experiments"
OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})

# ---- Fig: skill of the box-model stack, seen and new places ----
sp = pd.read_csv(ROOT / "split" / "split_results.csv")
me = pd.read_csv(ROOT / "memory" / "results.csv")
allr = pd.concat([sp, me[me.arm != "split-res"]])
order = [("ctx", "box\nalone"), ("denoise", "+ cons.\nlevel"), ("split", "+ broad-\ncast"), ("split-res", "+ resid.\noutput"),
         ("b0", "+ cons.\n(b0)"), ("mem", "+ node\nmemory")]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
x = range(len(order))
for st, mk, ls in [("seen", "o", "-"), ("unseen", "s", "--")]:
    m = [allr[(allr.arm == a) & (allr.stations == st)].skill.mean() for a, _ in order]
    sd = [allr[(allr.arm == a) & (allr.stations == st)].skill.std() for a, _ in order]
    ax.errorbar(x, m, yerr=sd, marker=mk, ls=ls, lw=1, ms=3.5, capsize=2, color="k" if st == "seen" else "0.45",
                label="known places" if st == "seen" else "new places")
ax.set_xticks(list(x))
ax.set_xticklabels([l for _, l in order], fontsize=6.5)
ax.set_ylabel("Skill (%)")
ax.grid(alpha=0.3, lw=0.4)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_stack.pdf")

# ---- Fig: messages vs. gain (event-triggered personal cloud forecasts) ----
fu = pd.read_csv(ROOT / "fusion" / "B_event_triggered.csv")
fig, ax = plt.subplots(figsize=(3.5, 2.2))
for how, st, lab in [("oracle", ":", "oracle"), ("disagreement", "-", "disagreement rule"), ("random", "--", "random")]:
    q = fu[fu.how == how]
    ax.plot(q.sent_pct, q.skill, st, color="k", lw=1.1, marker="o", ms=3, label=lab)
ax.set_xlabel("Hours with a personal cloud forecast sent (%)")
ax.set_ylabel("Skill (%)")
ax.grid(alpha=0.3, lw=0.4)
ax.legend(frameon=False)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_messages.pdf")
print(fu[fu.how.isin(["disagreement", "random"])].to_string())
