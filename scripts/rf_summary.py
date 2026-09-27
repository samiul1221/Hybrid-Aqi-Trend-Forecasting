"""Print a compact summary of an RF model's meta.json: size, parity, skill per horizon."""

import json
import sys

m = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    m["model_id"],
    m["config"],
    "size",
    m["size_bytes"],
    "parity",
    f"{m['parity_max_abs']:.2e}",
    "n_train",
    m["n_train"],
)
for split, r in m["metrics"].items():
    rf, pe, di = r["rf"]["point"], r["persistence"]["point"], r["diurnal_persistence"]["point"]
    print(f"== {split} (n={r['rf']['n']})")
    for k in rf:
        print(
            f"  {k}: RMSE rf {rf[k]['rmse']:.1f} | persist {pe[k]['rmse']:.1f} | diurnal {di[k]['rmse']:.1f} ug/m3"
            f" | skill rf {rf[k]['skill']:+.3f} diurnal {di[k]['skill']:+.3f} | DM p {rf[k]['dm_p']:.1e}"
        )
    a, ap = r["rf"]["aqi"], r["persistence"]["aqi"]
    print(
        "  AQI MAE h1..3 rf",
        [round(a[f"aqi_h{h}"]["mae"], 1) for h in (1, 2, 3)],
        "persist",
        [round(ap[f"aqi_h{h}"]["mae"], 1) for h in (1, 2, 3)],
        "| cat acc rf",
        [round(a[f"aqi_h{h}"]["category_accuracy"], 3) for h in (1, 2, 3)],
        "| band POD/FAR/CSI rf",
        {k: round(v, 2) for k, v in a["band_crossing"].items() if k != "events"},
        "events",
        a["band_crossing"]["events"],
    )
