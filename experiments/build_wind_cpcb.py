# Build the MEASURED station wind (CPCB WS/WD) on the same grid as data/4_features/grid_u*.npz.
# Why: every wind in the pipeline so far is ERA5 (~25 km grid, almost the same for all of Delhi).
# Output: data/4_features/wind_cpcb.npz  u, v [hour, station] (m/s, NaN = missing), plus coverage numbers.
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
g = np.load(ROOT / "data/4_features/grid_u0.npz")
stations, ts, W = list(g["stations"]), g["ts"], g["W"]
hours = pd.to_datetime(ts, unit="s", utc=True)


# one station: all yearly files -> WS, WD on the grid hours (CPCB timestamp = start of the IST hour)
def read_wind(sid):
    files = sorted((ROOT / "data/1_download/cpcb" / sid).glob("*.csv"))
    parts = []
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns
        if "WS (m/s)" not in cols or "WD (deg)" not in cols:
            continue
        parts.append(pd.read_csv(f, usecols=["Timestamp", "WS (m/s)", "WD (deg)"]))
    if not parts:
        return pd.DataFrame(index=hours, columns=["ws", "wd"], dtype=float)
    df = pd.concat(parts).rename(columns={"WS (m/s)": "ws", "WD (deg)": "wd"})
    df.index = pd.to_datetime(df.pop("Timestamp")).dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")
    df = df[~df.index.duplicated()].sort_index().reindex(hours)
    return df.apply(pd.to_numeric, errors="coerce")


# quality rules: physical range, and 6-hour stuck values (dead vanes repeat the same number)
def clean(df):
    bad = (df.ws < 0) | (df.ws > 30) | (df.wd < 0) | (df.wd > 360)
    for c in ("ws", "wd"):
        stuck = df[c].diff().eq(0).astype(int).rolling(6).sum().eq(6)
        bad |= stuck.reindex(df.index, fill_value=False)
    return df.mask(bad)


U = np.full((len(ts), len(stations)), np.nan, np.float32)
V = U.copy()
rows = []
for j, sid in enumerate(stations):
    d = clean(read_wind(sid))
    a = np.deg2rad(d.wd.to_numpy())                  # WD = direction the wind comes FROM
    U[:, j], V[:, j] = -d.ws * np.sin(a), -d.ws * np.cos(a)
    era_dir = np.rad2deg(np.arctan2(-W[:, j, 3], -W[:, j, 4])) % 360
    ok = np.isfinite(U[:, j]) & (d.ws.to_numpy() > 1) & (np.hypot(W[:, j, 3], W[:, j, 4]) > 1)
    diff = np.abs((d.wd.to_numpy()[ok] - era_dir[ok] + 180) % 360 - 180)
    rows.append(dict(station=sid, coverage_pct=round(100 * np.isfinite(U[:, j]).mean(), 1),
                     median_ws=round(float(np.nanmedian(d.ws)), 2),
                     era5_median_ws=round(float(np.nanmedian(np.hypot(W[:, j, 3], W[:, j, 4]))), 2),
                     dir_agree_45deg_pct=round(100 * float((diff < 45).mean()), 1) if ok.any() else np.nan))

rep = pd.DataFrame(rows)
print(rep.to_string())

# how different are the stations from each other at the same hour? (circular spread of direction, ws > 1 m/s)
def spread(u, v):
    s = np.hypot(u, v)
    ux, vx = np.where(s > 1, u / s, np.nan), np.where(s > 1, v / s, np.nan)
    n = np.isfinite(ux).sum(1)
    R = np.hypot(np.nanmean(ux, 1), np.nanmean(vx, 1))
    return np.nanmedian(np.rad2deg(np.sqrt(-2 * np.log(np.clip(R[n >= 8], 1e-6, 1)))))

print("median circular spread of wind direction across stations (deg): CPCB",
      round(float(spread(U, V)), 1), "| ERA5", round(float(spread(W[:, :, 3], W[:, :, 4])), 1))
print("distinct ERA5 wind series:", len({W[:5000, j, 3].tobytes() for j in range(len(stations))}), "of", len(stations))
np.savez_compressed(ROOT / "data/4_features/wind_cpcb.npz", u=U, v=V, ts=ts, stations=np.array(stations))
rep.to_csv(ROOT / "data/4_features/wind_cpcb_report.csv", index=False)
