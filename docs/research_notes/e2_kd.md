# E2 — output distillation (out_KD) vs no_KD (synthetic data; not results per [S11.3])

> FROZEN v1 result (ADR 0016). Teacher = interim LightGBM, which used **clean neighbour CPCB PM** as input ("clean-neighbour teacher", ADR 0019) — not comparable to the v2 sensor-domain graph teachers. CNN-B out_KD stopped after 2/3 seeds.

Cell = RMSE change vs no_KD in % (µg/m³, mean over seeds x 5 test time blocks); * = paired t-test p < 0.05. Deploy rule (stopped cloud session): no output significantly worse, mean RMSE lower, >= 1 output significantly better.

## RF-A (5 seeds x 5 test time blocks, paired t-test vs alpha1.0)

| variant | pm25_h1 | pm25_h2 | pm25_h3 | pm10_h1 | pm10_h2 | pm10_h3 | mean |
|---|---|---|---|---|---|---|---|
| alpha1.0 (RMSE 43.55) | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| alpha0.5 (RMSE 43.69) | +0.47* | +0.47* | +0.54* | -0.11 | +0.05 | +0.56* | +0.33 |
| alpha0.0 (RMSE 43.98) | +1.16* | +1.22* | +1.84* | -0.07 | +0.19 | +1.73* | +1.01 |

**Deploy:** `alpha1.0` — no KD variant qualifies.

## RF-B (5 seeds x 5 test time blocks, paired t-test vs alpha1.0)

| variant | pm25_h1 | pm25_h2 | pm25_h3 | pm10_h1 | pm10_h2 | pm10_h3 | mean |
|---|---|---|---|---|---|---|---|
| alpha1.0 (RMSE 76.19) | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| alpha0.5 (RMSE 75.89) | +0.03 | -0.18* | -0.17 | -0.16 | -0.49 | -0.87* | -0.31 |
| alpha0.0 (RMSE 75.82) | -0.14 | -0.09 | +0.15 | -0.23 | -0.96* | -0.81* | -0.35 |

**Deploy:** `alpha0.0` — no output significantly worse, mean RMSE lower, >=1 significantly better.

## CNN-A (3 seeds x 5 test time blocks, paired t-test vs lambda1.0)

| variant | pm25_h1 | pm25_h2 | pm25_h3 | pm10_h1 | pm10_h2 | pm10_h3 | mean |
|---|---|---|---|---|---|---|---|
| lambda1.0 (RMSE 44.18) | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| lambda0.5 (RMSE 43.75) | -1.65* | -1.31* | -0.77* | -1.80* | -0.60* | -0.63* | -1.13 |
| lambda0.7 (RMSE 43.87) | -1.17* | -1.05* | -0.49* | -1.30* | -0.43 | -0.46* | -0.81 |

**Deploy:** `lambda0.5` — no output significantly worse, mean RMSE lower, >=1 significantly better.

