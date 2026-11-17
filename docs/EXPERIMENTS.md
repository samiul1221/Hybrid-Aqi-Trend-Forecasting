# Experiment log: every hypothesis, how it was tested, what came out, what it means

Written 17 Nov 2026. This is the one file that lists **every experiment run so far** in this project (v1 training
run + experiments E1-E10 for v2): the question, the hypothesis, how it was implemented, the Kaggle notebook that ran
it, the result, the verdict, and the novelty (if any) that came out of it.

- The design itself is in `docs/ARCH_V3.md` (v2 kept for history). Ideas not yet run are in `docs/NOVELTY_IDEAS.md`.
- v1 details are in `docs/ABLATION_FINDINGS.md`.

> **All numbers are on simulated sensor boxes** (`simulator.py`) driven by real CPCB data (2018 - Sep 2026) and
> real ERA5 weather. No real box has been deployed. Treat every number as "evidence the design is worth building".

---

## 0. How to read the numbers

| term | meaning |
|---|---|
| **skill** | `1 − RMSE / RMSE(persistence)` in %. Persistence = "PM stays the same". 0 = no better than that lazy guess. Higher is better. |
| **% RMSE** (or "% error") | change of RMSE of arm A vs arm B. **Negative = A is better.** −1 % means 1 % smaller error. |
| **95 % CI** | week-block paired bootstrap (4 000 resamples of whole test weeks). If the CI contains 0, there is no real difference. |
| **seen / unseen** | the 27 training stations (test years) / the 5 hold-out stations never trained on (new places). |
| **test period** | Jan 2025 - Sep 2026. Train 2018-2023, validation 2024. |
| **kill threshold** | the pass / fail rule, written down **before** the run (E7-E9) and not changed afterwards. |
| **ctx** | the box-alone baseline student: GDC-CNN + raw "regime" context input (E3). Skill 16.0 seen, 17.5 unseen. |

Numbers in E6, E8, E9 and E10 use the **same student test windows**, so their skills can be compared across those
tables (ctx scores 16.01 seen in all of them). E1 (teacher, +1..+6 h), E2 (LightGBM probe) and E4 / E7 (saved
unit-0 forecasts) use other samples: compare them only inside their own table.

---

## 1. Where every experiment ran (Kaggle notebooks)

All notebooks are **private** on the Kaggle account `samiulseikh`. Each experiment notebook is **generated** by
`experiments/exp_<name>.py` (it re-uses notebook 04's own cells, so the data and the models are identical) and
pushed with `uv run python experiments/kaggle_exp.py push <name>`. Results are downloaded with
`... kaggle_exp.py get <name>` into `data/6_experiments/<name>/`.

| ID | Kaggle notebook | built by | machine | last run (UTC) | results folder | status |
|---|---|---|---|---|---|---|
| v1 | `samiulseikh/aqi-train` | `notebooks/04_train.ipynb` via `kaggle/kaggle_run.py` | T4 GPU | 4 Oct 2026 (results of 9 Oct) | `data/5_models/kaggle/` | done |
| E1 | `samiulseikh/aqi-exp-wind` | `exp_wind.py` (+ dataset `samiulseikh/aqi-wind-cpcb` from `build_wind_cpcb.py`) | T4 GPU | 14 Nov | `data/6_experiments/wind/` | done |
| E2 | `samiulseikh/aqi-exp-probe` | `exp_probe.py` | CPU | 14 Nov | `data/6_experiments/probe/` | done |
| E3 | `samiulseikh/aqi-exp-student` | `exp_student.py` | CPU | 14 Nov | `data/6_experiments/student/` | done |
| helper | `samiulseikh/aqi-exp-teachpred` | `exp_teachpred.py` | CPU | 14 Nov | `data/6_experiments/teachpred/` | done |
| E4 | none (run locally, no training) | `exp_fusion.py` | laptop | 14 Nov | `data/6_experiments/fusion/` | done |
| E5 | `samiulseikh/aqi-exp-netctx` | `exp_netctx.py` | CPU | 14 Nov | `data/6_experiments/netctx/` | done |
| E6 | `samiulseikh/aqi-exp-split` | `exp_split.py` | CPU | 14 Nov | `data/6_experiments/split/` | done |
| E7 | `samiulseikh/aqi-exp-trigger` | `exp_trigger.py` | CPU | 16 Nov | `data/6_experiments/trigger/` | done |
| E8 | `samiulseikh/aqi-exp-memory` | `exp_memory.py` | CPU | 16 Nov | `data/6_experiments/memory/` | done |
| E9 | `samiulseikh/aqi-exp-buffer` | `exp_buffer.py` | CPU | 16 Nov | `data/6_experiments/buffer/` | done |
| E10 | `samiulseikh/aqi-exp-moredata` | `exp_moredata.py` | T4 GPU | 16 Nov | `data/6_experiments/moredata/` | done |

Nothing is running on Kaggle now (checked 17 Nov). The older kernels `aqi-teacher` and `aqi-forecast` (2-4 Oct)
are earlier versions of the training notebook and are not used for any number here.

> **Note on E8-E10:** these three were first pushed as smoke tests (2 epochs, 4 000 windows). The smoke outputs are
> kept in `data/6_experiments/<name>_smoke/` and mean nothing. The real 3-seed (E10: 2-seed) results were
> downloaded on 17 Nov into `data/6_experiments/<name>/` and are the ones below.

---

## 2. All experiments at a glance

| ID | question | verdict | key number | novelty that came out |
|---|---|---|---|---|
| v1 | Does a graph teacher distilled into an MCU student work? | partly | teacher 20.8, student 15.0 (S0 14.0) | fair distillation; "you can't distil what the box can't see" |
| E1 | Is "wind adds nothing" real? | **yes, real** | graph −2.6 %; wind = placebo | placebo-controlled wind test (negative result) |
| E2 | Does wind-directed neighbour info help a simple model? Which sensors pay for their power? | no / fans + MQ-7 | neighbours −1.0 % at +1 h, upwind = downwind | sensor value per watt (all 128 subsets) |
| E3 | What makes the box-alone student better? | ctx input | −0.49 % | (fuzzy gate = interpretable only) |
| E4 | How should box and cloud forecasts be combined? | trigger + path re-base | 66 % of gain at 20 % of messages | event-triggered sending; path re-basing rule |
| E5 | Can a few bytes of network context help the box? | yes, 1 byte | −0.71 % fresh, no harm stale | byte-level context, no-regret (for 1 byte) |
| E6 | Can the box predict only its local difference from a cloud regional forecast? | **yes, best** | −1.58 %; unseen 17.5 → 19.9 | **split cloud-edge forecaster (main claim)** |
| E7 | Can a learned send rule beat the disagreement rule? | **fails** | +0.16 skill (needed +0.5) | negative: simple rule ≈ best possible |
| E8 | Does the node's per-hour memory of its offset help? | **passes** | −1.09 % seen, −0.89 % unseen vs b0 | **node memory (new supporting claim)** |
| E9 | Does a +9 h buffered path keep a stale broadcast useful? | **fails** | stale ≥ 2-3 h ≈ no broadcast; missing hurts | negative: robustness training has a price |
| E10 | Is the student short of training data? | **yes** | 1 M windows −1.06 %; width 64 adds 0 | (a fix, not a novelty) |

---

## 3. v1: the original teacher-student run (`aqi-train`, notebook 04)

**Kaggle:** `samiulseikh/aqi-train`, T4 GPU, `notebooks/04_train.ipynb` (Part A teacher, Part B student). Full
write-up: `docs/ABLATION_FINDINGS.md`.

**Hypotheses**
- H1: a graph over all stations (TI-STGNN) forecasts better than one station alone.
- H2: a wind-direction prior and an advection-diffusion-reaction (ADR) physics block improve the graph.
- H3: distillation (KD) passes the teacher's network knowledge into a 15k-parameter single-station student.
- H4: copying the teacher's spatial / physics states helps the student more than copying its forecast.
- H5: every sensor in the fixed pool is useful.

**How:** teacher variants T0 (own station) → T1 (+graph) → T2 (+wind prior) → T3 (+ADR), 3 seeds; out-of-fold
teacher outputs for the student; students S0 (alone), S1 (copy forecast), S2 (+ time features / attention),
S3 (+ neighbour state), S4 (+ physics state); sensor ablation (drop one sensor at a time); baselines: linear,
LightGBM, persistence.

**Results**

| hypothesis | result | verdict |
|---|---|---|
| H1 graph | T0 21.7 → T1 23.6 (1-6 h skill) | **supported** |
| H2 wind / ADR | T2 23.6, T3 23.8 (≤ 0.2, noise level) | **rejected** |
| H3 KD | S0 14.0 → S2-T0 15.0 (+0.9); teacher 20.8. Student keeps only 12-24 % of the teacher's lead | weak |
| H4 spatial KD | S3 / S4 worse than S1 / S2 | **rejected** |
| H5 sensors | MQ-7 −0.45, fans −0.26, SGP41 −0.14, BMP280 −0.09; MQ-135, light, rain ≈ 0 | partly |

Deployment check: starting from the box's own (uncalibrated) PM, the AQI colour is right only 43 % of the time
(71 % from the true PM). The level needs co-location calibration.

**What we learned:** the teacher's advantage is *seeing other nodes*. A box that sees only itself cannot be taught
that. This one finding motivated all of v2: **give the box a few bytes of network information instead of trying
to distil it.**

**Novelty out of it:** fair distillation (teacher and student see identical simulated node features, true PM is a
target only), and the negative result "distilling spatial / physics states into a single-station student hurts".

---

## 4. The v2 experiments, one by one

### E1. Wind re-test (is "wind adds nothing" real?)

- **Kaggle:** `samiulseikh/aqi-exp-wind` (T4 GPU). **Script:** `experiments/exp_wind.py`; measured wind built by
  `experiments/build_wind_cpcb.py` and uploaded as dataset `samiulseikh/aqi-wind-cpcb`.
- **Hypothesis:** v1's "wind adds nothing" may be an artefact of coarse ERA5 wind or of one seed. If wind direction
  matters, the ERA5 wind prior and measured CPCB wind should beat a **placebo** (wind rotated 180°).
- **How:** notebook 04 Part A with two small patches; arms T1 (graph, distance prior), T2-era5, T2-cpcb (measured
  station WS / WD, ERA5 where missing), T2-rot180 (placebo), and T0 / T0-nofuture / T0-noweather (no graph, to split
  the teacher's lead into "neighbours" vs "weather"). 3 seeds; scored overall, by wind-speed bin and by season.
- **Result (skill +1..+6 h, seen):**

| teacher | skill |
|---|---|
| T0 own station, no weather | 22.82 |
| T0 past weather only | 23.01 |
| T0 + future weather (hindcast) | 23.99 |
| **T1 + graph** | **25.97** |
| T2 + ERA5 wind | 26.11 |
| T2 + measured CPCB wind | 26.00 |
| T2 + wind rotated 180° (placebo) | 25.95 |

| comparison | % RMSE | 95 % CI |
|---|---|---|
| graph vs no graph | **−2.60** | −2.93 to −2.27 |
| future weather vs none | −1.27 | −1.61 to −0.94 |
| ERA5 wind vs no wind | −0.19 | −0.40 to +0.03 |
| ERA5 wind vs placebo | −0.21 | −0.43 to +0.01 |
| measured wind vs no wind | −0.04 | −0.24 to +0.15 |

- **Verdict:** the graph is real (−2.6 %). Wind direction is **not**: it equals the placebo. The learned wind
  weight β stays near 0.
- **Why:** ERA5 has only 4 distinct wind series for 32 stations; CPCB vanes disagree by a median 70°; Delhi moves
  as one airshed; at 0-2 m/s air needs 1-4 h to cross 5-15 km (details: `ARCH_V2.md` §11).
- **Novelty:** a **placebo-controlled** wind test (180° rotation, measured vs reanalysis wind, 3 seeds). An honest,
  publishable negative result. Also: the live cloud can use T1 (simpler, same accuracy).

### E2. Model-free probes (LightGBM) and sensor value per watt

- **Kaggle:** `samiulseikh/aqi-exp-probe` (CPU). **Script:** `experiments/exp_probe.py`.
- **Hypotheses:** P1, an upwind-weighted neighbour signal predicts a station's PM change better than a
  distance-weighted (or downwind, placebo) one. P2, a tuned wind-transport kernel finds a clear upwind effect
  (κ > 0). P3, some optional sensors are not worth their power.
- **How:** P1, LightGBM arms with neighbour features weighted by distance / upwind ERA5 / upwind measured / downwind.
  P2, differential evolution tunes `w = d^−p · exp(κ · cos(angle) · min(ws/ws0, 1))`. P3, all 2⁷ = 128 subsets of
  the 7 optional sensors (exhaustive, so no genetic algorithm needed), Pareto front of skill vs power.
- **Result:** distance-weighted neighbours −1.0 % at +1 h (CI −1.59 to −0.42); upwind ≈ downwind placebo;
  κ ≈ 0.69 but only ~0.06 % gain. Sensors (power TODO_VERIFY):

| best set at this power | W | skill |
|---|---|---|
| core (PM + T/RH + clock) | 0.55 | 17.74 |
| + fans | 0.56 | 18.42 |
| + fans, BMP280, SGP41, rain, light | 0.58 | 18.84 |
| + MQ-7 | 0.93 | 19.17 |
| + MQ-135 | 1.73 | 19.22 |

- **Verdict:** neighbours help as a **regional level**, not as directed transport. Fans and MQ-7 are worth it;
  MQ-135 adds +0.04 for +0.8 W.
- **Novelty:** **sensor value per watt** (exhaustive subset search), a design rule for the node. The finding
  "neighbours act as one regional level" led directly to E5 and E6.

### E3. What makes the box-alone student better?

- **Kaggle:** `samiulseikh/aqi-exp-student` (CPU). **Script:** `experiments/exp_student.py`.
- **Hypotheses:** instance normalisation throws away the absolute level and regime (PM level, RH, season, CO), so
  giving it back helps (ctx). A fuzzy Takagi-Sugeno gate (ANFIS-like) captures regimes better. Weather now / forecast
  helps the box.
- **How:** same GDC-CNN and data as notebook 04, 3 seeds per arm: S0, S0+ctx, FZ (4 Gaussian rules, each with its own
  linear head), S0+wx (weather now + next 3 h, hindcast), S0+wxnow, FZ+wx. Bootstrap vs S0.
- **Result:**

| student | skill seen | skill unseen | params |
|---|---|---|---|
| S0 | 15.64 | 17.16 | 14 278 |
| **S0 + ctx** | **16.02** | **17.49** | 14 790 |
| FZ (fuzzy) | 15.89 | 17.34 | 21 280 |
| S0 + weather now / past | 15.57 | 17.30 | |
| S0 + weather forecast | 14.85 | 16.52 | |

ctx vs S0: −0.49 % (CI −0.63 to −0.36). Fuzzy vs ctx: +0.02 % (no difference).
- **Verdict:** ctx kept (it is the "box-alone" reference from here on). Fuzzy = ctx but readable (strongest rule:
  "winter, humid, high PM, high CO"). Weather does not help the box.
- **Novelty:** none claimed; the fuzzy gate is an interpretability extra.

### Helper. Saved teacher forecasts (`aqi-exp-teachpred`)

- **Kaggle:** `samiulseikh/aqi-exp-teachpred` (CPU). **Script:** `experiments/exp_teachpred.py`.
- The best v1 teacher (T3, seed 2) forecasts every test hour at unit 0, so E4 could run without waiting for a GPU.

### E4. Cloud-box fusion and messaging

- **Where:** run locally, no training (`experiments/exp_fusion.py`) on the saved forecasts of E1 / helper and E3.
- **Hypotheses:** A, a stale cloud forecast is still useful if moved to "now" correctly. B, sending the personal
  cloud forecast only when cloud and box disagree (the cloud runs a twin of the student) beats random sending.
  C, online-adaptive blend weights (Bates-Granger, Jacobson / TCP-RTT style) beat fixed weights.
- **How:** A, link dies `a` hours ago, re-base the old forecast (on the box's sensor, or on its own path);
  B, send to the top-k % disagreement hours vs random vs oracle, and under a token bucket (messages per day);
  C, weights from recent errors against the box's own sensor.
- **Result:** box alone 15.66, cloud every hour 23.30.

| % of hours sent | disagreement | random | oracle |
|---|---|---|---|
| 5 | 18.10 | 16.01 | 21.68 |
| 10 | 19.31 | 16.42 | 24.25 |
| **20** | **20.72** | 17.10 | 27.19 |
| 50 | 22.63 | 19.41 | 30.01 |

Stale personal forecast: re-basing on the box's sensor gives **−71 % skill** (worse than nothing); re-basing on the
forecast's own path + fixed blend gives 19.4 at 1 h, 18.3 at 2 h, 17.8 at 3 h (box alone 15.9). Online weights =
fixed when stale, worse when fresh.
- **Verdict:** disagreement-triggered sending keeps ~66 % of the cloud gain at 20 % of the messages (random: ~19 %).
  Path re-basing works, sensor re-basing does not. Fixed blend weights are enough.
- **Novelty:** **event-triggered personal forecasts with a student twin and a token bucket** (applied from
  networking); the **path re-basing rule** for stale forecasts (small, but a real design rule).

### E5. A few bytes of network context (netctx)

- **Kaggle:** `samiulseikh/aqi-exp-netctx` (CPU). **Script:** `experiments/exp_netctx.py`.
- **Hypotheses:** N1, Delhi stations share one regional signal (clusters). N2, boxes can agree on the regional mean
  by radio gossip without the cloud. N3, **one byte** per hour (regional mean of the boxes' own ln PM2.5, 8-bit)
  helps the student, and is no-regret when stale or missing.
- **How:** N1, cluster stations by the correlation of their 3-h changes. N2, push-sum gossip within 10 km, 20
  rounds, 20 % packet loss. N3, students get the regional mean (float, 8-bit, per cluster, or gossip), trained with
  20 % missing and 0-3 h old context plus an age input; scored fresh, 1-3 h old and missing.
- **Result:** one cluster of 28 stations + 4 singletons. Gossip error 0.32 vs city mean (own reading 0.69).

| context | fresh | 1 h | 2 h | 3 h | missing |
|---|---|---|---|---|---|
| **regional mean, 8-bit** | **−0.71 %** | −0.20 % | +0.01 % | +0.09 % | +0.10 % |
| per-cluster, 8-bit | −0.70 % | −0.27 % | −0.02 % | +0.10 % | +0.09 % |
| gossip (no cloud) | −0.33 % | −0.10 % | +0.03 % | +0.13 % | +0.06 % |

- **Verdict:** 1 byte is enough (8-bit = float); per-cluster = city-wide (one airshed); gossip gives half; stale
  and missing are within noise of no context.
- **Novelty:** **byte-level regional context with staleness / dropout training, no-regret for a 1-byte level**.
  (E9 later shows that this no-regret does **not** carry over automatically to the full split model.)

### E6. Split cloud-edge forecaster (the main result)

- **Kaggle:** `samiulseikh/aqi-exp-split` (CPU). **Script:** `experiments/exp_split.py`.
- **Hypothesis:** the cloud, averaging many noisy boxes, can forecast the city-wide change well; the box should then
  only predict **its own local difference** (common-mode rejection). Also: custom trend / AQI losses improve
  direction or category accuracy.
- **How:**
  - Regional forecaster: LightGBM, 6 outputs (city-mean true ln-PM change, PM2.5 / PM10, +1..+3 h) from regional
    means of the boxes' own sensors + current ERA5 + clock; **out-of-fold** over 3 two-year blocks.
  - Student arms (GDC-CNN + ctx, 3 seeds): ctx; denoise (consensus filter `d_i = R + EWMA_24h(S_i − R)`);
    split (6 regional forecasts + gap to region as inputs); **split-res** (output = `R̂ + s · Dense6(...)`);
    ctx-trend (Huber + hinge on wrong direction); ctx-aqi (errors weighted near CPCB breakpoints).
  - All arms trained with a **fresh** broadcast (no staleness training).
- **Result:**

| arm | skill seen | skill unseen | direction acc. | % RMSE vs ctx (CI) |
|---|---|---|---|---|
| ctx | 16.01 | 17.48 | 69.8 % | |
| consensus filter | 16.83 | 18.56 | 70.4 % | −0.90 (−1.05 to −0.74) |
| split | 17.46 | 19.34 | 71.0 % | −1.43 (−1.72 to −1.14) |
| **split-res** | **17.73** | **19.92** | **71.3 %** | **−1.58 (−1.90 to −1.27)** |
| trend loss | 15.62 | 16.93 | 69.6 % | +0.64 (worse) |
| AQI loss | 15.59 | 16.97 | 69.6 % | +0.38 (worse) |
| regional forecast only | 11.06 | 14.44 | 67.3 % | |

- **Verdict:** **strongly supported.** Most of the gain comes from giving the box the broadcast; the residual output
  adds −0.16 % (not significant alone) and is kept for interpretability. Custom losses are worse.
- **Novelty:** **the split cloud-edge forecaster** (main contribution): the cloud forecasts the shared regional
  change in a few bytes, the MCU model predicts only the local difference. Largest gain at new places (+2.4 skill).
  Negative result: custom trend / AQI losses hurt.

### E7. Learned send rule

- **Kaggle:** `samiulseikh/aqi-exp-trigger` (CPU). **Script:** `experiments/exp_trigger.py`.
- **Hypothesis:** the oracle in E4 (27.2 at 20 %) is far above the disagreement rule (20.7). A model that predicts
  the benefit of sending, from what the cloud knows at hour t, can close part of that gap.
- **How:** target = (box error)² − (cloud error)² of that hour (true PM as target only). Inputs: both forecasts and
  their gaps, the teacher's +4..+6 h, the city-mean teacher forecast, the box's uploaded sensors, clock. LightGBM fitted
  on the **first half** of the test period, every rule scored on the **second half** (saved forecasts exist only for
  the test period). **Kill:** learned − disagreement < +0.5 skill at 20 %, or CI contains 0.
- **Result:** box alone 15.87, cloud always 23.53 (second half).

| % sent | random | disagreement | learned | oracle | learned vs disagreement (CI) |
|---|---|---|---|---|---|
| 5 | 16.22 | 18.38 | 18.44 | 21.98 | −0.07 (−0.23 to +0.09) |
| 10 | 16.61 | 19.59 | 19.63 | 24.55 | −0.05 (−0.23 to +0.14) |
| **20** | 17.31 | 20.97 | **21.12** | 27.48 | **−0.20 (−0.34 to −0.05)** |
| 50 | 19.58 | 22.87 | 22.98 | 30.23 | −0.14 (−0.25 to −0.03) |

Token bucket: best gain +0.26 skill at 4 and 6 messages / day; none at 1, 2 or 12.
- **Verdict:** **fails** (+0.16 skill vs +0.5 required). The cloud-box gap carries ~42 % of the model's importance;
  nothing else reaches 6 %.
- **Novelty:** negative result: **the simple disagreement rule is close to the best the cloud can do** with what it
  knows; the oracle gap is mostly unpredictable luck.

### E8. Node memory of its "usual self"

- **Kaggle:** `samiulseikh/aqi-exp-memory` (CPU, ~10 min per arm, 3 seeds). **Script:** `experiments/exp_memory.py`.
- **Hypothesis:** the consensus filter keeps one 24-h average offset from the region, blind to the clock. Local
  sources (traffic peaks, evening burning) repeat at the same hours and differ by site. A per-clock-hour memory of the
  box's offset from the region tells the student (a) how unusual it is now and (b) how its offset usually moves in
  the next 1-3 h. The gain must be **site-specific**, so another node's table (placebo) must not give it.
- **How:**
  - offset `o_i(t) = own ln PM_i(t) − R(t)` (R = regional mean from the broadcast; no true PM)
  - memory `M_i(c)` = EWMA over past days (half-life 10 days) of `o_i` at clock hour c (24 × 2 = 48 numbers in flash)
  - surprise = `o_i(t) − M_i(clock t)` (2 numbers); profile = `M_i(clock t+h) − M_i(clock t)`, h = 1..3 (6 numbers)
  - the entry for t+h was last updated at t+h−24, so there is no future data
  - arms: split-res; **b0** = split-res + consensus filter; mem = b0 + surprise + profile; mem-surprise;
    mem-profile; mem-placebo (every node reads the table of node i+7)
  - **Kill:** mem vs b0 weaker than −0.3 %, or CI contains 0, or mem does not beat placebo with CI excluding 0.
- **Result (3 seeds):**

| arm | skill seen | skill unseen | +1 h PM2.5 skill (seen) | direction acc. | category acc. |
|---|---|---|---|---|---|
| split-res | 17.73 | 19.92 | 11.45 | 71.26 % | 71.05 % |
| b0 (split-res + consensus) | 17.97 | 20.29 | 11.73 | 71.40 % | 71.14 % |
| mem-placebo | 18.03 | 20.31 | 11.77 | 71.43 % | 71.14 % |
| mem-surprise | 18.24 | 20.74 | 11.93 | 71.57 % | 71.17 % |
| mem-profile | 18.46 | 20.56 | 12.13 | 71.65 % | 71.29 % |
| **mem (both)** | **18.90** | **21.06** | **12.52** | **71.92 %** | **71.42 %** |

| comparison | seen % RMSE (CI) | unseen % RMSE (CI) |
|---|---|---|
| b0 vs split-res | −0.27 (−0.35 to −0.19) | −0.44 (−0.63 to −0.25) |
| **mem vs b0** | **−1.09 (−1.27 to −0.91)** | **−0.89 (−1.07 to −0.70)** |
| mem vs placebo | −0.92 (−1.08 to −0.76) | −0.77 (−0.96 to −0.58) |
| mem-surprise vs b0 | −0.38 | −0.61 |
| mem-profile vs b0 | −0.56 | −0.29 |
| placebo vs b0 | −0.17 | −0.12 |

- **Verdict:** **passes all three kill conditions**, on seen and unseen stations. Surprise and profile each help and
  together help more than either. The placebo gets only ~1/6 of the gain, so the gain is **site-specific**.
- **What we learned:** the box's own daily pattern of "how I differ from the city" is real information the student
  does not extract from its 24-h window by itself. The consensus filter, by contrast, adds only −0.27 % on top of
  split-res (its −0.90 % alone overlaps with what the broadcast already gives).
- **Novelty:** **node memory**: a 48-number per-clock-hour table of the node's offset from the regional broadcast,
  0 extra bytes on the link, ~300 parameters. It is the part of the split design that is about **the node itself**:
  the cloud gives the common part, the node remembers its own habits. Borrowed idea (error-correction / climatology
  anomalies), so claim it as an applied, placebo-controlled component, not a new algorithm. (Honest note: the cloud
  could also compute this table from the uploads; doing it on the node costs no bytes and keeps the table on the box.)

### E9. Buffered broadcast ("send the plan, not the next step")

- **Kaggle:** `samiulseikh/aqi-exp-buffer` (CPU, ~14 min per arm, 3 seeds). **Script:** `experiments/exp_buffer.py`.
- **Hypothesis:** a +1..+3 h broadcast is useless after 3 h because it no longer covers any future hour. A +1..+9 h
  regional path, shifted along by the box (`change = path[a+h] − path[a]`), should keep most of its value at 3 h.
- **How:** regional LightGBM forecasts 18 outputs (+1..+9 h × 2 pollutants). Arms (all split-res, residual output):
  ctx (no broadcast), short (+1..+3 h; at age a only horizons with a+h ≤ 3 exist), long (+1..+9 h). Broadcast arms
  also get the region-minus-own gap at broadcast time, the age and "have" flags. **All arms trained with a random
  age 0-6 h and 15 % missing** (so only ~1 in 8 training samples is fresh). Scored at ages 0, 1, 2, 3, 4, 6 h and
  missing. **Kill:** at 3 h, long keeps < 40 % of its fresh gain vs ctx, or CI of long vs short at 3 h contains 0;
  and long at age 0 must not be worse than short by > 0.2 %.
- **Result (skill, 3 seeds; ctx = 16.01 seen, 17.48 unseen at every age):**

| arm | stations | age 0 | 1 h | 2 h | 3 h | 4 h | 6 h | missing |
|---|---|---|---|---|---|---|---|---|
| short | seen | 16.19 | 15.82 | 15.98 | 16.01 | 16.01 | 15.99 | 16.00 |
| long | seen | **16.68** | 16.19 | 16.12 | 15.99 | 15.87 | 15.73 | **15.42** |
| short | unseen | 18.22 | 17.33 | 17.46 | 17.36 | 17.36 | 17.38 | 17.38 |
| long | unseen | **18.59** | 17.93 | 17.84 | 17.72 | 17.56 | 17.39 | **16.59** |

| % RMSE (CI) | age 0 | 1 h | 3 h | missing |
|---|---|---|---|---|
| long vs ctx, seen | −0.58 (−0.86 to −0.29) | +0.02 (n.s.) | +0.27 (+0.05 to +0.52) | **+1.01 (+0.77 to +1.28)** |
| long vs ctx, unseen | −1.17 (−1.48 to −0.85) | −0.34 (−0.59 to −0.08) | −0.09 (n.s.) | **+1.38 (+1.04 to +1.77)** |
| long vs short, seen | −0.45 (−0.67 to −0.26) | −0.32 | +0.14 (n.s.) | +0.87 |
| long vs short, unseen | −0.33 | −0.62 | −0.36 (−0.62 to −0.09) | +1.13 |

- **Verdict:** **fails.** At 3 h the long path keeps nothing on seen stations (slightly worse than ctx) and ~7 % of
  its fresh gain on unseen ones. Long does beat short for ages 0-3 h on unseen stations and 0-1 h on seen ones.
- **What we learned (more important than the idea itself):**
  1. **Staleness training costs most of the fresh gain.** The same split-res trained with a fresh broadcast (E6)
     scores 17.73; trained with random ages 0-6 h and 15 % missing it scores 16.19 (short) / 16.68 (long) when fresh.
     The model can't fully trust a broadcast that is usually old.
  2. **A missing broadcast now hurts** (+1.0 % seen, +1.4 % unseen for long). With the residual output
     `R̂ + correction`, a missing R̂ is set to 0 and the model has to produce the whole change from the correction
     alone, which it saw in only 15 % of training. So the "no-regret" result of E5 (1-byte level, plain input) does
     **not** carry over to the full split model.
  3. A longer path helps **even when fresh** (long vs short at age 0: −0.45 % seen), so the +4..+9 h part carries
     useful information about the regional trajectory.
- **Novelty:** negative / design result: **for a cloud-assisted MCU model, robustness to stale links has an accuracy
  price, and a residual output makes a missing broadcast harmful**. Fix to test next: keep two models on the box
  (split model when the broadcast is fresh, ctx student otherwise), see Section 7.

### E10. Does the student need more training data?

- **Kaggle:** `samiulseikh/aqi-exp-moredata` (T4 GPU, 2 seeds). **Script:** `experiments/exp_moredata.py`.
- **Hypothesis:** every student so far trained on 200 000 of the 2.6 M training windows. The teacher's lead over the
  student (~6 skill) is larger than the graph explains (~2), so the student may be short of data, or of width.
- **How:** the same S0+ctx student, same validation and test windows; train on 200k (reference), 1 M random training
  windows (27 training stations, 2018-23), and 1 M at width 64. No kill threshold (a diagnosis).
- **Result:**

| run | skill seen | skill unseen | params | % RMSE vs 200k, seen (CI) | unseen (CI) |
|---|---|---|---|---|---|
| 200k, width 32 | 15.96 | 17.46 | 14 790 | | |
| **1 M, width 32** | **16.82** | **18.07** | 14 790 | **−1.06 (−1.30 to −0.81)** | **−0.77 (−1.11 to −0.43)** |
| 1 M, width 64 | 16.58 | 17.97 | 48 006 | −1.00 | −0.84 |

1 M width 64 vs 1 M width 32: +0.06 % seen, −0.08 % unseen (no difference).
- **Verdict:** **the student is data-limited, not size-limited.** 5× more windows gives about as much as the ctx
  input and the consensus filter together; doubling the width gives nothing.
- **What we learned:** the 15k-parameter student is big enough for the ESP32; every arm of the final comparison
  should be trained on ≥ 1 M windows. Not a novelty, but it changes every number before the paper.

---

## 5. What we know now (the learnings in one place)

1. **The network is the information, not the physics.** Seeing other nodes is worth −2.6 % for the teacher; wind
   direction and ADR physics add nothing measurable at 1-6 h in Delhi (placebo-controlled).
2. **Delhi is one airshed** at this spacing: one cluster of 28 stations; per-cluster context = city-wide context.
3. **Distillation can't carry network knowledge into a box that can't see the network** (v1: +0.9 skill only).
   Sending the network's summary as a few bytes works much better (E6: +1.7 seen, +2.4 unseen).
4. **The box's job is the local difference.** The best box model stacks: cloud regional forecast (split-res) +
   consensus level + **node memory**. Skill on the same test windows:

| box model | skill seen | skill unseen |
|---|---|---|
| ctx (box alone) | 16.01 | 17.48 |
| + regional broadcast, residual (split-res, E6) | 17.73 | 19.92 |
| + consensus filter (b0, E8) | 17.97 | 20.29 |
| **+ node memory (mem, E8)** | **18.90** | **21.06** |

   From box alone to mem: about −3.4 % RMSE seen and −4.3 % unseen (derived from the skills on the same test windows;
   no direct bootstrap of mem vs ctx yet). Direction accuracy 69.8 % → 71.9 %.
5. **Data before size:** 5× more training windows gives −1.06 %; double width gives nothing (E10). This was not yet
   applied to the split / memory models, so the stack above should grow further.
6. **Messaging:** disagreement-triggered sending is near the best achievable (E7); 4-6 messages per day already give
   most of the benefit.
7. **Stale forecasts:** re-base on the forecast's own path, never on the box's sensor (E4).
8. **Robustness has a price (E9):** training one model for every broadcast age makes it weak when fresh, and the
   residual output makes a missing broadcast harmful. The "no-regret" claim is only proven for the 1-byte level (E5).
9. **What does not help:** custom trend / AQI losses, fuzzy gate (= ctx), weather on the box, online blend weights,
   learned send rule, spatial-state KD.
10. **Hardware:** fans and MQ-7 earn their place; MQ-135 adds ~0 for ~0.8 W (simulated; TODO_VERIFY power).

---

## 6. Novelty to include (ranked)

| rank | claim | evidence | how to word it |
|---|---|---|---|
| 1 | **Split cloud-edge forecaster** for low-cost AQI nodes: the cloud forecasts the shared regional change from all nodes (6-8 bytes), the MCU model predicts only the local difference | E6: −1.58 % (CI −1.90 to −1.27); unseen 17.5 → 19.9 | main contribution |
| 2 | **Node memory**: a per-clock-hour table of the node's offset from the broadcast, placebo-controlled | E8: −1.09 % seen / −0.89 % unseen on top of split + consensus; placebo gets ~1/6 | supporting component; "the node remembers its own habits" |
| 3 | **Bandwidth-accuracy curve** with event-triggered personal forecasts (student twin + token bucket), and the proof that a learned rule cannot beat the simple one | E4 + E7 | supporting; negative result included |
| 4 | **The price of robustness**: staleness training vs fresh accuracy; residual output and missing broadcasts | E5 vs E9 | design lesson (after the E11 fix is tested, it becomes "how to make it no-regret") |
| 5 | **Placebo-controlled negative results**: wind direction (180° rotation, measured vs ERA5), ADR, spatial KD, custom losses | E1, E2, v1, E6 | honest negative results section |
| 6 | **Sensor value per watt** (exhaustive 128-subset Pareto) | E2 | design rule for the node |
| 7 | Path re-basing for stale forecasts | E4 | small design rule |

**Do not claim:** "first" anything before the literature check; global + local residual forecasting in general
(hierarchical forecasting / MOS exist); dual prediction, send-on-delta, consensus, gossip, token bucket as
inventions (we apply and measure them); "no-regret" for the full split model (E9 shows the opposite so far).

**Paper sentence (updated):** *"A split cloud-edge forecaster: the cloud uses all nodes to predict the shared
regional change in a few bytes, the on-device model predicts only the local difference using a small memory of the
node's own daily habits. On simulated nodes this improves the on-device forecast by about 3.4 % RMSE at known sites
and 4.3 % at new sites, and a disagreement-triggered personal forecast recovers two thirds of the full cloud model's
gain at one fifth of the messages."*

---

## 7. What to run next (kill thresholds fixed now)

| step | experiment | why | cost | keep it if |
|---|---|---|---|---|
| **E11** | **Final combined model on 1 M windows**: ctx vs split-res vs b0 vs mem, all trained on 1 M windows, 3 seeds; plus mem + KD soft targets | turns E6 + E8 + E10 into one clean headline table with a direct mem-vs-ctx bootstrap | 1 GPU run (~3-4 h) | mem beats ctx by ≥ 3 % with CI excluding 0 on seen and unseen |
| **E12** | **Two models on the box** (fixes E9): fresh-trained mem model when the broadcast is ≤ 1 h old, ctx student otherwise; vs one model trained with mostly-fresh ages (70 % fresh, 15 % 1-2 h, 15 % missing) | make the split design truly no-regret without losing the fresh gain | 1 CPU run (~3 h) | fresh within 0.2 % of the fresh-only model AND missing within +0.1 % of ctx |
| E13 | **Fault robustness** (idea 14): stuck / spike / +30 % drift / dead PM sensor for 6 h, 24 h, 7 d; ctx vs mem vs mem with fault augmentation; virtual-sensor fallback for a dead sensor | new robustness claim; memory and consensus use the box's own PM, so faults matter more now | 1 CPU run | mem loses less skill than ctx under faults |
| E14 | **Node-count test**: broadcast built from 5, 10, 15, 27 nodes | how many nodes a city needs; supports "the nodes are essential" | 1 CPU run | at 10 nodes ≥ half of the full gain |
| E15 | Severe-event and interval metrics (idea 16) and forecast reconciliation (idea 7) | comparability with published tables; no training | < 1 h | (reporting) |
| later | Wind crossover horizon (idea 15, GPU 6-10 h); sub-hourly statistics (idea 9, needs 15-min data); real-box co-location | turn the wind negative into a horizon finding; real-world proof | | |

---

## 8. Caveats

- Simulated sensors decide how informative each sensor and each node is. Everything needs real-box confirmation.
- The teacher is a hindcast (ERA5 actual future weather). Live forecasts will be ~1 point worse (up to 3 at +6 h).
- E7 was fitted and scored inside the test period (first half / second half), because no 2024 forecasts were saved.
- Skills in Section 5 item 4 come from two runs (E6 and E8) on the same test windows; the derived −3.4 % / −4.3 %
  has no bootstrap CI yet (E11 gives it).
- E10 used 2 seeds, the rest 3.
- Citations outside `docs/papers/` are from memory and must be checked before use.
