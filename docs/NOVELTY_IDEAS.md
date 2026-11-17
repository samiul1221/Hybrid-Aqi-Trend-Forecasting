# Novelty ideas: what to add, why, and which experiments to run

Written 16 Nov 2026. Read `docs/ARCH_V2.md` first: this file only lists what could be **added** to it.
Everything here is on simulated sensor boxes, like the rest of the project.

> **Honest summary.** None of these ideas is a new algorithm. Each is a known technique from another field,
> applied to the split cloud-edge forecaster and measured. Expect 0.3-1 % RMSE per idea. The main novelty of
> the project stays the measured split system plus the honest negative results. The ideas below make that
> story wider (accuracy, robustness, bandwidth), not different.

---

## 1. Two things to know before picking

1. **The box sees almost nothing the cloud does not.** It uploads all 23 features every hour. With hourly data
   its only private assets are: its long memory, its offline hours, and its own forecast. The one truly
   node-private signal is **sub-hourly** data (idea 9), and the present data cannot test it.
2. **The student may simply be under-trained.** On the same windows the student (S0) scores 14.0 and the
   teacher 20.8, but the graph explains only about 1.9 of that gap and future weather about 1. The student
   trains on 200 000 of 2.6 M windows. This is an inference across different tables, not a verified fact;
   experiment E10 checks it. If true, it matters more than any idea below.

---

## 2. Shared rules for every experiment

| | |
|---|---|
| Baseline **B0** | split-res + consensus filter (the combined model of `ARCH_V2.md` step 13.1) |
| Metric | RMSE % change vs the baseline, 95 % week-block paired bootstrap CI; skill vs persistence |
| Setup | 3 seeds, seen (27) and unseen (5) stations, test 2025 - Sep 2026 |
| Tuning | on 2024 only (one exception: E7, see there) |
| Hard rules | true CPCB PM is a target only; no future data; teacher outputs out-of-fold; fixed sensor pool |
| Kill threshold | written down **before** the run and not changed afterwards |

A failed idea is reported as a negative result, not hidden.

---

## 3. Status at a glance

| # | idea | aim | experiment | status |
|---|---|---|---|---|
| 1 | Node memory | accuracy | E8 `exp_memory.py` | **done: passes** (−1.09 % seen, −0.89 % unseen; now in `ARCH_V3.md`) |
| 2 | Buffered broadcast | robustness | E9 `exp_buffer.py` | **done: fails** (see `EXPERIMENTS.md` E9) |
| 3 | Learned send rule | bandwidth | E7 `exp_trigger.py` | **done: fails its threshold** |
| 4 | Reference-free personalisation | accuracy at new sites | not written | idea |
| 5 | Network error feedback | robustness in episodes | not written | idea |
| 6 | Breadth and dispersion bytes | accuracy | not written | idea |
| 7 | Forecast reconciliation | accuracy | not written | idea (cheapest) |
| 8 | Kalman / kriging consensus | accuracy | not written | idea |
| 9 | Sub-hourly node statistics | accuracy | needs new data | idea (best "node is essential") |
| 10 | Serverless split (gossip) | robustness | not written | idea |
| 11 | Network level calibration | shown AQI | needs real boxes | idea |
| 12 | Node value | network design | fold into node-count test | idea |
| 13 | Analog days | accuracy in episodes | not written | idea (weak) |
| 14 | Fault-robustness, network as safety net | robustness | not written | **recommended next** |
| 15 | Wind crossover horizon | explains a negative result | not written | recommended |
| 16 | Severe-event and interval metrics | reporting | no training | recommended |
| 17 | Longer window, side task | accuracy | not written | after E8 |
| 18 | On-device measurements | evidence | hardware | needed for the paper |
| - | More training data (a check, not a novelty) | diagnosis | E10 `exp_moredata.py` | **done**: 1 M windows −1.06 %, width 64 = 0 |

Full results of E7-E10 and the next runs (E11-E15): `docs/EXPERIMENTS.md`.

Ideas 1-13 come from other fields. Ideas 14-18 come from the papers in `docs/papers/` (Section 6).

---

## 4. Ideas already under test

### Idea 1. Node memory of its "usual self" (E8)

- **In simple words:** the box keeps a table of its usual offset from the region for every clock hour. It tells
  the student "how unusual am I right now" and "how my offset usually moves in the next 1-3 h".
- **Borrowed from:** finance (error-correction models, Engle and Granger 1987), weather climatology and anomaly
  forecasting. As feature engineering it is probably not novel.
- **Why it could work:** the consensus filter keeps one average offset (24 h EWMA), blind to the hour of day.
  Local traffic and burning peaks repeat daily and differ by site. That is exactly the local correction the box
  must predict.
- **Formulas:**
  - offset `o_i(t) = own ln PM_i(t) − R(t)` (R arrives in the broadcast)
  - memory `M_i(c)` = EWMA over past days (half-life 10 days) of `o_i` at clock hour `c`
  - surprise = `o_i(t) − M_i(clock(t))` (2 numbers)
  - profile = `M_i(clock(t+h)) − M_i(clock(t))`, h = 1..3 (6 numbers)
  - the table entry used for hour t+h was last updated at t+h−24, so no future data is used
- **Arms:** split-res; b0; mem (both); mem-surprise; mem-profile; mem-placebo (the table of another node).
- **Kill threshold:** mem vs b0 weaker than −0.3 %, or CI contains 0, or mem does not beat the placebo with a CI
  that excludes 0.
- **Cost:** about 3.5 h CPU, 0 bytes on the link, about 300 extra parameters, 48 numbers in flash.
- **Risk:** the 24 h input window already shows yesterday's cycle, so the CNN may already know this.

### Idea 2. Buffered broadcast: "send the plan, not the next step" (E9)

- **In simple words:** the cloud broadcasts the regional path for +1..+9 h. When the link drops, the box shifts
  along the stored path.
- **Borrowed from:** control (packetized predictive control, Quevedo et al., to verify) and video buffering.
- **Why it could work:** today a 2-3 h old broadcast is worth nothing, because a +1..+3 h forecast made 3 hours
  ago covers no future hour. A +9 h path still covers the next 3 hours at age 6.
- **Formula:** change from now to now+h = `path[a + h] − path[a]`, where the path was made `a` hours ago (the
  same "path re-basing" that worked in the fusion experiment).
- **Arms:** ctx (no broadcast); short (+1..+3 h); long (+1..+9 h). All trained with random ages 0-6 h and 15 %
  missing. Scored at ages 0, 1, 2, 3, 4, 6 h and missing.
- **Kill threshold:** at age 3 h, long vs ctx keeps less than 40 % of long's fresh gain, or the CI of long vs
  short at age 3 h contains 0. Also long at age 0 must not be worse than short by more than 0.2 %.
- **Cost:** about 2 h CPU, 18 bytes per hour raw (about 8 if compressed), about 200 extra parameters.
- **Risk:** about half of the fresh gain may come from the regional level now, which cannot be buffered.

### Idea 3. Learned send rule (E7), result: FAILS

- **In simple words:** a model predicts when a personal cloud forecast will really help, instead of sending
  whenever cloud and box disagree.
- **Borrowed from:** economics (value of information), networking (age of information, Kaul et al. 2012).
- **Why it looked promising:** at 20 % of hours the disagreement rule gives 20.7 and an oracle 27.2.
- **Result (second half of the test period; box alone 15.87, cloud always 23.53):**

| % of hours sent | random | disagreement | learned | oracle | learned vs disagreement, RMSE % (95 % CI) |
|---|---|---|---|---|---|
| 5 | 16.22 | 18.38 | 18.44 | 21.98 | −0.07 (−0.23 to +0.09) |
| 10 | 16.61 | 19.59 | 19.63 | 24.55 | −0.05 (−0.23 to +0.14) |
| **20** | 17.31 | 20.97 | **21.12** | 27.48 | **−0.20 (−0.34 to −0.05)** |
| 30 | 18.11 | 21.90 | 22.06 | 29.03 | −0.21 (−0.35 to −0.05) |
| 50 | 19.58 | 22.87 | 22.98 | 30.23 | −0.14 (−0.25 to −0.03) |

| token bucket, messages per day | disagreement | learned | RMSE % (95 % CI) |
|---|---|---|---|
| 1 | 16.78 | 16.87 | −0.11 (−0.29 to +0.08) |
| 2 | 17.73 | 17.87 | −0.16 (−0.35 to +0.02) |
| 4 | 18.61 | 18.86 | −0.32 (−0.51 to −0.13) |
| 6 | 19.30 | 19.56 | −0.32 (−0.55 to −0.10) |
| 12 | 21.25 | 21.26 | −0.01 (−0.18 to +0.15) |

- **Verdict:** +0.16 skill at 20 % of hours against the +0.5 required. The effect is real (CI excludes 0) but
  tiny. The absolute cloud-box gap carries about 42 % of the model's importance and nothing else reaches 6 %.
- **What to write in the paper:** the simple disagreement rule is close to the best the cloud can do with what
  it knows; the oracle gap is mostly luck.
- **Deviation:** fitted on the first half of the test period and scored on the second half, because saved
  teacher and student forecasts exist only for the test period. Results: `data/6_experiments/trigger/`.

### Check E10. Does the student need more training data?

- **Arms:** 200k windows, width 32 (reference); 1 M windows, width 32; 1 M windows, width 64. Same validation
  and test windows, 2 seeds, Kaggle T4 GPU.
- **No kill threshold:** it is a diagnosis. If 1 M windows gives a large gain, retrain every arm of every
  experiment on the bigger set before writing the paper.

---

## 5. Ideas not yet tested

### Idea 4. Reference-free personalisation
- **Simple:** the cloud fine-tunes each box's last layer using the teacher's forecast as the label, so a new
  site needs no CPCB station.
- **From:** personalised federated learning (FedPer, to verify), test-time adaptation. Moderately novel for
  low-cost air-quality nodes (to verify).
- **Why:** one global 15k model serves every site, and teacher labels exist for every node for free.
- **Experiment:** B0; per-node head from 4, 12 and 52 weeks of 2024 teacher labels; a 4-byte "node passport"
  input (noise level, humidity slope, mean offset, daily amplitude); true-label head as an upper bound (seen
  stations only). Kill: weaker than −0.5 % on unseen stations.
- **Cost:** about 3 h, about 200 bytes per week, 0 extra parameters.
- **Risk:** global distillation gave only +0.9, and unseen sites already score higher than seen ones.

### Idea 5. Network error feedback
- **Simple:** feed the model its own recent forecast errors, scored against the network-denoised level (not
  against true PM).
- **From:** control (integral action), ARIMA moving-average terms. Not novel.
- **Why:** errors persist for days in episodes such as stubble smoke and Diwali.
- **Experiment:** B0; + regional error lags; + per-node error lags. Report an episode-only subset as well.
  Kill: weaker than −0.3 %.
- **Cost:** about 3 h, 2 bytes per hour, about 100 parameters.
- **Risk:** the same sensor noise that broke the online blend weights.

### Idea 6. Breadth and dispersion bytes
- **Simple:** broadcast how spread out the nodes are and what fraction is rising.
- **From:** finance (market breadth), biology (quorum sensing). Probably known.
- **Why:** high spread means a local-source regime (trust yourself); low spread means trust the region.
- **Experiment:** B0 + 2 bytes. Kill: weaker than −0.3 %.
- **Cost:** about 2 h, 2 bytes per hour, about 100 parameters.
- **Risk:** hour, season and fan wind already tell the regime.

### Idea 7. Forecast reconciliation (cheapest to test)
- **Simple:** the cloud's copies of all students must average to the regional forecast; broadcast the mismatch.
- **From:** econometrics (MinT, Wickramasuriya, Athanasopoulos and Hyndman 2019). Not novel as a method.
- **Why:** it removes the error that all students share.
- **Experiment:** post-hoc on saved forecasts, no training. Kill: weaker than −0.2 %.
- **Cost:** under 1 h, 6 bytes per hour, 0 parameters.
- **Risk:** split-res is already a top-down reconciliation.

### Idea 8. Kalman or kriging "analysis" instead of the EWMA consensus
- **Simple:** estimate each node's level with proper noise weights and its neighbours.
- **From:** data assimilation (optimal interpolation). Not novel.
- **Why:** the crude filter already gave −0.9 %.
- **Experiment:** EWMA vs Kalman vs kriging. Kill: weaker than −0.3 % against the EWMA version.
- **Cost:** about 2 h, same bytes, 0 parameters.
- **Risk:** one airshed, so neighbours add little beyond the mean.

### Idea 9. Sub-hourly node statistics (strongest "only the node can see this")
- **Simple:** within-hour PM slope, spikiness and fan gustiness as extra features.
- **From:** nowcasting; turbulence intensity in micrometeorology. Novelty unclear (to verify).
- **Why:** the end-of-hour trend leads the next hour; spikes mean a nearby source; gustiness measures mixing.
  The cloud never receives any of this.
- **Experiment:** needs 15-minute CPCB data (the portal offers it, to verify) or real boxes. Kill: weaker than
  −1 % at +1 h.
- **Cost:** new download and simulator changes, 0 bytes, about 200 parameters.
- **Risk:** cannot be tested today. Simulated within-hour noise would only test the simulator.

### Idea 10. Serverless split
- **Simple:** the box forecasts the regional part itself from the gossip average, with no cloud.
- **From:** swarm and gossip algorithms. Known.
- **Why:** it keeps the split design alive through multi-day cloud outages.
- **Experiment:** cloud split vs gossip split vs ctx. Kill: under 50 % of the cloud gain.
- **Cost:** about 4 h, 0 cloud bytes, about 1 000 parameters.
- **Risk:** the gossip test assumed 10 km links, which ESP32 Wi-Fi cannot do.

### Idea 11. Network level calibration for new nodes
- **Simple:** learn the true level from node features plus regional context at the 27 stations and apply it to
  new ones.
- **From:** low-cost sensor calibration, heavily published. Not novel.
- **Why:** the biggest error on the screen is the level, not the change: AQI colour is right 43 % of the time
  against 71 % with a true starting level.
- **Experiment:** category accuracy on the unseen stations.
- **Cost:** about 2 h, 2 bytes per hour.
- **Risk:** in simulation this only inverts the simulator. It needs real boxes.

### Idea 12. Node value
- **Simple:** leave one node out at a time to find which nodes the broadcast needs.
- **From:** Data Shapley (Ghorbani and Zou 2019); zoning by clustering (Illueca Fernández et al. 2024 recommend
  a new measurement point this way).
- **Why:** a placement rule, and robustness to losing a node.
- **Experiment:** fold into the planned node-count test (5, 10, 15, 27 nodes).
- **Risk:** a design rule, not an accuracy gain.

### Idea 13. Analog days
- **Simple:** the cloud finds past hours that look like now and broadcasts what happened next.
- **From:** the analog ensemble (Delle Monache et al. 2013).
- **Why:** might help in rare episodes.
- **Experiment:** kill weaker than −0.3 %. About 2 h.
- **Risk:** LightGBM already does roughly this.

---

## 6. Ideas from the papers in `docs/papers/`

Only the first 1-3 pages of each paper were read (abstract, introduction, contributions). Check the full text
before claiming that a paper lacks something.

### Idea 14. Fault-robustness test: the network as a safety net (recommended next)
- **From:** REDNet (Fu et al. 2026) makes robustness to corrupted inputs its headline; An et al. 2025 estimate
  PM at places with no sensor from nearby sensors ("virtual sensor").
- **Simple:** break the box's own PM sensor at test time and see whether the regional broadcast keeps the
  forecast usable.
- **Why:** the region is the average of many boxes, so it should not care about one faulty box. There is no
  robustness-to-fault table in the project yet, and the "Model B" fallback for a broken PM sensor is missing.
- **Experiment:**
  - faults: stuck value, spikes, +30 % drift, dead sensor, each for 6 h, 24 h and 7 days
  - arms: ctx (box alone); split-res; split-res trained with fault augmentation; for a dead sensor, the cloud
    sends a virtual-sensor estimate built from the other nodes
  - metric: skill under each fault, and the loss relative to the fault-free score
  - kill: split-res loses at least as much skill as ctx under faults (the broadcast gives no protection)
- **Cost:** 1 CPU run of about 3 h, same bytes, 0 extra parameters.
- **Risk:** the box model may lean on its own PM so much that the broadcast cannot rescue it without fault
  training.

### Idea 15. Wind crossover horizon
- **From:** Zhao et al. 2025 report their wind graph helps most at 24, 30 and 36 h; TopoFlow works at 12-96 h;
  TransNet up to 72 h; Wu et al. 2026 and Wang et al. 2024 use wind-direction graphs for ozone.
- **Simple:** repeat the placebo wind test at 6, 12 and 24 h.
- **Why:** the present test covers 1-6 h only, where wind adds nothing. Finding the horizon where it starts to
  help reconciles the negative result with these papers: "in a one-airshed city, wind direction pays only
  beyond X hours".
- **Experiment:** T1 vs T2 (ERA5 wind) vs T2 with wind rotated 180°, at each horizon, 3 seeds. Kill: no horizon
  up to 24 h where wind beats the placebo with a CI excluding 0 (then the negative result simply extends).
- **Cost:** 1 GPU run, about 6-10 h.
- **Risk:** ERA5 has only 4 distinct wind series for the 32 stations, so wind may never help in this data.
- **Note:** no placebo control was seen in the pages read of those papers. If the full texts confirm that, the
  rotated-wind control is a selling point.

### Idea 16. Severe-event and interval metrics (no training)
- **From:** TransNet reports recall, precision and false alarms at a "Very Unhealthy" threshold; Wu et al. report
  interval coverage and width.
- **Do:** recall, precision and false alarms for crossing into CPCB "Poor" or worse at +1..+3 h; coverage and
  width of the teacher's 10-90 % band after calibrating it on 2024 (it covers 75 % today, not 80 %).
- **Why:** it makes the tables comparable to published ones at no compute cost.

### Idea 17. Two cheap student arms
- **Longer window (from TSMNet, Tang and Guan 2025, which folds a series by its period):** 48 or 72 h of input
  instead of 24 h. It overlaps with idea 1, so wait for E8 first.
- **Side task (from Wang et al. 2024, where predicting weather alongside helps):** the student also predicts
  its own next-hour humidity, temperature and wind. The labels come from the box, so no CPCB station is needed.
- **Kill:** weaker than −0.3 % each. About 3 h CPU for both.

### Idea 18. On-device measurements
- **From:** Na et al. 2026 (the student's source) report 256 ms inference on an ESP32-S3.
- **Do:** latency, flash, RAM and energy of the student with the broadcast inputs on the WROOM-32.
- **Why:** reviewers of edge papers expect it.

### How the papers position the work
- **Closest:** Na et al. and REDNet both distil a big teacher into an edge model that then runs alone. In the
  pages read, neither sends cloud context to the device at inference. That is the split design's gap.
- **Bigger students are normal:** REDNet's student has 174k parameters against 15k here. This supports E10.
- **Simulated data is defensible:** Morales-García et al. 2025 use synthetic data to train graph models when real
  data is short, and show traffic data helps, which matches the MQ-7 finding.
- **Scale explains the physics result:** TransNet finds transport operators essential across 170 stations in a
  whole country; here 32 stations sit in one airshed at 1-3 h.
- **Low priority:** Shmuel et al. 2026 distil a boosting model into a readable formula. It could be done for the
  regional forecaster, but like the fuzzy gate it buys interpretability, not accuracy.

---

## 7. Suggested order

| step | what | why now |
|---|---|---|
| 1 | Read E8, E9, E10 when they finish | already paid for |
| 2 | If E10 shows a large gain, retrain the baseline on the bigger set | every later number depends on it |
| 3 | Idea 14 (fault robustness) | new robustness claim, one CPU run |
| 4 | Idea 7 (reconciliation) and idea 16 (metrics) | no training needed |
| 5 | Idea 15 (wind crossover) | turns a negative result into a finding |
| 6 | Idea 4 (personalisation) or idea 6 (breadth bytes) | only if steps 1-5 leave time |
| 7 | Ideas 9, 11, 18 | need 15-minute data or real boxes |

---

## 8. Where things are

| what | path |
|---|---|
| experiment scripts | `experiments/exp_trigger.py`, `exp_memory.py`, `exp_buffer.py`, `exp_moredata.py` |
| push / status / download | `uv run python experiments/kaggle_exp.py push|status|get <name>` |
| results | `data/6_experiments/<name>/` (`results.csv`, `bootstrap.csv`, the Kaggle log) |
| small change to an older script | `experiments/exp_split.py`: the regional forecaster's horizons are now the list `OUTPUTS` (behaviour unchanged) |

**Citations:** every paper named outside `docs/papers/` is from memory and must be verified at the publisher or
arXiv page before use.
