# 0010 — GDC-CNN student: preprocessing, export graph, results (P7)

Status: accepted (2026-09-27)

## Decisions
- Architecture exactly per [S13.3]/[S13.4]: shared Dense encoder + learned positional embedding;
  3 × [LayerNorm → causal left pad → DepthwiseConv1D(k=3, d=1/2/4) → Dense(2D) → GLU → Dense(D) + residual];
  flatten → Dense(6). Receptive field 15 ≥ 12 patches. Keras 3.15 / TF 2.21.
- Preprocessing outside the graph (`gdc_cnn/preprocess.py`, C twin in P10): instance norm on instnorm
  channels with σ = sqrt(var + 1e-5); patches P = 2 → 12 × (2·C) hour-major; target de-norm by the window
  σ of ln_pm25 / ln_pm10 **with a floor of 0.1** (a flat window would explode the normalised target).
- Model B: no PM σ exists → de-norm scale 1; the 5 [S08.2] extras are appended as constant channels
  (17 + 5 = 22 channels → patch dim 44). (Was an open question.)
- **Export graph uses batch size 1** (`model.export_model`): with a dynamic batch the converter emits
  SHAPE/GATHER/PACK/**REDUCE_PROD**; REDUCE_PROD is not in TFLite Micro ([S13.4] step 0 caught it).
  Static shapes leave only TFLM ops: ADD, DEPTHWISE_CONV_2D, FULLY_CONNECTED, LOGISTIC, MEAN, MUL, NEG,
  PAD, RESHAPE, RSQRT, SQUARED_DIFFERENCE, STRIDED_SLICE. LayerNorm kept (supported). TFLM op list is
  vendored in `ml/src/aqi_ml/export/tflm_ops.txt` (source URL inside; re-check vs esp-tflite-micro).
- Training: Adam 1e-3, batch 128, Huber δ = 1 on normalised targets, early stopping patience 10, D chosen
  from {16, 32, 64} by a short search on 15 % of train; final fit on 50 % of train windows (memory).

## Results (synthetic data, not results per [S11.3])
| | D | params | FP32 .tflite | arena upper bound | host parity | test PM2.5 skill h1/h2/h3 | hold-out |
|---|---|---|---|---|---|---|---|
| CNN-A | 16 | 4 838 | 34.7 KB | 67 KB | 7e-7 | +0.075/+0.118/+0.171 | +0.118/+0.160/+0.216 |
| CNN-B | 64 | 46 854 | 202 KB | 369 KB (!) | 2.5e-6 | +0.254/+0.241/+0.225 | +0.266/+0.254/+0.239 |

- CNN-A ≈ RF-A (better on hold-out stations, slightly worse PM10 on test); CNN-B clearly beats RF-B.
- The arena "upper bound" sums all activation tensors; TFLM reuses buffers, so the real arena is far
  smaller. **Measure `arena_used_bytes()` on the ESP32 (P10)**; fall back to D = 32 for CNN-B if > 128 KB.
- INT8 static PTQ ablation (E7): size 30 KB / 88 KB, host accuracy unchanged (0.2913 vs 0.2910 log RMSE A).
  FP32 remains the deployment default (Na et al.: INT8 slower on ESP32-S3 due to casts) — confirm on device.
