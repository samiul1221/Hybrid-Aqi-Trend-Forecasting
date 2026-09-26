# 0004 — How the feature schema and wire contracts are written down ([S08], [S16])

Status: accepted (2026-09-26). No behaviour change vs the spec; these are additions the spec needs
to be machine-readable.

## Feature schema (`shared/schema/feature_schema.yaml`)
- Each channel keeps the spec's `tf` text and gains a machine `op` (log1p, clip, offset, scale,
  log_ratio, base_diff_scale, cyc_sin, cyc_cos, measured_mask, identity). Python builder and C
  codegen read `op`, never parse `tf` strings.
- Schema hash = sha256 of canonical JSON (sorted keys, compact, UTF-8) of the parsed YAML, so it does
  not change with comments, key order or line endings.
- RF features [S08.3] are written as groups and expanded in a fixed order to exactly 50 names
  (`<src>_lag|rmean|rstd|diff|rainh<k>`). Definitions pinned for C parity: roll windows include hour t;
  roll_std is population std (ddof = 0); `rain_hours_last` counts `rain_flag` hours.
- `dust`, `wind_speed`, `rain_flag` are RF-only sources (`rf_aux_sources`), not sequence channels.
- Model B RF = non-PM groups (27) + the 5 [S08.2] extras = 32 features. Model B CNN keeps the 17
  `model_b: true` channels. **Open (P7):** the spec does not say how the 5 scalar extras enter the CNN.

## Contracts (`shared/contracts/*.schema.json`, draft 2020-12)
- Edge → cloud (ingest, events, registry): `additionalProperties: false` to catch drift; value ranges
  not enforced (cloud nulls and flags out-of-range values, [C06]).
- Cloud → edge (forecast, manifest, MQTT): open, because the edge must ignore unknown fields ([C14.1]).
- `ts` of records/forecasts must be an IST bucket (`HH:30:00Z`) — I02 enforced at the wire.
- Nullable where the spec implies "not ready": `proc.aqi_pm`, `edge_fc` (e.g. warm-up), `edge_fc.trend`,
  `edge_fc.aqi[i]`, `gas_subindex`, `neighbour_pm_estimate`.
- Added optional `sys.last_command_id` ([S16.7] ack) and `sensors.rtc` in the registry.
- Registry sensor fields are enums/consts: the sensor pool is fixed.
- MQTT payloads live in `mqtt.schema.json#/$defs/{manifest_notify,command,status}`; the MQTT forecast
  payload is `forecast.schema.json` (byte-identical to GET).
