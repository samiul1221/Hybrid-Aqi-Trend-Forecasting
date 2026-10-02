# 0019 — Teacher input must match the edge sensor domain

Status: accepted (2026-10-02, owner directive). Hard requirement for every v2 teacher (T0–T3).

- Teacher and student consume the same sensor-proxy features for local station information: the
  deterministic simulator + device pipeline + feature builder (no learned model in between). The teacher's
  node block is read from the same `hourly.parquet` as the student (audit guard G1).
- Clean CPCB/OpenAQ PM is a **target only** (y at t+1..t+6, and the base for µg/m³ metrics). It is never an
  input — not for the node, not for neighbours. Neighbours are the same virtual unit at the other stations.
- The teacher's privilege = other stations' sensor-proxy features, the wind graph, ADR transport, and the
  declared weather covariates (Open-Meteo past, ERA5 "oracle" future, ADR 0012).
- The v1 LightGBM teacher (ADR 0011) used clean neighbour CPCB PM; it is frozen as a baseline (ADR 0016)
  and labelled "clean-neighbour teacher" wherever its numbers appear.
- Evidence and guards: `docs/research_notes/teacher_input_audit.md`.
