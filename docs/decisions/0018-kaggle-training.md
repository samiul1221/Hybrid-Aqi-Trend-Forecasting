# 0018 — Teacher and KD training on Kaggle GPU (CR v2 R09)

Status: accepted (2026-10-02)

- The laptop has no GPU; teacher (T0–T3, OOF) and student KD runs go to private Kaggle notebooks.
- No Kaggle MCP server is configured; the **Kaggle CLI** (2.2.4, `uv tool install kaggle`) is used with
  `KAGGLE_API_TOKEN` from `.env` (never committed; placeholder in `.env.example`).
- Code reaches Kaggle as the `aqi-ml-src` dataset (zip of `aqi_ml` + `aqi_shared` + configs, tagged with
  the git sha); notebooks only call `aqi_ml` functions. Outputs are pulled into `data/kaggle_runs/<run_id>/`.
- Data building, simulator, RF baselines, tests, export and parity stay on the laptop CPU.
