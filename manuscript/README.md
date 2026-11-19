# SplitMem manuscript (IEEE journal format)

| file | what it is |
|---|---|
| `main.tex`, `refs.bib` | the paper (IEEEtran, BibTeX); sections follow the IEEE reference papers in `docs/papers/` |
| `SplitMem.pdf` | compiled paper (8 pages) |
| `figures/fig1_architecture.pdf` | Fig. 1, vector render of `docs/aqi-v3 final.drawio.svg` (`render_fig1.py`); editable source `fig1_architecture.drawio` |
| `figures/fig_detailed_architecture.drawio` | detailed shapes diagram (not used in the paper; for an appendix) |
| `figures/fig_stack.pdf`, `fig_messages.pdf` | result plots made by `make_figures.py` from `data/6_experiments/` |
| `tables/*.tex` | table bodies made by `make_tables.py` from the raw result CSVs, never edited by hand |

Rebuild: `python make_tables.py && python make_figures.py`, then compile `main.tex` (pdflatex, bibtex, pdflatex x2).

## Where each number comes from
| number in the paper | result file |
|---|---|
| stack table, Fig. 2 | `data/6_experiments/split/split_results.csv`, `split_bootstrap.csv`, `memory/results.csv`, `memory/bootstrap.csv` |
| memory ablation and placebo | `memory/results.csv`, `memory/bootstrap.csv` |
| cloud graph model and wind | `wind/wind_results.csv`, `wind_bootstrap.csv` |
| stale / missing broadcast | `buffer/results.csv` |
| messages, Fig. 3 | `fusion/B_event_triggered.csv`, `trigger/trigger_results.csv` |
| training size, width, losses | `moredata/bootstrap.csv`, `split/split_bootstrap.csv` |
| text only (no CSV read by a script): v1 copy result (20.8 / 14.0 / 15.0), sensor power subsets, clustering, CPCB category 43 % / 71 % | `docs/EXPERIMENTS.md`, `docs/ABLATION_FINDINGS.md`, `data/6_experiments/probe/`, `netctx/` |
| -3.4 % / -4.3 % RMSE (full stack vs box alone) | derived from mean skills (`tables/derived.txt`); no direct bootstrap yet |

## Before submission
- All sensors are simulated: say so in every claim (the paper does).
- Check `refs.bib`: entries 1-14 come from the PDFs in `docs/papers/` (end pages inferred from page counts, verify); the last 9 (Hinton, LightGBM, GAT, Transformer, GLU, ERA5, Huber, consensus, AdamW) were written from memory.
- Fill in department and e-mail addresses, and the AI-assistance disclosure if the venue requires it.
- Run the final stack on 1 M windows (E11) and the two-model fallback (E12); add a direct bootstrap for mem vs box alone.
- Phrase overlap with the reference PDFs (8-gram) is 0.00 % (`academic-humanize/scripts/overlap.py`); also run the venue's own similarity checker.
