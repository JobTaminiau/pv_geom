# Archived scripts

One-off spikes, benchmarks and diagnostics from the 0.1 development period
(May–August 2026). They are kept as a record of how design decisions were
reached, not as tools: most hard-code paths on the original development machine
and target the output schema of the time.

| Script | What it was for | Superseded by |
| --- | --- | --- |
| `spike.py`, `spike_roof.py`, `spike_validate.py` | LiDAR data-quality spikes behind the original thresholds | `pv-geom inspect-tile`, `tests/benchmark` |
| `_bench_1k.py`, `_coiled_bench_1k.py`, `_coiled_bench_10k.py`, `_coiled_smoke.py` | Timing and smoke runs on Phoenix | `scripts/build_benchmark.py`, `--dry-run` |
| `_review_bench.py`, `_aerial.py` | Six-panel QA figure over an ortho basemap | `pv-geom report` |
| `_vintage_impact.py` | Measured the roof-fit consensus floor and collar guard (2026-08-03) | `tests/benchmark` |
| `_build_coiled_env.py` | One-shot Coiled software-environment build | `pv_geom.coiled_env.ensure_software_env` |

`eda_outputs/` holds the figures and tables those scripts produced.
| `_label_sample.py`, `_label_analysis.py` | Ground-truth labelling of mounting types (2026-07-30) | archived with mounting classification |
