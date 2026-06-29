# Training Stats Dashboard

Local Streamlit dashboard for per-item training statistics JSONL dumps shaped like:

```json
{"iter": 1, "consumed_train_samples": 4, "stat": "grad_raw_moments_by_param", "values": {"decoder.layers.0.mlp.linear_fc1.weight": {"count": 22020096, "sum_1": 0.44, "sum_2": 4.65, "sum_3": 0.000008, "sum_4": 0.000011}}}
```

The app can load either a single JSONL file or a directory tree containing metric
subdirectories, such as:

```text
training_stats/
  param_raw_moments_by_param/rank3.jsonl
  grad_raw_moments_by_param/rank3.jsonl
  activation_raw_moments_by_layer/rank0.jsonl
  dgrad_raw_moments_by_layer/rank0.jsonl
```

## Generate Metrics

Add the raw-moment logging flags to a training run to produce dashboard inputs:

```bash
--statistics-log-dir /path/to/output \
--log-param-raw-moments-by-param \
--log-grad-raw-moments-by-param \
--log-activation-raw-moments-by-layer \
--log-dgrad-raw-moments-by-layer
```

Training writes JSONL files under `/path/to/output/training_stats/`. If
`--statistics-log-dir` is not set, the files are written under `--tensorboard-dir`
when available, then under `--save` as a fallback.

Parameter and gradient raw moments use `--tensorboard-log-interval`. Activation
and dgrad raw moments use `--activation-log-interval` when set, otherwise they
also use `--tensorboard-log-interval`. For example:

```bash
--tensorboard-log-interval 10 \
--activation-log-interval 100
```

## Setup

```bash
python3 -m pip install -r tools/training_stats_dashboard/requirements.txt
```

## Run

From the workspace root:

```bash
python3 -m streamlit run tools/training_stats_dashboard/app.py
```

The app defaults to `training_stats`. The sidebar path field accepts one or more
paths, one per line. Each path can be a single JSONL file, a `training_stats`
directory, or a parent directory containing multiple descendant `training_stats`
directories. When multiple runs are discovered, select the runs to load from the
sidebar before filtering metrics.

Override the default path with:

```bash
TRAINING_STATS_PATH=/path/to/training_stats \
  python3 -m streamlit run tools/training_stats_dashboard/app.py
```

For a multi-run parent directory:

```bash
TRAINING_STATS_PATH=~/tensorboard/memtests/muon_r10 \
  python3 -m streamlit run tools/training_stats_dashboard/app.py
```

## Data Cache

On launch, the dashboard automatically converts each selected JSONL source to a
cached Parquet dataset under
`tools/training_stats_dashboard/.cache/metric_parquet/`. The cache key includes
the source path, JSONL file sizes, mtimes, and the cache format version, so
changing a run's JSONL files triggers a fresh conversion. Each source JSONL file
is converted to its own Parquet part, and those parts are written in parallel
across all selected runs.

The Streamlit UI then queries those Parquet files through DuckDB and asks for
only the grouped rows needed by the active view. The Parquet files store raw
moments (`count`, `sum_1` through `sum_4`) plus metadata; statistics such as L2,
RMS, mean, variance, skewness, and kurtosis are derived after DuckDB aggregates
the moments.

The sidebar's `Parquet cache` expander shows whether each selected run reused an
existing cache file or created a new one. Large line charts are sampled by
distinct x-axis values before plotting; adjust `Max points per series` from the
sidebar when you need more detail.

By default, conversion uses up to 8 worker processes, capped by the number of
uncached JSONL files and available CPUs. Set `TRAINING_STATS_CACHE_WORKERS` to
control conversion parallelism:

```bash
TRAINING_STATS_CACHE_WORKERS=8 TRAINING_STATS_PATH=/path/to/runs \
  python3 -m streamlit run tools/training_stats_dashboard/app.py
```

## Notes

- As a research tool, this is intended to be modified as necessary by users.
  This directory is standalone, so coding agents can easily add many
  visualizations.
- Layer inference is heuristic and architecture-neutral. It recognizes common
  repeated-module patterns such as `layers.12`, `blocks.12`, `h.12`, and
  `layer1.0`.
- Items without an inferred layer are treated as global/unlayered and can be
  included from the sidebar.
- Extra scalar row metadata, such as `gradient_stage`, becomes an automatic
  sidebar filter.
