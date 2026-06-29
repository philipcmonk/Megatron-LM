# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# isort: off
import training_stats as _training_stats

_training_stats = importlib.reload(_training_stats)
import metric_store as _metric_store

_metric_store = importlib.reload(_metric_store)
# isort: on

discover_jsonl_paths = _training_stats.discover_jsonl_paths
natural_sort_key = _training_stats.natural_sort_key
ensure_parquet_caches = _metric_store.ensure_parquet_caches
MetricStore = _metric_store.MetricStore


DEFAULT_SOURCE_PATH = os.environ.get("TRAINING_STATS_PATH", "training_stats")
STATISTIC_COLUMNS = {
    "L2 norm": "l2_norm",
    "RMS": "rms",
    "Mean": "mean",
    "Variance": "variance",
    "Std dev": "std",
    "Skewness": "skewness",
    "Kurtosis": "kurtosis",
    "Excess kurtosis": "excess_kurtosis",
}


@dataclass(frozen=True)
class RunSource:
    label: str
    path: str
    jsonl_count: int


st.set_page_config(page_title="Training Stats Dashboard", layout="wide")


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def infer_run_label(source_path: Path, common_root: Path | None = None) -> str:
    """Return a compact run label for a JSONL file or training_stats directory."""

    source_path = source_path.resolve()

    if source_path.is_file():
        label_path = source_path.with_suffix("")
    elif source_path.name == "training_stats":
        label_path = source_path.parent
        if label_path.name in {"tensorboard", "tb"}:
            label_path = label_path.parent
    else:
        label_path = source_path

    if common_root is not None:
        try:
            relative = label_path.relative_to(common_root.resolve())
            label = str(relative)
            if label and label != ".":
                return label
        except ValueError:
            pass

    return label_path.name or str(label_path)


def is_training_stats_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name == "training_stats" and discover_jsonl_paths(path):
        return True
    metric_dirs = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.endswith(("_by_param", "_by_layer"))
    ]
    return bool(metric_dirs and discover_jsonl_paths(path))


def discover_run_sources_for_path(path: Path) -> list[RunSource]:
    if path.is_file():
        jsonl_count = 1 if path.suffix == ".jsonl" else len(discover_jsonl_paths(path))
        return [RunSource(infer_run_label(path), str(path), jsonl_count)] if jsonl_count else []

    if not path.is_dir():
        return []

    if is_training_stats_dir(path):
        jsonl_paths = discover_jsonl_paths(path)
        return [RunSource(infer_run_label(path), str(path), len(jsonl_paths))]

    training_stats_dirs = sorted(
        {
            candidate
            for candidate in path.rglob("training_stats")
            if candidate.is_dir() and discover_jsonl_paths(candidate)
        },
        key=lambda item: natural_sort_key(str(item)),
    )
    if training_stats_dirs:
        return [
            RunSource(
                infer_run_label(training_stats_dir, path),
                str(training_stats_dir),
                len(discover_jsonl_paths(training_stats_dir)),
            )
            for training_stats_dir in training_stats_dirs
        ]

    jsonl_paths = discover_jsonl_paths(path)
    if jsonl_paths:
        return [RunSource(infer_run_label(path), str(path), len(jsonl_paths))]

    return []


def parse_source_text(path_text: str) -> list[Path]:
    paths = []
    for line in path_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(resolve_path(line))
    return paths


def discover_run_sources(path_text: str) -> list[RunSource]:
    paths = parse_source_text(path_text)
    sources: list[RunSource] = []
    seen_paths: set[str] = set()
    for path in paths:
        for source in discover_run_sources_for_path(path):
            if source.path in seen_paths:
                continue
            sources.append(source)
            seen_paths.add(source.path)
    return uniquify_run_labels(sources)


def uniquify_run_labels(sources: list[RunSource]) -> list[RunSource]:
    label_counts: dict[str, int] = {}
    for source in sources:
        label_counts[source.label] = label_counts.get(source.label, 0) + 1

    if all(count == 1 for count in label_counts.values()):
        return sources

    next_index: dict[str, int] = {}
    unique_sources = []
    for source in sources:
        if label_counts[source.label] == 1:
            unique_sources.append(source)
            continue
        index = next_index.get(source.label, 1)
        next_index[source.label] = index + 1
        unique_sources.append(
            RunSource(f"{source.label} #{index}", source.path, source.jsonl_count)
        )
    return unique_sources


def sorted_unique(series: pd.Series) -> list:
    return sorted(series.dropna().unique().tolist(), key=natural_sort_key)


def selected_compare_columns_for_selection(
    run_labels: list[str], metric_names: list[str]
) -> list[str]:
    columns = []
    if len(run_labels) > 1:
        columns.append("run")
    if len(metric_names) > 1:
        columns.append("metric")
    return columns


def numeric_widget_value(value: object) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def numeric_slider_bounds(min_value: object, max_value: object) -> tuple[int | float, int | float]:
    min_numeric = numeric_widget_value(min_value)
    max_numeric = numeric_widget_value(max_value)
    if isinstance(min_numeric, int) and isinstance(max_numeric, int):
        return min_numeric, max_numeric
    return float(min_numeric), float(max_numeric)


def append_filter(
    filters: list[tuple[str, str, object]], column: str, op: str, value: object
) -> None:
    filters.append((column, op, value))


def with_filter(
    filters: list[tuple[str, str, object]] | tuple[tuple[str, str, object], ...],
    column: str,
    op: str,
    value: object,
) -> list[tuple[str, str, object]]:
    return [*filters, (column, op, value)]


def label_for_row(row: pd.Series, columns: list[str], fallback: str = "total") -> str:
    labels = []
    for column in columns:
        value = row.get(column)
        if pd.isna(value):
            continue
        labels.append(str(value))
    return " | ".join(labels) if labels else fallback


def add_series_label(
    df: pd.DataFrame, columns: list[str], fallback: str = "total", target_col: str = "series"
) -> pd.DataFrame:
    if df.empty:
        if target_col not in df.columns:
            df[target_col] = pd.Series(dtype=object)
        return df

    df = df.copy()
    df[target_col] = df.apply(lambda row: label_for_row(row, columns, fallback), axis=1)
    return df


def statistic_formula(statistic_name: str) -> str:
    formulas = {
        "L2 norm": "sqrt(sum_2)",
        "RMS": "sqrt(sum_2 / count)",
        "Mean": "sum_1 / count",
        "Variance": "sum_2 / count - mean^2",
        "Std dev": "sqrt(variance)",
        "Skewness": "E[(x - mean)^3] / variance^(3/2)",
        "Kurtosis": "E[(x - mean)^4] / variance^2",
        "Excess kurtosis": "kurtosis - 3",
    }
    return formulas.get(statistic_name, statistic_name)


def sort_by_series_and_x(df: pd.DataFrame, series_col: str, x_axis: str) -> pd.DataFrame:
    if df.empty:
        return df

    series_order = {value: index for index, value in enumerate(sorted_unique(df[series_col]))}
    df = df.copy()
    df["_series_order"] = df[series_col].map(series_order)
    df = df.sort_values(["_series_order", x_axis])
    return df.drop(columns=["_series_order"])


def add_plot_value(df: pd.DataFrame, series_col: str, mode: str, x_axis: str) -> pd.DataFrame:
    df = sort_by_series_and_x(df, series_col, x_axis)

    if mode == "Raw value":
        df["plot_value"] = df["metric_value"]
        return df

    first = df.groupby(series_col, dropna=False)["metric_value"].transform("first")
    if mode == "Relative to first step":
        df["plot_value"] = df["metric_value"] / first.mask(first == 0)
    elif mode == "Delta from first step":
        df["plot_value"] = df["metric_value"] - first
    elif mode == "Percent change from first step":
        df["plot_value"] = (df["metric_value"] / first.mask(first == 0) - 1.0) * 100.0
    else:
        df["plot_value"] = df["metric_value"]

    return df


def smooth_series(df: pd.DataFrame, series_col: str, x_axis: str, window: int) -> pd.DataFrame:
    if window <= 1 or df.empty:
        return df

    df = sort_by_series_and_x(df, series_col, x_axis)
    df["plot_value"] = df.groupby(series_col, dropna=False)["plot_value"].transform(
        lambda values: values.rolling(window, min_periods=1).mean()
    )
    return df


def line_chart(
    df: pd.DataFrame,
    x_axis: str,
    series_col: str,
    value_mode: str,
    title: str,
    log_y: bool,
    smoothing_window: int,
) -> None:
    if df.empty:
        st.info("No data for this selection.")
        return

    plot_df = add_plot_value(df, series_col, value_mode, x_axis)
    plot_df = smooth_series(plot_df, series_col, x_axis, smoothing_window)
    hover_data = {
        "metric_value": ":.6g",
        "item_count": True,
        "sample_count": True,
        "plot_value": ":.6g",
    }
    for column in ["run", "metric", "layer_label", "item_type", "item_family", "item_kind"]:
        if column in plot_df.columns and column != series_col:
            hover_data[column] = True

    fig = px.line(
        plot_df,
        x=x_axis,
        y="plot_value",
        color=series_col,
        title=title,
        hover_data=hover_data,
        log_y=log_y and value_mode in {"Raw value", "Relative to first step"},
    )
    fig.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend_title_text=series_col.replace("_", " "),
    )
    st.plotly_chart(fig, width="stretch")


def heatmap(
    df: pd.DataFrame, x_axis: str, y_axis: str, value_mode: str, title: str, color_label: str
) -> None:
    if df.empty:
        st.info("No data for this selection.")
        return

    plot_df = add_plot_value(df, y_axis, value_mode, x_axis)
    y_order = sorted_unique(plot_df[y_axis])
    pivot = plot_df.pivot_table(index=y_axis, columns=x_axis, values="plot_value", aggfunc="first")
    pivot = pivot.reindex(y_order)

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"x": x_axis, "y": y_axis.replace("_", " "), "color": color_label},
        title=title,
    )
    fig.update_layout(
        height=max(420, 28 * len(pivot.index)), margin={"l": 20, "r": 20, "t": 50, "b": 20}
    )
    st.plotly_chart(fig, width="stretch")


def fixed_step_label(x_axis: str, fixed_x: object) -> str:
    return f"{x_axis}={fixed_x}"


def layer_cross_section_chart(
    df: pd.DataFrame, color_col: str | None, title: str, log_y: bool, chart_mode: str
) -> None:
    if df.empty:
        st.info("No layer data for this selection.")
        return

    plot_df = df.sort_values(["layer_index"] + ([color_col] if color_col else [])).copy()
    plot_df["layer"] = plot_df["layer_index"].astype(int)

    chart_kwargs = {
        "x": "layer",
        "y": "metric_value",
        "title": title,
        "hover_data": {
            "layer_label": True,
            "metric_value": ":.6g",
            "item_count": True,
            "sample_count": True,
        },
        "log_y": log_y,
    }
    if color_col:
        chart_kwargs["color"] = color_col

    if chart_mode == "Bars":
        fig = px.bar(plot_df, **chart_kwargs, barmode="group")
    else:
        fig = px.line(plot_df, **chart_kwargs, markers=True)

    fig.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        xaxis_title="Layer",
        yaxis_title="Metric value",
    )
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")


RESERVED_FILTER_COLUMNS = {
    "run",
    "run_path",
    "source_file",
    "source_path",
    "source_line",
    "iter",
    "consumed_train_samples",
    "metric",
    "metric_scope",
    "stat",
    "loss_scale",
    "norm_type",
    "norm_type_label",
    "param",
    "item",
    "value",
    "value_sq",
    "count",
    "sum_1",
    "sum_2",
    "sum_3",
    "sum_4",
    "l2_norm",
    "rms",
    "mean",
    "variance",
    "std",
    "skewness",
    "kurtosis",
    "excess_kurtosis",
    "is_layered",
    "layer_index",
    "layer_stack",
    "layer_label",
    "param_type",
    "param_family",
    "param_kind",
    "item_type",
    "item_family",
    "item_kind",
    "tensor_endpoint",
    "tensor_index",
}


st.title("Training Stats Dashboard")

with st.sidebar:
    st.header("Data")
    path_text = st.text_area(
        "Training stats paths or parent directory",
        value=DEFAULT_SOURCE_PATH,
        height=96,
        help=(
            "Enter one path per line. A path can be a JSONL file, a training_stats directory, "
            "or a parent directory containing many training_stats directories."
        ),
    )

input_paths = parse_source_text(path_text)
missing_paths = [path for path in input_paths if not path.exists()]
if missing_paths:
    st.error("File not found:\n" + "\n".join(str(path) for path in missing_paths))
    st.stop()

run_sources = discover_run_sources(path_text)
if not run_sources:
    st.error("No JSONL files or training_stats directories were found for the supplied path(s).")
    st.stop()

run_source_by_label = {source.label: source for source in run_sources}
default_run_count = len(run_sources) if len(run_sources) <= 6 else 6
default_run_labels = [source.label for source in run_sources[:default_run_count]]

with st.sidebar:
    if len(run_sources) > 1:
        selected_run_labels = st.multiselect(
            "Runs",
            [source.label for source in run_sources],
            default=default_run_labels,
            help="Runs are discovered from JSONL files or descendant training_stats directories.",
        )
    else:
        selected_run_labels = [run_sources[0].label]
        st.caption(f"Run: `{run_sources[0].label}`")

    with st.expander("Discovered runs", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"run": source.label, "jsonl_files": source.jsonl_count, "path": source.path}
                    for source in run_sources
                ]
            ),
            width="stretch",
            hide_index=True,
        )

if not selected_run_labels:
    st.warning("Select at least one run.")
    st.stop()

selected_run_sources = [run_source_by_label[label] for label in selected_run_labels]

try:
    with st.spinner("Preparing Parquet cache..."):
        cache_entries = ensure_parquet_caches(
            tuple((source.label, source.path) for source in selected_run_sources)
        )
    store = MetricStore(cache_entries)
except Exception as exc:  # pragma: no cover - displayed in Streamlit
    st.exception(exc)
    st.stop()

store_columns = set(store.columns())

with st.sidebar:
    with st.expander("Loaded files", expanded=False):
        for source in selected_run_sources:
            st.markdown(f"**{source.label}**")
            source_path = Path(source.path)
            for jsonl_path in discover_jsonl_paths(source_path):
                display_path = str(
                    jsonl_path.relative_to(source_path) if source_path.is_dir() else jsonl_path.name
                )
                st.text(display_path)

    with st.expander("Parquet cache", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "run": entry.run_label,
                        "rows": entry.row_count,
                        "jsonl_files": entry.file_count,
                        "parquet_files": len(entry.parquet_paths),
                        "cache": "created" if entry.was_created else "reused",
                        "parquet_cache": entry.parquet_path,
                    }
                    for entry in cache_entries
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    st.header("View")
    selected_view = st.radio(
        "Dashboard view",
        ["Overview", "By Iter", "By Layer", "Heatmaps", "Raw Items", "Data"],
        key="dashboard_view",
        label_visibility="collapsed",
    )
    selected_statistic = st.selectbox("Statistic", list(STATISTIC_COLUMNS), index=0)
    statistic_col = STATISTIC_COLUMNS[selected_statistic]
    st.caption(f"Statistic formula: `{statistic_formula(selected_statistic)}`")
    log_y = st.checkbox("Log y-axis for line charts", value=False)

    st.header("Filters")

    filters: list[tuple[str, str, object]] = []

    metric_options = store.distinct("metric")
    default_metrics = metric_options if len(metric_options) <= 6 else metric_options[:1]
    selected_metrics = st.multiselect("Metrics", metric_options, default=default_metrics)
    if not selected_metrics:
        st.warning("Select at least one metric.")
        st.stop()
    append_filter(filters, "metric", "in", tuple(selected_metrics))
    selected_summary = store.summary(tuple(filters))

    x_axis_options = [
        column
        for column in ["iter", "consumed_train_samples", "source_line"]
        if column in store_columns and store.has_non_null(column, filters)
    ]
    if not x_axis_options:
        st.error("No usable step column found for this metric.")
        st.stop()
    x_axis = st.selectbox("X axis", x_axis_options, index=0)

    for column in store.metadata_filter_columns(RESERVED_FILTER_COLUMNS, filters):
        options = store.distinct(column, filters)
        selected_values = st.multiselect(column.replace("_", " ").title(), options, default=options)
        if selected_values != options:
            append_filter(filters, column, "in", tuple(selected_values))

    if store.count_rows(filters) == 0:
        st.warning("No rows match the selected metric filters.")
        st.stop()

    x_minmax = store.min_max(x_axis, filters)
    if x_minmax is None:
        st.error(f"No non-null `{x_axis}` values match the selected filters.")
        st.stop()
    min_x_raw, max_x_raw = x_minmax
    min_x, max_x = numeric_slider_bounds(min_x_raw, max_x_raw)
    if min_x < max_x:
        selected_x = st.slider("Step range", min_value=min_x, max_value=max_x, value=(min_x, max_x))
        append_filter(filters, x_axis, "range", selected_x)

    include_global = st.checkbox("Include global/unlayered items", value=True)

    layer_minmax = store.min_max("layer_index", with_filter(filters, "is_layered", "is_true", None))
    if layer_minmax is not None:
        layer_min, layer_max = numeric_slider_bounds(*layer_minmax)
        selected_layers = st.slider(
            "Layer index range", layer_min, layer_max, (layer_min, layer_max)
        )
        append_filter(
            filters,
            "layer_index",
            "layer_window",
            (selected_layers[0], selected_layers[1], include_global),
        )
    elif not include_global:
        append_filter(filters, "is_layered", "eq", True)

    family_options = store.distinct("item_family", filters)
    if family_options:
        selected_families = st.multiselect("Item families", family_options, default=family_options)
        if not selected_families:
            st.warning("Select at least one item family.")
            st.stop()
        if selected_families != family_options:
            append_filter(filters, "item_family", "in", tuple(selected_families))

    value_mode = st.selectbox(
        "Displayed value",
        [
            "Raw value",
            "Relative to first step",
            "Delta from first step",
            "Percent change from first step",
        ],
    )
    smoothing_window = st.slider("Rolling mean window", min_value=1, max_value=25, value=1)

    st.header("Series Limits")
    max_line_series = st.slider("Max line series", min_value=5, max_value=100, value=40, step=5)
    max_points_per_series = st.slider(
        "Max points per series", min_value=200, max_value=10000, value=2000, step=200
    )
    top_mover_count = st.slider("Top movers", min_value=5, max_value=50, value=15, step=5)

base_filters = tuple(filters)
filtered_row_count = store.count_rows(base_filters)
filtered_summary = store.summary(base_filters)

if filtered_row_count == 0:
    st.warning("No rows match the selected filters.")
    st.stop()

sampled_x_values = store.sampled_x_values(base_filters, x_axis, max_points_per_series)
chart_filters = list(base_filters)
if sampled_x_values is not None:
    append_filter(chart_filters, x_axis, "in", tuple(sampled_x_values))
chart_filters = tuple(chart_filters)

metric_display = (
    selected_metrics[0].replace("_", " ") if len(selected_metrics) == 1 else "selected metrics"
)
metric_title = f"{selected_statistic} of {metric_display}"

st.caption(
    f"Loaded {len(selected_run_sources)} run(s), "
    f"{sum(source.jsonl_count for source in selected_run_sources)} JSONL file(s), "
    f"{sum(entry.row_count for entry in cache_entries):,} cached row(s)"
)
if sampled_x_values is not None:
    st.caption(
        f"Line charts are sampled to {len(sampled_x_values):,} {x_axis} value(s) for rendering."
    )

metric_cols = st.columns(6)
metric_cols[0].metric("Steps", f"{selected_summary['steps']:,}")
metric_cols[1].metric("Runs", f"{len(selected_run_sources):,}")
metric_cols[2].metric("Metrics", f"{len(selected_metrics):,}")
metric_cols[3].metric("Layer Groups", f"{filtered_summary['layers']:,}")
metric_cols[4].metric("Filtered Rows", f"{filtered_row_count:,}")
metric_cols[5].metric("Statistic", selected_statistic)

compare_cols = selected_compare_columns_for_selection(selected_run_labels, selected_metrics)
group_base = [x_axis, *compare_cols]

if selected_view == "Overview":
    total_metric = add_series_label(
        store.aggregate(chart_filters, group_base, statistic_col, selected_statistic),
        compare_cols,
        "total",
    )
    family_metric = add_series_label(
        store.aggregate(
            chart_filters, [*group_base, "item_family"], statistic_col, selected_statistic
        ),
        [*compare_cols, "item_family"],
        "item_family",
    )

    left, right = st.columns([2, 1])
    with left:
        line_chart(
            total_metric,
            x_axis,
            "series",
            value_mode,
            f"Total {metric_title}",
            log_y,
            smoothing_window,
        )
    with right:
        movers = store.top_movers(
            base_filters, x_axis, statistic_col, [*compare_cols, "item"], top_mover_count
        )
        st.subheader("Largest Raw-Item Moves")
        mover_columns = [
            column
            for column in [
                "run",
                "metric",
                "item",
                "layer",
                "item_type",
                "tensor_role",
                "first",
                "last",
                "delta",
                "relative_change",
            ]
            if column in movers.columns
        ]
        st.dataframe(movers[mover_columns], width="stretch", hide_index=True)

    line_chart(
        family_metric,
        x_axis,
        "series",
        value_mode,
        f"{metric_title.title()} by Item Family",
        log_y,
        smoothing_window,
    )

elif selected_view == "By Iter":
    series_group = st.selectbox(
        "Series group",
        ["Layers", "Item Types", "Item Families", "Item Leaf Kinds"],
        key="by_iter_series_group",
    )

    if series_group == "Layers":
        layer_filters = with_filter(base_filters, "is_layered", "is_true", None)
        layer_options = store.distinct("layer_label", layer_filters)
        selected_layer_labels = st.multiselect(
            "Layers", layer_options, default=layer_options[:max_line_series]
        )
        selected_filters = with_filter(chart_filters, "is_layered", "is_true", None)
        if selected_layer_labels:
            append_filter(selected_filters, "layer_label", "in", tuple(selected_layer_labels))
        chart_df = add_series_label(
            store.aggregate(
                selected_filters, [*group_base, "layer_label"], statistic_col, selected_statistic
            ),
            [*compare_cols, "layer_label"],
            "layer",
        )
        title = f"{metric_title.title()} by Layer"
    elif series_group == "Item Types":
        type_options = store.distinct("item_type", base_filters)
        default_types = store.top_group_values(
            "item_type", base_filters, x_axis, compare_cols, statistic_col, max_line_series
        )
        selected_types = st.multiselect("Item types", type_options, default=default_types)
        selected_filters = list(chart_filters)
        if selected_types:
            append_filter(selected_filters, "item_type", "in", tuple(selected_types))
        chart_df = add_series_label(
            store.aggregate(
                selected_filters, [*group_base, "item_type"], statistic_col, selected_statistic
            ),
            [*compare_cols, "item_type"],
            "item_type",
        )
        title = f"{metric_title.title()} by Item Type"
    elif series_group == "Item Families":
        family_options_for_iter = store.distinct("item_family", base_filters)
        selected_iter_families = st.multiselect(
            "Item families",
            family_options_for_iter,
            default=family_options_for_iter[:max_line_series],
        )
        selected_filters = list(chart_filters)
        if selected_iter_families:
            append_filter(selected_filters, "item_family", "in", tuple(selected_iter_families))
        chart_df = add_series_label(
            store.aggregate(
                selected_filters, [*group_base, "item_family"], statistic_col, selected_statistic
            ),
            [*compare_cols, "item_family"],
            "item_family",
        )
        title = f"{metric_title.title()} by Item Family"
    else:
        kind_options = store.distinct("item_kind", base_filters)
        selected_kinds = st.multiselect(
            "Item leaf kinds", kind_options, default=kind_options[:max_line_series]
        )
        selected_filters = list(chart_filters)
        if selected_kinds:
            append_filter(selected_filters, "item_kind", "in", tuple(selected_kinds))
        chart_df = add_series_label(
            store.aggregate(
                selected_filters, [*group_base, "item_kind"], statistic_col, selected_statistic
            ),
            [*compare_cols, "item_kind"],
            "item_kind",
        )
        title = f"{metric_title.title()} by Item Leaf Kind"

    line_chart(chart_df, x_axis, "series", value_mode, title, log_y, smoothing_window)

elif selected_view == "By Layer":
    fixed_count = store.distinct_count(x_axis, base_filters)
    fixed_minmax = store.min_max(x_axis, base_filters)
    if not fixed_minmax:
        st.info("No steps available for the current filters.")
    else:
        if fixed_count <= 1000:
            fixed_options = store.distinct(x_axis, base_filters)
            fixed_x = st.select_slider(
                f"Fixed {x_axis}", options=fixed_options, value=fixed_options[-1]
            )
        else:
            fixed_min, fixed_max = numeric_slider_bounds(*fixed_minmax)
            fixed_x = st.number_input(
                f"Fixed {x_axis}", min_value=fixed_min, max_value=fixed_max, value=fixed_max
            )
        chart_mode = st.radio(
            "Chart type", ["Lines", "Bars"], horizontal=True, key="by_layer_chart_type"
        )

        fixed_label = fixed_step_label(x_axis, fixed_x)
        fixed_layer_filters = [
            *base_filters,
            (x_axis, "eq", fixed_x),
            ("is_layered", "is_true", None),
        ]

        fixed_layer_metric = store.aggregate(
            fixed_layer_filters,
            [*compare_cols, "layer_index", "layer_label"],
            statistic_col,
            selected_statistic,
        )
        fixed_layer_metric = add_series_label(fixed_layer_metric, compare_cols, "total")
        layer_cross_section_chart(
            fixed_layer_metric,
            "series" if compare_cols else None,
            f"{metric_title.title()} by Layer at {fixed_label}",
            log_y,
            chart_mode,
        )

        type_options = store.distinct("item_type", fixed_layer_filters)
        default_cross_types = store.top_group_values(
            "item_type",
            fixed_layer_filters,
            x_axis,
            compare_cols,
            statistic_col,
            min(6, max_line_series),
        )
        selected_cross_types = st.multiselect(
            "Item types by layer", type_options, default=default_cross_types
        )
        fixed_type_filters = list(fixed_layer_filters)
        if selected_cross_types:
            append_filter(fixed_type_filters, "item_type", "in", tuple(selected_cross_types))
        fixed_layer_type_metric = store.aggregate(
            fixed_type_filters,
            [*compare_cols, "layer_index", "layer_label", "item_type"],
            statistic_col,
            selected_statistic,
        )
        fixed_layer_type_metric = add_series_label(
            fixed_layer_type_metric, [*compare_cols, "item_type"], "item_type"
        )
        layer_cross_section_chart(
            fixed_layer_type_metric,
            "series",
            f"{metric_title.title()} Item Types by Layer at {fixed_label}",
            log_y,
            chart_mode,
        )

        family_options_for_layers = store.distinct("item_family", fixed_layer_filters)
        selected_cross_families = st.multiselect(
            "Item families by layer",
            family_options_for_layers,
            default=family_options_for_layers[: min(6, max_line_series)],
        )
        fixed_family_filters = list(fixed_layer_filters)
        if selected_cross_families:
            append_filter(fixed_family_filters, "item_family", "in", tuple(selected_cross_families))
        fixed_layer_family_metric = store.aggregate(
            fixed_family_filters,
            [*compare_cols, "layer_index", "layer_label", "item_family"],
            statistic_col,
            selected_statistic,
        )
        fixed_layer_family_metric = add_series_label(
            fixed_layer_family_metric, [*compare_cols, "item_family"], "item_family"
        )
        layer_cross_section_chart(
            fixed_layer_family_metric,
            "series",
            f"{metric_title.title()} Item Families by Layer at {fixed_label}",
            log_y,
            chart_mode,
        )

elif selected_view == "Heatmaps":
    heatmap_filters = list(base_filters)
    heatmap_controls = st.columns(2)
    with heatmap_controls[0]:
        heatmap_run_options = store.distinct("run", heatmap_filters)
        if len(heatmap_run_options) > 1:
            selected_heatmap_run = st.selectbox("Heatmap run", heatmap_run_options)
            append_filter(heatmap_filters, "run", "eq", selected_heatmap_run)
    with heatmap_controls[1]:
        heatmap_metric_options = store.distinct("metric", heatmap_filters)
        if len(heatmap_metric_options) > 1:
            selected_heatmap_metric = st.selectbox("Heatmap metric", heatmap_metric_options)
            append_filter(heatmap_filters, "metric", "eq", selected_heatmap_metric)

    sampled_heatmap_x = store.sampled_x_values(
        heatmap_filters, x_axis, min(max_points_per_series, 1000)
    )
    if sampled_heatmap_x is not None:
        append_filter(heatmap_filters, x_axis, "in", tuple(sampled_heatmap_x))
        st.caption(
            f"Heatmaps are sampled to {len(sampled_heatmap_x):,} {x_axis} value(s) for rendering."
        )

    heatmap_layer_metric = store.aggregate(
        with_filter(heatmap_filters, "is_layered", "is_true", None),
        [x_axis, "layer_label"],
        statistic_col,
        selected_statistic,
    )
    heatmap_type_metric = store.aggregate(
        heatmap_filters, [x_axis, "item_type"], statistic_col, selected_statistic
    )
    heatmap(
        heatmap_layer_metric,
        x_axis,
        "layer_label",
        value_mode,
        f"{metric_title.title()} Layer x Step Heatmap",
        value_mode,
    )
    heatmap(
        heatmap_type_metric,
        x_axis,
        "item_type",
        value_mode,
        f"{metric_title.title()} Item Type x Step Heatmap",
        value_mode,
    )

elif selected_view == "Raw Items":
    item_options = store.distinct("item", base_filters)
    default_items = store.top_group_values(
        "item", base_filters, x_axis, compare_cols, statistic_col, min(10, max_line_series)
    )
    selected_items = st.multiselect("Raw item series", item_options, default=default_items)
    raw_filters = list(chart_filters)
    if selected_items:
        append_filter(raw_filters, "item", "in", tuple(selected_items))
    raw_selected = store.raw_items(raw_filters, x_axis, statistic_col)
    raw_selected["statistic"] = selected_statistic
    raw_selected = add_series_label(raw_selected, [*compare_cols, "item"], "item")
    line_chart(
        raw_selected,
        x_axis,
        "series",
        value_mode,
        f"Raw {metric_title} by Item",
        log_y,
        smoothing_window,
    )

elif selected_view == "Data":
    st.subheader("Aggregated Data")
    dataset = st.selectbox(
        "Table", ["Layer metrics", "Item type metrics", "Family metrics", "Filtered long records"]
    )

    if dataset == "Layer metrics":
        table_df = add_series_label(
            store.aggregate(
                chart_filters, [*group_base, "layer_label"], statistic_col, selected_statistic
            ),
            [*compare_cols, "layer_label"],
            "layer",
        )
    elif dataset == "Item type metrics":
        table_df = add_series_label(
            store.aggregate(
                chart_filters, [*group_base, "item_type"], statistic_col, selected_statistic
            ),
            [*compare_cols, "item_type"],
            "item_type",
        )
    elif dataset == "Family metrics":
        table_df = add_series_label(
            store.aggregate(
                chart_filters, [*group_base, "item_family"], statistic_col, selected_statistic
            ),
            [*compare_cols, "item_family"],
            "item_family",
        )
    else:
        table_df = store.raw_records(base_filters, statistic_col, limit=50_000)
        if filtered_row_count > len(table_df):
            st.caption(
                f"Showing the first {len(table_df):,} of {filtered_row_count:,} filtered raw rows."
            )

    st.dataframe(table_df, width="stretch", hide_index=True)
    st.download_button(
        "Download CSV",
        data=table_df.to_csv(index=False),
        file_name=f"{dataset.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
