# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cached Parquet conversion and DuckDB queries for training metrics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import duckdb
except ImportError:  # pragma: no cover - displayed in Streamlit at runtime
    duckdb = None

from training_stats import discover_jsonl_paths, iter_metric_records, natural_sort_key

CACHE_FORMAT_VERSION = 1
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "metric_parquet"
BATCH_SIZE = 100_000

MOMENT_COLUMNS = ["count", "sum_1", "sum_2", "sum_3", "sum_4"]
DERIVED_STAT_COLUMNS = [
    "l2_norm",
    "rms",
    "mean",
    "variance",
    "std",
    "skewness",
    "kurtosis",
    "excess_kurtosis",
]

NUMERIC_COLUMNS = {
    "source_line",
    "iter",
    "consumed_train_samples",
    "count",
    "sum_1",
    "sum_2",
    "sum_3",
    "sum_4",
    "layer_index",
    "tensor_index",
}
BOOLEAN_COLUMNS = {"is_layered"}
BASE_COLUMNS = [
    "source_file",
    "source_path",
    "source_line",
    "iter",
    "consumed_train_samples",
    "metric",
    "metric_scope",
    "stat",
    "norm_type",
    "norm_type_label",
    "item",
    "param",
    *MOMENT_COLUMNS,
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
    "tensor_role",
    "tensor_index",
    "module_type",
]
EXCLUDED_RECORD_COLUMNS = {"value", "value_sq", *DERIVED_STAT_COLUMNS}


@dataclass(frozen=True)
class SourceFingerprint:
    key: str
    source_path: str
    files: tuple[dict[str, Any], ...]
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class ParquetCacheEntry:
    run_label: str
    run_path: str
    parquet_path: str
    parquet_paths: tuple[str, ...]
    manifest_path: str
    cache_key: str
    row_count: int
    file_count: int
    total_bytes: int
    created_at: float
    was_created: bool


@dataclass(frozen=True)
class CacheBuild:
    index: int
    run_label: str
    fingerprint: SourceFingerprint
    cache_dir: Path
    manifest_path: Path
    tmp_cache_dir: Path
    tmp_manifest_path: Path


Filter = tuple[str, str, Any]


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def quote_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def fingerprint_source(path: str | Path) -> SourceFingerprint:
    source_path = Path(path).expanduser().resolve()
    files = discover_jsonl_paths(source_path)
    file_entries = []
    total_bytes = 0
    base_path = source_path if source_path.is_dir() else source_path.parent

    for jsonl_path in files:
        resolved = jsonl_path.resolve()
        stat = resolved.stat()
        total_bytes += stat.st_size
        try:
            relative_path = str(resolved.relative_to(base_path))
        except ValueError:
            relative_path = resolved.name
        file_entries.append(
            {
                "path": str(resolved),
                "relative_path": relative_path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    payload = {
        "format": CACHE_FORMAT_VERSION,
        "source_path": str(source_path),
        "files": file_entries,
    }
    cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return SourceFingerprint(
        key=cache_key,
        source_path=str(source_path),
        files=tuple(file_entries),
        file_count=len(file_entries),
        total_bytes=total_bytes,
    )


def cache_paths(cache_key: str) -> tuple[Path, Path]:
    cache_dir = CACHE_DIR / cache_key[:2] / cache_key
    manifest_path = CACHE_DIR / cache_key[:2] / f"{cache_key}.manifest.json"
    return cache_dir, manifest_path


def part_file_name(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", relative_path.replace("/", "__")).strip(".-")
    stem = stem[:120] or "part"
    return f"{stem}.{digest}.parquet"


def discover_record_columns(path: str | Path) -> list[str]:
    seen = set(BASE_COLUMNS)
    for record in iter_metric_records(path):
        for column in record:
            if column not in EXCLUDED_RECORD_COLUMNS:
                seen.add(column)

    extras = sorted(seen - set(BASE_COLUMNS), key=natural_sort_key)
    return [*BASE_COLUMNS, *extras]


def stringify(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def normalize_batch(rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    df = df.loc[:, list(columns)]

    if "norm_type_label" in df.columns:
        df["norm_type_label"] = df["norm_type"].map(
            lambda value: "(none)" if value is None or pd.isna(value) else str(value)
        )

    for column in NUMERIC_COLUMNS.intersection(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in BOOLEAN_COLUMNS.intersection(df.columns):
        df[column] = df[column].astype("boolean")

    string_columns = set(df.columns) - NUMERIC_COLUMNS - BOOLEAN_COLUMNS
    for column in string_columns:
        df[column] = df[column].map(stringify).astype("string")

    return df


def write_parquet_cache(source_path: str | Path, parquet_path: Path, columns: Sequence[str]) -> int:
    row_count = 0
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal row_count, writer, rows
        if not rows:
            return

        df = normalize_batch(rows, columns)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        row_count += len(rows)
        rows = []

    try:
        for record in iter_metric_records(source_path):
            rows.append({column: record.get(column) for column in columns})
            if len(rows) >= BATCH_SIZE:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        empty_df = normalize_batch([], columns)
        pq.write_table(
            pa.Table.from_pandas(empty_df, preserve_index=False), parquet_path, compression="zstd"
        )

    return row_count


def cache_worker_count(file_count: int) -> int:
    configured_workers = os.environ.get("TRAINING_STATS_CACHE_WORKERS")
    if configured_workers:
        return max(1, min(file_count, int(configured_workers)))
    cpu_count = os.cpu_count() or 2
    return max(1, min(file_count, max(1, min(8, cpu_count - 1))))


def write_parquet_part(source_path: str, relative_path: str, parquet_path: str) -> dict[str, Any]:
    columns = discover_record_columns(source_path)
    row_count = write_parquet_cache(source_path, Path(parquet_path), columns)
    return {
        "source_path": source_path,
        "relative_path": relative_path,
        "parquet_file": Path(parquet_path).name,
        "parquet_path": str(Path(parquet_path)),
        "columns": columns,
        "row_count": row_count,
    }


def write_parquet_part_cli(
    source_path: str, relative_path: str, parquet_path: str, result_path: str
) -> None:
    result = write_parquet_part(source_path, relative_path, parquet_path)
    tmp_result_path = Path(result_path).with_suffix(".tmp.json")
    with tmp_result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)
    tmp_result_path.replace(result_path)


def convert_part_command(task: tuple[str, str, str], result_path: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "convert-part",
        task[0],
        task[1],
        task[2],
        str(result_path),
    ]


def run_subprocess_parts(
    tasks: list[tuple[str, str, str]], cache_dir: Path, worker_count: int
) -> list[dict[str, Any]]:
    pending = list(tasks)
    active: list[tuple[subprocess.Popen[str], tuple[str, str, str], Path]] = []
    results: list[dict[str, Any]] = []

    def stop_active_workers() -> None:
        for process, _, _ in active:
            if process.poll() is None:
                process.terminate()
        for process, _, _ in active:
            if process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()

    try:
        while pending or active:
            while pending and len(active) < worker_count:
                task = pending.pop(0)
                result_digest = hashlib.sha256(task[2].encode("utf-8")).hexdigest()[:12]
                result_path = cache_dir / f"{Path(task[2]).name}.{result_digest}.result.json"
                command = convert_part_command(task, result_path)
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                active.append((process, task, result_path))

            made_progress = False
            for process, task, result_path in list(active):
                return_code = process.poll()
                if return_code is None:
                    continue

                stdout, stderr = process.communicate()
                active.remove((process, task, result_path))
                made_progress = True
                if return_code != 0:
                    stop_active_workers()
                    detail = (stderr or stdout or "").strip()
                    if len(detail) > 4000:
                        detail = detail[-4000:]
                    raise RuntimeError(
                        "Parquet conversion worker failed for "
                        f"{task[1]} with exit code {return_code}."
                        + (f"\n\n{detail}" if detail else "")
                    )

                with result_path.open("r", encoding="utf-8") as handle:
                    results.append(json.load(handle))
                result_path.unlink(missing_ok=True)

            if not made_progress:
                time.sleep(0.1)
    except Exception:
        stop_active_workers()
        raise

    return sorted(results, key=lambda item: natural_sort_key(item["relative_path"]))


def parquet_part_tasks(
    fingerprint: SourceFingerprint, cache_dir: Path
) -> list[tuple[str, str, str]]:
    return [
        (
            file_entry["path"],
            file_entry["relative_path"],
            str(cache_dir / part_file_name(file_entry["relative_path"])),
        )
        for file_entry in fingerprint.files
    ]


def valid_manifest(
    manifest: dict[str, Any], fingerprint: SourceFingerprint, cache_dir: Path
) -> bool:
    if (
        manifest.get("cache_key") != fingerprint.key
        or manifest.get("format") != CACHE_FORMAT_VERSION
    ):
        return False
    parquet_files = manifest.get("parquet_files")
    if not isinstance(parquet_files, list) or not parquet_files:
        return False
    return all((cache_dir / item["parquet_file"]).exists() for item in parquet_files)


def entry_from_manifest(
    manifest: dict[str, Any],
    fingerprint: SourceFingerprint,
    cache_dir: Path,
    manifest_path: Path,
    run_label: str,
    was_created: bool,
) -> ParquetCacheEntry:
    parquet_paths = tuple(
        str(cache_dir / item["parquet_file"]) for item in manifest["parquet_files"]
    )
    return ParquetCacheEntry(
        run_label=run_label,
        run_path=fingerprint.source_path,
        parquet_path=str(cache_dir),
        parquet_paths=parquet_paths,
        manifest_path=str(manifest_path),
        cache_key=fingerprint.key,
        row_count=int(manifest.get("row_count", 0)),
        file_count=fingerprint.file_count,
        total_bytes=fingerprint.total_bytes,
        created_at=float(manifest.get("created_at", 0.0)),
        was_created=was_created,
    )


def read_valid_manifest(
    fingerprint: SourceFingerprint, cache_dir: Path, manifest_path: Path
) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest if valid_manifest(manifest, fingerprint, cache_dir) else None


def prepare_cache_build(
    index: int, path: str | Path, run_label: str
) -> ParquetCacheEntry | CacheBuild:
    fingerprint = fingerprint_source(path)
    if fingerprint.file_count == 0:
        raise ValueError(f"No JSONL files found under {path}")

    cache_dir, manifest_path = cache_paths(fingerprint.key)
    manifest = read_valid_manifest(fingerprint, cache_dir, manifest_path)
    if manifest is not None:
        return entry_from_manifest(
            manifest, fingerprint, cache_dir, manifest_path, run_label, was_created=False
        )

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_suffix = f"{os.getpid()}.{uuid.uuid4().hex}"
    return CacheBuild(
        index=index,
        run_label=run_label,
        fingerprint=fingerprint,
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        tmp_cache_dir=cache_dir.with_name(f"{cache_dir.name}.{tmp_suffix}.tmp"),
        tmp_manifest_path=manifest_path.with_name(f"{manifest_path.stem}.{tmp_suffix}.tmp.json"),
    )


def publish_cache_build(
    build: CacheBuild, parquet_files: list[dict[str, Any]], worker_count: int
) -> ParquetCacheEntry:
    parquet_files = sorted(parquet_files, key=lambda item: natural_sort_key(item["relative_path"]))
    row_count = sum(int(item["row_count"]) for item in parquet_files)
    created_at = time.time()
    manifest = {
        "format": CACHE_FORMAT_VERSION,
        "cache_key": build.fingerprint.key,
        "source_path": build.fingerprint.source_path,
        "files": list(build.fingerprint.files),
        "parquet_files": parquet_files,
        "row_count": row_count,
        "file_count": build.fingerprint.file_count,
        "total_bytes": build.fingerprint.total_bytes,
        "worker_count": worker_count,
        "created_at": created_at,
    }
    with build.tmp_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    existing_manifest = read_valid_manifest(build.fingerprint, build.cache_dir, build.manifest_path)
    if existing_manifest is not None:
        shutil.rmtree(build.tmp_cache_dir, ignore_errors=True)
        build.tmp_manifest_path.unlink(missing_ok=True)
        return entry_from_manifest(
            existing_manifest,
            build.fingerprint,
            build.cache_dir,
            build.manifest_path,
            build.run_label,
            was_created=False,
        )

    if build.cache_dir.exists():
        shutil.rmtree(build.cache_dir)
    build.tmp_cache_dir.replace(build.cache_dir)
    build.tmp_manifest_path.replace(build.manifest_path)
    return entry_from_manifest(
        manifest,
        build.fingerprint,
        build.cache_dir,
        build.manifest_path,
        build.run_label,
        was_created=True,
    )


def ensure_parquet_caches(run_sources: Sequence[tuple[str, str]]) -> list[ParquetCacheEntry]:
    entries: list[ParquetCacheEntry | None] = [None] * len(run_sources)
    builds: list[CacheBuild] = []
    all_tasks: list[tuple[str, str, str]] = []
    task_build_index: dict[str, int] = {}

    for index, (run_label, path) in enumerate(run_sources):
        prepared = prepare_cache_build(index, path, run_label)
        if isinstance(prepared, ParquetCacheEntry):
            entries[index] = prepared
            continue

        prepared.tmp_cache_dir.mkdir(parents=True, exist_ok=True)
        builds.append(prepared)
        for task in parquet_part_tasks(prepared.fingerprint, prepared.tmp_cache_dir):
            all_tasks.append(task)
            task_build_index[str(Path(task[2]).resolve())] = prepared.index

    if not builds:
        return [entry for entry in entries if entry is not None]

    worker_count = cache_worker_count(len(all_tasks))
    try:
        if worker_count == 1:
            results = [write_parquet_part(*task) for task in all_tasks]
        else:
            results = run_subprocess_parts(all_tasks, CACHE_DIR, worker_count)

        results_by_build: dict[int, list[dict[str, Any]]] = {build.index: [] for build in builds}
        for result in results:
            build_index = task_build_index[str(Path(result["parquet_path"]).resolve())]
            results_by_build[build_index].append(result)

        for build in builds:
            entries[build.index] = publish_cache_build(
                build, results_by_build[build.index], worker_count
            )
    except Exception:
        for build in builds:
            shutil.rmtree(build.tmp_cache_dir, ignore_errors=True)
            build.tmp_manifest_path.unlink(missing_ok=True)
        raise

    return [entry for entry in entries if entry is not None]


def compile_filters(filters: Sequence[Filter] | None) -> tuple[str, list[Any]]:
    if not filters:
        return "", []

    conditions: list[str] = []
    params: list[Any] = []
    for column, op, value in filters:
        quoted = quote_ident(column)
        if op == "in":
            values = list(value)
            non_null_values = [item for item in values if item is not None and not pd.isna(item)]
            has_null = len(non_null_values) != len(values)
            subconditions = []
            if non_null_values:
                placeholders = ", ".join("?" for _ in non_null_values)
                subconditions.append(f"{quoted} IN ({placeholders})")
                params.extend(non_null_values)
            if has_null:
                subconditions.append(f"{quoted} IS NULL")
            conditions.append("(" + " OR ".join(subconditions or ["FALSE"]) + ")")
        elif op == "eq":
            if value is None or pd.isna(value):
                conditions.append(f"{quoted} IS NULL")
            else:
                conditions.append(f"{quoted} = ?")
                params.append(value)
        elif op == "range":
            low, high = value
            conditions.append(f"{quoted} BETWEEN ? AND ?")
            params.extend([low, high])
        elif op == "not_null":
            conditions.append(f"{quoted} IS NOT NULL")
        elif op == "is_true":
            conditions.append(f"COALESCE({quoted}, FALSE)")
        elif op == "layer_window":
            low, high, include_global = value
            layered_condition = (
                "(COALESCE(\"is_layered\", FALSE) AND \"layer_index\" BETWEEN ? AND ?)"
            )
            params.extend([low, high])
            if include_global:
                conditions.append(f"({layered_condition} OR NOT COALESCE(\"is_layered\", FALSE))")
            else:
                conditions.append(layered_condition)
        else:
            raise ValueError(f"Unsupported filter operator: {op}")

    return "WHERE " + " AND ".join(conditions), params


class MetricStore:
    def __init__(self, entries: Sequence[ParquetCacheEntry]):
        if duckdb is None:
            raise RuntimeError(
                "DuckDB is not installed. Run `pip install -r requirements.txt` in the dashboard environment."
            )
        if not entries:
            raise ValueError("At least one Parquet cache entry is required.")

        self.entries = tuple(entries)
        self.record_columns = self._union_parquet_columns()
        self.con = duckdb.connect(database=":memory:")
        self.con.execute("PRAGMA threads=4")
        self._create_records_view()

    def _union_parquet_columns(self) -> list[str]:
        columns: list[str] = []
        seen: set[str] = set()
        for entry in self.entries:
            for parquet_path in entry.parquet_paths:
                schema = pq.read_schema(parquet_path)
                for column in schema.names:
                    if column in seen:
                        continue
                    columns.append(column)
                    seen.add(column)
        return columns

    def _create_records_view(self) -> None:
        selects = []
        for entry in self.entries:
            for parquet_path in entry.parquet_paths:
                entry_columns = set(pq.read_schema(parquet_path).names)
                projected_columns = []
                for column in self.record_columns:
                    if column in entry_columns:
                        projected_columns.append(quote_ident(column))
                    else:
                        projected_columns.append(f"NULL AS {quote_ident(column)}")
                selects.append(
                    "SELECT "
                    f"{quote_literal(entry.run_label)} AS run, "
                    f"{quote_literal(entry.run_path)} AS run_path, "
                    f"{', '.join(projected_columns)} "
                    f"FROM read_parquet({quote_literal(parquet_path)})"
                )
        self.con.execute("CREATE VIEW records AS " + " UNION ALL ".join(selects))

    def columns(self) -> list[str]:
        return self.con.execute("SELECT * FROM records LIMIT 0").fetchdf().columns.tolist()

    def query_df(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        return self.con.execute(sql, list(params or [])).fetchdf()

    def has_non_null(self, column: str, filters: Sequence[Filter] | None = None) -> bool:
        where_sql, params = compile_filters(filters)
        sql = f"SELECT COUNT(*) AS row_count FROM records {where_sql} AND {quote_ident(column)} IS NOT NULL"
        if not where_sql:
            sql = (
                f"SELECT COUNT(*) AS row_count FROM records WHERE {quote_ident(column)} IS NOT NULL"
            )
        return bool(self.con.execute(sql, params).fetchone()[0])

    def distinct(
        self, column: str, filters: Sequence[Filter] | None = None, limit: int | None = None
    ) -> list[Any]:
        where_sql, params = compile_filters(filters)
        limit_sql = f" LIMIT {int(limit)}" if limit else ""
        sql = f"SELECT DISTINCT {quote_ident(column)} AS value FROM records {where_sql}{limit_sql}"
        values = [row[0] for row in self.con.execute(sql, params).fetchall()]
        return sorted([value for value in values if value is not None], key=natural_sort_key)

    def min_max(
        self, column: str, filters: Sequence[Filter] | None = None
    ) -> tuple[Any, Any] | None:
        where_sql, params = compile_filters(filters)
        sql = f"SELECT MIN({quote_ident(column)}) AS min_value, MAX({quote_ident(column)}) AS max_value FROM records {where_sql}"
        min_value, max_value = self.con.execute(sql, params).fetchone()
        if min_value is None or max_value is None:
            return None
        return min_value, max_value

    def count_rows(self, filters: Sequence[Filter] | None = None) -> int:
        where_sql, params = compile_filters(filters)
        return int(
            self.con.execute(f"SELECT COUNT(*) FROM records {where_sql}", params).fetchone()[0]
        )

    def distinct_count(self, column: str, filters: Sequence[Filter] | None = None) -> int:
        where_sql, params = compile_filters(filters)
        sql = f"SELECT COUNT(DISTINCT {quote_ident(column)}) FROM records {where_sql}"
        return int(self.con.execute(sql, params).fetchone()[0])

    def summary(self, filters: Sequence[Filter] | None = None) -> dict[str, Any]:
        where_sql, params = compile_filters(filters)
        sql = f"""
            SELECT
                COUNT(*) AS records,
                COUNT(DISTINCT iter) AS steps,
                COUNT(DISTINCT item) AS items,
                COUNT(DISTINCT param) AS params,
                COUNT(DISTINCT CASE WHEN is_layered THEN layer_label ELSE NULL END) AS layers,
                COUNT(DISTINCT item_type) AS item_types,
                COUNT(DISTINCT param_type) AS param_types
            FROM records
            {where_sql}
        """
        row = self.con.execute(sql, params).fetchone()
        return {
            "records": int(row[0] or 0),
            "steps": int(row[1] or 0),
            "items": int(row[2] or 0),
            "params": int(row[3] or 0),
            "layers": int(row[4] or 0),
            "item_types": int(row[5] or 0),
            "param_types": int(row[6] or 0),
        }

    def metadata_filter_columns(
        self, reserved_columns: set[str], filters: Sequence[Filter] | None = None
    ) -> list[str]:
        columns = [
            column
            for column in self.columns()
            if column not in reserved_columns and column not in {"run", "run_path"}
        ]
        selected_columns = []
        where_sql, params = compile_filters(filters)
        for column in columns:
            sql = f"SELECT COUNT(DISTINCT {quote_ident(column)}) FROM records {where_sql} AND {quote_ident(column)} IS NOT NULL"
            if not where_sql:
                sql = f"SELECT COUNT(DISTINCT {quote_ident(column)}) FROM records WHERE {quote_ident(column)} IS NOT NULL"
            unique_count = int(self.con.execute(sql, params).fetchone()[0] or 0)
            if 1 <= unique_count <= 20:
                selected_columns.append(column)
        return sorted(selected_columns, key=natural_sort_key)

    def _grouped_stats_cte(
        self, filters: Sequence[Filter] | None, group_cols: Sequence[str]
    ) -> tuple[str, list[Any]]:
        where_sql, params = compile_filters(filters)
        group_select = ", ".join(quote_ident(column) for column in group_cols)
        group_prefix = f"{group_select}, " if group_select else ""
        group_by = f"GROUP BY {group_select}" if group_select else ""
        sql = f"""
            WITH grouped AS (
                SELECT
                    {group_prefix}
                    SUM("count") AS "count",
                    SUM(sum_1) AS sum_1,
                    SUM(sum_2) AS sum_2,
                    SUM(sum_3) AS sum_3,
                    SUM(sum_4) AS sum_4,
                    COUNT(DISTINCT item) AS item_count,
                    COUNT(*) AS sample_count
                FROM records
                {where_sql}
                {group_by}
            ),
            moments AS (
                SELECT
                    *,
                    sum_1 / NULLIF("count", 0.0) AS mean,
                    sum_2 / NULLIF("count", 0.0) AS raw_2,
                    sum_3 / NULLIF("count", 0.0) AS raw_3,
                    sum_4 / NULLIF("count", 0.0) AS raw_4
                FROM grouped
            ),
            stats AS (
                SELECT
                    *,
                    GREATEST(raw_2 - mean * mean, 0.0) AS variance,
                    raw_3 - 3.0 * mean * raw_2 + 2.0 * POWER(mean, 3) AS centered_3,
                    raw_4 - 4.0 * mean * raw_3 + 6.0 * mean * mean * raw_2 - 3.0 * POWER(mean, 4) AS centered_4
                FROM moments
            ),
            derived AS (
                SELECT
                    *,
                    SQRT(GREATEST(sum_2, 0.0)) AS l2_norm,
                    SQRT(GREATEST(raw_2, 0.0)) AS rms,
                    SQRT(variance) AS std,
                    CASE WHEN variance > 0 THEN centered_3 / POWER(SQRT(variance), 3) ELSE NULL END AS skewness,
                    CASE WHEN variance > 0 THEN centered_4 / POWER(variance, 2) ELSE NULL END AS kurtosis,
                    CASE WHEN variance > 0 THEN centered_4 / POWER(variance, 2) - 3.0 ELSE NULL END AS excess_kurtosis
                FROM stats
            )
        """
        return sql, params

    def aggregate(
        self,
        filters: Sequence[Filter] | None,
        group_cols: Sequence[str],
        statistic_col: str,
        statistic_name: str,
    ) -> pd.DataFrame:
        cte_sql, params = self._grouped_stats_cte(filters, group_cols)
        select_cols = [quote_ident(column) for column in group_cols]
        data_cols = [quote_ident(column) for column in MOMENT_COLUMNS + DERIVED_STAT_COLUMNS]
        order_sql = " ORDER BY " + ", ".join(select_cols) if select_cols else ""
        sql = f"""
            {cte_sql}
            SELECT
                {", ".join([*select_cols, *data_cols])},
                item_count,
                item_count AS param_count,
                sample_count,
                {quote_ident(statistic_col)} AS metric_value
            FROM derived
            {order_sql}
        """
        df = self.query_df(sql, params)
        if df.empty:
            for column in [
                *group_cols,
                *MOMENT_COLUMNS,
                *DERIVED_STAT_COLUMNS,
                "item_count",
                "param_count",
                "sample_count",
                "metric_value",
            ]:
                if column not in df.columns:
                    df[column] = pd.Series(dtype=float)
        df["statistic"] = statistic_name
        return df

    def top_group_values(
        self,
        group_col: str,
        filters: Sequence[Filter] | None,
        x_axis: str,
        compare_cols: Sequence[str],
        statistic_col: str,
        limit: int,
    ) -> list[Any]:
        group_cols = [x_axis, *compare_cols, group_col]
        cte_sql, params = self._grouped_stats_cte(filters, group_cols)
        sql = f"""
            {cte_sql}
            SELECT {quote_ident(group_col)} AS value, MAX({quote_ident(statistic_col)}) AS metric_value
            FROM derived
            GROUP BY {quote_ident(group_col)}
            ORDER BY metric_value DESC NULLS LAST
            LIMIT {int(limit)}
        """
        return [row[0] for row in self.con.execute(sql, params).fetchall() if row[0] is not None]

    def sampled_x_values(
        self, filters: Sequence[Filter] | None, x_axis: str, max_points: int
    ) -> list[Any] | None:
        max_points = max(int(max_points), 1)
        count = self.distinct_count(x_axis, filters)
        if count <= max_points:
            return None

        stride = max(math.ceil(count / max_points), 1)
        where_sql, params = compile_filters([*(filters or []), (x_axis, "not_null", None)])
        sql = f"""
            WITH x_values AS (
                SELECT
                    {quote_ident(x_axis)} AS x_value,
                    ROW_NUMBER() OVER (ORDER BY {quote_ident(x_axis)}) AS rn
                FROM (
                    SELECT DISTINCT {quote_ident(x_axis)}
                    FROM records
                    {where_sql}
                )
            )
            SELECT x_value
            FROM x_values
            WHERE rn = 1 OR rn = {count} OR ((rn - 1) % {stride}) = 0
            ORDER BY x_value
        """
        return [row[0] for row in self.con.execute(sql, params).fetchall()]

    def _row_stats_cte(self, filters: Sequence[Filter] | None) -> tuple[str, list[Any]]:
        where_sql, params = compile_filters(filters)
        sql = f"""
            WITH base AS (
                SELECT *
                FROM records
                {where_sql}
            ),
            moments AS (
                SELECT
                    *,
                    sum_1 / NULLIF("count", 0.0) AS mean,
                    sum_2 / NULLIF("count", 0.0) AS raw_2,
                    sum_3 / NULLIF("count", 0.0) AS raw_3,
                    sum_4 / NULLIF("count", 0.0) AS raw_4
                FROM base
            ),
            stats AS (
                SELECT
                    *,
                    GREATEST(raw_2 - mean * mean, 0.0) AS variance,
                    raw_3 - 3.0 * mean * raw_2 + 2.0 * POWER(mean, 3) AS centered_3,
                    raw_4 - 4.0 * mean * raw_3 + 6.0 * mean * mean * raw_2 - 3.0 * POWER(mean, 4) AS centered_4
                FROM moments
            ),
            derived AS (
                SELECT
                    *,
                    SQRT(GREATEST(sum_2, 0.0)) AS l2_norm,
                    SQRT(GREATEST(raw_2, 0.0)) AS rms,
                    SQRT(variance) AS std,
                    CASE WHEN variance > 0 THEN centered_3 / POWER(SQRT(variance), 3) ELSE NULL END AS skewness,
                    CASE WHEN variance > 0 THEN centered_4 / POWER(variance, 2) ELSE NULL END AS kurtosis,
                    CASE WHEN variance > 0 THEN centered_4 / POWER(variance, 2) - 3.0 ELSE NULL END AS excess_kurtosis
                FROM stats
            )
        """
        return sql, params

    def raw_items(
        self, filters: Sequence[Filter] | None, x_axis: str, statistic_col: str
    ) -> pd.DataFrame:
        cte_sql, params = self._row_stats_cte(filters)
        selected_cols = [
            "run",
            "metric",
            x_axis,
            "item",
            "layer_label",
            "item_type",
            "item_family",
            "item_kind",
            "tensor_role",
            *MOMENT_COLUMNS,
            *DERIVED_STAT_COLUMNS,
        ]
        select_sql = ", ".join(quote_ident(column) for column in dict.fromkeys(selected_cols))
        sql = f"""
            {cte_sql}
            SELECT
                {select_sql},
                {quote_ident(statistic_col)} AS metric_value,
                1 AS item_count,
                1 AS param_count,
                1 AS sample_count
            FROM derived
            ORDER BY run, metric, item, {quote_ident(x_axis)}
        """
        return self.query_df(sql, params)

    def top_movers(
        self,
        filters: Sequence[Filter] | None,
        x_axis: str,
        statistic_col: str,
        group_cols: Sequence[str],
        limit: int,
    ) -> pd.DataFrame:
        cte_sql, params = self._row_stats_cte(filters)
        partition_cols = ", ".join(quote_ident(column) for column in group_cols)
        select_group_cols = ", ".join(quote_ident(column) for column in group_cols)
        aux_selects = [
            "ANY_VALUE(layer) AS layer",
            "ANY_VALUE(item_type) AS item_type",
            "ANY_VALUE(item_family) AS item_family",
            "ANY_VALUE(tensor_role) AS tensor_role",
        ]
        if "run" not in group_cols:
            aux_selects.append("ANY_VALUE(run) AS run")
        if "metric" not in group_cols:
            aux_selects.append("ANY_VALUE(metric) AS metric")
        sql = f"""
            {cte_sql},
            rows AS (
                SELECT
                    {select_group_cols},
                    {quote_ident(x_axis)} AS x_value,
                    {quote_ident(statistic_col)} AS metric_value,
                    layer_label,
                    item_type,
                    item_family,
                    tensor_role,
                    run,
                    metric
                FROM derived
            ),
            ranked AS (
                SELECT
                    *,
                    FIRST_VALUE(metric_value) OVER first_window AS first,
                    FIRST_VALUE(metric_value) OVER last_window AS last,
                    MIN(metric_value) OVER group_window AS min,
                    MAX(metric_value) OVER group_window AS max,
                    FIRST_VALUE(layer_label) OVER first_window AS layer
                FROM rows
                WINDOW
                    group_window AS (PARTITION BY {partition_cols}),
                    first_window AS (PARTITION BY {partition_cols} ORDER BY x_value ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),
                    last_window AS (PARTITION BY {partition_cols} ORDER BY x_value DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
            )
            SELECT
                {select_group_cols},
                {", ".join(aux_selects)},
                ANY_VALUE(first) AS first,
                ANY_VALUE(last) AS last,
                ANY_VALUE(min) AS min,
                ANY_VALUE(max) AS max,
                ANY_VALUE(last) - ANY_VALUE(first) AS delta,
                ABS(ANY_VALUE(last) - ANY_VALUE(first)) AS abs_delta,
                (ANY_VALUE(last) - ANY_VALUE(first)) / NULLIF(ANY_VALUE(first), 0.0) AS relative_change
            FROM ranked
            GROUP BY {select_group_cols}
            ORDER BY abs_delta DESC NULLS LAST
            LIMIT {int(limit)}
        """
        return self.query_df(sql, params)

    def raw_records(
        self, filters: Sequence[Filter] | None, statistic_col: str, limit: int
    ) -> pd.DataFrame:
        cte_sql, params = self._row_stats_cte(filters)
        sql = f"""
            {cte_sql}
            SELECT
                *,
                {quote_ident(statistic_col)} AS metric_value
            FROM derived
            LIMIT {int(limit)}
        """
        return self.query_df(sql, params)


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "convert-part":
        if len(sys.argv) != 6:
            raise SystemExit(
                "usage: metric_store.py convert-part SOURCE_JSONL RELATIVE_PATH PARQUET_PATH RESULT_JSON"
            )
        write_parquet_part_cli(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        return
    raise SystemExit(
        "usage: metric_store.py convert-part SOURCE_JSONL RELATIVE_PATH PARQUET_PATH RESULT_JSON"
    )


if __name__ == "__main__":
    main()
