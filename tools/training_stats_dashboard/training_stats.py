# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Utilities for loading per-item training metric JSONL dumps."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

_INTEGER_RE = re.compile(r"^\d+$")
_TENSOR_ENDPOINT_RE = re.compile(r"^(input|output)(\d*)$")

_STRONG_LAYER_PREV_TOKENS = {
    "layer",
    "layers",
    "block",
    "blocks",
    "h",
    "resblocks",
    "transformer_blocks",
    "decoder_layers",
    "encoder_layers",
}

_STACK_CONTEXT_PREV_TOKENS = {"encoder", "decoder", "transformer", "module", "model", "backbone"}

_NON_LAYER_PREV_TOKENS = {
    "expert",
    "experts",
    "local_experts",
    "shared_experts",
    "head",
    "heads",
    "rank",
    "ranks",
    "shard",
    "shards",
    "adapter",
    "adapters",
}
_MIN_LAYER_TOKEN_SCORE = 6


def natural_sort_key(value: Any) -> list[Any]:
    """Return a key that sorts embedded numbers numerically."""

    text = "" if value is None else str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def split_param_name(name: str) -> list[str]:
    """Split a metric item path into structural tokens."""

    return [part for part in re.split(r"[./]", name) if part]


def infer_tensor_endpoint(parts: list[str]) -> dict[str, Any]:
    if not parts:
        return {
            "tensor_endpoint": None,
            "tensor_role": None,
            "tensor_index": None,
            "module_type": None,
        }

    endpoint = parts[-1]
    match = _TENSOR_ENDPOINT_RE.match(endpoint)
    if not match:
        return {
            "tensor_endpoint": endpoint,
            "tensor_role": None,
            "tensor_index": None,
            "module_type": ".".join(parts[:-1]) if len(parts) > 1 else None,
        }

    tensor_index = int(match.group(2)) if match.group(2) else None
    return {
        "tensor_endpoint": endpoint,
        "tensor_role": match.group(1),
        "tensor_index": tensor_index,
        "module_type": ".".join(parts[:-1]) if len(parts) > 1 else None,
    }


def _is_integer_token(value: str) -> bool:
    return bool(_INTEGER_RE.match(value))


def _is_non_layer_index_context(value: str) -> bool:
    value = value.lower()
    if value in _NON_LAYER_PREV_TOKENS:
        return True
    return any(marker in value for marker in ("expert", "head", "rank", "shard", "adapter"))


def find_layer_token(parts: list[str]) -> int | None:
    """Find the token index that most likely identifies a repeated model layer.

    This intentionally uses lightweight heuristics rather than architecture-specific
    names. Common LLM patterns such as ``model.layers.12.*``, ``decoder.layers.12.*``,
    ``transformer.h.12.*``, and ResNet-like ``layer1.0.*`` receive high scores. Bare
    module-list indices are not enough on their own because names can contain other
    indices, such as expert, head, rank, or shard ids.
    """

    best_index: int | None = None
    best_score = 0

    for index, token in enumerate(parts):
        if not _is_integer_token(token):
            continue

        previous = parts[index - 1].lower() if index > 0 else ""
        score = 0

        if _is_non_layer_index_context(previous):
            score -= 24
        if previous in _STRONG_LAYER_PREV_TOKENS:
            score += 12
        if previous in _STACK_CONTEXT_PREV_TOKENS:
            score += 6
        if "layer" in previous or "block" in previous:
            score += 8
        if previous == "layers" and index >= 2 and parts[index - 2].lower() == "mtp_model_layer":
            score += 20
        if index < len(parts) - 1:
            score += 1

        if score > best_score:
            best_index = index
            best_score = score

    return best_index if best_score >= _MIN_LAYER_TOKEN_SCORE else None


def infer_param_metadata(param_name: str) -> dict[str, Any]:
    """Infer layer and item-type metadata from a metric item name."""

    parts = split_param_name(param_name)
    layer_token_index = find_layer_token(parts)

    if layer_token_index is None:
        param_type = ".".join(parts) if parts else param_name
        endpoint_metadata = infer_tensor_endpoint(parts)
        return {
            "is_layered": False,
            "layer_index": None,
            "layer_stack": "(global)",
            "layer_label": "(global)",
            "param_type": param_type,
            "param_family": parts[0] if parts else "(unknown)",
            "param_kind": parts[-1] if parts else "(unknown)",
            "item_type": param_type,
            "item_family": parts[0] if parts else "(unknown)",
            "item_kind": parts[-1] if parts else "(unknown)",
            **endpoint_metadata,
        }

    layer_index = int(parts[layer_token_index])
    layer_stack_parts = parts[:layer_token_index]
    layer_stack = ".".join(layer_stack_parts) if layer_stack_parts else "layer"
    rest = parts[layer_token_index + 1 :]
    param_type = ".".join(rest) if rest else "(layer)"
    endpoint_metadata = infer_tensor_endpoint(rest)

    return {
        "is_layered": True,
        "layer_index": layer_index,
        "layer_stack": layer_stack,
        "layer_label": f"{layer_stack}.{layer_index}",
        "param_type": param_type,
        "param_family": rest[0] if rest else "(layer)",
        "param_kind": rest[-1] if rest else "(unknown)",
        "item_type": param_type,
        "item_family": rest[0] if rest else "(layer)",
        "item_kind": rest[-1] if rest else "(unknown)",
        **endpoint_metadata,
    }


def discover_jsonl_paths(path: str | Path) -> list[Path]:
    """Return JSONL files from a file or directory source."""

    path = Path(path)
    if path.is_file():
        return [path] if path.suffix == ".jsonl" else []
    if path.is_dir():
        return sorted(path.rglob("*.jsonl"), key=lambda item: natural_sort_key(str(item)))
    return []


def _payload_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    known_keys = {
        "values",
        "iter",
        "step",
        "global_step",
        "consumed_train_samples",
        "stat",
        "norm_type",
    }
    return {
        key: value
        for key, value in payload.items()
        if key not in known_keys and isinstance(value, (str, int, float, bool, type(None)))
    }


def normalize_value_moments(raw_value: Any) -> dict[str, float] | None:
    """Normalize a raw JSON value to count and power sums.

    The current training stats store dictionaries with ``count`` and ``sum_k``.
    Older dashboard inputs stored a scalar norm per parameter; those are mapped
    to a one-element distribution for compatibility.
    """

    if isinstance(raw_value, dict):
        count = raw_value.get("count", raw_value.get("sum_0"))
        if count is None:
            return None

        try:
            moments = {
                "count": float(count),
                "sum_1": float(raw_value.get("sum_1", 0.0)),
                "sum_2": float(raw_value.get("sum_2", 0.0)),
                "sum_3": float(raw_value.get("sum_3", 0.0)),
                "sum_4": float(raw_value.get("sum_4", 0.0)),
            }
        except (TypeError, ValueError):
            return None
        return moments

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None

    return {
        "count": 1.0,
        "sum_1": value,
        "sum_2": value * value,
        "sum_3": value**3,
        "sum_4": value**4,
    }


def iter_metric_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield one normalized record per parameter value from JSONL file(s)."""

    path = Path(path)
    files = discover_jsonl_paths(path)
    for jsonl_path in files:
        yield from iter_metric_file_records(jsonl_path)


def infer_metric_scope(metric: str | None) -> str:
    if not metric:
        return "item"
    if metric.endswith("_by_param"):
        return "param"
    if metric.endswith("_by_layer"):
        return "layer"
    return "item"


def iter_metric_file_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield one normalized record per parameter value in a single JSONL dump.

    An invalid unterminated final line is ignored because the file may be read while
    a training process is appending its next record. Invalid completed lines still
    raise ``JSONDecodeError``.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for source_line, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if not raw_line.endswith("\n"):
                    break
                raise
            values = payload.get("values")
            if not isinstance(values, dict):
                continue

            step = payload.get("iter", payload.get("step", payload.get("global_step", source_line)))
            consumed_train_samples = payload.get("consumed_train_samples")
            metric = payload.get("stat") or path.parent.name
            metric_scope = infer_metric_scope(metric)
            row_metadata = _payload_metadata(payload)

            for item_name, raw_value in values.items():
                moments = normalize_value_moments(raw_value)
                if moments is None:
                    continue

                metadata = infer_param_metadata(item_name)
                yield {
                    "source_file": path.name,
                    "source_path": str(path),
                    "source_line": source_line,
                    "iter": step,
                    "consumed_train_samples": consumed_train_samples,
                    "metric": metric,
                    "metric_scope": metric_scope,
                    "stat": metric,
                    "norm_type": payload.get("norm_type"),
                    "item": item_name,
                    "param": item_name,
                    **moments,
                    **row_metadata,
                    **metadata,
                }
