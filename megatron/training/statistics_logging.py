# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""JSON Lines logging for high-cardinality training statistics."""

import json
import os
from collections.abc import Iterable

import torch


def _get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def append_training_stat(log_dir: str, stat_name: str, record: dict, rank: int | None = None):
    """Append one JSONL record for a training statistic.

    Each rank writes to its own file under ``{log_dir}/training_stats/{stat_name}/``.
    Callers decide which ranks should write; this function only handles the file layout.
    """
    rank = _get_rank() if rank is None else rank
    stat_dir = os.path.join(log_dir, "training_stats", stat_name)
    os.makedirs(stat_dir, exist_ok=True)
    filepath = os.path.join(stat_dir, f"rank{rank}.jsonl")
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


def save_l2_norm_by_param(
    log_dir: str,
    stat_name: str,
    iteration: int,
    consumed_train_samples: int,
    values_by_param: Iterable[tuple[str, float]],
    rank: int | None = None,
    extra_record_fields: dict | None = None,
) -> None:
    """Append one per-parameter L2 norm record."""
    values = {name: float(norm) for name, norm in values_by_param}
    if not values:
        return

    record = {
        "iter": iteration,
        "consumed_train_samples": consumed_train_samples,
        "stat": stat_name,
        "norm_type": "l2",
        "values": values,
    }
    if extra_record_fields is not None:
        record.update(extra_record_fields)

    append_training_stat(
        log_dir,
        stat_name,
        record,
        rank=rank,
    )


def save_params_norm_by_param(
    log_dir: str,
    iteration: int,
    consumed_train_samples: int,
    params_norm_by_param: Iterable[tuple[str, float]],
    rank: int | None = None,
) -> None:
    """Append one per-parameter parameter L2 norm record."""
    save_l2_norm_by_param(
        log_dir,
        "params_norm_by_param",
        iteration,
        consumed_train_samples,
        params_norm_by_param,
        rank=rank,
    )


def save_grad_norm_by_param(
    log_dir: str,
    iteration: int,
    consumed_train_samples: int,
    grad_norm_by_param: Iterable[tuple[str, float]],
    rank: int | None = None,
) -> None:
    """Append one pre-clipping per-parameter gradient L2 norm record."""
    save_l2_norm_by_param(
        log_dir,
        "grad_norm_by_param",
        iteration,
        consumed_train_samples,
        grad_norm_by_param,
        rank=rank,
        extra_record_fields={"gradient_stage": "pre_clip"},
    )
