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


def save_params_norm_by_param(
    log_dir: str,
    iteration: int,
    consumed_train_samples: int,
    params_norm_by_param: Iterable[tuple[str, float]],
    rank: int | None = None,
):
    """Append one per-parameter L2 norm record."""
    values = {name: float(norm) for name, norm in params_norm_by_param}
    if not values:
        return

    append_training_stat(
        log_dir,
        "params_norm_by_param",
        {
            "iter": iteration,
            "consumed_train_samples": consumed_train_samples,
            "stat": "params_norm_by_param",
            "norm_type": "l2",
            "values": values,
        },
        rank=rank,
    )
