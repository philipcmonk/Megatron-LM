# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for high-cardinality per-parameter statistics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from megatron.core import parallel_state
from megatron.core.utils import unwrap_model

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from amp_C import multi_tensor_l2norm
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        from megatron.core.utils import (
            local_multi_tensor_applier as multi_tensor_applier,
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
        )


_LAYER_NAME_PATTERN = re.compile(r"layers\.(\d+)")
_GROUPED_EXPERT_PATTERN = re.compile(r"^(.*\.mlp\.experts\.linear_fc\d\.weight)(\d+)(.*)$")
_SEQUENTIAL_EXPERT_PATTERN = re.compile(r"^(.*\.mlp\.experts\.local_experts\.)(\d+)(\..*)$")


@dataclass(frozen=True)
class NamedTensorBucket:
    """Named tensors that should be reduced over the same process groups."""

    names: Sequence[str]
    tensors: Sequence[torch.Tensor]
    reduce_groups: tuple[torch.distributed.ProcessGroup | None, ...] = ()


class PerParameterStatRegistry:
    """Canonical parameter-name registry for per-parameter statistics."""

    def __init__(self, model_chunks: Iterable[torch.nn.Module] | torch.nn.Module):
        self.model_chunks = unwrap_model(_normalize_model_chunks(model_chunks))
        self.cache_key = tuple(id(model_chunk) for model_chunk in self.model_chunks)
        self.param_to_name = self._build_local_param_to_name()
        self.name_to_index = self._build_name_to_index()
        self.index_to_name = sorted(self.name_to_index, key=self.name_to_index.get)

    def name_for_param(self, param: torch.nn.Parameter) -> str:
        """Return the canonical name for ``param``."""
        return self.param_to_name[param]

    @property
    def num_params(self) -> int:
        """Number of globally known parameters."""
        return len(self.name_to_index)

    def _build_local_param_to_name(self) -> dict[torch.nn.Parameter, str]:
        param_to_name = {}
        num_experts = _get_num_moe_experts(self.model_chunks)
        for model_chunk in self.model_chunks:
            for local_name, param in model_chunk.named_parameters():
                param_to_name[param] = _canonical_param_name(
                    model_chunk, local_name, param, num_experts
                )
        return param_to_name

    def _build_name_to_index(self) -> dict[str, int]:
        local_names = list(self.param_to_name.values())
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered_names = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_names, local_names)
            all_names = set()
            for names in gathered_names:
                all_names.update(names)
        else:
            all_names = set(local_names)
        return {name: idx for idx, name in enumerate(sorted(all_names))}


def get_or_create_per_parameter_stat_registry(
    model_chunks: Iterable[torch.nn.Module] | torch.nn.Module,
) -> PerParameterStatRegistry:
    """Return a per-model cached parameter-stat registry."""
    unwrapped_model_chunks = unwrap_model(_normalize_model_chunks(model_chunks))
    if not unwrapped_model_chunks:
        raise ValueError("Cannot build a per-parameter stat registry for an empty model list.")

    cache_key = tuple(id(model_chunk) for model_chunk in unwrapped_model_chunks)
    cache_owner = unwrapped_model_chunks[0]
    registry = getattr(cache_owner, "_per_parameter_stat_registry", None)
    if registry is None or registry.cache_key != cache_key:
        registry = PerParameterStatRegistry(unwrapped_model_chunks)
        cache_owner._per_parameter_stat_registry = registry
    return registry


def reduce_l2_norm_by_param(
    registry: PerParameterStatRegistry,
    buckets: Sequence[NamedTensorBucket],
) -> tuple[list[tuple[str, float]], float]:
    """Reduce named tensor L2 norms by parameter name.

    Args:
        registry: Canonical parameter-name registry.
        buckets: Named tensor buckets with the process groups needed to assemble each bucket's
            local squared norms into global per-parameter squared norms.

    Returns:
        A ``(values, aggregate_norm)`` tuple. ``values`` is a list of ``(name, l2_norm)`` tuples
        ordered by canonical parameter index. ``aggregate_norm`` is the L2 norm reconstructed
        from the reduced per-parameter squared norms.
    """
    device = _select_device(buckets)
    norm_2 = torch.zeros(registry.num_params, dtype=torch.float32, device=device)

    for bucket in buckets:
        if len(bucket.names) != len(bucket.tensors):
            raise ValueError(
                f"NamedTensorBucket has {len(bucket.names)} names but {len(bucket.tensors)} tensors."
            )

        bucket_norm_2 = torch.zeros_like(norm_2)
        if bucket.names:
            indices = torch.tensor(
                [registry.name_to_index[name] for name in bucket.names],
                dtype=torch.long,
                device=device,
            )
            per_tensor_norm_2 = _local_l2_norm_squared(bucket.tensors, device)
            bucket_norm_2.index_add_(0, indices, per_tensor_norm_2)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for group in bucket.reduce_groups:
                torch.distributed.all_reduce(
                    bucket_norm_2, op=torch.distributed.ReduceOp.SUM, group=group
                )

        norm_2 += bucket_norm_2

    aggregate_norm = float(norm_2.sum().sqrt())
    norms = norm_2.sqrt().tolist()
    return [(name, norms[idx]) for idx, name in enumerate(registry.index_to_name)], aggregate_norm


def _select_device(buckets: Sequence[NamedTensorBucket]) -> torch.device:
    for bucket in buckets:
        if bucket.tensors:
            return bucket.tensors[0].device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _normalize_model_chunks(
    model_chunks: Iterable[torch.nn.Module] | torch.nn.Module,
) -> list[torch.nn.Module]:
    if isinstance(model_chunks, (list, tuple)):
        return list(model_chunks)
    return [model_chunks]


def _local_l2_norm_squared(tensors: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not tensors:
        return torch.zeros(0, dtype=torch.float32, device=device)

    if device.type == "cuda":
        dummy_overflow_buf = torch.zeros((1,), dtype=torch.int, device=device)
        _, per_tensor_norm = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [list(tensors)],
            True,  # per-parameter norm.
        )
        if per_tensor_norm is not None and per_tensor_norm.numel() == len(tensors):
            return per_tensor_norm.to(dtype=torch.float32, device=device) ** 2

    return torch.stack(
        [tensor.detach().to(dtype=torch.float32, device=device).pow(2).sum() for tensor in tensors]
    )


def _canonical_param_name(
    model_chunk: torch.nn.Module,
    local_name: str,
    param: torch.nn.Parameter,
    num_experts: int | None,
) -> str:
    name = _global_layer_param_name(model_chunk, local_name, param)
    return _global_expert_param_name(name, num_experts)


def _global_layer_param_name(
    model_chunk: torch.nn.Module, local_name: str, param: torch.nn.Parameter
) -> str:
    if "mtp" in local_name or _LAYER_NAME_PATTERN.search(local_name) is None:
        return local_name

    from megatron.core.transformer.transformer_layer import TransformerLayer

    for module in model_chunk.modules():
        if not isinstance(module, TransformerLayer):
            continue
        for module_param in module.parameters():
            if module_param is param:
                return _LAYER_NAME_PATTERN.sub(f"layers.{module.layer_number - 1}", local_name)
    return local_name


def _global_expert_param_name(local_name: str, num_experts: int | None) -> str:
    if not num_experts:
        return local_name

    expert_offset = _get_local_expert_offset(num_experts)
    if expert_offset == 0:
        return local_name

    grouped_match = _GROUPED_EXPERT_PATTERN.match(local_name)
    if grouped_match is not None:
        prefix, local_expert_index, suffix = grouped_match.groups()
        return f"{prefix}{int(local_expert_index) + expert_offset}{suffix}"

    sequential_match = _SEQUENTIAL_EXPERT_PATTERN.match(local_name)
    if sequential_match is not None:
        prefix, local_expert_index, suffix = sequential_match.groups()
        return f"{prefix}{int(local_expert_index) + expert_offset}{suffix}"

    return local_name


def _get_local_expert_offset(num_experts: int) -> int:
    expert_group = parallel_state.get_expert_model_parallel_group(check_initialized=False)
    if expert_group is None:
        return 0
    expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
    if expert_parallel_size <= 1:
        return 0
    local_experts = num_experts // expert_parallel_size
    return parallel_state.get_expert_model_parallel_rank() * local_experts


def _get_num_moe_experts(model_chunks: Sequence[torch.nn.Module]) -> int | None:
    if not model_chunks:
        return None
    config = getattr(model_chunks[0], "config", None)
    return getattr(config, "num_moe_experts", None)
