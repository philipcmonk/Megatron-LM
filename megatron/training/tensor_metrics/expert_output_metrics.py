# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor metrics for routed MoE expert outputs."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .core import (
    AllGather,
    AllReduce,
    CollectiveRequest,
    CollectiveStage,
    MetricResult,
    MetricSite,
    MetricStep,
    MetricTensor,
    Shard,
    TensorMetric,
)
from .definitions import LayerL2NormMetric

__all__ = ["LayerExpertOutputL2StatsMetric"]


@dataclass(frozen=True)
class _ExpertOutputContinuation:
    layer: str
    operation: str
    remaining_sum_axes: tuple[str, ...]
    gather_ep: bool


class LayerExpertOutputL2StatsMetric(TensorMetric):
    """Report the maximum and mean routed-expert output L2 norms in each MoE layer.

    The observed tensor contains one sum-of-squares value per local expert. These compact values
    are accumulated across microbatches and ranks that process different activation populations,
    then gathered across expert-parallel ranks to form one value per global expert.
    """

    name = "layer-expert-output-l2-stats"
    source_kinds = frozenset({"expert_output_squares"})

    def accepts(self, site: MetricSite) -> bool:
        """Select compact expert-output observations belonging to numbered layers."""
        return super().accepts(site) and LayerL2NormMetric._site_layer_label(site) is not None

    def prepare(self, values: Sequence[MetricTensor]) -> list[MetricTensor]:
        """Copy compact per-expert sum-of-squares vectors for deferred communication."""
        prepared = []
        for value in values:
            if value.tensor.ndim != 1:
                raise ValueError("Expert-output metrics require one value per local expert.")
            if not value.tensor.is_floating_point():
                raise ValueError("Expert-output sum-of-squares values must be floating point.")
            prepared.append(value.with_tensor(value.tensor.float().clone()))
        return prepared

    def start(self, values: Sequence[MetricTensor]) -> list[MetricStep]:
        """Start one exact expert-output reduction for each observed MoE layer."""
        values_by_layer: dict[str, list[MetricTensor]] = {}
        for value in values:
            layer = LayerL2NormMetric._layer_label(value)
            if layer is None:
                raise ValueError("An expert-output metric tensor must identify one logical layer.")
            values_by_layer.setdefault(layer, []).append(value)
        return [
            step
            for layer, layer_values in values_by_layer.items()
            for step in self._start_layer(layer, layer_values)
        ]

    def resume(self, values: Sequence[MetricTensor], continuation: object) -> MetricStep:
        """Continue population reductions, gather global experts, or finalize layer statistics."""
        if not isinstance(continuation, _ExpertOutputContinuation) or len(values) != 1:
            raise ValueError("Expert-output metric collective completion is invalid.")
        value = values[0]
        if continuation.remaining_sum_axes:
            axis = continuation.remaining_sum_axes[0]
            return CollectiveStage(
                (CollectiveRequest(value, axis, AllReduce(torch.distributed.ReduceOp.SUM)),),
                _ExpertOutputContinuation(
                    continuation.layer,
                    continuation.operation,
                    continuation.remaining_sum_axes[1:],
                    continuation.gather_ep,
                ),
            )
        if continuation.gather_ep:
            return CollectiveStage(
                (CollectiveRequest(value, "ep", AllGather(0)),),
                _ExpertOutputContinuation(
                    continuation.layer, continuation.operation, (), False
                ),
            )
        return self._finish(value.tensor, continuation.layer, continuation.operation)

    def _start_layer(self, layer: str, values: Sequence[MetricTensor]) -> list[MetricStep]:
        relations = values[0].rank_relations
        if any(value.rank_relations != relations for value in values[1:]):
            raise ValueError("Expert-output observations for one layer require identical layouts.")
        shape = values[0].tensor.shape
        if any(value.tensor.shape != shape for value in values[1:]):
            raise ValueError("Expert-output observations for one layer require equal expert counts.")

        local_squares = torch.stack(tuple(value.tensor for value in values)).sum(dim=0)
        sites = tuple(site for value in values for site in value.sites)
        value = MetricTensor(local_squares, sites, relations)
        sum_axes = tuple(
            relation.axis
            for relation in relations
            if relation.axis != "ep" and isinstance(relation.placement, Shard)
        )
        ep_relation = value.relation("ep")
        gather_ep = isinstance(ep_relation.placement, Shard)
        return [
            self._start_operation(value, layer, operation, sum_axes, gather_ep)
            for operation in ("max", "mean")
        ]

    def _start_operation(
        self,
        value: MetricTensor,
        layer: str,
        operation: str,
        sum_axes: tuple[str, ...],
        gather_ep: bool,
    ) -> MetricStep:
        if sum_axes:
            return CollectiveStage(
                (
                    CollectiveRequest(
                        value, sum_axes[0], AllReduce(torch.distributed.ReduceOp.SUM)
                    ),
                ),
                _ExpertOutputContinuation(layer, operation, sum_axes[1:], gather_ep),
            )
        if gather_ep:
            return CollectiveStage(
                (CollectiveRequest(value, "ep", AllGather(0)),),
                _ExpertOutputContinuation(layer, operation, (), False),
            )
        return self._finish(value.tensor, layer, operation)

    @staticmethod
    def _finish(expert_squares: torch.Tensor, layer: str, operation: str) -> MetricResult:
        if expert_squares.numel() == 0:
            raise ValueError("Expert-output metrics require at least one global expert.")
        expert_norms = expert_squares.clamp_min(0).sqrt()
        result = expert_norms.amax() if operation == "max" else expert_norms.mean()
        return MetricResult(result, f"{layer}/{operation}")
