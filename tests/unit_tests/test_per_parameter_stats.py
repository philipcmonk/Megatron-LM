# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.per_parameter_stats import (
    NamedTensorBucket,
    PerParameterStatRegistry,
    get_or_create_per_parameter_stat_registry,
    reduce_raw_moments_by_param,
)


class TwoParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.zeros(1))


def test_reduce_raw_moments_by_param_on_cpu():
    registry = PerParameterStatRegistry(TwoParamModel())

    values, aggregate_moments = reduce_raw_moments_by_param(
        registry,
        [
            NamedTensorBucket(
                names=["a", "a", "b"],
                tensors=[
                    torch.tensor([1.0, 2.0]),
                    torch.tensor([2.0, 4.0]),
                    torch.tensor([3.0]),
                ],
            )
        ],
    )

    assert dict(values) == {
        "a": {
            "count": pytest.approx(4.0),
            "sum_1": pytest.approx(9.0),
            "sum_2": pytest.approx(21.0),
            "sum_3": pytest.approx(81.0),
            "sum_4": pytest.approx(321.0),
        },
        "b": {
            "count": pytest.approx(1.0),
            "sum_1": pytest.approx(3.0),
            "sum_2": pytest.approx(9.0),
            "sum_3": pytest.approx(27.0),
            "sum_4": pytest.approx(81.0),
        },
    }
    assert aggregate_moments == {
        "count": pytest.approx(5.0),
        "sum_1": pytest.approx(12.0),
        "sum_2": pytest.approx(30.0),
        "sum_3": pytest.approx(108.0),
        "sum_4": pytest.approx(402.0),
    }


def test_reduce_raw_moments_by_param_rejects_mismatched_names_and_tensors():
    registry = PerParameterStatRegistry(TwoParamModel())

    with pytest.raises(ValueError, match="names but"):
        reduce_raw_moments_by_param(
            registry,
            [NamedTensorBucket(names=["a"], tensors=[torch.tensor([1.0]), torch.tensor([2.0])])],
        )


def test_registry_cache_is_per_model_identity():
    first_model = TwoParamModel()
    second_model = TwoParamModel()

    first_registry = get_or_create_per_parameter_stat_registry(first_model)
    assert get_or_create_per_parameter_stat_registry(first_model) is first_registry
    assert get_or_create_per_parameter_stat_registry(second_model) is not first_registry
