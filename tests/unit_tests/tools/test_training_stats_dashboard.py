# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import json
from pathlib import Path

import pytest

_TRAINING_STATS_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "training_stats_dashboard" / "training_stats.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "training_stats_dashboard_utils", _TRAINING_STATS_PATH
)
training_stats = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(training_stats)


def _metric_record(iteration=1):
    return {
        "iter": iteration,
        "stat": "param_raw_moments_by_param",
        "values": {
            "decoder.layers.0.mlp.weight": {
                "count": 1,
                "sum_1": 1,
                "sum_2": 1,
                "sum_3": 1,
                "sum_4": 1,
            }
        },
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "decoder.layers.3.self_attention.linear_qkv.weight",
            {
                "layer_index": 3,
                "layer_stack": "decoder.layers",
                "layer_label": "decoder.layers.3",
                "param_type": "self_attention.linear_qkv.weight",
                "param_family": "self_attention",
                "param_kind": "weight",
                "module_type": "self_attention.linear_qkv",
            },
        ),
        (
            "encoder.layers.2.mlp.linear_fc1.weight",
            {
                "layer_index": 2,
                "layer_stack": "encoder.layers",
                "layer_label": "encoder.layers.2",
                "param_type": "mlp.linear_fc1.weight",
                "param_family": "mlp",
                "param_kind": "weight",
                "module_type": "mlp.linear_fc1",
            },
        ),
        (
            "decoder.layers.3.mlp.experts.local_experts.7.linear_fc1.weight",
            {
                "layer_index": 3,
                "layer_stack": "decoder.layers",
                "layer_label": "decoder.layers.3",
                "param_type": "mlp.experts.local_experts.7.linear_fc1.weight",
                "param_family": "mlp",
                "param_kind": "weight",
                "module_type": "mlp.experts.local_experts.7.linear_fc1",
            },
        ),
        (
            "mtp.layers.0.mtp_model_layer.layers.1.mlp.experts.linear_fc1/output0",
            {
                "layer_index": 1,
                "layer_stack": "mtp.layers.0.mtp_model_layer.layers",
                "layer_label": "mtp.layers.0.mtp_model_layer.layers.1",
                "param_type": "mlp.experts.linear_fc1.output0",
                "param_family": "mlp",
                "param_kind": "output0",
                "module_type": "mlp.experts.linear_fc1",
                "tensor_endpoint": "output0",
                "tensor_role": "output",
                "tensor_index": 0,
            },
        ),
        (
            "vision.encoder.layer1.0.conv1.weight",
            {
                "layer_index": 0,
                "layer_stack": "vision.encoder.layer1",
                "layer_label": "vision.encoder.layer1.0",
                "param_type": "conv1.weight",
                "param_family": "conv1",
                "param_kind": "weight",
                "module_type": "conv1",
            },
        ),
    ],
)
def test_infer_param_metadata_extracts_layer_context(name, expected):
    metadata = training_stats.infer_param_metadata(name)

    assert metadata["is_layered"]
    for key, value in expected.items():
        assert metadata[key] == value


@pytest.mark.parametrize(
    "name",
    [
        "mlp.experts.local_experts.2.weight",
        "mlp.experts.local_experts.3.linear_fc1.weight",
        "embedding.word_embeddings.weight",
        "output_layer.weight",
        "router.experts.0.weight",
    ],
)
def test_infer_param_metadata_does_not_treat_non_layer_indices_as_layers(name):
    metadata = training_stats.infer_param_metadata(name)

    assert not metadata["is_layered"]
    assert metadata["layer_index"] is None
    assert metadata["layer_stack"] == "(global)"


def test_iter_metric_file_records_ignores_incomplete_final_line(tmp_path):
    path = tmp_path / "rank0.jsonl"
    path.write_text(json.dumps(_metric_record()) + '\n{"iter": 2, "values":')

    records = list(training_stats.iter_metric_file_records(path))

    assert len(records) == 1
    assert records[0]["iter"] == 1


def test_iter_metric_file_records_accepts_complete_final_line_without_newline(tmp_path):
    path = tmp_path / "rank0.jsonl"
    path.write_text(json.dumps(_metric_record()))

    records = list(training_stats.iter_metric_file_records(path))

    assert len(records) == 1
    assert records[0]["iter"] == 1


def test_iter_metric_file_records_rejects_malformed_completed_line(tmp_path):
    path = tmp_path / "rank0.jsonl"
    path.write_text('{"iter": 1, "values":}\n')

    with pytest.raises(json.JSONDecodeError):
        list(training_stats.iter_metric_file_records(path))
