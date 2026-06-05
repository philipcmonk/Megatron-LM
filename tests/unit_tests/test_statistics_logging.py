# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

from megatron.training.statistics_logging import save_params_norm_by_param


def _read_records(filepath):
    return [json.loads(line) for line in filepath.read_text().strip().split("\n")]


class TestSaveParamsNormByParam:
    def test_creates_jsonl(self, tmp_path):
        save_params_norm_by_param(
            str(tmp_path),
            iteration=100,
            consumed_train_samples=8192,
            params_norm_by_param=[
                ("decoder.layers.0.self_attention.linear_qkv.weight", 1.5),
                ("decoder.layers.0.mlp.linear_fc1.weight", 2.25),
            ],
            rank=7,
        )

        filepath = tmp_path / "training_stats" / "params_norm_by_param" / "rank7.jsonl"
        assert filepath.exists()

        records = _read_records(filepath)
        assert records == [
            {
                "iter": 100,
                "consumed_train_samples": 8192,
                "stat": "params_norm_by_param",
                "norm_type": "l2",
                "values": {
                    "decoder.layers.0.self_attention.linear_qkv.weight": 1.5,
                    "decoder.layers.0.mlp.linear_fc1.weight": 2.25,
                },
            }
        ]

    def test_appends_across_calls(self, tmp_path):
        save_params_norm_by_param(
            str(tmp_path),
            iteration=100,
            consumed_train_samples=8192,
            params_norm_by_param=[("layer.weight", 1.0)],
            rank=0,
        )
        save_params_norm_by_param(
            str(tmp_path),
            iteration=200,
            consumed_train_samples=16384,
            params_norm_by_param=[("layer.weight", 2.0)],
            rank=0,
        )

        filepath = tmp_path / "training_stats" / "params_norm_by_param" / "rank0.jsonl"
        records = _read_records(filepath)
        assert len(records) == 2
        assert records[0]["iter"] == 100
        assert records[0]["values"] == {"layer.weight": 1.0}
        assert records[1]["iter"] == 200
        assert records[1]["values"] == {"layer.weight": 2.0}

    def test_empty_values_do_not_create_file(self, tmp_path):
        save_params_norm_by_param(
            str(tmp_path),
            iteration=100,
            consumed_train_samples=8192,
            params_norm_by_param=[],
            rank=0,
        )

        filepath = tmp_path / "training_stats" / "params_norm_by_param" / "rank0.jsonl"
        assert not filepath.exists()
