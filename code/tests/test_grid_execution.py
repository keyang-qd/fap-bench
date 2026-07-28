from __future__ import annotations

import inspect
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from fapbench.grid import _completed_run_files
from fapbench.runner import run


def test_runner_accepts_preloaded_raw_panel() -> None:
    assert "preloaded_raw_panel" in inspect.signature(run).parameters


def test_completed_run_detection(tmp_path: Path) -> None:
    assert not _completed_run_files(tmp_path)
    pd.DataFrame([{"method": "fedavg", "mse": 1.0}]).to_csv(
        tmp_path / "benchmark_results.csv", index=False
    )
    (tmp_path / "heterogeneity.json").write_text(
        json.dumps({"label_wasserstein_mean": 0.1}), encoding="utf-8"
    )
    assert _completed_run_files(tmp_path)


def test_more_local_steps_holds_total_local_epoch_budget() -> None:
    path = Path(__file__).parents[1] / "configs" / "icaif_core.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenario = next(
        item for item in config["scenarios"] if item["name"] == "ff12_more_local_steps"
    )
    fed = scenario["overrides"]["federated"]
    assert fed["rounds"] * fed["local_epochs"] == 50


def test_collect_run_results_avoids_fragmentation(tmp_path: Path) -> None:
    from fapbench.grid import _collect_run_results

    pd.DataFrame([
        {"method": "fedavg", **{f"metric_{i}": float(i) for i in range(120)}}
    ]).to_csv(tmp_path / "benchmark_results.csv", index=False)
    (tmp_path / "heterogeneity.json").write_text(
        json.dumps({f"h{i}": float(i) for i in range(30)}), encoding="utf-8"
    )
    result = _collect_run_results(tmp_path, scenario="s", seed=11)
    assert result.loc[0, "scenario"] == "s"
    assert result.loc[0, "seed"] == 11
    assert result.loc[0, "heterogeneity_h29"] == 29.0


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    pd.DataFrame([{"method": "fedadam", "mse": 1.0}]).to_csv(
        tmp_path / "benchmark_results.csv", index=False
    )
    (tmp_path / "heterogeneity.json").write_text("{}", encoding="utf-8")
    completed = {"federated": {"server_lr": 0.01}, "output_dir": "old"}
    (tmp_path / "resolved_config.json").write_text(
        json.dumps(completed), encoding="utf-8"
    )
    expected = {"federated": {"server_lr": 0.003}, "output_dir": "new"}
    assert not _completed_run_files(tmp_path, expected_config=expected)
    expected["federated"]["server_lr"] = 0.01
    assert _completed_run_files(tmp_path, expected_config=expected)


def test_resume_accepts_yaml_date_after_json_roundtrip(tmp_path: Path) -> None:
    pd.DataFrame([{"method": "fedadam", "mse": 1.0}]).to_csv(
        tmp_path / "benchmark_results.csv", index=False
    )
    (tmp_path / "heterogeneity.json").write_text("{}", encoding="utf-8")
    completed = {
        "split": {"train_end": "2014-12-31", "val_end": "2018-12-31"},
        "data": {"path": "/tmp/panel.parquet"},
        "output_dir": "old",
    }
    (tmp_path / "resolved_config.json").write_text(
        json.dumps(completed), encoding="utf-8"
    )
    expected = {
        "split": {
            "train_end": date(2014, 12, 31),
            "val_end": date(2018, 12, 31),
        },
        "data": {"path": Path("/tmp/panel.parquet")},
        "output_dir": "new",
    }
    assert _completed_run_files(tmp_path, expected_config=expected)
