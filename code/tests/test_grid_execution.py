from __future__ import annotations

import inspect
import json
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
    path = Path(__file__).parents[1] / "configs" / "crsp_compustat_profitability_grid.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenario = next(
        item for item in config["scenarios"] if item["name"] == "ff12_more_local_steps"
    )
    fed = scenario["overrides"]["federated"]
    assert fed["rounds"] * fed["local_epochs"] == 50
