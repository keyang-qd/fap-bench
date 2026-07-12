from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from fapbench.algorithms import FederatedTrainer, train_centralized, train_local_models
from fapbench.algorithms.common import frame_to_arrays
from fapbench.config import load_config
from fapbench.data import PanelPreprocessor, PanelSpec, load_panel, prepare_panel, split_by_time
from fapbench.evaluation import (
    client_averaged_long_short_metrics,
    client_predictive_metrics,
    long_short_portfolio_metrics,
    partition_heterogeneity_report,
    predictive_metrics,
)
from fapbench.models import AssetPricingMLP
from fapbench.partition import (
    ClientSampler,
    assign_domain_clients,
    assign_iid_clients,
    build_feature_masks,
    temporal_unavailability,
)
from fapbench.utils import ensure_dir, set_seed, write_json


def _model_fn(config: dict[str, Any], input_dim: int):
    model_cfg = config.get("model", {})
    hidden = tuple(int(value) for value in model_cfg.get("hidden_dims", [64, 32]))
    return partial(
        AssetPricingMLP,
        input_dim=input_dim,
        hidden_dims=hidden,
        dropout=float(model_cfg.get("dropout", 0.1)),
        batch_norm=bool(model_cfg.get("batch_norm", False)),
    )


def _prediction_frame(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str,
    entity_col: str,
    client_col: str,
    device: str,
    feature_masks: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    for client_id, group in frame.groupby(client_col, sort=True):
        client_id = str(client_id)
        mask = None if feature_masks is None else feature_masks.get(client_id)
        x, y = frame_to_arrays(group, features, target_col, mask=mask)
        if len(y) != len(group):
            # Preprocessing should make all features finite. Fail explicitly if not.
            raise ValueError("Non-finite evaluation rows remained after preprocessing.")
        with torch.no_grad():
            prediction = model(torch.as_tensor(x, dtype=torch.float32, device=device_obj)).cpu().numpy()
        output = group[[date_col, entity_col, client_col]].copy()
        output[client_col] = output[client_col].astype(str)
        output["y_true"] = y
        output["y_pred"] = prediction
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def _local_prediction_frame(
    models: Mapping[str, torch.nn.Module],
    frame: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str,
    entity_col: str,
    client_col: str,
    device: str,
    feature_masks: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for client_id, group in frame.groupby(client_col, sort=True):
        client_id = str(client_id)
        if client_id not in models:
            continue
        rows.append(
            _prediction_frame(
                models[client_id], group, features, target_col, date_col,
                entity_col, client_col, device, feature_masks
            )
        )
    return pd.concat(rows, ignore_index=True)


def _evaluate_and_save(
    name: str,
    prediction_frame: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    data_cfg = config["data"]
    eval_cfg = config.get("evaluation", {})
    global_metrics = predictive_metrics(prediction_frame)
    client_frame, client_summary = client_predictive_metrics(prediction_frame)
    portfolio_frame, portfolio_metrics = long_short_portfolio_metrics(
        prediction_frame,
        date_col=data_cfg.get("date_col", "date"),
        entity_col=data_cfg.get("entity_col", "permno"),
        quantile=float(eval_cfg.get("portfolio_quantile", 0.1)),
        transaction_cost_bps=float(eval_cfg.get("transaction_cost_bps", 0.0)),
        periods_per_year=int(eval_cfg.get("periods_per_year", 12)),
    )
    client_portfolio_frame, client_portfolio_metrics = client_averaged_long_short_metrics(
        prediction_frame,
        client_col="client_id",
        date_col=data_cfg.get("date_col", "date"),
        entity_col=data_cfg.get("entity_col", "permno"),
        quantile=float(eval_cfg.get("portfolio_quantile", 0.1)),
        transaction_cost_bps=float(eval_cfg.get("transaction_cost_bps", 0.0)),
        periods_per_year=int(eval_cfg.get("periods_per_year", 12)),
    )
    prediction_frame.to_parquet(output_dir / f"predictions_{name}.parquet", index=False)
    client_frame.to_csv(output_dir / f"client_metrics_{name}.csv", index=False)
    portfolio_frame.to_csv(output_dir / f"portfolio_returns_{name}.csv", index=False)
    client_portfolio_frame.to_csv(output_dir / f"client_portfolio_returns_{name}.csv", index=False)
    prefixed_client_portfolio = {f"client_portfolio_{key}": value for key, value in client_portfolio_metrics.items()}
    result = {**global_metrics, **client_summary, **portfolio_metrics, **prefixed_client_portfolio}
    write_json(result, output_dir / f"metrics_{name}.json")
    return result


def run(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    torch.set_num_threads(int(config.get("torch_num_threads", 1)))
    output_dir = ensure_dir(config.get("output_dir", "outputs/run"))

    data_cfg = config["data"]
    raw = load_panel(data_cfg["path"])
    features_cfg = data_cfg.get("feature_columns")
    spec = PanelSpec(
        date_col=data_cfg.get("date_col", "date"),
        entity_col=data_cfg.get("entity_col", "permno"),
        target_col=data_cfg.get("target_col", "retadj_1mn"),
        target_shift_months=int(data_cfg.get("target_shift_months", 0)),
        feature_columns=tuple(features_cfg) if features_cfg else None,
        excluded_columns=tuple(data_cfg.get("excluded_columns", [])),
        min_feature_coverage=float(data_cfg.get("min_feature_coverage", 0.5)),
    )
    panel, features = prepare_panel(raw, spec)
    split_cfg = config.get("split", {})
    panel = split_by_time(
        panel,
        spec.date_col,
        train_end=split_cfg.get("train_end"),
        val_end=split_cfg.get("val_end"),
        ratios=split_cfg.get("ratios", [0.6, 0.2, 0.2]),
    )
    preprocessor = PanelPreprocessor(features, scaling=data_cfg.get("scaling", "global"))
    panel = preprocessor.fit_transform_splits(panel, date_col=spec.date_col)

    partition_cfg = config.get("partition", {})
    partition_type = partition_cfg.get("type", "iid").lower()
    if partition_type == "iid":
        panel = assign_iid_clients(
            panel, spec.entity_col, int(partition_cfg.get("num_clients", 5)), seed=seed
        )
    elif partition_type == "domain":
        panel = assign_domain_clients(
            panel,
            domain_col=partition_cfg["domain_col"],
            max_clients=partition_cfg.get("max_clients"),
            continuous_bins=partition_cfg.get("continuous_bins"),
            entity_col=spec.entity_col,
            persistent_entity=bool(partition_cfg.get("persistent_entity", True)),
        )
    else:
        raise ValueError("partition.type must be iid or domain.")

    clients = sorted(panel["client_id"].astype(str).unique().tolist())
    feature_cfg = config.get("feature_heterogeneity", {})
    feature_masks = None
    if bool(feature_cfg.get("enabled", False)):
        feature_masks = build_feature_masks(
            features,
            clients,
            common_fraction=float(feature_cfg.get("common_fraction", 1.0)),
            private_fraction=float(feature_cfg.get("private_fraction", 0.5)),
            seed=seed,
        )

    fed_cfg = config.get("federated", {})
    rounds = int(fed_cfg.get("rounds", 20))
    participation_rate = float(fed_cfg.get("participation_rate", 1.0))
    sampler = ClientSampler(clients, participation_rate, seed)
    schedule = sampler.schedule(rounds)
    temporal_h = temporal_unavailability(schedule, clients)

    heterogeneity = partition_heterogeneity_report(
        panel.loc[panel["split"] == "train"],
        features,
        spec.target_col,
        feature_masks=feature_masks,
        temporal_unavailability_value=temporal_h,
        seed=seed,
    )
    heterogeneity["feature_columns"] = features
    write_json(heterogeneity, output_dir / "heterogeneity.json")
    panel[[spec.date_col, spec.entity_col, "client_id", "split"]].to_parquet(
        output_dir / "partition_assignments.parquet", index=False
    )
    if feature_masks:
        mask_frame = pd.DataFrame(feature_masks, index=features).T
        mask_frame.index.name = "client_id"
        mask_frame.to_csv(output_dir / "feature_masks.csv")

    train = panel.loc[panel["split"] == "train"].copy()
    test = panel.loc[panel["split"] == "test"].copy()
    model_fn = _model_fn(config, len(features))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, Any]] = []

    baseline_cfg = config.get("baselines", {})
    if bool(baseline_cfg.get("centralized", True)):
        set_seed(seed)
        centralized = train_centralized(
            train, model_fn, features, spec.target_col,
            epochs=int(baseline_cfg.get("epochs", 5)),
            batch_size=int(fed_cfg.get("batch_size", 512)),
            learning_rate=float(fed_cfg.get("client_lr", 1e-3)),
            weight_decay=float(fed_cfg.get("weight_decay", 0.0)),
            device=device,
        )
        predictions = _prediction_frame(
            centralized, test, features, spec.target_col, spec.date_col,
            spec.entity_col, "client_id", device, feature_masks=None
        )
        metrics = _evaluate_and_save("centralized", predictions, output_dir, config)
        results.append({"method": "centralized", **metrics})

    if bool(baseline_cfg.get("local", True)):
        set_seed(seed)
        local_models = train_local_models(
            train, model_fn, features, spec.target_col,
            feature_masks=feature_masks,
            epochs=int(baseline_cfg.get("epochs", 5)),
            batch_size=int(fed_cfg.get("batch_size", 512)),
            learning_rate=float(fed_cfg.get("client_lr", 1e-3)),
            weight_decay=float(fed_cfg.get("weight_decay", 0.0)),
            device=device,
        )
        predictions = _local_prediction_frame(
            local_models, test, features, spec.target_col, spec.date_col,
            spec.entity_col, "client_id", device, feature_masks
        )
        metrics = _evaluate_and_save("local", predictions, output_dir, config)
        results.append({"method": "local", **metrics})

    for algorithm in fed_cfg.get("algorithms", ["fedavg", "fedprox", "scaffold", "fedadam"]):
        set_seed(seed)
        trainer = FederatedTrainer(
            algorithm=algorithm,
            model_fn=model_fn,
            feature_columns=features,
            target_col=spec.target_col,
            rounds=rounds,
            local_epochs=int(fed_cfg.get("local_epochs", 1)),
            batch_size=int(fed_cfg.get("batch_size", 512)),
            client_lr=float(fed_cfg.get("client_lr", 1e-3)),
            weight_decay=float(fed_cfg.get("weight_decay", 0.0)),
            proximal_mu=float(fed_cfg.get("proximal_mu", 0.01)),
            participation_rate=participation_rate,
            seed=seed,
            device=device,
            server_lr=float(fed_cfg.get("server_lr", 1e-2)),
            server_beta1=float(fed_cfg.get("server_beta1", 0.9)),
            server_beta2=float(fed_cfg.get("server_beta2", 0.99)),
            server_tau=float(fed_cfg.get("server_tau", 1e-8)),
        )
        model = trainer.fit(train, feature_masks=feature_masks)
        trainer.history.to_frame().to_csv(output_dir / f"training_{algorithm}.csv", index=False)
        predictions = _prediction_frame(
            model, test, features, spec.target_col, spec.date_col,
            spec.entity_col, "client_id", device, feature_masks
        )
        metrics = _evaluate_and_save(algorithm, predictions, output_dir, config)
        metrics["communicated_parameters"] = int(
            trainer.history.to_frame()["communicated_parameters"].sum()
        )
        metrics["communicated_bytes_fp32"] = 4 * metrics["communicated_parameters"]
        results.append({"method": algorithm, **metrics})

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(output_dir / "benchmark_results.csv", index=False)
    write_json(config, output_dir / "resolved_config.json")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FAP-Bench experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    args = parser.parse_args()
    output = run(load_config(args.config))
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    main()
