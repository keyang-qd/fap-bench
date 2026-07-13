from __future__ import annotations

import argparse
import logging
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from fapbench.algorithms import FederatedTrainer, train_centralized, train_local_models
from fapbench.algorithms.common import frame_to_arrays
from fapbench.config import load_config
from fapbench.data import (
    PanelPreprocessor,
    PanelSpec,
    load_panel,
    prepare_panel,
    split_by_time,
    validate_crsp_compustat_profitability_panel,
)
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
    add_ff12_column,
    apply_feature_masks_to_frame,
    assign_domain_clients,
    assign_iid_clients,
    build_feature_masks,
    temporal_unavailability,
)
from fapbench.utils import (
    configure_logger,
    ensure_dir,
    format_seconds,
    log_frame,
    log_stage,
    set_seed,
    write_json,
)


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


def _resolve_data_path(config: dict[str, Any]) -> Path:
    path = Path(config["data"]["path"])
    if path.is_absolute():
        return path
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    config_path = config.get("_config_path")
    if config_path:
        return (Path(config_path).parent / path).resolve()
    return cwd_candidate


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
    target_date_col: str = "target_date",
    logger: logging.Logger | None = None,
    log_name: str = "prediction",
    client_log_interval: int = 1,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    groups = list(frame.groupby(client_col, sort=True))
    started = time.perf_counter()
    if logger is not None:
        logger.info(
            "%s | start | rows=%s | clients=%s | features=%s | device=%s",
            log_name,
            f"{len(frame):,}",
            len(groups),
            len(features),
            device_obj,
        )

    for client_index, (client_id, group) in enumerate(groups, start=1):
        client_id = str(client_id)
        if feature_masks is not None and client_id not in feature_masks:
            raise KeyError(f"No feature mask is defined for client {client_id!r}.")
        mask = None if feature_masks is None else feature_masks[client_id]
        should_log = (
            logger is not None
            and client_log_interval > 0
            and (
                client_index == 1
                or client_index == len(groups)
                or client_index % client_log_interval == 0
            )
        )
        if should_log:
            logger.info(
                "%s | client=%s/%s | client_id=%s | rows=%s",
                log_name,
                client_index,
                len(groups),
                client_id,
                f"{len(group):,}",
            )
        x, y = frame_to_arrays(group, features, target_col, mask=mask)
        if len(y) != len(group):
            raise ValueError("Non-finite evaluation rows remained after preprocessing.")
        with torch.no_grad():
            prediction = (
                model(torch.as_tensor(x, dtype=torch.float32, device=device_obj))
                .cpu()
                .numpy()
            )
        output_columns = [date_col, entity_col, client_col]
        if target_date_col in group.columns and target_date_col not in output_columns:
            output_columns.append(target_date_col)
        output = group[output_columns].copy()
        output[client_col] = output[client_col].astype(str)
        output["y_true"] = y
        output["y_pred"] = prediction
        rows.append(output)
    if not rows:
        raise ValueError("No prediction rows were generated.")
    result = pd.concat(rows, ignore_index=True)
    if logger is not None:
        logger.info(
            "%s | done | rows=%s | elapsed=%s",
            log_name,
            f"{len(result):,}",
            format_seconds(time.perf_counter() - started),
        )
    return result


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
    target_date_col: str = "target_date",
    logger: logging.Logger | None = None,
    client_log_interval: int = 1,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    missing_clients: list[str] = []
    groups = list(frame.groupby(client_col, sort=True))
    started = time.perf_counter()
    if logger is not None:
        logger.info(
            "local prediction | start | rows=%s | clients=%s",
            f"{len(frame):,}",
            len(groups),
        )
    for client_index, (client_id, group) in enumerate(groups, start=1):
        client_id = str(client_id)
        if client_id not in models:
            missing_clients.append(client_id)
            continue
        rows.append(
            _prediction_frame(
                models[client_id],
                group,
                features,
                target_col,
                date_col,
                entity_col,
                client_col,
                device,
                feature_masks,
                target_date_col,
                logger=logger,
                log_name=f"local prediction client={client_id}",
                client_log_interval=0,
            )
        )
        if (
            logger is not None
            and client_log_interval > 0
            and (
                client_index == 1
                or client_index == len(groups)
                or client_index % client_log_interval == 0
            )
        ):
            logger.info(
                "local prediction | progress=%s/%s | client_id=%s",
                client_index,
                len(groups),
                client_id,
            )
    if missing_clients:
        raise ValueError(
            "Local evaluation contains clients with no training model: "
            f"{sorted(missing_clients)}"
        )
    if not rows:
        raise ValueError("No local prediction rows were generated.")
    result = pd.concat(rows, ignore_index=True)
    if logger is not None:
        logger.info(
            "local prediction | done | rows=%s | elapsed=%s",
            f"{len(result):,}",
            format_seconds(time.perf_counter() - started),
        )
    return result


def _evaluation_date_col(
    prediction_frame: pd.DataFrame, config: dict[str, Any]
) -> str:
    data_cfg = config["data"]
    eval_cfg = config.get("evaluation", {})
    requested = eval_cfg.get("date_col")
    if requested:
        if requested not in prediction_frame.columns:
            raise KeyError(f"evaluation.date_col={requested!r} is absent.")
        return str(requested)
    target_date_col = data_cfg.get("target_date_col", "target_date")
    if target_date_col in prediction_frame.columns:
        return target_date_col
    return data_cfg.get("date_col", "date")


def _evaluate_and_save(
    name: str,
    prediction_frame: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    data_cfg = config["data"]
    eval_cfg = config.get("evaluation", {})
    evaluation_date = _evaluation_date_col(prediction_frame, config)
    started = time.perf_counter()
    if logger is not None:
        logger.info(
            "%s | evaluation start | rows=%s | date_col=%s",
            name,
            f"{len(prediction_frame):,}",
            evaluation_date,
        )
    global_metrics = predictive_metrics(
        prediction_frame, date_col=evaluation_date
    )
    client_frame, client_summary = client_predictive_metrics(
        prediction_frame, date_col=evaluation_date
    )
    portfolio_frame, portfolio_metrics = long_short_portfolio_metrics(
        prediction_frame,
        date_col=evaluation_date,
        entity_col=data_cfg.get("entity_col", "permno"),
        quantile=float(eval_cfg.get("portfolio_quantile", 0.1)),
        transaction_cost_bps=float(eval_cfg.get("transaction_cost_bps", 0.0)),
        periods_per_year=int(eval_cfg.get("periods_per_year", 12)),
    )
    client_portfolio_frame, client_portfolio_metrics = (
        client_averaged_long_short_metrics(
            prediction_frame,
            client_col="client_id",
            date_col=evaluation_date,
            entity_col=data_cfg.get("entity_col", "permno"),
            quantile=float(eval_cfg.get("portfolio_quantile", 0.1)),
            transaction_cost_bps=float(
                eval_cfg.get("transaction_cost_bps", 0.0)
            ),
            periods_per_year=int(eval_cfg.get("periods_per_year", 12)),
        )
    )
    prediction_frame.to_parquet(
        output_dir / f"predictions_{name}.parquet", index=False
    )
    client_frame.to_csv(
        output_dir / f"client_metrics_{name}.csv", index=False
    )
    portfolio_frame.to_csv(
        output_dir / f"portfolio_returns_{name}.csv", index=False
    )
    client_portfolio_frame.to_csv(
        output_dir / f"client_portfolio_returns_{name}.csv", index=False
    )
    prefixed_client_portfolio = {
        f"client_portfolio_{key}": value
        for key, value in client_portfolio_metrics.items()
    }
    result = {
        **global_metrics,
        **client_summary,
        **portfolio_metrics,
        **prefixed_client_portfolio,
        "evaluation_date_col": evaluation_date,
    }
    write_json(result, output_dir / f"metrics_{name}.json")
    if logger is not None:
        highlights = []
        for key in ["mse", "oos_r2_zero", "ic", "rank_ic", "sharpe", "annualized_return"]:
            value = result.get(key)
            if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                highlights.append(f"{key}={float(value):.6f}")
        logger.info(
            "%s | evaluation done | %s | elapsed=%s",
            name,
            " | ".join(highlights) if highlights else "metrics saved",
            format_seconds(time.perf_counter() - started),
        )
    return result


def run(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    torch.set_num_threads(int(config.get("torch_num_threads", 1)))
    output_dir = ensure_dir(config.get("output_dir", "outputs/run"))
    logging_cfg = config.get("logging", {})
    logger = configure_logger(
        output_dir,
        name=f"fapbench.{abs(hash(str(output_dir.resolve())))}",
        level=logging_cfg.get("level", "INFO"),
        filename=logging_cfg.get("file", "run.log"),
        console=bool(logging_cfg.get("console", True)),
        overwrite=bool(logging_cfg.get("overwrite", False)),
    )
    run_started = time.perf_counter()
    data_cfg = config["data"]
    data_path = _resolve_data_path(config)
    device = config.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(
        "FAP-Bench run start | config=%s | output=%s | seed=%s | device=%s | torch_threads=%s",
        config.get("_config_path", "<in-memory>"),
        output_dir.resolve(),
        seed,
        device,
        torch.get_num_threads(),
    )
    logger.info(
        "Data path | %s | file_size=%.1f MB",
        data_path,
        data_path.stat().st_size / (1024**2) if data_path.exists() else float("nan"),
    )

    try:
        with log_stage(logger, "Load panel"):
            raw = load_panel(data_path)
            log_frame(
                logger,
                "Raw panel",
                raw,
                date_col=data_cfg.get("date_col", "date"),
                entity_col=data_cfg.get("entity_col", "permno"),
            )

        if data_cfg.get("source_profile") in {
            "crsp_compustat_profitability_v1",
            "crsp_compustat_profitability_v2",
        }:
            with log_stage(logger, "Validate source panel"):
                source_report = validate_crsp_compustat_profitability_panel(raw)
                write_json(source_report, output_dir / "source_validation.json")
                logger.info(
                    "Source validation | target_alignment=%s | duplicate_entity_month_rows=%s | has_sic=%s",
                    source_report.get("target_alignment_status"),
                    source_report.get("duplicate_entity_month_rows"),
                    source_report.get("has_sic"),
                )
                recommended_shift = source_report.get("recommended_target_shift_months")
                configured_shift = int(data_cfg.get("target_shift_months", 0))
                if recommended_shift is not None and configured_shift != recommended_shift:
                    raise ValueError(
                        "Configured target_shift_months conflicts with source validation: "
                        f"configured={configured_shift}, recommended={recommended_shift}."
                    )

        features_cfg = data_cfg.get("feature_columns")
        spec = PanelSpec(
            date_col=data_cfg.get("date_col", "date"),
            entity_col=data_cfg.get("entity_col", "permno"),
            target_col=data_cfg.get("target_col", "retadj_1mn"),
            target_shift_months=int(data_cfg.get("target_shift_months", 0)),
            feature_columns=tuple(features_cfg) if features_cfg else None,
            excluded_columns=tuple(data_cfg.get("excluded_columns", [])),
            min_feature_coverage=float(
                data_cfg.get("min_feature_coverage", 0.5)
            ),
            target_date_col=data_cfg.get("target_date_col", "target_date"),
            target_period_col=data_cfg.get(
                "target_period_col", "target_year_month"
            ),
            require_unique_entity_month=bool(
                data_cfg.get("require_unique_entity_month", True)
            ),
            validate_target_alignment=bool(
                data_cfg.get("validate_target_alignment", True)
            ),
        )
        with log_stage(logger, "Prepare panel and align target"):
            panel, features = prepare_panel(raw, spec)
            logger.info("Model features | count=%s | %s", len(features), ", ".join(features))
            log_frame(logger, "Prepared panel", panel, date_col=spec.date_col, entity_col=spec.entity_col)
        del raw

        split_cfg = config.get("split", {})
        split_date_col = split_cfg.get("date_col") or data_cfg.get("split_date_col")
        if split_date_col is None:
            split_date_col = (
                spec.target_date_col
                if spec.target_date_col in panel.columns
                else spec.date_col
            )
        with log_stage(logger, "Time split"):
            panel = split_by_time(
                panel,
                split_date_col,
                train_end=split_cfg.get("train_end"),
                val_end=split_cfg.get("val_end"),
                ratios=split_cfg.get("ratios", [0.6, 0.2, 0.2]),
            )
            split_counts = panel["split"].value_counts(dropna=False).to_dict()
            logger.info("Split counts | %s", " | ".join(f"{k}={v:,}" for k, v in split_counts.items()))

        with log_stage(logger, "Preprocess features"):
            preprocessor = PanelPreprocessor(
                features, scaling=data_cfg.get("scaling", "global")
            )
            panel = preprocessor.fit_transform_splits(
                panel, date_col=spec.date_col
            )
            logger.info("Preprocessing | scaling=%s", data_cfg.get("scaling", "global"))

        partition_cfg = config.get("partition", {})
        partition_type = partition_cfg.get("type", "iid").lower()
        with log_stage(logger, f"Assign {partition_type} clients"):
            if partition_type == "iid":
                panel = assign_iid_clients(
                    panel,
                    spec.entity_col,
                    int(partition_cfg.get("num_clients", 5)),
                    seed=seed,
                )
            elif partition_type == "domain":
                domain_col = partition_cfg["domain_col"]
                if partition_cfg.get("transform", "").lower() == "ff12":
                    output_col = partition_cfg.get("output_col", "ff12")
                    panel = add_ff12_column(
                        panel,
                        sic_col=domain_col,
                        output_col=output_col,
                        label_names=True,
                    )
                    before_drop = len(panel)
                    if bool(partition_cfg.get("drop_missing_domain", True)):
                        panel = panel.dropna(subset=[output_col]).copy()
                    logger.info(
                        "FF12 transform | output_col=%s | dropped_missing=%s",
                        output_col,
                        f"{before_drop - len(panel):,}",
                    )
                    domain_col = output_col
                panel = assign_domain_clients(
                    panel,
                    domain_col=domain_col,
                    max_clients=partition_cfg.get("max_clients"),
                    continuous_bins=partition_cfg.get("continuous_bins"),
                    entity_col=spec.entity_col,
                    persistent_entity=bool(
                        partition_cfg.get("persistent_entity", False)
                    ),
                    fit_split_col="split",
                    fit_split_value="train",
                )
            else:
                raise ValueError("partition.type must be iid or domain.")
            log_frame(logger, "Partitioned panel", panel, date_col=spec.date_col, entity_col=spec.entity_col, client_col="client_id")

        train = panel.loc[panel["split"] == "train"].copy()
        test = panel.loc[panel["split"] == "test"].copy()
        if train.empty or test.empty:
            raise ValueError("Training and test splits must both be non-empty.")
        clients = sorted(train["client_id"].astype(str).unique().tolist())
        evaluation_clients = set(test["client_id"].astype(str).unique())
        unseen_clients = evaluation_clients - set(clients)
        if unseen_clients:
            raise ValueError(
                f"Test clients absent from training: {sorted(unseen_clients)}"
            )
        log_frame(logger, "Training split", train, date_col=split_date_col, entity_col=spec.entity_col, client_col="client_id")
        log_frame(logger, "Test split", test, date_col=split_date_col, entity_col=spec.entity_col, client_col="client_id")
        client_sizes = train.groupby("client_id", observed=True).size().sort_values(ascending=False)
        logger.info(
            "Training client sizes | %s",
            " | ".join(f"{client}={int(size):,}" for client, size in client_sizes.items()),
        )

        feature_cfg = config.get("feature_heterogeneity", {})
        feature_masks = None
        if bool(feature_cfg.get("enabled", False)):
            with log_stage(logger, "Build feature heterogeneity masks"):
                feature_masks = build_feature_masks(
                    features,
                    clients,
                    common_fraction=float(
                        feature_cfg.get("common_fraction", 1.0)
                    ),
                    private_fraction=float(
                        feature_cfg.get("private_fraction", 0.5)
                    ),
                    seed=seed,
                )
                active_counts = {
                    client: int(np.asarray(mask).sum())
                    for client, mask in feature_masks.items()
                }
                logger.info(
                    "Feature masks | %s",
                    " | ".join(f"{client}={count}/{len(features)}" for client, count in active_counts.items()),
                )

        fed_cfg = config.get("federated", {})
        rounds = int(fed_cfg.get("rounds", 20))
        participation_rate = float(fed_cfg.get("participation_rate", 1.0))
        sampler = ClientSampler(clients, participation_rate, seed)
        schedule = sampler.schedule(rounds)
        temporal_h = temporal_unavailability(schedule, clients)

        with log_stage(logger, "Measure heterogeneity"):
            heterogeneity = partition_heterogeneity_report(
                train,
                features,
                spec.target_col,
                feature_masks=feature_masks,
                temporal_unavailability_value=temporal_h,
                seed=seed,
            )
            heterogeneity["feature_columns"] = features
            heterogeneity["split_date_col"] = split_date_col
            write_json(heterogeneity, output_dir / "heterogeneity.json")
            numeric_h = {
                key: value
                for key, value in heterogeneity.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            }
            logger.info(
                "Heterogeneity | %s",
                " | ".join(f"{key}={float(value):.6f}" for key, value in numeric_h.items()),
            )

        with log_stage(logger, "Save partition artifacts"):
            assignment_columns = [
                spec.date_col,
                spec.entity_col,
                "client_id",
                "split",
            ]
            if spec.target_date_col in panel.columns:
                assignment_columns.append(spec.target_date_col)
            if "client_id_domain" in panel.columns:
                assignment_columns.append("client_id_domain")
            panel[assignment_columns].to_parquet(
                output_dir / "partition_assignments.parquet", index=False
            )
            if feature_masks:
                mask_frame = pd.DataFrame(feature_masks, index=features).T
                mask_frame.index.name = "client_id"
                mask_frame.to_csv(output_dir / "feature_masks.csv")

        model_fn = _model_fn(config, len(features))
        client_optimizer = str(fed_cfg.get("client_optimizer", "sgd"))
        client_momentum = float(fed_cfg.get("client_momentum", 0.0))
        client_lr = float(fed_cfg.get("client_lr", 1e-2))
        results: list[dict[str, Any]] = []
        baseline_cfg = config.get("baselines", {})
        epoch_log_interval = int(logging_cfg.get("epoch_interval", 1))
        local_client_log_interval = int(logging_cfg.get("local_client_interval", 1))
        prediction_client_log_interval = int(logging_cfg.get("prediction_client_interval", 1))

        if bool(baseline_cfg.get("centralized", True)):
            method_name = "centralized_oracle" if feature_masks else "centralized"
            with log_stage(logger, f"Run {method_name}"):
                set_seed(seed)
                centralized = train_centralized(
                    train,
                    model_fn,
                    features,
                    spec.target_col,
                    epochs=int(baseline_cfg.get("epochs", 5)),
                    batch_size=int(fed_cfg.get("batch_size", 512)),
                    learning_rate=client_lr,
                    weight_decay=float(fed_cfg.get("weight_decay", 0.0)),
                    device=device,
                    optimizer_name=client_optimizer,
                    momentum=client_momentum,
                    logger=logger,
                    epoch_log_interval=epoch_log_interval,
                    log_name=method_name,
                )
                predictions = _prediction_frame(
                    centralized,
                    test,
                    features,
                    spec.target_col,
                    spec.date_col,
                    spec.entity_col,
                    "client_id",
                    device,
                    feature_masks=None,
                    target_date_col=spec.target_date_col,
                    logger=logger,
                    log_name=f"{method_name} prediction",
                    client_log_interval=prediction_client_log_interval,
                )
                metrics = _evaluate_and_save(
                    method_name, predictions, output_dir, config, logger
                )
                results.append({"method": method_name, **metrics})

        if feature_masks and bool(
            baseline_cfg.get("centralized_masked", True)
        ):
            with log_stage(logger, "Run centralized_masked"):
                set_seed(seed)
                masked_train = apply_feature_masks_to_frame(
                    train, features, feature_masks
                )
                masked_test = apply_feature_masks_to_frame(
                    test, features, feature_masks
                )
                centralized_masked = train_centralized(
                    masked_train,
                    model_fn,
                    features,
                    spec.target_col,
                    epochs=int(baseline_cfg.get("epochs", 5)),
                    batch_size=int(fed_cfg.get("batch_size", 512)),
                    learning_rate=client_lr,
                    weight_decay=float(fed_cfg.get("weight_decay", 0.0)),
                    device=device,
                    optimizer_name=client_optimizer,
                    momentum=client_momentum,
                    logger=logger,
                    epoch_log_interval=epoch_log_interval,
                    log_name="centralized_masked",
                )
                predictions = _prediction_frame(
                    centralized_masked,
                    masked_test,
                    features,
                    spec.target_col,
                    spec.date_col,
                    spec.entity_col,
                    "client_id",
                    device,
                    feature_masks=None,
                    target_date_col=spec.target_date_col,
                    logger=logger,
                    log_name="centralized_masked prediction",
                    client_log_interval=prediction_client_log_interval,
                )
                metrics = _evaluate_and_save(
                    "centralized_masked", predictions, output_dir, config, logger
                )
                results.append({"method": "centralized_masked", **metrics})

        if bool(baseline_cfg.get("local", True)):
            with log_stage(logger, "Run local baseline"):
                set_seed(seed)
                local_models = train_local_models(
                    train,
                    model_fn,
                    features,
                    spec.target_col,
                    feature_masks=feature_masks,
                    epochs=int(baseline_cfg.get("epochs", 5)),
                    batch_size=int(fed_cfg.get("batch_size", 512)),
                    learning_rate=client_lr,
                    weight_decay=float(fed_cfg.get("weight_decay", 0.0)),
                    device=device,
                    optimizer_name=client_optimizer,
                    momentum=client_momentum,
                    logger=logger,
                    client_log_interval=local_client_log_interval,
                    epoch_log_interval=int(logging_cfg.get("local_epoch_interval", 0)),
                )
                predictions = _local_prediction_frame(
                    local_models,
                    test,
                    features,
                    spec.target_col,
                    spec.date_col,
                    spec.entity_col,
                    "client_id",
                    device,
                    feature_masks,
                    target_date_col=spec.target_date_col,
                    logger=logger,
                    client_log_interval=prediction_client_log_interval,
                )
                metrics = _evaluate_and_save(
                    "local", predictions, output_dir, config, logger
                )
                results.append({"method": "local", **metrics})

        for algorithm in fed_cfg.get(
            "algorithms", ["fedavg", "fedprox", "scaffold", "fedadam"]
        ):
            with log_stage(logger, f"Run {algorithm}"):
                set_seed(seed)
                trainer = FederatedTrainer(
                    algorithm=algorithm,
                    model_fn=model_fn,
                    feature_columns=features,
                    target_col=spec.target_col,
                    rounds=rounds,
                    local_epochs=int(fed_cfg.get("local_epochs", 1)),
                    batch_size=int(fed_cfg.get("batch_size", 512)),
                    client_lr=client_lr,
                    weight_decay=float(fed_cfg.get("weight_decay", 0.0)),
                    proximal_mu=float(fed_cfg.get("proximal_mu", 0.01)),
                    participation_rate=participation_rate,
                    seed=seed,
                    device=device,
                    server_lr=float(fed_cfg.get("server_lr", 1e-2)),
                    server_beta1=float(fed_cfg.get("server_beta1", 0.9)),
                    server_beta2=float(fed_cfg.get("server_beta2", 0.99)),
                    server_tau=float(fed_cfg.get("server_tau", 1e-8)),
                    client_optimizer=client_optimizer,
                    client_momentum=client_momentum,
                    logger=logger,
                    round_log_interval=int(logging_cfg.get("round_interval", 1)),
                    log_client_details=bool(logging_cfg.get("federated_client_details", False)),
                    client_log_interval=int(logging_cfg.get("federated_client_interval", 1)),
                    local_epoch_log_interval=int(logging_cfg.get("federated_local_epoch_interval", 0)),
                )
                model = trainer.fit(train, feature_masks=feature_masks)
                training_history = trainer.history.to_frame()
                training_history.to_csv(
                    output_dir / f"training_{algorithm}.csv", index=False
                )
                predictions = _prediction_frame(
                    model,
                    test,
                    features,
                    spec.target_col,
                    spec.date_col,
                    spec.entity_col,
                    "client_id",
                    device,
                    feature_masks,
                    target_date_col=spec.target_date_col,
                    logger=logger,
                    log_name=f"{algorithm} prediction",
                    client_log_interval=prediction_client_log_interval,
                )
                metrics = _evaluate_and_save(
                    algorithm, predictions, output_dir, config, logger
                )
                metrics["communicated_parameters"] = int(
                    training_history["communicated_parameters"].sum()
                )
                metrics["communicated_bytes_fp32"] = (
                    4 * metrics["communicated_parameters"]
                )
                results.append({"method": algorithm, **metrics})

        with log_stage(logger, "Save benchmark summary"):
            results_frame = pd.DataFrame(results)
            results_frame.to_csv(
                output_dir / "benchmark_results.csv", index=False
            )
            write_json(config, output_dir / "resolved_config.json")
            logger.info(
                "Benchmark methods complete | %s",
                ", ".join(results_frame["method"].astype(str).tolist()),
            )
        logger.info(
            "FAP-Bench run complete | output=%s | elapsed=%s",
            output_dir.resolve(),
            format_seconds(time.perf_counter() - run_started),
        )
        return output_dir
    except Exception:
        logger.exception(
            "FAP-Bench run failed | elapsed=%s",
            format_seconds(time.perf_counter() - run_started),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FAP-Bench experiments."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a YAML configuration file."
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override logging.level from YAML, for example INFO or DEBUG.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Override logging.file from YAML.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.log_level or args.log_file:
        config.setdefault("logging", {})
        if args.log_level:
            config["logging"]["level"] = args.log_level
        if args.log_file:
            config["logging"]["file"] = args.log_file
    output = run(config)
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    main()
