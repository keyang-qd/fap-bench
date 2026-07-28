from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from fapbench.algorithms import FederatedTrainer, train_centralized, train_local_models
from fapbench.algorithms.common import frame_to_arrays, prepare_client_arrays
from fapbench.algorithms.parallel import ParallelClientRuntime
from fapbench.config import load_config
from fapbench.data import (
    PanelPreprocessor,
    PanelSpec,
    apply_universe_filters,
    client_local_rank_zscore,
    load_panel,
    prepare_panel,
    split_by_time,
    validate_crsp_compustat_profitability_panel,
)
from fapbench.evaluation import (
    apply_linear_prediction_calibrator,
    calibrate_predictions_within_client,
    client_averaged_long_short_metrics,
    client_portfolio_metrics,
    client_predictive_metrics,
    fit_linear_prediction_calibrator,
    long_short_client_exposures,
    long_short_portfolio_metrics,
    monthly_predictive_metrics,
    partition_heterogeneity_report,
    predictive_metrics,
)
from fapbench.models import AssetPricingMLP
from fapbench.selection import ValidationCheckpoint, combine_selection_histories
from fapbench.partition import (
    ClientSampler,
    add_ff12_column,
    add_sic_major_group_column,
    apply_client_size_imbalance,
    apply_partition_common_universe,
    apply_size_matched_uniform_control,
    apply_covariate_shift,
    apply_feature_masks_to_frame,
    apply_regression_label_skew,
    assign_domain_clients,
    assign_iid_clients,
    build_covariate_shift_matched_control_parameters,
    build_covariate_shift_parameters,
    build_feature_masks,
    build_feature_masks_matched_control,
    covariate_shift_frame,
    randomize_schedule_client_identity,
    regression_label_diagnostics,
    schedule_to_frame,
    temporal_availability_report,
    temporal_unavailability,
)
from fapbench.utils import (
    close_logger,
    configure_logger,
    ensure_dir,
    format_seconds,
    log_frame,
    log_stage,
    process_peak_rss_mb,
    set_seed,
    write_json,
)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _save_model_checkpoint(
    model: torch.nn.Module,
    path: Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": _cpu_state_dict(model),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def _save_local_checkpoints(
    models: Mapping[str, torch.nn.Module],
    path: Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "client_state_dicts": {
                str(client_id): _cpu_state_dict(model)
                for client_id, model in models.items()
            },
            "metadata": dict(metadata or {}),
        },
        path,
    )


def _local_tuning_grid(
    baseline_cfg: dict[str, Any],
    default_lr: float,
    default_weight_decay: float,
    *,
    selection_enabled: bool = True,
) -> list[tuple[float, float]]:
    """Candidate (learning_rate, weight_decay) pairs for the local-only baseline.

    By default local models inherit the federated client hyperparameters, which
    makes "federation helps" partly a statement about the budget-matching
    protocol rather than about collaboration: the local baseline never got a
    configuration chosen for it. Enabling ``baselines.local_tuning`` sweeps the
    listed values and selects per client on validation Rank IC, so each client
    is represented by the best local model it could have trained for itself.

    Returns a single-element list (the inherited configuration) when tuning is
    disabled, which keeps the default behaviour byte-identical.
    """
    tuning_cfg = baseline_cfg.get("local_tuning", {}) or {}
    if not bool(tuning_cfg.get("enabled", False)):
        return [(default_lr, default_weight_decay)]
    if not selection_enabled:
        # Without per-client validation there is no signal to select on, so the
        # sweep would silently keep whichever candidate happened to be first.
        raise ValueError(
            "baselines.local_tuning requires model_selection.enabled: per-client "
            "hyperparameter selection scores candidates on validation Rank IC, "
            "which is only computed when validation selection is on."
        )
    learning_rates = [
        float(value)
        for value in tuning_cfg.get("learning_rates", [default_lr])
    ] or [default_lr]
    weight_decays = [
        float(value)
        for value in tuning_cfg.get("weight_decays", [default_weight_decay])
    ] or [default_weight_decay]
    return [(lr, wd) for lr in learning_rates for wd in weight_decays]


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


def resolve_data_path(config: dict[str, Any]) -> Path:
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


def _prefix_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _resolve_baseline_epochs(
    config: dict[str, Any],
    effective_participation_rate: float | None = None,
) -> int:
    baseline_cfg = config.get("baselines", {})
    fed_cfg = config.get("federated", {})
    rounds = int(fed_cfg.get("rounds", 20))
    local_epochs = int(fed_cfg.get("local_epochs", 1))
    if effective_participation_rate is None:
        effective_participation_rate = float(
            fed_cfg.get("participation_rate", 1.0)
        )
    matched_passes = max(
        1,
        int(np.ceil(rounds * local_epochs * effective_participation_rate)),
    )
    if bool(baseline_cfg.get("match_federated_local_passes", False)):
        return matched_passes
    configured = baseline_cfg.get("epochs", 5)
    if configured is None:
        return matched_passes
    return int(configured)


def _validation_selector(
    name: str,
    validation_frame: pd.DataFrame,
    model_steps: int,
    config: dict[str, Any],
    features: list[str],
    target_col: str,
    date_col: str,
    entity_col: str,
    client_col: str,
    device: str,
    feature_masks: Mapping[str, np.ndarray] | None,
    target_date_col: str,
    logger: logging.Logger | None,
) -> ValidationCheckpoint | None:
    selection_cfg = config.get("model_selection", {})
    if not bool(selection_cfg.get("enabled", False)) or validation_frame.empty:
        return None
    metric = str(selection_cfg.get("metric", "rank_ic"))
    mode = str(
        selection_cfg.get(
            "mode", "min" if metric in {"mse", "mae"} else "max"
        )
    )
    evaluate_every = int(selection_cfg.get("evaluate_every", 5))
    evaluation_date = (
        target_date_col if target_date_col in validation_frame.columns else date_col
    )
    prediction_batch_size = int(
        config.get("performance", {}).get("prediction_batch_size", 65536)
    )

    # Validation is evaluated repeatedly. Materialize masked float32 arrays and
    # date metadata once rather than re-grouping/copying the DataFrame at every
    # checkpoint.
    cached_x: list[np.ndarray] = []
    cached_y: list[np.ndarray] = []
    cached_dates: list[pd.Series] = []
    for client_id, group in validation_frame.groupby(client_col, sort=True, observed=True):
        client_id = str(client_id)
        mask = None if feature_masks is None else feature_masks.get(client_id)
        x, y = frame_to_arrays(group, features, target_col, mask=mask)
        if len(y) != len(group):
            raise ValueError(
                "Non-finite validation rows remained after preprocessing; "
                "cannot build the validation cache safely."
            )
        cached_x.append(x)
        cached_y.append(y)
        cached_dates.append(group[evaluation_date].reset_index(drop=True))
    x_validation = np.ascontiguousarray(np.concatenate(cached_x, axis=0))
    y_validation = np.ascontiguousarray(np.concatenate(cached_y, axis=0))
    validation_dates = pd.concat(cached_dates, ignore_index=True)
    if logger is not None:
        logger.info(
            "%s | validation cache | rows=%s | memory=%.1f MB | batch_size=%s",
            name,
            f"{len(y_validation):,}",
            (x_validation.nbytes + y_validation.nbytes) / 1024**2,
            prediction_batch_size,
        )

    def score_fn(model: torch.nn.Module) -> float:
        device_obj = torch.device(device)
        model = model.to(device_obj)
        model.eval()
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for batch_start in range(0, len(x_validation), prediction_batch_size):
                xb = torch.as_tensor(
                    x_validation[batch_start : batch_start + prediction_batch_size],
                    dtype=torch.float32,
                    device=device_obj,
                )
                outputs.append(model(xb).detach().cpu().numpy())
        predictions = pd.DataFrame(
            {
                evaluation_date: validation_dates,
                "y_true": y_validation,
                "y_pred": np.concatenate(outputs),
            }
        )
        metrics = predictive_metrics(predictions, date_col=evaluation_date)
        if metric not in metrics:
            raise KeyError(
                f"model_selection.metric={metric!r} is unavailable. "
                f"Available metrics: {sorted(metrics)}"
            )
        return float(metrics[metric])

    return ValidationCheckpoint(
        score_fn=score_fn,
        metric=metric,
        mode=mode,
        evaluate_every=evaluate_every,
        total_steps=model_steps,
        logger=logger,
        name=name,
    )


def _evaluate_and_save(
    name: str,
    prediction_frame: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
    is_local: bool = False,
) -> dict[str, Any]:
    data_cfg = config["data"]
    eval_cfg = config.get("evaluation", {})
    evaluation_date = _evaluation_date_col(prediction_frame, config)
    entity_col = data_cfg.get("entity_col", "permno")
    client_col = str(eval_cfg.get("client_col", "client_id"))
    quantile = float(eval_cfg.get("portfolio_quantile", 0.1))
    transaction_cost_bps = float(eval_cfg.get("transaction_cost_bps", 0.0))
    periods_per_year = int(eval_cfg.get("periods_per_year", 12))
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
    monthly_metrics = monthly_predictive_metrics(
        prediction_frame, date_col=evaluation_date
    )
    client_frame, client_summary = client_predictive_metrics(
        prediction_frame, client_col=client_col, date_col=evaluation_date
    )

    score_frame = prediction_frame.copy()
    score_col = "y_pred"
    calibration = "none"
    uncalibrated_global_metrics: dict[str, Any] | None = None
    if is_local:
        calibration = str(
            eval_cfg.get("local_global_score_calibration", "percentile")
        ).lower()
        if calibration != "none":
            if bool(eval_cfg.get("report_uncalibrated_local_global", True)):
                _, uncalibrated_global_metrics = long_short_portfolio_metrics(
                    prediction_frame,
                    date_col=evaluation_date,
                    entity_col=entity_col,
                    client_col=client_col,
                    quantile=quantile,
                    transaction_cost_bps=transaction_cost_bps,
                    periods_per_year=periods_per_year,
                )
            score_frame = calibrate_predictions_within_client(
                prediction_frame,
                date_col=evaluation_date,
                client_col=client_col,
                prediction_col="y_pred",
                output_col="portfolio_score",
                method=calibration,
            )
            score_col = "portfolio_score"

    global_portfolio_frame, global_portfolio_metrics = long_short_portfolio_metrics(
        score_frame,
        date_col=evaluation_date,
        entity_col=entity_col,
        prediction_col=score_col,
        client_col=client_col,
        quantile=quantile,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
    )
    exposure_frame = long_short_client_exposures(
        score_frame,
        date_col=evaluation_date,
        client_col=client_col,
        prediction_col=score_col,
        quantile=quantile,
    )
    client_neutral_frame, client_neutral_metrics = (
        client_averaged_long_short_metrics(
            prediction_frame,
            client_col=client_col,
            date_col=evaluation_date,
            entity_col=entity_col,
            quantile=quantile,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=periods_per_year,
        )
    )
    per_client_portfolios, per_client_portfolio_summary = client_portfolio_metrics(
        prediction_frame,
        client_col=client_col,
        date_col=evaluation_date,
        entity_col=entity_col,
        quantile=quantile,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
    )

    saved_predictions = score_frame if score_col != "y_pred" else prediction_frame
    saved_predictions.to_parquet(
        output_dir / f"predictions_{name}.parquet", index=False
    )
    monthly_metrics.to_csv(
        output_dir / f"monthly_predictive_metrics_{name}.csv", index=False
    )
    client_frame.to_csv(
        output_dir / f"client_metrics_{name}.csv", index=False
    )
    global_portfolio_frame.to_csv(
        output_dir / f"portfolio_returns_global_{name}.csv", index=False
    )
    client_neutral_frame.to_csv(
        output_dir / f"portfolio_returns_client_neutral_{name}.csv", index=False
    )
    per_client_portfolios.to_csv(
        output_dir / f"client_portfolio_metrics_{name}.csv", index=False
    )
    exposure_frame.to_csv(
        output_dir / f"portfolio_client_exposures_{name}.csv", index=False
    )

    primary_mode = str(
        eval_cfg.get("primary_portfolio", "client_neutral")
    ).lower()
    if primary_mode == "global":
        primary_frame = global_portfolio_frame
        primary_metrics = global_portfolio_metrics
    elif primary_mode == "client_neutral":
        primary_frame = client_neutral_frame
        primary_metrics = client_neutral_metrics
    else:
        raise ValueError(
            "evaluation.primary_portfolio must be global or client_neutral."
        )
    # Legacy filename now points to the explicitly selected primary portfolio.
    primary_frame.to_csv(
        output_dir / f"portfolio_returns_{name}.csv", index=False
    )

    result: dict[str, Any] = {
        **global_metrics,
        **client_summary,
        **primary_metrics,
        **_prefix_metrics("global_portfolio_", global_portfolio_metrics),
        **_prefix_metrics("client_neutral_portfolio_", client_neutral_metrics),
        **_prefix_metrics("client_portfolio_", per_client_portfolio_summary),
        "evaluation_date_col": evaluation_date,
        "portfolio_primary_mode": primary_mode,
        "global_portfolio_score_calibration": calibration,
        "annual_return_definition": "arithmetic_monthly_mean_times_periods_per_year",
        "turnover_definition": "one_half_l1_change_in_portfolio_weights",
        "transaction_cost_application": "bps_times_monthly_turnover",
    }
    if uncalibrated_global_metrics is not None:
        result.update(
            _prefix_metrics(
                "uncalibrated_global_portfolio_",
                uncalibrated_global_metrics,
            )
        )
    result["evaluation_elapsed_seconds"] = float(time.perf_counter() - started)
    write_json(result, output_dir / f"metrics_{name}.json")
    if logger is not None:
        highlights = []
        for key in [
            "mse",
            "oos_r2_zero",
            "ic",
            "rank_ic",
            "net_sharpe",
            "client_portfolio_worst_net_sharpe",
        ]:
            value = result.get(key)
            if isinstance(value, (int, float, np.number)) and np.isfinite(value):
                highlights.append(f"{key}={float(value):.6f}")
        logger.info(
            "%s | evaluation done | primary_portfolio=%s | %s | elapsed=%s",
            name,
            primary_mode,
            " | ".join(highlights) if highlights else "metrics saved",
            format_seconds(time.perf_counter() - started),
        )
    return result


def run(
    config: dict[str, Any],
    *,
    preloaded_raw_panel: pd.DataFrame | None = None,
) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    torch.set_num_threads(int(config.get("torch_num_threads", 1)))
    try:
        torch.set_num_interop_threads(int(config.get("torch_num_interop_threads", 1)))
    except RuntimeError:
        pass
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
    data_path = resolve_data_path(config)
    device = config.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(
        "FAP-Bench run start | config=%s | output=%s | seed=%s | device=%s | torch_threads=%s | interop_threads=%s",
        config.get("_config_path", "<in-memory>"),
        output_dir.resolve(),
        seed,
        device,
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )
    logger.info(
        "Data path | %s | file_size=%.1f MB",
        data_path,
        data_path.stat().st_size / (1024**2) if data_path.exists() else float("nan"),
    )
    run_metadata: dict[str, Any] = {
        "status": "running",
        "seed": seed,
        "device": str(device),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "config_path": config.get("_config_path"),
        "output_dir": str(output_dir.resolve()),
        "data_path": str(data_path),
    }
    write_json(run_metadata, output_dir / "run_metadata.json")

    try:
        # Serialize the resolved configuration before any expensive work.  This
        # makes unsupported config types fail immediately rather than after a
        # long benchmark run.
        with log_stage(logger, "Validate and save resolved configuration"):
            write_json(config, output_dir / "resolved_config.json")

        with log_stage(logger, "Load panel"):
            if preloaded_raw_panel is None:
                raw = load_panel(data_path)
                panel_source = "disk"
            else:
                # prepare_panel() starts from frame.copy(), so the shared raw frame
                # remains unchanged and can safely be reused by later grid runs.
                raw = preloaded_raw_panel
                panel_source = "grid_memory_cache"
            logger.info("Panel source | %s", panel_source)
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

        universe_cfg = config.get("universe_filter", {})
        with log_stage(logger, "Apply investable-universe filters"):
            panel, universe_report = apply_universe_filters(
                panel,
                spec.date_col,
                enabled=bool(universe_cfg.get("enabled", False)),
                price_col=str(universe_cfg.get("price_col", "prc")),
                min_abs_price=universe_cfg.get("min_abs_price"),
                market_cap_col=str(universe_cfg.get("market_cap_col", "me")),
                exclude_bottom_market_cap_fraction=float(
                    universe_cfg.get("exclude_bottom_market_cap_fraction", 0.0)
                ),
                breakpoint_exchange_col=universe_cfg.get(
                    "breakpoint_exchange_col"
                ),
                breakpoint_exchange_values=tuple(
                    universe_cfg.get("breakpoint_exchange_values", [1])
                ),
                min_stocks_per_month=int(
                    universe_cfg.get("min_stocks_per_month", 0)
                ),
            )
            write_json(universe_report, output_dir / "universe_filter.json")
            logger.info(
                "Universe filter | enabled=%s | rows_before=%s | rows_after=%s | dropped=%s",
                universe_report.get("enabled"),
                f"{int(universe_report.get('n_rows_before', 0)):,}",
                f"{int(universe_report.get('n_rows_after', 0)):,}",
                f"{int(universe_report.get('n_rows_dropped', 0)):,}",
            )

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
                test_end=split_cfg.get("test_end"),
                ratios=split_cfg.get("ratios", [0.6, 0.2, 0.2]),
            )
            split_counts = panel["split"].value_counts(dropna=False).to_dict()
            logger.info("Split counts | %s", " | ".join(f"{k}={v:,}" for k, v in split_counts.items()))

        common_universe_cfg = config.get("partition_common_universe", {})
        common_universe_report: dict[str, Any] = {"enabled": False}
        if bool(common_universe_cfg.get("enabled", False)):
            with log_stage(logger, "Apply partition common universe"):
                panel, common_universe_report = apply_partition_common_universe(
                    panel,
                    required_columns=common_universe_cfg.get(
                        "required_columns", ["sic", "exchcd", "me"]
                    ),
                    finite_numeric_columns=common_universe_cfg.get(
                        "finite_numeric_columns", ["sic", "exchcd", "me"]
                    ),
                    positive_columns=common_universe_cfg.get(
                        "positive_columns", ["me"]
                    ),
                    ff12_sic_col=common_universe_cfg.get("ff12_sic_col", "sic"),
                )
                write_json(
                    common_universe_report,
                    output_dir / "partition_common_universe.json",
                )
                logger.info(
                    "Partition common universe | rows_before=%s | rows_after=%s | retained=%.4f",
                    f"{common_universe_report['n_rows_before']:,}",
                    f"{common_universe_report['n_rows_after']:,}",
                    common_universe_report["retained_fraction"],
                )

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
                        missing_action = "dropped"
                    else:
                        missing_label = str(
                            partition_cfg.get(
                                "missing_domain_label", "MissingSIC"
                            )
                        )
                        panel[output_col] = panel[output_col].fillna(
                            missing_label
                        )
                        missing_action = f"assigned_to_{missing_label}"
                    logger.info(
                        "FF12 transform | output_col=%s | missing_action=%s | dropped_missing=%s",
                        output_col,
                        missing_action,
                        f"{before_drop - len(panel):,}",
                    )
                    domain_col = output_col
                elif partition_cfg.get("transform", "").lower() == "sic_major_group":
                    output_col = partition_cfg.get("output_col", "sic_major_group")
                    panel = add_sic_major_group_column(
                        panel, sic_col=domain_col, output_col=output_col
                    )
                    before_drop = len(panel)
                    if bool(partition_cfg.get("drop_missing_domain", True)):
                        panel = panel.dropna(subset=[output_col]).copy()
                        missing_action = "dropped"
                    else:
                        missing_label = str(
                            partition_cfg.get(
                                "missing_domain_label", "MissingSIC"
                            )
                        )
                        panel[output_col] = panel[output_col].fillna(
                            missing_label
                        )
                        missing_action = f"assigned_to_{missing_label}"
                    logger.info(
                        "SIC-major-group transform | output_col=%s | missing_action=%s | dropped_missing=%s",
                        output_col,
                        missing_action,
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
                    missing_policy=str(
                        partition_cfg.get(
                            "missing_policy",
                            "drop" if bool(partition_cfg.get("drop_missing_domain", False)) else "separate",
                        )
                    ),
                )
            else:
                raise ValueError("partition.type must be iid or domain.")
            log_frame(logger, "Partitioned panel", panel, date_col=spec.date_col, entity_col=spec.entity_col, client_col="client_id")

        # Optional client-local standardization. Must run here: after clients
        # exist, before the splits are taken, so every split sees the same
        # transform. Off by default, so all previously published results are
        # byte-identical.
        client_local_standardization = bool(
            data_cfg.get("client_local_standardization", False)
        )
        if client_local_standardization:
            with log_stage(logger, "Client-local standardization"):
                panel = client_local_rank_zscore(
                    panel,
                    features,
                    client_col="client_id",
                    date_col=spec.date_col,
                )
                logger.info(
                    "Client-local standardization | features=%s | clients=%s",
                    len(features),
                    panel["client_id"].nunique(),
                )

        train = panel.loc[panel["split"] == "train"].copy()
        val = panel.loc[panel["split"] == "val"].copy()
        test = panel.loc[panel["split"] == "test"].copy()
        if train.empty or test.empty:
            raise ValueError("Training and test splits must both be non-empty.")
        if bool(config.get("model_selection", {}).get("enabled", False)) and val.empty:
            raise ValueError(
                "model_selection.enabled=true requires a non-empty validation split."
            )

        statistical_cfg = config.get("statistical_heterogeneity", {})
        statistical_report: dict[str, Any] = {"enabled": False}
        if bool(statistical_cfg.get("enabled", False)):
            statistical_report = {"enabled": True}
            matched_size_cfg = statistical_cfg.get("client_size_matched_control", {})
            size_cfg = statistical_cfg.get("client_size_imbalance", {})
            if bool(matched_size_cfg.get("enabled", False)) and bool(
                size_cfg.get("enabled", False)
            ):
                raise ValueError(
                    "Enable either client_size_matched_control or "
                    "client_size_imbalance, not both."
                )
            if bool(matched_size_cfg.get("enabled", False)):
                with log_stage(logger, "Apply matched uniform client-size control"):
                    rows_before = len(train)
                    train, size_report, matched_metadata = (
                        apply_size_matched_uniform_control(
                            train,
                            entity_col=spec.entity_col,
                            client_col="client_id",
                            matched_imbalance_ratio=float(
                                matched_size_cfg.get("matched_imbalance_ratio", 20.0)
                            ),
                            seed=seed,
                            min_entities_per_client=int(
                                matched_size_cfg.get("min_entities_per_client", 1)
                            ),
                        )
                    )
                    size_report.to_csv(
                        output_dir / "client_size_matched_control.csv", index=False
                    )
                    statistical_report.update(
                        {
                            "client_size_matched_control_enabled": True,
                            "client_size_matched_ratio": float(
                                matched_size_cfg.get("matched_imbalance_ratio", 20.0)
                            ),
                            "client_size_rows_before": int(rows_before),
                            "client_size_rows_after": int(len(train)),
                            "client_size_retained_fraction": float(
                                len(train) / rows_before
                            ),
                            **matched_metadata,
                        }
                    )
            if bool(size_cfg.get("enabled", False)):
                with log_stage(logger, "Apply client-size imbalance"):
                    rows_before = len(train)
                    train, size_report = apply_client_size_imbalance(
                        train,
                        entity_col=spec.entity_col,
                        client_col="client_id",
                        imbalance_ratio=float(size_cfg.get("imbalance_ratio", 5.0)),
                        seed=seed,
                        min_entities_per_client=int(
                            size_cfg.get("min_entities_per_client", 1)
                        ),
                    )
                    size_report.to_csv(
                        output_dir / "client_size_imbalance.csv", index=False
                    )
                    statistical_report.update(
                        {
                            "client_size_imbalance_enabled": True,
                            "client_size_imbalance_ratio_requested": float(
                                size_cfg.get("imbalance_ratio", 5.0)
                            ),
                            "client_size_rows_before": int(rows_before),
                            "client_size_rows_after": int(len(train)),
                            "client_size_retained_fraction": float(
                                len(train) / rows_before
                            ),
                        }
                    )
            label_cfg = statistical_cfg.get("label_skew", {})
            if bool(label_cfg.get("enabled", False)):
                with log_stage(logger, "Apply regression-label skew"):
                    rows_before = len(train)
                    finite_target = pd.to_numeric(
                        train[spec.target_col], errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan).dropna()
                    lower_tail = float(finite_target.quantile(0.05))
                    upper_tail = float(finite_target.quantile(0.95))
                    _, diagnostic_bin_edges = pd.qcut(
                        finite_target,
                        q=int(label_cfg.get("n_bins", 5)),
                        retbins=True,
                        duplicates="drop",
                    )
                    diagnostics_before = regression_label_diagnostics(
                        train,
                        target_col=spec.target_col,
                        date_col=split_date_col,
                        client_col="client_id",
                        entity_col=spec.entity_col,
                        stage="before",
                        lower_tail_threshold=lower_tail,
                        upper_tail_threshold=upper_tail,
                        n_bins=int(label_cfg.get("n_bins", 5)),
                        bin_edges=diagnostic_bin_edges,
                    )
                    train, label_report = apply_regression_label_skew(
                        train,
                        target_col=spec.target_col,
                        client_col="client_id",
                        n_bins=int(label_cfg.get("n_bins", 5)),
                        concentration=float(label_cfg.get("concentration", 1.0)),
                        retain_fraction=float(label_cfg.get("retain_fraction", 0.7)),
                        seed=seed,
                    )
                    diagnostics_after = regression_label_diagnostics(
                        train,
                        target_col=spec.target_col,
                        date_col=split_date_col,
                        client_col="client_id",
                        entity_col=spec.entity_col,
                        stage="after",
                        lower_tail_threshold=lower_tail,
                        upper_tail_threshold=upper_tail,
                        n_bins=int(label_cfg.get("n_bins", 5)),
                        bin_edges=diagnostic_bin_edges,
                    )
                    pd.concat(
                        [diagnostics_before, diagnostics_after], ignore_index=True
                    ).to_csv(
                        output_dir / "label_skew_diagnostics.csv", index=False
                    )
                    label_report.to_csv(
                        output_dir / "label_skew.csv", index=False
                    )
                    statistical_report.update(
                        {
                            "label_skew_enabled": True,
                            "label_skew_concentration": float(
                                label_cfg.get("concentration", 1.0)
                            ),
                            "label_skew_retain_fraction_requested": float(
                                label_cfg.get("retain_fraction", 0.7)
                            ),
                            "label_skew_rows_before": int(rows_before),
                            "label_skew_rows_after": int(len(train)),
                            "label_skew_retained_fraction": float(
                                len(train) / rows_before
                            ),
                            "label_skew_lower_tail_threshold": lower_tail,
                            "label_skew_upper_tail_threshold": upper_tail,
                        }
                    )

        clients = sorted(train["client_id"].astype(str).unique().tolist())
        evaluation_clients = set(test["client_id"].astype(str).unique())
        if not val.empty:
            evaluation_clients |= set(val["client_id"].astype(str).unique())
        unseen_clients = evaluation_clients - set(clients)
        if unseen_clients:
            raise ValueError(
                "Validation/test clients absent from training after controlled "
                f"heterogeneity: {sorted(unseen_clients)}"
            )

        covariate_cfg = statistical_cfg.get("covariate_shift", {})
        matched_covariate_cfg = statistical_cfg.get("covariate_shift_matched_control", {})
        if bool(covariate_cfg.get("enabled", False)) and bool(
            matched_covariate_cfg.get("enabled", False)
        ):
            raise ValueError(
                "Enable either covariate_shift or covariate_shift_matched_control, "
                "not both."
            )
        if bool(statistical_cfg.get("enabled", False)) and (
            bool(covariate_cfg.get("enabled", False))
            or bool(matched_covariate_cfg.get("enabled", False))
        ):
            matched_control = bool(matched_covariate_cfg.get("enabled", False))
            active_cfg = matched_covariate_cfg if matched_control else covariate_cfg
            stage_label = (
                "Apply matched uniform covariate-shift control"
                if matched_control
                else "Apply client-specific covariate shift"
            )
            with log_stage(logger, stage_label):
                builder = (
                    build_covariate_shift_matched_control_parameters
                    if matched_control
                    else build_covariate_shift_parameters
                )
                locations, scales = builder(
                    clients,
                    features,
                    location_std=float(active_cfg.get("location_std", 0.25)),
                    log_scale_std=float(active_cfg.get("log_scale_std", 0.15)),
                    seed=seed,
                )
                train = apply_covariate_shift(
                    train, features, locations, scales, client_col="client_id"
                )
                if not val.empty:
                    val = apply_covariate_shift(
                        val, features, locations, scales, client_col="client_id"
                    )
                test = apply_covariate_shift(
                    test, features, locations, scales, client_col="client_id"
                )
                covariate_shift_frame(features, locations, scales).to_csv(
                    output_dir / "covariate_shift_parameters.csv", index=False
                )
                statistical_report.update(
                    {
                        "covariate_shift_enabled": not matched_control,
                        "covariate_shift_matched_control_enabled": matched_control,
                        "covariate_location_std": float(
                            active_cfg.get("location_std", 0.25)
                        ),
                        "covariate_log_scale_std": float(
                            active_cfg.get("log_scale_std", 0.15)
                        ),
                    }
                )
        write_json(statistical_report, output_dir / "statistical_heterogeneity.json")

        log_frame(logger, "Training split", train, date_col=split_date_col, entity_col=spec.entity_col, client_col="client_id")
        if not val.empty:
            log_frame(logger, "Validation split", val, date_col=split_date_col, entity_col=spec.entity_col, client_col="client_id")
        log_frame(logger, "Test split", test, date_col=split_date_col, entity_col=spec.entity_col, client_col="client_id")
        client_sizes = train.groupby("client_id", observed=True).size().sort_values(ascending=False)
        logger.info(
            "Training client sizes | %s",
            " | ".join(f"{client}={int(size):,}" for client, size in client_sizes.items()),
        )

        feature_cfg = config.get("feature_heterogeneity", {})
        feature_masks = None
        if bool(feature_cfg.get("enabled", False)):
            shared_private_selection = bool(
                feature_cfg.get("shared_private_selection", False)
            )
            mask_builder = (
                build_feature_masks_matched_control
                if shared_private_selection
                else build_feature_masks
            )
            stage_label = (
                "Build matched-control (shared) feature heterogeneity masks"
                if shared_private_selection
                else "Build feature heterogeneity masks"
            )
            with log_stage(logger, stage_label):
                feature_masks = mask_builder(
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
        temporal_cfg = config.get("temporal_heterogeneity", {})
        temporal_enabled = bool(temporal_cfg.get("enabled", False))
        temporal_mode = (
            str(temporal_cfg.get("mode", "persistent")).lower()
            if temporal_enabled
            else "random"
        )
        reference_schedule = None
        if temporal_mode == "matched_random":
            reference_cfg = temporal_cfg.get("matched_reference", {})
            reference_mode = str(reference_cfg.get("mode", "persistent")).lower()
            reference_sampler = ClientSampler(
                clients,
                participation_rate,
                seed,
                mode=reference_mode,
                total_rounds=rounds,
                min_availability=float(
                    reference_cfg.get(
                        "min_availability", temporal_cfg.get("min_availability", 0.25)
                    )
                ),
                max_availability=float(
                    reference_cfg.get(
                        "max_availability", temporal_cfg.get("max_availability", 1.0)
                    )
                ),
                block_fraction=float(
                    reference_cfg.get(
                        "block_fraction", temporal_cfg.get("block_fraction", 0.35)
                    )
                ),
            )
            reference_schedule = reference_sampler.schedule(rounds)
            schedule = randomize_schedule_client_identity(
                reference_schedule, clients, seed=seed + 991
            )
            if schedule == reference_schedule:
                raise ValueError(
                    "The matched-random schedule could not alter temporal client "
                    "identity while preserving round and client totals."
                )
            schedule_to_frame(reference_schedule).to_csv(
                output_dir / "matched_reference_participation_schedule.csv",
                index=False,
            )
            expected_availability = {
                client: float(
                    np.mean(
                        [
                            client in set(map(str, active))
                            for active in reference_schedule.values()
                        ]
                    )
                )
                for client in clients
            }
        else:
            sampler = ClientSampler(
                clients,
                participation_rate,
                seed,
                mode=temporal_mode,
                total_rounds=rounds,
                min_availability=float(temporal_cfg.get("min_availability", 0.25)),
                max_availability=float(temporal_cfg.get("max_availability", 1.0)),
                block_fraction=float(temporal_cfg.get("block_fraction", 0.35)),
            )
            schedule = sampler.schedule(rounds)
            expected_availability = sampler.availability_probabilities()
        effective_participation_rate = float(
            np.mean([len(active) / len(clients) for active in schedule.values()])
        )
        temporal_h = temporal_unavailability(schedule, clients)
        participation_frame = schedule_to_frame(schedule)
        participation_frame.to_csv(
            output_dir / "participation_schedule.csv", index=False
        )
        client_availability, temporal_summary = temporal_availability_report(
            schedule, clients
        )
        client_availability["expected_availability"] = client_availability[
            "client_id"
        ].map(expected_availability)
        client_availability.to_csv(
            output_dir / "client_availability.csv", index=False
        )

        with log_stage(logger, "Measure heterogeneity"):
            heterogeneity = partition_heterogeneity_report(
                train,
                features,
                spec.target_col,
                feature_masks=feature_masks,
                temporal_unavailability_value=temporal_h,
                seed=seed,
            )
            heterogeneity.update(temporal_summary)
            heterogeneity["temporal_mode"] = temporal_mode
            if reference_schedule is not None:
                heterogeneity["temporal_matched_reference_mode"] = str(
                    temporal_cfg.get("matched_reference", {}).get(
                        "mode", "persistent"
                    )
                )
                heterogeneity["temporal_matched_active_slots"] = int(
                    sum(len(active) for active in reference_schedule.values())
                )
            heterogeneity["feature_columns"] = features
            heterogeneity["split_date_col"] = split_date_col
            for key, value in statistical_report.items():
                if isinstance(value, (int, float, bool, str)):
                    heterogeneity[f"controlled_{key}"] = value
            for key, value in common_universe_report.items():
                if isinstance(value, (int, float, bool, str)):
                    heterogeneity[f"common_universe_{key}"] = value
            write_json(heterogeneity, output_dir / "heterogeneity.json")
            numeric_h = {
                key: value
                for key, value in heterogeneity.items()
                if isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, bool)
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

        if bool(config.get("diagnostics_only", False)):
            diagnostics_result = pd.DataFrame(
                [
                    {
                        "method": "diagnostics_only",
                        "diagnostics_only": True,
                        "training_skipped": True,
                    }
                ]
            )
            diagnostics_result.to_csv(
                output_dir / "benchmark_results.csv", index=False
            )
            elapsed_seconds = float(time.perf_counter() - run_started)
            run_metadata.update(
                {
                    "status": "completed",
                    "elapsed_seconds": elapsed_seconds,
                    "methods": ["diagnostics_only"],
                    "n_clients": len(clients),
                    "n_features": len(features),
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(val)),
                    "test_rows": int(len(test)),
                    "diagnostics_only": True,
                    "process_peak_rss_mb": process_peak_rss_mb(),
                }
            )
            write_json(run_metadata, output_dir / "run_metadata.json")
            logger.info(
                "Diagnostics-only run complete | output=%s | elapsed=%s",
                output_dir.resolve(),
                format_seconds(elapsed_seconds),
            )
            return output_dir

        model_fn = _model_fn(config, len(features))
        checkpoint_dir = ensure_dir(output_dir / "checkpoints")
        client_optimizer = str(fed_cfg.get("client_optimizer", "sgd"))
        client_momentum = float(fed_cfg.get("client_momentum", 0.0))
        client_lr = float(fed_cfg.get("client_lr", 1e-2))
        local_epochs = int(fed_cfg.get("local_epochs", 1))
        baseline_epochs = _resolve_baseline_epochs(config, effective_participation_rate)
        results: list[dict[str, Any]] = []
        baseline_cfg = config.get("baselines", {})
        epoch_log_interval = int(logging_cfg.get("epoch_interval", 1))
        local_client_log_interval = int(logging_cfg.get("local_client_interval", 1))
        prediction_client_log_interval = int(logging_cfg.get("prediction_client_interval", 1))
        logger.info(
            "Training budget | baseline_epochs=%s | federated_rounds=%s | "
            "federated_local_epochs=%s | expected_global_data_passes=%s",
            baseline_epochs,
            rounds,
            local_epochs,
            rounds * local_epochs * effective_participation_rate,
        )

        if bool(baseline_cfg.get("centralized", True)):
            method_name = "centralized_oracle" if feature_masks else "centralized"
            method_started = time.perf_counter()
            with log_stage(logger, f"Run {method_name}"):
                set_seed(seed)
                selector = _validation_selector(
                    method_name,
                    val,
                    baseline_epochs,
                    config,
                    features,
                    spec.target_col,
                    spec.date_col,
                    spec.entity_col,
                    "client_id",
                    device,
                    None,
                    spec.target_date_col,
                    logger,
                )
                training_started = time.perf_counter()
                centralized = train_centralized(
                    train,
                    model_fn,
                    features,
                    spec.target_col,
                    epochs=baseline_epochs,
                    batch_size=int(fed_cfg.get("batch_size", 512)),
                    learning_rate=client_lr,
                    weight_decay=float(fed_cfg.get("weight_decay", 0.0)),
                    device=device,
                    optimizer_name=client_optimizer,
                    momentum=client_momentum,
                    logger=logger,
                    epoch_log_interval=epoch_log_interval,
                    log_name=method_name,
                    epoch_end_callback=selector,
                )
                training_elapsed = time.perf_counter() - training_started
                if selector is not None:
                    selector.restore(centralized)
                    selector.to_frame().to_csv(
                        output_dir / f"validation_history_{method_name}.csv",
                        index=False,
                    )
                _save_model_checkpoint(
                    centralized,
                    checkpoint_dir / f"{method_name}.pt",
                    {"method": method_name, "seed": seed},
                )
                prediction_started = time.perf_counter()
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
                prediction_elapsed = time.perf_counter() - prediction_started
                metrics = _evaluate_and_save(
                    method_name, predictions, output_dir, config, logger
                )
                metrics["training_elapsed_seconds"] = float(training_elapsed)
                metrics["prediction_elapsed_seconds"] = float(prediction_elapsed)
                metrics["training_epochs"] = baseline_epochs
                metrics["training_local_passes"] = baseline_epochs
                if selector is not None:
                    metrics.update(selector.metadata())
                metrics["method_elapsed_seconds"] = float(time.perf_counter() - method_started)
                metrics["process_peak_rss_mb"] = process_peak_rss_mb()
                write_json(metrics, output_dir / f"metrics_{method_name}.json")
                results.append({"method": method_name, **metrics})

        if feature_masks and bool(
            baseline_cfg.get("centralized_masked", True)
        ):
            method_started = time.perf_counter()
            with log_stage(logger, "Run centralized_masked"):
                set_seed(seed)
                masked_train = apply_feature_masks_to_frame(
                    train, features, feature_masks
                )
                masked_val = apply_feature_masks_to_frame(
                    val, features, feature_masks
                )
                masked_test = apply_feature_masks_to_frame(
                    test, features, feature_masks
                )
                selector = _validation_selector(
                    "centralized_masked",
                    masked_val,
                    baseline_epochs,
                    config,
                    features,
                    spec.target_col,
                    spec.date_col,
                    spec.entity_col,
                    "client_id",
                    device,
                    None,
                    spec.target_date_col,
                    logger,
                )
                training_started = time.perf_counter()
                centralized_masked = train_centralized(
                    masked_train,
                    model_fn,
                    features,
                    spec.target_col,
                    epochs=baseline_epochs,
                    batch_size=int(fed_cfg.get("batch_size", 512)),
                    learning_rate=client_lr,
                    weight_decay=float(fed_cfg.get("weight_decay", 0.0)),
                    device=device,
                    optimizer_name=client_optimizer,
                    momentum=client_momentum,
                    logger=logger,
                    epoch_log_interval=epoch_log_interval,
                    log_name="centralized_masked",
                    epoch_end_callback=selector,
                )
                training_elapsed = time.perf_counter() - training_started
                if selector is not None:
                    selector.restore(centralized_masked)
                    selector.to_frame().to_csv(
                        output_dir / "validation_history_centralized_masked.csv",
                        index=False,
                    )
                _save_model_checkpoint(
                    centralized_masked,
                    checkpoint_dir / "centralized_masked.pt",
                    {"method": "centralized_masked", "seed": seed},
                )
                prediction_started = time.perf_counter()
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
                prediction_elapsed = time.perf_counter() - prediction_started
                metrics = _evaluate_and_save(
                    "centralized_masked", predictions, output_dir, config, logger
                )
                metrics["training_elapsed_seconds"] = float(training_elapsed)
                metrics["prediction_elapsed_seconds"] = float(prediction_elapsed)
                metrics["training_epochs"] = baseline_epochs
                metrics["training_local_passes"] = baseline_epochs
                if selector is not None:
                    metrics.update(selector.metadata())
                metrics["method_elapsed_seconds"] = float(time.perf_counter() - method_started)
                metrics["process_peak_rss_mb"] = process_peak_rss_mb()
                write_json(
                    metrics, output_dir / "metrics_centralized_masked.json"
                )
                results.append({"method": "centralized_masked", **metrics})

        if bool(baseline_cfg.get("local", True)):
            method_started = time.perf_counter()
            with log_stage(logger, "Run local baseline"):
                local_grid = _local_tuning_grid(
                    baseline_cfg,
                    client_lr,
                    float(fed_cfg.get("weight_decay", 0.0)),
                    selection_enabled=bool(
                        config.get("model_selection", {}).get("enabled", False)
                    ),
                )
                best_local_models: dict[str, Any] = {}
                best_local_scores: dict[str, float] = {}
                best_local_config: dict[str, tuple[float, float]] = {}
                local_selectors: dict[str, ValidationCheckpoint] = {}
                training_started = time.perf_counter()
                for candidate_lr, candidate_wd in local_grid:
                    set_seed(seed)
                    candidate_selectors: dict[str, ValidationCheckpoint] = {}
                    candidate_callbacks: dict[str, ValidationCheckpoint] = {}
                    if bool(config.get("model_selection", {}).get("enabled", False)):
                        for client_id in clients:
                            client_val = val.loc[
                                val["client_id"].astype(str) == client_id
                            ].copy()
                            if client_val.empty:
                                continue
                            client_masks = (
                                None
                                if feature_masks is None
                                else {client_id: feature_masks[client_id]}
                            )
                            selector = _validation_selector(
                                f"local client={client_id}",
                                client_val,
                                baseline_epochs,
                                config,
                                features,
                                spec.target_col,
                                spec.date_col,
                                spec.entity_col,
                                "client_id",
                                device,
                                client_masks,
                                spec.target_date_col,
                                logger,
                            )
                            if selector is not None:
                                candidate_selectors[client_id] = selector
                                candidate_callbacks[client_id] = selector
                    candidate_models = train_local_models(
                        train,
                        model_fn,
                        features,
                        spec.target_col,
                        feature_masks=feature_masks,
                        epochs=baseline_epochs,
                        batch_size=int(fed_cfg.get("batch_size", 512)),
                        learning_rate=candidate_lr,
                        weight_decay=candidate_wd,
                        device=device,
                        optimizer_name=client_optimizer,
                        momentum=client_momentum,
                        logger=logger,
                        client_log_interval=local_client_log_interval,
                        epoch_log_interval=int(
                            logging_cfg.get("local_epoch_interval", 0)
                        ),
                        epoch_end_callbacks=candidate_callbacks or None,
                    )
                    for client_id, selector in candidate_selectors.items():
                        if client_id in candidate_models:
                            selector.restore(candidate_models[client_id])
                    # Keep, per client, the candidate with the best validation
                    # score. With a single candidate this reduces to the
                    # inherited-hyperparameter behaviour.
                    for client_id, model in candidate_models.items():
                        selector = candidate_selectors.get(client_id)
                        score = (
                            float(selector.best_score)
                            if selector is not None
                            and np.isfinite(selector.best_score)
                            else -np.inf
                        )
                        if (
                            client_id not in best_local_models
                            or score > best_local_scores[client_id]
                        ):
                            best_local_models[client_id] = model
                            best_local_scores[client_id] = score
                            best_local_config[client_id] = (candidate_lr, candidate_wd)
                            if selector is not None:
                                local_selectors[client_id] = selector
                local_models = best_local_models
                training_elapsed = time.perf_counter() - training_started
                if len(local_grid) > 1:
                    pd.DataFrame(
                        [
                            {
                                "client_id": client_id,
                                "selected_learning_rate": lr,
                                "selected_weight_decay": wd,
                                "validation_score": best_local_scores.get(
                                    client_id, float("nan")
                                ),
                            }
                            for client_id, (lr, wd) in sorted(
                                best_local_config.items()
                            )
                        ]
                    ).to_csv(
                        output_dir / "local_selected_hyperparameters.csv", index=False
                    )
                    logger.info(
                        "Local baseline tuned per client over %d candidate "
                        "configuration(s); selections written to "
                        "local_selected_hyperparameters.csv",
                        len(local_grid),
                    )
                local_selection_history = combine_selection_histories(
                    local_selectors
                )
                if not local_selection_history.empty:
                    local_selection_history.to_csv(
                        output_dir / "validation_history_local.csv", index=False
                    )
                _save_local_checkpoints(
                    local_models,
                    checkpoint_dir / "local.pt",
                    {"method": "local", "seed": seed},
                )
                prediction_started = time.perf_counter()
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
                prediction_elapsed = time.perf_counter() - prediction_started
                metrics = _evaluate_and_save(
                    "local",
                    predictions,
                    output_dir,
                    config,
                    logger,
                    is_local=True,
                )
                metrics["training_elapsed_seconds"] = float(training_elapsed)
                metrics["prediction_elapsed_seconds"] = float(prediction_elapsed)
                metrics["training_epochs"] = baseline_epochs
                metrics["training_local_passes"] = baseline_epochs
                if local_selectors:
                    selected_steps = [
                        selector.best_step
                        for selector in local_selectors.values()
                        if selector.best_step is not None
                    ]
                    best_scores = [
                        selector.best_score
                        for selector in local_selectors.values()
                        if selector.best_state is not None
                        and np.isfinite(selector.best_score)
                    ]
                    if selected_steps:
                        metrics["selected_step_mean"] = float(np.mean(selected_steps))
                        metrics["selected_step_min"] = int(np.min(selected_steps))
                        metrics["selected_step_max"] = int(np.max(selected_steps))
                    if best_scores:
                        metrics["validation_best_score_mean"] = float(
                            np.mean(best_scores)
                        )
                        metrics["validation_best_score_worst"] = float(
                            np.min(best_scores)
                            if next(iter(local_selectors.values())).mode == "max"
                            else np.max(best_scores)
                        )
                    metrics["selection_metric"] = next(
                        iter(local_selectors.values())
                    ).metric
                metrics["method_elapsed_seconds"] = float(time.perf_counter() - method_started)
                metrics["process_peak_rss_mb"] = process_peak_rss_mb()
                write_json(metrics, output_dir / "metrics_local.json")
                results.append({"method": "local", **metrics})

        performance_cfg = config.get("performance", {})
        parallel_client_count = int(performance_cfg.get("parallel_clients", 1))
        cache_started = time.perf_counter()
        federated_client_arrays = prepare_client_arrays(
            train,
            features,
            spec.target_col,
            client_col="client_id",
            feature_masks=feature_masks,
        )
        logger.info(
            "Federated client cache | clients=%s | rows=%s | memory=%.1f MB | elapsed=%s",
            len(federated_client_arrays),
            f"{sum(item.n_rows for item in federated_client_arrays.values()):,}",
            sum(item.nbytes for item in federated_client_arrays.values()) / 1024**2,
            format_seconds(time.perf_counter() - cache_started),
        )
        configured_batch_size = int(fed_cfg.get("batch_size", 512))
        client_budget = pd.DataFrame(
            [
                {
                    "client_id": client_id,
                    "n_rows": arrays.n_rows,
                    "batch_size": min(configured_batch_size, arrays.n_rows),
                    "batches_per_local_epoch": int(
                        np.ceil(arrays.n_rows / configured_batch_size)
                    ),
                    "optimizer_steps_per_selected_round": int(
                        np.ceil(arrays.n_rows / configured_batch_size)
                    )
                    * local_epochs,
                }
                for client_id, arrays in federated_client_arrays.items()
            ]
        ).sort_values("client_id")
        client_budget.to_csv(
            output_dir / "client_training_budget.csv", index=False
        )
        if parallel_client_count > 1 and torch.device(device).type == "cpu":
            runtime_context = ParallelClientRuntime(
                federated_client_arrays,
                max_workers=min(parallel_client_count, len(federated_client_arrays)),
                start_method=str(
                    performance_cfg.get("multiprocessing_start_method", "spawn")
                ),
                worker_torch_threads=int(
                    performance_cfg.get("worker_torch_threads", 1)
                ),
            )
            logger.info(
                "Reusable parallel runtime ready | workers=%s | reused_across_algorithms=true",
                runtime_context.max_workers,
            )
        else:
            runtime_context = nullcontext(None)

        with runtime_context as parallel_runtime:
            for algorithm in fed_cfg.get(
                "algorithms", ["fedavg", "fedprox", "scaffold", "fedadam"]
            ):
                method_started = time.perf_counter()
                with log_stage(logger, f"Run {algorithm}"):
                    set_seed(seed)
                    selector = _validation_selector(
                        algorithm,
                        val,
                        rounds,
                        config,
                        features,
                        spec.target_col,
                        spec.date_col,
                        spec.entity_col,
                        "client_id",
                        device,
                        feature_masks,
                        spec.target_date_col,
                        logger,
                    )
                    trainer = FederatedTrainer(
                        algorithm=algorithm,
                        model_fn=model_fn,
                        feature_columns=features,
                        target_col=spec.target_col,
                        rounds=rounds,
                        local_epochs=local_epochs,
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
                        parallel_clients=int(performance_cfg.get("parallel_clients", 1)),
                        multiprocessing_start_method=str(
                            performance_cfg.get("multiprocessing_start_method", "spawn")
                        ),
                        worker_torch_threads=int(
                            performance_cfg.get("worker_torch_threads", 1)
                        ),
                        cache_client_arrays=bool(
                            performance_cfg.get("cache_client_arrays", True)
                        ),
                    )
                    training_started = time.perf_counter()
                    model = trainer.fit(
                        train,
                        feature_masks=feature_masks,
                        round_end_callback=selector,
                        client_arrays=federated_client_arrays,
                        parallel_runtime=parallel_runtime,
                        client_schedule=schedule,
                    )
                    training_elapsed = time.perf_counter() - training_started
                    if selector is not None:
                        selector.restore(model)
                        selector.to_frame().to_csv(
                            output_dir / f"validation_history_{algorithm}.csv",
                            index=False,
                        )
                    _save_model_checkpoint(
                        model,
                        checkpoint_dir / f"{algorithm}.pt",
                        {"method": algorithm, "seed": seed},
                    )
                    training_history = trainer.history.to_frame()
                    training_history.to_csv(
                        output_dir / f"training_{algorithm}.csv", index=False
                    )
                    calibration_cfg = config.get("evaluation", {}).get(
                        "validation_prediction_calibration", {}
                    )
                    calibration_methods = {
                        str(value).lower()
                        for value in calibration_cfg.get("methods", ["fedadam"])
                    }
                    report_selected_validation = bool(
                        config.get("model_selection", {}).get(
                            "report_selected_validation_metrics", True
                        )
                    )
                    needs_calibration_validation = bool(
                        calibration_cfg.get("enabled", False)
                    ) and algorithm.lower() in calibration_methods
                    validation_predictions = None
                    if not val.empty and (
                        report_selected_validation or needs_calibration_validation
                    ):
                        validation_predictions = _prediction_frame(
                            model,
                            val,
                            features,
                            spec.target_col,
                            spec.date_col,
                            spec.entity_col,
                            "client_id",
                            device,
                            feature_masks,
                            target_date_col=spec.target_date_col,
                            logger=logger,
                            log_name=f"{algorithm} selected validation prediction",
                            client_log_interval=prediction_client_log_interval,
                        )
                        validation_predictions.to_parquet(
                            output_dir / f"validation_predictions_{algorithm}.parquet",
                            index=False,
                        )
                    prediction_started = time.perf_counter()
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
                    prediction_elapsed = time.perf_counter() - prediction_started
                    metrics = _evaluate_and_save(
                        algorithm, predictions, output_dir, config, logger
                    )
                    if validation_predictions is not None:
                        validation_date = _evaluation_date_col(
                            validation_predictions, config
                        )
                        metrics.update(
                            _prefix_metrics(
                                "validation_selected_",
                                predictive_metrics(
                                    validation_predictions,
                                    date_col=validation_date,
                                ),
                            )
                        )
                    metrics["training_elapsed_seconds"] = float(training_elapsed)
                    metrics["prediction_elapsed_seconds"] = float(prediction_elapsed)
                    metrics["training_rounds"] = rounds
                    metrics["training_local_epochs"] = local_epochs
                    metrics["training_selected_client_local_passes"] = (
                        rounds * local_epochs
                    )
                    metrics["training_effective_participation_rate"] = (
                        effective_participation_rate
                    )
                    metrics["training_expected_global_data_passes"] = (
                        rounds * local_epochs * effective_participation_rate
                    )
                    # Backward-compatible field now records full-dataset-equivalent passes.
                    metrics["training_local_passes"] = (
                        rounds * local_epochs * effective_participation_rate
                    )
                    metrics["communicated_parameters"] = int(
                        training_history["communicated_parameters"].sum()
                    )
                    metrics["communicated_bytes_fp32"] = (
                        4 * metrics["communicated_parameters"]
                    )
                    metrics["training_round_elapsed_seconds"] = float(
                        training_history["elapsed_seconds"].sum()
                    )
                    metrics["client_update_elapsed_seconds"] = float(
                        training_history["client_update_seconds"].sum()
                    )
                    metrics["server_aggregation_elapsed_seconds"] = float(
                        training_history["server_aggregation_seconds"].sum()
                    )
                    if "optimizer_steps_total" in training_history.columns:
                        metrics["optimizer_steps_total"] = int(
                            training_history["optimizer_steps_total"].sum()
                        )
                        metrics["optimizer_steps_per_round_mean"] = float(
                            training_history["optimizer_steps_total"].mean()
                        )
                        metrics["optimizer_steps_per_round_min"] = int(
                            training_history["optimizer_steps_total"].min()
                        )
                        metrics["optimizer_steps_per_round_max"] = int(
                            training_history["optimizer_steps_total"].max()
                        )
                        metrics["selected_client_steps_mean"] = float(
                            training_history["optimizer_steps_mean"].mean()
                        )
                        metrics["selected_client_steps_min"] = int(
                            training_history["optimizer_steps_min"].min()
                        )
                        metrics["selected_client_steps_max"] = int(
                            training_history["optimizer_steps_max"].max()
                        )
                    if selector is not None:
                        metrics.update(selector.metadata())
                        if selector.best_step is not None:
                            selected_row = training_history.loc[
                                training_history["round"] == selector.best_step
                            ]
                            if not selected_row.empty:
                                selected_communication = int(
                                    selected_row.iloc[-1][
                                        "cumulative_communicated_parameters"
                                    ]
                                )
                                metrics[
                                    "selected_checkpoint_communicated_parameters"
                                ] = selected_communication
                                metrics[
                                    "selected_checkpoint_communicated_bytes_fp32"
                                ] = 4 * selected_communication
                    metrics["method_elapsed_seconds"] = float(time.perf_counter() - method_started)
                    metrics["process_peak_rss_mb"] = process_peak_rss_mb()
                    write_json(metrics, output_dir / f"metrics_{algorithm}.json")
                    results.append({"method": algorithm, **metrics})

                    if bool(calibration_cfg.get("enabled", False)) and (
                        algorithm.lower() in calibration_methods
                    ):
                        if validation_predictions is None:
                            raise ValueError(
                                "Validation prediction calibration requires a "
                                "non-empty validation split."
                            )
                        calibration_started = time.perf_counter()
                        calibrator = fit_linear_prediction_calibrator(
                            validation_predictions,
                            nonnegative_slope=bool(
                                calibration_cfg.get("nonnegative_slope", True)
                            ),
                            ridge=float(calibration_cfg.get("ridge", 1.0e-12)),
                        )
                        calibrated_predictions = apply_linear_prediction_calibrator(
                            predictions, calibrator
                        )
                        suffix = str(calibration_cfg.get("method_suffix", "calibrated"))
                        calibrated_name = f"{algorithm}_{suffix}"
                        calibrated_metrics = _evaluate_and_save(
                            calibrated_name,
                            calibrated_predictions,
                            output_dir,
                            config,
                            logger,
                        )
                        shared_metadata_keys = [
                            key
                            for key in metrics
                            if key.startswith("training_")
                            or key.startswith("communicated_")
                            or key.startswith("optimizer_")
                            or key.startswith("selected_client_steps_")
                            or key.startswith("selected_")
                            or key.startswith("validation_")
                            or key in {
                                "selection_metric",
                                "selection_mode",
                                "client_update_elapsed_seconds",
                                "server_aggregation_elapsed_seconds",
                                "process_peak_rss_mb",
                            }
                        ]
                        for key in shared_metadata_keys:
                            calibrated_metrics[key] = metrics[key]
                        calibrated_metrics.update(
                            {
                                "evaluation_only_variant_of": algorithm,
                                "prediction_calibration_method": "validation_linear",
                                "prediction_calibration_fitted_on": "validation",
                                "prediction_calibration_intercept": calibrator.intercept,
                                "prediction_calibration_slope": calibrator.slope,
                                "prediction_calibration_raw_slope": calibrator.raw_slope,
                                "prediction_calibration_direction_valid": (
                                    calibrator.direction_valid
                                ),
                                "prediction_calibration_n_obs": calibrator.n_obs,
                                "prediction_calibration_nonnegative_slope": (
                                    calibrator.nonnegative_slope
                                ),
                                "prediction_elapsed_seconds": float(
                                    prediction_elapsed
                                    + time.perf_counter()
                                    - calibration_started
                                ),
                                "method_elapsed_seconds": float(
                                    time.perf_counter() - method_started
                                ),
                                "process_peak_rss_mb": process_peak_rss_mb(),
                            }
                        )
                        write_json(
                            calibrator.to_dict(),
                            output_dir / f"prediction_calibration_{algorithm}.json",
                        )
                        write_json(
                            calibrated_metrics,
                            output_dir / f"metrics_{calibrated_name}.json",
                        )
                        results.append(
                            {"method": calibrated_name, **calibrated_metrics}
                        )
        with log_stage(logger, "Save benchmark summary"):
            results_frame = pd.DataFrame(results)
            results_frame.to_csv(
                output_dir / "benchmark_results.csv", index=False
            )
            logger.info(
                "Benchmark methods complete | %s",
                ", ".join(results_frame["method"].astype(str).tolist()),
            )
        elapsed_seconds = float(time.perf_counter() - run_started)
        run_metadata.update(
            {
                "status": "completed",
                "elapsed_seconds": elapsed_seconds,
                "methods": results_frame["method"].astype(str).tolist(),
                "n_clients": len(clients),
                "n_features": len(features),
                "train_rows": int(len(train)),
                "validation_rows": int(len(val)),
                "test_rows": int(len(test)),
                "process_peak_rss_mb": process_peak_rss_mb(),
            }
        )
        write_json(run_metadata, output_dir / "run_metadata.json")
        logger.info(
            "FAP-Bench run complete | output=%s | elapsed=%s",
            output_dir.resolve(),
            format_seconds(elapsed_seconds),
        )
        return output_dir
    except Exception as exc:
        elapsed_seconds = float(time.perf_counter() - run_started)
        run_metadata.update(
            {
                "status": "failed",
                "elapsed_seconds": elapsed_seconds,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        write_json(run_metadata, output_dir / "run_metadata.json")
        logger.exception(
            "FAP-Bench run failed | elapsed=%s",
            format_seconds(elapsed_seconds),
        )
        raise
    finally:
        close_logger(logger)


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

    # Support a unified CLI: when a grid YAML is passed through --config,
    # dispatch to the grid runner instead of failing later on missing data keys.
    if "base_config" in config and "scenarios" in config:
        from fapbench.grid import run_grid

        output = run_grid(config)
        print(f"Grid results written to {output.resolve()}")
    else:
        output = run(config)
        print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    main()
