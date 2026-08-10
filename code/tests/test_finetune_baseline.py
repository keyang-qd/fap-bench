"""Per-client fine-tuning after federated training: the personalized arm.

Three reviewers made the same point -- the paper's headline economic finding is a
PER-CLIENT cost, and personalization is the method class designed to reduce it, so
evaluating none of that class leaves the reader unable to tell whether the cost is
intrinsic to heterogeneity or an artifact of insisting on a single global model.

What these tests pin is that the arm is genuinely *personalized* (each client ends
up with a different model, started from the shared global one) and that it is
strictly additive: every published config predates it and must be unaffected.
"""
from __future__ import annotations

import pandas as pd
import pytest


def test_finetune_is_disabled_by_default_in_every_shipped_config():
    """Strictly additive: enabling this must be an explicit opt-in."""
    from pathlib import Path

    import yaml

    for path in sorted(Path("configs").glob("*.yaml")):
        config = yaml.safe_load(path.read_text()) or {}
        for block in (config.get("baselines") or {}, (config.get("base_overrides") or {}).get("baselines") or {}):
            finetune = block.get("finetune_after_federated")
            if finetune is not None:
                assert isinstance(finetune, dict)
                # Only the dedicated grid may turn it on.
                if finetune.get("enabled"):
                    assert path.name == "icaif_finetune_baseline.yaml", (
                        f"{path.name} enables fine-tuning; only the dedicated grid may"
                    )


def test_finetune_rejects_a_zero_epoch_budget():
    """Zero epochs would silently emit the global model under a personalized
    label -- a null result that looks like a measured one."""
    from fapbench.config import load_config
    from fapbench.runner import run

    config = load_config("configs/synthetic_smoke.yaml")
    config.setdefault("baselines", {})["finetune_after_federated"] = {
        "enabled": True,
        "algorithms": ["fedavg"],
        "epochs": 0,
    }
    with pytest.raises(ValueError, match="epochs must be >= 1"):
        run(config)


def test_finetuned_arm_runs_and_differs_from_its_base_method(tmp_path):
    """End-to-end. The fine-tuned arm must appear as its own row, carry the base
    method's communication (fine-tuning uploads nothing), and actually differ from
    the global model it started from -- otherwise it is not personalizing."""
    import yaml

    from fapbench.config import load_config
    from fapbench.runner import run

    config = load_config("configs/synthetic_smoke.yaml")
    config.setdefault("baselines", {})["finetune_after_federated"] = {
        "enabled": True,
        "algorithms": ["fedavg"],
        "epochs": 3,
    }
    config["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    run(load_config(path))
    results = pd.read_csv(tmp_path / "out" / "benchmark_results.csv")

    assert "fedavg_finetuned" in set(results.method), "personalized arm must be reported"
    tuned = results.loc[results.method == "fedavg_finetuned"].iloc[0]
    base = results.loc[results.method == "fedavg"].iloc[0]

    assert tuned["finetune_base_method"] == "fedavg"
    assert int(tuned["finetune_epochs"]) == 3
    # Fine-tuning is local: it must inherit, not inflate, communication.
    if "communicated_bytes_fp32" in results:
        assert tuned["communicated_bytes_fp32"] == base["communicated_bytes_fp32"]
    # And it must actually be a different model.
    assert tuned["rank_ic"] != base["rank_ic"], (
        "fine-tuned arm is identical to its base method; it is not personalizing"
    )


def test_finetune_only_touches_the_named_algorithms(tmp_path):
    """Naming fedavg must not silently personalize scaffold as well."""
    import yaml

    from fapbench.config import load_config
    from fapbench.runner import run

    config = load_config("configs/synthetic_smoke.yaml")
    config["federated"]["algorithms"] = ["fedavg", "scaffold"]
    config.setdefault("baselines", {})["finetune_after_federated"] = {
        "enabled": True,
        "algorithms": ["fedavg"],
        "epochs": 2,
    }
    config["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    run(load_config(path))
    methods = set(pd.read_csv(tmp_path / "out" / "benchmark_results.csv").method)
    assert "fedavg_finetuned" in methods
    assert "scaffold_finetuned" not in methods
