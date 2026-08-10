from __future__ import annotations

from pathlib import Path

import yaml

from scripts.run_publication_experiments import (
    EXPANDED_FEATURES_PHASE,
    _ensure_expanded_panel,
    _effective_grid,
    _resolve_effective_data_path,
)


def test_data_path_override_skips_expanded_features_phase(tmp_path: Path) -> None:
    plan_path = tmp_path / "icaif_publication.yaml"
    plan_path.write_text("phases: []", encoding="utf-8")

    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump({"data": {"path": "/original/panel.parquet"}}),
        encoding="utf-8",
    )
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(
        yaml.safe_dump({"base_config": "base.yaml", "scenarios": [], "seeds": [11]}),
        encoding="utf-8",
    )

    expanded_phase = {"name": EXPANDED_FEATURES_PHASE, "grid": "grid.yaml"}
    config = _effective_grid(
        plan_path, expanded_phase, output_root=tmp_path, data_path=None
    )
    # No override requested for this phase -> base_overrides must not appear.
    assert "base_overrides" not in config
    assert _resolve_effective_data_path(config) == Path("/original/panel.parquet")

    other_phase = {"name": "core", "grid": "grid.yaml"}
    overridden = _effective_grid(
        plan_path,
        other_phase,
        output_root=tmp_path,
        data_path=Path("/override/panel.parquet"),
    )
    assert _resolve_effective_data_path(overridden) == Path("/override/panel.parquet")


def test_ensure_expanded_panel_skips_existing_file(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "expanded.parquet"
    target.write_bytes(b"placeholder")
    calls = []
    monkeypatch.setattr(
        "scripts.run_publication_experiments.data_recipe2",
        lambda **kwargs: calls.append(kwargs),
    )
    _ensure_expanded_panel(
        target, raw_data_dir=tmp_path, rebuild=False, log_level="ERROR"
    )
    assert calls == []


def test_ensure_expanded_panel_builds_when_missing(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "expanded.parquet"
    calls = []
    monkeypatch.setattr(
        "scripts.run_publication_experiments.data_recipe2",
        lambda **kwargs: calls.append(kwargs),
    )
    _ensure_expanded_panel(
        target, raw_data_dir=tmp_path, rebuild=False, log_level="ERROR"
    )
    assert len(calls) == 1
    assert calls[0]["include_expanded_characteristics"] is True
    assert calls[0]["output_path"] == target


def test_ensure_expanded_panel_rebuilds_when_forced(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "expanded.parquet"
    target.write_bytes(b"placeholder")
    calls = []
    monkeypatch.setattr(
        "scripts.run_publication_experiments.data_recipe2",
        lambda **kwargs: calls.append(kwargs),
    )
    _ensure_expanded_panel(
        target, raw_data_dir=tmp_path, rebuild=True, log_level="ERROR"
    )
    assert len(calls) == 1


def test_extended_phases_are_opt_in_but_complete():
    """The plan must be the complete manifest, without slowing the default run.

    Two things can go wrong and both are silent. If the extended grids leak into
    the default `phases`, the documented publication command grows from hours to
    days. If they are missing from the plan entirely, the paper's claim that every
    table regenerates from one command is false, because nine grids would only be
    discoverable by reading the README.
    """
    import yaml

    plan = yaml.safe_load(
        Path("configs/icaif_publication.yaml").read_text(encoding="utf-8")
    )
    default_names = {str(phase["name"]) for phase in plan["phases"]}
    extended_names = {str(phase["name"]) for phase in plan["extended_phases"]}
    assert not (default_names & extended_names), "a phase is listed in both lists"
    # Every grid config shipped for the ICAIF suite must appear in exactly one of
    # the two lists, so adding a grid without registering it fails here.
    shipped = {
        path.name
        for path in Path("configs").glob("icaif_*.yaml")
        if path.name != "icaif_publication.yaml"
    }
    registered = {
        str(phase["grid"])
        for phase in list(plan["phases"]) + list(plan["extended_phases"])
    }
    assert shipped == registered, (
        f"unregistered grids: {sorted(shipped - registered)}; "
        f"missing configs: {sorted(registered - shipped)}"
    )
