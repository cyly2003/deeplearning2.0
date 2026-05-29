"""Experiment configuration discovery and metadata loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from deeplearning2.paths import CONFIG_ROOT


EXPERIMENTS_ROOT = CONFIG_ROOT / "experiments"
EXPERIMENT_FAMILIES = ("baseline", "deep", "transfer", "ablation")
YAML_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class ExperimentRecord:
    """Normalized metadata extracted from an experiment config."""

    experiment_id: str
    family: str
    config_path: Path
    medium_scope: str | None = None
    split: str | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, str | None]:
        """Convert the record to a JSON-serializable payload."""

        return {
            "experiment_id": self.experiment_id,
            "family": self.family,
            "config_path": str(self.config_path),
            "medium_scope": self.medium_scope,
            "split": self.split,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ExperimentLaunchSpec:
    """Minimal launch-ready experiment configuration view."""

    experiment_id: str
    family: str
    config_path: Path
    medium_scope: str | None
    split: str | None
    body: dict[str, Any]


def discover_experiment_configs(root: Path = EXPERIMENTS_ROOT) -> list[Path]:
    """Return experiment config files from family subdirectories."""

    paths: list[Path] = []
    for family in EXPERIMENT_FAMILIES:
        family_dir = root / family
        if not family_dir.is_dir():
            continue
        for suffix in YAML_SUFFIXES:
            paths.extend(sorted(family_dir.glob(f"*{suffix}")))
    return sorted(set(paths))


def load_yaml_document(path: Path) -> dict[str, Any]:
    """Load a YAML document and return a dictionary payload."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping document in {path}.")
    return payload


def _infer_family_from_path(path: Path) -> str:
    for family in EXPERIMENT_FAMILIES:
        if family in path.parts:
            return family
    raise ValueError(f"Could not infer experiment family from path: {path}")


def _build_summary(family: str, body: dict[str, Any]) -> str:
    if family == "baseline":
        models = body.get("models", [])
        return f"models={len(models)}"
    if family == "deep":
        architecture = body.get("architecture", {})
        enabled = sorted(str(key) for key, value in architecture.items() if value)
        return "features=" + ",".join(enabled)
    if family == "transfer":
        stages = body.get("stages", [])
        pretrain_scope = body.get("pretrain_scope")
        finetune_scope = body.get("finetune_scope")
        return (
            f"stages={len(stages)}"
            f";pretrain={pretrain_scope or 'na'}"
            f";finetune={finetune_scope or 'na'}"
        )
    if family == "ablation":
        axes = body.get("axes", {})
        return f"axes={len(axes)}"
    return ""


def extract_experiment_record(path: Path) -> ExperimentRecord:
    """Load a config file and normalize its experiment metadata."""

    payload = load_yaml_document(path)
    family = _infer_family_from_path(path)
    body = payload.get("experiment")
    if not isinstance(body, dict):
        raise ValueError(f"Expected top-level 'experiment' mapping in {path}.")

    experiment_id = body.get("id")
    if not experiment_id:
        raise ValueError(f"Missing experiment.id in {path}.")

    declared_family = body.get("family")
    if declared_family and declared_family != family:
        raise ValueError(
            f"Experiment family mismatch in {path}: declared={declared_family!r}, inferred={family!r}."
        )

    return ExperimentRecord(
        experiment_id=str(experiment_id),
        family=family,
        config_path=path,
        medium_scope=body.get("medium_scope"),
        split=body.get("split"),
        summary=_build_summary(family, body),
    )


def load_experiment_records(root: Path = EXPERIMENTS_ROOT) -> list[ExperimentRecord]:
    """Discover and load all experiment metadata records."""

    return [extract_experiment_record(path) for path in discover_experiment_configs(root)]


def load_experiment_launch_spec(path: Path | str) -> ExperimentLaunchSpec:
    """Load a launch-ready experiment configuration payload."""

    resolved_path = Path(path)
    payload = load_yaml_document(resolved_path)
    family = _infer_family_from_path(resolved_path)
    body = payload.get("experiment")
    if not isinstance(body, dict):
        raise ValueError(f"Expected top-level 'experiment' mapping in {resolved_path}.")

    experiment_id = body.get("id")
    if not experiment_id:
        raise ValueError(f"Missing experiment.id in {resolved_path}.")

    declared_family = body.get("family")
    if declared_family and declared_family != family:
        raise ValueError(
            "Experiment family mismatch in "
            f"{resolved_path}: declared={declared_family!r}, inferred={family!r}."
        )

    return ExperimentLaunchSpec(
        experiment_id=str(experiment_id),
        family=family,
        config_path=resolved_path,
        medium_scope=body.get("medium_scope"),
        split=body.get("split"),
        body=body,
    )


def summarize_experiment_families(
    records: list[ExperimentRecord],
) -> dict[str, dict[str, int]]:
    """Aggregate experiment counts by family and declared metadata."""

    summary: dict[str, dict[str, int]] = {}
    for family in EXPERIMENT_FAMILIES:
        family_records = [record for record in records if record.family == family]
        summary[family] = {
            "count": len(family_records),
            "with_split": sum(1 for record in family_records if record.split),
            "with_medium_scope": sum(1 for record in family_records if record.medium_scope),
        }
    return summary
