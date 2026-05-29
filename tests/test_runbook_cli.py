import os
import json
import subprocess
import sys
from pathlib import Path

from deeplearning2.config.loader import (
    EXPERIMENT_FAMILIES,
    discover_experiment_configs,
    load_experiment_launch_spec,
    load_experiment_records,
    summarize_experiment_families,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(SRC_ROOT), str(PROJECT_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def test_loader_discovers_expected_experiment_families() -> None:
    configs = discover_experiment_configs()
    assert configs, "Expected experiment configs under configs/experiments."

    records = load_experiment_records()
    assert len(records) == len(configs)
    assert {record.family for record in records} == set(EXPERIMENT_FAMILIES)


def test_loader_extracts_known_metadata() -> None:
    records = {record.family: record for record in load_experiment_records()}

    assert records["baseline"].experiment_id == "baseline_water_screen"
    assert records["baseline"].split == "scaffold_holdout"
    assert records["deep"].experiment_id == "deep_direct_joint_water"
    assert records["transfer"].experiment_id == "transfer_pretrain_water_finetune_soil"
    assert records["ablation"].summary == "axes=3"


def test_loader_builds_launch_spec_for_baseline_config() -> None:
    spec = load_experiment_launch_spec(PROJECT_ROOT / "configs" / "experiments" / "baseline" / "baseline_water.yaml")

    assert spec.family == "baseline"
    assert spec.experiment_id == "baseline_water_screen"
    assert spec.medium_scope == "water"
    assert spec.split == "scaffold_holdout"
    assert spec.body["models"][0] == "ridge"


def test_summary_counts_each_family_once() -> None:
    summary = summarize_experiment_families(load_experiment_records())

    for family in EXPERIMENT_FAMILIES:
        assert summary[family]["count"] == 1
    assert summary["baseline"]["with_split"] == 1
    assert summary["ablation"]["with_split"] == 0


def test_runs_list_json_output() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deeplearning2.cli.main", "runs", "list", "--format", "json"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    payload = json.loads(result.stdout)
    assert len(payload) == 4
    assert payload[0]["experiment_id"]
    assert {item["family"] for item in payload} == set(EXPERIMENT_FAMILIES)


def test_runs_summary_text_output() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deeplearning2.cli.main", "runs", "summary"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    assert "family=baseline | count=1" in result.stdout
    assert "family=transfer | count=1" in result.stdout


def test_list_experiments_script_outputs_concise_inventory() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/list_experiments.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 4
    assert any(line.startswith("baseline\tbaseline_water_screen\t") for line in lines)


def test_runs_launch_outputs_baseline_placeholder_reports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deeplearning2.cli.main",
            "runs",
            "launch",
            "--family",
            "baseline",
            "--config",
            str(PROJECT_ROOT / "configs" / "experiments" / "baseline" / "baseline_water.yaml"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    payload = json.loads(result.stdout)
    assert len(payload) == 6
    assert payload[0]["runner_family"] == "baseline"
    assert payload[0]["config"]["medium_scope"] == "water"
    assert payload[0]["config"]["extra"]["launch_mode"] == "placeholder_baseline_launch"
