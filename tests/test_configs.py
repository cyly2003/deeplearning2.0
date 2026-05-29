from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}


def _config_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in CONFIG_SUFFIXES
    )


def _read_text_bundle(path: Path) -> str:
    parts = []
    for file_path in _config_files(path):
        parts.append(file_path.read_text(encoding="utf-8", errors="ignore").lower())
    return "\n".join(parts)


def _assert_tokens_present(text: str, tokens: list[str], scope: str) -> None:
    missing = [token for token in tokens if token.lower() not in text]
    assert not missing, f"Missing expected config tokens in {scope}: {', '.join(missing)}"


def test_config_root_exists() -> None:
    assert CONFIG_ROOT.is_dir(), (
        "Expected a top-level 'configs' directory. "
        "This project uses configuration as the contract for tasks, models, and experiments."
    )


def test_core_config_namespaces_exist() -> None:
    required_dirs = [
        CONFIG_ROOT / "data",
        CONFIG_ROOT / "tasks",
        CONFIG_ROOT / "models",
        CONFIG_ROOT / "experiments",
    ]

    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_dirs if not path.is_dir()]
    assert not missing, (
        "Missing required config namespaces: "
        + ", ".join(missing)
        + ". Keep data/task/model/experiment concerns separated for reproducible research."
    )


def test_data_config_points_to_curated_sqlite_view() -> None:
    data_dir = CONFIG_ROOT / "data"
    files = _config_files(data_dir)
    assert files, "Expected at least one configuration file under 'configs/data'."

    text = _read_text_bundle(data_dir)
    _assert_tokens_present(
        text,
        ["ecotox_clean.sqlite", "ecotox_toxicity_joined_curated"],
        "configs/data",
    )


def test_task_config_matches_authoritative_task_semantics() -> None:
    task_dir = CONFIG_ROOT / "tasks"
    files = _config_files(task_dir)
    assert files, "Expected at least one configuration file under 'configs/tasks'."

    text = _read_text_bundle(task_dir)
    _assert_tokens_present(
        text,
        ["species", "effect_type", "endpoint_observation", "effect_level"],
        "configs/tasks",
    )
    assert "nr" not in text, "Task configs should not register NR as a formal modeling task."


def test_model_config_reflects_grouped_rdkit_and_morgan_design() -> None:
    model_dir = CONFIG_ROOT / "models"
    files = _config_files(model_dir)
    assert files, "Expected at least one configuration file under 'configs/models'."

    text = _read_text_bundle(model_dir)
    _assert_tokens_present(
        text,
        ["rdkit", "morgan", "descriptor", "group", "context", "duration_h"],
        "configs/models",
    )


def test_experiment_config_layers_cover_research_lines() -> None:
    experiments_dir = CONFIG_ROOT / "experiments"
    required_subdirs = [
        experiments_dir / "baseline",
        experiments_dir / "deep",
        experiments_dir / "transfer",
        experiments_dir / "ablation",
    ]

    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_subdirs if not path.is_dir()]
    assert not missing, (
        "Missing experiment config layers: "
        + ", ".join(missing)
        + ". The repository should reserve separate spaces for baseline, deep, transfer, and ablation studies."
    )

    baseline_text = _read_text_bundle(experiments_dir / "baseline")
    deep_text = _read_text_bundle(experiments_dir / "deep")
    transfer_text = _read_text_bundle(experiments_dir / "transfer")
    ablation_text = _read_text_bundle(experiments_dir / "ablation")

    assert baseline_text, "Expected at least one baseline experiment config file."
    assert deep_text, "Expected at least one deep experiment config file."
    assert transfer_text, "Expected at least one transfer experiment config file."
    assert ablation_text, "Expected at least one ablation experiment config file."

    _assert_tokens_present(
        baseline_text,
        ["ridge", "randomforest"],
        "configs/experiments/baseline",
    )
    _assert_tokens_present(
        deep_text,
        ["rdkit", "morgan", "context"],
        "configs/experiments/deep",
    )
    _assert_tokens_present(
        transfer_text,
        ["pretrain", "finetune"],
        "configs/experiments/transfer",
    )
    _assert_tokens_present(
        ablation_text,
        ["rdkit only", "morgan only", "grouped"],
        "configs/experiments/ablation",
    )
