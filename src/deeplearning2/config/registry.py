"""Central registry for required configuration files."""

from __future__ import annotations

from pathlib import Path

from deeplearning2.paths import CONFIG_ROOT


CONFIG_FILES = {
    "project": CONFIG_ROOT / "project.yaml",
    "data": CONFIG_ROOT / "data" / "dataset.yaml",
    "tasks_semantics": CONFIG_ROOT / "tasks" / "task_semantics.yaml",
    "tasks_target_spaces": CONFIG_ROOT / "tasks" / "target_spaces.yaml",
    "features": CONFIG_ROOT / "features" / "chemical_features.yaml",
    "models_baseline": CONFIG_ROOT / "models" / "baseline.yaml",
    "models_deep": CONFIG_ROOT / "models" / "deep_multitask.yaml",
    "models_transfer": CONFIG_ROOT / "models" / "transfer_learning.yaml",
    "training_joint": CONFIG_ROOT / "training" / "joint_training.yaml",
    "evaluation": CONFIG_ROOT / "evaluation" / "evaluation.yaml",
    "inference": CONFIG_ROOT / "inference" / "prediction.yaml",
    "ssd": CONFIG_ROOT / "evaluation" / "ssd.yaml",
    "ablation": CONFIG_ROOT / "experiments" / "ablation_matrix.yaml",
    "experiment_baseline": CONFIG_ROOT / "experiments" / "baseline" / "baseline_water.yaml",
    "experiment_deep": CONFIG_ROOT / "experiments" / "deep" / "direct_joint_water.yaml",
    "experiment_transfer": CONFIG_ROOT / "experiments" / "transfer" / "pretrain_water_finetune_soil.yaml",
    "experiment_ablation": CONFIG_ROOT / "experiments" / "ablation" / "ablation_core_axes.yaml",
}

REQUIRED_CONFIG_FILES = tuple(CONFIG_FILES.values())
