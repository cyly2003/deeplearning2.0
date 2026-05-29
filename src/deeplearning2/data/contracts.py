"""Shared contracts for the curated dataset entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deeplearning2.config.loader import load_yaml_document
from deeplearning2.config.registry import CONFIG_FILES
from deeplearning2.data.dataset_schema import CORE_COLUMNS


@dataclass(frozen=True)
class DatasetEntrypointContract:
    """Authoritative dataset access contract for every research line."""

    source_type: str
    sqlite_path: Path
    curated_view: str
    rebuild_raw_join_by_default: bool
    required_core_columns: tuple[str, ...]
    target_value_column: str
    target_unit_column: str
    task_expression: str
    effect_level_is_task_id_component: bool
    exclude_nr: bool

    def __post_init__(self) -> None:
        if self.source_type != "sqlite":
            raise ValueError(f"Unsupported source_type={self.source_type!r}.")
        if not self.curated_view:
            raise ValueError("curated_view must be a non-empty SQLite relation name.")
        if not self.required_core_columns:
            raise ValueError("required_core_columns must not be empty.")
        if self.task_expression != "species + endpoint_semantics":
            raise ValueError(
                "Dataset task_expression must stay aligned with the project task contract."
            )
        if self.effect_level_is_task_id_component:
            raise ValueError("effect_level must not be part of the task ID.")
        if not self.exclude_nr:
            raise ValueError("NR exclusions must remain enabled for the formal dataset entrypoint.")


def load_dataset_entrypoint_contract(
    config_path: Path = CONFIG_FILES["data"],
) -> DatasetEntrypointContract:
    """Load the authoritative dataset contract from configuration."""

    payload = load_yaml_document(config_path)
    body = payload.get("dataset")
    if not isinstance(body, dict):
        raise ValueError(f"Expected top-level 'dataset' mapping in {config_path}.")

    target_columns = body.get("target_columns", {})
    if not isinstance(target_columns, dict):
        raise ValueError(f"Expected 'target_columns' mapping in {config_path}.")

    task_definition = body.get("task_definition", {})
    if not isinstance(task_definition, dict):
        raise ValueError(f"Expected 'task_definition' mapping in {config_path}.")

    required_columns = tuple(body.get("required_core_columns", ()))
    if required_columns != CORE_COLUMNS[:-2]:
        raise ValueError(
            "configs/data/dataset.yaml required_core_columns must match CORE_COLUMNS without "
            "the target value/unit columns."
        )

    return DatasetEntrypointContract(
        source_type=str(body.get("source_type")),
        sqlite_path=Path(str(body.get("sqlite_path"))),
        curated_view=str(body.get("curated_view")),
        rebuild_raw_join_by_default=bool(body.get("rebuild_raw_join_by_default", False)),
        required_core_columns=required_columns,
        target_value_column=str(target_columns.get("value")),
        target_unit_column=str(target_columns.get("unit")),
        task_expression=str(task_definition.get("expression")),
        effect_level_is_task_id_component=bool(
            task_definition.get("effect_level_is_task_id_component", True)
        ),
        exclude_nr=bool(task_definition.get("exclude_nr", False)),
    )
