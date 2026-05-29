"""Task definition helpers and shared task-semantics contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deeplearning2.config.loader import load_yaml_document
from deeplearning2.config.registry import CONFIG_FILES


TASK_ID_DEFINITION = "task = species + endpoint semantics"
TASK_ID_COMPONENTS = (
    "species",
    "effect_type",
    "endpoint_observation",
)
ENDPOINT_FLAGS = (
    "is_lethal",
    "is_chronic",
    "is_threshold_endpoint",
    "is_bioaccumulation",
)
CANONICAL_EFFECT_TYPE_OVERRIDES = {
    ("LC", "mortality"): "EC",
}


@dataclass(frozen=True)
class TaskSemanticsContract:
    """Authoritative task-semantics definition loaded from configuration."""

    formal_task_definition: str
    task_id_components: tuple[str, ...]
    allowed_effect_types: tuple[str, ...]
    allowed_endpoint_observations: tuple[str, ...]
    endpoint_flags: tuple[str, ...]
    effect_level_role: str
    effect_level_part_of_task_id: bool
    effect_level_required_for: tuple[str, ...]
    effect_level_forbidden_for: tuple[str, ...]
    exclude_not_reported_records: bool

    def __post_init__(self) -> None:
        if self.formal_task_definition != TASK_ID_DEFINITION:
            raise ValueError(
                "Task semantics must stay aligned with 'task = species + endpoint semantics'."
            )
        if self.task_id_components != TASK_ID_COMPONENTS:
            raise ValueError(
                f"task_id_components must remain {TASK_ID_COMPONENTS}, got {self.task_id_components}."
            )
        if self.effect_level_part_of_task_id:
            raise ValueError("effect_level must not be part of the formal task ID.")
        if not self.exclude_not_reported_records:
            raise ValueError("NR exclusions must remain enabled for formal modeling.")


def load_task_semantics_contract(
    config_path: Path = CONFIG_FILES["tasks_semantics"],
) -> TaskSemanticsContract:
    """Load the authoritative task-semantics contract from configuration."""

    payload = load_yaml_document(config_path)
    body = payload.get("task_semantics")
    if not isinstance(body, dict):
        raise ValueError(f"Expected top-level 'task_semantics' mapping in {config_path}.")

    effect_level = body.get("effect_level", {})
    if not isinstance(effect_level, dict):
        raise ValueError(f"Expected 'effect_level' mapping in {config_path}.")

    endpoint_semantics = body.get("endpoint_semantics", {})
    if not isinstance(endpoint_semantics, dict):
        raise ValueError(f"Expected 'endpoint_semantics' mapping in {config_path}.")

    return TaskSemanticsContract(
        formal_task_definition=str(body.get("formal_task_definition")),
        task_id_components=tuple(body.get("task_id_components", ())),
        allowed_effect_types=tuple(endpoint_semantics.get("effect_type", ())),
        allowed_endpoint_observations=tuple(endpoint_semantics.get("endpoint_observation", ())),
        endpoint_flags=tuple(body.get("flags", ())),
        effect_level_role=str(effect_level.get("role")),
        effect_level_part_of_task_id=bool(effect_level.get("part_of_task_id", True)),
        effect_level_required_for=tuple(effect_level.get("required_for", ())),
        effect_level_forbidden_for=tuple(effect_level.get("forbidden_for", ())),
        exclude_not_reported_records=bool(
            body.get("exclusions", {}).get("exclude_not_reported_records", False)
        ),
    )


def canonicalize_task_effect_type(effect_type: str, endpoint_observation: str) -> str:
    """Map raw effect types into the formal task-semantic identity space."""

    return CANONICAL_EFFECT_TYPE_OVERRIDES.get((effect_type, endpoint_observation), effect_type)


def validate_task_selection(
    species_id: str,
    effect_type: str,
    endpoint_observation: str,
    *,
    effect_level: str | int | None = None,
    contract: TaskSemanticsContract | None = None,
) -> None:
    """Validate task-defining inputs against the formal task semantics."""

    semantics = contract or load_task_semantics_contract()
    if not species_id:
        raise ValueError("species_id must be a non-empty string.")
    if effect_type not in semantics.allowed_effect_types:
        raise ValueError(
            f"Unsupported effect_type={effect_type!r}. Expected one of {semantics.allowed_effect_types}."
        )
    if endpoint_observation not in semantics.allowed_endpoint_observations:
        raise ValueError(
            "Unsupported endpoint_observation="
            f"{endpoint_observation!r}. Expected one of {semantics.allowed_endpoint_observations}."
        )
    if effect_type in semantics.effect_level_required_for and effect_level is None:
        raise ValueError(f"effect_level is required for effect_type={effect_type!r}.")
    if effect_type in semantics.effect_level_forbidden_for and effect_level is not None:
        raise ValueError(f"effect_level must not be provided for effect_type={effect_type!r}.")


def build_task_id(
    species_id: str,
    effect_type: str,
    endpoint_observation: str,
) -> str:
    """Build the formal task ID from species plus canonical endpoint semantics."""

    canonical_effect_type = canonicalize_task_effect_type(effect_type, endpoint_observation)
    return f"{species_id}__{canonical_effect_type}_{endpoint_observation}"
