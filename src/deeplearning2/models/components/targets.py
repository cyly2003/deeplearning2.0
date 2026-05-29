"""Target-space contracts shared across baseline, deep, and transfer lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deeplearning2.config.loader import load_yaml_document
from deeplearning2.config.registry import CONFIG_FILES
from deeplearning2.models.components.contracts import TargetContract


TARGET_SPACE_POLICY = "task_internal_unification_with_separate_training_and_output_spaces"
TARGET_SPACE_FAMILIES = (
    "EC_LC_ICx",
    "NOEC_LOEC",
    "BCF_BAF",
)
EFFECT_TYPE_TO_TARGET_FAMILY = {
    "EC": "EC_LC_ICx",
    "LC": "EC_LC_ICx",
    "ICx": "EC_LC_ICx",
    "NOEC": "NOEC_LOEC",
    "LOEC": "NOEC_LOEC",
    "BCF": "BCF_BAF",
    "BAF": "BCF_BAF",
}


@dataclass(frozen=True)
class TargetSpaceSpec:
    """Formal target-space definition for one endpoint family."""

    family_name: str
    training_space: str
    output_space: str
    effect_types: tuple[str, ...]
    uses_effect_level_features: bool

    def __post_init__(self) -> None:
        if self.family_name not in TARGET_SPACE_FAMILIES:
            raise ValueError(
                f"Unsupported family_name={self.family_name!r}. Expected one of {TARGET_SPACE_FAMILIES}."
            )
        if not self.effect_types:
            raise ValueError("effect_types must not be empty.")


def load_target_space_specs(
    config_path: Path = CONFIG_FILES["tasks_target_spaces"],
) -> tuple[TargetSpaceSpec, ...]:
    """Load target-space specifications from configuration."""

    payload = load_yaml_document(config_path)
    body = payload.get("target_spaces")
    if not isinstance(body, dict):
        raise ValueError(f"Expected top-level 'target_spaces' mapping in {config_path}.")

    policy = str(body.get("policy"))
    if policy != TARGET_SPACE_POLICY:
        raise ValueError(
            f"Unexpected target_spaces.policy={policy!r}. Expected {TARGET_SPACE_POLICY!r}."
        )

    families = body.get("families")
    if not isinstance(families, dict):
        raise ValueError(f"Expected 'families' mapping in {config_path}.")

    family_effect_types = {
        "EC_LC_ICx": ("EC", "LC", "ICx"),
        "NOEC_LOEC": ("NOEC", "LOEC"),
        "BCF_BAF": ("BCF", "BAF"),
    }

    specs: list[TargetSpaceSpec] = []
    for family_name in TARGET_SPACE_FAMILIES:
        family_body = families.get(family_name)
        if not isinstance(family_body, dict):
            raise ValueError(f"Missing target-space family {family_name!r} in {config_path}.")
        specs.append(
            TargetSpaceSpec(
                family_name=family_name,
                training_space=str(family_body.get("training_space")),
                output_space=str(family_body.get("output_space")),
                effect_types=family_effect_types[family_name],
                uses_effect_level_features=family_name == "EC_LC_ICx",
            )
        )
    return tuple(specs)


def resolve_target_space_spec(effect_type: str) -> TargetSpaceSpec:
    """Resolve a target-space specification from a formal effect type."""

    family_name = EFFECT_TYPE_TO_TARGET_FAMILY.get(effect_type)
    if family_name is None:
        raise ValueError(
            f"Unsupported effect_type={effect_type!r}. Expected one of {tuple(EFFECT_TYPE_TO_TARGET_FAMILY)}."
        )
    spec_map = {spec.family_name: spec for spec in load_target_space_specs()}
    return spec_map[family_name]


def build_target_contract(effect_type: str, *, notes: str = "") -> TargetContract:
    """Build the runner-facing target contract from formal target-space policy."""

    spec = resolve_target_space_spec(effect_type)
    return TargetContract(
        family=spec.family_name.lower(),
        training_space=spec.training_space,
        output_space=spec.output_space,
        uses_effect_level_features=spec.uses_effect_level_features,
        notes=notes,
    )
