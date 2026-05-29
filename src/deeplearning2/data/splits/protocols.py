"""Split protocol definitions used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deeplearning2.config.loader import load_yaml_document
from deeplearning2.config.registry import CONFIG_FILES


SPLIT_PROTOCOLS = (
    "scaffold_holdout",
    "chemical_id_holdout",
    "medium_transfer_split",
)
PRIMARY_SPLIT_PROTOCOL = "scaffold_holdout"


@dataclass(frozen=True)
class SplitProtocolSpec:
    """Formal split specification shared across all model families."""

    split_name: str
    split_group: str
    purpose: str
    is_primary: bool
    supports_scenarios: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split_name not in SPLIT_PROTOCOLS:
            raise ValueError(
                f"Unsupported split_name={self.split_name!r}. Expected one of {SPLIT_PROTOCOLS}."
            )


def load_primary_evaluation_scenarios(
    config_path: Path = CONFIG_FILES["evaluation"],
) -> tuple[str, ...]:
    """Load the evaluation scenarios that every split contract must support."""

    payload = load_yaml_document(config_path)
    body = payload.get("evaluation")
    if not isinstance(body, dict):
        raise ValueError(f"Expected top-level 'evaluation' mapping in {config_path}.")
    return tuple(body.get("primary_scenarios", ()))


def build_split_protocol_manifest() -> tuple[SplitProtocolSpec, ...]:
    """Return the formal split manifest for all research lines."""

    scenarios = load_primary_evaluation_scenarios()
    return (
        SplitProtocolSpec(
            split_name="scaffold_holdout",
            split_group="primary_research_split",
            purpose="known_species_new_chemicals_generalization",
            is_primary=True,
            supports_scenarios=scenarios,
        ),
        SplitProtocolSpec(
            split_name="chemical_id_holdout",
            split_group="supplementary_holdout",
            purpose="chemical_identity_generalization_cross_check",
            is_primary=False,
            supports_scenarios=scenarios,
        ),
        SplitProtocolSpec(
            split_name="medium_transfer_split",
            split_group="transfer_research_split",
            purpose="cross_medium_transfer",
            is_primary=False,
            supports_scenarios=scenarios,
        ),
    )


def get_split_protocol(split_name: str) -> SplitProtocolSpec:
    """Lookup a single split protocol from the formal manifest."""

    for spec in build_split_protocol_manifest():
        if spec.split_name == split_name:
            return spec
    raise ValueError(f"Unknown split_name={split_name!r}.")
