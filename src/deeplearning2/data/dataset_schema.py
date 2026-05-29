"""Authoritative dataset schema expectations for the curated SQLite entrypoint."""

from __future__ import annotations


CORE_COLUMNS = (
    "species_id",
    "species_name",
    "genus",
    "family",
    "taxon_group_l1",
    "taxon_group_l2",
    "taxon_group_l3",
    "organism_lifestage",
    "primary_medium",
    "duration_h",
    "effect_type",
    "effect_level",
    "endpoint",
    "endpoint_observation",
    "is_lethal",
    "is_chronic",
    "is_threshold_endpoint",
    "is_bioaccumulation",
    "smiles",
    "target_value",
    "target_unit",
)
