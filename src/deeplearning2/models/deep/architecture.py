"""High-level architecture contract for the multitask residual QSAR model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualArchitectureSpec:
    """Contract for the first-pass unified residual architecture."""

    chemical_encoder: str = "grouped_rdkit_plus_morgan"
    context_encoder: str = "structured_context_encoder"
    rule_adapter: str = "optional_rule_adapter"
    prediction_form: str = "y_pred = y_chemical + alpha * delta_context + beta * delta_rule"
