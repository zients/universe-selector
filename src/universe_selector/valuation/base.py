from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from universe_selector.providers.models import FundamentalFacts
    from universe_selector.valuation.models import (
        EffectiveValuationInputs,
        ValuationAssumptionSet,
        ValuationInputProvenance,
        ValuationModelInput,
        ValuationResult,
        ValuationScenarioResult,
    )


class ValuationModel(Protocol):
    model_id: str

    def validate_assumptions(self, assumptions: Mapping[str, object]) -> object:
        raise NotImplementedError

    def build_inputs(
        self,
        *,
        facts: FundamentalFacts,
        assumptions: ValuationAssumptionSet,
    ) -> tuple[EffectiveValuationInputs, ValuationInputProvenance]:
        raise NotImplementedError

    def value(self, model_input: ValuationModelInput) -> tuple[ValuationScenarioResult, ...]:
        raise NotImplementedError


class ValuationOutputRenderer(Protocol):
    def render_risk_disclosures(self, result: ValuationResult) -> list[str]:
        raise NotImplementedError

    def render_model_assumptions(self, result: ValuationResult) -> list[str]:
        raise NotImplementedError

    def render_effective_inputs(self, result: ValuationResult) -> list[str]:
        raise NotImplementedError

    def render_input_provenance(self, result: ValuationResult) -> list[str]:
        raise NotImplementedError

    def render_scenario_results(self, result: ValuationResult) -> list[str]:
        raise NotImplementedError
