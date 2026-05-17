from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from universe_selector.providers.models import FundamentalFacts
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
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
