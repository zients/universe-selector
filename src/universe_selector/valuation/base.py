from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from universe_selector.valuation.models import ValuationModelInput, ValuationScenarioResult


class ValuationModel(Protocol):
    model_id: str

    def validate_assumptions(self, assumptions: Mapping[str, object]) -> object:
        raise NotImplementedError

    def value(self, model_input: ValuationModelInput) -> tuple[ValuationScenarioResult, ...]:
        raise NotImplementedError
