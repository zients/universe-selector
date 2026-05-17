from __future__ import annotations

from pathlib import Path

from universe_selector.domain import Market, canonical_ticker
from universe_selector.errors import ValidationError
from universe_selector.providers.registry import get_fundamentals_registration
from universe_selector.valuation.assumptions import load_valuation_assumptions
from universe_selector.valuation.models import (
    ValuationModelInput,
    ValuationResult,
    ValuationRunInput,
)
from universe_selector.valuation.registry import get_valuation_model


def run_valuation(
    market: Market,
    ticker: str,
    model_id: str,
    assumptions_path: Path | None,
    fundamentals_provider_id: str,
) -> ValuationResult:
    normalized_ticker = canonical_ticker(ticker)
    model = get_valuation_model(model_id)
    registration = get_fundamentals_registration(fundamentals_provider_id, market)
    assumptions = load_valuation_assumptions(
        market=market,
        ticker=normalized_ticker,
        model_id=model_id,
        assumptions_path=assumptions_path,
    )
    provider = registration.factory()
    fundamentals = provider.load_fundamentals(market, normalized_ticker)
    facts = fundamentals.facts
    if assumptions.currency != facts.currency:
        raise ValidationError(
            f"assumptions currency {assumptions.currency} must match provider facts currency {facts.currency}"
        )
    effective_inputs, provenance = model.build_inputs(facts=facts, assumptions=assumptions)
    run_input = ValuationRunInput(
        market=market,
        ticker=normalized_ticker,
        model_id=model_id,
        fundamentals_metadata=fundamentals.metadata,
        raw_facts=facts,
        effective_inputs=effective_inputs,
        input_provenance=provenance,
        assumptions=assumptions,
    )
    scenario_results = model.value(
        ValuationModelInput(
            market=market,
            ticker=normalized_ticker,
            model_id=model_id,
            effective_inputs=effective_inputs,
            model_assumptions=assumptions.model_assumptions,
        )
    )
    return ValuationResult(run_input=run_input, scenario_results=scenario_results)
