from __future__ import annotations

from pathlib import Path

from universe_selector.domain import Market, canonical_ticker
from universe_selector.errors import ValidationError
from universe_selector.providers.registry import get_fundamentals_registration
from universe_selector.valuation.assumptions import load_valuation_assumptions
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    FcfDcfV1Assumptions,
    ValuationInputProvenance,
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
    starting_fcf, starting_fcf_source, starting_fcf_note = _resolve_starting_fcf(
        model_id=model_id,
        model_assumptions=assumptions.model_assumptions,
        provider_fcf=facts.free_cash_flow,
        fiscal_period_type=facts.fiscal_period_type,
    )

    effective_inputs = EffectiveValuationInputs(
        starting_fcf=starting_fcf,
        shares_outstanding=_effective_value("shares_outstanding", facts.shares_outstanding, assumptions.facts_overrides),
        net_debt=_effective_value("net_debt", facts.net_debt, assumptions.facts_overrides),
        reference_price=_effective_value("reference_price", facts.reference_price, assumptions.facts_overrides),
        currency=facts.currency,
        fiscal_period_type=facts.fiscal_period_type,
        fiscal_period_end=facts.fiscal_period_end,
        reference_price_as_of=(
            assumptions.as_of
            if assumptions.facts_overrides.get("reference_price") is not None
            else facts.reference_price_as_of
        ),
        reference_price_as_of_source=(
            "assumption_override"
            if assumptions.facts_overrides.get("reference_price") is not None
            else facts.reference_price_as_of_source
        ),
        reference_price_as_of_note=(
            assumptions.facts_override_notes.get("reference_price")
            if assumptions.facts_overrides.get("reference_price") is not None
            else facts.reference_price_as_of_note
        ),
    )
    provenance = ValuationInputProvenance(
        starting_fcf_source=starting_fcf_source,
        shares_outstanding_source=_source_for("shares_outstanding", assumptions.facts_overrides),
        net_debt_source=_source_for("net_debt", assumptions.facts_overrides),
        reference_price_source=_source_for("reference_price", assumptions.facts_overrides),
        starting_fcf_note=starting_fcf_note,
        shares_outstanding_note=_note_for("shares_outstanding", assumptions.facts_overrides, assumptions.facts_override_notes),
        net_debt_note=_note_for("net_debt", assumptions.facts_overrides, assumptions.facts_override_notes),
        reference_price_note=_note_for("reference_price", assumptions.facts_overrides, assumptions.facts_override_notes),
    )
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


def _effective_value(field: str, provider_value: float, overrides) -> float:
    override = overrides.get(field)
    if override is not None:
        return override
    return provider_value


def _resolve_starting_fcf(
    *,
    model_id: str,
    model_assumptions: object,
    provider_fcf: float,
    fiscal_period_type: str,
) -> tuple[float, str, str | None]:
    if model_id != "fcf_dcf_v1":
        return provider_fcf, "provider_fact", None
    if not isinstance(model_assumptions, FcfDcfV1Assumptions):
        raise ValidationError("fcf_dcf_v1 requires FcfDcfV1Assumptions")

    starting_fcf = model_assumptions.starting_fcf
    if starting_fcf.method == "override":
        assert starting_fcf.value is not None
        return starting_fcf.value, "assumption_override", starting_fcf.note
    if starting_fcf.method == "provider_ttm_fcf":
        return (
            provider_fcf,
            "provider_ttm_fcf",
            f"Provider raw FCF used as starting FCF proxy; fiscal_period_type={fiscal_period_type}.",
        )
    raise ValidationError(f"unsupported starting_fcf method: {starting_fcf.method}")


def _source_for(field: str, overrides) -> str:
    if overrides.get(field) is not None:
        return "assumption_override"
    return "provider_fact"


def _note_for(field: str, overrides, notes) -> str | None:
    if overrides.get(field) is not None:
        return notes.get(field)
    return None
