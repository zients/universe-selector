from __future__ import annotations

import math
from collections.abc import Mapping

from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    StartingFcfAssumption,
    ValuationAssumptionSet,
    ValuationInputProvenance,
)


_SCENARIO_ORDER = ("conservative", "base", "upside")
_STARTING_FCF_KEYS = frozenset({"method", "value", "note"})


def parse_starting_fcf(value: object) -> StartingFcfAssumption:
    if not isinstance(value, Mapping):
        raise ValidationError("starting_fcf must be a mapping")
    unknown_keys = sorted(set(value) - _STARTING_FCF_KEYS)
    if unknown_keys:
        raise ValidationError(f"unknown starting_fcf key: {unknown_keys[0]}")

    method = value.get("method")
    if method == "provider_ttm_fcf":
        if "value" in value:
            raise ValidationError("starting_fcf.value is only allowed when method is override")
        if "note" in value:
            raise ValidationError("starting_fcf.note is only allowed when method is override")
        return StartingFcfAssumption(method="provider_ttm_fcf", value=None, note=None)

    if method == "override":
        override_value = require_finite_float(value.get("value"), "starting_fcf.value")
        note = value.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValidationError("starting_fcf.note is required when method is override")
        return StartingFcfAssumption(method="override", value=override_value, note=note)

    raise ValidationError("starting_fcf.method must be provider_ttm_fcf or override")


def build_effective_inputs(
    *,
    facts: FundamentalFacts,
    assumptions: ValuationAssumptionSet,
    starting_fcf: StartingFcfAssumption,
) -> tuple[EffectiveValuationInputs, ValuationInputProvenance]:
    resolved_starting_fcf, starting_fcf_source, starting_fcf_note = _resolve_starting_fcf(
        starting_fcf=starting_fcf,
        provider_fcf=facts.free_cash_flow,
        fiscal_period_type=facts.fiscal_period_type,
    )
    effective_inputs = EffectiveValuationInputs(
        starting_fcf=resolved_starting_fcf,
        shares_outstanding=_effective_value(
            "shares_outstanding",
            facts.shares_outstanding,
            assumptions.facts_overrides,
        ),
        net_debt=_effective_value("net_debt", facts.net_debt, assumptions.facts_overrides),
        reference_price=_effective_value(
            "reference_price",
            facts.reference_price,
            assumptions.facts_overrides,
        ),
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
        shares_outstanding_note=_note_for(
            "shares_outstanding",
            assumptions.facts_overrides,
            assumptions.facts_override_notes,
        ),
        net_debt_note=_note_for("net_debt", assumptions.facts_overrides, assumptions.facts_override_notes),
        reference_price_note=_note_for(
            "reference_price",
            assumptions.facts_overrides,
            assumptions.facts_override_notes,
        ),
    )
    return effective_inputs, provenance


def require_finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be finite")
    return number


def require_rate(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field} must be a number")
    rate = float(value)
    if not math.isfinite(rate):
        raise ValidationError(f"{field} must be finite")
    return rate


def require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    return value


def require_literal(value: object, field: str, expected: str) -> str:
    if value != expected:
        raise ValidationError(f"{field} must be {expected}")
    return expected


def _resolve_starting_fcf(
    *,
    starting_fcf: StartingFcfAssumption,
    provider_fcf: float,
    fiscal_period_type: str,
) -> tuple[float, str, str | None]:
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


def _effective_value(field: str, provider_value: float, overrides: Mapping[str, float | None]) -> float:
    override = overrides.get(field)
    if override is not None:
        return override
    return provider_value


def _source_for(field: str, overrides: Mapping[str, float | None]) -> str:
    if overrides.get(field) is not None:
        return "assumption_override"
    return "provider_fact"


def _note_for(
    field: str,
    overrides: Mapping[str, float | None],
    notes: Mapping[str, str | None],
) -> str | None:
    if overrides.get(field) is not None:
        return notes.get(field)
    return None
