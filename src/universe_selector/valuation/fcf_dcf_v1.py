from __future__ import annotations

import math
from collections.abc import Mapping
from universe_selector.errors import ValidationError
from universe_selector.valuation.formatting import (
    _format_money,
    _format_note,
    _format_number,
    _format_pct,
    _lines_table,
    _markdown_text,
)
from universe_selector.providers.models import FundamentalFacts
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    FcfDcfV1Assumptions,
    StartingFcfAssumption,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationResult,
    ValuationScenarioAssumptions,
    ValuationScenarioResult,
)


_SCENARIO_ORDER = ("conservative", "base", "upside")
_MODEL_KEYS = frozenset(
    {
        "forecast_years",
        "terminal_method",
        "starting_fcf",
        "discount_rate_basis",
        "terminal_growth_basis",
        "scenarios",
    }
)
_STARTING_FCF_KEYS = frozenset({"method", "value", "note"})
_SCENARIO_KEYS = frozenset({"growth_rate", "discount_rate", "terminal_growth_rate", "note"})


class FcfDcfV1Model:
    model_id = "fcf_dcf_v1"

    def validate_assumptions(self, assumptions: Mapping[str, object]) -> FcfDcfV1Assumptions:
        unknown_keys = sorted(set(assumptions) - _MODEL_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown fcf_dcf_v1 key: {unknown_keys[0]}")

        forecast_years = _require_int(assumptions.get("forecast_years"), "forecast_years")
        if not 1 <= forecast_years <= 10:
            raise ValidationError("forecast_years must be from 1 through 10")

        terminal_method = assumptions.get("terminal_method")
        if terminal_method != "perpetual_growth":
            raise ValidationError("terminal_method must be perpetual_growth")
        starting_fcf = _parse_starting_fcf(assumptions.get("starting_fcf"))
        discount_rate_basis = _require_literal(
            assumptions.get("discount_rate_basis"),
            "discount_rate_basis",
            "nominal_wacc",
        )
        terminal_growth_basis = _require_literal(
            assumptions.get("terminal_growth_basis"),
            "terminal_growth_basis",
            "nominal_perpetual_growth",
        )

        scenarios_payload = assumptions.get("scenarios")
        if not isinstance(scenarios_payload, Mapping):
            raise ValidationError("scenarios must be a mapping")

        scenario_ids = set(scenarios_payload)
        if scenario_ids != set(_SCENARIO_ORDER):
            raise ValidationError("scenarios must contain conservative, base, and upside")

        scenarios: dict[str, ValuationScenarioAssumptions] = {}
        for scenario_id in _SCENARIO_ORDER:
            payload = scenarios_payload[scenario_id]
            if not isinstance(payload, Mapping):
                raise ValidationError(f"scenario {scenario_id} must be a mapping")
            scenarios[scenario_id] = self._parse_scenario(scenario_id, payload)

        return FcfDcfV1Assumptions(
            forecast_years=forecast_years,
            terminal_method=terminal_method,
            starting_fcf=starting_fcf,
            discount_rate_basis=discount_rate_basis,
            terminal_growth_basis=terminal_growth_basis,
            scenario_order=_SCENARIO_ORDER,
            scenarios=scenarios,
        )

    def build_inputs(
        self,
        *,
        facts: FundamentalFacts,
        assumptions: ValuationAssumptionSet,
    ) -> tuple[EffectiveValuationInputs, ValuationInputProvenance]:
        if assumptions.model_id != self.model_id:
            raise ValidationError(f"{self.model_id} cannot build inputs for {assumptions.model_id}")
        if not isinstance(assumptions.model_assumptions, FcfDcfV1Assumptions):
            raise ValidationError("fcf_dcf_v1 requires FcfDcfV1Assumptions")

        starting_fcf, starting_fcf_source, starting_fcf_note = _resolve_starting_fcf(
            model_assumptions=assumptions.model_assumptions,
            provider_fcf=facts.free_cash_flow,
            fiscal_period_type=facts.fiscal_period_type,
        )
        effective_inputs = EffectiveValuationInputs(
            starting_fcf=starting_fcf,
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

    def value(self, model_input: ValuationModelInput) -> tuple[ValuationScenarioResult, ...]:
        assumptions = model_input.model_assumptions
        if not isinstance(assumptions, FcfDcfV1Assumptions):
            raise ValidationError("fcf_dcf_v1 requires FcfDcfV1Assumptions")

        inputs = model_input.effective_inputs
        if inputs.shares_outstanding <= 0:
            raise ValidationError("shares_outstanding must be greater than zero")
        if inputs.reference_price <= 0:
            raise ValidationError("reference_price must be greater than zero")

        results = []
        for scenario_id in assumptions.scenario_order:
            scenario = assumptions.scenarios[scenario_id]
            projected_fcf = tuple(
                inputs.starting_fcf * (1.0 + scenario.growth_rate) ** year
                for year in range(1, assumptions.forecast_years + 1)
            )
            present_value_projected_fcf = tuple(
                fcf / (1.0 + scenario.discount_rate) ** year
                for year, fcf in enumerate(projected_fcf, start=1)
            )
            terminal_value = (
                projected_fcf[-1]
                * (1.0 + scenario.terminal_growth_rate)
                / (scenario.discount_rate - scenario.terminal_growth_rate)
            )
            present_value_terminal_value = terminal_value / (
                (1.0 + scenario.discount_rate) ** assumptions.forecast_years
            )
            enterprise_value = sum(present_value_projected_fcf) + present_value_terminal_value
            equity_value = enterprise_value - inputs.net_debt
            model_implied_value_per_share = equity_value / inputs.shares_outstanding
            model_implied_spread_to_reference_price = model_implied_value_per_share / inputs.reference_price - 1.0

            results.append(
                ValuationScenarioResult(
                    scenario_id=scenario_id,
                    projected_fcf=projected_fcf,
                    present_value_projected_fcf=present_value_projected_fcf,
                    terminal_value=terminal_value,
                    present_value_terminal_value=present_value_terminal_value,
                    enterprise_value=enterprise_value,
                    equity_value=equity_value,
                    model_implied_value_per_share=model_implied_value_per_share,
                    reference_price=inputs.reference_price,
                    model_implied_spread_to_reference_price=model_implied_spread_to_reference_price,
                )
            )
        return tuple(results)

    def _parse_scenario(
        self,
        scenario_id: str,
        payload: Mapping[str, object],
    ) -> ValuationScenarioAssumptions:
        unknown_keys = sorted(set(payload) - _SCENARIO_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown fcf_dcf_v1 scenario key: {unknown_keys[0]}")

        growth_rate = _require_rate(payload.get("growth_rate"), f"scenarios.{scenario_id}.growth_rate")
        discount_rate = _require_rate(payload.get("discount_rate"), f"scenarios.{scenario_id}.discount_rate")
        terminal_growth_rate = _require_rate(
            payload.get("terminal_growth_rate"),
            f"scenarios.{scenario_id}.terminal_growth_rate",
        )
        note = payload.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValidationError(f"scenarios.{scenario_id}.note is required")

        if not -1.0 < growth_rate <= 1.0:
            raise ValidationError(f"scenarios.{scenario_id}.growth_rate must be greater than -1.0 and at most 1.0")
        if not 0.0 < discount_rate <= 0.50:
            raise ValidationError(f"scenarios.{scenario_id}.discount_rate must be greater than 0.0 and at most 0.50")
        if not -0.05 <= terminal_growth_rate <= 0.05:
            raise ValidationError(
                f"scenarios.{scenario_id}.terminal_growth_rate must be between -0.05 and 0.05"
            )
        if discount_rate - terminal_growth_rate < 0.03:
            raise ValidationError(
                f"scenarios.{scenario_id}.discount_rate - terminal_growth_rate must be at least 0.03"
            )

        return ValuationScenarioAssumptions(
            scenario_id=scenario_id,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            note=note,
        )


class FcfDcfV1OutputRenderer:
    model_id = "fcf_dcf_v1"

    def render_risk_disclosures(self, result: ValuationResult) -> list[str]:
        del result
        return [
            (
                "- Model risk: fcf_dcf_v1 is a simplified constant-growth explicit forecast "
                "with perpetual-growth terminal value."
            ),
            (
                "- Sensitivity risk: outputs are highly sensitive to starting FCF, share count, "
                "discount rate, terminal growth, and terminal value assumptions."
            ),
        ]

    def render_model_assumptions(self, result: ValuationResult) -> list[str]:
        assumptions = result.run_input.assumptions.model_assumptions
        if not isinstance(assumptions, FcfDcfV1Assumptions):
            raise ValidationError("fcf_dcf_v1 requires FcfDcfV1Assumptions")
        return _render_fcf_dcf_assumptions(assumptions)

    def render_effective_inputs(self, result: ValuationResult) -> list[str]:
        effective = result.run_input.effective_inputs
        lines = [
            "",
            "## Effective Inputs",
            "",
            (
                "Starting FCF is used as an enterprise cash-flow proxy and is not verified "
                "unlevered FCFF. provider_ttm_fcf uses raw provider FCF as the starting FCF proxy. "
                "Use starting_fcf.method override when analyst-normalized FCF is needed."
            ),
            "",
        ]
        lines.extend(
            _lines_table(
                ("field", "value"),
                (
                    ("starting_fcf", _format_money(effective.starting_fcf, effective.currency)),
                    ("shares_outstanding", _format_number(effective.shares_outstanding)),
                    ("net_debt", _format_money(effective.net_debt, effective.currency)),
                    ("reference_price", _format_money(effective.reference_price, effective.currency)),
                    ("reference_price_as_of", effective.reference_price_as_of.isoformat()),
                    ("reference_price_as_of_source", effective.reference_price_as_of_source),
                    (
                        "reference_price_as_of_note",
                        _format_note(effective.reference_price_as_of_note),
                    ),
                    ("fiscal_period_end", effective.fiscal_period_end.isoformat()),
                    ("fiscal_period_type", effective.fiscal_period_type),
                ),
            )
        )
        return lines

    def render_input_provenance(self, result: ValuationResult) -> list[str]:
        provenance = result.run_input.input_provenance
        lines = [
            "",
            "## Input Provenance",
            "",
        ]
        lines.extend(
            _lines_table(
                ("field", "source", "note"),
                (
                    (
                        "starting_fcf",
                        provenance.starting_fcf_source,
                        _format_note(provenance.starting_fcf_note),
                    ),
                    (
                        "shares_outstanding",
                        provenance.shares_outstanding_source,
                        _format_note(provenance.shares_outstanding_note),
                    ),
                    ("net_debt", provenance.net_debt_source, _format_note(provenance.net_debt_note)),
                    (
                        "reference_price",
                        provenance.reference_price_source,
                        _format_note(provenance.reference_price_note),
                    ),
                ),
            )
        )
        return lines

    def render_scenario_results(self, result: ValuationResult) -> list[str]:
        effective = result.run_input.effective_inputs
        lines = [
            "",
            "## Scenario Results",
            "",
            (
                "Scenario rows are illustrative scenarios, not probabilities, forecasts, "
                "expected outcomes, target cases, or recommendations."
            ),
            "",
        ]
        lines.extend(
            _lines_table(
                (
                    "scenario",
                    "model-implied value per share",
                    "reference price",
                    "model-implied spread vs reference price",
                ),
                (
                    (
                        scenario.scenario_id,
                        _format_money(scenario.model_implied_value_per_share, effective.currency),
                        _format_money(scenario.reference_price, effective.currency),
                        _format_pct(scenario.model_implied_spread_to_reference_price),
                    )
                    for scenario in result.scenario_results
                ),
            )
        )
        return lines


def _render_fcf_dcf_assumptions(assumptions: FcfDcfV1Assumptions) -> list[str]:
    lines = [
        "## Model Assumptions",
        "",
        f"- forecast_years: {assumptions.forecast_years}",
        f"- terminal_method: {_markdown_text(assumptions.terminal_method)}",
        f"- starting_fcf_method: {_markdown_text(assumptions.starting_fcf.method)}",
        f"- discount_rate_basis: {_markdown_text(assumptions.discount_rate_basis)}",
        f"- terminal_growth_basis: {_markdown_text(assumptions.terminal_growth_basis)}",
        "",
    ]
    if assumptions.starting_fcf.method == "override":
        lines.extend(
            [
                f"- starting_fcf_value: {_format_number(assumptions.starting_fcf.value)}",
                f"- starting_fcf_note: {_markdown_text(assumptions.starting_fcf.note)}",
                "",
            ]
        )
    rows = []
    for scenario_id in assumptions.scenario_order:
        scenario = assumptions.scenarios[scenario_id]
        rows.append(
            (
                scenario.scenario_id,
                _format_pct(scenario.growth_rate),
                _format_pct(scenario.discount_rate),
                _format_pct(scenario.terminal_growth_rate),
                _format_note(scenario.note),
            )
        )
    lines.extend(
        _lines_table(
            ("scenario", "growth_rate", "discount_rate", "terminal_growth_rate", "note"),
            rows,
        )
    )
    return lines


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    return value


def _require_literal(value: object, field: str, expected: str) -> str:
    if value != expected:
        raise ValidationError(f"{field} must be {expected}")
    return expected


def _parse_starting_fcf(value: object) -> StartingFcfAssumption:
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
        override_value = _require_finite_float(value.get("value"), "starting_fcf.value")
        note = value.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValidationError("starting_fcf.note is required when method is override")
        return StartingFcfAssumption(method="override", value=override_value, note=note)

    raise ValidationError("starting_fcf.method must be provider_ttm_fcf or override")


def _resolve_starting_fcf(
    *,
    model_assumptions: FcfDcfV1Assumptions,
    provider_fcf: float,
    fiscal_period_type: str,
) -> tuple[float, str, str | None]:
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


def _require_finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be finite")
    return number


def _require_rate(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field} must be a number")
    rate = float(value)
    if not math.isfinite(rate):
        raise ValidationError(f"{field} must be finite")
    return rate
