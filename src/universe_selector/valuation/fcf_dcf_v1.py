from __future__ import annotations

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
from universe_selector.valuation.input_resolution import (
    _SCENARIO_ORDER,
    build_effective_inputs,
    parse_starting_fcf,
    require_int,
    require_literal,
    require_rate,
)
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    FcfDcfV1Assumptions,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationResult,
    ValuationScenarioAssumptions,
    ValuationScenarioResult,
)
from universe_selector.valuation.output_sections import (
    render_effective_inputs_section,
    render_input_provenance_section,
)


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
_SCENARIO_KEYS = frozenset({"growth_rate", "discount_rate", "terminal_growth_rate", "note"})


class FcfDcfV1Model:
    model_id = "fcf_dcf_v1"

    def validate_assumptions(self, assumptions: Mapping[str, object]) -> FcfDcfV1Assumptions:
        unknown_keys = sorted(set(assumptions) - _MODEL_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown fcf_dcf_v1 key: {unknown_keys[0]}")

        forecast_years = require_int(assumptions.get("forecast_years"), "forecast_years")
        if not 1 <= forecast_years <= 10:
            raise ValidationError("forecast_years must be from 1 through 10")

        terminal_method = assumptions.get("terminal_method")
        if terminal_method != "perpetual_growth":
            raise ValidationError("terminal_method must be perpetual_growth")
        starting_fcf = parse_starting_fcf(assumptions.get("starting_fcf"))
        discount_rate_basis = require_literal(
            assumptions.get("discount_rate_basis"),
            "discount_rate_basis",
            "nominal_wacc",
        )
        terminal_growth_basis = require_literal(
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

        return build_effective_inputs(
            facts=facts,
            assumptions=assumptions,
            starting_fcf=assumptions.model_assumptions.starting_fcf,
        )

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
                fcf / (1.0 + scenario.discount_rate) ** year for year, fcf in enumerate(projected_fcf, start=1)
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

        growth_rate = require_rate(payload.get("growth_rate"), f"scenarios.{scenario_id}.growth_rate")
        discount_rate = require_rate(payload.get("discount_rate"), f"scenarios.{scenario_id}.discount_rate")
        terminal_growth_rate = require_rate(
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
            raise ValidationError(f"scenarios.{scenario_id}.terminal_growth_rate must be between -0.05 and 0.05")
        if discount_rate - terminal_growth_rate < 0.03:
            raise ValidationError(f"scenarios.{scenario_id}.discount_rate - terminal_growth_rate must be at least 0.03")

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
        return render_effective_inputs_section(result)

    def render_input_provenance(self, result: ValuationResult) -> list[str]:
        return render_input_provenance_section(result)

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
