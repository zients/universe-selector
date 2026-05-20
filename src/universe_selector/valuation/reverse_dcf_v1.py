from __future__ import annotations

from collections.abc import Mapping

from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts
from universe_selector.valuation.formatting import (
    _format_money,
    _format_money_precise,
    _format_note,
    _format_number,
    _format_pct,
    _lines_table,
    _markdown_text,
)
from universe_selector.valuation.input_resolution import (
    _SCENARIO_ORDER,
    build_effective_inputs,
    parse_starting_fcf,
    require_finite_float,
    require_int,
    require_literal,
    require_rate,
)
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    ReverseDcfScenarioAssumptions,
    ReverseDcfV1Assumptions,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationResult,
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
        "implied_growth_basis",
        "solver_abs_tolerance",
        "solver_max_iterations",
        "scenarios",
    }
)
_SCENARIO_KEYS = frozenset(
    {
        "discount_rate",
        "terminal_growth_rate",
        "implied_growth_lower_bound",
        "implied_growth_upper_bound",
        "note",
    }
)


class ReverseDcfV1Model:
    model_id = "reverse_dcf_v1"

    def validate_assumptions(self, assumptions: Mapping[str, object]) -> ReverseDcfV1Assumptions:
        unknown_keys = sorted(set(assumptions) - _MODEL_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown reverse_dcf_v1 key: {unknown_keys[0]}")

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
        implied_growth_basis = require_literal(
            assumptions.get("implied_growth_basis"),
            "implied_growth_basis",
            "constant_explicit_fcf_growth",
        )
        solver_abs_tolerance = require_finite_float(
            assumptions.get("solver_abs_tolerance"),
            "solver_abs_tolerance",
        )
        if solver_abs_tolerance <= 0:
            raise ValidationError("solver_abs_tolerance must be greater than zero")
        solver_max_iterations = require_int(
            assumptions.get("solver_max_iterations"),
            "solver_max_iterations",
        )
        if not 1 <= solver_max_iterations <= 1000:
            raise ValidationError("solver_max_iterations must be from 1 through 1000")

        scenarios_payload = assumptions.get("scenarios")
        if not isinstance(scenarios_payload, Mapping):
            raise ValidationError("scenarios must be a mapping")
        scenario_ids = set(scenarios_payload)
        if scenario_ids != set(_SCENARIO_ORDER):
            raise ValidationError("scenarios must contain conservative, base, and upside")

        scenarios: dict[str, ReverseDcfScenarioAssumptions] = {}
        for scenario_id in _SCENARIO_ORDER:
            payload = scenarios_payload[scenario_id]
            if not isinstance(payload, Mapping):
                raise ValidationError(f"scenario {scenario_id} must be a mapping")
            scenarios[scenario_id] = self._parse_scenario(scenario_id, payload)

        return ReverseDcfV1Assumptions(
            forecast_years=forecast_years,
            terminal_method=terminal_method,
            starting_fcf=starting_fcf,
            discount_rate_basis=discount_rate_basis,
            terminal_growth_basis=terminal_growth_basis,
            implied_growth_basis=implied_growth_basis,
            solver_abs_tolerance=solver_abs_tolerance,
            solver_max_iterations=solver_max_iterations,
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
        if not isinstance(assumptions.model_assumptions, ReverseDcfV1Assumptions):
            raise ValidationError("reverse_dcf_v1 requires ReverseDcfV1Assumptions")

        return build_effective_inputs(
            facts=facts,
            assumptions=assumptions,
            starting_fcf=assumptions.model_assumptions.starting_fcf,
        )

    def value(self, model_input: ValuationModelInput) -> tuple[ValuationScenarioResult, ...]:
        assumptions = model_input.model_assumptions
        if not isinstance(assumptions, ReverseDcfV1Assumptions):
            raise ValidationError("reverse_dcf_v1 requires ReverseDcfV1Assumptions")

        inputs = model_input.effective_inputs
        _validate_reverse_dcf_inputs(inputs)
        reference_equity_value = inputs.reference_price * inputs.shares_outstanding
        reference_implied_enterprise_value = reference_equity_value + inputs.net_debt

        results = []
        for scenario_id in assumptions.scenario_order:
            scenario = assumptions.scenarios[scenario_id]
            implied_growth_rate = _solve_implied_growth(
                scenario_id=scenario_id,
                starting_fcf=inputs.starting_fcf,
                target_enterprise_value=reference_implied_enterprise_value,
                forecast_years=assumptions.forecast_years,
                discount_rate=scenario.discount_rate,
                terminal_growth_rate=scenario.terminal_growth_rate,
                lower_bound=scenario.implied_growth_lower_bound,
                upper_bound=scenario.implied_growth_upper_bound,
                abs_tolerance=assumptions.solver_abs_tolerance,
                max_iterations=assumptions.solver_max_iterations,
            )
            result = _value_at_growth(
                inputs=inputs,
                forecast_years=assumptions.forecast_years,
                growth_rate=implied_growth_rate,
                discount_rate=scenario.discount_rate,
                terminal_growth_rate=scenario.terminal_growth_rate,
            )
            solver_abs_residual = (
                abs(result.enterprise_value - reference_implied_enterprise_value) / inputs.shares_outstanding
            )
            results.append(
                ValuationScenarioResult(
                    scenario_id=scenario_id,
                    projected_fcf=result.projected_fcf,
                    present_value_projected_fcf=result.present_value_projected_fcf,
                    terminal_value=result.terminal_value,
                    present_value_terminal_value=result.present_value_terminal_value,
                    enterprise_value=result.enterprise_value,
                    equity_value=result.equity_value,
                    model_implied_value_per_share=result.model_implied_value_per_share,
                    reference_price=inputs.reference_price,
                    model_implied_spread_to_reference_price=(
                        result.model_implied_value_per_share / inputs.reference_price - 1.0
                    ),
                    model_metrics={
                        "implied_growth_rate": implied_growth_rate,
                        "solver_abs_residual": solver_abs_residual,
                    },
                )
            )
        return tuple(results)

    def _parse_scenario(
        self,
        scenario_id: str,
        payload: Mapping[str, object],
    ) -> ReverseDcfScenarioAssumptions:
        unknown_keys = sorted(set(payload) - _SCENARIO_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown reverse_dcf_v1 scenario key: {unknown_keys[0]}")

        discount_rate = require_rate(payload.get("discount_rate"), f"scenarios.{scenario_id}.discount_rate")
        terminal_growth_rate = require_rate(
            payload.get("terminal_growth_rate"),
            f"scenarios.{scenario_id}.terminal_growth_rate",
        )
        lower_bound = require_rate(
            payload.get("implied_growth_lower_bound"),
            f"scenarios.{scenario_id}.implied_growth_lower_bound",
        )
        upper_bound = require_rate(
            payload.get("implied_growth_upper_bound"),
            f"scenarios.{scenario_id}.implied_growth_upper_bound",
        )
        note = payload.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValidationError(f"scenarios.{scenario_id}.note is required")

        if not 0.0 < discount_rate <= 0.50:
            raise ValidationError(f"scenarios.{scenario_id}.discount_rate must be greater than 0.0 and at most 0.50")
        if not -0.05 <= terminal_growth_rate <= 0.05:
            raise ValidationError(f"scenarios.{scenario_id}.terminal_growth_rate must be between -0.05 and 0.05")
        if discount_rate - terminal_growth_rate < 0.03:
            raise ValidationError(f"scenarios.{scenario_id}.discount_rate - terminal_growth_rate must be at least 0.03")
        if not -1.0 < lower_bound <= 1.0:
            raise ValidationError(
                f"scenarios.{scenario_id}.implied_growth_lower_bound must be greater than -1.0 and at most 1.0"
            )
        if not -1.0 < upper_bound <= 1.0:
            raise ValidationError(
                f"scenarios.{scenario_id}.implied_growth_upper_bound must be greater than -1.0 and at most 1.0"
            )
        if lower_bound >= upper_bound:
            raise ValidationError(f"scenarios.{scenario_id}.implied growth lower bound must be below upper bound")

        return ReverseDcfScenarioAssumptions(
            scenario_id=scenario_id,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            implied_growth_lower_bound=lower_bound,
            implied_growth_upper_bound=upper_bound,
            note=note.strip(),
        )


class ReverseDcfV1OutputRenderer:
    model_id = "reverse_dcf_v1"

    def render_risk_disclosures(self, result: ValuationResult) -> list[str]:
        del result
        return [
            (
                "- Model risk: reverse_dcf_v1 solves the annual explicit-period FCF growth "
                "needed to reconcile the model to a reference price under stated assumptions."
            ),
            (
                "- Sensitivity risk: outputs are highly sensitive to starting FCF, share count, "
                "net debt, reference price, discount rate, terminal growth, and solver bounds."
            ),
        ]

    def render_model_assumptions(self, result: ValuationResult) -> list[str]:
        assumptions = result.run_input.assumptions.model_assumptions
        if not isinstance(assumptions, ReverseDcfV1Assumptions):
            raise ValidationError("reverse_dcf_v1 requires ReverseDcfV1Assumptions")

        lines = [
            "## Model Assumptions",
            "",
            f"- forecast_years: {assumptions.forecast_years}",
            f"- terminal_method: {_markdown_text(assumptions.terminal_method)}",
            f"- starting_fcf_method: {_markdown_text(assumptions.starting_fcf.method)}",
            f"- discount_rate_basis: {_markdown_text(assumptions.discount_rate_basis)}",
            f"- terminal_growth_basis: {_markdown_text(assumptions.terminal_growth_basis)}",
            f"- implied_growth_basis: {_markdown_text(assumptions.implied_growth_basis)}",
            f"- solver_abs_tolerance: {_format_number(assumptions.solver_abs_tolerance)}",
            f"- solver_max_iterations: {assumptions.solver_max_iterations}",
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
        lines.extend(
            _lines_table(
                (
                    "scenario",
                    "discount_rate",
                    "terminal_growth_rate",
                    "implied_growth_lower_bound",
                    "implied_growth_upper_bound",
                    "note",
                ),
                (
                    (
                        scenario.scenario_id,
                        _format_pct(scenario.discount_rate),
                        _format_pct(scenario.terminal_growth_rate),
                        _format_pct(scenario.implied_growth_lower_bound),
                        _format_pct(scenario.implied_growth_upper_bound),
                        _format_note(scenario.note),
                    )
                    for scenario in (assumptions.scenarios[item] for item in assumptions.scenario_order)
                ),
            )
        )
        return lines

    def render_effective_inputs(self, result: ValuationResult) -> list[str]:
        return render_effective_inputs_section(result)

    def render_input_provenance(self, result: ValuationResult) -> list[str]:
        return render_input_provenance_section(result)

    def render_scenario_results(self, result: ValuationResult) -> list[str]:
        effective = result.run_input.effective_inputs
        assumptions = result.run_input.assumptions.model_assumptions
        if not isinstance(assumptions, ReverseDcfV1Assumptions):
            raise ValidationError("reverse_dcf_v1 requires ReverseDcfV1Assumptions")

        reference_equity_value = effective.reference_price * effective.shares_outstanding
        reference_implied_enterprise_value = reference_equity_value + effective.net_debt
        lines = [
            "",
            "## Valuation Bridge",
            "",
        ]
        lines.extend(
            _lines_table(
                ("field", "value"),
                (
                    ("reference_equity_value", _format_money(reference_equity_value, effective.currency)),
                    ("net_debt", _format_money(effective.net_debt, effective.currency)),
                    (
                        "reference_implied_enterprise_value",
                        _format_money(reference_implied_enterprise_value, effective.currency),
                    ),
                ),
            )
        )
        lines.extend(
            [
                "",
                "## Scenario Results",
                "",
                (
                    "Reverse DCF implied growth is required to reconcile to the reference price "
                    "under the stated assumptions."
                ),
                (
                    "Scenario rows are assumption cases, not probabilities, expected outcomes, "
                    "or recommendations; the result is not a forecast or investment signal."
                ),
                (
                    "Solved growth applies only to the explicit forecast period; terminal growth is separate. "
                    "Displayed precision is formatting precision. "
                    "Spread is a descriptive reconciliation value, not an investment signal."
                ),
                "",
            ]
        )
        lines.extend(
            _lines_table(
                (
                    "scenario",
                    "implied annual FCF growth",
                    "absolute per-share reconciliation residual",
                    "discount_rate",
                    "terminal_growth_rate",
                    "model_implied_enterprise_value",
                    "model-implied value per share",
                    "reference price",
                    "spread vs reference price",
                    "note",
                ),
                (
                    (
                        scenario.scenario_id,
                        _format_pct(scenario.model_metrics["implied_growth_rate"]),
                        _format_money_precise(
                            scenario.model_metrics["solver_abs_residual"],
                            effective.currency,
                        ),
                        _format_pct(assumptions.scenarios[scenario.scenario_id].discount_rate),
                        _format_pct(assumptions.scenarios[scenario.scenario_id].terminal_growth_rate),
                        _format_money(scenario.enterprise_value, effective.currency),
                        _format_money(scenario.model_implied_value_per_share, effective.currency),
                        _format_money(scenario.reference_price, effective.currency),
                        _format_pct(scenario.model_implied_spread_to_reference_price),
                        _format_note(assumptions.scenarios[scenario.scenario_id].note),
                    )
                    for scenario in result.scenario_results
                ),
            )
        )
        return lines


def _validate_reverse_dcf_inputs(inputs: EffectiveValuationInputs) -> None:
    if inputs.starting_fcf <= 0:
        raise ValidationError("starting_fcf must be greater than zero")
    if inputs.shares_outstanding <= 0:
        raise ValidationError("shares_outstanding must be greater than zero")
    if inputs.reference_price <= 0:
        raise ValidationError("reference_price must be greater than zero")
    if inputs.reference_price * inputs.shares_outstanding + inputs.net_debt <= 0:
        raise ValidationError("reference-implied enterprise value must be greater than zero")


def _solve_implied_growth(
    *,
    scenario_id: str,
    starting_fcf: float,
    target_enterprise_value: float,
    forecast_years: int,
    discount_rate: float,
    terminal_growth_rate: float,
    lower_bound: float,
    upper_bound: float,
    abs_tolerance: float,
    max_iterations: int,
) -> float:
    lower_residual = (
        _enterprise_value_at_growth(
            starting_fcf=starting_fcf,
            forecast_years=forecast_years,
            growth_rate=lower_bound,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
        )
        - target_enterprise_value
    )
    if abs(lower_residual) <= abs_tolerance:
        return lower_bound

    upper_residual = (
        _enterprise_value_at_growth(
            starting_fcf=starting_fcf,
            forecast_years=forecast_years,
            growth_rate=upper_bound,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
        )
        - target_enterprise_value
    )
    if abs(upper_residual) <= abs_tolerance:
        return upper_bound

    if lower_residual * upper_residual > 0:
        raise ValidationError(
            "reverse_dcf_v1 scenario "
            f"{scenario_id} cannot bracket implied growth between {lower_bound:.4f} and {upper_bound:.4f}"
        )

    lower = lower_bound
    upper = upper_bound
    for _ in range(max_iterations):
        midpoint = (lower + upper) / 2.0
        midpoint_residual = (
            _enterprise_value_at_growth(
                starting_fcf=starting_fcf,
                forecast_years=forecast_years,
                growth_rate=midpoint,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth_rate,
            )
            - target_enterprise_value
        )
        if abs(midpoint_residual) <= abs_tolerance:
            return midpoint
        if lower_residual * midpoint_residual <= 0:
            upper = midpoint
            upper_residual = midpoint_residual
        else:
            lower = midpoint
            lower_residual = midpoint_residual

        if abs(upper - lower) <= 1e-12:
            break

    raise ValidationError(
        "reverse_dcf_v1 scenario "
        f"{scenario_id} implied growth solver did not converge after {max_iterations} iterations"
    )


def _value_at_growth(
    *,
    inputs: EffectiveValuationInputs,
    forecast_years: int,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
) -> ValuationScenarioResult:
    projected_fcf = tuple(inputs.starting_fcf * (1.0 + growth_rate) ** year for year in range(1, forecast_years + 1))
    present_value_projected_fcf = tuple(
        fcf / (1.0 + discount_rate) ** year for year, fcf in enumerate(projected_fcf, start=1)
    )
    terminal_value = projected_fcf[-1] * (1.0 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    present_value_terminal_value = terminal_value / ((1.0 + discount_rate) ** forecast_years)
    enterprise_value = sum(present_value_projected_fcf) + present_value_terminal_value
    equity_value = enterprise_value - inputs.net_debt
    model_implied_value_per_share = equity_value / inputs.shares_outstanding
    return ValuationScenarioResult(
        scenario_id="_internal",
        projected_fcf=projected_fcf,
        present_value_projected_fcf=present_value_projected_fcf,
        terminal_value=terminal_value,
        present_value_terminal_value=present_value_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        model_implied_value_per_share=model_implied_value_per_share,
        reference_price=inputs.reference_price,
        model_implied_spread_to_reference_price=model_implied_value_per_share / inputs.reference_price - 1.0,
    )


def _enterprise_value_at_growth(
    *,
    starting_fcf: float,
    forecast_years: int,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
) -> float:
    projected_fcf = tuple(starting_fcf * (1.0 + growth_rate) ** year for year in range(1, forecast_years + 1))
    present_value_projected_fcf = sum(
        fcf / (1.0 + discount_rate) ** year for year, fcf in enumerate(projected_fcf, start=1)
    )
    terminal_value = projected_fcf[-1] * (1.0 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    present_value_terminal_value = terminal_value / ((1.0 + discount_rate) ** forecast_years)
    return present_value_projected_fcf + present_value_terminal_value
