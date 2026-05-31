from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts
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
    ImpliedDiscountRateScenarioAssumptions,
    ImpliedDiscountRateV1Assumptions,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationScenarioResult,
)


_MODEL_KEYS = frozenset(
    {
        "forecast_years",
        "terminal_method",
        "starting_fcf",
        "growth_rate_basis",
        "terminal_growth_basis",
        "implied_discount_rate_basis",
        "solver_abs_tolerance",
        "solver_max_iterations",
        "scenarios",
    }
)
_SCENARIO_KEYS = frozenset(
    {
        "growth_rate",
        "terminal_growth_rate",
        "implied_discount_rate_lower_bound",
        "implied_discount_rate_upper_bound",
        "note",
    }
)


@dataclass(frozen=True)
class _ValueAtDiscountRate:
    projected_fcf: tuple[float, ...]
    present_value_projected_fcf: tuple[float, ...]
    terminal_value: float
    present_value_terminal_value: float
    enterprise_value: float
    equity_value: float
    model_implied_value_per_share: float


class ImpliedDiscountRateV1Model:
    model_id = "implied_discount_rate_v1"

    def validate_assumptions(self, assumptions: object) -> ImpliedDiscountRateV1Assumptions:
        if not isinstance(assumptions, Mapping):
            raise ValidationError("implied_discount_rate_v1 assumptions must be a mapping")
        unknown_keys = sorted(set(assumptions) - _MODEL_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown implied_discount_rate_v1 key: {unknown_keys[0]}")

        forecast_years = require_int(assumptions.get("forecast_years"), "forecast_years")
        if not 1 <= forecast_years <= 10:
            raise ValidationError("forecast_years must be from 1 through 10")

        terminal_method = require_literal(
            assumptions.get("terminal_method"),
            "terminal_method",
            "perpetual_growth",
        )
        starting_fcf = parse_starting_fcf(assumptions.get("starting_fcf"))
        growth_rate_basis = require_literal(
            assumptions.get("growth_rate_basis"),
            "growth_rate_basis",
            "constant_explicit_fcf_growth",
        )
        terminal_growth_basis = require_literal(
            assumptions.get("terminal_growth_basis"),
            "terminal_growth_basis",
            "nominal_perpetual_growth",
        )
        implied_discount_rate_basis = require_literal(
            assumptions.get("implied_discount_rate_basis"),
            "implied_discount_rate_basis",
            "nominal_wacc",
        )
        solver_abs_tolerance = require_finite_float(
            assumptions.get("solver_abs_tolerance"),
            "solver_abs_tolerance",
        )
        if solver_abs_tolerance <= 0.0:
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
        if set(scenarios_payload) != set(_SCENARIO_ORDER):
            raise ValidationError("scenarios must contain conservative, base, and upside")

        scenarios: dict[str, ImpliedDiscountRateScenarioAssumptions] = {}
        for scenario_id in _SCENARIO_ORDER:
            payload = scenarios_payload[scenario_id]
            if not isinstance(payload, Mapping):
                raise ValidationError(f"scenario {scenario_id} must be a mapping")
            scenarios[scenario_id] = self._parse_scenario(scenario_id, payload)

        if not (
            scenarios["conservative"].growth_rate <= scenarios["base"].growth_rate <= scenarios["upside"].growth_rate
        ):
            raise ValidationError("scenario growth_rate values must satisfy conservative <= base <= upside")
        if not (
            scenarios["conservative"].terminal_growth_rate
            <= scenarios["base"].terminal_growth_rate
            <= scenarios["upside"].terminal_growth_rate
        ):
            raise ValidationError("scenario terminal_growth_rate values must satisfy conservative <= base <= upside")

        return ImpliedDiscountRateV1Assumptions(
            forecast_years=forecast_years,
            terminal_method=terminal_method,
            starting_fcf=starting_fcf,
            growth_rate_basis=growth_rate_basis,
            terminal_growth_basis=terminal_growth_basis,
            implied_discount_rate_basis=implied_discount_rate_basis,
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
        if not isinstance(assumptions.model_assumptions, ImpliedDiscountRateV1Assumptions):
            raise ValidationError("implied_discount_rate_v1 requires ImpliedDiscountRateV1Assumptions")
        return build_effective_inputs(
            facts=facts,
            assumptions=assumptions,
            starting_fcf=assumptions.model_assumptions.starting_fcf,
        )

    def value(self, model_input: ValuationModelInput) -> tuple[ValuationScenarioResult, ...]:
        assumptions = model_input.model_assumptions
        if not isinstance(assumptions, ImpliedDiscountRateV1Assumptions):
            raise ValidationError("implied_discount_rate_v1 requires ImpliedDiscountRateV1Assumptions")

        inputs = model_input.effective_inputs
        _validate_implied_discount_rate_inputs(inputs)
        reference_equity_value = inputs.reference_price * inputs.shares_outstanding
        target_enterprise_value = reference_equity_value + inputs.net_debt
        if target_enterprise_value <= 0:
            raise ValidationError("reference-implied enterprise value must be greater than zero")

        results: list[ValuationScenarioResult] = []
        for scenario_id in assumptions.scenario_order:
            scenario = assumptions.scenarios[scenario_id]
            implied_discount_rate = _solve_implied_discount_rate(
                scenario_id=scenario_id,
                starting_fcf=inputs.starting_fcf,
                target_enterprise_value=target_enterprise_value,
                shares_outstanding=inputs.shares_outstanding,
                forecast_years=assumptions.forecast_years,
                growth_rate=scenario.growth_rate,
                terminal_growth_rate=scenario.terminal_growth_rate,
                lower_bound=scenario.implied_discount_rate_lower_bound,
                upper_bound=scenario.implied_discount_rate_upper_bound,
                abs_tolerance=assumptions.solver_abs_tolerance,
                max_iterations=assumptions.solver_max_iterations,
            )
            result = _value_at_discount_rate(
                inputs=inputs,
                forecast_years=assumptions.forecast_years,
                growth_rate=scenario.growth_rate,
                terminal_growth_rate=scenario.terminal_growth_rate,
                discount_rate=implied_discount_rate,
            )
            residual = abs(result.enterprise_value - target_enterprise_value) / inputs.shares_outstanding
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
                        "implied_discount_rate": implied_discount_rate,
                        "solver_abs_residual": residual,
                        "present_value_terminal_value_share_of_enterprise_value": (
                            result.present_value_terminal_value / result.enterprise_value
                        ),
                    },
                )
            )
        return tuple(results)

    def _parse_scenario(
        self,
        scenario_id: str,
        payload: Mapping[str, object],
    ) -> ImpliedDiscountRateScenarioAssumptions:
        unknown_keys = sorted(set(payload) - _SCENARIO_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown implied_discount_rate_v1 scenario key: {unknown_keys[0]}")

        growth_rate = require_rate(payload.get("growth_rate"), f"scenarios.{scenario_id}.growth_rate")
        terminal_growth_rate = require_rate(
            payload.get("terminal_growth_rate"),
            f"scenarios.{scenario_id}.terminal_growth_rate",
        )
        lower_bound = require_rate(
            payload.get("implied_discount_rate_lower_bound"),
            f"scenarios.{scenario_id}.implied_discount_rate_lower_bound",
        )
        upper_bound = require_rate(
            payload.get("implied_discount_rate_upper_bound"),
            f"scenarios.{scenario_id}.implied_discount_rate_upper_bound",
        )
        note = payload.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValidationError(f"scenarios.{scenario_id}.note is required")

        if not -1.0 < growth_rate <= 1.0:
            raise ValidationError(f"scenarios.{scenario_id}.growth_rate must be greater than -1.0 and at most 1.0")
        if not -0.05 <= terminal_growth_rate <= 0.05:
            raise ValidationError(f"scenarios.{scenario_id}.terminal_growth_rate must be between -0.05 and 0.05")
        if not 0.0 < lower_bound <= 0.50:
            raise ValidationError(
                f"scenarios.{scenario_id}.implied_discount_rate_lower_bound must be greater than 0.0 and at most 0.50"
            )
        if not 0.0 < upper_bound <= 0.50:
            raise ValidationError(
                f"scenarios.{scenario_id}.implied_discount_rate_upper_bound must be greater than 0.0 and at most 0.50"
            )
        if lower_bound >= upper_bound:
            raise ValidationError(f"scenarios.{scenario_id}.lower bound must be below upper bound")
        if lower_bound - terminal_growth_rate < 0.03:
            raise ValidationError(f"scenarios.{scenario_id}.lower_bound - terminal_growth_rate must be at least 0.03")

        return ImpliedDiscountRateScenarioAssumptions(
            scenario_id=scenario_id,
            growth_rate=growth_rate,
            terminal_growth_rate=terminal_growth_rate,
            implied_discount_rate_lower_bound=lower_bound,
            implied_discount_rate_upper_bound=upper_bound,
            note=note.strip(),
        )


def _value_at_discount_rate(
    *,
    inputs: EffectiveValuationInputs,
    forecast_years: int,
    growth_rate: float,
    terminal_growth_rate: float,
    discount_rate: float,
) -> _ValueAtDiscountRate:
    projected_fcf = tuple(inputs.starting_fcf * (1.0 + growth_rate) ** year for year in range(1, forecast_years + 1))
    present_value_projected_fcf = tuple(
        fcf / (1.0 + discount_rate) ** year for year, fcf in enumerate(projected_fcf, start=1)
    )
    terminal_value = projected_fcf[-1] * (1.0 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    present_value_terminal_value = terminal_value / ((1.0 + discount_rate) ** forecast_years)
    enterprise_value = sum(present_value_projected_fcf) + present_value_terminal_value
    equity_value = enterprise_value - inputs.net_debt
    return _ValueAtDiscountRate(
        projected_fcf=projected_fcf,
        present_value_projected_fcf=present_value_projected_fcf,
        terminal_value=terminal_value,
        present_value_terminal_value=present_value_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        model_implied_value_per_share=equity_value / inputs.shares_outstanding,
    )


def _solve_implied_discount_rate(
    *,
    scenario_id: str,
    starting_fcf: float,
    target_enterprise_value: float,
    shares_outstanding: float,
    forecast_years: int,
    growth_rate: float,
    terminal_growth_rate: float,
    lower_bound: float,
    upper_bound: float,
    abs_tolerance: float,
    max_iterations: int,
) -> float:
    def value_at(discount_rate: float) -> float:
        projected = tuple(starting_fcf * (1.0 + growth_rate) ** year for year in range(1, forecast_years + 1))
        pv_fcf = sum(fcf / (1.0 + discount_rate) ** year for year, fcf in enumerate(projected, start=1))
        terminal = projected[-1] * (1.0 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        return pv_fcf + terminal / ((1.0 + discount_rate) ** forecast_years)

    def residual(discount_rate: float) -> float:
        return (value_at(discount_rate) - target_enterprise_value) / shares_outstanding

    lower_residual = residual(lower_bound)
    if abs(lower_residual) <= abs_tolerance:
        return lower_bound
    upper_residual = residual(upper_bound)
    if abs(upper_residual) <= abs_tolerance:
        return upper_bound
    if lower_residual * upper_residual > 0:
        raise ValidationError(f"{scenario_id} reference-implied enterprise value is outside the solver bounds")

    lower = lower_bound
    upper = upper_bound
    for _ in range(max_iterations):
        midpoint = (lower + upper) / 2.0
        midpoint_residual = residual(midpoint)
        if abs(midpoint_residual) <= abs_tolerance:
            return midpoint
        if lower_residual * midpoint_residual > 0:
            lower = midpoint
            lower_residual = midpoint_residual
        else:
            upper = midpoint

    raise ValidationError(f"{scenario_id} implied discount rate solver did not converge")


def _validate_implied_discount_rate_inputs(inputs: EffectiveValuationInputs) -> None:
    if inputs.starting_fcf <= 0:
        raise ValidationError("starting_fcf must be greater than zero")
    if inputs.shares_outstanding <= 0:
        raise ValidationError("shares_outstanding must be greater than zero")
    if inputs.reference_price <= 0:
        raise ValidationError("reference_price must be greater than zero")
