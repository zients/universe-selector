from __future__ import annotations

from dataclasses import dataclass

from universe_selector.errors import ValidationError
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    ImpliedDiscountRateV1Assumptions,
    ValuationModelInput,
    ValuationScenarioResult,
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
