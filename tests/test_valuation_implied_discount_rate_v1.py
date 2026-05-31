from __future__ import annotations

from datetime import date

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    ImpliedDiscountRateScenarioAssumptions,
    ImpliedDiscountRateV1Assumptions,
    StartingFcfAssumption,
    ValuationModelInput,
)
from universe_selector.valuation.implied_discount_rate_v1 import (
    ImpliedDiscountRateV1Model,
    _solve_implied_discount_rate,
)


def _inputs(
    *,
    starting_fcf: float = 100.0,
    shares_outstanding: float = 10.0,
    net_debt: float = 100.0,
    reference_price: float = 124.82954545454542,
    currency: str = "USD",
) -> EffectiveValuationInputs:
    return EffectiveValuationInputs(
        starting_fcf=starting_fcf,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt,
        reference_price=reference_price,
        currency=currency,
        fiscal_period_type="ttm",
        fiscal_period_end=date(2025, 12, 31),
        reference_price_as_of=date(2026, 5, 15),
        reference_price_as_of_source="provider_reported",
        reference_price_as_of_note=None,
    )


def _assumptions(
    *,
    scenario_id: str = "base",
    growth_rate: float = 0.05,
    terminal_growth_rate: float = 0.02,
    lower_bound: float = 0.05,
    upper_bound: float = 0.25,
    solver_abs_tolerance: float = 0.000001,
    solver_max_iterations: int = 100,
) -> ImpliedDiscountRateV1Assumptions:
    return ImpliedDiscountRateV1Assumptions(
        forecast_years=2,
        terminal_method="perpetual_growth",
        starting_fcf=StartingFcfAssumption(method="provider_ttm_fcf", value=None, note=None),
        growth_rate_basis="constant_explicit_fcf_growth",
        terminal_growth_basis="nominal_perpetual_growth",
        implied_discount_rate_basis="nominal_wacc",
        solver_abs_tolerance=solver_abs_tolerance,
        solver_max_iterations=solver_max_iterations,
        scenario_order=(scenario_id,),
        scenarios={
            scenario_id: ImpliedDiscountRateScenarioAssumptions(
                scenario_id=scenario_id,
                growth_rate=growth_rate,
                terminal_growth_rate=terminal_growth_rate,
                implied_discount_rate_lower_bound=lower_bound,
                implied_discount_rate_upper_bound=upper_bound,
                note="unit test",
            )
        },
    )


def _value(inputs: EffectiveValuationInputs, assumptions: ImpliedDiscountRateV1Assumptions):
    return ImpliedDiscountRateV1Model().value(
        ValuationModelInput(
            market=Market.US,
            ticker="AAA",
            model_id="implied_discount_rate_v1",
            effective_inputs=inputs,
            model_assumptions=assumptions,
        )
    )


def test_implied_discount_rate_v1_solves_deterministic_rate_and_metrics() -> None:
    result = _value(_inputs(), _assumptions())[0]

    assert result.projected_fcf == pytest.approx((105.0, 110.25))
    assert result.present_value_projected_fcf == pytest.approx((95.45454545, 91.11570248))
    assert result.terminal_value == pytest.approx(1405.6875)
    assert result.present_value_terminal_value == pytest.approx(1161.72520661)
    assert result.enterprise_value == pytest.approx(1348.29545455)
    assert result.equity_value == pytest.approx(1248.29545455)
    assert result.model_implied_value_per_share == pytest.approx(124.82954545, abs=1e-6)
    assert result.reference_price == pytest.approx(124.82954545)
    assert result.model_implied_spread_to_reference_price == pytest.approx(0.0, abs=1e-7)
    assert result.model_metrics["implied_discount_rate"] == pytest.approx(0.10, abs=1e-8)
    assert result.model_metrics["solver_abs_residual"] == pytest.approx(0.0, abs=1e-5)
    assert result.model_metrics["present_value_terminal_value_share_of_enterprise_value"] == pytest.approx(0.86162510)


def test_implied_discount_rate_v1_accepts_endpoint_solution_within_tolerance() -> None:
    lower_result = _value(_inputs(), _assumptions(lower_bound=0.10, upper_bound=0.20))[0]
    upper_result = _value(_inputs(reference_price=49.791666666666664), _assumptions(upper_bound=0.20))[0]

    assert lower_result.model_metrics["implied_discount_rate"] == pytest.approx(0.10, abs=1e-8)
    assert lower_result.model_metrics["solver_abs_residual"] <= 0.000001
    assert upper_result.model_metrics["implied_discount_rate"] == pytest.approx(0.20, abs=1e-8)
    assert upper_result.model_metrics["solver_abs_residual"] <= 0.000001


def test_implied_discount_rate_v1_rejects_unbracketed_and_non_converged_solutions() -> None:
    with pytest.raises(ValidationError, match="outside.*solver bounds"):
        _value(_inputs(reference_price=1_000.0), _assumptions())
    with pytest.raises(ValidationError, match="outside.*solver bounds"):
        _value(_inputs(reference_price=1.0), _assumptions())

    with pytest.raises(ValidationError, match="did not converge"):
        _solve_implied_discount_rate(
            scenario_id="base",
            starting_fcf=100.0,
            target_enterprise_value=1348.29545455,
            shares_outstanding=10.0,
            forecast_years=2,
            growth_rate=0.05,
            terminal_growth_rate=0.02,
            lower_bound=0.05,
            upper_bound=0.25,
            abs_tolerance=0.000001,
            max_iterations=0,
        )
    with pytest.raises(ValidationError, match="did not converge"):
        _solve_implied_discount_rate(
            scenario_id="base",
            starting_fcf=100.0,
            target_enterprise_value=1348.29545455,
            shares_outstanding=10.0,
            forecast_years=2,
            growth_rate=0.05,
            terminal_growth_rate=0.02,
            lower_bound=0.05,
            upper_bound=0.25,
            abs_tolerance=0.000000000001,
            max_iterations=1,
        )


def test_implied_discount_rate_v1_rejects_unusable_runtime_inputs() -> None:
    with pytest.raises(ValidationError, match="starting_fcf"):
        _value(_inputs(starting_fcf=0.0), _assumptions())
    with pytest.raises(ValidationError, match="shares_outstanding"):
        _value(_inputs(shares_outstanding=0.0), _assumptions())
    with pytest.raises(ValidationError, match="reference_price"):
        _value(_inputs(reference_price=0.0), _assumptions())
    with pytest.raises(ValidationError, match="reference-implied enterprise value"):
        _value(_inputs(net_debt=-2_000.0, reference_price=100.0), _assumptions())


def test_implied_discount_rate_v1_rejects_wrong_assumption_type() -> None:
    with pytest.raises(ValidationError, match="ImpliedDiscountRateV1Assumptions"):
        ImpliedDiscountRateV1Model().value(
            ValuationModelInput(
                market=Market.US,
                ticker="AAA",
                model_id="implied_discount_rate_v1",
                effective_inputs=_inputs(),
                model_assumptions=object(),
            )
        )
