from __future__ import annotations

from datetime import date

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.valuation.exit_multiple_dcf_v1 import ExitMultipleDcfV1Model
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    ExitMultipleDcfScenarioAssumptions,
    ExitMultipleDcfV1Assumptions,
    StartingFcfAssumption,
    ValuationModelInput,
)


def _inputs(
    *,
    starting_fcf: float = 100.0,
    shares_outstanding: float = 10.0,
    net_debt: float = 150.0,
    reference_price: float = 20.0,
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
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_multiple: float = 5.0,
) -> ExitMultipleDcfV1Assumptions:
    return ExitMultipleDcfV1Assumptions(
        forecast_years=2,
        terminal_method="exit_multiple",
        starting_fcf=StartingFcfAssumption(method="provider_ttm_fcf", value=None, note=None),
        discount_rate_basis="nominal_wacc",
        exit_multiple_basis="ev_to_fcf",
        scenario_order=("base",),
        scenarios={
            "base": ExitMultipleDcfScenarioAssumptions(
                scenario_id="base",
                growth_rate=growth_rate,
                discount_rate=discount_rate,
                terminal_ev_to_fcf_multiple=terminal_multiple,
                note="unit test",
            )
        },
    )


def _value(inputs: EffectiveValuationInputs, assumptions: ExitMultipleDcfV1Assumptions):
    return ExitMultipleDcfV1Model().value(
        ValuationModelInput(
            market=Market.US,
            ticker="AAA",
            model_id="exit_multiple_dcf_v1",
            effective_inputs=inputs,
            model_assumptions=assumptions,
        )
    )


def test_exit_multiple_dcf_v1_computes_deterministic_model_implied_spread() -> None:
    result = _value(_inputs(), _assumptions())[0]

    assert result.projected_fcf == pytest.approx((105.0, 110.25))
    assert result.present_value_projected_fcf == pytest.approx((95.45454545, 91.11570248))
    assert result.terminal_value == pytest.approx(551.25)
    assert result.present_value_terminal_value == pytest.approx(455.57851240)
    assert result.enterprise_value == pytest.approx(642.14876033)
    assert result.equity_value == pytest.approx(492.14876033)
    assert result.model_implied_value_per_share == pytest.approx(49.21487603)
    assert result.reference_price == pytest.approx(20.0)
    assert result.model_implied_spread_to_reference_price == pytest.approx(1.46074380)
    assert result.model_metrics["terminal_ev_to_fcf_multiple"] == pytest.approx(5.0)
    assert result.model_metrics["final_year_fcf"] == pytest.approx(110.25)


def test_exit_multiple_dcf_v1_allows_negative_equity_value_without_clamping() -> None:
    result = _value(_inputs(net_debt=1_000.0), _assumptions())[0]

    assert result.enterprise_value == pytest.approx(642.14876033)
    assert result.equity_value == pytest.approx(-357.85123967)
    assert result.model_implied_value_per_share == pytest.approx(-35.78512397)
    assert result.model_implied_spread_to_reference_price == pytest.approx(-2.78925620)


def test_exit_multiple_dcf_v1_rejects_non_positive_starting_fcf_shares_and_reference_price() -> None:
    with pytest.raises(ValidationError, match="starting_fcf"):
        _value(_inputs(starting_fcf=0.0), _assumptions())
    with pytest.raises(ValidationError, match="shares_outstanding"):
        _value(_inputs(shares_outstanding=0.0), _assumptions())
    with pytest.raises(ValidationError, match="reference_price"):
        _value(_inputs(reference_price=0.0), _assumptions())


def test_exit_multiple_dcf_v1_rejects_wrong_assumption_type() -> None:
    with pytest.raises(ValidationError, match="ExitMultipleDcfV1Assumptions"):
        ExitMultipleDcfV1Model().value(
            ValuationModelInput(
                market=Market.US,
                ticker="AAA",
                model_id="exit_multiple_dcf_v1",
                effective_inputs=_inputs(),
                model_assumptions=object(),
            )
        )
