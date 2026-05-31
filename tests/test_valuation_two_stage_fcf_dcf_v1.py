from __future__ import annotations

from datetime import date

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    StartingFcfAssumption,
    TwoStageFcfDcfScenarioAssumptions,
    TwoStageFcfDcfV1Assumptions,
    ValuationModelInput,
)
from universe_selector.valuation.two_stage_fcf_dcf_v1 import TwoStageFcfDcfV1Model


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
    stage1_growth_rate: float = 0.10,
    stage2_growth_rate: float = 0.04,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.03,
) -> TwoStageFcfDcfV1Assumptions:
    return TwoStageFcfDcfV1Assumptions(
        stage1_years=2,
        stage2_years=2,
        terminal_method="perpetual_growth",
        starting_fcf=StartingFcfAssumption(method="provider_ttm_fcf", value=None, note=None),
        discount_rate_basis="nominal_wacc",
        terminal_growth_basis="nominal_perpetual_growth",
        scenario_order=("base",),
        scenarios={
            "base": TwoStageFcfDcfScenarioAssumptions(
                scenario_id="base",
                stage1_growth_rate=stage1_growth_rate,
                stage2_growth_rate=stage2_growth_rate,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth_rate,
                note="unit test",
            )
        },
    )


def _value(inputs: EffectiveValuationInputs, assumptions: TwoStageFcfDcfV1Assumptions):
    return TwoStageFcfDcfV1Model().value(
        ValuationModelInput(
            market=Market.US,
            ticker="AAA",
            model_id="two_stage_fcf_dcf_v1",
            effective_inputs=inputs,
            model_assumptions=assumptions,
        )
    )


def test_two_stage_fcf_dcf_v1_computes_deterministic_model_implied_spread() -> None:
    result = _value(_inputs(), _assumptions())[0]

    assert result.projected_fcf == pytest.approx((110.0, 121.0, 125.84, 130.8736))
    assert result.present_value_projected_fcf == pytest.approx((100.0, 100.0, 94.54545455, 89.38842975))
    assert result.terminal_value == pytest.approx(1925.71154286)
    assert result.present_value_terminal_value == pytest.approx(1315.28689492)
    assert result.enterprise_value == pytest.approx(1699.22077922)
    assert result.equity_value == pytest.approx(1549.22077922)
    assert result.model_implied_value_per_share == pytest.approx(154.92207792)
    assert result.reference_price == pytest.approx(20.0)
    assert result.model_implied_spread_to_reference_price == pytest.approx(6.74610390)
    assert result.model_metrics["stage1_final_fcf"] == pytest.approx(121.0)
    assert result.model_metrics["final_year_fcf"] == pytest.approx(130.8736)
    assert result.model_metrics["present_value_terminal_value_share_of_enterprise_value"] == pytest.approx(0.77405297)


def test_two_stage_fcf_dcf_v1_allows_negative_equity_value_without_clamping() -> None:
    result = _value(_inputs(net_debt=2_000.0), _assumptions())[0]

    assert result.enterprise_value == pytest.approx(1699.22077922)
    assert result.equity_value == pytest.approx(-300.77922078)
    assert result.model_implied_value_per_share == pytest.approx(-30.07792208)


def test_two_stage_fcf_dcf_v1_rejects_non_positive_starting_fcf_shares_and_reference_price() -> None:
    with pytest.raises(ValidationError, match="starting_fcf"):
        _value(_inputs(starting_fcf=0.0), _assumptions())
    with pytest.raises(ValidationError, match="shares_outstanding"):
        _value(_inputs(shares_outstanding=0.0), _assumptions())
    with pytest.raises(ValidationError, match="reference_price"):
        _value(_inputs(reference_price=0.0), _assumptions())


def test_two_stage_fcf_dcf_v1_rejects_wrong_assumption_type() -> None:
    with pytest.raises(ValidationError, match="TwoStageFcfDcfV1Assumptions"):
        TwoStageFcfDcfV1Model().value(
            ValuationModelInput(
                market=Market.US,
                ticker="AAA",
                model_id="two_stage_fcf_dcf_v1",
                effective_inputs=_inputs(),
                model_assumptions=object(),
            )
        )
