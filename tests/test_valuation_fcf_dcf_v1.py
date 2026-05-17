from __future__ import annotations

from datetime import date

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.valuation.fcf_dcf_v1 import FcfDcfV1Model
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    FcfDcfV1Assumptions,
    ValuationModelInput,
    ValuationScenarioAssumptions,
)
from universe_selector.valuation.registry import get_valuation_model, supported_valuation_model_ids


def _inputs(
    *,
    normalized_fcf: float = 100.0,
    shares_outstanding: float = 10.0,
    net_debt: float = 100.0,
    reference_price: float = 100.0,
) -> EffectiveValuationInputs:
    return EffectiveValuationInputs(
        normalized_fcf=normalized_fcf,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt,
        reference_price=reference_price,
        currency="USD",
        fiscal_period_type="ttm",
        fiscal_period_end=date(2026, 1, 1),
        reference_price_as_of=date(2026, 1, 2),
        reference_price_as_of_source="provider_reported",
        reference_price_as_of_note=None,
    )


def _assumptions(
    *,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
) -> FcfDcfV1Assumptions:
    return FcfDcfV1Assumptions(
        forecast_years=2,
        terminal_method="perpetual_growth",
        cash_flow_basis="normalized_fcf_enterprise_proxy",
        discount_rate_basis="nominal_wacc",
        terminal_growth_basis="nominal_perpetual_growth",
        scenario_order=("base",),
        scenarios={
            "base": ValuationScenarioAssumptions(
                scenario_id="base",
                growth_rate=growth_rate,
                discount_rate=discount_rate,
                terminal_growth_rate=terminal_growth_rate,
                note="unit test",
            )
        },
    )


def _value(inputs: EffectiveValuationInputs, assumptions: FcfDcfV1Assumptions):
    return FcfDcfV1Model().value(
        ValuationModelInput(
            market=Market.US,
            ticker="AAA",
            model_id="fcf_dcf_v1",
            effective_inputs=inputs,
            model_assumptions=assumptions,
        )
    )


def test_fcf_dcf_v1_computes_deterministic_model_implied_spread() -> None:
    result = _value(_inputs(), _assumptions())[0]

    assert result.projected_fcf == pytest.approx((105.0, 110.25))
    assert result.present_value_projected_fcf == pytest.approx((95.45454545, 91.11570248))
    assert result.terminal_value == pytest.approx(1405.6875)
    assert result.present_value_terminal_value == pytest.approx(1161.72520661)
    assert result.enterprise_value == pytest.approx(1348.29545455)
    assert result.equity_value == pytest.approx(1248.29545455)
    assert result.model_implied_value_per_share == pytest.approx(124.82954545)
    assert result.reference_price == pytest.approx(100.0)
    assert result.model_implied_spread_to_reference_price == pytest.approx(0.24829545)


def test_fcf_dcf_v1_adds_net_cash_and_handles_negative_fcf() -> None:
    result = _value(
        _inputs(normalized_fcf=-100.0, net_debt=-50.0, reference_price=10.0),
        _assumptions(growth_rate=0.0, discount_rate=0.10, terminal_growth_rate=0.02),
    )[0]

    assert result.enterprise_value < 0
    assert result.equity_value == pytest.approx(result.enterprise_value + 50.0)
    assert result.model_implied_spread_to_reference_price < -1


def test_fcf_dcf_v1_validates_positive_shares_and_reference_price() -> None:
    with pytest.raises(ValidationError, match="shares_outstanding"):
        _value(_inputs(shares_outstanding=0.0), _assumptions())

    with pytest.raises(ValidationError, match="reference_price"):
        _value(_inputs(reference_price=0.0), _assumptions())


def test_fcf_dcf_v1_validates_model_assumption_payload() -> None:
    model = FcfDcfV1Model()
    assumptions = model.validate_assumptions(
        {
            "forecast_years": 5,
            "terminal_method": "perpetual_growth",
            "cash_flow_basis": "normalized_fcf_enterprise_proxy",
            "discount_rate_basis": "nominal_wacc",
            "terminal_growth_basis": "nominal_perpetual_growth",
            "scenarios": {
                "base": {
                    "growth_rate": 0.05,
                    "discount_rate": 0.09,
                    "terminal_growth_rate": 0.025,
                    "note": "Middle illustrative scenario.",
                },
                "conservative": {
                    "growth_rate": -0.999,
                    "discount_rate": 0.50,
                    "terminal_growth_rate": -0.05,
                    "note": "Lower illustrative growth.",
                },
                "upside": {
                    "growth_rate": 1.0,
                    "discount_rate": 0.50,
                    "terminal_growth_rate": 0.05,
                    "note": "Higher illustrative growth.",
                },
            },
        }
    )

    assert assumptions.forecast_years == 5
    assert assumptions.cash_flow_basis == "normalized_fcf_enterprise_proxy"
    assert assumptions.discount_rate_basis == "nominal_wacc"
    assert assumptions.terminal_growth_basis == "nominal_perpetual_growth"
    assert assumptions.scenario_order == ("conservative", "base", "upside")
    assert tuple(assumptions.scenarios) == ("conservative", "base", "upside")
    assert assumptions.scenarios["conservative"].growth_rate == pytest.approx(-0.999)
    assert assumptions.scenarios["base"].discount_rate == pytest.approx(0.09)


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"forecast_years": 0}, "forecast_years"),
        ({"terminal_method": "exit_multiple"}, "terminal_method"),
        ({"cash_flow_basis": "raw_provider_fcf"}, "cash_flow_basis"),
        ({"discount_rate_basis": "cost_of_equity"}, "discount_rate_basis"),
        ({"terminal_growth_basis": "real_growth"}, "terminal_growth_basis"),
        ({"unexpected": "nope"}, "unknown fcf_dcf_v1 key"),
    ],
)
def test_fcf_dcf_v1_rejects_invalid_model_payload(patch: dict[str, object], message: str) -> None:
    payload: dict[str, object] = {
        "forecast_years": 5,
        "terminal_method": "perpetual_growth",
        "cash_flow_basis": "normalized_fcf_enterprise_proxy",
        "discount_rate_basis": "nominal_wacc",
        "terminal_growth_basis": "nominal_perpetual_growth",
        "scenarios": {
            "conservative": {
                "growth_rate": 0.03,
                "discount_rate": 0.10,
                "terminal_growth_rate": 0.02,
                "note": "Lower illustrative growth.",
            },
            "base": {
                "growth_rate": 0.05,
                "discount_rate": 0.09,
                "terminal_growth_rate": 0.025,
                "note": "Middle illustrative scenario.",
            },
            "upside": {
                "growth_rate": 0.07,
                "discount_rate": 0.085,
                "terminal_growth_rate": 0.03,
                "note": "Higher illustrative growth.",
            },
        },
    }
    payload.update(patch)

    with pytest.raises(ValidationError, match=message):
        FcfDcfV1Model().validate_assumptions(payload)


def test_fcf_dcf_v1_rejects_invalid_rates_and_scenarios() -> None:
    model = FcfDcfV1Model()
    payload = {
        "forecast_years": 5,
        "terminal_method": "perpetual_growth",
        "cash_flow_basis": "normalized_fcf_enterprise_proxy",
        "discount_rate_basis": "nominal_wacc",
        "terminal_growth_basis": "nominal_perpetual_growth",
        "scenarios": {
            "base": {
                "growth_rate": 0.05,
                "discount_rate": 0.02,
                "terminal_growth_rate": 0.025,
                "note": "Middle illustrative scenario.",
            }
        },
    }

    with pytest.raises(ValidationError, match="scenarios"):
        model.validate_assumptions(payload)

    payload["scenarios"] = {
        "conservative": {
            "growth_rate": 0.03,
            "discount_rate": 0.10,
            "terminal_growth_rate": 0.02,
            "note": "Lower illustrative growth.",
        },
        "base": {
            "growth_rate": 0.05,
            "discount_rate": 0.0,
            "terminal_growth_rate": 0.025,
            "note": "Middle illustrative scenario.",
        },
        "upside": {
            "growth_rate": 0.07,
            "discount_rate": 0.085,
            "terminal_growth_rate": 0.03,
            "note": "Higher illustrative growth.",
        },
    }

    with pytest.raises(ValidationError, match="discount_rate"):
        model.validate_assumptions(payload)


def test_valuation_model_registry_exposes_fcf_dcf_v1() -> None:
    assert supported_valuation_model_ids() == ("fcf_dcf_v1",)
    assert get_valuation_model("fcf_dcf_v1").model_id == "fcf_dcf_v1"

    with pytest.raises(ValidationError, match="unknown valuation model unknown"):
        get_valuation_model("unknown")
