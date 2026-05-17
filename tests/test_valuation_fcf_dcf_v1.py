from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts
from universe_selector.valuation.fcf_dcf_v1 import FcfDcfV1Model
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    FcfDcfV1Assumptions,
    StartingFcfAssumption,
    ValuationAssumptionSet,
    ValuationModelInput,
    ValuationScenarioAssumptions,
)
from universe_selector.valuation.registry import get_valuation_model, supported_valuation_model_ids


def _inputs(
    *,
    starting_fcf: float = 100.0,
    shares_outstanding: float = 10.0,
    net_debt: float = 100.0,
    reference_price: float = 100.0,
) -> EffectiveValuationInputs:
    return EffectiveValuationInputs(
        starting_fcf=starting_fcf,
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
        starting_fcf=StartingFcfAssumption(method="provider_ttm_fcf", value=None, note=None),
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


def _facts() -> FundamentalFacts:
    return FundamentalFacts(
        market=Market.US,
        ticker="AAA",
        currency="USD",
        reference_price=50.0,
        reference_price_as_of=date(2026, 5, 15),
        reference_price_as_of_source="provider_reported",
        reference_price_as_of_note=None,
        shares_outstanding=10.0,
        cash_and_cash_equivalents=30.0,
        total_debt=80.0,
        balance_sheet_as_of=date(2026, 3, 31),
        net_debt=50.0,
        operating_cash_flow=120.0,
        capital_expenditures=10.0,
        free_cash_flow=110.0,
        fiscal_period_end=date(2025, 12, 31),
        fiscal_period_type="ttm",
    )


def _assumption_set(model_assumptions: FcfDcfV1Assumptions) -> ValuationAssumptionSet:
    return ValuationAssumptionSet(
        schema_version=1,
        market=Market.US,
        ticker="AAA",
        purpose="research",
        as_of=date(2026, 5, 17),
        currency="USD",
        amount_unit="currency_units",
        assumption_source="analyst",
        prepared_by="Universe Selector",
        source_note="Unit test assumptions.",
        assumption_path="/tmp/valuation_assumptions/us/AAA.yaml",
        assumption_hash="abc123",
        facts_overrides={
            "shares_outstanding": None,
            "net_debt": None,
            "reference_price": 48.0,
        },
        facts_override_notes={
            "shares_outstanding": None,
            "net_debt": None,
            "reference_price": "Reference price override for scenario review.",
        },
        model_id="fcf_dcf_v1",
        model_assumptions=model_assumptions,
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


def test_fcf_dcf_v1_builds_effective_inputs_from_provider_ttm_fcf() -> None:
    effective, provenance = FcfDcfV1Model().build_inputs(
        facts=_facts(),
        assumptions=_assumption_set(_assumptions()),
    )

    assert effective.starting_fcf == pytest.approx(110.0)
    assert effective.shares_outstanding == pytest.approx(10.0)
    assert effective.net_debt == pytest.approx(50.0)
    assert effective.reference_price == pytest.approx(48.0)
    assert effective.reference_price_as_of == date(2026, 5, 17)
    assert effective.reference_price_as_of_source == "assumption_override"
    assert effective.reference_price_as_of_note == "Reference price override for scenario review."
    assert provenance.starting_fcf_source == "provider_ttm_fcf"
    assert provenance.starting_fcf_note == "Provider raw FCF used as starting FCF proxy; fiscal_period_type=ttm."
    assert provenance.reference_price_source == "assumption_override"
    assert provenance.reference_price_note == "Reference price override for scenario review."


def test_fcf_dcf_v1_builds_effective_inputs_from_override_starting_fcf() -> None:
    model_assumptions = replace(
        _assumptions(),
        starting_fcf=StartingFcfAssumption(
            method="override",
            value=100.0,
            note="Normalized for one-time working capital movement.",
        ),
    )

    effective, provenance = FcfDcfV1Model().build_inputs(
        facts=_facts(),
        assumptions=_assumption_set(model_assumptions),
    )

    assert effective.starting_fcf == pytest.approx(100.0)
    assert provenance.starting_fcf_source == "assumption_override"
    assert provenance.starting_fcf_note == "Normalized for one-time working capital movement."


def test_fcf_dcf_v1_adds_net_cash_and_handles_negative_fcf() -> None:
    result = _value(
        _inputs(starting_fcf=-100.0, net_debt=-50.0, reference_price=10.0),
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
            "starting_fcf": {"method": "provider_ttm_fcf"},
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
    assert assumptions.starting_fcf.method == "provider_ttm_fcf"
    assert assumptions.starting_fcf.value is None
    assert assumptions.starting_fcf.note is None
    assert assumptions.discount_rate_basis == "nominal_wacc"
    assert assumptions.terminal_growth_basis == "nominal_perpetual_growth"
    assert assumptions.scenario_order == ("conservative", "base", "upside")
    assert tuple(assumptions.scenarios) == ("conservative", "base", "upside")
    assert assumptions.scenarios["conservative"].growth_rate == pytest.approx(-0.999)
    assert assumptions.scenarios["base"].discount_rate == pytest.approx(0.09)


def test_fcf_dcf_v1_accepts_override_starting_fcf_payload() -> None:
    assumptions = FcfDcfV1Model().validate_assumptions(
        {
            "forecast_years": 5,
            "terminal_method": "perpetual_growth",
            "starting_fcf": {
                "method": "override",
                "value": 100_000_000_000.0,
                "note": "Adjusted for one-time working capital movement.",
            },
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
                    "growth_rate": 0.03,
                    "discount_rate": 0.10,
                    "terminal_growth_rate": 0.02,
                    "note": "Lower illustrative growth.",
                },
                "upside": {
                    "growth_rate": 0.07,
                    "discount_rate": 0.085,
                    "terminal_growth_rate": 0.03,
                    "note": "Higher illustrative growth.",
                },
            },
        }
    )

    assert assumptions.starting_fcf.method == "override"
    assert assumptions.starting_fcf.value == pytest.approx(100_000_000_000.0)
    assert assumptions.starting_fcf.note == "Adjusted for one-time working capital movement."


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"forecast_years": 0}, "forecast_years"),
        ({"terminal_method": "exit_multiple"}, "terminal_method"),
        ({"starting_fcf": {"method": "unknown"}}, "starting_fcf.method"),
        ({"starting_fcf": {"method": "provider_ttm_fcf", "value": 100.0}}, "starting_fcf.value"),
        ({"starting_fcf": {"method": "override", "value": 100.0}}, "starting_fcf.note"),
        ({"discount_rate_basis": "cost_of_equity"}, "discount_rate_basis"),
        ({"terminal_growth_basis": "real_growth"}, "terminal_growth_basis"),
        ({"unexpected": "nope"}, "unknown fcf_dcf_v1 key"),
    ],
)
def test_fcf_dcf_v1_rejects_invalid_model_payload(patch: dict[str, object], message: str) -> None:
    payload: dict[str, object] = {
        "forecast_years": 5,
        "terminal_method": "perpetual_growth",
        "starting_fcf": {"method": "provider_ttm_fcf"},
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
        "starting_fcf": {"method": "provider_ttm_fcf"},
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
