from __future__ import annotations

from datetime import date
from typing import cast

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    StartingFcfAssumption,
    TwoStageFcfDcfScenarioAssumptions,
    TwoStageFcfDcfV1Assumptions,
    ValuationAssumptionSet,
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


def _model_payload() -> dict[str, object]:
    return {
        "stage1_years": 2,
        "stage2_years": 2,
        "terminal_method": "perpetual_growth",
        "starting_fcf": {"method": "provider_ttm_fcf"},
        "discount_rate_basis": "nominal_wacc",
        "terminal_growth_basis": "nominal_perpetual_growth",
        "scenarios": {
            "conservative": {
                "stage1_growth_rate": 0.06,
                "stage2_growth_rate": 0.03,
                "discount_rate": 0.11,
                "terminal_growth_rate": 0.02,
                "note": " Lower assumption case. ",
            },
            "base": {
                "stage1_growth_rate": 0.10,
                "stage2_growth_rate": 0.04,
                "discount_rate": 0.10,
                "terminal_growth_rate": 0.03,
                "note": " Middle assumption case. ",
            },
            "upside": {
                "stage1_growth_rate": 0.14,
                "stage2_growth_rate": 0.06,
                "discount_rate": 0.09,
                "terminal_growth_rate": 0.035,
                "note": " Higher assumption case. ",
            },
        },
    }


def _scenarios(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], payload["scenarios"])


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


def _assumption_set(model_assumptions: TwoStageFcfDcfV1Assumptions) -> ValuationAssumptionSet:
    return ValuationAssumptionSet(
        schema_version=1,
        market=Market.US,
        ticker="AAA",
        default_model="two_stage_fcf_dcf_v1",
        purpose="research",
        as_of=date(2026, 5, 17),
        currency="USD",
        amount_unit="currency_units",
        assumption_source="analyst",
        prepared_by="Universe Selector",
        source_note="Unit test assumptions.",
        assumption_path="/tmp/valuation_assumptions/us/AAA.yaml",
        assumption_hash="abc123",
        facts_overrides={"shares_outstanding": None, "net_debt": None, "reference_price": 48.0},
        facts_override_notes={
            "shares_outstanding": None,
            "net_debt": None,
            "reference_price": "Reference price supplied for scenario review.",
        },
        model_id="two_stage_fcf_dcf_v1",
        model_assumptions=model_assumptions,
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


def test_two_stage_fcf_dcf_v1_validates_model_payload_trims_notes_and_freezes_scenarios() -> None:
    assumptions = TwoStageFcfDcfV1Model().validate_assumptions(_model_payload())

    assert assumptions.stage1_years == 2
    assert assumptions.stage2_years == 2
    assert assumptions.terminal_method == "perpetual_growth"
    assert assumptions.starting_fcf.method == "provider_ttm_fcf"
    assert assumptions.discount_rate_basis == "nominal_wacc"
    assert assumptions.terminal_growth_basis == "nominal_perpetual_growth"
    assert assumptions.scenario_order == ("conservative", "base", "upside")
    assert assumptions.scenarios["base"].note == "Middle assumption case."

    with pytest.raises(TypeError):
        assumptions.scenarios["base"] = assumptions.scenarios["conservative"]  # type: ignore[index]


def test_two_stage_fcf_dcf_v1_rejects_unknown_model_and_scenario_keys() -> None:
    payload = _model_payload()
    payload["unknown"] = "nope"
    with pytest.raises(ValidationError, match="unknown two_stage_fcf_dcf_v1 key"):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)

    payload = _model_payload()
    scenarios = dict(_scenarios(payload))
    base = dict(scenarios["base"])
    base["unknown"] = "nope"
    scenarios["base"] = base
    payload["scenarios"] = scenarios
    with pytest.raises(ValidationError, match="unknown two_stage_fcf_dcf_v1 scenario key"):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "scenarios",
    [
        {"conservative": {}, "base": {}},
        {"conservative": {}, "base": {}, "upside": {}, "aggressive": {}},
    ],
)
def test_two_stage_fcf_dcf_v1_rejects_missing_or_extra_scenario_ids(scenarios: dict[str, object]) -> None:
    payload = _model_payload()
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match="scenarios"):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("stage1_years", 0, "stage1_years"),
        ("stage1_years", 10, "stage1_years"),
        ("stage2_years", 0, "stage2_years"),
        ("stage2_years", 10, "stage2_years"),
    ],
)
def test_two_stage_fcf_dcf_v1_rejects_stage_year_bounds(field: str, value: int, message: str) -> None:
    payload = _model_payload()
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


def test_two_stage_fcf_dcf_v1_rejects_total_explicit_years_above_ten() -> None:
    payload = _model_payload()
    payload["stage1_years"] = 6
    payload["stage2_years"] = 5

    with pytest.raises(ValidationError, match="stage1_years \\+ stage2_years"):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("terminal_method", "exit_multiple", "terminal_method"),
        ("discount_rate_basis", "real_wacc", "discount_rate_basis"),
        ("terminal_growth_basis", "real_perpetual_growth", "terminal_growth_basis"),
    ],
)
def test_two_stage_fcf_dcf_v1_rejects_invalid_literals(field: str, value: str, message: str) -> None:
    payload = _model_payload()
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "starting_fcf",
    [
        {"method": "unsupported"},
        {"method": "override", "note": "missing value"},
        {"method": "override", "value": 100.0},
        {"method": "override", "value": 100.0, "note": ""},
        {"method": "override", "value": 100.0, "note": 123},
        {"method": "provider_ttm_fcf", "value": 100.0},
        {"method": "provider_ttm_fcf", "note": "not allowed"},
    ],
)
def test_two_stage_fcf_dcf_v1_rejects_invalid_starting_fcf_schema(starting_fcf: dict[str, object]) -> None:
    payload = _model_payload()
    payload["starting_fcf"] = starting_fcf

    with pytest.raises(ValidationError, match="starting_fcf"):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


def test_two_stage_fcf_dcf_v1_preserves_override_note_without_coercion() -> None:
    payload = _model_payload()
    payload["starting_fcf"] = {
        "method": "override",
        "value": 123.0,
        "note": " Normalized unlevered FCFF estimate. ",
    }

    assumptions = TwoStageFcfDcfV1Model().validate_assumptions(payload)

    assert assumptions.starting_fcf.value == pytest.approx(123.0)
    assert assumptions.starting_fcf.note == " Normalized unlevered FCFF estimate. "


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("stage1_growth_rate", -1.0, "stage1_growth_rate"),
        ("stage1_growth_rate", 1.01, "stage1_growth_rate"),
        ("stage1_growth_rate", "0.10", "stage1_growth_rate"),
        ("stage1_growth_rate", float("inf"), "stage1_growth_rate"),
        ("stage2_growth_rate", -1.0, "stage2_growth_rate"),
        ("stage2_growth_rate", 1.01, "stage2_growth_rate"),
        ("stage2_growth_rate", "0.04", "stage2_growth_rate"),
        ("stage2_growth_rate", float("inf"), "stage2_growth_rate"),
        ("terminal_growth_rate", -0.051, "terminal_growth_rate"),
        ("terminal_growth_rate", 0.051, "terminal_growth_rate"),
        ("terminal_growth_rate", "0.03", "terminal_growth_rate"),
        ("terminal_growth_rate", float("inf"), "terminal_growth_rate"),
        ("discount_rate", 0.0, "discount_rate"),
        ("discount_rate", 0.51, "discount_rate"),
        ("discount_rate", "0.10", "discount_rate"),
        ("discount_rate", float("inf"), "discount_rate"),
    ],
)
def test_two_stage_fcf_dcf_v1_rejects_invalid_scenario_rate_bounds(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _model_payload()
    scenarios = dict(_scenarios(payload))
    base = dict(scenarios["base"])
    base[field] = value
    scenarios["base"] = base
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match=message):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


def test_two_stage_fcf_dcf_v1_rejects_discount_terminal_growth_spread_below_three_points() -> None:
    payload = _model_payload()
    scenarios = dict(_scenarios(payload))
    base = dict(scenarios["base"])
    base["discount_rate"] = 0.05
    base["terminal_growth_rate"] = 0.03
    scenarios["base"] = base
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match="discount_rate - terminal_growth_rate"):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "note",
    ["", "   ", 123],
)
def test_two_stage_fcf_dcf_v1_rejects_invalid_scenario_notes(note: object) -> None:
    payload = _model_payload()
    scenarios = dict(_scenarios(payload))
    base = dict(scenarios["base"])
    base["note"] = note
    scenarios["base"] = base
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match="note"):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("stage1_growth_rate", 0.03, "stage1_growth_rate"),
        ("stage2_growth_rate", 0.025, "stage2_growth_rate"),
    ],
)
def test_two_stage_fcf_dcf_v1_rejects_non_fading_scenario_growth(
    field: str,
    value: float,
    message: str,
) -> None:
    payload = _model_payload()
    scenarios = dict(_scenarios(payload))
    base = dict(scenarios["base"])
    base[field] = value
    scenarios["base"] = base
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match=message):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "scenario_name, field, value, message",
    [
        ("conservative", "stage1_growth_rate", 0.11, "scenario stage1_growth_rate"),
        ("conservative", "stage2_growth_rate", 0.05, "scenario stage2_growth_rate"),
        ("base", "terminal_growth_rate", 0.015, "scenario terminal_growth_rate"),
        ("conservative", "discount_rate", 0.09, "scenario discount_rate"),
    ],
)
def test_two_stage_fcf_dcf_v1_rejects_non_monotonic_scenarios(
    scenario_name: str,
    field: str,
    value: float,
    message: str,
) -> None:
    payload = _model_payload()
    scenarios = dict(_scenarios(payload))
    patched = dict(scenarios[scenario_name])
    patched[field] = value
    scenarios[scenario_name] = patched
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match=message):
        TwoStageFcfDcfV1Model().validate_assumptions(payload)


def test_two_stage_fcf_dcf_v1_builds_effective_inputs_from_provider_ttm_fcf() -> None:
    assumptions = TwoStageFcfDcfV1Model().validate_assumptions(_model_payload())
    effective, provenance = TwoStageFcfDcfV1Model().build_inputs(
        facts=_facts(),
        assumptions=_assumption_set(assumptions),
    )

    assert effective.starting_fcf == pytest.approx(110.0)
    assert effective.shares_outstanding == pytest.approx(10.0)
    assert effective.net_debt == pytest.approx(50.0)
    assert effective.reference_price == pytest.approx(48.0)
    assert effective.reference_price_as_of == date(2026, 5, 17)
    assert effective.reference_price_as_of_source == "assumption_override"
    assert provenance.starting_fcf_source == "provider_ttm_fcf"
    assert "Provider raw FCF used as starting FCF proxy" in (provenance.starting_fcf_note or "")
    assert provenance.reference_price_source == "assumption_override"
    assert provenance.reference_price_note == "Reference price supplied for scenario review."


def test_two_stage_fcf_dcf_v1_builds_effective_inputs_from_override_starting_fcf() -> None:
    payload = _model_payload()
    payload["starting_fcf"] = {
        "method": "override",
        "value": 123.0,
        "note": "Normalized unlevered FCFF estimate.",
    }
    assumptions = TwoStageFcfDcfV1Model().validate_assumptions(payload)
    effective, provenance = TwoStageFcfDcfV1Model().build_inputs(
        facts=_facts(),
        assumptions=_assumption_set(assumptions),
    )

    assert effective.starting_fcf == pytest.approx(123.0)
    assert provenance.starting_fcf_source == "assumption_override"
    assert provenance.starting_fcf_note == "Normalized unlevered FCFF estimate."
