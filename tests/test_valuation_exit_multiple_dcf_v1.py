from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timezone

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts, FundamentalsMetadata
from universe_selector.valuation.exit_multiple_dcf_v1 import ExitMultipleDcfV1Model, ExitMultipleDcfV1OutputRenderer
from universe_selector.valuation.output import render_valuation_json, render_valuation_markdown
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    ExitMultipleDcfScenarioAssumptions,
    ExitMultipleDcfV1Assumptions,
    StartingFcfAssumption,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationResult,
    ValuationRunInput,
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


def _assumption_set(model_assumptions: ExitMultipleDcfV1Assumptions) -> ValuationAssumptionSet:
    return ValuationAssumptionSet(
        schema_version=1,
        market=Market.US,
        ticker="AAA",
        default_model="exit_multiple_dcf_v1",
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
        model_id="exit_multiple_dcf_v1",
        model_assumptions=model_assumptions,
    )


def _model_payload() -> dict[str, object]:
    return {
        "forecast_years": 5,
        "terminal_method": "exit_multiple",
        "starting_fcf": {"method": "provider_ttm_fcf"},
        "discount_rate_basis": "nominal_wacc",
        "exit_multiple_basis": "ev_to_fcf",
        "scenarios": {
            "conservative": {
                "growth_rate": 0.02,
                "discount_rate": 0.10,
                "terminal_ev_to_fcf_multiple": 10.0,
                "note": " Lower assumption case. ",
            },
            "base": {
                "growth_rate": 0.04,
                "discount_rate": 0.09,
                "terminal_ev_to_fcf_multiple": 14.0,
                "note": "Middle assumption case.",
            },
            "upside": {
                "growth_rate": 0.06,
                "discount_rate": 0.085,
                "terminal_ev_to_fcf_multiple": 18.0,
                "note": "Higher assumption case.",
            },
        },
    }


def test_exit_multiple_dcf_v1_validates_model_payload_and_trims_notes() -> None:
    assumptions = ExitMultipleDcfV1Model().validate_assumptions(_model_payload())

    assert assumptions.forecast_years == 5
    assert assumptions.terminal_method == "exit_multiple"
    assert assumptions.starting_fcf.method == "provider_ttm_fcf"
    assert assumptions.discount_rate_basis == "nominal_wacc"
    assert assumptions.exit_multiple_basis == "ev_to_fcf"
    assert assumptions.scenario_order == ("conservative", "base", "upside")
    assert tuple(assumptions.scenarios) == ("conservative", "base", "upside")
    assert assumptions.scenarios["conservative"].note == "Lower assumption case."


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"forecast_years": 0}, "forecast_years"),
        ({"terminal_method": "perpetual_growth"}, "terminal_method"),
        ({"starting_fcf": {"method": "unknown"}}, "starting_fcf.method"),
        ({"discount_rate_basis": "cost_of_equity"}, "discount_rate_basis"),
        ({"exit_multiple_basis": "ev_to_ebitda"}, "exit_multiple_basis"),
        ({"unexpected": "nope"}, "unknown exit_multiple_dcf_v1 key"),
    ],
)
def test_exit_multiple_dcf_v1_rejects_invalid_model_payload(patch: dict[str, object], message: str) -> None:
    payload = _model_payload()
    payload.update(patch)

    with pytest.raises(ValidationError, match=message):
        ExitMultipleDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "scenario_patch, message",
    [
        ({"growth_rate": -1.0}, "growth_rate"),
        ({"growth_rate": 1.01}, "growth_rate"),
        ({"discount_rate": 0.0}, "discount_rate"),
        ({"discount_rate": 0.51}, "discount_rate"),
        ({"terminal_ev_to_fcf_multiple": 0.0}, "terminal_ev_to_fcf_multiple"),
        ({"terminal_ev_to_fcf_multiple": 100.01}, "terminal_ev_to_fcf_multiple"),
        ({"note": "   "}, "note"),
        ({"extra": "nope"}, "unknown exit_multiple_dcf_v1 scenario key"),
    ],
)
def test_exit_multiple_dcf_v1_rejects_invalid_scenario_payload(
    scenario_patch: dict[str, object],
    message: str,
) -> None:
    payload = _model_payload()
    scenarios = dict(payload["scenarios"])  # type: ignore[arg-type]
    base = dict(scenarios["base"])  # type: ignore[index]
    base.update(scenario_patch)
    scenarios["base"] = base
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match=message):
        ExitMultipleDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "scenarios_patch",
    [
        {"base": _model_payload()["scenarios"]["base"]},  # type: ignore[index]
        {
            **_model_payload()["scenarios"],  # type: ignore[arg-type]
            "stress": {
                "growth_rate": 0.0,
                "discount_rate": 0.12,
                "terminal_ev_to_fcf_multiple": 8.0,
                "note": "Extra scenario.",
            },
        },
    ],
)
def test_exit_multiple_dcf_v1_rejects_missing_or_extra_scenario_ids(
    scenarios_patch: dict[str, object],
) -> None:
    payload = _model_payload()
    payload["scenarios"] = scenarios_patch

    with pytest.raises(ValidationError, match="scenarios must contain conservative, base, and upside"):
        ExitMultipleDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "scenario_name, field, value, message",
    [
        ("conservative", "growth_rate", 0.05, "scenario growth rates"),
        ("conservative", "terminal_ev_to_fcf_multiple", 15.0, "scenario exit multiples"),
        ("conservative", "discount_rate", 0.08, "scenario discount_rate"),
    ],
)
def test_exit_multiple_dcf_v1_rejects_non_monotonic_scenarios(
    scenario_name: str,
    field: str,
    value: float,
    message: str,
) -> None:
    payload = _model_payload()
    scenarios = dict(payload["scenarios"])  # type: ignore[arg-type]
    patched = dict(scenarios[scenario_name])  # type: ignore[index]
    patched[field] = value
    scenarios[scenario_name] = patched
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match=message):
        ExitMultipleDcfV1Model().validate_assumptions(payload)


def test_exit_multiple_dcf_v1_builds_effective_inputs_from_provider_ttm_fcf() -> None:
    effective, provenance = ExitMultipleDcfV1Model().build_inputs(
        facts=_facts(),
        assumptions=_assumption_set(_assumptions()),
    )

    assert effective.starting_fcf == pytest.approx(110.0)
    assert effective.shares_outstanding == pytest.approx(10.0)
    assert effective.net_debt == pytest.approx(50.0)
    assert effective.reference_price == pytest.approx(48.0)
    assert provenance.starting_fcf_source == "provider_ttm_fcf"
    assert provenance.reference_price_source == "assumption_override"


def test_exit_multiple_dcf_v1_builds_effective_inputs_from_override_starting_fcf() -> None:
    model_assumptions = replace(
        _assumptions(),
        starting_fcf=StartingFcfAssumption(
            method="override",
            value=100.0,
            note="Normalized for one-time working capital movement.",
        ),
    )

    effective, provenance = ExitMultipleDcfV1Model().build_inputs(
        facts=_facts(),
        assumptions=_assumption_set(model_assumptions),
    )

    assert effective.starting_fcf == pytest.approx(100.0)
    assert provenance.starting_fcf_source == "assumption_override"
    assert provenance.starting_fcf_note == "Normalized for one-time working capital movement."


def _valuation_result(
    *,
    assumptions: ExitMultipleDcfV1Assumptions | None = None,
    inputs: EffectiveValuationInputs | None = None,
) -> ValuationResult:
    inputs = inputs or _inputs()
    assumptions = assumptions or _assumptions()
    scenario_results = _value(inputs, assumptions)
    assumption_set = ValuationAssumptionSet(
        schema_version=1,
        market=Market.US,
        ticker="AAA",
        default_model="exit_multiple_dcf_v1",
        purpose="research",
        as_of=date(2026, 5, 17),
        currency=inputs.currency,
        amount_unit="currency_units",
        share_basis="ordinary_share",
        valuation_basis_note="Uses USD ordinary-share basis; no ADR ratio, board-lot, or currency adjustment is applied.",
        assumption_source="analyst",
        prepared_by="Universe Selector",
        source_note="Exit multiple DCF unit test assumptions.",
        assumption_path="/tmp/valuation_assumptions/us/AAA.yaml",
        assumption_hash="abc123",
        facts_overrides={"shares_outstanding": None, "net_debt": None, "reference_price": None},
        facts_override_notes={"shares_outstanding": None, "net_debt": None, "reference_price": None},
        model_id="exit_multiple_dcf_v1",
        model_assumptions=assumptions,
    )
    raw_facts = FundamentalFacts(
        market=Market.US,
        ticker="AAA",
        currency=inputs.currency,
        reference_price=inputs.reference_price,
        reference_price_as_of=inputs.reference_price_as_of,
        reference_price_as_of_source=inputs.reference_price_as_of_source,
        reference_price_as_of_note=inputs.reference_price_as_of_note,
        shares_outstanding=inputs.shares_outstanding,
        cash_and_cash_equivalents=30.0,
        total_debt=180.0,
        balance_sheet_as_of=date(2026, 3, 31),
        net_debt=inputs.net_debt,
        operating_cash_flow=120.0,
        capital_expenditures=20.0,
        free_cash_flow=inputs.starting_fcf,
        fiscal_period_end=inputs.fiscal_period_end,
        fiscal_period_type=inputs.fiscal_period_type,
    )
    return ValuationResult(
        run_input=ValuationRunInput(
            market=Market.US,
            ticker="AAA",
            model_id="exit_multiple_dcf_v1",
            fundamentals_metadata=FundamentalsMetadata(
                data_mode="live",
                fundamentals_provider_id="fake_fundamentals",
                fundamentals_source_ids=("fake-source",),
                data_fetch_started_at=datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
                latest_source_date=date(2026, 5, 15),
            ),
            raw_facts=raw_facts,
            effective_inputs=inputs,
            input_provenance=ValuationInputProvenance(
                starting_fcf_source="provider_ttm_fcf",
                shares_outstanding_source="provider_fact",
                net_debt_source="provider_fact",
                reference_price_source="provider_fact",
                starting_fcf_note="Provider raw FCF used as starting FCF proxy; fiscal_period_type=ttm.",
                shares_outstanding_note=None,
                net_debt_note=None,
                reference_price_note=None,
            ),
            assumptions=assumption_set,
        ),
        scenario_results=scenario_results,
    )


def _model_specific_markdown(result: ValuationResult) -> str:
    renderer = ExitMultipleDcfV1OutputRenderer()
    lines: list[str] = []
    lines.extend(renderer.render_risk_disclosures(result))
    lines.extend(renderer.render_model_assumptions(result))
    lines.extend(renderer.render_scenario_results(result))
    return "\n".join(lines)


def test_exit_multiple_dcf_v1_markdown_includes_disclosures_assumptions_and_results() -> None:
    markdown = _model_specific_markdown(_valuation_result())

    assert "analyst-supplied EV / FCF exit multiple" in markdown
    assert "not peer-derived" in markdown
    assert "raw starting proxy" in markdown
    assert "not clean unlevered FCFF" in markdown
    assert "terminal_method: exit_multiple" in markdown
    assert "exit_multiple_basis: ev_to_fcf" in markdown
    assert "| base | 5.00% | 10.00% | 5.0x | unit test |" in markdown
    assert "illustrative assumption cases, not probabilities, forecasts, expected outcomes" in markdown
    assert "not target prices, forecasts, expected returns, recommendations, or signals" in markdown
    assert "terminal_value" in markdown
    assert "$551.25" in markdown
    assert "$49.21" in markdown


def test_exit_multiple_dcf_v1_markdown_redacts_and_escapes_scenario_notes() -> None:
    assumptions = replace(
        _assumptions(),
        scenarios={
            "base": ExitMultipleDcfScenarioAssumptions(
                scenario_id="base",
                growth_rate=0.05,
                discount_rate=0.10,
                terminal_ev_to_fcf_multiple=5.0,
                note="buy | target price\nsecond line",
            )
        },
    )

    markdown = _model_specific_markdown(_valuation_result(assumptions=assumptions))

    assert "not target prices, forecasts, expected returns, recommendations, or signals" in markdown
    assert "redacted" in markdown.lower()
    assert "[redacted] \\| [redacted] second line" in markdown
    assert "buy \\|" not in markdown.lower()


def test_exit_multiple_dcf_v1_markdown_renders_negative_equity_outputs() -> None:
    markdown = _model_specific_markdown(_valuation_result(inputs=_inputs(net_debt=1_000.0)))

    assert "scenario_equity_value" in markdown
    assert "$-357.85" in markdown
    assert "$-35.79" in markdown


def test_exit_multiple_dcf_v1_full_markdown_uses_registered_renderer() -> None:
    markdown = render_valuation_markdown(_valuation_result())

    assert "model_id: exit_multiple_dcf_v1" in markdown
    assert "analyst-supplied EV / FCF exit multiple" in markdown
    assert "not target prices, forecasts, expected returns, recommendations, or signals" in markdown
    assert "| base | 5.0x | $110.25 | $551.25 |" in markdown


def test_exit_multiple_dcf_v1_json_contains_common_and_model_specific_fields() -> None:
    payload = json.loads(render_valuation_json(_valuation_result()))
    notes = "\n".join(payload["notes"])

    assert payload["model_id"] == "exit_multiple_dcf_v1"
    assert payload["model_assumptions"]["forecast_years"] == 2
    assert payload["model_assumptions"]["terminal_method"] == "exit_multiple"
    assert payload["model_assumptions"]["discount_rate_basis"] == "nominal_wacc"
    assert payload["model_assumptions"]["exit_multiple_basis"] == "ev_to_fcf"
    assert payload["model_assumptions"]["scenarios"]["base"]["growth_rate"] == 0.05
    assert payload["model_assumptions"]["scenarios"]["base"]["terminal_ev_to_fcf_multiple"] == 5.0
    assert payload["scenario_results"][0]["model_metrics"]["terminal_ev_to_fcf_multiple"] == 5.0
    assert payload["scenario_results"][0]["model_metrics"]["final_year_fcf"] == 110.25
    assert "analyst-supplied EV / FCF exit multiple" in notes
    assert "peer-derived" in notes
    assert "provider TTM FCF is a raw starting proxy" in notes
    assert "not clean unlevered FCFF" in notes
    assert "requires positive starting FCF" in notes
    assert "not probabilities, forecasts, expected outcomes, target cases" in notes
    assert "not target prices, forecasts, expected returns, recommendations, or signals" in notes
