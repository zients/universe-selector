from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime, timezone

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts, FundamentalsMetadata
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    MultipleValuationScenarioAssumptions,
    MultipleValuationV1Assumptions,
    StartingFcfAssumption,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationResult,
    ValuationRunInput,
)
from universe_selector.valuation.multiple_valuation_v1 import MultipleValuationV1Model
from universe_selector.valuation.output import render_valuation_markdown


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
        reference_price_as_of_note="Provider reported close.",
    )


def _assumptions(
    *,
    conservative_multiple: float = 4.0,
    base_multiple: float = 5.0,
    upside_multiple: float = 6.0,
    base_note: str = "Middle assumption case.",
) -> MultipleValuationV1Assumptions:
    return MultipleValuationV1Assumptions(
        starting_fcf=StartingFcfAssumption(method="provider_ttm_fcf", value=None, note=None),
        multiple_basis="ev_to_fcf",
        scenario_order=("conservative", "base", "upside"),
        scenarios={
            "conservative": MultipleValuationScenarioAssumptions(
                scenario_id="conservative",
                ev_to_fcf_multiple=conservative_multiple,
                note="Lower assumption case.",
            ),
            "base": MultipleValuationScenarioAssumptions(
                scenario_id="base",
                ev_to_fcf_multiple=base_multiple,
                note=base_note,
            ),
            "upside": MultipleValuationScenarioAssumptions(
                scenario_id="upside",
                ev_to_fcf_multiple=upside_multiple,
                note="Higher assumption case.",
            ),
        },
    )


def _value(inputs: EffectiveValuationInputs, assumptions: MultipleValuationV1Assumptions):
    return MultipleValuationV1Model().value(
        ValuationModelInput(
            market=Market.US,
            ticker="AAA",
            model_id="multiple_valuation_v1",
            effective_inputs=inputs,
            model_assumptions=assumptions,
        )
    )


def _model_payload() -> dict[str, object]:
    return {
        "starting_fcf": {"method": "provider_ttm_fcf"},
        "multiple_basis": "ev_to_fcf",
        "scenarios": {
            "base": {
                "ev_to_fcf_multiple": 8.0,
                "note": " Middle assumption case. ",
            },
            "conservative": {
                "ev_to_fcf_multiple": 6.0,
                "note": "Lower assumption case.",
            },
            "upside": {
                "ev_to_fcf_multiple": 10.0,
                "note": "Higher assumption case.",
            },
        },
    }


def _valuation_result(
    *,
    assumptions: MultipleValuationV1Assumptions | None = None,
    inputs: EffectiveValuationInputs | None = None,
) -> ValuationResult:
    inputs = inputs or _inputs()
    assumptions = assumptions or _assumptions(base_note="buy | target price\nsecond line")
    scenario_results = _value(inputs, assumptions)
    assumption_set = ValuationAssumptionSet(
        schema_version=1,
        market=Market.US,
        ticker="AAA",
        default_model="fcf_dcf_v1",
        purpose="research",
        as_of=date(2026, 5, 17),
        currency=inputs.currency,
        amount_unit="currency_units",
        share_basis="ordinary_share",
        valuation_basis_note="Uses USD ordinary-share basis; no ADR ratio, board-lot, or currency adjustment is applied.",
        assumption_source="analyst",
        prepared_by="Universe Selector",
        source_note="Multiple valuation test assumptions.",
        assumption_path="/tmp/valuation_assumptions/us/AAA.yaml",
        assumption_hash="abc123",
        facts_overrides={"shares_outstanding": None, "net_debt": None, "reference_price": None},
        facts_override_notes={"shares_outstanding": None, "net_debt": None, "reference_price": None},
        model_id="multiple_valuation_v1",
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
            model_id="multiple_valuation_v1",
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


def test_multiple_valuation_v1_validates_model_payload_trims_notes_and_freezes_scenarios() -> None:
    assumptions = MultipleValuationV1Model().validate_assumptions(_model_payload())

    assert assumptions.starting_fcf.method == "provider_ttm_fcf"
    assert assumptions.multiple_basis == "ev_to_fcf"
    assert assumptions.scenario_order == ("conservative", "base", "upside")
    assert tuple(assumptions.scenarios) == ("conservative", "base", "upside")
    assert assumptions.scenarios["base"].ev_to_fcf_multiple == pytest.approx(8.0)
    assert assumptions.scenarios["base"].note == "Middle assumption case."
    with pytest.raises(TypeError):
        assumptions.scenarios["base"] = assumptions.scenarios["conservative"]  # type: ignore[index]


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"multiple_basis": "price_to_fcf"}, "multiple_basis"),
        ({"starting_fcf": {"method": "unknown"}}, "starting_fcf.method"),
        ({"unexpected": "nope"}, "unknown multiple_valuation_v1 key"),
    ],
)
def test_multiple_valuation_v1_rejects_invalid_model_payload(patch: dict[str, object], message: str) -> None:
    payload = _model_payload()
    payload.update(patch)

    with pytest.raises(ValidationError, match=message):
        MultipleValuationV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "scenario_patch, message",
    [
        ({"ev_to_fcf_multiple": 0.0}, "ev_to_fcf_multiple"),
        ({"ev_to_fcf_multiple": 100.01}, "ev_to_fcf_multiple"),
        ({"note": "   "}, "note"),
        ({"extra": "nope"}, "unknown multiple_valuation_v1 scenario key"),
    ],
)
def test_multiple_valuation_v1_rejects_invalid_multiple_bounds_and_scenarios(
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
        MultipleValuationV1Model().validate_assumptions(payload)

    payload = _model_payload()
    payload["scenarios"] = {"base": dict(scenarios["base"])}  # type: ignore[index]
    with pytest.raises(ValidationError, match="scenarios must contain conservative, base, and upside"):
        MultipleValuationV1Model().validate_assumptions(payload)


def test_multiple_valuation_v1_rejects_non_monotonic_scenario_multiples() -> None:
    payload = _model_payload()
    scenarios = dict(payload["scenarios"])  # type: ignore[arg-type]
    conservative = dict(scenarios["conservative"])  # type: ignore[index]
    conservative["ev_to_fcf_multiple"] = 9.0
    scenarios["conservative"] = conservative
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match="conservative <= base <= upside"):
        MultipleValuationV1Model().validate_assumptions(payload)


def test_multiple_valuation_v1_calculates_ev_equity_per_share_spread_and_metrics() -> None:
    result = _value(_inputs(), _assumptions())[1]

    assert result.scenario_id == "base"
    assert result.enterprise_value == pytest.approx(500.0)
    assert result.equity_value == pytest.approx(350.0)
    assert result.model_implied_value_per_share == pytest.approx(35.0)
    assert result.reference_price == pytest.approx(20.0)
    assert result.model_implied_spread_to_reference_price == pytest.approx(0.75)
    assert result.projected_fcf == ()
    assert result.present_value_projected_fcf == ()
    assert result.terminal_value == pytest.approx(0.0)
    assert result.present_value_terminal_value == pytest.approx(0.0)
    assert result.model_metrics["ev_to_fcf_multiple"] == pytest.approx(5.0)


def test_multiple_valuation_v1_allows_negative_net_debt_and_visible_negative_equity() -> None:
    net_cash = _value(_inputs(net_debt=-50.0), _assumptions())[1]

    assert net_cash.enterprise_value == pytest.approx(500.0)
    assert net_cash.equity_value == pytest.approx(550.0)
    assert net_cash.model_implied_value_per_share == pytest.approx(55.0)

    negative_equity = _value(_inputs(net_debt=700.0), _assumptions())[1]

    assert negative_equity.enterprise_value == pytest.approx(500.0)
    assert negative_equity.equity_value == pytest.approx(-200.0)
    assert negative_equity.model_implied_value_per_share == pytest.approx(-20.0)
    assert negative_equity.model_implied_spread_to_reference_price == pytest.approx(-2.0)


def test_multiple_valuation_v1_rejects_non_positive_starting_fcf_shares_and_reference_price() -> None:
    with pytest.raises(ValidationError, match="starting_fcf"):
        _value(_inputs(starting_fcf=0.0), _assumptions())
    with pytest.raises(ValidationError, match="shares_outstanding"):
        _value(_inputs(shares_outstanding=0.0), _assumptions())
    with pytest.raises(ValidationError, match="reference_price"):
        _value(_inputs(reference_price=0.0), _assumptions())


def test_multiple_valuation_v1_build_inputs_rejects_non_positive_provider_and_override_starting_fcf() -> None:
    facts = FundamentalFacts(
        market=Market.US,
        ticker="AAA",
        currency="USD",
        reference_price=20.0,
        reference_price_as_of=date(2026, 5, 15),
        reference_price_as_of_source="provider_reported",
        reference_price_as_of_note=None,
        shares_outstanding=10.0,
        cash_and_cash_equivalents=30.0,
        total_debt=180.0,
        balance_sheet_as_of=date(2026, 3, 31),
        net_debt=150.0,
        operating_cash_flow=90.0,
        capital_expenditures=100.0,
        free_cash_flow=-10.0,
        fiscal_period_end=date(2025, 12, 31),
        fiscal_period_type="ttm",
    )
    assumption_set = ValuationAssumptionSet(
        schema_version=1,
        market=Market.US,
        ticker="AAA",
        default_model="fcf_dcf_v1",
        purpose="research",
        as_of=date(2026, 5, 17),
        currency="USD",
        amount_unit="currency_units",
        assumption_source="analyst",
        prepared_by="Universe Selector",
        source_note="Test assumptions.",
        assumption_path="/tmp/AAA.yaml",
        assumption_hash="abc123",
        facts_overrides={"shares_outstanding": None, "net_debt": None, "reference_price": None},
        facts_override_notes={"shares_outstanding": None, "net_debt": None, "reference_price": None},
        model_id="multiple_valuation_v1",
        model_assumptions=_assumptions(),
    )

    model = MultipleValuationV1Model()
    effective, _ = model.build_inputs(facts=facts, assumptions=assumption_set)
    with pytest.raises(ValidationError, match="starting_fcf"):
        model.value(
            ValuationModelInput(
                market=Market.US,
                ticker="AAA",
                model_id="multiple_valuation_v1",
                effective_inputs=effective,
                model_assumptions=assumption_set.model_assumptions,
            )
        )

    override_assumptions = replace(
        _assumptions(),
        starting_fcf=StartingFcfAssumption(method="override", value=-1.0, note="Normalized loss."),
    )
    effective, _ = model.build_inputs(
        facts=replace(facts, free_cash_flow=100.0),
        assumptions=replace(assumption_set, model_assumptions=override_assumptions),
    )
    with pytest.raises(ValidationError, match="starting_fcf"):
        model.value(
            ValuationModelInput(
                market=Market.US,
                ticker="AAA",
                model_id="multiple_valuation_v1",
                effective_inputs=effective,
                model_assumptions=override_assumptions,
            )
        )


def test_multiple_valuation_v1_output_includes_bridge_wording_and_redacted_metrics() -> None:
    result = _valuation_result()
    markdown = render_valuation_markdown(result)

    assert "model_id: multiple_valuation_v1" in markdown
    assert "EV / FCF multiple" in markdown
    assert "model-implied value per share" in markdown
    assert "reference price" in markdown
    assert "spread vs reference price" in markdown
    assert "analyst-supplied" in markdown
    assert "not peer-derived" in markdown
    assert "Spread is descriptive and is not an investment signal." in markdown
    assert "EV / FCF multiple valuation is not meaningful when starting FCF is zero or negative." in markdown
    assert "Displayed precision is formatting precision." in markdown
    assert "scenario_enterprise_value" in markdown
    assert "net_debt" in markdown
    assert "scenario_equity_value" in markdown
    assert "starting_fcf" in markdown
    assert "shares_outstanding" in markdown
    assert "reference_price_as_of" in markdown
    assert "reference_price_as_of_source" in markdown
    assert "Provider reported close." in markdown
    assert "currency: USD" in markdown
    assert "fiscal_period_end" in markdown
    assert "fiscal_period_type" in markdown
    assert "share_basis: ordinary_share" in markdown
    assert "valuation_basis_note: Uses USD ordinary-share basis" in markdown
    assert "[redacted] \\| [redacted] second line" in markdown

    for word in ("buy", "sell", "hold"):
        assert re.search(rf"\b{word}\b", markdown, flags=re.IGNORECASE) is None
    for phrase in ("target price", "fair value", "undervalued", "overvalued", "expected return"):
        assert phrase not in markdown.lower()


def test_multiple_valuation_v1_output_reads_displayed_multiple_from_model_metrics() -> None:
    result = _valuation_result()
    scenario = replace(
        result.scenario_results[1],
        model_metrics={
            **result.scenario_results[1].model_metrics,
            "ev_to_fcf_multiple": 7.25,
        },
    )
    result = replace(result, scenario_results=(scenario,))

    markdown = render_valuation_markdown(result)

    assert "7.25x" in markdown


def test_multiple_valuation_v1_tw_ordinary_share_twd_applies_no_ratio_lot_or_currency_adjustment() -> None:
    inputs = _inputs(
        starting_fcf=700_000_000_000.0,
        shares_outstanding=25_900_000_000.0,
        net_debt=-1_000_000_000_000.0,
        reference_price=800.0,
        currency="TWD",
    )
    result = _value(inputs, _assumptions(base_multiple=12.0))[1]

    assert result.enterprise_value == pytest.approx(8_400_000_000_000.0)
    assert result.equity_value == pytest.approx(9_400_000_000_000.0)
    assert result.model_implied_value_per_share == pytest.approx(362.934362934)
