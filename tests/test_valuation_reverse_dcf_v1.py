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
    ReverseDcfScenarioAssumptions,
    ReverseDcfV1Assumptions,
    StartingFcfAssumption,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationResult,
    ValuationRunInput,
)
from universe_selector.valuation.output import render_valuation_markdown
from universe_selector.valuation.reverse_dcf_v1 import ReverseDcfV1Model, _solve_implied_growth


def _inputs(
    *,
    starting_fcf: float = 100.0,
    shares_outstanding: float = 10.0,
    net_debt: float = 100.0,
    reference_price: float = 124.82954545,
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
    scenario_id: str = "base",
    lower_bound: float = -0.20,
    upper_bound: float = 0.30,
    note: str = "Middle assumption case.",
) -> ReverseDcfV1Assumptions:
    return ReverseDcfV1Assumptions(
        forecast_years=2,
        terminal_method="perpetual_growth",
        starting_fcf=StartingFcfAssumption(method="provider_ttm_fcf", value=None, note=None),
        discount_rate_basis="nominal_wacc",
        terminal_growth_basis="nominal_perpetual_growth",
        implied_growth_basis="constant_explicit_fcf_growth",
        solver_abs_tolerance=0.000001,
        solver_max_iterations=100,
        scenario_order=(scenario_id,),
        scenarios={
            scenario_id: ReverseDcfScenarioAssumptions(
                scenario_id=scenario_id,
                discount_rate=0.10,
                terminal_growth_rate=0.02,
                implied_growth_lower_bound=lower_bound,
                implied_growth_upper_bound=upper_bound,
                note=note,
            )
        },
    )


def _value(inputs: EffectiveValuationInputs, assumptions: ReverseDcfV1Assumptions):
    return ReverseDcfV1Model().value(
        ValuationModelInput(
            market=Market.US,
            ticker="AAA",
            model_id="reverse_dcf_v1",
            effective_inputs=inputs,
            model_assumptions=assumptions,
        )
    )


def _model_payload() -> dict[str, object]:
    return {
        "forecast_years": 5,
        "terminal_method": "perpetual_growth",
        "starting_fcf": {"method": "provider_ttm_fcf"},
        "discount_rate_basis": "nominal_wacc",
        "terminal_growth_basis": "nominal_perpetual_growth",
        "implied_growth_basis": "constant_explicit_fcf_growth",
        "solver_abs_tolerance": 0.000001,
        "solver_max_iterations": 100,
        "scenarios": {
            "base": {
                "discount_rate": 0.09,
                "terminal_growth_rate": 0.025,
                "implied_growth_lower_bound": -0.10,
                "implied_growth_upper_bound": 0.30,
                "note": " Middle assumption case. ",
            },
            "conservative": {
                "discount_rate": 0.10,
                "terminal_growth_rate": 0.02,
                "implied_growth_lower_bound": -0.20,
                "implied_growth_upper_bound": 0.20,
                "note": "Lower assumption case.",
            },
            "upside": {
                "discount_rate": 0.085,
                "terminal_growth_rate": 0.03,
                "implied_growth_lower_bound": -0.05,
                "implied_growth_upper_bound": 0.40,
                "note": "Higher assumption case.",
            },
        },
    }


def _valuation_result(
    *,
    assumptions: ReverseDcfV1Assumptions | None = None,
    inputs: EffectiveValuationInputs | None = None,
) -> ValuationResult:
    inputs = inputs or _inputs()
    assumptions = assumptions or _assumptions(note="buy | target price\nsecond line")
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
        source_note="Reverse DCF test assumptions.",
        assumption_path="/tmp/valuation_assumptions/us/AAA.yaml",
        assumption_hash="abc123",
        facts_overrides={"shares_outstanding": None, "net_debt": None, "reference_price": None},
        facts_override_notes={"shares_outstanding": None, "net_debt": None, "reference_price": None},
        model_id="reverse_dcf_v1",
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
        total_debt=130.0,
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
            model_id="reverse_dcf_v1",
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


def test_reverse_dcf_v1_validates_model_payload_and_trims_notes() -> None:
    assumptions = ReverseDcfV1Model().validate_assumptions(_model_payload())

    assert assumptions.forecast_years == 5
    assert assumptions.terminal_method == "perpetual_growth"
    assert assumptions.starting_fcf.method == "provider_ttm_fcf"
    assert assumptions.discount_rate_basis == "nominal_wacc"
    assert assumptions.terminal_growth_basis == "nominal_perpetual_growth"
    assert assumptions.implied_growth_basis == "constant_explicit_fcf_growth"
    assert assumptions.solver_abs_tolerance == pytest.approx(0.000001)
    assert assumptions.solver_max_iterations == 100
    assert assumptions.scenario_order == ("conservative", "base", "upside")
    assert tuple(assumptions.scenarios) == ("conservative", "base", "upside")
    assert assumptions.scenarios["base"].note == "Middle assumption case."
    with pytest.raises(TypeError):
        assumptions.scenarios["base"] = assumptions.scenarios["conservative"]  # type: ignore[index]


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"forecast_years": 0}, "forecast_years"),
        ({"terminal_method": "exit_multiple"}, "terminal_method"),
        ({"starting_fcf": {"method": "unknown"}}, "starting_fcf.method"),
        ({"discount_rate_basis": "cost_of_equity"}, "discount_rate_basis"),
        ({"terminal_growth_basis": "real_growth"}, "terminal_growth_basis"),
        ({"implied_growth_basis": "next_year_growth"}, "implied_growth_basis"),
        ({"solver_abs_tolerance": 0.0}, "solver_abs_tolerance"),
        ({"solver_max_iterations": 0}, "solver_max_iterations"),
        ({"unexpected": "nope"}, "unknown reverse_dcf_v1 key"),
    ],
)
def test_reverse_dcf_v1_rejects_invalid_model_payload(patch: dict[str, object], message: str) -> None:
    payload = _model_payload()
    payload.update(patch)

    with pytest.raises(ValidationError, match=message):
        ReverseDcfV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "scenario_patch, message",
    [
        ({"discount_rate": 0.0}, "discount_rate"),
        ({"discount_rate": 0.51}, "discount_rate"),
        ({"terminal_growth_rate": -0.06}, "terminal_growth_rate"),
        ({"terminal_growth_rate": 0.06}, "terminal_growth_rate"),
        ({"discount_rate": 0.04, "terminal_growth_rate": 0.02}, "discount_rate - terminal_growth_rate"),
        ({"implied_growth_lower_bound": -1.0}, "implied_growth_lower_bound"),
        ({"implied_growth_upper_bound": 1.01}, "implied_growth_upper_bound"),
        (
            {"implied_growth_lower_bound": 0.20, "implied_growth_upper_bound": 0.10},
            "implied growth lower bound must be below upper bound",
        ),
        ({"note": "   "}, "note"),
        ({"extra": "nope"}, "unknown reverse_dcf_v1 scenario key"),
    ],
)
def test_reverse_dcf_v1_rejects_invalid_rates_growth_bounds_and_scenarios(
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
        ReverseDcfV1Model().validate_assumptions(payload)

    payload = _model_payload()
    payload["scenarios"] = {"base": dict(scenarios["base"])}  # type: ignore[index]
    with pytest.raises(ValidationError, match="scenarios must contain conservative, base, and upside"):
        ReverseDcfV1Model().validate_assumptions(payload)


def test_reverse_dcf_v1_solves_deterministic_implied_growth_and_metrics() -> None:
    result = _value(_inputs(), _assumptions())[0]

    assert result.scenario_id == "base"
    assert result.model_metrics["implied_growth_rate"] == pytest.approx(0.05, abs=1e-8)
    assert result.model_metrics["solver_abs_residual"] == pytest.approx(0.0, abs=1e-5)
    assert result.projected_fcf == pytest.approx((105.0, 110.25))
    assert result.enterprise_value == pytest.approx(1348.2954545, rel=1e-8)
    assert result.equity_value == pytest.approx(1248.2954545, rel=1e-8)
    assert result.model_implied_value_per_share == pytest.approx(_inputs().reference_price, abs=1e-5)
    assert result.model_implied_spread_to_reference_price == pytest.approx(0.0, abs=1e-7)


def test_reverse_dcf_v1_accepts_endpoint_solution_and_negative_net_debt() -> None:
    endpoint_inputs = _inputs(net_debt=-50.0, reference_price=127.72727273)
    result = _value(endpoint_inputs, _assumptions(lower_bound=0.0, upper_bound=0.20))[0]

    assert result.model_metrics["implied_growth_rate"] == pytest.approx(0.0, abs=1e-8)
    assert result.equity_value == pytest.approx(result.enterprise_value + 50.0)
    assert result.model_implied_value_per_share == pytest.approx(endpoint_inputs.reference_price, abs=1e-5)


def test_reverse_dcf_v1_rejects_unusable_inputs_and_unbracketed_growth() -> None:
    with pytest.raises(ValidationError, match="starting_fcf"):
        _value(_inputs(starting_fcf=0.0), _assumptions())
    with pytest.raises(ValidationError, match="shares_outstanding"):
        _value(_inputs(shares_outstanding=0.0), _assumptions())
    with pytest.raises(ValidationError, match="reference_price"):
        _value(_inputs(reference_price=0.0), _assumptions())
    with pytest.raises(ValidationError, match="reference-implied enterprise value"):
        _value(_inputs(net_debt=-2_000.0, reference_price=100.0), _assumptions())
    with pytest.raises(ValidationError, match="base.*-0.2000.*0.3000"):
        _value(_inputs(reference_price=1_000.0), _assumptions())


def test_reverse_dcf_v1_exposes_non_convergence_error_path() -> None:
    with pytest.raises(ValidationError, match="did not converge"):
        _solve_implied_growth(
            scenario_id="base",
            starting_fcf=100.0,
            target_enterprise_value=1_348.2954545,
            forecast_years=2,
            discount_rate=0.10,
            terminal_growth_rate=0.02,
            lower_bound=-0.20,
            upper_bound=0.30,
            abs_tolerance=0.000001,
            max_iterations=0,
        )


def test_reverse_dcf_v1_solver_does_not_silently_return_above_tolerance() -> None:
    with pytest.raises(ValidationError, match="did not converge"):
        _solve_implied_growth(
            scenario_id="base",
            starting_fcf=700_000_000_000.0,
            target_enterprise_value=9_438_068_181_847.594,
            forecast_years=2,
            discount_rate=0.10,
            terminal_growth_rate=0.02,
            lower_bound=0.04,
            upper_bound=0.06,
            abs_tolerance=0.000001,
            max_iterations=100,
        )


def test_reverse_dcf_v1_output_includes_bridge_wording_and_redacted_metrics() -> None:
    result = _valuation_result()
    markdown = render_valuation_markdown(result)

    assert "model_id: reverse_dcf_v1" in markdown
    assert "implied annual FCF growth" in markdown
    assert "absolute per-share reconciliation residual" in markdown
    assert "discount_rate" in markdown
    assert "terminal_growth_rate" in markdown
    assert "implied growth is required to reconcile to the reference price under the stated assumptions" in markdown
    assert "Scenario rows are assumption cases" in markdown
    assert "not a forecast or investment signal" in markdown
    assert "Solved growth applies only to the explicit forecast period; terminal growth is separate." in markdown
    assert "Displayed precision is formatting precision." in markdown
    assert "Spread is a descriptive reconciliation value, not an investment signal." in markdown
    assert "reference_equity_value" in markdown
    assert "reference_implied_enterprise_value" in markdown
    assert "model_implied_enterprise_value" in markdown
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
    assert "$0.0000" in markdown
    assert "[redacted] \\| [redacted] second line" in markdown
    assert "target enterprise value" not in markdown.lower()

    for word in ("buy", "sell", "hold"):
        assert re.search(rf"\b{word}\b", markdown, flags=re.IGNORECASE) is None
    for phrase in ("target price", "fair value", "undervalued", "overvalued", "expected return"):
        assert phrase not in markdown.lower()


def test_reverse_dcf_v1_output_reads_displayed_metrics_from_model_metrics() -> None:
    result = _valuation_result()
    scenario = replace(
        result.scenario_results[0],
        model_metrics={
            **result.scenario_results[0].model_metrics,
            "implied_growth_rate": 0.1234,
            "solver_abs_residual": 0.98765,
        },
    )
    result = replace(result, scenario_results=(scenario,))

    markdown = render_valuation_markdown(result)

    assert "12.34%" in markdown
    assert "$0.9877" in markdown


def test_reverse_dcf_v1_tw_ordinary_share_twd_applies_no_ratio_lot_or_currency_adjustment() -> None:
    inputs = _inputs(
        starting_fcf=700_000_000_000.0,
        shares_outstanding=25_900_000_000.0,
        net_debt=-1_000_000_000_000.0,
        reference_price=403.014215514,
        currency="TWD",
    )
    assumptions = replace(
        _assumptions(lower_bound=0.04, upper_bound=0.06),
        solver_abs_tolerance=1_000_000.0,
    )

    result = _value(inputs, assumptions)[0]

    assert result.model_metrics["implied_growth_rate"] == pytest.approx(0.05, abs=1e-8)
    assert result.model_implied_value_per_share == pytest.approx(inputs.reference_price, abs=1e-5)
    assert result.equity_value == pytest.approx(result.enterprise_value + 1_000_000_000_000.0)
