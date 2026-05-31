from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from datetime import datetime, timezone
from typing import cast

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts, FundamentalsMetadata
from universe_selector.valuation.output import render_valuation_json, render_valuation_markdown
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    ImpliedDiscountRateScenarioAssumptions,
    ImpliedDiscountRateV1Assumptions,
    StartingFcfAssumption,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationResult,
    ValuationRunInput,
)
from universe_selector.valuation.implied_discount_rate_v1 import (
    ImpliedDiscountRateV1Model,
    ImpliedDiscountRateV1OutputRenderer,
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


def _model_payload() -> dict[str, object]:
    return {
        "forecast_years": 5,
        "terminal_method": "perpetual_growth",
        "starting_fcf": {"method": "provider_ttm_fcf"},
        "growth_rate_basis": "constant_explicit_fcf_growth",
        "terminal_growth_basis": "nominal_perpetual_growth",
        "implied_discount_rate_basis": "nominal_wacc",
        "solver_abs_tolerance": 0.000001,
        "solver_max_iterations": 100,
        "scenarios": {
            "conservative": {
                "growth_rate": 0.03,
                "terminal_growth_rate": 0.02,
                "implied_discount_rate_lower_bound": 0.05,
                "implied_discount_rate_upper_bound": 0.25,
                "note": " Lower diagnostic case. ",
            },
            "base": {
                "growth_rate": 0.05,
                "terminal_growth_rate": 0.025,
                "implied_discount_rate_lower_bound": 0.055,
                "implied_discount_rate_upper_bound": 0.25,
                "note": " Middle diagnostic case. ",
            },
            "upside": {
                "growth_rate": 0.07,
                "terminal_growth_rate": 0.03,
                "implied_discount_rate_lower_bound": 0.06,
                "implied_discount_rate_upper_bound": 0.25,
                "note": " Higher diagnostic case. ",
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


def _assumption_set(model_assumptions: ImpliedDiscountRateV1Assumptions) -> ValuationAssumptionSet:
    return ValuationAssumptionSet(
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
        source_note="Unit test assumptions.",
        assumption_path="/tmp/valuation_assumptions/us/AAA.yaml",
        assumption_hash="abc123",
        facts_overrides={"shares_outstanding": None, "net_debt": None, "reference_price": 48.0},
        facts_override_notes={
            "shares_outstanding": None,
            "net_debt": None,
            "reference_price": "Reference price supplied for scenario review.",
        },
        model_id="implied_discount_rate_v1",
        model_assumptions=model_assumptions,
    )


def _valuation_result(
    *,
    inputs: EffectiveValuationInputs | None = None,
    assumptions: ImpliedDiscountRateV1Assumptions | None = None,
) -> ValuationResult:
    inputs = inputs or _inputs()
    assumptions = assumptions or _assumptions()
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
            model_id="implied_discount_rate_v1",
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
            assumptions=_assumption_set(assumptions),
        ),
        scenario_results=_value(inputs, assumptions),
    )


def _renderer_markdown(result: ValuationResult) -> str:
    renderer = ImpliedDiscountRateV1OutputRenderer()
    lines: list[str] = []
    lines.extend(renderer.render_risk_disclosures(result))
    lines.extend(renderer.render_model_assumptions(result))
    lines.extend(renderer.render_effective_inputs(result))
    lines.extend(renderer.render_input_provenance(result))
    lines.extend(renderer.render_scenario_results(result))
    return "\n".join(lines)


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


def test_implied_discount_rate_v1_validates_model_payload_trims_notes_and_freezes_scenarios() -> None:
    assumptions = ImpliedDiscountRateV1Model().validate_assumptions(_model_payload())

    assert assumptions.forecast_years == 5
    assert assumptions.terminal_method == "perpetual_growth"
    assert assumptions.starting_fcf.method == "provider_ttm_fcf"
    assert assumptions.growth_rate_basis == "constant_explicit_fcf_growth"
    assert assumptions.terminal_growth_basis == "nominal_perpetual_growth"
    assert assumptions.implied_discount_rate_basis == "nominal_wacc"
    assert assumptions.solver_abs_tolerance == pytest.approx(0.000001)
    assert assumptions.solver_max_iterations == 100
    assert assumptions.scenario_order == ("conservative", "base", "upside")
    assert tuple(assumptions.scenarios) == ("conservative", "base", "upside")
    assert assumptions.scenarios["base"].note == "Middle diagnostic case."

    with pytest.raises(TypeError):
        assumptions.scenarios["base"] = assumptions.scenarios["conservative"]  # type: ignore[index]


def test_implied_discount_rate_v1_rejects_unknown_model_and_scenario_keys() -> None:
    payload = _model_payload()
    payload["unknown"] = "nope"
    with pytest.raises(ValidationError, match="unknown implied_discount_rate_v1 key"):
        ImpliedDiscountRateV1Model().validate_assumptions(payload)

    payload = _model_payload()
    scenarios = dict(_scenarios(payload))
    base = dict(scenarios["base"])
    base["unknown"] = "nope"
    scenarios["base"] = base
    payload["scenarios"] = scenarios
    with pytest.raises(ValidationError, match="unknown implied_discount_rate_v1 scenario key"):
        ImpliedDiscountRateV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "scenarios",
    [
        {"conservative": {}, "base": {}},
        {"conservative": {}, "base": {}, "upside": {}, "aggressive": {}},
    ],
)
def test_implied_discount_rate_v1_rejects_missing_or_extra_scenario_ids(scenarios: dict[str, object]) -> None:
    payload = _model_payload()
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match="scenarios"):
        ImpliedDiscountRateV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"forecast_years": 0}, "forecast_years"),
        ({"forecast_years": 11}, "forecast_years"),
        ({"terminal_method": "exit_multiple"}, "terminal_method"),
        ({"growth_rate_basis": "stage_growth"}, "growth_rate_basis"),
        ({"terminal_growth_basis": "real_growth"}, "terminal_growth_basis"),
        ({"implied_discount_rate_basis": "cost_of_equity"}, "implied_discount_rate_basis"),
        ({"solver_abs_tolerance": 0.0}, "solver_abs_tolerance"),
        ({"solver_abs_tolerance": float("inf")}, "solver_abs_tolerance"),
        ({"solver_max_iterations": 0}, "solver_max_iterations"),
        ({"solver_max_iterations": 1001}, "solver_max_iterations"),
    ],
)
def test_implied_discount_rate_v1_rejects_invalid_model_payload(patch: dict[str, object], message: str) -> None:
    payload = _model_payload()
    payload.update(patch)

    with pytest.raises(ValidationError, match=message):
        ImpliedDiscountRateV1Model().validate_assumptions(payload)


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
def test_implied_discount_rate_v1_rejects_invalid_starting_fcf_schema(starting_fcf: dict[str, object]) -> None:
    payload = _model_payload()
    payload["starting_fcf"] = starting_fcf

    with pytest.raises(ValidationError, match="starting_fcf"):
        ImpliedDiscountRateV1Model().validate_assumptions(payload)


def test_implied_discount_rate_v1_preserves_override_note_without_coercion() -> None:
    payload = _model_payload()
    payload["starting_fcf"] = {
        "method": "override",
        "value": 123.0,
        "note": " Normalized unlevered FCFF estimate. ",
    }

    assumptions = ImpliedDiscountRateV1Model().validate_assumptions(payload)

    assert assumptions.starting_fcf.value == pytest.approx(123.0)
    assert assumptions.starting_fcf.note == " Normalized unlevered FCFF estimate. "


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("growth_rate", -1.0, "growth_rate"),
        ("growth_rate", 1.01, "growth_rate"),
        ("growth_rate", "0.05", "growth_rate"),
        ("growth_rate", float("inf"), "growth_rate"),
        ("terminal_growth_rate", -0.051, "terminal_growth_rate"),
        ("terminal_growth_rate", 0.051, "terminal_growth_rate"),
        ("terminal_growth_rate", "0.02", "terminal_growth_rate"),
        ("terminal_growth_rate", float("inf"), "terminal_growth_rate"),
        ("implied_discount_rate_lower_bound", 0.0, "implied_discount_rate_lower_bound"),
        ("implied_discount_rate_lower_bound", 0.51, "implied_discount_rate_lower_bound"),
        ("implied_discount_rate_lower_bound", "0.05", "implied_discount_rate_lower_bound"),
        ("implied_discount_rate_lower_bound", float("inf"), "implied_discount_rate_lower_bound"),
        ("implied_discount_rate_upper_bound", 0.0, "implied_discount_rate_upper_bound"),
        ("implied_discount_rate_upper_bound", 0.51, "implied_discount_rate_upper_bound"),
        ("implied_discount_rate_upper_bound", "0.25", "implied_discount_rate_upper_bound"),
        ("implied_discount_rate_upper_bound", float("inf"), "implied_discount_rate_upper_bound"),
    ],
)
def test_implied_discount_rate_v1_rejects_invalid_scenario_rate_bounds(
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
        ImpliedDiscountRateV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "patch, message",
    [
        (
            {"implied_discount_rate_lower_bound": 0.10, "implied_discount_rate_upper_bound": 0.10},
            "lower bound must be below upper bound",
        ),
        (
            {"implied_discount_rate_lower_bound": 0.05, "terminal_growth_rate": 0.025},
            "lower_bound - terminal_growth_rate",
        ),
        ({"note": "   "}, "note"),
        ({"note": 123}, "note"),
    ],
)
def test_implied_discount_rate_v1_rejects_invalid_scenario_shape(
    patch: dict[str, object],
    message: str,
) -> None:
    payload = _model_payload()
    scenarios = dict(_scenarios(payload))
    base = dict(scenarios["base"])
    base.update(patch)
    scenarios["base"] = base
    payload["scenarios"] = scenarios

    with pytest.raises(ValidationError, match=message):
        ImpliedDiscountRateV1Model().validate_assumptions(payload)


@pytest.mark.parametrize(
    "scenario_name, field, value, message",
    [
        ("conservative", "growth_rate", 0.06, "scenario growth_rate"),
        ("base", "terminal_growth_rate", 0.015, "scenario terminal_growth_rate"),
    ],
)
def test_implied_discount_rate_v1_rejects_non_monotonic_scenarios(
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
        ImpliedDiscountRateV1Model().validate_assumptions(payload)


def test_implied_discount_rate_v1_builds_effective_inputs_from_provider_ttm_fcf() -> None:
    assumptions = ImpliedDiscountRateV1Model().validate_assumptions(_model_payload())
    effective, provenance = ImpliedDiscountRateV1Model().build_inputs(
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


def test_implied_discount_rate_v1_builds_effective_inputs_from_override_starting_fcf() -> None:
    payload = _model_payload()
    payload["starting_fcf"] = {
        "method": "override",
        "value": 123.0,
        "note": "Normalized unlevered FCFF estimate.",
    }
    assumptions = ImpliedDiscountRateV1Model().validate_assumptions(payload)
    effective, provenance = ImpliedDiscountRateV1Model().build_inputs(
        facts=_facts(),
        assumptions=_assumption_set(assumptions),
    )

    assert effective.starting_fcf == pytest.approx(123.0)
    assert provenance.starting_fcf_source == "assumption_override"
    assert provenance.starting_fcf_note == "Normalized unlevered FCFF estimate."


def test_implied_discount_rate_v1_renderer_discloses_assumptions_bridge_and_results() -> None:
    markdown = _renderer_markdown(_valuation_result())

    assert "diagnostic reconciliation" in markdown
    assert "not a company WACC estimate" in markdown
    assert "not a hurdle rate" in markdown
    assert "not a required return" in markdown
    assert "not an expected return" in markdown
    assert "not a recommendation" in markdown
    assert "not an investment signal" in markdown
    assert "provider-FCF-proxy EV/equity reconciliation" in markdown
    assert "provider TTM FCF is a raw starting proxy" in markdown
    assert "may not be analyst-normalized" in markdown
    assert "not clean unlevered FCFF" in markdown
    assert "accounting classification" in markdown
    assert "cyclicality" in markdown
    assert "working capital" in markdown
    assert "capex" in markdown
    assert "capital-structure effects" in markdown
    assert "override for normalized unlevered FCFF" in markdown
    assert "terminal-value dominance" in markdown
    assert "terminal-value share of EV" in markdown
    assert "No result is produced when reference-implied enterprise value is outside solver bounds" in markdown
    assert (
        "starting FCF, share count, net debt, reference price, forecast growth, terminal growth, and solver bounds"
        in markdown
    )
    assert "forecast_years: 2" in markdown
    assert "starting_fcf_method: provider_ttm_fcf" in markdown
    assert "growth_rate_basis: constant_explicit_fcf_growth" in markdown
    assert "implied_discount_rate_basis: nominal_wacc" in markdown
    assert "| base | 5.00% | 2.00% | 5.00% | 25.00% | unit test |" in markdown
    assert "reference_equity_value" in markdown
    assert "reference_implied_enterprise_value" in markdown
    assert "$1248.30" in markdown
    assert "$1348.30" in markdown
    assert "implied discount rate" in markdown
    assert "absolute per-share reconciliation residual" in markdown
    assert "forecast growth" in markdown
    assert "terminal growth" in markdown
    assert "final_year_fcf" in markdown
    assert "undiscounted_terminal_value" in markdown
    assert "present_value_terminal_value" in markdown
    assert "terminal_value_share_of_enterprise_value" in markdown
    assert "model_implied_enterprise_value" in markdown
    assert "model-implied value per share" in markdown
    assert "reference price" in markdown
    assert "spread vs reference price" in markdown
    assert "10.00%" in markdown
    assert "$110.25" in markdown
    assert "$1405.69" in markdown
    assert "$1161.73" in markdown
    assert "86.16%" in markdown
    assert "$0.0000" in markdown
    assert (
        "rows are diagnostic assumption cases, not probabilities, forecasts, expected outcomes, target cases, recommendations, or investment signals"
        in markdown
    )
    assert (
        "model-implied value per share, spread, and implied discount rate are reconciliation math only, "
        "not target prices, forecasts, expected returns, required returns, recommendations, or signals"
    ) in markdown


def test_implied_discount_rate_v1_renderer_renders_override_starting_fcf_assumptions() -> None:
    assumptions = replace(
        _assumptions(),
        starting_fcf=StartingFcfAssumption(
            method="override",
            value=123.0,
            note="Normalized unlevered FCFF estimate.",
        ),
    )

    markdown = _renderer_markdown(_valuation_result(assumptions=assumptions))

    assert "starting_fcf_method: override" in markdown
    assert "starting_fcf_value: 123.00" in markdown
    assert "starting_fcf_note: Normalized unlevered FCFF estimate." in markdown


def test_implied_discount_rate_v1_renderer_redacts_and_escapes_scenario_notes() -> None:
    assumptions = replace(
        _assumptions(),
        scenarios={
            "base": ImpliedDiscountRateScenarioAssumptions(
                scenario_id="base",
                growth_rate=0.05,
                terminal_growth_rate=0.02,
                implied_discount_rate_lower_bound=0.05,
                implied_discount_rate_upper_bound=0.25,
                note="buy | target price\nsecond line",
            )
        },
    )

    markdown = _renderer_markdown(_valuation_result(assumptions=assumptions))

    assert "[redacted] \\| [redacted] second line" in markdown
    assert "buy \\|" not in markdown.lower()


def test_implied_discount_rate_v1_registered_output_paths_render_markdown_and_json() -> None:
    result = _valuation_result()

    markdown = render_valuation_markdown(result)

    assert "model_id: implied_discount_rate_v1" in markdown
    assert "diagnostic reconciliation" in markdown
    assert "not a company WACC estimate" in markdown
    assert "not a required return" in markdown
    assert "not an expected return" in markdown
    assert "not a recommendation" in markdown
    assert "not an investment signal" in markdown
    assert "provider TTM FCF is a raw starting proxy" in markdown
    assert "## Valuation Bridge" in markdown
    assert "reference_implied_enterprise_value" in markdown
    assert "## Scenario Results" in markdown
    assert "terminal_value_share_of_enterprise_value" in markdown
    assert "No result is produced when reference-implied enterprise value is outside solver bounds" in markdown
    assert (
        "model-implied value per share, spread, and implied discount rate are reconciliation math only, "
        "not target prices, forecasts, expected returns, required returns, recommendations, or signals"
    ) in markdown

    payload = json.loads(render_valuation_json(result))
    notes = " ".join(payload["notes"])

    assert payload["model_id"] == "implied_discount_rate_v1"
    assert payload["valuation_bridge"] == {
        "reference_equity_value": pytest.approx(1248.2954545454543),
        "net_debt": pytest.approx(100.0),
        "reference_implied_enterprise_value": pytest.approx(1348.2954545454543),
    }
    assert payload["valuation_bridge"]["reference_implied_enterprise_value"] == pytest.approx(
        payload["effective_inputs"]["reference_price"] * payload["effective_inputs"]["shares_outstanding"]
        + payload["effective_inputs"]["net_debt"]
    )
    assert payload["model_assumptions"]["implied_discount_rate_basis"] == "nominal_wacc"
    assert payload["model_assumptions"]["scenarios"]["base"]["implied_discount_rate_lower_bound"] == 0.05
    assert payload["scenario_results"][0]["model_metrics"]["implied_discount_rate"] == pytest.approx(0.10)
    assert "not a company WACC estimate" in notes
    assert "not a required return" in notes
    assert "not an expected return" in notes
    assert "not a recommendation" in notes
    assert "not an investment signal" in notes
