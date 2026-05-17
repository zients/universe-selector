from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from universe_selector.domain import Market
from universe_selector.output.valuation import VALUATION_RESEARCH_DISCLAIMER, render_valuation_markdown
from universe_selector.providers.models import FundamentalFacts, FundamentalsMetadata
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    FcfDcfV1Assumptions,
    StartingFcfAssumption,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationResult,
    ValuationRunInput,
    ValuationScenarioAssumptions,
    ValuationScenarioResult,
)


def _result() -> ValuationResult:
    assumptions = ValuationAssumptionSet(
        schema_version=1,
        market=Market.US,
        ticker="AAPL",
        purpose="research",
        as_of=date(2026, 5, 16),
        currency="USD",
        amount_unit="currency_units",
        assumption_source="analyst",
        prepared_by="Universe Selector",
        source_note="Demonstration assumptions for schema validation only; not investment advice.",
        assumption_path="/repo/valuation_assumptions/us/AAPL.yaml",
        assumption_hash="abc123",
        facts_overrides={
            "shares_outstanding": None,
            "net_debt": None,
            "reference_price": 185.0,
        },
        facts_override_notes={
            "shares_outstanding": None,
            "net_debt": None,
            "reference_price": "Reference price supplied for scenario review.",
        },
        model_id="fcf_dcf_v1",
        model_assumptions=FcfDcfV1Assumptions(
            forecast_years=5,
            terminal_method="perpetual_growth",
            starting_fcf=StartingFcfAssumption(method="provider_ttm_fcf", value=None, note=None),
            discount_rate_basis="nominal_wacc",
            terminal_growth_basis="nominal_perpetual_growth",
            scenario_order=("conservative", "base", "upside"),
            scenarios={
                "conservative": ValuationScenarioAssumptions(
                    scenario_id="conservative",
                    growth_rate=0.03,
                    discount_rate=0.10,
                    terminal_growth_rate=0.02,
                    note="Lower illustrative scenario.",
                ),
                "base": ValuationScenarioAssumptions(
                    scenario_id="base",
                    growth_rate=0.05,
                    discount_rate=0.09,
                    terminal_growth_rate=0.025,
                    note="Middle illustrative scenario.",
                ),
                "upside": ValuationScenarioAssumptions(
                    scenario_id="upside",
                    growth_rate=0.07,
                    discount_rate=0.085,
                    terminal_growth_rate=0.03,
                    note="Higher illustrative scenario.",
                ),
            },
        ),
    )
    raw_facts = FundamentalFacts(
        market=Market.US,
        ticker="AAPL",
        currency="USD",
        reference_price=190.0,
        reference_price_as_of=date(2026, 5, 17),
        reference_price_as_of_source="fetch_date_fallback",
        reference_price_as_of_note="Provider quote timestamp unavailable; using fetch date.",
        shares_outstanding=15_000_000_000.0,
        cash_and_cash_equivalents=60_000_000_000.0,
        total_debt=110_000_000_000.0,
        balance_sheet_as_of=date(2026, 3, 31),
        net_debt=50_000_000_000.0,
        operating_cash_flow=120_000_000_000.0,
        capital_expenditures=10_000_000_000.0,
        free_cash_flow=110_000_000_000.0,
        fiscal_period_end=date(2025, 9, 30),
        fiscal_period_type="ttm",
    )
    effective_inputs = EffectiveValuationInputs(
        starting_fcf=110_000_000_000.0,
        shares_outstanding=15_000_000_000.0,
        net_debt=50_000_000_000.0,
        reference_price=185.0,
        currency="USD",
        fiscal_period_type="ttm",
        fiscal_period_end=date(2025, 9, 30),
        reference_price_as_of=date(2026, 5, 16),
        reference_price_as_of_source="assumption_override",
        reference_price_as_of_note="Reference price supplied for scenario review.",
    )
    return ValuationResult(
        run_input=ValuationRunInput(
            market=Market.US,
            ticker="AAPL",
            model_id="fcf_dcf_v1",
            fundamentals_metadata=FundamentalsMetadata(
                data_mode="live",
                fundamentals_provider_id="yfinance_fundamentals",
                fundamentals_source_ids=("yfinance", "quote", "quarterly_cash_flow", "balance_sheet"),
                data_fetch_started_at=datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
                latest_source_date=date(2026, 5, 17),
                source_risk_note=(
                    "yfinance third-party convenience data may be stale, incomplete, "
                    "restated, mapped inconsistently, or unavailable. Facts should be "
                    "independently verified before research use."
                ),
                field_mapping_note=(
                    "reference price from currentPrice/regularMarketPrice; shares from "
                    "sharesOutstanding; raw free cash flow from Operating Cash Flow minus "
                    "Capital Expenditure; cash and debt from quarterly balance sheet fields."
                ),
            ),
            raw_facts=raw_facts,
            effective_inputs=effective_inputs,
            input_provenance=ValuationInputProvenance(
                starting_fcf_source="provider_ttm_fcf",
                shares_outstanding_source="provider_fact",
                net_debt_source="provider_fact",
                reference_price_source="assumption_override",
                starting_fcf_note="Provider raw FCF used as starting FCF proxy; fiscal_period_type=ttm.",
                shares_outstanding_note=None,
                net_debt_note=None,
                reference_price_note="Reference price supplied for scenario review.",
            ),
            assumptions=assumptions,
        ),
        scenario_results=(
            ValuationScenarioResult(
                scenario_id="base",
                projected_fcf=(105_000_000_000.0, 110_250_000_000.0),
                present_value_projected_fcf=(96_330_275_229.36, 92_789_972_459.21),
                terminal_value=1_774_583_333_333.33,
                present_value_terminal_value=1_153_579_415_966.33,
                enterprise_value=1_342_699_663_654.90,
                equity_value=1_292_699_663_654.90,
                model_implied_value_per_share=86.18,
                reference_price=185.0,
                model_implied_spread_to_reference_price=-0.5342,
            ),
        ),
    )


def test_render_valuation_includes_context_disclosures_and_inputs() -> None:
    markdown = render_valuation_markdown(_result())

    assert markdown.startswith("# Universe Selector Valuation\n")
    assert VALUATION_RESEARCH_DISCLAIMER in markdown
    assert "This valuation output is ephemeral and is not persisted." in markdown
    assert "simplified constant-growth explicit forecast with perpetual-growth terminal value" in markdown
    assert (
        "highly sensitive to starting FCF, share count, discount rate, "
        "terminal growth, and terminal value assumptions"
    ) in markdown
    assert "third-party convenience data may be stale, incomplete, restated, mapped inconsistently, or unavailable" in markdown
    assert "Facts should be independently verified before research use." in markdown
    assert "## Provider Context" in markdown
    assert "fundamentals_provider_id: yfinance_fundamentals" in markdown
    assert "fundamentals_source_ids: yfinance, quote, quarterly_cash_flow, balance_sheet" in markdown
    assert "data_fetch_started_at: 2026-05-17T12:00:00+00:00" in markdown
    assert "latest_source_date: 2026-05-17" in markdown
    assert "facts_as_of" not in markdown
    assert "latest source date is the max of quote, cash-flow period, and balance-sheet dates" in markdown
    assert "## Assumption Context" in markdown
    assert "assumption_path: /repo/valuation_assumptions/us/AAPL.yaml" in markdown
    assert "assumption_hash: abc123" in markdown
    assert "assumption_source: analyst" in markdown
    assert "prepared_by: Universe Selector" in markdown
    assert "as_of: 2026-05-16" in markdown
    assert "currency: USD" in markdown
    assert "amount_unit: currency_units" in markdown
    assert "Demonstration assumptions for schema validation only; not investment advice." in markdown
    assert "forecast_years: 5" in markdown
    assert "terminal_method: perpetual_growth" in markdown
    assert "starting_fcf_method: provider_ttm_fcf" in markdown
    assert "discount_rate_basis: nominal_wacc" in markdown
    assert "terminal_growth_basis: nominal_perpetual_growth" in markdown
    assert "| conservative | 3.00% | 10.00% | 2.00% | Lower illustrative scenario. |" in markdown
    assert "## Raw Provider Facts" in markdown
    assert "free_cash_flow" in markdown
    assert "$110.00B" in markdown
    assert "balance_sheet_as_of" in markdown
    assert "2026-03-31" in markdown
    assert "## Effective Inputs" in markdown
    assert "starting FCF proxy" in markdown
    assert "not verified unlevered FCFF" in markdown
    assert "provider_ttm_fcf uses raw provider FCF as the starting FCF proxy." in markdown
    assert "Use starting_fcf.method override when analyst-normalized FCF is needed." in markdown
    assert "fetch_date_fallback" in markdown
    assert "Provider quote timestamp unavailable; using fetch date." in markdown


def test_render_valuation_includes_provenance_and_model_implied_scenarios_without_recommendations() -> None:
    markdown = render_valuation_markdown(_result())

    assert "## Input Provenance" in markdown
    assert "| starting_fcf | provider_ttm_fcf | Provider raw FCF used as starting FCF proxy; fiscal_period_type=ttm. |" in markdown
    assert "| shares_outstanding | provider_fact |  |" in markdown
    assert "| reference_price | assumption_override | Reference price supplied for scenario review. |" in markdown
    assert "model-implied spread vs reference price" in markdown
    assert "not probabilities, forecasts, expected outcomes, target cases, or recommendations" in markdown
    assert "| base | $86.18 | $185.00 | -53.42% |" in markdown

    for word in ("buy", "sell", "hold"):
        assert re.search(rf"\b{word}\b", markdown, flags=re.IGNORECASE) is None
    for phrase in ("target price", "fair value", "undervalued", "overvalued", "expected return"):
        assert phrase not in markdown.lower()


def test_render_valuation_redacts_prohibited_free_text_and_escapes_markdown_tables() -> None:
    result = _result()
    model_assumptions = result.run_input.assumptions.model_assumptions
    assert isinstance(model_assumptions, FcfDcfV1Assumptions)
    scenarios = dict(model_assumptions.scenarios)
    scenarios["base"] = replace(scenarios["base"], note="buy | target price\nsecond line")
    assumptions = replace(
        result.run_input.assumptions,
        source_note="sell | fair value\nsource line",
        model_assumptions=replace(model_assumptions, scenarios=scenarios),
    )
    raw_facts = replace(
        result.run_input.raw_facts,
        reference_price_as_of_note="hold | expected return\nprovider line",
    )
    provenance = replace(
        result.run_input.input_provenance,
        reference_price_note="undervalued | overvalued\nprovenance line",
    )
    result = replace(
        result,
        run_input=replace(
            result.run_input,
            ticker="sell",
            model_id="hold",
            assumptions=assumptions,
            raw_facts=raw_facts,
            input_provenance=provenance,
        ),
    )
    result = replace(
        result,
        run_input=replace(
            result.run_input,
            assumptions=replace(
                result.run_input.assumptions,
                assumption_path="/tmp/target price.yaml",
            ),
        ),
    )

    markdown = render_valuation_markdown(result)

    for word in ("buy", "sell", "hold"):
        assert re.search(rf"\b{word}\b", markdown, flags=re.IGNORECASE) is None
    for phrase in ("target price", "fair value", "undervalued", "overvalued", "expected return"):
        assert phrase not in markdown.lower()
    assert "\\|" in markdown
    assert "[redacted] \\| [redacted] second line" in markdown
    assert "[redacted] \\| [redacted] source line" in markdown
    assert "[redacted] \\| [redacted] provider line" in markdown
    assert "[redacted] \\| [redacted] provenance line" in markdown


def test_render_valuation_uses_currency_code_for_non_usd_amounts() -> None:
    result = _result()
    result = replace(
        result,
        run_input=replace(
            result.run_input,
            raw_facts=replace(result.run_input.raw_facts, currency="TWD"),
            effective_inputs=replace(result.run_input.effective_inputs, currency="TWD"),
        ),
    )

    markdown = render_valuation_markdown(result)

    assert "TWD 110.00B" in markdown
    assert "TWD 86.18" in markdown
    assert "$110.00B" not in markdown


def test_render_valuation_uses_provider_metadata_notes_without_yfinance_fallback() -> None:
    result = _result()
    result = replace(
        result,
        run_input=replace(
            result.run_input,
            fundamentals_metadata=replace(
                result.run_input.fundamentals_metadata,
                fundamentals_provider_id="custom_fundamentals",
                fundamentals_source_ids=("custom-facts",),
                source_risk_note="Custom provider facts can be delayed.",
                field_mapping_note="Custom field mapping is documented in the provider adapter.",
            ),
        ),
    )

    markdown = render_valuation_markdown(result)

    assert "fundamentals_provider_id: custom_fundamentals" in markdown
    assert "Source risk: Custom provider facts can be delayed." in markdown
    assert "field mapping: Custom field mapping is documented in the provider adapter." in markdown
    assert "yfinance" not in markdown


def test_valuation_output_submodule_import_does_not_import_report_config_or_provider_registry() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
import universe_selector.output.valuation
for module_name in (
    "universe_selector.output.report",
    "universe_selector.config",
    "universe_selector.providers.registry",
):
    assert module_name not in sys.modules, module_name
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_valuation_public_export_import_does_not_import_provider_config_or_report_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
from universe_selector.output import render_valuation_markdown
assert callable(render_valuation_markdown)
for module_name in (
    "universe_selector.output.report",
    "universe_selector.config",
    "universe_selector.providers",
    "universe_selector.providers.base",
    "universe_selector.providers.fixture",
    "universe_selector.providers.models",
    "universe_selector.providers.registration",
    "universe_selector.providers.registry",
):
    assert module_name not in sys.modules, module_name
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
