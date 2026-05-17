from __future__ import annotations

from collections.abc import Iterable
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from universe_selector.valuation.models import ValuationResult


VALUATION_RESEARCH_DISCLAIMER = (
    "This valuation report is for quantitative research only and is not investment advice."
)
_PROHIBITED_PHRASE_RE = re.compile(
    r"target price|fair value|undervalued|overvalued|expected return",
    re.IGNORECASE,
)
_PROHIBITED_WORD_RE = re.compile(r"\b(?:buy|sell|hold)\b", re.IGNORECASE)


def _sanitize_text(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    text = _PROHIBITED_PHRASE_RE.sub("[redacted]", text)
    return _PROHIBITED_WORD_RE.sub("[redacted]", text)


def _table_cell(value: object) -> str:
    return _sanitize_text(value).replace("|", r"\|")


def _markdown_text(value: object) -> str:
    return _sanitize_text(value).replace("|", r"\|")


def _compact_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:.2f}"


def _format_money(value: float, currency: str) -> str:
    compact = _compact_number(value)
    currency_code = _sanitize_text(currency.upper())
    if currency_code == "USD":
        return f"${compact}"
    if currency_code:
        return f"{currency_code} {compact}"
    return compact


def _format_number(value: float) -> str:
    return _compact_number(value)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_note(note: str | None) -> str:
    if note is None:
        return ""
    return _sanitize_text(note)


def _lines_table(headers: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_table_cell(cell) for cell in row) + " |" for row in rows)
    return lines


def _render_fcf_dcf_assumptions(assumptions: Any) -> list[str]:
    lines = [
        "## Model Assumptions",
        "",
        f"- forecast_years: {assumptions.forecast_years}",
        f"- terminal_method: {_markdown_text(assumptions.terminal_method)}",
        f"- starting_fcf_method: {_markdown_text(assumptions.starting_fcf.method)}",
        f"- discount_rate_basis: {_markdown_text(assumptions.discount_rate_basis)}",
        f"- terminal_growth_basis: {_markdown_text(assumptions.terminal_growth_basis)}",
        "",
    ]
    if assumptions.starting_fcf.method == "override":
        lines.extend(
            [
                f"- starting_fcf_value: {_format_number(assumptions.starting_fcf.value)}",
                f"- starting_fcf_note: {_markdown_text(assumptions.starting_fcf.note)}",
                "",
            ]
        )
    rows = []
    for scenario_id in assumptions.scenario_order:
        scenario = assumptions.scenarios[scenario_id]
        rows.append(
            (
                scenario.scenario_id,
                _format_pct(scenario.growth_rate),
                _format_pct(scenario.discount_rate),
                _format_pct(scenario.terminal_growth_rate),
                _format_note(scenario.note),
            )
        )
    lines.extend(
        _lines_table(
            ("scenario", "growth_rate", "discount_rate", "terminal_growth_rate", "note"),
            rows,
        )
    )
    return lines


def _render_model_assumptions(result: ValuationResult) -> list[str]:
    assumptions = result.run_input.assumptions.model_assumptions
    if assumptions.__class__.__name__ == "FcfDcfV1Assumptions":
        return _render_fcf_dcf_assumptions(assumptions)
    return ["## Model Assumptions", "", f"- model_id: {_markdown_text(result.run_input.model_id)}"]


def render_valuation_markdown(result: ValuationResult) -> str:
    run_input = result.run_input
    metadata = run_input.fundamentals_metadata
    raw_facts = run_input.raw_facts
    effective = run_input.effective_inputs
    provenance = run_input.input_provenance
    assumptions = run_input.assumptions

    lines = [
        "# Universe Selector Valuation",
        "",
        f"> {VALUATION_RESEARCH_DISCLAIMER}",
        "",
        "This valuation output is ephemeral and is not persisted.",
        "",
        "## Run Context",
        "",
        f"- market: {_markdown_text(run_input.market.value)}",
        f"- ticker: {_markdown_text(run_input.ticker)}",
        f"- model_id: {_markdown_text(run_input.model_id)}",
        f"- currency: {_markdown_text(effective.currency)}",
        "",
        "## Risk Disclosures",
        "",
        (
            "- Model risk: fcf_dcf_v1 is a simplified constant-growth explicit forecast "
            "with perpetual-growth terminal value."
        ),
        (
            "- Sensitivity risk: outputs are highly sensitive to starting FCF, share count, "
            "discount rate, terminal growth, and terminal value assumptions."
        ),
        (
            "- Source risk: yfinance third-party convenience data may be stale, incomplete, "
            "restated, mapped inconsistently, or unavailable. Facts should be independently "
            "verified before research use."
        ),
        "",
        "## Provider Context",
        "",
        f"- data_mode: {_markdown_text(metadata.data_mode)}",
        f"- fundamentals_provider_id: {_markdown_text(metadata.fundamentals_provider_id)}",
        f"- fundamentals_source_ids: {_markdown_text(', '.join(metadata.fundamentals_source_ids))}",
        f"- data_fetch_started_at: {metadata.data_fetch_started_at.isoformat()}",
        f"- latest_source_date: {metadata.latest_source_date.isoformat()}",
        (
            "- latest_source_date note: latest source date is the max of quote, "
            "cash-flow period, and balance-sheet dates; individual fact dates are shown below."
        ),
        (
            "- yfinance field mapping: reference price from currentPrice/regularMarketPrice; "
            "shares from sharesOutstanding; raw free cash flow from Operating Cash Flow minus "
            "Capital Expenditure; cash and debt from quarterly balance sheet fields."
        ),
        "",
        "## Assumption Context",
        "",
        f"- schema_version: {assumptions.schema_version}",
        f"- currency: {_markdown_text(assumptions.currency)}",
        f"- amount_unit: {_markdown_text(assumptions.amount_unit)}",
        f"- assumption_path: {_markdown_text(assumptions.assumption_path)}",
        f"- assumption_hash: {_markdown_text(assumptions.assumption_hash)}",
        f"- assumption_source: {_markdown_text(assumptions.assumption_source)}",
        f"- prepared_by: {_markdown_text(assumptions.prepared_by)}",
        f"- source_note: {_markdown_text(assumptions.source_note)}",
        f"- as_of: {assumptions.as_of.isoformat()}",
        "",
    ]

    lines.extend(_render_model_assumptions(result))
    lines.extend(
        [
            "",
            "## Raw Provider Facts",
            "",
        ]
    )
    lines.extend(
        _lines_table(
            ("field", "value"),
            (
                ("reference_price", _format_money(raw_facts.reference_price, raw_facts.currency)),
                ("reference_price_as_of", raw_facts.reference_price_as_of.isoformat()),
                ("reference_price_as_of_source", raw_facts.reference_price_as_of_source),
                (
                    "reference_price_as_of_note",
                    _format_note(raw_facts.reference_price_as_of_note),
                ),
                ("shares_outstanding", _format_number(raw_facts.shares_outstanding)),
                (
                    "cash_and_cash_equivalents",
                    _format_money(raw_facts.cash_and_cash_equivalents, raw_facts.currency),
                ),
                ("total_debt", _format_money(raw_facts.total_debt, raw_facts.currency)),
                ("balance_sheet_as_of", raw_facts.balance_sheet_as_of.isoformat()),
                ("net_debt", _format_money(raw_facts.net_debt, raw_facts.currency)),
                (
                    "operating_cash_flow",
                    _format_money(raw_facts.operating_cash_flow, raw_facts.currency),
                ),
                (
                    "capital_expenditures",
                    _format_money(raw_facts.capital_expenditures, raw_facts.currency),
                ),
                ("free_cash_flow", _format_money(raw_facts.free_cash_flow, raw_facts.currency)),
                ("fiscal_period_end", raw_facts.fiscal_period_end.isoformat()),
                ("fiscal_period_type", raw_facts.fiscal_period_type),
            ),
        )
    )
    lines.extend(
        [
            "",
            "## Effective Inputs",
            "",
            (
                "Starting FCF is used as an enterprise cash-flow proxy and is not verified "
                "unlevered FCFF. provider_ttm_fcf uses raw provider FCF as the starting FCF proxy. "
                "Use starting_fcf.method override when analyst-normalized FCF is needed."
            ),
            "",
        ]
    )
    lines.extend(
        _lines_table(
            ("field", "value"),
            (
                ("starting_fcf", _format_money(effective.starting_fcf, effective.currency)),
                ("shares_outstanding", _format_number(effective.shares_outstanding)),
                ("net_debt", _format_money(effective.net_debt, effective.currency)),
                ("reference_price", _format_money(effective.reference_price, effective.currency)),
                ("reference_price_as_of", effective.reference_price_as_of.isoformat()),
                ("reference_price_as_of_source", effective.reference_price_as_of_source),
                (
                    "reference_price_as_of_note",
                    _format_note(effective.reference_price_as_of_note),
                ),
                ("fiscal_period_end", effective.fiscal_period_end.isoformat()),
                ("fiscal_period_type", effective.fiscal_period_type),
            ),
        )
    )
    if raw_facts.reference_price_as_of_source == "fetch_date_fallback":
        lines.extend(
            [
                "",
                (
                    "Reference price date note: provider quote timestamp was unavailable; "
                    "the provider fetch date was used for raw facts."
                ),
            ]
        )
        if raw_facts.reference_price_as_of_note:
            lines.append(_markdown_text(raw_facts.reference_price_as_of_note))

    lines.extend(
        [
            "",
            "## Input Provenance",
            "",
        ]
    )
    lines.extend(
        _lines_table(
            ("field", "source", "note"),
            (
                (
                    "starting_fcf",
                    provenance.starting_fcf_source,
                    _format_note(provenance.starting_fcf_note),
                ),
                (
                    "shares_outstanding",
                    provenance.shares_outstanding_source,
                    _format_note(provenance.shares_outstanding_note),
                ),
                ("net_debt", provenance.net_debt_source, _format_note(provenance.net_debt_note)),
                (
                    "reference_price",
                    provenance.reference_price_source,
                    _format_note(provenance.reference_price_note),
                ),
            ),
        )
    )
    lines.extend(
        [
            "",
            "## Scenario Results",
            "",
            (
                "Scenario rows are illustrative scenarios, not probabilities, forecasts, "
                "expected outcomes, target cases, or recommendations."
            ),
            "",
        ]
    )
    lines.extend(
        _lines_table(
            (
                "scenario",
                "model-implied value per share",
                "reference price",
                "model-implied spread vs reference price",
            ),
            (
                (
                    scenario.scenario_id,
                    _format_money(scenario.model_implied_value_per_share, effective.currency),
                    _format_money(scenario.reference_price, effective.currency),
                    _format_pct(scenario.model_implied_spread_to_reference_price),
                )
                for scenario in result.scenario_results
            ),
        )
    )
    return "\n".join(lines) + "\n"


render_valuation = render_valuation_markdown
