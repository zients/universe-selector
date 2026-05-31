from __future__ import annotations

from typing import TYPE_CHECKING

from universe_selector.output.json import json_dumps
from universe_selector.valuation.formatting import (
    _format_money,
    _format_note,
    _format_number,
    _lines_table,
    _markdown_text,
)

if TYPE_CHECKING:
    from universe_selector.valuation.base import ValuationOutputRenderer
    from universe_selector.valuation.models import ValuationResult


VALUATION_RESEARCH_DISCLAIMER = "This valuation report is for quantitative research only and is not investment advice."


def _output_renderer(model_id: str) -> ValuationOutputRenderer:
    from universe_selector.valuation.registry import get_valuation_output_renderer

    return get_valuation_output_renderer(model_id)


def render_valuation_markdown(result: ValuationResult) -> str:
    run_input = result.run_input
    renderer = _output_renderer(run_input.model_id)
    metadata = run_input.fundamentals_metadata
    raw_facts = run_input.raw_facts
    effective = run_input.effective_inputs
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
    ]
    lines.extend(renderer.render_risk_disclosures(result))
    lines.extend(
        [
            f"- Source risk: {_markdown_text(metadata.source_risk_note)}",
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
        ]
    )
    if metadata.field_mapping_note:
        lines.append(f"- field mapping: {_markdown_text(metadata.field_mapping_note)}")
    lines.extend(
        [
            "",
            "## Assumption Context",
            "",
            f"- schema_version: {assumptions.schema_version}",
            f"- default_model: {_markdown_text(assumptions.default_model)}",
            f"- currency: {_markdown_text(assumptions.currency)}",
            f"- amount_unit: {_markdown_text(assumptions.amount_unit)}",
            f"- share_basis: {_markdown_text(assumptions.share_basis)}",
            f"- valuation_basis_note: {_markdown_text(assumptions.valuation_basis_note)}",
            f"- assumption_path: {_markdown_text(assumptions.assumption_path)}",
            f"- assumption_hash: {_markdown_text(assumptions.assumption_hash)}",
            f"- assumption_source: {_markdown_text(assumptions.assumption_source)}",
            f"- prepared_by: {_markdown_text(assumptions.prepared_by)}",
            f"- source_note: {_markdown_text(assumptions.source_note)}",
            f"- as_of: {assumptions.as_of.isoformat()}",
            "",
        ]
    )

    lines.extend(renderer.render_model_assumptions(result))
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
    lines.extend(renderer.render_effective_inputs(result))
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

    lines.extend(renderer.render_input_provenance(result))
    lines.extend(renderer.render_scenario_results(result))
    return "\n".join(lines) + "\n"


def _json_note_from_markdown_bullet(value: str) -> str:
    if value.startswith("- "):
        return value[2:]
    return value


def render_valuation_json(result: ValuationResult) -> str:
    run_input = result.run_input
    renderer = _output_renderer(run_input.model_id)
    metadata = run_input.fundamentals_metadata
    raw_facts = run_input.raw_facts
    effective = run_input.effective_inputs
    assumptions = run_input.assumptions
    payload = {
        "schema_version": 1,
        "artifact_type": "universe_selector_valuation",
        "ephemeral": True,
        "market": run_input.market,
        "ticker": run_input.ticker,
        "model_id": run_input.model_id,
        "currency": effective.currency,
        "provider_context": {
            "data_mode": metadata.data_mode,
            "fundamentals_provider_id": metadata.fundamentals_provider_id,
            "fundamentals_source_ids": metadata.fundamentals_source_ids,
            "data_fetch_started_at": metadata.data_fetch_started_at,
            "latest_source_date": metadata.latest_source_date,
            "source_risk_note": metadata.source_risk_note,
            "field_mapping_note": metadata.field_mapping_note,
        },
        "assumption_context": {
            "schema_version": assumptions.schema_version,
            "default_model": assumptions.default_model,
            "assumption_path": assumptions.assumption_path,
            "assumption_hash": assumptions.assumption_hash,
            "as_of": assumptions.as_of,
            "currency": assumptions.currency,
            "amount_unit": assumptions.amount_unit,
            "share_basis": assumptions.share_basis,
            "valuation_basis_note": assumptions.valuation_basis_note,
            "assumption_source": assumptions.assumption_source,
            "prepared_by": assumptions.prepared_by,
            "source_note": assumptions.source_note,
        },
        "raw_facts": raw_facts,
        "effective_inputs": effective,
        "input_provenance": run_input.input_provenance,
        "model_assumptions": assumptions.model_assumptions,
        "scenario_results": result.scenario_results,
        "notes": [
            VALUATION_RESEARCH_DISCLAIMER,
            "This valuation output is ephemeral and is not persisted.",
            *(_json_note_from_markdown_bullet(item) for item in renderer.render_risk_disclosures(result)),
        ],
    }
    return json_dumps(payload) + "\n"


render_valuation = render_valuation_markdown
