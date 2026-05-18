from __future__ import annotations

from typing import TYPE_CHECKING

from universe_selector.valuation.formatting import _format_money, _format_note, _format_number, _lines_table

if TYPE_CHECKING:
    from universe_selector.valuation.models import ValuationResult


def render_effective_inputs_section(result: ValuationResult) -> list[str]:
    effective = result.run_input.effective_inputs
    lines = [
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
    return lines


def render_input_provenance_section(result: ValuationResult) -> list[str]:
    provenance = result.run_input.input_provenance
    lines = [
        "",
        "## Input Provenance",
        "",
    ]
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
    return lines
