from __future__ import annotations

__all__ = [
    "render_inspect",
    "render_markdown_report",
    "render_valuation",
    "render_valuation_markdown",
]


def __getattr__(name: str):
    if name == "render_inspect":
        from universe_selector.output.inspect import render_inspect

        return render_inspect
    if name == "render_markdown_report":
        from universe_selector.output.report import render_markdown_report

        return render_markdown_report
    if name in {"render_valuation", "render_valuation_markdown"}:
        from universe_selector.output.valuation import render_valuation_markdown

        return render_valuation_markdown
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
