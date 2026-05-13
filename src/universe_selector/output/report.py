from __future__ import annotations

import re

import polars as pl

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.ranking_profiles import RankingProfile


FORBIDDEN_WORDS = ("buy", "sell", "hold", "recommendation", "target price", "expected return", "portfolio weight")
_ALLOWED_NO_ADVICE_PHRASES = ("holding-period recommendations",)
_FORBIDDEN_WORDING_PATTERN_SOURCES = {
    "buy": r"(?<!\w)buy(?!\w)",
    "sell": r"(?<!\w)sell(?!\w)",
    "hold": r"(?<!\w)hold(?!\w)",
    "recommendation": r"(?<!\w)recommendations?(?!\w)",
    "target price": r"(?<!\w)target prices?(?!\w)",
    "expected return": r"(?<!\w)expected returns?(?!\w)",
    "portfolio weight": r"(?<!\w)portfolio weights?(?!\w)",
}
_FORBIDDEN_WORDING_PATTERNS = tuple(
    (wording, re.compile(_FORBIDDEN_WORDING_PATTERN_SOURCES[wording])) for wording in FORBIDDEN_WORDS
)


def _find_forbidden_wording(content: str) -> str | None:
    lowered = content.lower()
    for allowed_phrase in _ALLOWED_NO_ADVICE_PHRASES:
        lowered = lowered.replace(allowed_phrase, " ")
    for wording, pattern in _FORBIDDEN_WORDING_PATTERNS:
        if pattern.search(lowered):
            return wording
    return None


def _format_provider_summary(provider_summary: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in provider_summary.items())


def _top_section(title: str, rankings: pl.DataFrame, horizon: str, top_n: int) -> str:
    if rankings.is_empty():
        return f"## {title}\n\nNo persisted candidates for this horizon.\n"
    rows = rankings.filter(pl.col("horizon") == horizon).sort("rank").head(top_n).to_dicts()
    if not rows:
        return f"## {title}\n\nNo persisted candidates for this horizon.\n"
    lines = [f"## {title}", "", "| rank | ticker | score |", "|---:|---|---:|"]
    for row in rows:
        lines.append(f"| {row['rank']} | {row['ticker']} | {row['score']:.4f} |")
    return "\n".join(lines) + "\n"


def render_markdown_report(
    *,
    run_id: str,
    market: Market,
    mode_label: str,
    provider_summary: dict[str, str],
    snapshot: pl.DataFrame,
    rankings: pl.DataFrame,
    config: AppConfig,
    profile: RankingProfile,
) -> str:
    survivor_count = snapshot.height
    empty_note = "\nSuccessful run with no persisted candidates." if survivor_count == 0 else ""
    ranking_sections = "\n\n".join(
        _top_section(
            f"Highest-ranked {horizon} candidates",
            rankings,
            horizon,
            config.report_top_n,
        )
        for horizon in profile.horizon_order
    )
    content = f"""# Universe Selector Report

## Run Context

- run_id: {run_id}
- market: {market.value}
- data mode: {mode_label}
- surviving candidate count: {survivor_count}{empty_note}

## Provider And Config Summary

{_format_provider_summary(provider_summary)}

## Candidate Summary

- persisted ticker snapshot rows: {snapshot.height}
- persisted ranking rows: {rankings.height}

{ranking_sections}

## Methodology Notes

- This is a pure quantitative run report.
- Ranking profiles compute finite scores; higher score ranks better.
- Scores are meaningful within the same run, market, profile, and horizon unless the profile documents otherwise.
- {profile.rank_interpretation_note}
- Filtered-out tickers and exclusion reasons are not persisted.
- This report is rendered during batch; report and inspect read persisted results only.
"""
    forbidden = _find_forbidden_wording(content)
    if forbidden is not None:
        raise ValueError(f"report contains forbidden wording: {forbidden}")
    return content
