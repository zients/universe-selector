from __future__ import annotations

import polars as pl

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.output.json import RANKING_CORE_KEYS, SNAPSHOT_CORE_KEYS, json_dumps, split_profile_metrics
from universe_selector.ranking_profiles import RankingProfile


REPORT_RESEARCH_DISCLAIMER = "This report is for quantitative research only and is not investment advice."


def _format_provider_summary(provider_summary: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in provider_summary.items())


def _top_section(
    title: str,
    rankings: pl.DataFrame,
    horizon: str,
    top_n: int,
) -> str:
    if rankings.is_empty():
        return f"## {title}\n\nNo persisted candidates for this horizon.\n"
    rows = rankings.filter(pl.col("horizon") == horizon).sort("rank").head(top_n).to_dicts()
    if not rows:
        return f"## {title}\n\nNo persisted candidates for this horizon.\n"
    lines = [f"## {title}", "", "| rank | ticker | score |", "|---:|---|---:|"]
    for row in rows:
        lines.append(f"| {row['rank']} | {row['ticker']} | {row['score']:.4f} |")
    return "\n".join(lines) + "\n"


def _snapshot_lookup(snapshot: pl.DataFrame) -> dict[str, dict[str, object]]:
    if snapshot.is_empty() or "ticker" not in snapshot.columns:
        return {}
    return {str(row["ticker"]): row for row in snapshot.to_dicts()}


def _full_json_snapshots(snapshot: pl.DataFrame, profile: RankingProfile) -> list[dict[str, object]]:
    if snapshot.is_empty():
        return []
    rows = snapshot.sort("ticker").to_dicts() if "ticker" in snapshot.columns else snapshot.to_dicts()
    return [
        split_profile_metrics(
            row,
            core_keys=SNAPSHOT_CORE_KEYS,
            metric_keys=profile.snapshot_metric_keys,
        )
        for row in rows
    ]


def _full_json_rankings(rankings: pl.DataFrame, profile: RankingProfile) -> list[dict[str, object]]:
    if rankings.is_empty():
        return []
    horizon_order = {horizon: index for index, horizon in enumerate(profile.horizon_order)}
    rows = rankings.to_dicts()
    rows.sort(
        key=lambda row: (
            horizon_order.get(str(row["horizon"]), len(horizon_order)),
            int(row["rank"]),
            str(row["ticker"]),
        )
    )
    return [
        split_profile_metrics(
            row,
            core_keys=RANKING_CORE_KEYS,
            metric_keys=profile.ranking_metric_keys,
        )
        for row in rows
    ]


def _top_json_rows(
    *,
    rankings: pl.DataFrame,
    snapshots_by_ticker: dict[str, dict[str, object]],
    horizon: str,
    top_n: int,
    profile: RankingProfile,
) -> list[dict[str, object]]:
    if rankings.is_empty():
        return []
    rows = rankings.filter(pl.col("horizon") == horizon).sort("rank").head(top_n).to_dicts()
    payloads = []
    for row in rows:
        ticker = str(row["ticker"])
        snapshot = snapshots_by_ticker.get(ticker, {"ticker": ticker})
        payloads.append(
            {
                "ticker": ticker,
                "ranking": split_profile_metrics(
                    row,
                    core_keys=RANKING_CORE_KEYS,
                    metric_keys=profile.ranking_metric_keys,
                ),
                "snapshot": split_profile_metrics(
                    snapshot,
                    core_keys=SNAPSHOT_CORE_KEYS,
                    metric_keys=profile.snapshot_metric_keys,
                ),
            }
        )
    return payloads


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
    return f"""# Universe Selector Report

> {REPORT_RESEARCH_DISCLAIMER}

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


def render_json_report(
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
    snapshots_by_ticker = _snapshot_lookup(snapshot)
    payload = {
        "schema_version": 1,
        "artifact_type": "universe_selector_report",
        "run_id": run_id,
        "market": market,
        "mode_label": mode_label,
        "ranking_profile": provider_summary.get("ranking_profile", config.ranking_profile),
        "ranking_config_hash": provider_summary.get("ranking_config_hash", config.ranking_config_hash()),
        "provider_summary": provider_summary,
        "candidate_summary": {
            "snapshot_rows": snapshot.height,
            "ranking_rows": rankings.height,
            "top_n": config.report_top_n,
        },
        "snapshots": _full_json_snapshots(snapshot, profile),
        "rankings": _full_json_rankings(rankings, profile),
        "top_horizons": {
            horizon: _top_json_rows(
                rankings=rankings,
                snapshots_by_ticker=snapshots_by_ticker,
                horizon=horizon,
                top_n=config.report_top_n,
                profile=profile,
            )
            for horizon in profile.horizon_order
        },
        "notes": [
            REPORT_RESEARCH_DISCLAIMER,
            "This is a pure quantitative run report.",
            "Ranking profiles compute finite scores; higher score ranks better.",
            "Scores are meaningful within the same run, market, profile, and horizon unless the profile documents otherwise.",
            profile.rank_interpretation_note,
            "Filtered-out tickers and exclusion reasons are not persisted.",
            "This report is rendered during batch; report and inspect read persisted results only.",
        ],
    }
    return json_dumps(payload) + "\n"
