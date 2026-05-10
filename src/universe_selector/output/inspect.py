from __future__ import annotations

from universe_selector.providers.models import ProviderMetadata
from universe_selector.ranking_profiles import RankingProfile


def render_inspect(
    *,
    run_id: str,
    resolution_mode: str,
    ticker: str,
    metadata: ProviderMetadata,
    snapshot: dict[str, object],
    rankings: list[dict[str, object]],
    profile: RankingProfile,
) -> str:
    lines = [
        "# Universe Selector Inspect",
        "",
        f"- run_id: {run_id}",
        f"- resolution mode: {resolution_mode}",
        f"- normalized ticker: {ticker}",
        "",
        "## Provider Metadata",
        "",
        f"- data_mode: {metadata.data_mode}",
        f"- listing_provider_id: {metadata.listing_provider_id}",
        f"- listing_source_id: {metadata.listing_source_id}",
        f"- ohlcv_provider_id: {metadata.ohlcv_provider_id}",
        f"- ohlcv_source_id: {metadata.ohlcv_source_id}",
        f"- provider_config_hash: {metadata.provider_config_hash}",
        f"- data_fetch_started_at: {metadata.data_fetch_started_at.isoformat()}",
        f"- market_timezone: {metadata.market_timezone}",
        f"- run_latest_bar_date: {metadata.run_latest_bar_date.isoformat()}",
        "",
        "## Raw Ranking Metrics",
        "",
    ]
    for key in profile.inspect_metric_keys:
        lines.append(f"- {key}: {snapshot[key]}")
    lines.extend(["", "## Horizon Rankings", ""])
    ranking_by_horizon = {str(row["horizon"]): row for row in rankings}
    for horizon in profile.horizon_order:
        row = ranking_by_horizon[horizon]
        metric_parts = [f"{key} {row[key]}" for key in profile.ranking_metric_keys]
        metric_text = ", ".join(metric_parts)
        lines.append(
            f"- {row['horizon']}: "
            f"rank {row['rank']}, "
            f"score {row['score']}, "
            f"{metric_text}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Ranking profiles compute finite scores; higher score ranks better.",
            "- Scores are meaningful within the same run, market, profile, and horizon unless the profile documents otherwise.",
            f"- {profile.rank_interpretation_note}",
            "- Absent tickers do not expose exclusion reasons.",
        ]
    )
    return "\n".join(lines) + "\n"
