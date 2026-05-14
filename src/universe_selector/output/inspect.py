from __future__ import annotations

from universe_selector.output.json import RANKING_CORE_KEYS, SNAPSHOT_CORE_KEYS, json_dumps, split_profile_metrics
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

    ranking_by_horizon = {str(row["horizon"]): row for row in rankings}
    lines.extend(["", "## Horizon Rankings", ""])
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


def render_inspect_json(
    *,
    run_id: str,
    resolution_mode: str,
    ticker: str,
    metadata: ProviderMetadata,
    snapshot: dict[str, object],
    rankings: list[dict[str, object]],
    profile: RankingProfile,
    ranking_profile: str,
    ranking_config_hash: str,
) -> str:
    horizon_order = {horizon: index for index, horizon in enumerate(profile.horizon_order)}
    ordered_rankings = sorted(
        rankings,
        key=lambda row: (
            horizon_order.get(str(row["horizon"]), len(horizon_order)),
            int(row["rank"]),
        ),
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "universe_selector_inspect",
        "run_id": run_id,
        "resolution_mode": resolution_mode,
        "normalized_ticker": ticker,
        "ranking_profile": ranking_profile,
        "ranking_config_hash": ranking_config_hash,
        "provider_metadata": metadata,
        "snapshot": split_profile_metrics(
            snapshot,
            core_keys=SNAPSHOT_CORE_KEYS,
            metric_keys=profile.snapshot_metric_keys,
        ),
        "rankings": [
            split_profile_metrics(
                row,
                core_keys=RANKING_CORE_KEYS,
                metric_keys=profile.ranking_metric_keys,
            )
            for row in ordered_rankings
        ],
        "notes": [
            "Ranking profiles compute finite scores; higher score ranks better.",
            "Scores are meaningful within the same run, market, profile, and horizon unless the profile documents otherwise.",
            profile.rank_interpretation_note,
            "Absent tickers do not expose exclusion reasons.",
        ],
    }
    return json_dumps(payload) + "\n"
