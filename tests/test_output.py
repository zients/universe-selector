from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.output.inspect import render_inspect
from universe_selector.output.report import FORBIDDEN_WORDS, render_markdown_report
from universe_selector.providers.models import ProviderMetadata
from universe_selector.ranking_profiles.sample_price_trend_v1 import SamplePriceTrendV1Profile


def _provider_metadata() -> ProviderMetadata:
    return ProviderMetadata(
        run_id="us-run",
        data_mode="fixture",
        listing_provider_id="fixture-listings-v1",
        listing_source_id="sample_basic/listings.csv",
        ohlcv_provider_id="fixture-ohlcv-v1",
        ohlcv_source_id="sample_basic/ohlcv.csv",
        provider_config_hash="fixture-sample-basic",
        data_fetch_started_at=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc),
        market_timezone="UTC",
        run_latest_bar_date=date(2026, 4, 24),
    )


def test_markdown_report_renders_sample_profile_sections_and_notes() -> None:
    profile = SamplePriceTrendV1Profile()
    content = render_markdown_report(
        run_id="us-sample",
        market=Market.US,
        mode_label="fixture",
        provider_summary={
            "data_mode": "fixture",
            "ranking_profile": "sample_price_trend_v1",
            "ranking_config_hash": "sample-hash",
        },
        snapshot=pl.DataFrame({"ticker": ["AAA", "BBB"]}),
        rankings=pl.DataFrame(
            {
                "horizon": ["midterm", "midterm", "longterm", "longterm"],
                "rank": [1, 2, 1, 2],
                "ticker": ["AAA", "BBB", "BBB", "AAA"],
                "final_rank_percentile": [90.0, 70.0, 85.0, 75.0],
            }
        ),
        config=AppConfig(data_mode="fixture", report_top_n=2),
        profile=profile,
    )

    assert content.startswith("# Universe Selector Report\n")
    assert "## Highest-ranked midterm candidates" in content
    assert "## Highest-ranked longterm candidates" in content
    assert "sample_price_trend_v1" in content
    assert profile.rank_interpretation_note in content
    assert "Filtered-out tickers and exclusion reasons are not persisted." in content
    for forbidden in FORBIDDEN_WORDS:
        assert forbidden not in content.lower()


def test_empty_report_is_structured_and_not_advice() -> None:
    content = render_markdown_report(
        run_id="us-run",
        market=Market.US,
        mode_label="fixture",
        provider_summary={"ranking_profile": "sample_price_trend_v1"},
        snapshot=pl.DataFrame(),
        rankings=pl.DataFrame(),
        config=AppConfig(data_mode="fixture"),
        profile=SamplePriceTrendV1Profile(),
    )

    assert "Successful run with no persisted candidates" in content
    assert "surviving candidate count: 0" in content
    for forbidden in FORBIDDEN_WORDS:
        assert forbidden not in content.lower()


def test_inspect_renders_sample_profile_metrics_and_rankings() -> None:
    profile = SamplePriceTrendV1Profile()
    output = render_inspect(
        run_id="us-run",
        resolution_mode="explicit run_id",
        ticker="AAA",
        metadata=_provider_metadata(),
        snapshot={
            "ticker": "AAA",
            "avg_traded_value_20d_local": 10_000_000.0,
            "return_60d": 0.20,
            "return_120d": 0.40,
        },
        rankings=[
            {
                "horizon": "midterm",
                "rank": 1,
                "return_60d_rank_percentile": 90.0,
                "return_120d_rank_percentile": 70.0,
                "final_rank_percentile": 90.0,
            },
            {
                "horizon": "longterm",
                "rank": 2,
                "return_60d_rank_percentile": 90.0,
                "return_120d_rank_percentile": 70.0,
                "final_rank_percentile": 70.0,
            },
        ],
        profile=profile,
    )

    assert "explicit run_id" in output
    assert "- return_60d: 0.2" in output
    assert "return_120d_rank_percentile 70.0" in output
    assert profile.rank_interpretation_note in output
    assert "Absent tickers do not expose exclusion reasons." in output
