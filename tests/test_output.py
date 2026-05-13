from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.output.inspect import render_inspect
from universe_selector.output.report import REPORT_RESEARCH_DISCLAIMER, render_markdown_report
from universe_selector.providers.models import ProviderMetadata
from universe_selector.ranking_profiles.liquidity_quality_v1 import LiquidityQualityV1Profile
from universe_selector.ranking_profiles.momentum_v1 import MomentumV1Profile
from universe_selector.ranking_profiles.sample_price_trend_v1 import SamplePriceTrendV1Profile
from universe_selector.ranking_profiles.trend_quality_v1 import TrendQualityV1Profile


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
                "score": [90.0, 70.0, 85.0, 75.0],
            }
        ),
        config=AppConfig(data_mode="fixture", report_top_n=2),
        profile=profile,
    )

    assert content.startswith("# Universe Selector Report\n")
    assert "## Highest-ranked midterm candidates" in content
    assert "## Highest-ranked longterm candidates" in content
    assert "| rank | ticker | score |" in content
    assert "sample_price_trend_v1" in content
    assert REPORT_RESEARCH_DISCLAIMER in content
    assert profile.rank_interpretation_note in content
    assert "Filtered-out tickers and exclusion reasons are not persisted." in content


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
    assert REPORT_RESEARCH_DISCLAIMER in content


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
                "score_return_60d": 90.0,
                "score_return_120d": 70.0,
                "score": 90.0,
            },
            {
                "horizon": "longterm",
                "rank": 2,
                "score_return_60d": 90.0,
                "score_return_120d": 70.0,
                "score": 70.0,
            },
        ],
        profile=profile,
    )

    assert "explicit run_id" in output
    assert "- return_60d: 0.2" in output
    assert "rank 1, score 90.0, score_return_60d 90.0" in output
    assert "score_return_120d 70.0" in output
    assert profile.rank_interpretation_note in output
    assert "Absent tickers do not expose exclusion reasons." in output


def test_inspect_renders_momentum_profile_metrics_and_rankings() -> None:
    profile = MomentumV1Profile()
    output = render_inspect(
        run_id="us-momentum",
        resolution_mode="explicit run_id",
        ticker="AAA",
        metadata=_provider_metadata(),
        snapshot={
            "ticker": "AAA",
            "momentum_return_12_1": 0.30,
            "momentum_return_6_1": 0.20,
            "volatility_12_1": 0.03,
            "volatility_6_1": 0.02,
            "risk_adjusted_momentum_12_1": 10.0,
            "risk_adjusted_momentum_6_1": 8.0,
            "short_term_strength_20d": 0.08,
        },
        rankings=[
            {
                "horizon": "swing",
                "rank": 1,
                "score_risk_adjusted_momentum_12_1": 10.0,
                "score_risk_adjusted_momentum_6_1": 8.0,
                "score_short_term_strength_20d": 4.0,
                "score": 8.0,
            },
            {
                "horizon": "midterm",
                "rank": 2,
                "score_risk_adjusted_momentum_12_1": 10.0,
                "score_risk_adjusted_momentum_6_1": 8.0,
                "score_short_term_strength_20d": 4.0,
                "score": 9.0,
            },
        ],
        profile=profile,
    )

    assert "## Horizon Rankings" in output
    assert "- momentum_return_12_1: 0.3" in output
    assert "- swing: rank 1, score 8.0, score_risk_adjusted_momentum_12_1 10.0" in output
    assert profile.rank_interpretation_note in output


def test_inspect_renders_liquidity_quality_metrics_through_generic_path() -> None:
    profile = LiquidityQualityV1Profile()
    snapshot = {
        "ticker": "AAA",
        **{key: 0.0 for key in profile.inspect_metric_keys},
    }
    snapshot.update(
        {
            "profile_metrics_version": 1.0,
            "avg_traded_value_20d_local": 25_000_000.0,
            "avg_traded_value_60d_local": 24_000_000.0,
            "amihud_illiquidity_60d": 1e-10,
        }
    )
    ranking_metrics = {key: 0.0 for key in profile.ranking_metric_keys}
    ranking_metrics.update(
        {
            "score_log_traded_value_20d": 1.0,
            "score_log_traded_value_60d": 0.9,
            "depth_score": 0.95,
            "friction_score": 0.85,
            "stability_score": 0.8,
            "tag_positive_deep_liquidity": 1.0,
        }
    )

    output = render_inspect(
        run_id="us-liquidity",
        resolution_mode="latest successful run",
        ticker="AAA",
        metadata=_provider_metadata(),
        snapshot=snapshot,
        rankings=[
            {"horizon": "composite", "rank": 1, "score": 0.90, **ranking_metrics},
            {"horizon": "shortterm", "rank": 1, "score": 0.88, **ranking_metrics},
            {"horizon": "stable", "rank": 1, "score": 0.86, **ranking_metrics},
        ],
        profile=profile,
    )

    assert "## Horizon Rankings" in output
    assert "- avg_traded_value_20d_local: 25000000.0" in output
    assert "- composite: rank 1, score 0.9, score_log_traded_value_20d 1.0" in output
    assert "tag_positive_deep_liquidity 1.0" in output
    assert "volume:" not in output
    assert profile.rank_interpretation_note in output


def test_report_and_inspect_render_trend_quality_interpretation_note() -> None:
    profile = TrendQualityV1Profile()
    snapshot = {"ticker": "AAA", **{key: 0.0 for key in profile.inspect_metric_keys}}
    snapshot.update({"profile_metrics_version": 1.0, "asof_bar_date_yyyymmdd": 20260507.0})
    ranking_metrics = {key: 0.0 for key in profile.ranking_metric_keys}
    ranking_metrics.update({"tag_structure_cap_active": 1.0, "structure_cap_score": 0.70})
    rankings = [
        {"horizon": "composite", "ticker": "AAA", "rank": 1, "score": 0.70, **ranking_metrics},
        {"horizon": "shortterm", "ticker": "AAA", "rank": 1, "score": 0.65, **ranking_metrics},
        {"horizon": "midterm", "ticker": "AAA", "rank": 1, "score": 0.60, **ranking_metrics},
    ]

    report = render_markdown_report(
        run_id="us-trend",
        market=Market.US,
        mode_label="fixture",
        provider_summary={"ranking_profile": "trend_quality_v1"},
        snapshot=pl.DataFrame({"ticker": ["AAA"]}),
        rankings=pl.DataFrame(rankings),
        config=AppConfig(data_mode="fixture", ranking_profile="trend_quality_v1"),
        profile=profile,
    )
    inspect = render_inspect(
        run_id="us-trend",
        resolution_mode="latest successful run",
        ticker="AAA",
        metadata=_provider_metadata(),
        snapshot=snapshot,
        rankings=rankings,
        profile=profile,
    )

    assert profile.rank_interpretation_note in report
    assert REPORT_RESEARCH_DISCLAIMER in report
    assert profile.rank_interpretation_note in inspect
    assert "tag_structure_cap_active 1.0" in inspect
