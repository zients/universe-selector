from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from types import MappingProxyType

import polars as pl

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.output.inspect import render_inspect, render_inspect_json
from universe_selector.output.json import json_dumps, to_jsonable
from universe_selector.output.report import REPORT_RESEARCH_DISCLAIMER, render_json_report, render_markdown_report
from universe_selector.providers.models import ProviderMetadata
from universe_selector.ranking_profiles.base_breakout_quality_v1 import BaseBreakoutQualityV1Profile
from universe_selector.ranking_profiles.defensive_compounder_quality_v1 import DefensiveCompounderQualityV1Profile
from universe_selector.ranking_profiles.liquidity_quality_v1 import LiquidityQualityV1Profile
from universe_selector.ranking_profiles.mean_reversion_quality_v1 import MeanReversionQualityV1Profile
from universe_selector.ranking_profiles.momentum_v1 import MomentumV1Profile
from universe_selector.ranking_profiles.momentum_quality_v1 import MomentumQualityV1Profile
from universe_selector.ranking_profiles.relative_strength_leader_v1 import RelativeStrengthLeaderV1Profile
from universe_selector.ranking_profiles.sample_price_trend_v1 import SamplePriceTrendV1Profile
from universe_selector.ranking_profiles.trend_quality_v1 import TrendQualityV1Profile
from universe_selector.ranking_profiles.trend_pullback_quality_v1 import TrendPullbackQualityV1Profile
from universe_selector.ranking_profiles.volatility_quality_v1 import VolatilityQualityV1Profile


def _provider_metadata() -> ProviderMetadata:
    return ProviderMetadata(
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


@dataclass(frozen=True)
class _JsonFixture:
    when: date
    values: tuple[int, ...]


def test_json_helpers_convert_domain_values_and_emit_compact_sorted_json() -> None:
    payload = {
        "z": MappingProxyType({"b": 2, "a": 1}),
        "a": _JsonFixture(date(2026, 5, 18), (3, 4)),
        "dt": datetime(2026, 5, 18, 1, 2, 3, tzinfo=timezone.utc),
    }

    converted = to_jsonable(payload)
    encoded = json_dumps(payload)

    assert converted == {
        "z": {"b": 2, "a": 1},
        "a": {"when": "2026-05-18", "values": [3, 4]},
        "dt": "2026-05-18T01:02:03+00:00",
    }
    assert encoded == ('{"a":{"values":[3,4],"when":"2026-05-18"},"dt":"2026-05-18T01:02:03+00:00","z":{"a":1,"b":2}}')
    assert json.loads(encoded)["a"]["when"] == "2026-05-18"


def _ranking_row(horizon: str, rank: int) -> dict[str, object]:
    return {
        "horizon": horizon,
        "rank": rank,
        "score": 90.0 - rank,
        "score_risk_adjusted_momentum_12_1": 90.0,
        "score_risk_adjusted_momentum_6_1": 85.0,
        "score_short_term_strength_20d": 80.0,
        "score_short_term_extension_20d": 95.0,
        "score_distance_from_ma20": 92.0,
        "score_volatility_12_1": 70.0,
        "score_volatility_6_1": 75.0,
        "moving_average_structure_score": 88.0,
        "drawdown_control_score": 82.0,
        "uptrend_consistency_score": 86.0,
        "trend_quality_score": 85.6,
        "momentum_blend_score": 87.25,
        "overheat_score": 93.8,
        "overheat_penalty_score": 69.0,
        "tag_risk_overheated": 1.0,
        "tag_risk_extended_from_ma20": 1.0,
        "tag_risk_high_volatility": 0.0,
        "tag_risk_large_drawdown": 0.0,
        "tag_risk_weak_trend_quality": 0.0,
        "tag_risk_thin_recent_volume": 0.0,
        "tag_risk_data_quality_warning": 0.0,
        "tag_positive_strong_momentum": 1.0,
        "tag_positive_stable_uptrend": 1.0,
        "tag_positive_early_breakout": 0.0,
    }


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


def test_markdown_report_does_not_render_ranking_tag_columns() -> None:
    profile = MomentumQualityV1Profile()
    content = render_markdown_report(
        run_id="us-v2",
        market=Market.US,
        mode_label="fixture",
        provider_summary={"ranking_profile": profile.profile_id},
        snapshot=pl.DataFrame({"ticker": ["AAA", "BBB"]}),
        rankings=pl.DataFrame(
            {
                "horizon": ["composite", "composite", "swing", "swing", "midterm", "midterm"],
                "rank": [1, 2, 1, 2, 1, 2],
                "ticker": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
                "score": [90.0, 80.0, 91.0, 81.0, 92.0, 82.0],
                "tag_risk_overheated": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "tag_risk_extended_from_ma20": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "tag_risk_high_volatility": [0.0] * 6,
                "tag_risk_large_drawdown": [0.0] * 6,
                "tag_risk_weak_trend_quality": [0.0] * 6,
                "tag_risk_thin_recent_volume": [0.0] * 6,
                "tag_risk_data_quality_warning": [0.0] * 6,
                "tag_positive_strong_momentum": [1.0] * 6,
                "tag_structure_cap_active": [1.0] * 6,
            }
        ),
        config=AppConfig(data_mode="fixture", report_top_n=2),
        profile=profile,
    )

    ranking_content = content.split("## Methodology Notes", 1)[0]
    assert "| rank | ticker | score |" in ranking_content
    assert "| rank | ticker | score | risk_tags |" not in ranking_content
    assert "risk_tags" not in ranking_content
    assert "overheated" not in ranking_content
    assert "extended_from_ma20" not in ranking_content
    assert "tag_risk_overheated" not in ranking_content
    assert "tag_positive_strong_momentum" not in ranking_content
    assert "tag_structure_cap_active" not in ranking_content
    assert "| 1 | AAA | 90.0000 |" in content
    assert "| 2 | BBB | 80.0000 |" in content
    assert ranking_content.count("| rank | ticker | score |") == len(profile.horizon_order)


def test_markdown_reports_use_plain_ranking_tables_for_any_profile() -> None:
    profiles = [
        SamplePriceTrendV1Profile(),
        MomentumV1Profile(),
        MomentumQualityV1Profile(),
        LiquidityQualityV1Profile(),
        TrendQualityV1Profile(),
        TrendPullbackQualityV1Profile(),
        VolatilityQualityV1Profile(),
        BaseBreakoutQualityV1Profile(),
        RelativeStrengthLeaderV1Profile(),
        MeanReversionQualityV1Profile(),
        DefensiveCompounderQualityV1Profile(),
    ]
    for profile in profiles:
        tag_metric_keys = [key for key in profile.ranking_metric_keys if key.startswith("tag_")]
        rankings = pl.DataFrame(
            {
                "horizon": list(profile.horizon_order),
                "rank": [1] * len(profile.horizon_order),
                "ticker": ["AAA"] * len(profile.horizon_order),
                "score": [1.0] * len(profile.horizon_order),
                **{key: [1.0] * len(profile.horizon_order) for key in tag_metric_keys},
            }
        )
        content = render_markdown_report(
            run_id=f"us-{profile.profile_id}",
            market=Market.US,
            mode_label="fixture",
            provider_summary={"ranking_profile": profile.profile_id},
            snapshot=pl.DataFrame({"ticker": ["AAA"]}),
            rankings=rankings,
            config=AppConfig(data_mode="fixture", ranking_profile=profile.profile_id),
            profile=profile,
        )

        ranking_content = content.split("## Methodology Notes", 1)[0]
        assert profile.rank_interpretation_note in content, profile.profile_id
        assert ranking_content.count("| rank | ticker | score |") == len(profile.horizon_order)
        assert "| rank | ticker | score | risk_tags |" not in ranking_content, profile.profile_id
        assert "tag_" not in ranking_content, profile.profile_id
        assert "risk_tags" not in ranking_content, profile.profile_id


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


def test_json_report_renders_top_n_horizon_rows_with_nested_metrics() -> None:
    profile = SamplePriceTrendV1Profile()
    content = render_json_report(
        run_id="us-sample",
        market=Market.US,
        mode_label="fixture",
        provider_summary={
            "data_mode": "fixture",
            "ranking_profile": "sample_price_trend_v1",
            "ranking_config_hash": "sample-hash",
        },
        snapshot=pl.DataFrame(
            {
                "run_id": ["us-sample", "us-sample"],
                "market": ["US", "US"],
                "ticker": ["AAA", "BBB"],
                "close": [10.0, 20.0],
                "adjusted_close": [10.0, 20.0],
                "avg_traded_value_20d_local": [10_000_000.0, 20_000_000.0],
                "return_60d": [0.20, 0.10],
                "return_120d": [0.40, 0.30],
            }
        ),
        rankings=pl.DataFrame(
            {
                "run_id": ["us-sample", "us-sample", "us-sample", "us-sample"],
                "market": ["US", "US", "US", "US"],
                "horizon": ["midterm", "midterm", "longterm", "longterm"],
                "rank": [1, 2, 1, 2],
                "ticker": ["AAA", "BBB", "BBB", "AAA"],
                "score": [90.0, 70.0, 85.0, 75.0],
                "score_return_60d": [90.0, 70.0, 70.0, 90.0],
                "score_return_120d": [70.0, 60.0, 85.0, 70.0],
            }
        ),
        config=AppConfig(data_mode="fixture", report_top_n=1),
        profile=profile,
    )

    payload = json.loads(content)

    assert payload["artifact_type"] == "universe_selector_report"
    assert payload["schema_version"] == 1
    assert payload["run_id"] == "us-sample"
    assert payload["market"] == "US"
    assert payload["ranking_profile"] == "sample_price_trend_v1"
    assert payload["ranking_config_hash"] == "sample-hash"
    assert payload["candidate_summary"] == {"snapshot_rows": 2, "ranking_rows": 4, "top_n": 1}
    assert [row["ticker"] for row in payload["snapshots"]] == ["AAA", "BBB"]
    assert [row["ticker"] for row in payload["rankings"]] == ["AAA", "BBB", "BBB", "AAA"]
    assert payload["rankings"][0]["metrics"]["score_return_60d"] == 90.0
    assert [row["ticker"] for row in payload["top_horizons"]["midterm"]] == ["AAA"]
    assert payload["top_horizons"]["midterm"][0]["ranking"]["metrics"]["score_return_60d"] == 90.0
    assert payload["top_horizons"]["midterm"][0]["snapshot"]["metrics"]["return_60d"] == 0.20
    assert [row["ticker"] for row in payload["top_horizons"]["longterm"]] == ["BBB"]
    assert (
        payload["notes"][-1] == "This report is rendered during batch; report and inspect read persisted results only."
    )


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


def test_inspect_json_separates_core_fields_from_profile_metrics() -> None:
    profile = SamplePriceTrendV1Profile()
    output = render_inspect_json(
        run_id="us-run",
        resolution_mode="explicit run_id",
        ticker="AAA",
        metadata=_provider_metadata(),
        snapshot={
            "run_id": "us-run",
            "market": "US",
            "ticker": "AAA",
            "close": 10.0,
            "adjusted_close": 10.0,
            "avg_traded_value_20d_local": 10_000_000.0,
            "return_60d": 0.20,
            "return_120d": 0.40,
        },
        rankings=[
            {
                "run_id": "us-run",
                "market": "US",
                "horizon": "midterm",
                "ticker": "AAA",
                "rank": 1,
                "score_return_60d": 90.0,
                "score_return_120d": 70.0,
                "score": 90.0,
            },
            {
                "run_id": "us-run",
                "market": "US",
                "horizon": "longterm",
                "ticker": "AAA",
                "rank": 2,
                "score_return_60d": 90.0,
                "score_return_120d": 70.0,
                "score": 70.0,
            },
        ],
        profile=profile,
        ranking_profile="sample_price_trend_v1",
        ranking_config_hash="sample-hash",
    )

    payload = json.loads(output)

    assert payload["artifact_type"] == "universe_selector_inspect"
    assert payload["schema_version"] == 1
    assert payload["ranking_profile"] == "sample_price_trend_v1"
    assert payload["ranking_config_hash"] == "sample-hash"
    assert payload["snapshot"]["close"] == 10.0
    assert payload["snapshot"]["metrics"]["return_60d"] == 0.20
    assert "return_60d" not in payload["snapshot"].keys() - {"metrics"}
    assert payload["rankings"][0]["metrics"]["score_return_60d"] == 90.0
    assert payload["provider_metadata"]["run_latest_bar_date"] == "2026-04-24"


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


def test_inspect_renders_momentum_quality_tags_through_generic_path() -> None:
    profile = MomentumQualityV1Profile()
    ranking_metrics = {key: 0.0 for key in profile.ranking_metric_keys}
    ranking_metrics.update(
        {
            "score_risk_adjusted_momentum_12_1": 90.0,
            "score_risk_adjusted_momentum_6_1": 85.0,
            "score_short_term_strength_20d": 80.0,
            "tag_risk_overheated": 1.0,
            "tag_positive_strong_momentum": 1.0,
        }
    )
    snapshot = {"ticker": "AAA", **{key: 0.0 for key in profile.inspect_metric_keys}}

    output = render_inspect(
        run_id="us-v2",
        resolution_mode="latest successful run",
        ticker="AAA",
        metadata=_provider_metadata(),
        snapshot=snapshot,
        rankings=[
            {"horizon": "composite", "rank": 1, "score": 0.90, **ranking_metrics},
            {"horizon": "swing", "rank": 1, "score": 0.88, **ranking_metrics},
            {"horizon": "midterm", "rank": 1, "score": 0.86, **ranking_metrics},
        ],
        profile=profile,
    )

    assert "## Horizon Rankings" in output
    assert "## Ranking Scores And Tags" not in output
    assert "- composite: rank 1, score 0.9, score_risk_adjusted_momentum_12_1 90.0" in output
    assert "tag_risk_overheated 1.0" in output
    assert "tag_positive_strong_momentum 1.0" in output
    tag_keys = [key for key in profile.ranking_metric_keys if key.startswith("tag_")]
    for key in tag_keys:
        assert f"{key} " in output


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


def test_report_tables_omit_and_inspect_renders_trend_quality_diagnostics() -> None:
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

    report_ranking_content = report.split("## Methodology Notes", 1)[0]
    assert profile.rank_interpretation_note in report
    assert "tag_structure_cap_active" not in report_ranking_content
    assert REPORT_RESEARCH_DISCLAIMER in report
    assert profile.rank_interpretation_note in inspect
    assert "tag_structure_cap_active 1.0" in inspect
