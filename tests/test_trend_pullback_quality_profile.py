from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.trend_pullback_quality_v1 import (
    TREND_PULLBACK_QUALITY_HORIZON_ORDER,
    TREND_PULLBACK_QUALITY_PROFILE_ID,
    TREND_PULLBACK_QUALITY_RANKING_METRIC_KEYS,
    TREND_PULLBACK_QUALITY_SCORE_METHOD,
    TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS,
    TrendPullbackQualityV1Profile,
)


def _listing(ticker: str = "AAA", market: Market = Market.US) -> ListingCandidate:
    return ListingCandidate(
        market=market,
        ticker=ticker,
        listing_symbol=ticker,
        exchange_segment="NASDAQ" if market is Market.US else "TWSE",
        listing_status="active",
        instrument_type="common_stock",
        source_id=f"unit:{ticker}",
    )


def _bars(
    ticker: str,
    latest: date,
    values: list[float],
    *,
    market: Market = Market.US,
    volume: float = 1_000_000.0,
    zero_volume_tail: int = 0,
    stale_tail: int = 0,
    invalid_ohlc: bool = False,
) -> pl.DataFrame:
    rows = []
    for index, adjusted_close in enumerate(values):
        close = adjusted_close
        high = close * 1.01
        low = close * 0.99
        if invalid_ohlc and index == len(values) - 2:
            high = low * 0.95
        if stale_tail and index >= len(values) - stale_tail:
            close = rows[-1]["close"] if rows else close
            adjusted_close = rows[-1]["adjusted_close"] if rows else adjusted_close
            high = close * 1.01
            low = close * 0.99
        rows.append(
            {
                "market": market.value,
                "ticker": ticker,
                "bar_date": latest - timedelta(days=len(values) - 1 - index),
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "adjusted_close": adjusted_close,
                "volume": 0.0 if index >= len(values) - zero_volume_tail else volume,
            }
        )
    return pl.DataFrame(rows)


def _strong_pullback_values(length: int = 253) -> list[float]:
    current = 50.0
    values = []
    for index in range(length):
        if index < length - 26:
            current *= 1.003
        else:
            current *= 0.9975
        values.append(current)
    return values


def _falling_knife_values(length: int = 253) -> list[float]:
    current = 50.0
    values = []
    for index in range(length):
        if index < length - 45:
            current *= 1.003
        else:
            current *= 0.982
        values.append(current)
    return values


def test_trend_pullback_quality_profile_public_api_and_payload() -> None:
    profile = TrendPullbackQualityV1Profile()

    assert TREND_PULLBACK_QUALITY_PROFILE_ID == "trend_pullback_quality_v1"
    assert profile.horizon_order == ("composite", "near_support", "trend_resume")
    assert profile.horizon_order == TREND_PULLBACK_QUALITY_HORIZON_ORDER
    assert profile.snapshot_metric_keys == TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS
    assert profile.ranking_metric_keys == TREND_PULLBACK_QUALITY_RANKING_METRIC_KEYS
    assert profile.inspect_metric_keys == profile.snapshot_metric_keys
    assert "pullback_from_120d_high" in profile.snapshot_metric_keys
    assert "tag_setup_healthy_pullback" in profile.ranking_metric_keys
    assert "volume" not in profile.snapshot_metric_keys
    assert profile.ranking_config_payload() == {
        "ranking_profile": "trend_pullback_quality_v1",
        "min_history_bars": 252,
        "price_floor": {"TW": 10.0, "US": 5.0},
        "liquidity_floor": {"TW": 50_000_000.0, "US": 10_000_000.0},
        "active_trading_min_days_60": {"TW": 50, "US": 55},
        "zero_volume_max_days_20": {"TW": 3, "US": 1},
        "stale_close_max_days_20": 5,
        "extreme_return_abs_cutoff": 0.80,
        "volatility_floor": 0.0001,
        "horizon_order": list(TREND_PULLBACK_QUALITY_HORIZON_ORDER),
        "snapshot_metric_keys": list(TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS),
        "ranking_metric_keys": list(TREND_PULLBACK_QUALITY_RANKING_METRIC_KEYS),
        "inspect_metric_keys": list(TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS),
        "stdev_ddof": 1,
        "score_method": TREND_PULLBACK_QUALITY_SCORE_METHOD,
    }


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (TrendPullbackQualityV1Profile(profile_id="bad"), "profile_id"),
        (TrendPullbackQualityV1Profile(min_history_bars=251), "min_history_bars"),
        (TrendPullbackQualityV1Profile(horizon_order=("composite",)), "horizon order"),
        (TrendPullbackQualityV1Profile(snapshot_metric_keys=("profile_metrics_version",)), "snapshot metric"),
        (TrendPullbackQualityV1Profile(ranking_metric_keys=("score_prior_strength",)), "ranking metric"),
        (TrendPullbackQualityV1Profile(score_method="bad"), "score_method"),
    ],
)
def test_trend_pullback_quality_profile_rejects_contract_changes(
    profile: TrendPullbackQualityV1Profile, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        profile.validate()


def test_trend_pullback_quality_builds_clean_pullback_snapshot() -> None:
    latest = date(2026, 5, 8)
    profile = TrendPullbackQualityV1Profile()
    bars = _bars("AAA", latest, _strong_pullback_values(), volume=1_000_000.0)

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.height == 1
    assert snapshot.columns == [
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        *profile.snapshot_metric_keys,
    ]
    row = snapshot.to_dicts()[0]
    assert row["ticker"] == "AAA"
    assert row["profile_metrics_version"] == 1.0
    assert row["asof_bar_date_yyyymmdd"] == 20260507.0
    assert row["return_120d_ex_recent_20d"] > 0.0
    assert -0.15 <= row["pullback_from_120d_high"] <= -0.03
    assert row["trend_slope_60d"] > 0.0
    assert row["sma_50d_vs_sma_200d"] > 0.0
    assert row["price_vs_sma_200d"] > 0.0
    assert row["avg_traded_value_20d_local"] >= profile.liquidity_floor[Market.US]
    for key in profile.snapshot_metric_keys:
        assert isinstance(row[key], int | float)
        assert math.isfinite(float(row[key]))


@pytest.mark.parametrize(
    ("ticker", "values", "kwargs"),
    [
        ("SHORT", _strong_pullback_values(251), {}),
        ("LOWPX", [value * 0.05 for value in _strong_pullback_values()], {}),
        ("LOWLIQ", _strong_pullback_values(), {"volume": 1_000.0}),
        ("ZEROTRADE", _strong_pullback_values(), {"volume": 0.0}),
        ("ZEROVOL", _strong_pullback_values(), {"zero_volume_tail": 5}),
        ("STALE", _strong_pullback_values(), {"stale_tail": 6}),
        ("BADHILO", _strong_pullback_values(), {"invalid_ohlc": True}),
        ("KNIFE", _falling_knife_values(), {}),
    ],
)
def test_trend_pullback_quality_excludes_unusable_or_broken_candidates(
    ticker: str, values: list[float], kwargs: dict[str, object]
) -> None:
    latest = date(2026, 5, 8)
    profile = TrendPullbackQualityV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing(ticker)],
        bars=_bars(ticker, latest, values, **kwargs),
        run_latest_bar_date=latest,
    )

    assert snapshot.is_empty()
    assert snapshot.schema == profile.empty_snapshot().schema


def _ranking_snapshot(rows: list[dict[str, object]]) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "run_id": "run",
        "market": "US",
        "ticker": "AAA",
        "close": 100.0,
        "adjusted_close": 100.0,
        "profile_metrics_version": 1.0,
        "asof_bar_date_yyyymmdd": 20260507.0,
        "avg_traded_value_20d_local": 20_000_000.0,
        "avg_traded_value_5d_local": 20_000_000.0,
        "median_traded_value_20d_local": 20_000_000.0,
        "traded_value_5d_to_20d_ratio": 1.0,
        "return_20d": -0.04,
        "return_60d": 0.12,
        "return_120d": 0.30,
        "return_120d_ex_recent_20d": 0.35,
        "risk_adjusted_return_120d_ex_recent_20d": 12.0,
        "volatility_20d": 0.02,
        "volatility_60d": 0.018,
        "volatility_20d_to_60d_ratio": 1.1,
        "trend_slope_60d": 0.002,
        "uptrend_r2_60d": 0.80,
        "trend_consistency_60d": 0.65,
        "sma_20d": 102.0,
        "sma_50d": 99.0,
        "sma_200d": 80.0,
        "price_vs_sma_20d": -0.02,
        "price_vs_sma_50d": 0.01,
        "price_vs_sma_200d": 0.25,
        "sma_20d_vs_sma_50d": 0.03,
        "sma_50d_vs_sma_200d": 0.23,
        "pullback_from_60d_high": -0.08,
        "pullback_from_120d_high": -0.10,
        "volatility_adjusted_pullback_120d": 1.2,
        "days_since_60d_high": 12.0,
        "days_since_120d_high": 18.0,
        "max_drawdown_120d": -0.12,
        "close_position_20d_range": 0.55,
        "low_10d_vs_sma_50d": -0.01,
        "zero_volume_days_20d": 0.0,
        "active_trading_days_60d": 60.0,
        "stale_close_days_20d": 0.0,
        "data_quality_extreme_return_flag": 0.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_trend_pullback_quality_assigns_rankings_tags_and_caps() -> None:
    profile = TrendPullbackQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "AAA"},
            {
                "ticker": "SHALLOW",
                "pullback_from_120d_high": -0.01,
                "pullback_from_60d_high": -0.005,
                "price_vs_sma_20d": 0.14,
                "price_vs_sma_50d": 0.18,
            },
            {
                "ticker": "DEEP",
                "pullback_from_120d_high": -0.24,
                "pullback_from_60d_high": -0.22,
                "price_vs_sma_50d": -0.04,
                "max_drawdown_120d": -0.28,
                "traded_value_5d_to_20d_ratio": 0.40,
                "volatility_20d_to_60d_ratio": 2.1,
            },
        ]
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.height == 9
    assert rankings["horizon"].unique().sort().to_list() == ["composite", "near_support", "trend_resume"]
    for horizon in profile.horizon_order:
        rows = rankings.filter(pl.col("horizon") == horizon).sort("rank")
        assert rows["rank"].to_list() == [1, 2, 3]
        assert rows["score"].is_sorted(descending=True)

    composite = {row["ticker"]: row for row in rankings.filter(pl.col("horizon") == "composite").to_dicts()}
    assert composite["AAA"]["rank"] == 1
    assert composite["AAA"]["tag_setup_healthy_pullback"] == 1.0
    assert composite["AAA"]["tag_setup_near_sma50"] == 1.0
    assert composite["SHALLOW"]["tag_risk_still_overheated"] == 1.0
    assert composite["SHALLOW"]["overheat_cap_score"] < composite["AAA"]["overheat_cap_score"]
    assert composite["DEEP"]["tag_risk_deep_drawdown"] == 1.0
    assert composite["DEEP"]["tag_risk_breakdown"] == 1.0
    assert composite["DEEP"]["tag_risk_liquidity_fade"] == 1.0
    assert composite["DEEP"]["tag_risk_volatility_spike"] == 1.0
    for row in rankings.to_dicts():
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_trend_pullback_quality_rejects_malformed_ranking_snapshot() -> None:
    profile = TrendPullbackQualityV1Profile()
    malformed = _ranking_snapshot([{"ticker": "AAA"}]).drop("pullback_from_120d_high")

    with pytest.raises(ValidationError, match="missing required ranking inputs"):
        profile.assign_rankings(malformed)
