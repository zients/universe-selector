from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.liquidity_quality_v1 import LiquidityQualityV1Profile


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
    *,
    market: Market = Market.US,
    close: float = 20.0,
    volume: float = 1_000_000.0,
    length: int = 63,
    zero_volume_tail: int = 0,
    volume_multiplier_tail_5: float = 1.0,
    range_pct: float = 0.02,
) -> pl.DataFrame:
    rows = []
    for index in range(length):
        bar_close = close * (1.0 + index * 0.001)
        bar_volume = volume
        if index >= length - zero_volume_tail:
            bar_volume = 0.0
        if index >= length - 5:
            bar_volume *= volume_multiplier_tail_5
        half_range = bar_close * range_pct / 2.0
        rows.append(
            {
                "market": market.value,
                "ticker": ticker,
                "bar_date": latest - timedelta(days=length - 1 - index),
                "open": bar_close,
                "high": bar_close + half_range,
                "low": bar_close - half_range,
                "close": bar_close,
                "adjusted_close": bar_close,
                "volume": bar_volume,
            }
        )
    return pl.DataFrame(rows)


def test_liquidity_quality_builds_snapshot_metrics() -> None:
    latest = date(2026, 5, 8)
    profile = LiquidityQualityV1Profile()
    bars = _bars("AAA", latest, close=20.0, volume=1_000_000.0)

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.height == 1
    row = snapshot.to_dicts()[0]
    assert row["profile_metrics_version"] == 1.0
    assert row["avg_traded_value_20d_local"] > 10_000_000.0
    assert row["avg_traded_value_60d_local"] > 10_000_000.0
    assert row["median_traded_value_20d_local"] > 10_000_000.0
    assert row["amihud_illiquidity_20d"] >= 0.0
    assert row["amihud_illiquidity_60d"] >= 0.0
    assert row["traded_value_cv_60d"] >= 0.0
    assert 0.0 < row["traded_value_concentration_60d"] < 1.0
    assert row["median_range_pct_20d"] == pytest.approx(0.02)
    assert row["median_range_pct_60d"] == pytest.approx(0.02)
    assert row["traded_value_5d_to_20d_ratio"] > 0.0
    assert row["zero_volume_days_20d"] == 0.0
    assert row["zero_volume_days_60d"] == 0.0
    assert row["active_trading_days_60d"] == 60.0
    assert row["stale_close_days_20d"] == 0.0
    assert row["stale_close_days_60d"] == 0.0
    assert row["data_quality_extreme_return_flag"] == 0.0


def test_liquidity_quality_filters_unusable_snapshot_rows() -> None:
    latest = date(2026, 5, 8)
    profile = LiquidityQualityV1Profile()
    bars = pl.concat(
        [
            _bars("LOWPRICE", latest, close=4.0, volume=3_000_000.0),
            _bars("LOW20D", latest, close=20.0, volume=100_000.0),
            _bars("LOW5D", latest, close=20.0, volume=1_000_000.0, volume_multiplier_tail_5=0.1),
            _bars("ZEROS", latest, close=20.0, volume=1_000_000.0, zero_volume_tail=2),
            _bars("STALE", latest - timedelta(days=1), close=20.0, volume=1_000_000.0),
        ]
    )

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[
            _listing("LOWPRICE"),
            _listing("LOW20D"),
            _listing("LOW5D"),
            _listing("ZEROS"),
            _listing("STALE"),
        ],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.is_empty()


def test_liquidity_quality_allows_limited_zero_volume_days_with_finite_amihud() -> None:
    latest = date(2026, 5, 8)
    profile = LiquidityQualityV1Profile()
    bars = _bars("AAA", latest, market=Market.TW, close=100.0, volume=1_000_000.0, zero_volume_tail=2)

    snapshot = profile.build_snapshot(
        run_id="tw-test",
        market=Market.TW,
        listings=[_listing("AAA", Market.TW)],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.height == 1
    row = snapshot.to_dicts()[0]
    assert row["zero_volume_days_20d"] == 2.0
    assert row["active_trading_days_60d"] == 58.0
    assert row["amihud_illiquidity_20d"] >= 0.0


def test_liquidity_quality_filters_invalid_ohlcv_values() -> None:
    latest = date(2026, 5, 8)
    profile = LiquidityQualityV1Profile()
    bars = _bars("AAA", latest, close=20.0, volume=1_000_000.0).with_columns(
        pl.when(pl.col("bar_date") == latest)
        .then(float("nan"))
        .otherwise(pl.col("close"))
        .alias("close")
    )

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.is_empty()


def _snapshot_rows(rows: list[dict[str, object]]) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "run_id": "run",
        "market": "US",
        "close": 20.0,
        "adjusted_close": 20.0,
        "profile_metrics_version": 1.0,
        "avg_traded_value_5d_local": 20_000_000.0,
        "avg_traded_value_20d_local": 20_000_000.0,
        "avg_traded_value_60d_local": 20_000_000.0,
        "median_traded_value_20d_local": 20_000_000.0,
        "median_traded_value_60d_local": 20_000_000.0,
        "amihud_illiquidity_20d": 1e-10,
        "amihud_illiquidity_60d": 1e-10,
        "traded_value_cv_60d": 0.20,
        "traded_value_concentration_60d": 0.05,
        "median_range_pct_20d": 0.02,
        "median_range_pct_60d": 0.02,
        "traded_value_5d_to_20d_ratio": 1.0,
        "zero_volume_days_20d": 0.0,
        "zero_volume_days_60d": 0.0,
        "active_trading_days_60d": 60.0,
        "stale_close_days_20d": 0.0,
        "stale_close_days_60d": 0.0,
        "data_quality_extreme_return_flag": 0.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_liquidity_quality_assigns_three_horizons_and_ranks_depth_first() -> None:
    profile = LiquidityQualityV1Profile()
    snapshot = _snapshot_rows(
        [
            {"ticker": "AAA", "avg_traded_value_20d_local": 100_000_000.0, "avg_traded_value_60d_local": 90_000_000.0},
            {"ticker": "BBB", "avg_traded_value_20d_local": 30_000_000.0, "avg_traded_value_60d_local": 30_000_000.0},
            {"ticker": "CCC", "avg_traded_value_20d_local": 12_000_000.0, "avg_traded_value_60d_local": 12_000_000.0},
        ]
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.height == 9
    assert rankings["horizon"].unique().sort().to_list() == ["composite", "shortterm", "stable"]
    composite = rankings.filter(pl.col("horizon") == "composite").sort("rank")
    assert composite["ticker"].to_list()[0] == "AAA"
    assert composite["rank"].to_list() == [1, 2, 3]
    assert composite["score"].is_sorted(descending=True)


def test_liquidity_quality_percentile_ties_use_average_position_and_final_tie_uses_ticker() -> None:
    profile = LiquidityQualityV1Profile()
    snapshot = _snapshot_rows(
        [
            {"ticker": "AAA", "avg_traded_value_20d_local": 100_000_000.0, "avg_traded_value_60d_local": 100_000_000.0},
            {"ticker": "BBB", "avg_traded_value_20d_local": 100_000_000.0, "avg_traded_value_60d_local": 100_000_000.0},
            {"ticker": "CCC", "avg_traded_value_20d_local": 20_000_000.0, "avg_traded_value_60d_local": 20_000_000.0},
        ]
    )

    rankings = profile.assign_rankings(snapshot)
    composite = rankings.filter(pl.col("horizon") == "composite").sort("rank")
    aaa = composite.filter(pl.col("ticker") == "AAA").to_dicts()[0]
    bbb = composite.filter(pl.col("ticker") == "BBB").to_dicts()[0]
    ccc = composite.filter(pl.col("ticker") == "CCC").to_dicts()[0]

    assert aaa["score_log_traded_value_20d"] == pytest.approx(0.75)
    assert bbb["score_log_traded_value_20d"] == pytest.approx(0.75)
    assert ccc["score_log_traded_value_20d"] == pytest.approx(0.0)
    assert composite["ticker"].to_list()[:2] == ["AAA", "BBB"]


def test_liquidity_quality_rejects_non_finite_snapshot_before_ranking_subset() -> None:
    profile = LiquidityQualityV1Profile()
    snapshot = _snapshot_rows(
        [
            {"ticker": "AAA"},
            {"ticker": "BBB", "amihud_illiquidity_60d": float("nan")},
        ]
    )

    with pytest.raises(ValidationError, match="non-finite ranking input"):
        profile.assign_rankings(snapshot)
