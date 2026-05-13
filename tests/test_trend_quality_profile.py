from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.trend_quality_v1 import TrendQualityV1Profile


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


def _trend_values(length: int = 253, *, start: float = 20.0, step: float = 0.002) -> list[float]:
    values: list[float] = []
    current = start
    for index in range(length):
        seasonal = 0.0008 if index % 5 in {0, 1, 2} else -0.0002
        current *= 1.0 + step + seasonal
        values.append(current)
    return values


def _bars(
    ticker: str,
    latest: date,
    *,
    market: Market = Market.US,
    adjusted_closes: list[float] | None = None,
    close_multiplier: float = 1.0,
    volume: float = 1_000_000.0,
    length: int = 253,
    zero_volume_tail: int = 0,
) -> pl.DataFrame:
    values = adjusted_closes if adjusted_closes is not None else _trend_values(length)
    rows = []
    for index, adjusted_close in enumerate(values):
        close = adjusted_close * close_multiplier
        bar_volume = 0.0 if index >= len(values) - zero_volume_tail else volume
        rows.append(
            {
                "market": market.value,
                "ticker": ticker,
                "bar_date": latest - timedelta(days=len(values) - 1 - index),
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adjusted_close": adjusted_close,
                "volume": bar_volume,
            }
        )
    return pl.DataFrame(rows)


def _std_for_test(values: list[float], *, ddof: int = 1) -> float:
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - ddof)
    return variance**0.5


def _max_drawdown_for_test(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def _ols_slope_r2_for_test(values: list[float]) -> tuple[float, float]:
    y_values = [math.log(value) for value in values]
    x_values = list(range(len(y_values)))
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    ss_xx = sum((value - x_mean) ** 2 for value in x_values)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)) / ss_xx
    intercept = y_mean - slope * x_mean
    total = sum((value - y_mean) ** 2 for value in y_values)
    if total == 0.0:
        return slope, 0.0
    residual = sum(
        (y - (intercept + slope * x)) ** 2
        for x, y in zip(x_values, y_values, strict=True)
    )
    return slope, 1.0 - residual / total


def test_trend_quality_builds_snapshot_metrics_with_common_asof_lag() -> None:
    latest = date(2026, 5, 8)
    adjusted = _trend_values()
    profile = TrendQualityV1Profile()
    bars = _bars("AAA", latest, adjusted_closes=adjusted)

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
    retained = adjusted[:-1][-252:]
    returns = [retained[index] / retained[index - 1] - 1.0 for index in range(1, len(retained))]
    slope, r2 = _ols_slope_r2_for_test(retained[-60:])
    assert row["asof_bar_date_yyyymmdd"] == 20260507.0
    assert row["profile_metrics_version"] == 1.0
    assert row["close"] == pytest.approx(retained[-1])
    assert row["adjusted_close"] == pytest.approx(retained[-1])
    assert row["avg_traded_value_20d_local"] == pytest.approx(
        sum(value * 1_000_000.0 for value in retained[-20:]) / 20.0
    )
    assert row["return_20d"] == pytest.approx(retained[-1] / retained[-21] - 1.0)
    assert row["return_60d"] == pytest.approx(retained[-1] / retained[-61] - 1.0)
    assert row["return_120d"] == pytest.approx(retained[-1] / retained[-121] - 1.0)
    assert row["volatility_60d"] == pytest.approx(_std_for_test(returns[-60:]))
    assert row["trend_slope_60d"] == pytest.approx(slope)
    assert row["trend_r2_60d"] == pytest.approx(r2)
    assert row["uptrend_r2_60d"] == pytest.approx(r2)
    assert row["trend_consistency_60d"] == pytest.approx(sum(1 for value in returns[-60:] if value > 0.0) / 60.0)
    assert row["price_vs_sma_50d"] == pytest.approx(retained[-1] / (sum(retained[-50:]) / 50.0) - 1.0)
    assert row["price_vs_sma_200d"] == pytest.approx(retained[-1] / (sum(retained[-200:]) / 200.0) - 1.0)
    assert row["sma_50d_vs_sma_200d"] == pytest.approx(
        (sum(retained[-50:]) / 50.0) / (sum(retained[-200:]) / 200.0) - 1.0
    )
    assert row["pct_below_120d_high"] == pytest.approx(retained[-1] / max(retained[-120:]) - 1.0)
    assert row["max_drawdown_120d"] == pytest.approx(_max_drawdown_for_test(retained[-120:]))
    assert row["active_trading_days_60d"] == 60.0
    assert row["zero_volume_days_20d"] == 0.0
    assert row["stale_close_days_20d"] == 0.0


def test_trend_quality_retains_exact_latest_252_rows_after_asof_exclusion() -> None:
    latest = date(2026, 5, 8)
    adjusted = _trend_values(260)
    profile = TrendQualityV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=_bars("AAA", latest, adjusted_closes=adjusted),
        run_latest_bar_date=latest,
    )

    retained = adjusted[:-1][-252:]
    row = snapshot.to_dicts()[0]
    assert row["return_120d"] == pytest.approx(retained[-1] / retained[-121] - 1.0)
    assert row["price_vs_sma_200d"] == pytest.approx(retained[-1] / (sum(retained[-200:]) / 200.0) - 1.0)


def test_trend_quality_omits_ticker_missing_common_asof_date() -> None:
    latest = date(2026, 5, 8)
    profile = TrendQualityV1Profile()
    bars = pl.concat([
        _bars("AAA", latest),
        _bars("STALE", latest - timedelta(days=2), length=253),
    ])

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA"), _listing("STALE")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot["ticker"].to_list() == ["AAA"]
    assert snapshot["asof_bar_date_yyyymmdd"].to_list() == [20260507.0]


def test_trend_quality_candidate_bars_filter_market_listing_and_run_latest_date() -> None:
    latest = date(2026, 5, 8)
    profile = TrendQualityV1Profile()
    bars = pl.concat([
        _bars("AAA", latest),
        _bars("UNLISTED", latest + timedelta(days=5), length=1),
        _bars("AAA", latest + timedelta(days=5), length=1),
        _bars("AAA", latest, market=Market.TW),
    ])

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot["ticker"].to_list() == ["AAA"]
    assert snapshot["asof_bar_date_yyyymmdd"].to_list() == [20260507.0]

    empty_missing_listing = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("MISSING")],
        bars=_bars("UNLISTED", latest),
        run_latest_bar_date=latest,
    )
    assert empty_missing_listing.is_empty()
    assert empty_missing_listing.columns == [
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        *profile.snapshot_metric_keys,
    ]

    empty_after_latest_exclusion = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("ONE")],
        bars=_bars("ONE", latest, length=1),
        run_latest_bar_date=latest,
    )
    assert empty_after_latest_exclusion.is_empty()


def test_trend_quality_duplicate_date_invalidates_only_that_ticker() -> None:
    latest = date(2026, 5, 8)
    profile = TrendQualityV1Profile()
    good = _bars("GOOD", latest)
    bad = _bars("BAD", latest)
    duplicate = bad.filter(pl.col("bar_date") == bad["bar_date"][10])
    bars = pl.concat([good, bad, duplicate])

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("GOOD"), _listing("BAD")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot["ticker"].to_list() == ["GOOD"]


def test_trend_quality_filters_ohlcv_and_market_threshold_failures() -> None:
    latest = date(2026, 5, 8)
    profile = TrendQualityV1Profile()
    inactive = _bars("INACTIVE", latest)
    inactive_zero_dates = inactive.sort("bar_date")["bar_date"].to_list()[-60:-54]
    inactive = inactive.with_columns(
        pl.when(pl.col("bar_date").is_in(inactive_zero_dates))
        .then(0.0)
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    bars = pl.concat([
        _bars("GOOD", latest),
        _bars("LOWPRICE", latest, close_multiplier=0.10),
        _bars("LOWLIQ", latest, volume=100_000.0),
        _bars("ZEROS", latest, zero_volume_tail=3),
        inactive,
        _bars("BADHIGH", latest).with_columns(
            pl.when(pl.col("bar_date") == latest - timedelta(days=1))
            .then(pl.col("low") - 1.0)
            .otherwise(pl.col("high"))
            .alias("high")
        ),
    ])

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[
            _listing("GOOD"),
            _listing("LOWPRICE"),
            _listing("LOWLIQ"),
            _listing("ZEROS"),
            _listing("INACTIVE"),
            _listing("BADHIGH"),
        ],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot["ticker"].to_list() == ["GOOD"]


def test_trend_quality_allows_tw_zero_volume_threshold() -> None:
    latest = date(2026, 5, 8)
    profile = TrendQualityV1Profile()
    bars = _bars("2330", latest, market=Market.TW, volume=3_000_000.0, zero_volume_tail=4)

    snapshot = profile.build_snapshot(
        run_id="tw-test",
        market=Market.TW,
        listings=[_listing("2330", Market.TW)],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.height == 1
    row = snapshot.to_dicts()[0]
    assert row["avg_traded_value_20d_local"] >= 50_000_000.0
    assert row["zero_volume_days_20d"] == 3.0
    assert row["active_trading_days_60d"] == 57.0


def test_trend_quality_filters_200_day_extreme_adjusted_return() -> None:
    latest = date(2026, 5, 8)
    profile = TrendQualityV1Profile()
    values = _trend_values()
    values[-100] = values[-101] * 1.90
    bars = _bars("SHOCK", latest, adjusted_closes=values)

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("SHOCK")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.is_empty()


def test_trend_quality_stale_close_uses_21st_prior_retained_bar() -> None:
    latest = date(2026, 5, 8)
    profile = TrendQualityV1Profile()
    values = _trend_values()
    retained_first_index = len(values) - 1 - 20
    values[retained_first_index] = values[retained_first_index - 1]
    bars = _bars("STALE", latest, adjusted_closes=values)
    bars = bars.with_columns(pl.col("adjusted_close").alias("close"))

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("STALE")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.to_dicts()[0]["stale_close_days_20d"] == 1.0


def _snapshot_rows(rows: list[dict[str, object]]) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "run_id": "run",
        "market": "US",
        "close": 20.0,
        "adjusted_close": 20.0,
        "profile_metrics_version": 1.0,
        "asof_bar_date_yyyymmdd": 20260507.0,
        "avg_traded_value_20d_local": 20_000_000.0,
        "return_20d": 0.03,
        "return_60d": 0.08,
        "return_120d": 0.14,
        "volatility_60d": 0.02,
        "trend_slope_60d": 0.001,
        "trend_r2_60d": 0.60,
        "uptrend_r2_60d": 0.60,
        "trend_consistency_60d": 0.60,
        "price_vs_sma_50d": 0.04,
        "price_vs_sma_200d": 0.10,
        "sma_50d_vs_sma_200d": 0.06,
        "pct_below_120d_high": -0.02,
        "max_drawdown_120d": -0.08,
        "zero_volume_days_20d": 0.0,
        "active_trading_days_60d": 60.0,
        "stale_close_days_20d": 0.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_trend_quality_assigns_three_horizons_and_ranks_upward_structure_first() -> None:
    profile = TrendQualityV1Profile()
    snapshot = _snapshot_rows([
        {
            "ticker": "CLEAN",
            "return_20d": 0.08,
            "return_60d": 0.18,
            "return_120d": 0.28,
            "trend_slope_60d": 0.003,
            "trend_r2_60d": 0.80,
            "uptrend_r2_60d": 0.80,
            "trend_consistency_60d": 0.70,
            "price_vs_sma_50d": 0.08,
            "price_vs_sma_200d": 0.18,
            "sma_50d_vs_sma_200d": 0.10,
            "pct_below_120d_high": -0.01,
            "max_drawdown_120d": -0.05,
        },
        {
            "ticker": "WEAK",
            "return_20d": 0.01,
            "return_60d": -0.03,
            "return_120d": 0.01,
            "trend_slope_60d": -0.001,
            "trend_r2_60d": 0.30,
            "uptrend_r2_60d": 0.0,
            "price_vs_sma_50d": -0.02,
            "price_vs_sma_200d": -0.04,
            "sma_50d_vs_sma_200d": -0.01,
            "pct_below_120d_high": -0.20,
            "max_drawdown_120d": -0.30,
        },
    ])

    rankings = profile.assign_rankings(snapshot)

    assert rankings.height == 6
    assert rankings["horizon"].unique().sort().to_list() == ["composite", "midterm", "shortterm"]
    for horizon in profile.horizon_order:
        ranked = rankings.filter(pl.col("horizon") == horizon).sort("rank")
        assert ranked["ticker"].to_list() == ["CLEAN", "WEAK"]
        assert ranked["rank"].to_list() == [1, 2]


def test_trend_quality_positive_only_slope_score_zeroes_nonpositive_slopes() -> None:
    profile = TrendQualityV1Profile()
    snapshot = _snapshot_rows([
        {"ticker": "NEG", "trend_slope_60d": -0.001, "uptrend_r2_60d": 0.0},
        {"ticker": "FLAT", "trend_slope_60d": 0.0, "uptrend_r2_60d": 0.0},
        {"ticker": "LOWPOS", "trend_slope_60d": 0.001, "uptrend_r2_60d": 0.60},
        {"ticker": "HIGHPOS", "trend_slope_60d": 0.002, "uptrend_r2_60d": 0.70},
    ])

    rows = profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite")
    by_ticker = {row["ticker"]: row for row in rows.to_dicts()}
    assert by_ticker["NEG"]["score_trend_slope_60d"] == 0.0
    assert by_ticker["FLAT"]["score_trend_slope_60d"] == 0.0
    assert by_ticker["LOWPOS"]["score_trend_slope_60d"] == 0.0
    assert by_ticker["HIGHPOS"]["score_trend_slope_60d"] == 1.0


def test_trend_quality_uptrend_r2_score_zeroes_nonpositive_slope_snapshot_values() -> None:
    profile = TrendQualityV1Profile()
    snapshot = _snapshot_rows([
        {"ticker": "BADSNAP", "trend_slope_60d": -0.001, "uptrend_r2_60d": 0.80},
    ])

    row = profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite").to_dicts()[0]

    assert row["score_uptrend_r2_60d"] == 0.0
    assert row["trend_cleanliness_cap_score"] == pytest.approx(0.35)


def test_trend_quality_percentile_ties_use_average_position_and_ticker_tiebreak() -> None:
    profile = TrendQualityV1Profile()
    snapshot = _snapshot_rows([
        {"ticker": "AAA", "return_20d": 0.10, "return_60d": 0.10},
        {"ticker": "BBB", "return_20d": 0.10, "return_60d": 0.10},
        {"ticker": "CCC", "return_20d": 0.01, "return_60d": 0.01},
    ])

    composite = profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite").sort("rank")
    aaa = composite.filter(pl.col("ticker") == "AAA").to_dicts()[0]
    bbb = composite.filter(pl.col("ticker") == "BBB").to_dicts()[0]
    assert aaa["score_return_20d"] == pytest.approx(0.75)
    assert bbb["score_return_20d"] == pytest.approx(0.75)
    assert composite["ticker"].to_list()[:2] == ["AAA", "BBB"]
    assert composite["rank"].to_list() == [1, 2, 3]


def test_trend_quality_uses_exact_component_cap_tag_and_horizon_formulas() -> None:
    profile = TrendQualityV1Profile()
    snapshot = _snapshot_rows([
        {
            "ticker": "AAA",
            "return_20d": 0.20,
            "return_60d": 0.10,
            "return_120d": 0.30,
            "trend_slope_60d": 0.002,
            "trend_r2_60d": 0.60,
            "uptrend_r2_60d": 0.60,
            "trend_consistency_60d": 0.70,
            "price_vs_sma_50d": 0.15,
            "price_vs_sma_200d": 0.12,
            "sma_50d_vs_sma_200d": 0.06,
            "pct_below_120d_high": -0.01,
            "max_drawdown_120d": -0.05,
        },
        {
            "ticker": "BBB",
            "return_20d": 0.10,
            "return_60d": 0.15,
            "return_120d": 0.10,
            "trend_slope_60d": 0.001,
            "trend_r2_60d": 0.55,
            "uptrend_r2_60d": 0.55,
            "trend_consistency_60d": 0.60,
            "price_vs_sma_50d": 0.05,
            "price_vs_sma_200d": 0.05,
            "sma_50d_vs_sma_200d": 0.02,
            "pct_below_120d_high": -0.05,
            "max_drawdown_120d": -0.10,
        },
        {
            "ticker": "CCC",
            "return_20d": -0.02,
            "return_60d": -0.02,
            "return_120d": -0.05,
            "trend_slope_60d": -0.001,
            "trend_r2_60d": 0.30,
            "uptrend_r2_60d": 0.0,
            "trend_consistency_60d": 0.40,
            "price_vs_sma_50d": -0.03,
            "price_vs_sma_200d": -0.04,
            "sma_50d_vs_sma_200d": -0.02,
            "pct_below_120d_high": -0.20,
            "max_drawdown_120d": -0.30,
        },
    ])

    rankings = profile.assign_rankings(snapshot)
    aaa = rankings.filter((pl.col("horizon") == "composite") & (pl.col("ticker") == "AAA")).to_dicts()[0]
    aaa_short = rankings.filter((pl.col("horizon") == "shortterm") & (pl.col("ticker") == "AAA")).to_dicts()[0]
    aaa_mid = rankings.filter((pl.col("horizon") == "midterm") & (pl.col("ticker") == "AAA")).to_dicts()[0]
    ccc = rankings.filter((pl.col("horizon") == "composite") & (pl.col("ticker") == "CCC")).to_dicts()[0]

    assert aaa["score_return_20d"] == 1.0
    assert aaa["score_return_60d"] == pytest.approx(0.5)
    assert aaa["score_return_120d"] == 1.0
    assert aaa["score_trend_slope_60d"] == 1.0
    assert aaa["score_uptrend_r2_60d"] == pytest.approx(0.60)
    assert aaa["score_trend_consistency_60d"] == 1.0
    assert aaa["score_price_vs_sma_50d"] == 1.0
    assert aaa["score_price_vs_sma_200d"] == 1.0
    assert aaa["score_sma_50d_vs_sma_200d"] == 1.0
    assert aaa["score_pct_below_120d_high"] == 1.0
    assert aaa["score_drawdown_control_120d"] == 1.0

    expected_strength = 0.35 * 1.0 + 0.25 * 0.5 + 0.25 * 1.0 + 0.15 * 1.0
    expected_cleanliness = 0.35 * 0.60 + 0.25 * 1.0 + 0.25 * 1.0 + 0.15 * 1.0
    expected_breakout = 1.0
    expected_composite = 0.40 * expected_strength + 0.30 * expected_cleanliness + 0.20 * expected_breakout + 0.10 * 1.0
    expected_shortterm = (
        0.20 * 1.0
        + 0.15 * 0.5
        + 0.15 * 1.0
        + 0.15 * 1.0
        + 0.10 * 1.0
        + 0.10 * 1.0
        + 0.05 * 0.60
        + 0.05 * 1.0
        + 0.05 * 1.0
    )
    expected_midterm = (
        0.25 * 1.0
        + 0.20 * 1.0
        + 0.15 * 1.0
        + 0.15 * 1.0
        + 0.10 * 1.0
        + 0.10 * expected_cleanliness
        + 0.05 * 0.60
    )

    assert aaa["trend_strength_score"] == pytest.approx(expected_strength)
    assert aaa["trend_cleanliness_score"] == pytest.approx(expected_cleanliness)
    assert aaa["breakout_position_score"] == pytest.approx(expected_breakout)
    assert aaa["drawdown_control_score"] == 1.0
    assert aaa["structure_cap_score"] == 1.0
    assert aaa["penalty_score"] == 0.0
    assert aaa["tag_structure_uptrend"] == 1.0
    assert aaa["tag_structure_consistent_uptrend"] == 1.0
    assert aaa["tag_structure_cap_active"] == 0.0
    assert aaa["score"] == pytest.approx(expected_composite)
    assert aaa_short["score"] == pytest.approx(expected_shortterm)
    assert aaa_mid["score"] == pytest.approx(expected_midterm)

    assert ccc["weak_structure_fail_count"] == 5.0
    assert ccc["hard_structure_cap_score"] == pytest.approx(0.15)
    assert ccc["penalty_score"] == pytest.approx(0.60)
    assert ccc["tag_structure_nonpositive_60d_slope"] == 1.0
    assert ccc["tag_structure_negative_60d_return"] == 1.0
    assert ccc["tag_structure_below_sma_50d"] == 1.0
    assert ccc["tag_structure_below_sma_200d"] == 1.0
    assert ccc["tag_structure_sma_50d_below_sma_200d"] == 1.0
    assert ccc["tag_structure_weak_trend_component"] == 1.0
    assert ccc["tag_structure_cap_active"] == 1.0


def test_trend_quality_ranks_within_run_market_horizon_partitions() -> None:
    profile = TrendQualityV1Profile()
    snapshot = _snapshot_rows([
        {"run_id": "run-1", "market": "US", "ticker": "AAA", "return_120d": 0.20},
        {"run_id": "run-1", "market": "US", "ticker": "BBB", "return_120d": 0.05},
        {"run_id": "run-1", "market": "TW", "ticker": "2330", "return_120d": 0.04},
        {"run_id": "run-1", "market": "TW", "ticker": "2454", "return_120d": 0.18},
        {"run_id": "run-2", "market": "US", "ticker": "DDD", "return_120d": 0.01},
        {"run_id": "run-2", "market": "US", "ticker": "EEE", "return_120d": 0.30},
    ])

    composite = profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite")

    assert composite.filter((pl.col("run_id") == "run-1") & (pl.col("market") == "US")).sort("rank")["rank"].to_list() == [1, 2]
    assert composite.filter((pl.col("run_id") == "run-1") & (pl.col("market") == "TW")).sort("rank")["rank"].to_list() == [1, 2]
    assert composite.filter((pl.col("run_id") == "run-2") & (pl.col("market") == "US")).sort("rank")["rank"].to_list() == [1, 2]


def test_trend_quality_caps_weak_noisy_flat_and_overextended_structures() -> None:
    profile = TrendQualityV1Profile()
    snapshot = _snapshot_rows([
        {"ticker": "LOWR2", "trend_r2_60d": 0.40, "uptrend_r2_60d": 0.40},
        {"ticker": "MODOVER", "return_20d": 0.35},
        {"ticker": "SEVOVER", "return_20d": 0.55},
        {"ticker": "WEAK1", "price_vs_sma_200d": 0.01},
        {"ticker": "WEAK2", "return_60d": 0.01, "price_vs_sma_200d": 0.01},
        {"ticker": "WEAK3", "trend_slope_60d": 0.0001, "return_60d": 0.01, "price_vs_sma_200d": 0.01},
        {"ticker": "BELOW50", "price_vs_sma_50d": -0.01},
    ])

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite").to_dicts()
    }

    assert rows["LOWR2"]["structure_cap_score"] == pytest.approx(0.70)
    assert rows["MODOVER"]["structure_cap_score"] == pytest.approx(0.75)
    assert rows["SEVOVER"]["structure_cap_score"] == pytest.approx(0.60)
    assert rows["WEAK1"]["structure_cap_score"] == pytest.approx(0.75)
    assert rows["WEAK2"]["structure_cap_score"] == pytest.approx(0.60)
    assert rows["WEAK3"]["structure_cap_score"] == pytest.approx(0.45)
    assert rows["BELOW50"]["hard_structure_cap_score"] == pytest.approx(0.40)
    assert rows["BELOW50"]["tag_structure_below_sma_50d"] == 1.0
    assert rows["BELOW50"]["tag_structure_weak_trend_component"] == 1.0
    assert rows["BELOW50"]["tag_structure_cap_active"] == 1.0


def test_trend_quality_applies_penalties_before_final_structure_cap() -> None:
    profile = TrendQualityV1Profile()
    snapshot = _snapshot_rows([
        {"ticker": "CAPPED", "price_vs_sma_50d": -0.01},
    ])

    row = profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite").to_dicts()[0]

    assert row["hard_structure_cap_score"] == pytest.approx(0.40)
    assert row["penalty_score"] == pytest.approx(0.10)
    assert row["score"] == pytest.approx(row["structure_cap_score"])


def test_trend_quality_penalty_reduces_uncapped_score() -> None:
    profile = TrendQualityV1Profile()
    snapshot = _snapshot_rows([
        {"ticker": "BASE"},
        {"ticker": "STALE", "stale_close_days_20d": 3.0},
    ])

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite").to_dicts()
    }

    assert rows["BASE"]["structure_cap_score"] == 1.0
    assert rows["STALE"]["structure_cap_score"] == 1.0
    assert rows["STALE"]["penalty_score"] == pytest.approx(0.10)
    assert rows["STALE"]["score"] == pytest.approx(rows["BASE"]["score"] - 0.10)


def test_trend_quality_empty_snapshot_returns_empty_ranking_schema() -> None:
    profile = TrendQualityV1Profile()

    rankings = profile.assign_rankings(pl.DataFrame())

    expected_schema = {
        "run_id": pl.String,
        "market": pl.String,
        "horizon": pl.String,
        "ticker": pl.String,
        **{key: pl.Float64 for key in profile.ranking_metric_keys},
        "score": pl.Float64,
        "rank": pl.Int64,
    }

    assert rankings.is_empty()
    assert rankings.columns == list(expected_schema)
    assert dict(rankings.schema) == expected_schema


def test_trend_quality_rejects_malformed_non_numeric_non_finite_and_mixed_asof_snapshots() -> None:
    profile = TrendQualityV1Profile()
    with pytest.raises(ValidationError, match="missing required ranking inputs"):
        profile.assign_rankings(pl.DataFrame({"run_id": ["run"], "market": ["US"], "ticker": ["AAA"]}))

    non_finite = _snapshot_rows([{"ticker": "AAA", "return_20d": float("nan")}])
    with pytest.raises(ValidationError, match="non-finite ranking input"):
        profile.assign_rankings(non_finite)

    non_numeric = _snapshot_rows([{"ticker": "AAA"}]).with_columns(pl.lit("bad").alias("return_20d"))
    with pytest.raises(ValidationError, match="non-numeric ranking input"):
        profile.assign_rankings(non_numeric)

    mixed_asof = _snapshot_rows([
        {"ticker": "AAA", "asof_bar_date_yyyymmdd": 20260507.0},
        {"ticker": "BBB", "asof_bar_date_yyyymmdd": 20260506.0},
    ])
    with pytest.raises(ValidationError, match="mixed as-of"):
        profile.assign_rankings(mixed_asof)


def test_trend_quality_declared_ranking_metrics_are_finite_and_ordered() -> None:
    profile = TrendQualityV1Profile()
    rankings = profile.assign_rankings(_snapshot_rows([
        {"ticker": "AAA", "return_120d": 0.20},
        {"ticker": "BBB", "return_120d": 0.05},
    ]))

    assert rankings.columns == [
        "run_id",
        "market",
        "horizon",
        "ticker",
        *profile.ranking_metric_keys,
        "score",
        "rank",
    ]
    for row in rankings.to_dicts():
        assert isinstance(row["rank"], int)
        assert math.isfinite(float(row["score"]))
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))
