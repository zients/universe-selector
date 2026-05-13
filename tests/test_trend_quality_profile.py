from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from universe_selector.domain import Market
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
