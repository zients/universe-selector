from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.volatility_quality_v1 import VolatilityQualityV1Profile


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


def _series(length: int = 126, *, shock_index: int | None = None) -> list[float]:
    values = []
    current = 20.0
    for index in range(length):
        step = 0.0015 if index % 2 == 0 else -0.0005
        current *= 1.0 + step
        if shock_index is not None and index == shock_index:
            current *= 1.95
        values.append(current)
    return values


def _bars(
    ticker: str,
    latest: date,
    *,
    market: Market = Market.US,
    adjusted_closes: list[float] | None = None,
    close: float = 20.0,
    volume: float = 1_000_000.0,
    length: int = 126,
    zero_volume_tail: int = 0,
    range_pct: float = 0.02,
) -> pl.DataFrame:
    values = adjusted_closes if adjusted_closes is not None else _series(length)
    rows = []
    for index, adjusted_close in enumerate(values):
        bar_close = adjusted_close if adjusted_closes is not None else close * (1.0 + index * 0.001)
        bar_volume = 0.0 if index >= len(values) - zero_volume_tail else volume
        half_range = bar_close * range_pct / 2.0
        rows.append(
            {
                "market": market.value,
                "ticker": ticker,
                "bar_date": latest - timedelta(days=len(values) - 1 - index),
                "open": bar_close,
                "high": bar_close + half_range,
                "low": bar_close - half_range,
                "close": bar_close,
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


def test_volatility_quality_builds_snapshot_metrics() -> None:
    latest = date(2026, 5, 8)
    adjusted = _series()
    profile = VolatilityQualityV1Profile()
    bars = _bars("AAA", latest, adjusted_closes=adjusted, range_pct=0.03)

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.height == 1
    row = snapshot.to_dicts()[0]
    returns = [adjusted[index] / adjusted[index - 1] - 1.0 for index in range(1, len(adjusted))]
    returns_20d = returns[-20:]
    returns_60d = returns[-60:]
    downside = [min(value, 0.0) for value in returns_60d]
    expected_vol_20d = _std_for_test(returns_20d)
    expected_vol_60d = _std_for_test(returns_60d)

    assert row["profile_metrics_version"] == 1.0
    assert row["avg_traded_value_20d_local"] > 10_000_000.0
    assert row["volatility_20d"] == pytest.approx(expected_vol_20d)
    assert row["volatility_60d"] == pytest.approx(expected_vol_60d)
    assert row["downside_volatility_60d"] == pytest.approx(
        (sum(value * value for value in downside) / 60.0) ** 0.5
    )
    assert row["volatility_20d_to_60d_ratio"] == pytest.approx(expected_vol_20d / expected_vol_60d)
    assert row["volatility_stability_60d"] == pytest.approx(abs(math.log(expected_vol_20d / expected_vol_60d)))
    assert row["max_drawdown_120d"] == pytest.approx(_max_drawdown_for_test(adjusted[-120:]))
    assert row["median_range_pct_20d"] == pytest.approx(0.03)
    assert row["median_range_pct_60d"] == pytest.approx(0.03)
    assert row["zero_volume_days_20d"] == 0.0
    assert row["active_trading_days_60d"] == 60.0
    assert row["stale_close_days_20d"] == 0.0
    assert row["data_quality_extreme_return_flag"] == 0.0


def test_volatility_quality_filters_unusable_snapshot_rows() -> None:
    latest = date(2026, 5, 8)
    profile = VolatilityQualityV1Profile()
    bars = pl.concat(
        [
            _bars("SHORT", latest, length=125),
            _bars("STALEDATE", latest - timedelta(days=1)),
            _bars("LOWPRICE", latest, close=4.0, volume=3_000_000.0),
            _bars("LOWLIQ", latest, close=20.0, volume=100_000.0),
            _bars("ZEROS", latest, zero_volume_tail=2),
            _bars("FLAT", latest, adjusted_closes=[20.0] * 126),
        ]
    )

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[
            _listing("SHORT"),
            _listing("STALEDATE"),
            _listing("LOWPRICE"),
            _listing("LOWLIQ"),
            _listing("ZEROS"),
            _listing("FLAT"),
        ],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.is_empty()


def test_volatility_quality_filters_zero_recent_volatility_without_crashing() -> None:
    latest = date(2026, 5, 8)
    profile = VolatilityQualityV1Profile()
    adjusted = _series(105) + [30.0] * 21
    bars = _bars("ZEROVOL20", latest, adjusted_closes=adjusted)

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("ZEROVOL20")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.is_empty()


def test_volatility_quality_filters_insufficient_active_trading_days() -> None:
    latest = date(2026, 5, 8)
    profile = VolatilityQualityV1Profile()
    bars = _bars("AAA", latest)
    zero_dates = set(bars.sort("bar_date")["bar_date"].to_list()[-60:-54])
    bars = bars.with_columns(
        pl.when(pl.col("bar_date").is_in(list(zero_dates)))
        .then(0.0)
        .otherwise(pl.col("volume"))
        .alias("volume")
    )

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.is_empty()


def test_volatility_quality_allows_tw_zero_volume_threshold() -> None:
    latest = date(2026, 5, 8)
    profile = VolatilityQualityV1Profile()
    bars = _bars("2330", latest, market=Market.TW, close=100.0, volume=1_000_000.0, zero_volume_tail=3)

    snapshot = profile.build_snapshot(
        run_id="tw-test",
        market=Market.TW,
        listings=[_listing("2330", Market.TW)],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.height == 1
    row = snapshot.to_dicts()[0]
    assert row["zero_volume_days_20d"] == 3.0
    assert row["active_trading_days_60d"] == 57.0


def test_volatility_quality_filters_invalid_ohlcv_values() -> None:
    latest = date(2026, 5, 8)
    profile = VolatilityQualityV1Profile()
    bars = _bars("BAD", latest).with_columns(
        pl.when(pl.col("bar_date") == latest)
        .then(pl.col("low") - 1.0)
        .otherwise(pl.col("high"))
        .alias("high")
    )

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("BAD")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.is_empty()


def test_volatility_quality_filters_open_outside_daily_range() -> None:
    latest = date(2026, 5, 8)
    profile = VolatilityQualityV1Profile()
    bars = _bars("BADOPEN", latest).with_columns(
        pl.when(pl.col("bar_date") == latest)
        .then(pl.col("high") + 1.0)
        .otherwise(pl.col("open"))
        .alias("open")
    )

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("BADOPEN")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot.is_empty()


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "adjusted_close", "volume"])
def test_volatility_quality_filters_non_finite_ohlcv_values(column: str) -> None:
    latest = date(2026, 5, 8)
    profile = VolatilityQualityV1Profile()
    bars = _bars("NAN", latest).with_columns(
        pl.when(pl.col("bar_date") == latest)
        .then(float("nan"))
        .otherwise(pl.col(column))
        .alias(column)
    )

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("NAN")],
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
        "avg_traded_value_20d_local": 20_000_000.0,
        "volatility_20d": 0.020,
        "volatility_60d": 0.025,
        "downside_volatility_60d": 0.010,
        "volatility_20d_to_60d_ratio": 0.8,
        "volatility_stability_60d": 0.22314355131420976,
        "max_drawdown_120d": -0.10,
        "median_range_pct_20d": 0.020,
        "median_range_pct_60d": 0.025,
        "zero_volume_days_20d": 0.0,
        "active_trading_days_60d": 60.0,
        "stale_close_days_20d": 0.0,
        "data_quality_extreme_return_flag": 0.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_volatility_quality_assigns_three_horizons_and_ranks_lower_volatility_first() -> None:
    profile = VolatilityQualityV1Profile()
    snapshot = _snapshot_rows(
        [
            {
                "ticker": "AAA",
                "volatility_20d": 0.010,
                "volatility_60d": 0.012,
                "downside_volatility_60d": 0.004,
                "max_drawdown_120d": -0.04,
                "median_range_pct_20d": 0.010,
                "median_range_pct_60d": 0.012,
                "volatility_stability_60d": 0.05,
            },
            {
                "ticker": "BBB",
                "volatility_20d": 0.020,
                "volatility_60d": 0.025,
                "downside_volatility_60d": 0.010,
                "max_drawdown_120d": -0.10,
                "median_range_pct_20d": 0.020,
                "median_range_pct_60d": 0.025,
                "volatility_stability_60d": 0.15,
            },
            {
                "ticker": "CCC",
                "volatility_20d": 0.040,
                "volatility_60d": 0.050,
                "downside_volatility_60d": 0.025,
                "max_drawdown_120d": -0.35,
                "median_range_pct_20d": 0.050,
                "median_range_pct_60d": 0.055,
                "volatility_stability_60d": 0.60,
            },
        ]
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.height == 9
    assert rankings["horizon"].unique().sort().to_list() == ["composite", "shortterm", "stable"]
    composite = rankings.filter(pl.col("horizon") == "composite").sort("rank")
    assert composite["ticker"].to_list() == ["AAA", "BBB", "CCC"]
    assert composite["rank"].to_list() == [1, 2, 3]
    assert composite["score"].is_sorted(descending=True)


def test_volatility_quality_uses_exact_component_and_horizon_formulas() -> None:
    profile = VolatilityQualityV1Profile()
    snapshot = _snapshot_rows(
        [
            {
                "ticker": "AAA",
                "volatility_20d": 0.010,
                "volatility_60d": 0.010,
                "downside_volatility_60d": 0.004,
                "max_drawdown_120d": -0.04,
                "median_range_pct_20d": 0.010,
                "median_range_pct_60d": 0.010,
                "volatility_stability_60d": 0.05,
            },
            {
                "ticker": "BBB",
                "volatility_20d": 0.020,
                "volatility_60d": 0.040,
                "downside_volatility_60d": 0.010,
                "max_drawdown_120d": -0.02,
                "median_range_pct_20d": 0.020,
                "median_range_pct_60d": 0.030,
                "volatility_stability_60d": 0.01,
            },
            {
                "ticker": "CCC",
                "volatility_20d": 0.030,
                "volatility_60d": 0.020,
                "downside_volatility_60d": 0.006,
                "max_drawdown_120d": -0.10,
                "median_range_pct_20d": 0.030,
                "median_range_pct_60d": 0.020,
                "volatility_stability_60d": 0.10,
            },
            {
                "ticker": "DDD",
                "volatility_20d": 0.040,
                "volatility_60d": 0.030,
                "downside_volatility_60d": 0.020,
                "max_drawdown_120d": -0.30,
                "median_range_pct_20d": 0.040,
                "median_range_pct_60d": 0.040,
                "volatility_stability_60d": 0.20,
            },
        ]
    )

    rankings = profile.assign_rankings(snapshot)
    composite = rankings.filter((pl.col("horizon") == "composite") & (pl.col("ticker") == "BBB")).to_dicts()[0]
    shortterm = rankings.filter((pl.col("horizon") == "shortterm") & (pl.col("ticker") == "BBB")).to_dicts()[0]
    stable = rankings.filter((pl.col("horizon") == "stable") & (pl.col("ticker") == "BBB")).to_dicts()[0]

    assert composite["score_low_volatility_20d"] == pytest.approx(2.0 / 3.0)
    assert composite["score_low_volatility_60d"] == pytest.approx(0.0)
    assert composite["score_downside_volatility_60d"] == pytest.approx(1.0 / 3.0)
    assert composite["score_drawdown_control_120d"] == pytest.approx(1.0)
    assert composite["score_range_tightness_20d"] == pytest.approx(2.0 / 3.0)
    assert composite["score_range_tightness_60d"] == pytest.approx(1.0 / 3.0)
    assert composite["score_volatility_stability_60d"] == pytest.approx(1.0)

    expected_volatility_control = 0.45 * 0.0 + 0.35 * (2.0 / 3.0) + 0.20 * (1.0 / 3.0)
    expected_trading_smoothness = 0.50 * (2.0 / 3.0) + 0.30 * (1.0 / 3.0) + 0.20 * 1.0
    expected_drawdown_quality = 1.0
    expected_composite = (
        0.45 * expected_volatility_control
        + 0.30 * expected_drawdown_quality
        + 0.25 * expected_trading_smoothness
    )
    expected_shortterm = 0.45 * (2.0 / 3.0) + 0.35 * (2.0 / 3.0) + 0.20 * 1.0
    expected_stable = 0.40 * 0.0 + 0.25 * (1.0 / 3.0) + 0.25 * 1.0 + 0.10 * (1.0 / 3.0)

    assert composite["volatility_control_score"] == pytest.approx(expected_volatility_control)
    assert composite["trading_smoothness_score"] == pytest.approx(expected_trading_smoothness)
    assert composite["drawdown_quality_score"] == pytest.approx(expected_drawdown_quality)
    assert composite["score"] == pytest.approx(expected_composite)
    assert shortterm["score"] == pytest.approx(expected_shortterm)
    assert stable["score"] == pytest.approx(expected_stable)


def test_volatility_quality_percentile_ties_use_average_position_and_ticker_tiebreak() -> None:
    profile = VolatilityQualityV1Profile()
    snapshot = _snapshot_rows(
        [
            {"ticker": "AAA", "volatility_20d": 0.010, "volatility_60d": 0.020},
            {"ticker": "BBB", "volatility_20d": 0.010, "volatility_60d": 0.020},
            {"ticker": "CCC", "volatility_20d": 0.040, "volatility_60d": 0.050},
        ]
    )

    rankings = profile.assign_rankings(snapshot)
    composite = rankings.filter(pl.col("horizon") == "composite").sort("rank")
    aaa = composite.filter(pl.col("ticker") == "AAA").to_dicts()[0]
    bbb = composite.filter(pl.col("ticker") == "BBB").to_dicts()[0]
    ccc = composite.filter(pl.col("ticker") == "CCC").to_dicts()[0]

    assert aaa["score_low_volatility_20d"] == pytest.approx(0.75)
    assert bbb["score_low_volatility_20d"] == pytest.approx(0.75)
    assert ccc["score_low_volatility_20d"] == pytest.approx(0.0)
    assert composite["ticker"].to_list()[:2] == ["AAA", "BBB"]


def test_volatility_quality_ranks_within_run_market_partitions() -> None:
    profile = VolatilityQualityV1Profile()
    snapshot = _snapshot_rows(
        [
            {"market": "US", "ticker": "AAA", "volatility_20d": 0.010, "volatility_60d": 0.010},
            {"market": "US", "ticker": "BBB", "volatility_20d": 0.030, "volatility_60d": 0.030},
            {"market": "TW", "ticker": "2330", "volatility_20d": 0.050, "volatility_60d": 0.050},
            {"market": "TW", "ticker": "2317", "volatility_20d": 0.020, "volatility_60d": 0.020},
        ]
    )

    composite = profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite")

    assert composite.filter(pl.col("market") == "US").sort("rank")["ticker"].to_list() == ["AAA", "BBB"]
    assert composite.filter(pl.col("market") == "TW").sort("rank")["ticker"].to_list() == ["2317", "2330"]
    assert composite.filter((pl.col("market") == "US") & (pl.col("ticker") == "AAA")).to_dicts()[0]["rank"] == 1
    assert composite.filter((pl.col("market") == "TW") & (pl.col("ticker") == "2317")).to_dicts()[0]["rank"] == 1


def test_volatility_quality_penalty_lowers_horizon_scores() -> None:
    profile = VolatilityQualityV1Profile()
    base_snapshot = _snapshot_rows([{"ticker": "AAA"}])
    penalized_snapshot = _snapshot_rows(
        [
            {
                "ticker": "AAA",
                "stale_close_days_20d": 3.0,
                "data_quality_extreme_return_flag": 1.0,
                "volatility_stability_60d": 0.80,
            }
        ]
    )

    base = profile.assign_rankings(base_snapshot).filter(pl.col("horizon") == "composite").to_dicts()[0]
    penalized = profile.assign_rankings(penalized_snapshot).filter(pl.col("horizon") == "composite").to_dicts()[0]

    assert base["penalty_score"] == 0.0
    assert penalized["penalty_score"] == pytest.approx(0.35)
    assert penalized["score"] == pytest.approx(base["score"] - 0.35)


def test_volatility_quality_empty_snapshot_returns_empty_ranking_schema() -> None:
    profile = VolatilityQualityV1Profile()

    rankings = profile.assign_rankings(pl.DataFrame())

    assert rankings.is_empty()
    assert rankings.schema["score_low_volatility_20d"] == pl.Float64
    assert rankings.schema["score"] == pl.Float64
    assert rankings.schema["rank"] == pl.Int64


def test_volatility_quality_rejects_malformed_or_non_finite_snapshot() -> None:
    profile = VolatilityQualityV1Profile()

    with pytest.raises(ValidationError, match="missing required ranking inputs"):
        profile.assign_rankings(pl.DataFrame({"run_id": ["run"], "market": ["US"], "ticker": ["AAA"]}))

    snapshot = _snapshot_rows([{"ticker": "AAA", "volatility_20d": float("nan")}])
    with pytest.raises(ValidationError, match="non-finite ranking input"):
        profile.assign_rankings(snapshot)

    non_numeric_snapshot = _snapshot_rows([{"ticker": "AAA"}]).with_columns(
        pl.lit("bad").alias("volatility_20d")
    )
    with pytest.raises(ValidationError, match="non-numeric ranking input"):
        profile.assign_rankings(non_numeric_snapshot)
