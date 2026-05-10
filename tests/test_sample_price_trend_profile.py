from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from universe_selector.domain import Market
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.sample_price_trend_v1 import SamplePriceTrendV1Profile


def _listing(ticker: str, market: Market = Market.US) -> ListingCandidate:
    return ListingCandidate(
        market=market,
        ticker=ticker,
        listing_symbol=ticker,
        exchange_segment="TEST",
        listing_status="active",
        instrument_type="common_stock",
        source_id=f"test:{ticker}",
    )


def _bars(ticker: str, latest: date, *, start_price: float, end_price: float, volume: int = 2_000_000) -> pl.DataFrame:
    rows = []
    for index in range(121):
        ratio = index / 120
        close = start_price + (end_price - start_price) * ratio
        rows.append(
            {
                "market": "US",
                "ticker": ticker,
                "bar_date": latest - timedelta(days=120 - index),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adjusted_close": close,
                "volume": volume,
            }
        )
    return pl.DataFrame(rows)


def test_sample_profile_builds_snapshot_from_fixture_provider(fixture_dir) -> None:
    from universe_selector.providers.fixture import FixtureProvider

    run_data = FixtureProvider(fixture_dir).load_run_data("sample-run", Market.US)
    profile = SamplePriceTrendV1Profile()

    snapshot = profile.build_snapshot(
        run_id="sample-run",
        market=Market.US,
        listings=run_data.listings,
        bars=run_data.bars,
        run_latest_bar_date=run_data.metadata.run_latest_bar_date,
    )

    assert not snapshot.is_empty()
    assert snapshot.columns == [
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        "avg_traded_value_20d_local",
        "return_60d",
        "return_120d",
    ]
    assert set(snapshot["ticker"].to_list()) <= {item.ticker for item in run_data.listings}
    assert snapshot["return_60d"].null_count() == 0
    assert snapshot["return_120d"].null_count() == 0


def test_sample_profile_assigns_rankings_per_run_and_market() -> None:
    profile = SamplePriceTrendV1Profile()
    snapshot = pl.DataFrame(
        {
            "run_id": ["run-1", "run-1", "run-2"],
            "market": ["US", "US", "US"],
            "ticker": ["AAA", "BBB", "CCC"],
            "close": [20.0, 20.0, 20.0],
            "adjusted_close": [20.0, 20.0, 20.0],
            "avg_traded_value_20d_local": [20_000_000.0, 20_000_000.0, 20_000_000.0],
            "return_60d": [0.20, 0.10, 0.50],
            "return_120d": [0.10, 0.30, 0.40],
        }
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.columns == [
        "run_id",
        "market",
        "horizon",
        "ticker",
        "score_return_60d",
        "score_return_120d",
        "score",
        "rank",
    ]
    assert rankings.filter(pl.col("run_id") == "run-1").height == 4
    midterm = rankings.filter((pl.col("run_id") == "run-1") & (pl.col("horizon") == "midterm")).sort("rank")
    longterm = rankings.filter((pl.col("run_id") == "run-1") & (pl.col("horizon") == "longterm")).sort("rank")
    assert midterm["ticker"].to_list() == ["AAA", "BBB"]
    assert longterm["ticker"].to_list() == ["BBB", "AAA"]
    assert midterm["score_return_60d"].to_list() == [0.20, 0.10]
    assert midterm["score_return_120d"].to_list() == [0.10, 0.30]
    assert longterm["score_return_60d"].to_list() == [0.10, 0.20]
    assert longterm["score_return_120d"].to_list() == [0.30, 0.10]
    assert midterm["score"].to_list() == midterm["score_return_60d"].to_list()
    assert longterm["score"].to_list() == longterm["score_return_120d"].to_list()
    assert rankings.filter(pl.col("run_id") == "run-2")["rank"].to_list() == [1, 1]


def test_sample_profile_filters_insufficient_or_invalid_rows() -> None:
    latest = date(2026, 4, 24)
    profile = SamplePriceTrendV1Profile()
    bars = pl.concat(
        [
            _bars("PASS", latest, start_price=10.0, end_price=20.0),
            _bars("LOWPRICE", latest, start_price=1.0, end_price=4.0),
            _bars("LOWVALUE", latest, start_price=10.0, end_price=20.0, volume=1),
        ]
    )

    snapshot = profile.build_snapshot(
        run_id="sample-run",
        market=Market.US,
        listings=[_listing("PASS"), _listing("LOWPRICE"), _listing("LOWVALUE")],
        bars=bars,
        run_latest_bar_date=latest,
    )

    assert snapshot["ticker"].to_list() == ["PASS"]


def test_sample_profile_rejects_non_finite_required_returns() -> None:
    profile = SamplePriceTrendV1Profile()
    snapshot = pl.DataFrame(
        {
            "run_id": ["run-1"],
            "market": ["US"],
            "ticker": ["AAA"],
            "close": [20.0],
            "adjusted_close": [20.0],
            "avg_traded_value_20d_local": [20_000_000.0],
            "return_60d": [float("nan")],
            "return_120d": [0.10],
        }
    )

    assert profile.assign_rankings(snapshot).is_empty()
