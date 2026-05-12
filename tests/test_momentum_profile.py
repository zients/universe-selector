from __future__ import annotations

import polars as pl

from universe_selector.domain import Market
from universe_selector.providers.fixture import FixtureProvider
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.momentum_v1 import MomentumV1Profile


def test_momentum_profile_builds_snapshot_from_fixture_provider(fixture_dir) -> None:
    run_data = FixtureProvider(fixture_dir).load_run_data("sample-run", Market.US)
    profile = MomentumV1Profile()

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
        "momentum_return_12_1",
        "momentum_return_6_1",
        "volatility_12_1",
        "volatility_6_1",
        "risk_adjusted_momentum_12_1",
        "risk_adjusted_momentum_6_1",
        "short_term_strength_20d",
    ]
    assert set(snapshot["ticker"].to_list()) <= {item.ticker for item in run_data.listings}
    assert snapshot.select(pl.exclude("run_id", "market", "ticker")).null_count().sum_horizontal().item() == 0


def test_momentum_profile_ignores_listings_from_other_markets(fixture_dir) -> None:
    run_data = FixtureProvider(fixture_dir).load_run_data("sample-run", Market.US)
    profile = MomentumV1Profile()
    cross_market_listing = ListingCandidate(
        market=Market.TW,
        ticker="AAA",
        listing_symbol="AAA",
        exchange_segment="TEST",
        listing_status="active",
        instrument_type="common_stock",
        source_id="test:cross-market-aaa",
    )

    snapshot = profile.build_snapshot(
        run_id="sample-run",
        market=Market.US,
        listings=[cross_market_listing],
        bars=run_data.bars,
        run_latest_bar_date=run_data.metadata.run_latest_bar_date,
    )

    assert snapshot.is_empty()


def test_momentum_profile_assigns_swing_and_midterm_rankings(fixture_dir) -> None:
    run_data = FixtureProvider(fixture_dir).load_run_data("sample-run", Market.US)
    profile = MomentumV1Profile()
    snapshot = profile.build_snapshot(
        run_id="sample-run",
        market=Market.US,
        listings=run_data.listings,
        bars=run_data.bars,
        run_latest_bar_date=run_data.metadata.run_latest_bar_date,
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.height == snapshot.height * 2
    assert rankings["horizon"].unique().sort().to_list() == ["midterm", "swing"]
    for horizon in profile.horizon_order:
        horizon_rankings = rankings.filter(pl.col("horizon") == horizon).sort("rank")
        assert horizon_rankings["rank"].to_list() == list(range(1, snapshot.height + 1))
        assert horizon_rankings["score"].is_sorted(descending=True)


def test_momentum_profile_uses_raw_weighted_scores() -> None:
    profile = MomentumV1Profile()
    snapshot = pl.DataFrame(
        {
            "run_id": ["run", "run"],
            "market": ["US", "US"],
            "ticker": ["AAA", "BBB"],
            "close": [100.0, 100.0],
            "adjusted_close": [100.0, 100.0],
            "avg_traded_value_20d_local": [10_000_000.0, 10_000_000.0],
            "momentum_return_12_1": [0.30, 0.20],
            "momentum_return_6_1": [0.10, 0.15],
            "volatility_12_1": [0.05, 0.05],
            "volatility_6_1": [0.05, 0.01],
            "risk_adjusted_momentum_12_1": [6.0, 3.0],
            "risk_adjusted_momentum_6_1": [2.0, 4.0],
            "short_term_strength_20d": [0.10, 0.02],
        }
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.columns == [
        "run_id",
        "market",
        "horizon",
        "ticker",
        "score_risk_adjusted_momentum_12_1",
        "score_risk_adjusted_momentum_6_1",
        "score_short_term_strength_20d",
        "score",
        "rank",
    ]
    aaa_midterm = rankings.filter((pl.col("horizon") == "midterm") & (pl.col("ticker") == "AAA")).to_dicts()[0]
    bbb_swing = rankings.filter((pl.col("horizon") == "swing") & (pl.col("ticker") == "BBB")).to_dicts()[0]

    assert aaa_midterm["score_risk_adjusted_momentum_12_1"] == 6.0
    assert aaa_midterm["score_risk_adjusted_momentum_6_1"] == 2.0
    assert aaa_midterm["score_short_term_strength_20d"] == 2.0
    assert aaa_midterm["score"] == 4.0
    assert aaa_midterm["rank"] == 1
    assert bbb_swing["score_short_term_strength_20d"] == 2.0
    assert bbb_swing["score"] == 3.3
    assert bbb_swing["rank"] == 1


def test_momentum_profile_assign_rankings_returns_empty_raw_score_schema_for_empty_snapshot() -> None:
    profile = MomentumV1Profile()

    rankings = profile.assign_rankings(pl.DataFrame())

    assert rankings.is_empty()
    assert rankings.columns == [
        "run_id",
        "market",
        "horizon",
        "ticker",
        "score_risk_adjusted_momentum_12_1",
        "score_risk_adjusted_momentum_6_1",
        "score_short_term_strength_20d",
        "score",
        "rank",
    ]


def test_momentum_profile_filters_non_finite_raw_score_inputs() -> None:
    profile = MomentumV1Profile()
    snapshot = pl.DataFrame(
        {
            "run_id": ["run", "run", "run", "run"],
            "market": ["US", "US", "US", "US"],
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "close": [100.0, 100.0, 100.0, 100.0],
            "adjusted_close": [100.0, 100.0, 100.0, 100.0],
            "avg_traded_value_20d_local": [10_000_000.0, 10_000_000.0, 10_000_000.0, 10_000_000.0],
            "momentum_return_12_1": [0.30, 0.20, 0.10, 0.10],
            "momentum_return_6_1": [0.10, 0.15, 0.05, 0.05],
            "volatility_12_1": [0.05, 0.05, 0.05, 0.05],
            "volatility_6_1": [0.05, 0.0, float("nan"), 0.0001],
            "risk_adjusted_momentum_12_1": [6.0, 3.0, float("inf"), 5.0],
            "risk_adjusted_momentum_6_1": [2.0, 4.0, 5.0, 4.0],
            "short_term_strength_20d": [0.10, 0.02, 0.01, 0.10],
        }
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings["ticker"].unique().to_list() == ["AAA"]
    assert rankings["score"].is_finite().all()
    assert rankings["score_short_term_strength_20d"].is_finite().all()
