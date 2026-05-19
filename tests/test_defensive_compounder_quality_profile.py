from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.defensive_compounder_quality_v1 import (
    DEFENSIVE_COMPOUNDER_QUALITY_HORIZON_ORDER,
    DEFENSIVE_COMPOUNDER_QUALITY_PROFILE_ID,
    DEFENSIVE_COMPOUNDER_QUALITY_RANKING_METRIC_KEYS,
    DEFENSIVE_COMPOUNDER_QUALITY_SCORE_METHOD,
    DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS,
    DefensiveCompounderQualityV1Profile,
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
        high = close * 1.008
        low = close * 0.992
        if invalid_ohlc and index == len(values) - 2:
            high = low * 0.95
        if stale_tail and index >= len(values) - stale_tail:
            close = rows[-1]["close"] if rows else close
            adjusted_close = rows[-1]["adjusted_close"] if rows else adjusted_close
            high = close * 1.008
            low = close * 0.992
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


def _compounder_values(length: int = 253) -> list[float]:
    values: list[float] = []
    for index in range(length):
        value = 40.0 + index * 0.08 + math.sin(index / 9.0) * 0.35
        values.append(value)
    return values


def _broken_values(length: int = 253) -> list[float]:
    values = _compounder_values(length)
    for offset in range(90):
        values[-90 + offset] *= 1.0 - offset * 0.0045
    return values


def test_defensive_compounder_quality_profile_public_api_and_payload() -> None:
    profile = DefensiveCompounderQualityV1Profile()

    assert DEFENSIVE_COMPOUNDER_QUALITY_PROFILE_ID == "defensive_compounder_quality_v1"
    assert profile.horizon_order == ("composite", "steady_compounder", "downside_control")
    assert profile.horizon_order == DEFENSIVE_COMPOUNDER_QUALITY_HORIZON_ORDER
    assert profile.snapshot_metric_keys == DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS
    assert profile.ranking_metric_keys == DEFENSIVE_COMPOUNDER_QUALITY_RANKING_METRIC_KEYS
    assert profile.inspect_metric_keys == profile.snapshot_metric_keys
    assert "downside_volatility_120d" in profile.snapshot_metric_keys
    assert "tag_risk_flat_no_growth" in profile.ranking_metric_keys
    assert "steady_compounder_cap_score" not in profile.ranking_metric_keys
    assert "volume" not in profile.snapshot_metric_keys
    assert profile.ranking_config_payload() == {
        "ranking_profile": "defensive_compounder_quality_v1",
        "min_history_bars": 252,
        "price_floor": {"TW": 10.0, "US": 5.0},
        "liquidity_floor": {"TW": 50_000_000.0, "US": 10_000_000.0},
        "active_trading_min_days_60": {"TW": 50, "US": 55},
        "zero_volume_max_days_20": {"TW": 3, "US": 1},
        "stale_close_max_days_20": 5,
        "extreme_return_abs_cutoff": 0.80,
        "volatility_floor": 0.0001,
        "horizon_order": list(DEFENSIVE_COMPOUNDER_QUALITY_HORIZON_ORDER),
        "snapshot_metric_keys": list(DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS),
        "ranking_metric_keys": list(DEFENSIVE_COMPOUNDER_QUALITY_RANKING_METRIC_KEYS),
        "inspect_metric_keys": list(DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS),
        "stdev_ddof": 1,
        "score_method": DEFENSIVE_COMPOUNDER_QUALITY_SCORE_METHOD,
    }


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (DefensiveCompounderQualityV1Profile(profile_id="bad"), "profile_id"),
        (DefensiveCompounderQualityV1Profile(min_history_bars=251), "min_history_bars"),
        (DefensiveCompounderQualityV1Profile(horizon_order=("composite",)), "horizon order"),
        (DefensiveCompounderQualityV1Profile(snapshot_metric_keys=("profile_metrics_version",)), "snapshot metric"),
        (DefensiveCompounderQualityV1Profile(ranking_metric_keys=("score_steady_return",)), "ranking metric"),
        (DefensiveCompounderQualityV1Profile(score_method="bad"), "score_method"),
    ],
)
def test_defensive_compounder_quality_profile_rejects_contract_changes(
    profile: DefensiveCompounderQualityV1Profile, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        profile.validate()


def test_defensive_compounder_quality_builds_steady_snapshot() -> None:
    latest = date(2026, 5, 8)
    profile = DefensiveCompounderQualityV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=_bars("AAA", latest, _compounder_values()),
        run_latest_bar_date=latest,
    )

    assert snapshot.height == 1
    row = snapshot.to_dicts()[0]
    assert row["ticker"] == "AAA"
    assert row["profile_metrics_version"] == 1.0
    assert row["asof_bar_date_yyyymmdd"] == 20260508.0
    assert row["return_120d"] > 0.0
    assert row["return_252d"] > 0.0
    assert row["price_vs_sma_200d"] >= -0.05
    assert row["max_drawdown_252d"] > -0.35
    assert row["positive_21d_return_ratio_252d"] >= 0.50
    for key in profile.snapshot_metric_keys:
        assert isinstance(row[key], int | float)
        assert math.isfinite(float(row[key]))


@pytest.mark.parametrize(
    ("ticker", "values", "kwargs"),
    [
        ("SHORT", _compounder_values(251), {}),
        ("LOWPX", [value * 0.05 for value in _compounder_values()], {}),
        ("LOWLIQ", _compounder_values(), {"volume": 1_000.0}),
        ("ZEROTRADE", _compounder_values(), {"volume": 0.0}),
        ("ZEROVOL", _compounder_values(), {"zero_volume_tail": 5}),
        ("STALE", _compounder_values(), {"stale_tail": 6}),
        ("BADHILO", _compounder_values(), {"invalid_ohlc": True}),
        ("BROKEN", _broken_values(), {}),
    ],
)
def test_defensive_compounder_quality_excludes_unusable_or_broken_candidates(
    ticker: str, values: list[float], kwargs: dict[str, object]
) -> None:
    latest = date(2026, 5, 8)
    profile = DefensiveCompounderQualityV1Profile()

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
        "avg_traded_value_5d_local": 22_000_000.0,
        "median_traded_value_20d_local": 20_000_000.0,
        "traded_value_5d_to_20d_ratio": 1.1,
        "return_60d": 0.10,
        "return_120d": 0.22,
        "return_252d": 0.38,
        "positive_21d_return_ratio_252d": 0.72,
        "trend_slope_120d": 0.0014,
        "uptrend_r2_120d": 0.72,
        "volatility_20d": 0.010,
        "volatility_60d": 0.012,
        "volatility_120d": 0.013,
        "volatility_20d_to_60d_ratio": 0.83,
        "downside_volatility_60d": 0.006,
        "downside_volatility_120d": 0.007,
        "max_drawdown_120d": -0.07,
        "max_drawdown_252d": -0.12,
        "range_tightness_20d": 0.86,
        "range_tightness_60d": 0.78,
        "price_vs_sma_50d": 0.04,
        "price_vs_sma_200d": 0.25,
        "sma_50d_vs_sma_200d": 0.20,
        "liquidity_stability_score_raw": 0.88,
        "zero_volume_days_20d": 0.0,
        "active_trading_days_60d": 60.0,
        "stale_close_days_20d": 0.0,
        "data_quality_extreme_return_flag": 0.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_defensive_compounder_quality_assigns_rankings_tags_and_caps() -> None:
    profile = DefensiveCompounderQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "STEADY"},
            {
                "ticker": "LOWDOWNSIDE",
                "return_120d": 0.12,
                "return_252d": 0.20,
                "downside_volatility_120d": 0.002,
                "max_drawdown_252d": -0.05,
            },
            {"ticker": "FLAT", "return_120d": 0.01, "return_252d": 0.02},
            {
                "ticker": "BROKEN",
                "return_120d": -0.04,
                "return_252d": -0.02,
                "price_vs_sma_200d": -0.08,
                "sma_50d_vs_sma_200d": -0.03,
                "max_drawdown_252d": -0.32,
            },
            {
                "ticker": "SPIKE",
                "volatility_20d": 0.045,
                "volatility_20d_to_60d_ratio": 2.10,
            },
        ]
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.height == 15
    assert rankings["horizon"].unique().sort().to_list() == [
        "composite",
        "downside_control",
        "steady_compounder",
    ]
    for horizon in profile.horizon_order:
        rows = rankings.filter(pl.col("horizon") == horizon).sort("rank")
        assert rows["rank"].to_list() == [1, 2, 3, 4, 5]
        assert rows["score"].is_sorted(descending=True)

    composite = {
        row["ticker"]: row
        for row in rankings.filter(pl.col("horizon") == "composite").to_dicts()
    }
    assert composite["STEADY"]["tag_positive_steady_compounder"] == 1.0
    assert composite["LOWDOWNSIDE"]["tag_positive_low_downside_volatility"] == 1.0
    assert composite["LOWDOWNSIDE"]["tag_positive_drawdown_control"] == 1.0
    assert composite["FLAT"]["tag_risk_flat_no_growth"] == 1.0
    assert composite["FLAT"]["growth_cap_score"] < composite["STEADY"]["growth_cap_score"]
    assert composite["BROKEN"]["tag_risk_broken_long_trend"] == 1.0
    assert composite["BROKEN"]["trend_structure_cap_score"] < composite["STEADY"]["trend_structure_cap_score"]
    assert composite["SPIKE"]["tag_risk_volatility_spike"] == 1.0
    assert composite["SPIKE"]["volatility_cap_score"] < composite["STEADY"]["volatility_cap_score"]
    for row in rankings.to_dicts():
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_defensive_compounder_quality_caps_low_volatility_no_growth_traps() -> None:
    profile = DefensiveCompounderQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "STEADY"},
            {
                "ticker": "FLATLOWVOL",
                "return_60d": 0.004,
                "return_120d": 0.01,
                "return_252d": 0.02,
                "volatility_60d": 0.002,
                "volatility_120d": 0.002,
                "downside_volatility_60d": 0.0005,
                "downside_volatility_120d": 0.0005,
                "max_drawdown_120d": -0.01,
                "max_drawdown_252d": -0.02,
                "range_tightness_20d": 0.98,
                "range_tightness_60d": 0.98,
            },
        ]
    )

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot)
        .filter(pl.col("horizon") == "downside_control")
        .to_dicts()
    }

    assert rows["FLATLOWVOL"]["tag_risk_flat_no_growth"] == 1.0
    assert rows["FLATLOWVOL"]["growth_cap_score"] <= 0.35
    assert rows["FLATLOWVOL"]["rank"] > rows["STEADY"]["rank"]


def test_defensive_compounder_quality_caps_momentum_without_defensive_traits() -> None:
    profile = DefensiveCompounderQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "STEADY"},
            {
                "ticker": "MOMENTUM",
                "return_60d": 0.80,
                "return_120d": 1.60,
                "return_252d": 3.00,
                "positive_21d_return_ratio_252d": 0.90,
                "trend_slope_120d": 0.006,
                "uptrend_r2_120d": 0.88,
                "volatility_20d": 0.028,
                "volatility_60d": 0.032,
                "volatility_120d": 0.034,
                "downside_volatility_60d": 0.022,
                "downside_volatility_120d": 0.025,
                "max_drawdown_120d": -0.17,
                "max_drawdown_252d": -0.22,
                "range_tightness_20d": 0.25,
                "range_tightness_60d": 0.20,
                "price_vs_sma_50d": 0.42,
                "price_vs_sma_200d": 1.30,
                "sma_50d_vs_sma_200d": 0.70,
            },
        ]
    )

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot)
        .filter(pl.col("horizon") == "composite")
        .to_dicts()
    }

    assert rows["MOMENTUM"]["score_steady_return"] > rows["STEADY"]["score_steady_return"]
    assert rows["MOMENTUM"]["score_low_volatility"] < rows["STEADY"]["score_low_volatility"]
    assert rows["MOMENTUM"]["score_downside_control"] < rows["STEADY"]["score_downside_control"]
    assert rows["MOMENTUM"]["volatility_cap_score"] <= 0.60
    assert rows["MOMENTUM"]["rank"] > rows["STEADY"]["rank"]


def test_defensive_compounder_quality_caps_low_volatility_lagging_growth_leaders() -> None:
    profile = DefensiveCompounderQualityV1Profile()

    cap = profile._volatility_cap_score(
        volatility_20d=0.031,
        volatility_20d_to_60d_ratio=1.10,
        score_low_volatility=0.39,
        score_downside_control=0.58,
    )

    assert cap <= 0.55


def test_defensive_compounder_quality_applies_hard_caps_to_defensive_risks() -> None:
    profile = DefensiveCompounderQualityV1Profile()

    volatility_cap = profile._volatility_cap_score(
        volatility_20d=0.045,
        volatility_20d_to_60d_ratio=1.20,
        score_low_volatility=0.70,
        score_downside_control=0.70,
    )
    drawdown_cap = profile._drawdown_cap_score(max_drawdown_120d=-0.21, max_drawdown_252d=-0.22)
    liquidity_cap = profile._liquidity_cap_score(
        stale_close_days_20d=1.0,
        zero_volume_days_20d=0.0,
        traded_value_5d_to_20d_ratio=1.00,
    )

    assert volatility_cap <= 0.40
    assert drawdown_cap <= 0.40
    assert liquidity_cap <= 0.35


def test_defensive_compounder_quality_composite_requires_steady_compounder_quality() -> None:
    profile = DefensiveCompounderQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "STEADY"},
            {
                "ticker": "DOWNSIDEONLY",
                "return_60d": 0.02,
                "return_120d": 0.08,
                "return_252d": 0.12,
                "positive_21d_return_ratio_252d": 0.52,
                "trend_slope_120d": 0.0001,
                "uptrend_r2_120d": 0.10,
                "volatility_20d": 0.004,
                "volatility_60d": 0.004,
                "volatility_120d": 0.004,
                "downside_volatility_60d": 0.0005,
                "downside_volatility_120d": 0.0005,
                "max_drawdown_120d": -0.02,
                "max_drawdown_252d": -0.03,
                "range_tightness_20d": 0.98,
                "range_tightness_60d": 0.98,
                "price_vs_sma_200d": 0.08,
                "sma_50d_vs_sma_200d": 0.04,
            },
        ]
    )

    by_horizon = {
        horizon: {
            row["ticker"]: row
            for row in profile.assign_rankings(snapshot)
            .filter(pl.col("horizon") == horizon)
            .to_dicts()
        }
        for horizon in profile.horizon_order
    }

    assert by_horizon["composite"]["DOWNSIDEONLY"]["tag_positive_steady_compounder"] == 0.0
    assert by_horizon["composite"]["DOWNSIDEONLY"]["score"] <= 0.45
    assert "steady_compounder_cap_score" not in by_horizon["composite"]["DOWNSIDEONLY"]
    assert by_horizon["composite"]["DOWNSIDEONLY"]["rank"] > by_horizon["composite"]["STEADY"]["rank"]
    assert by_horizon["downside_control"]["DOWNSIDEONLY"]["rank"] < by_horizon["downside_control"]["STEADY"]["rank"]


def test_defensive_compounder_quality_preserves_order_inside_composite_caps() -> None:
    profile = DefensiveCompounderQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "STEADY"},
            {
                "ticker": "MOMA",
                "return_60d": 0.80,
                "return_120d": 1.60,
                "return_252d": 3.00,
                "positive_21d_return_ratio_252d": 0.90,
                "trend_slope_120d": 0.006,
                "uptrend_r2_120d": 0.88,
                "volatility_20d": 0.028,
                "volatility_60d": 0.032,
                "volatility_120d": 0.034,
                "downside_volatility_60d": 0.022,
                "downside_volatility_120d": 0.025,
                "max_drawdown_120d": -0.17,
                "max_drawdown_252d": -0.22,
                "range_tightness_20d": 0.25,
                "range_tightness_60d": 0.20,
                "price_vs_sma_50d": 0.42,
                "price_vs_sma_200d": 1.30,
                "sma_50d_vs_sma_200d": 0.70,
            },
            {
                "ticker": "MOMB",
                "return_60d": 0.70,
                "return_120d": 1.40,
                "return_252d": 2.50,
                "positive_21d_return_ratio_252d": 0.86,
                "trend_slope_120d": 0.005,
                "uptrend_r2_120d": 0.82,
                "volatility_20d": 0.029,
                "volatility_60d": 0.033,
                "volatility_120d": 0.035,
                "downside_volatility_60d": 0.023,
                "downside_volatility_120d": 0.026,
                "max_drawdown_120d": -0.17,
                "max_drawdown_252d": -0.22,
                "range_tightness_20d": 0.25,
                "range_tightness_60d": 0.20,
                "price_vs_sma_50d": 0.42,
                "price_vs_sma_200d": 1.10,
                "sma_50d_vs_sma_200d": 0.60,
            },
        ]
    )

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot)
        .filter(pl.col("horizon") == "composite")
        .to_dicts()
    }

    assert rows["MOMA"]["tag_positive_steady_compounder"] == 0.0
    assert rows["MOMB"]["tag_positive_steady_compounder"] == 0.0
    assert rows["MOMA"]["score"] <= 0.45
    assert rows["MOMB"]["score"] <= 0.45
    assert rows["MOMA"]["score"] > rows["MOMB"]["score"]
    assert rows["MOMA"]["rank"] < rows["MOMB"]["rank"]


def test_defensive_compounder_quality_caps_stale_names_in_composite() -> None:
    profile = DefensiveCompounderQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "STEADY"},
            {
                "ticker": "STALE",
                "return_60d": 0.30,
                "return_120d": 0.55,
                "return_252d": 0.90,
                "positive_21d_return_ratio_252d": 0.85,
                "trend_slope_120d": 0.003,
                "uptrend_r2_120d": 0.80,
                "max_drawdown_120d": -0.08,
                "max_drawdown_252d": -0.12,
                "range_tightness_20d": 0.70,
                "range_tightness_60d": 0.68,
                "stale_close_days_20d": 1.0,
            },
        ]
    )

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot)
        .filter(pl.col("horizon") == "composite")
        .to_dicts()
    }

    assert rows["STALE"]["tag_risk_stale_or_illiquid"] == 1.0
    assert rows["STALE"]["tag_positive_steady_compounder"] == 0.0
    assert "steady_compounder_cap_score" not in rows["STALE"]
    assert rows["STALE"]["liquidity_cap_score"] <= 0.45
    assert rows["STALE"]["rank"] > rows["STEADY"]["rank"]


def test_defensive_compounder_quality_rejects_malformed_ranking_snapshot() -> None:
    profile = DefensiveCompounderQualityV1Profile()
    malformed = _ranking_snapshot([{"ticker": "AAA"}]).drop("return_120d")

    with pytest.raises(ValidationError, match="missing required ranking inputs"):
        profile.assign_rankings(malformed)
