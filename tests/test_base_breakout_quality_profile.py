from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.base_breakout_quality_v1 import (
    BASE_BREAKOUT_QUALITY_HORIZON_ORDER,
    BASE_BREAKOUT_QUALITY_PROFILE_ID,
    BASE_BREAKOUT_QUALITY_RANKING_METRIC_KEYS,
    BASE_BREAKOUT_QUALITY_SCORE_METHOD,
    BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS,
    BaseBreakoutQualityV1Profile,
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


def _constructive_base_values(length: int = 253) -> list[float]:
    values: list[float] = []
    for index in range(length):
        if index < length - 90:
            value = 45.0 + index * 0.20
        else:
            base_index = index - (length - 90)
            value = 77.0 + base_index * 0.035 + math.sin(base_index / 4.0) * 1.25
        values.append(value)
    values[-2] = max(values[-120:-2]) * 0.985
    values[-1] = values[-2] * 1.004
    return values


def _broken_base_values(length: int = 253) -> list[float]:
    values = _constructive_base_values(length)
    for offset in range(45):
        values[-45 + offset] *= 1.0 - offset * 0.006
    return values


def _confirmed_breakout_values(length: int = 253) -> list[float]:
    values = _constructive_base_values(length)
    prior_high = max(values[-121:-2])
    values[-2] = prior_high * 1.018
    values[-1] = values[-2] * 1.001
    return values


def _outer_readiness_values(length: int = 253) -> list[float]:
    values = _constructive_base_values(length)
    prior_high = max(values[-121:-2])
    values[-122] = prior_high * 0.74
    values[-121] = prior_high * 0.74
    for offset in range(120):
        if offset < 30:
            ratio = 0.74 + offset * (0.26 / 29.0)
        elif offset < 60:
            ratio = 0.94 - (offset - 30) * (0.14 / 29.0)
        else:
            ratio = 0.78 + (offset - 60) * (0.10 / 59.0) + math.sin(offset / 5.0) * 0.002
        values[-120 + offset] = prior_high * ratio
    values[-2] = prior_high * 0.880
    values[-1] = prior_high * 0.881
    return values


def test_base_breakout_quality_profile_public_api_and_payload() -> None:
    profile = BaseBreakoutQualityV1Profile()

    assert BASE_BREAKOUT_QUALITY_PROFILE_ID == "base_breakout_quality_v1"
    assert profile.horizon_order == ("composite", "near_breakout", "breakout_readiness")
    assert profile.horizon_order == BASE_BREAKOUT_QUALITY_HORIZON_ORDER
    assert profile.snapshot_metric_keys == BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS
    assert profile.ranking_metric_keys == BASE_BREAKOUT_QUALITY_RANKING_METRIC_KEYS
    assert profile.inspect_metric_keys == profile.snapshot_metric_keys
    assert "pct_below_120d_high" in profile.snapshot_metric_keys
    assert "tag_setup_valid_base" in profile.ranking_metric_keys
    assert "volume" not in profile.snapshot_metric_keys
    assert profile.ranking_config_payload() == {
        "ranking_profile": "base_breakout_quality_v1",
        "min_history_bars": 252,
        "price_floor": {"TW": 10.0, "US": 5.0},
        "liquidity_floor": {"TW": 50_000_000.0, "US": 10_000_000.0},
        "active_trading_min_days_60": {"TW": 50, "US": 55},
        "zero_volume_max_days_20": {"TW": 3, "US": 1},
        "stale_close_max_days_20": 5,
        "extreme_return_abs_cutoff": 0.80,
        "volatility_floor": 0.0001,
        "horizon_order": list(BASE_BREAKOUT_QUALITY_HORIZON_ORDER),
        "snapshot_metric_keys": list(BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS),
        "ranking_metric_keys": list(BASE_BREAKOUT_QUALITY_RANKING_METRIC_KEYS),
        "inspect_metric_keys": list(BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS),
        "stdev_ddof": 1,
        "score_method": BASE_BREAKOUT_QUALITY_SCORE_METHOD,
    }


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (BaseBreakoutQualityV1Profile(profile_id="bad"), "profile_id"),
        (BaseBreakoutQualityV1Profile(min_history_bars=251), "min_history_bars"),
        (BaseBreakoutQualityV1Profile(horizon_order=("composite",)), "horizon order"),
        (BaseBreakoutQualityV1Profile(snapshot_metric_keys=("profile_metrics_version",)), "snapshot metric"),
        (BaseBreakoutQualityV1Profile(ranking_metric_keys=("score_base_tightness",)), "ranking metric"),
        (BaseBreakoutQualityV1Profile(score_method="bad"), "score_method"),
    ],
)
def test_base_breakout_quality_profile_rejects_contract_changes(
    profile: BaseBreakoutQualityV1Profile, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        profile.validate()


def test_base_breakout_quality_builds_constructive_base_snapshot() -> None:
    latest = date(2026, 5, 8)
    profile = BaseBreakoutQualityV1Profile()
    bars = _bars("AAA", latest, _constructive_base_values(), volume=1_000_000.0)

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
    assert row["asof_bar_date_yyyymmdd"] == 20260508.0
    assert row["pct_below_120d_high"] >= -0.08
    assert 0.0 <= row["base_tightness_20d"] <= 1.0
    assert row["base_depth_120d"] <= 0.0
    assert row["price_vs_sma_200d"] > 0.0
    assert row["sma_50d_vs_sma_200d"] >= 0.0
    assert row["avg_traded_value_20d_local"] >= profile.liquidity_floor[Market.US]
    for key in profile.snapshot_metric_keys:
        assert isinstance(row[key], int | float)
        assert math.isfinite(float(row[key]))


def test_base_breakout_quality_build_snapshot_uses_aligned_outer_readiness_window() -> None:
    latest = date(2026, 5, 8)
    profile = BaseBreakoutQualityV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("OUTER")],
        bars=_bars("OUTER", latest, _outer_readiness_values(), volume=1_000_000.0),
        run_latest_bar_date=latest,
    )

    assert snapshot.height == 1
    row = snapshot.to_dicts()[0]
    assert row["ticker"] == "OUTER"
    assert row["asof_bar_date_yyyymmdd"] == 20260508.0
    assert -0.15 <= row["pct_below_120d_high"] < -0.08


def test_base_breakout_quality_measures_breakout_against_prior_high() -> None:
    latest = date(2026, 5, 8)
    profile = BaseBreakoutQualityV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=_bars("AAA", latest, _confirmed_breakout_values()),
        run_latest_bar_date=latest,
    )

    assert snapshot.height == 1
    row = snapshot.to_dicts()[0]
    assert row["pct_below_120d_high"] > 0.0
    assert row["pct_below_60d_high"] > 0.0

    confirmed = (
        profile.assign_rankings(snapshot)
        .filter((pl.col("horizon") == "breakout_readiness") & (pl.col("ticker") == "AAA"))
        .to_dicts()[0]
    )
    assert confirmed["tag_setup_valid_base"] == 1.0
    assert confirmed["tag_setup_confirmed_breakout"] == 1.0


@pytest.mark.parametrize(
    ("ticker", "values", "kwargs"),
    [
        ("SHORT", _constructive_base_values(251), {}),
        ("LOWPX", [value * 0.05 for value in _constructive_base_values()], {}),
        ("LOWLIQ", _constructive_base_values(), {"volume": 1_000.0}),
        ("ZEROTRADE", _constructive_base_values(), {"volume": 0.0}),
        ("ZEROVOL", _constructive_base_values(), {"zero_volume_tail": 5}),
        ("STALE", _constructive_base_values(), {"stale_tail": 6}),
        ("BADHILO", _constructive_base_values(), {"invalid_ohlc": True}),
        ("BROKEN", _broken_base_values(), {}),
    ],
)
def test_base_breakout_quality_excludes_unusable_or_broken_candidates(
    ticker: str, values: list[float], kwargs: dict[str, object]
) -> None:
    latest = date(2026, 5, 8)
    profile = BaseBreakoutQualityV1Profile()

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
        "return_20d": 0.04,
        "return_60d": 0.18,
        "return_120d": 0.30,
        "volatility_20d": 0.015,
        "volatility_60d": 0.020,
        "volatility_20d_to_60d_ratio": 0.75,
        "trend_slope_60d": 0.002,
        "uptrend_r2_60d": 0.65,
        "trend_consistency_60d": 0.62,
        "sma_20d": 99.0,
        "sma_50d": 95.0,
        "sma_200d": 80.0,
        "price_vs_sma_20d": 0.01,
        "price_vs_sma_50d": 0.05,
        "price_vs_sma_200d": 0.25,
        "sma_20d_vs_sma_50d": 0.04,
        "sma_50d_vs_sma_200d": 0.19,
        "pct_below_60d_high": -0.02,
        "pct_below_120d_high": -0.03,
        "base_depth_60d": -0.10,
        "base_depth_120d": -0.14,
        "base_tightness_20d": 0.82,
        "base_tightness_60d": 0.70,
        "close_position_20d_range": 0.72,
        "zero_volume_days_20d": 0.0,
        "active_trading_days_60d": 60.0,
        "stale_close_days_20d": 0.0,
        "data_quality_extreme_return_flag": 0.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_base_breakout_quality_assigns_rankings_tags_and_caps() -> None:
    profile = BaseBreakoutQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "AAA"},
            {
                "ticker": "CONFIRM",
                "pct_below_120d_high": 0.02,
                "pct_below_60d_high": 0.02,
                "return_20d": 0.12,
                "close_position_20d_range": 0.90,
                "traded_value_5d_to_20d_ratio": 1.50,
            },
            {
                "ticker": "EXTENDED",
                "pct_below_120d_high": 0.10,
                "pct_below_60d_high": 0.10,
                "return_20d": 0.42,
                "price_vs_sma_50d": 0.28,
                "base_depth_120d": -0.08,
            },
            {
                "ticker": "WEAKBASE",
                "base_depth_120d": -0.32,
                "base_depth_60d": -0.26,
                "base_tightness_20d": 0.20,
                "base_tightness_60d": 0.18,
                "volatility_20d_to_60d_ratio": 1.80,
            },
        ]
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.height == 12
    assert rankings["horizon"].unique().sort().to_list() == [
        "breakout_readiness",
        "composite",
        "near_breakout",
    ]
    for horizon in profile.horizon_order:
        rows = rankings.filter(pl.col("horizon") == horizon).sort("rank")
        assert rows["rank"].to_list() == [1, 2, 3, 4]
        assert rows["score"].is_sorted(descending=True)

    composite = {
        row["ticker"]: row
        for row in rankings.filter(pl.col("horizon") == "composite").to_dicts()
    }
    assert composite["AAA"]["tag_setup_valid_base"] == 1.0
    assert composite["AAA"]["tag_setup_near_breakout"] == 1.0
    assert composite["CONFIRM"]["tag_setup_confirmed_breakout"] == 1.0
    assert composite["EXTENDED"]["tag_risk_overextended_breakout"] == 1.0
    assert composite["EXTENDED"]["breakout_extension_cap_score"] < composite["AAA"]["breakout_extension_cap_score"]
    assert composite["WEAKBASE"]["tag_risk_weak_base"] == 1.0
    assert composite["WEAKBASE"]["base_depth_cap_score"] < composite["AAA"]["base_depth_cap_score"]
    for row in rankings.to_dicts():
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_base_breakout_quality_scores_proximity_inside_setup_window_only() -> None:
    profile = BaseBreakoutQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "INNER", "pct_below_120d_high": -0.03},
            {"ticker": "OUTER", "pct_below_120d_high": -0.12, "pct_below_60d_high": -0.12},
            {"ticker": "ABOVE", "pct_below_120d_high": 0.09, "pct_below_60d_high": 0.09},
        ]
    )

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot)
        .filter(pl.col("horizon") == "near_breakout")
        .to_dicts()
    }

    assert rows["OUTER"]["tag_setup_near_breakout"] == 1.0
    assert rows["OUTER"]["score_breakout_proximity"] > 0.0
    assert rows["ABOVE"]["tag_setup_near_breakout"] == 0.0
    assert rows["ABOVE"]["score_breakout_proximity"] == 0.0


def test_base_breakout_quality_rejects_weak_base_at_candidate_gate() -> None:
    profile = BaseBreakoutQualityV1Profile()

    assert not profile._is_persisted_setup(
        base_depth_120d=-0.32,
        base_depth_60d=-0.26,
        base_tightness_20d=0.20,
        base_tightness_60d=0.18,
        trend_slope_60d=0.002,
        price_vs_sma_200d=0.30,
        sma_50d_vs_sma_200d=0.20,
        return_60d=0.28,
        return_120d=0.45,
        pct_below_120d_high=-0.02,
        close_position_20d_range=0.80,
        traded_value_5d_to_20d_ratio=1.20,
    )

    assert profile._is_persisted_setup(
        base_depth_120d=-0.18,
        base_depth_60d=-0.10,
        base_tightness_20d=0.65,
        base_tightness_60d=0.55,
        trend_slope_60d=0.002,
        price_vs_sma_200d=0.16,
        sma_50d_vs_sma_200d=0.10,
        return_60d=0.08,
        return_120d=0.22,
        pct_below_120d_high=-0.02,
        close_position_20d_range=0.70,
        traded_value_5d_to_20d_ratio=1.10,
    )

    assert not profile._is_persisted_setup(
        base_depth_120d=-0.18,
        base_depth_60d=-0.10,
        base_tightness_20d=0.65,
        base_tightness_60d=0.55,
        trend_slope_60d=0.002,
        price_vs_sma_200d=0.16,
        sma_50d_vs_sma_200d=0.10,
        return_60d=0.08,
        return_120d=0.22,
        pct_below_120d_high=-0.16,
        close_position_20d_range=0.70,
        traded_value_5d_to_20d_ratio=1.10,
    )

    assert not profile._is_persisted_setup(
        base_depth_120d=-0.18,
        base_depth_60d=-0.10,
        base_tightness_20d=0.65,
        base_tightness_60d=0.55,
        trend_slope_60d=0.002,
        price_vs_sma_200d=0.16,
        sma_50d_vs_sma_200d=0.10,
        return_60d=0.08,
        return_120d=0.22,
        pct_below_120d_high=0.09,
        close_position_20d_range=0.92,
        traded_value_5d_to_20d_ratio=1.10,
    )


def test_base_breakout_quality_does_not_confirm_no_base_breakouts() -> None:
    profile = BaseBreakoutQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "AAA"},
            {
                "ticker": "NOBASE",
                "pct_below_120d_high": 0.02,
                "pct_below_60d_high": 0.02,
                "base_depth_120d": 0.0,
                "base_depth_60d": 0.0,
                "base_tightness_20d": 0.98,
                "base_tightness_60d": 0.97,
                "close_position_20d_range": 0.95,
                "traded_value_5d_to_20d_ratio": 1.40,
            },
        ]
    )

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot)
        .filter(pl.col("horizon") == "breakout_readiness")
        .to_dicts()
    }

    assert rows["NOBASE"]["tag_setup_valid_base"] == 0.0
    assert rows["NOBASE"]["tag_setup_confirmed_breakout"] == 0.0
    assert rows["NOBASE"]["tag_risk_weak_base"] == 1.0
    assert rows["NOBASE"]["base_depth_cap_score"] < rows["AAA"]["base_depth_cap_score"]
    assert rows["NOBASE"]["rank"] > rows["AAA"]["rank"]


def test_base_breakout_quality_uses_prior_strength_in_rankings() -> None:
    profile = BaseBreakoutQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "LEADER", "return_60d": 0.42, "return_120d": 0.68},
            {
                "ticker": "LAGGING",
                "return_60d": 0.04,
                "return_120d": 0.12,
                "price_vs_sma_200d": 0.09,
            },
        ]
    )

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot)
        .filter(pl.col("horizon") == "composite")
        .to_dicts()
    }

    assert rows["LEADER"]["score_prior_strength"] > rows["LAGGING"]["score_prior_strength"]
    assert rows["LEADER"]["prior_strength_score"] > rows["LAGGING"]["prior_strength_score"]
    assert rows["LEADER"]["rank"] < rows["LAGGING"]["rank"]


def test_base_breakout_quality_caps_weak_prior_strength_drift() -> None:
    profile = BaseBreakoutQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {
                "ticker": "LEADER",
                "return_60d": 0.24,
                "return_120d": 0.45,
                "price_vs_sma_200d": 0.30,
                "sma_50d_vs_sma_200d": 0.22,
                "trend_slope_60d": 0.003,
                "uptrend_r2_60d": 0.72,
                "trend_consistency_60d": 0.66,
            },
            {
                "ticker": "DRIFT",
                "return_20d": 0.03,
                "return_60d": 0.05,
                "return_120d": 0.06,
                "price_vs_sma_200d": 0.04,
                "sma_50d_vs_sma_200d": 0.02,
                "trend_slope_60d": 0.0005,
                "uptrend_r2_60d": 0.18,
                "trend_consistency_60d": 0.50,
                "pct_below_120d_high": -0.01,
                "pct_below_60d_high": -0.01,
                "base_depth_120d": -0.12,
                "base_depth_60d": -0.06,
                "base_tightness_20d": 0.90,
                "base_tightness_60d": 0.82,
                "close_position_20d_range": 0.86,
                "traded_value_5d_to_20d_ratio": 1.30,
            },
        ]
    )

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot)
        .filter(pl.col("horizon") == "composite")
        .to_dicts()
    }

    assert rows["LEADER"]["rank"] == 1
    assert rows["DRIFT"]["tag_setup_valid_base"] == 0.0
    assert rows["DRIFT"]["tag_risk_weak_base"] == 1.0
    assert rows["DRIFT"]["structure_cap_score"] <= 0.55
    assert rows["DRIFT"]["rank"] > rows["LEADER"]["rank"]


def test_base_breakout_quality_allows_mature_base_after_short_pause() -> None:
    profile = BaseBreakoutQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {
                "ticker": "PAUSE",
                "return_20d": 0.02,
                "return_60d": 0.015,
                "return_120d": 0.18,
                "price_vs_sma_200d": 0.10,
                "sma_50d_vs_sma_200d": 0.06,
                "trend_slope_60d": 0.0005,
                "uptrend_r2_60d": 0.05,
                "trend_consistency_60d": 0.53,
                "pct_below_120d_high": -0.04,
                "pct_below_60d_high": -0.04,
                "base_depth_120d": -0.20,
                "base_depth_60d": -0.12,
                "base_tightness_20d": 0.63,
                "base_tightness_60d": 0.59,
                "close_position_20d_range": 0.42,
                "traded_value_5d_to_20d_ratio": 1.10,
            },
        ]
    )

    row = (
        profile.assign_rankings(snapshot)
        .filter(pl.col("horizon") == "composite")
        .to_dicts()[0]
    )

    assert row["tag_setup_valid_base"] == 1.0
    assert row["tag_risk_weak_base"] == 0.0
    assert row["structure_cap_score"] > 0.55


def test_base_breakout_quality_rejects_malformed_ranking_snapshot() -> None:
    profile = BaseBreakoutQualityV1Profile()
    malformed = _ranking_snapshot([{"ticker": "AAA"}]).drop("pct_below_120d_high")

    with pytest.raises(ValidationError, match="missing required ranking inputs"):
        profile.assign_rankings(malformed)
