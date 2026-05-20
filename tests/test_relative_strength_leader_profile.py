from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.relative_strength_leader_v1 import (
    RELATIVE_STRENGTH_LEADER_HORIZON_ORDER,
    RELATIVE_STRENGTH_LEADER_PROFILE_ID,
    RELATIVE_STRENGTH_LEADER_RANKING_METRIC_KEYS,
    RELATIVE_STRENGTH_LEADER_SCORE_METHOD,
    RELATIVE_STRENGTH_LEADER_SNAPSHOT_METRIC_KEYS,
    RelativeStrengthLeaderV1Profile,
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


def _leader_values(length: int = 275) -> list[float]:
    values: list[float] = []
    for index in range(length):
        base = 45.0 + index * 0.18
        wave = math.sin(index / 7.0) * 0.55
        values.append(base + wave)
    values[-2] = max(values[-121:-2]) * 1.015
    values[-1] = values[-2] * 1.001
    return values


def _broken_values(length: int = 275) -> list[float]:
    values = _leader_values(length)
    for offset in range(80):
        values[-80 + offset] *= 1.0 - offset * 0.0045
    return values


def _std(values: list[float]) -> float:
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def test_relative_strength_leader_profile_public_api_and_payload() -> None:
    profile = RelativeStrengthLeaderV1Profile()

    assert RELATIVE_STRENGTH_LEADER_PROFILE_ID == "relative_strength_leader_v1"
    assert profile.horizon_order == (
        "composite",
        "shortterm_leader",
        "midterm_leader",
    )
    assert profile.horizon_order == RELATIVE_STRENGTH_LEADER_HORIZON_ORDER
    assert profile.snapshot_metric_keys == RELATIVE_STRENGTH_LEADER_SNAPSHOT_METRIC_KEYS
    assert profile.ranking_metric_keys == RELATIVE_STRENGTH_LEADER_RANKING_METRIC_KEYS
    assert profile.inspect_metric_keys == profile.snapshot_metric_keys
    assert "relative_strength_persistence_score" in profile.ranking_metric_keys
    assert "tag_risk_chasing_extension" in profile.ranking_metric_keys
    assert "volume" not in profile.snapshot_metric_keys
    assert profile.ranking_config_payload() == {
        "ranking_profile": "relative_strength_leader_v1",
        "min_history_bars": 274,
        "price_floor": {"TW": 10.0, "US": 5.0},
        "liquidity_floor": {"TW": 50_000_000.0, "US": 10_000_000.0},
        "active_trading_min_days_60": {"TW": 50, "US": 55},
        "zero_volume_max_days_20": {"TW": 3, "US": 1},
        "stale_close_max_days_20": 5,
        "extreme_return_abs_cutoff": 0.80,
        "volatility_floor": 0.0001,
        "horizon_order": list(RELATIVE_STRENGTH_LEADER_HORIZON_ORDER),
        "snapshot_metric_keys": list(RELATIVE_STRENGTH_LEADER_SNAPSHOT_METRIC_KEYS),
        "ranking_metric_keys": list(RELATIVE_STRENGTH_LEADER_RANKING_METRIC_KEYS),
        "inspect_metric_keys": list(RELATIVE_STRENGTH_LEADER_SNAPSHOT_METRIC_KEYS),
        "stdev_ddof": 1,
        "score_method": RELATIVE_STRENGTH_LEADER_SCORE_METHOD,
    }


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (RelativeStrengthLeaderV1Profile(profile_id="bad"), "profile_id"),
        (RelativeStrengthLeaderV1Profile(min_history_bars=273), "min_history_bars"),
        (RelativeStrengthLeaderV1Profile(horizon_order=("composite",)), "horizon order"),
        (RelativeStrengthLeaderV1Profile(snapshot_metric_keys=("profile_metrics_version",)), "snapshot metric"),
        (RelativeStrengthLeaderV1Profile(ranking_metric_keys=("score_relative_strength_20d",)), "ranking metric"),
        (RelativeStrengthLeaderV1Profile(score_method="bad"), "score_method"),
    ],
)
def test_relative_strength_leader_profile_rejects_contract_changes(
    profile: RelativeStrengthLeaderV1Profile, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        profile.validate()


def test_relative_strength_leader_builds_leader_snapshot() -> None:
    latest = date(2026, 5, 8)
    profile = RelativeStrengthLeaderV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=_bars("AAA", latest, _leader_values()),
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
    assert row["return_60d"] > 0.0
    assert row["return_120d"] > 0.0
    assert row["pct_below_120d_high"] > 0.0
    assert row["price_vs_sma_200d"] > 0.0
    assert row["max_drawdown_120d"] > -0.30
    for key in profile.snapshot_metric_keys:
        assert isinstance(row[key], int | float)
        assert math.isfinite(float(row[key]))


def test_relative_strength_leader_uses_matching_momentum_volatility_windows() -> None:
    latest = date(2026, 5, 8)
    profile = RelativeStrengthLeaderV1Profile()
    values = _leader_values()
    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=_bars("AAA", latest, values),
        run_latest_bar_date=latest,
    )

    row = snapshot.to_dicts()[0]
    returns = [values[index] / values[index - 1] - 1.0 for index in range(1, len(values))]
    volatility_6_1 = _std(returns[-147:-21])
    volatility_12_1 = _std(returns[-273:-21])
    expected_6_1 = row["momentum_return_6_1"] / volatility_6_1
    expected_12_1 = row["momentum_return_12_1"] / volatility_12_1

    assert row["risk_adjusted_momentum_6_1"] == pytest.approx(expected_6_1)
    assert row["risk_adjusted_momentum_12_1"] == pytest.approx(expected_12_1)


@pytest.mark.parametrize(
    ("ticker", "values", "kwargs"),
    [
        ("SHORT", _leader_values(273), {}),
        ("LOWPX", [value * 0.05 for value in _leader_values()], {}),
        ("LOWLIQ", _leader_values(), {"volume": 1_000.0}),
        ("ZEROTRADE", _leader_values(), {"volume": 0.0}),
        ("ZEROVOL", _leader_values(), {"zero_volume_tail": 5}),
        ("STALE", _leader_values(), {"stale_tail": 6}),
        ("BADHILO", _leader_values(), {"invalid_ohlc": True}),
        ("BROKEN", _broken_values(), {}),
    ],
)
def test_relative_strength_leader_excludes_unusable_or_broken_candidates(
    ticker: str, values: list[float], kwargs: dict[str, object]
) -> None:
    latest = date(2026, 5, 8)
    profile = RelativeStrengthLeaderV1Profile()

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
        "return_20d": 0.32,
        "return_60d": 0.46,
        "return_120d": 0.62,
        "momentum_return_6_1": 0.34,
        "momentum_return_12_1": 0.52,
        "volatility_20d": 0.018,
        "volatility_60d": 0.022,
        "volatility_126d": 0.024,
        "volatility_20d_to_60d_ratio": 0.82,
        "risk_adjusted_return_60d": 20.9,
        "risk_adjusted_return_120d": 25.8,
        "risk_adjusted_momentum_6_1": 15.0,
        "risk_adjusted_momentum_12_1": 21.0,
        "trend_slope_60d": 0.002,
        "uptrend_r2_60d": 0.65,
        "trend_consistency_60d": 0.62,
        "sma_20d": 98.0,
        "sma_50d": 92.0,
        "sma_200d": 74.0,
        "price_vs_sma_20d": 0.02,
        "price_vs_sma_50d": 0.09,
        "price_vs_sma_200d": 0.35,
        "sma_20d_vs_sma_50d": 0.065,
        "sma_50d_vs_sma_200d": 0.24,
        "pct_below_60d_high": -0.01,
        "pct_below_120d_high": -0.02,
        "max_drawdown_120d": -0.08,
        "largest_daily_return_60d": 0.045,
        "gain_concentration_60d": 0.18,
        "zero_volume_days_20d": 0.0,
        "active_trading_days_60d": 60.0,
        "stale_close_days_20d": 0.0,
        "data_quality_extreme_return_flag": 0.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_relative_strength_leader_assigns_rankings_tags_and_caps() -> None:
    profile = RelativeStrengthLeaderV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "LEADER"},
            {
                "ticker": "SHORTTERM",
                "return_20d": 0.20,
                "return_60d": 0.25,
                "return_120d": 0.16,
                "momentum_return_6_1": 0.12,
                "momentum_return_12_1": 0.10,
                "risk_adjusted_momentum_6_1": 4.0,
                "risk_adjusted_momentum_12_1": 3.0,
            },
            {
                "ticker": "CHASE",
                "return_20d": 0.30,
                "return_60d": 0.50,
                "return_120d": 0.72,
                "price_vs_sma_50d": 0.32,
                "pct_below_120d_high": 0.18,
                "largest_daily_return_60d": 0.24,
                "gain_concentration_60d": 0.48,
            },
            {
                "ticker": "FADE",
                "return_20d": -0.04,
                "return_60d": 0.30,
                "return_120d": 0.44,
                "traded_value_5d_to_20d_ratio": 0.62,
            },
        ]
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.height == 12
    assert rankings["horizon"].unique().sort().to_list() == [
        "composite",
        "midterm_leader",
        "shortterm_leader",
    ]
    for horizon in profile.horizon_order:
        rows = rankings.filter(pl.col("horizon") == horizon).sort("rank")
        assert rows["rank"].to_list() == [1, 2, 3, 4]
        assert rows["score"].is_sorted(descending=True)

    composite = {row["ticker"]: row for row in rankings.filter(pl.col("horizon") == "composite").to_dicts()}
    assert composite["LEADER"]["tag_positive_rs_leader"] == 1.0
    assert composite["LEADER"]["tag_positive_persistent_leader"] == 1.0
    assert (
        composite["LEADER"]["relative_strength_persistence_score"]
        > composite["SHORTTERM"]["relative_strength_persistence_score"]
    )
    assert composite["CHASE"]["tag_risk_chasing_extension"] == 1.0
    assert composite["CHASE"]["overheat_cap_score"] < composite["LEADER"]["overheat_cap_score"]
    assert composite["FADE"]["tag_risk_recent_rs_fade"] == 1.0
    assert composite["FADE"]["tag_risk_liquidity_fade"] == 1.0
    for row in rankings.to_dicts():
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_relative_strength_leader_positive_tag_requires_current_and_risk_adjusted_leadership() -> None:
    profile = RelativeStrengthLeaderV1Profile()
    snapshot = _ranking_snapshot(
        [
            {
                "ticker": "LEADER",
                "return_20d": 0.36,
                "return_60d": 0.70,
                "return_120d": 0.95,
                "risk_adjusted_momentum_6_1": 22.0,
                "risk_adjusted_momentum_12_1": 32.0,
            },
            {
                "ticker": "FADING",
                "return_20d": -0.02,
                "return_60d": 0.52,
                "return_120d": 0.76,
                "risk_adjusted_momentum_6_1": 4.0,
                "risk_adjusted_momentum_12_1": 4.0,
            },
            {
                "ticker": "RAWONLY",
                "return_20d": 0.34,
                "return_60d": 0.50,
                "return_120d": 0.72,
                "risk_adjusted_momentum_6_1": -1.0,
                "risk_adjusted_momentum_12_1": -1.0,
            },
        ]
    )

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite").to_dicts()
    }

    assert rows["LEADER"]["tag_positive_rs_leader"] == 1.0
    assert rows["FADING"]["tag_positive_rs_leader"] == 0.0
    assert rows["FADING"]["tag_risk_recent_rs_fade"] == 1.0
    assert rows["RAWONLY"]["tag_positive_rs_leader"] == 0.0
    assert rows["RAWONLY"]["score_risk_adjusted_momentum"] == 0.0


def test_relative_strength_leader_flags_absolute_risk_surfaces() -> None:
    profile = RelativeStrengthLeaderV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "LEADER"},
            {
                "ticker": "HIGHVOL",
                "volatility_60d": 0.075,
                "volatility_20d_to_60d_ratio": 1.05,
            },
            {"ticker": "DRAWDOWN", "max_drawdown_120d": -0.31},
            {
                "ticker": "WEAKTREND",
                "trend_slope_60d": -0.001,
                "price_vs_sma_50d": -0.01,
                "sma_50d_vs_sma_200d": -0.02,
            },
            {"ticker": "DATAWARN", "data_quality_extreme_return_flag": 1.0},
        ]
    )

    rows = {
        row["ticker"]: row
        for row in profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite").to_dicts()
    }

    assert rows["HIGHVOL"]["tag_risk_high_volatility"] == 1.0
    assert rows["HIGHVOL"]["volatility_cap_score"] < rows["LEADER"]["volatility_cap_score"]
    assert rows["DRAWDOWN"]["tag_risk_large_drawdown"] == 1.0
    assert rows["DRAWDOWN"]["drawdown_cap_score"] < rows["LEADER"]["drawdown_cap_score"]
    assert rows["WEAKTREND"]["tag_risk_weak_trend_structure"] == 1.0
    assert rows["WEAKTREND"]["trend_structure_cap_score"] < rows["LEADER"]["trend_structure_cap_score"]
    assert rows["DATAWARN"]["tag_risk_data_quality_warning"] == 1.0
    assert rows["DATAWARN"]["penalty_score"] > rows["LEADER"]["penalty_score"]


def test_relative_strength_leader_rejects_malformed_ranking_snapshot() -> None:
    profile = RelativeStrengthLeaderV1Profile()
    malformed = _ranking_snapshot([{"ticker": "AAA"}]).drop("return_120d")

    with pytest.raises(ValidationError, match="missing required ranking inputs"):
        profile.assign_rankings(malformed)
