from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.mean_reversion_quality_v1 import (
    MEAN_REVERSION_QUALITY_HORIZON_ORDER,
    MEAN_REVERSION_QUALITY_PROFILE_ID,
    MEAN_REVERSION_QUALITY_RANKING_METRIC_KEYS,
    MEAN_REVERSION_QUALITY_SCORE_METHOD,
    MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS,
    MeanReversionQualityV1Profile,
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


def _mean_reversion_values(length: int = 253) -> list[float]:
    values: list[float] = []
    for index in range(length):
        if index < length - 42:
            value = 52.0 + index * 0.13 + math.sin(index / 8.0) * 0.4
        elif index < length - 5:
            pullback_index = index - (length - 42)
            value = values[-1] * (1.0 - 0.0045 - pullback_index * 0.00025)
        else:
            value = values[-1] * 1.006
        values.append(value)
    return values


def _falling_knife_values(length: int = 253) -> list[float]:
    values = _mean_reversion_values(length)
    for offset in range(70):
        values[-70 + offset] *= 1.0 - offset * 0.006
    return values


def test_mean_reversion_quality_profile_public_api_and_payload() -> None:
    profile = MeanReversionQualityV1Profile()

    assert MEAN_REVERSION_QUALITY_PROFILE_ID == "mean_reversion_quality_v1"
    assert profile.horizon_order == ("composite", "oversold_bounce", "support_reversion")
    assert profile.horizon_order == MEAN_REVERSION_QUALITY_HORIZON_ORDER
    assert profile.snapshot_metric_keys == MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS
    assert profile.ranking_metric_keys == MEAN_REVERSION_QUALITY_RANKING_METRIC_KEYS
    assert profile.inspect_metric_keys == profile.snapshot_metric_keys
    assert "distance_from_sma_20d" in profile.snapshot_metric_keys
    assert "tag_risk_falling_knife" in profile.ranking_metric_keys
    assert "volume" not in profile.snapshot_metric_keys
    assert profile.ranking_config_payload() == {
        "ranking_profile": "mean_reversion_quality_v1",
        "min_history_bars": 252,
        "price_floor": {"TW": 10.0, "US": 5.0},
        "liquidity_floor": {"TW": 50_000_000.0, "US": 10_000_000.0},
        "active_trading_min_days_60": {"TW": 50, "US": 55},
        "zero_volume_max_days_20": {"TW": 3, "US": 1},
        "stale_close_max_days_20": 5,
        "extreme_return_abs_cutoff": 0.80,
        "volatility_floor": 0.0001,
        "horizon_order": list(MEAN_REVERSION_QUALITY_HORIZON_ORDER),
        "snapshot_metric_keys": list(MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS),
        "ranking_metric_keys": list(MEAN_REVERSION_QUALITY_RANKING_METRIC_KEYS),
        "inspect_metric_keys": list(MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS),
        "stdev_ddof": 1,
        "score_method": MEAN_REVERSION_QUALITY_SCORE_METHOD,
    }


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (MeanReversionQualityV1Profile(profile_id="bad"), "profile_id"),
        (MeanReversionQualityV1Profile(min_history_bars=251), "min_history_bars"),
        (MeanReversionQualityV1Profile(horizon_order=("composite",)), "horizon order"),
        (MeanReversionQualityV1Profile(snapshot_metric_keys=("profile_metrics_version",)), "snapshot metric"),
        (MeanReversionQualityV1Profile(ranking_metric_keys=("score_oversold_depth",)), "ranking metric"),
        (MeanReversionQualityV1Profile(score_method="bad"), "score_method"),
    ],
)
def test_mean_reversion_quality_profile_rejects_contract_changes(
    profile: MeanReversionQualityV1Profile, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        profile.validate()


def test_mean_reversion_quality_builds_oversold_snapshot() -> None:
    latest = date(2026, 5, 8)
    profile = MeanReversionQualityV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=_bars("AAA", latest, _mean_reversion_values()),
        run_latest_bar_date=latest,
    )

    assert snapshot.height == 1
    row = snapshot.to_dicts()[0]
    assert row["ticker"] == "AAA"
    assert row["profile_metrics_version"] == 1.0
    assert row["asof_bar_date_yyyymmdd"] == 20260508.0
    assert row["return_20d"] < 0.0
    assert row["distance_from_sma_20d"] < 0.0
    assert row["close_position_20d_range"] <= 0.70
    assert row["max_drawdown_120d"] > -0.40
    assert row["price_vs_sma_200d"] > -0.25
    for key in profile.snapshot_metric_keys:
        assert isinstance(row[key], int | float)
        assert math.isfinite(float(row[key]))


@pytest.mark.parametrize(
    ("ticker", "values", "kwargs"),
    [
        ("SHORT", _mean_reversion_values(251), {}),
        ("LOWPX", [value * 0.05 for value in _mean_reversion_values()], {}),
        ("LOWLIQ", _mean_reversion_values(), {"volume": 1_000.0}),
        ("ZEROTRADE", _mean_reversion_values(), {"volume": 0.0}),
        ("ZEROVOL", _mean_reversion_values(), {"zero_volume_tail": 5}),
        ("STALE", _mean_reversion_values(), {"stale_tail": 6}),
        ("BADHILO", _mean_reversion_values(), {"invalid_ohlc": True}),
        ("FALLING", _falling_knife_values(), {}),
    ],
)
def test_mean_reversion_quality_excludes_unusable_or_broken_candidates(
    ticker: str, values: list[float], kwargs: dict[str, object]
) -> None:
    latest = date(2026, 5, 8)
    profile = MeanReversionQualityV1Profile()

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
        "return_5d": 0.03,
        "return_20d": -0.08,
        "return_60d": 0.10,
        "distance_from_sma_20d": -0.04,
        "distance_from_sma_50d": -0.03,
        "distance_from_sma_200d": 0.18,
        "pct_below_60d_high": -0.12,
        "pct_below_120d_high": -0.14,
        "close_position_20d_range": 0.30,
        "downside_volatility_20d": 0.014,
        "volatility_20d": 0.020,
        "volatility_60d": 0.018,
        "volatility_20d_to_60d_ratio": 1.11,
        "max_drawdown_120d": -0.14,
        "max_drawdown_252d": -0.20,
        "support_proximity_score_raw": 0.82,
        "rebound_confirmation_score_raw": 0.70,
        "liquidity_continuity_score_raw": 0.90,
        "trend_slope_60d": 0.0015,
        "uptrend_r2_60d": 0.50,
        "sma_20d": 104.0,
        "sma_50d": 103.0,
        "sma_200d": 84.0,
        "price_vs_sma_20d": -0.04,
        "price_vs_sma_50d": -0.03,
        "price_vs_sma_200d": 0.18,
        "sma_50d_vs_sma_200d": 0.22,
        "zero_volume_days_20d": 0.0,
        "active_trading_days_60d": 60.0,
        "stale_close_days_20d": 0.0,
        "data_quality_extreme_return_flag": 0.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_mean_reversion_quality_assigns_rankings_tags_and_caps() -> None:
    profile = MeanReversionQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "QUALITY"},
            {
                "ticker": "BOUNCE",
                "return_5d": 0.08,
                "return_20d": -0.16,
                "distance_from_sma_20d": -0.09,
                "close_position_20d_range": 0.18,
                "rebound_confirmation_score_raw": 0.92,
            },
            {
                "ticker": "FALLING",
                "return_5d": -0.08,
                "return_20d": -0.28,
                "distance_from_sma_50d": -0.18,
                "close_position_20d_range": 0.05,
                "volatility_20d_to_60d_ratio": 2.30,
                "max_drawdown_120d": -0.33,
                "rebound_confirmation_score_raw": 0.10,
            },
            {
                "ticker": "BREAKDOWN",
                "price_vs_sma_200d": -0.28,
                "distance_from_sma_200d": -0.28,
                "trend_slope_60d": -0.001,
                "sma_50d_vs_sma_200d": -0.04,
            },
        ]
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.height == 12
    assert rankings["horizon"].unique().sort().to_list() == [
        "composite",
        "oversold_bounce",
        "support_reversion",
    ]
    for horizon in profile.horizon_order:
        rows = rankings.filter(pl.col("horizon") == horizon).sort("rank")
        assert rows["rank"].to_list() == [1, 2, 3, 4]
        assert rows["score"].is_sorted(descending=True)

    composite = {row["ticker"]: row for row in rankings.filter(pl.col("horizon") == "composite").to_dicts()}
    assert composite["QUALITY"]["tag_setup_oversold_quality"] == 1.0
    assert composite["QUALITY"]["tag_setup_near_support"] == 1.0
    assert composite["BOUNCE"]["tag_setup_rebound_confirmation"] == 1.0
    assert composite["FALLING"]["tag_risk_falling_knife"] == 1.0
    assert composite["FALLING"]["tag_risk_volatility_spike"] == 1.0
    assert composite["FALLING"]["falling_knife_cap_score"] < composite["QUALITY"]["falling_knife_cap_score"]
    assert composite["BREAKDOWN"]["tag_risk_breakdown"] == 1.0
    assert composite["BREAKDOWN"]["structure_cap_score"] < composite["QUALITY"]["structure_cap_score"]
    for row in rankings.to_dicts():
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_mean_reversion_quality_rejects_malformed_ranking_snapshot() -> None:
    profile = MeanReversionQualityV1Profile()
    malformed = _ranking_snapshot([{"ticker": "AAA"}]).drop("distance_from_sma_20d")

    with pytest.raises(ValidationError, match="missing required ranking inputs"):
        profile.assign_rankings(malformed)
