from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.momentum_quality_v1 import MomentumQualityV1Profile


def _listing(ticker: str = "AAA") -> ListingCandidate:
    return ListingCandidate(
        market=Market.US,
        ticker=ticker,
        listing_symbol=ticker,
        exchange_segment="NASDAQ",
        listing_status="active",
        instrument_type="common_stock",
        source_id=f"unit:{ticker}",
    )


def _bars(ticker: str, adjusted_closes: list[float], *, volume: float = 1_000_000.0) -> pl.DataFrame:
    start = date(2025, 1, 1)
    rows = []
    for index, adjusted_close in enumerate(adjusted_closes):
        close = adjusted_close
        rows.append(
            {
                "market": "US",
                "ticker": ticker,
                "bar_date": start + timedelta(days=index),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adjusted_close": adjusted_close,
                "volume": volume,
            }
        )
    return pl.DataFrame(rows)


def _trend_with_noise(length: int = 274) -> list[float]:
    return [100.0 + index * 0.5 + (0.75 if index % 3 == 0 else -0.25) for index in range(length)]


def _std_for_test(values: list[float], *, ddof: int = 1) -> float:
    average = sum(values) / len(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - ddof)
    return variance**0.5


def test_momentum_quality_profile_builds_required_raw_snapshot_metrics() -> None:
    adjusted = _trend_with_noise()
    bars = _bars("AAA", adjusted)
    profile = MomentumQualityV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=bars,
        run_latest_bar_date=bars["bar_date"][-1],
    )

    assert snapshot.height == 1
    row = snapshot.to_dicts()[0]
    assert row["profile_metrics_version"] == 2.0
    assert row["avg_traded_value_20d_local"] > 5_000_000.0
    assert row["avg_traded_value_5d_local"] > 5_000_000.0
    assert row["ma20"] > 0.0
    assert row["ma60"] > 0.0
    assert row["ma120"] > 0.0
    assert row["ma200"] > 0.0
    assert 0.0 <= row["moving_average_structure_raw_score"] <= 100.0
    assert row["max_drawdown_252d"] <= 0.0
    assert 0.0 <= row["above_ma60_ratio_126d"] <= 1.0
    assert 0.0 <= row["positive_21d_return_ratio_126d"] <= 1.0
    assert 0.0 <= row["uptrend_consistency_raw_score"] <= 100.0
    assert row["short_term_extension_20d"] == row["short_term_strength_20d"]
    assert row["distance_from_ma20"] > 0.0
    assert row["prior_60d_high_adjusted_close"] == max(adjusted[-61:-1])
    assert row["data_quality_extreme_return_flag"] == 0.0


def test_momentum_quality_profile_uses_specified_return_windows() -> None:
    adjusted = _trend_with_noise()
    bars = _bars("AAA", adjusted)
    profile = MomentumQualityV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=bars,
        run_latest_bar_date=bars["bar_date"][-1],
    )

    row = snapshot.to_dicts()[0]
    assert row["momentum_return_12_1"] == adjusted[-22] / adjusted[-274] - 1.0
    assert row["momentum_return_6_1"] == adjusted[-22] / adjusted[-148] - 1.0
    assert row["short_term_strength_20d"] == adjusted[-1] / adjusted[-21] - 1.0


def test_momentum_quality_profile_uses_specified_volatility_windows() -> None:
    adjusted = _trend_with_noise()
    bars = _bars("AAA", adjusted)
    profile = MomentumQualityV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=bars,
        run_latest_bar_date=bars["bar_date"][-1],
    )

    row = snapshot.to_dicts()[0]
    returns_12_1 = [adjusted[index] / adjusted[index - 1] - 1.0 for index in range(1, 253)]
    returns_6_1 = [adjusted[index] / adjusted[index - 1] - 1.0 for index in range(127, 253)]
    assert row["volatility_12_1"] == pytest.approx(_std_for_test(returns_12_1))
    assert row["volatility_6_1"] == pytest.approx(_std_for_test(returns_6_1))


def test_momentum_quality_profile_filters_unrankable_rows_before_snapshot() -> None:
    adjusted = _trend_with_noise(273)
    bars = _bars("AAA", adjusted)
    profile = MomentumQualityV1Profile()

    snapshot = profile.build_snapshot(
        run_id="us-test",
        market=Market.US,
        listings=[_listing("AAA")],
        bars=bars,
        run_latest_bar_date=bars["bar_date"][-1],
    )

    assert snapshot.is_empty()


def _ranking_snapshot(rows: list[dict[str, object]]) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "run_id": "run",
        "market": "US",
        "close": 100.0,
        "adjusted_close": 100.0,
        "profile_metrics_version": 2.0,
        "avg_traded_value_20d_local": 10_000_000.0,
        "avg_traded_value_5d_local": 10_000_000.0,
        "momentum_return_12_1": 0.20,
        "momentum_return_6_1": 0.10,
        "volatility_12_1": 0.02,
        "volatility_6_1": 0.02,
        "risk_adjusted_momentum_12_1": 10.0,
        "risk_adjusted_momentum_6_1": 5.0,
        "short_term_strength_20d": 0.05,
        "ma20": 100.0,
        "ma60": 95.0,
        "ma120": 90.0,
        "ma200": 85.0,
        "moving_average_structure_raw_score": 80.0,
        "max_drawdown_252d": -0.10,
        "above_ma60_ratio_126d": 0.80,
        "positive_21d_return_ratio_126d": 0.80,
        "uptrend_consistency_raw_score": 80.0,
        "short_term_extension_20d": 0.05,
        "distance_from_ma20": 0.05,
        "prior_60d_high_adjusted_close": 101.0,
        "data_quality_extreme_return_flag": 0.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def _twenty_row_overheat_snapshot() -> pl.DataFrame:
    rows = []
    for index in range(20):
        rows.append(
            {
                "ticker": f"T{index:02d}",
                "risk_adjusted_momentum_12_1": 10.0 + index,
                "risk_adjusted_momentum_6_1": 8.0 + index,
                "short_term_strength_20d": 0.02 + index * 0.01,
                "short_term_extension_20d": 0.02 + index * 0.01,
                "distance_from_ma20": 0.01 + index * 0.01,
                "moving_average_structure_raw_score": 70.0 + index,
                "max_drawdown_252d": -0.20 + index * 0.005,
                "uptrend_consistency_raw_score": 70.0 + index,
                "prior_60d_high_adjusted_close": 105.0,
            }
        )
    rows[-1].update(
        {
            "ticker": "AAA",
            "risk_adjusted_momentum_12_1": 50.0,
            "risk_adjusted_momentum_6_1": 45.0,
            "short_term_strength_20d": 0.80,
            "short_term_extension_20d": 0.80,
            "distance_from_ma20": 0.80,
            "moving_average_structure_raw_score": 100.0,
            "max_drawdown_252d": -0.05,
            "uptrend_consistency_raw_score": 100.0,
            "adjusted_close": 110.0,
            "prior_60d_high_adjusted_close": 105.0,
        }
    )
    return _ranking_snapshot(rows)


def test_momentum_quality_profile_assigns_composite_swing_midterm_rankings() -> None:
    profile = MomentumQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"ticker": "AAA", "risk_adjusted_momentum_12_1": 25.0, "risk_adjusted_momentum_6_1": 20.0},
            {"ticker": "BBB", "risk_adjusted_momentum_12_1": 15.0, "risk_adjusted_momentum_6_1": 10.0},
            {"ticker": "CCC", "risk_adjusted_momentum_12_1": 5.0, "risk_adjusted_momentum_6_1": 5.0},
        ]
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.height == 9
    assert rankings["horizon"].unique().sort().to_list() == ["composite", "midterm", "swing"]
    for horizon in profile.horizon_order:
        rows = rankings.filter(pl.col("horizon") == horizon).sort("rank")
        assert rows["rank"].to_list() == [1, 2, 3]
        assert rows["score"].is_sorted(descending=True)


def test_momentum_quality_profile_uses_raw_score_components() -> None:
    profile = MomentumQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {
                "ticker": "AAA",
                "risk_adjusted_momentum_12_1": 10.0,
                "risk_adjusted_momentum_6_1": 5.0,
                "short_term_strength_20d": 0.04,
                "short_term_extension_20d": 0.04,
                "distance_from_ma20": 0.03,
                "volatility_12_1": 0.03,
                "volatility_6_1": 0.02,
                "moving_average_structure_raw_score": 80.0,
                "max_drawdown_252d": -0.10,
                "uptrend_consistency_raw_score": 90.0,
            },
        ]
    )

    rankings = profile.assign_rankings(snapshot)
    row = rankings.filter((pl.col("horizon") == "composite") & (pl.col("ticker") == "AAA")).to_dicts()[0]

    assert row["score_risk_adjusted_momentum_12_1"] == 10.0
    assert row["score_risk_adjusted_momentum_6_1"] == 5.0
    assert row["score_short_term_strength_20d"] == 2.0
    assert row["score_short_term_extension_20d"] == 0.04
    assert row["score_distance_from_ma20"] == 0.03
    assert row["score_volatility_12_1"] == 0.03
    assert row["score_volatility_6_1"] == 0.02
    assert row["moving_average_structure_score"] == 0.80
    assert row["drawdown_control_score"] == 0.90
    assert row["uptrend_consistency_score"] == 0.90
    assert row["trend_quality_score"] == pytest.approx(0.86)
    assert row["momentum_blend_score"] == pytest.approx(7.45)
    assert row["overheat_score"] == pytest.approx(0.036)
    assert row["overheat_penalty_score"] == 0.0


def test_momentum_quality_profile_ranks_within_run_market_without_rewriting_scores() -> None:
    profile = MomentumQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {"market": "US", "ticker": "AAA", "risk_adjusted_momentum_12_1": 1.0},
            {"market": "US", "ticker": "BBB", "risk_adjusted_momentum_12_1": 2.0},
            {"market": "TW", "ticker": "2330", "risk_adjusted_momentum_12_1": 100.0},
            {"market": "TW", "ticker": "2317", "risk_adjusted_momentum_12_1": 200.0},
        ]
    )

    rankings = profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite")

    us_scores = rankings.filter(pl.col("market") == "US").sort("ticker")["score_risk_adjusted_momentum_12_1"].to_list()
    tw_scores = rankings.filter(pl.col("market") == "TW").sort("ticker")["score_risk_adjusted_momentum_12_1"].to_list()
    assert us_scores == [1.0, 2.0]
    assert tw_scores == [200.0, 100.0]
    assert rankings.filter(pl.col("market") == "US").sort("rank")["ticker"].to_list() == ["BBB", "AAA"]
    assert rankings.filter(pl.col("market") == "TW").sort("rank")["ticker"].to_list() == ["2317", "2330"]


def test_momentum_quality_profile_keeps_large_raw_score_outliers() -> None:
    profile = MomentumQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {
                "ticker": f"T{index:02d}",
                "risk_adjusted_momentum_12_1": 1000.0 if index == 19 else float(index),
            }
            for index in range(20)
        ]
    )

    rankings = profile.assign_rankings(snapshot).filter(pl.col("horizon") == "composite")
    leader = rankings.sort("score_risk_adjusted_momentum_12_1", descending=True).to_dicts()[0]

    assert leader["ticker"] == "T19"
    assert leader["score_risk_adjusted_momentum_12_1"] == 1000.0
    assert leader["score"] > 100.0


def test_momentum_quality_profile_returns_empty_raw_score_schema_for_empty_snapshot() -> None:
    profile = MomentumQualityV1Profile()

    rankings = profile.assign_rankings(pl.DataFrame())

    assert rankings.is_empty()
    assert rankings.schema["score_risk_adjusted_momentum_12_1"] == pl.Float64
    assert rankings.schema["score"] == pl.Float64
    assert rankings.schema["rank"] == pl.Int64


def test_momentum_quality_profile_filters_non_finite_raw_score_inputs() -> None:
    profile = MomentumQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {
                "ticker": "AAA",
                "risk_adjusted_momentum_12_1": float("nan"),
                "risk_adjusted_momentum_6_1": 5.0,
            },
            {
                "ticker": "BBB",
                "risk_adjusted_momentum_12_1": 10.0,
                "risk_adjusted_momentum_6_1": 5.0,
                "volatility_6_1": 0.0,
            },
            {
                "ticker": "CCC",
                "risk_adjusted_momentum_12_1": 10.0,
                "risk_adjusted_momentum_6_1": 5.0,
                "short_term_strength_20d": 0.10,
                "volatility_6_1": 0.0001,
            },
        ]
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.is_empty()


def test_momentum_quality_profile_filters_raw_score_overflow_outputs() -> None:
    profile = MomentumQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {
                "ticker": "AAA",
                "risk_adjusted_momentum_12_1": 10.0,
                "risk_adjusted_momentum_6_1": 5.0,
                "short_term_strength_20d": 1.0,
                "volatility_6_1": 1e-309,
            },
        ]
    )

    rankings = profile.assign_rankings(snapshot)

    assert rankings.is_empty()


def test_momentum_quality_profile_penalizes_severe_overheat() -> None:
    profile = MomentumQualityV1Profile()
    snapshot = _twenty_row_overheat_snapshot()

    rankings = profile.assign_rankings(snapshot)
    aaa = rankings.filter((pl.col("horizon") == "composite") & (pl.col("ticker") == "AAA")).to_dicts()[0]
    swing_aaa = rankings.filter((pl.col("horizon") == "swing") & (pl.col("ticker") == "AAA")).to_dicts()[0]

    assert aaa["overheat_score"] == pytest.approx(0.80)
    assert aaa["overheat_penalty_score"] > 0.0
    assert aaa["score"] < aaa["momentum_blend_score"]
    assert swing_aaa["score"] < swing_aaa["momentum_blend_score"]


def test_momentum_quality_profile_computes_risk_and_positive_tags() -> None:
    profile = MomentumQualityV1Profile()
    rows = []
    for index in range(20):
        rows.append(
            {
                "ticker": f"T{index:02d}",
                "risk_adjusted_momentum_12_1": 10.0 + index,
                "risk_adjusted_momentum_6_1": 10.0 + index,
                "short_term_strength_20d": 0.02 + index * 0.01,
                "volatility_12_1": 0.01 + index * 0.001,
                "volatility_6_1": 0.01 + index * 0.001,
                "moving_average_structure_raw_score": 70.0 + index,
                "max_drawdown_252d": -0.20 + index * 0.005,
                "uptrend_consistency_raw_score": 70.0 + index,
                "short_term_extension_20d": 0.02 + index * 0.01,
                "distance_from_ma20": 0.01 + index * 0.01,
                "avg_traded_value_5d_local": 10_000_000.0,
                "data_quality_extreme_return_flag": 0.0,
            }
        )
    rows[-1].update(
        {
            "ticker": "AAA",
            "adjusted_close": 110.0,
            "prior_60d_high_adjusted_close": 105.0,
            "short_term_extension_20d": 0.50,
            "distance_from_ma20": 0.50,
            "volatility_12_1": 0.20,
            "volatility_6_1": 0.20,
            "max_drawdown_252d": -0.35,
            "avg_traded_value_5d_local": 4_000_000.0,
            "data_quality_extreme_return_flag": 1.0,
        }
    )
    snapshot = _ranking_snapshot(rows)

    row = (
        profile.assign_rankings(snapshot)
        .filter((pl.col("horizon") == "composite") & (pl.col("ticker") == "AAA"))
        .to_dicts()[0]
    )

    assert row["tag_risk_overheated"] == 1.0
    assert row["tag_risk_extended_from_ma20"] == 1.0
    assert row["tag_risk_high_volatility"] == 1.0
    assert row["tag_risk_large_drawdown"] == 1.0
    assert row["tag_risk_thin_recent_volume"] == 1.0
    assert row["tag_risk_data_quality_warning"] == 1.0
    assert row["tag_positive_strong_momentum"] == 1.0


def test_momentum_quality_profile_flags_weak_trend_quality_and_early_breakout() -> None:
    profile = MomentumQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {
                "ticker": "AAA",
                "adjusted_close": 110.0,
                "prior_60d_high_adjusted_close": 105.0,
                "risk_adjusted_momentum_12_1": 20.0,
                "risk_adjusted_momentum_6_1": 18.0,
                "short_term_strength_20d": 0.06,
                "moving_average_structure_raw_score": 20.0,
                "max_drawdown_252d": -0.45,
                "uptrend_consistency_raw_score": 20.0,
            },
            {
                "ticker": "BBB",
                "adjusted_close": 100.0,
                "prior_60d_high_adjusted_close": 110.0,
                "risk_adjusted_momentum_12_1": 10.0,
                "risk_adjusted_momentum_6_1": 8.0,
                "short_term_strength_20d": 0.04,
                "moving_average_structure_raw_score": 80.0,
                "max_drawdown_252d": -0.05,
                "uptrend_consistency_raw_score": 80.0,
            },
            {
                "ticker": "CCC",
                "adjusted_close": 100.0,
                "prior_60d_high_adjusted_close": 110.0,
                "risk_adjusted_momentum_12_1": 5.0,
                "risk_adjusted_momentum_6_1": 4.0,
                "short_term_strength_20d": 0.02,
                "moving_average_structure_raw_score": 60.0,
                "max_drawdown_252d": -0.10,
                "uptrend_consistency_raw_score": 60.0,
            },
        ]
    )

    row = (
        profile.assign_rankings(snapshot)
        .filter((pl.col("horizon") == "composite") & (pl.col("ticker") == "AAA"))
        .to_dicts()[0]
    )

    assert row["tag_risk_weak_trend_quality"] == 1.0
    assert row["tag_positive_early_breakout"] == 1.0


def test_momentum_quality_profile_flags_stable_uptrend() -> None:
    profile = MomentumQualityV1Profile()
    snapshot = _ranking_snapshot(
        [
            {
                "ticker": "AAA",
                "moving_average_structure_raw_score": 100.0,
                "max_drawdown_252d": -0.03,
                "uptrend_consistency_raw_score": 100.0,
            },
            {
                "ticker": "BBB",
                "moving_average_structure_raw_score": 60.0,
                "max_drawdown_252d": -0.20,
                "uptrend_consistency_raw_score": 60.0,
            },
            {
                "ticker": "CCC",
                "moving_average_structure_raw_score": 40.0,
                "max_drawdown_252d": -0.35,
                "uptrend_consistency_raw_score": 40.0,
            },
        ]
    )

    row = (
        profile.assign_rankings(snapshot)
        .filter((pl.col("horizon") == "composite") & (pl.col("ticker") == "AAA"))
        .to_dicts()[0]
    )

    assert row["trend_quality_score"] >= 0.80
    assert row["tag_risk_large_drawdown"] == 0.0
    assert row["tag_positive_stable_uptrend"] == 1.0
