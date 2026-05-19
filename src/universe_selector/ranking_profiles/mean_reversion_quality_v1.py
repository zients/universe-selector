from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Literal, Mapping

import polars as pl

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles._alpha_common import (
    all_finite,
    band_score,
    clamp,
    downside_std,
    immutable_market_float_mapping,
    immutable_market_int_mapping,
    max_drawdown,
    mean,
    ols_slope_r2,
    percentile_scores,
    std,
    yyyymmdd,
)
from universe_selector.ranking_profiles.registration import RankingProfileRegistration


MEAN_REVERSION_QUALITY_PROFILE_ID = "mean_reversion_quality_v1"
MEAN_REVERSION_QUALITY_SCORE_METHOD = "market_relative_mean_reversion_quality_v1"
MEAN_REVERSION_QUALITY_RANK_INTERPRETATION_NOTE = (
    "Mean reversion quality scores are market-local relative rankings for short-term oversold "
    "setups with support, rebound, and falling-knife risk controls. They are ranking lenses, "
    "not buy signals, return forecasts, backtests, or holding-period recommendations."
)
MEAN_REVERSION_QUALITY_HORIZON_ORDER = ("composite", "oversold_bounce", "support_reversion")

MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS = (
    "profile_metrics_version",
    "asof_bar_date_yyyymmdd",
    "avg_traded_value_20d_local",
    "avg_traded_value_5d_local",
    "median_traded_value_20d_local",
    "traded_value_5d_to_20d_ratio",
    "return_5d",
    "return_20d",
    "return_60d",
    "distance_from_sma_20d",
    "distance_from_sma_50d",
    "distance_from_sma_200d",
    "pct_below_60d_high",
    "pct_below_120d_high",
    "close_position_20d_range",
    "downside_volatility_20d",
    "volatility_20d",
    "volatility_60d",
    "volatility_20d_to_60d_ratio",
    "max_drawdown_120d",
    "max_drawdown_252d",
    "support_proximity_score_raw",
    "rebound_confirmation_score_raw",
    "liquidity_continuity_score_raw",
    "trend_slope_60d",
    "uptrend_r2_60d",
    "sma_20d",
    "sma_50d",
    "sma_200d",
    "price_vs_sma_20d",
    "price_vs_sma_50d",
    "price_vs_sma_200d",
    "sma_50d_vs_sma_200d",
    "zero_volume_days_20d",
    "active_trading_days_60d",
    "stale_close_days_20d",
    "data_quality_extreme_return_flag",
)

MEAN_REVERSION_QUALITY_RANKING_METRIC_KEYS = (
    "score_oversold_depth",
    "score_support_proximity",
    "score_rebound_confirmation",
    "score_drawdown_control",
    "score_volatility_control",
    "score_liquidity_continuity",
    "score_trend_preservation",
    "oversold_quality_score",
    "support_reversion_score",
    "rebound_quality_score",
    "risk_control_score",
    "structure_cap_score",
    "falling_knife_cap_score",
    "volatility_cap_score",
    "drawdown_cap_score",
    "penalty_score",
    "tag_setup_oversold_quality",
    "tag_setup_near_support",
    "tag_setup_rebound_confirmation",
    "tag_risk_falling_knife",
    "tag_risk_breakdown",
    "tag_risk_deep_drawdown",
    "tag_risk_volatility_spike",
    "tag_risk_liquidity_fade",
    "tag_risk_no_reversion_setup",
    "tag_risk_data_quality_warning",
)

MEAN_REVERSION_QUALITY_SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    **{key: pl.Float64 for key in MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS},
}

MEAN_REVERSION_QUALITY_RANKING_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "horizon": pl.String,
    "ticker": pl.String,
    **{key: pl.Float64 for key in MEAN_REVERSION_QUALITY_RANKING_METRIC_KEYS},
    "score": pl.Float64,
    "rank": pl.Int64,
}


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=MEAN_REVERSION_QUALITY_SNAPSHOT_SCHEMA)


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema=MEAN_REVERSION_QUALITY_RANKING_SCHEMA)


@dataclass(frozen=True)
class MeanReversionQualityV1Profile:
    profile_id: Literal["mean_reversion_quality_v1"] = MEAN_REVERSION_QUALITY_PROFILE_ID
    min_history_bars: int = 252
    price_floor: Mapping[Market, float] = field(
        default_factory=lambda: {Market.TW: 10.0, Market.US: 5.0}
    )
    liquidity_floor: Mapping[Market, float] = field(
        default_factory=lambda: {Market.TW: 50_000_000.0, Market.US: 10_000_000.0}
    )
    active_trading_min_days_60: Mapping[Market, int] = field(
        default_factory=lambda: {Market.TW: 50, Market.US: 55}
    )
    zero_volume_max_days_20: Mapping[Market, int] = field(
        default_factory=lambda: {Market.TW: 3, Market.US: 1}
    )
    stale_close_max_days_20: int = 5
    extreme_return_abs_cutoff: float = 0.80
    volatility_floor: float = 0.0001
    snapshot_metric_keys: tuple[str, ...] = MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = MEAN_REVERSION_QUALITY_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS
    horizon_order: tuple[str, ...] = MEAN_REVERSION_QUALITY_HORIZON_ORDER
    stdev_ddof: int = 1
    score_method: str = MEAN_REVERSION_QUALITY_SCORE_METHOD
    rank_interpretation_note: str = MEAN_REVERSION_QUALITY_RANK_INTERPRETATION_NOTE

    def __post_init__(self) -> None:
        object.__setattr__(self, "price_floor", immutable_market_float_mapping(self.price_floor))
        object.__setattr__(self, "liquidity_floor", immutable_market_float_mapping(self.liquidity_floor))
        object.__setattr__(
            self,
            "active_trading_min_days_60",
            immutable_market_int_mapping(self.active_trading_min_days_60),
        )
        object.__setattr__(
            self,
            "zero_volume_max_days_20",
            immutable_market_int_mapping(self.zero_volume_max_days_20),
        )
        object.__setattr__(self, "snapshot_metric_keys", tuple(str(key) for key in self.snapshot_metric_keys))
        object.__setattr__(self, "ranking_metric_keys", tuple(str(key) for key in self.ranking_metric_keys))
        object.__setattr__(self, "inspect_metric_keys", tuple(str(key) for key in self.inspect_metric_keys))
        object.__setattr__(self, "horizon_order", tuple(str(horizon) for horizon in self.horizon_order))

    def validate(self) -> None:
        if self.profile_id != MEAN_REVERSION_QUALITY_PROFILE_ID:
            raise ValidationError("profile_id must be mean_reversion_quality_v1")
        if self.min_history_bars != 252:
            raise ValidationError("mean_reversion_quality_v1 requires min_history_bars to be 252")
        if self.snapshot_metric_keys != MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("snapshot metric keys must match mean_reversion_quality_v1")
        if self.ranking_metric_keys != MEAN_REVERSION_QUALITY_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match mean_reversion_quality_v1")
        if self.inspect_metric_keys != MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("inspect metric keys must match mean_reversion_quality_v1")
        if self.horizon_order != MEAN_REVERSION_QUALITY_HORIZON_ORDER:
            raise ValidationError("horizon order must be composite, oversold_bounce, support_reversion")
        for mapping_name in (
            "price_floor",
            "liquidity_floor",
            "active_trading_min_days_60",
            "zero_volume_max_days_20",
        ):
            if set(getattr(self, mapping_name)) != set(Market):
                raise ValidationError(f"{mapping_name} must contain exactly supported markets")
        if {market: self.price_floor[market] for market in Market} != {Market.TW: 10.0, Market.US: 5.0}:
            raise ValidationError("price_floor must match mean_reversion_quality_v1")
        if {market: self.liquidity_floor[market] for market in Market} != {
            Market.TW: 50_000_000.0,
            Market.US: 10_000_000.0,
        }:
            raise ValidationError("liquidity_floor must match mean_reversion_quality_v1")
        if {market: self.active_trading_min_days_60[market] for market in Market} != {
            Market.TW: 50,
            Market.US: 55,
        }:
            raise ValidationError("active_trading_min_days_60 must match mean_reversion_quality_v1")
        if {market: self.zero_volume_max_days_20[market] for market in Market} != {
            Market.TW: 3,
            Market.US: 1,
        }:
            raise ValidationError("zero_volume_max_days_20 must match mean_reversion_quality_v1")
        if self.stale_close_max_days_20 != 5:
            raise ValidationError("stale_close_max_days_20 must be 5")
        if self.extreme_return_abs_cutoff != 0.80:
            raise ValidationError("extreme_return_abs_cutoff must be 0.80")
        if self.volatility_floor != 0.0001:
            raise ValidationError("volatility_floor must be 0.0001")
        if self.stdev_ddof != 1:
            raise ValidationError("stdev_ddof must be 1")
        if self.score_method != MEAN_REVERSION_QUALITY_SCORE_METHOD:
            raise ValidationError(f"score_method must be {MEAN_REVERSION_QUALITY_SCORE_METHOD}")
        if self.rank_interpretation_note != MEAN_REVERSION_QUALITY_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match mean_reversion_quality_v1")

    def ranking_config_payload(self) -> dict[str, object]:
        return {
            "ranking_profile": self.profile_id,
            "min_history_bars": self.min_history_bars,
            "price_floor": {market.value: self.price_floor[market] for market in Market},
            "liquidity_floor": {market.value: self.liquidity_floor[market] for market in Market},
            "active_trading_min_days_60": {
                market.value: self.active_trading_min_days_60[market] for market in Market
            },
            "zero_volume_max_days_20": {
                market.value: self.zero_volume_max_days_20[market] for market in Market
            },
            "stale_close_max_days_20": self.stale_close_max_days_20,
            "extreme_return_abs_cutoff": self.extreme_return_abs_cutoff,
            "volatility_floor": self.volatility_floor,
            "horizon_order": list(self.horizon_order),
            "snapshot_metric_keys": list(self.snapshot_metric_keys),
            "ranking_metric_keys": list(self.ranking_metric_keys),
            "inspect_metric_keys": list(self.inspect_metric_keys),
            "stdev_ddof": self.stdev_ddof,
            "score_method": self.score_method,
        }

    def empty_snapshot(self) -> pl.DataFrame:
        return _empty_snapshot()

    def build_snapshot(
        self,
        *,
        run_id: str,
        market: Market,
        listings: list[ListingCandidate],
        bars: pl.DataFrame,
        run_latest_bar_date: date,
    ) -> pl.DataFrame:
        listed_tickers = {item.ticker for item in listings if item.market == market}
        if not listed_tickers or bars.is_empty():
            return _empty_snapshot()

        candidate_bars = bars.filter(
            (pl.col("market") == market.value)
            & (pl.col("ticker").is_in(list(listed_tickers)))
            & (pl.col("bar_date") <= run_latest_bar_date)
        )
        if candidate_bars.is_empty():
            return _empty_snapshot()
        profile_asof_bar_date = candidate_bars["bar_date"].max()
        rows: list[dict[str, object]] = []
        for ticker in sorted(listed_tickers):
            ticker_bars = (
                candidate_bars.filter((pl.col("ticker") == ticker) & (pl.col("bar_date") <= profile_asof_bar_date))
                .sort("bar_date")
            )
            if ticker_bars.is_empty() or ticker_bars.height < self.min_history_bars:
                continue
            if ticker_bars["bar_date"].n_unique() != ticker_bars.height:
                continue
            if ticker_bars["bar_date"][-1] != profile_asof_bar_date:
                continue

            retained = ticker_bars.tail(self.min_history_bars)
            opens = retained["open"].to_list()
            highs = retained["high"].to_list()
            lows = retained["low"].to_list()
            closes = retained["close"].to_list()
            adjusted_closes = retained["adjusted_close"].to_list()
            volumes = retained["volume"].to_list()
            if not all_finite(opens + highs + lows + closes + adjusted_closes + volumes):
                continue

            opens_float = [float(value) for value in opens]
            highs_float = [float(value) for value in highs]
            lows_float = [float(value) for value in lows]
            closes_float = [float(value) for value in closes]
            adjusted_closes_float = [float(value) for value in adjusted_closes]
            volumes_float = [float(value) for value in volumes]
            if any(value <= 0.0 for value in opens_float + highs_float + lows_float + closes_float + adjusted_closes_float):
                continue
            if any(value < 0.0 for value in volumes_float):
                continue
            if any(
                high < low or high < open_ or high < close or low > open_ or low > close
                for high, low, open_, close in zip(
                    highs_float, lows_float, opens_float, closes_float, strict=True
                )
            ):
                continue

            latest_close = closes_float[-1]
            latest_adjusted_close = adjusted_closes_float[-1]
            traded_values = [
                close * volume for close, volume in zip(closes_float, volumes_float, strict=True)
            ]
            traded_values_5d = traded_values[-5:]
            traded_values_20d = traded_values[-20:]
            avg_traded_value_5d_local = mean(traded_values_5d)
            avg_traded_value_20d_local = mean(traded_values_20d)
            if avg_traded_value_20d_local <= 0.0:
                continue
            median_traded_value_20d_local = float(median(traded_values_20d))
            traded_value_5d_to_20d_ratio = avg_traded_value_5d_local / avg_traded_value_20d_local
            active_trading_days_60d = float(sum(1 for value in volumes_float[-60:] if value > 0.0))
            zero_volume_days_20d = float(sum(1 for value in volumes_float[-20:] if value == 0.0))
            stale_close_days_20d = float(
                sum(
                    1
                    for index in range(len(closes_float) - 20, len(closes_float))
                    if closes_float[index] == closes_float[index - 1]
                    and adjusted_closes_float[index] == adjusted_closes_float[index - 1]
                )
            )

            if latest_close < self.price_floor[market]:
                continue
            if avg_traded_value_20d_local < self.liquidity_floor[market]:
                continue
            if active_trading_days_60d < self.active_trading_min_days_60[market]:
                continue
            if zero_volume_days_20d > self.zero_volume_max_days_20[market]:
                continue
            if stale_close_days_20d >= float(self.stale_close_max_days_20):
                continue

            returns = [
                adjusted_closes_float[index] / adjusted_closes_float[index - 1] - 1.0
                for index in range(1, len(adjusted_closes_float))
            ]
            if any(abs(value) > self.extreme_return_abs_cutoff for value in returns):
                continue
            returns_20d = returns[-20:]
            returns_60d = returns[-60:]
            volatility_20d = std(returns_20d, ddof=self.stdev_ddof)
            volatility_60d = std(returns_60d, ddof=self.stdev_ddof)
            downside_volatility_20d = downside_std(returns_20d, ddof=self.stdev_ddof)
            if volatility_20d is None or volatility_60d is None or downside_volatility_20d is None:
                continue
            if volatility_20d <= 0.0 or volatility_60d <= self.volatility_floor:
                continue

            return_5d = latest_adjusted_close / adjusted_closes_float[-6] - 1.0
            return_20d = latest_adjusted_close / adjusted_closes_float[-21] - 1.0
            return_60d = latest_adjusted_close / adjusted_closes_float[-61] - 1.0
            sma_20d = mean(adjusted_closes_float[-20:])
            sma_50d = mean(adjusted_closes_float[-50:])
            sma_200d = mean(adjusted_closes_float[-200:])
            distance_from_sma_20d = latest_adjusted_close / sma_20d - 1.0
            distance_from_sma_50d = latest_adjusted_close / sma_50d - 1.0
            distance_from_sma_200d = latest_adjusted_close / sma_200d - 1.0
            price_vs_sma_20d = distance_from_sma_20d
            price_vs_sma_50d = distance_from_sma_50d
            price_vs_sma_200d = distance_from_sma_200d
            sma_50d_vs_sma_200d = sma_50d / sma_200d - 1.0
            high_60d = max(adjusted_closes_float[-60:])
            high_120d = max(adjusted_closes_float[-120:])
            pct_below_60d_high = latest_adjusted_close / high_60d - 1.0
            pct_below_120d_high = latest_adjusted_close / high_120d - 1.0
            high_20d = max(adjusted_closes_float[-20:])
            low_20d = min(adjusted_closes_float[-20:])
            close_position_20d_range = (
                0.5
                if high_20d == low_20d
                else (latest_adjusted_close - low_20d) / (high_20d - low_20d)
            )
            max_drawdown_120d = max_drawdown(adjusted_closes_float[-120:])
            max_drawdown_252d = max_drawdown(adjusted_closes_float[-252:])
            trend_slope_60d, trend_r2_60d = ols_slope_r2(adjusted_closes_float[-60:])
            uptrend_r2_60d = trend_r2_60d if trend_slope_60d > 0.0 else 0.0
            volatility_20d_to_60d_ratio = volatility_20d / volatility_60d

            support_proximity_score_raw = max(
                band_score(
                    distance_from_sma_50d,
                    ideal_low=-0.06,
                    ideal_high=0.03,
                    outer_low=-0.16,
                    outer_high=0.12,
                ),
                band_score(
                    close_position_20d_range,
                    ideal_low=0.20,
                    ideal_high=0.55,
                    outer_low=0.0,
                    outer_high=0.90,
                ),
            )
            rebound_confirmation_score_raw = (
                0.45 * band_score(return_5d, ideal_low=0.0, ideal_high=0.08, outer_low=-0.10, outer_high=0.18)
                + 0.35 * band_score(close_position_20d_range, ideal_low=0.25, ideal_high=0.65, outer_low=0.0, outer_high=1.0)
                + 0.20 * band_score(traded_value_5d_to_20d_ratio, ideal_low=0.80, ideal_high=2.00, outer_low=0.45, outer_high=3.50)
            )
            liquidity_continuity_score_raw = band_score(
                traded_value_5d_to_20d_ratio,
                ideal_low=0.80,
                ideal_high=2.00,
                outer_low=0.45,
                outer_high=3.50,
            )

            has_reversion_setup = (
                return_20d < 0.0
                or distance_from_sma_20d < 0.0
                or close_position_20d_range <= 0.35
            )
            if not has_reversion_setup:
                continue
            if max_drawdown_120d <= -0.40 or max_drawdown_252d <= -0.50:
                continue
            if distance_from_sma_200d < -0.25 and trend_slope_60d <= 0.0:
                continue
            if volatility_20d_to_60d_ratio > 2.50 and close_position_20d_range <= 0.20:
                continue
            data_quality_extreme_return_flag = 1.0 if any(abs(value) > 0.50 for value in returns_60d) else 0.0

            computed = [
                avg_traded_value_20d_local,
                avg_traded_value_5d_local,
                median_traded_value_20d_local,
                traded_value_5d_to_20d_ratio,
                return_5d,
                return_20d,
                return_60d,
                distance_from_sma_20d,
                distance_from_sma_50d,
                distance_from_sma_200d,
                pct_below_60d_high,
                pct_below_120d_high,
                close_position_20d_range,
                downside_volatility_20d,
                volatility_20d,
                volatility_60d,
                volatility_20d_to_60d_ratio,
                max_drawdown_120d,
                max_drawdown_252d,
                support_proximity_score_raw,
                rebound_confirmation_score_raw,
                liquidity_continuity_score_raw,
                trend_slope_60d,
                uptrend_r2_60d,
                sma_20d,
                sma_50d,
                sma_200d,
                price_vs_sma_20d,
                price_vs_sma_50d,
                price_vs_sma_200d,
                sma_50d_vs_sma_200d,
                zero_volume_days_20d,
                active_trading_days_60d,
                stale_close_days_20d,
                data_quality_extreme_return_flag,
            ]
            if not all_finite(computed):
                continue

            rows.append(
                {
                    "run_id": run_id,
                    "market": market.value,
                    "ticker": ticker,
                    "close": latest_close,
                    "adjusted_close": latest_adjusted_close,
                    "profile_metrics_version": 1.0,
                    "asof_bar_date_yyyymmdd": yyyymmdd(profile_asof_bar_date),
                    "avg_traded_value_20d_local": avg_traded_value_20d_local,
                    "avg_traded_value_5d_local": avg_traded_value_5d_local,
                    "median_traded_value_20d_local": median_traded_value_20d_local,
                    "traded_value_5d_to_20d_ratio": traded_value_5d_to_20d_ratio,
                    "return_5d": return_5d,
                    "return_20d": return_20d,
                    "return_60d": return_60d,
                    "distance_from_sma_20d": distance_from_sma_20d,
                    "distance_from_sma_50d": distance_from_sma_50d,
                    "distance_from_sma_200d": distance_from_sma_200d,
                    "pct_below_60d_high": pct_below_60d_high,
                    "pct_below_120d_high": pct_below_120d_high,
                    "close_position_20d_range": close_position_20d_range,
                    "downside_volatility_20d": downside_volatility_20d,
                    "volatility_20d": volatility_20d,
                    "volatility_60d": volatility_60d,
                    "volatility_20d_to_60d_ratio": volatility_20d_to_60d_ratio,
                    "max_drawdown_120d": max_drawdown_120d,
                    "max_drawdown_252d": max_drawdown_252d,
                    "support_proximity_score_raw": support_proximity_score_raw,
                    "rebound_confirmation_score_raw": rebound_confirmation_score_raw,
                    "liquidity_continuity_score_raw": liquidity_continuity_score_raw,
                    "trend_slope_60d": trend_slope_60d,
                    "uptrend_r2_60d": uptrend_r2_60d,
                    "sma_20d": sma_20d,
                    "sma_50d": sma_50d,
                    "sma_200d": sma_200d,
                    "price_vs_sma_20d": price_vs_sma_20d,
                    "price_vs_sma_50d": price_vs_sma_50d,
                    "price_vs_sma_200d": price_vs_sma_200d,
                    "sma_50d_vs_sma_200d": sma_50d_vs_sma_200d,
                    "zero_volume_days_20d": zero_volume_days_20d,
                    "active_trading_days_60d": active_trading_days_60d,
                    "stale_close_days_20d": stale_close_days_20d,
                    "data_quality_extreme_return_flag": data_quality_extreme_return_flag,
                }
            )

        if not rows:
            return _empty_snapshot()
        return pl.DataFrame(rows, schema=MEAN_REVERSION_QUALITY_SNAPSHOT_SCHEMA).sort("ticker")

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        if snapshot.is_empty():
            return _empty_rankings()
        if not self._has_required_ranking_inputs(snapshot):
            raise ValidationError("mean_reversion_quality_v1 snapshot is missing required ranking inputs")
        for column in ("close", "adjusted_close", *MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS):
            if not snapshot.schema[column].is_numeric():
                raise ValidationError(
                    f"mean_reversion_quality_v1 snapshot contains non-numeric ranking input: {column}"
                )
            invalid_count = snapshot.filter(pl.col(column).is_null() | (~pl.col(column).is_finite())).height
            if invalid_count > 0:
                raise ValidationError(
                    f"mean_reversion_quality_v1 snapshot contains non-finite ranking input: {column}"
                )

        ranking_frames = []
        for partition in snapshot.partition_by(["run_id", "market"], maintain_order=True):
            if partition["asof_bar_date_yyyymmdd"].n_unique() != 1:
                raise ValidationError("mean_reversion_quality_v1 snapshot contains mixed as-of dates")
            ranking_frames.append(self._assign_single_run_market_rankings(partition))
        if not ranking_frames:
            return _empty_rankings()
        return (
            pl.concat(ranking_frames)
            .with_columns(
                pl.col("horizon")
                .replace_strict({horizon: index for index, horizon in enumerate(self.horizon_order)})
                .alias("_horizon_order")
            )
            .sort(["run_id", "market", "_horizon_order", "rank", "ticker"])
            .drop("_horizon_order")
            .select(self._ranking_columns())
        )

    def _has_required_ranking_inputs(self, snapshot: pl.DataFrame) -> bool:
        required = {
            "run_id",
            "market",
            "ticker",
            "close",
            "adjusted_close",
            *MEAN_REVERSION_QUALITY_SNAPSHOT_METRIC_KEYS,
        }
        return required.issubset(set(snapshot.columns))

    def _ranking_columns(self) -> list[str]:
        return [
            "run_id",
            "market",
            "horizon",
            "ticker",
            *self.ranking_metric_keys,
            "score",
            "rank",
        ]

    def _assign_single_run_market_rankings(self, frame: pl.DataFrame) -> pl.DataFrame:
        rows = frame.sort("ticker").to_dicts()
        score_drawdown_control = percentile_scores([float(row["max_drawdown_120d"]) for row in rows])
        score_volatility_control_rank = percentile_scores(
            [float(row["volatility_20d_to_60d_ratio"]) for row in rows],
            higher_is_better=False,
        )
        score_downside_volatility = percentile_scores(
            [float(row["downside_volatility_20d"]) for row in rows],
            higher_is_better=False,
        )
        score_trend_preservation_rank = percentile_scores([float(row["price_vs_sma_200d"]) for row in rows])

        scored_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows):
            return_5d = float(row["return_5d"])
            return_20d = float(row["return_20d"])
            distance_from_sma_20d = float(row["distance_from_sma_20d"])
            distance_from_sma_50d = float(row["distance_from_sma_50d"])
            distance_from_sma_200d = float(row["distance_from_sma_200d"])
            close_position_20d_range = clamp(float(row["close_position_20d_range"]), 0.0, 1.0)
            volatility_20d_to_60d_ratio = float(row["volatility_20d_to_60d_ratio"])
            max_drawdown_120d = float(row["max_drawdown_120d"])
            max_drawdown_252d = float(row["max_drawdown_252d"])
            support_proximity_score_raw = float(row["support_proximity_score_raw"])
            rebound_confirmation_score_raw = float(row["rebound_confirmation_score_raw"])
            liquidity_continuity_score_raw = float(row["liquidity_continuity_score_raw"])
            trend_slope_60d = float(row["trend_slope_60d"])
            uptrend_r2_60d = float(row["uptrend_r2_60d"])
            price_vs_sma_50d = float(row["price_vs_sma_50d"])
            price_vs_sma_200d = float(row["price_vs_sma_200d"])
            sma_50d_vs_sma_200d = float(row["sma_50d_vs_sma_200d"])
            traded_value_5d_to_20d_ratio = float(row["traded_value_5d_to_20d_ratio"])
            data_quality_extreme_return_flag = float(row["data_quality_extreme_return_flag"])

            score_oversold_depth = max(
                band_score(return_20d, ideal_low=-0.18, ideal_high=-0.04, outer_low=-0.35, outer_high=0.04),
                band_score(distance_from_sma_20d, ideal_low=-0.12, ideal_high=-0.02, outer_low=-0.25, outer_high=0.04),
                band_score(close_position_20d_range, ideal_low=0.10, ideal_high=0.40, outer_low=0.0, outer_high=0.75),
            )
            score_support_proximity = support_proximity_score_raw
            score_rebound_confirmation = rebound_confirmation_score_raw
            score_volatility_control = (
                0.55 * score_volatility_control_rank[index]
                + 0.45 * score_downside_volatility[index]
            )
            score_liquidity_continuity = liquidity_continuity_score_raw
            score_trend_preservation = (
                0.35 * score_trend_preservation_rank[index]
                + 0.25 * clamp((price_vs_sma_200d + 0.25) / 0.50, 0.0, 1.0)
                + 0.20 * clamp((sma_50d_vs_sma_200d + 0.08) / 0.28, 0.0, 1.0)
                + 0.20 * clamp((uptrend_r2_60d + 0.10) / 1.10, 0.0, 1.0)
            )
            oversold_quality_score = (
                0.45 * score_oversold_depth
                + 0.25 * score_support_proximity
                + 0.15 * score_rebound_confirmation
                + 0.15 * score_trend_preservation
            )
            support_reversion_score = (
                0.45 * score_support_proximity
                + 0.25 * score_oversold_depth
                + 0.15 * score_rebound_confirmation
                + 0.15 * score_trend_preservation
            )
            rebound_quality_score = (
                0.45 * score_rebound_confirmation
                + 0.25 * score_oversold_depth
                + 0.15 * score_support_proximity
                + 0.15 * score_liquidity_continuity
            )
            risk_control_score = (
                0.35 * score_drawdown_control[index]
                + 0.30 * score_volatility_control
                + 0.20 * score_liquidity_continuity
                + 0.15 * score_trend_preservation
            )

            tag_setup_oversold_quality = (
                1.0
                if score_oversold_depth >= 0.65
                and score_trend_preservation >= 0.45
                and max_drawdown_120d > -0.30
                else 0.0
            )
            tag_setup_near_support = 1.0 if score_support_proximity >= 0.65 else 0.0
            tag_setup_rebound_confirmation = 1.0 if score_rebound_confirmation >= 0.75 and return_5d > 0.0 else 0.0
            tag_risk_falling_knife = (
                1.0
                if return_5d < -0.04
                or close_position_20d_range < 0.12
                or (return_20d < -0.22 and score_rebound_confirmation < 0.35)
                else 0.0
            )
            tag_risk_breakdown = (
                1.0
                if price_vs_sma_200d < -0.25
                or (trend_slope_60d <= 0.0 and sma_50d_vs_sma_200d < 0.0)
                or price_vs_sma_50d < -0.16
                else 0.0
            )
            tag_risk_deep_drawdown = 1.0 if max_drawdown_120d < -0.30 or max_drawdown_252d < -0.40 else 0.0
            tag_risk_volatility_spike = 1.0 if volatility_20d_to_60d_ratio > 1.80 else 0.0
            tag_risk_liquidity_fade = 1.0 if traded_value_5d_to_20d_ratio < 0.75 else 0.0
            tag_risk_no_reversion_setup = (
                1.0
                if return_20d >= 0.0 and distance_from_sma_20d >= 0.0 and close_position_20d_range > 0.35
                else 0.0
            )
            tag_risk_data_quality_warning = 1.0 if data_quality_extreme_return_flag == 1.0 else 0.0

            structure_cap_score = self._structure_cap_score(
                price_vs_sma_200d=price_vs_sma_200d,
                price_vs_sma_50d=price_vs_sma_50d,
                sma_50d_vs_sma_200d=sma_50d_vs_sma_200d,
                trend_slope_60d=trend_slope_60d,
            )
            falling_knife_cap_score = self._falling_knife_cap_score(
                return_5d=return_5d,
                return_20d=return_20d,
                close_position_20d_range=close_position_20d_range,
                rebound_confirmation_score=score_rebound_confirmation,
            )
            volatility_cap_score = self._volatility_cap_score(
                volatility_20d_to_60d_ratio=volatility_20d_to_60d_ratio
            )
            drawdown_cap_score = self._drawdown_cap_score(
                max_drawdown_120d=max_drawdown_120d,
                max_drawdown_252d=max_drawdown_252d,
            )

            penalty_score = 0.0
            penalty_score += 0.18 * tag_risk_falling_knife
            penalty_score += 0.18 * tag_risk_breakdown
            penalty_score += 0.12 * tag_risk_deep_drawdown
            penalty_score += 0.12 * tag_risk_volatility_spike
            penalty_score += 0.08 * tag_risk_liquidity_fade
            penalty_score += 0.12 * tag_risk_no_reversion_setup
            penalty_score += 0.20 * tag_risk_data_quality_warning

            row.update(
                {
                    "score_oversold_depth": score_oversold_depth,
                    "score_support_proximity": score_support_proximity,
                    "score_rebound_confirmation": score_rebound_confirmation,
                    "score_drawdown_control": score_drawdown_control[index],
                    "score_volatility_control": score_volatility_control,
                    "score_liquidity_continuity": score_liquidity_continuity,
                    "score_trend_preservation": score_trend_preservation,
                    "oversold_quality_score": oversold_quality_score,
                    "support_reversion_score": support_reversion_score,
                    "rebound_quality_score": rebound_quality_score,
                    "risk_control_score": risk_control_score,
                    "structure_cap_score": structure_cap_score,
                    "falling_knife_cap_score": falling_knife_cap_score,
                    "volatility_cap_score": volatility_cap_score,
                    "drawdown_cap_score": drawdown_cap_score,
                    "penalty_score": penalty_score,
                    "tag_setup_oversold_quality": tag_setup_oversold_quality,
                    "tag_setup_near_support": tag_setup_near_support,
                    "tag_setup_rebound_confirmation": tag_setup_rebound_confirmation,
                    "tag_risk_falling_knife": tag_risk_falling_knife,
                    "tag_risk_breakdown": tag_risk_breakdown,
                    "tag_risk_deep_drawdown": tag_risk_deep_drawdown,
                    "tag_risk_volatility_spike": tag_risk_volatility_spike,
                    "tag_risk_liquidity_fade": tag_risk_liquidity_fade,
                    "tag_risk_no_reversion_setup": tag_risk_no_reversion_setup,
                    "tag_risk_data_quality_warning": tag_risk_data_quality_warning,
                }
            )
            scored_rows.append(row)

        ranking_rows: list[dict[str, object]] = []
        for horizon in self.horizon_order:
            horizon_rows = []
            for row in scored_rows:
                if horizon == "composite":
                    raw_score = (
                        0.30 * float(row["oversold_quality_score"])
                        + 0.25 * float(row["support_reversion_score"])
                        + 0.20 * float(row["rebound_quality_score"])
                        + 0.25 * float(row["risk_control_score"])
                    )
                elif horizon == "oversold_bounce":
                    raw_score = (
                        0.40 * float(row["oversold_quality_score"])
                        + 0.30 * float(row["rebound_quality_score"])
                        + 0.15 * float(row["support_reversion_score"])
                        + 0.15 * float(row["risk_control_score"])
                    )
                elif horizon == "support_reversion":
                    raw_score = (
                        0.40 * float(row["support_reversion_score"])
                        + 0.25 * float(row["oversold_quality_score"])
                        + 0.20 * float(row["risk_control_score"])
                        + 0.15 * float(row["rebound_quality_score"])
                    )
                else:
                    raise ValidationError(f"unknown horizon {horizon}")
                score = min(
                    raw_score - float(row["penalty_score"]),
                    float(row["structure_cap_score"]),
                    float(row["falling_knife_cap_score"]),
                    float(row["volatility_cap_score"]),
                    float(row["drawdown_cap_score"]),
                )
                ranking_values = [float(row[key]) for key in self.ranking_metric_keys] + [score]
                if not all_finite(ranking_values):
                    raise ValidationError("mean_reversion_quality_v1 produced a non-finite ranking metric")
                horizon_rows.append(
                    {
                        "run_id": row["run_id"],
                        "market": row["market"],
                        "horizon": horizon,
                        "ticker": row["ticker"],
                        **{key: row[key] for key in self.ranking_metric_keys},
                        "score": score,
                    }
                )
            horizon_rows.sort(key=lambda item: (-float(item["score"]), str(item["ticker"])))
            for rank, row in enumerate(horizon_rows, start=1):
                row["rank"] = rank
                ranking_rows.append(row)
        return pl.DataFrame(ranking_rows, schema=MEAN_REVERSION_QUALITY_RANKING_SCHEMA).select(
            self._ranking_columns()
        )

    def _structure_cap_score(
        self,
        *,
        price_vs_sma_200d: float,
        price_vs_sma_50d: float,
        sma_50d_vs_sma_200d: float,
        trend_slope_60d: float,
    ) -> float:
        if price_vs_sma_200d < -0.25 or (trend_slope_60d <= 0.0 and sma_50d_vs_sma_200d < 0.0):
            return 0.40
        if price_vs_sma_50d < -0.16 or sma_50d_vs_sma_200d < -0.04:
            return 0.60
        if price_vs_sma_50d < -0.10 or trend_slope_60d < 0.0:
            return 0.80
        return 1.0

    def _falling_knife_cap_score(
        self,
        *,
        return_5d: float,
        return_20d: float,
        close_position_20d_range: float,
        rebound_confirmation_score: float,
    ) -> float:
        if return_5d < -0.08 or (return_20d < -0.28 and close_position_20d_range < 0.12):
            return 0.35
        if return_5d < -0.04 or close_position_20d_range < 0.12 or rebound_confirmation_score < 0.25:
            return 0.55
        if return_20d < -0.22 or close_position_20d_range < 0.20:
            return 0.75
        return 1.0

    def _volatility_cap_score(self, *, volatility_20d_to_60d_ratio: float) -> float:
        if volatility_20d_to_60d_ratio > 2.30:
            return 0.40
        if volatility_20d_to_60d_ratio > 1.80:
            return 0.65
        if volatility_20d_to_60d_ratio > 1.45:
            return 0.82
        return 1.0

    def _drawdown_cap_score(self, *, max_drawdown_120d: float, max_drawdown_252d: float) -> float:
        if max_drawdown_120d < -0.38 or max_drawdown_252d < -0.48:
            return 0.40
        if max_drawdown_120d < -0.30 or max_drawdown_252d < -0.40:
            return 0.65
        if max_drawdown_120d < -0.24 or max_drawdown_252d < -0.34:
            return 0.82
        return 1.0


MEAN_REVERSION_QUALITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=MEAN_REVERSION_QUALITY_PROFILE_ID,
    factory=MeanReversionQualityV1Profile,
)
