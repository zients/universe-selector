from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Literal, Mapping

import polars as pl

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalsUniverseRunData, ListingCandidate
from universe_selector.ranking_profiles._alpha_common import (
    all_finite,
    band_score,
    clamp,
    immutable_market_float_mapping,
    immutable_market_int_mapping,
    max_drawdown,
    mean,
    metric_float,
    ols_slope_r2,
    percentile_scores,
    positive_only_percentile_scores,
    std,
    yyyymmdd,
)
from universe_selector.ranking_profiles.registration import RankingProfileRegistration


BASE_BREAKOUT_QUALITY_PROFILE_ID: Literal["base_breakout_quality_v1"] = "base_breakout_quality_v1"
BASE_BREAKOUT_QUALITY_SCORE_METHOD = "market_relative_base_breakout_quality_v1"
BASE_BREAKOUT_QUALITY_RANK_INTERPRETATION_NOTE = (
    "Base breakout quality scores are market-local relative scores for constructive bases near "
    "breakout or early breakout confirmation. They are ranking lenses, not buy signals, return "
    "forecasts, backtests, or holding-period recommendations."
)
BASE_BREAKOUT_QUALITY_HORIZON_ORDER = ("composite", "near_breakout", "breakout_readiness")
BASE_BREAKOUT_SETUP_PROXIMITY_MIN = -0.15
BASE_BREAKOUT_SETUP_PROXIMITY_MAX = 0.08
BASE_BREAKOUT_CONFIRMED_PROXIMITY_MIN = -0.005

BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS = (
    "profile_metrics_version",
    "asof_bar_date_yyyymmdd",
    "avg_traded_value_20d_local",
    "avg_traded_value_5d_local",
    "median_traded_value_20d_local",
    "traded_value_5d_to_20d_ratio",
    "return_20d",
    "return_60d",
    "return_120d",
    "volatility_20d",
    "volatility_60d",
    "volatility_20d_to_60d_ratio",
    "trend_slope_60d",
    "uptrend_r2_60d",
    "trend_consistency_60d",
    "sma_20d",
    "sma_50d",
    "sma_200d",
    "price_vs_sma_20d",
    "price_vs_sma_50d",
    "price_vs_sma_200d",
    "sma_20d_vs_sma_50d",
    "sma_50d_vs_sma_200d",
    "pct_below_60d_high",
    "pct_below_120d_high",
    "base_depth_60d",
    "base_depth_120d",
    "base_tightness_20d",
    "base_tightness_60d",
    "close_position_20d_range",
    "zero_volume_days_20d",
    "active_trading_days_60d",
    "stale_close_days_20d",
    "data_quality_extreme_return_flag",
)

BASE_BREAKOUT_QUALITY_RANKING_METRIC_KEYS = (
    "score_base_tightness",
    "score_base_depth",
    "score_breakout_proximity",
    "score_trend_structure",
    "score_volume_confirmation",
    "score_risk_control",
    "score_prior_strength",
    "base_quality_score",
    "breakout_setup_score",
    "trend_structure_score",
    "volume_confirmation_score",
    "risk_control_score",
    "prior_strength_score",
    "structure_cap_score",
    "breakout_extension_cap_score",
    "base_depth_cap_score",
    "false_breakout_cap_score",
    "penalty_score",
    "tag_setup_valid_base",
    "tag_setup_near_breakout",
    "tag_setup_confirmed_breakout",
    "tag_risk_false_breakout",
    "tag_risk_overextended_breakout",
    "tag_risk_weak_base",
    "tag_risk_breakdown",
    "tag_risk_liquidity_fade",
    "tag_risk_data_quality_warning",
)

BASE_BREAKOUT_QUALITY_SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    **{key: pl.Float64 for key in BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS},
}

BASE_BREAKOUT_QUALITY_RANKING_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "horizon": pl.String,
    "ticker": pl.String,
    **{key: pl.Float64 for key in BASE_BREAKOUT_QUALITY_RANKING_METRIC_KEYS},
    "score": pl.Float64,
    "rank": pl.Int64,
}


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=BASE_BREAKOUT_QUALITY_SNAPSHOT_SCHEMA)


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema=BASE_BREAKOUT_QUALITY_RANKING_SCHEMA)


@dataclass(frozen=True)
class BaseBreakoutQualityV1Profile:
    profile_id: Literal["base_breakout_quality_v1"] = BASE_BREAKOUT_QUALITY_PROFILE_ID
    min_history_bars: int = 252
    price_floor: Mapping[Market, float] = field(default_factory=lambda: {Market.TW: 10.0, Market.US: 5.0})
    liquidity_floor: Mapping[Market, float] = field(
        default_factory=lambda: {Market.TW: 50_000_000.0, Market.US: 10_000_000.0}
    )
    active_trading_min_days_60: Mapping[Market, int] = field(default_factory=lambda: {Market.TW: 50, Market.US: 55})
    zero_volume_max_days_20: Mapping[Market, int] = field(default_factory=lambda: {Market.TW: 3, Market.US: 1})
    stale_close_max_days_20: int = 5
    extreme_return_abs_cutoff: float = 0.80
    volatility_floor: float = 0.0001
    snapshot_metric_keys: tuple[str, ...] = BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = BASE_BREAKOUT_QUALITY_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS
    horizon_order: tuple[str, ...] = BASE_BREAKOUT_QUALITY_HORIZON_ORDER
    stdev_ddof: int = 1
    score_method: str = BASE_BREAKOUT_QUALITY_SCORE_METHOD
    rank_interpretation_note: str = BASE_BREAKOUT_QUALITY_RANK_INTERPRETATION_NOTE

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
        if self.profile_id != BASE_BREAKOUT_QUALITY_PROFILE_ID:
            raise ValidationError("profile_id must be base_breakout_quality_v1")
        if self.min_history_bars != 252:
            raise ValidationError("base_breakout_quality_v1 requires min_history_bars to be 252")
        if self.snapshot_metric_keys != BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("snapshot metric keys must match base_breakout_quality_v1")
        if self.ranking_metric_keys != BASE_BREAKOUT_QUALITY_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match base_breakout_quality_v1")
        if self.inspect_metric_keys != BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("inspect metric keys must match base_breakout_quality_v1")
        if self.horizon_order != BASE_BREAKOUT_QUALITY_HORIZON_ORDER:
            raise ValidationError("horizon order must be composite, near_breakout, breakout_readiness")
        for mapping_name in (
            "price_floor",
            "liquidity_floor",
            "active_trading_min_days_60",
            "zero_volume_max_days_20",
        ):
            if set(getattr(self, mapping_name)) != set(Market):
                raise ValidationError(f"{mapping_name} must contain exactly supported markets")
        if {market: self.price_floor[market] for market in Market} != {Market.TW: 10.0, Market.US: 5.0}:
            raise ValidationError("price_floor must match base_breakout_quality_v1")
        if {market: self.liquidity_floor[market] for market in Market} != {
            Market.TW: 50_000_000.0,
            Market.US: 10_000_000.0,
        }:
            raise ValidationError("liquidity_floor must match base_breakout_quality_v1")
        if {market: self.active_trading_min_days_60[market] for market in Market} != {
            Market.TW: 50,
            Market.US: 55,
        }:
            raise ValidationError("active_trading_min_days_60 must match base_breakout_quality_v1")
        if {market: self.zero_volume_max_days_20[market] for market in Market} != {
            Market.TW: 3,
            Market.US: 1,
        }:
            raise ValidationError("zero_volume_max_days_20 must match base_breakout_quality_v1")
        if self.stale_close_max_days_20 != 5:
            raise ValidationError("stale_close_max_days_20 must be 5")
        if self.extreme_return_abs_cutoff != 0.80:
            raise ValidationError("extreme_return_abs_cutoff must be 0.80")
        if self.volatility_floor != 0.0001:
            raise ValidationError("volatility_floor must be 0.0001")
        if self.stdev_ddof != 1:
            raise ValidationError("stdev_ddof must be 1")
        if self.score_method != BASE_BREAKOUT_QUALITY_SCORE_METHOD:
            raise ValidationError(f"score_method must be {BASE_BREAKOUT_QUALITY_SCORE_METHOD}")
        if self.rank_interpretation_note != BASE_BREAKOUT_QUALITY_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match base_breakout_quality_v1")

    def ranking_config_payload(self) -> dict[str, object]:
        return {
            "ranking_profile": self.profile_id,
            "min_history_bars": self.min_history_bars,
            "price_floor": {market.value: self.price_floor[market] for market in Market},
            "liquidity_floor": {market.value: self.liquidity_floor[market] for market in Market},
            "active_trading_min_days_60": {market.value: self.active_trading_min_days_60[market] for market in Market},
            "zero_volume_max_days_20": {market.value: self.zero_volume_max_days_20[market] for market in Market},
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
        fundamentals: FundamentalsUniverseRunData | None = None,
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
            ticker_bars = candidate_bars.filter(
                (pl.col("ticker") == ticker) & (pl.col("bar_date") <= profile_asof_bar_date)
            ).sort("bar_date")
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
            if any(
                value <= 0.0 for value in opens_float + highs_float + lows_float + closes_float + adjusted_closes_float
            ):
                continue
            if any(value < 0.0 for value in volumes_float):
                continue
            if any(
                high < low or high < open_ or high < close or low > open_ or low > close
                for high, low, open_, close in zip(highs_float, lows_float, opens_float, closes_float, strict=True)
            ):
                continue

            latest_close = closes_float[-1]
            latest_adjusted_close = adjusted_closes_float[-1]
            traded_values = [close * volume for close, volume in zip(closes_float, volumes_float, strict=True)]
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
            if volatility_20d is None or volatility_60d is None:
                continue
            if volatility_20d <= 0.0 or volatility_60d <= self.volatility_floor:
                continue

            trend_slope_60d, trend_r2_60d = ols_slope_r2(adjusted_closes_float[-60:])
            uptrend_r2_60d = trend_r2_60d if trend_slope_60d > 0.0 else 0.0
            trend_consistency_60d = float(sum(1 for value in returns_60d if value > 0.0)) / 60.0
            sma_20d = mean(adjusted_closes_float[-20:])
            sma_50d = mean(adjusted_closes_float[-50:])
            sma_200d = mean(adjusted_closes_float[-200:])
            price_vs_sma_20d = latest_adjusted_close / sma_20d - 1.0
            price_vs_sma_50d = latest_adjusted_close / sma_50d - 1.0
            price_vs_sma_200d = latest_adjusted_close / sma_200d - 1.0
            sma_20d_vs_sma_50d = sma_20d / sma_50d - 1.0
            sma_50d_vs_sma_200d = sma_50d / sma_200d - 1.0
            high_60d = max(adjusted_closes_float[-61:-1])
            high_120d = max(adjusted_closes_float[-121:-1])
            pct_below_60d_high = latest_adjusted_close / high_60d - 1.0
            pct_below_120d_high = latest_adjusted_close / high_120d - 1.0
            base_depth_60d = min(adjusted_closes_float[-60:]) / high_60d - 1.0
            base_depth_120d = min(adjusted_closes_float[-120:]) / high_120d - 1.0
            high_20d = max(adjusted_closes_float[-20:])
            low_20d = min(adjusted_closes_float[-20:])
            high_60d_range = max(adjusted_closes_float[-60:])
            low_60d_range = min(adjusted_closes_float[-60:])
            base_tightness_20d = 1.0 - clamp((high_20d / low_20d - 1.0) / 0.20, 0.0, 1.0)
            base_tightness_60d = 1.0 - clamp((high_60d_range / low_60d_range - 1.0) / 0.35, 0.0, 1.0)
            close_position_20d_range = (
                0.5 if high_20d == low_20d else (latest_adjusted_close - low_20d) / (high_20d - low_20d)
            )
            return_20d = latest_adjusted_close / adjusted_closes_float[-21] - 1.0
            return_60d = latest_adjusted_close / adjusted_closes_float[-61] - 1.0
            return_120d = latest_adjusted_close / adjusted_closes_float[-121] - 1.0
            max_drawdown_120d = max_drawdown(adjusted_closes_float[-120:])

            if price_vs_sma_200d <= 0.0 or sma_50d_vs_sma_200d < 0.0:
                continue
            if trend_slope_60d <= 0.0:
                continue
            if max_drawdown_120d <= -0.35:
                continue
            if (
                pct_below_120d_high < BASE_BREAKOUT_SETUP_PROXIMITY_MIN
                or pct_below_120d_high > BASE_BREAKOUT_SETUP_PROXIMITY_MAX
            ):
                continue
            data_quality_extreme_return_flag = 1.0 if any(abs(value) > 0.50 for value in returns_60d) else 0.0
            if not self._is_persisted_setup(
                base_depth_120d=base_depth_120d,
                base_depth_60d=base_depth_60d,
                base_tightness_20d=base_tightness_20d,
                base_tightness_60d=base_tightness_60d,
                trend_slope_60d=trend_slope_60d,
                price_vs_sma_200d=price_vs_sma_200d,
                sma_50d_vs_sma_200d=sma_50d_vs_sma_200d,
                return_60d=return_60d,
                return_120d=return_120d,
                pct_below_120d_high=pct_below_120d_high,
                close_position_20d_range=close_position_20d_range,
                traded_value_5d_to_20d_ratio=traded_value_5d_to_20d_ratio,
            ):
                continue

            computed = [
                avg_traded_value_20d_local,
                avg_traded_value_5d_local,
                median_traded_value_20d_local,
                traded_value_5d_to_20d_ratio,
                return_20d,
                return_60d,
                return_120d,
                volatility_20d,
                volatility_60d,
                volatility_20d_to_60d_ratio := volatility_20d / volatility_60d,
                trend_slope_60d,
                uptrend_r2_60d,
                trend_consistency_60d,
                sma_20d,
                sma_50d,
                sma_200d,
                price_vs_sma_20d,
                price_vs_sma_50d,
                price_vs_sma_200d,
                sma_20d_vs_sma_50d,
                sma_50d_vs_sma_200d,
                pct_below_60d_high,
                pct_below_120d_high,
                base_depth_60d,
                base_depth_120d,
                base_tightness_20d,
                base_tightness_60d,
                close_position_20d_range,
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
                    "return_20d": return_20d,
                    "return_60d": return_60d,
                    "return_120d": return_120d,
                    "volatility_20d": volatility_20d,
                    "volatility_60d": volatility_60d,
                    "volatility_20d_to_60d_ratio": volatility_20d_to_60d_ratio,
                    "trend_slope_60d": trend_slope_60d,
                    "uptrend_r2_60d": uptrend_r2_60d,
                    "trend_consistency_60d": trend_consistency_60d,
                    "sma_20d": sma_20d,
                    "sma_50d": sma_50d,
                    "sma_200d": sma_200d,
                    "price_vs_sma_20d": price_vs_sma_20d,
                    "price_vs_sma_50d": price_vs_sma_50d,
                    "price_vs_sma_200d": price_vs_sma_200d,
                    "sma_20d_vs_sma_50d": sma_20d_vs_sma_50d,
                    "sma_50d_vs_sma_200d": sma_50d_vs_sma_200d,
                    "pct_below_60d_high": pct_below_60d_high,
                    "pct_below_120d_high": pct_below_120d_high,
                    "base_depth_60d": base_depth_60d,
                    "base_depth_120d": base_depth_120d,
                    "base_tightness_20d": base_tightness_20d,
                    "base_tightness_60d": base_tightness_60d,
                    "close_position_20d_range": close_position_20d_range,
                    "zero_volume_days_20d": zero_volume_days_20d,
                    "active_trading_days_60d": active_trading_days_60d,
                    "stale_close_days_20d": stale_close_days_20d,
                    "data_quality_extreme_return_flag": data_quality_extreme_return_flag,
                }
            )

        if not rows:
            return _empty_snapshot()
        return pl.DataFrame(rows, schema=BASE_BREAKOUT_QUALITY_SNAPSHOT_SCHEMA).sort("ticker")

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        if snapshot.is_empty():
            return _empty_rankings()
        if not self._has_required_ranking_inputs(snapshot):
            raise ValidationError("base_breakout_quality_v1 snapshot is missing required ranking inputs")
        for column in ("close", "adjusted_close", *BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS):
            if not snapshot.schema[column].is_numeric():
                raise ValidationError(f"base_breakout_quality_v1 snapshot contains non-numeric ranking input: {column}")
            invalid_count = snapshot.filter(pl.col(column).is_null() | (~pl.col(column).is_finite())).height
            if invalid_count > 0:
                raise ValidationError(f"base_breakout_quality_v1 snapshot contains non-finite ranking input: {column}")

        ranking_frames = []
        for partition in snapshot.partition_by(["run_id", "market"], maintain_order=True):
            if partition["asof_bar_date_yyyymmdd"].n_unique() != 1:
                raise ValidationError("base_breakout_quality_v1 snapshot contains mixed as-of dates")
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
            *BASE_BREAKOUT_QUALITY_SNAPSHOT_METRIC_KEYS,
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
        score_trend_slope = positive_only_percentile_scores([metric_float(row["trend_slope_60d"]) for row in rows])
        score_sma_50d_vs_sma_200d = percentile_scores([metric_float(row["sma_50d_vs_sma_200d"]) for row in rows])
        score_price_vs_sma_200d = percentile_scores([metric_float(row["price_vs_sma_200d"]) for row in rows])
        score_trend_consistency = percentile_scores([metric_float(row["trend_consistency_60d"]) for row in rows])
        score_return_60d = positive_only_percentile_scores([metric_float(row["return_60d"]) for row in rows])
        score_return_120d = positive_only_percentile_scores([metric_float(row["return_120d"]) for row in rows])
        score_drawdown_control = percentile_scores([metric_float(row["base_depth_120d"]) for row in rows])
        score_volatility_ratio = percentile_scores(
            [metric_float(row["volatility_20d_to_60d_ratio"]) for row in rows],
            higher_is_better=False,
        )

        scored_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows):
            base_depth_60d = metric_float(row["base_depth_60d"])
            base_depth_120d = metric_float(row["base_depth_120d"])
            base_tightness_20d = metric_float(row["base_tightness_20d"])
            base_tightness_60d = metric_float(row["base_tightness_60d"])
            pct_below_120d_high = metric_float(row["pct_below_120d_high"])
            close_position_20d_range = clamp(metric_float(row["close_position_20d_range"]), 0.0, 1.0)
            return_20d = metric_float(row["return_20d"])
            price_vs_sma_20d = metric_float(row["price_vs_sma_20d"])
            price_vs_sma_50d = metric_float(row["price_vs_sma_50d"])
            price_vs_sma_200d = metric_float(row["price_vs_sma_200d"])
            sma_50d_vs_sma_200d = metric_float(row["sma_50d_vs_sma_200d"])
            trend_slope_60d = metric_float(row["trend_slope_60d"])
            uptrend_r2_60d = metric_float(row["uptrend_r2_60d"])
            trend_consistency_60d = metric_float(row["trend_consistency_60d"])
            traded_value_5d_to_20d_ratio = metric_float(row["traded_value_5d_to_20d_ratio"])
            volatility_20d_to_60d_ratio = metric_float(row["volatility_20d_to_60d_ratio"])
            stale_close_days_20d = metric_float(row["stale_close_days_20d"])
            data_quality_extreme_return_flag = metric_float(row["data_quality_extreme_return_flag"])

            score_base_tightness = 0.60 * base_tightness_20d + 0.40 * base_tightness_60d
            score_base_depth = band_score(
                base_depth_120d,
                ideal_low=-0.20,
                ideal_high=-0.06,
                outer_low=-0.35,
                outer_high=0.02,
            )
            score_breakout_proximity = band_score(
                pct_below_120d_high,
                ideal_low=-0.05,
                ideal_high=0.03,
                outer_low=BASE_BREAKOUT_SETUP_PROXIMITY_MIN,
                outer_high=BASE_BREAKOUT_SETUP_PROXIMITY_MAX,
            )
            score_volume_confirmation = band_score(
                traded_value_5d_to_20d_ratio,
                ideal_low=0.90,
                ideal_high=2.00,
                outer_low=0.45,
                outer_high=3.50,
            )
            score_prior_strength = 0.45 * score_return_60d[index] + 0.55 * score_return_120d[index]
            score_trend_structure = (
                0.20 * score_trend_slope[index]
                + 0.15 * uptrend_r2_60d
                + 0.15 * score_sma_50d_vs_sma_200d[index]
                + 0.15 * score_price_vs_sma_200d[index]
                + 0.10 * score_trend_consistency[index]
                + 0.25 * score_prior_strength
            )
            score_risk_control = (
                0.40 * score_drawdown_control[index]
                + 0.35 * score_volatility_ratio[index]
                + 0.25 * clamp(traded_value_5d_to_20d_ratio, 0.0, 1.0)
            )

            base_quality_score = 0.50 * score_base_tightness + 0.35 * score_base_depth + 0.15 * close_position_20d_range
            breakout_setup_score = (
                0.55 * score_breakout_proximity + 0.25 * close_position_20d_range + 0.20 * score_volume_confirmation
            )
            trend_structure_score = score_trend_structure
            volume_confirmation_score = score_volume_confirmation
            risk_control_score = score_risk_control
            prior_strength_score = score_prior_strength

            tag_setup_valid_base = (
                1.0
                if -0.30 <= base_depth_120d <= -0.02
                and base_depth_60d <= 0.0
                and base_tightness_20d >= 0.30
                and base_tightness_60d >= 0.25
                and trend_slope_60d > 0.0
                and price_vs_sma_200d > 0.0
                and price_vs_sma_200d >= 0.05
                and sma_50d_vs_sma_200d >= 0.0
                and metric_float(row["return_120d"]) >= 0.08
                and (metric_float(row["return_60d"]) >= 0.0 or metric_float(row["return_120d"]) >= 0.12)
                else 0.0
            )
            tag_setup_near_breakout = (
                1.0
                if tag_setup_valid_base == 1.0
                and BASE_BREAKOUT_SETUP_PROXIMITY_MIN <= pct_below_120d_high <= BASE_BREAKOUT_SETUP_PROXIMITY_MAX
                else 0.0
            )
            tag_setup_confirmed_breakout = (
                1.0
                if tag_setup_valid_base == 1.0
                and BASE_BREAKOUT_CONFIRMED_PROXIMITY_MIN <= pct_below_120d_high <= BASE_BREAKOUT_SETUP_PROXIMITY_MAX
                and close_position_20d_range >= 0.70
                and traded_value_5d_to_20d_ratio >= 0.90
                else 0.0
            )
            tag_risk_false_breakout = (
                1.0
                if pct_below_120d_high >= -0.01
                and (close_position_20d_range < 0.45 or traded_value_5d_to_20d_ratio < 0.75)
                else 0.0
            )
            tag_risk_overextended_breakout = (
                1.0 if return_20d > 0.35 or price_vs_sma_50d > 0.20 or pct_below_120d_high > 0.08 else 0.0
            )
            tag_risk_weak_base = (
                1.0
                if base_depth_120d < -0.30
                or base_depth_120d > -0.02
                or base_depth_60d > 0.0
                or base_tightness_20d < 0.25
                or base_tightness_60d < 0.20
                or metric_float(row["return_120d"]) < 0.08
                or price_vs_sma_200d < 0.05
                else 0.0
            )
            tag_risk_breakdown = (
                1.0
                if price_vs_sma_20d < -0.08
                or price_vs_sma_50d < -0.03
                or price_vs_sma_200d <= 0.0
                or sma_50d_vs_sma_200d < 0.0
                or trend_slope_60d <= 0.0
                else 0.0
            )
            tag_risk_liquidity_fade = 1.0 if traded_value_5d_to_20d_ratio < 0.75 else 0.0
            tag_risk_data_quality_warning = 1.0 if data_quality_extreme_return_flag == 1.0 else 0.0

            structure_cap_score = self._structure_cap_score(
                trend_slope_60d=trend_slope_60d,
                uptrend_r2_60d=uptrend_r2_60d,
                trend_consistency_60d=trend_consistency_60d,
                price_vs_sma_200d=price_vs_sma_200d,
                sma_50d_vs_sma_200d=sma_50d_vs_sma_200d,
                return_60d=metric_float(row["return_60d"]),
                return_120d=metric_float(row["return_120d"]),
            )
            breakout_extension_cap_score = self._breakout_extension_cap_score(
                return_20d=return_20d,
                price_vs_sma_50d=price_vs_sma_50d,
                pct_below_120d_high=pct_below_120d_high,
            )
            base_depth_cap_score = self._base_depth_cap_score(
                base_depth_120d=base_depth_120d,
                base_depth_60d=base_depth_60d,
                base_tightness_20d=base_tightness_20d,
                base_tightness_60d=base_tightness_60d,
            )
            false_breakout_cap_score = self._false_breakout_cap_score(
                close_position_20d_range=close_position_20d_range,
                traded_value_5d_to_20d_ratio=traded_value_5d_to_20d_ratio,
                volatility_20d_to_60d_ratio=volatility_20d_to_60d_ratio,
            )

            penalty_score = 0.0
            penalty_score += 0.14 * tag_risk_false_breakout
            penalty_score += 0.12 * tag_risk_overextended_breakout
            penalty_score += 0.14 * tag_risk_weak_base
            penalty_score += 0.16 * tag_risk_breakdown
            penalty_score += 0.08 * tag_risk_liquidity_fade
            penalty_score += 0.20 * tag_risk_data_quality_warning
            if stale_close_days_20d > 0.0:
                penalty_score += 0.03

            row.update(
                {
                    "score_base_tightness": score_base_tightness,
                    "score_base_depth": score_base_depth,
                    "score_breakout_proximity": score_breakout_proximity,
                    "score_trend_structure": score_trend_structure,
                    "score_volume_confirmation": score_volume_confirmation,
                    "score_risk_control": score_risk_control,
                    "score_prior_strength": score_prior_strength,
                    "base_quality_score": base_quality_score,
                    "breakout_setup_score": breakout_setup_score,
                    "trend_structure_score": trend_structure_score,
                    "volume_confirmation_score": volume_confirmation_score,
                    "risk_control_score": risk_control_score,
                    "prior_strength_score": prior_strength_score,
                    "structure_cap_score": structure_cap_score,
                    "breakout_extension_cap_score": breakout_extension_cap_score,
                    "base_depth_cap_score": base_depth_cap_score,
                    "false_breakout_cap_score": false_breakout_cap_score,
                    "penalty_score": penalty_score,
                    "tag_setup_valid_base": tag_setup_valid_base,
                    "tag_setup_near_breakout": tag_setup_near_breakout,
                    "tag_setup_confirmed_breakout": tag_setup_confirmed_breakout,
                    "tag_risk_false_breakout": tag_risk_false_breakout,
                    "tag_risk_overextended_breakout": tag_risk_overextended_breakout,
                    "tag_risk_weak_base": tag_risk_weak_base,
                    "tag_risk_breakdown": tag_risk_breakdown,
                    "tag_risk_liquidity_fade": tag_risk_liquidity_fade,
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
                        0.25 * metric_float(row["base_quality_score"])
                        + 0.25 * metric_float(row["breakout_setup_score"])
                        + 0.20 * metric_float(row["trend_structure_score"])
                        + 0.15 * metric_float(row["volume_confirmation_score"])
                        + 0.15 * metric_float(row["risk_control_score"])
                    )
                elif horizon == "near_breakout":
                    raw_score = (
                        0.35 * metric_float(row["base_quality_score"])
                        + 0.35 * metric_float(row["breakout_setup_score"])
                        + 0.15 * metric_float(row["trend_structure_score"])
                        + 0.10 * metric_float(row["volume_confirmation_score"])
                        + 0.05 * metric_float(row["risk_control_score"])
                    )
                elif horizon == "breakout_readiness":
                    raw_score = (
                        0.10 * metric_float(row["base_quality_score"])
                        + 0.35 * metric_float(row["breakout_setup_score"])
                        + 0.25 * metric_float(row["trend_structure_score"])
                        + 0.20 * metric_float(row["volume_confirmation_score"])
                        + 0.10 * metric_float(row["risk_control_score"])
                    )
                else:
                    raise ValidationError(f"unknown horizon {horizon}")
                score = min(
                    raw_score - metric_float(row["penalty_score"]),
                    metric_float(row["structure_cap_score"]),
                    metric_float(row["breakout_extension_cap_score"]),
                    metric_float(row["base_depth_cap_score"]),
                    metric_float(row["false_breakout_cap_score"]),
                )
                ranking_values = [metric_float(row[key]) for key in self.ranking_metric_keys] + [score]
                if not all_finite(ranking_values):
                    raise ValidationError("base_breakout_quality_v1 produced a non-finite ranking metric")
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
            horizon_rows.sort(key=lambda item: (-metric_float(item["score"]), str(item["ticker"])))
            for rank, row in enumerate(horizon_rows, start=1):
                row["rank"] = rank
                ranking_rows.append(row)
        return pl.DataFrame(ranking_rows, schema=BASE_BREAKOUT_QUALITY_RANKING_SCHEMA).select(self._ranking_columns())

    def _is_persisted_setup(
        self,
        *,
        base_depth_120d: float,
        base_depth_60d: float,
        base_tightness_20d: float,
        base_tightness_60d: float,
        trend_slope_60d: float,
        price_vs_sma_200d: float,
        sma_50d_vs_sma_200d: float,
        return_60d: float,
        return_120d: float,
        pct_below_120d_high: float,
        close_position_20d_range: float,
        traded_value_5d_to_20d_ratio: float,
    ) -> bool:
        valid_base = (
            -0.30 <= base_depth_120d <= -0.02
            and base_depth_60d <= 0.0
            and base_tightness_20d >= 0.30
            and base_tightness_60d >= 0.25
            and trend_slope_60d > 0.0
            and price_vs_sma_200d >= 0.05
            and sma_50d_vs_sma_200d >= 0.0
            and return_120d >= 0.08
            and (return_60d >= 0.0 or return_120d >= 0.12)
        )
        if not valid_base:
            return False
        if traded_value_5d_to_20d_ratio < 0.50:
            return False
        near_breakout = BASE_BREAKOUT_SETUP_PROXIMITY_MIN <= pct_below_120d_high <= BASE_BREAKOUT_SETUP_PROXIMITY_MAX
        confirmed_breakout = (
            BASE_BREAKOUT_CONFIRMED_PROXIMITY_MIN <= pct_below_120d_high <= BASE_BREAKOUT_SETUP_PROXIMITY_MAX
            and close_position_20d_range >= 0.70
            and traded_value_5d_to_20d_ratio >= 0.90
        )
        return near_breakout or confirmed_breakout

    def _structure_cap_score(
        self,
        *,
        trend_slope_60d: float,
        uptrend_r2_60d: float,
        trend_consistency_60d: float,
        price_vs_sma_200d: float,
        sma_50d_vs_sma_200d: float,
        return_60d: float,
        return_120d: float,
    ) -> float:
        failures = sum(
            1
            for failed in (
                trend_slope_60d <= 0.0,
                price_vs_sma_200d <= 0.0,
                sma_50d_vs_sma_200d < 0.0,
            )
            if failed
        )
        if failures >= 2:
            return 0.35
        if failures == 1:
            return 0.55
        if return_120d < 0.08 or price_vs_sma_200d < 0.05:
            return 0.45
        if return_60d < 0.0 and return_120d < 0.12:
            return 0.70
        if uptrend_r2_60d < 0.15 or trend_consistency_60d < 0.45:
            return 0.75
        return 1.0

    def _breakout_extension_cap_score(
        self,
        *,
        return_20d: float,
        price_vs_sma_50d: float,
        pct_below_120d_high: float,
    ) -> float:
        if return_20d > 0.50 or price_vs_sma_50d > 0.30 or pct_below_120d_high > 0.15:
            return 0.45
        if return_20d > 0.35 or price_vs_sma_50d > 0.20 or pct_below_120d_high > 0.08:
            return 0.65
        if return_20d > 0.25 or price_vs_sma_50d > 0.15 or pct_below_120d_high > 0.05:
            return 0.80
        return 1.0

    def _base_depth_cap_score(
        self,
        *,
        base_depth_120d: float,
        base_depth_60d: float,
        base_tightness_20d: float,
        base_tightness_60d: float,
    ) -> float:
        if base_depth_120d > -0.02 or base_depth_60d > -0.01:
            return 0.50
        if base_depth_120d > -0.04 or base_depth_60d > -0.02:
            return 0.70
        if base_depth_120d < -0.30 or base_tightness_20d < 0.20:
            return 0.55
        if base_depth_120d < -0.25 or base_tightness_60d < 0.25:
            return 0.70
        if base_depth_120d < -0.20 or base_tightness_20d < 0.35:
            return 0.85
        return 1.0

    def _false_breakout_cap_score(
        self,
        *,
        close_position_20d_range: float,
        traded_value_5d_to_20d_ratio: float,
        volatility_20d_to_60d_ratio: float,
    ) -> float:
        if close_position_20d_range < 0.20 or traded_value_5d_to_20d_ratio < 0.50:
            return 0.55
        if close_position_20d_range < 0.35 or volatility_20d_to_60d_ratio > 2.0:
            return 0.70
        if close_position_20d_range < 0.45 or volatility_20d_to_60d_ratio > 1.60:
            return 0.85
        return 1.0


BASE_BREAKOUT_QUALITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=BASE_BREAKOUT_QUALITY_PROFILE_ID,
    factory=lambda: BaseBreakoutQualityV1Profile(),
)
