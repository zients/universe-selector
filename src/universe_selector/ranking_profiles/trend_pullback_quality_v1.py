from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from statistics import median
from types import MappingProxyType
from typing import Literal, Mapping

import polars as pl

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.registration import RankingProfileRegistration


TREND_PULLBACK_QUALITY_PROFILE_ID = "trend_pullback_quality_v1"
TREND_PULLBACK_QUALITY_SCORE_METHOD = "market_relative_trend_pullback_quality_v1"
TREND_PULLBACK_QUALITY_RANK_INTERPRETATION_NOTE = (
    "Trend pullback quality scores are market-local relative scores for strong stocks in orderly "
    "pullbacks. They are ranking lenses, not buy signals, return forecasts, backtests, or holding-period "
    "recommendations."
)
TREND_PULLBACK_QUALITY_HORIZON_ORDER = ("composite", "near_support", "trend_resume")

TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS = (
    "profile_metrics_version",
    "asof_bar_date_yyyymmdd",
    "avg_traded_value_20d_local",
    "avg_traded_value_5d_local",
    "median_traded_value_20d_local",
    "traded_value_5d_to_20d_ratio",
    "return_20d",
    "return_60d",
    "return_120d",
    "return_120d_ex_recent_20d",
    "risk_adjusted_return_120d_ex_recent_20d",
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
    "pullback_from_60d_high",
    "pullback_from_120d_high",
    "volatility_adjusted_pullback_120d",
    "days_since_60d_high",
    "days_since_120d_high",
    "max_drawdown_120d",
    "close_position_20d_range",
    "low_10d_vs_sma_50d",
    "zero_volume_days_20d",
    "active_trading_days_60d",
    "stale_close_days_20d",
    "data_quality_extreme_return_flag",
)

TREND_PULLBACK_QUALITY_RANKING_METRIC_KEYS = (
    "score_prior_strength",
    "score_trend_intact",
    "score_pullback_depth",
    "score_volatility_adjusted_pullback",
    "score_support_proximity",
    "score_pullback_stability",
    "score_liquidity_continuity",
    "prior_strength_score",
    "trend_intact_score",
    "pullback_setup_score",
    "support_quality_score",
    "risk_control_score",
    "structure_cap_score",
    "pullback_depth_cap_score",
    "overheat_cap_score",
    "breakdown_cap_score",
    "penalty_score",
    "tag_setup_healthy_pullback",
    "tag_setup_near_sma50",
    "tag_setup_trend_resume",
    "tag_risk_false_pullback",
    "tag_risk_breakdown",
    "tag_risk_deep_drawdown",
    "tag_risk_liquidity_fade",
    "tag_risk_still_overheated",
    "tag_risk_volatility_spike",
    "tag_risk_data_quality_warning",
)

TREND_PULLBACK_QUALITY_SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    **{key: pl.Float64 for key in TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS},
}

TREND_PULLBACK_QUALITY_RANKING_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "horizon": pl.String,
    "ticker": pl.String,
    **{key: pl.Float64 for key in TREND_PULLBACK_QUALITY_RANKING_METRIC_KEYS},
    "score": pl.Float64,
    "rank": pl.Int64,
}


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=TREND_PULLBACK_QUALITY_SNAPSHOT_SCHEMA)


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema=TREND_PULLBACK_QUALITY_RANKING_SCHEMA)


def _all_finite(values: list[object]) -> bool:
    return all(
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in values
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float], *, ddof: int) -> float | None:
    if len(values) <= ddof:
        return None
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - ddof)
    return math.sqrt(variance)


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def _ols_slope_r2(values: list[float]) -> tuple[float, float]:
    y_values = [math.log(value) for value in values]
    x_values = list(range(len(y_values)))
    x_mean = _mean([float(value) for value in x_values])
    y_mean = _mean(y_values)
    ss_xx = sum((float(value) - x_mean) ** 2 for value in x_values)
    slope = sum(
        (float(x) - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)
    ) / ss_xx
    intercept = y_mean - slope * x_mean
    total = sum((value - y_mean) ** 2 for value in y_values)
    if total == 0.0:
        return slope, 0.0
    residual = sum(
        (y - (intercept + slope * float(x))) ** 2
        for x, y in zip(x_values, y_values, strict=True)
    )
    return slope, 1.0 - residual / total


def _yyyymmdd(value: date) -> float:
    return float(value.year * 10_000 + value.month * 100 + value.day)


def _percentile_scores(values: list[float], *, higher_is_better: bool = True) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=higher_is_better)
    scores = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank_position = (position + end - 1) / 2.0
        score = 1.0 - average_rank_position / (len(values) - 1)
        for original_index, _value in indexed[position:end]:
            scores[original_index] = score
        position = end
    return scores


def _positive_only_percentile_scores(values: list[float]) -> list[float]:
    scores = [0.0] * len(values)
    positive_pairs = [(index, value) for index, value in enumerate(values) if value > 0.0]
    if not positive_pairs:
        return scores
    positive_scores = _percentile_scores([value for _index, value in positive_pairs])
    for (original_index, _value), score in zip(positive_pairs, positive_scores, strict=True):
        scores[original_index] = score
    return scores


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _band_score(value: float, *, ideal_low: float, ideal_high: float, outer_low: float, outer_high: float) -> float:
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        if value <= outer_low:
            return 0.0
        return (value - outer_low) / (ideal_low - outer_low)
    if value >= outer_high:
        return 0.0
    return (outer_high - value) / (outer_high - ideal_high)


def _most_recent_days_since_high(values: list[float]) -> float:
    high = max(values)
    for reverse_index, value in enumerate(reversed(values)):
        if value == high:
            return float(reverse_index)
    return 0.0


def _immutable_market_float_mapping(value: Mapping[Market, float]) -> Mapping[Market, float]:
    return MappingProxyType({market: float(value[market]) for market in Market})


def _immutable_market_int_mapping(value: Mapping[Market, int]) -> Mapping[Market, int]:
    return MappingProxyType({market: int(value[market]) for market in Market})


@dataclass(frozen=True)
class TrendPullbackQualityV1Profile:
    profile_id: Literal["trend_pullback_quality_v1"] = TREND_PULLBACK_QUALITY_PROFILE_ID
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
    snapshot_metric_keys: tuple[str, ...] = TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = TREND_PULLBACK_QUALITY_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS
    horizon_order: tuple[str, ...] = TREND_PULLBACK_QUALITY_HORIZON_ORDER
    stdev_ddof: int = 1
    score_method: str = TREND_PULLBACK_QUALITY_SCORE_METHOD
    rank_interpretation_note: str = TREND_PULLBACK_QUALITY_RANK_INTERPRETATION_NOTE

    def __post_init__(self) -> None:
        object.__setattr__(self, "price_floor", _immutable_market_float_mapping(self.price_floor))
        object.__setattr__(self, "liquidity_floor", _immutable_market_float_mapping(self.liquidity_floor))
        object.__setattr__(
            self,
            "active_trading_min_days_60",
            _immutable_market_int_mapping(self.active_trading_min_days_60),
        )
        object.__setattr__(
            self,
            "zero_volume_max_days_20",
            _immutable_market_int_mapping(self.zero_volume_max_days_20),
        )
        object.__setattr__(self, "snapshot_metric_keys", tuple(str(key) for key in self.snapshot_metric_keys))
        object.__setattr__(self, "ranking_metric_keys", tuple(str(key) for key in self.ranking_metric_keys))
        object.__setattr__(self, "inspect_metric_keys", tuple(str(key) for key in self.inspect_metric_keys))
        object.__setattr__(self, "horizon_order", tuple(str(horizon) for horizon in self.horizon_order))

    def validate(self) -> None:
        if self.profile_id != TREND_PULLBACK_QUALITY_PROFILE_ID:
            raise ValidationError("profile_id must be trend_pullback_quality_v1")
        if self.min_history_bars != 252:
            raise ValidationError("trend_pullback_quality_v1 requires min_history_bars to be 252")
        if self.snapshot_metric_keys != TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("snapshot metric keys must match trend_pullback_quality_v1")
        if self.ranking_metric_keys != TREND_PULLBACK_QUALITY_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match trend_pullback_quality_v1")
        if self.inspect_metric_keys != TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("inspect metric keys must match trend_pullback_quality_v1")
        if self.horizon_order != TREND_PULLBACK_QUALITY_HORIZON_ORDER:
            raise ValidationError("horizon order must be composite, near_support, trend_resume")
        for mapping_name in (
            "price_floor",
            "liquidity_floor",
            "active_trading_min_days_60",
            "zero_volume_max_days_20",
        ):
            if set(getattr(self, mapping_name)) != set(Market):
                raise ValidationError(f"{mapping_name} must contain exactly supported markets")
        if {market: self.price_floor[market] for market in Market} != {Market.TW: 10.0, Market.US: 5.0}:
            raise ValidationError("price_floor must match trend_pullback_quality_v1")
        if {market: self.liquidity_floor[market] for market in Market} != {
            Market.TW: 50_000_000.0,
            Market.US: 10_000_000.0,
        }:
            raise ValidationError("liquidity_floor must match trend_pullback_quality_v1")
        if {market: self.active_trading_min_days_60[market] for market in Market} != {
            Market.TW: 50,
            Market.US: 55,
        }:
            raise ValidationError("active_trading_min_days_60 must match trend_pullback_quality_v1")
        if {market: self.zero_volume_max_days_20[market] for market in Market} != {
            Market.TW: 3,
            Market.US: 1,
        }:
            raise ValidationError("zero_volume_max_days_20 must match trend_pullback_quality_v1")
        if self.stale_close_max_days_20 != 5:
            raise ValidationError("stale_close_max_days_20 must be 5")
        if self.extreme_return_abs_cutoff != 0.80:
            raise ValidationError("extreme_return_abs_cutoff must be 0.80")
        if self.volatility_floor != 0.0001:
            raise ValidationError("volatility_floor must be 0.0001")
        if self.stdev_ddof != 1:
            raise ValidationError("stdev_ddof must be 1")
        if self.score_method != TREND_PULLBACK_QUALITY_SCORE_METHOD:
            raise ValidationError(f"score_method must be {TREND_PULLBACK_QUALITY_SCORE_METHOD}")
        if self.rank_interpretation_note != TREND_PULLBACK_QUALITY_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match trend_pullback_quality_v1")

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
        if candidate_bars.filter(pl.col("bar_date") == run_latest_bar_date).height > 0:
            candidate_bars = candidate_bars.filter(pl.col("bar_date") != run_latest_bar_date)
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
            if not _all_finite(opens + highs + lows + closes + adjusted_closes + volumes):
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
            avg_traded_value_5d_local = _mean(traded_values_5d)
            avg_traded_value_20d_local = _mean(traded_values_20d)
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
            volatility_20d = _std(returns_20d, ddof=self.stdev_ddof)
            volatility_60d = _std(returns_60d, ddof=self.stdev_ddof)
            if volatility_20d is None or volatility_60d is None:
                continue
            if volatility_20d <= 0.0 or volatility_60d <= self.volatility_floor:
                continue

            trend_slope_60d, trend_r2_60d = _ols_slope_r2(adjusted_closes_float[-60:])
            uptrend_r2_60d = trend_r2_60d if trend_slope_60d > 0.0 else 0.0
            trend_consistency_60d = float(sum(1 for value in returns_60d if value > 0.0)) / 60.0
            sma_20d = _mean(adjusted_closes_float[-20:])
            sma_50d = _mean(adjusted_closes_float[-50:])
            sma_200d = _mean(adjusted_closes_float[-200:])
            price_vs_sma_20d = latest_adjusted_close / sma_20d - 1.0
            price_vs_sma_50d = latest_adjusted_close / sma_50d - 1.0
            price_vs_sma_200d = latest_adjusted_close / sma_200d - 1.0
            sma_20d_vs_sma_50d = sma_20d / sma_50d - 1.0
            sma_50d_vs_sma_200d = sma_50d / sma_200d - 1.0
            pullback_from_60d_high = latest_adjusted_close / max(adjusted_closes_float[-60:]) - 1.0
            pullback_from_120d_high = latest_adjusted_close / max(adjusted_closes_float[-120:]) - 1.0
            volatility_adjusted_pullback_120d = abs(pullback_from_120d_high) / (
                volatility_60d * math.sqrt(20.0)
            )
            days_since_60d_high = _most_recent_days_since_high(adjusted_closes_float[-60:])
            days_since_120d_high = _most_recent_days_since_high(adjusted_closes_float[-120:])
            max_drawdown_120d = _max_drawdown(adjusted_closes_float[-120:])
            close_20d_low = min(adjusted_closes_float[-20:])
            close_20d_high = max(adjusted_closes_float[-20:])
            close_position_20d_range = (
                0.5
                if close_20d_high == close_20d_low
                else (latest_adjusted_close - close_20d_low) / (close_20d_high - close_20d_low)
            )
            low_10d_vs_sma_50d = min(lows_float[-10:]) / sma_50d - 1.0
            return_20d = latest_adjusted_close / adjusted_closes_float[-21] - 1.0
            return_60d = latest_adjusted_close / adjusted_closes_float[-61] - 1.0
            return_120d = latest_adjusted_close / adjusted_closes_float[-121] - 1.0
            return_120d_ex_recent_20d = adjusted_closes_float[-21] / adjusted_closes_float[-121] - 1.0
            risk_adjusted_return_120d_ex_recent_20d = return_120d_ex_recent_20d / volatility_60d

            if trend_slope_60d <= 0.0 or sma_50d_vs_sma_200d <= 0.0 or price_vs_sma_200d <= 0.0:
                continue
            if pullback_from_120d_high < -0.30:
                continue

            computed = [
                avg_traded_value_20d_local,
                avg_traded_value_5d_local,
                median_traded_value_20d_local,
                traded_value_5d_to_20d_ratio,
                return_20d,
                return_60d,
                return_120d,
                return_120d_ex_recent_20d,
                risk_adjusted_return_120d_ex_recent_20d,
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
                pullback_from_60d_high,
                pullback_from_120d_high,
                volatility_adjusted_pullback_120d,
                days_since_60d_high,
                days_since_120d_high,
                max_drawdown_120d,
                close_position_20d_range,
                low_10d_vs_sma_50d,
                zero_volume_days_20d,
                active_trading_days_60d,
                stale_close_days_20d,
                0.0,
            ]
            if not _all_finite(computed):
                continue

            rows.append(
                {
                    "run_id": run_id,
                    "market": market.value,
                    "ticker": ticker,
                    "close": latest_close,
                    "adjusted_close": latest_adjusted_close,
                    "profile_metrics_version": 1.0,
                    "asof_bar_date_yyyymmdd": _yyyymmdd(profile_asof_bar_date),
                    "avg_traded_value_20d_local": avg_traded_value_20d_local,
                    "avg_traded_value_5d_local": avg_traded_value_5d_local,
                    "median_traded_value_20d_local": median_traded_value_20d_local,
                    "traded_value_5d_to_20d_ratio": traded_value_5d_to_20d_ratio,
                    "return_20d": return_20d,
                    "return_60d": return_60d,
                    "return_120d": return_120d,
                    "return_120d_ex_recent_20d": return_120d_ex_recent_20d,
                    "risk_adjusted_return_120d_ex_recent_20d": risk_adjusted_return_120d_ex_recent_20d,
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
                    "pullback_from_60d_high": pullback_from_60d_high,
                    "pullback_from_120d_high": pullback_from_120d_high,
                    "volatility_adjusted_pullback_120d": volatility_adjusted_pullback_120d,
                    "days_since_60d_high": days_since_60d_high,
                    "days_since_120d_high": days_since_120d_high,
                    "max_drawdown_120d": max_drawdown_120d,
                    "close_position_20d_range": close_position_20d_range,
                    "low_10d_vs_sma_50d": low_10d_vs_sma_50d,
                    "zero_volume_days_20d": zero_volume_days_20d,
                    "active_trading_days_60d": active_trading_days_60d,
                    "stale_close_days_20d": stale_close_days_20d,
                    "data_quality_extreme_return_flag": 0.0,
                }
            )

        if not rows:
            return _empty_snapshot()
        return pl.DataFrame(rows, schema=TREND_PULLBACK_QUALITY_SNAPSHOT_SCHEMA).sort("ticker")

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        if snapshot.is_empty():
            return _empty_rankings()
        if not self._has_required_ranking_inputs(snapshot):
            raise ValidationError("trend_pullback_quality_v1 snapshot is missing required ranking inputs")
        for column in ("close", "adjusted_close", *TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS):
            if not snapshot.schema[column].is_numeric():
                raise ValidationError(
                    f"trend_pullback_quality_v1 snapshot contains non-numeric ranking input: {column}"
                )
            invalid_count = snapshot.filter(pl.col(column).is_null() | (~pl.col(column).is_finite())).height
            if invalid_count > 0:
                raise ValidationError(
                    f"trend_pullback_quality_v1 snapshot contains non-finite ranking input: {column}"
                )

        ranking_frames = []
        for partition in snapshot.partition_by(["run_id", "market"], maintain_order=True):
            if partition["asof_bar_date_yyyymmdd"].n_unique() != 1:
                raise ValidationError("trend_pullback_quality_v1 snapshot contains mixed as-of dates")
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
            *TREND_PULLBACK_QUALITY_SNAPSHOT_METRIC_KEYS,
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
        score_return_120d_ex_recent = _percentile_scores(
            [float(row["return_120d_ex_recent_20d"]) for row in rows]
        )
        score_return_60d = _percentile_scores([float(row["return_60d"]) for row in rows])
        score_risk_adjusted_return = _percentile_scores(
            [float(row["risk_adjusted_return_120d_ex_recent_20d"]) for row in rows]
        )
        score_trend_slope = _positive_only_percentile_scores([float(row["trend_slope_60d"]) for row in rows])
        score_sma_50d_vs_sma_200d = _percentile_scores(
            [float(row["sma_50d_vs_sma_200d"]) for row in rows]
        )
        score_price_vs_sma_200d = _percentile_scores([float(row["price_vs_sma_200d"]) for row in rows])
        score_trend_consistency = _percentile_scores([float(row["trend_consistency_60d"]) for row in rows])
        score_drawdown_control = _percentile_scores([float(row["max_drawdown_120d"]) for row in rows])
        score_volatility_ratio = _percentile_scores(
            [float(row["volatility_20d_to_60d_ratio"]) for row in rows],
            higher_is_better=False,
        )
        score_stale_close = _percentile_scores(
            [float(row["stale_close_days_20d"]) for row in rows],
            higher_is_better=False,
        )

        scored_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows):
            pullback_from_120d_high = float(row["pullback_from_120d_high"])
            pullback_from_60d_high = float(row["pullback_from_60d_high"])
            volatility_adjusted_pullback_120d = float(row["volatility_adjusted_pullback_120d"])
            days_since_60d_high = float(row["days_since_60d_high"])
            return_120d_ex_recent_20d = float(row["return_120d_ex_recent_20d"])
            trend_slope_60d = float(row["trend_slope_60d"])
            uptrend_r2_60d = float(row["uptrend_r2_60d"])
            trend_consistency_60d = float(row["trend_consistency_60d"])
            price_vs_sma_20d = float(row["price_vs_sma_20d"])
            price_vs_sma_50d = float(row["price_vs_sma_50d"])
            price_vs_sma_200d = float(row["price_vs_sma_200d"])
            sma_50d_vs_sma_200d = float(row["sma_50d_vs_sma_200d"])
            max_drawdown_120d = float(row["max_drawdown_120d"])
            close_position_20d_range = _clamp(float(row["close_position_20d_range"]), 0.0, 1.0)
            low_10d_vs_sma_50d = float(row["low_10d_vs_sma_50d"])
            traded_value_5d_to_20d_ratio = float(row["traded_value_5d_to_20d_ratio"])
            volatility_20d_to_60d_ratio = float(row["volatility_20d_to_60d_ratio"])
            stale_close_days_20d = float(row["stale_close_days_20d"])
            data_quality_extreme_return_flag = float(row["data_quality_extreme_return_flag"])

            score_prior_strength = score_return_120d_ex_recent[index]
            prior_strength_score = (
                0.50 * score_return_120d_ex_recent[index]
                + 0.30 * score_return_60d[index]
                + 0.20 * score_risk_adjusted_return[index]
            )
            trend_intact_score = (
                0.25 * score_trend_slope[index]
                + 0.20 * uptrend_r2_60d
                + 0.20 * score_sma_50d_vs_sma_200d[index]
                + 0.20 * score_price_vs_sma_200d[index]
                + 0.15 * score_trend_consistency[index]
            )
            score_trend_intact = trend_intact_score
            score_pullback_depth = _band_score(
                pullback_from_120d_high,
                ideal_low=-0.15,
                ideal_high=-0.05,
                outer_low=-0.30,
                outer_high=0.0,
            )
            score_volatility_adjusted_pullback = _band_score(
                volatility_adjusted_pullback_120d,
                ideal_low=0.8,
                ideal_high=1.8,
                outer_low=0.0,
                outer_high=3.2,
            )
            score_pullback_stability = _band_score(
                days_since_60d_high,
                ideal_low=3.0,
                ideal_high=30.0,
                outer_low=0.0,
                outer_high=60.0,
            )
            pullback_setup_score = (
                0.55 * score_pullback_depth
                + 0.25 * score_volatility_adjusted_pullback
                + 0.20 * score_pullback_stability
            )
            score_support_proximity = _band_score(
                price_vs_sma_50d,
                ideal_low=-0.02,
                ideal_high=0.05,
                outer_low=-0.10,
                outer_high=0.18,
            )
            score_price_vs_sma_20d = _band_score(
                price_vs_sma_20d,
                ideal_low=-0.03,
                ideal_high=0.04,
                outer_low=-0.12,
                outer_high=0.16,
            )
            score_low_10d_vs_sma_50d = _band_score(
                low_10d_vs_sma_50d,
                ideal_low=-0.03,
                ideal_high=0.03,
                outer_low=-0.12,
                outer_high=0.12,
            )
            support_quality_score = (
                0.45 * score_support_proximity
                + 0.25 * score_price_vs_sma_20d
                + 0.20 * close_position_20d_range
                + 0.10 * score_low_10d_vs_sma_50d
            )
            score_liquidity_continuity = self._liquidity_continuity_score(
                traded_value_5d_to_20d_ratio=traded_value_5d_to_20d_ratio,
                stale_close_days_20d=stale_close_days_20d,
            )
            risk_control_score = (
                0.35 * score_drawdown_control[index]
                + 0.25 * score_volatility_ratio[index]
                + 0.25 * score_liquidity_continuity
                + 0.15 * score_stale_close[index]
            )

            structure_cap_score = self._structure_cap_score(
                trend_slope_60d=trend_slope_60d,
                uptrend_r2_60d=uptrend_r2_60d,
                trend_consistency_60d=trend_consistency_60d,
                price_vs_sma_200d=price_vs_sma_200d,
                sma_50d_vs_sma_200d=sma_50d_vs_sma_200d,
                return_120d_ex_recent_20d=return_120d_ex_recent_20d,
            )
            pullback_depth_cap_score = self._pullback_depth_cap_score(
                pullback_from_120d_high=pullback_from_120d_high
            )
            overheat_cap_score = self._overheat_cap_score(
                pullback_from_120d_high=pullback_from_120d_high,
                price_vs_sma_20d=price_vs_sma_20d,
                price_vs_sma_50d=price_vs_sma_50d,
            )
            breakdown_cap_score = self._breakdown_cap_score(
                price_vs_sma_50d=price_vs_sma_50d,
                low_10d_vs_sma_50d=low_10d_vs_sma_50d,
                close_position_20d_range=close_position_20d_range,
                max_drawdown_120d=max_drawdown_120d,
            )

            tag_setup_healthy_pullback = (
                1.0
                if -0.15 <= pullback_from_120d_high <= -0.03
                and return_120d_ex_recent_20d > 0.0
                and trend_slope_60d > 0.0
                and sma_50d_vs_sma_200d > 0.0
                and price_vs_sma_200d > 0.0
                else 0.0
            )
            tag_setup_near_sma50 = 1.0 if -0.02 <= price_vs_sma_50d <= 0.05 else 0.0
            tag_setup_trend_resume = (
                1.0
                if tag_setup_healthy_pullback == 1.0
                and close_position_20d_range >= 0.50
                and price_vs_sma_20d >= -0.05
                else 0.0
            )
            tag_risk_false_pullback = (
                1.0 if pullback_from_120d_high > -0.03 or return_120d_ex_recent_20d <= 0.05 else 0.0
            )
            tag_risk_breakdown = (
                1.0
                if price_vs_sma_50d < -0.03
                or low_10d_vs_sma_50d < -0.06
                or close_position_20d_range < 0.25
                else 0.0
            )
            tag_risk_deep_drawdown = (
                1.0 if pullback_from_120d_high <= -0.20 or max_drawdown_120d <= -0.25 else 0.0
            )
            tag_risk_liquidity_fade = 1.0 if traded_value_5d_to_20d_ratio < 0.75 else 0.0
            tag_risk_still_overheated = (
                1.0
                if pullback_from_120d_high > -0.03 or price_vs_sma_20d > 0.10 or price_vs_sma_50d > 0.12
                else 0.0
            )
            tag_risk_volatility_spike = 1.0 if volatility_20d_to_60d_ratio > 1.75 else 0.0
            tag_risk_data_quality_warning = 1.0 if data_quality_extreme_return_flag == 1.0 else 0.0

            penalty_score = 0.0
            penalty_score += 0.12 * tag_risk_false_pullback
            penalty_score += 0.15 * tag_risk_breakdown
            penalty_score += 0.12 * tag_risk_deep_drawdown
            penalty_score += 0.08 * tag_risk_liquidity_fade
            penalty_score += 0.10 * tag_risk_still_overheated
            penalty_score += 0.08 * tag_risk_volatility_spike
            penalty_score += 0.20 * tag_risk_data_quality_warning

            row.update(
                {
                    "score_prior_strength": score_prior_strength,
                    "score_trend_intact": score_trend_intact,
                    "score_pullback_depth": score_pullback_depth,
                    "score_volatility_adjusted_pullback": score_volatility_adjusted_pullback,
                    "score_support_proximity": score_support_proximity,
                    "score_pullback_stability": score_pullback_stability,
                    "score_liquidity_continuity": score_liquidity_continuity,
                    "prior_strength_score": prior_strength_score,
                    "trend_intact_score": trend_intact_score,
                    "pullback_setup_score": pullback_setup_score,
                    "support_quality_score": support_quality_score,
                    "risk_control_score": risk_control_score,
                    "structure_cap_score": structure_cap_score,
                    "pullback_depth_cap_score": pullback_depth_cap_score,
                    "overheat_cap_score": overheat_cap_score,
                    "breakdown_cap_score": breakdown_cap_score,
                    "penalty_score": penalty_score,
                    "tag_setup_healthy_pullback": tag_setup_healthy_pullback,
                    "tag_setup_near_sma50": tag_setup_near_sma50,
                    "tag_setup_trend_resume": tag_setup_trend_resume,
                    "tag_risk_false_pullback": tag_risk_false_pullback,
                    "tag_risk_breakdown": tag_risk_breakdown,
                    "tag_risk_deep_drawdown": tag_risk_deep_drawdown,
                    "tag_risk_liquidity_fade": tag_risk_liquidity_fade,
                    "tag_risk_still_overheated": tag_risk_still_overheated,
                    "tag_risk_volatility_spike": tag_risk_volatility_spike,
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
                        0.25 * float(row["prior_strength_score"])
                        + 0.25 * float(row["trend_intact_score"])
                        + 0.25 * float(row["pullback_setup_score"])
                        + 0.15 * float(row["support_quality_score"])
                        + 0.10 * float(row["risk_control_score"])
                    )
                elif horizon == "near_support":
                    raw_score = (
                        0.20 * float(row["prior_strength_score"])
                        + 0.20 * float(row["trend_intact_score"])
                        + 0.25 * float(row["pullback_setup_score"])
                        + 0.25 * float(row["support_quality_score"])
                        + 0.10 * float(row["risk_control_score"])
                    )
                elif horizon == "trend_resume":
                    raw_score = (
                        0.25 * float(row["prior_strength_score"])
                        + 0.30 * float(row["trend_intact_score"])
                        + 0.15 * float(row["pullback_setup_score"])
                        + 0.20 * float(row["support_quality_score"])
                        + 0.10 * float(row["risk_control_score"])
                    )
                else:
                    raise ValidationError(f"unknown horizon {horizon}")
                score = min(
                    raw_score - float(row["penalty_score"]),
                    float(row["structure_cap_score"]),
                    float(row["pullback_depth_cap_score"]),
                    float(row["overheat_cap_score"]),
                    float(row["breakdown_cap_score"]),
                )
                ranking_values = [float(row[key]) for key in self.ranking_metric_keys] + [score]
                if not _all_finite(ranking_values):
                    raise ValidationError("trend_pullback_quality_v1 produced a non-finite ranking metric")
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
        return pl.DataFrame(ranking_rows, schema=TREND_PULLBACK_QUALITY_RANKING_SCHEMA).select(
            self._ranking_columns()
        )

    def _liquidity_continuity_score(
        self,
        *,
        traded_value_5d_to_20d_ratio: float,
        stale_close_days_20d: float,
    ) -> float:
        ratio_score = _clamp(traded_value_5d_to_20d_ratio, 0.0, 1.0)
        stale_penalty = _clamp(stale_close_days_20d / float(self.stale_close_max_days_20), 0.0, 1.0)
        return _clamp(ratio_score - 0.50 * stale_penalty, 0.0, 1.0)

    def _structure_cap_score(
        self,
        *,
        trend_slope_60d: float,
        uptrend_r2_60d: float,
        trend_consistency_60d: float,
        price_vs_sma_200d: float,
        sma_50d_vs_sma_200d: float,
        return_120d_ex_recent_20d: float,
    ) -> float:
        failures = sum(
            1
            for failed in (
                trend_slope_60d <= 0.0,
                price_vs_sma_200d <= 0.0,
                sma_50d_vs_sma_200d <= 0.0,
                return_120d_ex_recent_20d <= 0.0,
            )
            if failed
        )
        if failures >= 3:
            return 0.25
        if failures == 2:
            return 0.40
        if failures == 1:
            return 0.60
        if uptrend_r2_60d < 0.20 or trend_consistency_60d < 0.45:
            return 0.75
        return 1.0

    def _pullback_depth_cap_score(self, *, pullback_from_120d_high: float) -> float:
        if pullback_from_120d_high > -0.01:
            return 0.45
        if pullback_from_120d_high > -0.03:
            return 0.65
        if pullback_from_120d_high < -0.25:
            return 0.55
        if pullback_from_120d_high < -0.20:
            return 0.70
        if pullback_from_120d_high < -0.18:
            return 0.85
        return 1.0

    def _overheat_cap_score(
        self,
        *,
        pullback_from_120d_high: float,
        price_vs_sma_20d: float,
        price_vs_sma_50d: float,
    ) -> float:
        if pullback_from_120d_high > -0.01 or price_vs_sma_20d > 0.14 or price_vs_sma_50d > 0.18:
            return 0.45
        if pullback_from_120d_high > -0.03 or price_vs_sma_20d > 0.10 or price_vs_sma_50d > 0.12:
            return 0.60
        if price_vs_sma_20d > 0.08 or price_vs_sma_50d > 0.10:
            return 0.75
        return 1.0

    def _breakdown_cap_score(
        self,
        *,
        price_vs_sma_50d: float,
        low_10d_vs_sma_50d: float,
        close_position_20d_range: float,
        max_drawdown_120d: float,
    ) -> float:
        if price_vs_sma_50d < -0.08 or low_10d_vs_sma_50d < -0.10 or close_position_20d_range < 0.10:
            return 0.40
        if price_vs_sma_50d < -0.03 or low_10d_vs_sma_50d < -0.06 or close_position_20d_range < 0.25:
            return 0.60
        if max_drawdown_120d <= -0.25:
            return 0.75
        return 1.0


TREND_PULLBACK_QUALITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=TREND_PULLBACK_QUALITY_PROFILE_ID,
    factory=TrendPullbackQualityV1Profile,
)
