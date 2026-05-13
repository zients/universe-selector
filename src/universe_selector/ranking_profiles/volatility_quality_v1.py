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


VOLATILITY_QUALITY_PROFILE_ID = "volatility_quality_v1"
VOLATILITY_QUALITY_SCORE_METHOD = "market_percentile_volatility_quality_v1"
VOLATILITY_QUALITY_RANK_INTERPRETATION_NOTE = (
    "Volatility quality scores rank market-local lower realized volatility, downside volatility, "
    "range tightness, and drawdown control; high scores do not imply future returns or lower future risk."
)
VOLATILITY_QUALITY_HORIZON_ORDER = ("composite", "shortterm", "stable")

VOLATILITY_QUALITY_SNAPSHOT_METRIC_KEYS = (
    "profile_metrics_version",
    "avg_traded_value_20d_local",
    "volatility_20d",
    "volatility_60d",
    "downside_volatility_60d",
    "volatility_20d_to_60d_ratio",
    "volatility_stability_60d",
    "max_drawdown_120d",
    "median_range_pct_20d",
    "median_range_pct_60d",
    "zero_volume_days_20d",
    "active_trading_days_60d",
    "stale_close_days_20d",
    "data_quality_extreme_return_flag",
)

VOLATILITY_QUALITY_RANKING_METRIC_KEYS = (
    "score_low_volatility_20d",
    "score_low_volatility_60d",
    "score_downside_volatility_60d",
    "score_drawdown_control_120d",
    "score_range_tightness_20d",
    "score_range_tightness_60d",
    "score_volatility_stability_60d",
    "volatility_control_score",
    "trading_smoothness_score",
    "drawdown_quality_score",
    "penalty_score",
)

VOLATILITY_QUALITY_SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    **{key: pl.Float64 for key in VOLATILITY_QUALITY_SNAPSHOT_METRIC_KEYS},
}

VOLATILITY_QUALITY_RANKING_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "horizon": pl.String,
    "ticker": pl.String,
    **{key: pl.Float64 for key in VOLATILITY_QUALITY_RANKING_METRIC_KEYS},
    "score": pl.Float64,
    "rank": pl.Int64,
}


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=VOLATILITY_QUALITY_SNAPSHOT_SCHEMA)


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema=VOLATILITY_QUALITY_RANKING_SCHEMA)


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


def _downside_volatility(values: list[float]) -> float:
    downside = [min(value, 0.0) for value in values]
    return math.sqrt(sum(value * value for value in downside) / len(values))


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def _immutable_market_float_mapping(value: Mapping[Market, float]) -> Mapping[Market, float]:
    return MappingProxyType({market: float(value[market]) for market in Market})


def _immutable_market_int_mapping(value: Mapping[Market, int]) -> Mapping[Market, int]:
    return MappingProxyType({market: int(value[market]) for market in Market})


@dataclass(frozen=True)
class VolatilityQualityV1Profile:
    profile_id: Literal["volatility_quality_v1"] = VOLATILITY_QUALITY_PROFILE_ID
    min_history_bars: int = 126
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
    volatility_floor: float = 0.0001
    snapshot_metric_keys: tuple[str, ...] = VOLATILITY_QUALITY_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = VOLATILITY_QUALITY_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = VOLATILITY_QUALITY_SNAPSHOT_METRIC_KEYS
    horizon_order: tuple[str, ...] = VOLATILITY_QUALITY_HORIZON_ORDER
    stdev_ddof: int = 1
    score_method: str = VOLATILITY_QUALITY_SCORE_METHOD
    rank_interpretation_note: str = VOLATILITY_QUALITY_RANK_INTERPRETATION_NOTE

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
        if self.profile_id != VOLATILITY_QUALITY_PROFILE_ID:
            raise ValidationError("profile_id must be volatility_quality_v1")
        if self.min_history_bars != 126:
            raise ValidationError("volatility_quality_v1 requires min_history_bars to be 126")
        if self.snapshot_metric_keys != VOLATILITY_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("snapshot metric keys must match volatility_quality_v1")
        if self.ranking_metric_keys != VOLATILITY_QUALITY_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match volatility_quality_v1")
        if self.inspect_metric_keys != VOLATILITY_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("inspect metric keys must match volatility_quality_v1")
        if self.horizon_order != VOLATILITY_QUALITY_HORIZON_ORDER:
            raise ValidationError("horizon order must be composite, shortterm, stable")
        for mapping_name in (
            "price_floor",
            "liquidity_floor",
            "active_trading_min_days_60",
            "zero_volume_max_days_20",
        ):
            if set(getattr(self, mapping_name)) != set(Market):
                raise ValidationError(f"{mapping_name} must contain exactly supported markets")
        if {market: self.price_floor[market] for market in Market} != {Market.TW: 10.0, Market.US: 5.0}:
            raise ValidationError("price_floor must match volatility_quality_v1")
        if {market: self.liquidity_floor[market] for market in Market} != {
            Market.TW: 50_000_000.0,
            Market.US: 10_000_000.0,
        }:
            raise ValidationError("liquidity_floor must match volatility_quality_v1")
        if {market: self.active_trading_min_days_60[market] for market in Market} != {
            Market.TW: 50,
            Market.US: 55,
        }:
            raise ValidationError("active_trading_min_days_60 must match volatility_quality_v1")
        if {market: self.zero_volume_max_days_20[market] for market in Market} != {
            Market.TW: 3,
            Market.US: 1,
        }:
            raise ValidationError("zero_volume_max_days_20 must match volatility_quality_v1")
        if self.volatility_floor != 0.0001:
            raise ValidationError("volatility_floor must be 0.0001")
        if self.stdev_ddof != 1:
            raise ValidationError("stdev_ddof must be 1")
        if self.score_method != VOLATILITY_QUALITY_SCORE_METHOD:
            raise ValidationError(f"score_method must be {VOLATILITY_QUALITY_SCORE_METHOD}")
        if self.rank_interpretation_note != VOLATILITY_QUALITY_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match volatility_quality_v1")

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
            "volatility_floor": self.volatility_floor,
            "horizon_order": list(self.horizon_order),
            "snapshot_metric_keys": list(self.snapshot_metric_keys),
            "ranking_metric_keys": list(self.ranking_metric_keys),
            "inspect_metric_keys": list(self.inspect_metric_keys),
            "stdev_ddof": self.stdev_ddof,
            "score_method": self.score_method,
        }

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

        filtered_bars = bars.filter(
            (pl.col("market") == market.value)
            & (pl.col("ticker").is_in(list(listed_tickers)))
            & (pl.col("bar_date") <= run_latest_bar_date)
        ).sort(["ticker", "bar_date"])

        rows: list[dict[str, object]] = []
        for ticker in sorted(listed_tickers):
            ticker_bars = filtered_bars.filter(pl.col("ticker") == ticker)
            if ticker_bars.height < self.min_history_bars:
                continue
            if ticker_bars["bar_date"][-1] != run_latest_bar_date:
                continue

            tail = ticker_bars.tail(self.min_history_bars)
            opens = tail["open"].to_list()
            highs = tail["high"].to_list()
            lows = tail["low"].to_list()
            closes = tail["close"].to_list()
            adjusted_closes = tail["adjusted_close"].to_list()
            volumes = tail["volume"].to_list()
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

            traded_values = [close * volume for close, volume in zip(closes_float, volumes_float, strict=True)]
            latest_close = closes_float[-1]
            latest_adjusted_close = adjusted_closes_float[-1]
            avg_traded_value_20d_local = _mean(traded_values[-20:])
            volumes_20d = volumes_float[-20:]
            volumes_60d = volumes_float[-60:]
            zero_volume_days_20d = float(sum(1 for value in volumes_20d if value == 0.0))
            active_trading_days_60d = float(sum(1 for value in volumes_60d if value > 0.0))

            if latest_close < self.price_floor[market]:
                continue
            if avg_traded_value_20d_local < self.liquidity_floor[market]:
                continue
            if active_trading_days_60d < self.active_trading_min_days_60[market]:
                continue
            if zero_volume_days_20d > self.zero_volume_max_days_20[market]:
                continue

            returns = [
                adjusted_closes_float[index] / adjusted_closes_float[index - 1] - 1.0
                for index in range(1, len(adjusted_closes_float))
            ]
            returns_20d = returns[-20:]
            returns_60d = returns[-60:]
            volatility_20d = _std(returns_20d, ddof=self.stdev_ddof)
            volatility_60d = _std(returns_60d, ddof=self.stdev_ddof)
            if volatility_20d is None or volatility_60d is None:
                continue
            if volatility_20d <= 0.0 or volatility_60d <= self.volatility_floor:
                continue

            downside_volatility_60d = _downside_volatility(returns_60d)
            volatility_20d_to_60d_ratio = volatility_20d / volatility_60d
            volatility_stability_60d = abs(math.log(volatility_20d_to_60d_ratio))
            max_drawdown_120d = _max_drawdown(adjusted_closes_float[-120:])
            range_pct = [
                (high - low) / close
                for high, low, close in zip(highs_float, lows_float, closes_float, strict=True)
            ]
            median_range_pct_20d = float(median(range_pct[-20:]))
            median_range_pct_60d = float(median(range_pct[-60:]))
            stale_close_days_20d = float(
                sum(
                    1
                    for index in range(len(adjusted_closes_float) - 20, len(adjusted_closes_float))
                    if adjusted_closes_float[index] == adjusted_closes_float[index - 1]
                )
            )
            data_quality_extreme_return_flag = 1.0 if any(abs(value) > 0.80 for value in returns_60d) else 0.0

            computed = [
                avg_traded_value_20d_local,
                volatility_20d,
                volatility_60d,
                downside_volatility_60d,
                volatility_20d_to_60d_ratio,
                volatility_stability_60d,
                max_drawdown_120d,
                median_range_pct_20d,
                median_range_pct_60d,
                zero_volume_days_20d,
                active_trading_days_60d,
                stale_close_days_20d,
                data_quality_extreme_return_flag,
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
                    "avg_traded_value_20d_local": avg_traded_value_20d_local,
                    "volatility_20d": volatility_20d,
                    "volatility_60d": volatility_60d,
                    "downside_volatility_60d": downside_volatility_60d,
                    "volatility_20d_to_60d_ratio": volatility_20d_to_60d_ratio,
                    "volatility_stability_60d": volatility_stability_60d,
                    "max_drawdown_120d": max_drawdown_120d,
                    "median_range_pct_20d": median_range_pct_20d,
                    "median_range_pct_60d": median_range_pct_60d,
                    "zero_volume_days_20d": zero_volume_days_20d,
                    "active_trading_days_60d": active_trading_days_60d,
                    "stale_close_days_20d": stale_close_days_20d,
                    "data_quality_extreme_return_flag": data_quality_extreme_return_flag,
                }
            )

        if not rows:
            return _empty_snapshot()
        return pl.DataFrame(rows, schema=VOLATILITY_QUALITY_SNAPSHOT_SCHEMA).sort("ticker")

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        return _empty_rankings()


VOLATILITY_QUALITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=VOLATILITY_QUALITY_PROFILE_ID,
    factory=VolatilityQualityV1Profile,
)
