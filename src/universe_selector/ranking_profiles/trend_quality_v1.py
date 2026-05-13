from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Literal, Mapping

import polars as pl

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import ListingCandidate
from universe_selector.ranking_profiles.registration import RankingProfileRegistration


TREND_QUALITY_PROFILE_ID = "trend_quality_v1"
TREND_QUALITY_SCORE_METHOD = "market_relative_trend_quality_v1"
TREND_QUALITY_RANK_INTERPRETATION_NOTE = (
    "Trend quality scores are market-local weighted relative scores built from percentile/absolute "
    "structure components, structure penalties, and weak-structure caps, with descriptive non-exclusive "
    "structure tags provided alongside the scores. Scores and ranks compare candidates only within the "
    "same run, market, profile, and horizon; a top rank in a weak eligible universe can still be the "
    "least-bad candidate rather than a clean upward trend. Rows with tag_structure_weak_trend_component "
    "or tag_structure_cap_active should not be interpreted as clean upward trends. Horizons are ranking "
    "lenses, not forecasts or holding-period recommendations. Scores may be negative or capped; high "
    "scores do not imply future returns or investment suitability."
)
TREND_QUALITY_HORIZON_ORDER = ("composite", "shortterm", "midterm")

TREND_QUALITY_SNAPSHOT_METRIC_KEYS = (
    "profile_metrics_version",
    "asof_bar_date_yyyymmdd",
    "avg_traded_value_20d_local",
    "return_20d",
    "return_60d",
    "return_120d",
    "volatility_60d",
    "trend_slope_60d",
    "trend_r2_60d",
    "uptrend_r2_60d",
    "trend_consistency_60d",
    "price_vs_sma_50d",
    "price_vs_sma_200d",
    "sma_50d_vs_sma_200d",
    "pct_below_120d_high",
    "max_drawdown_120d",
    "zero_volume_days_20d",
    "active_trading_days_60d",
    "stale_close_days_20d",
)

TREND_QUALITY_RANKING_METRIC_KEYS = (
    "score_return_20d",
    "score_return_60d",
    "score_return_120d",
    "score_trend_slope_60d",
    "score_uptrend_r2_60d",
    "score_trend_consistency_60d",
    "score_price_vs_sma_50d",
    "score_price_vs_sma_200d",
    "score_sma_50d_vs_sma_200d",
    "score_pct_below_120d_high",
    "score_drawdown_control_120d",
    "trend_strength_score",
    "trend_cleanliness_score",
    "breakout_position_score",
    "drawdown_control_score",
    "trend_cleanliness_cap_score",
    "hard_structure_cap_score",
    "weak_structure_fail_count",
    "weak_positive_structure_count",
    "trend_magnitude_cap_score",
    "overextension_cap_score",
    "structure_cap_score",
    "penalty_score",
    "tag_structure_uptrend",
    "tag_structure_breakout_proximity",
    "tag_structure_consistent_uptrend",
    "tag_structure_nonpositive_60d_slope",
    "tag_structure_negative_60d_return",
    "tag_structure_below_sma_50d",
    "tag_structure_below_sma_200d",
    "tag_structure_sma_50d_below_sma_200d",
    "tag_structure_weak_trend_component",
    "tag_structure_large_drawdown",
    "tag_structure_overextended",
    "tag_structure_cap_active",
    "tag_data_stale_trading",
)

TREND_QUALITY_SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    **{key: pl.Float64 for key in TREND_QUALITY_SNAPSHOT_METRIC_KEYS},
}

TREND_QUALITY_RANKING_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "horizon": pl.String,
    "ticker": pl.String,
    **{key: pl.Float64 for key in TREND_QUALITY_RANKING_METRIC_KEYS},
    "score": pl.Float64,
    "rank": pl.Int64,
}


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=TREND_QUALITY_SNAPSHOT_SCHEMA)


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema=TREND_QUALITY_RANKING_SCHEMA)


def _all_finite(values: list[object]) -> bool:
    return all(
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in values
    )


def _immutable_market_float_mapping(value: Mapping[Market, float]) -> Mapping[Market, float]:
    return MappingProxyType({market: float(value[market]) for market in Market})


def _immutable_market_int_mapping(value: Mapping[Market, int]) -> Mapping[Market, int]:
    return MappingProxyType({market: int(value[market]) for market in Market})


@dataclass(frozen=True)
class TrendQualityV1Profile:
    profile_id: Literal["trend_quality_v1"] = TREND_QUALITY_PROFILE_ID
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
    volatility_floor: float = 0.0001
    snapshot_metric_keys: tuple[str, ...] = TREND_QUALITY_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = TREND_QUALITY_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = TREND_QUALITY_SNAPSHOT_METRIC_KEYS
    horizon_order: tuple[str, ...] = TREND_QUALITY_HORIZON_ORDER
    stdev_ddof: int = 1
    score_method: str = TREND_QUALITY_SCORE_METHOD
    rank_interpretation_note: str = TREND_QUALITY_RANK_INTERPRETATION_NOTE

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
        if self.profile_id != TREND_QUALITY_PROFILE_ID:
            raise ValidationError("profile_id must be trend_quality_v1")
        if self.min_history_bars != 252:
            raise ValidationError("trend_quality_v1 requires min_history_bars to be 252")
        if self.snapshot_metric_keys != TREND_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("snapshot metric keys must match trend_quality_v1")
        if self.ranking_metric_keys != TREND_QUALITY_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match trend_quality_v1")
        if self.inspect_metric_keys != TREND_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("inspect metric keys must match trend_quality_v1")
        if self.horizon_order != TREND_QUALITY_HORIZON_ORDER:
            raise ValidationError("horizon order must be composite, shortterm, midterm")
        for mapping_name in (
            "price_floor",
            "liquidity_floor",
            "active_trading_min_days_60",
            "zero_volume_max_days_20",
        ):
            if set(getattr(self, mapping_name)) != set(Market):
                raise ValidationError(f"{mapping_name} must contain exactly supported markets")
        if {market: self.price_floor[market] for market in Market} != {Market.TW: 10.0, Market.US: 5.0}:
            raise ValidationError("price_floor must match trend_quality_v1")
        if {market: self.liquidity_floor[market] for market in Market} != {
            Market.TW: 50_000_000.0,
            Market.US: 10_000_000.0,
        }:
            raise ValidationError("liquidity_floor must match trend_quality_v1")
        if {market: self.active_trading_min_days_60[market] for market in Market} != {
            Market.TW: 50,
            Market.US: 55,
        }:
            raise ValidationError("active_trading_min_days_60 must match trend_quality_v1")
        if {market: self.zero_volume_max_days_20[market] for market in Market} != {
            Market.TW: 3,
            Market.US: 1,
        }:
            raise ValidationError("zero_volume_max_days_20 must match trend_quality_v1")
        if self.volatility_floor != 0.0001:
            raise ValidationError("volatility_floor must be 0.0001")
        if self.stdev_ddof != 1:
            raise ValidationError("stdev_ddof must be 1")
        if self.score_method != TREND_QUALITY_SCORE_METHOD:
            raise ValidationError(f"score_method must be {TREND_QUALITY_SCORE_METHOD}")
        if self.rank_interpretation_note != TREND_QUALITY_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match trend_quality_v1")

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
        _ = (run_id, market, listings, bars, run_latest_bar_date)
        return _empty_snapshot()

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        if snapshot.is_empty():
            return _empty_rankings()
        return _empty_rankings()


TREND_QUALITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=TREND_QUALITY_PROFILE_ID,
    factory=TrendQualityV1Profile,
)
