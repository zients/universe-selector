from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
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
        return _empty_snapshot()

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        return _empty_rankings()


VOLATILITY_QUALITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=VOLATILITY_QUALITY_PROFILE_ID,
    factory=VolatilityQualityV1Profile,
)
