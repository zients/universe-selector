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


LIQUIDITY_QUALITY_PROFILE_ID = "liquidity_quality_v1"
LIQUIDITY_QUALITY_SCORE_METHOD = "market_percentile_liquidity_quality_v1"
LIQUIDITY_QUALITY_RANK_INTERPRETATION_NOTE = (
    "Liquidity quality scores rank market-local traded value, friction proxies, and traded value stability; "
    "traded value metrics are local currency amounts."
)
LIQUIDITY_QUALITY_HORIZON_ORDER = ("composite", "shortterm", "stable")

LIQUIDITY_QUALITY_SNAPSHOT_METRIC_KEYS = (
    "profile_metrics_version",
    "avg_traded_value_5d_local",
    "avg_traded_value_20d_local",
    "avg_traded_value_60d_local",
    "median_traded_value_20d_local",
    "median_traded_value_60d_local",
    "amihud_illiquidity_20d",
    "amihud_illiquidity_60d",
    "traded_value_cv_60d",
    "traded_value_concentration_60d",
    "median_range_pct_20d",
    "median_range_pct_60d",
    "traded_value_5d_to_20d_ratio",
    "zero_volume_days_20d",
    "zero_volume_days_60d",
    "active_trading_days_60d",
    "stale_close_days_20d",
    "stale_close_days_60d",
    "data_quality_extreme_return_flag",
)

LIQUIDITY_QUALITY_RANKING_METRIC_KEYS = (
    "score_log_traded_value_20d",
    "score_log_traded_value_60d",
    "score_amihud_20d",
    "score_amihud_60d",
    "score_traded_value_stability_60d",
    "score_traded_value_concentration_60d",
    "score_range_tightness_20d",
    "score_range_tightness_60d",
    "score_recent_traded_value_ratio",
    "score_trading_continuity_60d",
    "depth_score",
    "friction_score",
    "stability_score",
    "penalty_score",
    "tag_risk_thin_liquidity",
    "tag_risk_recent_liquidity_fade",
    "tag_risk_traded_value_spike",
    "tag_risk_high_impact",
    "tag_risk_wide_range",
    "tag_risk_stale_trading",
    "tag_risk_data_quality_warning",
    "tag_positive_deep_liquidity",
    "tag_positive_stable_liquidity",
)

LIQUIDITY_QUALITY_SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    **{key: pl.Float64 for key in LIQUIDITY_QUALITY_SNAPSHOT_METRIC_KEYS},
}

LIQUIDITY_QUALITY_RANKING_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "horizon": pl.String,
    "ticker": pl.String,
    **{key: pl.Float64 for key in LIQUIDITY_QUALITY_RANKING_METRIC_KEYS},
    "score": pl.Float64,
    "rank": pl.Int64,
}


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=LIQUIDITY_QUALITY_SNAPSHOT_SCHEMA)


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema=LIQUIDITY_QUALITY_RANKING_SCHEMA)


def _all_finite(values: list[object]) -> bool:
    return all(
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in values
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _cv(values: list[float]) -> float:
    average = _mean(values)
    if average <= 0.0:
        return 0.0
    return _std(values) / average


def _immutable_market_float_mapping(value: Mapping[Market, float]) -> Mapping[Market, float]:
    return MappingProxyType({market: float(value[market]) for market in Market})


def _immutable_market_int_mapping(value: Mapping[Market, int]) -> Mapping[Market, int]:
    return MappingProxyType({market: int(value[market]) for market in Market})


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class LiquidityQualityV1Profile:
    profile_id: Literal["liquidity_quality_v1"] = LIQUIDITY_QUALITY_PROFILE_ID
    min_history_bars: int = 63
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
    snapshot_metric_keys: tuple[str, ...] = LIQUIDITY_QUALITY_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = LIQUIDITY_QUALITY_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = LIQUIDITY_QUALITY_SNAPSHOT_METRIC_KEYS
    horizon_order: tuple[str, ...] = LIQUIDITY_QUALITY_HORIZON_ORDER
    score_method: str = LIQUIDITY_QUALITY_SCORE_METHOD
    rank_interpretation_note: str = LIQUIDITY_QUALITY_RANK_INTERPRETATION_NOTE

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
        if self.profile_id != LIQUIDITY_QUALITY_PROFILE_ID:
            raise ValidationError("profile_id must be liquidity_quality_v1")
        if self.min_history_bars != 63:
            raise ValidationError("liquidity_quality_v1 requires min_history_bars to be 63")
        if self.snapshot_metric_keys != LIQUIDITY_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("snapshot metric keys must match liquidity_quality_v1")
        if self.ranking_metric_keys != LIQUIDITY_QUALITY_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match liquidity_quality_v1")
        if self.inspect_metric_keys != LIQUIDITY_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("inspect metric keys must match liquidity_quality_v1")
        if self.horizon_order != LIQUIDITY_QUALITY_HORIZON_ORDER:
            raise ValidationError("horizon order must be composite, shortterm, stable")
        for mapping_name in (
            "price_floor",
            "liquidity_floor",
            "active_trading_min_days_60",
            "zero_volume_max_days_20",
        ):
            if set(getattr(self, mapping_name)) != set(Market):
                raise ValidationError(f"{mapping_name} must contain exactly supported markets")
        if self.score_method != LIQUIDITY_QUALITY_SCORE_METHOD:
            raise ValidationError(f"score_method must be {LIQUIDITY_QUALITY_SCORE_METHOD}")
        if self.rank_interpretation_note != LIQUIDITY_QUALITY_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match liquidity_quality_v1")

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
            "horizon_order": list(self.horizon_order),
            "snapshot_metric_keys": list(self.snapshot_metric_keys),
            "ranking_metric_keys": list(self.ranking_metric_keys),
            "inspect_metric_keys": list(self.inspect_metric_keys),
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

            traded_values = [
                close * volume for close, volume in zip(closes_float, volumes_float, strict=True)
            ]
            latest_close = closes_float[-1]
            latest_adjusted_close = adjusted_closes_float[-1]
            traded_values_5d = traded_values[-5:]
            traded_values_20d = traded_values[-20:]
            traded_values_60d = traded_values[-60:]
            volumes_20d = volumes_float[-20:]
            volumes_60d = volumes_float[-60:]
            closes_20d = closes_float[-20:]
            closes_60d = closes_float[-60:]
            adjusted_60d = adjusted_closes_float[-60:]

            avg_traded_value_5d_local = _mean(traded_values_5d)
            avg_traded_value_20d_local = _mean(traded_values_20d)
            avg_traded_value_60d_local = _mean(traded_values_60d)
            median_traded_value_20d_local = float(median(traded_values_20d))
            median_traded_value_60d_local = float(median(traded_values_60d))
            zero_volume_days_20d = float(sum(1 for value in volumes_20d if value == 0.0))
            zero_volume_days_60d = float(sum(1 for value in volumes_60d if value == 0.0))
            active_trading_days_60d = float(sum(1 for value in volumes_60d if value > 0.0))
            stale_close_days_20d = float(
                sum(1 for index in range(1, len(closes_20d)) if closes_20d[index] == closes_20d[index - 1])
            )
            stale_close_days_60d = float(
                sum(1 for index in range(1, len(closes_60d)) if closes_60d[index] == closes_60d[index - 1])
            )

            if latest_close < self.price_floor[market]:
                continue
            if avg_traded_value_20d_local < self.liquidity_floor[market]:
                continue
            if avg_traded_value_5d_local < self.liquidity_floor[market] * 0.50:
                continue
            if active_trading_days_60d < self.active_trading_min_days_60[market]:
                continue
            if zero_volume_days_20d > self.zero_volume_max_days_20[market]:
                continue

            returns_20d = [
                adjusted_closes_float[index] / adjusted_closes_float[index - 1] - 1.0
                for index in range(len(adjusted_closes_float) - 20, len(adjusted_closes_float))
            ]
            returns_60d = [
                adjusted_60d[index] / adjusted_60d[index - 1] - 1.0
                for index in range(1, len(adjusted_60d))
            ]
            traded_values_for_returns_20d = traded_values[-20:]
            traded_values_for_returns_60d = traded_values_60d[1:]
            amihud_observations_20d = [
                (ret, traded_value)
                for ret, traded_value in zip(returns_20d, traded_values_for_returns_20d, strict=True)
                if traded_value > 0.0
            ]
            amihud_observations_60d = [
                (ret, traded_value)
                for ret, traded_value in zip(returns_60d, traded_values_for_returns_60d, strict=True)
                if traded_value > 0.0
            ]
            if not amihud_observations_20d or not amihud_observations_60d:
                continue

            amihud_illiquidity_20d = _mean(
                [abs(ret) / traded_value for ret, traded_value in amihud_observations_20d]
            )
            amihud_illiquidity_60d = _mean(
                [abs(ret) / traded_value for ret, traded_value in amihud_observations_60d]
            )
            traded_value_cv_60d = _cv(traded_values_60d)
            traded_value_concentration_60d = max(traded_values_60d) / sum(traded_values_60d)
            range_pct = [
                (high - low) / close
                for high, low, close in zip(highs_float, lows_float, closes_float, strict=True)
            ]
            median_range_pct_20d = float(median(range_pct[-20:]))
            median_range_pct_60d = float(median(range_pct[-60:]))
            traded_value_5d_to_20d_ratio = avg_traded_value_5d_local / avg_traded_value_20d_local
            data_quality_extreme_return_flag = (
                1.0 if any(abs(value) > 0.80 for value in returns_60d) else 0.0
            )

            computed = [
                avg_traded_value_5d_local,
                avg_traded_value_20d_local,
                avg_traded_value_60d_local,
                median_traded_value_20d_local,
                median_traded_value_60d_local,
                amihud_illiquidity_20d,
                amihud_illiquidity_60d,
                traded_value_cv_60d,
                traded_value_concentration_60d,
                median_range_pct_20d,
                median_range_pct_60d,
                traded_value_5d_to_20d_ratio,
                zero_volume_days_20d,
                zero_volume_days_60d,
                active_trading_days_60d,
                stale_close_days_20d,
                stale_close_days_60d,
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
                    "avg_traded_value_5d_local": avg_traded_value_5d_local,
                    "avg_traded_value_20d_local": avg_traded_value_20d_local,
                    "avg_traded_value_60d_local": avg_traded_value_60d_local,
                    "median_traded_value_20d_local": median_traded_value_20d_local,
                    "median_traded_value_60d_local": median_traded_value_60d_local,
                    "amihud_illiquidity_20d": amihud_illiquidity_20d,
                    "amihud_illiquidity_60d": amihud_illiquidity_60d,
                    "traded_value_cv_60d": traded_value_cv_60d,
                    "traded_value_concentration_60d": traded_value_concentration_60d,
                    "median_range_pct_20d": median_range_pct_20d,
                    "median_range_pct_60d": median_range_pct_60d,
                    "traded_value_5d_to_20d_ratio": traded_value_5d_to_20d_ratio,
                    "zero_volume_days_20d": zero_volume_days_20d,
                    "zero_volume_days_60d": zero_volume_days_60d,
                    "active_trading_days_60d": active_trading_days_60d,
                    "stale_close_days_20d": stale_close_days_20d,
                    "stale_close_days_60d": stale_close_days_60d,
                    "data_quality_extreme_return_flag": data_quality_extreme_return_flag,
                }
            )

        if not rows:
            return _empty_snapshot()
        return pl.DataFrame(rows, schema=LIQUIDITY_QUALITY_SNAPSHOT_SCHEMA).sort("ticker")

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        if snapshot.is_empty():
            return _empty_rankings()
        if not self._has_required_ranking_inputs(snapshot):
            raise ValidationError("liquidity_quality_v1 snapshot is missing required ranking inputs")
        for column in LIQUIDITY_QUALITY_SNAPSHOT_METRIC_KEYS:
            invalid_count = snapshot.filter(pl.col(column).is_null() | (~pl.col(column).is_finite())).height
            if invalid_count > 0:
                raise ValidationError(
                    f"liquidity_quality_v1 snapshot contains non-finite ranking input: {column}"
                )
        ranking_frames = []
        for partition in snapshot.partition_by(["run_id", "market"], maintain_order=True):
            ranking_frames.append(self._assign_single_run_market_rankings(partition))
        return (
            pl.concat(ranking_frames)
            .sort(["run_id", "market", "horizon", "rank"])
            .select(self._ranking_columns())
        )

    def _has_required_ranking_inputs(self, snapshot: pl.DataFrame) -> bool:
        required = {"run_id", "market", "ticker", *LIQUIDITY_QUALITY_SNAPSHOT_METRIC_KEYS}
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
        log_traded_value_20d = [math.log1p(float(row["avg_traded_value_20d_local"])) for row in rows]
        log_traded_value_60d = [math.log1p(float(row["avg_traded_value_60d_local"])) for row in rows]
        amihud_20d = [math.log(max(float(row["amihud_illiquidity_20d"]), 1e-300)) for row in rows]
        amihud_60d = [math.log(max(float(row["amihud_illiquidity_60d"]), 1e-300)) for row in rows]
        traded_value_cv = [float(row["traded_value_cv_60d"]) for row in rows]
        concentration = [float(row["traded_value_concentration_60d"]) for row in rows]
        range_20d = [float(row["median_range_pct_20d"]) for row in rows]
        range_60d = [float(row["median_range_pct_60d"]) for row in rows]

        score_log_traded_value_20d = _percentile_scores(log_traded_value_20d)
        score_log_traded_value_60d = _percentile_scores(log_traded_value_60d)
        score_amihud_20d = _percentile_scores(amihud_20d, higher_is_better=False)
        score_amihud_60d = _percentile_scores(amihud_60d, higher_is_better=False)
        score_traded_value_stability_60d = _percentile_scores(traded_value_cv, higher_is_better=False)
        score_traded_value_concentration_60d = _percentile_scores(concentration, higher_is_better=False)
        score_range_tightness_20d = _percentile_scores(range_20d, higher_is_better=False)
        score_range_tightness_60d = _percentile_scores(range_60d, higher_is_better=False)

        scored_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows):
            market = Market(str(row["market"]))
            ratio = float(row["traded_value_5d_to_20d_ratio"])
            score_recent_traded_value_ratio = _clamp(ratio, 0.0, 2.0) / 2.0
            continuity_raw = (
                float(row["active_trading_days_60d"]) / 60.0
                - float(row["zero_volume_days_60d"]) / 60.0
                - float(row["stale_close_days_60d"]) / 60.0
            )
            score_trading_continuity_60d = _clamp(continuity_raw, 0.0, 1.0)
            depth_score = (
                0.55 * score_log_traded_value_20d[index]
                + 0.35 * score_log_traded_value_60d[index]
                + 0.10 * score_recent_traded_value_ratio
            )
            friction_score = (
                0.55 * score_amihud_60d[index]
                + 0.30 * score_range_tightness_60d[index]
                + 0.15 * score_trading_continuity_60d
            )
            stability_score = (
                0.45 * score_traded_value_stability_60d[index]
                + 0.30 * score_traded_value_concentration_60d[index]
                + 0.25 * score_recent_traded_value_ratio
            )
            penalty_score = 0.0
            if ratio < 0.75:
                penalty_score += 0.10
            if ratio > 3.00:
                penalty_score += 0.05
            if float(row["zero_volume_days_20d"]) > 0.0:
                penalty_score += 0.05
            if float(row["traded_value_concentration_60d"]) > 0.20:
                penalty_score += 0.05
            if score_amihud_60d[index] <= 0.20:
                penalty_score += 0.05
            if score_range_tightness_60d[index] <= 0.20:
                penalty_score += 0.05
            if float(row["stale_close_days_20d"]) >= 5.0:
                penalty_score += 0.10
            if float(row["data_quality_extreme_return_flag"]) == 1.0:
                penalty_score += 0.20

            row.update(
                {
                    "score_log_traded_value_20d": score_log_traded_value_20d[index],
                    "score_log_traded_value_60d": score_log_traded_value_60d[index],
                    "score_amihud_20d": score_amihud_20d[index],
                    "score_amihud_60d": score_amihud_60d[index],
                    "score_traded_value_stability_60d": score_traded_value_stability_60d[index],
                    "score_traded_value_concentration_60d": score_traded_value_concentration_60d[index],
                    "score_range_tightness_20d": score_range_tightness_20d[index],
                    "score_range_tightness_60d": score_range_tightness_60d[index],
                    "score_recent_traded_value_ratio": score_recent_traded_value_ratio,
                    "score_trading_continuity_60d": score_trading_continuity_60d,
                    "depth_score": depth_score,
                    "friction_score": friction_score,
                    "stability_score": stability_score,
                    "penalty_score": penalty_score,
                    "tag_risk_thin_liquidity": (
                        1.0
                        if float(row["avg_traded_value_20d_local"]) <= self.liquidity_floor[market] * 1.25
                        else 0.0
                    ),
                    "tag_risk_recent_liquidity_fade": 1.0 if ratio < 0.75 else 0.0,
                    "tag_risk_traded_value_spike": (
                        1.0 if float(row["traded_value_concentration_60d"]) > 0.20 else 0.0
                    ),
                    "tag_risk_high_impact": 1.0 if score_amihud_60d[index] <= 0.20 else 0.0,
                    "tag_risk_wide_range": 1.0 if score_range_tightness_60d[index] <= 0.20 else 0.0,
                    "tag_risk_stale_trading": 1.0 if float(row["stale_close_days_60d"]) > 0.0 else 0.0,
                    "tag_risk_data_quality_warning": (
                        1.0 if float(row["data_quality_extreme_return_flag"]) == 1.0 else 0.0
                    ),
                    "tag_positive_deep_liquidity": 1.0 if depth_score >= 0.80 else 0.0,
                    "tag_positive_stable_liquidity": 1.0 if stability_score >= 0.75 else 0.0,
                }
            )
            scored_rows.append(row)

        ranking_rows: list[dict[str, object]] = []
        for horizon in self.horizon_order:
            horizon_rows = []
            for row in scored_rows:
                if horizon == "composite":
                    score = (
                        0.45 * float(row["depth_score"])
                        + 0.35 * float(row["friction_score"])
                        + 0.20 * float(row["stability_score"])
                        - float(row["penalty_score"])
                    )
                elif horizon == "shortterm":
                    score = (
                        0.45 * float(row["score_log_traded_value_20d"])
                        + 0.25 * float(row["score_recent_traded_value_ratio"])
                        + 0.20 * float(row["score_amihud_20d"])
                        + 0.10 * float(row["score_range_tightness_20d"])
                        - float(row["penalty_score"])
                    )
                elif horizon == "stable":
                    score = (
                        0.35 * float(row["score_log_traded_value_60d"])
                        + 0.25 * float(row["score_traded_value_stability_60d"])
                        + 0.20 * float(row["score_traded_value_concentration_60d"])
                        + 0.20 * float(row["score_trading_continuity_60d"])
                        - float(row["penalty_score"])
                    )
                else:
                    raise ValidationError(f"unknown horizon {horizon}")
                if not math.isfinite(score):
                    raise ValidationError("liquidity_quality_v1 produced a non-finite ranking score")
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
        return pl.DataFrame(ranking_rows, schema=LIQUIDITY_QUALITY_RANKING_SCHEMA).select(
            self._ranking_columns()
        )


LIQUIDITY_QUALITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=LIQUIDITY_QUALITY_PROFILE_ID,
    factory=LiquidityQualityV1Profile,
)
