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
            traded_values_20d = [
                close * volume for close, volume in zip(closes_float[-20:], volumes_float[-20:], strict=True)
            ]
            avg_traded_value_20d_local = _mean(traded_values_20d)
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
            if stale_close_days_20d >= 5.0:
                continue

            returns = [
                adjusted_closes_float[index] / adjusted_closes_float[index - 1] - 1.0
                for index in range(1, len(adjusted_closes_float))
            ]
            returns_60d = returns[-60:]
            volatility_60d = _std(returns_60d, ddof=self.stdev_ddof)
            if volatility_60d is None or volatility_60d <= self.volatility_floor:
                continue
            extreme_returns_200d = [
                adjusted_closes_float[index] / adjusted_closes_float[index - 1] - 1.0
                for index in range(len(adjusted_closes_float) - 200, len(adjusted_closes_float))
            ]
            if any(abs(value) > 0.80 for value in extreme_returns_200d):
                continue

            trend_slope_60d, trend_r2_60d = _ols_slope_r2(adjusted_closes_float[-60:])
            uptrend_r2_60d = trend_r2_60d if trend_slope_60d > 0.0 else 0.0
            trend_consistency_60d = float(sum(1 for value in returns_60d if value > 0.0)) / 60.0
            sma_50d = _mean(adjusted_closes_float[-50:])
            sma_200d = _mean(adjusted_closes_float[-200:])
            price_vs_sma_50d = latest_adjusted_close / sma_50d - 1.0
            price_vs_sma_200d = latest_adjusted_close / sma_200d - 1.0
            sma_50d_vs_sma_200d = sma_50d / sma_200d - 1.0
            pct_below_120d_high = latest_adjusted_close / max(adjusted_closes_float[-120:]) - 1.0
            max_drawdown_120d = _max_drawdown(adjusted_closes_float[-120:])
            return_20d = latest_adjusted_close / adjusted_closes_float[-21] - 1.0
            return_60d = latest_adjusted_close / adjusted_closes_float[-61] - 1.0
            return_120d = latest_adjusted_close / adjusted_closes_float[-121] - 1.0

            computed = [
                avg_traded_value_20d_local,
                return_20d,
                return_60d,
                return_120d,
                volatility_60d,
                trend_slope_60d,
                trend_r2_60d,
                uptrend_r2_60d,
                trend_consistency_60d,
                price_vs_sma_50d,
                price_vs_sma_200d,
                sma_50d_vs_sma_200d,
                pct_below_120d_high,
                max_drawdown_120d,
                zero_volume_days_20d,
                active_trading_days_60d,
                stale_close_days_20d,
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
                    "return_20d": return_20d,
                    "return_60d": return_60d,
                    "return_120d": return_120d,
                    "volatility_60d": volatility_60d,
                    "trend_slope_60d": trend_slope_60d,
                    "trend_r2_60d": trend_r2_60d,
                    "uptrend_r2_60d": uptrend_r2_60d,
                    "trend_consistency_60d": trend_consistency_60d,
                    "price_vs_sma_50d": price_vs_sma_50d,
                    "price_vs_sma_200d": price_vs_sma_200d,
                    "sma_50d_vs_sma_200d": sma_50d_vs_sma_200d,
                    "pct_below_120d_high": pct_below_120d_high,
                    "max_drawdown_120d": max_drawdown_120d,
                    "zero_volume_days_20d": zero_volume_days_20d,
                    "active_trading_days_60d": active_trading_days_60d,
                    "stale_close_days_20d": stale_close_days_20d,
                }
            )

        if not rows:
            return _empty_snapshot()
        return pl.DataFrame(rows, schema=TREND_QUALITY_SNAPSHOT_SCHEMA).sort("ticker")

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        if snapshot.is_empty():
            return _empty_rankings()
        if not self._has_required_ranking_inputs(snapshot):
            raise ValidationError("trend_quality_v1 snapshot is missing required ranking inputs")
        for column in TREND_QUALITY_SNAPSHOT_METRIC_KEYS:
            if not snapshot.schema[column].is_numeric():
                raise ValidationError(f"trend_quality_v1 snapshot contains non-numeric ranking input: {column}")
            invalid_count = snapshot.filter(pl.col(column).is_null() | (~pl.col(column).is_finite())).height
            if invalid_count > 0:
                raise ValidationError(f"trend_quality_v1 snapshot contains non-finite ranking input: {column}")

        ranking_frames = []
        for partition in snapshot.partition_by(["run_id", "market"], maintain_order=True):
            if partition["asof_bar_date_yyyymmdd"].n_unique() != 1:
                raise ValidationError("trend_quality_v1 snapshot contains mixed as-of dates")
            ranking_frames.append(self._assign_single_run_market_rankings(partition))
        if not ranking_frames:
            return _empty_rankings()
        return (
            pl.concat(ranking_frames)
            .with_columns(pl.col("horizon").replace_strict({horizon: index for index, horizon in enumerate(self.horizon_order)}).alias("_horizon_order"))
            .sort(["run_id", "market", "_horizon_order", "rank", "ticker"])
            .drop("_horizon_order")
            .select(self._ranking_columns())
        )

    def _has_required_ranking_inputs(self, snapshot: pl.DataFrame) -> bool:
        required = {"run_id", "market", "ticker", *TREND_QUALITY_SNAPSHOT_METRIC_KEYS}
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
        score_return_20d = _percentile_scores([float(row["return_20d"]) for row in rows])
        score_return_60d = _percentile_scores([float(row["return_60d"]) for row in rows])
        score_return_120d = _percentile_scores([float(row["return_120d"]) for row in rows])
        score_trend_slope_60d = _positive_only_percentile_scores(
            [float(row["trend_slope_60d"]) for row in rows]
        )
        score_trend_consistency_60d = _percentile_scores(
            [float(row["trend_consistency_60d"]) for row in rows]
        )
        score_price_vs_sma_50d = _percentile_scores([float(row["price_vs_sma_50d"]) for row in rows])
        score_price_vs_sma_200d = _percentile_scores([float(row["price_vs_sma_200d"]) for row in rows])
        score_sma_50d_vs_sma_200d = _percentile_scores([float(row["sma_50d_vs_sma_200d"]) for row in rows])
        score_pct_below_120d_high = _percentile_scores([float(row["pct_below_120d_high"]) for row in rows])
        score_drawdown_control_120d = _percentile_scores([float(row["max_drawdown_120d"]) for row in rows])

        scored_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows):
            trend_slope_60d = float(row["trend_slope_60d"])
            return_20d = float(row["return_20d"])
            return_60d = float(row["return_60d"])
            price_vs_sma_50d = float(row["price_vs_sma_50d"])
            price_vs_sma_200d = float(row["price_vs_sma_200d"])
            sma_50d_vs_sma_200d = float(row["sma_50d_vs_sma_200d"])
            max_drawdown_120d = float(row["max_drawdown_120d"])
            uptrend_r2_60d = float(row["uptrend_r2_60d"])
            trend_r2_60d = float(row["trend_r2_60d"])
            trend_consistency_60d = float(row["trend_consistency_60d"])
            stale_close_days_20d = float(row["stale_close_days_20d"])

            score_uptrend_r2_60d = uptrend_r2_60d if trend_slope_60d > 0.0 else 0.0
            trend_strength_score = (
                0.35 * score_return_120d[index]
                + 0.25 * score_return_60d[index]
                + 0.25 * score_trend_slope_60d[index]
                + 0.15 * score_price_vs_sma_200d[index]
            )
            raw_trend_cleanliness_score = (
                0.35 * score_uptrend_r2_60d
                + 0.25 * score_trend_consistency_60d[index]
                + 0.25 * score_sma_50d_vs_sma_200d[index]
                + 0.15 * score_drawdown_control_120d[index]
            )
            trend_cleanliness_cap_score = self._trend_cleanliness_cap_score(
                trend_slope_60d=trend_slope_60d,
                uptrend_r2_60d=uptrend_r2_60d,
            )
            trend_cleanliness_score = min(raw_trend_cleanliness_score, trend_cleanliness_cap_score)
            breakout_position_score = (
                0.40 * score_pct_below_120d_high[index]
                + 0.25 * score_return_20d[index]
                + 0.20 * score_price_vs_sma_50d[index]
                + 0.15 * score_sma_50d_vs_sma_200d[index]
            )
            drawdown_control_score = score_drawdown_control_120d[index]
            weak_structure_fail_count = float(
                sum(
                    1
                    for failed in (
                        trend_slope_60d <= 0.0,
                        return_60d < 0.0,
                        price_vs_sma_50d < 0.0,
                        price_vs_sma_200d < 0.0,
                        sma_50d_vs_sma_200d < 0.0,
                    )
                    if failed
                )
            )
            weak_positive_structure_count = float(
                sum(
                    1
                    for weak in (
                        0.0 < trend_slope_60d < 0.0002,
                        0.0 <= return_60d < 0.02,
                        0.0 <= price_vs_sma_50d < 0.01,
                        0.0 <= price_vs_sma_200d < 0.02,
                        0.0 <= sma_50d_vs_sma_200d < 0.01,
                    )
                    if weak
                )
            )
            hard_structure_cap_score = self._hard_structure_cap_score(
                weak_structure_fail_count=weak_structure_fail_count,
                max_drawdown_120d=max_drawdown_120d,
            )
            trend_magnitude_cap_score = self._trend_magnitude_cap_score(
                weak_structure_fail_count=weak_structure_fail_count,
                weak_positive_structure_count=weak_positive_structure_count,
            )
            overextension_cap_score = self._overextension_cap_score(
                return_20d=return_20d,
                price_vs_sma_50d=price_vs_sma_50d,
            )
            structure_cap_score = min(
                hard_structure_cap_score,
                trend_magnitude_cap_score,
                trend_cleanliness_cap_score,
                overextension_cap_score,
            )
            penalty_score = 0.0
            if stale_close_days_20d >= 3.0:
                penalty_score += 0.10
            severely_overextended = return_20d > 0.50 or price_vs_sma_50d > 0.30
            moderately_overextended = return_20d > 0.30 or price_vs_sma_50d > 0.20
            if severely_overextended:
                penalty_score += 0.15
            elif moderately_overextended:
                penalty_score += 0.10
            if trend_slope_60d <= 0.0:
                penalty_score += 0.10
            if return_60d < 0.0:
                penalty_score += 0.10
            if price_vs_sma_50d < 0.0:
                penalty_score += 0.10
            if price_vs_sma_200d < 0.0:
                penalty_score += 0.10
            if sma_50d_vs_sma_200d < 0.0:
                penalty_score += 0.10
            if weak_structure_fail_count == 0.0:
                penalty_score += 0.05 * weak_positive_structure_count
            if trend_slope_60d > 0.0 and uptrend_r2_60d < 0.50:
                penalty_score += 0.05
            if max_drawdown_120d <= -0.25:
                penalty_score += 0.10

            row.update(
                {
                    "score_return_20d": score_return_20d[index],
                    "score_return_60d": score_return_60d[index],
                    "score_return_120d": score_return_120d[index],
                    "score_trend_slope_60d": score_trend_slope_60d[index],
                    "score_uptrend_r2_60d": score_uptrend_r2_60d,
                    "score_trend_consistency_60d": score_trend_consistency_60d[index],
                    "score_price_vs_sma_50d": score_price_vs_sma_50d[index],
                    "score_price_vs_sma_200d": score_price_vs_sma_200d[index],
                    "score_sma_50d_vs_sma_200d": score_sma_50d_vs_sma_200d[index],
                    "score_pct_below_120d_high": score_pct_below_120d_high[index],
                    "score_drawdown_control_120d": score_drawdown_control_120d[index],
                    "trend_strength_score": trend_strength_score,
                    "trend_cleanliness_score": trend_cleanliness_score,
                    "breakout_position_score": breakout_position_score,
                    "drawdown_control_score": drawdown_control_score,
                    "trend_cleanliness_cap_score": trend_cleanliness_cap_score,
                    "hard_structure_cap_score": hard_structure_cap_score,
                    "weak_structure_fail_count": weak_structure_fail_count,
                    "weak_positive_structure_count": weak_positive_structure_count,
                    "trend_magnitude_cap_score": trend_magnitude_cap_score,
                    "overextension_cap_score": overextension_cap_score,
                    "structure_cap_score": structure_cap_score,
                    "penalty_score": penalty_score,
                    "tag_structure_uptrend": (
                        1.0
                        if trend_slope_60d > 0.0
                        and return_60d > 0.0
                        and price_vs_sma_50d > 0.0
                        and sma_50d_vs_sma_200d > 0.0
                        else 0.0
                    ),
                    "tag_structure_breakout_proximity": 1.0 if float(row["pct_below_120d_high"]) >= -0.03 else 0.0,
                    "tag_structure_consistent_uptrend": (
                        1.0
                        if trend_slope_60d > 0.0
                        and return_60d > 0.0
                        and trend_r2_60d >= 0.50
                        and trend_consistency_60d >= 0.55
                        else 0.0
                    ),
                    "tag_structure_nonpositive_60d_slope": 1.0 if trend_slope_60d <= 0.0 else 0.0,
                    "tag_structure_negative_60d_return": 1.0 if return_60d < 0.0 else 0.0,
                    "tag_structure_below_sma_50d": 1.0 if price_vs_sma_50d < 0.0 else 0.0,
                    "tag_structure_below_sma_200d": 1.0 if price_vs_sma_200d < 0.0 else 0.0,
                    "tag_structure_sma_50d_below_sma_200d": (
                        1.0 if sma_50d_vs_sma_200d < 0.0 else 0.0
                    ),
                    "tag_structure_weak_trend_component": 1.0 if weak_structure_fail_count > 0.0 else 0.0,
                    "tag_structure_large_drawdown": 1.0 if max_drawdown_120d <= -0.25 else 0.0,
                    "tag_structure_overextended": 1.0 if moderately_overextended else 0.0,
                    "tag_structure_cap_active": 1.0 if structure_cap_score < 1.0 else 0.0,
                    "tag_data_stale_trading": 1.0 if stale_close_days_20d >= 3.0 else 0.0,
                }
            )
            scored_rows.append(row)

        ranking_rows: list[dict[str, object]] = []
        for horizon in self.horizon_order:
            horizon_rows = []
            for row in scored_rows:
                if horizon == "composite":
                    raw_score = (
                        0.40 * float(row["trend_strength_score"])
                        + 0.30 * float(row["trend_cleanliness_score"])
                        + 0.20 * float(row["breakout_position_score"])
                        + 0.10 * float(row["drawdown_control_score"])
                    )
                elif horizon == "shortterm":
                    raw_score = (
                        0.20 * float(row["score_return_20d"])
                        + 0.15 * float(row["score_return_60d"])
                        + 0.15 * float(row["score_pct_below_120d_high"])
                        + 0.15 * float(row["score_price_vs_sma_50d"])
                        + 0.10 * float(row["score_sma_50d_vs_sma_200d"])
                        + 0.10 * float(row["score_trend_slope_60d"])
                        + 0.05 * float(row["score_uptrend_r2_60d"])
                        + 0.05 * float(row["score_trend_consistency_60d"])
                        + 0.05 * float(row["score_drawdown_control_120d"])
                    )
                elif horizon == "midterm":
                    raw_score = (
                        0.25 * float(row["score_return_120d"])
                        + 0.20 * float(row["score_trend_slope_60d"])
                        + 0.15 * float(row["score_sma_50d_vs_sma_200d"])
                        + 0.15 * float(row["score_drawdown_control_120d"])
                        + 0.10 * float(row["score_trend_consistency_60d"])
                        + 0.10 * float(row["trend_cleanliness_score"])
                        + 0.05 * float(row["score_uptrend_r2_60d"])
                    )
                else:
                    raise ValidationError(f"unknown horizon {horizon}")
                score = min(raw_score - float(row["penalty_score"]), float(row["structure_cap_score"]))
                ranking_values = [float(row[key]) for key in self.ranking_metric_keys] + [score]
                if not _all_finite(ranking_values):
                    raise ValidationError("trend_quality_v1 produced a non-finite ranking metric")
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
        return pl.DataFrame(ranking_rows, schema=TREND_QUALITY_RANKING_SCHEMA).select(
            self._ranking_columns()
        )

    def _trend_cleanliness_cap_score(self, *, trend_slope_60d: float, uptrend_r2_60d: float) -> float:
        if trend_slope_60d <= 0.0:
            return 0.35
        if uptrend_r2_60d < 0.10:
            return 0.45
        if uptrend_r2_60d < 0.25:
            return 0.65
        if uptrend_r2_60d < 0.50:
            return 0.70
        return 1.0

    def _hard_structure_cap_score(
        self,
        *,
        weak_structure_fail_count: float,
        max_drawdown_120d: float,
    ) -> float:
        large_drawdown = max_drawdown_120d <= -0.25
        if weak_structure_fail_count >= 5.0 and large_drawdown:
            return 0.15
        if weak_structure_fail_count >= 5.0:
            return 0.20
        if weak_structure_fail_count == 4.0 and large_drawdown:
            return 0.20
        if weak_structure_fail_count == 4.0:
            return 0.25
        if weak_structure_fail_count == 3.0 and large_drawdown:
            return 0.25
        if weak_structure_fail_count == 3.0:
            return 0.30
        if weak_structure_fail_count == 2.0 and large_drawdown:
            return 0.30
        if weak_structure_fail_count == 2.0:
            return 0.35
        if weak_structure_fail_count == 1.0 and large_drawdown:
            return 0.35
        if weak_structure_fail_count == 1.0:
            return 0.40
        if weak_structure_fail_count == 0.0 and large_drawdown:
            return 0.65
        return 1.0

    def _trend_magnitude_cap_score(
        self,
        *,
        weak_structure_fail_count: float,
        weak_positive_structure_count: float,
    ) -> float:
        if weak_structure_fail_count != 0.0:
            return 1.0
        if weak_positive_structure_count >= 3.0:
            return 0.45
        if weak_positive_structure_count == 2.0:
            return 0.60
        if weak_positive_structure_count == 1.0:
            return 0.75
        return 1.0

    def _overextension_cap_score(self, *, return_20d: float, price_vs_sma_50d: float) -> float:
        if return_20d > 0.50 or price_vs_sma_50d > 0.30:
            return 0.60
        if return_20d > 0.30 or price_vs_sma_50d > 0.20:
            return 0.75
        return 1.0


TREND_QUALITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=TREND_QUALITY_PROFILE_ID,
    factory=TrendQualityV1Profile,
)
