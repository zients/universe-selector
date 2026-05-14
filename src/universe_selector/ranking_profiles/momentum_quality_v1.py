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


MOMENTUM_QUALITY_PROFILE_ID = "momentum_quality_v1"
MOMENTUM_QUALITY_SCORE_METHOD = "raw_weighted_momentum_quality_v1"
MOMENTUM_QUALITY_RANK_INTERPRETATION_NOTE = (
    "Momentum quality scores are raw weighted composites; risk tags describe absolute risk conditions and scores are not capped at 100."
)
MOMENTUM_QUALITY_HORIZON_ORDER = ("composite", "swing", "midterm")

MOMENTUM_QUALITY_SNAPSHOT_METRIC_KEYS = (
    "profile_metrics_version",
    "avg_traded_value_20d_local",
    "avg_traded_value_5d_local",
    "momentum_return_12_1",
    "momentum_return_6_1",
    "volatility_12_1",
    "volatility_6_1",
    "risk_adjusted_momentum_12_1",
    "risk_adjusted_momentum_6_1",
    "short_term_strength_20d",
    "ma20",
    "ma60",
    "ma120",
    "ma200",
    "moving_average_structure_raw_score",
    "max_drawdown_252d",
    "above_ma60_ratio_126d",
    "positive_21d_return_ratio_126d",
    "uptrend_consistency_raw_score",
    "short_term_extension_20d",
    "distance_from_ma20",
    "prior_60d_high_adjusted_close",
    "data_quality_extreme_return_flag",
)

MOMENTUM_QUALITY_RANKING_METRIC_KEYS = (
    "score_risk_adjusted_momentum_12_1",
    "score_risk_adjusted_momentum_6_1",
    "score_short_term_strength_20d",
    "score_short_term_extension_20d",
    "score_distance_from_ma20",
    "score_volatility_12_1",
    "score_volatility_6_1",
    "moving_average_structure_score",
    "drawdown_control_score",
    "uptrend_consistency_score",
    "trend_quality_score",
    "momentum_blend_score",
    "overheat_score",
    "overheat_penalty_score",
    "tag_risk_overheated",
    "tag_risk_extended_from_ma20",
    "tag_risk_high_volatility",
    "tag_risk_large_drawdown",
    "tag_risk_weak_trend_quality",
    "tag_risk_thin_recent_volume",
    "tag_risk_data_quality_warning",
    "tag_positive_strong_momentum",
    "tag_positive_stable_uptrend",
    "tag_positive_early_breakout",
)

MOMENTUM_QUALITY_RAW_FACTOR_KEYS = (
    "risk_adjusted_momentum_12_1",
    "risk_adjusted_momentum_6_1",
    "short_term_strength_20d",
    "moving_average_structure_raw_score",
    "max_drawdown_252d",
    "uptrend_consistency_raw_score",
    "short_term_extension_20d",
    "distance_from_ma20",
    "volatility_12_1",
    "volatility_6_1",
    "avg_traded_value_20d_local",
    "avg_traded_value_5d_local",
    "prior_60d_high_adjusted_close",
    "data_quality_extreme_return_flag",
)

MOMENTUM_QUALITY_SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    **{key: pl.Float64 for key in MOMENTUM_QUALITY_SNAPSHOT_METRIC_KEYS},
}

MOMENTUM_QUALITY_RANKING_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "horizon": pl.String,
    "ticker": pl.String,
    **{key: pl.Float64 for key in MOMENTUM_QUALITY_RANKING_METRIC_KEYS},
    "score": pl.Float64,
    "rank": pl.Int64,
}


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


def _moving_average(values: list[float], window: int, end_index: int) -> float:
    start = end_index - window + 1
    return _mean(values[start : end_index + 1])


def _moving_average_slope(values: list[float], window: int, end_index: int, lag: int) -> float:
    return _moving_average(values, window, end_index) - _moving_average(values, window, end_index - lag)


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=MOMENTUM_QUALITY_SNAPSHOT_SCHEMA)


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema=MOMENTUM_QUALITY_RANKING_SCHEMA)


def _immutable_market_float_mapping(value: Mapping[Market, float]) -> Mapping[Market, float]:
    return MappingProxyType({market: float(value[market]) for market in Market})


def _moving_average_structure_score(adjusted_closes: list[float], end_index: int) -> tuple[float, float, float, float, float]:
    ma20 = _moving_average(adjusted_closes, 20, end_index)
    ma60 = _moving_average(adjusted_closes, 60, end_index)
    ma120 = _moving_average(adjusted_closes, 120, end_index)
    ma200 = _moving_average(adjusted_closes, 200, end_index)
    score = 0.0
    if adjusted_closes[end_index] > ma20:
        score += 15.0
    if ma20 > ma60:
        score += 20.0
    if ma60 > ma120:
        score += 20.0
    if ma120 > ma200:
        score += 20.0
    if _moving_average_slope(adjusted_closes, 60, end_index, 20) > 0.0:
        score += 10.0
    if _moving_average_slope(adjusted_closes, 120, end_index, 20) > 0.0:
        score += 10.0
    if _moving_average_slope(adjusted_closes, 200, end_index, 20) > 0.0:
        score += 5.0
    return ma20, ma60, ma120, ma200, score


def _above_ma60_ratio(adjusted_closes: list[float], end_index: int) -> float:
    start_index = end_index - 125
    above_count = 0
    for index in range(start_index, end_index + 1):
        if adjusted_closes[index] > _moving_average(adjusted_closes, 60, index):
            above_count += 1
    return above_count / 126.0


def _positive_21d_return_ratio(adjusted_closes: list[float], end_index: int) -> float:
    start_index = end_index - 126
    positive_count = 0
    for index in range(start_index, end_index, 21):
        if adjusted_closes[index + 21] / adjusted_closes[index] - 1.0 > 0.0:
            positive_count += 1
    return positive_count / 6.0


@dataclass(frozen=True)
class MomentumQualityV1Profile:
    profile_id: Literal["momentum_quality_v1"] = MOMENTUM_QUALITY_PROFILE_ID
    min_history_bars: int = 274
    volatility_floor: float = 0.001
    active_trading_min_days_274: int = 230
    zero_volume_max_days_20: int = 2
    price_floor: Mapping[Market, float] = field(
        default_factory=lambda: {Market.TW: 10.0, Market.US: 5.0}
    )
    liquidity_floor: Mapping[Market, float] = field(
        default_factory=lambda: {Market.TW: 20_000_000.0, Market.US: 5_000_000.0}
    )
    snapshot_metric_keys: tuple[str, ...] = MOMENTUM_QUALITY_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = MOMENTUM_QUALITY_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = MOMENTUM_QUALITY_SNAPSHOT_METRIC_KEYS
    horizon_order: tuple[str, ...] = MOMENTUM_QUALITY_HORIZON_ORDER
    stdev_ddof: int = 1
    score_method: str = MOMENTUM_QUALITY_SCORE_METHOD
    rank_interpretation_note: str = MOMENTUM_QUALITY_RANK_INTERPRETATION_NOTE

    def __post_init__(self) -> None:
        object.__setattr__(self, "price_floor", _immutable_market_float_mapping(self.price_floor))
        object.__setattr__(self, "liquidity_floor", _immutable_market_float_mapping(self.liquidity_floor))
        object.__setattr__(self, "snapshot_metric_keys", tuple(str(key) for key in self.snapshot_metric_keys))
        object.__setattr__(self, "ranking_metric_keys", tuple(str(key) for key in self.ranking_metric_keys))
        object.__setattr__(self, "inspect_metric_keys", tuple(str(key) for key in self.inspect_metric_keys))
        object.__setattr__(self, "horizon_order", tuple(str(horizon) for horizon in self.horizon_order))

    def validate(self) -> None:
        if self.profile_id != MOMENTUM_QUALITY_PROFILE_ID:
            raise ValidationError("profile_id must be momentum_quality_v1")
        if self.min_history_bars != 274:
            raise ValidationError("momentum_quality_v1 requires min_history_bars to be 274")
        if self.snapshot_metric_keys != MOMENTUM_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("snapshot metric keys must match momentum_quality_v1")
        if self.ranking_metric_keys != MOMENTUM_QUALITY_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match momentum_quality_v1")
        if self.inspect_metric_keys != MOMENTUM_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("inspect metric keys must match momentum_quality_v1 raw metrics")
        if self.horizon_order != MOMENTUM_QUALITY_HORIZON_ORDER:
            raise ValidationError("horizon order must be composite, swing, midterm")
        if set(self.price_floor) != set(Market):
            raise ValidationError("price_floor must contain exactly supported markets")
        if set(self.liquidity_floor) != set(Market):
            raise ValidationError("liquidity_floor must contain exactly supported markets")
        if self.stdev_ddof != 1:
            raise ValidationError("stdev_ddof must be 1")
        if self.score_method != MOMENTUM_QUALITY_SCORE_METHOD:
            raise ValidationError(f"score_method must be {MOMENTUM_QUALITY_SCORE_METHOD}")
        if self.rank_interpretation_note != MOMENTUM_QUALITY_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match momentum_quality_v1")

    def ranking_config_payload(self) -> dict[str, object]:
        return {
            "ranking_profile": self.profile_id,
            "min_history_bars": self.min_history_bars,
            "price_floor": {market.value: self.price_floor[market] for market in Market},
            "liquidity_floor": {market.value: self.liquidity_floor[market] for market in Market},
            "active_trading_min_days_274": self.active_trading_min_days_274,
            "zero_volume_max_days_20": self.zero_volume_max_days_20,
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

            latest_bar_date = ticker_bars["bar_date"][-1]
            if latest_bar_date != run_latest_bar_date:
                continue

            tail = ticker_bars.tail(self.min_history_bars)
            closes = tail["close"].to_list()
            adjusted_closes = tail["adjusted_close"].to_list()
            volumes = tail["volume"].to_list()
            if not _all_finite(closes + adjusted_closes + volumes):
                continue

            closes_float = [float(value) for value in closes]
            adjusted_closes_float = [float(value) for value in adjusted_closes]
            volumes_float = [float(value) for value in volumes]
            if any(value <= 0.0 for value in closes_float):
                continue
            if any(value <= 0.0 for value in adjusted_closes_float):
                continue
            if any(value < 0.0 for value in volumes_float):
                continue

            latest_20_volumes = volumes_float[-20:]
            if sum(1 for value in latest_20_volumes if value == 0.0) > self.zero_volume_max_days_20:
                continue
            if sum(1 for value in volumes_float if value > 0.0) < self.active_trading_min_days_274:
                continue

            returns = [
                adjusted_closes_float[index] / adjusted_closes_float[index - 1] - 1.0
                for index in range(1, len(adjusted_closes_float))
            ]
            returns_12_1 = returns[0:252]
            returns_6_1 = returns[126:252]
            volatility_12_1 = _std(returns_12_1, ddof=self.stdev_ddof)
            volatility_6_1 = _std(returns_6_1, ddof=self.stdev_ddof)
            if volatility_12_1 is None or volatility_6_1 is None:
                continue
            if volatility_12_1 <= self.volatility_floor or volatility_6_1 <= self.volatility_floor:
                continue

            avg_traded_value_20d_local = sum(
                close * volume
                for close, volume in zip(closes_float[-20:], volumes_float[-20:], strict=True)
            ) / 20.0
            avg_traded_value_5d_local = sum(
                close * volume
                for close, volume in zip(closes_float[-5:], volumes_float[-5:], strict=True)
            ) / 5.0
            latest_close = closes_float[-1]
            if avg_traded_value_20d_local < self.liquidity_floor[market]:
                continue
            if latest_close < self.price_floor[market]:
                continue

            end_index = len(adjusted_closes_float) - 1
            adjusted_close_t = adjusted_closes_float[-1]
            momentum_return_12_1 = adjusted_closes_float[-22] / adjusted_closes_float[-274] - 1.0
            momentum_return_6_1 = adjusted_closes_float[-22] / adjusted_closes_float[-148] - 1.0
            short_term_strength_20d = adjusted_close_t / adjusted_closes_float[-21] - 1.0
            risk_adjusted_momentum_12_1 = momentum_return_12_1 / volatility_12_1
            risk_adjusted_momentum_6_1 = momentum_return_6_1 / volatility_6_1
            ma20, ma60, ma120, ma200, moving_average_structure_raw_score = (
                _moving_average_structure_score(adjusted_closes_float, end_index)
            )
            max_drawdown_252d = _max_drawdown(adjusted_closes_float[-252:])
            above_ma60_ratio_126d = _above_ma60_ratio(adjusted_closes_float, end_index)
            positive_21d_return_ratio_126d = _positive_21d_return_ratio(
                adjusted_closes_float, end_index
            )
            uptrend_consistency_raw_score = 100.0 * (
                0.70 * above_ma60_ratio_126d + 0.30 * positive_21d_return_ratio_126d
            )
            distance_from_ma20 = adjusted_close_t / ma20 - 1.0
            prior_60d_high_adjusted_close = max(adjusted_closes_float[-61:-1])
            data_quality_extreme_return_flag = (
                1.0 if any(abs(value) > 0.80 for value in returns) else 0.0
            )

            computed_factors = [
                avg_traded_value_20d_local,
                avg_traded_value_5d_local,
                momentum_return_12_1,
                momentum_return_6_1,
                volatility_12_1,
                volatility_6_1,
                risk_adjusted_momentum_12_1,
                risk_adjusted_momentum_6_1,
                short_term_strength_20d,
                ma20,
                ma60,
                ma120,
                ma200,
                moving_average_structure_raw_score,
                max_drawdown_252d,
                above_ma60_ratio_126d,
                positive_21d_return_ratio_126d,
                uptrend_consistency_raw_score,
                short_term_strength_20d,
                distance_from_ma20,
                prior_60d_high_adjusted_close,
                data_quality_extreme_return_flag,
            ]
            if not _all_finite(computed_factors):
                continue

            rows.append(
                {
                    "run_id": run_id,
                    "market": market.value,
                    "ticker": ticker,
                    "close": latest_close,
                    "adjusted_close": adjusted_close_t,
                    "profile_metrics_version": 2.0,
                    "avg_traded_value_20d_local": avg_traded_value_20d_local,
                    "avg_traded_value_5d_local": avg_traded_value_5d_local,
                    "momentum_return_12_1": momentum_return_12_1,
                    "momentum_return_6_1": momentum_return_6_1,
                    "volatility_12_1": volatility_12_1,
                    "volatility_6_1": volatility_6_1,
                    "risk_adjusted_momentum_12_1": risk_adjusted_momentum_12_1,
                    "risk_adjusted_momentum_6_1": risk_adjusted_momentum_6_1,
                    "short_term_strength_20d": short_term_strength_20d,
                    "ma20": ma20,
                    "ma60": ma60,
                    "ma120": ma120,
                    "ma200": ma200,
                    "moving_average_structure_raw_score": moving_average_structure_raw_score,
                    "max_drawdown_252d": max_drawdown_252d,
                    "above_ma60_ratio_126d": above_ma60_ratio_126d,
                    "positive_21d_return_ratio_126d": positive_21d_return_ratio_126d,
                    "uptrend_consistency_raw_score": uptrend_consistency_raw_score,
                    "short_term_extension_20d": short_term_strength_20d,
                    "distance_from_ma20": distance_from_ma20,
                    "prior_60d_high_adjusted_close": prior_60d_high_adjusted_close,
                    "data_quality_extreme_return_flag": data_quality_extreme_return_flag,
                }
            )

        if not rows:
            return _empty_snapshot()
        return pl.DataFrame(rows, schema=MOMENTUM_QUALITY_SNAPSHOT_SCHEMA).sort("ticker")

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

    def _has_required_ranking_inputs(self, snapshot: pl.DataFrame) -> bool:
        required_columns = {
            "run_id",
            "market",
            "ticker",
            "adjusted_close",
            *MOMENTUM_QUALITY_RAW_FACTOR_KEYS,
        }
        return required_columns.issubset(set(snapshot.columns))

    def _drop_non_finite_required_factors(self, frame: pl.DataFrame) -> pl.DataFrame:
        result = frame
        for column in ("adjusted_close", *MOMENTUM_QUALITY_RAW_FACTOR_KEYS):
            result = result.filter(pl.col(column).is_not_null() & pl.col(column).is_finite())
        return result.filter(
            (pl.col("volatility_12_1") > self.volatility_floor)
            & (pl.col("volatility_6_1") > self.volatility_floor)
            & (pl.col("avg_traded_value_20d_local") > 0.0)
        )

    def _with_raw_scores(self, frame: pl.DataFrame) -> pl.DataFrame:
        rows = frame.to_dicts()
        for row in rows:
            volatility_6_1 = float(row["volatility_6_1"])
            row["score_risk_adjusted_momentum_12_1"] = float(row["risk_adjusted_momentum_12_1"])
            row["score_risk_adjusted_momentum_6_1"] = float(row["risk_adjusted_momentum_6_1"])
            row["score_short_term_strength_20d"] = (
                float(row["short_term_strength_20d"]) / volatility_6_1
            )
            row["score_short_term_extension_20d"] = float(row["short_term_extension_20d"])
            row["score_distance_from_ma20"] = float(row["distance_from_ma20"])
            row["score_volatility_12_1"] = float(row["volatility_12_1"])
            row["score_volatility_6_1"] = volatility_6_1
            row["moving_average_structure_score"] = (
                float(row["moving_average_structure_raw_score"]) / 100.0
            )
            row["drawdown_control_score"] = 1.0 + float(row["max_drawdown_252d"])
            row["uptrend_consistency_score"] = float(row["uptrend_consistency_raw_score"]) / 100.0
            row["trend_quality_score"] = (
                0.40 * float(row["moving_average_structure_score"])
                + 0.30 * float(row["drawdown_control_score"])
                + 0.30 * float(row["uptrend_consistency_score"])
            )
            row["momentum_blend_score"] = (
                0.55 * float(row["score_risk_adjusted_momentum_12_1"])
                + 0.35 * float(row["score_risk_adjusted_momentum_6_1"])
                + 0.10 * float(row["score_short_term_strength_20d"])
            )
            row["overheat_score"] = (
                0.60 * float(row["score_short_term_extension_20d"])
                + 0.40 * float(row["score_distance_from_ma20"])
            )
            row["overheat_penalty_score"] = (
                max(0.0, (float(row["score_short_term_extension_20d"]) - 0.25) * 20.0)
                + max(0.0, (float(row["score_distance_from_ma20"]) - 0.20) * 20.0)
            )

        finite_rows = [
            row
            for row in rows
            if _all_finite(
                [
                    row[key]
                    for key in self.ranking_metric_keys
                    if not key.startswith("tag_")
                ]
            )
        ]
        return pl.DataFrame(finite_rows) if finite_rows else pl.DataFrame()

    def _with_tags(self, frame: pl.DataFrame) -> pl.DataFrame:
        rows = frame.to_dicts()
        for row in rows:
            row["tag_risk_overheated"] = (
                1.0
                if float(row["score_short_term_extension_20d"]) >= 0.30
                or float(row["score_distance_from_ma20"]) >= 0.25
                else 0.0
            )
            row["tag_risk_extended_from_ma20"] = (
                1.0 if float(row["score_distance_from_ma20"]) >= 0.15 else 0.0
            )
            row["tag_risk_high_volatility"] = (
                1.0
                if float(row["score_volatility_12_1"]) >= 0.06
                or float(row["score_volatility_6_1"]) >= 0.06
                else 0.0
            )
            row["tag_risk_large_drawdown"] = (
                1.0 if float(row["max_drawdown_252d"]) <= -0.30 else 0.0
            )
            row["tag_risk_weak_trend_quality"] = (
                1.0
                if float(row["momentum_blend_score"]) >= 7.0
                and float(row["trend_quality_score"]) <= 0.45
                else 0.0
            )
            row["tag_risk_thin_recent_volume"] = (
                1.0
                if float(row["avg_traded_value_5d_local"])
                / float(row["avg_traded_value_20d_local"])
                < 0.50
                else 0.0
            )
            row["tag_risk_data_quality_warning"] = (
                1.0 if float(row["data_quality_extreme_return_flag"]) == 1.0 else 0.0
            )
            row["tag_positive_strong_momentum"] = (
                1.0 if float(row["momentum_blend_score"]) >= 7.0 else 0.0
            )
            row["tag_positive_stable_uptrend"] = (
                1.0
                if float(row["trend_quality_score"]) >= 0.80
                and float(row["tag_risk_large_drawdown"]) == 0.0
                else 0.0
            )
            row["tag_positive_early_breakout"] = (
                1.0
                if float(row["adjusted_close"]) >= float(row["prior_60d_high_adjusted_close"])
                and 5.0 <= float(row["momentum_blend_score"]) <= 20.0
                and float(row["overheat_score"]) < 0.20
                and float(row["tag_risk_overheated"]) == 0.0
                else 0.0
            )
        return pl.DataFrame(rows) if rows else frame

    def _horizon_score(self, row: dict[str, object], horizon: str) -> float:
        if horizon == "composite":
            return (
                float(row["momentum_blend_score"])
                + 0.50 * float(row["trend_quality_score"])
                - 0.40 * float(row["overheat_penalty_score"])
            )
        if horizon == "swing":
            return (
                0.45 * float(row["score_risk_adjusted_momentum_6_1"])
                + 0.25 * float(row["score_risk_adjusted_momentum_12_1"])
                + 0.20 * float(row["score_short_term_strength_20d"])
                + 0.10 * float(row["trend_quality_score"])
                - 0.60 * float(row["overheat_penalty_score"])
            )
        if horizon == "midterm":
            return (
                0.45 * float(row["score_risk_adjusted_momentum_12_1"])
                + 0.35 * float(row["score_risk_adjusted_momentum_6_1"])
                + 0.20 * float(row["trend_quality_score"])
            )
        raise ValidationError(f"unknown horizon {horizon}")

    def _assign_single_run_market_rankings(self, frame: pl.DataFrame) -> pl.DataFrame:
        scored = self._with_tags(self._with_raw_scores(frame))
        scored_rows = [
            row
            for row in scored.to_dicts()
            if _all_finite([row[key] for key in self.ranking_metric_keys])
        ]
        ranking_rows: list[dict[str, object]] = []
        for horizon in self.horizon_order:
            horizon_rows = []
            for row in scored_rows:
                score = self._horizon_score(row, horizon)
                if not math.isfinite(score):
                    continue
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
            horizon_rows.sort(key=lambda row: (-float(row["score"]), str(row["ticker"])))
            for rank, row in enumerate(horizon_rows, start=1):
                row["rank"] = rank
                ranking_rows.append(row)
        return pl.DataFrame(ranking_rows, schema=MOMENTUM_QUALITY_RANKING_SCHEMA).select(
            self._ranking_columns()
        )

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        if snapshot.is_empty() or not self._has_required_ranking_inputs(snapshot):
            return _empty_rankings()
        eligible = self._drop_non_finite_required_factors(snapshot)
        if eligible.height == 0:
            return _empty_rankings()

        ranking_frames = []
        for partition in eligible.partition_by(["run_id", "market"], maintain_order=True):
            ranking_frames.append(self._assign_single_run_market_rankings(partition))
        return (
            pl.concat(ranking_frames)
            .sort(["run_id", "market", "horizon", "rank"])
            .select(self._ranking_columns())
        )


MOMENTUM_QUALITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=MOMENTUM_QUALITY_PROFILE_ID,
    factory=MomentumQualityV1Profile,
)
