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
from universe_selector.ranking_profiles.rank_math import percentile_rank
from universe_selector.ranking_profiles.registration import RankingProfileRegistration


SAMPLE_PRICE_TREND_PROFILE_ID = "sample_price_trend_v1"
SAMPLE_PRICE_TREND_PERCENTILE_METHOD = "average_rank_100_times_rank_minus_half_over_n"
SAMPLE_PRICE_TREND_RANK_INTERPRETATION_NOTE = (
    "Sample profile scores are run-local normalized return factor scores."
)

SAMPLE_PRICE_TREND_SNAPSHOT_METRIC_KEYS = (
    "avg_traded_value_20d_local",
    "return_60d",
    "return_120d",
)
SAMPLE_PRICE_TREND_RANKING_METRIC_KEYS = (
    "score_return_60d",
    "score_return_120d",
)
SAMPLE_PRICE_TREND_INSPECT_METRIC_KEYS = SAMPLE_PRICE_TREND_SNAPSHOT_METRIC_KEYS
SAMPLE_PRICE_TREND_HORIZON_ORDER = ("midterm", "longterm")
SAMPLE_PRICE_TREND_RETURN_WINDOWS = {"midterm": 60, "longterm": 120}

SAMPLE_PRICE_TREND_SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    "avg_traded_value_20d_local": pl.Float64,
    "return_60d": pl.Float64,
    "return_120d": pl.Float64,
}

SAMPLE_PRICE_TREND_RANKING_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "horizon": pl.String,
    "ticker": pl.String,
    "score_return_60d": pl.Float64,
    "score_return_120d": pl.Float64,
    "score": pl.Float64,
    "rank": pl.Int64,
}


def _immutable_market_float_mapping(value: Mapping[object, float]) -> Mapping[object, float]:
    return MappingProxyType(dict(value))


def _immutable_return_windows(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


def _is_real_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _finite_float(value: object) -> float | None:
    if not _is_real_number(value):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=SAMPLE_PRICE_TREND_SNAPSHOT_SCHEMA)


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema=SAMPLE_PRICE_TREND_RANKING_SCHEMA)


def _finite_real_values(values: list[object]) -> list[float] | None:
    result = []
    for value in values:
        numeric = _finite_float(value)
        if numeric is None:
            return None
        result.append(numeric)
    return result


def _positive_real_values(values: list[object]) -> list[float] | None:
    result = _finite_real_values(values)
    if result is None or any(value <= 0 for value in result):
        return None
    return result


def _non_negative_real_values(values: list[object]) -> list[float] | None:
    result = _finite_real_values(values)
    if result is None or any(value < 0 for value in result):
        return None
    return result


@dataclass(frozen=True)
class SamplePriceTrendV1Profile:
    profile_id: Literal["sample_price_trend_v1"] = SAMPLE_PRICE_TREND_PROFILE_ID
    min_history_bars: int = 121
    return_windows: Mapping[str, object] = field(
        default_factory=lambda: dict(SAMPLE_PRICE_TREND_RETURN_WINDOWS)
    )
    price_floor: Mapping[Market, float] = field(
        default_factory=lambda: {Market.TW: 10.0, Market.US: 5.0}
    )
    liquidity_floor: Mapping[Market, float] = field(
        default_factory=lambda: {Market.TW: 20_000_000.0, Market.US: 5_000_000.0}
    )
    snapshot_metric_keys: tuple[str, ...] = SAMPLE_PRICE_TREND_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = SAMPLE_PRICE_TREND_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = SAMPLE_PRICE_TREND_INSPECT_METRIC_KEYS
    horizon_order: tuple[str, ...] = SAMPLE_PRICE_TREND_HORIZON_ORDER
    percentile_method: str = SAMPLE_PRICE_TREND_PERCENTILE_METHOD
    rank_interpretation_note: str = SAMPLE_PRICE_TREND_RANK_INTERPRETATION_NOTE

    def __post_init__(self) -> None:
        object.__setattr__(self, "return_windows", _immutable_return_windows(self.return_windows))
        object.__setattr__(self, "price_floor", _immutable_market_float_mapping(self.price_floor))
        object.__setattr__(self, "liquidity_floor", _immutable_market_float_mapping(self.liquidity_floor))
        object.__setattr__(self, "snapshot_metric_keys", tuple(str(key) for key in self.snapshot_metric_keys))
        object.__setattr__(self, "ranking_metric_keys", tuple(str(key) for key in self.ranking_metric_keys))
        object.__setattr__(self, "inspect_metric_keys", tuple(str(key) for key in self.inspect_metric_keys))
        object.__setattr__(self, "horizon_order", tuple(str(horizon) for horizon in self.horizon_order))

    def validate(self) -> None:
        if self.profile_id != SAMPLE_PRICE_TREND_PROFILE_ID:
            raise ValidationError("profile_id must be sample_price_trend_v1")
        if self.min_history_bars != 121:
            raise ValidationError("min_history_bars must be 121")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in self.return_windows.values()):
            raise ValidationError("return_windows values must be exact integers")
        if dict(self.return_windows) != SAMPLE_PRICE_TREND_RETURN_WINDOWS:
            raise ValidationError("return_windows must be midterm=60 and longterm=120")
        if self.horizon_order != SAMPLE_PRICE_TREND_HORIZON_ORDER:
            raise ValidationError("horizon_order must be midterm then longterm")
        if self.snapshot_metric_keys != SAMPLE_PRICE_TREND_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("snapshot metric keys must match sample_price_trend_v1")
        if self.ranking_metric_keys != SAMPLE_PRICE_TREND_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match sample_price_trend_v1")
        if self.inspect_metric_keys != SAMPLE_PRICE_TREND_INSPECT_METRIC_KEYS:
            raise ValidationError("inspect metric keys must match sample_price_trend_v1")
        if not set(self.inspect_metric_keys).issubset(set(self.snapshot_metric_keys)):
            raise ValidationError("inspect metric keys must be persisted snapshot metrics")
        if set(self.price_floor) != set(Market):
            raise ValidationError("price_floor must contain exactly supported markets")
        if set(self.liquidity_floor) != set(Market):
            raise ValidationError("liquidity_floor must contain exactly supported markets")
        for value in self.price_floor.values():
            numeric = _finite_float(value)
            if numeric is None or numeric <= 0:
                raise ValidationError("price_floor values must be finite positive numbers")
        for value in self.liquidity_floor.values():
            numeric = _finite_float(value)
            if numeric is None or numeric <= 0:
                raise ValidationError("liquidity_floor values must be finite positive numbers")
        if self.percentile_method != SAMPLE_PRICE_TREND_PERCENTILE_METHOD:
            raise ValidationError(f"percentile_method must be {SAMPLE_PRICE_TREND_PERCENTILE_METHOD}")
        if self.rank_interpretation_note != SAMPLE_PRICE_TREND_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match sample_price_trend_v1")

    def ranking_config_payload(self) -> dict[str, object]:
        return {
            "ranking_profile": self.profile_id,
            "min_history_bars": self.min_history_bars,
            "return_windows": {
                "midterm": self.return_windows["midterm"],
                "longterm": self.return_windows["longterm"],
            },
            "price_floor": {market.value: self.price_floor[market] for market in Market},
            "liquidity_floor": {market.value: self.liquidity_floor[market] for market in Market},
            "horizon_order": list(self.horizon_order),
            "snapshot_metric_keys": list(self.snapshot_metric_keys),
            "ranking_metric_keys": list(self.ranking_metric_keys),
            "inspect_metric_keys": list(self.inspect_metric_keys),
            "percentile_method": self.percentile_method,
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
            last_20_closes = tail["close"].tail(20).to_list()
            last_20_volumes = tail["volume"].tail(20).to_list()
            adjusted_closes = tail["adjusted_close"].to_list()
            denominator_values = [
                tail["adjusted_close"][-1],
                tail["adjusted_close"][-61],
                tail["adjusted_close"][-121],
            ]

            denominator_adjusted_close_values = _positive_real_values(denominator_values)
            last_20_close_values = _positive_real_values(last_20_closes)
            last_20_volume_values = _non_negative_real_values(last_20_volumes)
            if _finite_real_values(adjusted_closes) is None:
                continue
            if denominator_adjusted_close_values is None:
                continue
            if last_20_close_values is None:
                continue
            if last_20_volume_values is None:
                continue

            latest_close = last_20_close_values[-1]
            latest_adjusted_close = denominator_adjusted_close_values[0]
            adjusted_close_60 = denominator_adjusted_close_values[1]
            adjusted_close_120 = denominator_adjusted_close_values[2]
            if latest_close < self.price_floor[market]:
                continue

            avg_traded_value_20d_local = sum(
                close * volume for close, volume in zip(last_20_close_values, last_20_volume_values, strict=True)
            ) / 20.0
            if avg_traded_value_20d_local < self.liquidity_floor[market]:
                continue

            return_60d = latest_adjusted_close / adjusted_close_60 - 1.0
            return_120d = latest_adjusted_close / adjusted_close_120 - 1.0
            if _finite_real_values([avg_traded_value_20d_local, return_60d, return_120d]) is None:
                continue

            rows.append(
                {
                    "run_id": run_id,
                    "market": market.value,
                    "ticker": ticker,
                    "close": latest_close,
                    "adjusted_close": latest_adjusted_close,
                    "avg_traded_value_20d_local": avg_traded_value_20d_local,
                    "return_60d": return_60d,
                    "return_120d": return_120d,
                }
            )

        if not rows:
            return _empty_snapshot()
        return pl.DataFrame(rows, schema=SAMPLE_PRICE_TREND_SNAPSHOT_SCHEMA)

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

    def _with_return_scores(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.Series(
                "score_return_60d",
                percentile_rank(frame["return_60d"]).to_list(),
            ),
            pl.Series(
                "score_return_120d",
                percentile_rank(frame["return_120d"]).to_list(),
            ),
        )

    def _drop_non_finite_required_returns(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        return snapshot.filter(
            pl.col("return_60d").is_not_null()
            & pl.col("return_60d").is_finite()
            & pl.col("return_120d").is_not_null()
            & pl.col("return_120d").is_finite()
        )

    def _assign_single_run_market_rankings(self, frame: pl.DataFrame) -> pl.DataFrame:
        with_scores = self._with_return_scores(frame)
        horizon_to_column = {
            "midterm": "score_return_60d",
            "longterm": "score_return_120d",
        }
        horizon_frames = []
        for horizon in self.horizon_order:
            score_column = horizon_to_column[horizon]
            horizon_frame = (
                with_scores.with_columns(
                    pl.lit(horizon).alias("horizon"),
                    pl.col(score_column).alias("score"),
                )
                .sort(["score", "ticker"], descending=[True, False])
                .with_row_index("rank", offset=1)
                .with_columns(pl.col("rank").cast(pl.Int64))
                .select(self._ranking_columns())
            )
            horizon_frames.append(horizon_frame)
        return pl.concat(horizon_frames)

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        if snapshot.is_empty() or any(column not in snapshot.columns for column in ("return_60d", "return_120d")):
            return _empty_rankings()
        eligible = self._drop_non_finite_required_returns(snapshot)
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


SAMPLE_PRICE_TREND_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=SAMPLE_PRICE_TREND_PROFILE_ID,
    factory=SamplePriceTrendV1Profile,
)
