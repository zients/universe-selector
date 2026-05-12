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


MOMENTUM_PROFILE_ID = "momentum_v1"
MOMENTUM_SCORE_METHOD = "raw_weighted_momentum_v1"
MOMENTUM_RANK_INTERPRETATION_NOTE = (
    "Momentum scores are raw weighted momentum composites; high scores do not guarantee positive absolute momentum or an uptrend."
)

MOMENTUM_SCORE_INPUT_COLUMNS = (
    "risk_adjusted_momentum_12_1",
    "risk_adjusted_momentum_6_1",
    "short_term_strength_20d",
    "volatility_6_1",
)

MOMENTUM_RANKING_METRIC_KEYS = (
    "score_risk_adjusted_momentum_12_1",
    "score_risk_adjusted_momentum_6_1",
    "score_short_term_strength_20d",
)

MOMENTUM_INSPECT_METRIC_KEYS = (
    "momentum_return_12_1",
    "momentum_return_6_1",
    "volatility_12_1",
    "volatility_6_1",
    "risk_adjusted_momentum_12_1",
    "risk_adjusted_momentum_6_1",
    "short_term_strength_20d",
)

MOMENTUM_HORIZON_WEIGHTS = {
    "midterm": {
        "score_risk_adjusted_momentum_12_1": 0.50,
        "score_risk_adjusted_momentum_6_1": 0.50,
    },
    "swing": {
        "score_risk_adjusted_momentum_6_1": 0.50,
        "score_risk_adjusted_momentum_12_1": 0.30,
        "score_short_term_strength_20d": 0.20,
    },
}

MOMENTUM_SNAPSHOT_METRIC_KEYS = (
    "close",
    "adjusted_close",
    "avg_traded_value_20d_local",
    *MOMENTUM_INSPECT_METRIC_KEYS,
)
MOMENTUM_PERSISTED_SNAPSHOT_METRIC_KEYS = (
    "avg_traded_value_20d_local",
    *MOMENTUM_INSPECT_METRIC_KEYS,
)
MOMENTUM_HORIZON_ORDER = ("swing", "midterm")

SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    "avg_traded_value_20d_local": pl.Float64,
    "momentum_return_12_1": pl.Float64,
    "momentum_return_6_1": pl.Float64,
    "volatility_12_1": pl.Float64,
    "volatility_6_1": pl.Float64,
    "risk_adjusted_momentum_12_1": pl.Float64,
    "risk_adjusted_momentum_6_1": pl.Float64,
    "short_term_strength_20d": pl.Float64,
}


def _all_finite(values: list[object]) -> bool:
    for value in values:
        if value is None:
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(numeric):
            return False
    return True


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=SNAPSHOT_SCHEMA)


def _immutable_market_float_mapping(value: Mapping[Market, float]) -> Mapping[Market, float]:
    return MappingProxyType({market: float(value[market]) for market in Market})


def _immutable_nested_float_mapping(value: Mapping[str, Mapping[str, float]]) -> Mapping[str, Mapping[str, float]]:
    return MappingProxyType(
        {
            str(name): MappingProxyType({str(column): float(weight) for column, weight in weights.items()})
            for name, weights in value.items()
        }
    )


@dataclass(frozen=True)
class MomentumV1Profile:
    profile_id: Literal["momentum_v1"] = MOMENTUM_PROFILE_ID
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
    horizon_weights: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {
            horizon: dict(weights) for horizon, weights in MOMENTUM_HORIZON_WEIGHTS.items()
        }
    )
    score_input_columns: tuple[str, ...] = MOMENTUM_SCORE_INPUT_COLUMNS
    snapshot_metric_keys: tuple[str, ...] = MOMENTUM_PERSISTED_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = MOMENTUM_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = MOMENTUM_INSPECT_METRIC_KEYS
    horizon_order: tuple[str, ...] = MOMENTUM_HORIZON_ORDER
    stdev_ddof: int = 1
    score_method: str = MOMENTUM_SCORE_METHOD
    rank_interpretation_note: str = MOMENTUM_RANK_INTERPRETATION_NOTE

    def __post_init__(self) -> None:
        object.__setattr__(self, "price_floor", _immutable_market_float_mapping(self.price_floor))
        object.__setattr__(self, "liquidity_floor", _immutable_market_float_mapping(self.liquidity_floor))
        object.__setattr__(self, "horizon_weights", _immutable_nested_float_mapping(self.horizon_weights))
        object.__setattr__(
            self,
            "score_input_columns",
            tuple(str(column) for column in self.score_input_columns),
        )
        object.__setattr__(
            self,
            "inspect_metric_keys",
            tuple(str(key) for key in self.inspect_metric_keys),
        )
        object.__setattr__(
            self,
            "snapshot_metric_keys",
            tuple(str(key) for key in self.snapshot_metric_keys),
        )
        object.__setattr__(
            self,
            "ranking_metric_keys",
            tuple(str(key) for key in self.ranking_metric_keys),
        )
        object.__setattr__(
            self,
            "horizon_order",
            tuple(str(horizon) for horizon in self.horizon_order),
        )

    def validate(self) -> None:
        if self.profile_id != MOMENTUM_PROFILE_ID:
            raise ValidationError("profile_id must be momentum_v1")
        if self.min_history_bars != 274:
            raise ValidationError("momentum_v1 requires min_history_bars to be 274")
        if set(self.horizon_weights.keys()) != {"swing", "midterm"}:
            raise ValidationError("momentum_v1 horizons must be swing and midterm")
        for horizon, weights in self.horizon_weights.items():
            total = round(sum(weights.values()), 10)
            if total != 1.0:
                raise ValidationError(f"{horizon} horizon weights must sum to 1.0")
        if self.score_input_columns != MOMENTUM_SCORE_INPUT_COLUMNS:
            raise ValidationError("score input columns must match momentum_v1")
        ranking_metric_values = set(self.ranking_metric_keys)
        for horizon, weights in self.horizon_weights.items():
            missing_columns = set(weights.keys()) - ranking_metric_values
            if missing_columns:
                raise ValidationError(
                    f"{horizon} horizon weights reference unknown ranking metric columns: "
                    f"{', '.join(sorted(missing_columns))}"
                )
        unknown_inspect_keys = set(self.inspect_metric_keys) - set(MOMENTUM_SNAPSHOT_METRIC_KEYS)
        if unknown_inspect_keys:
            raise ValidationError(
                f"inspect metric keys are not in the momentum snapshot shape: "
                f"{', '.join(sorted(unknown_inspect_keys))}"
            )
        if self.snapshot_metric_keys != MOMENTUM_PERSISTED_SNAPSHOT_METRIC_KEYS:
            raise ValidationError(
                "snapshot metric keys must match momentum_v1 persisted metrics"
            )
        if not set(self.inspect_metric_keys).issubset(set(self.snapshot_metric_keys)):
            missing_inspect_keys = set(self.inspect_metric_keys) - set(self.snapshot_metric_keys)
            raise ValidationError(
                f"inspect metric keys must be persisted snapshot metrics: "
                f"{', '.join(sorted(missing_inspect_keys))}"
            )
        if tuple(self.ranking_metric_keys) != MOMENTUM_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match momentum_v1")
        if len(set(self.horizon_order)) != len(self.horizon_order) or set(self.horizon_order) != set(
            self.horizon_weights.keys()
        ):
            raise ValidationError("horizon order must match configured horizons")
        if self.stdev_ddof != 1:
            raise ValidationError("stdev_ddof must be 1")
        if self.score_method != MOMENTUM_SCORE_METHOD:
            raise ValidationError(f"score_method must be {MOMENTUM_SCORE_METHOD}")
        if self.rank_interpretation_note != MOMENTUM_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match momentum_v1 output text")

    def ranking_config_payload(self) -> dict[str, object]:
        return {
            "active_trading_min_days_274": self.active_trading_min_days_274,
            "horizon_weights": {
                horizon: dict(weights) for horizon, weights in self.horizon_weights.items()
            },
            "liquidity_floor": {market.value: self.liquidity_floor[market] for market in Market},
            "min_history_bars": self.min_history_bars,
            "price_floor": {market.value: self.price_floor[market] for market in Market},
            "ranking_profile": self.profile_id,
            "ranking_metric_keys": list(self.ranking_metric_keys),
            "score_input_columns": list(self.score_input_columns),
            "score_method": self.score_method,
            "stdev_ddof": self.stdev_ddof,
            "volatility_floor": self.volatility_floor,
            "zero_volume_max_days_20": self.zero_volume_max_days_20,
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

            adjusted_close_t = adjusted_closes_float[-1]
            adjusted_close_t_20 = adjusted_closes_float[-21]
            adjusted_close_t_21 = adjusted_closes_float[-22]
            adjusted_close_t_147 = adjusted_closes_float[-148]
            adjusted_close_t_273 = adjusted_closes_float[-274]
            returns = [
                adjusted_closes_float[index] / adjusted_closes_float[index - 1] - 1.0
                for index in range(1, len(adjusted_closes_float))
            ]
            returns_12_1 = returns[0:252]
            returns_6_1 = returns[126:252]

            volatility_12_1 = pl.Series(returns_12_1).std(ddof=self.stdev_ddof)
            volatility_6_1 = pl.Series(returns_6_1).std(ddof=self.stdev_ddof)
            if volatility_12_1 is None or volatility_6_1 is None:
                continue
            if volatility_12_1 <= self.volatility_floor or volatility_6_1 <= self.volatility_floor:
                continue

            avg_traded_value_20d_local = sum(
                close * volume
                for close, volume in zip(closes_float[-20:], volumes_float[-20:], strict=True)
            ) / 20.0
            latest_close = closes_float[-1]
            if avg_traded_value_20d_local < self.liquidity_floor[market]:
                continue
            if latest_close < self.price_floor[market]:
                continue

            momentum_return_12_1 = adjusted_close_t_21 / adjusted_close_t_273 - 1.0
            momentum_return_6_1 = adjusted_close_t_21 / adjusted_close_t_147 - 1.0
            short_term_strength_20d = adjusted_close_t / adjusted_close_t_20 - 1.0
            risk_adjusted_momentum_12_1 = momentum_return_12_1 / volatility_12_1
            risk_adjusted_momentum_6_1 = momentum_return_6_1 / volatility_6_1

            computed_factors = [
                avg_traded_value_20d_local,
                momentum_return_12_1,
                momentum_return_6_1,
                volatility_12_1,
                volatility_6_1,
                risk_adjusted_momentum_12_1,
                risk_adjusted_momentum_6_1,
                short_term_strength_20d,
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
                    "avg_traded_value_20d_local": avg_traded_value_20d_local,
                    "momentum_return_12_1": momentum_return_12_1,
                    "momentum_return_6_1": momentum_return_6_1,
                    "volatility_12_1": volatility_12_1,
                    "volatility_6_1": volatility_6_1,
                    "risk_adjusted_momentum_12_1": risk_adjusted_momentum_12_1,
                    "risk_adjusted_momentum_6_1": risk_adjusted_momentum_6_1,
                    "short_term_strength_20d": short_term_strength_20d,
                }
            )

        if not rows:
            return _empty_snapshot()
        return pl.DataFrame(rows, schema=SNAPSHOT_SCHEMA).sort("ticker")

    def _drop_non_finite_required_score_inputs(self, frame: pl.DataFrame) -> pl.DataFrame:
        result = frame
        for column in self.score_input_columns:
            result = result.filter(pl.col(column).is_not_null() & pl.col(column).is_finite())
        return result.filter(pl.col("volatility_6_1") > self.volatility_floor)

    def _with_score_components(self, frame: pl.DataFrame) -> pl.DataFrame:
        with_scores = frame.with_columns(
            pl.col("risk_adjusted_momentum_12_1").alias("score_risk_adjusted_momentum_12_1"),
            pl.col("risk_adjusted_momentum_6_1").alias("score_risk_adjusted_momentum_6_1"),
            (pl.col("short_term_strength_20d") / pl.col("volatility_6_1")).alias(
                "score_short_term_strength_20d"
            ),
        )
        result = with_scores
        for column in self.ranking_metric_keys:
            result = result.filter(pl.col(column).is_not_null() & pl.col(column).is_finite())
        return result

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
        with_scores = self._with_score_components(frame)
        horizon_frames: list[pl.DataFrame] = []
        for horizon, weights in self.horizon_weights.items():
            weighted = None
            for column, weight in weights.items():
                term = pl.col(column) * weight
                weighted = term if weighted is None else weighted + term
            horizon_frame = (
                with_scores.with_columns(
                    pl.lit(horizon).alias("horizon"),
                    weighted.alias("score"),
                )
                .filter(pl.col("score").is_not_null() & pl.col("score").is_finite())
                .sort(["score", "ticker"], descending=[True, False])
                .with_row_index("rank", offset=1)
                .with_columns(pl.col("rank").cast(pl.Int64))
                .select(self._ranking_columns())
            )
            horizon_frames.append(horizon_frame)
        return pl.concat(horizon_frames)

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        empty_schema = {
            "run_id": pl.String,
            "market": pl.String,
            "horizon": pl.String,
            "ticker": pl.String,
            **{column: pl.Float64 for column in self.ranking_metric_keys},
            "score": pl.Float64,
            "rank": pl.Int64,
        }
        if snapshot.is_empty() or any(column not in snapshot.columns for column in self.score_input_columns):
            return pl.DataFrame(schema=empty_schema)
        eligible = self._drop_non_finite_required_score_inputs(snapshot)
        if eligible.height == 0:
            return pl.DataFrame(schema=empty_schema)

        ranking_frames = []
        for partition in eligible.partition_by(["run_id", "market"], maintain_order=True):
            ranking_frames.append(self._assign_single_run_market_rankings(partition))
        return (
            pl.concat(ranking_frames)
            .sort(["run_id", "market", "horizon", "rank"])
            .select(self._ranking_columns())
        )


MOMENTUM_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=MOMENTUM_PROFILE_ID,
    factory=MomentumV1Profile,
)
