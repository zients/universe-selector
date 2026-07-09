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
    downside_std,
    immutable_market_float_mapping,
    immutable_market_int_mapping,
    max_drawdown,
    mean,
    ols_slope_r2,
    percentile_scores,
    positive_only_percentile_scores,
    std,
    yyyymmdd,
)
from universe_selector.ranking_profiles.registration import RankingProfileRegistration


DEFENSIVE_COMPOUNDER_QUALITY_PROFILE_ID = "defensive_compounder_quality_v1"
DEFENSIVE_COMPOUNDER_QUALITY_SCORE_METHOD = "market_relative_defensive_compounder_quality_v1"
DEFENSIVE_COMPOUNDER_QUALITY_RANK_INTERPRETATION_NOTE = (
    "Defensive compounder quality scores are OHLCV-only market-local relative rankings for steady "
    "positive price behavior, downside control, drawdown control, and long-trend preservation. "
    "They are not fundamental quality measures, buy signals, forecasts, backtests, or holding-period recommendations."
)
DEFENSIVE_COMPOUNDER_QUALITY_HORIZON_ORDER = ("composite", "steady_compounder", "downside_control")

DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS = (
    "profile_metrics_version",
    "asof_bar_date_yyyymmdd",
    "avg_traded_value_20d_local",
    "avg_traded_value_5d_local",
    "median_traded_value_20d_local",
    "traded_value_5d_to_20d_ratio",
    "return_60d",
    "return_120d",
    "return_252d",
    "positive_21d_return_ratio_252d",
    "trend_slope_120d",
    "uptrend_r2_120d",
    "volatility_20d",
    "volatility_60d",
    "volatility_120d",
    "volatility_20d_to_60d_ratio",
    "downside_volatility_60d",
    "downside_volatility_120d",
    "max_drawdown_120d",
    "max_drawdown_252d",
    "range_tightness_20d",
    "range_tightness_60d",
    "price_vs_sma_50d",
    "price_vs_sma_200d",
    "sma_50d_vs_sma_200d",
    "liquidity_stability_score_raw",
    "zero_volume_days_20d",
    "active_trading_days_60d",
    "stale_close_days_20d",
    "data_quality_extreme_return_flag",
)

DEFENSIVE_COMPOUNDER_QUALITY_RANKING_METRIC_KEYS = (
    "score_steady_return",
    "score_positive_return_persistence",
    "score_trend_durability",
    "score_low_volatility",
    "score_downside_control",
    "score_drawdown_control",
    "score_range_tightness",
    "score_liquidity_stability",
    "steady_compounder_score",
    "downside_control_score",
    "trend_quality_score",
    "risk_control_score",
    "trend_structure_cap_score",
    "growth_cap_score",
    "volatility_cap_score",
    "drawdown_cap_score",
    "liquidity_cap_score",
    "penalty_score",
    "tag_positive_steady_compounder",
    "tag_positive_low_downside_volatility",
    "tag_positive_drawdown_control",
    "tag_risk_flat_no_growth",
    "tag_risk_broken_long_trend",
    "tag_risk_large_drawdown",
    "tag_risk_volatility_spike",
    "tag_risk_stale_or_illiquid",
    "tag_risk_data_quality_warning",
)

DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    **{key: pl.Float64 for key in DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS},
}

DEFENSIVE_COMPOUNDER_QUALITY_RANKING_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "horizon": pl.String,
    "ticker": pl.String,
    **{key: pl.Float64 for key in DEFENSIVE_COMPOUNDER_QUALITY_RANKING_METRIC_KEYS},
    "score": pl.Float64,
    "rank": pl.Int64,
}


def _empty_snapshot() -> pl.DataFrame:
    return pl.DataFrame(schema=DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_SCHEMA)


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema=DEFENSIVE_COMPOUNDER_QUALITY_RANKING_SCHEMA)


@dataclass(frozen=True)
class DefensiveCompounderQualityV1Profile:
    profile_id: Literal["defensive_compounder_quality_v1"] = DEFENSIVE_COMPOUNDER_QUALITY_PROFILE_ID
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
    snapshot_metric_keys: tuple[str, ...] = DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = DEFENSIVE_COMPOUNDER_QUALITY_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS
    horizon_order: tuple[str, ...] = DEFENSIVE_COMPOUNDER_QUALITY_HORIZON_ORDER
    stdev_ddof: int = 1
    score_method: str = DEFENSIVE_COMPOUNDER_QUALITY_SCORE_METHOD
    rank_interpretation_note: str = DEFENSIVE_COMPOUNDER_QUALITY_RANK_INTERPRETATION_NOTE

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
        if self.profile_id != DEFENSIVE_COMPOUNDER_QUALITY_PROFILE_ID:
            raise ValidationError("profile_id must be defensive_compounder_quality_v1")
        if self.min_history_bars != 252:
            raise ValidationError("defensive_compounder_quality_v1 requires min_history_bars to be 252")
        if self.snapshot_metric_keys != DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("snapshot metric keys must match defensive_compounder_quality_v1")
        if self.ranking_metric_keys != DEFENSIVE_COMPOUNDER_QUALITY_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match defensive_compounder_quality_v1")
        if self.inspect_metric_keys != DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("inspect metric keys must match defensive_compounder_quality_v1")
        if self.horizon_order != DEFENSIVE_COMPOUNDER_QUALITY_HORIZON_ORDER:
            raise ValidationError("horizon order must be composite, steady_compounder, downside_control")
        for mapping_name in (
            "price_floor",
            "liquidity_floor",
            "active_trading_min_days_60",
            "zero_volume_max_days_20",
        ):
            if set(getattr(self, mapping_name)) != set(Market):
                raise ValidationError(f"{mapping_name} must contain exactly supported markets")
        if {market: self.price_floor[market] for market in Market} != {Market.TW: 10.0, Market.US: 5.0}:
            raise ValidationError("price_floor must match defensive_compounder_quality_v1")
        if {market: self.liquidity_floor[market] for market in Market} != {
            Market.TW: 50_000_000.0,
            Market.US: 10_000_000.0,
        }:
            raise ValidationError("liquidity_floor must match defensive_compounder_quality_v1")
        if {market: self.active_trading_min_days_60[market] for market in Market} != {
            Market.TW: 50,
            Market.US: 55,
        }:
            raise ValidationError("active_trading_min_days_60 must match defensive_compounder_quality_v1")
        if {market: self.zero_volume_max_days_20[market] for market in Market} != {
            Market.TW: 3,
            Market.US: 1,
        }:
            raise ValidationError("zero_volume_max_days_20 must match defensive_compounder_quality_v1")
        if self.stale_close_max_days_20 != 5:
            raise ValidationError("stale_close_max_days_20 must be 5")
        if self.extreme_return_abs_cutoff != 0.80:
            raise ValidationError("extreme_return_abs_cutoff must be 0.80")
        if self.volatility_floor != 0.0001:
            raise ValidationError("volatility_floor must be 0.0001")
        if self.stdev_ddof != 1:
            raise ValidationError("stdev_ddof must be 1")
        if self.score_method != DEFENSIVE_COMPOUNDER_QUALITY_SCORE_METHOD:
            raise ValidationError(f"score_method must be {DEFENSIVE_COMPOUNDER_QUALITY_SCORE_METHOD}")
        if self.rank_interpretation_note != DEFENSIVE_COMPOUNDER_QUALITY_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match defensive_compounder_quality_v1")

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
            returns_120d = returns[-120:]
            volatility_20d = std(returns_20d, ddof=self.stdev_ddof)
            volatility_60d = std(returns_60d, ddof=self.stdev_ddof)
            volatility_120d = std(returns_120d, ddof=self.stdev_ddof)
            downside_volatility_60d = downside_std(returns_60d, ddof=self.stdev_ddof)
            downside_volatility_120d = downside_std(returns_120d, ddof=self.stdev_ddof)
            if (
                volatility_20d is None
                or volatility_60d is None
                or volatility_120d is None
                or downside_volatility_60d is None
                or downside_volatility_120d is None
            ):
                continue
            if (
                volatility_20d <= 0.0
                or volatility_60d <= self.volatility_floor
                or volatility_120d <= self.volatility_floor
            ):
                continue

            return_60d = latest_adjusted_close / adjusted_closes_float[-61] - 1.0
            return_120d = latest_adjusted_close / adjusted_closes_float[-121] - 1.0
            return_252d = latest_adjusted_close / adjusted_closes_float[-252] - 1.0
            if return_120d <= 0.0 or return_252d <= 0.0:
                continue
            positive_21d_returns = [
                adjusted_closes_float[index] / adjusted_closes_float[index - 21] - 1.0
                for index in range(21, len(adjusted_closes_float))
            ]
            positive_21d_return_ratio_252d = float(sum(1 for value in positive_21d_returns if value > 0.0)) / float(
                len(positive_21d_returns)
            )
            trend_slope_120d, trend_r2_120d = ols_slope_r2(adjusted_closes_float[-120:])
            uptrend_r2_120d = trend_r2_120d if trend_slope_120d > 0.0 else 0.0
            max_drawdown_120d = max_drawdown(adjusted_closes_float[-120:])
            max_drawdown_252d = max_drawdown(adjusted_closes_float[-252:])
            sma_50d = mean(adjusted_closes_float[-50:])
            sma_200d = mean(adjusted_closes_float[-200:])
            price_vs_sma_50d = latest_adjusted_close / sma_50d - 1.0
            price_vs_sma_200d = latest_adjusted_close / sma_200d - 1.0
            sma_50d_vs_sma_200d = sma_50d / sma_200d - 1.0
            range_tightness_20d = 1.0 - clamp(
                (max(adjusted_closes_float[-20:]) / min(adjusted_closes_float[-20:]) - 1.0) / 0.18, 0.0, 1.0
            )
            range_tightness_60d = 1.0 - clamp(
                (max(adjusted_closes_float[-60:]) / min(adjusted_closes_float[-60:]) - 1.0) / 0.28, 0.0, 1.0
            )
            volatility_20d_to_60d_ratio = volatility_20d / volatility_60d
            liquidity_stability_score_raw = band_score(
                traded_value_5d_to_20d_ratio,
                ideal_low=0.80,
                ideal_high=1.60,
                outer_low=0.45,
                outer_high=3.00,
            )

            if max_drawdown_252d <= -0.35:
                continue
            if price_vs_sma_200d < -0.05:
                continue
            if sma_50d_vs_sma_200d < -0.05 and return_120d <= 0.0:
                continue
            data_quality_extreme_return_flag = 1.0 if any(abs(value) > 0.50 for value in returns_60d) else 0.0

            computed = [
                avg_traded_value_20d_local,
                avg_traded_value_5d_local,
                median_traded_value_20d_local,
                traded_value_5d_to_20d_ratio,
                return_60d,
                return_120d,
                return_252d,
                positive_21d_return_ratio_252d,
                trend_slope_120d,
                uptrend_r2_120d,
                volatility_20d,
                volatility_60d,
                volatility_120d,
                volatility_20d_to_60d_ratio,
                downside_volatility_60d,
                downside_volatility_120d,
                max_drawdown_120d,
                max_drawdown_252d,
                range_tightness_20d,
                range_tightness_60d,
                price_vs_sma_50d,
                price_vs_sma_200d,
                sma_50d_vs_sma_200d,
                liquidity_stability_score_raw,
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
                    "return_60d": return_60d,
                    "return_120d": return_120d,
                    "return_252d": return_252d,
                    "positive_21d_return_ratio_252d": positive_21d_return_ratio_252d,
                    "trend_slope_120d": trend_slope_120d,
                    "uptrend_r2_120d": uptrend_r2_120d,
                    "volatility_20d": volatility_20d,
                    "volatility_60d": volatility_60d,
                    "volatility_120d": volatility_120d,
                    "volatility_20d_to_60d_ratio": volatility_20d_to_60d_ratio,
                    "downside_volatility_60d": downside_volatility_60d,
                    "downside_volatility_120d": downside_volatility_120d,
                    "max_drawdown_120d": max_drawdown_120d,
                    "max_drawdown_252d": max_drawdown_252d,
                    "range_tightness_20d": range_tightness_20d,
                    "range_tightness_60d": range_tightness_60d,
                    "price_vs_sma_50d": price_vs_sma_50d,
                    "price_vs_sma_200d": price_vs_sma_200d,
                    "sma_50d_vs_sma_200d": sma_50d_vs_sma_200d,
                    "liquidity_stability_score_raw": liquidity_stability_score_raw,
                    "zero_volume_days_20d": zero_volume_days_20d,
                    "active_trading_days_60d": active_trading_days_60d,
                    "stale_close_days_20d": stale_close_days_20d,
                    "data_quality_extreme_return_flag": data_quality_extreme_return_flag,
                }
            )

        if not rows:
            return _empty_snapshot()
        return pl.DataFrame(rows, schema=DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_SCHEMA).sort("ticker")

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        if snapshot.is_empty():
            return _empty_rankings()
        if not self._has_required_ranking_inputs(snapshot):
            raise ValidationError("defensive_compounder_quality_v1 snapshot is missing required ranking inputs")
        for column in ("close", "adjusted_close", *DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS):
            if not snapshot.schema[column].is_numeric():
                raise ValidationError(
                    f"defensive_compounder_quality_v1 snapshot contains non-numeric ranking input: {column}"
                )
            invalid_count = snapshot.filter(pl.col(column).is_null() | (~pl.col(column).is_finite())).height
            if invalid_count > 0:
                raise ValidationError(
                    f"defensive_compounder_quality_v1 snapshot contains non-finite ranking input: {column}"
                )

        ranking_frames = []
        for partition in snapshot.partition_by(["run_id", "market"], maintain_order=True):
            if partition["asof_bar_date_yyyymmdd"].n_unique() != 1:
                raise ValidationError("defensive_compounder_quality_v1 snapshot contains mixed as-of dates")
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
            *DEFENSIVE_COMPOUNDER_QUALITY_SNAPSHOT_METRIC_KEYS,
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
        score_return_60d = positive_only_percentile_scores([float(row["return_60d"]) for row in rows])
        score_return_120d = positive_only_percentile_scores([float(row["return_120d"]) for row in rows])
        score_return_252d = positive_only_percentile_scores([float(row["return_252d"]) for row in rows])
        score_trend_slope = positive_only_percentile_scores([float(row["trend_slope_120d"]) for row in rows])
        score_low_volatility_60d = percentile_scores(
            [float(row["volatility_60d"]) for row in rows],
            higher_is_better=False,
        )
        score_low_volatility_120d = percentile_scores(
            [float(row["volatility_120d"]) for row in rows],
            higher_is_better=False,
        )
        score_downside_volatility_60d = percentile_scores(
            [float(row["downside_volatility_60d"]) for row in rows],
            higher_is_better=False,
        )
        score_downside_volatility_120d = percentile_scores(
            [float(row["downside_volatility_120d"]) for row in rows],
            higher_is_better=False,
        )
        score_drawdown_120d = percentile_scores([float(row["max_drawdown_120d"]) for row in rows])
        score_drawdown_252d = percentile_scores([float(row["max_drawdown_252d"]) for row in rows])

        scored_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows):
            return_60d = float(row["return_60d"])
            return_120d = float(row["return_120d"])
            return_252d = float(row["return_252d"])
            positive_21d_return_ratio_252d = float(row["positive_21d_return_ratio_252d"])
            trend_slope_120d = float(row["trend_slope_120d"])
            uptrend_r2_120d = float(row["uptrend_r2_120d"])
            volatility_20d = float(row["volatility_20d"])
            volatility_20d_to_60d_ratio = float(row["volatility_20d_to_60d_ratio"])
            max_drawdown_120d = float(row["max_drawdown_120d"])
            max_drawdown_252d = float(row["max_drawdown_252d"])
            range_tightness_20d = float(row["range_tightness_20d"])
            range_tightness_60d = float(row["range_tightness_60d"])
            price_vs_sma_200d = float(row["price_vs_sma_200d"])
            sma_50d_vs_sma_200d = float(row["sma_50d_vs_sma_200d"])
            traded_value_5d_to_20d_ratio = float(row["traded_value_5d_to_20d_ratio"])
            liquidity_stability_score_raw = float(row["liquidity_stability_score_raw"])
            stale_close_days_20d = float(row["stale_close_days_20d"])
            zero_volume_days_20d = float(row["zero_volume_days_20d"])
            data_quality_extreme_return_flag = float(row["data_quality_extreme_return_flag"])

            score_steady_return = (
                0.20 * score_return_60d[index] + 0.35 * score_return_120d[index] + 0.45 * score_return_252d[index]
            )
            score_positive_return_persistence = positive_21d_return_ratio_252d
            score_trend_durability = (
                0.30 * score_trend_slope[index]
                + 0.25 * uptrend_r2_120d
                + 0.25 * clamp((price_vs_sma_200d + 0.05) / 0.45, 0.0, 1.0)
                + 0.20 * clamp((sma_50d_vs_sma_200d + 0.05) / 0.35, 0.0, 1.0)
            )
            score_low_volatility = 0.45 * score_low_volatility_60d[index] + 0.55 * score_low_volatility_120d[index]
            score_downside_control = (
                0.45 * score_downside_volatility_60d[index] + 0.55 * score_downside_volatility_120d[index]
            )
            score_drawdown_control = 0.45 * score_drawdown_120d[index] + 0.55 * score_drawdown_252d[index]
            score_range_tightness = 0.45 * range_tightness_20d + 0.55 * range_tightness_60d
            score_liquidity_stability = liquidity_stability_score_raw

            steady_compounder_score = (
                0.35 * score_steady_return
                + 0.25 * score_positive_return_persistence
                + 0.25 * score_trend_durability
                + 0.15 * score_range_tightness
            )
            downside_control_score = (
                0.30 * score_downside_control
                + 0.30 * score_drawdown_control
                + 0.25 * score_low_volatility
                + 0.15 * score_range_tightness
            )
            trend_quality_score = score_trend_durability
            risk_control_score = (
                0.35 * downside_control_score
                + 0.25 * score_drawdown_control
                + 0.20 * score_liquidity_stability
                + 0.20 * score_range_tightness
            )

            tag_positive_steady_compounder = (
                1.0
                if return_120d >= 0.10
                and return_252d >= 0.18
                and score_trend_durability >= 0.45
                and max_drawdown_120d > -0.15
                and max_drawdown_252d > -0.20
                and volatility_20d <= 0.035
                and score_range_tightness >= 0.20
                and stale_close_days_20d == 0.0
                and zero_volume_days_20d == 0.0
                and traded_value_5d_to_20d_ratio >= 0.75
                else 0.0
            )
            tag_positive_low_downside_volatility = 1.0 if score_downside_control >= 0.70 else 0.0
            tag_positive_drawdown_control = 1.0 if max_drawdown_252d > -0.18 and score_drawdown_control >= 0.60 else 0.0
            tag_risk_flat_no_growth = 1.0 if return_120d < 0.03 or return_252d < 0.06 else 0.0
            tag_risk_broken_long_trend = (
                1.0
                if price_vs_sma_200d < -0.05
                or sma_50d_vs_sma_200d < 0.0
                or (trend_slope_120d <= 0.0 and return_120d <= 0.0)
                else 0.0
            )
            tag_risk_large_drawdown = 1.0 if max_drawdown_252d < -0.25 or max_drawdown_120d < -0.20 else 0.0
            tag_risk_volatility_spike = 1.0 if volatility_20d_to_60d_ratio > 1.60 or volatility_20d > 0.04 else 0.0
            tag_risk_stale_or_illiquid = (
                1.0
                if stale_close_days_20d > 0.0 or zero_volume_days_20d > 0.0 or traded_value_5d_to_20d_ratio < 0.75
                else 0.0
            )
            tag_risk_data_quality_warning = 1.0 if data_quality_extreme_return_flag == 1.0 else 0.0

            trend_structure_cap_score = self._trend_structure_cap_score(
                price_vs_sma_200d=price_vs_sma_200d,
                sma_50d_vs_sma_200d=sma_50d_vs_sma_200d,
                trend_slope_120d=trend_slope_120d,
                return_120d=return_120d,
            )
            steady_compounder_cap_score = self._steady_compounder_cap_score(
                tag_positive_steady_compounder=tag_positive_steady_compounder,
                steady_compounder_score=steady_compounder_score,
                score_steady_return=score_steady_return,
                score_positive_return_persistence=score_positive_return_persistence,
                score_trend_durability=score_trend_durability,
                score_low_volatility=score_low_volatility,
                score_downside_control=score_downside_control,
                score_drawdown_control=score_drawdown_control,
                score_range_tightness=score_range_tightness,
            )
            growth_cap_score = self._growth_cap_score(
                return_60d=return_60d,
                return_120d=return_120d,
                return_252d=return_252d,
                positive_21d_return_ratio_252d=positive_21d_return_ratio_252d,
            )
            volatility_cap_score = self._volatility_cap_score(
                volatility_20d=volatility_20d,
                volatility_20d_to_60d_ratio=volatility_20d_to_60d_ratio,
                score_low_volatility=score_low_volatility,
                score_downside_control=score_downside_control,
            )
            drawdown_cap_score = self._drawdown_cap_score(
                max_drawdown_120d=max_drawdown_120d,
                max_drawdown_252d=max_drawdown_252d,
            )
            liquidity_cap_score = self._liquidity_cap_score(
                stale_close_days_20d=stale_close_days_20d,
                zero_volume_days_20d=zero_volume_days_20d,
                traded_value_5d_to_20d_ratio=traded_value_5d_to_20d_ratio,
            )

            penalty_score = 0.0
            penalty_score += 0.12 * tag_risk_flat_no_growth
            penalty_score += 0.18 * tag_risk_broken_long_trend
            penalty_score += 0.14 * tag_risk_large_drawdown
            penalty_score += 0.12 * tag_risk_volatility_spike
            penalty_score += 0.12 * tag_risk_stale_or_illiquid
            penalty_score += 0.20 * tag_risk_data_quality_warning

            row.update(
                {
                    "score_steady_return": score_steady_return,
                    "score_positive_return_persistence": score_positive_return_persistence,
                    "score_trend_durability": score_trend_durability,
                    "score_low_volatility": score_low_volatility,
                    "score_downside_control": score_downside_control,
                    "score_drawdown_control": score_drawdown_control,
                    "score_range_tightness": score_range_tightness,
                    "score_liquidity_stability": score_liquidity_stability,
                    "steady_compounder_score": steady_compounder_score,
                    "downside_control_score": downside_control_score,
                    "trend_quality_score": trend_quality_score,
                    "risk_control_score": risk_control_score,
                    "trend_structure_cap_score": trend_structure_cap_score,
                    "steady_compounder_cap_score": steady_compounder_cap_score,
                    "growth_cap_score": growth_cap_score,
                    "volatility_cap_score": volatility_cap_score,
                    "drawdown_cap_score": drawdown_cap_score,
                    "liquidity_cap_score": liquidity_cap_score,
                    "penalty_score": penalty_score,
                    "tag_positive_steady_compounder": tag_positive_steady_compounder,
                    "tag_positive_low_downside_volatility": tag_positive_low_downside_volatility,
                    "tag_positive_drawdown_control": tag_positive_drawdown_control,
                    "tag_risk_flat_no_growth": tag_risk_flat_no_growth,
                    "tag_risk_broken_long_trend": tag_risk_broken_long_trend,
                    "tag_risk_large_drawdown": tag_risk_large_drawdown,
                    "tag_risk_volatility_spike": tag_risk_volatility_spike,
                    "tag_risk_stale_or_illiquid": tag_risk_stale_or_illiquid,
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
                        0.40 * float(row["steady_compounder_score"])
                        + 0.25 * float(row["downside_control_score"])
                        + 0.20 * float(row["trend_quality_score"])
                        + 0.15 * float(row["risk_control_score"])
                    )
                elif horizon == "steady_compounder":
                    raw_score = (
                        0.45 * float(row["steady_compounder_score"])
                        + 0.25 * float(row["trend_quality_score"])
                        + 0.15 * float(row["downside_control_score"])
                        + 0.15 * float(row["risk_control_score"])
                    )
                elif horizon == "downside_control":
                    raw_score = (
                        0.45 * float(row["downside_control_score"])
                        + 0.25 * float(row["risk_control_score"])
                        + 0.15 * float(row["steady_compounder_score"])
                        + 0.15 * float(row["trend_quality_score"])
                    )
                else:
                    raise ValidationError(f"unknown horizon {horizon}")
                adjusted_score = raw_score - float(row["penalty_score"])
                cap_score = min(
                    float(row["trend_structure_cap_score"]),
                    float(row["growth_cap_score"]),
                    float(row["volatility_cap_score"]),
                    float(row["drawdown_cap_score"]),
                    float(row["liquidity_cap_score"]),
                )
                if horizon in ("composite", "steady_compounder"):
                    cap_score = min(cap_score, float(row["steady_compounder_cap_score"]))
                score = self._capped_score(adjusted_score, cap_score)
                ranking_values = [float(row[key]) for key in self.ranking_metric_keys] + [score]
                if not all_finite(ranking_values):
                    raise ValidationError("defensive_compounder_quality_v1 produced a non-finite ranking metric")
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
        return pl.DataFrame(ranking_rows, schema=DEFENSIVE_COMPOUNDER_QUALITY_RANKING_SCHEMA).select(
            self._ranking_columns()
        )

    def _capped_score(self, raw_score: float, cap_score: float) -> float:
        if raw_score <= cap_score:
            return raw_score
        if cap_score <= 0.0:
            return cap_score
        cap_range = max(1.0 - cap_score, 1e-9)
        capped_position = clamp((raw_score - cap_score) / cap_range, 0.0, 1.0)
        return cap_score * (0.90 + 0.10 * capped_position)

    def _trend_structure_cap_score(
        self,
        *,
        price_vs_sma_200d: float,
        sma_50d_vs_sma_200d: float,
        trend_slope_120d: float,
        return_120d: float,
    ) -> float:
        if price_vs_sma_200d < -0.10 or (sma_50d_vs_sma_200d < -0.05 and return_120d <= 0.0):
            return 0.35
        if price_vs_sma_200d < -0.05 or sma_50d_vs_sma_200d < 0.0 or trend_slope_120d <= 0.0:
            return 0.60
        if price_vs_sma_200d < 0.0:
            return 0.80
        return 1.0

    def _volatility_cap_score(
        self,
        *,
        volatility_20d: float,
        volatility_20d_to_60d_ratio: float,
        score_low_volatility: float,
        score_downside_control: float,
    ) -> float:
        cap = 1.0
        if volatility_20d > 0.07 or volatility_20d_to_60d_ratio > 2.20:
            cap = min(cap, 0.35)
        elif volatility_20d > 0.04 or volatility_20d_to_60d_ratio > 1.60:
            cap = min(cap, 0.40)
        elif volatility_20d > 0.03 or volatility_20d_to_60d_ratio > 1.30:
            cap = min(cap, 0.70)
        if score_low_volatility < 0.35 and score_downside_control < 0.45:
            cap = min(cap, 0.55)
        elif score_low_volatility < 0.40:
            cap = min(cap, 0.55)
        elif score_low_volatility < 0.50 or score_downside_control < 0.45:
            cap = min(cap, 0.70)
        elif score_low_volatility < 0.60 or score_downside_control < 0.55:
            cap = min(cap, 0.82)
        return cap

    def _steady_compounder_cap_score(
        self,
        *,
        tag_positive_steady_compounder: float,
        steady_compounder_score: float,
        score_steady_return: float,
        score_positive_return_persistence: float,
        score_trend_durability: float,
        score_low_volatility: float,
        score_downside_control: float,
        score_drawdown_control: float,
        score_range_tightness: float,
    ) -> float:
        if tag_positive_steady_compounder == 1.0:
            return 1.0
        if score_low_volatility < 0.50 or score_downside_control < 0.55 or score_drawdown_control < 0.60:
            return 0.45
        if (
            steady_compounder_score < 0.45
            or score_steady_return < 0.25
            or score_positive_return_persistence < 0.55
            or score_trend_durability < 0.45
            or score_range_tightness < 0.10
        ):
            return 0.45
        if (
            steady_compounder_score < 0.55
            or score_steady_return < 0.35
            or score_trend_durability < 0.55
            or score_range_tightness < 0.20
        ):
            return 0.65
        return 0.82

    def _growth_cap_score(
        self,
        *,
        return_60d: float,
        return_120d: float,
        return_252d: float,
        positive_21d_return_ratio_252d: float,
    ) -> float:
        if return_120d <= 0.0 or return_252d <= 0.0:
            return 0.35
        if return_120d < 0.03 or return_252d < 0.06:
            return 0.35
        if return_60d < 0.0 or positive_21d_return_ratio_252d < 0.50:
            return 0.75
        if return_120d < 0.06 or return_252d < 0.10:
            return 0.85
        return 1.0

    def _drawdown_cap_score(self, *, max_drawdown_120d: float, max_drawdown_252d: float) -> float:
        if max_drawdown_252d < -0.35 or max_drawdown_120d < -0.28:
            return 0.30
        if max_drawdown_252d < -0.25 or max_drawdown_120d < -0.20:
            return 0.40
        if max_drawdown_252d < -0.18 or max_drawdown_120d < -0.15:
            return 0.70
        return 1.0

    def _liquidity_cap_score(
        self,
        *,
        stale_close_days_20d: float,
        zero_volume_days_20d: float,
        traded_value_5d_to_20d_ratio: float,
    ) -> float:
        if stale_close_days_20d >= 3.0 or zero_volume_days_20d >= 2.0 or traded_value_5d_to_20d_ratio < 0.50:
            return 0.30
        if stale_close_days_20d > 0.0 or zero_volume_days_20d > 0.0 or traded_value_5d_to_20d_ratio < 0.75:
            return 0.35
        return 1.0


DEFENSIVE_COMPOUNDER_QUALITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=DEFENSIVE_COMPOUNDER_QUALITY_PROFILE_ID,
    factory=DefensiveCompounderQualityV1Profile,
)
