from __future__ import annotations

import math
from datetime import date
from types import MappingProxyType
from typing import Mapping

from universe_selector.domain import Market


def all_finite(values: list[object]) -> bool:
    return all(
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in values
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float], *, ddof: int) -> float | None:
    if len(values) <= ddof:
        return None
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - ddof)
    return math.sqrt(variance)


def downside_std(values: list[float], *, ddof: int) -> float | None:
    downside = [min(value, 0.0) for value in values]
    return std(downside, ddof=ddof)


def max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def ols_slope_r2(values: list[float]) -> tuple[float, float]:
    y_values = [math.log(value) for value in values]
    x_values = list(range(len(y_values)))
    x_mean = mean([float(value) for value in x_values])
    y_mean = mean(y_values)
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


def yyyymmdd(value: date) -> float:
    return float(value.year * 10_000 + value.month * 100 + value.day)


def percentile_scores(values: list[float], *, higher_is_better: bool = True) -> list[float]:
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


def positive_only_percentile_scores(values: list[float]) -> list[float]:
    scores = [0.0] * len(values)
    positive_pairs = [(index, value) for index, value in enumerate(values) if value > 0.0]
    if not positive_pairs:
        return scores
    positive_scores = percentile_scores([value for _index, value in positive_pairs])
    for (original_index, _value), score in zip(positive_pairs, positive_scores, strict=True):
        scores[original_index] = score
    return scores


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def band_score(
    value: float,
    *,
    ideal_low: float,
    ideal_high: float,
    outer_low: float,
    outer_high: float,
) -> float:
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        if value <= outer_low:
            return 0.0
        return (value - outer_low) / (ideal_low - outer_low)
    if value >= outer_high:
        return 0.0
    return (outer_high - value) / (outer_high - ideal_high)


def immutable_market_float_mapping(value: Mapping[Market, float]) -> Mapping[Market, float]:
    return MappingProxyType({market: float(value[market]) for market in Market})


def immutable_market_int_mapping(value: Mapping[Market, int]) -> Mapping[Market, int]:
    return MappingProxyType({market: int(value[market]) for market in Market})
