from __future__ import annotations

import math

import polars as pl


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def percentile_rank(values: pl.Series) -> pl.Series:
    finite_values = [float(value) for value in values.to_list() if _is_finite(value)]
    n = len(finite_values)
    if n == 0:
        return pl.Series(values.name, [])
    if n == 1 or len(set(finite_values)) == 1:
        return pl.Series(values.name, [50.0 for _ in finite_values])

    sorted_values = sorted(finite_values)
    rank_by_value: dict[float, float] = {}
    index = 0
    while index < n:
        value = sorted_values[index]
        end = index
        while end < n and sorted_values[end] == value:
            end += 1
        first_rank = index + 1
        last_rank = end
        average_rank = (first_rank + last_rank) / 2.0
        rank_by_value[value] = 100.0 * (average_rank - 0.5) / n
        index = end

    return pl.Series(values.name, [rank_by_value[value] for value in finite_values])
