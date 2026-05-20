from __future__ import annotations

from datetime import date
from typing import Protocol

import polars as pl

from universe_selector.domain import Market
from universe_selector.providers.models import ListingCandidate


class RankingProfile(Protocol):
    @property
    def profile_id(self) -> str: ...

    @property
    def snapshot_metric_keys(self) -> tuple[str, ...]: ...

    @property
    def ranking_metric_keys(self) -> tuple[str, ...]: ...

    @property
    def inspect_metric_keys(self) -> tuple[str, ...]: ...

    @property
    def horizon_order(self) -> tuple[str, ...]: ...

    @property
    def rank_interpretation_note(self) -> str: ...

    def validate(self) -> None: ...

    def ranking_config_payload(self) -> dict[str, object]: ...

    def build_snapshot(
        self,
        *,
        run_id: str,
        market: Market,
        listings: list[ListingCandidate],
        bars: pl.DataFrame,
        run_latest_bar_date: date,
    ) -> pl.DataFrame: ...

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame: ...
