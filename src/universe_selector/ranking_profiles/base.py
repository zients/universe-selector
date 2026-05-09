from __future__ import annotations

from datetime import date
from typing import Protocol

import polars as pl

from universe_selector.domain import Market
from universe_selector.providers.models import ListingCandidate


class RankingProfile(Protocol):
    profile_id: str
    snapshot_metric_keys: tuple[str, ...]
    ranking_metric_keys: tuple[str, ...]
    inspect_metric_keys: tuple[str, ...]
    horizon_order: tuple[str, ...]
    rank_interpretation_note: str

    def validate(self) -> None:
        ...

    def ranking_config_payload(self) -> dict[str, object]:
        ...

    def build_snapshot(
        self,
        *,
        run_id: str,
        market: Market,
        listings: list[ListingCandidate],
        bars: pl.DataFrame,
        run_latest_bar_date: date,
    ) -> pl.DataFrame:
        ...

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        ...
