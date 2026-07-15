from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import polars as pl

from universe_selector.domain import Market
from universe_selector.providers.context import ProviderRunContext
from universe_selector.providers.models import (
    FundamentalsRunData,
    FundamentalsUniverseRunData,
    ListingCandidate,
    ProviderDataRequirements,
    ProviderRunData,
)


class MarketDataProvider(ABC):
    @abstractmethod
    def load_run_data(self, market: Market, requirements: ProviderDataRequirements | None = None) -> ProviderRunData:
        raise NotImplementedError


class ListingProvider(Protocol):
    provider_id: str
    source_ids: tuple[str, ...]

    def load_listings(self, context: ProviderRunContext, market: Market) -> list[ListingCandidate]:
        raise NotImplementedError


class OhlcvProvider(Protocol):
    provider_id: str
    source_ids: tuple[str, ...]

    def load_ohlcv(
        self,
        context: ProviderRunContext,
        market: Market,
        listings: list[ListingCandidate],
    ) -> pl.DataFrame:
        raise NotImplementedError


class FundamentalsProvider(Protocol):
    provider_id: str
    source_ids: tuple[str, ...]

    def load_fundamentals(
        self,
        market: Market,
        ticker: str,
        *,
        listing: ListingCandidate | None = None,
    ) -> FundamentalsRunData:
        raise NotImplementedError

    def load_fundamentals_for_listings(
        self,
        context: ProviderRunContext,
        market: Market,
        listings: list[ListingCandidate],
    ) -> FundamentalsUniverseRunData:
        raise NotImplementedError
