from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from universe_selector.domain import Market


@dataclass(frozen=True)
class ListingCandidate:
    market: Market
    ticker: str
    listing_symbol: str
    exchange_segment: str
    listing_status: str
    instrument_type: str
    source_id: str


@dataclass(frozen=True)
class ProviderMetadata:
    data_mode: str
    listing_provider_id: str
    listing_source_id: str
    ohlcv_provider_id: str
    ohlcv_source_id: str
    provider_config_hash: str
    data_fetch_started_at: datetime
    market_timezone: str
    run_latest_bar_date: date


@dataclass(frozen=True)
class ProviderRunData:
    metadata: ProviderMetadata
    listings: list[ListingCandidate]
    bars: pl.DataFrame
