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


@dataclass(frozen=True)
class FundamentalsMetadata:
    data_mode: str
    fundamentals_provider_id: str
    fundamentals_source_ids: tuple[str, ...]
    data_fetch_started_at: datetime
    facts_as_of: date


@dataclass(frozen=True)
class FundamentalFacts:
    market: Market
    ticker: str
    currency: str
    reference_price: float
    reference_price_as_of: date
    reference_price_as_of_source: str
    reference_price_as_of_note: str | None
    shares_outstanding: float
    cash_and_cash_equivalents: float
    total_debt: float
    balance_sheet_as_of: date
    net_debt: float
    operating_cash_flow: float
    capital_expenditures: float
    free_cash_flow: float
    fiscal_period_end: date
    fiscal_period_type: str


@dataclass(frozen=True)
class FundamentalsRunData:
    metadata: FundamentalsMetadata
    facts: FundamentalFacts
