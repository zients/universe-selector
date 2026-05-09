from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from urllib.request import urlopen

from universe_selector.domain import Market, canonical_ticker
from universe_selector.errors import ProviderDataError
from universe_selector.providers.context import ProviderRunContext
from universe_selector.providers.models import ListingCandidate
from universe_selector.providers.registration import ListingProviderRegistration


NASDAQ_LISTED_SOURCE_ID = "nasdaqtrader:nasdaqlisted"
OTHER_LISTED_SOURCE_ID = "nasdaqtrader:otherlisted"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


@dataclass(frozen=True)
class _ParsedListing:
    ticker: str
    listing_symbol: str
    exchange_segment: str
    listing_status: str
    instrument_type: str
    source_id: str
    identity: tuple[str, str, str, str, str, str]


def _default_fetch_text(source_id: str) -> str:
    url = {
        NASDAQ_LISTED_SOURCE_ID: NASDAQ_LISTED_URL,
        OTHER_LISTED_SOURCE_ID: OTHER_LISTED_URL,
    }[source_id]
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _rows(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(StringIO(text), delimiter="|")
    if reader.fieldnames is None:
        return []
    rows = []
    for row in reader:
        values = {str(key): (value or "").strip() for key, value in row.items() if key is not None}
        if _is_summary_row(values):
            continue
        rows.append(values)
    return rows


def _is_summary_row(row: dict[str, str]) -> bool:
    return any(value.lower().startswith("file creation time") for value in row.values())


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return value.strip()
    return ""


def _source_symbol(row: dict[str, str], source_id: str) -> str:
    if source_id == NASDAQ_LISTED_SOURCE_ID:
        return _field(row, "Symbol")
    return _field(row, "NASDAQ Symbol", "ACT Symbol", "CQS Symbol")


def _exchange_segment(row: dict[str, str], source_id: str) -> str:
    if source_id == NASDAQ_LISTED_SOURCE_ID:
        return "NASDAQ"
    exchange = _field(row, "Exchange")
    return exchange or "OTHER"


def _is_test_or_etf(row: dict[str, str]) -> bool:
    return (
        _field(row, "Test Issue").upper() != "N"
        or _field(row, "ETF").upper() != "N"
        or _field(row, "NextShares").upper() == "Y"
    )


def _has_bad_financial_status(row: dict[str, str], source_id: str) -> bool:
    if source_id != NASDAQ_LISTED_SOURCE_ID:
        return False
    status = _field(row, "Financial Status").upper()
    return status not in {"", "N"}


def _suffix_kind(symbol: str) -> str | None:
    upper = symbol.upper()
    suffixes = (
        (".WS", "warrant"),
        ("-WS", "warrant"),
        (".W", "warrant"),
        ("-W", "warrant"),
        (".U", "unit"),
        ("-U", "unit"),
        (".R", "right"),
        ("-R", "right"),
        (".P", "preferred"),
        ("-P", "preferred"),
    )
    for suffix, kind in suffixes:
        if upper.endswith(suffix) and len(upper) > len(suffix):
            return kind
    return None


def _name_kind(name: str) -> str | None:
    lowered = name.lower()
    if re.search(r"\bwarrants?\b", lowered):
        return "warrant"
    if re.search(r"\bunits?\b", lowered):
        return "unit"
    if re.search(r"\brights?\b", lowered):
        return "right"
    if re.search(
        r"\bpreferred\s+(stock|shares?)\b|\bpreference\s+shares?\b|\bdepositary\s+shares?.*\bpreferred\s+stock\b",
        lowered,
    ):
        return "preferred"
    return None


def _non_common_kind(symbol: str, name: str) -> str | None:
    name_kind = _name_kind(name)
    suffix_kind = _suffix_kind(symbol)
    if name_kind is not None:
        return name_kind
    if suffix_kind is not None and name_kind == suffix_kind:
        return suffix_kind
    return None


def _has_explicit_common_stock(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("common stock", "ordinary shares", "common shares"))


def _is_depositary_receipt(symbol: str, name: str) -> bool:
    upper_name = name.upper()
    return any(
        re.search(pattern, upper_name)
        for pattern in (
            r"\bADR\b",
            r"\bADS\b",
            r"AMERICAN DEPOSITARY RECEIPTS?",
            r"AMERICAN DEPOSITARY SHARES?",
        )
    )


def _is_common_stock(name: str) -> bool:
    return _has_explicit_common_stock(name)


def _parse_row(row: dict[str, str], source_id: str) -> _ParsedListing | None:
    if _is_test_or_etf(row) or _has_bad_financial_status(row, source_id):
        return None
    symbol = _source_symbol(row, source_id)
    if not symbol:
        return None
    security_name = _field(row, "Security Name")
    if not security_name:
        return None
    if _non_common_kind(symbol, security_name) is not None:
        return None

    if _is_depositary_receipt(symbol, security_name):
        instrument_type = "depositary_receipt"
    elif _is_common_stock(security_name):
        instrument_type = "common_stock"
    else:
        return None

    ticker = canonical_ticker(symbol)
    exchange_segment = _exchange_segment(row, source_id)
    listing_status = "active"
    identity = (ticker, symbol, exchange_segment, listing_status, instrument_type, source_id)
    return _ParsedListing(
        ticker=ticker,
        listing_symbol=symbol,
        exchange_segment=exchange_segment,
        listing_status=listing_status,
        instrument_type=instrument_type,
        source_id=source_id,
        identity=identity,
    )


def parse_nasdaq_trader_listings(nasdaqlisted_text: str, otherlisted_text: str) -> list[ListingCandidate]:
    parsed: dict[str, _ParsedListing] = {}
    for source_id, text in (
        (NASDAQ_LISTED_SOURCE_ID, nasdaqlisted_text),
        (OTHER_LISTED_SOURCE_ID, otherlisted_text),
    ):
        for row in _rows(text):
            listing = _parse_row(row, source_id)
            if listing is None:
                continue
            existing = parsed.get(listing.ticker)
            if existing is None:
                parsed[listing.ticker] = listing
                continue
            if existing.identity != listing.identity:
                raise ProviderDataError(f"duplicate canonical ticker in Nasdaq Trader listings: {listing.ticker}")

    return [
        ListingCandidate(
            market=Market.US,
            ticker=listing.ticker,
            listing_symbol=listing.listing_symbol,
            exchange_segment=listing.exchange_segment,
            listing_status=listing.listing_status,
            instrument_type=listing.instrument_type,
            source_id=listing.source_id,
        )
        for listing in sorted(parsed.values(), key=lambda item: item.ticker)
    ]


class NasdaqTraderListingProvider:
    provider_id = "nasdaq_trader"
    source_ids = (NASDAQ_LISTED_SOURCE_ID, OTHER_LISTED_SOURCE_ID)

    def __init__(self, fetch_text: Callable[[str], str] = _default_fetch_text) -> None:
        self._fetch_text = fetch_text

    def load_listings(self, context: ProviderRunContext, market: Market) -> list[ListingCandidate]:
        del context
        if market is not Market.US:
            raise ProviderDataError("Nasdaq Trader listing provider supports only US market")
        return parse_nasdaq_trader_listings(
            self._fetch_text(NASDAQ_LISTED_SOURCE_ID),
            self._fetch_text(OTHER_LISTED_SOURCE_ID),
        )


def _nasdaq_trader_factory(_config: object) -> NasdaqTraderListingProvider:
    return NasdaqTraderListingProvider()


NASDAQ_TRADER_LISTING_REGISTRATION = ListingProviderRegistration(
    provider_id=NasdaqTraderListingProvider.provider_id,
    supported_markets=frozenset({Market.US}),
    source_ids=NasdaqTraderListingProvider.source_ids,
    factory=_nasdaq_trader_factory,
)
