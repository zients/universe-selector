from __future__ import annotations

import re
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.request import urlopen

from universe_selector.domain import Market, canonical_ticker
from universe_selector.errors import ProviderDataError
from universe_selector.providers.context import ProviderRunContext
from universe_selector.providers.models import ListingCandidate
from universe_selector.providers.registration import ListingProviderRegistration


TWSE_ISIN_STR_MODE_2_SOURCE_ID = "twse:isin:strMode=2"
TWSE_ISIN_STR_MODE_4_SOURCE_ID = "twse:isin:strMode=4"
TWSE_ISIN_STR_MODE_2_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
TWSE_ISIN_STR_MODE_4_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"

_SECTION_STOCK = "股票"
_SOURCE_MARKET_RULES = {
    TWSE_ISIN_STR_MODE_2_SOURCE_ID: ("上市", "TWSE"),
    TWSE_ISIN_STR_MODE_4_SOURCE_ID: ("上櫃", "TPEX"),
}
_SOURCE_URLS = {
    TWSE_ISIN_STR_MODE_2_SOURCE_ID: TWSE_ISIN_STR_MODE_2_URL,
    TWSE_ISIN_STR_MODE_4_SOURCE_ID: TWSE_ISIN_STR_MODE_4_URL,
}


def _twse_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    # TWCA's root chain fails OpenSSL strict X.509 checks, while normal CA and
    # hostname verification succeeds. Keep verification on and relax only strict.
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


@dataclass(frozen=True)
class _ParsedTwListing:
    ticker: str
    listing_symbol: str
    exchange_segment: str
    source_id: str


class _TableRowParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() in {"td", "th"} and self._current_row is not None:
            self._current_cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_cell_parts is not None:
            self._current_cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell_parts is not None:
            self._current_row.append(_clean_text("".join(self._current_cell_parts)))
            self._current_cell_parts = None
        elif tag == "tr" and self._current_row is not None:
            row = [cell for cell in self._current_row if cell]
            if row:
                self.rows.append(row)
            self._current_row = None
            self._current_cell_parts = None


def _default_fetch_bytes(source_id: str) -> bytes:
    with urlopen(_SOURCE_URLS[source_id], timeout=30, context=_twse_ssl_context()) as response:
        return response.read()


def parse_twse_isin_listings(str_mode_2_html: bytes, str_mode_4_html: bytes) -> list[ListingCandidate]:
    parsed: dict[str, _ParsedTwListing] = {}
    for source_id, html in (
        (TWSE_ISIN_STR_MODE_2_SOURCE_ID, str_mode_2_html),
        (TWSE_ISIN_STR_MODE_4_SOURCE_ID, str_mode_4_html),
    ):
        for listing in _parse_source(source_id, html):
            existing = parsed.get(listing.ticker)
            if existing is not None and existing != listing:
                raise ProviderDataError(f"duplicate canonical ticker in TWSE ISIN listings: {listing.ticker}")
            parsed[listing.ticker] = listing

    return [
        ListingCandidate(
            market=Market.TW,
            ticker=listing.ticker,
            listing_symbol=listing.listing_symbol,
            exchange_segment=listing.exchange_segment,
            listing_status="active",
            instrument_type="common_stock",
            source_id=listing.source_id,
        )
        for listing in sorted(parsed.values(), key=lambda item: item.ticker)
    ]


class TwseIsinListingProvider:
    provider_id = "twse_isin"
    source_ids = (TWSE_ISIN_STR_MODE_2_SOURCE_ID, TWSE_ISIN_STR_MODE_4_SOURCE_ID)

    def __init__(self, fetch_bytes: Callable[[str], bytes] = _default_fetch_bytes) -> None:
        self._fetch_bytes = fetch_bytes

    def load_listings(self, context: ProviderRunContext, market: Market) -> list[ListingCandidate]:
        del context
        if market is not Market.TW:
            raise ProviderDataError("TWSE ISIN listing provider supports only TW market")
        return parse_twse_isin_listings(
            self._fetch_bytes(TWSE_ISIN_STR_MODE_2_SOURCE_ID),
            self._fetch_bytes(TWSE_ISIN_STR_MODE_4_SOURCE_ID),
        )


def _twse_isin_factory(_config: object) -> TwseIsinListingProvider:
    return TwseIsinListingProvider()


TWSE_ISIN_LISTING_REGISTRATION = ListingProviderRegistration(
    provider_id=TwseIsinListingProvider.provider_id,
    supported_markets=frozenset({Market.TW}),
    source_ids=TwseIsinListingProvider.source_ids,
    factory=_twse_isin_factory,
)


def _parse_source(source_id: str, html: bytes) -> list[_ParsedTwListing]:
    expected_market, exchange_segment = _SOURCE_MARKET_RULES[source_id]
    rows = _html_rows(html)
    listings = []
    current_section = ""
    for row in rows:
        section = _section_name(row)
        if section is not None:
            current_section = section
            continue
        if current_section != _SECTION_STOCK:
            continue
        if _is_header_row(row):
            continue
        if _market_text(row) != expected_market:
            continue
        listing_symbol = _listing_symbol(row[0])
        if listing_symbol is None:
            continue
        listings.append(
            _ParsedTwListing(
                ticker=canonical_ticker(listing_symbol),
                listing_symbol=listing_symbol,
                exchange_segment=exchange_segment,
                source_id=source_id,
            )
        )
    return listings


def _html_rows(html: bytes) -> list[list[str]]:
    parser = _TableRowParser()
    parser.feed(html.decode("ms950"))
    parser.close()
    return parser.rows


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _section_name(row: list[str]) -> str | None:
    if len(row) == 1 and not _looks_like_listing_cell(row[0]):
        return row[0]
    return None


def _looks_like_listing_cell(value: str) -> bool:
    return re.match(r"^[0-9A-Za-z]+\s+", value) is not None


def _is_header_row(row: list[str]) -> bool:
    return bool(row) and row[0].startswith("有價證券代號")


def _market_text(row: list[str]) -> str:
    for cell in row[1:]:
        if cell in {"上市", "上櫃"}:
            return cell
    return ""


def _listing_symbol(value: str) -> str | None:
    match = re.match(r"^(?P<symbol>[0-9A-Za-z]+)\s+", value)
    if match is None:
        return None
    return match.group("symbol").upper()
