from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.providers.context import build_provider_run_context
from universe_selector.providers.nasdaq_trader import (
    NASDAQ_LISTED_SOURCE_ID,
    OTHER_LISTED_SOURCE_ID,
    NasdaqTraderListingProvider,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "nasdaq_trader"


def _sample_texts() -> dict[str, str]:
    return {
        NASDAQ_LISTED_SOURCE_ID: (FIXTURE_DIR / "nasdaqlisted_sample.txt").read_text(),
        OTHER_LISTED_SOURCE_ID: (FIXTURE_DIR / "otherlisted_sample.txt").read_text(),
    }


def _provider(texts: dict[str, str] | None = None) -> NasdaqTraderListingProvider:
    payloads = texts or _sample_texts()

    def fetch_text(source_id: str) -> str:
        return payloads[source_id]

    return NasdaqTraderListingProvider(fetch_text=fetch_text)


def _context():
    return build_provider_run_context(
        market=Market.US,
        data_fetch_started_at=datetime.fromisoformat("2026-05-03T13:30:00+00:00"),
        ticker_limit=None,
    )


def test_nasdaq_trader_provider_includes_common_stocks_adrs_and_class_shares() -> None:
    listings = _provider().load_listings(_context(), Market.US)
    rows = {item.ticker: item for item in listings}

    assert rows["AAPL"].instrument_type == "common_stock"
    assert rows["AAPL"].listing_symbol == "AAPL"
    assert rows["BABA"].instrument_type == "depositary_receipt"
    assert rows["BABA"].listing_symbol == "BABA"
    assert rows["BRK.B"].instrument_type == "common_stock"
    assert rows["BRK.B"].listing_symbol == "BRK.B"
    assert rows["SAFE"].instrument_type == "common_stock"
    assert rows["PBNK"].instrument_type == "common_stock"
    assert rows["FUBC"].instrument_type == "common_stock"
    assert rows["ADSK"].instrument_type == "common_stock"
    assert [item.ticker for item in listings] == sorted(rows)


def test_nasdaq_trader_provider_excludes_non_common_rows_and_ambiguous_rows() -> None:
    listings = _provider().load_listings(_context(), Market.US)
    tickers = {item.ticker for item in listings}

    assert {
        "ETFZ",
        "NEXT",
        "TEST",
        "SUSP",
        "AMBIG",
        "PREFP",
        "WARRW",
        "UNITU",
        "RIGHTR",
        "ADSP",
        "NTEST",
        "NOTE",
        "WCTX.W",
        "UCTX-U",
        "RCTX.R",
        "PCTX-P",
        "WSCTX.WS",
        "WSCTX-WS",
    }.isdisjoint(tickers)


def test_nasdaq_trader_provider_does_not_over_exclude_w_u_r_symbols() -> None:
    listings = _provider().load_listings(_context(), Market.US)
    tickers = {item.ticker for item in listings}

    assert {"W", "U", "R", "POWER", "UNITED", "KEEPW", "CUSIPR", "CTX.W", "CTX-W"}.issubset(tickers)


def test_nasdaq_trader_provider_rejects_non_us_market() -> None:
    with pytest.raises(ProviderDataError, match="US"):
        _provider().load_listings(_context(), Market.TW)


def test_nasdaq_trader_provider_rejects_unsafe_duplicate_canonical_tickers() -> None:
    texts = _sample_texts()
    texts[OTHER_LISTED_SOURCE_ID] = (
        texts[OTHER_LISTED_SOURCE_ID]
        + "AAPL|Apple Holdings Different Common Stock|N|AAPL|N|100|N|AAPL\n"
    )

    with pytest.raises(ProviderDataError, match="duplicate canonical ticker"):
        _provider(texts).load_listings(_context(), Market.US)


def test_nasdaq_trader_provider_allows_safe_equivalent_duplicates() -> None:
    listings = _provider().load_listings(_context(), Market.US)

    assert [item.ticker for item in listings].count("SAFE") == 1
