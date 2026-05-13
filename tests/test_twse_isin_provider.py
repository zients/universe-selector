from __future__ import annotations

import ssl
from datetime import datetime, timezone
from pathlib import Path

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.providers import twse_isin as twse_isin_module
from universe_selector.providers.context import build_provider_run_context
from universe_selector.providers.twse_isin import (
    TWSE_ISIN_STR_MODE_2_SOURCE_ID,
    TWSE_ISIN_STR_MODE_4_SOURCE_ID,
    TwseIsinListingProvider,
    parse_twse_isin_listings,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "twse_isin"


def _payloads() -> dict[str, bytes]:
    return {
        TWSE_ISIN_STR_MODE_2_SOURCE_ID: (FIXTURE_DIR / "str_mode_2_sample.html").read_bytes(),
        TWSE_ISIN_STR_MODE_4_SOURCE_ID: (FIXTURE_DIR / "str_mode_4_sample.html").read_bytes(),
    }


def _context():
    return build_provider_run_context(
        market=Market.TW,
        data_fetch_started_at=datetime(2026, 5, 3, 13, 30, tzinfo=timezone.utc),
        ticker_limit=None,
    )


def test_twse_isin_fixtures_are_ms950_big5_bytes() -> None:
    payload = (FIXTURE_DIR / "str_mode_2_sample.html").read_bytes()

    assert "台積電" in payload.decode("ms950")
    with pytest.raises(UnicodeDecodeError):
        payload.decode("utf-8")


def test_twse_isin_parser_includes_only_twse_and_tpex_stock_sections() -> None:
    payloads = _payloads()

    listings = parse_twse_isin_listings(
        payloads[TWSE_ISIN_STR_MODE_2_SOURCE_ID],
        payloads[TWSE_ISIN_STR_MODE_4_SOURCE_ID],
    )

    assert [item.ticker for item in listings] == ["1101", "1234", "2330", "6543"]
    rows = {item.ticker: item for item in listings}
    assert rows["2330"].listing_symbol == "2330"
    assert rows["2330"].exchange_segment == "TWSE"
    assert rows["2330"].source_id == TWSE_ISIN_STR_MODE_2_SOURCE_ID
    assert rows["1234"].listing_symbol == "1234"
    assert rows["1234"].exchange_segment == "TPEX"
    assert rows["1234"].source_id == TWSE_ISIN_STR_MODE_4_SOURCE_ID
    assert {item.market for item in listings} == {Market.TW}
    assert {item.listing_status for item in listings} == {"active"}
    assert {item.instrument_type for item in listings} == {"common_stock"}


def test_twse_isin_parser_excludes_non_common_sections_and_pre_stock_rows() -> None:
    payloads = _payloads()

    listings = parse_twse_isin_listings(
        payloads[TWSE_ISIN_STR_MODE_2_SOURCE_ID],
        payloads[TWSE_ISIN_STR_MODE_4_SOURCE_ID],
    )

    tickers = {item.ticker for item in listings}
    assert {
        "0050",
        "006201",
        "01001T",
        "020001",
        "030001",
        "2881A",
        "700001",
        "9105",
        "9999",
    }.isdisjoint(tickers)


def test_twse_isin_provider_fetches_both_sources_and_returns_sorted_rows() -> None:
    payloads = _payloads()
    calls = []

    def fetch_bytes(source_id: str) -> bytes:
        calls.append(source_id)
        return payloads[source_id]

    adapter = TwseIsinListingProvider(fetch_bytes=fetch_bytes)

    listings = adapter.load_listings(_context(), Market.TW)

    assert adapter.provider_id == "twse_isin"
    assert adapter.source_ids == (TWSE_ISIN_STR_MODE_2_SOURCE_ID, TWSE_ISIN_STR_MODE_4_SOURCE_ID)
    assert calls == [TWSE_ISIN_STR_MODE_2_SOURCE_ID, TWSE_ISIN_STR_MODE_4_SOURCE_ID]
    assert [item.ticker for item in listings] == sorted(item.ticker for item in listings)


def test_twse_isin_default_fetch_keeps_verification_without_x509_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_exc_info: object) -> None:
            return None

        def read(self) -> bytes:
            return b"twse-html"

    def fake_urlopen(url: str, *, timeout: int, context: ssl.SSLContext | None = None) -> Response:
        captured["url"] = url
        captured["timeout"] = timeout
        captured["context"] = context
        return Response()

    monkeypatch.setattr(twse_isin_module, "urlopen", fake_urlopen)

    payload = twse_isin_module._default_fetch_bytes(TWSE_ISIN_STR_MODE_2_SOURCE_ID)

    assert payload == b"twse-html"
    assert captured["timeout"] == 30
    context = captured["context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.verify_flags & ssl.VERIFY_X509_STRICT == 0


def test_twse_isin_provider_rejects_non_tw_market() -> None:
    adapter = TwseIsinListingProvider(fetch_bytes=lambda _source_id: b"")

    with pytest.raises(ProviderDataError, match="TW"):
        adapter.load_listings(_context(), Market.US)
