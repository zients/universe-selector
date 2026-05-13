from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.providers.models import ListingCandidate
from universe_selector.errors import ProviderDataError
from universe_selector.providers.context import build_provider_run_context
from universe_selector.providers.yfinance_ohlcv import YFinanceOhlcvProvider


def _context(market: Market = Market.US):
    return build_provider_run_context(
        market=market,
        data_fetch_started_at=datetime(2026, 5, 3, 13, 30, tzinfo=timezone.utc),
        ticker_limit=None,
    )


def _listing(
    ticker: str,
    listing_symbol: str | None = None,
    *,
    market: Market = Market.US,
    exchange_segment: str = "NASDAQ",
) -> ListingCandidate:
    return ListingCandidate(
        market=market,
        ticker=ticker,
        listing_symbol=listing_symbol or ticker,
        exchange_segment=exchange_segment,
        listing_status="active",
        instrument_type="common_stock",
        source_id=f"unit:{ticker}",
    )


class DownloadProbe:
    def __init__(self, result=None) -> None:
        self.calls = []
        self.result = result if result is not None else pl.DataFrame()

    def __call__(self, tickers, **kwargs):
        self.calls.append((tickers, kwargs))
        return self.result


class BatchDownloadProbe:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, tickers, **kwargs):
        self.calls.append((list(tickers), kwargs))
        return _multi_ticker_frame_for(list(tickers))


def _single_ticker_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [9.0, 10.0],
            "High": [10.0, 11.0],
            "Low": [8.0, 9.0],
            "Close": [9.5, 10.5],
            "Adj Close": [9.25, 10.25],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2026-05-01", "2026-05-04"]),
    )


def _multi_ticker_frame() -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["AAPL", "BRK-B"]],
    )
    return pd.DataFrame(
        [
            [9.0, 90.0, 10.0, 91.0, 8.0, 89.0, 9.5, 90.5, 9.25, 90.25, 1000, 2000],
            [10.0, None, 11.0, None, 9.0, None, 10.5, None, 10.25, None, 1100, None],
        ],
        columns=columns,
        index=pd.to_datetime(["2026-05-01", "2026-05-04"]),
    )


def _multi_ticker_frame_for(request_symbols: list[str]) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], request_symbols],
    )
    row = []
    for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        for index, _request_symbol in enumerate(request_symbols):
            if column == "Volume":
                row.append(1000 + index)
            else:
                row.append(10.0 + index)
    return pd.DataFrame([row], columns=columns, index=pd.to_datetime(["2026-05-01"]))


def _tw_multi_ticker_frame() -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["2330.TW", "1234.TWO"]],
    )
    return pd.DataFrame(
        [
            [600.0, 90.0, 610.0, 91.0, 590.0, 89.0, 605.0, 90.5, 605.0, 90.25, 1000, 2000],
            [610.0, None, 620.0, None, 600.0, None, 615.0, None, 615.0, None, 1100, None],
        ],
        columns=columns,
        index=pd.to_datetime(["2026-05-01", "2026-05-04"]),
    )


def test_yfinance_us_symbol_mapping_is_owned_by_adapter() -> None:
    download = DownloadProbe(_multi_ticker_frame())
    adapter = YFinanceOhlcvProvider(download=download)

    adapter.load_ohlcv(
        _context(),
        Market.US,
        [
            _listing("AAPL", listing_symbol="AAPL LISTING SYMBOL"),
            _listing("BRK.B", listing_symbol="BRK.B LISTING SYMBOL"),
        ],
    )

    tickers, _kwargs = download.calls[0]
    assert tickers == ["AAPL", "BRK-B"]


def test_yfinance_skips_us_symbols_with_unsupported_punctuation() -> None:
    download = DownloadProbe(_single_ticker_frame())
    adapter = YFinanceOhlcvProvider(download=download)

    adapter.load_ohlcv(
        _context(),
        Market.US,
        [
            _listing("AAPL"),
            _listing("BAD^"),
            _listing("BAD/"),
            _listing("BAD$"),
            _listing("BAD SPACE"),
            _listing(" BAD"),
            _listing("BAD "),
            _listing("BAD@"),
            _listing("BAD-"),
            _listing(".BAD"),
            _listing("BAD."),
            _listing("A..B"),
        ],
    )

    tickers, _kwargs = download.calls[0]
    assert tickers == ["AAPL"]


def test_yfinance_raises_when_all_candidates_are_unmappable() -> None:
    adapter = YFinanceOhlcvProvider(download=DownloadProbe())

    with pytest.raises(ProviderDataError, match="no yfinance-mappable"):
        adapter.load_ohlcv(_context(), Market.US, [_listing("BAD^"), _listing("BAD/")])


def test_yfinance_download_window_and_options_are_explicit() -> None:
    download = DownloadProbe(_single_ticker_frame())
    adapter = YFinanceOhlcvProvider(download=download)

    adapter.load_ohlcv(_context(), Market.US, [_listing("AAPL")])

    tickers, kwargs = download.calls[0]
    assert tickers == ["AAPL"]
    assert kwargs["start"] == date(2025, 3, 9)
    assert kwargs["end"] == date(2026, 5, 4)
    assert (kwargs["end"] - kwargs["start"]).days == 421
    assert kwargs["interval"] == "1d"
    assert kwargs["auto_adjust"] is False
    assert kwargs["back_adjust"] is False
    assert kwargs["actions"] is False
    assert kwargs["repair"] is False
    assert kwargs["progress"] is False
    assert kwargs["multi_level_index"] is True
    assert kwargs["threads"] is True
    assert "period" not in kwargs


def test_yfinance_downloads_request_symbols_in_configured_batches() -> None:
    download = BatchDownloadProbe()
    adapter = YFinanceOhlcvProvider(download=download, batch_size=2)

    bars = adapter.load_ohlcv(
        _context(),
        Market.US,
        [_listing("AAA"), _listing("BBB"), _listing("BRK.B"), _listing("CCC"), _listing("DDD")],
    )

    assert [tickers for tickers, _kwargs in download.calls] == [
        ["AAA", "BBB"],
        ["BRK-B", "CCC"],
        ["DDD"],
    ]
    assert all(kwargs["threads"] is True for _tickers, kwargs in download.calls)
    assert set(bars["ticker"].to_list()) == {"AAA", "BBB", "BRK.B", "CCC", "DDD"}


def test_yfinance_lookback_keeps_sample_profile_bar_requirement_buffer() -> None:
    adapter = YFinanceOhlcvProvider(download=DownloadProbe())

    assert adapter.lookback_days == 420
    assert adapter.lookback_days > 274


def test_yfinance_normalizes_multi_ticker_dataframe_to_canonical_bars() -> None:
    adapter = YFinanceOhlcvProvider(download=DownloadProbe(_multi_ticker_frame()))

    bars = adapter.load_ohlcv(_context(), Market.US, [_listing("AAPL"), _listing("BRK.B")])

    assert bars.columns == [
        "market",
        "ticker",
        "bar_date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]
    assert bars.to_dicts() == [
        {
            "market": "US",
            "ticker": "AAPL",
            "bar_date": date(2026, 5, 1),
            "open": 9.0,
            "high": 10.0,
            "low": 8.0,
            "close": 9.5,
            "adjusted_close": 9.25,
            "volume": 1000.0,
        },
        {
            "market": "US",
            "ticker": "AAPL",
            "bar_date": date(2026, 5, 4),
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "adjusted_close": 10.25,
            "volume": 1100.0,
        },
        {
            "market": "US",
            "ticker": "BRK.B",
            "bar_date": date(2026, 5, 1),
            "open": 90.0,
            "high": 91.0,
            "low": 89.0,
            "close": 90.5,
            "adjusted_close": 90.25,
            "volume": 2000.0,
        },
    ]


def test_yfinance_normalizes_single_ticker_dataframe_to_same_shape() -> None:
    adapter = YFinanceOhlcvProvider(download=DownloadProbe(_single_ticker_frame()))

    bars = adapter.load_ohlcv(_context(), Market.US, [_listing("AAPL")])

    assert bars.select(["market", "ticker", "bar_date", "close", "adjusted_close", "volume"]).to_dicts() == [
        {
            "market": "US",
            "ticker": "AAPL",
            "bar_date": date(2026, 5, 1),
            "close": 9.5,
            "adjusted_close": 9.25,
            "volume": 1000.0,
        },
        {
            "market": "US",
            "ticker": "AAPL",
            "bar_date": date(2026, 5, 4),
            "close": 10.5,
            "adjusted_close": 10.25,
            "volume": 1100.0,
        },
    ]


@pytest.mark.parametrize("missing_column", ["Close", "Adj Close", "Volume"])
def test_yfinance_requires_canonical_price_and_volume_columns(missing_column: str) -> None:
    frame = _single_ticker_frame().drop(columns=[missing_column])
    adapter = YFinanceOhlcvProvider(download=DownloadProbe(frame))

    with pytest.raises(ProviderDataError, match=missing_column):
        adapter.load_ohlcv(_context(), Market.US, [_listing("AAPL")])


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Adj Close": [1.0], "Volume": [1.0]}),
    ],
)
def test_yfinance_rejects_empty_or_all_null_date_results(frame: pd.DataFrame) -> None:
    adapter = YFinanceOhlcvProvider(download=DownloadProbe(frame))

    with pytest.raises(ProviderDataError, match="OHLCV provider returned no usable bars"):
        adapter.load_ohlcv(_context(), Market.US, [_listing("AAPL")])


def test_yfinance_allows_equal_close_and_adjusted_close() -> None:
    frame = _single_ticker_frame()
    frame["Adj Close"] = frame["Close"]
    adapter = YFinanceOhlcvProvider(download=DownloadProbe(frame))

    bars = adapter.load_ohlcv(_context(), Market.US, [_listing("AAPL")])

    assert bars["close"].to_list() == bars["adjusted_close"].to_list()


def test_yfinance_tw_symbol_mapping_uses_exchange_segment_and_preserves_canonical_tickers() -> None:
    download = DownloadProbe(_tw_multi_ticker_frame())
    adapter = YFinanceOhlcvProvider(download=download)

    bars = adapter.load_ohlcv(
        _context(Market.TW),
        Market.TW,
        [
            _listing("2330", market=Market.TW, exchange_segment="TWSE"),
            _listing("1234", market=Market.TW, exchange_segment="TPEX"),
        ],
    )

    tickers, _kwargs = download.calls[0]
    assert tickers == ["2330.TW", "1234.TWO"]
    assert bars.select(["market", "ticker", "bar_date"]).to_dicts() == [
        {"market": "TW", "ticker": "1234", "bar_date": date(2026, 5, 1)},
        {"market": "TW", "ticker": "2330", "bar_date": date(2026, 5, 1)},
        {"market": "TW", "ticker": "2330", "bar_date": date(2026, 5, 4)},
    ]


def test_yfinance_tw_unknown_exchange_segment_raises_without_download() -> None:
    download = DownloadProbe(_single_ticker_frame())
    adapter = YFinanceOhlcvProvider(download=download)

    with pytest.raises(ProviderDataError, match="unsupported TW exchange segment"):
        adapter.load_ohlcv(
            _context(Market.TW),
            Market.TW,
            [_listing("9999", market=Market.TW, exchange_segment="EMERGING")],
        )

    assert download.calls == []


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "1000"])
def test_yfinance_rejects_invalid_direct_batch_size(batch_size: object) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        YFinanceOhlcvProvider(download=DownloadProbe(), batch_size=batch_size)  # type: ignore[arg-type]
