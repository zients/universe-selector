from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import polars as pl

from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.providers.context import ProviderRunContext
from universe_selector.providers.models import ListingCandidate
from universe_selector.providers.registration import OhlcvProviderRegistration


# Keep a generous daily-bar lookback to cover weekends and market holidays
# while keeping live smoke runs bounded.
YFINANCE_LOOKBACK_DAYS = 420
CANONICAL_OHLCV_COLUMNS = [
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
YFINANCE_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def _default_download(tickers: list[str], **kwargs: object) -> Any:
    import yfinance as yf

    return yf.download(tickers, **kwargs)


class YFinanceOhlcvProvider:
    provider_id = "yfinance"
    source_ids = ("yahoo-finance:yfinance-download",)
    lookback_days = YFINANCE_LOOKBACK_DAYS

    def __init__(self, download: Callable[..., Any] = _default_download, *, batch_size: int = 1000) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._download = download
        self.batch_size = batch_size

    def load_ohlcv(
        self,
        context: ProviderRunContext,
        market: Market,
        listings: list[ListingCandidate],
    ) -> pl.DataFrame:
        canonical_to_request, request_to_canonical = self._symbol_maps(listings, market)
        if not canonical_to_request:
            raise ProviderDataError("no yfinance-mappable listings")

        frames = []
        request_symbols = list(canonical_to_request.values())
        for batch in _batches(request_symbols, self.batch_size):
            frame = self._download(
                batch,
                start=context.market_fetch_date - timedelta(days=self.lookback_days),
                end=context.market_fetch_date + timedelta(days=1),
                interval="1d",
                auto_adjust=False,
                back_adjust=False,
                actions=False,
                repair=False,
                progress=False,
                threads=True,
                multi_level_index=True,
            )
            batch_request_to_canonical = {request_symbol: request_to_canonical[request_symbol] for request_symbol in batch}
            try:
                frames.append(self._normalize_download(frame, market, batch_request_to_canonical))
            except ProviderDataError as exc:
                if str(exc) != "OHLCV provider returned no usable bars":
                    raise
        if not frames:
            raise ProviderDataError("OHLCV provider returned no usable bars")
        return pl.concat(frames).select(CANONICAL_OHLCV_COLUMNS).sort(["ticker", "bar_date"])

    def _symbol_maps(
        self,
        listings: list[ListingCandidate],
        market: Market,
    ) -> tuple[dict[str, str], dict[str, str]]:
        canonical_to_request = {}
        request_to_canonical = {}
        for listing in listings:
            request_symbol = self._request_symbol(listing, market)
            if request_symbol is None:
                continue
            existing = request_to_canonical.get(request_symbol)
            if existing is not None and existing != listing.ticker:
                raise ProviderDataError(f"duplicate yfinance request symbol: {request_symbol}")
            canonical_to_request[listing.ticker] = request_symbol
            request_to_canonical[request_symbol] = listing.ticker
        return canonical_to_request, request_to_canonical

    def _request_symbol(self, listing: ListingCandidate, market: Market) -> str | None:
        ticker = listing.ticker.upper()
        if ticker != listing.ticker:
            return None
        if market is Market.US:
            if re.search(r"[\^/\$\s-]", ticker):
                return None
            if re.fullmatch(r"[A-Z0-9]+", ticker):
                return ticker
            if re.fullmatch(r"[A-Z0-9]+\.[A-Z0-9]+", ticker):
                return ticker.replace(".", "-")
            return None
        if market is Market.TW:
            if not re.fullmatch(r"[A-Z0-9]+", ticker):
                return None
            exchange_segment = listing.exchange_segment.upper()
            if exchange_segment == "TWSE":
                return f"{ticker}.TW"
            if exchange_segment == "TPEX":
                return f"{ticker}.TWO"
            raise ProviderDataError(f"unsupported TW exchange segment for yfinance: {listing.exchange_segment}")
        return None

    def _normalize_download(
        self,
        frame: Any,
        market: Market,
        request_to_canonical: dict[str, str],
    ) -> pl.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ProviderDataError("OHLCV provider returned no usable bars")
        rows = []
        if isinstance(frame.columns, pd.MultiIndex):
            rows = self._multi_ticker_rows(frame, market, request_to_canonical)
        elif len(request_to_canonical) == 1:
            request_symbol, canonical_ticker = next(iter(request_to_canonical.items()))
            del request_symbol
            rows = self._single_ticker_rows(frame, market, canonical_ticker)
        else:
            raise ProviderDataError("yfinance multi-ticker result must use MultiIndex columns")
        if not rows:
            raise ProviderDataError("OHLCV provider returned no usable bars")
        return pl.DataFrame(rows).select(CANONICAL_OHLCV_COLUMNS).sort(["ticker", "bar_date"])

    def _multi_ticker_rows(
        self,
        frame: pd.DataFrame,
        market: Market,
        request_to_canonical: dict[str, str],
    ) -> list[dict[str, object]]:
        rows = []
        for request_symbol, canonical_ticker in request_to_canonical.items():
            if request_symbol not in frame.columns.get_level_values(1):
                continue
            ticker_frame = frame.xs(request_symbol, axis=1, level=1, drop_level=True)
            rows.extend(self._single_ticker_rows(ticker_frame, market, canonical_ticker))
        return rows

    def _single_ticker_rows(
        self,
        frame: pd.DataFrame,
        market: Market,
        canonical_ticker: str,
    ) -> list[dict[str, object]]:
        self._require_columns(frame)
        rows = []
        for index, row in frame.iterrows():
            bar_date = _bar_date_from_index(index)
            if bar_date is None:
                continue
            if any(pd.isna(row[column]) for column in YFINANCE_REQUIRED_COLUMNS):
                continue
            rows.append(
                {
                    "market": market.value,
                    "ticker": canonical_ticker,
                    "bar_date": bar_date,
                    "open": _float_or_none(row["Open"]),
                    "high": _float_or_none(row["High"]),
                    "low": _float_or_none(row["Low"]),
                    "close": float(row["Close"]),
                    "adjusted_close": float(row["Adj Close"]),
                    "volume": float(row["Volume"]),
                }
            )
        return rows

    def _require_columns(self, frame: pd.DataFrame) -> None:
        missing = [column for column in YFINANCE_REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ProviderDataError(f"yfinance result missing required columns: {', '.join(missing)}")


def _bar_date_from_index(index: object) -> date | None:
    if pd.isna(index):
        return None
    if isinstance(index, int | float):
        return None
    if isinstance(index, pd.Timestamp):
        return index.date()
    if isinstance(index, datetime):
        return index.date()
    if isinstance(index, date):
        return index
    parsed = pd.to_datetime(index, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _float_or_none(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _batches(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def _yfinance_factory(config: object) -> YFinanceOhlcvProvider:
    return YFinanceOhlcvProvider(batch_size=getattr(config, "live_yfinance_batch_size"))


YFINANCE_OHLCV_REGISTRATION = OhlcvProviderRegistration(
    provider_id=YFinanceOhlcvProvider.provider_id,
    supported_markets=frozenset({Market.US, Market.TW}),
    source_ids=YFinanceOhlcvProvider.source_ids,
    factory=_yfinance_factory,
)
