from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import pytest

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.providers.models import ListingCandidate
from universe_selector.errors import ProviderDataError
from universe_selector.providers.live import LiveMarketDataProvider
from universe_selector.providers.registration import ListingProviderRegistration, OhlcvProviderRegistration


def _listing(ticker: str, *, market: Market = Market.US, exchange_segment: str = "NASDAQ") -> ListingCandidate:
    return ListingCandidate(
        market=market,
        ticker=ticker,
        listing_symbol=ticker,
        exchange_segment=exchange_segment,
        listing_status="active",
        instrument_type="common_stock",
        source_id=f"unit:{ticker}",
    )


class FakeListingProvider:
    def __init__(
        self,
        listings: list[ListingCandidate],
        *,
        provider_id: str = "nasdaq_trader",
        source_ids: tuple[str, ...] = ("nasdaqtrader:nasdaqlisted", "nasdaqtrader:otherlisted"),
    ) -> None:
        self.provider_id = provider_id
        self.source_ids = source_ids
        self.listings = listings
        self.contexts = []
        self.markets = []

    def load_listings(self, context, market: Market) -> list[ListingCandidate]:
        self.contexts.append(context)
        self.markets.append(market)
        return list(self.listings)


class FakeOhlcvProvider:
    provider_id = "yfinance"
    source_ids = ("yahoo-finance:yfinance-download",)

    def __init__(self, bars: pl.DataFrame) -> None:
        self.bars = bars
        self.contexts = []
        self.markets = []
        self.listings = []

    def load_ohlcv(self, context, market: Market, listings: list[ListingCandidate]) -> pl.DataFrame:
        self.contexts.append(context)
        self.markets.append(market)
        self.listings.append(list(listings))
        return self.bars


def _provider(
    *,
    config: AppConfig,
    listing_provider: FakeListingProvider,
    ohlcv_provider: FakeOhlcvProvider,
    now: datetime = datetime(2026, 5, 3, 13, 30, tzinfo=timezone.utc),
    clock=None,
) -> LiveMarketDataProvider:
    listing_registration = ListingProviderRegistration(
        provider_id=listing_provider.provider_id,
        supported_markets=frozenset({Market.US}),
        source_ids=listing_provider.source_ids,
        factory=lambda _config: listing_provider,
    )
    ohlcv_registration = OhlcvProviderRegistration(
        provider_id=ohlcv_provider.provider_id,
        supported_markets=frozenset({Market.US}),
        source_ids=ohlcv_provider.source_ids,
        factory=lambda _config: ohlcv_provider,
    )
    return LiveMarketDataProvider(
        config,
        listing_registration_resolver=lambda provider_id, market: listing_registration,
        ohlcv_registration_resolver=lambda provider_id: ohlcv_registration,
        clock=clock or (lambda: now),
    )


def test_live_provider_reuses_one_context_and_applies_sorted_ticker_limit() -> None:
    config = AppConfig(
        data_mode="live",
        live_ticker_limit=2,
    )
    listing_provider = FakeListingProvider([_listing("CCC"), _listing("AAA"), _listing("BBB")])
    ohlcv_provider = FakeOhlcvProvider(
        pl.DataFrame(
            {
                "market": ["US", "US"],
                "ticker": ["AAA", "BBB"],
                "bar_date": [date(2026, 5, 1), date(2026, 5, 2)],
                "open": [10.0, 11.0],
                "high": [10.0, 11.0],
                "low": [10.0, 11.0],
                "close": [10.0, 11.0],
                "adjusted_close": [10.0, 11.0],
                "volume": [1000, 1000],
            }
        )
    )
    clock_calls = 0

    def clock() -> datetime:
        nonlocal clock_calls
        clock_calls += 1
        return datetime(2026, 5, 3, 13, 30, tzinfo=timezone.utc)

    provider_data = _provider(
        config=config,
        listing_provider=listing_provider,
        ohlcv_provider=ohlcv_provider,
        clock=clock,
    ).load_run_data(Market.US)

    assert clock_calls == 1
    assert listing_provider.contexts == ohlcv_provider.contexts
    context = listing_provider.contexts[0]
    assert context.data_fetch_started_at == datetime(2026, 5, 3, 13, 30, tzinfo=timezone.utc)
    assert context.market_timezone == "America/New_York"
    assert context.market_fetch_date == date(2026, 5, 3)
    assert context.ticker_limit == 2
    assert [item.ticker for item in provider_data.listings] == ["AAA", "BBB"]
    assert [item.ticker for item in ohlcv_provider.listings[0]] == ["AAA", "BBB"]
    assert provider_data.metadata.run_latest_bar_date == date(2026, 5, 2)
    assert provider_data.metadata.data_fetch_started_at == context.data_fetch_started_at
    assert not hasattr(provider_data.metadata, "run_id")


def test_live_provider_drops_null_dates_before_latest_date_and_returned_bars() -> None:
    config = AppConfig(
        data_mode="live",
    )
    listing_provider = FakeListingProvider([_listing("AAA"), _listing("BBB")])
    ohlcv_provider = FakeOhlcvProvider(
        pl.DataFrame(
            {
                "market": ["US", "US", "US"],
                "ticker": ["AAA", "AAA", "BBB"],
                "bar_date": [date(2026, 5, 1), None, date(2026, 5, 3)],
                "open": [10.0, 10.5, 11.0],
                "high": [10.0, 10.5, 11.0],
                "low": [10.0, 10.5, 11.0],
                "close": [10.0, 10.5, 11.0],
                "adjusted_close": [10.0, 10.5, 11.0],
                "volume": [1000, 1000, 1000],
            }
        )
    )

    provider_data = _provider(
        config=config,
        listing_provider=listing_provider,
        ohlcv_provider=ohlcv_provider,
    ).load_run_data(Market.US)

    assert provider_data.metadata.run_latest_bar_date == date(2026, 5, 3)
    assert provider_data.bars["bar_date"].null_count() == 0
    assert provider_data.bars.height == 2


@pytest.mark.parametrize("bars", [pl.DataFrame(), pl.DataFrame({"bar_date": [None, None]})])
def test_live_provider_rejects_empty_or_all_null_date_bars(bars: pl.DataFrame) -> None:
    config = AppConfig(
        data_mode="live",
    )
    listing_provider = FakeListingProvider([_listing("AAA")])
    ohlcv_provider = FakeOhlcvProvider(bars)

    with pytest.raises(ProviderDataError, match="OHLCV provider returned no usable bars"):
        _provider(
            config=config,
            listing_provider=listing_provider,
            ohlcv_provider=ohlcv_provider,
        ).load_run_data(Market.US)


def test_live_provider_allows_partial_bars_and_builds_metadata_from_provider_config() -> None:
    config = AppConfig(
        data_mode="live",
    )
    listing_provider = FakeListingProvider([_listing("AAA"), _listing("BBB")])
    ohlcv_provider = FakeOhlcvProvider(
        pl.DataFrame(
            {
                "market": ["US"],
                "ticker": ["BBB"],
                "bar_date": [date(2026, 5, 3)],
                "open": [11.0],
                "high": [11.0],
                "low": [11.0],
                "close": [11.0],
                "adjusted_close": [11.0],
                "volume": [1000],
            }
        )
    )
    provider = _provider(
        config=config,
        listing_provider=listing_provider,
        ohlcv_provider=ohlcv_provider,
        now=datetime(2026, 5, 3, 17, 30, tzinfo=timezone.utc),
    )

    provider_data = provider.load_run_data(Market.US)

    assert [item.ticker for item in provider_data.listings] == ["AAA", "BBB"]
    assert provider_data.bars["ticker"].to_list() == ["BBB"]
    assert provider_data.metadata.data_mode == "live"
    assert provider_data.metadata.listing_provider_id == "nasdaq_trader"
    assert provider_data.metadata.listing_source_id == "nasdaqtrader:nasdaqlisted+nasdaqtrader:otherlisted"
    assert provider_data.metadata.ohlcv_provider_id == "yfinance"
    assert provider_data.metadata.ohlcv_source_id == "yahoo-finance:yfinance-download"
    assert provider_data.metadata.provider_config_hash == config.provider_config_hash()
    assert provider_data.metadata.market_timezone == "America/New_York"
    assert provider_data.metadata.run_latest_bar_date == date(2026, 5, 3)


def test_live_provider_applies_ticker_limit_after_sorted_tw_listings() -> None:
    config = AppConfig(
        data_mode="live",
        live_ticker_limit=2,
        live_listing_provider={Market.US: "nasdaq_trader", Market.TW: "twse_isin"},
    )
    listing_provider = FakeListingProvider(
        [
            _listing("6543", market=Market.TW, exchange_segment="TPEX"),
            _listing("2330", market=Market.TW, exchange_segment="TWSE"),
            _listing("1234", market=Market.TW, exchange_segment="TPEX"),
        ],
        provider_id="twse_isin",
        source_ids=("twse:isin:strMode=2", "twse:isin:strMode=4"),
    )
    ohlcv_provider = FakeOhlcvProvider(
        pl.DataFrame(
            {
                "market": ["TW", "TW"],
                "ticker": ["1234", "2330"],
                "bar_date": [date(2026, 5, 1), date(2026, 5, 2)],
                "open": [100.0, 600.0],
                "high": [101.0, 601.0],
                "low": [99.0, 599.0],
                "close": [100.0, 600.0],
                "adjusted_close": [100.0, 600.0],
                "volume": [1000, 1000],
            }
        )
    )
    provider = _provider(
        config=config,
        listing_provider=listing_provider,
        ohlcv_provider=ohlcv_provider,
    )

    provider_data = provider.load_run_data(Market.TW)

    assert [item.ticker for item in provider_data.listings] == ["1234", "2330"]
    assert [item.ticker for item in ohlcv_provider.listings[0]] == ["1234", "2330"]
    assert provider_data.metadata.listing_provider_id == "twse_isin"
    assert provider_data.metadata.listing_source_id == "twse:isin:strMode=2+twse:isin:strMode=4"
    assert provider_data.metadata.market_timezone == "Asia/Taipei"
    assert provider_data.metadata.run_latest_bar_date == date(2026, 5, 2)
