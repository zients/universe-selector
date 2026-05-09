from __future__ import annotations

import pytest

import universe_selector.providers as providers
from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.base import ListingProvider, OhlcvProvider
from universe_selector.providers.nasdaq_trader import (
    NASDAQ_TRADER_LISTING_REGISTRATION,
    NasdaqTraderListingProvider,
)
from universe_selector.providers.registration import (
    ListingProviderRegistration,
    OhlcvProviderRegistration,
    build_listing_provider_registration_map,
    build_ohlcv_provider_registration_map,
)
from universe_selector.providers.registry import (
    get_listing_registration,
    get_ohlcv_registration,
    supported_listing_provider_ids,
    supported_ohlcv_provider_ids,
)
from universe_selector.providers.twse_isin import (
    TWSE_ISIN_LISTING_REGISTRATION,
    TwseIsinListingProvider,
)
from universe_selector.providers.yfinance_ohlcv import YFINANCE_OHLCV_REGISTRATION, YFinanceOhlcvProvider


def test_provider_adapter_contracts_live_in_base_module() -> None:
    assert ListingProvider.__module__ == "universe_selector.providers.base"
    assert OhlcvProvider.__module__ == "universe_selector.providers.base"


def test_providers_package_root_exposes_registration_contracts() -> None:
    assert providers.ListingProviderRegistration is ListingProviderRegistration
    assert providers.OhlcvProviderRegistration is OhlcvProviderRegistration


def test_provider_modules_own_their_registrations() -> None:
    assert NASDAQ_TRADER_LISTING_REGISTRATION.provider_id == "nasdaq_trader"
    assert TWSE_ISIN_LISTING_REGISTRATION.provider_id == "twse_isin"
    assert YFINANCE_OHLCV_REGISTRATION.provider_id == "yfinance"
    assert isinstance(NASDAQ_TRADER_LISTING_REGISTRATION, ListingProviderRegistration)
    assert isinstance(TWSE_ISIN_LISTING_REGISTRATION, ListingProviderRegistration)
    assert isinstance(YFINANCE_OHLCV_REGISTRATION, OhlcvProviderRegistration)


def test_provider_registration_maps_are_immutable() -> None:
    listing_map = build_listing_provider_registration_map((NASDAQ_TRADER_LISTING_REGISTRATION,))
    ohlcv_map = build_ohlcv_provider_registration_map((YFINANCE_OHLCV_REGISTRATION,))

    with pytest.raises(TypeError):
        listing_map["probe"] = NASDAQ_TRADER_LISTING_REGISTRATION

    with pytest.raises(TypeError):
        ohlcv_map["probe"] = YFINANCE_OHLCV_REGISTRATION


def test_provider_registration_maps_reject_duplicate_provider_ids() -> None:
    with pytest.raises(ValueError, match="duplicate listing provider registration nasdaq_trader"):
        build_listing_provider_registration_map(
            (NASDAQ_TRADER_LISTING_REGISTRATION, NASDAQ_TRADER_LISTING_REGISTRATION)
        )

    with pytest.raises(ValueError, match="duplicate OHLCV provider registration yfinance"):
        build_ohlcv_provider_registration_map((YFINANCE_OHLCV_REGISTRATION, YFINANCE_OHLCV_REGISTRATION))


def test_provider_registrations_have_specific_types() -> None:
    assert isinstance(get_listing_registration("nasdaq_trader", Market.US), ListingProviderRegistration)
    assert isinstance(get_listing_registration("twse_isin", Market.TW), ListingProviderRegistration)
    assert isinstance(get_ohlcv_registration("yfinance"), OhlcvProviderRegistration)


def test_listing_provider_registry_exposes_supported_market_descriptors() -> None:
    us_registration = get_listing_registration("nasdaq_trader", Market.US)
    tw_registration = get_listing_registration("twse_isin", Market.TW)

    assert us_registration.provider_id == "nasdaq_trader"
    assert us_registration.supported_markets == frozenset({Market.US})
    assert us_registration.source_ids == (
        "nasdaqtrader:nasdaqlisted",
        "nasdaqtrader:otherlisted",
    )
    assert callable(us_registration.factory)

    assert tw_registration.provider_id == "twse_isin"
    assert tw_registration.supported_markets == frozenset({Market.TW})
    assert tw_registration.source_ids == (
        "twse:isin:strMode=2",
        "twse:isin:strMode=4",
    )
    assert callable(tw_registration.factory)


def test_ohlcv_provider_registry_exposes_yfinance_descriptor() -> None:
    registration = get_ohlcv_registration("yfinance")

    assert registration.provider_id == "yfinance"
    assert registration.supported_markets == frozenset({Market.US, Market.TW})
    assert registration.source_ids == ("yahoo-finance:yfinance-download",)
    assert callable(registration.factory)


def test_us_live_provider_registry_factories_return_concrete_adapters() -> None:
    config = AppConfig()

    assert isinstance(get_listing_registration("nasdaq_trader", Market.US).factory(config), NasdaqTraderListingProvider)
    assert isinstance(get_ohlcv_registration("yfinance").factory(config), YFinanceOhlcvProvider)


def test_tw_live_provider_registry_factory_returns_concrete_adapter() -> None:
    assert isinstance(get_listing_registration("twse_isin", Market.TW).factory(AppConfig()), TwseIsinListingProvider)


def test_yfinance_registry_factory_applies_yfinance_batch_config() -> None:
    registration = get_ohlcv_registration("yfinance")
    provider = registration.factory(AppConfig(live_yfinance_batch_size=123))

    assert provider.batch_size == 123


def test_supported_provider_ids_are_market_specific_and_sorted() -> None:
    assert supported_listing_provider_ids(Market.US) == ("nasdaq_trader",)
    assert supported_listing_provider_ids(Market.TW) == ("twse_isin",)
    assert supported_ohlcv_provider_ids() == ("yfinance",)


def test_unknown_listing_provider_id_reports_supported_ids() -> None:
    with pytest.raises(ValidationError, match="nasdaq_trader"):
        get_listing_registration("unknown", Market.US)

    with pytest.raises(ValidationError, match="twse_isin"):
        get_listing_registration("unknown", Market.TW)


def test_listing_provider_registered_for_other_market_reports_supported_ids() -> None:
    with pytest.raises(ValidationError, match="nasdaq_trader"):
        get_listing_registration("twse_isin", Market.US)

    with pytest.raises(ValidationError, match="twse_isin"):
        get_listing_registration("nasdaq_trader", Market.TW)


def test_unknown_ohlcv_provider_id_reports_supported_ids() -> None:
    with pytest.raises(ValidationError, match="yfinance"):
        get_ohlcv_registration("unknown")
