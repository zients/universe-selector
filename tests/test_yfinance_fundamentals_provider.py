from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.providers.yfinance_fundamentals import YFinanceFundamentalsProvider


def valid_payload() -> dict[str, object]:
    return {
        "currency": "USD",
        "reference_price": 190.0,
        "reference_price_as_of": "2026-05-15",
        "reference_price_as_of_source": "provider_reported",
        "reference_price_as_of_note": None,
        "shares_outstanding": 10_000_000_000.0,
        "cash_and_cash_equivalents": 60_000_000_000.0,
        "total_debt": 110_000_000_000.0,
        "balance_sheet_as_of": "2026-03-31",
        "operating_cash_flow": 120_000_000_000.0,
        "capital_expenditures": -10_000_000_000.0,
        "fiscal_period_end": "2025-09-30",
        "fiscal_period_type": "ttm",
    }


def test_yfinance_fundamentals_uses_injected_fetcher_and_normalizes_contract(monkeypatch) -> None:
    def fail_ticker(*args, **kwargs):
        raise AssertionError("yfinance.Ticker must not be called when fetcher is injected")

    monkeypatch.setattr("yfinance.Ticker", fail_ticker)
    requested = []

    def fetcher(symbol: str) -> dict[str, object]:
        requested.append(symbol)
        return valid_payload()

    provider = YFinanceFundamentalsProvider(
        fetcher=fetcher,
        clock=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
    )
    data = provider.load_fundamentals(Market.US, "AAPL")

    assert requested == ["AAPL"]
    assert data.metadata.data_mode == "live"
    assert data.metadata.fundamentals_provider_id == "yfinance_fundamentals"
    assert data.metadata.fundamentals_source_ids == ("yahoo-finance:yfinance-ticker",)
    assert data.metadata.latest_source_date == date(2026, 5, 15)
    assert "yfinance third-party convenience data" in data.metadata.source_risk_note
    assert "currentPrice/regularMarketPrice" in data.metadata.field_mapping_note
    assert data.facts.market is Market.US
    assert data.facts.ticker == "AAPL"
    assert data.facts.reference_price_as_of_source == "provider_reported"
    assert data.facts.reference_price_as_of_note is None
    assert data.facts.capital_expenditures == 10_000_000_000.0
    assert data.facts.free_cash_flow == 110_000_000_000.0
    assert data.facts.net_debt == 50_000_000_000.0
    assert data.facts.balance_sheet_as_of == date(2026, 3, 31)
    assert data.facts.fiscal_period_type == "ttm"


def test_yfinance_fundamentals_rejects_unsupported_market_before_fetch() -> None:
    def fetcher(symbol: str) -> dict[str, object]:
        raise AssertionError(f"fetcher should not be called for {symbol}")

    provider = YFinanceFundamentalsProvider(fetcher=fetcher)

    with pytest.raises(ProviderDataError, match="unsupported fundamentals provider for TW"):
        provider.load_fundamentals(Market.TW, "2330")


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"reference_price": float("nan")}, "reference_price"),
        ({"reference_price_as_of": "bad-date"}, "reference_price_as_of"),
        ({"reference_price_as_of_source": "unknown"}, "reference_price_as_of_source"),
        ({"fiscal_period_type": "quarterly"}, "fiscal_period_type"),
    ],
)
def test_yfinance_fundamentals_rejects_invalid_payload_fields(patch: dict[str, object], message: str) -> None:
    payload = valid_payload()
    payload.update(patch)
    provider = YFinanceFundamentalsProvider(fetcher=lambda _symbol: payload)

    with pytest.raises(ProviderDataError, match=message):
        provider.load_fundamentals(Market.US, "AAPL")


def test_yfinance_fundamentals_rejects_missing_required_field() -> None:
    payload = valid_payload()
    del payload["shares_outstanding"]
    provider = YFinanceFundamentalsProvider(fetcher=lambda _symbol: payload)

    with pytest.raises(ProviderDataError, match="shares_outstanding"):
        provider.load_fundamentals(Market.US, "AAPL")


def test_default_yfinance_adapter_derives_ttm_from_quarterly_cash_flow(monkeypatch) -> None:
    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            assert symbol == "AAPL"
            self.info = {
                "currency": "USD",
                "financialCurrency": "USD",
                "currentPrice": 190.0,
                "sharesOutstanding": 10_000_000_000.0,
                "totalCash": 68_000_000_000.0,
                "regularMarketTime": 1778841600,
            }

        def get_cash_flow(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if freq == "quarterly":
                return pd.DataFrame(
                    {
                        pd.Timestamp("2026-03-31"): {
                            "Operating Cash Flow": 31_000_000_000.0,
                            "Capital Expenditure": -2_500_000_000.0,
                        },
                        pd.Timestamp("2025-12-31"): {
                            "Operating Cash Flow": 30_000_000_000.0,
                            "Capital Expenditure": -2_500_000_000.0,
                        },
                        pd.Timestamp("2025-09-30"): {
                            "Operating Cash Flow": 29_000_000_000.0,
                            "Capital Expenditure": -2_500_000_000.0,
                        },
                        pd.Timestamp("2025-06-30"): {
                            "Operating Cash Flow": 28_000_000_000.0,
                            "Capital Expenditure": -2_500_000_000.0,
                        },
                    }
                )
            raise AssertionError(f"unexpected freq {freq}")

        def get_balance_sheet(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert freq == "quarterly"
            assert pretty is True
            return pd.DataFrame(
                {
                    pd.Timestamp("2026-03-31"): {
                        "Cash And Cash Equivalents": 45_000_000_000,
                        "Total Debt": 110_000_000_000,
                    }
                }
            )

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    provider = YFinanceFundamentalsProvider(clock=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc))

    data = provider.load_fundamentals(Market.US, "AAPL")

    assert data.facts.operating_cash_flow == 118_000_000_000.0
    assert data.facts.capital_expenditures == 10_000_000_000.0
    assert data.facts.free_cash_flow == 108_000_000_000.0
    assert data.facts.cash_and_cash_equivalents == 45_000_000_000.0
    assert data.facts.balance_sheet_as_of == date(2026, 3, 31)
    assert data.facts.fiscal_period_type == "ttm"
    assert data.facts.fiscal_period_end == date(2026, 3, 31)
    assert data.facts.reference_price_as_of == date(2026, 5, 15)
    assert data.facts.reference_price_as_of_source == "provider_reported"
    assert data.metadata.latest_source_date == date(2026, 5, 15)


def test_default_yfinance_adapter_falls_back_to_yearly_and_fetch_date_price_timestamp(monkeypatch) -> None:
    calls = []

    class FakeTicker:
        info = {
            "currency": "USD",
            "financialCurrency": "USD",
            "regularMarketPrice": 190.0,
            "sharesOutstanding": 10_000_000_000.0,
        }

        def __init__(self, symbol: str) -> None:
            del symbol

        def get_cash_flow(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            calls.append(freq)
            if freq == "quarterly":
                return pd.DataFrame()
            if freq == "yearly":
                return pd.DataFrame(
                    {
                        pd.Timestamp("2025-09-30"): {
                            "Operating Cash Flow": 120_000_000_000.0,
                            "Capital Expenditures": -10_000_000_000.0,
                        }
                    }
                )
            raise AssertionError(f"unexpected freq {freq}")

        def get_balance_sheet(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert freq == "quarterly"
            assert pretty is True
            return pd.DataFrame(
                {
                    pd.Timestamp("2025-09-30"): {
                        "Cash And Cash Equivalents": Decimal("60000000000"),
                        "Total Debt": Decimal("110000000000"),
                    }
                }
            )

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    provider = YFinanceFundamentalsProvider(clock=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc))

    data = provider.load_fundamentals(Market.US, "AAPL")

    assert calls == ["quarterly", "yearly"]
    assert data.facts.fiscal_period_type == "annual"
    assert data.facts.cash_and_cash_equivalents == 60_000_000_000.0
    assert data.facts.net_debt == 50_000_000_000.0
    assert data.facts.balance_sheet_as_of == date(2025, 9, 30)
    assert data.facts.reference_price_as_of == date(2026, 5, 17)
    assert data.facts.reference_price_as_of_source == "fetch_date_fallback"
    assert data.facts.reference_price_as_of_note == (
        "yfinance did not provide a usable quote timestamp; using fetch date, "
        "not a provider-reported quote timestamp."
    )
    assert data.metadata.latest_source_date == date(2026, 5, 17)


def test_default_yfinance_adapter_rejects_quote_and_financial_currency_mismatch(monkeypatch) -> None:
    class FakeTicker:
        info = {
            "currency": "USD",
            "financialCurrency": "TWD",
            "regularMarketPrice": 190.0,
            "sharesOutstanding": 10_000_000_000.0,
        }

        def __init__(self, symbol: str) -> None:
            del symbol

        def get_cash_flow(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del freq, pretty, kwargs
            return pd.DataFrame()

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    provider = YFinanceFundamentalsProvider()

    with pytest.raises(ProviderDataError, match="financialCurrency"):
        provider.load_fundamentals(Market.US, "AAPL")


def test_yfinance_fundamentals_rejects_nat_dates_and_defaults_nat_quote_timestamp(monkeypatch) -> None:
    payload = valid_payload()
    payload["fiscal_period_end"] = pd.NaT
    provider = YFinanceFundamentalsProvider(fetcher=lambda _symbol: payload)

    with pytest.raises(ProviderDataError, match="fiscal_period_end"):
        provider.load_fundamentals(Market.US, "AAPL")

    class FakeTicker:
        info = {
            "currency": "USD",
            "financialCurrency": "USD",
            "regularMarketPrice": 190.0,
            "sharesOutstanding": 10_000_000_000.0,
            "regularMarketTime": pd.NaT,
        }

        def __init__(self, symbol: str) -> None:
            del symbol

        def get_cash_flow(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if freq == "quarterly":
                return pd.DataFrame()
            return pd.DataFrame(
                {
                    pd.Timestamp("2025-09-30"): {
                        "Operating Cash Flow": 120_000_000_000.0,
                        "Capital Expenditures": -10_000_000_000.0,
                    }
                }
            )

        def get_balance_sheet(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert freq == "quarterly"
            assert pretty is True
            return pd.DataFrame(
                {
                    pd.Timestamp("2025-09-30"): {
                        "Cash And Cash Equivalents": 60_000_000_000.0,
                        "Total Debt": 110_000_000_000.0,
                    }
                }
            )

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    provider = YFinanceFundamentalsProvider(clock=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc))
    data = provider.load_fundamentals(Market.US, "AAPL")

    assert data.facts.reference_price_as_of == date(2026, 5, 17)
    assert data.facts.reference_price_as_of_source == "fetch_date_fallback"
