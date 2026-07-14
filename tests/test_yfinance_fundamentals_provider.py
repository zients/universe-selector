from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.providers.context import build_provider_run_context
from universe_selector.providers.models import ListingCandidate
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


def _install_fake_yfinance(monkeypatch: pytest.MonkeyPatch, ticker_cls: type[object]) -> None:
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=ticker_cls))


def _listing(ticker: str) -> ListingCandidate:
    return ListingCandidate(
        market=Market.US,
        ticker=ticker,
        listing_symbol=ticker,
        exchange_segment="NASDAQ",
        listing_status="active",
        instrument_type="common_stock",
        source_id=f"unit:{ticker}",
    )


def _tw_listing(ticker: str, exchange_segment: str) -> ListingCandidate:
    return ListingCandidate(
        market=Market.TW,
        ticker=ticker,
        listing_symbol=ticker,
        exchange_segment=exchange_segment,
        listing_status="active",
        instrument_type="common_stock",
        source_id=f"unit:{ticker}",
    )


def _context():
    return build_provider_run_context(
        market=Market.US,
        data_fetch_started_at=datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
        ticker_limit=None,
    )


def _quarterly_frame(row_values: dict[str, tuple[float, float, float, float]]) -> pd.DataFrame:
    columns = [
        pd.Timestamp("2025-06-30"),
        pd.Timestamp("2025-09-30"),
        pd.Timestamp("2025-12-31"),
        pd.Timestamp("2026-03-31"),
    ]
    return pd.DataFrame(
        {column: {row: values[index] for row, values in row_values.items()} for index, column in enumerate(columns)}
    )


def _annual_frame(row_values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({pd.Timestamp("2025-12-31"): row_values})


def _balance_sheet_frame(
    *,
    assets: float = 200.0,
    equity: float = 100.0,
    debt: float = 25.0,
    cash: float = 10.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            pd.Timestamp("2026-03-31"): {
                "Total Assets": assets,
                "Stockholders Equity": equity,
                "Total Debt": debt,
                "Cash And Cash Equivalents": cash,
            }
        }
    )


def _quarterly_income_frame(
    *,
    revenue: float = 100.0,
    gross_profit: float = 60.0,
    operating_income: float = 30.0,
    net_income: float = 20.0,
) -> pd.DataFrame:
    return _quarterly_frame(
        {
            "Total Revenue": (revenue / 4.0,) * 4,
            "Gross Profit": (gross_profit / 4.0,) * 4,
            "Operating Income": (operating_income / 4.0,) * 4,
            "Net Income": (net_income / 4.0,) * 4,
        }
    )


def _quarterly_cash_flow_frame(
    *,
    operating_cash_flow: float = 24.0,
    capital_expenditure: float = -4.0,
) -> pd.DataFrame:
    return _quarterly_frame(
        {
            "Operating Cash Flow": (operating_cash_flow / 4.0,) * 4,
            "Capital Expenditure": (capital_expenditure / 4.0,) * 4,
        }
    )


def _annual_income_frame(
    *,
    revenue: float = 100.0,
    gross_profit: float = 50.0,
    operating_income: float = -5.0,
    net_income: float = -10.0,
) -> pd.DataFrame:
    return _annual_frame(
        {
            "Total Revenue": revenue,
            "Gross Profit": gross_profit,
            "Operating Income": operating_income,
            "Net Income": net_income,
        }
    )


def _annual_cash_flow_frame(
    *,
    operating_cash_flow: float = 2.0,
    capital_expenditure: float = -5.0,
) -> pd.DataFrame:
    return _annual_frame(
        {
            "Operating Cash Flow": operating_cash_flow,
            "Capital Expenditure": capital_expenditure,
        }
    )


def test_yfinance_fundamentals_uses_injected_fetcher_and_normalizes_contract() -> None:
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
    assert "yfinance" not in sys.modules


def test_yfinance_fundamentals_maps_us_class_share_to_yfinance_symbol() -> None:
    requested = []

    def fetcher(symbol: str) -> dict[str, object]:
        requested.append(symbol)
        return valid_payload()

    provider = YFinanceFundamentalsProvider(fetcher=fetcher)
    data = provider.load_fundamentals(Market.US, "BRK.B")

    assert requested == ["BRK-B"]
    assert data.facts.ticker == "BRK.B"


@pytest.mark.parametrize(
    ("ticker", "exchange_segment", "expected_symbol"),
    [
        ("2330", "TWSE", "2330.TW"),
        ("6488", "TPEX", "6488.TWO"),
    ],
)
def test_yfinance_single_fundamentals_maps_resolved_tw_listing(
    ticker: str,
    exchange_segment: str,
    expected_symbol: str,
) -> None:
    requested = []

    def fetcher(symbol: str) -> dict[str, object]:
        requested.append(symbol)
        payload = valid_payload()
        payload["currency"] = "TWD"
        return payload

    provider = YFinanceFundamentalsProvider(fetcher=fetcher)
    data = provider.load_fundamentals(
        Market.TW,
        ticker,
        listing=_tw_listing(ticker, exchange_segment),
    )

    assert requested == [expected_symbol]
    assert data.facts.market is Market.TW
    assert data.facts.ticker == ticker
    assert data.facts.currency == "TWD"


def test_yfinance_single_tw_fundamentals_requires_matching_listing() -> None:
    def fetcher(symbol: str) -> dict[str, object]:
        raise AssertionError(f"fetcher must not be invoked: {symbol}")

    provider = YFinanceFundamentalsProvider(fetcher=fetcher)

    with pytest.raises(ProviderDataError, match="requires resolved listing identity"):
        provider.load_fundamentals(Market.TW, "2330")

    with pytest.raises(ProviderDataError, match="does not match requested ticker"):
        provider.load_fundamentals(
            Market.TW,
            "2330",
            listing=_tw_listing("6488", "TPEX"),
        )

    with pytest.raises(ProviderDataError, match="unsupported TW exchange segment"):
        provider.load_fundamentals(
            Market.TW,
            "2330",
            listing=_tw_listing("2330", "EMERGING"),
        )


def test_yfinance_universe_fundamentals_maps_tpex_to_two_and_preserves_canonical_tickers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = []

    class FakeTicker:
        info = {"financialCurrency": "TWD"}

        def __init__(self, symbol: str) -> None:
            requested.append(symbol)

        def get_income_stmt(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if freq == "quarterly":
                return _quarterly_income_frame()
            raise AssertionError(f"unexpected freq {freq}")

        def get_cash_flow(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if freq == "quarterly":
                return _quarterly_cash_flow_frame()
            raise AssertionError(f"unexpected freq {freq}")

        def get_balance_sheet(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            assert freq == "quarterly"
            return _balance_sheet_frame()

    _install_fake_yfinance(monkeypatch, FakeTicker)
    context = build_provider_run_context(
        market=Market.TW,
        data_fetch_started_at=datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
        ticker_limit=None,
    )
    listings = [
        ListingCandidate(
            market=Market.TW,
            ticker="2330",
            listing_symbol="2330",
            exchange_segment="TWSE",
            listing_status="active",
            instrument_type="common_stock",
            source_id="unit:2330",
        ),
        ListingCandidate(
            market=Market.TW,
            ticker="1240",
            listing_symbol="1240",
            exchange_segment="TPEX",
            listing_status="active",
            instrument_type="common_stock",
            source_id="unit:1240",
        ),
    ]
    provider = YFinanceFundamentalsProvider(clock=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc))

    data = provider.load_fundamentals_for_listings(context, Market.TW, listings)

    assert requested == ["2330.TW", "1240.TWO"]
    assert data.facts["ticker"].to_list() == ["1240", "2330"]
    assert data.coverage.requested_count == 2
    assert data.coverage.returned_count == 2
    assert data.coverage.missing_count == 0
    assert data.coverage.invalid_count == 0


def test_yfinance_universe_fundamentals_maps_us_class_share_to_yfinance_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = []

    class FakeTicker:
        info = {"financialCurrency": "USD"}

        def __init__(self, symbol: str) -> None:
            requested.append(symbol)

        def get_income_stmt(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if freq == "quarterly":
                return _quarterly_income_frame()
            raise AssertionError(f"unexpected freq {freq}")

        def get_cash_flow(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if freq == "quarterly":
                return _quarterly_cash_flow_frame()
            raise AssertionError(f"unexpected freq {freq}")

        def get_balance_sheet(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            assert freq == "quarterly"
            return _balance_sheet_frame()

    _install_fake_yfinance(monkeypatch, FakeTicker)
    provider = YFinanceFundamentalsProvider(clock=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc))

    data = provider.load_fundamentals_for_listings(_context(), Market.US, [_listing("BRK.B")])

    assert requested == ["BRK-B"]
    assert data.facts["ticker"].to_list() == ["BRK.B"]


def test_yfinance_universe_fundamentals_normalizes_quarterly_ttm(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = []

    class FakeTicker:
        info = {"financialCurrency": "USD"}

        def __init__(self, symbol: str) -> None:
            requested.append(symbol)

        def get_income_stmt(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if freq == "quarterly":
                return _quarterly_income_frame()
            raise AssertionError(f"unexpected freq {freq}")

        def get_cash_flow(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if freq == "quarterly":
                return _quarterly_cash_flow_frame()
            raise AssertionError(f"unexpected freq {freq}")

        def get_balance_sheet(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            assert freq == "quarterly"
            return _balance_sheet_frame()

    _install_fake_yfinance(monkeypatch, FakeTicker)
    provider = YFinanceFundamentalsProvider(clock=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc))

    data = provider.load_fundamentals_for_listings(_context(), Market.US, [_listing("AAA")])

    expected_columns = {
        "market",
        "ticker",
        "currency",
        "fiscal_period_end",
        "balance_sheet_as_of",
        "fiscal_period_type",
        "revenue_ttm",
        "gross_profit_ttm",
        "operating_income_ttm",
        "net_income_ttm",
        "total_assets",
        "shareholders_equity",
        "total_debt",
        "cash_and_cash_equivalents",
        "operating_cash_flow_ttm",
        "capital_expenditures_ttm",
        "free_cash_flow_ttm",
        "roe",
        "roa",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "debt_to_equity",
        "fcf_margin",
        "tag_fundamentals_annual_fallback",
        "tag_negative_net_income",
        "tag_negative_fcf",
    }
    row = data.facts.to_dicts()[0]

    assert requested == ["AAA"]
    assert set(data.facts.columns) == expected_columns
    assert data.coverage.requested_count == 1
    assert data.coverage.returned_count == 1
    assert data.coverage.missing_count == 0
    assert data.coverage.invalid_count == 0
    assert row["market"] == "US"
    assert row["ticker"] == "AAA"
    assert row["currency"] == "USD"
    assert row["fiscal_period_type"] == "ttm"
    assert row["revenue_ttm"] == pytest.approx(100.0)
    assert row["free_cash_flow_ttm"] == pytest.approx(20.0)
    assert row["roe"] == pytest.approx(0.20)
    assert row["roa"] == pytest.approx(0.10)
    assert row["gross_margin"] == pytest.approx(0.60)
    assert row["operating_margin"] == pytest.approx(0.30)
    assert row["net_margin"] == pytest.approx(0.20)
    assert row["debt_to_equity"] == pytest.approx(0.25)
    assert row["fcf_margin"] == pytest.approx(0.20)
    assert row["tag_fundamentals_annual_fallback"] == 0.0
    assert row["tag_negative_net_income"] == 0.0
    assert row["tag_negative_fcf"] == 0.0
    assert data.metadata.fundamentals_provider_id == "yfinance_fundamentals"
    assert data.metadata.latest_source_date == date(2026, 3, 31)


def test_yfinance_universe_fundamentals_tracks_mixed_coverage_and_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self.info = {"currency": "USD", "financialCurrency": "USD"}
            if symbol == "EEE":
                self.info = {"currency": "USD", "financialCurrency": "TWD"}

        def get_income_stmt(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if self.symbol == "AAA" and freq == "quarterly":
                return _quarterly_income_frame()
            if self.symbol == "BBB":
                if freq == "quarterly":
                    return pd.DataFrame()
                if freq == "yearly":
                    return _annual_income_frame()
            if self.symbol == "DDD" and freq == "quarterly":
                return _quarterly_income_frame(revenue=0.0)
            if self.symbol == "EEE" and freq == "quarterly":
                return _quarterly_income_frame()
            return pd.DataFrame()

        def get_cash_flow(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if self.symbol == "AAA" and freq == "quarterly":
                return _quarterly_cash_flow_frame()
            if self.symbol == "BBB":
                if freq == "quarterly":
                    return pd.DataFrame()
                if freq == "yearly":
                    return _annual_cash_flow_frame()
            if self.symbol in {"DDD", "EEE"} and freq == "quarterly":
                return _quarterly_cash_flow_frame()
            return pd.DataFrame()

        def get_balance_sheet(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            assert freq == "quarterly"
            if self.symbol == "CCC":
                return pd.DataFrame()
            return _balance_sheet_frame()

    _install_fake_yfinance(monkeypatch, FakeTicker)
    provider = YFinanceFundamentalsProvider(clock=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc))

    data = provider.load_fundamentals_for_listings(
        _context(),
        Market.US,
        [_listing("AAA"), _listing("BBB"), _listing("CCC"), _listing("DDD"), _listing("EEE")],
    )

    rows = {row["ticker"]: row for row in data.facts.to_dicts()}

    assert data.coverage.requested_count == 5
    assert data.coverage.returned_count == 2
    assert data.coverage.missing_count == 1
    assert data.coverage.invalid_count == 2
    assert data.facts["ticker"].to_list() == ["AAA", "BBB"]
    assert rows["BBB"]["fiscal_period_type"] == "annual"
    assert rows["BBB"]["tag_fundamentals_annual_fallback"] == 1.0
    assert rows["BBB"]["tag_negative_net_income"] == 1.0
    assert rows["BBB"]["tag_negative_fcf"] == 1.0
    assert rows["BBB"]["net_margin"] == pytest.approx(-0.10)
    assert rows["BBB"]["fcf_margin"] == pytest.approx(-0.03)


def test_yfinance_universe_fundamentals_rejects_negative_debt_and_cash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self.info = {"currency": "USD", "financialCurrency": "USD"}

        def get_income_stmt(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if freq == "quarterly":
                return _quarterly_income_frame()
            return pd.DataFrame()

        def get_cash_flow(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            if freq == "quarterly":
                return _quarterly_cash_flow_frame()
            return pd.DataFrame()

        def get_balance_sheet(self, *, freq: str = "yearly", pretty: bool = False, **kwargs) -> pd.DataFrame:
            del kwargs
            assert pretty is True
            assert freq == "quarterly"
            if self.symbol == "NEGDEBT":
                return _balance_sheet_frame(debt=-1.0)
            return _balance_sheet_frame(cash=-1.0)

    _install_fake_yfinance(monkeypatch, FakeTicker)
    provider = YFinanceFundamentalsProvider(clock=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc))

    data = provider.load_fundamentals_for_listings(
        _context(),
        Market.US,
        [_listing("NEGDEBT"), _listing("NEGCASH")],
    )

    assert data.facts.is_empty()
    assert data.coverage.requested_count == 2
    assert data.coverage.returned_count == 0
    assert data.coverage.missing_count == 0
    assert data.coverage.invalid_count == 2


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

    _install_fake_yfinance(monkeypatch, FakeTicker)
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

    _install_fake_yfinance(monkeypatch, FakeTicker)
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
        "yfinance did not provide a usable quote timestamp; using fetch date, not a provider-reported quote timestamp."
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

    _install_fake_yfinance(monkeypatch, FakeTicker)
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

    _install_fake_yfinance(monkeypatch, FakeTicker)
    provider = YFinanceFundamentalsProvider(clock=lambda: datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc))
    data = provider.load_fundamentals(Market.US, "AAPL")

    assert data.facts.reference_price_as_of == date(2026, 5, 17)
    assert data.facts.reference_price_as_of_source == "fetch_date_fallback"
