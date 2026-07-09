from __future__ import annotations

import math
import numbers
import re
from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import polars as pl

from universe_selector.domain import Market, canonical_ticker
from universe_selector.errors import ProviderDataError
from universe_selector.providers.context import ProviderRunContext
from universe_selector.providers.models import (
    FundamentalFacts,
    FundamentalsCoverage,
    FundamentalsMetadata,
    FundamentalsRunData,
    FundamentalsUniverseRunData,
    ListingCandidate,
)
from universe_selector.providers.registration import FundamentalsProviderRegistration


YFinanceFundamentalsFetcher = Callable[[str], Mapping[str, object]]
_SOURCE_IDS = ("yahoo-finance:yfinance-ticker",)
_REQUIRED_KEYS = (
    "currency",
    "reference_price",
    "reference_price_as_of",
    "reference_price_as_of_source",
    "reference_price_as_of_note",
    "shares_outstanding",
    "cash_and_cash_equivalents",
    "total_debt",
    "balance_sheet_as_of",
    "operating_cash_flow",
    "capital_expenditures",
    "fiscal_period_end",
    "fiscal_period_type",
)
_FETCH_DATE_FALLBACK_NOTE = (
    "yfinance did not provide a usable quote timestamp; using fetch date, not a provider-reported quote timestamp."
)
_SOURCE_RISK_NOTE = (
    "yfinance third-party convenience data may be stale, incomplete, restated, "
    "mapped inconsistently, or unavailable. Facts should be independently verified "
    "before research use."
)
_FIELD_MAPPING_NOTE = (
    "reference price from currentPrice/regularMarketPrice; shares from sharesOutstanding; "
    "raw free cash flow from Operating Cash Flow minus Capital Expenditure; cash and debt "
    "from quarterly balance sheet fields."
)
_UNIVERSE_FIELD_MAPPING_NOTE = (
    "profitability facts from yfinance income statement, cash flow, and quarterly balance sheet; "
    "TTM uses latest four quarterly periods when available and falls back to latest annual flow fields."
)
_UNIVERSE_FACT_SCHEMA = {
    "market": pl.String,
    "ticker": pl.String,
    "currency": pl.String,
    "fiscal_period_end": pl.Date,
    "balance_sheet_as_of": pl.Date,
    "fiscal_period_type": pl.String,
    "revenue_ttm": pl.Float64,
    "gross_profit_ttm": pl.Float64,
    "operating_income_ttm": pl.Float64,
    "net_income_ttm": pl.Float64,
    "total_assets": pl.Float64,
    "shareholders_equity": pl.Float64,
    "total_debt": pl.Float64,
    "cash_and_cash_equivalents": pl.Float64,
    "operating_cash_flow_ttm": pl.Float64,
    "capital_expenditures_ttm": pl.Float64,
    "free_cash_flow_ttm": pl.Float64,
    "roe": pl.Float64,
    "roa": pl.Float64,
    "gross_margin": pl.Float64,
    "operating_margin": pl.Float64,
    "net_margin": pl.Float64,
    "debt_to_equity": pl.Float64,
    "fcf_margin": pl.Float64,
    "tag_fundamentals_annual_fallback": pl.Float64,
    "tag_negative_net_income": pl.Float64,
    "tag_negative_fcf": pl.Float64,
}


class _MissingFundamentalsError(Exception):
    pass


class YFinanceFundamentalsProvider:
    provider_id = "yfinance_fundamentals"
    source_ids = _SOURCE_IDS

    def __init__(
        self,
        fetcher: YFinanceFundamentalsFetcher | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._fetcher = fetcher or self._default_fetcher

    def load_fundamentals(self, market: Market, ticker: str) -> FundamentalsRunData:
        normalized_ticker = canonical_ticker(ticker)
        request_symbol = _request_symbol(market, normalized_ticker)
        fetch_started_at = self._clock()
        payload = self._fetcher(request_symbol)
        facts = self._normalize_payload(market, normalized_ticker, payload)
        latest_source_date = max(facts.fiscal_period_end, facts.reference_price_as_of, facts.balance_sheet_as_of)

        return FundamentalsRunData(
            metadata=FundamentalsMetadata(
                data_mode="live",
                fundamentals_provider_id=self.provider_id,
                fundamentals_source_ids=self.source_ids,
                data_fetch_started_at=fetch_started_at,
                latest_source_date=latest_source_date,
                source_risk_note=_SOURCE_RISK_NOTE,
                field_mapping_note=_FIELD_MAPPING_NOTE,
            ),
            facts=facts,
        )

    def load_fundamentals_for_listings(
        self,
        context: ProviderRunContext,
        market: Market,
        listings: list[ListingCandidate],
    ) -> FundamentalsUniverseRunData:
        import yfinance as yf

        rows: list[dict[str, object]] = []
        missing_count = 0
        invalid_count = 0
        requested_tickers = [item.ticker for item in listings if item.market is market]
        for ticker in requested_tickers:
            normalized_ticker = canonical_ticker(ticker)
            try:
                ticker_obj = yf.Ticker(_request_symbol(market, normalized_ticker))
                rows.append(self._normalize_universe_ticker(market, normalized_ticker, ticker_obj))
            except _MissingFundamentalsError:
                missing_count += 1
            except ProviderDataError:
                invalid_count += 1

        rows.sort(key=lambda row: str(row["ticker"]))
        facts = pl.DataFrame(rows, schema=_UNIVERSE_FACT_SCHEMA) if rows else pl.DataFrame(schema=_UNIVERSE_FACT_SCHEMA)
        latest_source_date = context.market_fetch_date
        if rows:
            latest_source_date = max(
                max(row["fiscal_period_end"], row["balance_sheet_as_of"])  # type: ignore[type-var]
                for row in rows
            )
        return FundamentalsUniverseRunData(
            metadata=FundamentalsMetadata(
                data_mode="live",
                fundamentals_provider_id=self.provider_id,
                fundamentals_source_ids=self.source_ids,
                data_fetch_started_at=context.data_fetch_started_at,
                latest_source_date=latest_source_date,
                source_risk_note=_SOURCE_RISK_NOTE,
                field_mapping_note=_UNIVERSE_FIELD_MAPPING_NOTE,
            ),
            facts=facts,
            coverage=FundamentalsCoverage(
                requested_count=len(requested_tickers),
                returned_count=len(rows),
                missing_count=missing_count,
                invalid_count=invalid_count,
            ),
        )

    def _default_fetcher(self, symbol: str) -> Mapping[str, object]:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = _mapping_or_empty(getattr(ticker, "info", {}))
        currency = info.get("currency")
        financial_currency = info.get("financialCurrency")
        if financial_currency is not None and currency != financial_currency:
            raise ProviderDataError(
                f"currency must match financialCurrency for yfinance fundamentals: {currency} != {financial_currency}"
            )

        cash_flow = ticker.get_cash_flow(freq="quarterly", pretty=True)
        period_type = "ttm"
        cash_flow_values = _cash_flow_from_frame(cash_flow, periods=4)
        if cash_flow_values is None:
            cash_flow = ticker.get_cash_flow(freq="yearly", pretty=True)
            period_type = "annual"
            cash_flow_values = _cash_flow_from_frame(cash_flow, periods=1)
        if cash_flow_values is None:
            raise ProviderDataError("yfinance returned no usable fundamentals")
        balance_sheet_values = _balance_sheet_from_frame(ticker.get_balance_sheet(freq="quarterly", pretty=True))
        if balance_sheet_values is None:
            raise ProviderDataError("yfinance returned no usable balance sheet facts")

        reference_price_as_of = _reference_price_as_of(info, self._clock)
        if reference_price_as_of is None:
            reference_price_as_of = self._clock().date()
            reference_price_as_of_source = "fetch_date_fallback"
            reference_price_as_of_note: str | None = _FETCH_DATE_FALLBACK_NOTE
        else:
            reference_price_as_of_source = "provider_reported"
            reference_price_as_of_note = None

        return {
            "currency": currency,
            "reference_price": _first_present(info, ("currentPrice", "regularMarketPrice")),
            "reference_price_as_of": reference_price_as_of,
            "reference_price_as_of_source": reference_price_as_of_source,
            "reference_price_as_of_note": reference_price_as_of_note,
            "shares_outstanding": info.get("sharesOutstanding"),
            "cash_and_cash_equivalents": balance_sheet_values["cash_and_cash_equivalents"],
            "total_debt": balance_sheet_values["total_debt"],
            "balance_sheet_as_of": balance_sheet_values["balance_sheet_as_of"],
            "operating_cash_flow": cash_flow_values["operating_cash_flow"],
            "capital_expenditures": cash_flow_values["capital_expenditures"],
            "fiscal_period_end": cash_flow_values["fiscal_period_end"],
            "fiscal_period_type": period_type,
        }

    def _normalize_payload(
        self,
        market: Market,
        ticker: str,
        payload: Mapping[str, object],
    ) -> FundamentalFacts:
        for key in _REQUIRED_KEYS:
            if key not in payload:
                raise ProviderDataError(f"yfinance fundamentals missing required field: {key}")

        currency = payload["currency"]
        if not isinstance(currency, str) or not currency.strip():
            raise ProviderDataError("currency must be a non-empty string")

        reference_price = _required_float(payload["reference_price"], "reference_price")
        if reference_price <= 0:
            raise ProviderDataError("reference_price must be greater than zero")
        reference_price_as_of = _required_date(payload["reference_price_as_of"], "reference_price_as_of")
        reference_price_as_of_source = payload["reference_price_as_of_source"]
        if reference_price_as_of_source not in {"provider_reported", "fetch_date_fallback"}:
            raise ProviderDataError("reference_price_as_of_source must be provider_reported or fetch_date_fallback")
        reference_price_as_of_note = payload["reference_price_as_of_note"]
        if reference_price_as_of_note is not None and not isinstance(reference_price_as_of_note, str):
            raise ProviderDataError("reference_price_as_of_note must be a string or null")

        shares_outstanding = _required_float(payload["shares_outstanding"], "shares_outstanding")
        if shares_outstanding <= 0:
            raise ProviderDataError("shares_outstanding must be greater than zero")
        cash = _required_float(payload["cash_and_cash_equivalents"], "cash_and_cash_equivalents")
        debt = _required_float(payload["total_debt"], "total_debt")
        balance_sheet_as_of = _required_date(payload["balance_sheet_as_of"], "balance_sheet_as_of")
        operating_cash_flow = _required_float(payload["operating_cash_flow"], "operating_cash_flow")
        capex = abs(_required_float(payload["capital_expenditures"], "capital_expenditures"))
        fiscal_period_end = _required_date(payload["fiscal_period_end"], "fiscal_period_end")
        fiscal_period_type = payload["fiscal_period_type"]
        if fiscal_period_type not in {"ttm", "annual"}:
            raise ProviderDataError("fiscal_period_type must be ttm or annual")

        return FundamentalFacts(
            market=market,
            ticker=ticker,
            currency=currency,
            reference_price=reference_price,
            reference_price_as_of=reference_price_as_of,
            reference_price_as_of_source=reference_price_as_of_source,
            reference_price_as_of_note=reference_price_as_of_note,
            shares_outstanding=shares_outstanding,
            cash_and_cash_equivalents=cash,
            total_debt=debt,
            balance_sheet_as_of=balance_sheet_as_of,
            net_debt=debt - cash,
            operating_cash_flow=operating_cash_flow,
            capital_expenditures=capex,
            free_cash_flow=operating_cash_flow - capex,
            fiscal_period_end=fiscal_period_end,
            fiscal_period_type=fiscal_period_type,
        )

    def _normalize_universe_ticker(self, market: Market, ticker: str, ticker_obj: object) -> dict[str, object]:
        info = _mapping_or_empty(getattr(ticker_obj, "info", {}))
        quote_currency = info.get("currency")
        financial_currency = info.get("financialCurrency")
        quote_currency_text = quote_currency.strip() if isinstance(quote_currency, str) else None
        financial_currency_text = financial_currency.strip() if isinstance(financial_currency, str) else None
        if quote_currency_text and financial_currency_text and quote_currency_text != financial_currency_text:
            raise ProviderDataError(
                "currency must match financialCurrency for yfinance fundamentals: "
                f"{quote_currency_text} != {financial_currency_text}"
            )
        currency = financial_currency_text or quote_currency_text
        if currency is None:
            raise ProviderDataError("currency must be a non-empty string")

        income_statement = ticker_obj.get_income_stmt(freq="quarterly", pretty=True)
        cash_flow = ticker_obj.get_cash_flow(freq="quarterly", pretty=True)
        flow_values = _universe_flow_values_from_frames(income_statement, cash_flow, periods=4)
        fiscal_period_type = "ttm"
        if flow_values is None:
            income_statement = ticker_obj.get_income_stmt(freq="yearly", pretty=True)
            cash_flow = ticker_obj.get_cash_flow(freq="yearly", pretty=True)
            flow_values = _universe_flow_values_from_frames(income_statement, cash_flow, periods=1)
            fiscal_period_type = "annual"
        if flow_values is None:
            raise _MissingFundamentalsError

        balance_sheet_values = _universe_balance_sheet_from_frame(
            ticker_obj.get_balance_sheet(freq="quarterly", pretty=True)
        )
        if balance_sheet_values is None:
            raise _MissingFundamentalsError

        revenue = flow_values["revenue_ttm"]
        total_assets = balance_sheet_values["total_assets"]
        shareholders_equity = balance_sheet_values["shareholders_equity"]
        if revenue <= 0:
            raise ProviderDataError("revenue_ttm must be greater than zero")
        if total_assets <= 0:
            raise ProviderDataError("total_assets must be greater than zero")
        if shareholders_equity <= 0:
            raise ProviderDataError("shareholders_equity must be greater than zero")
        if balance_sheet_values["total_debt"] < 0:
            raise ProviderDataError("total_debt must be non-negative")
        if balance_sheet_values["cash_and_cash_equivalents"] < 0:
            raise ProviderDataError("cash_and_cash_equivalents must be non-negative")

        operating_cash_flow = flow_values["operating_cash_flow_ttm"]
        capital_expenditures = abs(flow_values["capital_expenditures_ttm"])
        free_cash_flow = operating_cash_flow - capital_expenditures
        net_income = flow_values["net_income_ttm"]
        values: dict[str, float] = {
            "revenue_ttm": float(flow_values["revenue_ttm"]),
            "gross_profit_ttm": float(flow_values["gross_profit_ttm"]),
            "operating_income_ttm": float(flow_values["operating_income_ttm"]),
            "net_income_ttm": float(flow_values["net_income_ttm"]),
            "total_assets": float(balance_sheet_values["total_assets"]),
            "shareholders_equity": float(balance_sheet_values["shareholders_equity"]),
            "total_debt": float(balance_sheet_values["total_debt"]),
            "cash_and_cash_equivalents": float(balance_sheet_values["cash_and_cash_equivalents"]),
            "operating_cash_flow_ttm": float(flow_values["operating_cash_flow_ttm"]),
            "capital_expenditures_ttm": capital_expenditures,
            "free_cash_flow_ttm": free_cash_flow,
            "roe": net_income / shareholders_equity,
            "roa": net_income / total_assets,
            "gross_margin": flow_values["gross_profit_ttm"] / revenue,
            "operating_margin": flow_values["operating_income_ttm"] / revenue,
            "net_margin": net_income / revenue,
            "debt_to_equity": balance_sheet_values["total_debt"] / shareholders_equity,
            "fcf_margin": free_cash_flow / revenue,
        }
        for field, value in values.items():
            if not math.isfinite(float(value)):
                raise ProviderDataError(f"{field} must be finite")

        return {
            "market": market.value,
            "ticker": ticker,
            "currency": currency,
            "fiscal_period_end": flow_values["fiscal_period_end"],
            "balance_sheet_as_of": balance_sheet_values["balance_sheet_as_of"],
            "fiscal_period_type": fiscal_period_type,
            "revenue_ttm": values["revenue_ttm"],
            "gross_profit_ttm": values["gross_profit_ttm"],
            "operating_income_ttm": values["operating_income_ttm"],
            "net_income_ttm": values["net_income_ttm"],
            "total_assets": values["total_assets"],
            "shareholders_equity": values["shareholders_equity"],
            "total_debt": values["total_debt"],
            "cash_and_cash_equivalents": values["cash_and_cash_equivalents"],
            "operating_cash_flow_ttm": values["operating_cash_flow_ttm"],
            "capital_expenditures_ttm": capital_expenditures,
            "free_cash_flow_ttm": free_cash_flow,
            "roe": values["roe"],
            "roa": values["roa"],
            "gross_margin": values["gross_margin"],
            "operating_margin": values["operating_margin"],
            "net_margin": values["net_margin"],
            "debt_to_equity": values["debt_to_equity"],
            "fcf_margin": values["fcf_margin"],
            "tag_fundamentals_annual_fallback": 1.0 if fiscal_period_type == "annual" else 0.0,
            "tag_negative_net_income": 1.0 if net_income < 0 else 0.0,
            "tag_negative_fcf": 1.0 if free_cash_flow < 0 else 0.0,
        }


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _first_present(payload: Mapping[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _request_symbol(market: Market, ticker: str) -> str:
    if market is Market.US:
        if re.fullmatch(r"[A-Z0-9]+\.[A-Z0-9]+", ticker):
            return ticker.replace(".", "-")
        return ticker
    if market is Market.TW:
        if ticker.endswith((".TW", ".TWO")):
            return ticker
        return f"{ticker}.TW"
    raise ProviderDataError(
        f"unsupported fundamentals provider for {market.value}: {YFinanceFundamentalsProvider.provider_id}"
    )


def _cash_flow_from_frame(frame: object, *, periods: int) -> dict[str, object] | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    operating_row = _row_for(frame, ("Operating Cash Flow",))
    capex_row = _row_for(frame, ("Capital Expenditure", "Capital Expenditures"))
    if operating_row is None or capex_row is None:
        return None

    columns = sorted(
        (_date_from_value(column), column) for column in frame.columns if _date_from_value(column) is not None
    )
    if len(columns) < periods:
        return None
    selected = columns[-periods:]
    operating_cash_flow = 0.0
    capital_expenditures = 0.0
    for _, column in selected:
        operating_value = _float_or_none(operating_row[column])
        capex_value = _float_or_none(capex_row[column])
        if operating_value is None or capex_value is None:
            return None
        operating_cash_flow += operating_value
        capital_expenditures += capex_value
    return {
        "operating_cash_flow": operating_cash_flow,
        "capital_expenditures": capital_expenditures,
        "fiscal_period_end": selected[-1][0],
    }


def _balance_sheet_from_frame(frame: object) -> dict[str, object] | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    cash_row = _row_for(frame, ("Cash And Cash Equivalents",))
    debt_row = _row_for(frame, ("Total Debt",))
    if cash_row is None or debt_row is None:
        return None

    columns = sorted(
        (_date_from_value(column), column) for column in frame.columns if _date_from_value(column) is not None
    )
    if not columns:
        return None
    balance_sheet_as_of, column = columns[-1]
    cash = _float_or_none(cash_row[column])
    debt = _float_or_none(debt_row[column])
    if cash is None or debt is None:
        return None
    return {
        "cash_and_cash_equivalents": cash,
        "total_debt": debt,
        "balance_sheet_as_of": balance_sheet_as_of,
    }


def _universe_flow_values_from_frames(
    income_statement: object,
    cash_flow: object,
    *,
    periods: int,
) -> dict[str, float | date] | None:
    if not isinstance(income_statement, pd.DataFrame) or income_statement.empty:
        return None
    if not isinstance(cash_flow, pd.DataFrame) or cash_flow.empty:
        return None
    rows = {
        "revenue_ttm": _row_for(income_statement, ("Total Revenue",)),
        "gross_profit_ttm": _row_for(income_statement, ("Gross Profit",)),
        "operating_income_ttm": _row_for(income_statement, ("Operating Income", "Operating Income Loss")),
        "net_income_ttm": _row_for(
            income_statement,
            (
                "Net Income",
                "Net Income Common Stockholders",
                "Net Income From Continuing Operation Net Minority Interest",
            ),
        ),
        "operating_cash_flow_ttm": _row_for(cash_flow, ("Operating Cash Flow",)),
        "capital_expenditures_ttm": _row_for(cash_flow, ("Capital Expenditure", "Capital Expenditures")),
    }
    if any(row is None for row in rows.values()):
        return None

    columns = sorted(
        (_date_from_value(column), column)
        for column in income_statement.columns
        if _date_from_value(column) is not None and column in cash_flow.columns
    )
    if len(columns) < periods:
        return None
    selected = columns[-periods:]
    result: dict[str, float | date] = {key: 0.0 for key in rows}
    for _, column in selected:
        for key, row in rows.items():
            assert row is not None
            value = _float_or_none(row[column])
            if value is None:
                raise ProviderDataError(f"{key} must be finite")
            result[key] = float(result[key]) + value
    result["fiscal_period_end"] = selected[-1][0]
    return result


def _universe_balance_sheet_from_frame(frame: object) -> dict[str, float | date] | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    rows = {
        "total_assets": _row_for(frame, ("Total Assets",)),
        "shareholders_equity": _row_for(
            frame,
            ("Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"),
        ),
        "total_debt": _row_for(frame, ("Total Debt",)),
        "cash_and_cash_equivalents": _row_for(
            frame,
            ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
        ),
    }
    if any(row is None for row in rows.values()):
        return None
    columns = sorted(
        (_date_from_value(column), column) for column in frame.columns if _date_from_value(column) is not None
    )
    if not columns:
        return None
    balance_sheet_as_of, column = columns[-1]
    result: dict[str, float | date] = {"balance_sheet_as_of": balance_sheet_as_of}
    for key, row in rows.items():
        assert row is not None
        value = _float_or_none(row[column])
        if value is None:
            raise ProviderDataError(f"{key} must be finite")
        result[key] = value
    return result


def _row_for(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series[Any] | None:
    for name in names:
        if name in frame.index:
            return frame.loc[name]
    return None


def _reference_price_as_of(info: Mapping[str, object], clock: Callable[[], datetime]) -> date | None:
    del clock
    raw = info.get("regularMarketTime")
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except TypeError:
        pass
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, int | float) and math.isfinite(float(raw)):
        return datetime.fromtimestamp(float(raw), timezone.utc).date()
    return None


def _required_float(value: object, field: str) -> float:
    number = _float_or_none(value)
    if number is None:
        raise ProviderDataError(f"{field} must be a finite number")
    return number


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if not isinstance(value, numbers.Number):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _required_date(value: object, field: str) -> date:
    parsed = _date_from_value(value)
    if parsed is None:
        raise ProviderDataError(f"{field} must be an ISO date")
    return parsed


def _date_from_value(value: object) -> date | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


YFINANCE_FUNDAMENTALS_REGISTRATION = FundamentalsProviderRegistration(
    provider_id=YFinanceFundamentalsProvider.provider_id,
    supported_markets=frozenset({Market.US, Market.TW}),
    source_ids=YFinanceFundamentalsProvider.source_ids,
    factory=YFinanceFundamentalsProvider,
)
