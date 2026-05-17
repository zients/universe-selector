from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from universe_selector.domain import Market, canonical_ticker
from universe_selector.errors import ProviderDataError
from universe_selector.providers.models import FundamentalFacts, FundamentalsMetadata, FundamentalsRunData
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
    "yfinance did not provide a usable quote timestamp; using fetch date, "
    "not a provider-reported quote timestamp."
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
        if market is not Market.US:
            raise ProviderDataError(f"unsupported fundamentals provider for {market.value}: {self.provider_id}")

        normalized_ticker = canonical_ticker(ticker)
        fetch_started_at = self._clock()
        payload = self._fetcher(normalized_ticker)
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


def _cash_flow_from_frame(frame: object, *, periods: int) -> dict[str, object] | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    operating_row = _row_for(frame, ("Operating Cash Flow",))
    capex_row = _row_for(frame, ("Capital Expenditure", "Capital Expenditures"))
    if operating_row is None or capex_row is None:
        return None

    columns = sorted((_date_from_value(column), column) for column in frame.columns if _date_from_value(column) is not None)
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

    columns = sorted((_date_from_value(column), column) for column in frame.columns if _date_from_value(column) is not None)
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
    supported_markets=frozenset({Market.US}),
    source_ids=YFinanceFundamentalsProvider.source_ids,
    factory=YFinanceFundamentalsProvider,
)
