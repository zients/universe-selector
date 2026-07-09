from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from universe_selector.domain import Market, canonical_ticker
from universe_selector.errors import ProviderDataError
from universe_selector.providers.base import MarketDataProvider
from universe_selector.providers.models import (
    FundamentalsCoverage,
    FundamentalsMetadata,
    FundamentalsUniverseRunData,
    ListingCandidate,
    ProviderDataRequirements,
    ProviderMetadata,
    ProviderRunData,
)


class FixtureProvider(MarketDataProvider):
    def __init__(self, fixture_dir: str | Path) -> None:
        self.fixture_dir = Path(fixture_dir)

    def load_run_data(self, market: Market, requirements: ProviderDataRequirements | None = None) -> ProviderRunData:
        requirements = requirements or ProviderDataRequirements()
        metadata = self._read_metadata(market)
        listings = self._read_listings(market)
        bars = self._read_ohlcv(
            market,
            [item.ticker for item in listings],
            metadata.run_latest_bar_date,
        )
        if requirements.fundamentals and not (self.fixture_dir / "fundamentals.csv").exists():
            raise ProviderDataError("fixture fundamentals are required but unavailable")
        fundamentals = None
        if requirements.fundamentals:
            fundamentals = self._read_fundamentals(market, [item.ticker for item in listings], metadata)
            metadata = replace(
                metadata,
                fundamentals_provider_id=fundamentals.metadata.fundamentals_provider_id,
                fundamentals_source_id="+".join(fundamentals.metadata.fundamentals_source_ids),
                fundamentals_latest_source_date=fundamentals.metadata.latest_source_date,
                fundamentals_source_risk_note=fundamentals.metadata.source_risk_note,
                fundamentals_field_mapping_note=fundamentals.metadata.field_mapping_note,
                fundamentals_requested_count=fundamentals.coverage.requested_count,
                fundamentals_returned_count=fundamentals.coverage.returned_count,
                fundamentals_missing_count=fundamentals.coverage.missing_count,
                fundamentals_invalid_count=fundamentals.coverage.invalid_count,
            )
        return ProviderRunData(metadata=metadata, listings=listings, bars=bars, fundamentals=fundamentals)

    def _read_metadata(self, market: Market) -> ProviderMetadata:
        del market
        payload = json.loads((self.fixture_dir / "metadata.json").read_text())
        return ProviderMetadata(
            data_mode=payload["data_mode"],
            listing_provider_id=payload["listing_provider_id"],
            listing_source_id=payload["listing_source_id"],
            ohlcv_provider_id=payload["ohlcv_provider_id"],
            ohlcv_source_id=payload["ohlcv_source_id"],
            provider_config_hash=payload["provider_config_hash"],
            data_fetch_started_at=datetime.now(timezone.utc),
            market_timezone=payload["market_timezone"],
            run_latest_bar_date=date.fromisoformat(payload["run_latest_bar_date"]),
        )

    def _read_listings(self, market: Market) -> list[ListingCandidate]:
        listings = pl.read_csv(
            self.fixture_dir / "listings.csv",
            schema_overrides={"market": pl.Utf8, "ticker": pl.Utf8, "listing_symbol": pl.Utf8},
        )
        rows = listings.filter(pl.col("market") == market.value).sort("ticker").to_dicts()
        return [
            ListingCandidate(
                market=market,
                ticker=canonical_ticker(str(row["ticker"])),
                listing_symbol=str(row["listing_symbol"]).strip(),
                exchange_segment=str(row["exchange_segment"]),
                listing_status=str(row["listing_status"]),
                instrument_type=str(row["instrument_type"]),
                source_id=str(row["source_id"]),
            )
            for row in rows
        ]

    def _read_ohlcv(self, market: Market, tickers: list[str], run_latest_bar_date: date) -> pl.DataFrame:
        requested_tickers = [canonical_ticker(ticker) for ticker in tickers]
        bars = self._read_ohlcv_csv()
        required_columns = {
            "market",
            "ticker",
            "bar_date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        }
        missing_columns = required_columns - set(bars.columns)
        if missing_columns:
            raise ProviderDataError(
                f"OHLCV fixture data missing required columns: {', '.join(sorted(missing_columns))}"
            )
        removed_columns = {"tradable_close", "price_basis"} & set(bars.columns)
        if removed_columns:
            raise ProviderDataError(f"OHLCV fixture data uses removed columns: {', '.join(sorted(removed_columns))}")
        filtered = bars.filter((pl.col("market") == market.value) & (pl.col("ticker").is_in(requested_tickers)))

        duplicates = filtered.group_by(["market", "ticker", "bar_date"]).len().filter(pl.col("len") > 1)
        if duplicates.height > 0:
            raise ProviderDataError("duplicate OHLCV bars in fixture data")

        if filtered.filter(pl.col("bar_date") > run_latest_bar_date).height > 0:
            raise ProviderDataError("OHLCV fixture data contains rows after run_latest_bar_date")
        filtered = filtered.filter(pl.col("bar_date") <= run_latest_bar_date)

        return filtered.sort(["ticker", "bar_date"])

    def _read_ohlcv_csv(self) -> pl.DataFrame:
        return pl.read_csv(
            self.fixture_dir / "ohlcv.csv",
            try_parse_dates=True,
            schema_overrides={
                "market": pl.Utf8,
                "ticker": pl.Utf8,
            },
        )

    def _read_fundamentals(
        self,
        market: Market,
        tickers: list[str],
        metadata: ProviderMetadata,
    ) -> FundamentalsUniverseRunData:
        facts = pl.read_csv(
            self.fixture_dir / "fundamentals.csv",
            try_parse_dates=True,
            schema_overrides={"market": pl.Utf8, "ticker": pl.Utf8, "currency": pl.Utf8},
        )
        required_columns = {
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
        missing_columns = required_columns - set(facts.columns)
        if missing_columns:
            raise ProviderDataError(
                f"fundamentals fixture data missing required columns: {', '.join(sorted(missing_columns))}"
            )

        requested_tickers = [canonical_ticker(ticker) for ticker in tickers]
        filtered = facts.filter((pl.col("market") == market.value) & (pl.col("ticker").is_in(requested_tickers))).sort(
            "ticker"
        )
        latest_source_date = metadata.run_latest_bar_date
        if not filtered.is_empty():
            dates = filtered.select(
                pl.max_horizontal(pl.col("fiscal_period_end"), pl.col("balance_sheet_as_of")).alias("source_date")
            )
            source_date = dates["source_date"].max()
            if isinstance(source_date, date):
                latest_source_date = source_date
        return FundamentalsUniverseRunData(
            metadata=FundamentalsMetadata(
                data_mode="fixture",
                fundamentals_provider_id="fixture-fundamentals-v1",
                fundamentals_source_ids=("sample_basic/fundamentals.csv",),
                data_fetch_started_at=metadata.data_fetch_started_at,
                latest_source_date=latest_source_date,
                source_risk_note="Fixture fundamentals are deterministic sample data for offline tests.",
                field_mapping_note="Fixture fundamentals already use the normalized fundamentals universe schema.",
            ),
            facts=filtered,
            coverage=FundamentalsCoverage(
                requested_count=len(requested_tickers),
                returned_count=filtered.height,
                missing_count=len(requested_tickers) - filtered.height,
                invalid_count=0,
            ),
        )
