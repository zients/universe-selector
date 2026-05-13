from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from universe_selector.domain import Market, canonical_ticker
from universe_selector.errors import ProviderDataError
from universe_selector.providers.base import MarketDataProvider
from universe_selector.providers.models import ListingCandidate, ProviderMetadata, ProviderRunData


class FixtureProvider(MarketDataProvider):
    def __init__(self, fixture_dir: str | Path) -> None:
        self.fixture_dir = Path(fixture_dir)

    def load_run_data(self, market: Market) -> ProviderRunData:
        metadata = self._read_metadata(market)
        listings = self._read_listings(market)
        bars = self._read_ohlcv(
            market,
            [item.ticker for item in listings],
            metadata.run_latest_bar_date,
        )
        return ProviderRunData(metadata=metadata, listings=listings, bars=bars)

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
            raise ProviderDataError(f"OHLCV fixture data missing required columns: {', '.join(sorted(missing_columns))}")
        removed_columns = {"tradable_close", "price_basis"} & set(bars.columns)
        if removed_columns:
            raise ProviderDataError(f"OHLCV fixture data uses removed columns: {', '.join(sorted(removed_columns))}")
        filtered = bars.filter((pl.col("market") == market.value) & (pl.col("ticker").is_in(requested_tickers)))

        duplicates = (
            filtered.group_by(["market", "ticker", "bar_date"])
            .len()
            .filter(pl.col("len") > 1)
        )
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
