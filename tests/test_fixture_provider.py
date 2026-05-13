from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.providers.fixture import FixtureProvider


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_basic"


def test_fixture_provider_loads_metadata_listings_and_bars(fixture_dir: Path) -> None:
    provider = FixtureProvider(fixture_dir)

    us_data = provider.load_run_data(Market.US)
    tw_data = provider.load_run_data(Market.TW)

    assert us_data.metadata.run_latest_bar_date == date(2026, 4, 24)
    assert us_data.metadata.market_timezone == "UTC"
    assert us_data.metadata.data_mode == "fixture"
    assert not hasattr(us_data.metadata, "run_id")
    assert [item.ticker for item in us_data.listings] == ["AAA", "BBB", "CCC", "LOWVOL", "SHORT"]
    assert [item.listing_symbol for item in us_data.listings] == ["AAA", "BBB", "CCC", "LOWVOL", "SHORT"]
    assert [item.ticker for item in us_data.listings] == sorted(item.ticker for item in us_data.listings)
    assert [item.ticker for item in tw_data.listings] == ["2317", "2330", "2454", "TLOWVOL", "TSHORT"]
    assert [item.listing_symbol for item in tw_data.listings] == ["2317", "2330", "2454", "TLOWVOL", "TSHORT"]
    assert "adjusted_close" in us_data.bars.columns
    assert "adjusted_close" in tw_data.bars.columns
    assert "tradable_close" not in us_data.bars.columns
    assert "price_basis" not in us_data.bars.columns
    assert us_data.bars.filter(us_data.bars["ticker"] == "AAA").height == 274
    assert tw_data.bars.filter(tw_data.bars["ticker"] == "2330").height == 274


def test_fixture_provider_rejects_duplicate_bars(tmp_path: Path, fixture_dir: Path) -> None:
    temp_fixture_dir = tmp_path / "sample_basic"
    shutil.copytree(fixture_dir, temp_fixture_dir)

    ohlcv_path = temp_fixture_dir / "ohlcv.csv"
    first_data_line = ohlcv_path.read_text().splitlines()[1]
    with ohlcv_path.open("a") as file:
        file.write(f"{first_data_line}\n")

    provider = FixtureProvider(temp_fixture_dir)

    with pytest.raises(ProviderDataError):
        provider.load_run_data(Market.US)


def test_fixture_provider_rejects_rows_after_run_latest_bar_date(tmp_path: Path, fixture_dir: Path) -> None:
    temp_fixture_dir = tmp_path / "sample_basic"
    shutil.copytree(fixture_dir, temp_fixture_dir)

    ohlcv_path = temp_fixture_dir / "ohlcv.csv"
    ohlcv_path.write_text(
        ohlcv_path.read_text()
        + "US,AAA,2026-04-27,10.0,10.0,10.0,10.0,10.0,2000000\n"
    )

    provider = FixtureProvider(temp_fixture_dir)

    with pytest.raises(ProviderDataError, match="run_latest_bar_date"):
        provider.load_run_data(Market.US)


def test_fixture_provider_rejects_old_price_schema(tmp_path: Path, fixture_dir: Path) -> None:
    temp_fixture_dir = tmp_path / "sample_basic"
    shutil.copytree(fixture_dir, temp_fixture_dir)

    ohlcv_path = temp_fixture_dir / "ohlcv.csv"
    ohlcv_path.write_text(
        "market,ticker,bar_date,open,high,low,close,tradable_close,volume,price_basis\n"
        "US,AAA,2026-04-24,10.0,10.0,10.0,10.0,10.0,2000000,provider_adjusted_close\n"
    )

    provider = FixtureProvider(temp_fixture_dir)

    with pytest.raises(ProviderDataError, match="adjusted_close"):
        provider.load_run_data(Market.US)
