from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_basic"


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_dir: Path) -> Path:
    db_path = tmp_path / "universe_selector.duckdb"
    lock_path = tmp_path / "batch.lock"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data_mode: fixture
duckdb_path: {db_path}
lock_path: {lock_path}
fixture_dir: {fixture_dir}
live:
  listing_provider:
    US: nasdaq_trader
    TW: twse_isin
  ohlcv_provider: yfinance
  ticker_limit: null
  yfinance:
    batch_size: 200
ranking:
  profile: sample_price_trend_v1
report:
  top_n: 100
""".lstrip()
    )
    monkeypatch.chdir(tmp_path)
    os.environ.pop("YF_API_KEY", None)
    return db_path
