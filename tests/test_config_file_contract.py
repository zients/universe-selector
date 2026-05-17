from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import yaml

from universe_selector.config import AppConfig, load_config
from universe_selector.domain import Market


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_EXAMPLE_PATH = REPO_ROOT / "config.example.yaml"
EXPECTED_CONFIG_EXAMPLE: dict[str, object] = {
    "data_mode": "live",
    "duckdb_path": ".universe-selector/universe_selector.duckdb",
    "lock_path": ".universe-selector/batch.lock",
    "fixture_dir": "tests/fixtures/sample_basic",
    "live": {
        "listing_provider": {
            "US": "nasdaq_trader",
            "TW": "twse_isin",
        },
        "ohlcv_provider": "yfinance",
        "fundamentals_provider": "yfinance_fundamentals",
        "ticker_limit": None,
        "yfinance": {
            "batch_size": 200,
        },
    },
    "ranking": {"profile": "sample_price_trend_v1"},
    "report": {"top_n": 100},
}
def test_config_example_loads_and_matches_app_config_defaults(monkeypatch, tmp_path) -> None:
    assert CONFIG_EXAMPLE_PATH.is_file()
    assert yaml.safe_load(CONFIG_EXAMPLE_PATH.read_text()) == EXPECTED_CONFIG_EXAMPLE

    (tmp_path / "config.yaml").write_bytes(CONFIG_EXAMPLE_PATH.read_bytes())
    monkeypatch.chdir(tmp_path)

    actual = load_config()
    expected = AppConfig(ranking_profile="sample_price_trend_v1")

    assert actual.data_mode == expected.data_mode
    assert actual.duckdb_path == expected.duckdb_path
    assert actual.lock_path == expected.lock_path
    assert actual.fixture_dir == expected.fixture_dir
    assert actual.ranking_profile == expected.ranking_profile
    assert actual.report_top_n == expected.report_top_n
    assert actual.live_listing_provider[Market.US] == expected.live_listing_provider[Market.US]
    assert actual.live_listing_provider[Market.TW] == expected.live_listing_provider[Market.TW]
    assert actual.live_ohlcv_provider == expected.live_ohlcv_provider
    assert actual.live_fundamentals_provider == expected.live_fundamentals_provider
    assert actual.live_ticker_limit == expected.live_ticker_limit
    assert actual.live_yfinance_batch_size == expected.live_yfinance_batch_size
    assert actual.ranking_config_hash() == expected.ranking_config_hash()
    assert actual.provider_config_hash() == expected.provider_config_hash()


def test_config_yaml_files_are_ignored_at_root_and_nested_paths() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "config.yaml", "nested/config.yaml"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["config.yaml", "nested/config.yaml"]


def test_config_example_is_packaged_in_built_wheel(tmp_path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_path = next(tmp_path.glob("*.whl"))

    with zipfile.ZipFile(wheel_path) as wheel:
        packaged_example = wheel.read("universe_selector/config.example.yaml")

    assert packaged_example == CONFIG_EXAMPLE_PATH.read_bytes()
