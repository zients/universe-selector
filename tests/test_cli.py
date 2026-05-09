from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from universe_selector.cli import app


RUN_ID_RE = re.compile(r"run_id: (?P<run_id>(?:tw|us)-[0-9a-f-]+)")
runner = CliRunner()


def _write_cli_config(tmp_path: Path, fixture_dir: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        f"""
data_mode: fixture
duckdb_path: {tmp_path / "runs.duckdb"}
lock_path: {tmp_path / "batch.lock"}
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
  top_n: 2
""".lstrip()
    )


def _run_id_from(output: str) -> str:
    match = RUN_ID_RE.search(output)
    assert match is not None
    return match.group("run_id")


def test_cli_batch_report_and_inspect_use_sample_profile(monkeypatch, tmp_path: Path, fixture_dir: Path) -> None:
    _write_cli_config(tmp_path, fixture_dir)
    monkeypatch.chdir(tmp_path)

    batch = runner.invoke(app, ["batch", "us"])
    assert batch.exit_code == 0, batch.output
    run_id = _run_id_from(batch.output)

    report = runner.invoke(app, ["report", "us"])
    assert report.exit_code == 0, report.output
    assert f"run_id: {run_id}" in report.output
    assert "# Universe Selector Report" in report.output
    assert "ranking_profile: sample_price_trend_v1" in report.output

    explicit_report = runner.invoke(app, ["report", "--run-id", run_id])
    assert explicit_report.exit_code == 0, explicit_report.output
    assert "resolution mode: explicit run_id" in explicit_report.output

    inspect = runner.invoke(app, ["inspect", "us", "--ticker", "AAA"])
    assert inspect.exit_code == 0, inspect.output
    assert "# Universe Selector Inspect" in inspect.output
    assert "- normalized ticker: AAA" in inspect.output
    assert "- return_60d:" in inspect.output
    assert "- return_120d:" in inspect.output


def test_cli_reports_config_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["report", "us"])

    assert result.exit_code != 0
    assert "config file not found: config.yaml" in result.output
