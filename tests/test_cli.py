from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from universe_selector.cli import app
from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.persistence.repository import DuckDbRepository
from universe_selector.persistence.schema import apply_migrations
from universe_selector.pipeline import BatchResult


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


def _seed_profile_resolution_runs(tmp_path: Path) -> tuple[str, str]:
    db_path = tmp_path / "runs.duckdb"
    repo = DuckDbRepository(str(db_path))
    connection = repo.connect()
    apply_migrations(connection)

    sample_run_id = "us-00000000-0000-4000-8000-000000000001"
    other_run_id = "us-00000000-0000-4000-8000-000000000002"
    connection.execute(
        """
        insert into run_log(run_id, market, status, created_at, ranking_profile, ranking_config_hash, error_message)
        values
          (?, 'US', 'successful', timestamp '2026-01-01 00:00:00', 'sample_price_trend_v1', 'sample-hash', null),
          (?, 'US', 'successful', timestamp '2026-01-02 00:00:00', 'other_profile', 'other-hash', null)
        """,
        [sample_run_id, other_run_id],
    )
    connection.execute(
        """
        insert into report_artifacts(run_id, format, content)
        values
          (?, 'markdown', ?),
          (?, 'markdown', ?)
        """,
        [
            sample_run_id,
            "# sample report\nranking_profile: sample_price_trend_v1\n",
            other_run_id,
            "# other report\nranking_profile: other_profile\n",
        ],
    )
    connection.execute(
        """
        insert into run_provider_metadata(
            run_id, data_mode, listing_provider_id, listing_source_id, ohlcv_provider_id,
            ohlcv_source_id, provider_config_hash, data_fetch_started_at, market_timezone,
            run_latest_bar_date
        )
        values (?, 'fixture', 'fixture-listings-v1', 'sample_basic/listings.csv', 'fixture-ohlcv-v1',
                'sample_basic/ohlcv.csv', 'fixture-sample-basic', timestamp '2026-01-01 00:00:00',
                'UTC', date '2026-01-01')
        """,
        [sample_run_id],
    )
    connection.execute(
        """
        insert into run_ticker_snapshot(run_id, market, ticker, close, adjusted_close, metrics_json)
        values (?, 'US', 'AAA', 10.0, 10.0,
                '{"avg_traded_value_20d_local":1000.0,"return_60d":0.1,"return_120d":0.2}')
        """,
        [sample_run_id],
    )
    connection.execute(
        """
        insert into run_rankings(run_id, market, horizon, ticker, score, rank, metrics_json)
        values
          (?, 'US', 'midterm', 'AAA', 0.1, 1, '{"score_return_60d":0.1,"score_return_120d":0.2}'),
          (?, 'US', 'longterm', 'AAA', 0.2, 1, '{"score_return_60d":0.1,"score_return_120d":0.2}')
        """,
        [sample_run_id, sample_run_id],
    )
    connection.close()
    return sample_run_id, other_run_id


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


def test_cli_batch_ranking_profile_overrides_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "universe_selector.cli.load_config",
        lambda: AppConfig(ranking_profile="other_profile"),
    )

    def fake_run_batch(market: Market, config: AppConfig) -> BatchResult:
        captured["market"] = market
        captured["ranking_profile"] = config.ranking_profile
        return BatchResult(run_id="us-00000000-0000-4000-8000-000000000003", market=market)

    monkeypatch.setattr("universe_selector.cli.run_batch", fake_run_batch)

    result = runner.invoke(app, ["batch", "us", "--ranking-profile", "sample_price_trend_v1"])

    assert result.exit_code == 0, result.output
    assert captured == {"market": Market.US, "ranking_profile": "sample_price_trend_v1"}


def test_cli_report_and_inspect_ranking_profile_filter_latest_run(monkeypatch, tmp_path: Path) -> None:
    sample_run_id, other_run_id = _seed_profile_resolution_runs(tmp_path)
    monkeypatch.setattr(
        "universe_selector.cli.load_config",
        lambda: AppConfig(duckdb_path=str(tmp_path / "runs.duckdb"), ranking_profile="other_profile"),
    )

    report = runner.invoke(app, ["report", "us", "--ranking-profile", "sample_price_trend_v1"])
    assert report.exit_code == 0, report.output
    assert f"run_id: {sample_run_id}" in report.output
    assert other_run_id not in report.output
    assert "# sample report" in report.output
    assert "# other report" not in report.output

    inspect = runner.invoke(app, ["inspect", "us", "--ticker", "AAA", "--ranking-profile", "sample_price_trend_v1"])
    assert inspect.exit_code == 0, inspect.output
    assert f"run_id: {sample_run_id}" in inspect.output
    assert other_run_id not in inspect.output
    assert "- normalized ticker: AAA" in inspect.output


def test_cli_rejects_ranking_profile_with_explicit_run_id(monkeypatch, tmp_path: Path, fixture_dir: Path) -> None:
    _write_cli_config(tmp_path, fixture_dir)
    monkeypatch.chdir(tmp_path)

    batch = runner.invoke(app, ["batch", "us"])
    assert batch.exit_code == 0, batch.output
    run_id = _run_id_from(batch.output)

    valid_report = runner.invoke(app, ["report", "--run-id", run_id, "--ranking-profile", "sample_price_trend_v1"])
    assert valid_report.exit_code != 0
    assert "do not provide --ranking-profile with --run-id" in valid_report.output

    valid_inspect = runner.invoke(
        app,
        ["inspect", "--run-id", run_id, "--ticker", "AAA", "--ranking-profile", "sample_price_trend_v1"],
    )
    assert valid_inspect.exit_code != 0
    assert "do not provide --ranking-profile with --run-id" in valid_inspect.output

    report = runner.invoke(app, ["report", "--run-id", run_id, "--ranking-profile", "unknown_profile"])
    assert report.exit_code != 0
    assert "do not provide --ranking-profile with --run-id" in report.output
    assert "unknown ranking profile" not in report.output

    inspect = runner.invoke(
        app,
        ["inspect", "--run-id", run_id, "--ticker", "AAA", "--ranking-profile", "unknown_profile"],
    )
    assert inspect.exit_code != 0
    assert "do not provide --ranking-profile with --run-id" in inspect.output
    assert "unknown ranking profile" not in inspect.output


def test_cli_run_id_ranking_profile_conflict_does_not_load_config(monkeypatch) -> None:
    def fail_load_config() -> AppConfig:
        raise AssertionError("load_config must not run before run-id ranking-profile conflict validation")

    monkeypatch.setattr("universe_selector.cli.load_config", fail_load_config)
    run_id = "us-00000000-0000-4000-8000-000000000001"

    report = runner.invoke(app, ["report", "--run-id", run_id, "--ranking-profile", "unknown_profile"])
    assert report.exit_code != 0
    assert "do not provide --ranking-profile with --run-id" in report.output
    assert "unknown ranking profile" not in report.output
    assert "load_config must not run" not in report.output

    inspect = runner.invoke(app, ["inspect", "--run-id", run_id, "--ticker", "AAA", "--ranking-profile", "unknown_profile"])
    assert inspect.exit_code != 0
    assert "do not provide --ranking-profile with --run-id" in inspect.output
    assert "unknown ranking profile" not in inspect.output
    assert "load_config must not run" not in inspect.output


def test_cli_rejects_unknown_ranking_profile_override(monkeypatch, tmp_path: Path, fixture_dir: Path) -> None:
    _write_cli_config(tmp_path, fixture_dir)
    monkeypatch.chdir(tmp_path)

    batch = runner.invoke(app, ["batch", "us", "--ranking-profile", "unknown_profile"])
    assert batch.exit_code != 0
    assert "unknown ranking profile unknown_profile" in batch.output

    report = runner.invoke(app, ["report", "us", "--ranking-profile", "unknown_profile"])
    assert report.exit_code != 0
    assert "unknown ranking profile unknown_profile" in report.output

    inspect = runner.invoke(app, ["inspect", "us", "--ticker", "AAA", "--ranking-profile", "unknown_profile"])
    assert inspect.exit_code != 0
    assert "unknown ranking profile unknown_profile" in inspect.output


def test_cli_reports_config_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["report", "us"])

    assert result.exit_code != 0
    assert "config file not found: config.yaml" in result.output
