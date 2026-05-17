from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from universe_selector.cli import app
from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.persistence.repository import DuckDbRepository
from universe_selector.persistence.schema import apply_migrations
from universe_selector.pipeline import BatchResult, FailedBatchResult, MultiProfileBatchError


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
  fundamentals_provider: yfinance_fundamentals
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
        return BatchResult(
            run_id="us-00000000-0000-4000-8000-000000000003",
            market=market,
            ranking_profile=config.ranking_profile,
        )

    monkeypatch.setattr("universe_selector.cli.run_batch", fake_run_batch)

    result = runner.invoke(app, ["batch", "us", "--ranking-profile", "sample_price_trend_v1"])

    assert result.exit_code == 0, result.output
    assert captured == {"market": Market.US, "ranking_profile": "sample_price_trend_v1"}


def test_cli_batch_single_profile_output_stays_run_id_and_market_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "universe_selector.cli.load_config",
        lambda: AppConfig(ranking_profile="sample_price_trend_v1"),
    )

    def fake_run_batch(market: Market, config: AppConfig) -> BatchResult:
        return BatchResult(
            run_id="us-00000000-0000-4000-8000-000000000003",
            market=market,
            ranking_profile=config.ranking_profile,
        )

    monkeypatch.setattr("universe_selector.cli.run_batch", fake_run_batch)

    no_override = runner.invoke(app, ["batch", "us"])
    assert no_override.exit_code == 0, no_override.output
    assert no_override.output.splitlines() == [
        "run_id: us-00000000-0000-4000-8000-000000000003",
        "market: US",
    ]

    one_override = runner.invoke(app, ["batch", "us", "--ranking-profile", "momentum_v1"])
    assert one_override.exit_code == 0, one_override.output
    assert one_override.output.splitlines() == [
        "run_id: us-00000000-0000-4000-8000-000000000003",
        "market: US",
    ]


def test_cli_batch_accepts_repeated_ranking_profiles(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "universe_selector.cli.load_config",
        lambda: AppConfig(ranking_profile="sample_price_trend_v1"),
    )

    def fail_run_batch(market: Market, config: AppConfig) -> BatchResult:
        raise AssertionError("single-profile run_batch must not be used")

    def fake_run_batch_profiles(market: Market, config: AppConfig, profile_ids: tuple[str, ...]):
        captured["market"] = market
        captured["base_profile"] = config.ranking_profile
        captured["profile_ids"] = profile_ids
        return (
            BatchResult("us-00000000-0000-4000-8000-000000000001", market, "trend_quality_v1"),
            BatchResult("us-00000000-0000-4000-8000-000000000002", market, "momentum_v1"),
        )

    monkeypatch.setattr("universe_selector.cli.run_batch", fail_run_batch)
    monkeypatch.setattr("universe_selector.cli.run_batch_profiles", fake_run_batch_profiles)

    result = runner.invoke(
        app,
        [
            "batch",
            "us",
            "--ranking-profile",
            "trend_quality_v1",
            "--ranking-profile",
            "momentum_v1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "market": Market.US,
        "base_profile": "sample_price_trend_v1",
        "profile_ids": ("trend_quality_v1", "momentum_v1"),
    }
    assert result.output.splitlines() == [
        "run_id: us-00000000-0000-4000-8000-000000000001",
        "ranking_profile: trend_quality_v1",
        "run_id: us-00000000-0000-4000-8000-000000000002",
        "ranking_profile: momentum_v1",
        "market: US",
    ]


def test_cli_batch_rejects_duplicate_repeated_ranking_profiles(monkeypatch) -> None:
    monkeypatch.setattr("universe_selector.cli.load_config", lambda: AppConfig())

    def fail_run_batch_profiles(*args, **kwargs):
        raise AssertionError("pipeline must not run for duplicate profiles")

    monkeypatch.setattr("universe_selector.cli.run_batch_profiles", fail_run_batch_profiles)

    result = runner.invoke(
        app,
        ["batch", "us", "--ranking-profile", "momentum_v1", "--ranking-profile", "momentum_v1"],
    )

    assert result.exit_code != 0
    assert "duplicate ranking profile momentum_v1" in result.output


def test_cli_batch_rejects_unknown_repeated_ranking_profile(monkeypatch) -> None:
    monkeypatch.setattr("universe_selector.cli.load_config", lambda: AppConfig())

    def fail_run_batch_profiles(*args, **kwargs):
        raise AssertionError("pipeline must not run for unknown profiles")

    monkeypatch.setattr("universe_selector.cli.run_batch_profiles", fail_run_batch_profiles)

    result = runner.invoke(
        app,
        ["batch", "us", "--ranking-profile", "unknown_profile", "--ranking-profile", "momentum_v1"],
    )

    assert result.exit_code != 0
    assert "unknown ranking profile unknown_profile" in result.output


def test_cli_batch_prints_partial_results_on_multi_profile_failure(monkeypatch) -> None:
    monkeypatch.setattr("universe_selector.cli.load_config", lambda: AppConfig())

    def fake_run_batch_profiles(market: Market, config: AppConfig, profile_ids: tuple[str, ...]):
        raise MultiProfileBatchError(
            completed_results=(
                BatchResult("us-00000000-0000-4000-8000-000000000001", market, "trend_quality_v1"),
            ),
            failed_result=FailedBatchResult(
                run_id="us-00000000-0000-4000-8000-000000000002",
                market=market,
                ranking_profile="momentum_v1",
                error_message="boom",
            ),
            exit_code=ValidationError.exit_code,
        )

    monkeypatch.setattr("universe_selector.cli.run_batch_profiles", fake_run_batch_profiles)

    result = runner.invoke(
        app,
        ["batch", "us", "--ranking-profile", "trend_quality_v1", "--ranking-profile", "momentum_v1"],
    )

    assert result.exit_code == ValidationError.exit_code
    assert result.output.splitlines() == [
        "run_id: us-00000000-0000-4000-8000-000000000001",
        "ranking_profile: trend_quality_v1",
        "failed_run_id: us-00000000-0000-4000-8000-000000000002",
        "failed_ranking_profile: momentum_v1",
        "error: boom",
        "market: US",
    ]


def test_cli_batch_prints_first_failed_profile_without_completed_results(monkeypatch) -> None:
    monkeypatch.setattr("universe_selector.cli.load_config", lambda: AppConfig())

    def fake_run_batch_profiles(market: Market, config: AppConfig, profile_ids: tuple[str, ...]):
        raise MultiProfileBatchError(
            completed_results=(),
            failed_result=FailedBatchResult(
                run_id="us-00000000-0000-4000-8000-000000000001",
                market=market,
                ranking_profile="trend_quality_v1",
                error_message="first profile failed",
            ),
            exit_code=ValidationError.exit_code,
        )

    monkeypatch.setattr("universe_selector.cli.run_batch_profiles", fake_run_batch_profiles)

    result = runner.invoke(
        app,
        ["batch", "us", "--ranking-profile", "trend_quality_v1", "--ranking-profile", "momentum_v1"],
    )

    assert result.exit_code == ValidationError.exit_code
    assert result.output.splitlines() == [
        "failed_run_id: us-00000000-0000-4000-8000-000000000001",
        "failed_ranking_profile: trend_quality_v1",
        "error: first profile failed",
        "market: US",
    ]


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


def test_cli_run_id_ranking_profile_conflict_precedes_run_id_parsing() -> None:
    report = runner.invoke(app, ["report", "--run-id", "bad-id", "--ranking-profile", "sample_price_trend_v1"])
    assert report.exit_code != 0
    assert "do not provide --ranking-profile with --run-id" in report.output

    inspect = runner.invoke(
        app,
        ["inspect", "--run-id", "bad-id", "--ticker", "AAA", "--ranking-profile", "sample_price_trend_v1"],
    )
    assert inspect.exit_code != 0
    assert "do not provide --ranking-profile with --run-id" in inspect.output


def test_cli_market_run_id_conflict_precedes_ranking_profile_conflict() -> None:
    report = runner.invoke(
        app,
        ["report", "us", "--run-id", "us-00000000-0000-4000-8000-000000000001", "--ranking-profile", "sample_price_trend_v1"],
    )
    assert report.exit_code != 0
    assert "provide either MARKET or --run-id, not both" in report.output
    assert "do not provide --ranking-profile with --run-id" not in report.output

    inspect = runner.invoke(
        app,
        [
            "inspect",
            "us",
            "--run-id",
            "us-00000000-0000-4000-8000-000000000001",
            "--ticker",
            "AAA",
            "--ranking-profile",
            "sample_price_trend_v1",
        ],
    )
    assert inspect.exit_code != 0
    assert "provide either MARKET or --run-id, not both" in inspect.output
    assert "do not provide --ranking-profile with --run-id" not in inspect.output


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


def _install_value_cli_no_persistence_or_ranking_guards(monkeypatch) -> None:
    monkeypatch.setattr(
        "universe_selector.cli.ensure_runtime_dirs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ensure_runtime_dirs")),
        raising=False,
    )
    monkeypatch.setattr(
        "universe_selector.cli.DuckDbRepository",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DuckDbRepository")),
    )
    monkeypatch.setattr(
        "universe_selector.cli.validate_schema",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validate_schema")),
    )
    monkeypatch.setattr(
        "universe_selector.cli.run_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_batch")),
    )
    monkeypatch.setattr(
        "universe_selector.cli.run_batch_profiles",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_batch_profiles")),
    )
    monkeypatch.setattr(
        "universe_selector.cli._read_repo",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_read_repo")),
    )
    monkeypatch.setattr(
        "universe_selector.cli.render_inspect",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("render_inspect")),
    )
    monkeypatch.setattr(
        "universe_selector.cli.get_ranking_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("get_ranking_profile")),
    )


def test_cli_value_reads_config_provider_and_prints_markdown(monkeypatch) -> None:
    _install_value_cli_no_persistence_or_ranking_guards(monkeypatch)
    captured: dict[str, object] = {}
    sentinel_result = object()

    def fake_run_valuation(
        *,
        market: Market,
        ticker: str,
        model_id: str,
        assumptions_path: Path | None,
        fundamentals_provider_id: str,
    ):
        captured["market"] = market
        captured["ticker"] = ticker
        captured["model_id"] = model_id
        captured["assumptions_path"] = assumptions_path
        captured["fundamentals_provider_id"] = fundamentals_provider_id
        return sentinel_result

    monkeypatch.setattr(
        "universe_selector.cli.load_config",
        lambda: AppConfig(live_fundamentals_provider="fake_fundamentals"),
        raising=False,
    )
    monkeypatch.setattr("universe_selector.cli.get_valuation_model", lambda model_id: object(), raising=False)
    monkeypatch.setattr("universe_selector.cli.run_valuation", fake_run_valuation, raising=False)
    monkeypatch.setattr(
        "universe_selector.cli.render_valuation_markdown",
        lambda result: "# rendered valuation\n" if result is sentinel_result else "wrong\n",
        raising=False,
    )

    result = runner.invoke(app, ["value", "us", "--ticker", "aapl"])

    assert result.exit_code == 0, result.output
    assert result.output == "# rendered valuation\n"
    assert captured["market"] is Market.US
    assert captured["ticker"] == "AAPL"
    assert captured["model_id"] == "fcf_dcf_v1"
    assert captured["assumptions_path"] is None
    assert captured["fundamentals_provider_id"] == "fake_fundamentals"


def test_cli_value_passes_model_and_default_assumptions_path_ownership(monkeypatch) -> None:
    _install_value_cli_no_persistence_or_ranking_guards(monkeypatch)
    captured: dict[str, object] = {}

    def fake_run_valuation(
        *,
        market: Market,
        ticker: str,
        model_id: str,
        assumptions_path: Path | None,
        fundamentals_provider_id: str,
    ):
        captured["market"] = market
        captured["ticker"] = ticker
        captured["model_id"] = model_id
        captured["assumptions_path"] = assumptions_path
        captured["fundamentals_provider_id"] = fundamentals_provider_id
        return object()

    monkeypatch.setattr(
        "universe_selector.cli.load_config",
        lambda: AppConfig(live_fundamentals_provider="fake_fundamentals"),
        raising=False,
    )
    monkeypatch.setattr("universe_selector.cli.get_valuation_model", lambda model_id: object(), raising=False)
    monkeypatch.setattr("universe_selector.cli.run_valuation", fake_run_valuation, raising=False)
    monkeypatch.setattr("universe_selector.cli.render_valuation_markdown", lambda result: "ok\n", raising=False)

    result = runner.invoke(app, ["value", "us", "--ticker", "aapl", "--model", "fcf_dcf_v1"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "market": Market.US,
        "ticker": "AAPL",
        "model_id": "fcf_dcf_v1",
        "assumptions_path": None,
        "fundamentals_provider_id": "fake_fundamentals",
    }


def test_cli_value_passes_explicit_assumptions_path(monkeypatch, tmp_path: Path) -> None:
    _install_value_cli_no_persistence_or_ranking_guards(monkeypatch)
    captured: dict[str, object] = {}
    assumptions_path = tmp_path / "valuation_assumptions" / "us" / "AAPL.yaml"

    def fake_run_valuation(
        *,
        market: Market,
        ticker: str,
        model_id: str,
        assumptions_path: Path | None,
        fundamentals_provider_id: str,
    ):
        captured["market"] = market
        captured["ticker"] = ticker
        captured["model_id"] = model_id
        captured["assumptions_path"] = assumptions_path
        captured["fundamentals_provider_id"] = fundamentals_provider_id
        return object()

    monkeypatch.setattr(
        "universe_selector.cli.load_config",
        lambda: AppConfig(live_fundamentals_provider="fake_fundamentals"),
        raising=False,
    )
    monkeypatch.setattr("universe_selector.cli.get_valuation_model", lambda model_id: object(), raising=False)
    monkeypatch.setattr("universe_selector.cli.run_valuation", fake_run_valuation, raising=False)
    monkeypatch.setattr("universe_selector.cli.render_valuation_markdown", lambda result: "ok\n", raising=False)

    result = runner.invoke(
        app,
        ["value", "us", "--ticker", "aapl", "--assumptions", str(assumptions_path)],
    )

    assert result.exit_code == 0, result.output
    assert captured["market"] is Market.US
    assert captured["ticker"] == "AAPL"
    assert captured["model_id"] == "fcf_dcf_v1"
    assert captured["assumptions_path"] == assumptions_path
    assert captured["fundamentals_provider_id"] == "fake_fundamentals"


def test_cli_value_rejects_unknown_model(monkeypatch) -> None:
    _install_value_cli_no_persistence_or_ranking_guards(monkeypatch)

    def fail_model_lookup(model_id: str):
        raise ValidationError(f"unknown valuation model {model_id}")

    def fail_run_valuation(*args, **kwargs):
        raise AssertionError("run_valuation must not run for an unknown model")

    monkeypatch.setattr(
        "universe_selector.cli.load_config",
        lambda: (_ for _ in ()).throw(AssertionError("load_config must not run before model validation")),
        raising=False,
    )
    monkeypatch.setattr("universe_selector.cli.get_valuation_model", fail_model_lookup, raising=False)
    monkeypatch.setattr("universe_selector.cli.run_valuation", fail_run_valuation, raising=False)
    monkeypatch.setattr(
        "universe_selector.cli.render_valuation_markdown",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("render_valuation_markdown")),
        raising=False,
    )

    result = runner.invoke(app, ["value", "us", "--ticker", "AAPL", "--model", "unknown_model"])

    assert result.exit_code == ValidationError.exit_code
    assert "unknown valuation model unknown_model" in result.output
