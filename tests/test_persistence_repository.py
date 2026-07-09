from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.persistence.repository import DuckDbRepository
from universe_selector.persistence.schema import (
    SCHEMA_MIGRATIONS_SQL,
    _migration_checksum,
    _migration_sql,
    apply_migrations,
    validate_schema,
)
from universe_selector.providers.models import ProviderMetadata


def _metadata() -> ProviderMetadata:
    return ProviderMetadata(
        data_mode="fixture",
        listing_provider_id="fixture-listings-v1",
        listing_source_id="sample_basic/listings.csv",
        ohlcv_provider_id="fixture-ohlcv-v1",
        ohlcv_source_id="sample_basic/ohlcv.csv",
        provider_config_hash="fixture-sample-basic",
        data_fetch_started_at=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc),
        market_timezone="UTC",
        run_latest_bar_date=date(2026, 4, 24),
    )


def _sample_snapshot(run_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": [run_id],
            "market": ["US"],
            "ticker": ["AAA"],
            "close": [20.0],
            "adjusted_close": [20.0],
            "avg_traded_value_20d_local": [20_000_000.0],
            "return_60d": [0.20],
            "return_120d": [0.40],
        }
    )


def _sample_rankings(run_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": [run_id, run_id],
            "market": ["US", "US"],
            "horizon": ["midterm", "longterm"],
            "ticker": ["AAA", "AAA"],
            "score_return_60d": [0.20, 0.20],
            "score_return_120d": [0.40, 0.40],
            "score": [0.20, 0.40],
            "rank": [1, 1],
        }
    )


def test_repository_persists_profile_scores_outside_0_to_100_range(tmp_path: Path) -> None:
    run_id = "us-00000000-0000-4000-8000-000000000123"
    repo = DuckDbRepository(tmp_path / "runs.duckdb")
    apply_migrations(repo.connect())
    repo.create_running_run(run_id, Market.US, AppConfig(data_mode="fixture"))

    snapshot = pl.DataFrame(
        {
            "run_id": [run_id],
            "market": ["US"],
            "ticker": ["AAA"],
            "close": [20.0],
            "adjusted_close": [20.0],
            "avg_traded_value_20d_local": [20_000_000.0],
            "return_60d": [0.20],
            "return_120d": [0.40],
        }
    )
    rankings = pl.DataFrame(
        {
            "run_id": [run_id, run_id],
            "market": ["US", "US"],
            "horizon": ["midterm", "longterm"],
            "ticker": ["AAA", "AAA"],
            "score_return_60d": [123.45, 123.45],
            "score_return_120d": [-12.5, -12.5],
            "score": [123.45, -12.5],
            "rank": [1, 1],
        }
    )

    repo.mark_successful_run(
        run_id=run_id,
        metadata=_metadata(),
        snapshot=snapshot,
        rankings=rankings,
        markdown="# Report\n",
    )

    stored_scores = (
        repo.connect()
        .execute(
            "select score from run_rankings where run_id = ? order by horizon",
            [run_id],
        )
        .fetchall()
    )
    payload = repo.read_inspect_payload(run_id, "AAA", profile=AppConfig().selected_ranking_profile)
    stored_metadata = repo.read_provider_metadata(run_id)

    assert stored_scores == [(-12.5,), (123.45,)]
    assert {row["score"] for row in payload.rankings} == {123.45, -12.5}
    assert not hasattr(stored_metadata, "run_id")
    assert stored_metadata.provider_config_hash == "fixture-sample-basic"


def test_repository_persists_fundamentals_provider_metadata(tmp_path: Path) -> None:
    run_id = "us-00000000-0000-4000-8000-000000000126"
    repo = DuckDbRepository(tmp_path / "runs.duckdb")
    apply_migrations(repo.connect())
    repo.create_running_run(run_id, Market.US, AppConfig(data_mode="fixture"))

    metadata = ProviderMetadata(
        data_mode="fixture",
        listing_provider_id="fixture-listings-v1",
        listing_source_id="sample_basic/listings.csv",
        ohlcv_provider_id="fixture-ohlcv-v1",
        ohlcv_source_id="sample_basic/ohlcv.csv",
        provider_config_hash="fixture-with-fundamentals",
        data_fetch_started_at=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc),
        market_timezone="UTC",
        run_latest_bar_date=date(2026, 4, 24),
        fundamentals_provider_id="yfinance_fundamentals",
        fundamentals_source_id="yahoo-finance:yfinance-ticker",
        fundamentals_latest_source_date=date(2026, 3, 31),
        fundamentals_source_risk_note="unit risk note",
        fundamentals_field_mapping_note="unit mapping note",
        fundamentals_requested_count=3,
        fundamentals_returned_count=2,
        fundamentals_missing_count=1,
        fundamentals_invalid_count=0,
    )

    repo.mark_successful_run(
        run_id=run_id,
        metadata=metadata,
        snapshot=_sample_snapshot(run_id),
        rankings=_sample_rankings(run_id),
        markdown="# Report\n",
    )

    stored = repo.read_provider_metadata(run_id)

    assert stored.fundamentals_provider_id == "yfinance_fundamentals"
    assert stored.fundamentals_source_id == "yahoo-finance:yfinance-ticker"
    assert stored.fundamentals_latest_source_date == date(2026, 3, 31)
    assert stored.fundamentals_requested_count == 3
    assert stored.fundamentals_returned_count == 2
    assert stored.fundamentals_missing_count == 1
    assert stored.fundamentals_invalid_count == 0


def test_repository_stores_and_reads_json_report_artifact(tmp_path: Path) -> None:
    run_id = "us-00000000-0000-4000-8000-000000000124"
    repo = DuckDbRepository(tmp_path / "runs.duckdb")
    apply_migrations(repo.connect())
    repo.create_running_run(run_id, Market.US, AppConfig(data_mode="fixture"))

    snapshot = pl.DataFrame(
        {
            "run_id": [run_id],
            "market": ["US"],
            "ticker": ["AAA"],
            "close": [20.0],
            "adjusted_close": [20.0],
            "avg_traded_value_20d_local": [20_000_000.0],
            "return_60d": [0.20],
            "return_120d": [0.40],
        }
    )
    rankings = pl.DataFrame(
        {
            "run_id": [run_id, run_id],
            "market": ["US", "US"],
            "horizon": ["midterm", "longterm"],
            "ticker": ["AAA", "AAA"],
            "score_return_60d": [123.45, 123.45],
            "score_return_120d": [-12.5, -12.5],
            "score": [123.45, -12.5],
            "rank": [1, 1],
        }
    )

    repo.mark_successful_run(
        run_id=run_id,
        metadata=_metadata(),
        snapshot=snapshot,
        rankings=rankings,
        markdown="# Report\n",
        json_report='{"artifact_type":"universe_selector_report"}\n',
    )

    assert repo.read_report_markdown(run_id) == "# Report\n"
    assert repo.read_report_artifact(run_id, "markdown") == "# Report\n"
    assert repo.read_report_artifact(run_id, "json") == '{"artifact_type":"universe_selector_report"}\n'


def test_report_json_migration_upgrades_existing_markdown_artifacts(tmp_path: Path) -> None:
    run_id = "us-00000000-0000-4000-8000-000000000125"
    repo = DuckDbRepository(tmp_path / "runs.duckdb")
    connection = repo.connect()
    connection.execute(SCHEMA_MIGRATIONS_SQL)
    connection.execute(_migration_sql("001_initial"))
    connection.execute(
        "insert into schema_migrations(version, name, checksum) values (1, '001_initial', ?)",
        [_migration_checksum("001_initial")],
    )
    connection.execute(
        """
        insert into run_log(run_id, market, status, created_at, ranking_profile, ranking_config_hash, error_message)
        values (?, 'US', 'successful', timestamp '2026-01-01 00:00:00', 'sample_price_trend_v1', 'sample-hash', null)
        """,
        [run_id],
    )
    connection.execute(
        "insert into report_artifacts(run_id, format, content) values (?, 'markdown', ?)",
        [run_id, "# Old report\n"],
    )

    apply_migrations(connection)
    validate_schema(connection)
    connection.execute(
        "insert into report_artifacts(run_id, format, content) values (?, 'json', ?)",
        [run_id, '{"artifact_type":"universe_selector_report"}\n'],
    )

    assert repo.read_report_artifact(run_id, "markdown") == "# Old report\n"
    assert repo.read_report_artifact(run_id, "json") == '{"artifact_type":"universe_selector_report"}\n'


def test_fundamentals_metadata_migration_preserves_v2_successful_runs(tmp_path: Path) -> None:
    run_id = "us-00000000-0000-4000-8000-000000000127"
    repo = DuckDbRepository(tmp_path / "runs.duckdb")
    connection = repo.connect()
    connection.execute(SCHEMA_MIGRATIONS_SQL)
    for version, name in ((1, "001_initial"), (2, "002_report_json_artifacts")):
        connection.execute(_migration_sql(name))
        connection.execute(
            "insert into schema_migrations(version, name, checksum) values (?, ?, ?)",
            [version, name, _migration_checksum(name)],
        )
    connection.execute(
        """
        insert into run_log(run_id, market, status, created_at, ranking_profile, ranking_config_hash, error_message)
        values (?, 'US', 'successful', timestamp '2026-01-01 00:00:00', 'sample_price_trend_v1', 'sample-hash', null)
        """,
        [run_id],
    )
    connection.execute(
        """
        insert into run_provider_metadata(
          run_id, data_mode, listing_provider_id, listing_source_id,
          ohlcv_provider_id, ohlcv_source_id, provider_config_hash,
          data_fetch_started_at, market_timezone, run_latest_bar_date
        )
        values (
          ?, 'fixture', 'fixture-listings-v1', 'sample_basic/listings.csv',
          'fixture-ohlcv-v1', 'sample_basic/ohlcv.csv', 'fixture-sample-basic',
          timestamp '2026-04-24 12:00:00', 'UTC', date '2026-04-24'
        )
        """,
        [run_id],
    )
    connection.execute(
        """
        insert into run_ticker_snapshot(run_id, market, ticker, close, adjusted_close, metrics_json)
        values (?, 'US', 'AAA', 20.0, 20.0, ?)
        """,
        [run_id, '{"avg_traded_value_20d_local":20000000.0,"return_120d":0.4,"return_60d":0.2}'],
    )
    for horizon, score in (("midterm", 0.2), ("longterm", 0.4)):
        connection.execute(
            """
            insert into run_rankings(run_id, market, horizon, ticker, score, rank, metrics_json)
            values (?, 'US', ?, 'AAA', ?, 1, ?)
            """,
            [run_id, horizon, score, '{"score_return_120d":0.4,"score_return_60d":0.2}'],
        )
    connection.execute(
        "insert into report_artifacts(run_id, format, content) values (?, 'markdown', ?)",
        [run_id, "# Old report\n"],
    )
    connection.execute(
        "insert into report_artifacts(run_id, format, content) values (?, 'json', ?)",
        [run_id, '{"artifact_type":"universe_selector_report"}\n'],
    )

    apply_migrations(connection)
    validate_schema(connection)
    metadata = repo.read_provider_metadata(run_id)
    inspect_payload = repo.read_inspect_payload(run_id, "AAA", profile=AppConfig().selected_ranking_profile)

    assert metadata.fundamentals_provider_id is None
    assert metadata.fundamentals_requested_count is None
    assert repo.read_report_markdown(run_id) == "# Old report\n"
    assert inspect_payload.snapshot["ticker"] == "AAA"
