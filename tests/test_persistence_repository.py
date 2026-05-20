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
