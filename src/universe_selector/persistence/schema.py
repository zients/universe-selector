from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb

from universe_selector.errors import DuckDbBusyError, SchemaError


MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, "001_initial"),
)

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "schema_migrations": {"version", "name", "checksum", "applied_at"},
    "run_log": {
        "run_id",
        "market",
        "status",
        "created_at",
        "ranking_profile",
        "ranking_config_hash",
        "error_message",
    },
    "run_provider_metadata": {
        "run_id",
        "data_mode",
        "listing_provider_id",
        "listing_source_id",
        "ohlcv_provider_id",
        "ohlcv_source_id",
        "provider_config_hash",
        "data_fetch_started_at",
        "market_timezone",
        "run_latest_bar_date",
    },
    "run_ticker_snapshot": {
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        "metrics_json",
    },
    "run_rankings": {
        "run_id",
        "market",
        "horizon",
        "ticker",
        "final_rank_percentile",
        "rank",
        "metrics_json",
    },
    "report_artifacts": {"run_id", "format", "content"},
}

FORBIDDEN_COLUMNS: dict[str, set[str]] = {}

SCHEMA_MIGRATIONS_SQL = """
create table if not exists schema_migrations (
  version integer primary key,
  name varchar not null,
  checksum varchar not null,
  applied_at timestamp not null default current_timestamp
);
"""


def map_duckdb_error(exc: Exception) -> Exception:
    message = str(exc).lower()
    busy_tokens = ("lock", "locked", "busy", "concurrent", "same database file with a different configuration")
    if isinstance(exc, (duckdb.IOException, duckdb.ConnectionException)) and any(token in message for token in busy_tokens):
        return DuckDbBusyError("database is busy; another process may be using it")
    return exc


def _execute(connection: duckdb.DuckDBPyConnection, sql: str, parameters: list[object] | None = None):
    try:
        return connection.execute(sql, parameters or [])
    except Exception as exc:
        mapped = map_duckdb_error(exc)
        if mapped is not exc:
            raise mapped from exc
        raise


def _migration_path(name: str) -> Path:
    return Path(__file__).parent / "migrations" / f"{name}.sql"


def _migration_sql(name: str) -> str:
    return _migration_path(name).read_text(encoding="utf-8")


def _migration_checksum(name: str) -> str:
    return hashlib.sha256(_migration_sql(name).encode("utf-8")).hexdigest()


def migration_checksum() -> str:
    return _migration_checksum(MIGRATIONS[-1][1])


def _validate_required_tables_and_columns(connection: duckdb.DuckDBPyConnection) -> None:
    for table_name, expected_columns in REQUIRED_COLUMNS.items():
        try:
            rows = _execute(connection, f"describe {table_name}").fetchall()
        except duckdb.CatalogException as exc:
            raise SchemaError(f"required schema object is missing: {table_name}") from exc
        actual_columns = {row[0] for row in rows}
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise SchemaError(f"required schema object is missing columns: {table_name}.{missing}")
        removed_columns = FORBIDDEN_COLUMNS.get(table_name, set()) & actual_columns
        if removed_columns:
            removed = ", ".join(sorted(removed_columns))
            raise SchemaError(f"required schema object has removed columns: {table_name}.{removed}")


def apply_migrations(connection: duckdb.DuckDBPyConnection) -> None:
    _execute(connection, "begin")
    try:
        _execute(connection, SCHEMA_MIGRATIONS_SQL)
        for version, name in MIGRATIONS:
            checksum = _migration_checksum(name)
            existing = _execute(
                connection,
                "select name, checksum from schema_migrations where version = ?",
                [version],
            ).fetchone()
            if existing is not None:
                if existing != (name, checksum):
                    raise SchemaError("applied migration checksum does not match current SQL file")
                continue
            _execute(connection, _migration_sql(name))
            _execute(
                connection,
                "insert into schema_migrations(version, name, checksum) values (?, ?, ?)",
                [version, name, checksum],
            )
        _execute(connection, "commit")
    except Exception:
        _execute(connection, "rollback")
        raise


def validate_schema(connection: duckdb.DuckDBPyConnection) -> None:
    try:
        rows = _execute(
            connection,
            "select version, name, checksum from schema_migrations order by version",
        ).fetchall()
    except duckdb.CatalogException as exc:
        raise SchemaError("schema has not been initialized") from exc
    expected_rows = [
        (version, name, _migration_checksum(name))
        for version, name in MIGRATIONS
    ]
    if rows != expected_rows:
        raise SchemaError("schema version is missing or invalid")
    _validate_required_tables_and_columns(connection)
