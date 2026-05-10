from __future__ import annotations

import json
import math
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.errors import DataIntegrityError, NotFoundError, SchemaError
from universe_selector.persistence.schema import map_duckdb_error, validate_schema
from universe_selector.providers.models import ProviderMetadata
from universe_selector.ranking_profiles import RankingProfile, get_ranking_profile


SNAPSHOT_COLUMNS = [
    "run_id",
    "market",
    "ticker",
    "close",
    "adjusted_close",
    "metrics_json",
]

SNAPSHOT_CORE_COLUMNS = [
    "run_id",
    "market",
    "ticker",
    "close",
    "adjusted_close",
]

RANKING_COLUMNS = [
    "run_id",
    "market",
    "horizon",
    "ticker",
    "score",
    "rank",
    "metrics_json",
]

RANKING_CORE_COLUMNS = [
    "run_id",
    "market",
    "horizon",
    "ticker",
    "score",
    "rank",
]

PROVIDER_METADATA_COLUMNS = [
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
]

@dataclass(frozen=True)
class ResolvedRun:
    run_id: str
    market: Market
    ranking_profile: str
    ranking_config_hash: str


@dataclass(frozen=True)
class InspectPayload:
    metadata: ProviderMetadata
    snapshot: dict[str, object]
    rankings: list[dict[str, object]]


def _execute(connection: duckdb.DuckDBPyConnection, sql: str, parameters: list[object] | None = None):
    try:
        return connection.execute(sql, parameters or [])
    except Exception as exc:
        mapped = map_duckdb_error(exc)
        if mapped is not exc:
            raise mapped from exc
        raise


def _utc_now_millis() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=(now.microsecond // 1000) * 1000)


def _rows_to_dicts(columns: list[str], rows: list[tuple[Any, ...]]) -> list[dict[str, object]]:
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _is_json_number(value: object) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    return True


def _finite_float(value: object, label: str) -> float:
    if not _is_json_number(value):
        raise DataIntegrityError(f"{label} must be a JSON number")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise DataIntegrityError(f"{label} must be finite") from exc
    if not math.isfinite(numeric):
        raise DataIntegrityError(f"{label} must be finite")
    return numeric


def _metrics_json(row: dict[str, object], keys: tuple[str, ...], label: str) -> str:
    payload: dict[str, float] = {}
    for key in keys:
        if key not in row:
            raise DataIntegrityError(f"{label} metric is required: {key}")
        payload[key] = _finite_float(row[key], f"{label} metric {key}")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _parse_metrics_json(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, str):
        raise DataIntegrityError(f"{label} metrics_json must be a JSON object")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"{label} metrics_json is invalid") from exc
    if not isinstance(payload, dict):
        raise DataIntegrityError(f"{label} metrics_json must be a JSON object")
    result: dict[str, float] = {}
    for key, item in payload.items():
        result[str(key)] = _finite_float(item, f"{label} metric {key}")
    return result


def _metadata_from_row(row: tuple[Any, ...]) -> ProviderMetadata:
    payload = dict(zip(PROVIDER_METADATA_COLUMNS, row, strict=True))
    data_fetch_started_at = payload["data_fetch_started_at"]
    if not isinstance(data_fetch_started_at, datetime):
        raise DataIntegrityError("provider metadata data_fetch_started_at must be a timestamp")
    if data_fetch_started_at.tzinfo is None or data_fetch_started_at.utcoffset() is None:
        data_fetch_started_at = data_fetch_started_at.replace(tzinfo=timezone.utc)
    else:
        data_fetch_started_at = data_fetch_started_at.astimezone(timezone.utc)
    return ProviderMetadata(
        run_id=str(payload["run_id"]),
        data_mode=str(payload["data_mode"]),
        listing_provider_id=str(payload["listing_provider_id"]),
        listing_source_id=str(payload["listing_source_id"]),
        ohlcv_provider_id=str(payload["ohlcv_provider_id"]),
        ohlcv_source_id=str(payload["ohlcv_source_id"]),
        provider_config_hash=str(payload["provider_config_hash"]),
        data_fetch_started_at=data_fetch_started_at,
        market_timezone=str(payload["market_timezone"]),
        run_latest_bar_date=payload["run_latest_bar_date"],
    )


def _metadata_timestamp_for_storage(metadata: ProviderMetadata) -> datetime:
    data_fetch_started_at = metadata.data_fetch_started_at
    if data_fetch_started_at.tzinfo is None or data_fetch_started_at.utcoffset() is None:
        return data_fetch_started_at
    return data_fetch_started_at.astimezone(timezone.utc).replace(tzinfo=None)


def _column_values(frame: pl.DataFrame, column: str) -> set[object]:
    if frame.is_empty() or column not in frame.columns:
        return set()
    return set(frame[column].unique().to_list())


def _require_column_value(frame: pl.DataFrame, column: str, expected: object, label: str) -> None:
    if not frame.is_empty() and column not in frame.columns:
        raise DataIntegrityError(f"{label} is required")
    values = _column_values(frame, column)
    if values and values != {expected}:
        raise DataIntegrityError(f"{label} must match run context")


def _validate_snapshot_ranking_consistency(
    snapshot: pl.DataFrame,
    rankings: pl.DataFrame,
    profile: RankingProfile,
) -> None:
    snapshot_tickers = _column_values(snapshot, "ticker")
    ranking_tickers = _column_values(rankings, "ticker")
    if snapshot_tickers != ranking_tickers:
        raise DataIntegrityError("snapshot and ranking tickers must match")
    if snapshot.is_empty() and rankings.is_empty():
        return
    if snapshot.height != len(snapshot_tickers):
        raise DataIntegrityError("snapshot tickers must be unique")

    expected_horizons = set(profile.horizon_order)
    if rankings.height != len(snapshot_tickers) * len(expected_horizons):
        raise DataIntegrityError("rankings must contain exactly one row per profile horizon per ticker")

    ranking_rows = rankings.select(["ticker", "horizon", "rank"]).to_dicts()
    for ticker in snapshot_tickers:
        horizons = {str(row["horizon"]) for row in ranking_rows if row["ticker"] == ticker}
        if horizons != expected_horizons:
            raise DataIntegrityError("rankings must contain exactly the profile horizons for every ticker")

    expected_ranks = list(range(1, len(snapshot_tickers) + 1))
    for horizon in expected_horizons:
        ranks = []
        for row in ranking_rows:
            if str(row["horizon"]) != horizon:
                continue
            rank = row["rank"]
            if not isinstance(rank, int):
                raise DataIntegrityError("ranking ranks must be integers")
            ranks.append(rank)
        ranks.sort()
        if ranks != expected_ranks:
            raise DataIntegrityError("ranking ranks must be unique and contiguous within each horizon")


def _snapshot_storage_rows(snapshot: pl.DataFrame, profile: RankingProfile) -> list[dict[str, object]]:
    rows = []
    for row in snapshot.to_dicts():
        stored = {column: row[column] for column in SNAPSHOT_CORE_COLUMNS}
        stored["close"] = _finite_float(stored["close"], "snapshot close")
        stored["adjusted_close"] = _finite_float(stored["adjusted_close"], "snapshot adjusted_close")
        stored["metrics_json"] = _metrics_json(row, profile.snapshot_metric_keys, "snapshot")
        rows.append(stored)
    return rows


def _ranking_storage_rows(rankings: pl.DataFrame, profile: RankingProfile) -> list[dict[str, object]]:
    rows = []
    for row in rankings.to_dicts():
        stored = {column: row[column] for column in RANKING_CORE_COLUMNS}
        stored["score"] = _finite_float(
            stored["score"],
            "ranking score",
        )
        stored["metrics_json"] = _metrics_json(row, profile.ranking_metric_keys, "ranking")
        rows.append(stored)
    return rows


class DuckDbRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._connection: duckdb.DuckDBPyConnection | None = None

    def connect(self, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        if read_only and not Path(self.db_path).exists():
            raise SchemaError("schema has not been initialized")
        if self._connection is None:
            if not read_only:
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                self._connection = duckdb.connect(self.db_path, read_only=read_only)
            except Exception as exc:
                mapped = map_duckdb_error(exc)
                if mapped is not exc:
                    raise mapped from exc
                raise
        return self._connection

    def _read_connection(self) -> duckdb.DuckDBPyConnection:
        connection = self.connect(read_only=True)
        validate_schema(connection)
        return connection

    def create_running_run(self, run_id: str, market: Market, config: AppConfig) -> None:
        connection = self.connect()
        _execute(connection, "begin")
        try:
            _execute(
                connection,
                """
                insert into run_log(
                  run_id, market, status, created_at, ranking_profile,
                  ranking_config_hash, error_message
                )
                values (?, ?, 'running', ?, ?, ?, null)
                """,
                [
                    run_id,
                    market.value,
                    _utc_now_millis(),
                    config.ranking_profile,
                    config.ranking_config_hash(),
                ],
            )
            _execute(connection, "commit")
        except Exception:
            with suppress(Exception):
                _execute(connection, "rollback")
            raise

    def mark_successful_run(
        self,
        *,
        run_id: str,
        metadata: ProviderMetadata,
        snapshot: pl.DataFrame,
        rankings: pl.DataFrame,
        markdown: str,
    ) -> None:
        connection = self.connect()
        _execute(connection, "begin")
        try:
            running_row = _execute(
                connection,
                "select market, ranking_profile from run_log where run_id = ? and status = 'running'",
                [run_id],
            ).fetchone()
            if running_row is None:
                raise DataIntegrityError("run is not running")
            run_market = running_row[0]
            profile = get_ranking_profile(str(running_row[1]))
            profile.validate()
            if metadata.run_id != run_id:
                raise DataIntegrityError("provider metadata run_id must match run context")
            _require_column_value(snapshot, "run_id", run_id, "snapshot run_id")
            _require_column_value(snapshot, "market", run_market, "snapshot market")
            _require_column_value(rankings, "run_id", run_id, "ranking run_id")
            _require_column_value(rankings, "market", run_market, "ranking market")
            _validate_snapshot_ranking_consistency(snapshot, rankings, profile)
            # DuckDB rejects updating a referenced parent row after child inserts;
            # the surrounding transaction still makes later payload failures atomic.
            _execute(
                connection,
                "update run_log set status = 'successful', error_message = null where run_id = ? and status = 'running'",
                [run_id],
            )
            _execute(
                connection,
                """
                insert into run_provider_metadata(
                  run_id, data_mode, listing_provider_id, listing_source_id,
                  ohlcv_provider_id, ohlcv_source_id, provider_config_hash,
                  data_fetch_started_at, market_timezone, run_latest_bar_date
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    metadata.data_mode,
                    metadata.listing_provider_id,
                    metadata.listing_source_id,
                    metadata.ohlcv_provider_id,
                    metadata.ohlcv_source_id,
                    metadata.provider_config_hash,
                    _metadata_timestamp_for_storage(metadata),
                    metadata.market_timezone,
                    metadata.run_latest_bar_date,
                ],
            )

            if not snapshot.is_empty():
                for row in _snapshot_storage_rows(snapshot, profile):
                    _execute(
                        connection,
                        f"insert into run_ticker_snapshot({', '.join(SNAPSHOT_COLUMNS)}) values ({', '.join(['?'] * len(SNAPSHOT_COLUMNS))})",
                        [row[column] for column in SNAPSHOT_COLUMNS],
                    )

            if not rankings.is_empty():
                for row in _ranking_storage_rows(rankings, profile):
                    _execute(
                        connection,
                        f"insert into run_rankings({', '.join(RANKING_COLUMNS)}) values ({', '.join(['?'] * len(RANKING_COLUMNS))})",
                        [row[column] for column in RANKING_COLUMNS],
                    )

            _execute(
                connection,
                "insert into report_artifacts(run_id, format, content) values (?, 'markdown', ?)",
                [run_id, markdown],
            )
            _execute(connection, "commit")
        except Exception:
            with suppress(Exception):
                _execute(connection, "rollback")
            raise

    def mark_failed_run(self, run_id: str, message: str) -> None:
        connection = self.connect()
        _execute(connection, "begin")
        try:
            _execute(
                connection,
                "update run_log set status = 'failed', error_message = ? where run_id = ? and status = 'running'",
                [message, run_id],
            )
            _execute(connection, "commit")
        except Exception:
            with suppress(Exception):
                _execute(connection, "rollback")
            raise

    def resolve_latest_successful_run(
        self,
        market: Market,
        *,
        ranking_profile: str | None = None,
    ) -> ResolvedRun:
        parameters: list[object] = [market.value]
        profile_filter = ""
        if ranking_profile is not None:
            profile_filter = " and ranking_profile = ?"
            parameters.append(ranking_profile)
        row = _execute(
            self._read_connection(),
            f"""
            select run_id, market, ranking_profile, ranking_config_hash
            from run_log
            where market = ? and status = 'successful'{profile_filter}
            order by created_at desc, run_id desc
            limit 1
            """,
            parameters,
        ).fetchone()
        if row is None:
            raise NotFoundError(f"run not found for market {market.value}")
        return ResolvedRun(
            run_id=row[0],
            market=Market(row[1]),
            ranking_profile=str(row[2]),
            ranking_config_hash=str(row[3]),
        )

    def resolve_successful_run(self, run_id: str) -> ResolvedRun:
        row = _execute(
            self._read_connection(),
            """
            select run_id, market, ranking_profile, ranking_config_hash
            from run_log
            where run_id = ? and status = 'successful'
            """,
            [run_id],
        ).fetchone()
        if row is None:
            raise NotFoundError(f"run not found: {run_id}")
        return ResolvedRun(
            run_id=row[0],
            market=Market(row[1]),
            ranking_profile=str(row[2]),
            ranking_config_hash=str(row[3]),
        )

    def read_report_markdown(self, run_id: str) -> str:
        rows = _execute(
            self._read_connection(),
            "select content from report_artifacts where run_id = ? and format = 'markdown'",
            [run_id],
        ).fetchall()
        if len(rows) != 1:
            raise DataIntegrityError(f"expected exactly one markdown report artifact for run {run_id}")
        return rows[0][0]

    def read_provider_metadata(self, run_id: str) -> ProviderMetadata:
        rows = _execute(
            self._read_connection(),
            f"select {', '.join(PROVIDER_METADATA_COLUMNS)} from run_provider_metadata where run_id = ?",
            [run_id],
        ).fetchall()
        if len(rows) != 1:
            raise DataIntegrityError(f"expected exactly one provider metadata row for run {run_id}")
        return _metadata_from_row(rows[0])

    def read_inspect_payload(
        self,
        run_id: str,
        ticker: str,
        *,
        profile: RankingProfile,
    ) -> InspectPayload:
        metadata = self.read_provider_metadata(run_id)
        snapshot_rows = _execute(
            self._read_connection(),
            f"select {', '.join(SNAPSHOT_COLUMNS)} from run_ticker_snapshot where run_id = ? and ticker = ?",
            [run_id, ticker],
        ).fetchall()
        ranking_rows = _execute(
            self._read_connection(),
            f"select {', '.join(RANKING_COLUMNS)} from run_rankings where run_id = ? and ticker = ? order by horizon",
            [run_id, ticker],
        ).fetchall()

        if not snapshot_rows and not ranking_rows:
            raise NotFoundError(f"ticker not in this run's persisted candidate set: {ticker}")
        if len(snapshot_rows) != 1:
            raise DataIntegrityError(f"inspect rows are inconsistent for run {run_id} ticker {ticker}")

        snapshot = _rows_to_dicts(SNAPSHOT_COLUMNS, snapshot_rows)[0]
        snapshot_metrics = _parse_metrics_json(snapshot.pop("metrics_json"), "snapshot")
        missing_snapshot_metrics = set(profile.snapshot_metric_keys) - set(snapshot_metrics)
        if missing_snapshot_metrics:
            raise DataIntegrityError(
                f"inspect snapshot metrics are missing: {', '.join(sorted(missing_snapshot_metrics))}"
            )
        snapshot.update(snapshot_metrics)

        rankings = []
        for row in _rows_to_dicts(RANKING_COLUMNS, ranking_rows):
            ranking_metrics = _parse_metrics_json(row.pop("metrics_json"), "ranking")
            missing_ranking_metrics = set(profile.ranking_metric_keys) - set(ranking_metrics)
            if missing_ranking_metrics:
                raise DataIntegrityError(
                    f"inspect ranking metrics are missing: {', '.join(sorted(missing_ranking_metrics))}"
                )
            row.update(ranking_metrics)
            rankings.append(row)

        horizons = {str(row["horizon"]) for row in rankings}
        expected_horizons = set(profile.horizon_order)
        if horizons != expected_horizons or len(rankings) != len(expected_horizons):
            raise DataIntegrityError(f"inspect horizon rows are inconsistent for run {run_id} ticker {ticker}")

        return InspectPayload(
            metadata=metadata,
            snapshot=snapshot,
            rankings=rankings,
        )
