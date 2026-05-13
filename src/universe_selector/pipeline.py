from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from universe_selector.config import AppConfig, ensure_runtime_dirs
from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError, UniverseSelectorError, ValidationError
from universe_selector.identifiers import make_run_id
from universe_selector.locking import batch_lock
from universe_selector.output.report import render_markdown_report
from universe_selector.persistence.repository import DuckDbRepository
from universe_selector.persistence.schema import apply_migrations
from universe_selector.providers.base import MarketDataProvider
from universe_selector.providers.fixture import FixtureProvider
from universe_selector.providers.models import ProviderMetadata, ProviderRunData
from universe_selector.ranking_profiles import get_ranking_profile


@dataclass(frozen=True)
class BatchResult:
    run_id: str
    market: Market
    ranking_profile: str


@dataclass(frozen=True)
class FailedBatchResult:
    run_id: str
    market: Market
    ranking_profile: str
    error_message: str


class MultiProfileBatchError(UniverseSelectorError):
    def __init__(
        self,
        *,
        completed_results: tuple[BatchResult, ...],
        failed_result: FailedBatchResult,
        exit_code: int,
    ) -> None:
        super().__init__(failed_result.error_message)
        self.completed_results = completed_results
        self.failed_result = failed_result
        self.exit_code = exit_code


def _provider_for(config: AppConfig) -> MarketDataProvider:
    if config.data_mode == "fixture":
        return FixtureProvider(config.fixture_dir)

    from universe_selector.providers.live import LiveMarketDataProvider

    return LiveMarketDataProvider(config)


def _validate_provider_data(provider_data: ProviderRunData) -> None:
    if not provider_data.listings:
        raise ProviderDataError("listing provider returned no usable listings")
    if provider_data.bars.is_empty():
        raise ProviderDataError("OHLCV provider returned no usable bars")


def _provider_summary(metadata: ProviderMetadata, config: AppConfig) -> dict[str, str]:
    return {
        "data_mode": metadata.data_mode,
        "listing_provider_id": metadata.listing_provider_id,
        "listing_source_id": metadata.listing_source_id,
        "ohlcv_provider_id": metadata.ohlcv_provider_id,
        "ohlcv_source_id": metadata.ohlcv_source_id,
        "provider_config_hash": metadata.provider_config_hash,
        "data_fetch_started_at": metadata.data_fetch_started_at.isoformat(),
        "market_timezone": metadata.market_timezone,
        "run_latest_bar_date": metadata.run_latest_bar_date.isoformat(),
        "ranking_profile": config.ranking_profile,
        "ranking_config_hash": config.ranking_config_hash(),
    }


def _validate_profile_ids(profile_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(profile_id) for profile_id in profile_ids)
    if len(normalized) < 2:
        raise ValidationError("provide at least two ranking profiles for multi-profile batch")

    seen: set[str] = set()
    for profile_id in normalized:
        if profile_id in seen:
            raise ValidationError(f"duplicate ranking profile {profile_id}")
        seen.add(profile_id)
        get_ranking_profile(profile_id).validate()
    return normalized


def _run_profile_from_provider_data(
    *,
    repo: DuckDbRepository,
    run_id: str,
    market: Market,
    config: AppConfig,
    provider_data: ProviderRunData,
    create_running: bool = True,
) -> BatchResult:
    try:
        if create_running:
            repo.create_running_run(run_id, market, config)

        profile = config.selected_ranking_profile
        snapshot = profile.build_snapshot(
            run_id=run_id,
            market=market,
            listings=provider_data.listings,
            bars=provider_data.bars,
            run_latest_bar_date=provider_data.metadata.run_latest_bar_date,
        )
        rankings = profile.assign_rankings(snapshot)
        metadata = provider_data.metadata
        markdown = render_markdown_report(
            run_id=run_id,
            market=market,
            mode_label=config.data_mode,
            provider_summary=_provider_summary(metadata, config),
            snapshot=snapshot,
            rankings=rankings,
            config=config,
            profile=profile,
        )
        repo.mark_successful_run(
            run_id=run_id,
            metadata=metadata,
            snapshot=snapshot,
            rankings=rankings,
            markdown=markdown,
        )
        return BatchResult(run_id=run_id, market=market, ranking_profile=config.ranking_profile)
    except Exception as exc:
        repo.mark_failed_run(run_id, str(exc) or exc.__class__.__name__)
        raise


def run_batch(market: Market, config: AppConfig) -> BatchResult:
    ensure_runtime_dirs(config)
    with batch_lock(config.lock_path):
        repo = DuckDbRepository(config.duckdb_path)
        apply_migrations(repo.connect())

        run_id = make_run_id(market)
        repo.create_running_run(run_id, market, config)
        try:
            provider = _provider_for(config)
            provider_data = provider.load_run_data(market)
            _validate_provider_data(provider_data)
        except Exception as exc:
            repo.mark_failed_run(run_id, str(exc))
            raise

        return _run_profile_from_provider_data(
            repo=repo,
            run_id=run_id,
            market=market,
            config=config,
            provider_data=provider_data,
            create_running=False,
        )


def run_batch_profiles(
    market: Market,
    base_config: AppConfig,
    profile_ids: tuple[str, ...],
) -> tuple[BatchResult, ...]:
    normalized_profile_ids = _validate_profile_ids(profile_ids)
    profile_configs = tuple(replace(base_config, ranking_profile=profile_id) for profile_id in normalized_profile_ids)
    for profile_config in profile_configs:
        profile_config.validate()

    ensure_runtime_dirs(base_config)
    with batch_lock(base_config.lock_path):
        repo = DuckDbRepository(base_config.duckdb_path)
        try:
            apply_migrations(repo.connect())
            provider = _provider_for(base_config)
            provider_data = provider.load_run_data(market)
            _validate_provider_data(provider_data)

            completed: list[BatchResult] = []
            for config in profile_configs:
                run_id = make_run_id(market)
                try:
                    result = _run_profile_from_provider_data(
                        repo=repo,
                        run_id=run_id,
                        market=market,
                        config=config,
                        provider_data=provider_data,
                    )
                except Exception as exc:
                    exit_code = exc.exit_code if isinstance(exc, UniverseSelectorError) else UniverseSelectorError.exit_code
                    raise MultiProfileBatchError(
                        completed_results=tuple(completed),
                        failed_result=FailedBatchResult(
                            run_id=run_id,
                            market=market,
                            ranking_profile=config.ranking_profile,
                            error_message=str(exc) or exc.__class__.__name__,
                        ),
                        exit_code=exit_code,
                    ) from exc
                completed.append(result)
            return tuple(completed)
        finally:
            repo.close()
