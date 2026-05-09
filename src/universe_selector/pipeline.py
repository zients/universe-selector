from __future__ import annotations

from dataclasses import dataclass

from universe_selector.config import AppConfig, ensure_runtime_dirs
from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.identifiers import make_run_id
from universe_selector.locking import batch_lock
from universe_selector.output.report import render_markdown_report
from universe_selector.persistence.repository import DuckDbRepository
from universe_selector.persistence.schema import apply_migrations
from universe_selector.providers.base import MarketDataProvider
from universe_selector.providers.fixture import FixtureProvider


@dataclass(frozen=True)
class BatchResult:
    run_id: str
    market: Market


def _provider_for(config: AppConfig) -> MarketDataProvider:
    if config.data_mode == "fixture":
        return FixtureProvider(config.fixture_dir)

    from universe_selector.providers.live import LiveMarketDataProvider

    return LiveMarketDataProvider(config)


def run_batch(market: Market, config: AppConfig) -> BatchResult:
    ensure_runtime_dirs(config)
    with batch_lock(config.lock_path):
        repo = DuckDbRepository(config.duckdb_path)
        apply_migrations(repo.connect())

        run_id = make_run_id(market)
        repo.create_running_run(run_id, market, config)
        try:
            provider = _provider_for(config)
            provider_data = provider.load_run_data(run_id, market)
            if not provider_data.listings:
                raise ProviderDataError("listing provider returned no usable listings")
            if provider_data.bars.is_empty():
                raise ProviderDataError("OHLCV provider returned no usable bars")

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
            provider_summary = {
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
            markdown = render_markdown_report(
                run_id=run_id,
                market=market,
                mode_label=config.data_mode,
                provider_summary=provider_summary,
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
            return BatchResult(run_id=run_id, market=market)
        except Exception as exc:
            repo.mark_failed_run(run_id, str(exc))
            raise
