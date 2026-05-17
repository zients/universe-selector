from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Annotated, TypeVar

import typer

from universe_selector.config import AppConfig, load_config, load_live_fundamentals_provider_id
from universe_selector.domain import Market, canonical_market, canonical_ticker
from universe_selector.errors import NotFoundError, UniverseSelectorError, ValidationError
from universe_selector.identifiers import parse_run_id
from universe_selector.output.inspect import render_inspect
from universe_selector.output.valuation import render_valuation_markdown
from universe_selector.persistence.repository import DuckDbRepository, ResolvedRun
from universe_selector.persistence.schema import validate_schema
from universe_selector.pipeline import BatchResult, MultiProfileBatchError, run_batch, run_batch_profiles
from universe_selector.ranking_profiles import get_ranking_profile
from universe_selector.valuation.registry import get_valuation_model
from universe_selector.valuation.service import run_valuation


app = typer.Typer(no_args_is_help=True)
T = TypeVar("T")


def _exit_with_error(exc: Exception) -> None:
    if isinstance(exc, UniverseSelectorError):
        message = str(exc)
        code = exc.exit_code
    else:
        message = str(exc) or exc.__class__.__name__
        code = UniverseSelectorError.exit_code
    typer.echo(message)
    raise typer.Exit(code=code)


def _guard(action: Callable[[], T]) -> T:
    try:
        return action()
    except typer.Exit:
        raise
    except Exception as exc:
        _exit_with_error(exc)
        raise


def _read_repo(config: AppConfig) -> DuckDbRepository:
    repo = DuckDbRepository(config.duckdb_path)
    validate_schema(repo.connect(read_only=True))
    return repo


def _config_with_cli_overrides(
    config: AppConfig,
    *,
    ranking_profile: str | None = None,
) -> AppConfig:
    updates: dict[str, object] = {}
    if ranking_profile is not None:
        updates["ranking_profile"] = ranking_profile
    if not updates:
        return config
    overridden = replace(config, **updates)
    overridden.validate()
    return overridden


def _validate_read_resolution_request(
    *,
    market: str | None,
    run_id: str | None,
    ranking_profile: str | None = None,
) -> None:
    if market and run_id:
        raise ValidationError("provide either MARKET or --run-id, not both")
    if run_id is not None and ranking_profile is not None:
        raise ValidationError("do not provide --ranking-profile with --run-id")


def _resolution_request(market: str | None, run_id: str | None) -> tuple[str, str | Market]:
    if market and run_id:
        raise ValidationError("provide either MARKET or --run-id, not both")
    if not market and not run_id:
        raise ValidationError("provide MARKET or --run-id")
    if run_id:
        parsed = parse_run_id(run_id)
        return "explicit run_id", parsed.run_id

    resolved_market = canonical_market(market or "")
    return "resolved latest successful run", resolved_market


def _resolve_run(
    repo: DuckDbRepository,
    resolution_mode: str,
    target: str | Market,
    *,
    ranking_profile: str | None,
) -> ResolvedRun:
    if resolution_mode == "explicit run_id":
        return repo.resolve_successful_run(target)

    assert isinstance(target, Market)
    return repo.resolve_latest_successful_run(target, ranking_profile=ranking_profile)


def _resolve_readable_run(
    repo: DuckDbRepository,
    resolution_mode: str,
    target: str | Market,
    *,
    ranking_profile: str | None,
) -> ResolvedRun:
    try:
        return _resolve_run(repo, resolution_mode, target, ranking_profile=ranking_profile)
    except NotFoundError as exc:
        if resolution_mode == "explicit run_id":
            raise NotFoundError(f"No readable successful run found for run_id {target}") from exc
        assert isinstance(target, Market)
        raise NotFoundError(f"No readable successful run found for market {target.value}") from exc


def _profile_overrides(values: list[str] | None) -> tuple[str, ...]:
    return tuple(values or ())


def _validate_multi_profile_overrides(profile_ids: tuple[str, ...]) -> None:
    seen: set[str] = set()
    for profile_id in profile_ids:
        if profile_id in seen:
            raise ValidationError(f"duplicate ranking profile {profile_id}")
        seen.add(profile_id)
        get_ranking_profile(profile_id).validate()


def _echo_batch_result(result: BatchResult, *, include_profile: bool) -> None:
    typer.echo(f"run_id: {result.run_id}")
    if include_profile:
        typer.echo(f"ranking_profile: {result.ranking_profile}")


def _echo_batch_results(results: tuple[BatchResult, ...], *, include_profile: bool) -> None:
    for result in results:
        _echo_batch_result(result, include_profile=include_profile)


def _echo_batch_results_with_market(results: tuple[BatchResult, ...], *, include_profile: bool) -> None:
    _echo_batch_results(results, include_profile=include_profile)
    if results:
        typer.echo(f"market: {results[0].market.value}")


@app.command()
def batch(
    market: Annotated[str, typer.Argument()],
    ranking_profile: Annotated[list[str] | None, typer.Option("--ranking-profile")] = None,
) -> None:
    def action() -> None:
        config = load_config()
        overrides = _profile_overrides(ranking_profile)
        resolved_market = canonical_market(market)
        if len(overrides) == 0:
            result = run_batch(resolved_market, config)
            _echo_batch_result(result, include_profile=False)
            typer.echo(f"market: {result.market.value}")
            return
        if len(overrides) == 1:
            profile_config = _config_with_cli_overrides(config, ranking_profile=overrides[0])
            result = run_batch(resolved_market, profile_config)
            _echo_batch_result(result, include_profile=False)
            typer.echo(f"market: {result.market.value}")
            return

        _validate_multi_profile_overrides(overrides)
        try:
            results = run_batch_profiles(resolved_market, config, overrides)
        except MultiProfileBatchError as exc:
            _echo_batch_results(exc.completed_results, include_profile=True)
            typer.echo(f"failed_run_id: {exc.failed_result.run_id}")
            typer.echo(f"failed_ranking_profile: {exc.failed_result.ranking_profile}")
            typer.echo(f"error: {exc.failed_result.error_message}")
            typer.echo(f"market: {exc.failed_result.market.value}")
            raise typer.Exit(code=exc.exit_code)
        _echo_batch_results_with_market(results, include_profile=True)

    _guard(action)


@app.command()
def report(
    market: Annotated[str | None, typer.Argument()] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    ranking_profile: Annotated[str | None, typer.Option("--ranking-profile")] = None,
) -> None:
    def action() -> None:
        _validate_read_resolution_request(market=market, run_id=run_id, ranking_profile=ranking_profile)
        resolution_mode, target = _resolution_request(market, run_id)
        config = _config_with_cli_overrides(load_config(), ranking_profile=ranking_profile)
        resolution_ranking_profile = None if resolution_mode == "explicit run_id" else config.ranking_profile
        repo = _read_repo(config)
        resolved = _resolve_readable_run(
            repo,
            resolution_mode,
            target,
            ranking_profile=resolution_ranking_profile,
        )
        markdown = repo.read_report_markdown(resolved.run_id)
        typer.echo(f"resolution mode: {resolution_mode}")
        typer.echo(f"run_id: {resolved.run_id}")
        typer.echo(markdown, nl=False)

    _guard(action)


@app.command()
def inspect(
    ticker: Annotated[str, typer.Option("--ticker")],
    market: Annotated[str | None, typer.Argument()] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    ranking_profile: Annotated[str | None, typer.Option("--ranking-profile")] = None,
) -> None:
    def action() -> None:
        _validate_read_resolution_request(market=market, run_id=run_id, ranking_profile=ranking_profile)
        normalized_ticker = canonical_ticker(ticker)
        resolution_mode, target = _resolution_request(market, run_id)
        config = _config_with_cli_overrides(load_config(), ranking_profile=ranking_profile)
        resolution_ranking_profile = None if resolution_mode == "explicit run_id" else config.ranking_profile
        repo = _read_repo(config)
        resolved = _resolve_readable_run(
            repo,
            resolution_mode,
            target,
            ranking_profile=resolution_ranking_profile,
        )
        profile = get_ranking_profile(resolved.ranking_profile)
        try:
            payload = repo.read_inspect_payload(
                resolved.run_id,
                normalized_ticker,
                profile=profile,
            )
        except NotFoundError as exc:
            raise NotFoundError(
                f"Normalized ticker {normalized_ticker} is not in this run's persisted candidate set"
            ) from exc
        typer.echo(
            render_inspect(
                run_id=resolved.run_id,
                resolution_mode=resolution_mode,
                ticker=normalized_ticker,
                metadata=payload.metadata,
                snapshot=payload.snapshot,
                rankings=payload.rankings,
                profile=profile,
            ),
            nl=False,
        )

    _guard(action)


@app.command()
def value(
    market: Annotated[str, typer.Argument()],
    ticker: Annotated[str, typer.Option("--ticker")],
    model: Annotated[str, typer.Option("--model")] = "fcf_dcf_v1",
    assumptions: Annotated[Path | None, typer.Option("--assumptions")] = None,
) -> None:
    def action() -> None:
        resolved_market = canonical_market(market)
        normalized_ticker = canonical_ticker(ticker)
        get_valuation_model(model)
        result = run_valuation(
            market=resolved_market,
            ticker=normalized_ticker,
            model_id=model,
            assumptions_path=assumptions,
            fundamentals_provider_id=load_live_fundamentals_provider_id(),
        )
        typer.echo(render_valuation_markdown(result), nl=False)

    _guard(action)
