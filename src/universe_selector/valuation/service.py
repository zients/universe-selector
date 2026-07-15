from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from universe_selector.domain import Market, canonical_ticker
from universe_selector.errors import ProviderDataError, ValidationError
from universe_selector.providers.context import build_provider_run_context
from universe_selector.providers.models import ListingCandidate
from universe_selector.providers.registry import get_fundamentals_registration, get_listing_registration
from universe_selector.valuation.assumptions import load_valuation_assumptions
from universe_selector.valuation.models import (
    ValuationModelInput,
    ValuationResult,
    ValuationRunInput,
)
from universe_selector.valuation.registry import get_valuation_model


def validate_value_ticker(market: Market, ticker: str) -> None:
    if market is Market.TW and ticker.endswith((".TW", ".TWO")):
        raise ValidationError(f"TW value ticker must be a canonical bare ticker without provider suffix: {ticker}")


def _select_listing(
    market: Market,
    ticker: str,
    listings: list[ListingCandidate],
) -> ListingCandidate:
    matches: list[ListingCandidate] = []
    for listing in listings:
        if listing.market is not market:
            continue
        try:
            listing_ticker = canonical_ticker(listing.ticker)
        except ValidationError as exc:
            raise ProviderDataError(f"listing provider returned invalid ticker for {market.value}") from exc
        if listing_ticker == ticker:
            matches.append(listing)
    if len(matches) != 1:
        raise ProviderDataError(
            f"expected exactly one listing for {market.value} ticker {ticker}; found {len(matches)}"
        )
    return matches[0]


def run_valuation(
    market: Market,
    ticker: str,
    model_id: str | None,
    assumptions_path: Path | None,
    fundamentals_provider_id: str,
    *,
    listing_provider_id: str | None = None,
) -> ValuationResult:
    normalized_ticker = canonical_ticker(ticker)
    validate_value_ticker(market, normalized_ticker)
    model = get_valuation_model(model_id) if model_id is not None else None
    fundamentals_registration = get_fundamentals_registration(fundamentals_provider_id, market)

    listing_registration = None
    if market is Market.TW:
        if listing_provider_id is None:
            raise ValidationError("TW value requires a configured listing provider")
        listing_registration = get_listing_registration(listing_provider_id, market)

    assumptions = load_valuation_assumptions(
        market=market,
        ticker=normalized_ticker,
        model_id=model_id,
        assumptions_path=assumptions_path,
    )
    model = model or get_valuation_model(assumptions.model_id)
    fundamentals_provider = fundamentals_registration.factory()

    if listing_registration is None:
        fundamentals = fundamentals_provider.load_fundamentals(market, normalized_ticker)
    else:
        context = build_provider_run_context(
            market=market,
            data_fetch_started_at=datetime.now(timezone.utc),
            ticker_limit=None,
        )
        listing_provider = listing_registration.factory(None)
        listing = _select_listing(
            market,
            normalized_ticker,
            listing_provider.load_listings(context, market),
        )
        fundamentals = fundamentals_provider.load_fundamentals(
            market,
            normalized_ticker,
            listing=listing,
        )

    facts = fundamentals.facts
    if assumptions.currency != facts.currency:
        raise ValidationError(
            f"assumptions currency {assumptions.currency} must match provider facts currency {facts.currency}"
        )
    effective_inputs, provenance = model.build_inputs(facts=facts, assumptions=assumptions)
    run_input = ValuationRunInput(
        market=market,
        ticker=normalized_ticker,
        model_id=assumptions.model_id,
        fundamentals_metadata=fundamentals.metadata,
        raw_facts=facts,
        effective_inputs=effective_inputs,
        input_provenance=provenance,
        assumptions=assumptions,
    )
    scenario_results = model.value(
        ValuationModelInput(
            market=market,
            ticker=normalized_ticker,
            model_id=assumptions.model_id,
            effective_inputs=effective_inputs,
            model_assumptions=assumptions.model_assumptions,
        )
    )
    return ValuationResult(run_input=run_input, scenario_results=scenario_results)
