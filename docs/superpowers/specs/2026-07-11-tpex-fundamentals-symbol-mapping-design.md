# TPEX Fundamentals Symbol Mapping Design

## Goal

Make the live Yahoo Finance universe-fundamentals request for every Taiwan TPEX listing use the `.TWO` suffix while preserving the canonical ticker stored in returned facts.

## Scope

This change is limited to request-symbol selection inside `YFinanceFundamentalsProvider.load_fundamentals_for_listings()`.

- TWSE universe listings continue to request `{ticker}.TW`.
- TPEX universe listings request `{ticker}.TWO`.
- Returned universe facts continue to store the canonical listing ticker without a Yahoo Finance suffix.
- Coverage counts, normalization, missing/invalid handling, metadata, ranking, persistence, and command boundaries remain unchanged.
- Single-ticker `load_fundamentals()` behavior remains unchanged because it has no listing exchange segment with which to distinguish TWSE from TPEX.
- US symbol mapping remains unchanged.

Listing completeness, ADR handling, scoring, and performance are explicitly out of scope.

## Root Cause

`load_fundamentals_for_listings()` currently filters the incoming `ListingCandidate` values into a list of ticker strings before building Yahoo Finance requests. That discards `ListingCandidate.exchange_segment`, so the ticker-only `_request_symbol()` helper applies its default Taiwan `.TW` suffix to both TWSE and TPEX listings.

The OHLCV provider demonstrates the required working pattern: retain each `ListingCandidate` until request-symbol selection, then map `TWSE` to `.TW` and `TPEX` to `.TWO`.

## Considered Approaches

1. Add a small listing-aware helper for universe fundamentals and delegate unchanged cases to the existing ticker-only helper. This is the selected approach because it fixes the information-loss boundary while keeping the change local.
2. Extract a shared Yahoo Finance symbol-mapping module for OHLCV and fundamentals. This could reduce duplication, but it would broaden the refactor and risk changing US filtering or single-ticker behavior.
3. Add an inline TPEX conditional inside the universe loop. This would be a smaller textual diff, but it would mix symbol policy with fetching and be harder to test and maintain.

## Design

`load_fundamentals_for_listings()` will retain the market-matching `ListingCandidate` objects instead of reducing them to strings. For each listing it will:

1. Canonicalize `listing.ticker` exactly as today.
2. Select the Yahoo Finance request symbol with a private listing-aware helper.
3. For `Market.TW`, use the upper-cased `exchange_segment` to map `TWSE` to `.TW` and `TPEX` to `.TWO`.
4. For all non-TW behavior, delegate to the existing `_request_symbol()` helper so US class-share mapping is untouched.
5. Normalize the fetched data with the canonical ticker, so Yahoo Finance suffixes never enter persisted facts.

Unsupported TW exchange segments will raise the same `ProviderDataError` message pattern already used by the OHLCV provider. Because the universe loader already classifies `ProviderDataError` as invalid coverage, no coverage or error-handling policy changes are needed.

## Testing

A focused provider regression test will supply one TWSE listing and one TPEX listing through `load_fundamentals_for_listings()` using the existing fake `yfinance.Ticker` pattern. It will prove that:

- requests are `2330.TW` and `1240.TWO`;
- returned fact tickers remain `2330` and `1240`;
- both successful listings are reflected in the existing coverage counts.

The test must be observed failing before production code changes, then passing after the minimal implementation. Existing single-ticker TW and universe US mapping tests must remain green. Final verification will run the focused provider tests and all repository quality gates required by `AGENTS.md`.

## Acceptance Criteria

- Every TPEX listing passed to universe fundamentals uses `.TWO` for the Yahoo Finance request.
- TWSE universe requests still use `.TW`.
- Canonical tickers, US mappings, and single-ticker behavior are unchanged.
- No changes are made to listing completeness, ADR handling, ranking/scoring, persistence schema, or performance.
- The focused regression test and the full repository gates pass.
