# Provider-Neutral TW Single-Ticker Valuation Resolution Design

## Goal

Make `value tw --ticker <canonical bare ticker>` use the same public Taiwan
ticker identity as `inspect`, while resolving the exact listing through the
configured live listing provider before loading fundamentals. Provider-specific
symbols such as Yahoo Finance `.TW` and `.TWO` suffixes must remain inside the
fundamentals adapter.

## Public Contract

The command boundary identifies a security with the market argument and a
canonical ticker:

```text
market=tw, ticker=2330
market=tw, ticker=6488
```

`value tw --ticker 2330` and `inspect tw --ticker 2330` therefore use the same
ticker identity. Yahoo Finance request symbols such as `2330.TW` and
`6488.TWO` are transport details, not public tickers, assumptions keys, or
persisted identities.

For `Market.TW`, provider-suffixed inputs such as `2330.TW` and `6488.TWO` are
invalid at the `value` boundary. The error directs the user to pass the
canonical bare ticker. This is an explicit provider-neutral contract, not a
compatibility fallback.

## Scope

This change is limited to live, ephemeral single-ticker valuation resolution:

- For TW, `value` reads the configured listing provider in addition to the
  configured fundamentals provider.
- A TW bare ticker is resolved to one exact `ListingCandidate` before the
  fundamentals request.
- The single-ticker fundamentals contract receives the canonical ticker and
  resolved listing identity, allowing each adapter to select its own request
  symbol.
- Yahoo Finance maps `TWSE` to `.TW` and `TPEX` to `.TWO`, while returned facts
  retain the canonical bare ticker.
- The default assumptions path remains
  `valuation_assumptions/tw/{canonical_ticker}.yaml`.
- US single-ticker valuation behavior remains unchanged.

`batch`, `report`, `inspect`, ranking profiles, DuckDB schemas, persisted
tickers, report artifacts, and ranking provider metadata are unchanged.
`value` remains live and ephemeral and must not read or write ranking runs.

## Root Cause

Universe fundamentals loading already retains each `ListingCandidate` and can
map `TWSE` and `TPEX` correctly. The single-ticker valuation path instead calls
`FundamentalsProvider.load_fundamentals(market, ticker)` without listing
identity. `YFinanceFundamentalsProvider` therefore defaults every bare Taiwan
ticker to `.TW`, which is wrong for TPEX securities.

The ticker alone cannot reliably determine whether a Taiwan security trades on
TWSE or TPEX. Trying both Yahoo suffixes would make provider errors ambiguous
and couple valuation orchestration to Yahoo Finance.

## Considered Approaches

1. Resolve the canonical ticker through the configured listing provider and
   pass the resulting `ListingCandidate` to the fundamentals adapter. This is
   selected because it preserves the project identity boundary, reuses the
   authoritative listing source, and remains fundamentals-provider-neutral.
2. Require a complete Yahoo Finance symbol at the CLI. This avoids a listing
   request but makes `value` inconsistent with `inspect`, leaks one provider's
   transport format into assumptions and user workflows, and obstructs future
   provider replacement.
3. Probe `.TW` and then `.TWO`. This is rejected because it adds provider calls,
   cannot cleanly distinguish a missing security from a provider outage or
   malformed response, and embeds Yahoo-specific policy in orchestration.

## Design

### Minimal live-value configuration

The existing value-only configuration loader will be replaced or extended by a
typed minimal loader that reads only:

- `live.fundamentals_provider` for every market; and
- `live.listing_provider.TW` when the requested market is TW.

For TW, both registrations are validated for the requested market. US retains
its current fundamentals-only minimal configuration contract. `value` must not
load or validate unrelated ranking, report, DuckDB, lock, fixture, OHLCV, or
batch ticker-limit settings. Provider factories receive only the configuration
needed by this live-value path.

Explicit model validation remains before configuration loading, preserving the
current fail-fast behavior for an unknown `--model`.

### Resolution and request flow

`run_valuation` will perform these steps in order:

1. Canonicalize the input ticker and reject a provider-suffixed TW ticker.
2. Validate the selected listing and fundamentals registrations for the market.
3. Load and validate the valuation assumptions, including the selected model.
4. Build a live provider context with `ticker_limit=None`.
5. Load listings from the configured listing provider and select the one exact
   candidate whose market and canonical ticker match the request.
6. Pass the canonical ticker and resolved `ListingCandidate` to the configured
   fundamentals provider.
7. Validate currency and run the valuation model exactly as today.

Assumptions must be validated before either listing or fundamentals network
access. The batch-only `live.ticker_limit` must not be applied to exact
single-ticker resolution because the target could fall outside an arbitrary
prefix of the listing set.

US valuation continues through its existing ticker-only request behavior and
does not gain a listing lookup as part of this TW-focused change.

### Listing resolution

Resolution considers only candidates whose `ListingCandidate.market` equals the
requested market and whose canonical ticker exactly equals the requested
ticker. Exactly one match is required.

- No match raises a provider-data error naming the market and canonical ticker.
- Multiple matches raise a provider-data error instead of selecting
  arbitrarily.
- The service never guesses an exchange segment and never probes provider
  symbols.

The configured listing provider remains responsible for listing status and
instrument-type policy. The valuation service does not introduce a second set
of listing filters.

### Fundamentals provider boundary

The single-ticker fundamentals interface will accept resolved listing identity
for TW requests. A TW request without a matching `ListingCandidate` fails
closed; the provider no longer silently defaults a bare Taiwan ticker to
`.TW`.

`YFinanceFundamentalsProvider` will reuse the existing listing-aware mapping
used by universe fundamentals:

- `exchange_segment=TWSE` requests `{ticker}.TW`;
- `exchange_segment=TPEX` requests `{ticker}.TWO`; and
- an unsupported TW segment raises `ProviderDataError`.

Normalization continues to receive the canonical ticker, so the suffix never
enters `FundamentalFacts.ticker`, valuation output, assumptions identity, or
persistence.

### Output and command boundaries

The valuation markdown and JSON schemas remain unchanged. Their ticker is the
canonical bare ticker. This change adds no DuckDB columns and performs no
ranking-run lookup or persistence.

## Error Handling

The following failures are explicit and do not trigger fallback requests:

- provider-suffixed TW input;
- missing or unsupported live TW listing-provider configuration;
- missing or unsupported fundamentals-provider configuration;
- missing, duplicate, or mismatched listing identity;
- unsupported TW exchange segment;
- existing fundamentals normalization, currency, assumptions, and model
  validation errors.

An unknown explicit model still fails before configuration access. Valid
provider configuration is checked before assumptions loading, while assumptions
loading completes before any network access.

## Testing

Implementation follows test-driven development. Tests must first demonstrate
the current failure, then prove:

- minimal TW value configuration loads and validates both provider selections
  without requiring unrelated application configuration, while US remains
  fundamentals-only;
- `value tw --ticker 2330` resolves a TWSE listing and Yahoo requests
  `2330.TW`;
- `value tw --ticker 6488` resolves a TPEX listing and Yahoo requests
  `6488.TWO`;
- returned facts, assumptions identity, and valuation output retain `2330` or
  `6488` without provider suffixes;
- provider-suffixed TW input fails with a canonical-ticker validation message;
- missing and duplicate listing matches fail before fundamentals access;
- invalid or missing assumptions fail before listing and fundamentals network
  access;
- single-ticker resolution ignores the batch ticker limit;
- US ticker and class-share request mapping remains unchanged;
- `value` does not read or write ranking persistence; and
- existing universe fundamentals, ranking, report, and inspect behavior remains
  green.

Targeted tests cover configuration, CLI orchestration, valuation service, and
the Yahoo fundamentals adapter. Final verification runs all repository quality
gates required by `AGENTS.md`.

## Acceptance Criteria

- `value tw --ticker <bare ticker>` and `inspect tw --ticker <bare ticker>` use
  the same canonical public identity.
- TWSE and TPEX value requests are selected from configured listing identity,
  not inferred from the ticker or discovered by probing Yahoo Finance.
- The fundamentals provider, not CLI or valuation orchestration, owns its
  transport-symbol mapping.
- Yahoo requests TWSE as `.TW` and TPEX as `.TWO`, while all project-facing
  ticker values remain canonical and suffix-free.
- Missing or ambiguous listing identity and unsupported exchange segments fail
  closed without fallback.
- Assumptions are validated before live provider access.
- `value` stays live and ephemeral; ranking persistence and read-command
  boundaries are unchanged.
- Targeted tests and the complete format, lint, type-check, and pytest gates
  pass.
