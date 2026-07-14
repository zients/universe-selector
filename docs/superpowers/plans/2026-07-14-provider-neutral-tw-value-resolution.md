# Provider-Neutral TW Value Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve a canonical bare Taiwan `value` ticker through the configured listing provider and give the resolved listing identity to the fundamentals adapter, without exposing Yahoo suffixes or changing ranking persistence.

**Architecture:** Add a minimal value-only provider selection that conditionally includes a TW listing provider. Extend the single-ticker fundamentals boundary with optional listing identity, resolve one exact TW `ListingCandidate` in valuation orchestration after assumptions validation, and keep transport-symbol mapping inside the Yahoo adapter. US and all persisted ranking flows retain their existing behavior.

**Tech Stack:** Python 3.14, dataclasses, Typer, provider protocols/registrations, pytest, Ruff, mypy, uv.

---

## Completion Status (2026-07-15)

Remote branch: `origin/feature/value-tw-listing-resolution`

Implementation checkpoint: `096666dcd0e7cd909ab7c554145a172d54f34009`

- Task 1 is complete. Minimal value-provider selection is implemented and has
  passed independent spec-compliance and code-quality reviews.
- Task 2 is complete. Single-ticker fundamentals requires TW listing identity;
  Yahoo maps TWSE to `.TW` and TPEX to `.TWO`. Its review finding was fixed and
  re-reviewed successfully.
- Task 3 is complete. Valuation orchestration validates a bare TW ticker,
  resolves one exact listing after assumptions validation, and passes the
  listing to fundamentals. Its resumed independent spec-compliance and fresh
  code-quality reviews both passed.
- Task 4 is complete. The CLI uses `load_live_value_provider_selection`, passes
  both provider IDs to `run_valuation`, preserves explicit model validation
  before configuration access, removes the legacy minimal loader, and documents
  the provider-neutral Taiwan ticker contract. Its TDD implementation passed
  spec review; two code-quality test findings were fixed and re-reviewed.
- The final independent whole-branch review found one minor provider-error
  classification issue. Commit `096666d` fixed it with a focused RED/GREEN
  regression, and final re-review reported no remaining findings.
- Final controller verification passed Ruff formatting, Ruff checks, mypy, and
  the full `712 passed` pytest suite.

The implementation is complete and `value tw --ticker <canonical bare ticker>`
is wired end to end while remaining live and ephemeral. No ranking persistence,
report/inspect, or DuckDB schema boundary changed.

## File Structure

- `src/universe_selector/config.py`: define and load the minimal provider selection used only by `value`.
- `src/universe_selector/providers/base.py`: declare optional resolved listing identity on the single-ticker fundamentals protocol.
- `src/universe_selector/providers/yfinance_fundamentals.py`: require listing identity for TW and own `.TW`/`.TWO` request mapping.
- `src/universe_selector/valuation/service.py`: validate canonical TW input, resolve one listing, and pass it to fundamentals after assumptions validation.
- `src/universe_selector/cli.py`: wire the minimal selection into `run_valuation` without loading full application config.
- `docs/valuation.md` and `README.md`: document provider-neutral canonical TW ticker behavior.
- `tests/test_config.py`, `tests/test_yfinance_fundamentals_provider.py`, `tests/test_valuation_service.py`, `tests/test_cli.py`, and `tests/test_readme_contract.py`: offline regression coverage for each boundary.

### Task 1: Minimal live-value provider selection

**Files:**
- Modify: `src/universe_selector/config.py:12-19,238-250`
- Test: `tests/test_config.py:1-20,167-176`

- [ ] **Step 1: Write failing configuration tests**

Replace the old minimal-loader import and test with the following selection tests. They prove that TW requires only its listing provider plus fundamentals, while US remains fundamentals-only:

```python
from universe_selector.config import (
    AppConfig,
    LiveValueProviderSelection,
    load_config,
    load_live_value_provider_selection,
)


def test_load_live_value_provider_selection_reads_minimal_tw_config(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        """
live:
  listing_provider:
    TW: twse_isin
  fundamentals_provider: yfinance_fundamentals
""".lstrip()
    )
    monkeypatch.chdir(tmp_path)

    assert load_live_value_provider_selection(Market.TW) == LiveValueProviderSelection(
        fundamentals_provider_id="yfinance_fundamentals",
        listing_provider_id="twse_isin",
    )


def test_load_live_value_provider_selection_keeps_us_fundamentals_only(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        """
live:
  fundamentals_provider: yfinance_fundamentals
""".lstrip()
    )
    monkeypatch.chdir(tmp_path)

    assert load_live_value_provider_selection(Market.US) == LiveValueProviderSelection(
        fundamentals_provider_id="yfinance_fundamentals",
        listing_provider_id=None,
    )


def test_load_live_value_provider_selection_requires_tw_listing_provider(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        """
live:
  fundamentals_provider: yfinance_fundamentals
""".lstrip()
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="config missing required key: live.listing_provider.TW"):
        load_live_value_provider_selection(Market.TW)


@pytest.mark.parametrize(
    ("config_text", "message"),
    (
        (
            "live:\n  listing_provider:\n    TW: unknown\n  fundamentals_provider: yfinance_fundamentals\n",
            "unsupported listing provider for TW: unknown",
        ),
        (
            "live:\n  listing_provider:\n    TW: twse_isin\n  fundamentals_provider: unknown\n",
            "unsupported fundamentals provider for TW: unknown",
        ),
    ),
)
def test_load_live_value_provider_selection_validates_tw_registrations(
    monkeypatch,
    tmp_path: Path,
    config_text: str,
    message: str,
) -> None:
    (tmp_path / "config.yaml").write_text(config_text)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match=message):
        load_live_value_provider_selection(Market.TW)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_config.py -k live_value_provider_selection
```

Expected: collection fails because `LiveValueProviderSelection` and `load_live_value_provider_selection` do not exist.

- [ ] **Step 3: Implement the typed minimal loader**

Import the market-aware fundamentals registration resolver, add the frozen
selection type near `AppConfig`, and add the new loader alongside
`load_live_fundamentals_provider_id`. Keeping the old loader until Task 4 keeps
the CLI import valid between task commits.

```python
from universe_selector.providers.registry import (
    get_fundamentals_provider_registration,
    get_fundamentals_registration,
    get_listing_registration,
    get_ohlcv_registration,
)


@dataclass(frozen=True)
class LiveValueProviderSelection:
    fundamentals_provider_id: str
    listing_provider_id: str | None


def load_live_value_provider_selection(market: Market) -> LiveValueProviderSelection:
    loaded = _load_config_mapping()
    live = loaded.get("live")
    if not isinstance(live, dict):
        raise ValidationError("config key live must be a mapping")
    if "fundamentals_provider" not in live:
        raise ValidationError("config missing required key: live.fundamentals_provider")

    fundamentals_provider_id = _parse_provider_id(
        live["fundamentals_provider"],
        label="live.fundamentals_provider",
    )
    get_fundamentals_registration(fundamentals_provider_id, market)

    listing_provider_id = None
    if market is Market.TW:
        listing_provider = live.get("listing_provider")
        if not isinstance(listing_provider, dict):
            raise ValidationError("config missing required key: live.listing_provider.TW")
        if market.value not in listing_provider:
            raise ValidationError("config missing required key: live.listing_provider.TW")
        listing_provider_id = _parse_provider_id(
            listing_provider[market.value],
            label="live.listing_provider.TW",
        )
        get_listing_registration(listing_provider_id, market)

    return LiveValueProviderSelection(
        fundamentals_provider_id=fundamentals_provider_id,
        listing_provider_id=listing_provider_id,
    )
```

Do not call `load_config`, parse `live.ticker_limit`, or require `live.listing_provider.US` in this loader.

- [ ] **Step 4: Run focused and full configuration tests**

Run:

```bash
uv run pytest -q tests/test_config.py
uv run mypy src/universe_selector/config.py
```

Expected: all configuration tests pass and mypy reports no issues.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/universe_selector/config.py tests/test_config.py
git commit -m "feat(value): load TW listing provider"
```

### Task 2: Listing-aware single-ticker fundamentals boundary

**Files:**
- Modify: `src/universe_selector/providers/base.py:46-59`
- Modify: `src/universe_selector/providers/yfinance_fundamentals.py:109-128,407-427`
- Test: `tests/test_yfinance_fundamentals_provider.py:155-217`

- [ ] **Step 1: Replace the ambiguous TW single-ticker test with listing-aware cases**

Add this helper and tests using the existing `valid_payload()` fixture:

```python
def _tw_listing(ticker: str, exchange_segment: str) -> ListingCandidate:
    return ListingCandidate(
        market=Market.TW,
        ticker=ticker,
        listing_symbol=ticker,
        exchange_segment=exchange_segment,
        listing_status="active",
        instrument_type="common_stock",
        source_id=f"unit:{ticker}",
    )


@pytest.mark.parametrize(
    ("ticker", "exchange_segment", "expected_symbol"),
    (("2330", "TWSE", "2330.TW"), ("6488", "TPEX", "6488.TWO")),
)
def test_yfinance_single_fundamentals_maps_resolved_tw_listing(
    ticker: str,
    exchange_segment: str,
    expected_symbol: str,
) -> None:
    requested: list[str] = []

    def fetcher(symbol: str) -> dict[str, object]:
        requested.append(symbol)
        payload = valid_payload()
        payload["currency"] = "TWD"
        return payload

    provider = YFinanceFundamentalsProvider(fetcher=fetcher)
    data = provider.load_fundamentals(
        Market.TW,
        ticker,
        listing=_tw_listing(ticker, exchange_segment),
    )

    assert requested == [expected_symbol]
    assert data.facts.market is Market.TW
    assert data.facts.ticker == ticker
    assert data.facts.currency == "TWD"


def test_yfinance_single_tw_fundamentals_requires_matching_listing() -> None:
    provider = YFinanceFundamentalsProvider(
        fetcher=lambda symbol: (_ for _ in ()).throw(AssertionError(f"unexpected fetch {symbol}"))
    )

    with pytest.raises(ProviderDataError, match="requires resolved listing identity"):
        provider.load_fundamentals(Market.TW, "6488")

    with pytest.raises(ProviderDataError, match="does not match requested ticker"):
        provider.load_fundamentals(Market.TW, "6488", listing=_tw_listing("2330", "TWSE"))

    with pytest.raises(ProviderDataError, match="unsupported TW exchange segment"):
        provider.load_fundamentals(Market.TW, "6488", listing=_tw_listing("6488", "EMERGING"))
```

Keep the existing US `AAPL` and `BRK.B` tests unchanged.

- [ ] **Step 2: Run the provider tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_yfinance_fundamentals_provider.py \
  -k 'single_fundamentals_maps_resolved_tw_listing or single_tw_fundamentals_requires_matching_listing'
```

Expected: calls fail because `load_fundamentals` does not accept `listing` and still defaults bare TW tickers to `.TW`.

- [ ] **Step 3: Extend the protocol and Yahoo adapter**

Change the protocol signature:

```python
def load_fundamentals(
    self,
    market: Market,
    ticker: str,
    *,
    listing: ListingCandidate | None = None,
) -> FundamentalsRunData:
    raise NotImplementedError
```

Change the Yahoo method to select its request symbol through a single-listing helper:

```python
def load_fundamentals(
    self,
    market: Market,
    ticker: str,
    *,
    listing: ListingCandidate | None = None,
) -> FundamentalsRunData:
    normalized_ticker = canonical_ticker(ticker)
    request_symbol = _request_symbol_for_single_listing(market, normalized_ticker, listing)
    fetch_started_at = self._clock()
    payload = self._fetcher(request_symbol)
    facts = self._normalize_payload(market, normalized_ticker, payload)
    latest_source_date = max(
        facts.fiscal_period_end,
        facts.reference_price_as_of,
        facts.balance_sheet_as_of,
    )

    return FundamentalsRunData(
        metadata=FundamentalsMetadata(
            data_mode="live",
            fundamentals_provider_id=self.provider_id,
            fundamentals_source_ids=self.source_ids,
            data_fetch_started_at=fetch_started_at,
            latest_source_date=latest_source_date,
            source_risk_note=_SOURCE_RISK_NOTE,
            field_mapping_note=_FIELD_MAPPING_NOTE,
        ),
        facts=facts,
    )
```

Add this private helper beside `_request_symbol_for_listing`:

```python
def _request_symbol_for_single_listing(
    market: Market,
    ticker: str,
    listing: ListingCandidate | None,
) -> str:
    if market is not Market.TW:
        return _request_symbol(market, ticker)
    if listing is None:
        raise ProviderDataError("TW single-ticker fundamentals requires resolved listing identity")
    if listing.market is not market or canonical_ticker(listing.ticker) != ticker:
        raise ProviderDataError(
            f"resolved listing {listing.ticker} does not match requested ticker {ticker} for {market.value}"
        )
    return _request_symbol_for_listing(market, listing, ticker)
```

Remove the TW auto-suffix branch from `_request_symbol`; a TW single request can reach Yahoo only through `_request_symbol_for_single_listing`. Leave US class-share conversion unchanged.

- [ ] **Step 4: Run focused provider and registry tests**

Run:

```bash
uv run pytest -q tests/test_yfinance_fundamentals_provider.py tests/test_provider_registry.py
uv run mypy src/universe_selector/providers
```

Expected: all tests pass; TWSE requests `.TW`, TPEX requests `.TWO`, facts stay suffix-free, and US mapping remains green.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/universe_selector/providers/base.py \
  src/universe_selector/providers/yfinance_fundamentals.py \
  tests/test_yfinance_fundamentals_provider.py
git commit -m "fix(value): require TW listing identity"
```

### Task 3: Resolve TW listing identity in valuation orchestration

**Files:**
- Modify: `src/universe_selector/valuation/service.py:1-61`
- Test: `tests/test_valuation_service.py:1-55,470-520`

- [ ] **Step 1: Add service fakes and failing resolution tests**

Import `replace` from `dataclasses`, plus `ProviderDataError`,
`ProviderRunContext`, `ListingCandidate`, and `ListingProviderRegistration`.
Keep the fake's existing pair-shaped `requests` list so US assertions remain
unchanged. Add a listing record and replace the method with:

```python
self.requests: list[tuple[Market, str]] = []
self.listing_requests: list[ListingCandidate | None] = []

def load_fundamentals(
    self,
    market: Market,
    ticker: str,
    *,
    listing: ListingCandidate | None = None,
) -> FundamentalsRunData:
    self.requests.append((market, ticker))
    self.listing_requests.append(listing)
    return FundamentalsRunData(
        metadata=FundamentalsMetadata(
            data_mode="live",
            fundamentals_provider_id=self.provider_id,
            fundamentals_source_ids=self.source_ids,
            data_fetch_started_at=datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
            latest_source_date=date(2026, 5, 15),
        ),
        facts=self._facts,
    )
```

Add this listing fake:

```python
class FakeListingProvider:
    provider_id = "fake_listing"
    source_ids = ("fake-listings",)

    def __init__(self, listings: list[ListingCandidate]) -> None:
        self._listings = listings
        self.requests: list[tuple[ProviderRunContext, Market]] = []

    def load_listings(self, context: ProviderRunContext, market: Market) -> list[ListingCandidate]:
        self.requests.append((context, market))
        return self._listings


def _tw_listing(ticker: str, exchange_segment: str = "TPEX") -> ListingCandidate:
    return ListingCandidate(
        market=Market.TW,
        ticker=ticker,
        listing_symbol=ticker,
        exchange_segment=exchange_segment,
        listing_status="active",
        instrument_type="common_stock",
        source_id=f"unit:{ticker}",
    )


def _install_tw_value_providers(
    monkeypatch,
    listings: list[ListingCandidate],
) -> tuple[FakeListingProvider, FakeFundamentalsProvider]:
    fake_listing_provider = FakeListingProvider(listings)
    fake_fundamentals_provider = FakeFundamentalsProvider(
        replace(_facts(), market=Market.TW, ticker="2330", currency="TWD")
    )
    listing_registration = ListingProviderRegistration(
        provider_id=fake_listing_provider.provider_id,
        supported_markets=frozenset({Market.TW}),
        source_ids=fake_listing_provider.source_ids,
        factory=lambda _config: fake_listing_provider,
    )
    fundamentals_registration = FundamentalsProviderRegistration(
        provider_id=fake_fundamentals_provider.provider_id,
        supported_markets=frozenset({Market.TW}),
        source_ids=fake_fundamentals_provider.source_ids,
        factory=lambda: fake_fundamentals_provider,
    )
    monkeypatch.setattr(
        "universe_selector.valuation.service.get_listing_registration",
        lambda provider_id, market: listing_registration,
    )
    monkeypatch.setattr(
        "universe_selector.valuation.service.get_fundamentals_registration",
        lambda provider_id, market: fundamentals_registration,
    )
    return fake_listing_provider, fake_fundamentals_provider
```

Replace the existing TW service test with this complete setup and call:

```python
listing = _tw_listing("2330", "TWSE")
fake_listing_provider, fake_provider = _install_tw_value_providers(monkeypatch, [listing])

result = run_valuation(
    Market.TW,
    "2330",
    "fcf_dcf_v1",
    TW_FIXTURE,
    "fake_fundamentals",
    listing_provider_id="fake_listing",
)

assert len(fake_listing_provider.requests) == 1
assert fake_listing_provider.requests[0][0].ticker_limit is None
assert fake_provider.requests == [(Market.TW, "2330")]
assert fake_provider.listing_requests == [listing]
assert result.run_input.ticker == "2330"
assert result.run_input.assumptions.assumption_path.endswith("valuation_assumptions/tw/2330.yaml")
```

Add missing, duplicate, suffixed-input, and pre-network assumptions tests:

```python
@pytest.mark.parametrize("listings", ([], [_tw_listing("2330"), _tw_listing("2330")]))
def test_run_valuation_rejects_missing_or_duplicate_tw_listing_before_fundamentals(
    monkeypatch,
    listings: list[ListingCandidate],
) -> None:
    _, fake_provider = _install_tw_value_providers(monkeypatch, listings)

    with pytest.raises(ProviderDataError, match="expected exactly one listing for TW ticker 2330"):
        run_valuation(
            Market.TW,
            "2330",
            "fcf_dcf_v1",
            TW_FIXTURE,
            "fake_fundamentals",
            listing_provider_id="fake_listing",
        )
    assert fake_provider.requests == []
    assert fake_provider.listing_requests == []


@pytest.mark.parametrize("ticker", ("2330.TW", "6488.TWO"))
def test_run_valuation_rejects_provider_suffixed_tw_ticker_before_provider_access(ticker: str) -> None:
    with pytest.raises(ValidationError, match="canonical bare ticker"):
        run_valuation(
            Market.TW,
            ticker,
            "fcf_dcf_v1",
            TW_FIXTURE,
            "yfinance_fundamentals",
            listing_provider_id="twse_isin",
        )


def test_run_valuation_validates_assumptions_before_tw_listing_network(monkeypatch, tmp_path: Path) -> None:
    fake_listing_provider, fake_provider = _install_tw_value_providers(
        monkeypatch,
        [_tw_listing("2330")],
    )

    with pytest.raises(ValidationError, match="missing valuation assumptions file"):
        run_valuation(
            Market.TW,
            "2330",
            "fcf_dcf_v1",
            tmp_path / "missing.yaml",
            "yfinance_fundamentals",
            listing_provider_id="fake_listing",
        )
    assert fake_listing_provider.requests == []
    assert fake_provider.requests == []
    assert fake_provider.listing_requests == []
```

- [ ] **Step 2: Run service tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_valuation_service.py -k 'tw or listing or suffixed'
```

Expected: tests fail because `run_valuation` has no `listing_provider_id` and never loads listings.

- [ ] **Step 3: Implement canonical validation and exact resolver**

Add imports for UTC time, listing provider types, context construction, and registration resolution. Add these helpers:

```python
def _validate_value_ticker(market: Market, ticker: str) -> None:
    if market is Market.TW and ticker.endswith((".TW", ".TWO")):
        raise ValidationError(
            f"TW value ticker must be a canonical bare ticker without provider suffix: {ticker}"
        )


def _select_listing(
    market: Market,
    ticker: str,
    listings: list[ListingCandidate],
) -> ListingCandidate:
    matches = [
        listing
        for listing in listings
        if listing.market is market and canonical_ticker(listing.ticker) == ticker
    ]
    if len(matches) != 1:
        raise ProviderDataError(
            f"expected exactly one listing for {market.value} ticker {ticker}; found {len(matches)}"
        )
    return matches[0]
```

Extend `run_valuation` without breaking existing US callers:

```python
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
    _validate_value_ticker(market, normalized_ticker)
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
```

Provider registrations are resolved before assumptions; provider factories and network methods run only after assumptions validation. Do not read `live.ticker_limit`, DuckDB, or ranking runs.

- [ ] **Step 4: Run service and provider tests**

Run:

```bash
uv run pytest -q tests/test_valuation_service.py tests/test_yfinance_fundamentals_provider.py
uv run mypy src/universe_selector/valuation/service.py tests/test_valuation_service.py
```

Expected: all tests pass, including existing US valuation tests without a listing provider.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/universe_selector/valuation/service.py tests/test_valuation_service.py
git commit -m "feat(value): resolve TW listing identity"
```

### Task 4: CLI wiring, public documentation, and complete verification

**Files:**
- Modify: `src/universe_selector/cli.py:10,312-340`
- Modify: `tests/test_cli.py:694-958`
- Modify: `README.md:71,264-275`
- Modify: `docs/valuation.md:1-16,35-49`
- Modify: `tests/test_readme_contract.py:37-113`

- [ ] **Step 1: Write failing CLI wiring tests**

Import `LiveValueProviderSelection`. In every value CLI fake, change the
signature and captured fields to this exact shape:

```python
def fake_run_valuation(
    *,
    market: Market,
    ticker: str,
    model_id: str | None,
    assumptions_path: Path | None,
    fundamentals_provider_id: str,
    listing_provider_id: str | None = None,
):
    captured["market"] = market
    captured["ticker"] = ticker
    captured["model_id"] = model_id
    captured["assumptions_path"] = assumptions_path
    captured["fundamentals_provider_id"] = fundamentals_provider_id
    captured["listing_provider_id"] = listing_provider_id
    return sentinel_result
```

Replace each monkeypatch of `load_live_fundamentals_provider_id` with:

```python
monkeypatch.setattr(
    "universe_selector.cli.load_live_value_provider_selection",
    lambda market: LiveValueProviderSelection(
        fundamentals_provider_id="fake_fundamentals",
        listing_provider_id=None,
    ),
)
```

Add a TW wiring test:

```python
def test_cli_value_tw_passes_canonical_ticker_and_both_provider_ids(monkeypatch) -> None:
    _install_value_cli_no_persistence_or_ranking_guards(monkeypatch)
    captured: dict[str, object] = {}

    def fake_run_valuation(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "universe_selector.cli.load_live_value_provider_selection",
        lambda market: LiveValueProviderSelection(
            fundamentals_provider_id="fake_fundamentals",
            listing_provider_id="fake_listing",
        ),
    )
    monkeypatch.setattr("universe_selector.cli.run_valuation", fake_run_valuation)
    monkeypatch.setattr("universe_selector.cli.render_valuation_markdown", lambda result: "ok\n")

    result = runner.invoke(app, ["value", "tw", "--ticker", "6488"])

    assert result.exit_code == 0, result.output
    assert captured["market"] is Market.TW
    assert captured["ticker"] == "6488"
    assert captured["fundamentals_provider_id"] == "fake_fundamentals"
    assert captured["listing_provider_id"] == "fake_listing"
```

Update the unknown-model ordering test so `load_live_value_provider_selection` raises `AssertionError` if called; the CLI must still reject the model first.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_cli.py -k 'cli_value'
```

Expected: tests fail because CLI still imports the old provider-ID loader and does not pass `listing_provider_id`.

- [ ] **Step 3: Wire the typed provider selection**

Change the config import and value action:

```python
from universe_selector.config import AppConfig, load_config, load_live_value_provider_selection


if model is not None:
    get_valuation_model(model)
provider_selection = load_live_value_provider_selection(resolved_market)
result = run_valuation(
    market=resolved_market,
    ticker=normalized_ticker,
    model_id=model,
    assumptions_path=assumptions,
    fundamentals_provider_id=provider_selection.fundamentals_provider_id,
    listing_provider_id=provider_selection.listing_provider_id,
)
```

After all CLI tests use the new selection loader, remove the now-unused
`load_live_fundamentals_provider_id` function from `config.py`. Verify no stale
references remain:

```bash
rg -n "load_live_fundamentals_provider_id" src tests
```

Expected: no matches. Keep model validation before provider selection and keep
all persistence guards in the CLI tests.

- [ ] **Step 4: Document the public provider-neutral ticker contract**

Add this paragraph near the command examples in `docs/valuation.md`:

```markdown
For Taiwan, `value` accepts the same canonical bare ticker as `inspect`, for
example `2330` or `6488`. It resolves the ticker through
`live.listing_provider.TW`; provider-specific request symbols such as `.TW` and
`.TWO` remain internal to the configured fundamentals provider. Do not append a
provider suffix to `--ticker`.
```

Add this exact paragraph near the `value` examples in `README.md`:

```markdown
For Taiwan, `value` accepts the same canonical bare ticker as `inspect` and
resolves it through `live.listing_provider.TW`. Provider suffixes such as `.TW`
and `.TWO` stay internal to the fundamentals provider. Do not append a provider
suffix to `--ticker`.
```

Assert these phrases in `tests/test_readme_contract.py`:

```python
assert "same canonical bare ticker as `inspect`" in text
assert "live.listing_provider.TW" in text
assert "Do not append a provider suffix" in text
```

- [ ] **Step 5: Run all targeted tests**

Run:

```bash
uv run pytest -q \
  tests/test_config.py \
  tests/test_yfinance_fundamentals_provider.py \
  tests/test_valuation_service.py \
  tests/test_cli.py \
  tests/test_readme_contract.py \
  tests/test_live_provider.py \
  tests/test_pipeline.py
```

Expected: all targeted tests pass offline.

- [ ] **Step 6: Run all repository quality gates**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

Expected: all formatting, lint, type-check, and test gates pass. The test count is at least the 697-test clean baseline plus the new regression cases.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/universe_selector/cli.py tests/test_cli.py \
  src/universe_selector/config.py README.md docs/valuation.md \
  tests/test_readme_contract.py
git commit -m "feat(value): wire TW listing resolution"
```

## Final Completion Evidence

Before declaring completion, record:

```bash
git status --short
git log --oneline ae3b7c5..HEAD
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

The branch must be clean; the design and plan commits plus all implementation commits must be present; and every gate must exit zero.
