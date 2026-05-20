# Extending

Universe Selector is designed around two extension points: ranking profiles and
providers.

## Ranking Profiles

To add a ranking profile:

- Add a module under `src/universe_selector/ranking_profiles/`.
- Implement the `RankingProfile` protocol from `ranking_profiles/base.py`.
- Define a stable `profile_id` and include it in `ranking_config_payload()`.
- Implement `build_snapshot()` to turn provider data into one persisted row per
  surviving ticker.
- Implement `assign_rankings()` to produce one ranking row per ticker and
  profile horizon.
- Declare `snapshot_metric_keys`, `ranking_metric_keys`, and
  `inspect_metric_keys`; these keys control what is persisted in `metrics_json`
  and what `inspect` can print.
- Create a `RankingProfileRegistration` and add it to
  `ranking_profiles/registry.py`.
- Add tests for validation, snapshot construction, ranking assignment,
  persistence, report, and inspect behavior.

## Providers

To add providers:

- Use `providers/models.py` for the data contracts.
- A listing provider returns `ListingCandidate` records for a market.
- An OHLCV provider returns canonical daily bars with `market`, `ticker`,
  `bar_date`, `open`, `high`, `low`, `close`, `adjusted_close`, and `volume`.
- Add a provider registration in the provider module and include it in
  `providers/registry.py`.
- Keep provider-specific source IDs stable, because they are part of persisted
  provider metadata and provider config hashes.
- Add tests for registration, provider parsing/normalization, and error cases.

Do not add profile-specific metrics as DuckDB columns. Keep profile-specific
values behind the metric key declarations so multiple profiles can coexist in
the same persistence model.
