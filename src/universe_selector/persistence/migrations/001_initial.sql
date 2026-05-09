create table run_log (
    run_id varchar primary key,
    market varchar not null check (market in ('TW', 'US')),
    status varchar not null check (status in ('running', 'successful', 'failed')),
    created_at timestamp not null,
    ranking_profile varchar not null,
    ranking_config_hash varchar not null,
    error_message varchar,
    check (
        (market = 'TW' and starts_with(run_id, 'tw-'))
        or (market = 'US' and starts_with(run_id, 'us-'))
    )
);

create table run_provider_metadata (
    run_id varchar primary key references run_log(run_id),
    data_mode varchar not null check (data_mode in ('fixture', 'live')),
    listing_provider_id varchar not null,
    listing_source_id varchar not null,
    ohlcv_provider_id varchar not null,
    ohlcv_source_id varchar not null,
    provider_config_hash varchar not null,
    data_fetch_started_at timestamp not null,
    market_timezone varchar not null,
    run_latest_bar_date date not null
);

create table run_ticker_snapshot (
    run_id varchar not null references run_log(run_id),
    market varchar not null check (market in ('TW', 'US')),
    ticker varchar not null,
    close double not null,
    adjusted_close double not null,
    metrics_json varchar not null,
    primary key (run_id, ticker)
);

create table run_rankings (
    run_id varchar not null references run_log(run_id),
    market varchar not null check (market in ('TW', 'US')),
    horizon varchar not null,
    ticker varchar not null,
    final_rank_percentile double not null check (final_rank_percentile between 0 and 100),
    rank integer not null check (rank >= 1),
    metrics_json varchar not null,
    primary key (run_id, market, horizon, ticker)
);

create table report_artifacts (
    run_id varchar not null references run_log(run_id),
    format varchar not null check (format = 'markdown'),
    content varchar not null,
    primary key (run_id, format)
);
