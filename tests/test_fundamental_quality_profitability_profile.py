from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.providers.models import (
    FundamentalsCoverage,
    FundamentalsMetadata,
    FundamentalsUniverseRunData,
    ListingCandidate,
)
from universe_selector.ranking_profiles import get_ranking_profile_registration
from universe_selector.ranking_profiles.fundamental_quality_profitability_v1 import (
    FUNDAMENTAL_QUALITY_PROFITABILITY_RANKING_METRIC_KEYS,
    FUNDAMENTAL_QUALITY_PROFITABILITY_SNAPSHOT_METRIC_KEYS,
    FundamentalQualityProfitabilityV1Profile,
    _percentile_scores,
)


def _listing(ticker: str, market: Market = Market.US) -> ListingCandidate:
    return ListingCandidate(
        market=market,
        ticker=ticker,
        listing_symbol=ticker,
        exchange_segment="TEST",
        listing_status="active",
        instrument_type="common_stock",
        source_id=f"test:{ticker}",
    )


def _bars(ticker: str, latest: date, *, close: float = 20.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market": ["US"],
            "ticker": [ticker],
            "bar_date": [latest],
            "open": [close],
            "high": [close],
            "low": [close],
            "close": [close],
            "adjusted_close": [close],
            "volume": [1_000_000],
        }
    )


def _fundamental_row(
    ticker: str,
    *,
    fiscal_period_end: date = date(2026, 3, 31),
    balance_sheet_as_of: date = date(2026, 3, 31),
    revenue_ttm: float = 100.0,
    net_income_ttm: float = 20.0,
    total_assets: float = 200.0,
    shareholders_equity: float = 100.0,
    total_debt: float = 50.0,
    cash_and_cash_equivalents: float = 10.0,
    free_cash_flow_ttm: float = 20.0,
    roe: float = 0.20,
    roa: float = 0.10,
    operating_margin: float = 0.30,
    net_margin: float = 0.20,
    fcf_margin: float = 0.20,
    debt_to_equity: float = 0.50,
    annual_fallback: float = 0.0,
    negative_net_income: float = 0.0,
    negative_fcf: float = 0.0,
    return_on_equity_canary: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "market": "US",
        "ticker": ticker,
        "currency": "USD",
        "fiscal_period_end": fiscal_period_end,
        "balance_sheet_as_of": balance_sheet_as_of,
        "fiscal_period_type": "annual" if annual_fallback else "ttm",
        "revenue_ttm": revenue_ttm,
        "gross_profit_ttm": 60.0,
        "operating_income_ttm": operating_margin * revenue_ttm,
        "net_income_ttm": net_income_ttm,
        "total_assets": total_assets,
        "shareholders_equity": shareholders_equity,
        "total_debt": total_debt,
        "cash_and_cash_equivalents": cash_and_cash_equivalents,
        "operating_cash_flow_ttm": free_cash_flow_ttm + 5.0,
        "capital_expenditures_ttm": 5.0,
        "free_cash_flow_ttm": free_cash_flow_ttm,
        "roe": roe,
        "roa": roa,
        "gross_margin": 0.60,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "debt_to_equity": debt_to_equity,
        "fcf_margin": fcf_margin,
        "tag_fundamentals_annual_fallback": annual_fallback,
        "tag_negative_net_income": negative_net_income,
        "tag_negative_fcf": negative_fcf,
    }
    if return_on_equity_canary is not None:
        row["returnOnEquity"] = return_on_equity_canary
    return row


def _fundamentals(rows: list[dict[str, object]]) -> FundamentalsUniverseRunData:
    return FundamentalsUniverseRunData(
        metadata=FundamentalsMetadata(
            data_mode="fixture",
            fundamentals_provider_id="unit_fundamentals",
            fundamentals_source_ids=("unit:fundamentals",),
            data_fetch_started_at=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc),
            latest_source_date=date(2026, 3, 31),
        ),
        facts=pl.DataFrame(rows),
        coverage=FundamentalsCoverage(
            requested_count=len(rows),
            returned_count=len(rows),
            missing_count=0,
            invalid_count=0,
        ),
    )


def _snapshot_for(rows: list[dict[str, object]], *, latest: date = date(2026, 4, 24)) -> pl.DataFrame:
    profile = FundamentalQualityProfitabilityV1Profile()
    tickers = [str(row["ticker"]) for row in rows]
    return profile.build_snapshot(
        run_id="fundamental-run",
        market=Market.US,
        listings=[_listing(ticker) for ticker in tickers],
        bars=pl.concat([_bars(ticker, latest) for ticker in tickers]),
        run_latest_bar_date=latest,
        fundamentals=_fundamentals(rows),
    )


def test_fundamental_quality_profile_contract() -> None:
    profile = FundamentalQualityProfitabilityV1Profile()
    registration = get_ranking_profile_registration("fundamental_quality_profitability_v1")
    payload = profile.ranking_config_payload()

    assert profile.profile_id == "fundamental_quality_profitability_v1"
    assert profile.horizon_order == ("composite",)
    assert registration.data_requirements.fundamentals is True
    assert profile.snapshot_metric_keys == FUNDAMENTAL_QUALITY_PROFITABILITY_SNAPSHOT_METRIC_KEYS
    assert profile.ranking_metric_keys == FUNDAMENTAL_QUALITY_PROFITABILITY_RANKING_METRIC_KEYS
    assert profile.inspect_metric_keys == profile.snapshot_metric_keys
    assert payload["eligibility_rules_version"] == 1
    assert payload["scoring_weights"] == {
        "profitability": {
            "score_roe": 0.35,
            "score_roa": 0.25,
            "score_operating_margin": 0.25,
            "score_net_margin": 0.15,
        },
        "cash_generation": {"score_fcf_margin": 0.70, "score_positive_fcf": 0.30},
        "balance_sheet": {"score_low_debt_to_equity": 1.0},
        "composite": {"score_profitability": 0.60, "score_cash_generation": 0.25, "score_balance_sheet": 0.15},
    }


def test_fundamental_quality_profile_filters_stale_and_invalid_rows() -> None:
    latest = date(2026, 4, 24)

    snapshot = _snapshot_for(
        [
            _fundamental_row("PASS"),
            _fundamental_row("STALE", fiscal_period_end=latest - timedelta(days=551)),
            _fundamental_row("ZERO_REV", revenue_ttm=0.0),
            _fundamental_row("NEG_EQUITY", shareholders_equity=-1.0),
            _fundamental_row("NEG_DEBT", total_debt=-1.0, debt_to_equity=-0.01),
            _fundamental_row("NEG_CASH", cash_and_cash_equivalents=-1.0),
            _fundamental_row("NEG_INCOME", net_income_ttm=-10.0, net_margin=-0.10, negative_net_income=1.0),
            _fundamental_row("NEG_FCF", free_cash_flow_ttm=-5.0, fcf_margin=-0.05, negative_fcf=1.0),
        ],
        latest=latest,
    ).sort("ticker")

    assert snapshot["ticker"].to_list() == ["NEG_FCF", "NEG_INCOME", "PASS"]
    assert snapshot.filter(pl.col("ticker") == "NEG_INCOME")["tag_negative_net_income"].item() == 1.0
    assert snapshot.filter(pl.col("ticker") == "NEG_FCF")["tag_negative_fcf"].item() == 1.0


def test_fundamental_quality_profile_fails_when_no_eligible_fundamentals() -> None:
    latest = date(2026, 4, 24)
    profile = FundamentalQualityProfitabilityV1Profile()

    with pytest.raises(ProviderDataError, match="fundamentals provider returned no eligible fundamentals for US"):
        profile.build_snapshot(
            run_id="fundamental-run",
            market=Market.US,
            listings=[_listing("STALE")],
            bars=_bars("STALE", latest),
            run_latest_bar_date=latest,
            fundamentals=_fundamentals([_fundamental_row("STALE", fiscal_period_end=latest - timedelta(days=551))]),
        )


def test_fundamental_quality_percentile_scores() -> None:
    assert _percentile_scores({"AAA": 0.30, "BBB": 0.20, "CCC": 0.10}, higher_is_better=True) == {
        "AAA": 100.0,
        "BBB": 50.0,
        "CCC": 0.0,
    }
    assert _percentile_scores({"AAA": 0.5, "BBB": 1.0, "CCC": 1.0}, higher_is_better=False) == {
        "AAA": 100.0,
        "BBB": 25.0,
        "CCC": 25.0,
    }
    assert _percentile_scores({"AAA": 0.30}, higher_is_better=True) == {"AAA": 100.0}


def test_fundamental_quality_profile_assigns_exact_scores_and_ranks() -> None:
    profile = FundamentalQualityProfitabilityV1Profile()
    snapshot = _snapshot_for(
        [
            _fundamental_row(
                "AAA", roe=0.30, roa=0.15, operating_margin=0.30, net_margin=0.20, fcf_margin=0.20, debt_to_equity=0.50
            ),
            _fundamental_row(
                "BBB", roe=0.20, roa=0.10, operating_margin=0.20, net_margin=0.10, fcf_margin=0.10, debt_to_equity=1.00
            ),
            _fundamental_row(
                "CCC",
                roe=0.10,
                roa=0.05,
                operating_margin=0.10,
                net_margin=0.05,
                free_cash_flow_ttm=-2.0,
                fcf_margin=-0.02,
                debt_to_equity=2.00,
                negative_fcf=1.0,
            ),
        ]
    )

    rankings = profile.assign_rankings(snapshot).sort("rank")
    bbb = rankings.filter(pl.col("ticker") == "BBB").to_dicts()[0]

    assert rankings["ticker"].to_list() == ["AAA", "BBB", "CCC"]
    assert bbb["score_roe"] == pytest.approx(50.0)
    assert bbb["score_roa"] == pytest.approx(50.0)
    assert bbb["score_operating_margin"] == pytest.approx(50.0)
    assert bbb["score_net_margin"] == pytest.approx(50.0)
    assert bbb["score_fcf_margin"] == pytest.approx(50.0)
    assert bbb["score_positive_fcf"] == pytest.approx(100.0)
    assert bbb["score_low_debt_to_equity"] == pytest.approx(50.0)
    assert bbb["score_profitability"] == pytest.approx(50.0)
    assert bbb["score_cash_generation"] == pytest.approx(65.0)
    assert bbb["score_balance_sheet"] == pytest.approx(50.0)
    assert bbb["score"] == pytest.approx(53.75)


def test_fundamental_quality_profile_ties_sort_by_ticker() -> None:
    profile = FundamentalQualityProfitabilityV1Profile()
    snapshot = _snapshot_for([_fundamental_row("BBB"), _fundamental_row("AAA")])

    rankings = profile.assign_rankings(snapshot).sort("rank")

    assert rankings["ticker"].to_list() == ["AAA", "BBB"]


def test_fundamental_quality_profile_uses_normalized_fields_not_yahoo_raw_canary() -> None:
    profile = FundamentalQualityProfitabilityV1Profile()
    snapshot = _snapshot_for(
        [
            _fundamental_row("AAA", roe=0.30, return_on_equity_canary=-999.0),
            _fundamental_row("BBB", roe=0.10, return_on_equity_canary=999.0),
        ]
    )
    rankings = profile.assign_rankings(snapshot)

    assert snapshot.filter(pl.col("ticker") == "AAA")["roe"].item() == pytest.approx(0.30)
    assert rankings.filter(pl.col("ticker") == "AAA")["score_roe"].item() == pytest.approx(100.0)
