from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Mapping, cast

import polars as pl

from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError, ValidationError
from universe_selector.providers.models import FundamentalsUniverseRunData, ListingCandidate
from universe_selector.ranking_profiles.base import RankingProfileDataRequirements
from universe_selector.ranking_profiles.registration import RankingProfileRegistration


FUNDAMENTAL_QUALITY_PROFITABILITY_PROFILE_ID = "fundamental_quality_profitability_v1"
FUNDAMENTAL_QUALITY_PROFITABILITY_RANK_INTERPRETATION_NOTE = (
    "Fundamental quality scores rank normalized profitability, cash generation, and balance-sheet quality; "
    "high scores do not imply future returns."
)
FUNDAMENTAL_QUALITY_PROFITABILITY_HORIZON_ORDER = ("composite",)
FUNDAMENTAL_QUALITY_PROFITABILITY_METRICS_VERSION = 1.0
FUNDAMENTAL_QUALITY_PROFITABILITY_STALENESS_DAYS = 550
FUNDAMENTAL_QUALITY_PROFITABILITY_ELIGIBILITY_RULES_VERSION = 1
FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS = {
    "profitability": {
        "score_roe": 0.35,
        "score_roa": 0.25,
        "score_operating_margin": 0.25,
        "score_net_margin": 0.15,
    },
    "cash_generation": {
        "score_fcf_margin": 0.70,
        "score_positive_fcf": 0.30,
    },
    "balance_sheet": {
        "score_low_debt_to_equity": 1.0,
    },
    "composite": {
        "score_profitability": 0.60,
        "score_cash_generation": 0.25,
        "score_balance_sheet": 0.15,
    },
}

FUNDAMENTAL_QUALITY_PROFITABILITY_SNAPSHOT_METRIC_KEYS = (
    "profile_metrics_version",
    "revenue_ttm",
    "net_income_ttm",
    "total_assets",
    "shareholders_equity",
    "total_debt",
    "free_cash_flow_ttm",
    "roe",
    "roa",
    "operating_margin",
    "net_margin",
    "fcf_margin",
    "debt_to_equity",
    "tag_fundamentals_annual_fallback",
    "tag_negative_net_income",
    "tag_negative_fcf",
)
FUNDAMENTAL_QUALITY_PROFITABILITY_RANKING_METRIC_KEYS = (
    "score_roe",
    "score_roa",
    "score_operating_margin",
    "score_net_margin",
    "score_fcf_margin",
    "score_positive_fcf",
    "score_low_debt_to_equity",
    "score_profitability",
    "score_cash_generation",
    "score_balance_sheet",
    "tag_fundamentals_annual_fallback",
    "tag_negative_net_income",
    "tag_negative_fcf",
)
FUNDAMENTAL_QUALITY_PROFITABILITY_INSPECT_METRIC_KEYS = FUNDAMENTAL_QUALITY_PROFITABILITY_SNAPSHOT_METRIC_KEYS

FUNDAMENTAL_QUALITY_PROFITABILITY_SNAPSHOT_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "ticker": pl.String,
    "close": pl.Float64,
    "adjusted_close": pl.Float64,
    **{key: pl.Float64 for key in FUNDAMENTAL_QUALITY_PROFITABILITY_SNAPSHOT_METRIC_KEYS},
}
FUNDAMENTAL_QUALITY_PROFITABILITY_RANKING_SCHEMA = {
    "run_id": pl.String,
    "market": pl.String,
    "horizon": pl.String,
    "ticker": pl.String,
    **{key: pl.Float64 for key in FUNDAMENTAL_QUALITY_PROFITABILITY_RANKING_METRIC_KEYS},
    "score": pl.Float64,
    "rank": pl.Int64,
}


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _fresh_enough(run_latest_bar_date: date, fiscal_period_end: date, balance_sheet_as_of: date) -> bool:
    return max((run_latest_bar_date - fiscal_period_end).days, (run_latest_bar_date - balance_sheet_as_of).days) <= (
        FUNDAMENTAL_QUALITY_PROFITABILITY_STALENESS_DAYS
    )


def _empty_rankings() -> pl.DataFrame:
    return pl.DataFrame(schema=FUNDAMENTAL_QUALITY_PROFITABILITY_RANKING_SCHEMA)


def _percentile_scores(values_by_ticker: Mapping[str, float], *, higher_is_better: bool) -> dict[str, float]:
    if not values_by_ticker:
        return {}
    if len(values_by_ticker) == 1:
        ticker = next(iter(values_by_ticker))
        return {ticker: 100.0}

    ordered = sorted(
        values_by_ticker.items(),
        key=lambda item: item[1],
        reverse=higher_is_better,
    )
    scores: dict[str, float] = {}
    n = len(ordered)
    index = 0
    while index < n:
        value = ordered[index][1]
        end = index + 1
        while end < n and ordered[end][1] == value:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        score = 100.0 * (n - average_rank) / (n - 1)
        for ticker, _ in ordered[index:end]:
            scores[ticker] = score
        index = end
    return scores


@dataclass(frozen=True)
class FundamentalQualityProfitabilityV1Profile:
    profile_id: Literal["fundamental_quality_profitability_v1"] = "fundamental_quality_profitability_v1"
    snapshot_metric_keys: tuple[str, ...] = FUNDAMENTAL_QUALITY_PROFITABILITY_SNAPSHOT_METRIC_KEYS
    ranking_metric_keys: tuple[str, ...] = FUNDAMENTAL_QUALITY_PROFITABILITY_RANKING_METRIC_KEYS
    inspect_metric_keys: tuple[str, ...] = FUNDAMENTAL_QUALITY_PROFITABILITY_INSPECT_METRIC_KEYS
    horizon_order: tuple[str, ...] = FUNDAMENTAL_QUALITY_PROFITABILITY_HORIZON_ORDER
    rank_interpretation_note: str = FUNDAMENTAL_QUALITY_PROFITABILITY_RANK_INTERPRETATION_NOTE

    def validate(self) -> None:
        if self.profile_id != FUNDAMENTAL_QUALITY_PROFITABILITY_PROFILE_ID:
            raise ValidationError("profile_id must be fundamental_quality_profitability_v1")
        if self.snapshot_metric_keys != FUNDAMENTAL_QUALITY_PROFITABILITY_SNAPSHOT_METRIC_KEYS:
            raise ValidationError("snapshot metric keys must match fundamental_quality_profitability_v1")
        if self.ranking_metric_keys != FUNDAMENTAL_QUALITY_PROFITABILITY_RANKING_METRIC_KEYS:
            raise ValidationError("ranking metric keys must match fundamental_quality_profitability_v1")
        if self.inspect_metric_keys != self.snapshot_metric_keys:
            raise ValidationError("inspect metric keys must match snapshot metric keys")
        if self.horizon_order != FUNDAMENTAL_QUALITY_PROFITABILITY_HORIZON_ORDER:
            raise ValidationError("horizon_order must be composite")
        if self.rank_interpretation_note != FUNDAMENTAL_QUALITY_PROFITABILITY_RANK_INTERPRETATION_NOTE:
            raise ValidationError("rank_interpretation_note must match fundamental_quality_profitability_v1")

    def ranking_config_payload(self) -> dict[str, object]:
        return {
            "ranking_profile": self.profile_id,
            "metrics_version": FUNDAMENTAL_QUALITY_PROFITABILITY_METRICS_VERSION,
            "eligibility_rules_version": FUNDAMENTAL_QUALITY_PROFITABILITY_ELIGIBILITY_RULES_VERSION,
            "staleness_days": FUNDAMENTAL_QUALITY_PROFITABILITY_STALENESS_DAYS,
            "scoring_weights": FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS,
            "horizon_order": list(self.horizon_order),
            "snapshot_metric_keys": list(self.snapshot_metric_keys),
            "ranking_metric_keys": list(self.ranking_metric_keys),
            "inspect_metric_keys": list(self.inspect_metric_keys),
        }

    def build_snapshot(
        self,
        *,
        run_id: str,
        market: Market,
        listings: list[ListingCandidate],
        bars: pl.DataFrame,
        run_latest_bar_date: date,
        fundamentals: FundamentalsUniverseRunData | None = None,
    ) -> pl.DataFrame:
        if fundamentals is None:
            raise ProviderDataError(f"ranking profile {self.profile_id} requires fundamentals")

        listed_tickers = {item.ticker for item in listings if item.market == market}
        if not listed_tickers or bars.is_empty() or fundamentals.facts.is_empty():
            raise ProviderDataError(f"fundamentals provider returned no eligible fundamentals for {market.value}")

        filtered_bars = bars.filter(
            (pl.col("market") == market.value)
            & (pl.col("ticker").is_in(list(listed_tickers)))
            & (pl.col("bar_date") <= run_latest_bar_date)
        ).sort(["ticker", "bar_date"])
        facts = fundamentals.facts.filter(
            (pl.col("market") == market.value) & (pl.col("ticker").is_in(list(listed_tickers)))
        ).sort("ticker")

        rows: list[dict[str, object]] = []
        for fact in facts.to_dicts():
            ticker = str(fact["ticker"])
            ticker_bars = filtered_bars.filter(pl.col("ticker") == ticker)
            if ticker_bars.is_empty() or ticker_bars["bar_date"][-1] != run_latest_bar_date:
                continue
            fiscal_period_end = fact["fiscal_period_end"]
            balance_sheet_as_of = fact["balance_sheet_as_of"]
            if not isinstance(fiscal_period_end, date) or not isinstance(balance_sheet_as_of, date):
                continue
            if not _fresh_enough(run_latest_bar_date, fiscal_period_end, balance_sheet_as_of):
                continue

            numeric: dict[str, float] = {}
            required_numeric = (
                "revenue_ttm",
                "net_income_ttm",
                "total_assets",
                "shareholders_equity",
                "total_debt",
                "cash_and_cash_equivalents",
                "free_cash_flow_ttm",
                "roe",
                "roa",
                "operating_margin",
                "net_margin",
                "fcf_margin",
                "debt_to_equity",
                "tag_fundamentals_annual_fallback",
                "tag_negative_net_income",
                "tag_negative_fcf",
            )
            for key in required_numeric:
                value = _finite_float(fact.get(key))
                if value is None:
                    break
                numeric[key] = value
            if len(numeric) != len(required_numeric):
                continue
            if numeric["revenue_ttm"] <= 0 or numeric["total_assets"] <= 0 or numeric["shareholders_equity"] <= 0:
                continue
            if numeric["total_debt"] < 0 or numeric["cash_and_cash_equivalents"] < 0:
                continue

            close = _finite_float(ticker_bars["close"][-1])
            adjusted_close = _finite_float(ticker_bars["adjusted_close"][-1])
            if close is None or adjusted_close is None:
                continue

            rows.append(
                {
                    "run_id": run_id,
                    "market": market.value,
                    "ticker": ticker,
                    "close": close,
                    "adjusted_close": adjusted_close,
                    "profile_metrics_version": FUNDAMENTAL_QUALITY_PROFITABILITY_METRICS_VERSION,
                    **{
                        key: value
                        for key, value in numeric.items()
                        if key in FUNDAMENTAL_QUALITY_PROFITABILITY_SNAPSHOT_METRIC_KEYS
                    },
                }
            )

        if not rows:
            raise ProviderDataError(f"fundamentals provider returned no eligible fundamentals for {market.value}")
        return pl.DataFrame(rows, schema=FUNDAMENTAL_QUALITY_PROFITABILITY_SNAPSHOT_SCHEMA).sort("ticker")

    def assign_rankings(self, snapshot: pl.DataFrame) -> pl.DataFrame:
        if snapshot.is_empty():
            return _empty_rankings()

        rows = snapshot.to_dicts()
        values = {
            metric: {str(row["ticker"]): float(row[metric]) for row in rows}
            for metric in ("roe", "roa", "operating_margin", "net_margin", "fcf_margin", "debt_to_equity")
        }
        score_roe = _percentile_scores(values["roe"], higher_is_better=True)
        score_roa = _percentile_scores(values["roa"], higher_is_better=True)
        score_operating_margin = _percentile_scores(values["operating_margin"], higher_is_better=True)
        score_net_margin = _percentile_scores(values["net_margin"], higher_is_better=True)
        score_fcf_margin = _percentile_scores(values["fcf_margin"], higher_is_better=True)
        score_low_debt_to_equity = _percentile_scores(values["debt_to_equity"], higher_is_better=False)

        ranking_rows = []
        for row in rows:
            ticker = str(row["ticker"])
            score_positive_fcf = 100.0 if float(row["free_cash_flow_ttm"]) > 0 else 0.0
            score_profitability = (
                FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS["profitability"]["score_roe"] * score_roe[ticker]
                + FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS["profitability"]["score_roa"] * score_roa[ticker]
                + FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS["profitability"]["score_operating_margin"]
                * score_operating_margin[ticker]
                + FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS["profitability"]["score_net_margin"]
                * score_net_margin[ticker]
            )
            score_cash_generation = (
                FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS["cash_generation"]["score_fcf_margin"]
                * score_fcf_margin[ticker]
                + FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS["cash_generation"]["score_positive_fcf"]
                * score_positive_fcf
            )
            score_balance_sheet = score_low_debt_to_equity[ticker]
            score = (
                FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS["composite"]["score_profitability"]
                * score_profitability
                + FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS["composite"]["score_cash_generation"]
                * score_cash_generation
                + FUNDAMENTAL_QUALITY_PROFITABILITY_SCORING_WEIGHTS["composite"]["score_balance_sheet"]
                * score_balance_sheet
            )
            ranking_rows.append(
                {
                    "run_id": row["run_id"],
                    "market": row["market"],
                    "horizon": "composite",
                    "ticker": ticker,
                    "score_roe": score_roe[ticker],
                    "score_roa": score_roa[ticker],
                    "score_operating_margin": score_operating_margin[ticker],
                    "score_net_margin": score_net_margin[ticker],
                    "score_fcf_margin": score_fcf_margin[ticker],
                    "score_positive_fcf": score_positive_fcf,
                    "score_low_debt_to_equity": score_balance_sheet,
                    "score_profitability": score_profitability,
                    "score_cash_generation": score_cash_generation,
                    "score_balance_sheet": score_balance_sheet,
                    "tag_fundamentals_annual_fallback": float(row["tag_fundamentals_annual_fallback"]),
                    "tag_negative_net_income": float(row["tag_negative_net_income"]),
                    "tag_negative_fcf": float(row["tag_negative_fcf"]),
                    "score": score,
                    "rank": 0,
                    "roe": float(row["roe"]),
                    "free_cash_flow_ttm": float(row["free_cash_flow_ttm"]),
                }
            )

        ranking_rows.sort(
            key=lambda row: (
                -float(row["score"]),
                -float(row["score_profitability"]),
                -float(row["score_cash_generation"]),
                -float(row["score_balance_sheet"]),
                -float(row["roe"]),
                -float(row["free_cash_flow_ttm"]),
                str(row["ticker"]),
            )
        )
        for rank, row in enumerate(ranking_rows, start=1):
            row["rank"] = rank
            del row["roe"]
            del row["free_cash_flow_ttm"]
        return pl.DataFrame(ranking_rows, schema=FUNDAMENTAL_QUALITY_PROFITABILITY_RANKING_SCHEMA)


FUNDAMENTAL_QUALITY_PROFITABILITY_V1_REGISTRATION = RankingProfileRegistration(
    profile_id=FUNDAMENTAL_QUALITY_PROFITABILITY_PROFILE_ID,
    factory=FundamentalQualityProfitabilityV1Profile,
    data_requirements=RankingProfileDataRequirements(fundamentals=True),
)
