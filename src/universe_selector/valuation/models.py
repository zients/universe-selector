from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from universe_selector.domain import Market
from universe_selector.providers.models import FundamentalFacts, FundamentalsMetadata


@dataclass(frozen=True)
class ValuationScenarioAssumptions:
    scenario_id: str
    growth_rate: float
    discount_rate: float
    terminal_growth_rate: float
    note: str


@dataclass(frozen=True)
class FcfDcfV1Assumptions:
    forecast_years: int
    terminal_method: str
    cash_flow_basis: str
    discount_rate_basis: str
    terminal_growth_basis: str
    scenario_order: tuple[str, ...]
    scenarios: Mapping[str, ValuationScenarioAssumptions]


@dataclass(frozen=True)
class ValuationAssumptionSet:
    schema_version: int
    market: Market
    ticker: str
    purpose: str
    as_of: date
    currency: str
    amount_unit: str
    assumption_source: str
    prepared_by: str
    source_note: str
    assumption_path: str
    assumption_hash: str
    facts_overrides: Mapping[str, float | None]
    facts_override_notes: Mapping[str, str | None]
    model_id: str
    model_assumptions: object


@dataclass(frozen=True)
class EffectiveValuationInputs:
    normalized_fcf: float
    shares_outstanding: float
    net_debt: float
    reference_price: float
    currency: str
    fiscal_period_type: str
    fiscal_period_end: date
    reference_price_as_of: date
    reference_price_as_of_source: str
    reference_price_as_of_note: str | None


@dataclass(frozen=True)
class ValuationInputProvenance:
    normalized_fcf_source: str
    shares_outstanding_source: str
    net_debt_source: str
    reference_price_source: str
    normalized_fcf_note: str | None
    shares_outstanding_note: str | None
    net_debt_note: str | None
    reference_price_note: str | None


@dataclass(frozen=True)
class ValuationModelInput:
    market: Market
    ticker: str
    model_id: str
    effective_inputs: EffectiveValuationInputs
    model_assumptions: object


@dataclass(frozen=True)
class ValuationRunInput:
    market: Market
    ticker: str
    model_id: str
    fundamentals_metadata: FundamentalsMetadata
    raw_facts: FundamentalFacts
    effective_inputs: EffectiveValuationInputs
    input_provenance: ValuationInputProvenance
    assumptions: ValuationAssumptionSet


@dataclass(frozen=True)
class ValuationScenarioResult:
    scenario_id: str
    projected_fcf: tuple[float, ...]
    present_value_projected_fcf: tuple[float, ...]
    terminal_value: float
    present_value_terminal_value: float
    enterprise_value: float
    equity_value: float
    model_implied_value_per_share: float
    reference_price: float
    model_implied_spread_to_reference_price: float


@dataclass(frozen=True)
class ValuationResult:
    run_input: ValuationRunInput
    scenario_results: tuple[ValuationScenarioResult, ...]
