from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType

from universe_selector.domain import Market
from universe_selector.providers.models import FundamentalFacts, FundamentalsMetadata


def _empty_model_metrics() -> Mapping[str, float]:
    return MappingProxyType({})


@dataclass(frozen=True)
class ValuationScenarioAssumptions:
    scenario_id: str
    growth_rate: float
    discount_rate: float
    terminal_growth_rate: float
    note: str


@dataclass(frozen=True)
class StartingFcfAssumption:
    method: str
    value: float | None
    note: str | None


@dataclass(frozen=True)
class FcfDcfV1Assumptions:
    forecast_years: int
    terminal_method: str
    starting_fcf: StartingFcfAssumption
    discount_rate_basis: str
    terminal_growth_basis: str
    scenario_order: tuple[str, ...]
    scenarios: Mapping[str, ValuationScenarioAssumptions]


@dataclass(frozen=True)
class ExitMultipleDcfScenarioAssumptions:
    scenario_id: str
    growth_rate: float
    discount_rate: float
    terminal_ev_to_fcf_multiple: float
    note: str


@dataclass(frozen=True)
class ExitMultipleDcfV1Assumptions:
    forecast_years: int
    terminal_method: str
    starting_fcf: StartingFcfAssumption
    discount_rate_basis: str
    exit_multiple_basis: str
    scenario_order: tuple[str, ...]
    scenarios: Mapping[str, ExitMultipleDcfScenarioAssumptions]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenarios", MappingProxyType(dict(self.scenarios)))


@dataclass(frozen=True)
class TwoStageFcfDcfScenarioAssumptions:
    scenario_id: str
    stage1_growth_rate: float
    stage2_growth_rate: float
    discount_rate: float
    terminal_growth_rate: float
    note: str


@dataclass(frozen=True)
class TwoStageFcfDcfV1Assumptions:
    stage1_years: int
    stage2_years: int
    terminal_method: str
    starting_fcf: StartingFcfAssumption
    discount_rate_basis: str
    terminal_growth_basis: str
    scenario_order: tuple[str, ...]
    scenarios: Mapping[str, TwoStageFcfDcfScenarioAssumptions]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenarios", MappingProxyType(dict(self.scenarios)))


@dataclass(frozen=True)
class ImpliedDiscountRateScenarioAssumptions:
    scenario_id: str
    growth_rate: float
    terminal_growth_rate: float
    implied_discount_rate_lower_bound: float
    implied_discount_rate_upper_bound: float
    note: str


@dataclass(frozen=True)
class ImpliedDiscountRateV1Assumptions:
    forecast_years: int
    terminal_method: str
    starting_fcf: StartingFcfAssumption
    growth_rate_basis: str
    terminal_growth_basis: str
    implied_discount_rate_basis: str
    solver_abs_tolerance: float
    solver_max_iterations: int
    scenario_order: tuple[str, ...]
    scenarios: Mapping[str, ImpliedDiscountRateScenarioAssumptions]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenarios", MappingProxyType(dict(self.scenarios)))


@dataclass(frozen=True)
class ReverseDcfScenarioAssumptions:
    scenario_id: str
    discount_rate: float
    terminal_growth_rate: float
    implied_growth_lower_bound: float
    implied_growth_upper_bound: float
    note: str


@dataclass(frozen=True)
class ReverseDcfV1Assumptions:
    forecast_years: int
    terminal_method: str
    starting_fcf: StartingFcfAssumption
    discount_rate_basis: str
    terminal_growth_basis: str
    implied_growth_basis: str
    solver_abs_tolerance: float
    solver_max_iterations: int
    scenario_order: tuple[str, ...]
    scenarios: Mapping[str, ReverseDcfScenarioAssumptions]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenarios", MappingProxyType(dict(self.scenarios)))


@dataclass(frozen=True)
class MultipleValuationScenarioAssumptions:
    scenario_id: str
    ev_to_fcf_multiple: float
    note: str


@dataclass(frozen=True)
class MultipleValuationV1Assumptions:
    starting_fcf: StartingFcfAssumption
    multiple_basis: str
    scenario_order: tuple[str, ...]
    scenarios: Mapping[str, MultipleValuationScenarioAssumptions]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenarios", MappingProxyType(dict(self.scenarios)))


@dataclass(frozen=True)
class ValuationAssumptionSet:
    schema_version: int
    market: Market
    ticker: str
    default_model: str
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
    share_basis: str = "ordinary_share"
    valuation_basis_note: str = "Uses ordinary-share basis; no ADR ratio, board-lot, or currency adjustment is applied."


@dataclass(frozen=True)
class EffectiveValuationInputs:
    starting_fcf: float
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
    starting_fcf_source: str
    shares_outstanding_source: str
    net_debt_source: str
    reference_price_source: str
    starting_fcf_note: str | None
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
    model_metrics: Mapping[str, float] = field(default_factory=_empty_model_metrics)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_metrics", MappingProxyType(dict(self.model_metrics)))


@dataclass(frozen=True)
class ValuationResult:
    run_input: ValuationRunInput
    scenario_results: tuple[ValuationScenarioResult, ...]
