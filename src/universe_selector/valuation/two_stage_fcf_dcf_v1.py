from __future__ import annotations

from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts
from universe_selector.valuation.input_resolution import build_effective_inputs
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    TwoStageFcfDcfV1Assumptions,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationScenarioResult,
)


class TwoStageFcfDcfV1Model:
    model_id = "two_stage_fcf_dcf_v1"

    def validate_assumptions(self, assumptions):
        raise NotImplementedError

    def build_inputs(
        self,
        *,
        facts: FundamentalFacts,
        assumptions: ValuationAssumptionSet,
    ) -> tuple[EffectiveValuationInputs, ValuationInputProvenance]:
        if assumptions.model_id != self.model_id:
            raise ValidationError(f"{self.model_id} cannot build inputs for {assumptions.model_id}")
        if not isinstance(assumptions.model_assumptions, TwoStageFcfDcfV1Assumptions):
            raise ValidationError("two_stage_fcf_dcf_v1 requires TwoStageFcfDcfV1Assumptions")
        return build_effective_inputs(
            facts=facts,
            assumptions=assumptions,
            starting_fcf=assumptions.model_assumptions.starting_fcf,
        )

    def value(self, model_input: ValuationModelInput) -> tuple[ValuationScenarioResult, ...]:
        assumptions = model_input.model_assumptions
        if not isinstance(assumptions, TwoStageFcfDcfV1Assumptions):
            raise ValidationError("two_stage_fcf_dcf_v1 requires TwoStageFcfDcfV1Assumptions")

        inputs = model_input.effective_inputs
        _validate_two_stage_inputs(inputs)
        results: list[ValuationScenarioResult] = []
        total_years = assumptions.stage1_years + assumptions.stage2_years
        for scenario_id in assumptions.scenario_order:
            scenario = assumptions.scenarios[scenario_id]
            projected_fcf_values: list[float] = []
            fcf = inputs.starting_fcf
            for year in range(1, total_years + 1):
                growth_rate = (
                    scenario.stage1_growth_rate if year <= assumptions.stage1_years else scenario.stage2_growth_rate
                )
                fcf *= 1.0 + growth_rate
                projected_fcf_values.append(fcf)

            projected_fcf = tuple(projected_fcf_values)
            present_value_projected_fcf = tuple(
                projected / (1.0 + scenario.discount_rate) ** year
                for year, projected in enumerate(projected_fcf, start=1)
            )
            stage1_final_fcf = projected_fcf[assumptions.stage1_years - 1]
            final_year_fcf = projected_fcf[-1]
            terminal_value = (
                final_year_fcf
                * (1.0 + scenario.terminal_growth_rate)
                / (scenario.discount_rate - scenario.terminal_growth_rate)
            )
            present_value_terminal_value = terminal_value / ((1.0 + scenario.discount_rate) ** total_years)
            enterprise_value = sum(present_value_projected_fcf) + present_value_terminal_value
            equity_value = enterprise_value - inputs.net_debt
            model_implied_value_per_share = equity_value / inputs.shares_outstanding

            results.append(
                ValuationScenarioResult(
                    scenario_id=scenario_id,
                    projected_fcf=projected_fcf,
                    present_value_projected_fcf=present_value_projected_fcf,
                    terminal_value=terminal_value,
                    present_value_terminal_value=present_value_terminal_value,
                    enterprise_value=enterprise_value,
                    equity_value=equity_value,
                    model_implied_value_per_share=model_implied_value_per_share,
                    reference_price=inputs.reference_price,
                    model_implied_spread_to_reference_price=(
                        model_implied_value_per_share / inputs.reference_price - 1.0
                    ),
                    model_metrics={
                        "stage1_final_fcf": stage1_final_fcf,
                        "final_year_fcf": final_year_fcf,
                        "present_value_terminal_value_share_of_enterprise_value": (
                            present_value_terminal_value / enterprise_value
                        ),
                    },
                )
            )
        return tuple(results)


def _validate_two_stage_inputs(inputs: EffectiveValuationInputs) -> None:
    if inputs.starting_fcf <= 0:
        raise ValidationError("starting_fcf must be greater than zero")
    if inputs.shares_outstanding <= 0:
        raise ValidationError("shares_outstanding must be greater than zero")
    if inputs.reference_price <= 0:
        raise ValidationError("reference_price must be greater than zero")
