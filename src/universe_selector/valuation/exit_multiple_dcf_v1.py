from __future__ import annotations

from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts
from universe_selector.valuation.input_resolution import build_effective_inputs
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    ExitMultipleDcfV1Assumptions,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationScenarioResult,
)


class ExitMultipleDcfV1Model:
    model_id = "exit_multiple_dcf_v1"

    def build_inputs(
        self,
        *,
        facts: FundamentalFacts,
        assumptions: ValuationAssumptionSet,
    ) -> tuple[EffectiveValuationInputs, ValuationInputProvenance]:
        if assumptions.model_id != self.model_id:
            raise ValidationError(f"{self.model_id} cannot build inputs for {assumptions.model_id}")
        if not isinstance(assumptions.model_assumptions, ExitMultipleDcfV1Assumptions):
            raise ValidationError("exit_multiple_dcf_v1 requires ExitMultipleDcfV1Assumptions")
        return build_effective_inputs(
            facts=facts,
            assumptions=assumptions,
            starting_fcf=assumptions.model_assumptions.starting_fcf,
        )

    def value(self, model_input: ValuationModelInput) -> tuple[ValuationScenarioResult, ...]:
        assumptions = model_input.model_assumptions
        if not isinstance(assumptions, ExitMultipleDcfV1Assumptions):
            raise ValidationError("exit_multiple_dcf_v1 requires ExitMultipleDcfV1Assumptions")

        inputs = model_input.effective_inputs
        _validate_exit_multiple_inputs(inputs)
        results: list[ValuationScenarioResult] = []
        for scenario_id in assumptions.scenario_order:
            scenario = assumptions.scenarios[scenario_id]
            projected_fcf = tuple(
                inputs.starting_fcf * (1.0 + scenario.growth_rate) ** year
                for year in range(1, assumptions.forecast_years + 1)
            )
            present_value_projected_fcf = tuple(
                fcf / (1.0 + scenario.discount_rate) ** year for year, fcf in enumerate(projected_fcf, start=1)
            )
            final_year_fcf = projected_fcf[-1]
            terminal_value = final_year_fcf * scenario.terminal_ev_to_fcf_multiple
            present_value_terminal_value = terminal_value / (
                (1.0 + scenario.discount_rate) ** assumptions.forecast_years
            )
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
                        "terminal_ev_to_fcf_multiple": scenario.terminal_ev_to_fcf_multiple,
                        "final_year_fcf": final_year_fcf,
                    },
                )
            )
        return tuple(results)


def _validate_exit_multiple_inputs(inputs: EffectiveValuationInputs) -> None:
    if inputs.starting_fcf <= 0:
        raise ValidationError(
            "starting_fcf must be greater than zero; EV / FCF exit multiple DCF is not meaningful "
            "when starting FCF is zero or negative"
        )
    if inputs.shares_outstanding <= 0:
        raise ValidationError("shares_outstanding must be greater than zero")
    if inputs.reference_price <= 0:
        raise ValidationError("reference_price must be greater than zero")
