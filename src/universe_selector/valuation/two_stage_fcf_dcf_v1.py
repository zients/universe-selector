from __future__ import annotations

from collections.abc import Mapping

from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts
from universe_selector.valuation.input_resolution import (
    _SCENARIO_ORDER,
    build_effective_inputs,
    parse_starting_fcf,
    require_int,
    require_literal,
    require_rate,
)
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    TwoStageFcfDcfScenarioAssumptions,
    TwoStageFcfDcfV1Assumptions,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationScenarioResult,
)


_MODEL_KEYS = frozenset(
    {
        "stage1_years",
        "stage2_years",
        "terminal_method",
        "starting_fcf",
        "discount_rate_basis",
        "terminal_growth_basis",
        "scenarios",
    }
)
_SCENARIO_KEYS = frozenset(
    {
        "stage1_growth_rate",
        "stage2_growth_rate",
        "discount_rate",
        "terminal_growth_rate",
        "note",
    }
)


class TwoStageFcfDcfV1Model:
    model_id = "two_stage_fcf_dcf_v1"

    def validate_assumptions(self, assumptions: object) -> TwoStageFcfDcfV1Assumptions:
        if not isinstance(assumptions, Mapping):
            raise ValidationError("two_stage_fcf_dcf_v1 assumptions must be a mapping")
        unknown_keys = sorted(set(assumptions) - _MODEL_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown two_stage_fcf_dcf_v1 key: {unknown_keys[0]}")

        stage1_years = require_int(assumptions.get("stage1_years"), "stage1_years")
        stage2_years = require_int(assumptions.get("stage2_years"), "stage2_years")
        if not 1 <= stage1_years <= 9:
            raise ValidationError("stage1_years must be from 1 through 9")
        if not 1 <= stage2_years <= 9:
            raise ValidationError("stage2_years must be from 1 through 9")
        if not 2 <= stage1_years + stage2_years <= 10:
            raise ValidationError("stage1_years + stage2_years must be from 2 through 10")

        terminal_method = require_literal(
            assumptions.get("terminal_method"),
            "terminal_method",
            "perpetual_growth",
        )
        starting_fcf = parse_starting_fcf(assumptions.get("starting_fcf"))
        discount_rate_basis = require_literal(
            assumptions.get("discount_rate_basis"),
            "discount_rate_basis",
            "nominal_wacc",
        )
        terminal_growth_basis = require_literal(
            assumptions.get("terminal_growth_basis"),
            "terminal_growth_basis",
            "nominal_perpetual_growth",
        )

        scenarios_payload = assumptions.get("scenarios")
        if not isinstance(scenarios_payload, Mapping):
            raise ValidationError("scenarios must be a mapping")
        if set(scenarios_payload) != set(_SCENARIO_ORDER):
            raise ValidationError("scenarios must contain conservative, base, and upside")

        scenarios: dict[str, TwoStageFcfDcfScenarioAssumptions] = {}
        for scenario_id in _SCENARIO_ORDER:
            payload = scenarios_payload[scenario_id]
            if not isinstance(payload, Mapping):
                raise ValidationError(f"scenario {scenario_id} must be a mapping")
            scenarios[scenario_id] = self._parse_scenario(scenario_id, payload)

        if not (
            scenarios["conservative"].stage1_growth_rate
            <= scenarios["base"].stage1_growth_rate
            <= scenarios["upside"].stage1_growth_rate
        ):
            raise ValidationError("scenario stage1_growth_rate values must satisfy conservative <= base <= upside")
        if not (
            scenarios["conservative"].stage2_growth_rate
            <= scenarios["base"].stage2_growth_rate
            <= scenarios["upside"].stage2_growth_rate
        ):
            raise ValidationError("scenario stage2_growth_rate values must satisfy conservative <= base <= upside")
        if not (
            scenarios["conservative"].terminal_growth_rate
            <= scenarios["base"].terminal_growth_rate
            <= scenarios["upside"].terminal_growth_rate
        ):
            raise ValidationError("scenario terminal_growth_rate values must satisfy conservative <= base <= upside")
        if not (
            scenarios["conservative"].discount_rate
            >= scenarios["base"].discount_rate
            >= scenarios["upside"].discount_rate
        ):
            raise ValidationError("scenario discount_rate values must satisfy conservative >= base >= upside")

        return TwoStageFcfDcfV1Assumptions(
            stage1_years=stage1_years,
            stage2_years=stage2_years,
            terminal_method=terminal_method,
            starting_fcf=starting_fcf,
            discount_rate_basis=discount_rate_basis,
            terminal_growth_basis=terminal_growth_basis,
            scenario_order=_SCENARIO_ORDER,
            scenarios=scenarios,
        )

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

    def _parse_scenario(
        self,
        scenario_id: str,
        payload: Mapping[str, object],
    ) -> TwoStageFcfDcfScenarioAssumptions:
        unknown_keys = sorted(set(payload) - _SCENARIO_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown two_stage_fcf_dcf_v1 scenario key: {unknown_keys[0]}")

        stage1_growth_rate = require_rate(
            payload.get("stage1_growth_rate"),
            f"scenarios.{scenario_id}.stage1_growth_rate",
        )
        stage2_growth_rate = require_rate(
            payload.get("stage2_growth_rate"),
            f"scenarios.{scenario_id}.stage2_growth_rate",
        )
        discount_rate = require_rate(payload.get("discount_rate"), f"scenarios.{scenario_id}.discount_rate")
        terminal_growth_rate = require_rate(
            payload.get("terminal_growth_rate"),
            f"scenarios.{scenario_id}.terminal_growth_rate",
        )
        note = payload.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValidationError(f"scenarios.{scenario_id}.note is required")

        if not -1.0 < stage1_growth_rate <= 1.0:
            raise ValidationError(
                f"scenarios.{scenario_id}.stage1_growth_rate must be greater than -1.0 and at most 1.0"
            )
        if not -1.0 < stage2_growth_rate <= 1.0:
            raise ValidationError(
                f"scenarios.{scenario_id}.stage2_growth_rate must be greater than -1.0 and at most 1.0"
            )
        if not -0.05 <= terminal_growth_rate <= 0.05:
            raise ValidationError(f"scenarios.{scenario_id}.terminal_growth_rate must be between -0.05 and 0.05")
        if not 0.0 < discount_rate <= 0.50:
            raise ValidationError(f"scenarios.{scenario_id}.discount_rate must be greater than 0.0 and at most 0.50")
        if discount_rate - terminal_growth_rate < 0.03:
            raise ValidationError(f"scenarios.{scenario_id}.discount_rate - terminal_growth_rate must be at least 0.03")
        if stage1_growth_rate < stage2_growth_rate:
            raise ValidationError(f"scenarios.{scenario_id}.stage1_growth_rate must be at least stage2_growth_rate")
        if stage2_growth_rate < terminal_growth_rate:
            raise ValidationError(f"scenarios.{scenario_id}.stage2_growth_rate must be at least terminal_growth_rate")

        return TwoStageFcfDcfScenarioAssumptions(
            scenario_id=scenario_id,
            stage1_growth_rate=stage1_growth_rate,
            stage2_growth_rate=stage2_growth_rate,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            note=note.strip(),
        )


def _validate_two_stage_inputs(inputs: EffectiveValuationInputs) -> None:
    if inputs.starting_fcf <= 0:
        raise ValidationError("starting_fcf must be greater than zero")
    if inputs.shares_outstanding <= 0:
        raise ValidationError("shares_outstanding must be greater than zero")
    if inputs.reference_price <= 0:
        raise ValidationError("reference_price must be greater than zero")
