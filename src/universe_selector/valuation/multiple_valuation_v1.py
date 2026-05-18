from __future__ import annotations

from collections.abc import Mapping

from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts
from universe_selector.valuation.formatting import (
    _format_money,
    _format_multiple,
    _format_note,
    _format_pct,
    _lines_table,
    _markdown_text,
)
from universe_selector.valuation.input_resolution import (
    _SCENARIO_ORDER,
    build_effective_inputs,
    parse_starting_fcf,
    require_literal,
    require_rate,
)
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    MultipleValuationScenarioAssumptions,
    MultipleValuationV1Assumptions,
    ValuationAssumptionSet,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationResult,
    ValuationScenarioResult,
)
from universe_selector.valuation.output_sections import (
    render_effective_inputs_section,
    render_input_provenance_section,
)


_MODEL_KEYS = frozenset({"starting_fcf", "multiple_basis", "scenarios"})
_SCENARIO_KEYS = frozenset({"ev_to_fcf_multiple", "note"})


class MultipleValuationV1Model:
    model_id = "multiple_valuation_v1"

    def validate_assumptions(self, assumptions: Mapping[str, object]) -> MultipleValuationV1Assumptions:
        unknown_keys = sorted(set(assumptions) - _MODEL_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown multiple_valuation_v1 key: {unknown_keys[0]}")

        starting_fcf = parse_starting_fcf(assumptions.get("starting_fcf"))
        multiple_basis = require_literal(
            assumptions.get("multiple_basis"),
            "multiple_basis",
            "ev_to_fcf",
        )
        scenarios_payload = assumptions.get("scenarios")
        if not isinstance(scenarios_payload, Mapping):
            raise ValidationError("scenarios must be a mapping")
        scenario_ids = set(scenarios_payload)
        if scenario_ids != set(_SCENARIO_ORDER):
            raise ValidationError("scenarios must contain conservative, base, and upside")

        scenarios: dict[str, MultipleValuationScenarioAssumptions] = {}
        for scenario_id in _SCENARIO_ORDER:
            payload = scenarios_payload[scenario_id]
            if not isinstance(payload, Mapping):
                raise ValidationError(f"scenario {scenario_id} must be a mapping")
            scenarios[scenario_id] = self._parse_scenario(scenario_id, payload)

        if not (
            scenarios["conservative"].ev_to_fcf_multiple
            <= scenarios["base"].ev_to_fcf_multiple
            <= scenarios["upside"].ev_to_fcf_multiple
        ):
            raise ValidationError("scenario multiples must satisfy conservative <= base <= upside")

        return MultipleValuationV1Assumptions(
            starting_fcf=starting_fcf,
            multiple_basis=multiple_basis,
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
        if not isinstance(assumptions.model_assumptions, MultipleValuationV1Assumptions):
            raise ValidationError("multiple_valuation_v1 requires MultipleValuationV1Assumptions")

        return build_effective_inputs(
            facts=facts,
            assumptions=assumptions,
            starting_fcf=assumptions.model_assumptions.starting_fcf,
        )

    def value(self, model_input: ValuationModelInput) -> tuple[ValuationScenarioResult, ...]:
        assumptions = model_input.model_assumptions
        if not isinstance(assumptions, MultipleValuationV1Assumptions):
            raise ValidationError("multiple_valuation_v1 requires MultipleValuationV1Assumptions")

        inputs = model_input.effective_inputs
        _validate_multiple_inputs(inputs)
        results = []
        for scenario_id in assumptions.scenario_order:
            scenario = assumptions.scenarios[scenario_id]
            enterprise_value = inputs.starting_fcf * scenario.ev_to_fcf_multiple
            equity_value = enterprise_value - inputs.net_debt
            model_implied_value_per_share = equity_value / inputs.shares_outstanding
            results.append(
                ValuationScenarioResult(
                    scenario_id=scenario_id,
                    projected_fcf=(),
                    present_value_projected_fcf=(),
                    terminal_value=0.0,
                    present_value_terminal_value=0.0,
                    enterprise_value=enterprise_value,
                    equity_value=equity_value,
                    model_implied_value_per_share=model_implied_value_per_share,
                    reference_price=inputs.reference_price,
                    model_implied_spread_to_reference_price=(
                        model_implied_value_per_share / inputs.reference_price - 1.0
                    ),
                    model_metrics={"ev_to_fcf_multiple": scenario.ev_to_fcf_multiple},
                )
            )
        return tuple(results)

    def _parse_scenario(
        self,
        scenario_id: str,
        payload: Mapping[str, object],
    ) -> MultipleValuationScenarioAssumptions:
        unknown_keys = sorted(set(payload) - _SCENARIO_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown multiple_valuation_v1 scenario key: {unknown_keys[0]}")

        ev_to_fcf_multiple = require_rate(
            payload.get("ev_to_fcf_multiple"),
            f"scenarios.{scenario_id}.ev_to_fcf_multiple",
        )
        note = payload.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValidationError(f"scenarios.{scenario_id}.note is required")
        if not 0.0 < ev_to_fcf_multiple <= 100.0:
            raise ValidationError(
                f"scenarios.{scenario_id}.ev_to_fcf_multiple must be greater than 0.0 and at most 100.0"
            )

        return MultipleValuationScenarioAssumptions(
            scenario_id=scenario_id,
            ev_to_fcf_multiple=ev_to_fcf_multiple,
            note=note.strip(),
        )


class MultipleValuationV1OutputRenderer:
    model_id = "multiple_valuation_v1"

    def render_risk_disclosures(self, result: ValuationResult) -> list[str]:
        del result
        return [
            (
                "- Model risk: multiple_valuation_v1 applies analyst-supplied EV / FCF multiples "
                "to starting FCF and does not infer a peer-derived multiple."
            ),
            (
                "- Sensitivity risk: outputs are highly sensitive to starting FCF, share count, "
                "net debt, reference price, and selected EV / FCF multiples."
            ),
        ]

    def render_model_assumptions(self, result: ValuationResult) -> list[str]:
        assumptions = result.run_input.assumptions.model_assumptions
        if not isinstance(assumptions, MultipleValuationV1Assumptions):
            raise ValidationError("multiple_valuation_v1 requires MultipleValuationV1Assumptions")

        lines = [
            "## Model Assumptions",
            "",
            f"- starting_fcf_method: {_markdown_text(assumptions.starting_fcf.method)}",
            f"- multiple_basis: {_markdown_text(assumptions.multiple_basis)}",
            "",
        ]
        if assumptions.starting_fcf.method == "override":
            lines.extend(
                [
                    f"- starting_fcf_value: {_markdown_text(assumptions.starting_fcf.value)}",
                    f"- starting_fcf_note: {_markdown_text(assumptions.starting_fcf.note)}",
                    "",
                ]
            )
        lines.extend(
            _lines_table(
                ("scenario", "EV / FCF multiple", "note"),
                (
                    (
                        scenario.scenario_id,
                        _format_multiple(scenario.ev_to_fcf_multiple),
                        _format_note(scenario.note),
                    )
                    for scenario in (assumptions.scenarios[item] for item in assumptions.scenario_order)
                ),
            )
        )
        return lines

    def render_effective_inputs(self, result: ValuationResult) -> list[str]:
        return render_effective_inputs_section(result)

    def render_input_provenance(self, result: ValuationResult) -> list[str]:
        return render_input_provenance_section(result)

    def render_scenario_results(self, result: ValuationResult) -> list[str]:
        assumptions = result.run_input.assumptions.model_assumptions
        if not isinstance(assumptions, MultipleValuationV1Assumptions):
            raise ValidationError("multiple_valuation_v1 requires MultipleValuationV1Assumptions")
        effective = result.run_input.effective_inputs
        lines = [
            "",
            "## Scenario Results",
            "",
            (
                "EV / FCF multiple valuation uses analyst-supplied multiples and is not peer-derived. "
                "EV / FCF multiple valuation is not meaningful when starting FCF is zero or negative."
            ),
            (
                "Scenario rows are assumption cases, not probabilities, expected outcomes, or recommendations. "
                "Spread is descriptive and is not an investment signal. "
                "Displayed precision is formatting precision."
            ),
            "",
        ]
        lines.extend(
            _lines_table(
                (
                    "scenario",
                    "EV / FCF multiple",
                    "scenario_enterprise_value",
                    "net_debt",
                    "scenario_equity_value",
                    "model-implied value per share",
                    "reference price",
                    "spread vs reference price",
                    "note",
                ),
                (
                    (
                        scenario.scenario_id,
                        _format_multiple(scenario.model_metrics["ev_to_fcf_multiple"]),
                        _format_money(scenario.enterprise_value, effective.currency),
                        _format_money(effective.net_debt, effective.currency),
                        _format_money(scenario.equity_value, effective.currency),
                        _format_money(scenario.model_implied_value_per_share, effective.currency),
                        _format_money(scenario.reference_price, effective.currency),
                        _format_pct(scenario.model_implied_spread_to_reference_price),
                        _format_note(assumptions.scenarios[scenario.scenario_id].note),
                    )
                    for scenario in result.scenario_results
                ),
            )
        )
        return lines


def _validate_multiple_inputs(inputs: EffectiveValuationInputs) -> None:
    if inputs.starting_fcf <= 0:
        raise ValidationError(
            "starting_fcf must be greater than zero; EV / FCF multiple valuation is not meaningful "
            "when starting FCF is zero or negative"
        )
    if inputs.shares_outstanding <= 0:
        raise ValidationError("shares_outstanding must be greater than zero")
    if inputs.reference_price <= 0:
        raise ValidationError("reference_price must be greater than zero")
