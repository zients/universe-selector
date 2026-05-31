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
from universe_selector.valuation.formatting import (
    _format_money,
    _format_multiple,
    _format_note,
    _format_number,
    _format_pct,
    _lines_table,
    _markdown_text,
)
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    ExitMultipleDcfScenarioAssumptions,
    ExitMultipleDcfV1Assumptions,
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


_MODEL_KEYS = frozenset(
    {
        "forecast_years",
        "terminal_method",
        "starting_fcf",
        "discount_rate_basis",
        "exit_multiple_basis",
        "scenarios",
    }
)
_SCENARIO_KEYS = frozenset({"growth_rate", "discount_rate", "terminal_ev_to_fcf_multiple", "note"})


class ExitMultipleDcfV1Model:
    model_id = "exit_multiple_dcf_v1"

    def validate_assumptions(self, assumptions: Mapping[str, object]) -> ExitMultipleDcfV1Assumptions:
        unknown_keys = sorted(set(assumptions) - _MODEL_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown exit_multiple_dcf_v1 key: {unknown_keys[0]}")

        forecast_years = require_int(assumptions.get("forecast_years"), "forecast_years")
        if not 1 <= forecast_years <= 10:
            raise ValidationError("forecast_years must be from 1 through 10")

        terminal_method = require_literal(assumptions.get("terminal_method"), "terminal_method", "exit_multiple")
        starting_fcf = parse_starting_fcf(assumptions.get("starting_fcf"))
        discount_rate_basis = require_literal(
            assumptions.get("discount_rate_basis"),
            "discount_rate_basis",
            "nominal_wacc",
        )
        exit_multiple_basis = require_literal(
            assumptions.get("exit_multiple_basis"),
            "exit_multiple_basis",
            "ev_to_fcf",
        )

        scenarios_payload = assumptions.get("scenarios")
        if not isinstance(scenarios_payload, Mapping):
            raise ValidationError("scenarios must be a mapping")
        if set(scenarios_payload) != set(_SCENARIO_ORDER):
            raise ValidationError("scenarios must contain conservative, base, and upside")

        scenarios: dict[str, ExitMultipleDcfScenarioAssumptions] = {}
        for scenario_id in _SCENARIO_ORDER:
            payload = scenarios_payload[scenario_id]
            if not isinstance(payload, Mapping):
                raise ValidationError(f"scenario {scenario_id} must be a mapping")
            scenarios[scenario_id] = self._parse_scenario(scenario_id, payload)

        if not (
            scenarios["conservative"].growth_rate <= scenarios["base"].growth_rate <= scenarios["upside"].growth_rate
        ):
            raise ValidationError("scenario growth rates must satisfy conservative <= base <= upside")
        if not (
            scenarios["conservative"].terminal_ev_to_fcf_multiple
            <= scenarios["base"].terminal_ev_to_fcf_multiple
            <= scenarios["upside"].terminal_ev_to_fcf_multiple
        ):
            raise ValidationError("scenario exit multiples must satisfy conservative <= base <= upside")
        if not (
            scenarios["conservative"].discount_rate
            >= scenarios["base"].discount_rate
            >= scenarios["upside"].discount_rate
        ):
            raise ValidationError("scenario discount_rate values must satisfy conservative >= base >= upside")

        return ExitMultipleDcfV1Assumptions(
            forecast_years=forecast_years,
            terminal_method=terminal_method,
            starting_fcf=starting_fcf,
            discount_rate_basis=discount_rate_basis,
            exit_multiple_basis=exit_multiple_basis,
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
                        "present_value_terminal_value": present_value_terminal_value,
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
    ) -> ExitMultipleDcfScenarioAssumptions:
        unknown_keys = sorted(set(payload) - _SCENARIO_KEYS)
        if unknown_keys:
            raise ValidationError(f"unknown exit_multiple_dcf_v1 scenario key: {unknown_keys[0]}")

        growth_rate = require_rate(payload.get("growth_rate"), f"scenarios.{scenario_id}.growth_rate")
        discount_rate = require_rate(payload.get("discount_rate"), f"scenarios.{scenario_id}.discount_rate")
        terminal_multiple = require_rate(
            payload.get("terminal_ev_to_fcf_multiple"),
            f"scenarios.{scenario_id}.terminal_ev_to_fcf_multiple",
        )
        note = payload.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValidationError(f"scenarios.{scenario_id}.note is required")

        if not -1.0 < growth_rate <= 1.0:
            raise ValidationError(f"scenarios.{scenario_id}.growth_rate must be greater than -1.0 and at most 1.0")
        if not 0.0 < discount_rate <= 0.50:
            raise ValidationError(f"scenarios.{scenario_id}.discount_rate must be greater than 0.0 and at most 0.50")
        if not 0.0 < terminal_multiple <= 100.0:
            raise ValidationError(
                f"scenarios.{scenario_id}.terminal_ev_to_fcf_multiple must be greater than 0.0 and at most 100.0"
            )

        return ExitMultipleDcfScenarioAssumptions(
            scenario_id=scenario_id,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            terminal_ev_to_fcf_multiple=terminal_multiple,
            note=note.strip(),
        )


class ExitMultipleDcfV1OutputRenderer:
    model_id = "exit_multiple_dcf_v1"

    def render_risk_disclosures(self, result: ValuationResult) -> list[str]:
        del result
        return [
            (
                "- Model risk: exit_multiple_dcf_v1 uses an analyst-supplied EV / FCF exit multiple "
                "to calculate terminal value and does not infer a peer-derived multiple."
            ),
            (
                "- FCF quality risk: provider TTM FCF is a raw starting proxy, may not be "
                "analyst-normalized, is not clean unlevered FCFF, and may be affected by "
                "accounting classification, cyclicality, working capital, capex, and capital-structure effects; "
                "use starting_fcf.method override for normalized unlevered FCFF with a note."
            ),
            (
                "- Terminal-value risk: exit-multiple terminal value can dominate enterprise value; "
                "present-value terminal value and terminal-value share of EV should be reviewed."
            ),
            (
                "- Sensitivity risk: outputs are highly sensitive to starting FCF, share count, net debt, "
                "discount rate, forecast growth, and exit multiple assumptions."
            ),
            (
                "- Starting FCF limitation: exit-multiple DCF requires positive starting FCF because "
                "EV / FCF exit multiple terminal value is not meaningful when starting FCF is zero or negative."
            ),
            (
                "- Scenario risk: scenario rows are illustrative assumption cases, not probabilities, "
                "forecasts, expected outcomes, target cases, recommendations, or investment signals."
            ),
            (
                "- Output interpretation risk: model-implied value per share and spread are illustrative "
                "scenario math only, not target prices, forecasts, expected returns, recommendations, or signals."
            ),
        ]

    def render_model_assumptions(self, result: ValuationResult) -> list[str]:
        assumptions = result.run_input.assumptions.model_assumptions
        if not isinstance(assumptions, ExitMultipleDcfV1Assumptions):
            raise ValidationError("exit_multiple_dcf_v1 requires ExitMultipleDcfV1Assumptions")

        lines = [
            "## Model Assumptions",
            "",
            f"- forecast_years: {assumptions.forecast_years}",
            f"- terminal_method: {_markdown_text(assumptions.terminal_method)}",
            f"- starting_fcf_method: {_markdown_text(assumptions.starting_fcf.method)}",
            f"- discount_rate_basis: {_markdown_text(assumptions.discount_rate_basis)}",
            f"- exit_multiple_basis: {_markdown_text(assumptions.exit_multiple_basis)}",
            "",
        ]
        if assumptions.starting_fcf.method == "override":
            if assumptions.starting_fcf.value is None or assumptions.starting_fcf.note is None:
                raise ValidationError("exit_multiple_dcf_v1 override starting_fcf requires value and note")
            lines.extend(
                [
                    f"- starting_fcf_value: {_format_number(assumptions.starting_fcf.value)}",
                    f"- starting_fcf_note: {_markdown_text(assumptions.starting_fcf.note)}",
                    "",
                ]
            )
        lines.extend(
            _lines_table(
                ("scenario", "growth_rate", "discount_rate", "terminal EV / FCF multiple", "note"),
                (
                    (
                        scenario.scenario_id,
                        _format_pct(scenario.growth_rate),
                        _format_pct(scenario.discount_rate),
                        _format_multiple(scenario.terminal_ev_to_fcf_multiple),
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
        if not isinstance(assumptions, ExitMultipleDcfV1Assumptions):
            raise ValidationError("exit_multiple_dcf_v1 requires ExitMultipleDcfV1Assumptions")
        effective = result.run_input.effective_inputs
        lines = [
            "",
            "## Scenario Results",
            "",
            (
                "Exit-multiple DCF uses analyst-supplied EV / FCF exit multiples and is not peer-derived. "
                "EV / FCF exit multiple terminal value is not meaningful when starting FCF is zero or negative."
            ),
            (
                "Scenario rows are illustrative assumption cases, not probabilities, forecasts, expected outcomes, "
                "target cases, recommendations, or investment signals."
            ),
            (
                "Model-implied value per share and spread are illustrative scenario math only, not target prices, "
                "forecasts, expected returns, recommendations, or signals. "
                "Negative equity value and negative per-share values are rendered when net debt exceeds enterprise value."
            ),
            "",
        ]
        lines.extend(
            _lines_table(
                (
                    "scenario",
                    "terminal EV / FCF multiple",
                    "final_year_fcf",
                    "terminal_value",
                    "present_value_terminal_value",
                    "terminal_value_share_of_enterprise_value",
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
                        _format_multiple(scenario.model_metrics["terminal_ev_to_fcf_multiple"]),
                        _format_money(scenario.model_metrics["final_year_fcf"], effective.currency),
                        _format_money(scenario.terminal_value, effective.currency),
                        _format_money(scenario.model_metrics["present_value_terminal_value"], effective.currency),
                        _format_pct(scenario.model_metrics["present_value_terminal_value_share_of_enterprise_value"]),
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
