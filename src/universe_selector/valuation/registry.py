from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from universe_selector.errors import ValidationError
from universe_selector.valuation.base import ValuationModel, ValuationOutputRenderer
from universe_selector.valuation.fcf_dcf_v1 import FcfDcfV1Model, FcfDcfV1OutputRenderer
from universe_selector.valuation.multiple_valuation_v1 import (
    MultipleValuationV1Model,
    MultipleValuationV1OutputRenderer,
)
from universe_selector.valuation.reverse_dcf_v1 import ReverseDcfV1Model, ReverseDcfV1OutputRenderer


@dataclass(frozen=True, init=False)
class ValuationModelRegistration:
    model_id: str
    output_renderer_factory: Callable[[], ValuationOutputRenderer]
    model_factory: Callable[[], ValuationModel]
    _model: ValuationModel | None = field(default=None, repr=False)

    def __init__(
        self,
        model_id: str,
        model: ValuationModel | None = None,
        output_renderer_factory: Callable[[], ValuationOutputRenderer] | None = None,
        model_factory: Callable[[], ValuationModel] | None = None,
    ) -> None:
        if output_renderer_factory is None:
            raise TypeError("output_renderer_factory is required")
        if model is not None and model_factory is not None:
            raise TypeError("model and model_factory cannot both be provided")
        if model_factory is None:
            if model is None:
                raise TypeError("model or model_factory is required")
            model_factory = lambda model=model: model

        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "output_renderer_factory", output_renderer_factory)
        object.__setattr__(self, "model_factory", model_factory)
        object.__setattr__(self, "_model", model)

    @property
    def model(self) -> ValuationModel:
        if self._model is not None:
            return self._model
        return self.model_factory()


def _build_registrations(
    registrations: tuple[ValuationModelRegistration, ...],
) -> Mapping[str, ValuationModelRegistration]:
    result: dict[str, ValuationModelRegistration] = {}
    for registration in registrations:
        if registration.model_id in result:
            raise ValidationError(f"duplicate valuation model id {registration.model_id}")

        model = registration.model_factory()
        if model.model_id != registration.model_id:
            raise ValidationError(f"model factory for {registration.model_id} returned model_id {model.model_id}")

        renderer = registration.output_renderer_factory()
        if renderer.model_id != registration.model_id:
            raise ValidationError(f"renderer factory for {registration.model_id} returned model_id {renderer.model_id}")

        result[registration.model_id] = registration
    return MappingProxyType(result)


_REGISTRATIONS: Mapping[str, ValuationModelRegistration] = _build_registrations(
    (
        ValuationModelRegistration(
            model_id="fcf_dcf_v1",
            model_factory=FcfDcfV1Model,
            output_renderer_factory=FcfDcfV1OutputRenderer,
        ),
        ValuationModelRegistration(
            model_id="multiple_valuation_v1",
            model_factory=MultipleValuationV1Model,
            output_renderer_factory=MultipleValuationV1OutputRenderer,
        ),
        ValuationModelRegistration(
            model_id="reverse_dcf_v1",
            model_factory=ReverseDcfV1Model,
            output_renderer_factory=ReverseDcfV1OutputRenderer,
        ),
    )
)


def _registration(model_id: str) -> ValuationModelRegistration:
    registration = _REGISTRATIONS.get(model_id)
    if registration is None:
        supported = ", ".join(supported_valuation_model_ids())
        raise ValidationError(f"unknown valuation model {model_id}; supported models: {supported}")
    return registration


def supported_valuation_model_ids() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRATIONS))


def get_valuation_model(model_id: str) -> ValuationModel:
    return _registration(model_id).model_factory()


def get_valuation_output_renderer(model_id: str) -> ValuationOutputRenderer:
    return _registration(model_id).output_renderer_factory()
