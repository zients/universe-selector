from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from universe_selector.errors import ValidationError
from universe_selector.valuation.base import ValuationModel, ValuationOutputRenderer
from universe_selector.valuation.fcf_dcf_v1 import FcfDcfV1Model, FcfDcfV1OutputRenderer


@dataclass(frozen=True)
class ValuationModelRegistration:
    model_id: str
    model: ValuationModel
    output_renderer_factory: Callable[[], ValuationOutputRenderer]


_REGISTRATIONS: Mapping[str, ValuationModelRegistration] = MappingProxyType(
    {
        "fcf_dcf_v1": ValuationModelRegistration(
            model_id="fcf_dcf_v1",
            model=FcfDcfV1Model(),
            output_renderer_factory=FcfDcfV1OutputRenderer,
        )
    }
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
    return _registration(model_id).model


def get_valuation_output_renderer(model_id: str) -> ValuationOutputRenderer:
    return _registration(model_id).output_renderer_factory()
