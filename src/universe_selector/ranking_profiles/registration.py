from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from universe_selector.ranking_profiles.base import RankingProfile


@dataclass(frozen=True)
class RankingProfileRegistration:
    profile_id: str
    factory: Callable[[], RankingProfile]

    def create_profile(self) -> RankingProfile:
        return self.factory()


def build_ranking_profile_registration_map(
    registrations: Iterable[RankingProfileRegistration],
) -> Mapping[str, RankingProfileRegistration]:
    result: dict[str, RankingProfileRegistration] = {}
    for registration in registrations:
        if registration.profile_id in result:
            raise ValueError(f"duplicate ranking profile registration {registration.profile_id}")
        result[registration.profile_id] = registration
    return MappingProxyType(result)
