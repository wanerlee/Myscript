"""Basic NPU MoE cache contracts for design review, not a runtime implementation.

One target forward at a time. Weight copies finish before returning.
Each layer has R fixed resident slots and one temporary slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from torch import Tensor

ForwardMode = Literal["prefill", "decode", "verify"]


@dataclass
class CacheConfig:
    resident_slots_per_layer: int
    update_interval: int = 4
    min_usage: float = 2.0
    replacement_ratio: float = 1.5
    max_swaps_per_rank: int = 96
    decay: float = 0.7


@dataclass
class ExpertWeights:
    w13: Tensor
    w2: Tensor
    extras: dict[str, Tensor] = field(default_factory=dict)
    # Scale, offset and bias components required by the selected format.


@dataclass
class LayerCache:
    cpu_weights: dict[int, ExpertWeights]  # This rank's owned expert ids.
    slots: list[ExpertWeights]  # R+1 device buffers, allocated once.
    slot_experts: list[int | None]
    resident: dict[int, int]  # expert_id -> slot_id in [0, R).
    usage: dict[int, float]


class MoeExpertCache(Protocol):
    config: CacheConfig
    layers: dict[int, LayerCache]
    mode: ForwardMode

    def initialize(
        self,
        cpu_weights: dict[int, dict[int, ExpertWeights]],
        offline_usage: dict[int, dict[int, float]],
        config: CacheConfig,
    ) -> None:
        """Allocate pools and synchronously load top-R experts per layer.

        Inputs contain this rank's participating layers and owned experts.
        Offline scores select initial residents; online usage starts at zero.
        """
        ...

    def begin_forward(self, mode: ForwardMode) -> None:
        """Set mode; prefill also saves and clears resident mappings."""
        ...

    def get_slot(self, layer_id: int, expert_id: int) -> int:
        """Decode/verify: return resident slot or load into temporary slot R.

        A miss waits for previous compute and finishes its copy before return.
        Submit this expert's compute before requesting the next expert slot.
        """
        ...

    def load_prefill_group(
        self, layer_id: int, expert_ids: list[int]
    ) -> list[int]:
        """Synchronously load up to R+1 unique experts into slots 0..n-1.

        Submit this group's compute before loading the next group.
        """
        ...

    def add_usage(self, layer_id: int, counts: dict[int, int]) -> None:
        """Decode/verify only: add valid route counts once per layer."""
        ...

    def end_forward(self) -> None:
        """Wait for compute, then restore prefill or advance decode counter.

        Every fourth decode/verify forward replaces selected residents in
        place and decays usage. All copying and table updates finish here.
        """
        ...


class CachedMoERunner(Protocol):
    def run(
        self, layer_id: int, dispatch_output: Any, cache: MoeExpertCache
    ) -> Any:
        """Group tokens, compute using slot weights, return original CombineInput."""
        ...
