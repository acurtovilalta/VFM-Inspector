"""Shared public data structures for MedVFM-Inspector."""

from dataclasses import dataclass
from typing import Any, Literal

import torch

RepresentationType = Literal["cls", "patch_tokens", "register_tokens"]


@dataclass(frozen=True)
class Sample:
    """One labeled image sample with a stable identifier."""

    sample_id: str
    image: Any
    label: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetInfo:
    """Basic metadata describing a labeled image dataset."""

    name: str
    task_type: str
    class_names: list[str]
    split_sizes: dict[str, int]


@dataclass(frozen=True)
class ModelInfo:
    """Basic metadata describing a loaded vision model."""

    name: str
    num_layers: int
    hidden_size: int | None = None
    revision: str | None = None
    architecture: str | None = None
    patch_size: int | tuple[int, int] | None = None
    num_register_tokens: int = 0


@dataclass(frozen=True)
class LayerRepresentation:
    """Token representations for one transformer block."""

    layer_name: str
    layer_index: int
    cls: torch.Tensor | None = None
    patch_tokens: torch.Tensor | None = None
    register_tokens: torch.Tensor | None = None


@dataclass(frozen=True)
class ModelRepresentations:
    """Layer-wise representations produced by a model adapter."""

    model: ModelInfo
    layers: tuple[LayerRepresentation, ...]


@dataclass(frozen=True)
class ExtractedRepresentations:
    """Dataset-aligned layer representations ready for caching or analysis."""

    model: ModelInfo
    dataset: DatasetInfo
    split: str
    sample_ids: tuple[str, ...]
    labels: tuple[int, ...]
    layers: tuple[LayerRepresentation, ...]
