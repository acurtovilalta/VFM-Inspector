"""Minimal model adapter abstraction."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from medvfm_inspector.types import ModelInfo, ModelRepresentations


class ModelAdapter(ABC):
    """Interface for vision models that expose layer-wise representations."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Return metadata for the loaded model."""

    @property
    @abstractmethod
    def layer_names(self) -> tuple[str, ...]:
        """Return names for the transformer blocks exposed by this adapter."""

    @abstractmethod
    def preprocess(self, images: Sequence[Any]) -> dict[str, torch.Tensor]:
        """Convert a batch of images into tensor inputs for the model."""

    @abstractmethod
    def extract_from_inputs(
        self, model_inputs: Mapping[str, torch.Tensor]
    ) -> ModelRepresentations:
        """Run the model and return layer-wise representations."""

    def extract(self, images: Sequence[Any]) -> ModelRepresentations:
        """Preprocess images and extract layer-wise representations."""
        return self.extract_from_inputs(self.preprocess(images))
