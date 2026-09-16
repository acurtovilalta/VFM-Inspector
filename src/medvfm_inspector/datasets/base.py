"""Minimal dataset adapter abstraction."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from medvfm_inspector.types import DatasetInfo, Sample

SplitName = Literal["train", "val", "test"]


class DatasetAdapter(ABC):
    """Interface for datasets that provide stable labeled image samples."""

    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Return metadata for this dataset."""

    @abstractmethod
    def get_split(self, split: SplitName) -> Sequence[Sample]:
        """Return a deterministic split sequence."""
