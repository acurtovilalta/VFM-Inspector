"""Dataset adapter interfaces and implementations."""

from medvfm_inspector.datasets.base import DatasetAdapter, SplitName
from medvfm_inspector.datasets.medmnist import MedMNISTAdapter, MedMNISTSplit

__all__ = ["DatasetAdapter", "MedMNISTAdapter", "MedMNISTSplit", "SplitName"]
