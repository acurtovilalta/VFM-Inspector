"""Linear CKA over cached CLS representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from medvfm_inspector.analysis._features import cls_matrix
from medvfm_inspector.types import ExtractedRepresentations, ModelInfo


@dataclass(frozen=True)
class CKALayerMetadata:
    """Layer identity included with a CKA matrix."""

    layer_index: int
    layer_name: str


@dataclass(frozen=True)
class CKAResult:
    """Layer-by-layer linear CKA result."""

    source_model: ModelInfo
    target_model: ModelInfo
    source_layers: tuple[CKALayerMetadata, ...]
    target_layers: tuple[CKALayerMetadata, ...]
    matrix: NDArray[np.float64]


def linear_cka(
    source: ExtractedRepresentations,
    target: ExtractedRepresentations,
) -> CKAResult:
    """Compute feature-space linear CKA between two representation sets."""
    _validate_sample_alignment(source, target)
    matrix = np.empty((len(source.layers), len(target.layers)), dtype=np.float64)

    for source_index, source_layer in enumerate(source.layers):
        source_features = _center(cls_matrix(source_layer))
        for target_index, target_layer in enumerate(target.layers):
            target_features = _center(cls_matrix(target_layer))
            matrix[source_index, target_index] = _linear_cka_value(
                source_features, target_features
            )

    return CKAResult(
        source_model=source.model,
        target_model=target.model,
        source_layers=tuple(
            CKALayerMetadata(layer.layer_index, layer.layer_name)
            for layer in source.layers
        ),
        target_layers=tuple(
            CKALayerMetadata(layer.layer_index, layer.layer_name)
            for layer in target.layers
        ),
        matrix=matrix,
    )


def within_model_cka(representations: ExtractedRepresentations) -> CKAResult:
    """Compute layer-to-layer linear CKA within one representation set."""
    return linear_cka(representations, representations)


def _validate_sample_alignment(
    source: ExtractedRepresentations, target: ExtractedRepresentations
) -> None:
    if source.sample_ids != target.sample_ids:
        raise ValueError("CKA requires identical sample IDs in the same order.")


def _center(features: NDArray[np.float64]) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], features - features.mean(axis=0, keepdims=True))


def _linear_cka_value(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    cross_covariance = source.T @ target
    numerator = float(np.linalg.norm(cross_covariance, ord="fro") ** 2)
    source_norm = float(np.linalg.norm(source.T @ source, ord="fro"))
    target_norm = float(np.linalg.norm(target.T @ target, ord="fro"))
    denominator = source_norm * target_norm
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
