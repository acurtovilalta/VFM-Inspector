"""PCA over cached CLS representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from medvfm_inspector.analysis._features import cls_matrix
from medvfm_inspector.types import ExtractedRepresentations, ModelInfo


@dataclass(frozen=True)
class PCAResult:
    """PCA coordinates and metadata for one layer."""

    model: ModelInfo
    layer_index: int
    layer_name: str
    sample_ids: tuple[str, ...]
    labels: tuple[int, ...]
    coordinates: NDArray[np.float64]
    explained_variance_ratio: tuple[float, ...]


def run_pca(
    representations: ExtractedRepresentations,
    *,
    n_components: int = 2,
) -> tuple[PCAResult, ...]:
    """Project each layer's CLS representations with PCA."""
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    results: list[PCAResult] = []
    for layer in representations.layers:
        features = cls_matrix(layer)
        max_components = min(features.shape[0], features.shape[1])
        if n_components > max_components:
            raise ValueError(
                f"n_components={n_components} exceeds layer {layer.layer_index} "
                f"limit {max_components}."
            )
        estimator = PCA(n_components=n_components)
        coordinates = estimator.fit_transform(features)
        results.append(
            PCAResult(
                model=representations.model,
                layer_index=layer.layer_index,
                layer_name=layer.layer_name,
                sample_ids=representations.sample_ids,
                labels=representations.labels,
                coordinates=coordinates,
                explained_variance_ratio=tuple(
                    float(value) for value in estimator.explained_variance_ratio_
                ),
            )
        )
    return tuple(results)
