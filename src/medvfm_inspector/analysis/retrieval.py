"""Exact nearest-neighbor retrieval over cached CLS representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from medvfm_inspector.analysis._features import cls_matrix
from medvfm_inspector.types import ExtractedRepresentations, LayerRepresentation


@dataclass(frozen=True)
class NeighborResult:
    """One retrieved nearest neighbor."""

    rank: int
    sample_index: int
    sample_id: str
    label: int | None
    score: float


def nearest_neighbors(
    representations: ExtractedRepresentations,
    *,
    query: int | str,
    layer_index: int,
    top_k: int = 5,
) -> tuple[NeighborResult, ...]:
    """Return top-k cosine nearest neighbors for a query sample."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    layer = _layer_by_index(representations, layer_index)
    features = cls_matrix(layer)
    query_index = _query_index(representations.sample_ids, query)

    normalized = _l2_normalize(features)
    similarities = normalized @ normalized[query_index]
    candidates = [
        (index, float(similarities[index]))
        for index in range(features.shape[0])
        if index != query_index
    ]
    candidates.sort(key=lambda item: (-item[1], representations.sample_ids[item[0]]))

    neighbors: list[NeighborResult] = []
    for rank, (sample_index, score) in enumerate(candidates[:top_k], start=1):
        label = None
        if representations.labels:
            label = representations.labels[sample_index]
        neighbors.append(
            NeighborResult(
                rank=rank,
                sample_index=sample_index,
                sample_id=representations.sample_ids[sample_index],
                label=label,
                score=score,
            )
        )
    return tuple(neighbors)


def _layer_by_index(
    representations: ExtractedRepresentations, layer_index: int
) -> LayerRepresentation:
    matches = [
        layer for layer in representations.layers if layer.layer_index == layer_index
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one layer {layer_index}; found {len(matches)}.")
    return matches[0]


def _query_index(sample_ids: tuple[str, ...], query: int | str) -> int:
    if isinstance(query, int):
        if query < 0 or query >= len(sample_ids):
            raise ValueError(f"Query index out of range: {query}")
        return query

    try:
        return sample_ids.index(query)
    except ValueError as exc:
        raise ValueError(f"Unknown query sample ID: {query}") from exc


def _l2_normalize(features: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return cast(NDArray[np.float64], features / np.maximum(norms, 1e-12))
