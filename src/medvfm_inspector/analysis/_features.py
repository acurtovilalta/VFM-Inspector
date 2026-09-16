"""Helpers for reading standardized cached representations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medvfm_inspector.types import ExtractedRepresentations, LayerRepresentation


def labels_array(representations: ExtractedRepresentations) -> NDArray[np.int_]:
    """Return representation labels as a NumPy integer array."""
    return np.asarray(representations.labels, dtype=int)


def cls_matrix(layer: LayerRepresentation) -> NDArray[np.float64]:
    """Return a layer CLS tensor as a two-dimensional NumPy array."""
    if layer.cls is None:
        raise ValueError(f"Layer {layer.layer_index} has no CLS representations.")
    if layer.cls.ndim != 2:
        raise ValueError(
            f"Layer {layer.layer_index} CLS representations must be 2D; "
            f"got shape {tuple(layer.cls.shape)}."
        )
    return layer.cls.detach().cpu().numpy()


def matching_eval_layer(
    train_layer: LayerRepresentation,
    evaluation: ExtractedRepresentations,
) -> LayerRepresentation:
    """Return the evaluation layer matching a training layer index."""
    matches = [
        layer
        for layer in evaluation.layers
        if layer.layer_index == train_layer.layer_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one eval layer {train_layer.layer_index}; found {len(matches)}."
        )
    return matches[0]
