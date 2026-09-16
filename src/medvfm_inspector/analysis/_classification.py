"""Small classification metric helpers for analysis modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


@dataclass(frozen=True)
class ClassificationMetrics:
    """Common classification metrics for layer-wise analyses."""

    accuracy: float
    balanced_accuracy: float
    auroc: float | None


def classification_metrics(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    *,
    probabilities: NDArray[np.float64] | None = None,
    classes: NDArray[Any] | None = None,
) -> ClassificationMetrics:
    """Compute standard classification metrics with graceful AUROC handling."""
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        auroc=_auroc(y_true, probabilities=probabilities, classes=classes),
    )


def _auroc(
    y_true: NDArray[np.int_],
    *,
    probabilities: NDArray[np.float64] | None,
    classes: NDArray[Any] | None,
) -> float | None:
    if probabilities is None or classes is None or np.unique(y_true).size < 2:
        return None

    try:
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return None
