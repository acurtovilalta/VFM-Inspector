"""Layer-wise k-NN classification over cached CLS representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from medvfm_inspector.analysis._classification import classification_metrics
from medvfm_inspector.analysis._features import (
    cls_matrix,
    labels_array,
    matching_eval_layer,
)
from medvfm_inspector.types import ExtractedRepresentations

DistanceMetric = Literal["cosine", "euclidean"]


@dataclass(frozen=True)
class KNNResult:
    """Metrics for one layer-wise k-NN classifier."""

    layer_index: int
    layer_name: str
    train_sample_count: int
    evaluation_sample_count: int
    k: int
    metric: DistanceMetric
    accuracy: float
    balanced_accuracy: float
    auroc: float | None


def run_knn_classification(
    *,
    train: ExtractedRepresentations,
    evaluation: ExtractedRepresentations,
    k: int = 5,
    metric: DistanceMetric = "cosine",
) -> tuple[KNNResult, ...]:
    """Evaluate one k-NN classifier per selected layer."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > len(train.sample_ids):
        raise ValueError("k cannot exceed the number of training samples.")

    y_train = labels_array(train)
    y_eval = labels_array(evaluation)
    results: list[KNNResult] = []

    for train_layer in train.layers:
        eval_layer = matching_eval_layer(train_layer, evaluation)
        x_train = cls_matrix(train_layer)
        x_eval = cls_matrix(eval_layer)
        classifier = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k, metric=metric),
        )
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_eval)
        probabilities = None
        classes = None
        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(x_eval)
            knn = cast(Any, classifier.named_steps["kneighborsclassifier"])
            classes = knn.classes_
        metrics = classification_metrics(
            y_eval,
            predictions,
            probabilities=probabilities,
            classes=classes,
        )
        results.append(
            KNNResult(
                layer_index=train_layer.layer_index,
                layer_name=train_layer.layer_name,
                train_sample_count=len(train.sample_ids),
                evaluation_sample_count=len(evaluation.sample_ids),
                k=k,
                metric=metric,
                accuracy=metrics.accuracy,
                balanced_accuracy=metrics.balanced_accuracy,
                auroc=metrics.auroc,
            )
        )

    return tuple(results)
