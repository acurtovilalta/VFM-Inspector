"""Layer-wise linear probing over cached CLS representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from medvfm_inspector.analysis._classification import classification_metrics
from medvfm_inspector.analysis._features import (
    cls_matrix,
    labels_array,
    matching_eval_layer,
)
from medvfm_inspector.types import ExtractedRepresentations


@dataclass(frozen=True)
class LinearProbeResult:
    """Metrics for one layer-wise linear probe."""

    layer_index: int
    layer_name: str
    train_sample_count: int
    evaluation_sample_count: int
    accuracy: float
    balanced_accuracy: float
    auroc: float | None


@dataclass(frozen=True)
class LinearProbeConfig:
    """Reproducibility settings for linear probing."""

    random_seed: int = 0
    max_iter: int = 1000


def run_linear_probe(
    *,
    train: ExtractedRepresentations,
    evaluation: ExtractedRepresentations,
    random_seed: int = 0,
    max_iter: int = 1000,
) -> tuple[LinearProbeResult, ...]:
    """Train and evaluate one logistic-regression probe per selected layer."""
    config = LinearProbeConfig(random_seed=random_seed, max_iter=max_iter)
    y_train = labels_array(train)
    y_eval = labels_array(evaluation)
    results: list[LinearProbeResult] = []

    for train_layer in train.layers:
        eval_layer = matching_eval_layer(train_layer, evaluation)
        x_train = cls_matrix(train_layer)
        x_eval = cls_matrix(eval_layer)
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=config.max_iter,
                random_state=config.random_seed,
            ),
        )
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_eval)
        probabilities = None
        classes = None
        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(x_eval)
            logistic_regression = cast(
                Any, classifier.named_steps["logisticregression"]
            )
            classes = logistic_regression.classes_
        metrics = classification_metrics(
            y_eval,
            predictions,
            probabilities=probabilities,
            classes=classes,
        )
        results.append(
            LinearProbeResult(
                layer_index=train_layer.layer_index,
                layer_name=train_layer.layer_name,
                train_sample_count=len(train.sample_ids),
                evaluation_sample_count=len(evaluation.sample_ids),
                accuracy=metrics.accuracy,
                balanced_accuracy=metrics.balanced_accuracy,
                auroc=metrics.auroc,
            )
        )

    return tuple(results)
