"""Thin orchestration for comparing cached model representations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from medvfm_inspector.analysis.cka import CKAResult, linear_cka
from medvfm_inspector.analysis.knn import DistanceMetric, run_knn_classification
from medvfm_inspector.analysis.linear_probe import run_linear_probe
from medvfm_inspector.analysis.pca import run_pca
from medvfm_inspector.types import (
    DatasetInfo,
    ExtractedRepresentations,
    LayerRepresentation,
    ModelInfo,
)

AnalysisName = Literal["linear_probe", "knn", "cka", "pca"]
_DEFAULT_ANALYSES: tuple[AnalysisName, ...] = (
    "linear_probe",
    "knn",
    "cka",
    "pca",
)


@dataclass(frozen=True)
class ModelComparisonInput:
    """Train/evaluation representations for one already-extracted model."""

    name: str
    train: ExtractedRepresentations
    evaluation: ExtractedRepresentations


@dataclass(frozen=True)
class ComparedClassificationResult:
    """Layer-wise classification result with model identity attached."""

    model_name: str
    model: ModelInfo
    layer_index: int
    layer_name: str
    train_sample_count: int
    evaluation_sample_count: int
    accuracy: float
    balanced_accuracy: float
    auroc: float | None
    analysis: Literal["linear_probe", "knn"]
    k: int | None = None
    metric: str | None = None


@dataclass(frozen=True)
class ComparedCKAResult:
    """Cross-model CKA result with model-pair identity attached."""

    source_model_name: str
    target_model_name: str
    result: CKAResult


@dataclass(frozen=True)
class ComparedPCAResult:
    """PCA coordinates with model identity attached."""

    model_name: str
    model: ModelInfo
    layer_index: int
    layer_name: str
    sample_ids: tuple[str, ...]
    labels: tuple[int, ...]
    coordinates: NDArray[np.float64]
    explained_variance_ratio: tuple[float, ...]


@dataclass(frozen=True)
class ModelComparisonResult:
    """Structured outputs for a multi-model representation comparison."""

    dataset: DatasetInfo
    train_split: str
    evaluation_split: str
    models: tuple[ModelComparisonInput, ...]
    linear_probe: tuple[ComparedClassificationResult, ...]
    knn: tuple[ComparedClassificationResult, ...]
    cka: tuple[ComparedCKAResult, ...]
    pca: tuple[ComparedPCAResult, ...]


def compare_models(
    inputs: Sequence[ModelComparisonInput],
    *,
    analyses: Sequence[AnalysisName] = _DEFAULT_ANALYSES,
    random_seed: int = 0,
    knn_k: int = 5,
    knn_metric: DistanceMetric = "cosine",
    pca_n_components: int = 2,
    pca_layer_indices: Sequence[int] | None = None,
) -> ModelComparisonResult:
    """Run selected analyses over compatible cached model representations."""
    model_inputs = tuple(inputs)
    _validate_inputs(model_inputs)
    requested = _normalize_analyses(analyses)

    linear_results: list[ComparedClassificationResult] = []
    knn_results: list[ComparedClassificationResult] = []
    cka_results: list[ComparedCKAResult] = []
    pca_results: list[ComparedPCAResult] = []

    for model_input in model_inputs:
        if "linear_probe" in requested:
            linear_results.extend(_linear_results(model_input, random_seed=random_seed))
        if "knn" in requested:
            knn_results.extend(_knn_results(model_input, k=knn_k, metric=knn_metric))
        if "pca" in requested:
            pca_representations = _select_layers(
                model_input.evaluation, pca_layer_indices
            )
            pca_results.extend(
                _pca_results(
                    model_input,
                    representations=pca_representations,
                    n_components=pca_n_components,
                )
            )

    if "cka" in requested:
        for source_index, source in enumerate(model_inputs):
            for target in model_inputs[source_index + 1 :]:
                cka_results.append(
                    ComparedCKAResult(
                        source_model_name=source.name,
                        target_model_name=target.name,
                        result=linear_cka(source.evaluation, target.evaluation),
                    )
                )

    first = model_inputs[0]
    return ModelComparisonResult(
        dataset=first.evaluation.dataset,
        train_split=first.train.split,
        evaluation_split=first.evaluation.split,
        models=model_inputs,
        linear_probe=tuple(linear_results),
        knn=tuple(knn_results),
        cka=tuple(cka_results),
        pca=tuple(pca_results),
    )


def _linear_results(
    model_input: ModelComparisonInput, *, random_seed: int
) -> tuple[ComparedClassificationResult, ...]:
    return tuple(
        ComparedClassificationResult(
            model_name=model_input.name,
            model=model_input.evaluation.model,
            layer_index=result.layer_index,
            layer_name=result.layer_name,
            train_sample_count=result.train_sample_count,
            evaluation_sample_count=result.evaluation_sample_count,
            accuracy=result.accuracy,
            balanced_accuracy=result.balanced_accuracy,
            auroc=result.auroc,
            analysis="linear_probe",
        )
        for result in run_linear_probe(
            train=model_input.train,
            evaluation=model_input.evaluation,
            random_seed=random_seed,
        )
    )


def _knn_results(
    model_input: ModelComparisonInput, *, k: int, metric: DistanceMetric
) -> tuple[ComparedClassificationResult, ...]:
    return tuple(
        ComparedClassificationResult(
            model_name=model_input.name,
            model=model_input.evaluation.model,
            layer_index=result.layer_index,
            layer_name=result.layer_name,
            train_sample_count=result.train_sample_count,
            evaluation_sample_count=result.evaluation_sample_count,
            accuracy=result.accuracy,
            balanced_accuracy=result.balanced_accuracy,
            auroc=result.auroc,
            analysis="knn",
            k=result.k,
            metric=result.metric,
        )
        for result in run_knn_classification(
            train=model_input.train,
            evaluation=model_input.evaluation,
            k=k,
            metric=metric,
        )
    )


def _pca_results(
    model_input: ModelComparisonInput,
    *,
    representations: ExtractedRepresentations,
    n_components: int,
) -> tuple[ComparedPCAResult, ...]:
    return tuple(
        ComparedPCAResult(
            model_name=model_input.name,
            model=result.model,
            layer_index=result.layer_index,
            layer_name=result.layer_name,
            sample_ids=result.sample_ids,
            labels=result.labels,
            coordinates=result.coordinates,
            explained_variance_ratio=result.explained_variance_ratio,
        )
        for result in run_pca(representations, n_components=n_components)
    )


def _validate_inputs(inputs: tuple[ModelComparisonInput, ...]) -> None:
    if len(inputs) < 2:
        raise ValueError("At least two model inputs are required for comparison.")
    names = [model_input.name for model_input in inputs]
    if len(set(names)) != len(names):
        raise ValueError("Model comparison input names must be unique.")

    first = inputs[0]
    for model_input in inputs:
        if model_input.train.model != model_input.evaluation.model:
            raise ValueError(
                f"Train/evaluation model metadata differs for {model_input.name}."
            )
        if model_input.train.dataset != first.train.dataset:
            raise ValueError("Training caches must use compatible datasets.")
        if model_input.evaluation.dataset != first.evaluation.dataset:
            raise ValueError("Evaluation caches must use compatible datasets.")
        if model_input.train.split != first.train.split:
            raise ValueError("Training caches must use the same split.")
        if model_input.evaluation.split != first.evaluation.split:
            raise ValueError("Evaluation caches must use the same split.")
        if model_input.train.sample_ids != first.train.sample_ids:
            raise ValueError("Training caches must use the same sample IDs.")
        if model_input.evaluation.sample_ids != first.evaluation.sample_ids:
            raise ValueError("Evaluation caches must use the same sample IDs.")
        if model_input.train.labels != first.train.labels:
            raise ValueError("Training caches must use the same labels.")
        if model_input.evaluation.labels != first.evaluation.labels:
            raise ValueError("Evaluation caches must use the same labels.")


def _normalize_analyses(analyses: Sequence[AnalysisName]) -> set[AnalysisName]:
    requested = set(analyses)
    if not requested:
        raise ValueError("At least one analysis must be selected.")
    unknown = requested - set(_DEFAULT_ANALYSES)
    if unknown:
        raise ValueError(f"Unsupported analyses: {sorted(unknown)}")
    return requested


def _select_layers(
    representations: ExtractedRepresentations,
    layer_indices: Sequence[int] | None,
) -> ExtractedRepresentations:
    if layer_indices is None:
        return representations

    selected = tuple(layer_indices)
    layers: list[LayerRepresentation] = []
    for layer_index in selected:
        matches = [
            layer
            for layer in representations.layers
            if layer.layer_index == layer_index
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one layer {layer_index}; found {len(matches)}.")
        layers.append(matches[0])

    return ExtractedRepresentations(
        model=representations.model,
        dataset=representations.dataset,
        split=representations.split,
        sample_ids=representations.sample_ids,
        labels=representations.labels,
        layers=tuple(layers),
    )
