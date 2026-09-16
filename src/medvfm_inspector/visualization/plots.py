"""Matplotlib visualizations for analysis results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medvfm_inspector.analysis.cka import CKAResult
from medvfm_inspector.analysis.comparison import (
    ComparedCKAResult,
    ComparedClassificationResult,
    ComparedPCAResult,
)
from medvfm_inspector.analysis.pca import PCAResult

MetricName = Literal["accuracy", "balanced_accuracy", "auroc"]


def plot_layer_metric(
    results: Sequence[ComparedClassificationResult],
    output_path: str | Path,
    *,
    metric: MetricName = "balanced_accuracy",
    title: str | None = None,
) -> Path:
    """Plot one layer-wise metric line per model."""
    path = _prepare_path(output_path)
    grouped = _group_classification_results(results, metric)
    if not grouped:
        raise ValueError(f"No plottable {metric} values were provided.")

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for model_name in sorted(grouped):
        model_results = sorted(grouped[model_name], key=lambda item: item.layer_index)
        x_values: list[int] = []
        y_values: list[float] = []
        for result in model_results:
            value = _metric_value(result, metric)
            if value is not None:
                x_values.append(result.layer_index)
                y_values.append(value)
        ax.plot(x_values, y_values, marker="o", label=model_name)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or metric.replace("_", " ").title())
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cka_heatmap(
    cka: ComparedCKAResult | CKAResult,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot a CKA matrix heatmap."""
    path = _prepare_path(output_path)
    result = cka.result if isinstance(cka, ComparedCKAResult) else cka
    default_title = "Linear CKA"
    if isinstance(cka, ComparedCKAResult):
        default_title = f"CKA: {cka.source_model_name} vs {cka.target_model_name}"

    source_name = result.source_model.name
    target_name = result.target_model.name
    if isinstance(cka, ComparedCKAResult):
        source_name = cka.source_model_name
        target_name = cka.target_model_name

    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    image = ax.imshow(result.matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(image, ax=ax, label="CKA")
    ax.set_xlabel(target_name)
    ax.set_ylabel(source_name)
    ax.set_title(title or default_title)
    ax.set_xticks(np.arange(len(result.target_layers)))
    ax.set_yticks(np.arange(len(result.source_layers)))
    ax.set_xticklabels([str(layer.layer_index) for layer in result.target_layers])
    ax.set_yticklabels([str(layer.layer_index) for layer in result.source_layers])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pca_scatter(
    pca: ComparedPCAResult | PCAResult,
    output_path: str | Path,
    *,
    class_names: Sequence[str] | None = None,
    title: str | None = None,
) -> Path:
    """Plot two-dimensional PCA coordinates."""
    path = _prepare_path(output_path)
    coordinates = pca.coordinates
    if coordinates.shape[1] < 2:
        raise ValueError("PCA scatter requires at least two components.")

    labels = np.asarray(pca.labels, dtype=int)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    if labels.size:
        colormap = plt.get_cmap("tab10")
        for color_index, label in enumerate(sorted(set(labels.tolist()))):
            mask = labels == label
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                color=colormap(color_index % colormap.N),
                label=_class_label(label, class_names),
                s=28,
                edgecolors="none",
            )
        ax.legend(title="Class label")
    else:
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=28,
            edgecolors="none",
        )
    model_name = (
        pca.model_name if isinstance(pca, ComparedPCAResult) else pca.model.name
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or f"PCA: {model_name} layer {pca.layer_index}")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _class_label(label: int, class_names: Sequence[str] | None) -> str:
    if class_names is not None and 0 <= label < len(class_names):
        return class_names[label]
    return str(label)


def _prepare_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _group_classification_results(
    results: Sequence[ComparedClassificationResult], metric: MetricName
) -> dict[str, list[ComparedClassificationResult]]:
    grouped: dict[str, list[ComparedClassificationResult]] = {}
    for result in results:
        if _metric_value(result, metric) is None:
            continue
        grouped.setdefault(result.model_name, []).append(result)
    return grouped


def _metric_value(
    result: ComparedClassificationResult, metric: MetricName
) -> float | None:
    if metric == "accuracy":
        return result.accuracy
    if metric == "balanced_accuracy":
        return result.balanced_accuracy
    return result.auroc
