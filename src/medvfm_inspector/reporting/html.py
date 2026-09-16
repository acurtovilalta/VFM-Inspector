"""Deterministic HTML reports for model comparison results."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medvfm_inspector.analysis.comparison import (
    ComparedClassificationResult,
    ModelComparisonResult,
)
from medvfm_inspector.visualization import MetricName
from medvfm_inspector.visualization.plots import (
    plot_cka_heatmap,
    plot_layer_metric,
    plot_pca_scatter,
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Filesystem outputs produced for one comparison report."""

    experiment_name: str
    output_dir: Path
    results_dir: Path
    figures_dir: Path
    report_path: Path
    summary_path: Path
    figure_paths: tuple[Path, ...]


def write_comparison_report(
    comparison: ModelComparisonResult,
    output_dir: str | Path,
    *,
    experiment_name: str = "comparison",
    metric: MetricName = "balanced_accuracy",
) -> ReportArtifacts:
    """Write figures, a JSON summary, and a locally portable HTML report."""
    root = Path(output_dir) / experiment_name
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = _write_figures(comparison, figures_dir, metric=metric)
    summary = _summary_payload(comparison, figure_paths, root, metric=metric)
    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report_path = root / "report.html"
    report_path.write_text(_html_document(comparison, summary), encoding="utf-8")
    return ReportArtifacts(
        experiment_name=experiment_name,
        output_dir=root,
        results_dir=results_dir,
        figures_dir=figures_dir,
        report_path=report_path,
        summary_path=summary_path,
        figure_paths=tuple(figure_paths),
    )


def _write_figures(
    comparison: ModelComparisonResult,
    figures_dir: Path,
    *,
    metric: MetricName,
) -> list[Path]:
    figure_paths: list[Path] = []
    if comparison.linear_probe:
        figure_paths.append(
            plot_layer_metric(
                comparison.linear_probe,
                figures_dir / f"linear_probe_{metric}.png",
                metric=metric,
                title="Linear probe",
            )
        )
    if comparison.knn:
        figure_paths.append(
            plot_layer_metric(
                comparison.knn,
                figures_dir / f"knn_{metric}.png",
                metric=metric,
                title="k-NN",
            )
        )
    for index, cka in enumerate(comparison.cka):
        filename = (
            f"cka_{index:03d}_{_slug(cka.source_model_name)}_"
            f"vs_{_slug(cka.target_model_name)}.png"
        )
        figure_paths.append(plot_cka_heatmap(cka, figures_dir / filename))
    for index, pca in enumerate(comparison.pca):
        filename = (
            f"pca_{index:03d}_{_slug(pca.model_name)}_layer_{pca.layer_index}.png"
        )
        figure_paths.append(
            plot_pca_scatter(
                pca,
                figures_dir / filename,
                class_names=comparison.dataset.class_names,
            )
        )
    return figure_paths


def _summary_payload(
    comparison: ModelComparisonResult,
    figure_paths: list[Path],
    root: Path,
    *,
    metric: MetricName,
) -> dict[str, Any]:
    return {
        "experiment": root.name,
        "dataset": comparison.dataset.name,
        "train_split": comparison.train_split,
        "evaluation_split": comparison.evaluation_split,
        "metric": metric,
        "models": [
            {
                "name": model_input.name,
                "checkpoint": model_input.evaluation.model.name,
                "num_layers": model_input.evaluation.model.num_layers,
                "hidden_size": model_input.evaluation.model.hidden_size,
                "train_samples": len(model_input.train.sample_ids),
                "evaluation_samples": len(model_input.evaluation.sample_ids),
            }
            for model_input in comparison.models
        ],
        "linear_probe_best": [
            _best_result_payload(result, metric=metric)
            for result in _best_by_model(comparison.linear_probe, metric=metric)
        ],
        "knn_best": [
            _best_result_payload(result, metric=metric)
            for result in _best_by_model(comparison.knn, metric=metric)
        ],
        "cka_pairs": [
            {
                "source_model": cka.source_model_name,
                "target_model": cka.target_model_name,
                "shape": list(cka.result.matrix.shape),
            }
            for cka in comparison.cka
        ],
        "pca": [
            {
                "model": pca.model_name,
                "layer_index": pca.layer_index,
                "explained_variance_ratio": list(pca.explained_variance_ratio),
            }
            for pca in comparison.pca
        ],
        "figures": [path.relative_to(root).as_posix() for path in sorted(figure_paths)],
    }


def _html_document(comparison: ModelComparisonResult, summary: dict[str, Any]) -> str:
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(summary['experiment'])}</title>",
        "<style>",
        _css(),
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{_escape(summary['experiment'])}</h1>",
        "<section>",
        "<h2>Experiment Overview</h2>",
        "<dl>",
        f"<dt>Dataset</dt><dd>{_escape(comparison.dataset.name)}</dd>",
        f"<dt>Train split</dt><dd>{_escape(comparison.train_split)}</dd>",
        f"<dt>Evaluation split</dt><dd>{_escape(comparison.evaluation_split)}</dd>",
        "</dl>",
        "</section>",
        _model_table(comparison),
        _classification_section("Linear Probing Results", summary["linear_probe_best"]),
        _classification_section("k-NN Results", summary["knn_best"]),
        _cka_section(summary["cka_pairs"]),
        _pca_section(summary["pca"]),
        _figure_section(summary["figures"]),
        _factual_summary(summary),
        "</body>",
        "</html>",
    ]
    return "\n".join(lines) + "\n"


def _model_table(comparison: ModelComparisonResult) -> str:
    rows = []
    for model_input in comparison.models:
        model = model_input.evaluation.model
        rows.append(
            "<tr>"
            f"<td>{_escape(model_input.name)}</td>"
            f"<td>{_escape(model.name)}</td>"
            f"<td>{model.num_layers}</td>"
            f"<td>{_escape(str(model.hidden_size))}</td>"
            f"<td>{len(model_input.train.sample_ids)}</td>"
            f"<td>{len(model_input.evaluation.sample_ids)}</td>"
            "</tr>"
        )
    return (
        "<section>\n<h2>Model Metadata</h2>\n<table>"
        "<thead><tr><th>Model</th><th>Checkpoint</th><th>Layers</th>"
        "<th>Hidden size</th><th>Train samples</th><th>Eval samples</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>\n</section>"
    )


def _classification_section(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"<section>\n<h2>{_escape(title)}</h2>\n<p>No results.</p>\n</section>"
    body = "".join(
        "<tr>"
        f"<td>{_escape(row['model'])}</td>"
        f"<td>{row['layer_index']}</td>"
        f"<td>{_format_metric(row['accuracy'])}</td>"
        f"<td>{_format_metric(row['balanced_accuracy'])}</td>"
        f"<td>{_format_metric(row['auroc'])}</td>"
        "</tr>"
        for row in rows
    )
    return (
        f"<section>\n<h2>{_escape(title)}</h2>\n<table>"
        "<thead><tr><th>Model</th><th>Best layer</th><th>Accuracy</th>"
        "<th>Balanced accuracy</th><th>AUROC</th></tr></thead>"
        f"<tbody>{body}</tbody></table>\n</section>"
    )


def _cka_section(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<section>\n<h2>CKA Comparisons</h2>\n<p>No results.</p>\n</section>"
    items = "".join(
        "<li>"
        f"{_escape(row['source_model'])} vs {_escape(row['target_model'])}: "
        f"{row['shape'][0]} x {row['shape'][1]}"
        "</li>"
        for row in rows
    )
    return f"<section>\n<h2>CKA Comparisons</h2>\n<ul>{items}</ul>\n</section>"


def _pca_section(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<section>\n<h2>PCA</h2>\n<p>No results.</p>\n</section>"
    items = "".join(_pca_item(row) for row in rows)
    return f"<section>\n<h2>PCA</h2>\n<ul>{items}</ul>\n</section>"


def _pca_item(row: dict[str, Any]) -> str:
    variance = ", ".join(
        _format_metric(value) for value in row["explained_variance_ratio"]
    )
    return f"<li>{_escape(row['model'])} layer {row['layer_index']}: {variance}</li>"


def _figure_section(figures: list[str]) -> str:
    if not figures:
        return "<section>\n<h2>Figures</h2>\n<p>No figures.</p>\n</section>"
    images = "".join(
        "<figure>"
        f'<img src="{_escape(path)}" alt="{_escape(path)}">'
        f"<figcaption>{_escape(path)}</figcaption>"
        "</figure>"
        for path in figures
    )
    return f"<section>\n<h2>Figures</h2>\n{images}\n</section>"


def _factual_summary(summary: dict[str, Any]) -> str:
    sentences: list[str] = []
    linear_best = summary["linear_probe_best"]
    if linear_best:
        best = max(linear_best, key=lambda row: row[summary["metric"]] or -1.0)
        sentences.append(
            f"{best['model']} achieved the highest linear-probe "
            f"{summary['metric'].replace('_', ' ')} of "
            f"{_format_metric(best[summary['metric']])} at layer {best['layer_index']}."
        )
    knn_best = summary["knn_best"]
    if knn_best:
        best = max(knn_best, key=lambda row: row[summary["metric"]] or -1.0)
        sentences.append(
            f"{best['model']} achieved the highest k-NN "
            f"{summary['metric'].replace('_', ' ')} of "
            f"{_format_metric(best[summary['metric']])} at layer {best['layer_index']}."
        )
    if summary["cka_pairs"]:
        sentences.append(
            f"{len(summary['cka_pairs'])} cross-model CKA comparison(s) were computed."
        )
    if summary["pca"]:
        sentences.append(f"{len(summary['pca'])} PCA projection(s) were generated.")
    text = " ".join(sentences) if sentences else "No analysis results were provided."
    return f"<section>\n<h2>Factual Summary</h2>\n<p>{_escape(text)}</p>\n</section>"


def _best_by_model(
    results: tuple[ComparedClassificationResult, ...], *, metric: MetricName
) -> list[ComparedClassificationResult]:
    by_model: dict[str, ComparedClassificationResult] = {}
    for result in sorted(results, key=lambda item: (item.model_name, item.layer_index)):
        value = _metric_value(result, metric)
        if value is None:
            continue
        current = by_model.get(result.model_name)
        if current is None or value > (_metric_value(current, metric) or -1.0):
            by_model[result.model_name] = result
    return [by_model[model_name] for model_name in sorted(by_model)]


def _best_result_payload(
    result: ComparedClassificationResult, *, metric: MetricName
) -> dict[str, Any]:
    return {
        "model": result.model_name,
        "checkpoint": result.model.name,
        "layer_index": result.layer_index,
        "layer_name": result.layer_name,
        "accuracy": result.accuracy,
        "balanced_accuracy": result.balanced_accuracy,
        "auroc": result.auroc,
        "selection_metric": metric,
    }


def _metric_value(
    result: ComparedClassificationResult, metric: MetricName
) -> float | None:
    if metric == "accuracy":
        return result.accuracy
    if metric == "balanced_accuracy":
        return result.balanced_accuracy
    return result.auroc


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug or "item"


def _css() -> str:
    return """
body {
  font-family: system-ui, sans-serif;
  line-height: 1.45;
  margin: 2rem;
  color: #202124;
}
section { margin: 2rem 0; }
table { border-collapse: collapse; width: 100%; max-width: 960px; }
th, td { border: 1px solid #d0d7de; padding: 0.45rem 0.6rem; text-align: left; }
th { background: #f6f8fa; }
figure { margin: 1rem 0; }
img { max-width: 760px; width: 100%; height: auto; border: 1px solid #d0d7de; }
figcaption { color: #57606a; font-size: 0.9rem; margin-top: 0.25rem; }
""".strip()
