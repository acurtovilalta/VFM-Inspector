import json
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import torch
from matplotlib.figure import Figure

from medvfm_inspector.analysis import ModelComparisonInput, compare_models
from medvfm_inspector.reporting import write_comparison_report
from medvfm_inspector.types import (
    DatasetInfo,
    ExtractedRepresentations,
    LayerRepresentation,
    ModelInfo,
)
from medvfm_inspector.visualization import (
    plot_cka_heatmap,
    plot_layer_metric,
    plot_pca_scatter,
)


def _representations(
    *,
    model_name: str,
    split: str,
    labels: tuple[int, ...],
    offset: float = 0.0,
) -> ExtractedRepresentations:
    sample_ids = tuple(f"sample-{index}" for index in range(len(labels)))
    signal = torch.tensor(
        [
            [-2.0 + offset, -1.0],
            [-1.0 + offset, 1.0],
            [1.0 + offset, -1.0],
            [2.0 + offset, 1.0],
        ]
    )
    noise = torch.tensor(
        [
            [0.0 + offset, 0.0],
            [0.0 + offset, 0.0],
            [0.0 + offset, 0.0],
            [0.0 + offset, 0.0],
        ]
    )
    return ExtractedRepresentations(
        model=ModelInfo(name=model_name, num_layers=2, hidden_size=2),
        dataset=DatasetInfo(
            name="synthetic-med",
            task_type="binary-class",
            class_names=["normal", "finding"],
            split_sizes={"train": len(labels), "val": len(labels), "test": 0},
        ),
        split=split,
        sample_ids=sample_ids,
        labels=labels,
        layers=(
            LayerRepresentation(layer_name="block_0", layer_index=0, cls=noise),
            LayerRepresentation(layer_name="block_1", layer_index=1, cls=signal),
        ),
    )


def _comparison():
    labels = (0, 0, 1, 1)
    inputs = (
        ModelComparisonInput(
            name="DINOv2-base",
            train=_representations(
                model_name="facebook/dinov2-base", split="train", labels=labels
            ),
            evaluation=_representations(
                model_name="facebook/dinov2-base", split="val", labels=labels
            ),
        ),
        ModelComparisonInput(
            name="RAD-DINO",
            train=_representations(
                model_name="microsoft/rad-dino",
                split="train",
                labels=labels,
                offset=0.1,
            ),
            evaluation=_representations(
                model_name="microsoft/rad-dino",
                split="val",
                labels=labels,
                offset=0.1,
            ),
        ),
    )
    return compare_models(
        inputs,
        analyses=("linear_probe", "knn", "cka", "pca"),
        knn_k=1,
        knn_metric="euclidean",
        pca_layer_indices=(1,),
    )


def test_plots_generate_png_files_without_gui(tmp_path: Path) -> None:
    comparison = _comparison()
    metric_path = plot_layer_metric(
        comparison.linear_probe, tmp_path / "linear.png", metric="balanced_accuracy"
    )
    heatmap_path = plot_cka_heatmap(comparison.cka[0], tmp_path / "cka.png")
    pca_path = plot_pca_scatter(
        comparison.pca[0],
        tmp_path / "pca.png",
        class_names=comparison.dataset.class_names,
    )

    assert matplotlib.get_backend().lower() == "agg"
    for path in (metric_path, heatmap_path, pca_path):
        assert path.exists()
        assert path.stat().st_size > 0
        image = mpimg.imread(path)
        assert image.shape[0] > 0
        assert image.shape[1] > 0


def test_cka_heatmap_uses_expected_matrix_dimensions(tmp_path: Path) -> None:
    comparison = _comparison()

    path = plot_cka_heatmap(comparison.cka[0], tmp_path / "cka.png")

    assert comparison.cka[0].result.matrix.shape == (2, 2)
    assert path.exists()


def test_pca_plot_uses_discrete_class_legend(tmp_path: Path, monkeypatch) -> None:
    captured: list[Figure] = []

    def fake_savefig(self, output_path, *args, **kwargs):
        captured.append(self)
        Path(output_path).write_bytes(b"png")

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    comparison = _comparison()

    plot_pca_scatter(
        comparison.pca[0],
        tmp_path / "pca.png",
        class_names=comparison.dataset.class_names,
    )

    figure = captured[0]
    assert len(figure.axes) == 1
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["normal", "finding"]


def test_cka_heatmap_uses_comparison_display_model_names(
    tmp_path: Path, monkeypatch
) -> None:
    captured: list[Figure] = []

    def fake_savefig(self, output_path, *args, **kwargs):
        captured.append(self)
        Path(output_path).write_bytes(b"png")

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    comparison = _comparison()

    plot_cka_heatmap(comparison.cka[0], tmp_path / "cka.png")

    axis = captured[0].axes[0]
    assert axis.get_xlabel() == "RAD-DINO"
    assert axis.get_ylabel() == "DINOv2-base"


def test_html_report_generation_writes_expected_structure_and_content(
    tmp_path: Path,
) -> None:
    comparison = _comparison()

    artifacts = write_comparison_report(
        comparison, tmp_path, experiment_name="demo-comparison"
    )

    assert artifacts.output_dir == tmp_path / "demo-comparison"
    assert artifacts.results_dir == artifacts.output_dir / "results"
    assert artifacts.figures_dir == artifacts.output_dir / "figures"
    assert artifacts.report_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.figure_paths
    assert all(path.exists() for path in artifacts.figure_paths)

    html = artifacts.report_path.read_text(encoding="utf-8")
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert "synthetic-med" in html
    assert "DINOv2-base" in html
    assert "RAD-DINO" in html
    assert "Linear Probing Results" in html
    assert "balanced accuracy" in html
    assert summary["dataset"] == "synthetic-med"
    assert summary["cka_pairs"][0]["shape"] == [2, 2]


def test_html_report_content_is_deterministic(tmp_path: Path) -> None:
    comparison = _comparison()

    first = write_comparison_report(comparison, tmp_path / "a", experiment_name="same")
    second = write_comparison_report(comparison, tmp_path / "b", experiment_name="same")

    assert first.report_path.read_text(
        encoding="utf-8"
    ) == second.report_path.read_text(encoding="utf-8")
    assert first.summary_path.read_text(
        encoding="utf-8"
    ) == second.summary_path.read_text(encoding="utf-8")
