import pytest
import torch

from medvfm_inspector.analysis import ModelComparisonInput, compare_models
from medvfm_inspector.types import (
    DatasetInfo,
    ExtractedRepresentations,
    LayerRepresentation,
    ModelInfo,
)


def _representations(
    *,
    model_name: str,
    split: str,
    labels: tuple[int, ...],
    offset: float = 0.0,
    dataset_name: str = "synthetic",
) -> ExtractedRepresentations:
    sample_ids = tuple(f"sample-{index}" for index in range(len(labels)))
    signal = torch.tensor(
        [
            [-2.0 + offset, 0.0],
            [-1.0 + offset, 0.0],
            [1.0 + offset, 0.0],
            [2.0 + offset, 0.0],
        ]
    )
    noise = torch.zeros(len(labels), 2) + offset
    return ExtractedRepresentations(
        model=ModelInfo(name=model_name, num_layers=2, hidden_size=2),
        dataset=DatasetInfo(
            name=dataset_name,
            task_type="binary-class",
            class_names=["negative", "positive"],
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


def _input(name: str, offset: float = 0.0) -> ModelComparisonInput:
    model_name = f"{name}-checkpoint"
    labels = (0, 0, 1, 1)
    return ModelComparisonInput(
        name=name,
        train=_representations(
            model_name=model_name, split="train", labels=labels, offset=offset
        ),
        evaluation=_representations(
            model_name=model_name, split="val", labels=labels, offset=offset
        ),
    )


def test_compare_models_runs_selected_analyses_for_multiple_models() -> None:
    result = compare_models(
        [_input("dinov2"), _input("rad-dino", offset=0.1)],
        analyses=("linear_probe", "knn", "cka", "pca"),
        knn_k=1,
        knn_metric="euclidean",
        pca_layer_indices=(1,),
    )

    assert result.dataset.name == "synthetic"
    assert result.train_split == "train"
    assert result.evaluation_split == "val"
    assert [item.model_name for item in result.linear_probe] == [
        "dinov2",
        "dinov2",
        "rad-dino",
        "rad-dino",
    ]
    assert result.linear_probe[1].layer_index == 1
    assert result.linear_probe[1].balanced_accuracy == 1.0
    assert result.knn[1].model_name == "dinov2"
    assert result.knn[1].k == 1
    assert result.knn[1].metric == "euclidean"
    assert len(result.cka) == 1
    assert result.cka[0].source_model_name == "dinov2"
    assert result.cka[0].target_model_name == "rad-dino"
    assert result.cka[0].result.matrix.shape == (2, 2)
    assert len(result.pca) == 2
    assert all(item.layer_index == 1 for item in result.pca)


def test_compare_models_can_run_only_requested_analysis() -> None:
    result = compare_models(
        [_input("dinov2"), _input("dinov3", offset=0.2)], analyses=("cka",)
    )

    assert result.linear_probe == ()
    assert result.knn == ()
    assert result.pca == ()
    assert len(result.cka) == 1


def test_compare_models_rejects_incompatible_datasets() -> None:
    first = _input("dinov2")
    second = ModelComparisonInput(
        name="dinov3",
        train=_representations(
            model_name="dinov3-checkpoint",
            split="train",
            labels=(0, 0, 1, 1),
            dataset_name="other",
        ),
        evaluation=_representations(
            model_name="dinov3-checkpoint",
            split="val",
            labels=(0, 0, 1, 1),
            dataset_name="other",
        ),
    )

    with pytest.raises(ValueError, match="compatible datasets"):
        compare_models([first, second], analyses=("cka",))


def test_compare_models_rejects_sample_misalignment() -> None:
    first = _input("dinov2")
    second = _input("dinov3")
    shifted_eval = ExtractedRepresentations(
        model=second.evaluation.model,
        dataset=second.evaluation.dataset,
        split=second.evaluation.split,
        sample_ids=("sample-1", "sample-0", "sample-2", "sample-3"),
        labels=second.evaluation.labels,
        layers=second.evaluation.layers,
    )

    with pytest.raises(ValueError, match="same sample IDs"):
        compare_models(
            [first, ModelComparisonInput("dinov3", second.train, shifted_eval)],
            analyses=("cka",),
        )
