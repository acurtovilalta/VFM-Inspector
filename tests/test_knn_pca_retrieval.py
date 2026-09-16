import pytest
import torch

from medvfm_inspector.analysis import nearest_neighbors, run_knn_classification, run_pca
from medvfm_inspector.types import (
    DatasetInfo,
    ExtractedRepresentations,
    LayerRepresentation,
    ModelInfo,
)


def _representations(
    labels: tuple[int, ...], layer0: torch.Tensor, layer1: torch.Tensor
) -> ExtractedRepresentations:
    return ExtractedRepresentations(
        model=ModelInfo(name="synthetic", num_layers=2, hidden_size=2),
        dataset=DatasetInfo(
            name="synthetic",
            task_type="binary-class",
            class_names=["negative", "positive"],
            split_sizes={"train": len(labels), "val": len(labels), "test": 0},
        ),
        split="train",
        sample_ids=tuple(f"sample-{index}" for index in range(len(labels))),
        labels=labels,
        layers=(
            LayerRepresentation(layer_name="block_0", layer_index=0, cls=layer0),
            LayerRepresentation(layer_name="block_1", layer_index=1, cls=layer1),
        ),
    )


def test_knn_evaluates_layers_with_cls_representations() -> None:
    train = _representations(
        (0, 0, 1, 1),
        layer0=torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        layer1=torch.tensor([[-2.0, 0.0], [-1.5, 0.0], [1.5, 0.0], [2.0, 0.0]]),
    )
    evaluation = _representations(
        (0, 1),
        layer0=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        layer1=torch.tensor([[-1.7, 0.0], [1.7, 0.0]]),
    )

    results = run_knn_classification(
        train=train, evaluation=evaluation, k=1, metric="euclidean"
    )

    assert [result.layer_index for result in results] == [0, 1]
    assert results[1].accuracy == 1.0
    assert results[1].balanced_accuracy == 1.0
    assert results[1].k == 1
    assert results[1].metric == "euclidean"


def test_pca_returns_coordinates_and_layer_metadata() -> None:
    representations = _representations(
        (0, 0, 1, 1),
        layer0=torch.tensor([[-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        layer1=torch.tensor([[-1.0, -1.0], [-0.5, -0.5], [0.5, 0.5], [1.0, 1.0]]),
    )

    results = run_pca(representations, n_components=2)

    assert len(results) == 2
    assert results[0].model == representations.model
    assert results[0].sample_ids == representations.sample_ids
    assert results[0].labels == representations.labels
    assert results[0].coordinates.shape == (4, 2)
    assert len(results[0].explained_variance_ratio) == 2


def test_nearest_neighbors_uses_cosine_similarity_and_excludes_query() -> None:
    representations = _representations(
        (0, 0, 1, 1),
        layer0=torch.zeros(4, 2),
        layer1=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]),
    )

    neighbors = nearest_neighbors(
        representations, query="sample-0", layer_index=1, top_k=2
    )

    assert [neighbor.sample_id for neighbor in neighbors] == ["sample-1", "sample-2"]
    assert [neighbor.rank for neighbor in neighbors] == [1, 2]
    assert neighbors[0].label == 0
    assert neighbors[0].score > neighbors[1].score


def test_nearest_neighbors_rejects_unknown_query_id() -> None:
    representations = _representations(
        (0, 1),
        layer0=torch.zeros(2, 2),
        layer1=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    with pytest.raises(ValueError, match="Unknown query sample ID"):
        nearest_neighbors(representations, query="missing", layer_index=1)
