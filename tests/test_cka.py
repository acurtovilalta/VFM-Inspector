import pytest
import torch

from medvfm_inspector.analysis import linear_cka, within_model_cka
from medvfm_inspector.types import (
    DatasetInfo,
    ExtractedRepresentations,
    LayerRepresentation,
    ModelInfo,
)


def _representations(
    cls_layers: tuple[torch.Tensor, ...],
    *,
    sample_ids: tuple[str, ...] = ("a", "b", "c", "d"),
    model_name: str = "synthetic",
) -> ExtractedRepresentations:
    return ExtractedRepresentations(
        model=ModelInfo(name=model_name, num_layers=len(cls_layers)),
        dataset=DatasetInfo(
            name="synthetic",
            task_type="binary-class",
            class_names=["negative", "positive"],
            split_sizes={"train": len(sample_ids), "val": 0, "test": 0},
        ),
        split="train",
        sample_ids=sample_ids,
        labels=tuple(index % 2 for index in range(len(sample_ids))),
        layers=tuple(
            LayerRepresentation(
                layer_name=f"block_{index}", layer_index=index, cls=features
            )
            for index, features in enumerate(cls_layers)
        ),
    )


def test_linear_cka_identical_representations_are_near_one() -> None:
    features = torch.tensor([[-1.0, 0.0], [-0.5, 1.0], [0.5, -1.0], [1.0, 0.0]])
    result = linear_cka(
        _representations((features,), model_name="a"),
        _representations((features.clone(),), model_name="b"),
    )

    assert result.matrix.shape == (1, 1)
    assert result.matrix[0, 0] == pytest.approx(1.0)
    assert result.source_layers[0].layer_index == 0
    assert result.target_layers[0].layer_name == "block_0"


def test_linear_cka_unrelated_synthetic_representations_are_lower() -> None:
    source = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])
    target = torch.tensor([[1.0], [-1.0], [-1.0], [1.0]])

    result = linear_cka(_representations((source,)), _representations((target,)))

    assert result.matrix[0, 0] == pytest.approx(0.0)


def test_linear_cka_supports_different_hidden_dimensions() -> None:
    source = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])
    target = torch.tensor(
        [[-1.0, 0.0, 1.0], [-0.5, 1.0, 0.0], [0.5, 0.0, -1.0], [1.0, -1.0, 0.0]]
    )

    result = linear_cka(_representations((source,)), _representations((target,)))

    assert result.matrix.shape == (1, 1)
    assert 0.0 <= result.matrix[0, 0] <= 1.0


def test_linear_cka_rejects_mismatched_sample_ids() -> None:
    features = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])

    with pytest.raises(ValueError, match="identical sample IDs"):
        linear_cka(
            _representations((features,), sample_ids=("a", "b", "c", "d")),
            _representations((features,), sample_ids=("a", "b", "c", "x")),
        )


def test_linear_cka_rejects_same_samples_in_different_order() -> None:
    features = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])

    with pytest.raises(ValueError, match="same order"):
        linear_cka(
            _representations((features,), sample_ids=("a", "b", "c", "d")),
            _representations((features,), sample_ids=("b", "a", "c", "d")),
        )


def test_within_model_cka_uses_same_implementation() -> None:
    first = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])
    second = torch.tensor([[1.0], [-1.0], [-1.0], [1.0]])

    result = within_model_cka(_representations((first, second)))

    assert result.matrix.shape == (2, 2)
    assert result.matrix[0, 0] == pytest.approx(1.0)
    assert result.matrix[1, 1] == pytest.approx(1.0)
