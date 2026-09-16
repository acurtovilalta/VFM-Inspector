import torch

from medvfm_inspector.analysis import run_linear_probe
from medvfm_inspector.types import (
    DatasetInfo,
    ExtractedRepresentations,
    LayerRepresentation,
    ModelInfo,
)


def _representations(
    labels: tuple[int, ...], layer0: torch.Tensor, layer1: torch.Tensor
) -> ExtractedRepresentations:
    sample_ids = tuple(f"sample-{index}" for index in range(len(labels)))
    return ExtractedRepresentations(
        model=ModelInfo(name="synthetic", num_layers=2, hidden_size=2),
        dataset=DatasetInfo(
            name="synthetic",
            task_type="binary-class",
            class_names=["negative", "positive"],
            split_sizes={"train": len(labels), "val": len(labels), "test": 0},
        ),
        split="train",
        sample_ids=sample_ids,
        labels=labels,
        layers=(
            LayerRepresentation(layer_name="block_0", layer_index=0, cls=layer0),
            LayerRepresentation(layer_name="block_1", layer_index=1, cls=layer1),
        ),
    )


def test_linear_probe_evaluates_each_layer_and_identifies_predictive_layer() -> None:
    train_labels = (0, 0, 0, 1, 1, 1)
    eval_labels = (0, 0, 1, 1)
    train = _representations(
        train_labels,
        layer0=torch.zeros(6, 2),
        layer1=torch.tensor(
            [[-3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        ),
    )
    evaluation = _representations(
        eval_labels,
        layer0=torch.zeros(4, 2),
        layer1=torch.tensor([[-2.5, 0.0], [-1.5, 0.0], [1.5, 0.0], [2.5, 0.0]]),
    )

    results = run_linear_probe(train=train, evaluation=evaluation, random_seed=13)

    assert [result.layer_index for result in results] == [0, 1]
    assert results[0].train_sample_count == 6
    assert results[0].evaluation_sample_count == 4
    assert results[1].accuracy == 1.0
    assert results[1].balanced_accuracy == 1.0
    assert results[1].auroc == 1.0
    assert results[1].accuracy > results[0].accuracy


def test_linear_probe_supports_multiclass_auroc() -> None:
    labels = (0, 0, 1, 1, 2, 2)
    features = torch.tensor(
        [
            [-3.0, 0.0],
            [-2.5, 0.0],
            [0.0, 3.0],
            [0.0, 2.5],
            [3.0, 0.0],
            [2.5, 0.0],
        ]
    )
    train = _representations(labels, layer0=torch.zeros(6, 2), layer1=features)
    evaluation = _representations(labels, layer0=torch.zeros(6, 2), layer1=features)

    result = run_linear_probe(train=train, evaluation=evaluation)[1]

    assert result.accuracy == 1.0
    assert result.balanced_accuracy == 1.0
    assert result.auroc is not None
    assert result.auroc > 0.9


def test_linear_probe_raises_when_cls_representations_are_missing() -> None:
    train = _representations(
        (0, 1),
        layer0=torch.zeros(2, 2),
        layer1=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
    )
    train = ExtractedRepresentations(
        model=train.model,
        dataset=train.dataset,
        split=train.split,
        sample_ids=train.sample_ids,
        labels=train.labels,
        layers=(LayerRepresentation(layer_name="block_0", layer_index=0),),
    )

    try:
        run_linear_probe(train=train, evaluation=train)
    except ValueError as exc:
        assert "no CLS" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing CLS representations.")
