import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest
import torch
from PIL import Image

from medvfm_inspector.datasets import DatasetAdapter, SplitName
from medvfm_inspector.types import (
    DatasetInfo,
    ExtractedRepresentations,
    LayerRepresentation,
    ModelInfo,
    Sample,
)


def _demo_module() -> ModuleType:
    path = Path("examples/pneumoniamnist_comparison.py")
    spec = importlib.util.spec_from_file_location("pneumoniamnist_comparison", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load demo module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TinyDataset(DatasetAdapter):
    def __init__(self) -> None:
        self._splits = {
            split: tuple(
                Sample(
                    sample_id=f"{split}-{index}",
                    image=Image.new("RGB", (4, 4)),
                    label=index % 2,
                    metadata={"split": split},
                )
                for index in range(4)
            )
            for split in ("train", "val", "test")
        }
        self._info = DatasetInfo(
            name="tiny",
            task_type="binary-class",
            class_names=["normal", "finding"],
            split_sizes={"train": 4, "val": 4, "test": 4},
        )

    @property
    def info(self) -> DatasetInfo:
        return self._info

    def get_split(self, split: SplitName) -> Sequence[Sample]:
        return self._splits[split]


def _result(module: ModuleType, model_name: str, layer_index: int, score: float):
    return module.ComparedClassificationResult(
        model_name=model_name,
        model=module.ModelComparisonInput(
            name=model_name,
            train=_empty_reps(),
            evaluation=_empty_reps(),
        ).evaluation.model,
        layer_index=layer_index,
        layer_name=f"block_{layer_index}",
        train_sample_count=4,
        evaluation_sample_count=4,
        accuracy=score,
        balanced_accuracy=score,
        auroc=score,
        analysis="linear_probe",
    )


def _empty_reps() -> ExtractedRepresentations:
    return ExtractedRepresentations(
        model=ModelInfo(name="tiny-model", num_layers=2, hidden_size=2),
        dataset=DatasetInfo(
            name="tiny",
            task_type="binary-class",
            class_names=["normal", "finding"],
            split_sizes={"train": 0, "val": 0, "test": 0},
        ),
        split="val",
        sample_ids=(),
        labels=(),
        layers=(
            LayerRepresentation(
                layer_name="block_0", layer_index=0, cls=torch.zeros(0, 2)
            ),
        ),
    )


def test_sample_limited_dataset_preserves_first_samples_per_split() -> None:
    module = _demo_module()
    dataset = module.SampleLimitedDataset(TinyDataset(), sample_limit=2)

    assert dataset.info.name == "tiny-first-2"
    assert dataset.info.split_sizes == {"train": 2, "val": 2, "test": 2}
    assert [sample.sample_id for sample in dataset.get_split("train")] == [
        "train-0",
        "train-1",
    ]
    assert [sample.sample_id for sample in dataset.get_split("test")] == [
        "test-0",
        "test-1",
    ]


def test_best_layer_indices_use_validation_probe_and_knn_winners() -> None:
    module = _demo_module()
    linear = (
        _result(module, "model-a", 0, 0.5),
        _result(module, "model-a", 1, 0.9),
        _result(module, "model-b", 0, 0.8),
    )
    knn = (
        _result(module, "model-a", 0, 0.7),
        _result(module, "model-b", 1, 0.6),
    )

    assert module.best_layer_indices(linear, knn) == (0, 1)


def test_demo_help_documents_quick_mode_and_dinov3_access(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _demo_module()

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--sample-limit" in help_text
    assert "--no-sample-limit" in help_text
    assert "Quick mode" in help_text
    assert "DINOv3" in help_text
    assert "huggingface-cli" in help_text
    assert "login" in help_text
