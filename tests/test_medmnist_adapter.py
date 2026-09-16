import pytest
from PIL import Image

from medvfm_inspector.datasets import DatasetAdapter, MedMNISTAdapter


class IncompleteDatasetAdapter(DatasetAdapter):
    pass


def _fake_info() -> dict[str, object]:
    return {
        "task": "binary-class",
        "label": {"0": "normal", "1": "pneumonia"},
        "python_class": "PneumoniaMNIST",
    }


def _fake_splits() -> dict[str, list[tuple[Image.Image, list[int]]]]:
    return {
        "train": [(Image.new("RGB", (224, 224)), [0]) for _ in range(2)],
        "val": [(Image.new("RGB", (224, 224)), [1])],
        "test": [(Image.new("RGB", (224, 224)), [0]) for _ in range(3)],
    }


def test_dataset_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        IncompleteDatasetAdapter()


def test_medmnist_adapter_reports_split_metadata() -> None:
    adapter = MedMNISTAdapter(datasets=_fake_splits(), info=_fake_info())

    assert adapter.info.name == "pneumoniamnist-224"
    assert adapter.info.task_type == "binary-class"
    assert adapter.info.class_names == ["normal", "pneumonia"]
    assert adapter.info.split_sizes == {"train": 2, "val": 1, "test": 3}


def test_medmnist_split_returns_stable_samples() -> None:
    adapter = MedMNISTAdapter(datasets=_fake_splits(), info=_fake_info())
    train = adapter.get_split("train")

    first = train[0]
    second = train[1]

    assert len(train) == 2
    assert first.sample_id == "pneumoniamnist:train:0"
    assert second.sample_id == "pneumoniamnist:train:1"
    assert isinstance(first.image, Image.Image)
    assert first.image.size == (224, 224)
    assert first.label == 0
    assert first.metadata == {"split": "train", "index": 0}


def test_medmnist_adapter_rejects_unsupported_tasks() -> None:
    info = _fake_info()
    info["task"] = "multi-label"

    with pytest.raises(ValueError, match="Unsupported MedMNIST task"):
        MedMNISTAdapter(datasets=_fake_splits(), info=info)


def test_medmnist_adapter_rejects_multivalue_labels() -> None:
    splits = _fake_splits()
    splits["train"] = [(Image.new("RGB", (224, 224)), [0, 1])]
    adapter = MedMNISTAdapter(datasets=splits, info=_fake_info())

    with pytest.raises(ValueError, match="single-label"):
        adapter.get_split("train")[0]
