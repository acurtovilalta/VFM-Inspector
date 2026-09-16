"""MedMNIST dataset adapter."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, cast, overload

from PIL import Image

from medvfm_inspector.datasets.base import DatasetAdapter, SplitName
from medvfm_inspector.types import DatasetInfo, Sample

_SUPPORTED_SPLITS: tuple[SplitName, ...] = ("train", "val", "test")


class MedMNISTSplit(Sequence[Sample]):
    """Lazy sequence wrapper over one MedMNIST split."""

    def __init__(
        self, dataset: Sequence[tuple[Any, Any]], *, dataset_name: str, split: str
    ):
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._split = split

    def __len__(self) -> int:
        return len(self._dataset)

    @overload
    def __getitem__(self, index: int) -> Sample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Sample]: ...

    def __getitem__(self, index: int | slice) -> Sample | Sequence[Sample]:
        if isinstance(index, slice):
            return [self[item_index] for item_index in range(*index.indices(len(self)))]

        image, label = self._dataset[index]
        return Sample(
            sample_id=f"{self._dataset_name}:{self._split}:{index}",
            image=_as_pil_image(image),
            label=_single_label(label),
            metadata={"split": self._split, "index": index},
        )


class MedMNISTAdapter(DatasetAdapter):
    """Adapter for 2D single-label MedMNIST datasets."""

    def __init__(
        self,
        name: str = "pneumoniamnist",
        *,
        size: int = 224,
        as_rgb: bool = True,
        download: bool = False,
        root: str | None = None,
        datasets: dict[SplitName, Sequence[tuple[Any, Any]]] | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        self._name = name.lower()
        self._size = size
        self._as_rgb = as_rgb
        self._download = download
        self._root = root
        if self._root is not None:
            Path(self._root).mkdir(parents=True, exist_ok=True)
        self._raw_info = info if info is not None else self._load_info(self._name)
        self._validate_task(self._raw_info)
        self._splits = datasets if datasets is not None else self._load_splits()
        self._info = DatasetInfo(
            name=f"{self._name}-{self._size}",
            task_type=str(self._raw_info.get("task", "unknown")),
            class_names=_class_names(self._raw_info),
            split_sizes={
                split: len(self._splits[split]) for split in _SUPPORTED_SPLITS
            },
        )

    @property
    def info(self) -> DatasetInfo:
        """Return metadata for the MedMNIST dataset."""
        return self._info

    def get_split(self, split: SplitName) -> Sequence[Sample]:
        """Return a deterministic sequence for the requested official split."""
        if split not in _SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        return MedMNISTSplit(self._splits[split], dataset_name=self._name, split=split)

    @staticmethod
    def _load_info(name: str) -> dict[str, Any]:
        medmnist = import_module("medmnist")
        info = cast(dict[str, Any], medmnist.INFO)

        if name not in info:
            raise ValueError(f"Unsupported MedMNIST dataset: {name}")
        return cast(dict[str, Any], info[name])

    def _load_splits(self) -> dict[SplitName, Sequence[tuple[Any, Any]]]:
        medmnist = import_module("medmnist")

        class_name = self._raw_info.get("python_class")
        if not isinstance(class_name, str):
            raise ValueError(f"MedMNIST metadata for {self._name} has no python_class.")

        dataset_class = getattr(medmnist, class_name)
        splits: dict[SplitName, Sequence[tuple[Any, Any]]] = {}
        for split in _SUPPORTED_SPLITS:
            kwargs: dict[str, Any] = {
                "split": split,
                "download": self._download,
                "size": self._size,
                "as_rgb": self._as_rgb,
            }
            if self._root is not None:
                kwargs["root"] = self._root
            splits[split] = cast(Sequence[tuple[Any, Any]], dataset_class(**kwargs))
        return splits

    @staticmethod
    def _validate_task(info: dict[str, Any]) -> None:
        task = info.get("task")
        if task not in {"binary-class", "multi-class"}:
            raise ValueError(f"Unsupported MedMNIST task for v0.1: {task}")


def _as_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if hasattr(image, "convert"):
        converted = image.convert("RGB")
        if isinstance(converted, Image.Image):
            return converted
    raise TypeError("MedMNIST samples must provide PIL-compatible images.")


def _single_label(label: Any) -> int:
    if isinstance(label, int):
        return label
    if hasattr(label, "tolist"):
        label = label.tolist()
    while isinstance(label, list):
        if len(label) != 1:
            raise ValueError("Only single-label MedMNIST targets are supported.")
        label = label[0]
    return int(label)


def _class_names(info: dict[str, Any]) -> list[str]:
    labels = info.get("label")
    if not isinstance(labels, dict):
        return []
    return [str(labels[key]) for key in sorted(labels, key=lambda value: int(value))]
