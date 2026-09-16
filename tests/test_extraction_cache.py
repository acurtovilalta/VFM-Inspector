from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from medvfm_inspector.datasets import DatasetAdapter
from medvfm_inspector.extraction import (
    CacheMetadataError,
    ExtractionConfig,
    cache_status,
    extract_representations,
    load_representations,
    save_representations,
)
from medvfm_inspector.models import ModelAdapter
from medvfm_inspector.types import (
    DatasetInfo,
    LayerRepresentation,
    ModelInfo,
    ModelRepresentations,
    Sample,
)


class TinyDataset(DatasetAdapter):
    def __init__(self, sample_count: int = 5) -> None:
        self.samples = tuple(
            Sample(
                sample_id=f"sample-{index}",
                image=Image.new("RGB", (4, 4), color=(index, 0, 0)),
                label=index % 2,
                metadata={},
            )
            for index in range(sample_count)
        )
        self._info = DatasetInfo(
            name="tiny",
            task_type="binary-class",
            class_names=["zero", "one"],
            split_sizes={"train": sample_count, "val": 0, "test": 0},
        )

    @property
    def info(self) -> DatasetInfo:
        return self._info

    def get_split(self, split: str) -> Sequence[Sample]:
        assert split == "train"
        return self.samples


class TinyAdapter(ModelAdapter):
    def __init__(self, *, with_register_tokens: bool = False) -> None:
        self.with_register_tokens = with_register_tokens
        self.extract_calls = 0
        self.batch_sizes: list[int] = []
        self._info = ModelInfo(
            name="tiny-model",
            architecture="tiny",
            num_layers=2,
            hidden_size=3,
            num_register_tokens=1 if with_register_tokens else 0,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def layer_names(self) -> tuple[str, ...]:
        return ("block_0", "block_1")

    def preprocess(self, images: Sequence[Any]) -> dict[str, torch.Tensor]:
        values = [image.getpixel((0, 0))[0] for image in images]
        return {"values": torch.tensor(values, dtype=torch.float32)}

    def extract_from_inputs(
        self, model_inputs: Mapping[str, torch.Tensor]
    ) -> ModelRepresentations:
        values = model_inputs["values"]
        self.extract_calls += 1
        self.batch_sizes.append(int(values.shape[0]))
        layers: list[LayerRepresentation] = []
        for layer_index in range(2):
            base = values[:, None] + (layer_index * 100)
            cls = torch.cat([base, base + 1, base + 2], dim=1)
            patch_tokens = torch.stack([cls + 10, cls + 20], dim=1)
            register_tokens = None
            if self.with_register_tokens:
                register_tokens = (cls + 30)[:, None, :]
            layers.append(
                LayerRepresentation(
                    layer_name=f"block_{layer_index}",
                    layer_index=layer_index,
                    cls=cls,
                    patch_tokens=patch_tokens,
                    register_tokens=register_tokens,
                )
            )
        return ModelRepresentations(model=self.info, layers=tuple(layers))


def test_default_extraction_is_cls_only_and_preserves_order(tmp_path: Path) -> None:
    dataset = TinyDataset(sample_count=5)
    adapter = TinyAdapter()

    extracted = extract_representations(
        model=adapter,
        dataset=dataset,
        split="train",
        batch_size=2,
        cache_dir=tmp_path,
    )

    assert adapter.extract_calls == 3
    assert adapter.batch_sizes == [2, 2, 1]
    assert extracted.sample_ids == tuple(f"sample-{index}" for index in range(5))
    assert extracted.labels == (0, 1, 0, 1, 0)
    assert len(extracted.layers) == 2
    assert extracted.layers[0].cls is not None
    assert extracted.layers[0].cls.shape == (5, 3)
    assert extracted.layers[0].patch_tokens is None
    assert extracted.layers[0].register_tokens is None
    assert not (tmp_path / "layers" / "layer_000_patch_tokens.npy").exists()
    torch.testing.assert_close(
        extracted.layers[1].cls[0], torch.tensor([100, 101, 102], dtype=torch.float32)
    )


def test_extraction_can_include_cls_and_patch_tokens() -> None:
    extracted = extract_representations(
        model=TinyAdapter(),
        dataset=TinyDataset(sample_count=3),
        split="train",
        batch_size=2,
        representation_types=("cls", "patch_tokens"),
    )

    first_layer = extracted.layers[0]
    assert first_layer.cls is not None
    assert first_layer.patch_tokens is not None
    assert first_layer.cls.shape == (3, 3)
    assert first_layer.patch_tokens.shape == (3, 2, 3)
    torch.testing.assert_close(
        first_layer.patch_tokens[0, 0], torch.tensor([10, 11, 12], dtype=torch.float32)
    )


def test_extraction_can_select_layer_indices() -> None:
    extracted = extract_representations(
        model=TinyAdapter(),
        dataset=TinyDataset(sample_count=2),
        split="train",
        layer_indices=(1,),
    )

    assert len(extracted.layers) == 1
    assert extracted.layers[0].layer_index == 1
    assert extracted.layers[0].layer_name == "block_1"
    assert extracted.layers[0].cls is not None
    torch.testing.assert_close(
        extracted.layers[0].cls[0], torch.tensor([100, 101, 102], dtype=torch.float32)
    )


def test_extraction_can_select_dinov3_register_tokens() -> None:
    extracted = extract_representations(
        model=TinyAdapter(with_register_tokens=True),
        dataset=TinyDataset(sample_count=3),
        split="train",
        batch_size=2,
        representation_types=("register_tokens",),
    )

    first_layer = extracted.layers[0]
    assert first_layer.cls is None
    assert first_layer.patch_tokens is None
    assert first_layer.register_tokens is not None
    assert first_layer.register_tokens.shape == (3, 1, 3)
    torch.testing.assert_close(
        first_layer.register_tokens[0, 0],
        torch.tensor([30, 31, 32], dtype=torch.float32),
    )


def test_cache_save_load_round_trip(tmp_path: Path) -> None:
    extraction = ExtractionConfig(
        batch_size=2,
        representation_types=("cls", "patch_tokens", "register_tokens"),
    )
    extracted = extract_representations(
        model=TinyAdapter(with_register_tokens=True),
        dataset=TinyDataset(sample_count=3),
        split="train",
        batch_size=2,
        representation_types=("cls", "patch_tokens", "register_tokens"),
    )

    metadata = save_representations(extracted, tmp_path, extraction=extraction)
    loaded = load_representations(tmp_path, expected=metadata)

    assert loaded.sample_ids == extracted.sample_ids
    assert loaded.labels == extracted.labels
    assert loaded.model == extracted.model
    assert loaded.dataset == extracted.dataset
    assert len(loaded.layers) == len(extracted.layers)
    for loaded_layer, extracted_layer in zip(
        loaded.layers, extracted.layers, strict=True
    ):
        assert loaded_layer.cls is not None
        assert extracted_layer.cls is not None
        torch.testing.assert_close(loaded_layer.cls, extracted_layer.cls)
        assert loaded_layer.patch_tokens is not None
        assert extracted_layer.patch_tokens is not None
        torch.testing.assert_close(
            loaded_layer.patch_tokens, extracted_layer.patch_tokens
        )
        assert loaded_layer.register_tokens is not None
        assert extracted_layer.register_tokens is not None
        torch.testing.assert_close(
            loaded_layer.register_tokens, extracted_layer.register_tokens
        )


def test_extraction_reuses_valid_cache(tmp_path: Path) -> None:
    dataset = TinyDataset(sample_count=4)
    first_adapter = TinyAdapter()
    first = extract_representations(
        model=first_adapter,
        dataset=dataset,
        split="train",
        batch_size=2,
        cache_dir=tmp_path,
    )
    second_adapter = TinyAdapter()

    second = extract_representations(
        model=second_adapter,
        dataset=dataset,
        split="train",
        batch_size=2,
        cache_dir=tmp_path,
        reuse_cache=True,
    )

    assert first_adapter.extract_calls == 2
    assert second_adapter.extract_calls == 0
    assert first.layers[0].cls is not None
    assert second.layers[0].cls is not None
    torch.testing.assert_close(second.layers[0].cls, first.layers[0].cls)


def test_incompatible_cache_metadata_is_detected(tmp_path: Path) -> None:
    extracted = extract_representations(
        model=TinyAdapter(), dataset=TinyDataset(sample_count=2), split="train"
    )
    metadata = save_representations(
        extracted, tmp_path, extraction=ExtractionConfig(batch_size=32)
    )
    incompatible = type(metadata)(
        schema_version=metadata.schema_version,
        model=ModelInfo(name="other", num_layers=2),
        dataset=metadata.dataset,
        split=metadata.split,
        num_samples=metadata.num_samples,
        extraction=metadata.extraction,
    )

    status = cache_status(tmp_path, incompatible)

    assert status.valid is False
    with pytest.raises(CacheMetadataError):
        load_representations(tmp_path, expected=incompatible)


def test_overwriting_cache_removes_unselected_tensor_files(tmp_path: Path) -> None:
    extracted_with_patches = extract_representations(
        model=TinyAdapter(),
        dataset=TinyDataset(sample_count=2),
        split="train",
        representation_types=("cls", "patch_tokens"),
    )
    save_representations(
        extracted_with_patches,
        tmp_path,
        extraction=ExtractionConfig(
            batch_size=32, representation_types=("cls", "patch_tokens")
        ),
    )
    assert (tmp_path / "layers" / "layer_000_patch_tokens.npy").exists()

    extracted_cls_only = extract_representations(
        model=TinyAdapter(), dataset=TinyDataset(sample_count=2), split="train"
    )
    save_representations(
        extracted_cls_only, tmp_path, extraction=ExtractionConfig(batch_size=32)
    )

    assert (tmp_path / "layers" / "layer_000_cls.npy").exists()
    assert not (tmp_path / "layers" / "layer_000_patch_tokens.npy").exists()


def test_cache_metadata_differs_when_extraction_selection_differs(
    tmp_path: Path,
) -> None:
    dataset = TinyDataset(sample_count=2)
    adapter = TinyAdapter()
    extracted = extract_representations(model=adapter, dataset=dataset, split="train")
    metadata = save_representations(
        extracted, tmp_path, extraction=ExtractionConfig(batch_size=32)
    )
    incompatible = type(metadata)(
        schema_version=metadata.schema_version,
        model=metadata.model,
        dataset=metadata.dataset,
        split=metadata.split,
        num_samples=metadata.num_samples,
        extraction=ExtractionConfig(
            batch_size=32,
            representation_types=("cls", "patch_tokens"),
        ),
    )

    layer_incompatible = type(metadata)(
        schema_version=metadata.schema_version,
        model=metadata.model,
        dataset=metadata.dataset,
        split=metadata.split,
        num_samples=metadata.num_samples,
        extraction=ExtractionConfig(batch_size=32, layer_indices=(1,)),
    )

    status = cache_status(tmp_path, incompatible)
    layer_status = cache_status(tmp_path, layer_incompatible)

    assert status.valid is False
    assert layer_status.valid is False
    with pytest.raises(CacheMetadataError):
        load_representations(tmp_path, expected=incompatible)
