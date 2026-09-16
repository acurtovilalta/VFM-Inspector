"""Explicit filesystem cache for extracted representations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from medvfm_inspector.types import (
    DatasetInfo,
    ExtractedRepresentations,
    LayerRepresentation,
    ModelInfo,
    RepresentationType,
)

SCHEMA_VERSION = "0.2"
_VALID_REPRESENTATION_TYPES: tuple[RepresentationType, ...] = (
    "cls",
    "patch_tokens",
    "register_tokens",
)


class CacheMetadataError(RuntimeError):
    """Raised when cached representation metadata is incompatible."""


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration values that affect one extraction run."""

    batch_size: int = 32
    representation_types: tuple[RepresentationType, ...] = ("cls",)
    layer_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        object.__setattr__(
            self,
            "representation_types",
            _normalize_representation_types(self.representation_types),
        )
        object.__setattr__(
            self,
            "layer_indices",
            _normalize_layer_indices(self.layer_indices),
        )


@dataclass(frozen=True)
class CacheMetadata:
    """Metadata used to validate a representation cache."""

    schema_version: str
    model: ModelInfo
    dataset: DatasetInfo
    split: str
    num_samples: int
    extraction: ExtractionConfig


@dataclass(frozen=True)
class CacheStatus:
    """Result of checking whether a cache can be reused."""

    valid: bool
    reason: str | None = None


def expected_metadata(
    *,
    model: ModelInfo,
    dataset: DatasetInfo,
    split: str,
    num_samples: int,
    extraction: ExtractionConfig,
) -> CacheMetadata:
    """Build cache metadata for the current extraction request."""
    return CacheMetadata(
        schema_version=SCHEMA_VERSION,
        model=model,
        dataset=dataset,
        split=split,
        num_samples=num_samples,
        extraction=extraction,
    )


def cache_status(cache_dir: str | Path, expected: CacheMetadata) -> CacheStatus:
    """Return whether a cache directory matches the expected metadata."""
    metadata_path = Path(cache_dir) / "metadata.json"
    if not metadata_path.exists():
        return CacheStatus(valid=False, reason="metadata.json is missing")

    try:
        observed = _read_metadata(metadata_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return CacheStatus(valid=False, reason=f"metadata is unreadable: {exc}")

    if observed != expected:
        return CacheStatus(valid=False, reason="metadata does not match request")
    return CacheStatus(valid=True)


def save_representations(
    representations: ExtractedRepresentations,
    cache_dir: str | Path,
    *,
    extraction: ExtractionConfig,
) -> CacheMetadata:
    """Save extracted representations as JSON metadata plus NumPy arrays."""
    path = Path(cache_dir)
    layers_dir = path / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    _clear_layer_tensors(layers_dir)

    metadata = expected_metadata(
        model=representations.model,
        dataset=representations.dataset,
        split=representations.split,
        num_samples=len(representations.sample_ids),
        extraction=extraction,
    )

    _write_json(path / "metadata.json", _metadata_to_json(metadata, representations))
    with (path / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id, label in zip(
            representations.sample_ids, representations.labels, strict=True
        ):
            handle.write(json.dumps({"sample_id": sample_id, "label": label}) + "\n")

    for layer in representations.layers:
        stem = f"layer_{layer.layer_index:03d}"
        if layer.cls is not None:
            _save_tensor(layers_dir / f"{stem}_cls.npy", layer.cls)
        if layer.patch_tokens is not None:
            _save_tensor(layers_dir / f"{stem}_patch_tokens.npy", layer.patch_tokens)
        if layer.register_tokens is not None:
            _save_tensor(
                layers_dir / f"{stem}_register_tokens.npy", layer.register_tokens
            )

    return metadata


def load_representations(
    cache_dir: str | Path,
    *,
    expected: CacheMetadata | None = None,
) -> ExtractedRepresentations:
    """Load extracted representations from a cache directory."""
    path = Path(cache_dir)
    metadata_json = _read_json(path / "metadata.json")
    metadata = _metadata_from_json(metadata_json)
    if expected is not None and metadata != expected:
        raise CacheMetadataError(
            "Cached representation metadata does not match request."
        )

    sample_ids: list[str] = []
    labels: list[int] = []
    with (path / "samples.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_ids.append(str(row["sample_id"]))
            labels.append(int(row["label"]))

    layers = tuple(
        _load_layer(path / "layers", layer) for layer in metadata_json["layers"]
    )
    return ExtractedRepresentations(
        model=metadata.model,
        dataset=metadata.dataset,
        split=metadata.split,
        sample_ids=tuple(sample_ids),
        labels=tuple(labels),
        layers=layers,
    )


def _metadata_to_json(
    metadata: CacheMetadata, representations: ExtractedRepresentations
) -> dict[str, Any]:
    return {
        **asdict(metadata),
        "layers": [
            {
                "layer_name": layer.layer_name,
                "layer_index": layer.layer_index,
                "has_cls": layer.cls is not None,
                "has_patch_tokens": layer.patch_tokens is not None,
                "has_register_tokens": layer.register_tokens is not None,
            }
            for layer in representations.layers
        ],
    }


def _metadata_from_json(payload: dict[str, Any]) -> CacheMetadata:
    return CacheMetadata(
        schema_version=str(payload["schema_version"]),
        model=ModelInfo(**payload["model"]),
        dataset=DatasetInfo(**payload["dataset"]),
        split=str(payload["split"]),
        num_samples=int(payload["num_samples"]),
        extraction=ExtractionConfig(**payload["extraction"]),
    )


def _read_metadata(path: Path) -> CacheMetadata:
    return _metadata_from_json(_read_json(path))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return cast(dict[str, Any], json.load(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _clear_layer_tensors(layers_dir: Path) -> None:
    for tensor_path in layers_dir.glob("*.npy"):
        tensor_path.unlink()


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().clone().numpy()
    np.save(path, array)


def _load_tensor(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path))


def _load_layer(layers_dir: Path, layer_json: dict[str, Any]) -> LayerRepresentation:
    layer_index = int(layer_json["layer_index"])
    stem = f"layer_{layer_index:03d}"
    cls = None
    if bool(layer_json.get("has_cls", True)):
        cls = _load_tensor(layers_dir / f"{stem}_cls.npy")

    patch_tokens = None
    if bool(layer_json.get("has_patch_tokens", True)):
        patch_tokens = _load_tensor(layers_dir / f"{stem}_patch_tokens.npy")

    register_tokens = None
    if bool(layer_json["has_register_tokens"]):
        register_tokens = _load_tensor(layers_dir / f"{stem}_register_tokens.npy")

    return LayerRepresentation(
        layer_name=str(layer_json["layer_name"]),
        layer_index=layer_index,
        cls=cls,
        patch_tokens=patch_tokens,
        register_tokens=register_tokens,
    )


def _normalize_representation_types(
    representation_types: tuple[RepresentationType, ...],
) -> tuple[RepresentationType, ...]:
    if not representation_types:
        raise ValueError("At least one representation type must be selected.")

    unknown = sorted(set(representation_types) - set(_VALID_REPRESENTATION_TYPES))
    if unknown:
        raise ValueError(f"Unsupported representation type(s): {unknown}")

    requested = set(representation_types)
    return tuple(
        representation_type
        for representation_type in _VALID_REPRESENTATION_TYPES
        if representation_type in requested
    )


def _normalize_layer_indices(
    layer_indices: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    if layer_indices is None:
        return None
    if not layer_indices:
        raise ValueError("At least one layer index must be selected.")
    if any(index < 0 for index in layer_indices):
        raise ValueError("Layer indices must be non-negative.")
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError("Layer indices must be unique.")
    return tuple(sorted(layer_indices))
