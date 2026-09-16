"""Batched representation extraction."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

from medvfm_inspector.datasets.base import DatasetAdapter, SplitName
from medvfm_inspector.extraction.cache import (
    ExtractionConfig,
    cache_status,
    expected_metadata,
    load_representations,
    save_representations,
)
from medvfm_inspector.models.base import ModelAdapter
from medvfm_inspector.types import (
    DatasetInfo,
    ExtractedRepresentations,
    LayerRepresentation,
    RepresentationType,
    Sample,
)


def extract_representations(
    *,
    model: ModelAdapter,
    dataset: DatasetAdapter,
    split: SplitName,
    batch_size: int = 32,
    representation_types: Sequence[RepresentationType] = ("cls",),
    layer_indices: Sequence[int] | None = None,
    cache_dir: str | Path | None = None,
    reuse_cache: bool = True,
) -> ExtractedRepresentations:
    """Extract or load cached layer representations for one dataset split."""
    extraction = ExtractionConfig(
        batch_size=batch_size,
        representation_types=tuple(representation_types),
        layer_indices=None if layer_indices is None else tuple(layer_indices),
    )

    split_data = dataset.get_split(split)
    metadata = expected_metadata(
        model=model.info,
        dataset=dataset.info,
        split=split,
        num_samples=len(split_data),
        extraction=extraction,
    )
    if cache_dir is not None and reuse_cache:
        status = cache_status(cache_dir, metadata)
        if status.valid:
            return load_representations(cache_dir, expected=metadata)

    extracted = _extract_split(
        model=model,
        dataset_info=dataset.info,
        split=split,
        samples=split_data,
        extraction=extraction,
    )
    if cache_dir is not None:
        save_representations(extracted, cache_dir, extraction=extraction)
    return extracted


def _extract_split(
    *,
    model: ModelAdapter,
    dataset_info: DatasetInfo,
    split: str,
    samples: Sequence[Sample],
    extraction: ExtractionConfig,
) -> ExtractedRepresentations:
    sample_ids: list[str] = []
    labels: list[int] = []
    selected_indices: tuple[int, ...] | None = None
    layer_names: tuple[str, ...] | None = None
    cls_by_layer: list[list[torch.Tensor]] | None = None
    patch_by_layer: list[list[torch.Tensor]] | None = None
    register_by_layer: list[list[torch.Tensor] | None] | None = None

    include_cls = "cls" in extraction.representation_types
    include_patch_tokens = "patch_tokens" in extraction.representation_types
    include_register_tokens = "register_tokens" in extraction.representation_types

    for batch in _batches(samples, extraction.batch_size):
        sample_ids.extend(sample.sample_id for sample in batch)
        labels.extend(sample.label for sample in batch)
        batch_representations = model.extract([sample.image for sample in batch])

        if selected_indices is None:
            selected_indices = _resolve_layer_indices(
                extraction.layer_indices, len(batch_representations.layers)
            )
            layer_names = tuple(
                batch_representations.layers[index].layer_name
                for index in selected_indices
            )
            if include_cls:
                cls_by_layer = [[] for _ in selected_indices]
            if include_patch_tokens:
                patch_by_layer = [[] for _ in selected_indices]
            register_by_layer = [
                []
                if include_register_tokens
                and batch_representations.layers[index].register_tokens is not None
                else None
                for index in selected_indices
            ]

        if selected_indices is None or register_by_layer is None:
            raise RuntimeError("Extraction buffers were not initialized.")

        for output_index, layer_index in enumerate(selected_indices):
            layer = batch_representations.layers[layer_index]
            if include_cls:
                if cls_by_layer is None:
                    raise RuntimeError("CLS extraction buffer was not initialized.")
                if layer.cls is None:
                    raise RuntimeError(f"Layer {layer.layer_name} has no CLS tensor.")
                cls_by_layer[output_index].append(layer.cls.detach().cpu())
            if include_patch_tokens:
                if patch_by_layer is None:
                    raise RuntimeError(
                        "Patch-token extraction buffer was not initialized."
                    )
                if layer.patch_tokens is None:
                    raise RuntimeError(
                        f"Layer {layer.layer_name} has no patch-token tensor."
                    )
                patch_by_layer[output_index].append(layer.patch_tokens.detach().cpu())
            register_buffer = register_by_layer[output_index]
            if register_buffer is not None:
                if layer.register_tokens is None:
                    raise RuntimeError(
                        "Register-token presence changed between batches."
                    )
                register_buffer.append(layer.register_tokens.detach().cpu())

    if selected_indices is None or register_by_layer is None:
        raise ValueError("Cannot extract representations from an empty split.")
    if layer_names is None:
        raise ValueError("Cannot extract representations without model layers.")

    layers = tuple(
        LayerRepresentation(
            layer_name=layer_names[output_index],
            layer_index=layer_index,
            cls=_cat_optional_tensors(
                None if cls_by_layer is None else cls_by_layer[output_index]
            ),
            patch_tokens=_cat_optional_tensors(
                None if patch_by_layer is None else patch_by_layer[output_index]
            ),
            register_tokens=_cat_optional_tensors(register_by_layer[output_index]),
        )
        for output_index, layer_index in enumerate(selected_indices)
    )
    return ExtractedRepresentations(
        model=model.info,
        dataset=dataset_info,
        split=split,
        sample_ids=tuple(sample_ids),
        labels=tuple(labels),
        layers=layers,
    )


def _resolve_layer_indices(
    layer_indices: tuple[int, ...] | None, layer_count: int
) -> tuple[int, ...]:
    indices = tuple(range(layer_count)) if layer_indices is None else layer_indices
    invalid = [index for index in indices if index >= layer_count]
    if invalid:
        raise ValueError(
            f"Selected layer index out of range for {layer_count} layers: {invalid}"
        )
    return indices


def _cat_optional_tensors(tensors: list[torch.Tensor] | None) -> torch.Tensor | None:
    if tensors is None:
        return None
    return torch.cat(tensors, dim=0)


def _batches(samples: Sequence[Sample], batch_size: int) -> Iterator[list[Sample]]:
    for start in range(0, len(samples), batch_size):
        yield [
            samples[index]
            for index in range(start, min(start + batch_size, len(samples)))
        ]
