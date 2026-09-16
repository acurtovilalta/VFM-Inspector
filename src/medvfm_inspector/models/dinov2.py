"""DINOv2 model adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch

from medvfm_inspector.models._hf import (
    HfModel,
    ImageProcessor,
    build_model_info,
    load_hugging_face_components,
    require_hidden_state_shape,
    resolve_device,
    transformer_block_states,
)
from medvfm_inspector.models.base import ModelAdapter
from medvfm_inspector.types import LayerRepresentation, ModelInfo, ModelRepresentations


class DinoV2Adapter(ModelAdapter):
    """Adapter for Hugging Face DINOv2 vision models."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        *,
        revision: str | None = None,
        device: str | torch.device = "cpu",
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self._device = resolve_device(device)
        self._model_name = model_name
        self._revision = revision
        self._processor = (
            cast(ImageProcessor, processor) if processor is not None else None
        )
        self._model = cast(HfModel, model) if model is not None else None

        if self._processor is None or self._model is None:
            self._load_from_hugging_face()

        model = self._require_model()
        self._model = model.to(self._device).eval()
        self._info = build_model_info(
            self._model.config,
            model_name=model_name,
            revision=revision,
            default_architecture="dinov2",
        )
        self._layer_names = tuple(
            f"block_{layer_index}" for layer_index in range(self._info.num_layers)
        )

    @property
    def info(self) -> ModelInfo:
        """Return metadata for the loaded DINOv2 model."""
        return self._info

    @property
    def layer_names(self) -> tuple[str, ...]:
        """Return one name per exposed transformer block."""
        return self._layer_names

    def preprocess(self, images: Sequence[Any]) -> dict[str, torch.Tensor]:
        """Preprocess a batch of PIL images for DINOv2."""
        if not images:
            raise ValueError("DinoV2Adapter.preprocess requires at least one image.")

        processor = self._require_processor()
        encoded = processor(images=images, return_tensors="pt")
        return {name: tensor.to(self._device) for name, tensor in encoded.items()}

    def extract_from_inputs(
        self, model_inputs: Mapping[str, torch.Tensor]
    ) -> ModelRepresentations:
        """Extract CLS and patch-token representations from each DINOv2 block."""
        inputs = {
            name: tensor.to(self._device) for name, tensor in model_inputs.items()
        }
        with torch.inference_mode():
            outputs = self._require_model()(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "DINOv2 model did not return hidden states. Ensure the model supports "
                "output_hidden_states=True."
            )

        block_states = transformer_block_states(
            tuple(hidden_states),
            expected_layers=self.info.num_layers,
            model_name="DINOv2",
        )
        layers = tuple(
            _layer_representation(layer_name, layer_index, hidden_state)
            for layer_index, (layer_name, hidden_state) in enumerate(
                zip(self.layer_names, block_states, strict=True)
            )
        )
        return ModelRepresentations(model=self.info, layers=layers)

    def _load_from_hugging_face(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        processor, model = load_hugging_face_components(
            self._model_name, self._revision
        )
        if self._processor is None:
            self._processor = processor
        if self._model is None:
            self._model = model

    def _require_processor(self) -> ImageProcessor:
        if self._processor is None:
            raise RuntimeError("DINOv2 image processor has not been loaded.")
        return self._processor

    def _require_model(self) -> HfModel:
        if self._model is None:
            raise RuntimeError("DINOv2 model has not been loaded.")
        return self._model


def _layer_representation(
    layer_name: str, layer_index: int, hidden_state: torch.Tensor
) -> LayerRepresentation:
    require_hidden_state_shape(hidden_state, layer_name)
    if hidden_state.shape[1] < 2:
        raise RuntimeError(
            f"Expected hidden state for {layer_name} to include one CLS token and "
            "at least one patch token."
        )

    return LayerRepresentation(
        layer_name=layer_name,
        layer_index=layer_index,
        cls=hidden_state[:, 0, :].detach(),
        patch_tokens=hidden_state[:, 1:, :].detach(),
    )
