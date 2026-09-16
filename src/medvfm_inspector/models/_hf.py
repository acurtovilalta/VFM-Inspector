"""Small Hugging Face helpers for model adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

import torch

from medvfm_inspector.types import ModelInfo


class ImageProcessor(Protocol):
    """Callable image processor returning PyTorch tensors."""

    def __call__(
        self, images: Sequence[Any], return_tensors: str
    ) -> Mapping[str, torch.Tensor]: ...


class HfModel(Protocol):
    """Minimal Hugging Face model surface needed by adapters."""

    config: Any

    def eval(self) -> HfModel: ...

    def to(self, device: torch.device) -> HfModel: ...

    def __call__(self, **kwargs: Any) -> Any: ...


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve and validate a CPU or CUDA device."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false."
        )
    if resolved.type != "cpu" and resolved.type != "cuda":
        raise ValueError("Model adapters currently support only CPU and CUDA devices.")
    return resolved


def load_hugging_face_components(
    model_name: str, revision: str | None
) -> tuple[ImageProcessor, HfModel]:
    """Load an image processor and model from Hugging Face Transformers."""
    from transformers import AutoImageProcessor, AutoModel

    image_processor_loader = cast(Any, AutoImageProcessor)
    model_loader = cast(Any, AutoModel)

    load_kwargs: dict[str, str] = {}
    if revision is not None:
        load_kwargs["revision"] = revision

    processor = cast(
        ImageProcessor,
        image_processor_loader.from_pretrained(model_name, **load_kwargs),
    )
    model = cast(HfModel, model_loader.from_pretrained(model_name, **load_kwargs))
    return processor, model


def build_model_info(
    config: Any,
    *,
    model_name: str,
    revision: str | None,
    default_architecture: str,
    num_register_tokens: int = 0,
) -> ModelInfo:
    """Build common model metadata from a Hugging Face config."""
    num_layers = getattr(config, "num_hidden_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("Model config must define a positive num_hidden_layers.")

    return ModelInfo(
        name=model_name,
        revision=revision,
        architecture=getattr(config, "model_type", None) or default_architecture,
        num_layers=num_layers,
        hidden_size=getattr(config, "hidden_size", None),
        patch_size=getattr(config, "patch_size", None),
        num_register_tokens=num_register_tokens,
    )


def transformer_block_states(
    hidden_states: tuple[torch.Tensor, ...], *, expected_layers: int, model_name: str
) -> tuple[torch.Tensor, ...]:
    """Drop the embedding output and return one tensor per transformer block."""
    expected_with_embeddings = expected_layers + 1
    if len(hidden_states) != expected_with_embeddings:
        raise RuntimeError(
            f"{model_name} hidden states should contain the embedding output plus "
            f"one tensor per transformer block; expected {expected_with_embeddings}, "
            f"got {len(hidden_states)}."
        )
    return hidden_states[1:]


def require_hidden_state_shape(hidden_state: torch.Tensor, layer_name: str) -> None:
    """Validate that a hidden state has batch, token, and hidden dimensions."""
    if hidden_state.ndim != 3:
        raise RuntimeError(
            f"Expected hidden state for {layer_name} to have shape "
            "(batch, tokens, hidden_size)."
        )
