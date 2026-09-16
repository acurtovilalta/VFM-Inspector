from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from medvfm_inspector.models import DinoV2Adapter, ModelAdapter
from medvfm_inspector.types import (
    LayerRepresentation,
    ModelInfo,
    ModelRepresentations,
)


class IncompleteAdapter(ModelAdapter):
    pass


class MinimalAdapter(ModelAdapter):
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(name="minimal", num_layers=1)

    @property
    def layer_names(self) -> tuple[str, ...]:
        return ("block_0",)

    def preprocess(self, images: Sequence[Any]) -> dict[str, torch.Tensor]:
        return {"pixel_values": torch.zeros(len(images), 3, 4, 4)}

    def extract_from_inputs(
        self, model_inputs: Mapping[str, torch.Tensor]
    ) -> ModelRepresentations:
        batch_size = model_inputs["pixel_values"].shape[0]
        cls = torch.zeros(batch_size, 2)
        patch_tokens = torch.zeros(batch_size, 1, 2)
        return ModelRepresentations(
            model=self.info,
            layers=(
                LayerRepresentation(
                    layer_name="block_0",
                    layer_index=0,
                    cls=cls,
                    patch_tokens=patch_tokens,
                ),
            ),
        )


@dataclass
class FakeConfig:
    num_hidden_layers: int = 2
    hidden_size: int = 4
    patch_size: int = 14
    model_type: str = "dinov2"


class FakeProcessor:
    def __call__(
        self, images: list[Image.Image], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.ones(len(images), 3, 8, 8)}


class FakeDinoV2Model:
    def __init__(self, config: FakeConfig | None = None) -> None:
        self.config = config or FakeConfig()
        self.device = torch.device("cpu")
        self.eval_called = False
        self.forward_called = False
        self.grad_enabled_during_forward: bool | None = None
        self.inference_mode_during_forward: bool | None = None

    def to(self, device: torch.device) -> "FakeDinoV2Model":
        self.device = device
        return self

    def eval(self) -> "FakeDinoV2Model":
        self.eval_called = True
        return self

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        assert kwargs["output_hidden_states"] is True
        assert kwargs["return_dict"] is True
        pixel_values = kwargs["pixel_values"]
        self.forward_called = True
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        self.inference_mode_during_forward = torch.is_inference_mode_enabled()

        batch_size = pixel_values.shape[0]
        token_count = 1 + 3
        hidden_size = self.config.hidden_size
        shape = (batch_size, token_count, hidden_size)
        hidden_states = [torch.full(shape, -1.0, device=pixel_values.device)]
        for layer_index in range(self.config.num_hidden_layers):
            start = layer_index * batch_size * token_count * hidden_size
            layer = torch.arange(
                start,
                start + batch_size * token_count * hidden_size,
                dtype=torch.float32,
                device=pixel_values.device,
            ).reshape(shape)
            hidden_states.append(layer)
        return SimpleNamespace(hidden_states=tuple(hidden_states))


def test_model_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        IncompleteAdapter()


def test_model_adapter_extract_delegates_preprocess_and_forward() -> None:
    adapter = MinimalAdapter()
    image = Image.new("RGB", (8, 8))

    representations = adapter.extract([image])

    assert representations.model.name == "minimal"
    assert len(representations.layers) == 1
    assert representations.layers[0].cls.shape == (1, 2)


def test_dinov2_adapter_metadata_layer_count_and_eval_mode() -> None:
    model = FakeDinoV2Model(FakeConfig(num_hidden_layers=3))
    adapter = DinoV2Adapter(model=model, processor=FakeProcessor())

    assert adapter.info.name == "facebook/dinov2-small"
    assert adapter.info.num_layers == 3
    assert adapter.info.hidden_size == 4
    assert adapter.info.patch_size == 14
    assert adapter.layer_names == ("block_0", "block_1", "block_2")
    assert model.eval_called


def test_dinov2_adapter_output_shapes_and_cls_patch_separation() -> None:
    model = FakeDinoV2Model()
    adapter = DinoV2Adapter(model=model, processor=FakeProcessor(), device="cpu")
    images = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]

    representations = adapter.extract(images)

    assert model.forward_called
    assert model.grad_enabled_during_forward is False
    assert model.inference_mode_during_forward is True
    assert len(representations.layers) == 2

    first_layer = representations.layers[0]
    assert first_layer.layer_name == "block_0"
    assert first_layer.layer_index == 0
    assert first_layer.cls.shape == (2, 4)
    assert first_layer.patch_tokens.shape == (2, 3, 4)

    expected_state = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    torch.testing.assert_close(first_layer.cls, expected_state[:, 0, :])
    torch.testing.assert_close(first_layer.patch_tokens, expected_state[:, 1:, :])


def test_dinov2_adapter_runs_on_cpu() -> None:
    adapter = DinoV2Adapter(
        model=FakeDinoV2Model(), processor=FakeProcessor(), device="cpu"
    )
    image = Image.new("RGB", (8, 8))

    representations = adapter.extract([image])

    assert representations.layers[0].cls.device.type == "cpu"
    assert representations.layers[0].patch_tokens.device.type == "cpu"
