from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from medvfm_inspector.models import RadDinoAdapter


@dataclass
class FakeRadDinoConfig:
    num_hidden_layers: int = 2
    hidden_size: int = 6
    patch_size: int = 14
    model_type: str = "dinov2"


class FakeProcessor:
    def __call__(
        self, images: list[Image.Image], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.ones(len(images), 3, 32, 32)}


class FakeRadDinoModel:
    def __init__(self, config: FakeRadDinoConfig | None = None) -> None:
        self.config = config or FakeRadDinoConfig()
        self.eval_called = False
        self.forward_called = False
        self.grad_enabled_during_forward: bool | None = None
        self.inference_mode_during_forward: bool | None = None

    def to(self, device: torch.device) -> "FakeRadDinoModel":
        self.device = device
        return self

    def eval(self) -> "FakeRadDinoModel":
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
        token_count = 1 + 4
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


def test_rad_dino_adapter_metadata_layer_count_and_eval_mode() -> None:
    model = FakeRadDinoModel(FakeRadDinoConfig(num_hidden_layers=3))
    adapter = RadDinoAdapter(model=model, processor=FakeProcessor())

    assert adapter.info.name == "microsoft/rad-dino"
    assert adapter.info.num_layers == 3
    assert adapter.info.hidden_size == 6
    assert adapter.info.patch_size == 14
    assert adapter.info.num_register_tokens == 0
    assert adapter.layer_names == ("block_0", "block_1", "block_2")
    assert model.eval_called


def test_rad_dino_adapter_extracts_cls_and_patch_tokens() -> None:
    model = FakeRadDinoModel()
    adapter = RadDinoAdapter(model=model, processor=FakeProcessor(), device="cpu")
    images = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]

    representations = adapter.extract(images)

    assert model.forward_called
    assert model.grad_enabled_during_forward is False
    assert model.inference_mode_during_forward is True
    assert len(representations.layers) == 2

    first_layer = representations.layers[0]
    assert first_layer.cls.shape == (2, 6)
    assert first_layer.register_tokens is None
    assert first_layer.patch_tokens.shape == (2, 4, 6)

    expected_state = torch.arange(2 * 5 * 6, dtype=torch.float32).reshape(2, 5, 6)
    torch.testing.assert_close(first_layer.cls, expected_state[:, 0, :])
    torch.testing.assert_close(first_layer.patch_tokens, expected_state[:, 1:, :])


def test_rad_dino_adapter_runs_on_cpu() -> None:
    adapter = RadDinoAdapter(
        model=FakeRadDinoModel(), processor=FakeProcessor(), device="cpu"
    )
    image = Image.new("RGB", (32, 32))

    representations = adapter.extract([image])

    assert representations.layers[0].cls.device.type == "cpu"
    assert representations.layers[0].patch_tokens.device.type == "cpu"
