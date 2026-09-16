from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from medvfm_inspector.models import DinoV3Adapter


@dataclass
class FakeDinoV3Config:
    num_hidden_layers: int = 2
    hidden_size: int = 5
    patch_size: int = 16
    model_type: str = "dinov3_vit"
    num_register_tokens: int = 4


class FakeProcessor:
    def __call__(
        self, images: list[Image.Image], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.ones(len(images), 3, 16, 16)}


class FakeDinoV3Model:
    def __init__(self, config: FakeDinoV3Config | None = None) -> None:
        self.config = config or FakeDinoV3Config()
        self.eval_called = False
        self.forward_called = False
        self.grad_enabled_during_forward: bool | None = None
        self.inference_mode_during_forward: bool | None = None

    def to(self, device: torch.device) -> "FakeDinoV3Model":
        self.device = device
        return self

    def eval(self) -> "FakeDinoV3Model":
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
        token_count = 1 + self.config.num_register_tokens + 3
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


def test_dinov3_adapter_metadata_layer_count_and_eval_mode() -> None:
    model = FakeDinoV3Model(FakeDinoV3Config(num_hidden_layers=3))
    adapter = DinoV3Adapter(model=model, processor=FakeProcessor())

    assert adapter.info.name == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert adapter.info.num_layers == 3
    assert adapter.info.hidden_size == 5
    assert adapter.info.patch_size == 16
    assert adapter.info.num_register_tokens == 4
    assert adapter.layer_names == ("block_0", "block_1", "block_2")
    assert model.eval_called


def test_dinov3_adapter_extracts_cls_register_and_patch_boundaries() -> None:
    model = FakeDinoV3Model()
    adapter = DinoV3Adapter(model=model, processor=FakeProcessor(), device="cpu")
    images = [Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))]

    representations = adapter.extract(images)

    assert model.forward_called
    assert model.grad_enabled_during_forward is False
    assert model.inference_mode_during_forward is True
    assert len(representations.layers) == 2

    first_layer = representations.layers[0]
    assert first_layer.cls.shape == (2, 5)
    assert first_layer.register_tokens is not None
    assert first_layer.register_tokens.shape == (2, 4, 5)
    assert first_layer.patch_tokens.shape == (2, 3, 5)

    expected_state = torch.arange(2 * 8 * 5, dtype=torch.float32).reshape(2, 8, 5)
    torch.testing.assert_close(first_layer.cls, expected_state[:, 0, :])
    torch.testing.assert_close(first_layer.register_tokens, expected_state[:, 1:5, :])
    torch.testing.assert_close(first_layer.patch_tokens, expected_state[:, 5:, :])


def test_dinov3_adapter_runs_on_cpu() -> None:
    adapter = DinoV3Adapter(
        model=FakeDinoV3Model(), processor=FakeProcessor(), device="cpu"
    )
    image = Image.new("RGB", (16, 16))

    representations = adapter.extract([image])

    assert representations.layers[0].cls.device.type == "cpu"
    assert representations.layers[0].register_tokens is not None
    assert representations.layers[0].register_tokens.device.type == "cpu"
    assert representations.layers[0].patch_tokens.device.type == "cpu"
