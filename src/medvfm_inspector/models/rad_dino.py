"""RAD-DINO model adapter."""

from __future__ import annotations

from typing import Any

import torch

from medvfm_inspector.models.dinov2 import DinoV2Adapter


class RadDinoAdapter(DinoV2Adapter):
    """Adapter for the Hugging Face Microsoft RAD-DINO model."""

    def __init__(
        self,
        model_name: str = "microsoft/rad-dino",
        *,
        revision: str | None = None,
        device: str | torch.device = "cpu",
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            revision=revision,
            device=device,
            model=model,
            processor=processor,
        )
