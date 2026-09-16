"""Model adapter interfaces and implementations."""

from medvfm_inspector.models.base import ModelAdapter
from medvfm_inspector.models.dinov2 import DinoV2Adapter
from medvfm_inspector.models.dinov3 import DinoV3Adapter
from medvfm_inspector.models.rad_dino import RadDinoAdapter

__all__ = ["DinoV2Adapter", "DinoV3Adapter", "ModelAdapter", "RadDinoAdapter"]
