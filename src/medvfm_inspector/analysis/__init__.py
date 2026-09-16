"""Representation analysis utilities."""

from medvfm_inspector.analysis.cka import (
    CKALayerMetadata,
    CKAResult,
    linear_cka,
    within_model_cka,
)
from medvfm_inspector.analysis.comparison import (
    ComparedCKAResult,
    ComparedClassificationResult,
    ComparedPCAResult,
    ModelComparisonInput,
    ModelComparisonResult,
    compare_models,
)
from medvfm_inspector.analysis.knn import KNNResult, run_knn_classification
from medvfm_inspector.analysis.linear_probe import (
    LinearProbeConfig,
    LinearProbeResult,
    run_linear_probe,
)
from medvfm_inspector.analysis.pca import PCAResult, run_pca
from medvfm_inspector.analysis.retrieval import NeighborResult, nearest_neighbors

__all__ = [
    "CKALayerMetadata",
    "CKAResult",
    "ComparedCKAResult",
    "ComparedClassificationResult",
    "ComparedPCAResult",
    "KNNResult",
    "LinearProbeConfig",
    "ModelComparisonInput",
    "ModelComparisonResult",
    "LinearProbeResult",
    "NeighborResult",
    "PCAResult",
    "linear_cka",
    "compare_models",
    "nearest_neighbors",
    "run_knn_classification",
    "run_linear_probe",
    "run_pca",
    "within_model_cka",
]
