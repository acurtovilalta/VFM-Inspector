"""Reproducible PneumoniaMNIST-224 comparison demo.

Quick-mode metrics from small sample limits are functional smoke-test outputs only;
they are not meaningful benchmark results.
"""

from __future__ import annotations

import argparse
import gc
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from medvfm_inspector.analysis import (
    ComparedClassificationResult,
    ModelComparisonInput,
    ModelComparisonResult,
    compare_models,
)
from medvfm_inspector.datasets import DatasetAdapter, MedMNISTAdapter, SplitName
from medvfm_inspector.extraction import extract_representations
from medvfm_inspector.models import (
    DinoV2Adapter,
    DinoV3Adapter,
    ModelAdapter,
    RadDinoAdapter,
)
from medvfm_inspector.reporting import ReportArtifacts, write_comparison_report
from medvfm_inspector.types import DatasetInfo, ExtractedRepresentations, Sample

SPLITS: tuple[SplitName, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class DemoModelSpec:
    """One model included in the reproducible demo."""

    display_name: str
    checkpoint: str
    adapter_class: type[ModelAdapter]


MODEL_SPECS: tuple[DemoModelSpec, ...] = (
    DemoModelSpec("DINOv2-base", "facebook/dinov2-base", DinoV2Adapter),
    DemoModelSpec(
        "DINOv3 ViT-B/16",
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        DinoV3Adapter,
    ),
    DemoModelSpec("RAD-DINO", "microsoft/rad-dino", RadDinoAdapter),
)


class SampleLimitedDataset(DatasetAdapter):
    """Small deterministic view over a dataset for quick demo runs."""

    def __init__(self, base: DatasetAdapter, sample_limit: int) -> None:
        if sample_limit <= 0:
            raise ValueError("sample_limit must be positive.")
        self._base = base
        self._splits = {
            split: tuple(base.get_split(split)[:sample_limit]) for split in SPLITS
        }
        self._info = DatasetInfo(
            name=f"{base.info.name}-first-{sample_limit}",
            task_type=base.info.task_type,
            class_names=base.info.class_names,
            split_sizes={split: len(self._splits[split]) for split in SPLITS},
        )

    @property
    def info(self) -> DatasetInfo:
        """Return metadata for the limited dataset view."""
        return self._info

    def get_split(self, split: SplitName) -> Sequence[Sample]:
        """Return the first samples from the requested official split."""
        return self._splits[split]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the PneumoniaMNIST-224 comparison demo."""
    args = _parse_args(argv)
    torch.manual_seed(args.random_seed)

    output_root = Path(args.output_dir)
    experiment_dir = output_root / args.experiment_name
    cache_root = experiment_dir / "cache"

    print("MedVFM-Inspector PneumoniaMNIST-224 comparison demo")
    if args.sample_limit is not None:
        print(
            "Quick mode enabled: metrics from limited samples are not meaningful "
            "benchmark results."
        )

    dataset = _dataset(data_root=args.data_root, download=args.download)
    if args.sample_limit is not None:
        dataset = SampleLimitedDataset(dataset, args.sample_limit)

    print(f"Dataset: {dataset.info.name}")
    print(f"Split sizes: {dataset.info.split_sizes}")

    extracted = _extract_all_models(
        dataset=dataset,
        cache_root=cache_root,
        device=args.device,
        batch_size=args.batch_size,
        reuse_cache=not args.no_cache_reuse,
    )
    inputs = tuple(
        ModelComparisonInput(
            name=spec.display_name,
            train=extracted[spec.display_name]["train"],
            evaluation=extracted[spec.display_name]["val"],
        )
        for spec in MODEL_SPECS
    )
    min_train_samples = min(len(model_input.train.sample_ids) for model_input in inputs)
    knn_k = min(args.knn_k, min_train_samples)
    if knn_k != args.knn_k:
        print(f"Adjusted k-NN k from {args.knn_k} to {knn_k} for sample count.")

    comparison_without_pca = compare_models(
        inputs,
        analyses=("linear_probe", "knn", "cka"),
        random_seed=args.random_seed,
        knn_k=knn_k,
        knn_metric="cosine",
    )
    pca_layers = best_layer_indices(
        comparison_without_pca.linear_probe,
        comparison_without_pca.knn,
    )
    pca_comparison = compare_models(
        inputs,
        analyses=("pca",),
        pca_layer_indices=pca_layers,
    )
    comparison = combine_with_pca(comparison_without_pca, pca_comparison)
    artifacts = write_comparison_report(
        comparison,
        output_root,
        experiment_name=args.experiment_name,
    )

    _print_run_summary(
        extracted=extracted,
        comparison=comparison,
        artifacts=artifacts,
        cache_root=cache_root,
        pca_layers=pca_layers,
    )


def best_layer_indices(
    linear_probe: Sequence[ComparedClassificationResult],
    knn: Sequence[ComparedClassificationResult],
) -> tuple[int, ...]:
    """Return validation-selected layers for PCA from best probe/k-NN results."""
    selected = {
        result.layer_index
        for result in [*_best_by_model(linear_probe), *_best_by_model(knn)]
    }
    return tuple(sorted(selected))


def combine_with_pca(
    comparison: ModelComparisonResult,
    pca_comparison: ModelComparisonResult,
) -> ModelComparisonResult:
    """Attach PCA results to an existing comparison result."""
    return ModelComparisonResult(
        dataset=comparison.dataset,
        train_split=comparison.train_split,
        evaluation_split=comparison.evaluation_split,
        models=comparison.models,
        linear_probe=comparison.linear_probe,
        knn=comparison.knn,
        cka=comparison.cka,
        pca=pca_comparison.pca,
    )


def _extract_all_models(
    *,
    dataset: DatasetAdapter,
    cache_root: Path,
    device: str,
    batch_size: int,
    reuse_cache: bool,
) -> dict[str, dict[SplitName, ExtractedRepresentations]]:
    extracted: dict[str, dict[SplitName, ExtractedRepresentations]] = {}
    for spec in MODEL_SPECS:
        print(f"Loading {spec.display_name}: {spec.checkpoint}")
        adapter = _load_adapter(spec, device=device)
        model_reps: dict[SplitName, ExtractedRepresentations] = {}
        for split in SPLITS:
            cache_dir = cache_root / _slug(spec.display_name) / split
            print(f"Extracting {spec.display_name} {split} -> {cache_dir}")
            model_reps[split] = extract_representations(
                model=adapter,
                dataset=dataset,
                split=split,
                batch_size=batch_size,
                representation_types=("cls",),
                cache_dir=cache_dir,
                reuse_cache=reuse_cache,
            )
        extracted[spec.display_name] = model_reps
        del adapter
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return extracted


def _load_adapter(spec: DemoModelSpec, *, device: str) -> ModelAdapter:
    try:
        return spec.adapter_class(model_name=spec.checkpoint, device=device)
    except Exception as exc:
        if spec.adapter_class is DinoV3Adapter:
            raise SystemExit(_dinov3_access_message(spec.checkpoint, exc)) from exc
        raise


def _dataset(*, data_root: str, download: bool) -> DatasetAdapter:
    return MedMNISTAdapter(
        name="pneumoniamnist",
        size=224,
        root=data_root,
        download=download,
        as_rgb=True,
    )


def _best_by_model(
    results: Sequence[ComparedClassificationResult],
) -> tuple[ComparedClassificationResult, ...]:
    best: dict[str, ComparedClassificationResult] = {}
    for result in sorted(results, key=lambda item: (item.model_name, item.layer_index)):
        current = best.get(result.model_name)
        if current is None or result.balanced_accuracy > current.balanced_accuracy:
            best[result.model_name] = result
    return tuple(best[model_name] for model_name in sorted(best))


def _print_run_summary(
    *,
    extracted: dict[str, dict[SplitName, ExtractedRepresentations]],
    comparison: ModelComparisonResult,
    artifacts: ReportArtifacts,
    cache_root: Path,
    pca_layers: tuple[int, ...],
) -> None:
    print("\nObserved smoke details")
    print(f"Cache root: {cache_root}")
    print(f"PCA layers: {pca_layers}")
    for model_name, splits in extracted.items():
        print(f"{model_name} sample counts:")
        for split in SPLITS:
            reps = splits[split]
            shape = None
            if reps.layers and reps.layers[0].cls is not None:
                shape = tuple(reps.layers[0].cls.shape)
            print(f"  {split}: {len(reps.sample_ids)} samples, layer0 CLS {shape}")
    print("Best linear probe balanced accuracy:")
    for result in _best_by_model(comparison.linear_probe):
        print(
            f"  {result.model_name}: layer {result.layer_index}, "
            f"balanced_accuracy={result.balanced_accuracy:.4f}"
        )
    print("Best k-NN balanced accuracy:")
    for result in _best_by_model(comparison.knn):
        print(
            f"  {result.model_name}: layer {result.layer_index}, "
            f"balanced_accuracy={result.balanced_accuracy:.4f}"
        )
    print("CKA matrix dimensions:")
    for cka in comparison.cka:
        print(
            f"  {cka.source_model_name} vs {cka.target_model_name}: "
            f"{tuple(cka.result.matrix.shape)}"
        )
    print(f"Report: {artifacts.report_path}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible PneumoniaMNIST-224 comparison demo for "
            "DINOv2-base, DINOv3 ViT-B/16, and RAD-DINO."
        ),
        epilog=(
            "Quick mode is enabled by default through --sample-limit. Metrics from "
            "limited samples verify the pipeline only and are not benchmark "
            "results. DINOv3 may require accepting Hugging Face access terms and "
            "running `huggingface-cli login` before use."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where caches, figures, results, and the report are written.",
    )
    parser.add_argument(
        "--experiment-name",
        default="pneumoniamnist_comparison",
        help="Subdirectory name for this demo run.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Directory used by MedMNIST for local dataset files.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device passed to each model adapter, for example cpu or cuda.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images processed per extraction batch.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used by deterministic analysis steps.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=24,
        help="Limit each split to the first N samples for quick smoke runs.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="Number of neighbors for layer-wise k-NN classification.",
    )
    parser.add_argument(
        "--no-sample-limit",
        action="store_true",
        help="Process the full official train/validation/test splits.",
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Require the MedMNIST dataset to already exist under --data-root.",
    )
    parser.add_argument(
        "--no-cache-reuse",
        action="store_true",
        help="Recompute representations even when a valid cache exists.",
    )
    parser.set_defaults(download=True)
    args = parser.parse_args(argv)
    if args.no_sample_limit:
        args.sample_limit = None
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    if args.knn_k <= 0:
        parser.error("--knn-k must be positive")
    return args


def _dinov3_access_message(checkpoint: str, exc: Exception) -> str:
    return (
        f"Could not load DINOv3 checkpoint {checkpoint!r}. Official DINOv3 "
        "checkpoints on Hugging Face may require accepting the model access "
        "conditions and authenticating locally. Visit the checkpoint page, accept "
        "the terms for your Hugging Face account, then run `huggingface-cli login` "
        f"before retrying. Original error: {exc}"
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()


if __name__ == "__main__":
    main()
