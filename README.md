# MedVFM-Inspector

MedVFM-Inspector is a model-agnostic microscope for probing, comparing, and
auditing representations learned by vision foundation models, with first-class
support for medical imaging.

It helps researchers and engineers answer practical questions such as which
layers are most useful for a medical classification task, how representations
change across checkpoints, and whether cached embeddings can be reused for
repeatable analyses and reports. The project is under active development; v0.1.0
is the first public release target.

## Capabilities

- Model adapters for DINOv2, DINOv3 ViT, and RAD-DINO Hugging Face checkpoints.
- MedMNIST support, including PneumoniaMNIST-224 train/validation/test splits.
- Batched representation extraction with explicit local caching.
- Layer-wise linear probing and k-NN classification.
- PCA, nearest-neighbor retrieval, and linear CKA.
- Static HTML comparison reports with generated figures.

## Architecture

```text
model adapter + dataset adapter
  -> batched representation extraction
  -> explicit disk cache
  -> analysis
  -> visualization/report
```

Model-specific token semantics live in model adapters. Dataset-specific loading
lives in dataset adapters. Analysis consumes standardized cached
representations, so expensive model forward passes do not need to be rerun.

## Installation

MedVFM-Inspector uses `uv` for dependency management:

```bash
uv sync --dev
uv run medvfm --help
```

Useful development checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/medvfm_inspector
uv run pytest
```

## Minimal DINOv2 Example

```python
from PIL import Image

from medvfm_inspector.models import DinoV2Adapter

image = Image.new("RGB", (224, 224))
adapter = DinoV2Adapter(model_name="facebook/dinov2-small", device="cpu")
representations = adapter.extract([image])

print(representations.model_info.name)
print(representations.layers[0].cls.shape)
```

## PneumoniaMNIST-224 Demo

The reproducible demo compares:

- `facebook/dinov2-base`
- `facebook/dinov3-vitb16-pretrain-lvd1689m`
- `microsoft/rad-dino`

It loads PneumoniaMNIST-224, extracts CLS representations for all layers, reuses
valid caches, runs layer-wise linear probing and k-NN on the validation split,
computes cross-model CKA, generates selected PCA plots, and writes an HTML
comparison report.

DINOv3 may require accepting the Hugging Face model access conditions and
authenticating locally:

```bash
huggingface-cli login
```

Quick mode limits each split so contributors can verify the full path without a
large run. Quick-mode metrics are functional smoke-test outputs only, not
meaningful benchmark results.

```bash
uv run python examples/pneumoniamnist_comparison.py \
  --device cpu \
  --batch-size 2 \
  --sample-limit 8 \
  --output-dir outputs
```

For the full experiment, remove the sample limit:

```bash
uv run python examples/pneumoniamnist_comparison.py \
  --device cuda \
  --batch-size 16 \
  --no-sample-limit \
  --output-dir outputs
```

Expected output structure:

```text
outputs/pneumoniamnist_comparison/
├── cache/
├── figures/
├── results/
│   └── summary.json
└── report.html
```

The report contains the experiment overview, model metadata, best validation
linear-probe and k-NN layers, CKA heatmaps, selected PCA plots, and a concise
summary of computed results.

## Release Checklist

Before tagging a release:

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src/medvfm_inspector
uv run pytest
uv run python examples/pneumoniamnist_comparison.py --help
```

Then run a quick real demo, inspect the generated HTML report, confirm the
version, and create the git tag:

```bash
git tag v0.1.0
```
