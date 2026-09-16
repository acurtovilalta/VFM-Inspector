# MedVFM-Inspector: Product and Implementation Specification

> Status: implementation blueprint
> Target repository: `medvfm-inspector`
> Python package: `medvfm`
> CLI executable: `medvfm`
> First public milestone: `v0.1.0`
> Stable milestone: `v1.0.0`
> Last verified: 2026-09-04

## 1. Project mission

MedVFM-Inspector is an open-source workbench for understanding what vision foundation models encode, where information appears across layers, and how representations change across models or after fine-tuning.

The tool should answer questions such as:

- Which transformer layer is most useful for a downstream task?
- Is the final layer actually the best representation for this dataset?
- How linearly separable is the target at each layer?
- How does k-NN performance evolve through the network?
- How similar are representations across layers or across two checkpoints?
- Does a domain-specific model reorganize representations compared with its general-domain parent?
- In v1.0: which layers encode nuisance variables such as site, scanner, acquisition protocol, age, or sex?
- In v1.0: did fine-tuning improve target information while also increasing shortcut or site information?
- In v1.0: what changed at the patch-token level?

The project is primarily a representation-analysis tool, not a training framework.

A useful one-sentence description for the README:

> **MedVFM-Inspector is a model-agnostic microscope for probing, comparing, and auditing representations learned by vision foundation models, with first-class support for medical imaging.**

## 2. Product principles

1. **Analysis first, training second.**
   The core package consumes pretrained or fine-tuned checkpoints. Full model training is out of scope for v0.1.

2. **Model adapters, not model-specific scripts.**
   DINOv2 and RAD-DINO are the first tested models, but the architecture must make new backbones easy to add.

3. **Medical-first, not medical-only.**
   The core abstractions must also work on ordinary computer-vision datasets.

4. **Separate extraction from analysis.**
   Expensive forward passes are cached. Probes, CKA, visualizations, and reports should be rerunnable without re-extracting embeddings.

5. **No test-set tuning.**
   Hyperparameters are selected on the validation split. The test split is evaluated only after selection.

6. **Reproducible by default.**
   Every run stores the resolved config, seed, package versions, model identifier/revision when available, dataset description, and output schema.

7. **Static artifacts before dashboards.**
   v0.1 produces a self-contained HTML report. A richer interactive dashboard is a v1.0 feature.

8. **No clinical claims.**
   This is research and engineering tooling, not a diagnostic system.

---

# PART I - v0.1.0

## 3. v0.1 goal

v0.1 must be a small but genuinely useful end-to-end product.

A user should be able to run:

```bash
uv run medvfm inspect -c configs/demo/dinov2_base_pneumoniamnist.yaml
uv run medvfm inspect -c configs/demo/dinov3_vitb16_pneumoniamnist.yaml
uv run medvfm inspect -c configs/demo/rad_dino_pneumoniamnist.yaml
uv run medvfm compare \
  runs/dinov2_base_pneumoniamnist \
  runs/dinov3_vitb16_pneumoniamnist
uv run medvfm compare \
  runs/dinov3_vitb16_pneumoniamnist \
  runs/rad_dino_pneumoniamnist
```

and receive:

- cached layer-wise embeddings,
- layer-wise linear-probe results,
- layer-wise k-NN results,
- PCA visualizations,
- within-model and cross-model CKA,
- nearest-neighbor examples,
- a machine-readable summary,
- and a self-contained HTML report.

v0.1 is complete only when the core workflow works from a clean environment using public data and an ungated checkpoint, and the DINOv3 path is additionally validated when the user has accepted the official checkpoint access conditions.

## 4. Scope of v0.1

### Must have

- 2D single-label classification datasets.
- Hugging Face Transformers vision encoders that expose hidden states.
- Explicitly tested DINOv2-family support.
- Explicitly tested DINOv3 ViT support.
- Explicitly tested `facebook/dinov2-small`.
- Explicitly tested `facebook/dinov2-base`.
- Explicitly tested `facebook/dinov3-vitb16-pretrain-lvd1689m`.
- Explicitly tested `microsoft/rad-dino`.
- Register-token-aware patch pooling for DINOv3.
- MedMNIST 2D adapter with 224x224 support.
- Generic CSV image dataset adapter.
- Layer-wise CLS-token extraction.
- Layer-wise mean-patch-token extraction.
- Embedding cache on disk.
- Linear probes per layer.
- k-NN probes per layer.
- Classification metrics.
- Linear CKA.
- PCA.
- Nearest-neighbor retrieval.
- Static/self-contained HTML report.
- Model-to-model/checkpoint-to-checkpoint comparison.
- CLI.
- YAML configuration.
- Unit tests and CPU integration tests.
- Reproducibility manifest.
- GitHub Actions CI.
- Clean README with a reproducible demo.

### Explicitly out of scope for v0.1

Do not implement these before v0.1 is stable:

- model fine-tuning,
- LoRA/PEFT training,
- segmentation,
- 3D volumes,
- multi-label classification,
- multimodal image-text analysis,
- scanner/site nuisance probing,
- OOD group evaluation,
- patch-token heatmaps,
- attention rollout,
- Streamlit/Gradio UI,
- distributed multi-GPU extraction,
- arbitrary hooks into every possible PyTorch model,
- an experiment tracking server,
- a database,
- an LLM-generated report.

Avoid scope creep. The v0.1 value proposition is layer-wise representation inspection and model comparison.

### DINOv3-specific v0.1 decision

DINOv3 is an exception to the otherwise conservative model-support scope: it is included in v0.1 because it materially strengthens the project's scientific and portfolio value. Support only the **ViT DINOv3 family** in v0.1. Do not add DINOv3 ConvNeXt, segmentation heads, or every released DINOv3 variant before the core workflow is stable.

The canonical v0.1 DINOv3 checkpoint is:

```text
facebook/dinov3-vitb16-pretrain-lvd1689m
```

Use ViT-B/16 for the main comparison because it is a much more informative capacity-level counterpart to DINOv2-base and RAD-DINO than comparing models with very different parameter scales. Do not hard-code scientific superiority: the tool should measure whether DINOv3 is better on each dataset and at which layer.

---

## 5. Recommended repository structure

```text
medvfm-inspector/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── demo-smoke.yml
├── assets/
│   └── demo/
│       ├── report_preview.png
│       └── README.md
├── configs/
│   ├── demo/
│   │   ├── dinov2_small_pneumoniamnist.yaml
│   │   ├── dinov2_base_pneumoniamnist.yaml
│   │   ├── dinov3_vitb16_pneumoniamnist.yaml
│   │   ├── rad_dino_pneumoniamnist.yaml
│   │   ├── compare_dinov2_base_dinov3.yaml
│   │   └── compare_dinov3_rad_dino.yaml
│   └── examples/
│       └── csv_dataset.yaml
├── docs/
│   ├── architecture.md
│   ├── adding_a_dataset.md
│   ├── adding_a_model.md
│   ├── output_format.md
│   └── methodology.md
├── examples/
│   ├── inspect_medmnist.py
│   └── inspect_custom_csv.py
├── src/
│   └── medvfm/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── constants.py
│       ├── logging.py
│       ├── registry.py
│       ├── types.py
│       ├── utils/
│       │   ├── hashing.py
│       │   ├── io.py
│       │   ├── reproducibility.py
│       │   └── validation.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── hf_hidden_states.py
│       │   ├── dinov2.py
│       │   └── dinov3.py
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── csv_images.py
│       │   └── medmnist.py
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── cache.py
│       │   ├── extractor.py
│       │   └── pooling.py
│       ├── probes/
│       │   ├── __init__.py
│       │   ├── knn.py
│       │   └── linear.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── classification.py
│       │   └── cka.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── geometry.py
│       │   ├── neighbors.py
│       │   └── comparison.py
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── cka.py
│       │   ├── embeddings.py
│       │   ├── neighbors.py
│       │   └── probes.py
│       └── report/
│           ├── __init__.py
│           ├── builder.py
│           ├── schema.py
│           └── templates/
│               └── report.html.j2
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_cache.py
│   │   ├── test_cka.py
│   │   ├── test_config.py
│   │   ├── test_metrics.py
│   │   └── test_pooling.py
│   └── integration/
│       ├── test_csv_dataset.py
│       ├── test_local_hf_model.py
│       └── test_pipeline_tiny.py
├── .gitignore
├── .pre-commit-config.yaml
├── AGENTS.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

Notes:

- Use a `src/` layout.
- Do not commit generated `runs/` directories or model weights.
- Keep the public package name short: `medvfm`.
- `AGENTS.md` should tell coding agents to read this specification, run tests before and after changes, and not expand v0.1 scope without an issue.

---

## 6. Core architecture

### 6.1 Data flow

```text
YAML config
   |
   v
Config validation
   |
   +--> DatasetAdapter ---> Sample objects
   |
   +--> ModelAdapter -----> preprocessor/model
                              |
                              v
                       FeatureExtractor
                              |
                              v
                     Disk Embedding Cache
                              |
              +---------------+----------------+
              |               |                |
              v               v                v
         Linear probe        k-NN             CKA
              |               |                |
              +---------------+----------------+
                              |
                              v
                       Analysis artifacts
                              |
                              v
                         HTML report
```

The most important architectural rule is that analysis code reads cached embeddings and does not depend on a live model object.

### 6.2 Shared data types

Use typed dataclasses or Pydantic models for public data structures.

Minimum types:

```python
@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: Any
    target: int
    metadata: dict[str, Any]

@dataclass(frozen=True)
class LayerFeatures:
    layer_name: str
    layer_index: int
    cls: torch.Tensor | None
    mean_patch: torch.Tensor | None

@dataclass(frozen=True)
class DatasetInfo:
    name: str
    task_type: str
    class_names: list[str]
    split_sizes: dict[str, int]

@dataclass(frozen=True)
class ModelInfo:
    identifier: str
    revision: str | None
    architecture: str
    num_layers: int
    hidden_size: int | None
    patch_size: int | tuple[int, int] | None = None
    num_register_tokens: int = 0
```

Do not pass unstructured dictionaries between major modules when a stable schema can be defined.

---

## 7. Model adapter contract

Create an abstract `ModelAdapter`.

Conceptual API:

```python
class ModelAdapter(ABC):
    @property
    def info(self) -> ModelInfo: ...

    @property
    def layer_names(self) -> list[str]: ...

    def preprocess(self, images: list[Any]) -> dict[str, torch.Tensor]: ...

    @torch.inference_mode()
    def extract_layers(
        self,
        model_inputs: dict[str, torch.Tensor],
    ) -> list[LayerFeatures]: ...
```

### v0.1 adapters

#### `HFHiddenStatesAdapter`

Purpose:
- generic best-effort support for Hugging Face vision models whose forward pass supports `output_hidden_states=True`;
- assumes a CLS-like global token at token index 0 unless overridden;
- should fail with an actionable error if the output shape is incompatible.

#### `DinoV2Adapter`

Purpose:
- tested adapter for DINOv2 and RAD-DINO;
- owns DINO-specific token semantics;
- excludes the embedding output from "transformer blocks" by default;
- exposes encoder blocks dynamically from model config rather than hard-coding layer count;
- supports:
  - `pooling: cls`
  - `pooling: mean_patch`
  - `pooling: both`

For Hugging Face DINOv2, hidden states are available from the bare model. Use `output_hidden_states=True` and `return_dict=True`.

#### `DinoV3Adapter`

Purpose:
- first-class v0.1 support for DINOv3 Vision Transformer checkpoints;
- first tested checkpoint: `facebook/dinov3-vitb16-pretrain-lvd1689m`;
- load through Hugging Face Transformers `AutoImageProcessor` + `AutoModel`;
- require a Transformers version with official DINOv3 support (`transformers>=4.56.0`);
- infer `num_hidden_layers`, `patch_size`, and `num_register_tokens` from model config;
- understand DINOv3 token order as `[CLS] + [register tokens] + [patch tokens]`;
- CLS pooling uses token index `0`;
- mean-patch pooling starts at index `1 + num_register_tokens`, so register tokens are never averaged into patch features;
- exposes the same `cls`, `mean_patch`, and `both` pooling choices as DINOv2.

DINOv3 support is important in v0.1 because one of the project's most useful comparisons is whether a newer general-domain foundation model provides stronger medical representations than both DINOv2 and a domain-specific DINOv2 derivative such as RAD-DINO.

**Access constraint:** official Meta DINOv3 checkpoints on Hugging Face are gated. Users must accept the DINOv3 license/access conditions before downloading them. Therefore DINOv3 must be supported and tested, but it must not be required for offline CI or for the ungated quick-start smoke test.

### Important implementation detail

Hugging Face hidden-state tuples commonly contain:

```text
embedding output + block 0 + block 1 + ... + block N-1
```

Do not label the embedding output as transformer block 0. Store it only if:

```yaml
extraction:
  include_embedding_layer: true
```

Default: `false`.

### Failure behavior

Never silently return a wrong token slice.

If a model uses register tokens or nonstandard special tokens and the adapter does not know how to identify patch tokens, raise a clear exception such as:

```text
PatchTokenLayoutError:
Model exposes nonstandard special tokens. CLS extraction is available,
but mean-patch extraction requires an explicit adapter.
```

Register-token-aware DINOv2 support remains a v1.0 item. DINOv3 register-token handling is required in v0.1 because the official DINOv3 ViT checkpoints use register tokens.

---

## 8. Dataset adapter contract

Conceptual API:

```python
class DatasetAdapter(ABC):
    @property
    def info(self) -> DatasetInfo: ...

    def get_split(self, split: Literal["train", "val", "test"]) -> Dataset: ...
```

Each yielded item should resolve to a `Sample`.

### 8.1 `MedMNISTAdapter`

v0.1 supports 2D single-label MedMNIST datasets.

Configuration example:

```yaml
dataset:
  type: medmnist
  name: pneumoniamnist
  size: 224
  as_rgb: true
  download: true
```

Requirements:

- preserve official train/validation/test splits;
- `size: 224` for the demo;
- convert grayscale images to RGB when required by the model processor;
- expose human-readable class names where available;
- reject unsupported multi-label tasks in v0.1 with an explicit error.

### 8.2 `CSVImageDatasetAdapter`

Generic layout:

```csv
path,label,patient_id,site,scanner
images/img001.png,0,p001,A,S1
images/img002.png,1,p002,A,S2
```

v0.1 uses only `path` and `label` for probing but preserves other columns as metadata for future v1.0 nuisance analysis.

Configuration:

```yaml
dataset:
  type: csv
  root: /data/my_dataset
  train_csv: splits/train.csv
  val_csv: splits/val.csv
  test_csv: splits/test.csv
  image_column: path
  label_column: label
  id_column: patient_id
  metadata_columns:
    - site
    - scanner
```

Validation:

- file exists;
- labels are consistent across splits;
- no duplicate `sample_id` across splits unless explicitly allowed;
- report class counts;
- warn about severe class imbalance;
- do not implement patient-level split generation in v0.1.

---

## 9. Configuration system

Use YAML + Pydantic v2.

Example:

```yaml
run:
  name: dinov2_base_pneumoniamnist
  output_dir: runs
  seed: 42

model:
  adapter: dinov2
  source: huggingface
  identifier: facebook/dinov2-base
  revision: null
  device: auto
  dtype: auto

dataset:
  type: medmnist
  name: pneumoniamnist
  size: 224
  as_rgb: true
  download: true

extraction:
  batch_size: 32
  num_workers: 4
  pooling:
    - cls
    - mean_patch
  splits:
    - train
    - val
    - test
  include_embedding_layer: false
  overwrite_cache: false

linear_probe:
  enabled: true
  standardize: true
  c_values: [0.01, 0.1, 1.0, 10.0]
  selection_metric: balanced_accuracy
  max_iter: 2000

knn:
  enabled: true
  k_values: [1, 5, 20]
  selection_metric: balanced_accuracy
  metric: cosine

analysis:
  pca:
    enabled: true
    n_components: 2
  cka:
    enabled: true
    max_samples: 5000
  neighbors:
    enabled: true
    k: 5
    num_anchors: 12

report:
  enabled: true
  include_thumbnails: true
  self_contained: true
```

Rules:

- Config validation must happen before model download.
- Save the fully resolved config to the run directory.
- CLI flags may override a small number of operational fields such as device and output directory.
- Do not build a large Hydra configuration hierarchy in v0.1. Pydantic + YAML is sufficient.

---

## 10. Feature extraction

### 10.1 Global representations

For every requested transformer block, extract:

1. **CLS representation**
   ```python
   hidden_state[:, 0, :]
   ```

2. **Mean patch representation**
   Mean over patch tokens only, excluding known special tokens.

Store each pooling type separately.

### 10.2 Device behavior

Accepted config values:

```text
auto
cpu
cuda
cuda:0
mps
```

`auto` priority:

```text
CUDA -> MPS -> CPU
```

Do not make CUDA mandatory.

Use `torch.inference_mode()`.

Mixed precision:
- use a safe inference dtype when configured;
- convert embeddings to float32 before writing unless the cache config explicitly requests another dtype.

### 10.3 Determinism

Seed:
- Python `random`;
- NumPy;
- PyTorch.

Record:
- seed;
- torch version;
- transformers version;
- device;
- dtype;
- model identifier;
- model revision/commit hash when available.

### 10.4 Cache format

Prefer simple, inspectable formats over a custom database.

Recommended run structure:

```text
runs/<run_name>/
├── config.resolved.yaml
├── manifest.json
├── dataset.json
├── model.json
├── samples.parquet
├── embeddings/
│   ├── train/
│   │   ├── layer_00_cls.npy
│   │   ├── layer_00_mean_patch.npy
│   │   └── ...
│   ├── val/
│   └── test/
├── metrics/
│   ├── linear_probe.csv
│   ├── knn.csv
│   ├── cka.npy
│   └── summary.json
├── figures/
│   ├── linear_probe.html
│   ├── knn.html
│   ├── pca_layer_00.html
│   ├── pca_layer_11.html
│   └── cka.html
├── neighbors/
│   └── neighbors.json
└── report/
    └── index.html
```

Use `.npy` so arrays can be memory-mapped.

`samples.parquet` must preserve the row order corresponding to every embedding matrix.

### 10.5 Cache identity

A cache must not be reused if any representation-affecting input changes.

Cache key should include at least:

- model identifier;
- model revision if known;
- adapter name/version;
- dataset identity;
- split;
- image preprocessing config;
- image size;
- pooling method;
- layer set.

Store the hash inputs in `manifest.json` so cache behavior is debuggable.

---

## 11. Linear probe

Use scikit-learn.

### Pipeline

For each layer and pooling method:

```text
train embeddings
    -> optional StandardScaler
    -> LogisticRegression
    -> choose C using validation split only
    -> evaluate chosen model on test split
```

Default `C` grid:

```python
[0.01, 0.1, 1.0, 10.0]
```

Default selection metric:

```text
balanced_accuracy
```

Store train/validation/test results separately.

### Metrics

For binary and multiclass single-label tasks:

- accuracy;
- balanced accuracy;
- macro F1;
- AUROC when defined;
- validation-selected `C`.

For multiclass AUROC use macro one-vs-rest when probabilities are available.

Never hide undefined metrics. Store `NaN` plus an explanatory warning.

### Output schema

`metrics/linear_probe.csv`:

```text
layer_index
layer_name
pooling
split
accuracy
balanced_accuracy
macro_f1
auroc
selected_c
n_samples
```

---

## 12. k-NN probe

Use normalized embeddings and cosine distance.

For each layer/pooling:

1. fit on train embeddings;
2. choose `k` from `[1, 5, 20]` using validation data;
3. evaluate on test data.

Output:

`metrics/knn.csv`

Columns:

```text
layer_index
layer_name
pooling
split
k
accuracy
balanced_accuracy
macro_f1
```

For v0.1 use scikit-learn. Do not require FAISS.

---

## 13. CKA representation similarity

Implement **linear CKA** as a first-class metric.

Required modes:

### A. Within-model CKA

Compare all selected layers of one model:

```text
layer x layer -> CKA matrix
```

This reveals representational stages and redundant blocks.

### B. Cross-model CKA

Given two compatible runs evaluated on the same samples:

```text
model A layers x model B layers -> CKA matrix
```

This is central to the DINOv2-base vs RAD-DINO demo.

### Requirements

- require aligned sample IDs;
- fail loudly if sample ordering cannot be reconciled;
- allow deterministic subsampling with `max_samples`;
- test invariance properties on synthetic matrices;
- document the exact formula in `docs/methodology.md`.

Store:
- raw matrix as `.npy`;
- row/column layer labels;
- Plotly heatmap.

Do not claim CKA proves semantic equivalence. Describe it as representation similarity.

---

## 14. Representation geometry

### PCA

PCA is mandatory in v0.1.

Generate interactive Plotly scatter plots for:
- first transformer block;
- middle block;
- final block;
- best linear-probe block.

Color by class label.

Avoid rendering every layer into the default report because that becomes visually noisy. The raw analysis API may support arbitrary layers.

### UMAP

UMAP is optional and should be an optional dependency:

```text
uv add "medvfm[umap]"
```

Do not make UMAP necessary for the default install or CI.

---

## 15. Nearest-neighbor retrieval

For a small set of deterministic anchors:

- select examples using the configured seed;
- compute cosine-nearest neighbors;
- show anchor + top-k neighbors;
- display target labels;
- indicate whether each neighbor label matches the anchor.

Default representation:
- best linear-probe layer using CLS pooling;
- fall back to final-layer CLS if probes are disabled.

The purpose is qualitative representation inspection, not image retrieval benchmarking.

For private/custom datasets:

```yaml
report:
  include_thumbnails: false
```

must prevent images from being embedded into the report.

---

## 16. Model comparison

CLI:

```bash
uv run medvfm compare RUN_A RUN_B
```

Requirements:

- no new forward pass if compatible cached embeddings exist;
- validate dataset/sample compatibility;
- compute cross-model CKA;
- compare layer-wise probe curves;
- compare best-layer performance;
- compare final-layer performance;
- generate `comparison/report/index.html`.

Example summary:

```json
{
  "model_a": "facebook/dinov2-base",
  "model_b": "microsoft/rad-dino",
  "dataset": "pneumoniamnist-224",
  "pooling": "cls",
  "model_a_best_layer": 8,
  "model_b_best_layer": 10,
  "model_a_best_balanced_accuracy": 0.81,
  "model_b_best_balanced_accuracy": 0.86,
  "cross_model_cka_mean": 0.63
}
```

The numbers above are examples only. Never hard-code expected scientific results.

---

## 17. HTML report

v0.1 should generate a self-contained static HTML report using Jinja2 + Plotly.

### Required sections

1. **Run overview**
   - model;
   - dataset;
   - seed;
   - device;
   - date;
   - package versions.

2. **Dataset summary**
   - task;
   - class names;
   - split sizes;
   - class distribution.

3. **Model summary**
   - architecture;
   - hidden size;
   - transformer blocks;
   - pooling strategies.

4. **Layer-wise target information**
   - linear-probe curve;
   - k-NN curve;
   - best layer;
   - final layer;
   - best-vs-final delta.

5. **Representation similarity**
   - within-model CKA.

6. **Embedding geometry**
   - selected PCA plots.

7. **Nearest neighbors**
   - qualitative examples if allowed.

8. **Machine-readable fingerprint**
   - link/embedded JSON summary.

### Report summary language

Use deterministic templates, not an LLM.

Example:

```text
The highest validation balanced accuracy was obtained at block 8.
The final block was 3.2 percentage points lower.
Blocks 8-10 formed the most similar late-layer cluster by linear CKA.
```

Only make statements directly supported by computed metrics.

---

## 18. Model fingerprint

Create `metrics/summary.json`.

v0.1 schema:

```json
{
  "schema_version": "0.1",
  "model": {},
  "dataset": {},
  "representation": {
    "pooling": "cls",
    "best_linear_probe_layer": 8,
    "best_linear_probe_metric": 0.0,
    "final_layer_metric": 0.0,
    "best_knn_layer": 7,
    "best_knn_metric": 0.0
  },
  "cka": {
    "mean_off_diagonal": 0.0
  }
}
```

The goal is to make fingerprints easy to compare programmatically later.

---

## 19. CLI

Use Typer.

### Commands required in v0.1

```bash
uv run medvfm inspect -c CONFIG
uv run medvfm compare RUN_A RUN_B
uv run medvfm report RUN_DIR
uv run medvfm version
```

### `inspect`

Stages:

```text
validate config
-> initialize run
-> load dataset
-> load model
-> extract/cache embeddings
-> run probes
-> run analyses
-> build report
```

Print a concise final summary:

```text
Run complete: runs/rad_dino_pneumoniamnist
Best linear-probe block: 10
Balanced accuracy: 0.xxx
Report: runs/rad_dino_pneumoniamnist/report/index.html
```

### Resume behavior

If valid embeddings already exist:
- skip extraction;
- print that cache is being reused;
- rerun downstream analyses when requested.

No silent overwriting.

---

## 20. v0.1 demo strategy

There should be two levels of demo.

### 20.1 Fast smoke demo

**Model**
- `facebook/dinov2-small`

**Dataset**
- `PneumoniaMNIST`, size 224, RGB

Purpose:
- minimal download;
- validates installation and end-to-end behavior;
- suitable for a README quick start.

Command:

```bash
uv run medvfm inspect -c configs/demo/dinov2_small_pneumoniamnist.yaml
```

Allow an optional config field to cap samples for a smoke run:

```yaml
dataset:
  max_samples_per_split: 1000
```

Sampling must be deterministic and stratified.

### 20.2 Portfolio/scientific demo

**Models**
- `facebook/dinov2-base`
- `facebook/dinov3-vitb16-pretrain-lvd1689m`
- `microsoft/rad-dino`

**Dataset**
- `PneumoniaMNIST`, size 224, RGB

Why this three-way comparison:
- DINOv2-base provides the established general-domain baseline;
- DINOv3 ViT-B/16 is a newer general-domain foundation model of roughly comparable model scale and is especially interesting because practical medical-imaging experience suggests it may transfer better;
- RAD-DINO is a chest-X-ray-specific DINOv2-derived model;
- together, the comparison asks a stronger question than a simple domain-adaptation demo: **does a newer generic foundation model outperform an older generic model and/or a medically specialized model, and at which layers?**

Primary comparisons:

```text
DINOv2-base <-> DINOv3 ViT-B/16   # foundation-model generation comparison
DINOv2-base <-> RAD-DINO          # general vs medical domain adaptation
DINOv3 ViT-B/16 <-> RAD-DINO      # newer generic vs medical-specialized
```

Scientific interpretation:
- do not frame DINOv3 vs RAD-DINO as a controlled adaptation experiment because they differ in pretraining recipe, data, architecture details, and patch size;
- compare layer-wise downstream decodability, best-layer location, final-layer quality, CKA geometry, and nearest-neighbor behavior;
- treat the three models as representation families, not as identical-training-condition baselines.

Deliverables:

```text
assets/demo/dinov2_dinov3_raddino_probe_curve.png
assets/demo/dinov2_vs_dinov3_cka.png
assets/demo/dinov2_vs_raddino_cka.png
assets/demo/dinov3_vs_raddino_cka.png
assets/demo/report_preview.png
```

DINOv3 access note:
- keep DINOv2-small as the ungated quick-start demo;
- document how to authenticate with Hugging Face and accept the DINOv3 checkpoint terms;
- if DINOv3 credentials are unavailable, the rest of the v0.1 demo must still run.

Do not commit embedding caches.

### 20.3 Optional second v0.1 demo

**Dataset**
- `PathMNIST`, size 224

**Models**
- `facebook/dinov2-base`
- `facebook/dinov3-vitb16-pretrain-lvd1689m`

Purpose:
- prove the tool is not chest-X-ray-specific;
- exercise multiclass metrics;
- test whether the DINOv3 advantage, if observed on PneumoniaMNIST, persists in a different medical-imaging domain;
- provide a cleaner DINOv2-vs-DINOv3 comparison without a chest-X-ray-specialized model in the mix.

This is optional for v0.1 release, but desirable. Keep it out of blocking CI because the DINOv3 checkpoint is gated.

---

## 21. Testing strategy

### 21.1 Unit tests

Required:

- CKA known/invariance cases;
- pooling excludes correct special tokens;
- DINOv3 pooling excludes all register tokens and retains all patch tokens;
- DINOv3 adapter derives `num_register_tokens` from config rather than assuming a constant;
- cache invalidation;
- config validation;
- classification metrics;
- sample alignment;
- deterministic subsampling;
- run manifest serialization.

### 21.2 Integration tests

CI must not depend on downloading large public checkpoints.

Create a tiny local Transformers ViT during the test:

1. instantiate a very small random ViT config;
2. save it with `save_pretrained()` to a temp directory;
3. load it through the same adapter path used for real models;
4. create a tiny synthetic image dataset;
5. run the complete pipeline;
6. assert expected output files and schemas exist.

This tests the real adapter/CLI without network dependency.

### 21.3 Optional slow tests

Mark external-download tests:

```python
@pytest.mark.slow
```

They may test:
- DINOv2-small;
- a tiny MedMNIST subset.

Do not run them in normal pull-request CI.

---

## 22. Engineering standards

### Tooling

Recommended:

- Python 3.10+
- `uv` for environment management and lock file; run project commands through `uv run`
- `hatchling` or a similarly lightweight PEP 517 build backend
- PyTorch
- Transformers `>=4.56.0` (required for released-package DINOv3 support)
- MedMNIST
- NumPy
- pandas
- PyArrow
- scikit-learn
- Pillow
- Plotly
- Jinja2
- Typer
- Pydantic v2
- PyYAML
- Rich
- pytest
- Ruff
- mypy

Optional:
- `umap-learn`

Avoid adding dependencies without a concrete feature need.

### Style

- complete type hints for public APIs;
- docstrings for public classes/functions;
- Ruff formatting/linting;
- no notebook-only implementation;
- no global mutable registries that make tests order-dependent;
- actionable exceptions rather than bare `assert`;
- no `except Exception: pass`;
- no hidden network calls after artifacts are cached.

### GitHub Actions

`ci.yml`:
- install locked environment with `uv sync --locked --dev`;
- `uv run ruff check .`;
- `uv run ruff format --check .`;
- `uv run mypy src/medvfm_inspector`;
- `uv run pytest -m "not slow"`;
- `uv build`.

`demo-smoke.yml`:
- manual workflow dispatch or scheduled;
- may run the small public demo;
- should not block ordinary PRs if external services are unavailable.

---

## 23. README requirements for v0.1

The README is part of the product.

Top section should immediately show:

1. one-sentence value proposition;
2. report screenshot;
3. three core questions the project answers;
4. install;
5. five-line quick start;
6. supported models/datasets;
7. example output;
8. architecture graphic;
9. how to add a model/dataset adapter;
10. methodology caveats.

Suggested opening:

```markdown
# MedVFM-Inspector

A model-agnostic microscope for probing, comparing, and auditing
representations learned by vision foundation models, with first-class
support for medical imaging.

**Where is task information encoded? Is the final layer actually best?
What changed after domain adaptation or fine-tuning?**
```

Do not market it as a benchmark unless a benchmark protocol is actually added later.

---

## 24. v0.1 implementation phases

Do these in order.

### Phase 0 - Scaffold

Deliver:
- repository;
- `pyproject.toml`;
- `src/` package;
- Typer CLI;
- Pydantic config loading;
- logging;
- test setup;
- Ruff/mypy/pytest CI.

Acceptance:
```bash
uv sync --dev
uv run medvfm version
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/medvfm_inspector
```

### Phase 1 - Dataset and model adapters

Deliver:
- base interfaces;
- MedMNIST adapter;
- CSV adapter;
- HF hidden-state adapter;
- DINOv2 adapter;
- DINOv3 ViT adapter with register-token-aware patch pooling.

Acceptance:
- tiny local HF model integration test passes;
- MedMNIST adapter unit tests work without requiring full demo download.

### Phase 2 - Extraction and caching

Deliver:
- CLS extraction;
- mean-patch extraction;
- `.npy` cache;
- `samples.parquet`;
- manifest/hash;
- resume behavior.

Acceptance:
- second identical run performs no model forward pass;
- changing model revision/pooling/image size invalidates relevant cache.

### Phase 3 - Probes

Deliver:
- linear probe;
- validation-only C selection;
- k-NN;
- metrics CSVs.

Acceptance:
- synthetic linearly separable dataset achieves expected high performance;
- test labels never influence hyperparameter selection.

### Phase 4 - Representation analysis

Deliver:
- linear CKA;
- PCA;
- nearest neighbors.

Acceptance:
- CKA tests pass;
- cross-run sample alignment checks work.

### Phase 5 - Report

Deliver:
- Jinja2 report;
- Plotly figures;
- summary JSON.

Acceptance:
- report opens without a running Python server;
- no broken local absolute paths;
- report respects `include_thumbnails: false`.

### Phase 6 - Comparison workflow

Deliver:
- `uv run medvfm compare`;
- cross-model CKA;
- probe-curve comparison;
- comparison report.

Acceptance:
- works with two compatible synthetic runs;
- fails clearly on incompatible datasets/sample sets.

### Phase 7 - Public demo and release

Deliver:
- DINOv2-small ungated smoke config;
- DINOv2-base vs DINOv3 ViT-B/16 vs RAD-DINO portfolio demo;
- README images;
- DINOv3 access/authentication note;
- documentation;
- `v0.1.0` release.

---

## 25. Definition of done for v0.1.0

All of the following must be true:

- [ ] Package installs from a clean checkout.
- [ ] `uv run medvfm version` works.
- [ ] Unit tests pass on CPU.
- [ ] Integration pipeline test passes without network.
- [ ] Ruff passes.
- [ ] mypy passes on the package.
- [ ] DINOv2-small + PneumoniaMNIST-224 smoke demo runs.
- [ ] DINOv2-base + PneumoniaMNIST-224 runs.
- [ ] DINOv3 ViT-B/16 + PneumoniaMNIST-224 runs when checkpoint access is available.
- [ ] DINOv3 mean-patch pooling excludes register tokens correctly.
- [ ] RAD-DINO + PneumoniaMNIST-224 runs.
- [ ] Model comparison runs without re-extraction.
- [ ] Linear probe is computed at every transformer block.
- [ ] k-NN is computed at every transformer block.
- [ ] CKA matrix is generated.
- [ ] PCA plots are generated.
- [ ] Nearest-neighbor examples are generated.
- [ ] Report is self-contained.
- [ ] `summary.json` follows a documented schema.
- [ ] Resolved config and reproducibility manifest are saved.
- [ ] README contains a real report screenshot.
- [ ] No large weights or embeddings are committed.
- [ ] Public API and methodology are documented.
- [ ] License and contribution guide exist.

---

# PART II - v1.0.0

## 26. v1.0 goal

v1.0 should turn the v0.1 representation workbench into a general **foundation-model auditing and comparison platform**.

The core conceptual extension is:

```text
v0.1:
Where is target information encoded?

v1.0:
What information is encoded, where is it encoded,
how does it change after adaptation, and is any of it undesirable?
```

---

## 27. v1.0 feature set

### 27.1 Broader model ecosystem

Add tested adapters/support for:

- broader Hugging Face ViT-style encoders;
- DINOv2 with register tokens;
- broader DINOv3 variants, including optional ConvNeXt support where layer semantics can be defined cleanly;
- `timm`;
- OpenCLIP;
- BiomedCLIP visual encoder.

v1.0 demo model set:

- `facebook/dinov2-base`
- `facebook/dinov3-vitb16-pretrain-lvd1689m`
- `microsoft/rad-dino`
- `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- one DINOv2-with-registers checkpoint to test generalized special-token handling

Architecture rule:
- adapters own token semantics;
- the analysis layer never assumes token 0 is CLS unless the adapter declares it.

### 27.2 Nuisance-variable probing

This is one of the signature v1.0 features.

Given dataset metadata:

```yaml
metadata_columns:
  - site
  - scanner
  - sex
  - age
```

Run a probe at every layer for each metadata variable.

Supported variable types:

- binary categorical;
- multiclass categorical;
- continuous.

Methods:

```text
categorical -> logistic regression
continuous  -> ridge regression
```

Metrics:

```text
categorical:
- balanced accuracy
- macro F1
- AUROC when defined

continuous:
- R2
- MAE
```

Report should display target and nuisance curves together.

Example fingerprint:

```text
Target: pneumonia
  best target block: 10
  final target balanced accuracy: 0.xx

Nuisance: site
  strongest site block: 11
  site balanced accuracy: 0.xx
  warning: site information rises strongly in late layers
```

Do not label something a "shortcut" solely because it is predictable. Use terms such as "nuisance-variable predictability" unless the experiment establishes shortcut behavior.

### 27.3 Group and OOD evaluation

Support a group column:

```yaml
groups:
  column: site
```

Features:

- per-group downstream metrics;
- worst-group accuracy/balanced accuracy;
- train-on-source/evaluate-on-target;
- leave-one-group-out evaluation;
- layer-wise OOD probe curves.

This makes the tool relevant to domain shift and multi-center medical imaging.

### 27.4 Checkpoint/fine-tuning drift

Compare:

```text
pretrained
vs
linear-head training
vs
LoRA checkpoint
vs
full fine-tuned checkpoint
```

MedVFM-Inspector does not need to perform the training.

Compute:

- cross-checkpoint CKA;
- per-layer average cosine similarity;
- per-sample embedding shift;
- effective rank;
- neighborhood preservation;
- best-layer migration;
- target-predictability delta;
- nuisance-predictability delta.

Example report statement:

```text
Full fine-tuning increased target separability by X points but reduced
cross-checkpoint CKA most strongly in blocks 8-11.
Site predictability also increased in blocks 9-11.
```

Again, only generate this statement if directly supported by computed values.

### 27.5 Patch-token inspection

Add spatial representation inspection for ViT-like models.

Minimum features:

- patch-token norm maps;
- CLS-to-patch cosine-similarity maps;
- patch PCA projections;
- patch nearest-neighbor retrieval across images.

Optional where architecture supports it:
- attention rollout.

Never present patch maps as validated clinical explanations. Call them representation visualizations.

### 27.6 Multi-label tasks

Add:
- multi-label linear probing;
- macro/micro AUROC;
- average precision;
- per-label curves.

This enables ChestMNIST and more realistic chest-X-ray tasks.

### 27.7 Interactive explorer

Add an optional local UI:

```bash
uv add "medvfm[dashboard]"
uv run medvfm dashboard RUN_DIR
```

Preferred implementation:
- Streamlit, unless a lighter solution is clearly superior at implementation time.

Views:

- model/dataset overview;
- layer selector;
- target probe curve;
- nuisance probe curves;
- CKA heatmap;
- PCA/UMAP;
- sample browser;
- nearest neighbors;
- patch-token explorer;
- checkpoint comparison.

The static HTML report must remain available. Do not require the dashboard for reproducibility.

### 27.8 Plugin registry

External packages should be able to register:

- model adapters;
- dataset adapters;
- probes;
- report panels.

Use Python entry points rather than editing a central hard-coded registry.

Example conceptual entry points:

```toml
[project.entry-points."medvfm.models"]
my_model = "my_package:MyModelAdapter"

[project.entry-points."medvfm.datasets"]
my_dataset = "my_package:MyDatasetAdapter"
```

### 27.9 Scaling

For larger datasets:

- chunked extraction;
- memory-mapped arrays;
- configurable sample caps for expensive analyses;
- deterministic CKA subsampling;
- optional FAISS for large nearest-neighbor search;
- optional mixed-precision cache storage.

Multi-GPU/distributed extraction may be added only if profiling shows it is necessary.

---

## 28. v1.0 demo suite

### Demo A - Foundation-model generation and domain specialization

**Dataset**
- PneumoniaMNIST-224

**Models**
- DINOv2-base
- DINOv3 ViT-B/16
- RAD-DINO

Questions:
- Which layers transfer best for each model?
- Does DINOv3 improve medical downstream separability relative to DINOv2?
- At which layers does DINOv3 diverge most strongly from DINOv2?
- How does the DINOv2 -> RAD-DINO domain adaptation alter layer geometry?
- Can a newer generic foundation model match or exceed the medically specialized encoder?
- Does the best downstream layer move across model families?

Interpret DINOv2 vs RAD-DINO as the adaptation-oriented comparison. Interpret DINOv3 comparisons as cross-family representation comparisons rather than controlled adaptation experiments.

### Demo B - Cross-family biomedical model

**Dataset**
- PathMNIST-224

**Models**
- DINOv2-base
- BiomedCLIP visual encoder

Questions:
- Does biomedical vision-language pretraining change layer-wise class separability?
- How different is representation geometry across model families?

Do not imply the models had identical pretraining data or objectives. This is an inspection demo, not necessarily a controlled scientific comparison.

### Demo C - Synthetic shortcut/site audit

Create a reproducible wrapper around a public medical dataset, preferably PneumoniaMNIST.

Generate a synthetic nuisance variable such as `site` and apply a controlled image transformation correlated with the site/label only under specified splits.

Possible transformations:
- corner marker;
- brightness/window shift;
- border artifact.

The wrapper must record the synthetic nuisance metadata.

Purpose:
- demonstrate nuisance probing;
- demonstrate group/OOD evaluation;
- demonstrate that the tool can detect where nuisance information becomes linearly decodable.

Clearly label this as a synthetic stress test, not a real hospital-site experiment.

---

## 29. v1.0 fingerprint schema

Example:

```json
{
  "schema_version": "1.0",
  "model": {},
  "dataset": {},
  "target": {
    "name": "pneumonia",
    "best_layer": 10,
    "best_metric": 0.0,
    "final_layer_metric": 0.0
  },
  "nuisance_variables": {
    "site": {
      "variable_type": "categorical",
      "most_predictive_layer": 11,
      "best_metric": 0.0
    }
  },
  "representation": {
    "effective_rank_by_layer": [],
    "within_model_cka_mean": 0.0
  },
  "comparison": {
    "reference_checkpoint": null,
    "cross_checkpoint_cka_by_layer": [],
    "neighborhood_preservation_by_layer": []
  },
  "ood": {
    "group_column": null,
    "worst_group_metric_by_layer": []
  }
}
```

Version the schema independently from package internals.

---

## 30. Scientific and methodological caveats

Document these prominently.

### Linear probe

A linear probe measures linear decodability under a particular sample size, preprocessing pipeline, regularization choice, and evaluation protocol. It does not prove that a model "uses" that information.

### k-NN

k-NN performance depends on metric, normalization, and k. It is a nonparametric view of local representation geometry, not a complete measure of representation quality.

### CKA

CKA measures representation similarity under its mathematical assumptions. High CKA does not imply identical semantics or identical downstream behavior.

### PCA/UMAP

2D projections can distort geometry. They are exploratory visualizations.

### Nuisance probes

Predictability of scanner/site metadata does not by itself establish a harmful shortcut. It identifies information encoded in the representation and motivates controlled follow-up experiments.

### Medical use

No report should contain wording suggesting clinical validation or diagnostic safety.

---

## 31. Non-goals through v1.0

Unless the project direction changes deliberately, do not turn MedVFM-Inspector into:

- a generic experiment tracker;
- a foundation-model training framework;
- a medical data lake;
- a DICOM viewer;
- a model-serving platform;
- a clinical decision-support system;
- a leaderboard site;
- an explainability library covering every method;
- a replacement for MONAI, FiftyOne, or Hugging Face.

Integrate with existing ecosystems where useful rather than cloning them.

---

## 32. Suggested public roadmap

```text
v0.1 - Layer microscope
  DINOv2 + DINOv3 ViT + RAD-DINO + MedMNIST
  hidden states + register-aware pooling + linear probe + kNN + CKA + PCA + report

v0.2 - More backbones
  timm + OpenCLIP/BiomedCLIP + stronger adapter API

v0.3 - Fine-tuning drift
  base vs adapted checkpoint analysis

v0.4 - Metadata audit
  nuisance-variable probing + group metrics

v0.5 - Spatial inspection
  patch-token views

v0.6 - Interactive explorer
  local dashboard

v1.0 - Stable auditing platform
  stable schemas + plugin API + documented methodology + demo suite
```

This roadmap may be compressed, but v0.1 should remain intentionally narrow.

---

# PART III - CODING-AGENT INSTRUCTIONS

## 33. Rules for Codex or another coding agent

When implementing from this specification:

1. Read this entire file before editing code.
2. Implement v0.1 before any v1.0-only feature.
3. Prefer a working vertical slice over many empty abstractions.
4. Add tests with every core module.
5. Run relevant tests after each implementation step.
6. Do not leave placeholder functions in the main execution path.
7. Do not silently catch model/dataset incompatibilities.
8. Do not tune on test data.
9. Do not download models or datasets during unit tests.
10. Do not commit model weights, datasets, embedding caches, or generated `runs/`.
11. Keep expensive computation resumable.
12. Keep extraction independent from analysis.
13. Preserve sample IDs and row ordering explicitly.
14. Use deterministic seeds.
15. Record enough metadata to reproduce a run.
16. If implementation reality conflicts with this spec, document the conflict in an issue/ADR rather than hiding it.
17. Avoid premature plugin systems before the basic adapter interfaces are proven.
18. Keep public APIs typed and documented.
19. Make error messages tell the user how to fix the problem.
20. Treat report numbers as scientific outputs: never fabricate or hard-code them.

---

## 34. First Codex task

The first implementation task should be limited to **Phase 0 + the interfaces needed for Phase 1**.

Suggested task:

```text
Read MEDVFM_INSPECTOR_SPEC.md.

Implement Phase 0 of v0.1:
- create the src-layout Python package;
- create pyproject.toml, uv dependency groups, and uv setup;
- implement `medvfm version`;
- implement Pydantic YAML config loading with the v0.1 config sections;
- add structured logging;
- create abstract ModelAdapter and DatasetAdapter interfaces;
- create shared typed data structures;
- configure Ruff, mypy, pytest, and GitHub Actions;
- add unit tests for config validation and CLI version;
- add AGENTS.md pointing coding agents back to this specification.

Do not implement feature extraction, probes, CKA, reports, or v1.0 features yet.

Before finishing, run:
- `uv sync --dev`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy src/medvfm_inspector`
- `uv run pytest`

Report what was implemented, tests run, and any deviations from the spec.
```

After Phase 0 is clean, give Codex Phase 1 from Section 24 rather than asking it to implement the entire project in a single pass.

---

## 35. External references verified for this plan

These are implementation/reference links, not dependencies that should be copied blindly.

### Models

DINOv2 small:
- https://huggingface.co/facebook/dinov2-small

DINOv2 base:
- https://huggingface.co/facebook/dinov2-base

Hugging Face DINOv2 documentation:
- https://huggingface.co/docs/transformers/model_doc/dinov2

DINOv3 ViT-B/16 (LVD-1689M):
- https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

Hugging Face DINOv3 documentation:
- https://huggingface.co/docs/transformers/model_doc/dinov3

Official DINOv3 repository:
- https://github.com/facebookresearch/dinov3

DINOv3 implementation notes:
- Transformers support is available in released versions starting with `transformers>=4.56.0`.
- Official Hugging Face checkpoints are gated and require accepting Meta's DINOv3 access/license conditions.
- DINOv3 ViT hidden states contain a CLS token, register tokens, and patch tokens; patch pooling must skip `1 + num_register_tokens` prefix tokens.

RAD-DINO:
- https://huggingface.co/microsoft/rad-dino

BiomedCLIP:
- https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

DINOv2 with registers:
- https://huggingface.co/facebook/dinov2-with-registers-small

### Datasets

MedMNIST repository:
- https://github.com/MedMNIST/MedMNIST

MedMNIST supports standardized 2D datasets and larger 64/128/224 variants suitable for foundation-model experiments. Use the official API/download path.

---

## 36. Final product test

Before calling any milestone "done", ask whether a new user can answer this without reading the source code:

> "I have a vision foundation model and a labeled image dataset. Which layers contain useful target information, what does the representation geometry look like, and how does this checkpoint differ from another one?"

For v0.1, the answer must be:

```text
Install MedVFM-Inspector, write one YAML file, run one command,
and open the generated report.
```

For v1.0, add:

> "Which layers also encode nuisance variables, and what changed after adaptation?"

That is the product.
