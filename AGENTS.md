# MedVFM-Inspector — Codex Instructions

## Project goal

MedVFM-Inspector is an open-source toolkit for inspecting and comparing representations learned by vision foundation models, with particular emphasis on medical imaging.

The full product specification and roadmap are in:

`docs/MEDVFM_INSPECTOR_SPEC.md`

Read that document before making architectural decisions.

## Development strategy

Do NOT attempt to implement the entire v0.1 specification at once.

Work incrementally, one milestone at a time.

For every task:

1. Inspect the existing repository before changing anything.
2. Identify the smallest implementation needed for the requested milestone.
3. Do not implement features from future milestones unless they are required for the current task.
4. Prefer simple, extensible abstractions over premature generalization.
5. Add or update tests for new functionality.
6. Run the relevant tests before finishing.
7. Run formatting/linting checks if configured.
8. Summarize:

   * files created or changed,
   * architectural decisions,
   * tests run,
   * anything intentionally deferred.

## Important constraints

* Python project.
* Use type hints for public interfaces.
* Prefer small composable modules.
* Avoid large monolithic classes.
* Model-specific behavior must live behind model adapters.
* Dataset-specific behavior must live behind dataset adapters.
* Analysis methods should consume standardized extracted representations rather than depend directly on a specific model.
* Avoid unnecessary dependencies.
* Do not silently download very large datasets or checkpoints in tests.
* Unit tests must use lightweight mocks or tiny synthetic inputs where possible.
* Keep CPU-only test execution possible.
* GPU support should be optional.
* Maintain reproducibility by exposing random seeds where relevant.

## Environment and commands

Use `uv` as the repository environment manager. Prefer `uv sync --dev` to
create or update the local environment, and run project commands through
`uv run`, for example `uv run pytest`, `uv run ruff check .`,
`uv run ruff format --check .`, and `uv run mypy src/medvfm_inspector`.
Do not document or assume direct global `pytest`, `ruff`, `mypy`, or `python`
commands when a `uv run ...` equivalent is available.

## Scope control

The roadmap in `docs/MEDVFM_INSPECTOR_SPEC.md` describes the eventual project, not the scope of every task.

Only implement the milestone explicitly requested in the current Codex prompt.

If a feature belongs to a later milestone, leave a clean extension point rather than implementing it early.

## Quality bar

Before considering a milestone complete:

* tests for the new functionality pass,
* existing tests continue to pass,
* public APIs have concise docstrings,
* imports are clean,
* no dead experimental code remains,
* documentation reflects user-facing behavior.

Do not make unrelated refactors while implementing a milestone.

## Engineering principles

Prioritize simplicity, readability, reproducibility, and efficiency.

### Avoid overengineering

* Implement only what the current milestone requires.
* Prefer the simplest design that cleanly solves the problem.
* Do not add abstractions, configuration, registries, factories, or dependencies without a concrete need.
* Prefer composition and small interfaces over complex inheritance.
* Do not refactor unrelated code.

### Code quality

* Prefer clear, explicit code over clever code.
* Keep functions and classes small and focused.
* Use type hints for public APIs.
* Avoid duplicated logic and unnecessary wrappers.
* Add dependencies only when they solve a real problem.

### ML efficiency

* Use `model.eval()` and `torch.inference_mode()` for extraction.
* Process data in batches.
* Avoid unnecessary CPU/GPU transfers.
* Do not reload models unnecessarily.
* Cache representations when appropriate.
* Do not keep the full dataset in memory unnecessarily.
* Optimize only when there is evidence of a bottleneck.

### Architecture

Keep the main flow:

`model + dataset → representation extraction → cached representations → analysis → report`

Analysis modules should work on standardized representations and should not reload models.

### Testing

* Add tests for new behavior.
* Keep unit tests small, deterministic, and CPU-compatible.
* Do not download large models or datasets in normal CI.

### Scope

Before adding a class, abstraction, dependency, or feature, ask:

> Is this necessary for the current milestone?

If not, do not add it.

Before finishing a task, run:

```bash
ruff check .
ruff format --check .
pytest
```

Then summarize:

* what changed,
* tests run,
* new dependencies,
* anything intentionally deferred.

