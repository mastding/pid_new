# PID V2 Skill/Provider Refactor Plan

## 1. Background

The current project already has a usable layering pattern:

- `backend/api`
  Protocol and transport layer.
- `backend/core/pipeline`
  End-to-end tuning orchestration.
- `backend/core/algorithms`
  Deterministic algorithm implementation.
- `backend/core/skills`
  LLM-facing skill abstraction and registry.

This structure is workable today, but it is not yet ideal for long-term extensibility because:

1. `pipeline` still depends directly on specific algorithm modules.
2. `skills` and `algorithms` are close in responsibility and can drift over time.
3. Replacing one algorithm with another currently requires touching orchestration code.
4. Rule sets such as loop priors, parameter constraints, and score caps are not yet first-class architecture objects.

The target direction is:

`pipeline` controls the flow, `skills` expose business capabilities, `providers` implement concrete algorithms, and `policies` host rules and priors.

## 2. Refactor Objectives

The refactor should achieve the following:

- Keep the main tuning workflow stable.
- Make core algorithms pluggable without rewriting the pipeline.
- Allow different providers per capability, loop type, or plant.
- Make skill definitions business-oriented instead of algorithm-oriented.
- Move heuristic rules and priors into dedicated policy modules.
- Preserve backward compatibility during migration.

## 3. Target Architecture

```text
backend/
  api/

  core/
    application/
      tuning_pipeline.py
      session_app_service.py

    skills/
      base.py
      registry.py
      common/
      window/
      dead_time/
      identification/
      tuning/
      evaluation/
      reporting/

    providers/
      common/
      window/
      dead_time/
      identification/
      tuning/
      evaluation/

    policies/
      loop_priors.py
      constraints.py
      scoring_rules.py
      model_selection_rules.py

    shared/
      dto.py
      enums.py
      provider_registry.py
      errors.py
```

## 4. Layer Responsibilities

### 4.1 `api`

Responsibilities:

- Receive uploaded files and user input.
- Return SSE or JSON responses.
- Convert request models into application-layer calls.

Should not:

- Contain algorithm logic.
- Choose algorithm implementations directly.

### 4.2 `application`

Responsibilities:

- Define use-case flows such as the tuning pipeline.
- Coordinate stage execution order and fallbacks.
- Publish stage events.
- Persist session history or execution traces.

Should not:

- Implement model fitting, scoring, or tuning formulas.
- Know the internals of any specific provider.

### 4.3 `skills`

Responsibilities:

- Expose business capabilities with stable input/output contracts.
- Validate inputs.
- Read and write shared execution context.
- Select and call providers.
- Aggregate multi-provider or multi-round outputs when needed.

Should not:

- Duplicate algorithm logic already owned by providers.
- Become a second orchestration layer competing with `application`.

### 4.4 `providers`

Responsibilities:

- Implement concrete algorithms.
- Stay deterministic and testable.
- Remain replaceable behind stable interfaces.

Examples:

- history-rule-based window detection
- cross-correlation dead-time estimation
- transfer-function fitting
- ARX identification
- IMC tuning
- Lambda tuning
- closed-loop simulation evaluation

### 4.5 `policies`

Responsibilities:

- Store configurable priors, constraints, and scoring rules.
- Centralize business heuristics.
- Prevent algorithm providers from embedding scattered rule logic.

Examples:

- loop-specific soft/hard time-constant priors
- tuning parameter physical constraints
- score cap rules
- preferred model families by loop type

## 5. Current-to-Target Mapping

### 5.1 Current algorithm modules

- `backend/core/algorithms/data_analysis.py`
  Split over time into:
  - `providers/common/dataset_loader.py`
  - `providers/window/history_rule_based.py`
- `backend/core/algorithms/system_id.py`
  First becomes:
  - `providers/identification/transfer_function_fit.py`
- `backend/core/algorithms/pid_tuning.py`
  Split over time into:
  - `providers/tuning/imc.py`
  - `providers/tuning/lambda_tuning.py`
  - `providers/tuning/zn.py`
  - `providers/tuning/chr.py`
- `backend/core/algorithms/pid_evaluation.py`
  First becomes:
  - `providers/evaluation/closed_loop_sim.py`

### 5.2 Current skill modules

The current `backend/core/skills` package already provides two important assets:

- a context-aware skill abstraction
- a registry mechanism

These should be preserved and evolved, not removed.

### 5.3 Current pipeline

`backend/core/pipeline/runner.py` should gradually stop calling:

- `load_and_prepare_dataset()`
- `fit_best_model()`
- `select_best_strategy()`
- `evaluate_pid_params()`

directly, and instead call skills such as:

- `load_dataset_skill`
- `detect_windows_skill`
- `identify_model_skill`
- `generate_tuning_candidates_skill`
- `evaluate_tuning_skill`

## 6. Provider Interface Drafts

### 6.1 Window detection provider

```python
class WindowDetectionProvider(Protocol):
    name: str

    def detect(
        self,
        df: pd.DataFrame,
        dt: float,
        loop_type: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...
```

Expected result shape:

```python
{
    "candidate_windows": [...],
    "step_events": [...],
    "provider": "history_rule_based",
    "meta": {...},
}
```

### 6.2 Dead-time estimation provider

```python
class DeadTimeEstimationProvider(Protocol):
    name: str

    def estimate(
        self,
        mv: np.ndarray,
        pv: np.ndarray,
        dt: float,
        loop_type: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...
```

Expected result shape:

```python
{
    "L": 12.5,
    "confidence": 0.78,
    "provider": "cross_correlation",
    "meta": {...},
}
```

### 6.3 Identification provider

```python
class IdentificationProvider(Protocol):
    name: str

    def identify(
        self,
        *,
        window: dict[str, Any],
        loop_type: str,
        model_pool: list[str] | None = None,
        constraints: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...
```

Expected result shape:

```python
{
    "best_model": {...},
    "attempts": [...],
    "provider": "transfer_function_fit",
    "meta": {...},
}
```

### 6.4 Tuning provider

```python
class TuningProvider(Protocol):
    name: str

    def tune(
        self,
        *,
        model: dict[str, Any],
        loop_type: str,
        constraints: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...
```

Expected result shape:

```python
{
    "candidates": [...],
    "recommended": {...},
    "provider": "lambda_tuning",
    "meta": {...},
}
```

### 6.5 Evaluation provider

```python
class EvaluationProvider(Protocol):
    name: str

    def evaluate(
        self,
        *,
        model: dict[str, Any],
        pid_params: dict[str, Any],
        loop_type: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...
```

Expected result shape:

```python
{
    "performance_score": 7.6,
    "final_rating": 7.8,
    "readiness_score": 7.4,
    "details": {...},
    "provider": "closed_loop_sim",
}
```

## 7. Provider Registry Draft

```python
class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, dict[str, object]] = {}

    def register(self, category: str, provider: object) -> None:
        self._providers.setdefault(category, {})
        name = getattr(provider, "name", None)
        if not name:
            raise ValueError("provider must define a name")
        self._providers[category][name] = provider

    def get(self, category: str, name: str):
        return self._providers[category][name]

    def list_names(self, category: str) -> list[str]:
        return sorted(self._providers.get(category, {}).keys())
```

Design notes:

- `SkillRegistry` and `ProviderRegistry` should coexist.
- Skills represent business capabilities.
- Providers represent implementation choices inside a capability.

## 8. Policy Modules

These should be extracted from algorithm modules over time.

### 8.1 `loop_priors.py`

Use this module for:

- hard and soft minimum time constants
- preferred time-constant ranges
- preferred model families per loop type
- reality-check ranges

### 8.2 `constraints.py`

Use this module for:

- physical tuning limits
- loop-specific Ti/PB constraints
- identification constraint presets

### 8.3 `scoring_rules.py`

Use this module for:

- fit score composition
- physical confidence adjustments
- evaluation score caps
- readiness and robustness scoring

### 8.4 `model_selection_rules.py`

Use this module for:

- model family priority by loop type
- lower-bound hit penalties
- cross-round fallback selection rules

## 9. Migration Strategy

The refactor should be incremental.

### Phase 1: Add registries and wrappers

- keep existing algorithm files
- introduce `ProviderRegistry`
- wrap current algorithm entry points as first-generation providers

### Phase 2: Route skills through providers

- keep existing `BaseSkill` and `SkillRegistry`
- update skills to choose providers through the new registry

### Phase 3: Let pipeline call skills

- gradually replace direct algorithm calls in `runner.py`
- keep output payloads compatible with the current frontend

### Phase 4: Move rules into policies

- extract priors, constraints, and score logic
- make provider logic thinner and easier to test

### Phase 5: Split large algorithm files

- split identification, tuning, and evaluation into provider-specific files
- keep legacy wrappers temporarily for compatibility

## 10. First Concrete Refactor Targets

The first practical steps should be:

1. Add `backend/core/shared/provider_registry.py`.
2. Create provider wrappers for:
   - current window detection
   - current dead-time estimation
   - current transfer-function identification
   - current tuning methods
   - current evaluation engine
3. Add the first batch of business-oriented skills.
4. Let the tuning pipeline call those skills instead of direct algorithm functions.

## 11. Success Criteria

The refactor can be considered successful when:

- a new window detection algorithm can be added without changing the pipeline
- a new identification provider can be selected by config or skill input
- tuning and evaluation rules are not scattered across multiple providers
- skill interfaces remain stable while provider implementations evolve
- unit tests can validate each provider independently
- end-to-end pipeline tests still pass without frontend changes
