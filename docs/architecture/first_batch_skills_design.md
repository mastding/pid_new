# First Batch Skills Design

## 1. Scope

This document defines the first five high-value skills to introduce into the current `pid_v2` project.

Selection criteria:

- high reuse across the tuning workflow
- strong need for algorithm replacement in the future
- clear business-level meaning
- easy to introduce without rewriting the whole system

The first batch is:

1. `detect_windows_skill`
2. `estimate_dead_time_skill`
3. `identify_model_skill`
4. `generate_tuning_candidates_skill`
5. `evaluate_tuning_skill`

## 2. Shared Skill Design Principles

All skills in this batch should:

- inherit from the current `BaseSkill`
- use the current context mechanism
- return a stable output envelope
- expose the chosen provider name
- separate warnings from hard failure

Recommended standard return shape:

```python
{
    "success": True,
    "data": {...},
    "warnings": [...],
    "reasoning": "...",
}
```

Recommended fields inside `data` when possible:

```python
{
    "provider": "provider_name",
    "meta": {
        "version": "v1",
        "policy_set": "default",
    },
    ...
}
```

## 3. Skill 1: `detect_windows_skill`

### 3.1 Business purpose

Detect candidate identification windows from historical loop data.

This skill should answer:

- which windows are available
- how they were detected
- which windows are usable for identification
- how those windows are ranked

### 3.2 Why this skill is worth doing

- window logic will likely change often
- multiple providers will be useful
- this is a natural capability boundary

### 3.3 Input fields

```python
class DetectWindowsInputs(BaseModel):
    provider: str | None = Field(None, description="window provider name")
    loop_type: str | None = Field(None, description="flow/pressure/temperature/level")
    mode: str = Field("auto", description="auto or specified provider mode")
    max_windows: int = Field(8, description="max number of windows to return")
    include_unusable: bool = Field(True, description="whether to include low-quality windows")
```

### 3.4 Output fields

```python
{
    "provider": "history_rule_based",
    "candidate_count": 6,
    "usable_count": 4,
    "step_event_count": 6,
    "windows": [
        {
            "index": 0,
            "source": "mv_change_1",
            "type": "mv_ramp",
            "score": 0.8324,
            "usable": True,
            "mv_span": 8.21,
            "pv_span": 3.15,
            "corr": 0.61,
        }
    ],
    "meta": {...}
}
```

### 3.5 Context dependencies

Reads:

- `ctx.cleaned_df`
- `ctx.dt`
- `ctx.loop_type`

Writes:

- `ctx.candidate_windows`
- `ctx.data_profile["window_detection"]`

### 3.6 Provider dependencies

Initial providers:

- `history_rule_based`
  Migrated from current data-analysis window logic.

Future providers:

- `change_point_based`
- `hybrid`

### 3.7 Migration source

Main migration source:

- `backend/core/algorithms/data_analysis.py`
  - current candidate window generation
  - current usable-window heuristics
- `backend/core/skills/data_understanding/detect_candidate_windows.py`
  - current skill envelope pattern

### 3.8 Notes

- this skill should detect windows only
- window selection should remain a separate skill

## 4. Skill 2: `estimate_dead_time_skill`

### 4.1 Business purpose

Estimate dead time for a chosen window or signal segment.

This skill should answer:

- what dead time is estimated
- how confident the estimate is
- which estimation method was used

### 4.2 Why this skill is worth doing

- dead-time estimation is a distinct technical capability
- it is a common algorithm replacement point
- it can be reused by identification and diagnostics

### 4.3 Input fields

```python
class EstimateDeadTimeInputs(BaseModel):
    provider: str | None = Field(None, description="dead-time provider name")
    window_index: int | None = Field(None, description="candidate window index")
    use_selected_window: bool = Field(True, description="use selected window from context")
    force_positive_lag_only: bool = Field(True, description="apply positive-lag assumption")
```

### 4.4 Output fields

```python
{
    "provider": "cross_correlation",
    "window_index": 2,
    "L": 60.0,
    "confidence": 0.79,
    "meta": {
        "peak_corr": 0.41,
        "positive_lag_only": True,
    }
}
```

### 4.5 Context dependencies

Reads:

- `ctx.cleaned_df`
- `ctx.dt`
- `ctx.candidate_windows`
- `ctx.selected_window_index`

Writes:

- `ctx.data_profile["dead_time"]`
- optionally `ctx.model` hint fields or side-channel metadata

### 4.6 Provider dependencies

Initial providers:

- `cross_correlation`

Future providers:

- `reaction_curve`
- `hybrid_dead_time`

### 4.7 Migration source

Main migration source:

- `backend/core/algorithms/system_id.py`
  - current `_estimate_dead_time()`

### 4.8 Notes

- this skill should not fit the full process model
- it should remain independent and reusable

## 5. Skill 3: `identify_model_skill`

### 5.1 Business purpose

Identify the best process model for one or more windows.

This skill should answer:

- which model is best
- what alternative attempts were tried
- how fit and confidence were determined
- which provider produced the result

### 5.2 Why this skill is worth doing

- this is the core technical capability of the product
- it is the most likely place for future algorithm replacement
- it enables side-by-side evaluation of providers

### 5.3 Input fields

```python
class IdentifyModelInputs(BaseModel):
    provider: str | None = Field(None, description="identification provider")
    window_indices: list[int] | None = Field(None, description="specific candidate windows to use")
    use_usable_windows_only: bool = Field(True, description="restrict to usable windows")
    loop_type: str | None = Field(None, description="loop type override")
    model_pool: list[str] | None = Field(None, description="allowed model types")
    hint_L: float | None = Field(None, description="dead-time initial hint")
```

### 5.4 Output fields

```python
{
    "provider": "transfer_function_fit",
    "best_model": {
        "model_type": "FOPDT",
        "K": 0.96,
        "T1": 194.69,
        "L": 60.0,
        "fit_score": 15.2,
        "confidence": 0.81,
    },
    "attempts": [...],
    "round_summary": {
        "window_count": 4,
        "attempt_count": 16,
    },
    "meta": {...}
}
```

### 5.5 Context dependencies

Reads:

- `ctx.cleaned_df`
- `ctx.dt`
- `ctx.candidate_windows`
- `ctx.loop_type`
- `ctx.selected_window_index`

Writes:

- `ctx.model`
- `ctx.confidence`
- `ctx.data_profile["identification"]`

### 5.6 Provider dependencies

Initial providers:

- `transfer_function_fit`

Future providers:

- `arx`
- `state_space`
- `hybrid_identification`

### 5.7 Migration source

Main migration source:

- `backend/core/algorithms/system_id.py`
  - model simulation
  - model fitting
  - fit score logic
  - confidence logic

Related future integration points:

- `backend/core/pipeline/identification_advisor.py`
- `backend/core/pipeline/identification_refinement_advisor.py`

### 5.8 Notes

- this skill should not own LLM review logic
- LLM review should stay in separate review/refinement skills

## 6. Skill 4: `generate_tuning_candidates_skill`

### 6.1 Business purpose

Generate PID tuning candidates from an identified process model.

This skill should answer:

- which tuning strategies were tried
- what PID candidates were produced
- which candidate is recommended before evaluation

### 6.2 Why this skill is worth doing

- tuning formulas will evolve
- different plants may want different candidate sets
- strategy generation should be independent from final selection policy

### 6.3 Input fields

```python
class GenerateTuningCandidatesInputs(BaseModel):
    provider_names: list[str] | None = Field(None, description="specific tuning providers")
    recommended_strategy: str | None = Field(None, description="preferred strategy if available")
    loop_type: str | None = Field(None, description="loop type override")
    constraints_profile: str | None = Field(None, description="constraint preset name")
```

### 6.4 Output fields

```python
{
    "providers": ["imc", "lambda_tuning", "zn", "chr"],
    "candidate_count": 4,
    "recommended": {
        "strategy": "LAMBDA",
        "Kp": 0.7967,
        "Ki": 0.003546,
        "Kd": 0.0,
        "Ti": 224.69,
        "Td": 0.0,
        "PB": 125.52,
    },
    "candidates": [...],
    "warnings": [...],
    "meta": {...}
}
```

### 6.5 Context dependencies

Reads:

- `ctx.model`
- `ctx.loop_type`

Writes:

- `ctx.pid_params`
- `ctx.data_profile["tuning_candidates"]`

### 6.6 Provider dependencies

Initial providers:

- `imc`
- `lambda_tuning`
- `zn`
- `chr`

Future providers:

- `cohen_coon`
- `amigo`
- `plant_specific_rules`

### 6.7 Migration source

Main migration source:

- `backend/core/algorithms/pid_tuning.py`
  - `tune_fo`
  - `tune_fopdt`
  - `tune_sopdt`
  - `tune_ipdt`
  - candidate selection scaffolding

### 6.8 Notes

- candidate generation and final recommendation can be combined in phase 1
- later they can split into two skills if strategy governance becomes complex

## 7. Skill 5: `evaluate_tuning_skill`

### 7.1 Business purpose

Evaluate a selected PID candidate against the identified model and supporting guardrails.

This skill should answer:

- how the PID performs in simulation
- what the final scores are
- whether reality-check divergence exists
- whether score caps were applied

### 7.2 Why this skill is worth doing

- evaluation policy will continue to change
- score caps, reality checks, and physical guards need stable contracts
- this is where engineering governance is enforced

### 7.3 Input fields

```python
class EvaluateTuningInputs(BaseModel):
    provider: str | None = Field(None, description="evaluation provider")
    use_recommended_pid: bool = Field(True, description="evaluate recommended pid from context")
    pid_strategy: str | None = Field(None, description="evaluate a named candidate strategy")
    loop_type: str | None = Field(None, description="loop type override")
    tuning_unreliable: bool = Field(False, description="whether tuning is already flagged unreliable")
    review_unreliable_reason: str | None = Field(None, description="upstream reliability warning")
```

### 7.4 Output fields

```python
{
    "provider": "closed_loop_sim",
    "performance_score": 7.75,
    "final_rating": 7.85,
    "readiness_score": 7.86,
    "robustness_score": 6.25,
    "score_caps_applied": [],
    "metrics": {
        "overshoot": 0.0,
        "settling_time": 1110.0,
        "steady_state_error": 0.0,
    },
    "meta": {
        "reality_check_t": 300.0,
        "reality_check_diverged": False,
    }
}
```

### 7.5 Context dependencies

Reads:

- `ctx.model`
- `ctx.pid_params`
- `ctx.loop_type`

Writes:

- `ctx.evaluation`
- `ctx.data_profile["evaluation"]`

### 7.6 Provider dependencies

Initial providers:

- `closed_loop_sim`

Future providers:

- `robust_monte_carlo_eval`
- `historical_playback_eval`

### 7.7 Migration source

Main migration source:

- `backend/core/algorithms/pid_evaluation.py`
  - closed-loop simulation
  - performance scoring
  - readiness scoring
  - reality check
  - score caps

### 7.8 Notes

- score-cap reasoning must remain visible in output
- this skill should preserve a machine-readable explanation trail

## 8. Recommended Introduction Order

The recommended implementation order is:

1. `detect_windows_skill`
2. `identify_model_skill`
3. `generate_tuning_candidates_skill`
4. `evaluate_tuning_skill`
5. `estimate_dead_time_skill`

Reasoning:

- window detection and identification unlock the most architecture value first
- tuning and evaluation then complete the end-to-end path
- dead-time estimation can be introduced as a reusable sub-capability once the provider pattern is stable

## 9. Minimum Provider Set for Phase 1

The first provider set should be small:

- window
  - `history_rule_based`
- dead_time
  - `cross_correlation`
- identification
  - `transfer_function_fit`
- tuning
  - `imc`
  - `lambda_tuning`
  - `zn`
  - `chr`
- evaluation
  - `closed_loop_sim`

This keeps the first migration low-risk while establishing the long-term structure.

## 10. Phase-1 Success Criteria

Phase 1 succeeds when:

- the tuning pipeline can reach final evaluation through skills
- each skill declares and exposes its provider
- current frontend payloads remain compatible
- new provider implementations can be added without changing the pipeline stage order
- unit tests can run provider-by-provider and skill-by-skill
