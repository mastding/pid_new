/** Pipeline stage event from SSE stream */
export interface StageEvent {
  type: 'stage';
  stage: string;
  status: 'running' | 'done';
  data?: Record<string, unknown>;
}

export interface ErrorEvent {
  type: 'error';
  message: string;
  stage?: string;
  error_code?: string;
}

export interface ResultEvent {
  type: 'result';
  data: TuningResult;
}

export interface SessionStartEvent {
  type: 'session_start';
  task_id: string;
  kind: 'tune' | 'consult';
}

export interface LlmThinkingEvent {
  type: 'llm_thinking';
  stage: string;
  round?: number;
  model: string;
  reasoning_content: string;
  raw_text: string;
}

export type PipelineEvent =
  | StageEvent
  | ErrorEvent
  | ResultEvent
  | SessionStartEvent
  | LlmThinkingEvent
  | { type: 'done' };

/** Closed-loop simulation trace */
export interface SimulationTrace {
  pv_history: number[];
  mv_history: number[];
  sp_history: number[];
  dt: number;
}

/** Day 4: LLM 窗口顾问的选择元数据 */
export interface WindowSelectionMeta {
  mode: 'llm' | 'deterministic' | 'fallback_deterministic' | 'user_override' | 'blocked';
  chosen_index: number;
  deterministic_index: number;
  deterministic_score: number;
  reasoning: string;
  formal_identification_allowed?: boolean;
  diagnostic_identification_allowed?: boolean;
  stop_reason?: string | null;
  window_policy?: WindowSelectionPolicy;
  window_candidate_decision?: WindowCandidateDecision;
  window_policy_results?: WindowPolicyResult[];
  window_algorithm_family_summaries?: WindowAlgorithmFamilySummary[];
  agreed_with_deterministic?: boolean;
  llm_reasoning_chain_len?: number;
  ontology_context_used?: boolean;
  ontology_context_source?: 'mcp' | 'frontend' | 'none';
  ontology_mcp_server?: string;
  ontology_mcp_tool?: string;
  ontology_mcp_query?: string;
  ontology_mcp_content_preview?: string;
  ontology_mcp_content_raw?: string;
  ontology_mcp_content_chars?: number;
  ontology_mcp_error?: string;
  // 原本错误地放在 data_analysis 阶段，现归到 window_selection
  candidate_window_count?: number;
  usable_window_count_pre_policy?: number;
  step_event_count?: number;
  algorithm_filter?: string[] | null;
  window_detection_meta?: Record<string, unknown>;
  policy_adjusted_candidate_windows?: number;
  policy_adjusted_usable_windows?: number;
  ontology_evidence?: Array<{
    fact: string;
    source: string;
  }>;
  window_judgements?: Array<{
    index: number;
    pool_index?: number;
    verdict: 'preferred' | 'acceptable' | 'risk';
    reason: string;
    window_source?: string;
    window_quality_score?: number;
  }>;
  chosen_window_summary?: {
    source: string;
    score: number;
    n_points: number;
  };
  deterministic_window_summary?: {
    source: string;
    score: number;
    n_points: number;
  };
}

export interface WindowSelectionPolicy {
  loop_id?: string;
  loop_type?: string;
  policy_version?: string;
  confidence?: number;
  preferred_algorithm_families?: string[];
  deprioritized_algorithm_families?: string[];
  disabled_algorithm_families?: string[];
  algorithm_plan?: WindowAlgorithmPlanItem[];
  min_mv_excitation?: number | null;
  min_sp_excitation?: number | null;
  min_pv_response?: number | null;
  max_mv_saturation_ratio?: number | null;
  max_pv_noise_ratio?: number | null;
  max_drift_ratio?: number | null;
  expected_dead_time_range_s?: [number, number] | number[] | null;
  expected_time_constant_range_s?: [number, number] | number[] | null;
  expected_gain_sign?: 'positive' | 'negative' | 'unknown';
  min_window_points?: number;
  min_window_duration_s?: number;
  max_window_points?: number | null;
  pre_window_s?: number | null;
  post_window_s?: number | null;
  steady_scan_window_s?: number | null;
  steady_scan_step_s?: number | null;
  merge_gap_s?: number | null;
  max_candidates_per_family?: number;
  allowed_operating_states?: string[];
  avoid_operating_states?: string[];
  llm_policy_raw_text?: string;
  llm_policy_reasoning_content?: string;
  scoring_weights?: Record<string, number>;
  hard_guards?: Array<Record<string, unknown>>;
  soft_penalties?: Array<Record<string, unknown>>;
  field_usage?: WindowPolicyFieldUsage[];
  rationale?: string;
  ontology_facts?: {
    source?: string;
    confidence?: number;
    process_direction?: 'positive' | 'negative' | 'unknown';
    evidence?: Array<Record<string, unknown>>;
    raw_answer?: string | null;
  };
}

export interface WindowAlgorithmPlanItem {
  family?: string;
  state?: 'preferred' | 'available' | 'deprioritized' | 'disabled';
  reason?: string;
  consumed_policy_fields?: string[];
  consumed_policy_field_labels?: string[];
}

export interface WindowPolicyFieldUsage {
  field: string;
  label?: string;
  status: 'consumed' | 'downstream_hint' | 'display_only';
  consumed_by?: string[];
  note?: string;
}

export interface WindowCandidateDecision {
  selected_window_indices: number[];
  rejected_window_indices: number[];
  fallback_window_indices: number[];
  formal_identification_allowed: boolean;
  diagnostic_identification_allowed: boolean;
  stop_reason?: string | null;
  primary_reason: string;
  ontology_evidence?: Array<Record<string, unknown>>;
  data_evidence?: Array<Record<string, unknown>>;
  window_judgements?: Array<Record<string, unknown>>;
  recommended_identification_plan?: Record<string, unknown>;
  risk_flags?: string[];
}

export interface WindowPolicyResult {
  index: number;
  window_source?: string;
  algorithm_family?: string;
  original_score?: number;
  policy_score?: number;
  ontology_consistency_score?: number;
  usable_before_policy?: boolean;
  usable_after_policy?: boolean;
  policy_violations?: Array<{
    level?: 'hard' | 'soft';
    code?: string;
    message?: string;
  }>;
}

export interface WindowAlgorithmFamilySummary {
  family?: string;
  provider?: string;
  run_state?: 'ran' | 'disabled' | 'skipped' | string;
  policy_state?: 'preferred' | 'available' | 'deprioritized' | 'disabled' | string;
  policy_reason?: string;
  event_count?: number;
  window_count?: number;
  usable_count?: number;
  best_score?: number;
}

/** Final tuning result */
export interface TuningResult {
  data_analysis: {
    data_points: number;
    sampling_time: number;
    step_events: unknown[];
    candidate_windows: CandidateWindow[];
    quality_metrics?: unknown;
  };
  formal_identification_blocked?: boolean;
  block_reason?: string | null;
  diagnostic_identification_allowed?: boolean;
  pipeline_status?: string;
  model: {
    model_type: string;
    K: number;
    T: number;
    T1: number;
    T2: number;
    L: number;
    r2_score: number;
    normalized_rmse: number;
    confidence: number;
    confidence_quality: string;
    window_source: string;
    selection_reason: string;
    fit_preview: FitPreview;
    candidates: ModelCandidate[];
    attempts?: IdentificationAttempt[];
    algorithm_comparison?: WindowAlgorithmFitSummary[];
  };
  pid_params: {
    Kp: number;
    Ki: number;
    Kd: number;
    Ti: number;
    Td: number;
    strategy: string;
    candidates: StrategyCandidate[];
  };
  evaluation: {
    passed: boolean;
    performance_score: number;
    final_rating: number;
    readiness_score: number;
    robustness_score: number;
    is_stable: boolean;
    overshoot_percent: number;
    settling_time_s: number;
    steady_state_error: number;
    oscillation_count: number;
    mv_saturation_pct: number;
    recommendation: string;
    reality_check_score?: number;
    reality_check_typical_T?: number;
    reality_check_diverged?: boolean;
    score_caps_applied?: string[];
    simulation: SimulationTrace;
  };
  loop_type: string;
  loop_name: string;
  window_selection?: WindowSelectionMeta;
  model_review?: ModelReviewMeta | null;
}

/** LLM 模型评审结果（identification 后的 verdict）
 *
 * Phase 1 起，verdict 只有 accept / downgrade 两种 —— 不再 reject 中止流程。
 * downgrade 仍会跑完整定，但带"不可信"标记并收紧评分上限。
 */
export interface ModelReviewMeta {
  verdict: 'accept' | 'downgrade';
  reason: string;
  concerns: string[];
  fallback?: boolean;
  error_type?: string | null;
  error_message?: string | null;
  raw_text?: string;
}

export interface IdentificationRefinementMeta {
  round: number;
  retry: boolean;
  source?: string;
  rationale: string;
  force_window_index?: number | null;
  force_model_types?: string[];
  hint_L?: number | null;
  recommended_algorithm?: string;
  recommended_algorithm_label?: string;
  recommended_window_source?: string;
  evidence?: WindowAlgorithmFitSummary;
}

/** identification 阶段的单次拟合尝试（某个窗口 × 某个模型类型） */
export interface IdentificationAttempt {
  model_type: string;
  window_source: string;
  window_algorithm?: string;
  window_algorithm_label?: string;
  window_quality_score?: number;
  window_score_breakdown?: Record<string, number>;
  success: boolean;
  round?: number;
  K?: number;
  T?: number;
  T1?: number;
  T2?: number;
  L?: number;
  zeta?: number | null;
  r2_score?: number;
  normalized_rmse?: number;
  fit_score?: number;
  confidence?: number;
  degenerate_T?: boolean;
  fit_preview?: FitPreview;
  error?: string;
}

export interface WindowAlgorithmFitSummary {
  algorithm: string;
  algorithm_label: string;
  window_source: string;
  model_type: string;
  fit_score: number;
  r2_score: number;
  normalized_rmse: number;
  confidence: number;
  window_quality_score: number;
}

export interface CandidateWindow {
  index: number;
  start: number;
  end: number;
  n_points: number;
  score: number;
  amplitude: number;
  window_usable_for_id: boolean;
  source: string;
}

export interface FitPreview {
  points?: Array<{
    index: number;
    time?: string;
    pv: number;
    pv_fit: number;
    mv: number;
  }>;
  model_type?: string;
  x_axis?: 'timestamp' | 'index';
  t?: number[];
  pv_actual?: number[];
  pv_fitted?: number[];
}

export interface ModelCandidate {
  model_type: string;
  K: number;
  T: number;
  L: number;
  r2_score: number;
  fit_score: number;
}

export interface StrategyCandidate {
  strategy: string;
  Kp: number;
  Ki: number;
  Kd: number;
  Ti: number;
  Td: number;
  description: string;
  is_recommended: boolean;
}

/** Consultant session context (sent with each consult request) */
export interface ConsultSession {
  csv_path: string;
  loop_type: string;
  dt: number;
  model_type: string;
  model_K: number;
  model_T: number;
  model_T1: number;
  model_T2: number;
  model_L: number;
  model_r2: number;
  model_nrmse: number;
  model_confidence: number;
  n_windows: number;
  Kp: number;
  Ki: number;
  Kd: number;
  Ti: number;
  Td: number;
  tuning_strategy: string;
}
