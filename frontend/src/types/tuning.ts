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

export type PipelineEvent = StageEvent | ErrorEvent | ResultEvent | { type: 'done' };

/** Closed-loop simulation trace */
export interface SimulationTrace {
  pv_history: number[];
  mv_history: number[];
  sp_history: number[];
  dt: number;
}

/** Day 4: LLM 窗口顾问的选择元数据 */
export interface WindowSelectionMeta {
  mode: 'llm' | 'deterministic' | 'fallback_deterministic' | 'user_override';
  chosen_index: number;
  deterministic_index: number;
  deterministic_score: number;
  reasoning: string;
  agreed_with_deterministic?: boolean;
  llm_reasoning_chain_len?: number;
  chosen_window_summary?: {
    source: string;
    score: number;
    n_points: number;
  };
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

/** LLM 模型评审结果（identification 后的 verdict） */
export interface ModelReviewMeta {
  verdict: 'accept' | 'downgrade' | 'reject';
  reason: string;
  concerns: string[];
}

/** identification 阶段的单次拟合尝试（某个窗口 × 某个模型类型） */
export interface IdentificationAttempt {
  model_type: string;
  window_source: string;
  success: boolean;
  K?: number;
  T?: number;
  T1?: number;
  T2?: number;
  L?: number;
  zeta?: number;
  r2_score?: number;
  normalized_rmse?: number;
  fit_score?: number;
  confidence?: number;
  degenerate_T?: boolean;
  error?: string;
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
