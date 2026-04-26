import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 120_000,
});

/** Upload CSV and run tuning pipeline with SSE streaming */
export function tunePidStream(
  file: File,
  params: {
    loop_type?: string;
    loop_name?: string;
    plant_type?: string;
    scenario?: string;
    control_object?: string;
    selected_loop_prefix?: string;
    selected_window_index?: number;
    use_llm_advisor?: boolean;
  },
  onEvent: (event: Record<string, unknown>) => void,
): AbortController {
  const controller = new AbortController();
  const formData = new FormData();
  formData.append('file', file);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      formData.append(key, String(value));
    }
  });

  fetch('/api/tune/stream', {
    method: 'POST',
    body: formData,
    signal: controller.signal,
  }).then(async (response) => {
    const reader = response.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const text = line.replace(/^data: /, '').trim();
        if (text) {
          try {
            onEvent(JSON.parse(text));
          } catch {
            // skip malformed events
          }
        }
      }
    }
  });

  return controller;
}

/** Inspect CSV for PID loops */
export async function inspectLoops(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post('/data/inspect-loops', formData);
  return data;
}

/** Inspect candidate identification windows */
export async function inspectWindows(file: File, loopPrefix?: string, loopType?: string) {
  const formData = new FormData();
  formData.append('file', file);
  if (loopPrefix) formData.append('loop_prefix', loopPrefix);
  if (loopType) formData.append('loop_type', loopType);
  const { data } = await api.post('/data/inspect-windows', formData);
  return data;
}

export interface LoopSeriesPoint {
  t: string | number;
  pv: number;
  sv: number | null;
  mv: number;
}

export interface LoopSeriesResp {
  csv_path: string;
  loop_prefix: string;
  x_axis: 'timestamp' | 't';
  dt: number;
  total_points: number;
  sampled_points: number;
  points: LoopSeriesPoint[];
  error?: string;
}

export async function getLoopSeries(params: {
  csv_path: string;
  loop_prefix?: string;
  start_time?: string;
  end_time?: string;
  max_points?: number;
}) {
  const { data } = await api.get<LoopSeriesResp>('/data/series', { params });
  return data;
}

/** 根据 DCS 位号前缀推断回路类型。识别 FIC/PIC/TIC/LIC 四类，识别不到返回 null。 */
export interface HistoryLoop {
  loop_id: string;
  loop_prefix: string;
  loop_type: string;
  dataset_id: string;
  source_filename: string;
  rows: number;
  sampling_time: number;
  start_time?: string | null;
  end_time?: string | null;
  pv_min?: number | null;
  pv_max?: number | null;
  mv_min?: number | null;
  mv_max?: number | null;
  window_count?: number;
  usable_window_count?: number;
  best_window_source?: string;
  best_window_score?: number | null;
}

export interface HistoryImportResp {
  dataset_id: string;
  imported_count: number;
  loops: HistoryLoop[];
  errors: Array<{ filename: string; error: string }>;
}

export async function importHistoryFiles(files: File[]) {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  const { data } = await api.post<HistoryImportResp>('/history/import', formData);
  return data;
}

export async function listHistoryLoops() {
  const { data } = await api.get<{ total: number; items: HistoryLoop[] }>('/history/loops');
  return data;
}

export async function getHistoryLoopSeries(loopId: string, params: {
  start_time?: string;
  end_time?: string;
  max_points?: number;
} = {}) {
  const { data } = await api.get<LoopSeriesResp & { loop_id: string; loop_type: string }>(
    `/history/loops/${encodeURIComponent(loopId)}/series`,
    { params },
  );
  return data;
}

export interface HistoryLoopAssessment {
  loop_id: string;
  loop_type: string;
  summary?: {
    decision?: string;
    decision_text?: string;
    recommended_next_action?: string;
    recommended_next_action_text?: string;
  };
  performance?: {
    score?: number;
    level?: string;
    monitoring_score?: number | null;
    stability_score?: number | null;
    constraint_score?: number | null;
    pv_mv_behavior_score?: number | null;
  };
  tuning_readiness?: {
    score?: number;
    level?: string;
    decision?: string;
    recommended_next_action?: string;
    blocking_reasons?: Array<{
      type: string;
      severity: string;
      message: string;
      evidence?: Record<string, unknown>;
    }>;
    gate_checks?: Array<{
      name: string;
      passed: boolean;
      severity: string;
      message: string;
      evidence?: Record<string, unknown>;
    }>;
  };
  identification_suitability?: {
    score?: number;
    level?: string;
    excitation_score?: number | null;
    response_observability_score?: number | null;
    direction_confidence?: number | null;
    window_count?: number;
    usable_window_count?: number;
    best_window_score?: number | null;
    best_window_source?: string;
  };
  operating_condition?: HistoryLoopFeatures['operating_condition_profile'];
  data_quality: {
    score: number;
    level: string;
    missing_ratio: number;
    continuity_score: number;
    noise_score: number;
    saturation_score: number;
  };
  identifiability: {
    score: number;
    level: string;
    window_count: number;
    usable_window_count: number;
    best_window_score?: number | null;
    best_window_source?: string;
    best_window_reasons?: string[];
  };
  diagnostics: {
    pv_range?: Record<string, unknown>;
    noise?: Record<string, unknown>;
    saturation?: Record<string, unknown>;
    deadzone?: Record<string, unknown>;
    oscillation?: Record<string, unknown>;
    disturbance?: Record<string, unknown>;
    flags: Array<{ type: string; severity: string; message: string }>;
  };
  readiness: {
    score: number;
    level: string;
    recommendations: string[];
  };
  error?: string;
}

export async function getHistoryLoopAssessment(loopId: string) {
  const { data } = await api.get<HistoryLoopAssessment>(
    `/history/loops/${encodeURIComponent(loopId)}/assessment`,
  );
  return data;
}

export interface RawSeriesStats {
  available?: boolean;
  count?: number;
  missing_ratio?: number;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  span?: number | null;
  p95_abs_step?: number | null;
  max_abs_step?: number | null;
  flat_step_ratio?: number | null;
  active_step_ratio?: number | null;
  move_count_per_hour?: number | null;
  direction_reversal_per_hour?: number | null;
  total_travel?: number | null;
  travel_per_hour?: number | null;
}

export interface HistoryLoopFeatures {
  identity: {
    loop_id?: string;
    loop_type?: string;
    source_filename?: string | null;
    dataset_id?: string | null;
    [key: string]: unknown;
  };
  data_profile: {
    row_count?: number;
    valid_row_count?: number;
    time_start?: string | null;
    time_end?: string | null;
    duration_h?: number | null;
    sample_time_median_s?: number | null;
    sample_interval_p95_s?: number | null;
    sample_interval_p99_s?: number | null;
    irregular_sample_ratio?: number | null;
    long_gap_count?: number;
    duplicate_timestamp_count?: number;
    [key: string]: unknown;
  };
  pv_stats?: RawSeriesStats | null;
  mv_stats?: RawSeriesStats | null;
  sp_stats?: RawSeriesStats | null;
  scale_profile?: {
    pv?: Record<string, number | null | undefined>;
    mv?: Record<string, number | null | undefined>;
    sp?: Record<string, number | null | undefined>;
    pv_range_type?: string;
    mv_scale_type?: string;
    normalization?: Record<string, number | null | undefined>;
    [key: string]: unknown;
  };
  process_prior?: {
    process_direction?: string;
    process_direction_confidence?: number | null;
    k_sign_constraint?: string;
    static_gain_hint?: number | null;
    gain_sample_count?: number | null;
    gain_variability?: number | null;
    response_lag_hint_s?: number | null;
    time_constant_prior_min_s?: number | null;
    time_constant_prior_max_s?: number | null;
    time_constant_prior_basis?: string;
    [key: string]: unknown;
  };
  excitation_profile?: {
    mv_excitation_span?: number | null;
    mv_effective_excitation_span?: number | null;
    mv_excitation_event_count?: number | null;
    mv_ramp_event_count?: number | null;
    pv_response_after_mv_ratio?: number | null;
    saturation_free_ratio?: number | null;
    usable_excitation_ratio?: number | null;
    excitation_level?: string;
    [key: string]: unknown;
  };
  actuator_profile?: {
    mv_resolution_hint?: number | null;
    mv_deadband_hint_ratio?: number | null;
    mv_stiction_hint?: boolean;
    longest_mv_stuck_duration_s?: number | null;
    mv_rate_limit_hint?: boolean;
    mv_saturation_margin_low?: number | null;
    mv_saturation_margin_high?: number | null;
    mv_saturation_ratio?: number | null;
    [key: string]: unknown;
  };
  operating_condition_profile?: {
    condition_label?: string;
    confidence?: number | null;
    tuning_suitability?: string;
    reason_codes?: string[];
    evidence?: Array<{
      name?: string;
      value?: string | number | boolean | null;
      status?: string;
      detail?: string;
      [key: string]: unknown;
    }>;
    segment_summary?: Array<{
      label?: string;
      duration_s?: number | null;
      ratio?: number | null;
      tuning_usable?: boolean;
      reason?: string;
      [key: string]: unknown;
    }>;
    recommendations?: string[];
    ontology_context?: {
      status?: string;
      loop_type_hint?: string;
      requires_fields?: string[];
      [key: string]: unknown;
    };
    debug?: Record<string, unknown>;
    [key: string]: unknown;
  };
  pv_mv_relation_raw?: {
    estimated_direction_raw?: string;
    process_direction?: string;
    process_direction_confidence?: number | null;
    process_direction_basis?: string;
    cross_correlation_peak_abs?: number | null;
    best_lag_corr_pv_mv?: number | null;
    best_lag_s_pv_mv?: number | null;
    best_lag_corr_dpv_dmv?: number | null;
    best_lag_s_dpv_dmv?: number | null;
    [key: string]: unknown;
  };
  constraint_raw?: {
    mv_saturation_ratio?: number | null;
    mv_high_saturation_ratio?: number | null;
    mv_low_saturation_ratio?: number | null;
    longest_mv_saturation_duration_s?: number | null;
    mv_saturation_segment_count?: number | null;
    pv_near_observed_min_ratio?: number | null;
    pv_near_observed_max_ratio?: number | null;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface HistoryLoopMonitoringAlert {
  type: string;
  severity: string;
  message: string;
}

export interface HistoryLoopMonitoringEvent {
  type: string;
  severity: string;
  name: string;
  message: string;
  status?: string;
  source?: string;
  recommendation?: string;
  evidence?: Record<string, unknown>;
}

export interface HistoryLoopMonitoringSnapshot {
  status: string;
  overall_score: number;
  data_health?: {
    status?: string;
    score?: number;
    missing_ratio?: number | null;
    irregular_sample_ratio?: number | null;
    duplicate_timestamp_ratio?: number | null;
    long_gap_count?: number;
    pv_outlier_count?: number | null;
    mv_outlier_count?: number | null;
    pv_noise_ratio?: number | null;
    pv_snr_db?: number | null;
    pv_spike_count?: number | null;
  };
  stability?: {
    status?: string;
    score?: number;
    oscillation_detected?: boolean;
    oscillation_severity?: string;
    oscillation_confidence?: number | null;
    pv_dominant_period_s?: number | null;
    pv_dominant_power_ratio?: number | null;
    pv_zero_crossing_per_hour?: number | null;
    phase_hint?: string | null;
  };
  pv_mv_behavior?: {
    status?: string;
    score?: number;
    mv_direction_reversal_per_hour?: number | null;
    mv_travel_per_hour?: number | null;
    mv_spike_count?: number | null;
    mv_spike_ratio?: number | null;
  };
  constraints?: {
    status?: string;
    score?: number;
    mv_saturation_ratio?: number | null;
    mv_high_saturation_ratio?: number | null;
    mv_low_saturation_ratio?: number | null;
    longest_mv_saturation_duration_s?: number | null;
    mv_saturation_segment_count?: number | null;
    reason?: string;
  };
  tracking?: {
    sp_available?: boolean;
    status?: string;
    score?: number;
    reason?: string;
  };
  response_observability?: {
    status?: string;
    score?: number;
    estimated_direction_raw?: string;
    process_direction?: string;
    process_direction_confidence?: number | null;
    process_direction_basis?: string;
    cross_correlation_peak_abs?: number | null;
  };
  operating_condition?: {
    condition_label?: string;
    confidence?: number | null;
    tuning_suitability?: string;
    evidence?: Array<{
      name?: string;
      value?: string | number | boolean | null;
      status?: string;
      detail?: string;
      [key: string]: unknown;
    }>;
    recommendations?: string[];
    [key: string]: unknown;
  };
  noise?: Record<string, unknown>;
  oscillation?: Record<string, unknown>;
  events?: HistoryLoopMonitoringEvent[];
  alerts?: HistoryLoopMonitoringAlert[];
  [key: string]: unknown;
}

export interface HistoryLoopMonitoring {
  loop_id: string;
  features: HistoryLoopFeatures;
  monitoring: HistoryLoopMonitoringSnapshot;
}

export async function fetchHistoryLoopFeatures(loopId: string) {
  const { data } = await api.get<HistoryLoopFeatures>(
    `/history/loops/${encodeURIComponent(loopId)}/features`,
  );
  return data;
}

export async function fetchHistoryLoopMonitoring(loopId: string) {
  const { data } = await api.get<HistoryLoopMonitoring>(
    `/history/loops/${encodeURIComponent(loopId)}/monitoring`,
  );
  return data;
}

export interface HistoryWindowPreviewPoint {
  t: string | number;
  pv: number;
  mv: number;
}

export interface HistoryWindow {
  index: number;
  source: string;
  type: string;
  algorithm?: string;
  algorithm_label?: string;
  selection_basis?: string;
  start_idx: number;
  end_idx: number;
  n_points: number;
  start_time?: string | null;
  end_time?: string | null;
  usable: boolean;
  score: number;
  amplitude: number;
  mv_span: number;
  pv_span: number;
  corr: number;
  reasons: string[];
  score_breakdown?: {
    mv_excitation?: number;
    pv_response?: number;
    lag_correlation?: number;
    saturation_penalty?: number;
    drift_penalty?: number;
  };
  quality_metrics?: Record<string, number>;
  preview: HistoryWindowPreviewPoint[];
}

export async function getHistoryLoopWindows(loopId: string) {
  const { data } = await api.get<{
    loop_id: string;
    loop_type: string;
    dt: number;
    total: number;
    usable_count: number;
    algorithm_summary?: Record<string, { total: number; usable: number }>;
    windows: HistoryWindow[];
    error?: string;
  }>(`/history/loops/${encodeURIComponent(loopId)}/windows`);
  return data;
}

export function tuneHistoryLoopStream(
  loopId: string,
  params: {
    loop_type?: string;
    loop_name?: string;
    plant_type?: string;
    scenario?: string;
    control_object?: string;
    selected_window_index?: number;
    use_llm_advisor?: boolean;
  },
  onEvent: (event: Record<string, unknown>) => void,
): AbortController {
  const controller = new AbortController();
  const formData = new FormData();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      formData.append(key, String(value));
    }
  });

  fetch(`/api/history/loops/${encodeURIComponent(loopId)}/tune/stream`, {
    method: 'POST',
    body: formData,
    signal: controller.signal,
  }).then(async (response) => {
    const reader = response.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const text = line.replace(/^data: /, '').trim();
        if (text) {
          try {
            onEvent(JSON.parse(text));
          } catch {
            // skip malformed events
          }
        }
      }
    }
  });

  return controller;
}

export function inferLoopTypeFromPrefix(prefix: string): string | null {
  if (!prefix) return null;
  const m = prefix.toUpperCase().match(/(?:^|_)([FPTL])IC?[A-Z]?_?\d/);
  if (!m) return null;
  return { F: 'flow', P: 'pressure', T: 'temperature', L: 'level' }[m[1]] ?? null;
}

/** Get system config */
export async function getSystemConfig() {
  const { data } = await api.get('/system-config');
  return data;
}

// ── 模型配置 ────────────────────────────────────────────────────────────

export interface ModelConfig {
  model_api_url: string;
  model_api_key: string;
  model_name: string;
}

export async function fetchModelConfig() {
  const { data } = await api.get<ModelConfig>('/model-config');
  return data;
}

export async function updateModelConfig(body: {
  model_api_url?: string | null;
  model_api_key?: string | null;
  model_name?: string | null;
}) {
  const { data } = await api.put<{ status: string; config: ModelConfig }>(
    '/model-config',
    body,
  );
  return data;
}

export async function testModelConfig() {
  const { data } = await api.post<{ status: string; message: string }>(
    '/model-config/test',
  );
  return data;
}

/** Stream PID consultant chat events via SSE.
 *
 * Sends conversation messages + a session context (csv_path, current model,
 * current PID). The agent can call the 5 tools to refine results iteratively.
 */
export function consultStream(
  body: {
    messages: { role: string; content: string }[];
    session: Record<string, unknown>;
    max_iterations?: number;
  },
  onEvent: (event: Record<string, unknown>) => void,
): AbortController {
  const controller = new AbortController();

  fetch('/api/consult/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: controller.signal,
  }).then(async (response) => {
    const reader = response.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const text = line.replace(/^data: /, '').trim();
        if (text) {
          try {
            onEvent(JSON.parse(text));
          } catch {
            // skip malformed events
          }
        }
      }
    }
  });

  return controller;
}

// ── Session history ─────────────────────────────────────────────────────────

export interface SessionMeta {
  task_id: string;
  kind: 'tune' | 'consult';
  created_at: string;
  ended_at?: string;
  duration_s?: number;
  n_events?: number;
  status?: 'ok' | 'error' | string;
  last_stage?: string;
  error?: string;
  csv_name?: string;
  loop_type?: string;
  loop_name?: string;
  use_llm_advisor?: boolean;
  user_prompt?: string;
  summary?: {
    passed?: boolean;
    performance_score?: number;
    final_rating?: number;
    is_stable?: boolean;
    decay_ratio?: number;
    overshoot_percent?: number;
    model_type?: string;
    r2_score?: number;
    confidence?: number;
    strategy?: string;
    selection_mode?: string;
    chosen_index?: number;
    deterministic_index?: number;
    agreed_with_deterministic?: boolean;
  };
}

export interface SessionDetail {
  meta: SessionMeta;
  events: Array<Record<string, unknown> & { _seq?: number; _t?: number }>;
}

export async function listSessions(params: { limit?: number; kind?: 'tune' | 'consult' } = {}) {
  const { data } = await api.get<{ total: number; items: SessionMeta[] }>('/sessions', {
    params,
  });
  return data;
}

export async function getSession(taskId: string) {
  const { data } = await api.get<SessionDetail>(`/sessions/${taskId}`);
  return data;
}

export async function deleteSession(taskId: string) {
  const { data } = await api.delete<{ deleted: string }>(`/sessions/${taskId}`);
  return data;
}

export default api;
