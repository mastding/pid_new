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

export interface AssistantSessionMessage {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  reasoning_summary?: string;
  raw_events?: Array<Record<string, unknown>>;
  created_at?: string;
}

export interface AssistantSessionSummary {
  id: string;
  kind: 'assistant';
  title: string;
  loop_id?: string | null;
  created_at?: string;
  updated_at?: string;
  message_count?: number;
}

export interface AssistantSession extends AssistantSessionSummary {
  messages: AssistantSessionMessage[];
}

export async function listAssistantSessions(limit = 100) {
  const { data } = await api.get<{ total: number; items: AssistantSessionSummary[] }>(
    '/assistant/sessions',
    { params: { limit } },
  );
  return data;
}

export async function createAssistantSession(body: { title?: string; loop_id?: string | null }) {
  const { data } = await api.post<{ session: AssistantSession }>('/assistant/sessions', body);
  return data.session;
}

export async function getAssistantSession(sessionId: string) {
  const { data } = await api.get<{ session: AssistantSession }>(
    `/assistant/sessions/${encodeURIComponent(sessionId)}`,
  );
  return data.session;
}

export async function updateAssistantSession(sessionId: string, body: { title?: string; loop_id?: string | null }) {
  const { data } = await api.put<{ session: AssistantSession }>(
    `/assistant/sessions/${encodeURIComponent(sessionId)}`,
    body,
  );
  return data.session;
}

export async function deleteAssistantSession(sessionId: string) {
  const { data } = await api.delete<{ deleted: string }>(
    `/assistant/sessions/${encodeURIComponent(sessionId)}`,
  );
  return data;
}

export function assistantSessionStream(
  sessionId: string,
  body: { message: string; context: Record<string, unknown> },
  onEvent: (event: Record<string, unknown>) => void,
  onError?: (error: Error) => void,
): AbortController {
  const controller = new AbortController();
  fetch(`/api/assistant/sessions/${encodeURIComponent(sessionId)}/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: controller.signal,
  }).then(async (response) => {
    if (!response.ok) {
      throw new Error(`assistant session stream failed: ${response.status}`);
    }
    const reader = response.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split('\n\n');
      buffer = chunks.pop() || '';
      for (const chunk of chunks) {
        const text = chunk.replace(/^data: /, '').trim();
        if (!text) continue;
        try {
          onEvent(JSON.parse(text));
        } catch {
          // skip malformed SSE payloads
        }
      }
    }
  }).catch((error) => {
    if (!controller.signal.aborted) {
      onError?.(error as Error);
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

export async function getHistoryLoopAssessment(loopId: string, params?: HistoryTimeRangeParams) {
  const { data } = await api.get<HistoryLoopAssessment>(
    `/history/loops/${encodeURIComponent(loopId)}/assessment`,
    { params },
  );
  return data;
}

export interface HistoryLoopTuningPrior {
  loop_id: string;
  loop_type: string;
  start_time?: string | null;
  end_time?: string | null;
  features?: HistoryLoopFeatures;
  monitoring?: HistoryLoopMonitoring;
  assessment?: HistoryLoopAssessment;
  core_context?: Record<string, unknown>;
  ontology?: {
    source?: string;
    server_id?: string;
    server_name?: string;
    tool?: string;
    query?: string;
    content?: string;
    error?: string;
    [key: string]: unknown;
  };
  prompt: string;
  review?: string;
  advisory_only?: boolean;
  error?: string;
}

export async function getHistoryLoopTuningPrior(loopId: string, params?: HistoryTimeRangeParams) {
  const { data } = await api.get<HistoryLoopTuningPrior>(
    `/history/loops/${encodeURIComponent(loopId)}/tuning-prior`,
    { params },
  );
  return data;
}

export async function getHistoryLoopTuningPriorCore(loopId: string, params?: HistoryTimeRangeParams) {
  const { data } = await api.get<HistoryLoopTuningPrior>(
    `/history/loops/${encodeURIComponent(loopId)}/tuning-prior/core`,
    { params },
  );
  return data;
}

export async function getHistoryLoopTuningPriorOntology(loopId: string, params?: HistoryTimeRangeParams) {
  const { data } = await api.get<HistoryLoopTuningPrior>(
    `/history/loops/${encodeURIComponent(loopId)}/tuning-prior/ontology`,
    { params },
  );
  return data;
}

export async function reviewHistoryLoopTuningPrior(
  loopId: string,
  body: { core_context: Record<string, unknown>; ontology?: Record<string, unknown> | null },
) {
  const { data } = await api.post<HistoryLoopTuningPrior>(
    `/history/loops/${encodeURIComponent(loopId)}/tuning-prior/review`,
    body,
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
    mv_deadband_lagged_ratio?: number | null;
    mv_deadband_event_count?: number | null;
    mv_deadband_events_total?: number | null;
    mv_deadband_estimated_width?: number | null;
    mv_deadband_lag_used_s?: number | null;
    mv_deadband_step_threshold?: number | null;
    mv_hysteresis_ratio?: number | null;
    mv_hysteresis_hint?: boolean;
    mv_hysteresis_up_gain?: number | null;
    mv_hysteresis_down_gain?: number | null;
    mv_hysteresis_sample_count?: number | null;
    mv_stiction_hint?: boolean;
    mv_stiction_score?: number | null;
    mv_stuck_hint?: boolean;
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
  frequency_raw?: {
    pv_dominant_period_s?: number | null;
    pv_dominant_power_ratio?: number | null;
    pv_zero_crossing_per_hour?: number | null;
    [key: string]: unknown;
  };
  oscillation_raw?: {
    detected?: boolean;
    confidence?: number | null;
    pv_dominant_period_s?: number | null;
    pv_dominant_power_ratio?: number | null;
    pv_zero_crossing_per_hour?: number | null;
    phase_hint?: string | null;
    [key: string]: unknown;
  };
  performance_raw?: {
    harris_index?: number | null;
    harris_degradation_index?: number | null;
    harris_benchmark_ratio?: number | null;
    harris_actual_variance?: number | null;
    harris_min_variance_floor?: number | null;
    harris_error_basis?: string;
    cpk?: number | null;
    cpk_lsl?: number | null;
    cpk_usl?: number | null;
    cpk_basis?: string;
    oscillation_index?: number | null;
    oscillation_period_s?: number | null;
    oscillation_power_ratio?: number | null;
    oscillation_zero_crossing_per_hour?: number | null;
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

export interface HistoryTimeRangeParams {
  start_time?: string;
  end_time?: string;
}

export type HistoryRiskLevel = 'high' | 'medium' | 'low' | 'potential' | 'handled';

export interface HistoryRiskAlertRow {
  key: string;
  level: HistoryRiskLevel | string;
  type: string;
  type_label?: string;
  loop_id: string;
  loop_type?: string;
  asset?: string;
  description?: string;
  trigger?: string;
  duration?: string;
  risk_score?: number;
  found_at?: string | null;
  alert_count?: number;
  overall_score?: number | null;
  status?: string;
}

export interface HistoryRiskDistributionRow {
  label: string;
  value: number;
}

export interface HistoryRiskTrendRow {
  date: string;
  high?: number;
  medium?: number;
  low?: number;
  potential?: number;
  handled?: number;
}

export interface HistoryRiskAlertsResp {
  time_range: string;
  start_time?: string | null;
  end_time?: string | null;
  asset: string;
  total_loops: number;
  loaded_count: number;
  total_risk: number;
  counts: Record<string, number>;
  items: HistoryRiskAlertRow[];
  type_distribution: HistoryRiskDistributionRow[];
  asset_distribution: HistoryRiskDistributionRow[];
  trend: HistoryRiskTrendRow[];
  asset_options: Array<{ label: string; value: string }>;
  errors?: Array<Record<string, unknown>>;
}

export async function fetchHistoryRiskAlerts(params: {
  asset?: string;
  time_range?: string;
  start_time?: string;
  end_time?: string;
} = {}) {
  const { data } = await api.get<HistoryRiskAlertsResp>('/history/risk-alerts', { params });
  return data;
}

export async function fetchHistoryLoopFeatures(loopId: string, params?: HistoryTimeRangeParams) {
  const { data } = await api.get<HistoryLoopFeatures>(
    `/history/loops/${encodeURIComponent(loopId)}/features`,
    { params },
  );
  return data;
}

export async function fetchHistoryLoopMonitoring(loopId: string, params?: HistoryTimeRangeParams) {
  const { data } = await api.get<HistoryLoopMonitoring>(
    `/history/loops/${encodeURIComponent(loopId)}/monitoring`,
    { params },
  );
  return data;
}

export interface HistoryLoopHarris {
  loop_id: string;
  loop_type: string;
  start_time?: string | null;
  end_time?: string | null;
  success: boolean;
  reasoning?: string;
  warnings?: string[];
  harris?: {
    eta?: number | null;
    level?: string;
    confidence?: number | null;
    recommend_tuning?: boolean;
    deadtime_used?: {
      value_s?: number | null;
      samples?: number | null;
      source?: string;
      confidence?: number | null;
      attempts?: Array<Record<string, unknown>>;
      evidence?: Record<string, unknown>;
    };
    ar_model?: {
      order?: number | null;
      residual_acf_lag1?: number | null;
      residual_std?: number | null;
      aic?: number | null;
      provider?: string;
      bumped_max_order?: boolean;
    };
    variance_decomp?: {
      sigma2_y?: number | null;
      sigma2_mv?: number | null;
      sigma2_a?: number | null;
    };
    error_basis?: string;
    data_window?: {
      n_samples?: number | null;
      duration_s?: number | null;
      sp_change_ratio?: number | null;
    };
    threshold_used?: Record<string, number>;
    risk_flags?: string[];
    abort_reason?: string | null;
    confidence_components?: Record<string, number>;
    meta?: Record<string, unknown>;
    [key: string]: unknown;
  };
  error?: string;
}

export async function fetchHistoryLoopHarris(loopId: string, params?: HistoryTimeRangeParams & {
  error_basis?: string;
  force_deadtime_samples?: number;
  ar_order_override?: number;
}) {
  const { data } = await api.get<HistoryLoopHarris>(
    `/history/loops/${encodeURIComponent(loopId)}/harris`,
    { params },
  );
  return data;
}

export interface HistoryLoopCpk {
  loop_id: string;
  loop_type: string;
  start_time?: string | null;
  end_time?: string | null;
  success: boolean;
  cpk?: {
    value?: number | null;
    level?: string;
    status?: string;
    reason?: string;
    cpl?: number | null;
    cpu?: number | null;
    recommend_action?: string;
  };
  data_window?: {
    n_samples?: number | null;
    duration_s?: number | null;
    sample_time_s?: number | null;
    time_start?: string | null;
    time_end?: string | null;
  };
  limits?: {
    lsl?: number | null;
    usl?: number | null;
    source?: string;
    lsl_column?: string | null;
    usl_column?: string | null;
  };
  statistics?: {
    mean?: number | null;
    std?: number | null;
    min?: number | null;
    max?: number | null;
  };
  warnings?: string[];
  reasoning?: string;
  error?: string;
}

export async function fetchHistoryLoopCpk(loopId: string, params?: HistoryTimeRangeParams & { refresh_ontology?: boolean }) {
  const { data } = await api.get<HistoryLoopCpk>(
    `/history/loops/${encodeURIComponent(loopId)}/cpk`,
    { params },
  );
  return data;
}

export interface HistoryLoopOntologyRule {
  rule_id: string;
  title: string;
  status: 'pass' | 'warn' | 'blocked' | 'unknown' | string;
  severity?: string;
  blocking?: boolean;
  evidence?: unknown[];
  missing_fields?: string[];
  action?: string;
}

export interface HistoryLoopOntologyRules {
  loop_id: string;
  loop_type: string;
  start_time?: string | null;
  end_time?: string | null;
  rule_pack_version: string;
  ontology_facts?: Record<string, unknown>;
  rules: HistoryLoopOntologyRule[];
  summary?: {
    decision?: string;
    blocking_count?: number;
    warning_count?: number;
    advisory_only?: boolean;
    message?: string;
  };
  metrics?: Record<string, unknown>;
  error?: string;
}

export async function fetchHistoryLoopOntologyRules(loopId: string, params?: HistoryTimeRangeParams & { refresh_ontology?: boolean }) {
  const { data } = await api.get<HistoryLoopOntologyRules>(
    `/history/loops/${encodeURIComponent(loopId)}/ontology-rules`,
    { params },
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

export async function getHistoryLoopWindows(loopId: string, params?: HistoryTimeRangeParams) {
  const { data } = await api.get<{
    loop_id: string;
    loop_type: string;
    dt: number;
    total: number;
    usable_count: number;
    algorithm_summary?: Record<string, { total: number; usable: number }>;
    windows: HistoryWindow[];
    error?: string;
  }>(`/history/loops/${encodeURIComponent(loopId)}/windows`, { params });
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
    /** 提前结束流水线：
     *  "window_selection" — 数据分析菜单（跑到选窗为止）
     *  "identification"  — 系统辨识菜单（跑到辨识 + 精修结束）
     *  undefined         — 跑完全流程（整定菜单） */
    stop_after?: 'window_selection' | 'identification';
    /** 候选窗口算法白名单（如 ["sv_step", "mv_step"]）；不传或空数组表示不过滤。 */
    algorithm_filter?: string[];
    /** 可选本体上下文：窗口 LLM 顾问用它结合工艺先验判断窗口是否合理。 */
    ontology_context?: string;
    start_time?: string;
    end_time?: string;
    auto_tuning_task_id?: string;
  },
  onEvent: (event: Record<string, unknown>) => void,
): AbortController {
  const controller = new AbortController();
  const formData = new FormData();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null) return;
    // 数组以逗号拼接（后端 Form 字符串解析后 split 回去）
    if (Array.isArray(value)) {
      if (value.length === 0) return;
      formData.append(key, value.join(','));
      return;
    }
    formData.append(key, String(value));
  });

  fetch(`/api/history/loops/${encodeURIComponent(loopId)}/tune/stream`, {
    method: 'POST',
    body: formData,
    signal: controller.signal,
  }).then(async (response) => {
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      onEvent({
        type: 'error',
        stage: 'network',
        message: `SSE 请求失败：HTTP ${response.status}${text ? `，${text.slice(0, 200)}` : ''}`,
      });
      return;
    }
    const reader = response.body?.getReader();
    if (!reader) {
      onEvent({ type: 'error', stage: 'network', message: 'SSE 请求失败：后端没有返回事件流' });
      return;
    }
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
  }).catch((error) => {
    if (controller.signal.aborted) return;
    onEvent({ type: 'error', stage: 'network', message: `SSE 请求异常：${String(error)}` });
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

export interface PolicyConfig {
  loop_priors: {
    model_order: Record<string, string[]>;
    min_reasonable_t: Record<string, number>;
    reality_t_ranges: Record<string, { min: number; max: number }>;
  };
  refinement: {
    fallback_rule: {
      min_confidence: number;
      min_r2: number;
      min_window_quality: number;
      max_model_pool_size: number;
    };
    model_fallbacks: Record<string, string[]>;
  };
}

export async function fetchPolicyConfig() {
  const { data } = await api.get<PolicyConfig>('/policy-config');
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

// ── MCP 服务配置 ───────────────────────────────────────────────────────────────

export interface PromptConfig {
  assistant_system_prompt: string;
  assistant_developer_prompt: string;
  assistant_response_schema: string;
  window_policy_system_prompt: string;
  window_policy_user_prompt_template: string;
  identification_review_system_prompt: string;
  identification_review_user_prompt_template: string;
  updated_at: string;
}

export async function fetchPromptConfig() {
  const { data } = await api.get<PromptConfig>('/prompt-config');
  return data;
}

export async function updatePromptConfig(body: {
  assistant_system_prompt?: string | null;
  assistant_developer_prompt?: string | null;
  assistant_response_schema?: string | null;
  window_policy_system_prompt?: string | null;
  window_policy_user_prompt_template?: string | null;
  identification_review_system_prompt?: string | null;
  identification_review_user_prompt_template?: string | null;
}) {
  const { data } = await api.put<{ status: string; config: PromptConfig }>(
    '/prompt-config',
    body,
  );
  return data;
}

export async function resetPromptConfig() {
  const { data } = await api.post<{ status: string; config: PromptConfig }>(
    '/prompt-config/reset',
  );
  return data;
}

export interface SkillEffectInfo {
  key?: string;
  description?: string;
}

export interface SkillMetadataInfo {
  name: string;
  description: string;
  risk_level: 'low' | 'medium' | 'high' | string;
  preconditions: string[];
  effects: SkillEffectInfo[];
  stage: string;
  deterministic_gate: boolean;
}

export async function listSkills() {
  const { data } = await api.get<{
    total: number;
    items: SkillMetadataInfo[];
    summary: {
      stages: Record<string, number>;
      risks: Record<string, number>;
    };
  }>('/skills');
  return data;
}

// ── 实时评估快照 / 自动整定任务 ────────────────────────────────────────────────

export interface RealtimeMetricResult {
  name: string;
  value?: number | null;
  level?: string | null;
  confidence?: number | null;
  success?: boolean;
  raw?: Record<string, unknown>;
}

export interface RealtimeDiagnosisResult {
  diagnosis_id?: string;
  root_cause: string;
  confidence?: number | null;
  severity?: string;
  evidence?: unknown[];
  action?: string;
}

export interface RealtimeSkillTrace {
  trace_id?: string;
  skill_name: string;
  risk_level: string;
  status: string;
  inputs_summary?: Record<string, unknown>;
  outputs_summary?: Record<string, unknown>;
  guard?: Record<string, unknown>;
  duration_ms?: number;
  created_at?: string;
}

export interface RealtimeAssessmentSnapshot {
  snapshot_id: string;
  loop_id: string;
  asset_id: string;
  loop_type: string;
  created_at: string;
  time_window: {
    range: string;
    start_time?: string | null;
    end_time?: string | null;
  };
  risk_level: HistoryRiskLevel | 'normal' | string;
  score?: number | null;
  ontology?: {
    context_id?: string;
    case_id?: string | null;
    source?: string;
    pv_spec_limits?: Record<string, unknown>;
    relation_hints?: string[];
    missing_fields?: string[];
    facts?: Record<string, unknown>;
  };
  metrics?: RealtimeMetricResult[];
  monitoring?: HistoryLoopMonitoring;
  assessment?: HistoryLoopAssessment;
  diagnosis?: RealtimeDiagnosisResult[];
  decision?: {
    decision?: string;
    need_tuning?: boolean;
    blocked?: boolean;
    priority?: string;
    reason_codes?: string[];
    required_confirmations?: string[];
    summary?: string;
  };
  skill_trace?: RealtimeSkillTrace[];
}

export async function runRealtimeAssessment(body: {
  loop_ids?: string[] | null;
  asset_id?: string | null;
  time_range?: string;
  start_time?: string | null;
  end_time?: string | null;
  force_refresh?: boolean;
  include_formal_metrics?: boolean;
  auto_create_tasks?: boolean;
}) {
  const { data } = await api.post<{
    total: number;
    saved: number;
    items: RealtimeAssessmentSnapshot[];
    tasks?: AutoTuningTask[];
    errors: Array<Record<string, unknown>>;
  }>('/realtime-assessments/run', body);
  return data;
}

export async function fetchLatestRealtimeAssessments(params: {
  asset_id?: string;
  loop_id?: string;
  risk_level?: string;
  decision?: string;
  limit?: number;
} = {}) {
  const { data } = await api.get<{
    total: number;
    items: RealtimeAssessmentSnapshot[];
    summary: {
      risk_levels: Record<string, number>;
      decisions: Record<string, number>;
      assets: Record<string, number>;
    };
  }>('/realtime-assessments/latest', { params });
  return data;
}

export async function fetchRealtimeAssessment(snapshotId: string) {
  const { data } = await api.get<RealtimeAssessmentSnapshot>(
    `/realtime-assessments/${encodeURIComponent(snapshotId)}`,
  );
  return data;
}

export async function createAutoTuningTaskFromAssessment(snapshotId: string, body: {
  confirm?: boolean;
  trigger_mode?: string;
  reason?: string | null;
} = {}) {
  const { data } = await api.post<{ task: Record<string, unknown> }>(
    `/realtime-assessments/${encodeURIComponent(snapshotId)}/tuning-task`,
    body,
  );
  return data;
}

export interface AutoTuningTask {
  task_id: string;
  snapshot_id: string;
  loop_id: string;
  asset_id: string;
  status: 'pending_review' | 'pending' | 'blocked' | 'running' | 'completed' | 'failed' | string;
  trigger_mode: string;
  trigger_reason?: string | null;
  created_at: string;
  updated_at: string;
  assessment_decision?: Record<string, unknown>;
  source_snapshot?: Record<string, unknown>;
  result?: Record<string, unknown>;
}

export interface PreparedAutoTuningTask {
  task: AutoTuningTask;
  guard: {
    allowed: boolean;
    blocked?: boolean;
    requires_confirmation?: boolean;
    reason?: string;
  };
  tuning_request: {
    loop_id?: string;
    loop_type?: string;
    loop_name?: string;
    start_time?: string | null;
    end_time?: string | null;
    selected_window_index?: number | null;
    use_llm_advisor?: boolean;
    ontology_context?: Record<string, unknown>;
  };
}

export async function listAutoTuningTasks(params: {
  status?: string;
  loop_id?: string;
  asset_id?: string;
  limit?: number;
} = {}) {
  const { data } = await api.get<{ total: number; items: AutoTuningTask[] }>(
    '/auto-tuning/tasks',
    { params },
  );
  return data;
}

export async function prepareAutoTuningTask(taskId: string, body: {
  confirm?: boolean;
  use_llm_advisor?: boolean;
  selected_window_index?: number | null;
} = {}) {
  const { data } = await api.post<PreparedAutoTuningTask>(
    `/auto-tuning/tasks/${encodeURIComponent(taskId)}/prepare`,
    body,
  );
  return data;
}

export interface RealtimeMonitorConfig {
  config_id: string;
  enabled: boolean;
  asset_id?: string | null;
  loop_ids: string[];
  time_range: string;
  interval_seconds: number;
  include_formal_metrics: boolean;
  auto_create_tasks: boolean;
  updated_at?: string;
  last_run_at?: string;
  last_result?: Record<string, unknown>;
}

export async function getRealtimeMonitorConfig() {
  const { data } = await api.get<RealtimeMonitorConfig>('/realtime-monitor/config');
  return data;
}

export async function updateRealtimeMonitorConfig(body: Partial<RealtimeMonitorConfig>) {
  const { data } = await api.put<RealtimeMonitorConfig>('/realtime-monitor/config', body);
  return data;
}

export async function runRealtimeMonitorTick(body: { force?: boolean } = {}) {
  const { data } = await api.post<{
    status: 'completed' | 'skipped' | string;
    reason?: string;
    config: RealtimeMonitorConfig;
    result?: {
      total: number;
      saved: number;
      items: RealtimeAssessmentSnapshot[];
      tasks?: AutoTuningTask[];
      errors: Array<Record<string, unknown>>;
    };
  }>('/realtime-monitor/tick', body);
  return data;
}

export async function getRealtimeMonitorScheduler() {
  const { data } = await api.get<{
    running: boolean;
    poll_seconds: number;
    config: RealtimeMonitorConfig;
  }>('/realtime-monitor/scheduler');
  return data;
}

export type McpTransport = 'stdio' | 'sse' | 'streamable-http';

export interface McpServerConfig {
  id: string;
  name: string;
  url: string;
  transport: McpTransport;
  raw_json: string;
  enabled: boolean;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface McpServerPayload {
  name: string;
  url?: string;
  transport: McpTransport;
  raw_json?: string;
  enabled?: boolean;
  description?: string;
}

export interface McpOntologyQueryConfig {
  query_template: string;
  updated_at: string;
  placeholders: Array<{ name: string; description: string }>;
  preview: string;
}

export async function listMcpServers() {
  const { data } = await api.get<{ total: number; items: McpServerConfig[] }>(
    '/mcp-servers',
  );
  return data;
}

export async function fetchMcpOntologyQueryConfig() {
  const { data } = await api.get<McpOntologyQueryConfig>('/mcp-ontology-query-config');
  return data;
}

export async function updateMcpOntologyQueryConfig(body: { query_template?: string | null }) {
  const { data } = await api.put<{ status: string; config: McpOntologyQueryConfig }>(
    '/mcp-ontology-query-config',
    body,
  );
  return data;
}

export async function resetMcpOntologyQueryConfig() {
  const { data } = await api.post<{ status: string; config: McpOntologyQueryConfig }>(
    '/mcp-ontology-query-config/reset',
  );
  return data;
}

export async function createMcpServer(body: McpServerPayload) {
  const { data } = await api.post<{ status: string; server: McpServerConfig }>(
    '/mcp-servers',
    body,
  );
  return data;
}

export async function updateMcpServer(
  id: string,
  body: Partial<McpServerPayload>,
) {
  const { data } = await api.put<{ status: string; server: McpServerConfig }>(
    `/mcp-servers/${encodeURIComponent(id)}`,
    body,
  );
  return data;
}

export async function deleteMcpServer(id: string) {
  const { data } = await api.delete<{ status: string; deleted: string }>(
    `/mcp-servers/${encodeURIComponent(id)}`,
  );
  return data;
}

export async function testMcpServer(id: string) {
  const { data } = await api.post<{ status: string; message: string }>(
    `/mcp-servers/${encodeURIComponent(id)}/test`,
  );
  return data;
}

export interface McpToolInfo {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  outputSchema?: Record<string, unknown>;
  [key: string]: unknown;
}

export async function listMcpServerTools(id: string) {
  const { data } = await api.get<{ server_id: string; total: number; items: McpToolInfo[] }>(
    `/mcp-servers/${encodeURIComponent(id)}/tools`,
  );
  return data;
}

export async function callMcpServerTool(
  id: string,
  toolName: string,
  argumentsPayload: Record<string, unknown>,
) {
  const { data } = await api.post<{
    status: string;
    server_id: string;
    tool: string;
    result: Record<string, unknown>;
  }>(
    `/mcp-servers/${encodeURIComponent(id)}/tools/${encodeURIComponent(toolName)}/call`,
    { arguments: argumentsPayload },
  );
  return data;
}

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
