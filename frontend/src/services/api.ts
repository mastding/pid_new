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

export interface HistoryWindowPreviewPoint {
  t: string | number;
  pv: number;
  mv: number;
}

export interface HistoryWindow {
  index: number;
  source: string;
  type: string;
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
  preview: HistoryWindowPreviewPoint[];
}

export async function getHistoryLoopWindows(loopId: string) {
  const { data } = await api.get<{
    loop_id: string;
    loop_type: string;
    dt: number;
    total: number;
    usable_count: number;
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
