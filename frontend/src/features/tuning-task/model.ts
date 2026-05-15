import type {
  IdentificationAttempt,
  IdentificationRefinementMeta,
  LlmThinkingEvent,
  PipelineEvent,
  TuningResult,
  WindowAlgorithmFitSummary,
} from '@/types/tuning';

export const TUNING_STAGE_KEYS: string[] = [
  'data_analysis',
  'ontology_policy',
  'window_selection',
  'identification',
  'model_review',
  'identification_refinement',
  'tuning',
  'evaluation',
];

export const TUNING_STAGE_LABELS: Record<string, string> = {
  data_analysis: '数据分析',
  ontology_policy: '本体策略',
  window_selection: '窗口选择',
  identification: '模型辨识',
  model_review: '大模型评审',
  identification_refinement: '精修建议',
  tuning: 'PID 整定',
  evaluation: '性能评估',
};

export type TaskStatus = 'idle' | 'running' | 'done' | 'error';

export interface TaskEventLog {
  id: number;
  label: string;
  detail?: string;
}

export type TaskStageStatusMap = Record<string, 'running' | 'done'>;
export type TaskStageDataMap = Record<string, Record<string, unknown>>;

export interface TaskStageCardModel {
  stage: string;
  index: number;
  status: 'done' | 'running' | 'wait';
  label: string;
  summary: string;
  isCurrent: boolean;
}

function formatStageNumber(value?: number | null, digits = 2) {
  return value === null || value === undefined || Number.isNaN(value) ? '-' : value.toFixed(digits);
}

export function summarizeTaskStage(stage: string, data?: Record<string, unknown>) {
  if (!data) return '等待执行';
  if (stage === 'data_analysis') {
    const dt = data.sampling_time as number | undefined;
    const dtTxt = typeof dt === 'number' ? `${dt}s` : '-';
    const profile = data.data_profile as { data_profile?: { duration_h?: number } } | undefined;
    const dur = profile?.data_profile?.duration_h;
    const durTxt = typeof dur === 'number' ? ` / ${dur.toFixed(1)} h` : '';
    return `${data.data_points ?? '-'} 点 / 采样 ${dtTxt}${durTxt}`;
  }
  if (stage === 'ontology_policy') {
    const src = data.source as string | undefined;
    const ontologySrc = data.ontology_source as string | undefined;
    const conf = typeof data.confidence === 'number' ? `${Math.round(data.confidence * 100)}%` : '-';
    const srcLabel = src === 'llm' ? '大模型改写策略' : src === 'default' ? '默认策略' : src ?? '-';
    const ontLabel = ontologySrc === 'mcp' ? '本体上下文已注入'
      : ontologySrc === 'frontend' ? '前端注入'
      : ontologySrc === 'default' ? '无本体上下文'
      : ontologySrc ?? '-';
    return `${srcLabel} · ${ontLabel} · 置信度 ${conf}`;
  }
  if (stage === 'window_selection') {
    const mode = data.mode as string | undefined;
    const modeLabel = mode === 'llm' ? '大模型选窗'
      : mode === 'fallback_deterministic' ? '大模型失败后回退确定性'
      : mode === 'deterministic' ? '确定性选窗'
      : mode === 'user_override' ? '工程师手动指定'
      : mode === 'blocked' ? '阻断'
      : mode ?? '-';
    const candidates = (data.candidate_window_count as number | undefined)
      ?? (data.policy_adjusted_candidate_windows as number | undefined);
    const usable = data.policy_adjusted_usable_windows as number | undefined;
    const agreedRaw = data.agreed_with_deterministic;
    const agreed = typeof agreedRaw === 'boolean' ? (agreedRaw ? '与算法一致' : '与算法分歧') : null;
    const evidenceCount = Array.isArray(data.ontology_evidence) ? (data.ontology_evidence as unknown[]).length : 0;
    const judgementCount = Array.isArray(data.window_judgements) ? (data.window_judgements as unknown[]).length : 0;
    const parts: string[] = [];
    if (candidates !== undefined) parts.push(`候选 ${candidates} / 可用 ${usable ?? '-'}`);
    parts.push(`${modeLabel} #${data.chosen_index ?? '-'}（算法 #${data.deterministic_index ?? '-'}）`);
    if (agreed) parts.push(agreed);
    if (evidenceCount) parts.push(`本体证据 ${evidenceCount} 项`);
    if (judgementCount) parts.push(`窗口判断 ${judgementCount} 项`);
    return parts.join(' · ');
  }
  if (stage === 'identification') {
    const r2 = typeof data.r2_score === 'number' ? data.r2_score.toFixed(3) : '-';
    const confidence = typeof data.confidence === 'number' ? `${Math.round(data.confidence * 100)}%` : '-';
    return `R${data.round ?? 0} ${data.model_type ?? '-'} / R²=${r2} / 置信度 ${confidence}`;
  }
  if (stage === 'model_review') {
    return `${data.verdict ?? '-'}：${data.reason ?? ''}`;
  }
  if (stage === 'identification_refinement') {
    return `${data.retry ? '继续重试' : '停止重试'}：${data.rationale ?? ''}`;
  }
  if (stage === 'tuning') {
    return `${data.strategy ?? '-'} / Kp=${formatStageNumber(data.Kp as number | undefined, 3)}`;
  }
  if (stage === 'evaluation') {
    return `${data.passed ? '可上线' : '需优化'} / 综合 ${formatStageNumber(data.final_rating as number | undefined, 1)}`;
  }
  return '已完成';
}

export function buildTaskStageCards({
  taskStatus,
  taskCurrentStage,
  taskStageStatus,
  taskStageData,
}: {
  taskStatus: TaskStatus;
  taskCurrentStage?: string;
  taskStageStatus: TaskStageStatusMap;
  taskStageData: TaskStageDataMap;
}): TaskStageCardModel[] {
  return TUNING_STAGE_KEYS.map((stage, index) => {
    const rawStatus = taskStageStatus[stage];
    const derivedStatus: TaskStageCardModel['status'] = taskStatus === 'done' || rawStatus === 'done' || taskStageData[stage]
      ? 'done'
      : rawStatus === 'running'
        ? 'running'
        : 'wait';
    return {
      stage,
      index,
      status: derivedStatus,
      label: TUNING_STAGE_LABELS[stage],
      summary: derivedStatus === 'running' ? '运行中...' : summarizeTaskStage(stage, taskStageData[stage]),
      isCurrent: taskCurrentStage === stage && taskStatus === 'running',
    };
  });
}

function objectPayload(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

export function prependTaskEventLog(prev: TaskEventLog[], event: PipelineEvent): TaskEventLog[] {
  return [
    {
      id: Date.now() + Math.random(),
      label: eventLabel(event),
      detail: event.type === 'stage' && event.data ? JSON.stringify(event.data) : undefined,
    },
    ...prev,
  ].slice(0, 30);
}

export function mergeRunningStageData(
  prev: TaskStageDataMap,
  stage: string,
  data: unknown,
): TaskStageDataMap {
  return {
    ...prev,
    [stage]: {
      ...(prev[stage] ?? {}),
      ...objectPayload(data),
    },
  };
}

export function clearRunningStageData(prev: TaskStageDataMap, stage: string): TaskStageDataMap {
  if (!(stage in prev)) return prev;
  const next = { ...prev };
  delete next[stage];
  return next;
}

export function mergeDoneStageData(
  prev: TaskStageDataMap,
  stage: string,
  data: unknown,
): TaskStageDataMap {
  return {
    ...prev,
    [stage]: {
      ...(prev[stage] ?? {}),
      ...objectPayload(data),
    },
  };
}

export function upsertRefinement(
  prev: IdentificationRefinementMeta[],
  refinement: IdentificationRefinementMeta,
): IdentificationRefinementMeta[] {
  return [
    ...prev.filter((item) => item.round !== refinement.round),
    refinement,
  ].sort((a, b) => a.round - b.round);
}

export function mergeIdentificationAttempts(
  prev: IdentificationAttempt[],
  data: unknown,
): IdentificationAttempt[] {
  const payload = objectPayload(data);
  const round = typeof payload.round === 'number' ? payload.round : 0;
  const attempts = ((payload.attempts as IdentificationAttempt[] | undefined) ?? []).map((attempt) => ({
    ...attempt,
    round: typeof attempt.round === 'number' ? attempt.round : round,
  }));
  return [
    ...prev.filter((attempt) => (attempt.round ?? 0) !== round),
    ...attempts,
  ].sort((a, b) => {
    const roundDiff = (a.round ?? 0) - (b.round ?? 0);
    if (roundDiff !== 0) return roundDiff;
    return (b.fit_score ?? -1e12) - (a.fit_score ?? -1e12);
  });
}

export function upsertThinkingEvent(
  prev: LlmThinkingEvent[],
  event: LlmThinkingEvent,
): LlmThinkingEvent[] {
  return [
    ...prev.filter((item) => !(item.stage === event.stage && item.round === event.round)),
    event,
  ];
}

export function eventLabel(event: PipelineEvent) {
  if (event.type === 'stage') {
    const stageLabel = TUNING_STAGE_LABELS[event.stage] ?? event.stage;
    return `${stageLabel} ${event.status === 'running' ? '运行中' : '完成'}`;
  }
  if (event.type === 'llm_thinking') {
    return `${TUNING_STAGE_LABELS[event.stage] ?? event.stage} 大模型思考`;
  }
  if (event.type === 'session_start') return `任务已创建：${event.task_id}`;
  if (event.type === 'result') return '完整结果已返回';
  if (event.type === 'error') return `流程异常：${event.message}`;
  return '流程结束';
}

export function formatEventDetail(detail?: string) {
  if (!detail) return '';
  try {
    return JSON.stringify(JSON.parse(detail), null, 2);
  } catch {
    return detail;
  }
}

export function attemptFitKey(attempt: IdentificationAttempt) {
  return [
    attempt.round ?? 0,
    attempt.window_source ?? '',
    attempt.window_algorithm ?? '',
    attempt.model_type ?? '',
    attempt.fit_score ?? '',
  ].join('|');
}

export function getTaskAlgorithmComparison(
  taskStageData: TaskStageDataMap,
  taskResult: TuningResult | null,
): WindowAlgorithmFitSummary[] {
  const identificationStage = taskStageData.identification ?? {};
  const stageComparison = identificationStage.algorithm_comparison;
  const resultComparison = taskResult?.model?.algorithm_comparison;
  const source = Array.isArray(stageComparison) ? stageComparison : resultComparison;
  return Array.isArray(source) ? source as WindowAlgorithmFitSummary[] : [];
}

export function getFitPreviewAttempts(
  taskAttempts: IdentificationAttempt[],
  taskResult: TuningResult | null,
): IdentificationAttempt[] {
  const attemptsWithPreview = taskAttempts
    .filter((attempt) => attempt.success && !!attempt.fit_preview?.points?.length)
    .sort((a, b) => {
      const roundDiff = (b.round ?? 0) - (a.round ?? 0);
      if (roundDiff) return roundDiff;
      return (b.fit_score ?? -9999) - (a.fit_score ?? -9999);
    });
  if (attemptsWithPreview.length) return attemptsWithPreview;

  const model = taskResult?.model;
  if (!model?.fit_preview?.points?.length) return [];
  return [{
    success: true,
    round: 0,
    model_type: model.model_type,
    window_source: model.window_source,
    K: model.K,
    T: model.T,
    T1: model.T1,
    T2: model.T2,
    L: model.L,
    r2_score: model.r2_score,
    normalized_rmse: model.normalized_rmse,
    confidence: model.confidence,
    fit_preview: model.fit_preview,
  }];
}

export function getSelectedFitAttempt(
  attempts: IdentificationAttempt[],
  selectedKey?: string,
) {
  if (!attempts.length) return undefined;
  return attempts.find((attempt) => attemptFitKey(attempt) === selectedKey) ?? attempts[0];
}

export function buildFitPreviewChartData(attempt?: IdentificationAttempt) {
  const points = attempt?.fit_preview?.points ?? [];
  return points.flatMap((point) => {
    const x = point.time ?? point.index;
    return [
      { t: x, value: point.pv, series: 'PV 实测' },
      { t: x, value: point.pv_fit, series: 'PV 仿真' },
      { t: x, value: point.mv, series: 'MV' },
    ];
  });
}

export function getDeterministicRefinement(refinements: IdentificationRefinementMeta[]) {
  return [...refinements].reverse().find((item) => item.source === 'deterministic_algorithm_policy');
}
