import type { IdentificationAttempt, PipelineEvent } from '@/types/tuning';

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
