import type { ModuleKey, SubKey } from '@/features/app-shell/navigation';

export interface AssistantAction {
  label: string;
  target: ModuleKey;
  sub: SubKey;
  loopId?: string;
}

export interface AssistantEventItem {
  id: string;
  type: string;
  title: string;
  detail?: string;
}

export interface AssistantMessage {
  id: number;
  role: 'user' | 'assistant';
  text: string;
  reasoning?: string;
  loading?: boolean;
  error?: string;
  actions?: AssistantAction[];
  eventLog?: AssistantEventItem[];
}

export function buildDialogueActions(loopId?: string | null): AssistantAction[] {
  return [
    { label: '查看趋势与频谱', target: 'monitor', sub: 'trend_spectrum', loopId: loopId || undefined },
    { label: '生成整定先验', target: 'tuning', sub: 'tuning_prior', loopId: loopId || undefined },
    { label: '进入整定任务', target: 'tuning', sub: 'tuning_task', loopId: loopId || undefined },
  ];
}

export function normalizeAssistantAction(value: string, loopId?: string | null): AssistantAction | null {
  const raw = value.trim();
  const text = raw.replace(/^[-•\d.\s]+/, '');
  if (!text) return null;
  const startsLikeAction = /^[-•\d.\s]*(进入|查看|打开|前往|跳转|发起|建议进入|建议查看)/.test(raw);
  if (!startsLikeAction || text.length > 32) return null;
  if (text.includes('趋势') || text.includes('频谱')) {
    return { label: text, target: 'monitor', sub: 'trend_spectrum', loopId: loopId || undefined };
  }
  if (text.includes('画像')) {
    return { label: text, target: 'monitor', sub: 'loop_profile', loopId: loopId || undefined };
  }
  if (text.includes('先验')) {
    return { label: text, target: 'tuning', sub: 'tuning_prior', loopId: loopId || undefined };
  }
  if (text.includes('窗口')) {
    return { label: text, target: 'tuning', sub: 'id_windows', loopId: loopId || undefined };
  }
  if (text.includes('整定任务') || text.includes('整定页面') || text.includes('发起整定') || text.includes('进入整定')) {
    return { label: text, target: 'tuning', sub: 'tuning_task', loopId: loopId || undefined };
  }
  return null;
}

export function formatAssistantEvent(event: Record<string, unknown>): AssistantEventItem | null {
  const type = String(event.type || '');
  if (!type || type === 'answer_delta' || type === 'done') return null;
  if (type === 'thinking_step' || type === 'reasoning_delta') {
    const content = String(event.content || '').trim();
    return {
      id: `${Date.now()}-${Math.random()}`,
      type,
      title: content || '模型正在结合会话历史、当前回路和可用监控指标生成判断。',
    };
  }
  if (type === 'tool_event') {
    return {
      id: `${Date.now()}-${Math.random()}`,
      type,
      title: String(event.name || 'tool'),
      detail: `状态：${String(event.status || 'ok')}`,
    };
  }
  if (type === 'error') {
    return {
      id: `${Date.now()}-${Math.random()}`,
      type,
      title: '调用异常',
      detail: String(event.message || ''),
    };
  }
  return {
    id: `${Date.now()}-${Math.random()}`,
    type,
    title: type,
    detail: JSON.stringify(event),
  };
}
