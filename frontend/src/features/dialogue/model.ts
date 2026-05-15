import type { ModuleKey, SubKey } from '@/features/app-shell/navigation';
import type { DashboardLoopRow, DashboardStats } from '@/features/dashboard/model';
import type {
  AssistantSession,
  HistoryLoop,
  HistoryLoopAssessment,
  HistoryLoopFeatures,
  HistoryLoopMonitoring,
  HistoryLoopMonitoringSnapshot,
} from '@/services/api';

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

export function buildAssistantContext({
  activeModule,
  activeSub,
  assessment,
  currentSubLabel,
  dashboardRows,
  dashboardStats,
  loopFeatures,
  loopMonitoring,
  monitoringByLoopId,
  scopedLoopCount,
  selectedLoop,
  selectedLoopId,
}: {
  activeModule: ModuleKey;
  activeSub: SubKey;
  assessment: HistoryLoopAssessment | null;
  currentSubLabel: string;
  dashboardRows: DashboardLoopRow[];
  dashboardStats: DashboardStats;
  loopFeatures: HistoryLoopFeatures | null;
  loopMonitoring: HistoryLoopMonitoring | null;
  monitoringByLoopId: Record<string, HistoryLoopMonitoring>;
  scopedLoopCount: number;
  selectedLoop?: HistoryLoop;
  selectedLoopId?: string;
}) {
  const riskRows = dashboardRows
    .filter((row) => row.alertCount > 0 || row.snapshot?.status === 'warning' || row.snapshot?.status === 'alarm' || row.snapshot?.status === 'critical')
    .slice(0, 8)
    .map((row) => ({
      loop_id: row.loop.loop_id,
      loop_type: row.loop.loop_type,
      status: row.snapshot?.status,
      overall_score: row.snapshot?.overall_score,
      alerts: row.snapshot?.alerts,
      events: row.snapshot?.events,
    }));

  const selectedMonitoring: HistoryLoopMonitoringSnapshot | null =
    loopMonitoring?.monitoring ?? (selectedLoopId ? monitoringByLoopId[selectedLoopId]?.monitoring : null);

  return {
    loop_id: selectedLoop?.loop_id ?? selectedLoopId ?? null,
    start_time: selectedLoop?.start_time ?? null,
    end_time: selectedLoop?.end_time ?? null,
    page: { module: activeModule, sub: activeSub, title: currentSubLabel },
    scope: {
      loop_count: scopedLoopCount,
      avg_score: dashboardStats.avgScore,
      normal_count: dashboardStats.normalCount,
      warning_count: dashboardStats.warningCount,
      alarm_count: dashboardStats.alarmCount,
    },
    selected_loop: selectedLoop ? {
      loop_id: selectedLoop.loop_id,
      loop_type: selectedLoop.loop_type,
      start_time: selectedLoop.start_time,
      end_time: selectedLoop.end_time,
    } : null,
    selected_monitoring: selectedMonitoring,
    selected_features: loopFeatures,
    selected_assessment: assessment,
    risk_loops: riskRows,
    safety_rules: [
      '不要直接执行整定、窗口候选或 PID 参数修改。',
      '需要操作时只输出建议动作，由用户点击后进入对应页面确认。',
      '缺少上下文时必须明确说明。',
    ],
  };
}

export function mapAssistantSessionMessages(session: AssistantSession | null): AssistantMessage[] {
  if (!session?.messages?.length) return [];
  return session.messages.map((item, index) => ({
    id: Number(`${index + 1}${String(item.created_at || '').replace(/\D/g, '').slice(-6)}`) || Date.now() + index,
    role: item.role,
    text: item.content,
    reasoning: item.reasoning_summary,
    loading: false,
    actions: item.role === 'assistant' ? buildDialogueActions(session.loop_id) : undefined,
    eventLog: item.role === 'assistant'
      ? (item.raw_events ?? []).map((event) => formatAssistantEvent(event)).filter(Boolean) as AssistantEventItem[]
      : undefined,
  }));
}
