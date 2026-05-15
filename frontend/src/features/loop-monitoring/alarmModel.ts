import type { HistoryLoopAssessment, HistoryLoopMonitoring } from '@/services/api';
import type { TaskStatus } from '@/features/tuning-task/model';

export interface RailAlarmRow {
  key: string;
  time: string;
  level: string;
  name: string;
  value: string;
  status: string;
  recommendation: string;
  evidence: string;
}

export function buildRailAlarms({
  assessment,
  dataSourceType,
  loopMonitoring,
  taskId,
  taskStartedAt,
  taskStatus,
  monitoringStatusText,
  scorePercent,
}: {
  assessment: HistoryLoopAssessment | null;
  dataSourceType: string;
  loopMonitoring: HistoryLoopMonitoring | null;
  taskId?: string;
  taskStartedAt?: string;
  taskStatus: TaskStatus;
  monitoringStatusText: (status?: string) => string;
  scorePercent: (value?: number) => number;
}): RailAlarmRow[] {
  const monitoringEvents = loopMonitoring?.monitoring.events ?? [];
  if (monitoringEvents.length) {
    return monitoringEvents.map((event, index) => ({
      key: `monitoring-event-${index}`,
      time: '当前',
      level: event.severity || '提示',
      name: event.name || event.type || 'monitoring',
      value: event.message,
      status: event.status || 'new',
      recommendation: event.recommendation || '',
      evidence: event.evidence ? JSON.stringify(event.evidence) : '',
    }));
  }

  const monitoringAlerts = loopMonitoring?.monitoring.alerts ?? [];
  if (monitoringAlerts.length) {
    return monitoringAlerts.map((alert, index) => ({
      key: `monitoring-${index}`,
      time: '当前',
      level: alert.severity || '提示',
      name: alert.type || 'monitoring',
      value: alert.message,
      status: monitoringStatusText(loopMonitoring?.monitoring.status),
      recommendation: '',
      evidence: '',
    }));
  }

  const flags = assessment?.diagnostics.flags ?? [];
  if (flags.length) {
    return flags.map((flag, index) => ({
      key: `flag-${index}`,
      time: '当前',
      level: flag.severity || '提示',
      name: flag.type,
      value: flag.message,
      status: '待确认',
      recommendation: '',
      evidence: '',
    }));
  }

  return [
    {
      key: 'monitoring',
      time: '当前',
      level: '中',
      name: '监控分',
      value: loopMonitoring?.monitoring?.overall_score === undefined
        ? '等待监控快照'
        : `${scorePercent(loopMonitoring.monitoring.overall_score)}%`,
      status: '跟踪',
      recommendation: '',
      evidence: '',
    },
    {
      key: 'source',
      time: '当前',
      level: '低',
      name: '数据源',
      value: dataSourceType === 'history' ? '历史文件导入' : '历史仓库/实时库',
      status: '正常',
      recommendation: '',
      evidence: '',
    },
    {
      key: 'task',
      time: taskStartedAt ? new Date(taskStartedAt).toLocaleTimeString() : '未启动',
      level: taskStatus === 'error' ? '高' : '低',
      name: '整定任务',
      value: taskId ? `任务 ${taskId}` : '暂无运行任务',
      status: taskStatus === 'done' ? '完成' : taskStatus === 'running' ? '运行' : '空闲',
      recommendation: '',
      evidence: '',
    },
  ];
}
