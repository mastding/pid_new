import type { ReactNode } from 'react';

export type DashboardWidgetKey =
  | 'kpi_total'
  | 'kpi_loaded'
  | 'kpi_normal'
  | 'kpi_warning'
  | 'kpi_alarm'
  | 'kpi_score'
  | 'kpi_alerts'
  | 'health'
  | 'asset'
  | 'type'
  | 'metrics'
  | 'top'
  | 'abnormal'
  | 'alerts'
  | 'trend'
  | 'snapshot'
  | 'quick';

export const DASHBOARD_KPI_WIDGET_KEYS: DashboardWidgetKey[] = [
  'kpi_total',
  'kpi_loaded',
  'kpi_normal',
  'kpi_warning',
  'kpi_alarm',
  'kpi_score',
  'kpi_alerts',
];

export const DASHBOARD_WIDGET_OPTIONS: Array<{ label: string; value: DashboardWidgetKey }> = [
  { label: '回路总数', value: 'kpi_total' },
  { label: '已监控回路', value: 'kpi_loaded' },
  { label: '正常回路', value: 'kpi_normal' },
  { label: '关注回路', value: 'kpi_warning' },
  { label: '告警回路', value: 'kpi_alarm' },
  { label: '监控均分', value: 'kpi_score' },
  { label: '监控告警', value: 'kpi_alerts' },
  { label: '回路健康分布', value: 'health' },
  { label: '回路类型分布', value: 'type' },
  { label: '关键指标均值', value: 'metrics' },
  { label: '告警统计', value: 'alerts' },
  { label: '性能评分 TOP5', value: 'top' },
  { label: '异常回路列表', value: 'abnormal' },
  { label: '选中回路趋势', value: 'trend' },
  { label: '快捷操作', value: 'quick' },
  { label: '选中回路快照', value: 'snapshot' },
  { label: '回路按装置分布', value: 'asset' },
];

export const ALL_DASHBOARD_WIDGET_KEYS = DASHBOARD_WIDGET_OPTIONS.map((item) => item.value);
export const DEFAULT_DASHBOARD_WIDGET_KEYS = ALL_DASHBOARD_WIDGET_KEYS.filter((item) => item !== 'kpi_warning');
export const DASHBOARD_WIDGET_KEY_SET = new Set<DashboardWidgetKey>(ALL_DASHBOARD_WIDGET_KEYS);
export const DASHBOARD_WIDGET_STORAGE_KEY = 'pid_v2_dashboard_widgets';

export type DashboardWidgetDefinition = {
  title: string;
  className: string;
  content: ReactNode;
  weight: number;
  minWidth: number;
};

export type DashboardLoopRow = {
  loop: {
    loop_id: string;
    loop_type?: string;
  };
  snapshot?: {
    overall_score?: number;
    status?: string;
    alerts?: Array<{
      type?: string;
      severity?: string;
    }>;
  };
  alertCount: number;
};

export type DashboardSliceInput = {
  label: string;
  value: number;
  color: string;
};

export type DashboardSlice = DashboardSliceInput & {
  percent: number;
};

export function makeDashboardSlices(items: DashboardSliceInput[]): DashboardSlice[] {
  const total = items.reduce((sum, item) => sum + item.value, 0);
  return items.map((item) => ({ ...item, percent: total > 0 ? item.value / total : 0 }));
}

export function dashboardConicGradient(items: Array<{ value: number; color: string }>) {
  const total = items.reduce((sum, item) => sum + item.value, 0);
  if (!total) return 'conic-gradient(#26364d 0 100%)';
  let cursor = 0;
  return `conic-gradient(${items.map((item) => {
    const start = cursor;
    cursor += (item.value / total) * 100;
    return `${item.color} ${start}% ${cursor}%`;
  }).join(', ')})`;
}

export const normalizeDashboardWidgetKeys = (input: unknown): DashboardWidgetKey[] => {
  const items = Array.isArray(input) ? input : [];
  const normalized = items.flatMap((item) => (item === 'kpis' ? DASHBOARD_KPI_WIDGET_KEYS : [item]));
  const result: DashboardWidgetKey[] = [];
  normalized.forEach((item) => {
    if (DASHBOARD_WIDGET_KEY_SET.has(item as DashboardWidgetKey) && !result.includes(item as DashboardWidgetKey)) {
      result.push(item as DashboardWidgetKey);
    }
  });
  const isLegacyDefault =
    result.length === ALL_DASHBOARD_WIDGET_KEYS.length &&
    ALL_DASHBOARD_WIDGET_KEYS.every((item) => result.includes(item));
  if (isLegacyDefault) return DEFAULT_DASHBOARD_WIDGET_KEYS;
  return result.length ? result : DEFAULT_DASHBOARD_WIDGET_KEYS;
};
