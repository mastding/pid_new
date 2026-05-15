import type { ReactNode } from 'react';
import dayjs from 'dayjs';

import type { HistoryLoop, HistoryLoopMonitoring } from '@/services/api';
import { DashboardAbnormalLoopsWidget } from '@/features/dashboard/DashboardAbnormalLoopsWidget';
import { DashboardAlertStatsWidget } from '@/features/dashboard/DashboardAlertStatsWidget';
import { DashboardBarsWidget, type DashboardBarRow } from '@/features/dashboard/DashboardBarsWidget';
import { DashboardConfigModal } from '@/features/dashboard/DashboardConfigModal';
import { DashboardDonutWidget } from '@/features/dashboard/DashboardDonutWidget';
import { DashboardHeader } from '@/features/dashboard/DashboardHeader';
import { DashboardKpiWidget } from '@/features/dashboard/DashboardKpiWidget';
import { DashboardQuickActionsWidget } from '@/features/dashboard/DashboardQuickActionsWidget';
import { DashboardSnapshotWidget } from '@/features/dashboard/DashboardSnapshotWidget';
import { DashboardTopLoopsWidget } from '@/features/dashboard/DashboardTopLoopsWidget';
import { DashboardTrendWidget } from '@/features/dashboard/DashboardTrendWidget';
import { DashboardWidgetGrid } from '@/features/dashboard/DashboardWidgetGrid';
import {
  countByLabel,
  countDashboardAlertSeverities,
  getAbnormalDashboardRows,
  getRealDashboardRows,
  getTopHealthyDashboardRows,
  makeDashboardSlices,
  type DashboardLoopRow,
  type DashboardStats,
  type DashboardWidgetDefinition,
  type DashboardWidgetKey,
} from '@/features/dashboard/model';

interface DashboardCockpitPanelProps {
  scopedLoops: HistoryLoop[];
  scopedLoopCount: number;
  dashboardRows: DashboardLoopRow[];
  dashboardStats: DashboardStats;
  selectedLoopId?: string;
  selectedLoop?: HistoryLoop;
  monitoring?: HistoryLoopMonitoring['monitoring'];
  assetTypeLabel: string;
  assetTagColor: string;
  pathLabel: string;
  loopTypeLabels: Record<string, string>;
  widgetKeys: DashboardWidgetKey[];
  draggedWidgetKey: DashboardWidgetKey | null;
  configOpen: boolean;
  trend: ReactNode;
  assetNameForLoop: (loop: HistoryLoop) => string;
  scorePercent: (value?: number) => number;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  statusColor: (status?: string) => string;
  statusText: (status?: string) => string;
  onOpenConfig: () => void;
  onCloseConfig: () => void;
  onWidgetKeysChange: (keys: DashboardWidgetKey[]) => void;
  onDragStart: (key: DashboardWidgetKey) => void;
  onDrop: (source: DashboardWidgetKey, target: DashboardWidgetKey) => void;
  onDragEnd: () => void;
  onHide: (key: DashboardWidgetKey) => void;
  onSwitchAsset: () => void;
  onSelectLoop: (loopId: string) => void;
  onViewLoop: (loopId: string) => void;
  onCreateTuningTask: () => void;
  onViewLoopProfile: () => void;
  onViewTrendSpectrum: () => void;
  onOpenDiagnosis: () => void;
}

export function DashboardCockpitPanel({
  scopedLoops,
  scopedLoopCount,
  dashboardRows,
  dashboardStats,
  selectedLoop,
  monitoring,
  assetTypeLabel,
  assetTagColor,
  pathLabel,
  loopTypeLabels,
  widgetKeys,
  draggedWidgetKey,
  configOpen,
  trend,
  assetNameForLoop,
  scorePercent,
  formatPercentValue,
  statusColor,
  statusText,
  onOpenConfig,
  onCloseConfig,
  onWidgetKeysChange,
  onDragStart,
  onDrop,
  onDragEnd,
  onHide,
  onSwitchAsset,
  onSelectLoop,
  onViewLoop,
  onCreateTuningTask,
  onViewLoopProfile,
  onViewTrendSpectrum,
  onOpenDiagnosis,
}: DashboardCockpitPanelProps) {
  const dashboardScore = dashboardStats.avgScore === undefined ? undefined : scorePercent(dashboardStats.avgScore);
  const loopCount = Math.max(scopedLoopCount, 1);
  const loadedCount = dashboardRows.filter((row) => row.snapshot).length;
  const pendingCount = Math.max(scopedLoopCount - loadedCount, 0);
  const warningTotal = dashboardStats.warningCount + dashboardStats.alarmCount;
  const realRows = getRealDashboardRows(dashboardRows);
  const compactTime = (value?: string | null) => value ? dayjs(value).format('MM-DD HH:mm') : '-';
  const compactDataRange = `${compactTime(dashboardStats.dataStart)} ~ ${compactTime(dashboardStats.dataEnd)}`;
  const typeCounts = countByLabel(scopedLoops, (loop) => loopTypeLabels[loop.loop_type] ?? loop.loop_type ?? '未知');
  const assetCounts = countByLabel(scopedLoops, assetNameForLoop);
  const alertSeverityCounts = countDashboardAlertSeverities(realRows);
  const statusSlices = makeDashboardSlices([
    { label: '正常', value: dashboardStats.normalCount, color: '#22c55e' },
    { label: '关注', value: dashboardStats.warningCount, color: '#facc15' },
    { label: '告警', value: dashboardStats.alarmCount, color: '#ef4444' },
    { label: '待加载', value: pendingCount, color: '#64748b' },
  ]);
  const typePalette = ['#22c55e', '#facc15', '#60a5fa', '#a78bfa', '#fb923c', '#14b8a6'];
  const typeSlices = makeDashboardSlices(Object.entries(typeCounts).map(([label, value], index) => ({
    label,
    value,
    color: typePalette[index % typePalette.length],
  })));
  const assetRows = Object.entries(assetCounts)
    .map(([label, value]) => ({ label, value, percent: value / loopCount }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 6);
  const indicatorRows = [
    { label: '数据健康', value: realRows.length ? realRows.reduce((sum, row) => sum + (row.snapshot?.data_health?.score ?? 0), 0) / realRows.length : undefined, color: '#38bdf8' },
    { label: '稳定性', value: realRows.length ? realRows.reduce((sum, row) => sum + (row.snapshot?.stability?.score ?? 0), 0) / realRows.length : undefined, color: '#22c55e' },
    { label: 'PV/MV 行为', value: realRows.length ? realRows.reduce((sum, row) => sum + (row.snapshot?.pv_mv_behavior?.score ?? 0), 0) / realRows.length : undefined, color: '#facc15' },
    { label: '约束', value: realRows.length ? realRows.reduce((sum, row) => sum + (row.snapshot?.constraints?.score ?? 0), 0) / realRows.length : undefined, color: '#fb923c' },
    { label: '响应可观测', value: realRows.length ? realRows.reduce((sum, row) => sum + (row.snapshot?.response_observability?.score ?? 0), 0) / realRows.length : undefined, color: '#a78bfa' },
  ];
  const topHealthyRows = getTopHealthyDashboardRows(realRows);
  const abnormalRows = getAbnormalDashboardRows(realRows);
  const alertRows = Object.entries(alertSeverityCounts)
    .map(([label, value], index) => ({ label, value, color: ['#ef4444', '#fb923c', '#facc15', '#60a5fa'][index % 4] }))
    .sort((a, b) => b.value - a.value);
  const assetBarRows: DashboardBarRow[] = assetRows.map((item) => ({
    label: item.label,
    percent: Math.max(4, item.percent * 100),
    trailing: `${item.value} (${formatPercentValue(item.percent, 1)})`,
  }));
  const indicatorBarRows: DashboardBarRow[] = indicatorRows.map((item) => {
    const pct = item.value === undefined ? 0 : scorePercent(item.value);
    return {
      label: item.label,
      percent: Math.max(0, Math.min(100, pct)),
      color: item.color,
      trailing: item.value === undefined ? '-' : `${pct}%`,
    };
  });
  const kpiItems: Array<{
    key: DashboardWidgetKey;
    label: string;
    value: number | string;
    suffix: string;
    color: string;
    sub: string;
  }> = [
    { key: 'kpi_total', label: '回路总数', value: scopedLoopCount, suffix: '个', color: '#60a5fa', sub: `范围 ${compactDataRange}` },
    { key: 'kpi_loaded', label: '已监控回路', value: loadedCount, suffix: '个', color: '#22d3ee', sub: `覆盖率 ${formatPercentValue(scopedLoopCount ? loadedCount / scopedLoopCount : 0, 1)}` },
    { key: 'kpi_normal', label: '正常回路', value: dashboardStats.normalCount, suffix: '个', color: '#22c55e', sub: `占比 ${formatPercentValue(dashboardStats.normalCount / loopCount, 1)}` },
    { key: 'kpi_warning', label: '关注回路', value: dashboardStats.warningCount, suffix: '个', color: '#facc15', sub: `占比 ${formatPercentValue(dashboardStats.warningCount / loopCount, 1)}` },
    { key: 'kpi_alarm', label: '告警回路', value: dashboardStats.alarmCount, suffix: '个', color: '#ef4444', sub: `占比 ${formatPercentValue(dashboardStats.alarmCount / loopCount, 1)}` },
    { key: 'kpi_score', label: '监控均分', value: dashboardScore ?? '-', suffix: dashboardScore === undefined ? '' : '分', color: '#38bdf8', sub: dashboardScore === undefined ? '暂无快照' : '后端快照均值' },
    { key: 'kpi_alerts', label: '监控告警', value: dashboardStats.alertCount, suffix: '条', color: '#a78bfa', sub: warningTotal ? '需要处理' : '当前平稳' },
  ];
  const kpiWidgetMap = Object.fromEntries(kpiItems.map((item) => [
    item.key,
    {
      title: item.label,
      className: 'cockpit-kpi dashboard-widget-kpi',
      weight: 1,
      minWidth: 132,
      content: (
        <DashboardKpiWidget
          label={item.label}
          value={item.value}
          suffix={item.suffix}
          color={item.color}
          sub={item.sub}
        />
      ),
    },
  ])) as Partial<Record<DashboardWidgetKey, DashboardWidgetDefinition>>;
  const dashboardWidgetMap: Partial<Record<DashboardWidgetKey, DashboardWidgetDefinition>> = {
    ...kpiWidgetMap,
    health: {
      title: '回路健康分布',
      className: 'cockpit-card dashboard-widget-medium',
      weight: 2,
      minWidth: 320,
      content: (
        <DashboardDonutWidget
          title="回路健康分布"
          total={scopedLoopCount}
          totalLabel="总回路数"
          slices={statusSlices}
          formatPercent={(value) => formatPercentValue(value, 1)}
        />
      ),
    },
    asset: {
      title: '回路按装置分布',
      className: 'cockpit-card dashboard-widget-medium',
      weight: 2,
      minWidth: 320,
      content: (
        <DashboardBarsWidget
          title="回路按装置分布"
          rows={assetBarRows}
          emptyDescription="暂无回路"
        />
      ),
    },
    type: {
      title: '回路类型分布',
      className: 'cockpit-card dashboard-widget-medium',
      weight: 2,
      minWidth: 320,
      content: (
        <DashboardDonutWidget
          title="回路类型分布"
          total={scopedLoopCount}
          totalLabel="总回路数"
          slices={typeSlices}
          formatPercent={(value) => formatPercentValue(value, 1)}
        />
      ),
    },
    metrics: {
      title: '关键指标均值',
      className: 'cockpit-card dashboard-widget-medium',
      weight: 2,
      minWidth: 320,
      content: (
        <DashboardBarsWidget
          title="关键指标均值"
          rows={indicatorBarRows}
          variant="metric"
        />
      ),
    },
    top: {
      title: '性能评分 TOP5',
      className: 'cockpit-card wide dashboard-widget-wide',
      weight: 3,
      minWidth: 430,
      content: (
        <DashboardTopLoopsWidget
          rows={topHealthyRows}
          loopTypeLabels={loopTypeLabels}
          scorePercent={scorePercent}
          statusColor={statusColor}
          statusText={statusText}
          onSelectLoop={onSelectLoop}
          onViewLoop={onViewLoop}
        />
      ),
    },
    abnormal: {
      title: '异常回路列表',
      className: 'cockpit-card wide dashboard-widget-wide',
      weight: 3,
      minWidth: 430,
      content: (
        <DashboardAbnormalLoopsWidget
          rows={abnormalRows}
          statusText={statusText}
          onViewLoop={onViewLoop}
        />
      ),
    },
    alerts: {
      title: '告警统计',
      className: 'cockpit-card alerts dashboard-widget-medium',
      weight: 2,
      minWidth: 300,
      content: (
        <DashboardAlertStatsWidget
          total={dashboardStats.alertCount}
          rows={alertRows}
        />
      ),
    },
    trend: {
      title: '选中回路真实趋势',
      className: 'cockpit-card trend dashboard-widget-wide',
      weight: 4,
      minWidth: 520,
      content: (
        <DashboardTrendWidget
          loopId={selectedLoop?.loop_id}
          trend={trend}
        />
      ),
    },
    snapshot: {
      title: '选中回路监控快照',
      className: 'cockpit-card snapshot dashboard-widget-medium',
      weight: 2,
      minWidth: 320,
      content: (
        <DashboardSnapshotWidget
          status={monitoring?.status}
          overallScore={monitoring?.overall_score}
          dataHealthScore={monitoring?.data_health?.score}
          stabilityScore={monitoring?.stability?.score}
          behaviorScore={monitoring?.pv_mv_behavior?.score}
          constraintScore={monitoring?.constraints?.score}
          scorePercent={scorePercent}
          statusColor={statusColor}
          statusText={statusText}
        />
      ),
    },
    quick: {
      title: '快捷操作',
      className: 'cockpit-card quick dashboard-widget-medium',
      weight: 2,
      minWidth: 300,
      content: (
        <DashboardQuickActionsWidget
          onCreateTuningTask={onCreateTuningTask}
          onViewLoopProfile={onViewLoopProfile}
          onViewTrendSpectrum={onViewTrendSpectrum}
          onOpenDiagnosis={onOpenDiagnosis}
        />
      ),
    },
  };

  return (
    <div className="dashboard-cockpit">
      <DashboardHeader
        assetTypeLabel={assetTypeLabel}
        assetTagColor={assetTagColor}
        pathLabel={pathLabel}
        onOpenConfig={onOpenConfig}
        onSwitchAsset={onSwitchAsset}
      />

      <DashboardWidgetGrid
        widgetKeys={widgetKeys}
        widgetMap={dashboardWidgetMap}
        draggedWidgetKey={draggedWidgetKey}
        onDragStart={onDragStart}
        onDrop={onDrop}
        onDragEnd={onDragEnd}
        onHide={onHide}
      />

      <DashboardConfigModal
        open={configOpen}
        widgetKeys={widgetKeys}
        onChange={onWidgetKeysChange}
        onClose={onCloseConfig}
      />
    </div>
  );
}
