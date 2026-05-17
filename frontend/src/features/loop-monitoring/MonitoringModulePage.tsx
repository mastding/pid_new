import type { Dispatch, ReactNode, SetStateAction } from 'react';
import type { Dayjs } from 'dayjs';
import type {
  HistoryLoop,
  HistoryLoopAssessment,
  HistoryLoopFeatures,
  HistoryLoopMonitoring,
  HistoryTimeRangeParams,
  LoopSeriesResp,
} from '@/services/api';
import type { SubKey } from '@/features/app-shell/navigation';
import type {
  DashboardLoopRow,
  DashboardStats,
  DashboardWidgetKey,
} from '@/features/dashboard/model';
import { DashboardCockpitPanel } from '@/features/dashboard/DashboardCockpitPanel';
import { AlarmEventsPanel } from '@/features/loop-monitoring/AlarmEventsPanel';
import type { RailAlarmRow } from '@/features/loop-monitoring/alarmModel';
import { LoopBoardPanel } from '@/features/loop-monitoring/LoopBoardPanel';
import { LoopProfilePanel } from '@/features/loop-monitoring/LoopProfilePanel';
import { RiskAlertsPanel } from '@/features/loop-monitoring/RiskAlertsPanel';
import { TrendSpectrumPanel } from '@/features/loop-monitoring/TrendSpectrumPanel';
import type {
  FeatureRangePreset,
  TrendPointLimit,
  TrendPreset,
} from '@/features/monitoring/pageConfig';

interface MonitoringModulePageProps {
  activeSub: SubKey;
  alertSeverityColor: (severity?: string) => string;
  assessment: HistoryLoopAssessment | null;
  assetNameForLoop: (loop: HistoryLoop, fallback?: string) => string;
  buildFeatureRangeParams: (loop?: HistoryLoop) => HistoryTimeRangeParams;
  configOpen: boolean;
  dashboardRows: DashboardLoopRow[];
  dashboardStats: DashboardStats;
  draggedWidgetKey: DashboardWidgetKey | null;
  featureCustomRange: [Dayjs | null, Dayjs | null] | null;
  featureLoading: boolean;
  featureRangeOptions: Array<{ label: string; value: string; seconds?: number }>;
  featureRangePreset: string;
  formatCpkBasis: (value?: string) => string;
  formatHarrisBasis: (value?: string) => string;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatOscillationEvidence: (detected?: boolean, confidence?: number | null) => string;
  formatOscillationPhaseHint: (detected?: boolean, phaseHint?: string | null) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  formatProcessDirection: (value?: string | null) => string;
  formatProcessDirectionBasis: (value?: string | null) => string;
  formatRange: (min?: number | null, max?: number | null, digits?: number) => string;
  loading: boolean;
  loopFeatures: HistoryLoopFeatures | null;
  loopTable: ReactNode;
  loopTypeLabels: Record<string, string>;
  monitoring?: HistoryLoopMonitoring['monitoring'];
  monitoringStatusColor: (status?: string) => string;
  monitoringStatusText: (status?: string) => string;
  oscillationDetected: boolean;
  pathLabel: string;
  railAlarms: RailAlarmRow[];
  scopedLoopCount: number;
  scopedLoops: HistoryLoop[];
  scorePercent: (value?: number) => number;
  selectedAssetTagColor: string;
  selectedAssetTypeLabel: string;
  selectedLoop?: HistoryLoop;
  selectedLoopId?: string;
  series: LoopSeriesResp | null;
  seriesLoading: boolean;
  tagColor: (level?: string) => string;
  taskId?: string;
  taskStatus: string;
  trend: ReactNode;
  trendChart: ReactNode;
  trendCustomRange: [Dayjs | null, Dayjs | null] | null;
  trendPointLimit: string;
  trendPointLimitOptions: Array<{ label: string; value: string; seconds?: number }>;
  trendPreset: string;
  trendPresetOptions: Array<{ label: string; value: string; seconds?: number }>;
  trendSplitYAxis: boolean;
  widgetKeys: DashboardWidgetKey[];
  loadLoopFeatures: (loopId: string, params?: HistoryTimeRangeParams) => void;
  loadLoopMonitoring: (loopId: string, params?: HistoryTimeRangeParams) => void;
  loadLoops: () => void;
  loadSeries: (loopId: string, loop?: HistoryLoop) => void;
  moveDashboardWidget: (source: DashboardWidgetKey, target: DashboardWidgetKey) => void;
  onCreateTuningTask: () => void;
  onOpenConfig: () => void;
  onOpenDiagnosis: () => void;
  onSwitchAsset: () => void;
  onViewLoopProfile: () => void;
  onViewTrendSpectrum: () => void;
  setConfigOpen: (open: boolean) => void;
  setDraggedWidgetKey: (key: DashboardWidgetKey | null) => void;
  setFeatureCustomRange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  setFeatureRangePreset: Dispatch<SetStateAction<FeatureRangePreset>>;
  setSelectedLoopId: Dispatch<SetStateAction<string | undefined>>;
  setTrendCustomRange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  setTrendPointLimit: Dispatch<SetStateAction<TrendPointLimit>>;
  setTrendPreset: Dispatch<SetStateAction<TrendPreset>>;
  setTrendSplitYAxis: Dispatch<SetStateAction<boolean>>;
  setWidgetKeys: (keys: DashboardWidgetKey[]) => void;
  hideDashboardWidget: (key: DashboardWidgetKey) => void;
  onViewLoop: (loopId: string) => void;
}

export function MonitoringModulePage({
  activeSub,
  alertSeverityColor,
  assessment,
  assetNameForLoop,
  buildFeatureRangeParams,
  configOpen,
  dashboardRows,
  dashboardStats,
  draggedWidgetKey,
  featureCustomRange,
  featureLoading,
  featureRangeOptions,
  featureRangePreset,
  formatCpkBasis,
  formatHarrisBasis,
  formatNumber,
  formatOscillationEvidence,
  formatOscillationPhaseHint,
  formatPercentValue,
  formatProcessDirection,
  formatProcessDirectionBasis,
  formatRange,
  hideDashboardWidget,
  loading,
  loadLoopFeatures,
  loadLoopMonitoring,
  loadLoops,
  loadSeries,
  loopFeatures,
  loopTable,
  loopTypeLabels,
  monitoring,
  monitoringStatusColor,
  monitoringStatusText,
  moveDashboardWidget,
  onCreateTuningTask,
  onOpenConfig,
  onOpenDiagnosis,
  onSwitchAsset,
  onViewLoop,
  onViewLoopProfile,
  onViewTrendSpectrum,
  oscillationDetected,
  pathLabel,
  railAlarms,
  scopedLoopCount,
  scopedLoops,
  scorePercent,
  selectedAssetTagColor,
  selectedAssetTypeLabel,
  selectedLoop,
  selectedLoopId,
  series,
  seriesLoading,
  setConfigOpen,
  setDraggedWidgetKey,
  setFeatureCustomRange,
  setFeatureRangePreset,
  setSelectedLoopId,
  setTrendCustomRange,
  setTrendPointLimit,
  setTrendPreset,
  setTrendSplitYAxis,
  setWidgetKeys,
  tagColor,
  taskId,
  taskStatus,
  trend,
  trendChart,
  trendCustomRange,
  trendPointLimit,
  trendPointLimitOptions,
  trendPreset,
  trendPresetOptions,
  trendSplitYAxis,
  widgetKeys,
}: MonitoringModulePageProps) {
  const monitoringAlerts = monitoring?.alerts ?? [];

  switch (activeSub) {
    case 'dashboard':
      return (
        <DashboardCockpitPanel
          scopedLoops={scopedLoops}
          scopedLoopCount={scopedLoopCount}
          dashboardRows={dashboardRows}
          dashboardStats={dashboardStats}
          selectedLoopId={selectedLoopId}
          selectedLoop={selectedLoop}
          monitoring={monitoring}
          assetTypeLabel={selectedAssetTypeLabel}
          assetTagColor={selectedAssetTagColor}
          pathLabel={pathLabel}
          loopTypeLabels={loopTypeLabels}
          widgetKeys={widgetKeys}
          draggedWidgetKey={draggedWidgetKey}
          configOpen={configOpen}
          trend={trend}
          assetNameForLoop={(loop) => assetNameForLoop(loop, '未归属')}
          scorePercent={scorePercent}
          formatPercentValue={formatPercentValue}
          statusColor={monitoringStatusColor}
          statusText={monitoringStatusText}
          onOpenConfig={onOpenConfig}
          onCloseConfig={() => setConfigOpen(false)}
          onWidgetKeysChange={setWidgetKeys}
          onDragStart={setDraggedWidgetKey}
          onDrop={moveDashboardWidget}
          onDragEnd={() => setDraggedWidgetKey(null)}
          onHide={hideDashboardWidget}
          onSwitchAsset={onSwitchAsset}
          onSelectLoop={setSelectedLoopId}
          onViewLoop={onViewLoop}
          onCreateTuningTask={onCreateTuningTask}
          onViewLoopProfile={onViewLoopProfile}
          onViewTrendSpectrum={onViewTrendSpectrum}
          onOpenDiagnosis={onOpenDiagnosis}
        />
      );
    case 'loop_board':
      return (
        <LoopBoardPanel
          selectedLoop={selectedLoop}
          monitoring={monitoring}
          alertCount={monitoringAlerts.length}
          loading={loading}
          loopTable={loopTable}
          scorePercent={scorePercent}
          statusColor={monitoringStatusColor}
          statusText={monitoringStatusText}
          onRefresh={loadLoops}
        />
      );
    case 'loop_profile':
      return (
        <LoopProfilePanel
          selectedLoopId={selectedLoopId}
          selectedLoop={selectedLoop}
          scopedLoops={scopedLoops}
          loopFeatures={loopFeatures}
          assessment={assessment}
          monitoring={monitoring}
          featureRangePreset={featureRangePreset}
          featureCustomRange={featureCustomRange}
          featureRangeOptions={featureRangeOptions}
          featureLoading={featureLoading}
          loopTypeLabel={loopTypeLabels}
          onLoopChange={setSelectedLoopId}
          onRangePresetChange={(value) => setFeatureRangePreset(value as FeatureRangePreset)}
          onCustomRangeChange={setFeatureCustomRange}
          onRefresh={() => {
            if (!selectedLoopId) return;
            const params = buildFeatureRangeParams(selectedLoop);
            loadLoopFeatures(selectedLoopId, params);
            loadLoopMonitoring(selectedLoopId, params);
          }}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
          formatRange={formatRange}
          scorePercent={scorePercent}
          tagColor={tagColor}
          formatProcessDirection={formatProcessDirection}
          formatProcessDirectionBasis={formatProcessDirectionBasis}
          formatHarrisBasis={formatHarrisBasis}
          formatCpkBasis={formatCpkBasis}
          monitoringStatusColor={monitoringStatusColor}
          monitoringStatusText={monitoringStatusText}
        />
      );
    case 'trend_spectrum':
      return (
        <TrendSpectrumPanel
          selectedLoopId={selectedLoopId}
          selectedLoop={selectedLoop}
          scopedLoops={scopedLoops}
          series={series}
          seriesLoading={seriesLoading}
          trendPreset={trendPreset}
          trendPointLimit={trendPointLimit}
          trendSplitYAxis={trendSplitYAxis}
          trendCustomRange={trendCustomRange}
          trendPresetOptions={trendPresetOptions}
          trendPointLimitOptions={trendPointLimitOptions}
          loopTypeLabel={loopTypeLabels}
          assessment={assessment}
          monitoring={monitoring}
          oscillationDetected={oscillationDetected}
          chart={trendChart}
          onLoopChange={setSelectedLoopId}
          onTrendPresetChange={(value) => setTrendPreset(value as TrendPreset)}
          onTrendPointLimitChange={(value) => setTrendPointLimit(value as TrendPointLimit)}
          onTrendSplitYAxisChange={setTrendSplitYAxis}
          onTrendCustomRangeChange={setTrendCustomRange}
          onRefresh={() => selectedLoopId && loadSeries(selectedLoopId, selectedLoop)}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
          formatOscillationEvidence={formatOscillationEvidence}
          formatOscillationPhaseHint={formatOscillationPhaseHint}
        />
      );
    case 'risk_alerts':
      return (
        <RiskAlertsPanel
          dashboardRows={dashboardRows}
          scopedLoops={scopedLoops}
          pathLabel={pathLabel}
          loading={loading}
          loopTypeLabels={loopTypeLabels}
          assetNameForLoop={(loop) => assetNameForLoop(loop, '未归属')}
          formatPercentValue={formatPercentValue}
          scorePercent={scorePercent}
          onRefresh={loadLoops}
          onViewLoop={onViewLoop}
          onViewTrendSpectrum={(loopId) => {
            setSelectedLoopId(loopId);
            onViewTrendSpectrum();
          }}
          onOpenDiagnosis={onOpenDiagnosis}
          onCreateTuningTask={onCreateTuningTask}
        />
      );
    case 'alarm_events':
      return (
        <AlarmEventsPanel
          railAlarms={railAlarms}
          monitoringStatus={monitoring?.status}
          monitoringEventCount={monitoring?.events?.length ?? monitoringAlerts.length}
          diagnosticFlagCount={assessment?.diagnostics.flags.length ?? 0}
          taskLabel={taskId ? taskStatus : '暂无任务'}
          pathLabel={pathLabel}
          monitoringStatusText={monitoringStatusText}
          monitoringStatusColor={monitoringStatusColor}
          alertSeverityColor={alertSeverityColor}
        />
      );
    default:
      return null;
  }
}
