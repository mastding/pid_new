import { useEffect } from 'react';
import type { DashboardWidgetKey } from '@/features/dashboard/model';
import type { SubKey } from '@/features/app-shell/navigation';
import type { HistoryLoop, HistoryTimeRangeParams } from '@/services/api';

interface UseMonitoringPageEffectsOptions {
  activeSub: SubKey;
  buildFeatureRangeParams: (loop?: HistoryLoop) => HistoryTimeRangeParams;
  buildTuningRangeParams: (loop?: HistoryLoop) => HistoryTimeRangeParams;
  enabledDashboardWidgets: Set<DashboardWidgetKey>;
  isSettingsView: boolean;
  loadAssessment: (loopId: string, params?: HistoryTimeRangeParams) => void;
  loadLoopFeatures: (loopId: string, params?: HistoryTimeRangeParams) => void;
  loadLoopMonitoring: (loopId: string, params?: HistoryTimeRangeParams) => void;
  loadSeries: (loopId: string, loop?: HistoryLoop) => void;
  loadWindows: (loopId: string, params?: HistoryTimeRangeParams) => void;
  resetTuningPrior: () => void;
  selectedLoop?: HistoryLoop;
  selectedLoopId?: string;
  shouldLoadAssessmentDetail: boolean;
  shouldLoadFeatureDetail: boolean;
  shouldLoadMonitoringDetail: boolean;
  shouldLoadWindowDetail: boolean;
  tuningPriorCustomRange: unknown;
  tuningPriorRangePreset: unknown;
}

export function useMonitoringPageEffects({
  activeSub,
  buildFeatureRangeParams,
  buildTuningRangeParams,
  enabledDashboardWidgets,
  isSettingsView,
  loadAssessment,
  loadLoopFeatures,
  loadLoopMonitoring,
  loadSeries,
  loadWindows,
  resetTuningPrior,
  selectedLoop,
  selectedLoopId,
  shouldLoadAssessmentDetail,
  shouldLoadFeatureDetail,
  shouldLoadMonitoringDetail,
  shouldLoadWindowDetail,
  tuningPriorCustomRange,
  tuningPriorRangePreset,
}: UseMonitoringPageEffectsOptions) {
  useEffect(() => {
    if (!selectedLoopId || isSettingsView) return;
    const shouldLoadDashboardTrend = activeSub === 'dashboard' && enabledDashboardWidgets.has('trend');
    if (activeSub !== 'trend_spectrum' && !shouldLoadDashboardTrend) return;
    loadSeries(selectedLoopId, selectedLoop);
  }, [activeSub, enabledDashboardWidgets, isSettingsView, loadSeries, selectedLoop, selectedLoopId]);

  useEffect(() => {
    if (!selectedLoopId || isSettingsView) return;
    if (!shouldLoadAssessmentDetail && !shouldLoadWindowDetail) return;
    if (shouldLoadAssessmentDetail) {
      const params = activeSub === 'tuning_task' ? buildTuningRangeParams(selectedLoop) : undefined;
      loadAssessment(selectedLoopId, params);
    }
    if (shouldLoadWindowDetail) loadWindows(selectedLoopId);
  }, [
    activeSub,
    buildTuningRangeParams,
    isSettingsView,
    loadAssessment,
    loadWindows,
    selectedLoop,
    selectedLoopId,
    shouldLoadAssessmentDetail,
    shouldLoadWindowDetail,
  ]);

  useEffect(() => {
    if (!selectedLoopId || isSettingsView) return;
    if (!shouldLoadFeatureDetail && !shouldLoadMonitoringDetail) return;
    const featureParams = buildFeatureRangeParams(selectedLoop);
    if (shouldLoadMonitoringDetail) {
      loadLoopMonitoring(selectedLoopId, featureParams);
      return;
    }
    loadLoopFeatures(selectedLoopId, featureParams);
  }, [
    buildFeatureRangeParams,
    isSettingsView,
    loadLoopFeatures,
    loadLoopMonitoring,
    selectedLoop,
    selectedLoopId,
    shouldLoadFeatureDetail,
    shouldLoadMonitoringDetail,
  ]);

  useEffect(() => {
    if (activeSub !== 'tuning_prior') return;
    resetTuningPrior();
  }, [activeSub, resetTuningPrior, selectedLoopId, tuningPriorRangePreset, tuningPriorCustomRange]);
}
