import { useMemo } from 'react';
import type { HistoryLoop, HistoryLoopAssessment, HistoryLoopMonitoring } from '@/services/api';
import {
  buildDashboardRows,
  summarizeDashboardRows,
} from '@/features/dashboard/model';
import {
  buildFitPreviewChartData,
  buildTuningGate,
  getDeterministicRefinement,
  getFitPreviewAttempts,
  getSelectedFitAttempt,
  getTaskAlgorithmComparison,
  type TaskStatus,
} from '@/features/tuning-task/model';
import { buildRailAlarms } from '@/features/loop-monitoring/alarmModel';
import type { IdentificationAttempt, IdentificationRefinementMeta, TuningResult } from '@/types/tuning';
import type { TaskStageDataMap } from '@/features/tuning-task/model';

interface UseMonitoringDerivedStateOptions {
  assessment: HistoryLoopAssessment | null;
  dataSourceType: string;
  loopMonitoring: HistoryLoopMonitoring | null;
  monitoringByLoopId: Record<string, HistoryLoopMonitoring>;
  scopedLoops: HistoryLoop[];
  selectedFitAttemptKey?: string;
  taskAttempts: IdentificationAttempt[];
  taskId?: string;
  taskRefinements: IdentificationRefinementMeta[];
  taskResult: TuningResult | null;
  taskStageData: TaskStageDataMap;
  taskStartedAt?: string;
  taskStatus: TaskStatus;
  monitoringStatusText: (status?: string) => string;
  scorePercent: (value?: number) => number;
}

export function useMonitoringDerivedState({
  assessment,
  dataSourceType,
  loopMonitoring,
  monitoringByLoopId,
  scopedLoops,
  selectedFitAttemptKey,
  taskAttempts,
  taskId,
  taskRefinements,
  taskResult,
  taskStageData,
  taskStartedAt,
  taskStatus,
  monitoringStatusText,
  scorePercent,
}: UseMonitoringDerivedStateOptions) {
  const dashboardRows = useMemo(
    () => buildDashboardRows(scopedLoops, monitoringByLoopId),
    [monitoringByLoopId, scopedLoops],
  );

  const dashboardWorstLoopId = useMemo(
    () => (dashboardRows.find((row) => row.snapshot) ?? dashboardRows[0])?.loop.loop_id,
    [dashboardRows],
  );

  const dashboardStats = useMemo(
    () => summarizeDashboardRows(dashboardRows, scopedLoops),
    [dashboardRows, scopedLoops],
  );

  const taskAlgorithmComparison = useMemo(
    () => getTaskAlgorithmComparison(taskStageData, taskResult),
    [taskResult, taskStageData],
  );

  const fitPreviewAttempts = useMemo(() => {
    return getFitPreviewAttempts(taskAttempts, taskResult);
  }, [taskAttempts, taskResult]);

  const selectedFitAttempt = useMemo(() => {
    return getSelectedFitAttempt(fitPreviewAttempts, selectedFitAttemptKey);
  }, [fitPreviewAttempts, selectedFitAttemptKey]);

  const fitPreviewChartData = useMemo(() => {
    return buildFitPreviewChartData(selectedFitAttempt);
  }, [selectedFitAttempt]);

  const deterministicRefinement = useMemo(
    () => getDeterministicRefinement(taskRefinements),
    [taskRefinements],
  );

  const tuningGate = useMemo(() => {
    return buildTuningGate(assessment);
  }, [assessment]);

  const railAlarms = useMemo(() => {
    return buildRailAlarms({
      assessment,
      dataSourceType,
      loopMonitoring,
      taskId,
      taskStartedAt,
      taskStatus,
      monitoringStatusText,
      scorePercent,
    });
  }, [assessment, dataSourceType, loopMonitoring, taskId, taskStartedAt, taskStatus, monitoringStatusText, scorePercent]);

  return {
    dashboardRows,
    dashboardStats,
    dashboardWorstLoopId,
    deterministicRefinement,
    fitPreviewAttempts,
    fitPreviewChartData,
    railAlarms,
    selectedFitAttempt,
    taskAlgorithmComparison,
    tuningGate,
  };
}
