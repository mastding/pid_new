import { useCallback } from 'react';
import type { ModuleKey, SubKey } from '@/features/app-shell/navigation';
import type { ViewMode } from '@/features/app-shell/useAppShellState';
import type { DashboardLoopRow, DashboardStats } from '@/features/dashboard/model';
import type {
  HistoryLoop,
  HistoryLoopAssessment,
  HistoryLoopFeatures,
  HistoryLoopMonitoring,
} from '@/services/api';
import {
  buildAssistantContext,
  type AssistantAction,
} from '@/features/dialogue/model';
import { useAssistantDialogue } from '@/features/dialogue/useAssistantDialogue';

interface UseMonitoringAssistantOptions {
  activeModule: ModuleKey;
  activeSub: SubKey;
  assessment: HistoryLoopAssessment | null;
  currentSubLabel: string;
  dashboardRows: DashboardLoopRow[];
  dashboardStats: DashboardStats;
  enabled: boolean;
  loopFeatures: HistoryLoopFeatures | null;
  loopMonitoring: HistoryLoopMonitoring | null;
  monitoringByLoopId: Record<string, HistoryLoopMonitoring>;
  scopedLoopCount: number;
  selectedLoop?: HistoryLoop;
  selectedLoopId?: string;
  setViewMode: (mode: ViewMode) => void;
  switchTo: (moduleKey: ModuleKey, subKey: SubKey) => void;
  onSelectLoop: (loopId?: string) => void;
}

export function useMonitoringAssistant({
  activeModule,
  activeSub,
  assessment,
  currentSubLabel,
  dashboardRows,
  dashboardStats,
  enabled,
  loopFeatures,
  loopMonitoring,
  monitoringByLoopId,
  scopedLoopCount,
  selectedLoop,
  selectedLoopId,
  setViewMode,
  switchTo,
  onSelectLoop,
}: UseMonitoringAssistantOptions) {
  const runAssistantAction = useCallback((action: AssistantAction) => {
    if (action.loopId) onSelectLoop(action.loopId);
    switchTo(action.target, action.sub);
    setViewMode('classic');
  }, [onSelectLoop, setViewMode, switchTo]);

  const buildContext = useCallback(() => {
    return buildAssistantContext({
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
    });
  }, [
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
  ]);

  const assistant = useAssistantDialogue({
    enabled,
    selectedLoop,
    selectedLoopId,
    buildContext,
    onSelectLoop,
  });

  return {
    ...assistant,
    runAssistantAction,
  };
}
