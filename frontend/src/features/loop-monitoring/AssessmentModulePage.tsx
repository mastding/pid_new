import type { Dispatch, ReactNode, SetStateAction } from 'react';
import type { HistoryLoop, HistoryLoopAssessment, HistoryLoopFeatures, HistoryLoopMonitoring } from '@/services/api';
import type { SubKey } from '@/features/app-shell/navigation';
import type { TuningResult } from '@/types/tuning';
import { ActuatorStatusPanel } from '@/features/loop-monitoring/ActuatorStatusPanel';
import { ConstraintMonitorPanel } from '@/features/loop-monitoring/ConstraintMonitorPanel';
import { OperatingConditionPanel } from '@/features/loop-monitoring/OperatingConditionPanel';
import { PerformanceScorePanel } from '@/features/loop-monitoring/PerformanceScorePanel';
import { TuningReadinessPanel } from '@/features/loop-monitoring/TuningReadinessPanel';

interface AssessmentModulePageProps {
  activeSub: SubKey;
  assessment: HistoryLoopAssessment | null;
  assessmentCards: ReactNode;
  assessmentLoading: boolean;
  loopFeatures: HistoryLoopFeatures | null;
  monitoring?: HistoryLoopMonitoring['monitoring'];
  scopedLoops: HistoryLoop[];
  selectedLoop?: HistoryLoop;
  selectedLoopId?: string;
  taskResult: TuningResult | null;
  onLoopChange: Dispatch<SetStateAction<string | undefined>>;
  conditionEvidenceDetail: (value?: string) => string;
  conditionEvidenceName: (value?: string) => string;
  conditionRecommendationText: (value?: string) => string;
  evidenceStatusColor: (status?: string) => string;
  evidenceStatusText: (status?: string) => string;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  loopTypeLabel: (loop: HistoryLoop) => string;
  monitoringStatusColor: (status?: string) => string;
  monitoringStatusText: (status?: string) => string;
  operatingConditionText: (value?: string) => string;
  scorePercent: (value?: number) => number;
  tagColor: (level?: string) => string;
  tuningSuitabilityColor: (value?: string) => string;
  tuningSuitabilityText: (value?: string) => string;
  yesNo: (value?: boolean | null) => string;
}

export function AssessmentModulePage({
  activeSub,
  assessment,
  assessmentCards,
  assessmentLoading,
  conditionEvidenceDetail,
  conditionEvidenceName,
  conditionRecommendationText,
  evidenceStatusColor,
  evidenceStatusText,
  formatNumber,
  formatPercentValue,
  loopFeatures,
  loopTypeLabel,
  monitoring,
  monitoringStatusColor,
  monitoringStatusText,
  operatingConditionText,
  onLoopChange,
  scopedLoops,
  scorePercent,
  selectedLoop,
  selectedLoopId,
  tagColor,
  taskResult,
  tuningSuitabilityColor,
  tuningSuitabilityText,
  yesNo,
}: AssessmentModulePageProps) {
  switch (activeSub) {
    case 'tuning_readiness':
      return (
        <TuningReadinessPanel
          assessment={assessment}
          showDetails={activeSub === 'tuning_readiness'}
          assessmentCards={assessmentCards}
          tagColor={tagColor}
        />
      );
    case 'performance_score':
      return (
        <PerformanceScorePanel
          assessment={assessment}
          assessmentLoading={assessmentLoading}
          taskResult={taskResult}
          formatNumber={formatNumber}
          scorePercent={scorePercent}
          tagColor={tagColor}
        />
      );
    case 'condition_recognition':
      return (
        <OperatingConditionPanel
          conditionProfile={loopFeatures?.operating_condition_profile}
          selectedLoop={selectedLoop}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
          operatingConditionText={operatingConditionText}
          tuningSuitabilityText={tuningSuitabilityText}
          tuningSuitabilityColor={tuningSuitabilityColor}
          conditionEvidenceName={conditionEvidenceName}
          conditionEvidenceDetail={conditionEvidenceDetail}
          conditionRecommendationText={conditionRecommendationText}
          evidenceStatusText={evidenceStatusText}
          evidenceStatusColor={evidenceStatusColor}
        />
      );
    case 'actuator_status':
      return (
        <ActuatorStatusPanel
          selectedLoopId={selectedLoopId}
          scopedLoops={scopedLoops}
          loopFeatures={loopFeatures}
          onLoopChange={onLoopChange}
          loopTypeLabel={loopTypeLabel}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
          yesNo={yesNo}
        />
      );
    case 'constraint_monitor':
      return (
        <ConstraintMonitorPanel
          loopFeatures={loopFeatures}
          monitoring={monitoring}
          scorePercent={scorePercent}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
          monitoringStatusText={monitoringStatusText}
          monitoringStatusColor={monitoringStatusColor}
        />
      );
    default:
      return null;
  }
}
