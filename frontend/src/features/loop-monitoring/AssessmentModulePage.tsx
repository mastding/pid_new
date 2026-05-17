import type { Dispatch, ReactNode, SetStateAction } from 'react';
import type { Dayjs } from 'dayjs';
import type { HistoryLoop, HistoryLoopAssessment, HistoryLoopFeatures, HistoryLoopMonitoring } from '@/services/api';
import type { HistoryTimeRangeParams } from '@/services/api';
import type { SubKey } from '@/features/app-shell/navigation';
import type { FeatureRangePreset } from '@/features/monitoring/pageConfig';
import { ActuatorStatusPanel } from '@/features/loop-monitoring/ActuatorStatusPanel';
import { ConstraintMonitorPanel } from '@/features/loop-monitoring/ConstraintMonitorPanel';
import { OperatingConditionPanel } from '@/features/loop-monitoring/OperatingConditionPanel';
import { PerformanceScorePanel } from '@/features/loop-monitoring/PerformanceScorePanel';
import { TuningReadinessPanel } from '@/features/loop-monitoring/TuningReadinessPanel';

interface AssessmentModulePageProps {
  activeSub: SubKey;
  assessment: HistoryLoopAssessment | null;
  assessmentCards: ReactNode;
  loopFeatures: HistoryLoopFeatures | null;
  monitoring?: HistoryLoopMonitoring['monitoring'];
  scopedLoops: HistoryLoop[];
  selectedLoop?: HistoryLoop;
  selectedLoopId?: string;
  buildFeatureRangeParams: (loop?: HistoryLoop) => HistoryTimeRangeParams;
  featureCustomRange: [Dayjs | null, Dayjs | null] | null;
  featureRangeOptions: Array<{ label: string; value: FeatureRangePreset; seconds?: number }>;
  featureRangePreset: FeatureRangePreset;
  onCustomRangeChange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  onRangePresetChange: Dispatch<SetStateAction<FeatureRangePreset>>;
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
  buildFeatureRangeParams,
  conditionEvidenceDetail,
  conditionEvidenceName,
  conditionRecommendationText,
  evidenceStatusColor,
  evidenceStatusText,
  featureCustomRange,
  featureRangeOptions,
  featureRangePreset,
  formatNumber,
  formatPercentValue,
  loopFeatures,
  loopTypeLabel,
  monitoring,
  monitoringStatusColor,
  monitoringStatusText,
  operatingConditionText,
  onCustomRangeChange,
  onLoopChange,
  onRangePresetChange,
  scopedLoops,
  scorePercent,
  selectedLoop,
  selectedLoopId,
  tagColor,
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
          buildFeatureRangeParams={buildFeatureRangeParams}
          featureCustomRange={featureCustomRange}
          featureRangeOptions={featureRangeOptions}
          featureRangePreset={featureRangePreset}
          selectedLoop={selectedLoop}
          selectedLoopId={selectedLoopId}
          scopedLoops={scopedLoops}
          onCustomRangeChange={onCustomRangeChange}
          onLoopChange={(loopId) => onLoopChange(loopId)}
          onRangePresetChange={(value) => onRangePresetChange(value as FeatureRangePreset)}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
          loopTypeLabel={loopTypeLabel}
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
