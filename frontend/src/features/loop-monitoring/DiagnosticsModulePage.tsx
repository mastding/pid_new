import type { SubKey } from '@/features/app-shell/navigation';
import type { Dayjs } from 'dayjs';
import type { Dispatch, SetStateAction } from 'react';
import type { HistoryLoop, HistoryLoopAssessment, HistoryLoopMonitoringSnapshot } from '@/services/api';
import type { FeatureRangePreset } from '@/features/monitoring/pageConfig';
import {
  DiagnosisOverviewPanel,
  DiagnosisPlanPanel,
  OscillationDiagnosisPanel,
  type DiagnosisPlanKey,
} from '@/features/loop-monitoring/DiagnosisPanels';

interface DiagnosticsModulePageProps {
  activeSub: SubKey;
  assessment: HistoryLoopAssessment | null;
  monitoring?: HistoryLoopMonitoringSnapshot;
  selectedLoopId?: string;
  scopedLoops: HistoryLoop[];
  featureRangePreset: string;
  featureCustomRange: [Dayjs | null, Dayjs | null] | null;
  featureRangeOptions: Array<{ label: string; value: string; seconds?: number }>;
  onLoopChange: Dispatch<SetStateAction<string | undefined>>;
  onRangePresetChange: Dispatch<SetStateAction<FeatureRangePreset>>;
  onCustomRangeChange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  loopTypeLabel: (loop: HistoryLoop) => string;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
}

export function DiagnosticsModulePage({
  activeSub,
  assessment,
  monitoring,
  selectedLoopId,
  scopedLoops,
  featureRangePreset,
  featureCustomRange,
  featureRangeOptions,
  onLoopChange,
  onRangePresetChange,
  onCustomRangeChange,
  loopTypeLabel,
  formatNumber,
  formatPercentValue,
}: DiagnosticsModulePageProps) {
  switch (activeSub) {
    case 'diagnosis_overview':
      return (
        <DiagnosisOverviewPanel
          assessment={assessment}
          monitoring={monitoring}
          selectedLoopId={selectedLoopId}
          scopedLoops={scopedLoops}
          featureRangePreset={featureRangePreset}
          featureCustomRange={featureCustomRange}
          featureRangeOptions={featureRangeOptions}
          onLoopChange={onLoopChange}
          onRangePresetChange={onRangePresetChange}
          onCustomRangeChange={onCustomRangeChange}
          loopTypeLabel={loopTypeLabel}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
        />
      );
    case 'oscillation_diagnosis':
      return (
        <OscillationDiagnosisPanel
          assessment={assessment}
          monitoring={monitoring}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
        />
      );
    case 'pid_diagnosis':
    case 'valve_diagnosis':
    case 'measurement_noise_diagnosis':
    case 'process_disturbance_diagnosis':
      return <DiagnosisPlanPanel activeSub={activeSub as DiagnosisPlanKey} />;
    default:
      return null;
  }
}
