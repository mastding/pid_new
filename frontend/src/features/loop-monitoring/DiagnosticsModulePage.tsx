import type { SubKey } from '@/features/app-shell/navigation';
import type { HistoryLoopAssessment, HistoryLoopMonitoringSnapshot } from '@/services/api';
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
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
}

export function DiagnosticsModulePage({
  activeSub,
  assessment,
  monitoring,
  formatNumber,
  formatPercentValue,
}: DiagnosticsModulePageProps) {
  switch (activeSub) {
    case 'diagnosis_overview':
      return <DiagnosisOverviewPanel assessment={assessment} />;
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
