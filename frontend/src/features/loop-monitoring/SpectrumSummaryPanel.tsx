import { Descriptions, Empty } from 'antd';
import type { HistoryLoopAssessment, HistoryLoopMonitoringSnapshot } from '@/services/api';

interface SpectrumSummaryPanelProps {
  assessment: HistoryLoopAssessment | null;
  monitoring?: HistoryLoopMonitoringSnapshot;
  oscillationDetected: boolean;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  formatOscillationEvidence: (detected: boolean, confidence?: number | null) => string;
  formatOscillationPhaseHint: (detected: boolean, phaseHint?: string | null) => string;
}

export function SpectrumSummaryPanel({
  assessment,
  monitoring,
  oscillationDetected,
  formatNumber,
  formatPercentValue,
  formatOscillationEvidence,
  formatOscillationPhaseHint,
}: SpectrumSummaryPanelProps) {
  return (
    <section className="agent-panel compact-facts">
      <div className="panel-title">频谱与振荡监测</div>
      {assessment || monitoring ? (
        <Descriptions bordered column={4} size="small" className="industrial-descriptions">
          <Descriptions.Item label="是否振荡">
            {monitoring?.stability?.oscillation_detected ?? assessment?.diagnostics.oscillation?.detected ? '检测到' : '未检测到'}
          </Descriptions.Item>
          <Descriptions.Item label="严重度">{monitoring?.stability?.oscillation_severity ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="振荡证据">{formatOscillationEvidence(oscillationDetected, monitoring?.stability?.oscillation_confidence)}</Descriptions.Item>
          <Descriptions.Item label="主周期">
            {String(monitoring?.stability?.pv_dominant_period_s ?? assessment?.diagnostics.oscillation?.period_sec ?? '-')}s
          </Descriptions.Item>
          <Descriptions.Item label="主频能量">{formatPercentValue(monitoring?.stability?.pv_dominant_power_ratio, 1)}</Descriptions.Item>
          <Descriptions.Item label="零交叉">{formatNumber(monitoring?.stability?.pv_zero_crossing_per_hour, 2)}/h</Descriptions.Item>
          <Descriptions.Item label="相位关系">{formatOscillationPhaseHint(oscillationDetected, monitoring?.stability?.phase_hint)}</Descriptions.Item>
          <Descriptions.Item label="PV SNR">
            {formatNumber(
              monitoring?.data_health?.pv_snr_db
                ?? (typeof assessment?.diagnostics.noise?.snr_db === 'number' ? assessment.diagnostics.noise.snr_db : undefined),
              2,
            )} dB
          </Descriptions.Item>
        </Descriptions>
      ) : (
        <Empty description="暂无频谱分析" />
      )}
    </section>
  );
}
