import { Descriptions, Empty, Tag, Typography } from 'antd';
import type { HistoryLoopAssessment, HistoryLoopMonitoringSnapshot } from '@/services/api';

interface LoopProfileDataQualityPanelProps {
  assessment: HistoryLoopAssessment | null;
  monitoring?: HistoryLoopMonitoringSnapshot;
  scorePercent: (value?: number) => number;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  tagColor: (level?: string) => string;
}

export function LoopProfileDataQualityPanel({
  assessment,
  monitoring,
  scorePercent,
  formatNumber,
  formatPercentValue,
  tagColor,
}: LoopProfileDataQualityPanelProps) {
  return (
    <section className="agent-panel compact-facts">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">数据质量指标</div>
          <Typography.Text type="secondary">查看缺失、连续性、采样异常、噪声和离群点。</Typography.Text>
        </div>
        {assessment?.data_quality ? (
          <Tag color={tagColor(assessment.data_quality.level)}>{assessment.data_quality.level}</Tag>
        ) : null}
      </div>
      {assessment ? (
        <Descriptions bordered column={4} size="small" className="industrial-descriptions">
          <Descriptions.Item label="缺失比例">{(assessment.data_quality.missing_ratio * 100).toFixed(2)}%</Descriptions.Item>
          <Descriptions.Item label="连续性">{scorePercent(assessment.data_quality.continuity_score)}%</Descriptions.Item>
          <Descriptions.Item label="噪声得分">{scorePercent(assessment.data_quality.noise_score)}%</Descriptions.Item>
          <Descriptions.Item label="采样不规则">{formatPercentValue(monitoring?.data_health?.irregular_sample_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="长间隔">{monitoring?.data_health?.long_gap_count ?? 0} 个</Descriptions.Item>
          <Descriptions.Item label="重复时间戳">{formatPercentValue(monitoring?.data_health?.duplicate_timestamp_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="PV噪声比">{formatPercentValue(monitoring?.data_health?.pv_noise_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="PV SNR">{formatNumber(monitoring?.data_health?.pv_snr_db, 2)} dB</Descriptions.Item>
          <Descriptions.Item label="PV尖峰">{monitoring?.data_health?.pv_spike_count ?? 0} 个</Descriptions.Item>
          <Descriptions.Item label="PV离群">{monitoring?.data_health?.pv_outlier_count ?? 0} 个</Descriptions.Item>
          <Descriptions.Item label="MV离群">{monitoring?.data_health?.mv_outlier_count ?? 0} 个</Descriptions.Item>
        </Descriptions>
      ) : (
        <Empty description="暂无数据质量指标" />
      )}
    </section>
  );
}
