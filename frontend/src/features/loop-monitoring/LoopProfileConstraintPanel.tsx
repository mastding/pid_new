import { Descriptions, Empty, Tag } from 'antd';
import type { HistoryLoopFeatures, HistoryLoopMonitoringSnapshot } from '@/services/api';

interface LoopProfileConstraintPanelProps {
  loopFeatures: HistoryLoopFeatures | null;
  monitoring?: HistoryLoopMonitoringSnapshot;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  statusColor: (status?: string) => string;
  statusText: (status?: string) => string;
}

export function LoopProfileConstraintPanel({
  loopFeatures,
  monitoring,
  formatNumber,
  formatPercentValue,
  statusColor,
  statusText,
}: LoopProfileConstraintPanelProps) {
  return (
    <section className="agent-panel compact-facts">
      <div className="panel-title">约束与饱和 constraint_raw</div>
      {loopFeatures ? (
        <Descriptions bordered size="small" column={4} className="industrial-descriptions">
          <Descriptions.Item label="MV饱和比例">{formatPercentValue(loopFeatures.constraint_raw?.mv_saturation_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="高限饱和">{formatPercentValue(loopFeatures.constraint_raw?.mv_high_saturation_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="低限饱和">{formatPercentValue(loopFeatures.constraint_raw?.mv_low_saturation_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="最长饱和">{formatNumber(loopFeatures.constraint_raw?.longest_mv_saturation_duration_s, 1)}s</Descriptions.Item>
          <Descriptions.Item label="饱和段数">{loopFeatures.constraint_raw?.mv_saturation_segment_count ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="PV近低限">{formatPercentValue(loopFeatures.constraint_raw?.pv_near_observed_min_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="PV近高限">{formatPercentValue(loopFeatures.constraint_raw?.pv_near_observed_max_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="监控状态">
            <Tag color={statusColor(monitoring?.constraints?.status)}>
              {statusText(monitoring?.constraints?.status)}
            </Tag>
          </Descriptions.Item>
        </Descriptions>
      ) : (
        <Empty description="暂无约束统计" />
      )}
    </section>
  );
}
