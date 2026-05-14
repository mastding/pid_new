import { Descriptions, Empty } from 'antd';
import type { HistoryLoopFeatures, HistoryLoopMonitoringSnapshot } from '@/services/api';

interface LoopProfilePvMvPanelProps {
  loopFeatures: HistoryLoopFeatures | null;
  monitoring?: HistoryLoopMonitoringSnapshot;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  formatProcessDirection: (value?: string) => string;
  formatProcessDirectionBasis: (value?: string) => string;
}

export function LoopProfilePvMvPanel({
  loopFeatures,
  monitoring,
  formatNumber,
  formatPercentValue,
  formatProcessDirection,
  formatProcessDirectionBasis,
}: LoopProfilePvMvPanelProps) {
  return (
    <section className="agent-panel compact-facts">
      <div className="panel-title">PV / MV 原始分布</div>
      {loopFeatures ? (
        <Descriptions bordered size="small" column={4} className="industrial-descriptions">
          <Descriptions.Item label="PV 均值">{formatNumber(loopFeatures.pv_stats?.mean, 3)}</Descriptions.Item>
          <Descriptions.Item label="PV 标准差">{formatNumber(loopFeatures.pv_stats?.std, 3)}</Descriptions.Item>
          <Descriptions.Item label="PV 跨度">{formatNumber(loopFeatures.pv_stats?.span, 3)}</Descriptions.Item>
          <Descriptions.Item label="PV P95跳变">{formatNumber(loopFeatures.pv_stats?.p95_abs_step, 3)}</Descriptions.Item>
          <Descriptions.Item label="MV 均值">{formatNumber(loopFeatures.mv_stats?.mean, 3)}</Descriptions.Item>
          <Descriptions.Item label="MV 标准差">{formatNumber(loopFeatures.mv_stats?.std, 3)}</Descriptions.Item>
          <Descriptions.Item label="MV 跨度">{formatNumber(loopFeatures.mv_stats?.span, 3)}</Descriptions.Item>
          <Descriptions.Item label="MV P95跳变">{formatNumber(loopFeatures.mv_stats?.p95_abs_step, 3)}</Descriptions.Item>
          <Descriptions.Item label="MV 活跃比例">{formatPercentValue(loopFeatures.mv_stats?.active_step_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="MV 平坦比例">{formatPercentValue(loopFeatures.mv_stats?.flat_step_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="MV 反向频次">{formatNumber(loopFeatures.mv_stats?.direction_reversal_per_hour, 2)}/h</Descriptions.Item>
          <Descriptions.Item label="MV 总行程">{formatNumber(loopFeatures.mv_stats?.total_travel, 2)}</Descriptions.Item>
          <Descriptions.Item label="过程作用方向">
            {formatProcessDirection(
              monitoring?.response_observability?.process_direction
                ?? String(loopFeatures.pv_mv_relation_raw?.process_direction ?? loopFeatures.pv_mv_relation_raw?.estimated_direction_raw ?? ''),
            )}
          </Descriptions.Item>
          <Descriptions.Item label="方向置信度">
            {formatPercentValue(
              monitoring?.response_observability?.process_direction_confidence
                ?? (typeof loopFeatures.pv_mv_relation_raw?.process_direction_confidence === 'number'
                  ? loopFeatures.pv_mv_relation_raw.process_direction_confidence
                  : undefined),
              1,
            )}
          </Descriptions.Item>
          <Descriptions.Item label="方向证据">
            {formatProcessDirectionBasis(
              monitoring?.response_observability?.process_direction_basis
                ?? String(loopFeatures.pv_mv_relation_raw?.process_direction_basis ?? ''),
            )}
          </Descriptions.Item>
          <Descriptions.Item label="滞后相关峰值">{formatNumber(loopFeatures.pv_mv_relation_raw?.cross_correlation_peak_abs as number | undefined, 3)}</Descriptions.Item>
        </Descriptions>
      ) : (
        <Empty description="暂无 PV / MV 统计" />
      )}
    </section>
  );
}
