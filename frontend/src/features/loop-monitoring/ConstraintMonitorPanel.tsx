import { Descriptions, Tag, Typography } from 'antd';

import type { HistoryLoopFeatures, HistoryLoopMonitoringSnapshot } from '@/services/api';

interface ConstraintMonitorPanelProps {
  loopFeatures: HistoryLoopFeatures | null;
  monitoring?: HistoryLoopMonitoringSnapshot;
  scorePercent: (value?: number) => number;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  monitoringStatusText: (status?: string) => string;
  monitoringStatusColor: (status?: string) => string;
}

export function ConstraintMonitorPanel({
  loopFeatures,
  monitoring,
  scorePercent,
  formatNumber,
  formatPercentValue,
  monitoringStatusText,
  monitoringStatusColor,
}: ConstraintMonitorPanelProps) {
  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">约束与饱和</div>
            <Typography.Text type="secondary">
              监测 MV 上下限触碰、连续饱和和饱和期间 PV 是否仍无法回到目标。
            </Typography.Text>
          </div>
          <Tag color={monitoringStatusColor(monitoring?.constraints?.status)}>
            {monitoringStatusText(monitoring?.constraints?.status)}
          </Tag>
        </div>
        <Descriptions bordered column={3} size="small" className="industrial-descriptions">
          <Descriptions.Item label="约束评分">
            {monitoring?.constraints?.score === undefined ? '-' : `${scorePercent(monitoring.constraints.score)}%`}
          </Descriptions.Item>
          <Descriptions.Item label="MV饱和比例">
            {formatPercentValue(loopFeatures?.constraint_raw?.mv_saturation_ratio, 2)}
          </Descriptions.Item>
          <Descriptions.Item label="MV范围">
            {loopFeatures?.mv_stats ? `${formatNumber(loopFeatures.mv_stats.min, 3)} ~ ${formatNumber(loopFeatures.mv_stats.max, 3)}` : '-'}
          </Descriptions.Item>
          <Descriptions.Item label="PV范围">
            {loopFeatures?.pv_stats ? `${formatNumber(loopFeatures.pv_stats.min, 3)} ~ ${formatNumber(loopFeatures.pv_stats.max, 3)}` : '-'}
          </Descriptions.Item>
          <Descriptions.Item label="约束影响">
            {monitoring?.constraints?.status === 'normal' ? '暂无明显约束风险' : '存在约束风险或后端尚未返回详细原因'}
          </Descriptions.Item>
        </Descriptions>
      </section>
    </div>
  );
}
