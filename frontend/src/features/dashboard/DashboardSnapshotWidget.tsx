import { Descriptions, Tag } from 'antd';

interface DashboardSnapshotWidgetProps {
  status?: string;
  overallScore?: number;
  dataHealthScore?: number;
  stabilityScore?: number;
  behaviorScore?: number;
  constraintScore?: number;
  scorePercent: (value?: number) => number;
  statusColor: (status?: string) => string;
  statusText: (status?: string) => string;
}

function formatScore(value: number | undefined, scorePercent: (value?: number) => number) {
  return value === undefined ? '-' : `${scorePercent(value)}%`;
}

export function DashboardSnapshotWidget({
  status,
  overallScore,
  dataHealthScore,
  stabilityScore,
  behaviorScore,
  constraintScore,
  scorePercent,
  statusColor,
  statusText,
}: DashboardSnapshotWidgetProps) {
  return (
    <>
      <div className="cockpit-card-title">选中回路监控快照</div>
      <Descriptions bordered size="small" column={2} className="industrial-descriptions cockpit-descriptions">
        <Descriptions.Item label="监控状态"><Tag color={statusColor(status)}>{statusText(status)}</Tag></Descriptions.Item>
        <Descriptions.Item label="综合分">{formatScore(overallScore, scorePercent)}</Descriptions.Item>
        <Descriptions.Item label="数据健康">{formatScore(dataHealthScore, scorePercent)}</Descriptions.Item>
        <Descriptions.Item label="稳定性">{formatScore(stabilityScore, scorePercent)}</Descriptions.Item>
        <Descriptions.Item label="PV/MV行为">{formatScore(behaviorScore, scorePercent)}</Descriptions.Item>
        <Descriptions.Item label="约束饱和">{formatScore(constraintScore, scorePercent)}</Descriptions.Item>
      </Descriptions>
    </>
  );
}
