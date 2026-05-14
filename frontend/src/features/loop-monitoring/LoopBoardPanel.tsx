import type { ReactNode } from 'react';
import { Button, Space, Statistic, Tag, Typography } from 'antd';
import { SyncOutlined } from '@ant-design/icons';
import type { HistoryLoop, HistoryLoopMonitoringSnapshot } from '@/services/api';

interface LoopBoardPanelProps {
  selectedLoop?: HistoryLoop;
  monitoring?: HistoryLoopMonitoringSnapshot;
  alertCount: number;
  loading: boolean;
  loopTable: ReactNode;
  scorePercent: (value?: number) => number;
  statusColor: (status?: string) => string;
  statusText: (status?: string) => string;
  onRefresh: () => void;
}

function formatScore(value: number | undefined, scorePercent: (value?: number) => number) {
  return value === undefined ? '-' : scorePercent(value);
}

function scoreSuffix(value: number | undefined) {
  return value === undefined ? undefined : '%';
}

export function LoopBoardPanel({
  selectedLoop,
  monitoring,
  alertCount,
  loading,
  loopTable,
  scorePercent,
  statusColor,
  statusText,
  onRefresh,
}: LoopBoardPanelProps) {
  return (
    <section className="agent-panel">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">全局回路看板</div>
          <Typography.Text type="secondary">
            当前选中：{selectedLoop?.loop_id ?? '-'} · 监控状态：
            <Tag color={statusColor(monitoring?.status)}>{statusText(monitoring?.status)}</Tag>
            综合分：{monitoring?.overall_score === undefined ? '-' : `${scorePercent(monitoring.overall_score)}%`}
          </Typography.Text>
        </div>
        <Space>
          <Tag color={statusColor(monitoring?.status)}>
            {statusText(monitoring?.status)}
          </Tag>
          <Button icon={<SyncOutlined />} onClick={onRefresh} loading={loading}>刷新</Button>
        </Space>
      </div>
      <div className="kpi-grid compact-kpi">
        <Statistic
          title="监控综合分"
          value={formatScore(monitoring?.overall_score, scorePercent)}
          suffix={scoreSuffix(monitoring?.overall_score)}
        />
        <Statistic
          title="数据健康"
          value={formatScore(monitoring?.data_health?.score, scorePercent)}
          suffix={scoreSuffix(monitoring?.data_health?.score)}
        />
        <Statistic
          title="PV/MV行为"
          value={formatScore(monitoring?.pv_mv_behavior?.score, scorePercent)}
          suffix={scoreSuffix(monitoring?.pv_mv_behavior?.score)}
        />
        <Statistic title="监控告警" value={alertCount} suffix="条" />
      </div>
      {loopTable}
    </section>
  );
}
