import { Button, Table, Tag } from 'antd';
import type { DashboardLoopRow } from './model';

interface DashboardTopLoopsWidgetProps {
  rows: DashboardLoopRow[];
  loopTypeLabels: Record<string, string>;
  scorePercent: (value?: number) => number;
  statusColor: (status?: string) => string;
  statusText: (status?: string) => string;
  onSelectLoop: (loopId: string) => void;
  onViewLoop: (loopId: string) => void;
}

export function DashboardTopLoopsWidget({
  rows,
  loopTypeLabels,
  scorePercent,
  statusColor,
  statusText,
  onSelectLoop,
  onViewLoop,
}: DashboardTopLoopsWidgetProps) {
  return (
    <>
      <div className="cockpit-card-title">性能评分 TOP5</div>
      <Table<DashboardLoopRow>
        className="cockpit-table"
        size="small"
        pagination={false}
        rowKey={(row) => row.loop.loop_id}
        dataSource={rows}
        columns={[
          { title: '排名', width: 64, render: (_value, _row, index) => index + 1 },
          {
            title: '回路名称',
            render: (_value, row) => (
              <Button type="link" onClick={() => onSelectLoop(row.loop.loop_id)}>{row.loop.loop_id}</Button>
            ),
          },
          { title: '类型', width: 100, render: (_value, row) => loopTypeLabels[row.loop.loop_type ?? ''] ?? row.loop.loop_type ?? '-' },
          { title: '健康评分', width: 110, render: (_value, row) => `${scorePercent(row.snapshot?.overall_score)}%` },
          { title: '状态', width: 90, render: (_value, row) => <Tag color={statusColor(row.snapshot?.status)}>{statusText(row.snapshot?.status)}</Tag> },
          { title: '操作', width: 90, render: (_value, row) => <Button size="small" onClick={() => onViewLoop(row.loop.loop_id)}>查看</Button> },
        ]}
      />
    </>
  );
}
