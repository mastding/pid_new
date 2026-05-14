import { Button, Table } from 'antd';
import type { DashboardLoopRow } from './model';

interface DashboardAbnormalLoopsWidgetProps {
  rows: DashboardLoopRow[];
  statusText: (status?: string) => string;
  onViewLoop: (loopId: string) => void;
}

export function DashboardAbnormalLoopsWidget({
  rows,
  statusText,
  onViewLoop,
}: DashboardAbnormalLoopsWidgetProps) {
  return (
    <>
      <div className="cockpit-card-title">异常回路列表</div>
      <Table<DashboardLoopRow>
        className="cockpit-table"
        size="small"
        pagination={false}
        rowKey={(row) => row.loop.loop_id}
        dataSource={rows}
        locale={{ emptyText: '当前监控快照未发现异常回路' }}
        columns={[
          { title: '回路名称', render: (_value, row) => row.loop.loop_id },
          { title: '异常类型', width: 160, render: (_value, row) => row.snapshot?.alerts?.[0]?.type || statusText(row.snapshot?.status) },
          { title: '严重度', width: 100, render: (_value, row) => row.snapshot?.alerts?.[0]?.severity || row.snapshot?.status || '-' },
          { title: '告警数', width: 90, render: (_value, row) => row.alertCount },
          { title: '操作', width: 90, render: (_value, row) => <Button size="small" onClick={() => onViewLoop(row.loop.loop_id)}>查看</Button> },
        ]}
      />
    </>
  );
}
