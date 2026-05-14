import { Empty } from 'antd';

export type DashboardAlertRow = {
  label: string;
  value: number;
  color: string;
};

interface DashboardAlertStatsWidgetProps {
  total: number;
  rows: DashboardAlertRow[];
}

export function DashboardAlertStatsWidget({ total, rows }: DashboardAlertStatsWidgetProps) {
  return (
    <>
      <div className="cockpit-card-title">告警统计</div>
      <strong className="cockpit-alert-total">{total}</strong>
      <span>当前监控快照告警总数</span>
      <div className="cockpit-bars">
        {rows.length ? rows.map((item) => (
          <div className="cockpit-bar" key={item.label}>
            <span>{item.label}</span>
            <em>
              <i style={{ width: `${Math.max(8, (item.value / Math.max(total, 1)) * 100)}%`, background: item.color }} />
            </em>
            <b>{item.value}</b>
          </div>
        )) : (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无告警" />
        )}
      </div>
    </>
  );
}
