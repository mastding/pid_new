import type { ReactNode } from 'react';
import { Typography } from 'antd';

interface DashboardTrendWidgetProps {
  loopId?: string;
  trend: ReactNode;
}

export function DashboardTrendWidget({ loopId, trend }: DashboardTrendWidgetProps) {
  return (
    <>
      <div className="cockpit-card-title">选中回路真实趋势</div>
      <Typography.Text type="secondary">{loopId ?? '-'} · 来自后端历史趋势接口</Typography.Text>
      {trend}
    </>
  );
}
