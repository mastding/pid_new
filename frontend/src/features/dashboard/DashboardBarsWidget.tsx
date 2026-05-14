import { Empty } from 'antd';

export type DashboardBarRow = {
  label: string;
  percent: number;
  trailing: string;
  color?: string;
};

interface DashboardBarsWidgetProps {
  title: string;
  rows: DashboardBarRow[];
  variant?: 'metric';
  emptyDescription?: string;
}

export function DashboardBarsWidget({
  title,
  rows,
  variant,
  emptyDescription = '暂无数据',
}: DashboardBarsWidgetProps) {
  return (
    <>
      <div className="cockpit-card-title">{title}</div>
      <div className={`cockpit-bars${variant ? ` ${variant}` : ''}`}>
        {rows.length ? rows.map((item) => (
          <div className="cockpit-bar" key={item.label}>
            <span>{item.label}</span>
            <em>
              <i style={{ width: `${item.percent}%`, background: item.color }} />
            </em>
            <b>{item.trailing}</b>
          </div>
        )) : (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyDescription} />
        )}
      </div>
    </>
  );
}
