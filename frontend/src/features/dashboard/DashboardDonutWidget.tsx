import { dashboardConicGradient, type DashboardSlice } from './model';

interface DashboardDonutWidgetProps {
  title: string;
  total: number;
  totalLabel: string;
  slices: DashboardSlice[];
  formatPercent: (value: number) => string;
}

export function DashboardDonutWidget({
  title,
  total,
  totalLabel,
  slices,
  formatPercent,
}: DashboardDonutWidgetProps) {
  return (
    <>
      <div className="cockpit-card-title">{title}</div>
      <div className="cockpit-donut-row">
        <div className="cockpit-donut" style={{ background: dashboardConicGradient(slices) }}>
          <strong>{total}</strong>
          <span>{totalLabel}</span>
        </div>
        <div className="cockpit-legend">
          {slices.map((item) => (
            <span key={item.label}>
              <i style={{ background: item.color }} />
              {item.label}
              <b>{item.value}</b>
              <em>{formatPercent(item.percent)}</em>
            </span>
          ))}
        </div>
      </div>
    </>
  );
}
