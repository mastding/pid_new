interface DashboardKpiWidgetProps {
  label: string;
  value: number | string;
  suffix: string;
  color: string;
  sub: string;
}

export function DashboardKpiWidget({
  label,
  value,
  suffix,
  color,
  sub,
}: DashboardKpiWidgetProps) {
  return (
    <>
      <i style={{ background: color }} />
      <div>
        <span>{label}</span>
        <strong>{value}<em>{suffix}</em></strong>
        <small>{sub}</small>
      </div>
    </>
  );
}
