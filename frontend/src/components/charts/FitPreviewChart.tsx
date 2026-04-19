import { Line } from '@ant-design/charts';
import type { FitPreview } from '@/types/tuning';

interface Props {
  fitPreview: FitPreview;
  dt?: number;
  height?: number;
}

interface ChartPoint {
  t: number;
  value: number;
  series: string;
}

export default function FitPreviewChart({ fitPreview, dt = 1, height = 220 }: Props) {
  const { t: times, pv_actual, pv_fitted } = fitPreview;
  if (!pv_actual || !pv_fitted || pv_actual.length === 0) return null;

  const n = Math.min(
    times?.length ?? pv_actual.length,
    pv_actual.length,
    pv_fitted.length,
  );

  const data: ChartPoint[] = [];
  for (let i = 0; i < n; i++) {
    const t = times ? parseFloat(times[i].toFixed(1)) : parseFloat((i * dt).toFixed(1));
    data.push({ t, value: parseFloat(pv_actual[i].toFixed(4)), series: '实测 PV' });
    data.push({ t, value: parseFloat(pv_fitted[i].toFixed(4)), series: '模型拟合' });
  }

  const config = {
    data,
    xField: 't',
    yField: 'value',
    colorField: 'series',
    height,
    smooth: false,
    legend: { position: 'top-right' as const },
    xAxis: { title: { text: '时间 (s)' } },
    yAxis: { title: { text: 'PV' } },
    color: ['#1890ff', '#ff4d4f'],
    tooltip: { shared: true, showCrosshairs: true },
  };

  return <Line {...config} />;
}
