import { Line } from '@ant-design/charts';
import type { SimulationTrace } from '@/types/tuning';

interface Props {
  simulation: SimulationTrace;
  title?: string;
  height?: number;
  /** Show MV axis (right). Default true */
  showMv?: boolean;
}

interface ChartPoint {
  t: number;
  value: number;
  series: string;
}

export default function SimulationChart({
  simulation,
  height = 280,
  showMv = true,
}: Props) {
  const { pv_history, mv_history, sp_history, dt } = simulation;
  const n = Math.min(pv_history.length, mv_history.length, sp_history.length);
  if (n === 0) return null;

  const data: ChartPoint[] = [];
  for (let i = 0; i < n; i++) {
    const t = parseFloat((i * dt).toFixed(1));
    data.push({ t, value: parseFloat(sp_history[i].toFixed(3)), series: 'SP' });
    data.push({ t, value: parseFloat(pv_history[i].toFixed(3)), series: 'PV' });
    if (showMv) {
      data.push({ t, value: parseFloat(mv_history[i].toFixed(3)), series: 'MV' });
    }
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
    yAxis: {
      title: { text: 'PV / SP (%)' },
    },
    color: ['#1890ff', '#52c41a', '#faad14'],
    tooltip: {
      shared: true,
      showCrosshairs: true,
    },
    // Down-sample if too many points to avoid browser slowdown
    sampling: n > 500 ? 'lttb' as const : undefined,
  };

  return <Line {...config} />;
}
