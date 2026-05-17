import { Line } from '@ant-design/charts';
import { Alert, Empty, Space, Tabs, Tag, Typography } from 'antd';
import type { SimulationScenarioPack, SimulationTrace, SimulationTraceMap } from '@/types/tuning';

interface TuningSimulationPanelProps {
  simulation?: SimulationTrace | null;
  traces?: SimulationTraceMap;
  scenario?: SimulationScenarioPack;
  formatNumber: (value?: number | null, digits?: number) => string;
}

interface ChartPoint {
  t: number;
  value: number;
  series: string;
}

function buildSeries(trace: SimulationTrace, includeMv: boolean) {
  const n = Math.min(
    trace.pv_history?.length ?? 0,
    trace.sp_history?.length ?? 0,
    includeMv ? (trace.mv_history?.length ?? 0) : Number.MAX_SAFE_INTEGER,
  );
  const rows: ChartPoint[] = [];
  for (let i = 0; i < n; i += 1) {
    const t = Number((i * trace.dt).toFixed(2));
    rows.push({ t, value: Number(trace.sp_history[i].toFixed(4)), series: 'SP' });
    rows.push({ t, value: Number(trace.pv_history[i].toFixed(4)), series: 'PV' });
    if (includeMv) rows.push({ t, value: Number(trace.mv_history[i].toFixed(4)), series: 'MV' });
  }
  return rows;
}

function mvSaturation(trace: SimulationTrace) {
  const mv = trace.mv_history ?? [];
  if (!mv.length) return { pct: 0, count: 0, firstTime: null as number | null };
  let count = 0;
  let firstTime: number | null = null;
  for (let i = 0; i < mv.length; i += 1) {
    if (mv[i] <= 0.5 || mv[i] >= 99.5) {
      count += 1;
      if (firstTime === null) firstTime = i * trace.dt;
    }
  }
  return { pct: (count / mv.length) * 100, count, firstTime };
}

function traceItems(traces?: SimulationTraceMap, fallback?: SimulationTrace | null) {
  const items = [
    ['nominal_sp_step', '标称阶跃'],
    ['reverse_sp_step', '反向阶跃'],
    ['robustness_worst_case', '鲁棒性最差'],
  ] as const;
  const available = items
    .map(([key, label]) => ({ key, label, trace: traces?.[key] }))
    .filter((item) => (item.trace?.pv_history?.length ?? 0) > 0);
  if (available.length) return available;
  if (fallback?.pv_history?.length) return [{ key: 'nominal_sp_step', label: '标称阶跃', trace: fallback }];
  return [];
}

function SimulationTraceChart({
  trace,
  scenario,
  formatNumber,
}: {
  trace: SimulationTrace;
  scenario?: SimulationScenarioPack;
  formatNumber: (value?: number | null, digits?: number) => string;
}) {
  const pvData = buildSeries(trace, false);
  const mvData = buildSeries(trace, true).filter((item) => item.series === 'MV');
  const sat = mvSaturation(trace);
  const scenarioMeta = scenario?.scenarios?.find((item) => item.id === trace.scenario_id);

  const commonLine = {
    xField: 't',
    yField: 'value',
    colorField: 'series',
    theme: 'classic' as const,
    height: 260,
    padding: [28, 28, 48, 58] as [number, number, number, number],
    axis: {
      x: { title: '时间(s)', labelAutoHide: true, labelAutoRotate: true },
      y: { title: '' },
    },
    legend: {
      color: {
        position: 'top' as const,
        itemLabelFill: '#334155',
        itemLabelFontWeight: 600,
      },
    },
    slider: { height: 24 },
  };

  return (
    <div className="simulation-review-block">
      <div className="simulation-review-meta">
        <Space wrap>
          <Tag color={trace.is_stable === false ? 'red' : 'green'}>{trace.is_stable === false ? '不稳定' : '稳定'}</Tag>
          {trace.score != null && <Tag color="blue">评分 {formatNumber(trace.score, 1)}</Tag>}
          {trace.overshoot_percent != null && <Tag color="orange">超调 {formatNumber(trace.overshoot_percent, 1)}%</Tag>}
          {trace.settling_time_s != null && <Tag color="purple">调节时间 {formatNumber(trace.settling_time_s, 1)}s</Tag>}
          {scenarioMeta && (
            <Tag color="cyan">
              SP {formatNumber(scenarioMeta.sp_initial, 3)} → {formatNumber(scenarioMeta.sp_final, 3)}
            </Tag>
          )}
          <Tag color={sat.pct > 5 ? 'red' : sat.pct > 0 ? 'orange' : 'green'}>MV饱和 {formatNumber(sat.pct, 1)}%</Tag>
        </Space>
      </div>

      {sat.pct > 0 && (
        <Alert
          className="agent-alert"
          type={sat.pct > 5 ? 'error' : 'warning'}
          showIcon
          message="MV 饱和区间"
          description={`共有 ${sat.count} 个采样点接近 0%/100% 输出限幅，首次出现在 ${formatNumber(sat.firstTime, 1)}s。`}
        />
      )}

      <div className="simulation-chart-grid">
        <div className="simulation-chart-panel">
          <div className="simulation-chart-title">PV / SP 响应</div>
          <Line
            {...commonLine}
            data={pvData}
            color={['#2563eb', '#14b8a6']}
            scale={{ color: { range: ['#2563eb', '#14b8a6'] } }}
            style={{ lineWidth: 2.1 }}
          />
        </div>
        <div className="simulation-chart-panel">
          <div className="simulation-chart-title">MV 输出</div>
          <Line
            {...commonLine}
            data={mvData}
            height={220}
            color={['#f97316']}
            scale={{ color: { range: ['#f97316'] } }}
            style={{ lineWidth: 2.1 }}
          />
          <div className="simulation-limit-note">
            <span>限幅参考：0% / 100%</span>
            <span>接近限幅按 ≤0.5% 或 ≥99.5% 标记</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function TuningSimulationPanel({
  simulation,
  traces,
  scenario,
  formatNumber,
}: TuningSimulationPanelProps) {
  const items = traceItems(traces, simulation);
  if (!items.length) return <Empty description="暂无闭环仿真曲线" />;

  return (
    <div className="simulation-review">
      <div className="simulation-review-head">
        <div>
          <div className="simulation-review-title">闭环仿真曲线</div>
          <Typography.Text type="secondary">
            PV/SP 与 MV 分图展示，专家复核时重点看超调、调节时间、MV饱和和鲁棒性最差场景。
          </Typography.Text>
        </div>
        <Tag color="geekblue">{scenario?.basis || '仿真评估'}</Tag>
      </div>
      <Tabs
        items={items.map((item) => ({
          key: item.key,
          label: item.label,
          children: item.trace ? (
            <SimulationTraceChart trace={item.trace} scenario={scenario} formatNumber={formatNumber} />
          ) : null,
        }))}
      />
    </div>
  );
}
