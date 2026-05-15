import { Line } from '@ant-design/charts';
import { Empty } from 'antd';

export interface LoopTrendPoint {
  t: string | number;
  value: number;
  series: string;
}

function formatChartTooltipValue(value: unknown, digits = 3) {
  if (typeof value === 'number') return Number.isNaN(value) ? '-' : value.toFixed(digits);
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(digits) : value;
  }
  return '-';
}

export function chartSeriesColor(series?: string) {
  if (!series) return '#35a7ff';
  if (series.includes('MV')) return '#ff9f43';
  if (series.includes('SV') || series.includes('SP')) return '#28d7c5';
  if (series.includes('仿真') || series.includes('拟合')) return '#28d7c5';
  return '#35a7ff';
}

export const chartLineTooltip = {
  title: (datum: { t?: string | number }) => `X：${datum?.t ?? '-'}`,
  items: [
    (datum: { series?: string; value?: unknown }) => {
      const value = formatChartTooltipValue(datum.value);
      const series = datum.series || '数值';
      return {
        name: `${series}：${value}`,
        value: '',
        color: chartSeriesColor(series),
      };
    },
  ],
};

interface LoopTrendChartProps {
  data: LoopTrendPoint[];
  height?: number;
  splitYAxis?: boolean;
  xAxisMode?: string;
}

function TrendLine({
  data,
  height,
  colors,
  xAxisMode,
}: {
  data: LoopTrendPoint[];
  height: number;
  colors: string[];
  xAxisMode?: string;
}) {
  return (
    <div className="chart-shell">
      <Line
        height={height}
        data={data}
        xField="t"
        yField="value"
        colorField="series"
        theme="classic"
        color={colors}
        scale={{ color: { range: colors } }}
        style={{ lineWidth: 2.1 }}
        padding={[28, 28, 58, 58]}
        axis={{
          x: {
            title: '',
            titleFill: '#334155',
            titleFontSize: 12,
            titleFontWeight: 700,
            labelFill: '#334155',
            labelFontSize: 11,
            labelAutoHide: true,
            labelAutoRotate: true,
            lineStroke: '#cbd5e1',
            tickStroke: '#cbd5e1',
          },
          y: {
            title: '',
            titleFill: '#334155',
            titleFontSize: 12,
            titleFontWeight: 700,
            labelFill: '#334155',
            labelFontSize: 12,
            lineStroke: '#cbd5e1',
            tickStroke: '#cbd5e1',
            gridStroke: '#d8e2ee',
            gridLineDash: [4, 4],
          },
        }}
        legend={{
          color: {
            position: 'top',
            itemLabelFill: '#334155',
            itemLabelFontSize: 13,
            itemLabelFontWeight: 600,
            markerSize: 10,
          },
        }}
        slider={{
          height: 28,
          textStyle: { fill: '#64748b' },
          trendCfg: { lineStyle: { stroke: colors[0] ?? '#35a7ff' } },
          handlerStyle: { fill: '#ffffff', stroke: '#7fb8ff' },
        }}
        xAxis={{
          type: xAxisMode === 'timestamp' ? 'timeCat' : 'linear',
          title: { text: '', style: { fill: '#334155', fontSize: 12, fontWeight: 700 } },
          label: { autoHide: true, autoRotate: true, style: { fill: '#334155', fontSize: 11, fontWeight: 600 } },
          line: { style: { stroke: '#cbd5e1' } },
          tickLine: { style: { stroke: '#cbd5e1' } },
        }}
        yAxis={{
          title: { text: '', style: { fill: '#334155', fontSize: 12, fontWeight: 700 } },
          label: { style: { fill: '#334155', fontSize: 12, fontWeight: 600 } },
          line: { style: { stroke: '#cbd5e1' } },
          tickLine: { style: { stroke: '#cbd5e1' } },
          grid: { line: { style: { stroke: '#d8e2ee', lineDash: [4, 4] } } },
        }}
        tooltip={chartLineTooltip}
      />
    </div>
  );
}

export function LoopTrendChart({
  data,
  height = 360,
  splitYAxis = false,
  xAxisMode,
}: LoopTrendChartProps) {
  if (!data.length) return <Empty description="暂无趋势数据" />;

  const pvTrendData = data.filter((item) => item.series === 'PV' || item.series === 'SV');
  const mvTrendData = data.filter((item) => item.series === 'MV');

  return (
    <>
      <div className="chart-axis-note">
        <span>X 轴：时间 / 采样点</span>
        <span>{splitYAxis ? '分轴：上图 PV/SV，下图 MV，各自坐标' : 'Y 轴：PV / SV / MV 数值'}</span>
      </div>
      {splitYAxis ? (
        <div className="split-trend-grid">
          <div className="split-trend-panel">
            <div className="split-trend-title">PV / SV 趋势</div>
            <TrendLine
              data={pvTrendData}
              height={Math.max(260, Math.floor(height * 0.58))}
              colors={['#35a7ff', '#ff9f43']}
              xAxisMode={xAxisMode}
            />
          </div>
          <div className="split-trend-panel">
            <div className="split-trend-title">MV 趋势</div>
            <TrendLine
              data={mvTrendData}
              height={Math.max(220, Math.floor(height * 0.46))}
              colors={['#28d7c5']}
              xAxisMode={xAxisMode}
            />
          </div>
        </div>
      ) : (
        <TrendLine
          data={data}
          height={height}
          colors={['#35a7ff', '#28d7c5', '#ff9f43']}
          xAxisMode={xAxisMode}
        />
      )}
    </>
  );
}
