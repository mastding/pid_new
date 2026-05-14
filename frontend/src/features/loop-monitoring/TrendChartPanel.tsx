import type { ReactNode } from 'react';
import { Empty, Space, Tag, Typography } from 'antd';
import type { HistoryLoop, LoopSeriesResp } from '@/services/api';

interface TrendChartPanelProps {
  selectedLoop?: HistoryLoop;
  series: LoopSeriesResp | null;
  loading: boolean;
  chart: ReactNode;
}

export function TrendChartPanel({
  selectedLoop,
  series,
  loading,
  chart,
}: TrendChartPanelProps) {
  return (
    <section className="agent-panel chart-panel">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">趋势曲线</div>
          <Typography.Text type="secondary">PV/MV 长周期趋势。</Typography.Text>
        </div>
        <Space wrap>
          <Tag color="blue">{series?.sampled_points ?? 0} 点</Tag>
          <Tag color="cyan">{selectedLoop?.sampling_time ?? '-'}s</Tag>
        </Space>
      </div>
      {loading ? <Empty description="正在加载趋势数据..." /> : chart}
    </section>
  );
}
