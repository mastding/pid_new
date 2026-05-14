import { Descriptions, Empty } from 'antd';
import type { HistoryLoopFeatures } from '@/services/api';

interface LoopProfileRawStatsPanelProps {
  loopFeatures: HistoryLoopFeatures | null;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
}

export function LoopProfileRawStatsPanel({
  loopFeatures,
  formatNumber,
  formatPercentValue,
}: LoopProfileRawStatsPanelProps) {
  return (
    <section className="agent-panel compact-facts">
      <div className="panel-title">原始统计</div>
      {loopFeatures ? (
        <Descriptions bordered size="small" column={4} className="industrial-descriptions">
          <Descriptions.Item label="行数">{loopFeatures.data_profile?.row_count ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="有效行">{loopFeatures.data_profile?.valid_row_count ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="中位采样">{formatNumber(loopFeatures.data_profile?.sample_time_median_s, 1)}s</Descriptions.Item>
          <Descriptions.Item label="P95 间隔">{formatNumber(loopFeatures.data_profile?.sample_interval_p95_s, 1)}s</Descriptions.Item>
          <Descriptions.Item label="P99 间隔">{formatNumber(loopFeatures.data_profile?.sample_interval_p99_s, 1)}s</Descriptions.Item>
          <Descriptions.Item label="不规则采样">{formatPercentValue(loopFeatures.data_profile?.irregular_sample_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="长间隔数">{loopFeatures.data_profile?.long_gap_count ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="重复时间戳">{loopFeatures.data_profile?.duplicate_timestamp_count ?? '-'}</Descriptions.Item>
        </Descriptions>
      ) : (
        <Empty description="暂无画像数据" />
      )}
    </section>
  );
}
