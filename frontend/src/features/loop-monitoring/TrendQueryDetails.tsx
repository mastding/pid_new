import { Descriptions } from 'antd';
import type { HistoryLoop, LoopSeriesResp } from '@/services/api';

interface TrendQueryDetailsProps {
  selectedLoop?: HistoryLoop;
  loopTypeLabel: string;
  series: LoopSeriesResp | null;
  rangeLabel: string;
  pointLimitLabel: string;
}

export function TrendQueryDetails({
  selectedLoop,
  loopTypeLabel,
  series,
  rangeLabel,
  pointLimitLabel,
}: TrendQueryDetailsProps) {
  return (
    <Descriptions bordered size="small" column={4} className="industrial-descriptions">
      <Descriptions.Item label="当前回路">{selectedLoop?.loop_id ?? '-'}</Descriptions.Item>
      <Descriptions.Item label="回路类型">{selectedLoop ? loopTypeLabel : '-'}</Descriptions.Item>
      <Descriptions.Item label="原始时间范围" span={2}>
        {selectedLoop?.start_time || '-'} ~ {selectedLoop?.end_time || '-'}
      </Descriptions.Item>
      <Descriptions.Item label="当前显示点数">{series?.sampled_points ?? 0}/{series?.total_points ?? 0}</Descriptions.Item>
      <Descriptions.Item label="采样周期">{selectedLoop?.sampling_time ?? '-'}s</Descriptions.Item>
      <Descriptions.Item label="时间筛选" span={2}>{rangeLabel}</Descriptions.Item>
      <Descriptions.Item label="点数模式">{pointLimitLabel}</Descriptions.Item>
      <Descriptions.Item label="显示说明" span={3}>
        {series && series.sampled_points < series.total_points
          ? `当前为抽样趋势，后端从 ${series.total_points} 点中返回 ${series.sampled_points} 点。`
          : '当前时间范围内为全量点显示。'}
      </Descriptions.Item>
    </Descriptions>
  );
}
