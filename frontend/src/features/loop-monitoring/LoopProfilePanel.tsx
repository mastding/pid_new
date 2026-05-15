import type { Dispatch, SetStateAction } from 'react';
import type { Dayjs } from 'dayjs';
import { Button, DatePicker, Descriptions, Empty, Select, Space, Tag, Typography } from 'antd';
import { SyncOutlined } from '@ant-design/icons';

import type { HistoryLoop, HistoryLoopAssessment, HistoryLoopFeatures, HistoryLoopMonitoring } from '@/services/api';
import { LoopProfileConstraintPanel } from '@/features/loop-monitoring/LoopProfileConstraintPanel';
import { LoopProfileDataQualityPanel } from '@/features/loop-monitoring/LoopProfileDataQualityPanel';
import { LoopProfilePerformancePanel } from '@/features/loop-monitoring/LoopProfilePerformancePanel';
import { LoopProfilePvMvPanel } from '@/features/loop-monitoring/LoopProfilePvMvPanel';
import { LoopProfileRawStatsPanel } from '@/features/loop-monitoring/LoopProfileRawStatsPanel';

type FeatureRangeOption = {
  label: string;
  value: string;
  seconds?: number;
};

interface LoopProfilePanelProps {
  selectedLoopId?: string;
  selectedLoop?: HistoryLoop;
  scopedLoops: HistoryLoop[];
  loopFeatures: HistoryLoopFeatures | null;
  assessment: HistoryLoopAssessment | null;
  monitoring?: HistoryLoopMonitoring['monitoring'];
  featureRangePreset: string;
  featureCustomRange: [Dayjs | null, Dayjs | null] | null;
  featureRangeOptions: FeatureRangeOption[];
  featureLoading: boolean;
  loopTypeLabel: Record<string, string>;
  onLoopChange: Dispatch<SetStateAction<string | undefined>>;
  onRangePresetChange: Dispatch<SetStateAction<string>>;
  onCustomRangeChange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  onRefresh: () => void;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  formatRange: (min?: number | null, max?: number | null, digits?: number) => string;
  scorePercent: (value?: number) => number;
  tagColor: (level?: string) => string;
  formatProcessDirection: (value?: string | null) => string;
  formatProcessDirectionBasis: (value?: string | null) => string;
  formatHarrisBasis: (value?: string) => string;
  formatCpkBasis: (value?: string) => string;
  monitoringStatusColor: (status?: string) => string;
  monitoringStatusText: (status?: string) => string;
}

export function LoopProfilePanel({
  selectedLoopId,
  selectedLoop,
  scopedLoops,
  loopFeatures,
  assessment,
  monitoring,
  featureRangePreset,
  featureCustomRange,
  featureRangeOptions,
  featureLoading,
  loopTypeLabel,
  onLoopChange,
  onRangePresetChange,
  onCustomRangeChange,
  onRefresh,
  formatNumber,
  formatPercentValue,
  formatRange,
  scorePercent,
  tagColor,
  formatProcessDirection,
  formatProcessDirectionBasis,
  formatHarrisBasis,
  formatCpkBasis,
  monitoringStatusColor,
  monitoringStatusText,
}: LoopProfilePanelProps) {
  return (
    <div className="page-stack">
      <section className="agent-panel profile-panel compact-profile">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">单回路画像</div>
            <Typography.Text type="secondary">集中展示资产信息、量程、采样、原始统计与约束饱和摘要。</Typography.Text>
          </div>
          <Space wrap>
            <Select
              showSearch
              size="small"
              style={{ minWidth: 300 }}
              placeholder="选择回路"
              value={selectedLoopId}
              onChange={onLoopChange}
              optionFilterProp="label"
              options={scopedLoops.map((loop) => ({
                value: loop.loop_id,
                label: `${loop.loop_id} · ${loopTypeLabel[loop.loop_type] ?? loop.loop_type}`,
              }))}
            />
            <Select
              size="small"
              style={{ width: 140 }}
              value={featureRangePreset}
              onChange={onRangePresetChange}
              options={featureRangeOptions.map((item) => ({ label: item.label, value: item.value }))}
            />
            {featureRangePreset === 'custom' && (
              <DatePicker.RangePicker
                size="small"
                showTime
                value={featureCustomRange}
                onChange={onCustomRangeChange}
              />
            )}
            <Button
              size="small"
              icon={<SyncOutlined />}
              loading={featureLoading}
              onClick={onRefresh}
            >
              刷新区间指标
            </Button>
            <Tag color="blue">{selectedLoop ? loopTypeLabel[selectedLoop.loop_type] ?? selectedLoop.loop_type : '-'}</Tag>
            <Tag color={loopFeatures ? 'cyan' : 'default'}>{loopFeatures ? '画像已加载' : '画像待加载'}</Tag>
          </Space>
        </div>
        {selectedLoop ? (
          <div className="profile-compact-grid">
            <div className="profile-nameplate">
              <span>回路位号</span>
              <strong>{selectedLoop.loop_id}</strong>
              <em>{selectedLoop.source_filename || '历史导入回路数据'}</em>
            </div>
            <Descriptions bordered size="small" column={4} className="industrial-descriptions">
              <Descriptions.Item label="采样周期">{formatNumber(loopFeatures?.data_profile?.sample_time_median_s ?? selectedLoop.sampling_time, 0)}s</Descriptions.Item>
              <Descriptions.Item label="数据点数">{loopFeatures?.data_profile?.row_count ?? selectedLoop.rows}</Descriptions.Item>
              <Descriptions.Item label="有效点数">{loopFeatures?.data_profile?.valid_row_count ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="总时长">{loopFeatures?.data_profile?.duration_h === undefined ? '-' : `${formatNumber(loopFeatures.data_profile.duration_h, 1)}h`}</Descriptions.Item>
              <Descriptions.Item label="PV 范围">{formatRange(loopFeatures?.pv_stats?.min ?? selectedLoop.pv_min, loopFeatures?.pv_stats?.max ?? selectedLoop.pv_max, 2)}</Descriptions.Item>
              <Descriptions.Item label="MV 范围">{formatRange(loopFeatures?.mv_stats?.min ?? selectedLoop.mv_min, loopFeatures?.mv_stats?.max ?? selectedLoop.mv_max, 2)}</Descriptions.Item>
              <Descriptions.Item label="开始时间">{loopFeatures?.data_profile?.time_start || selectedLoop.start_time || '-'}</Descriptions.Item>
              <Descriptions.Item label="结束时间">{loopFeatures?.data_profile?.time_end || selectedLoop.end_time || '-'}</Descriptions.Item>
            </Descriptions>
          </div>
        ) : <Empty description="暂无选中回路" />}
      </section>
      <LoopProfileRawStatsPanel
        loopFeatures={loopFeatures}
        formatNumber={formatNumber}
        formatPercentValue={formatPercentValue}
      />
      <LoopProfileDataQualityPanel
        assessment={assessment}
        monitoring={monitoring}
        scorePercent={scorePercent}
        formatNumber={formatNumber}
        formatPercentValue={formatPercentValue}
        tagColor={tagColor}
      />
      <LoopProfilePvMvPanel
        loopFeatures={loopFeatures}
        monitoring={monitoring}
        formatNumber={formatNumber}
        formatPercentValue={formatPercentValue}
        formatProcessDirection={formatProcessDirection}
        formatProcessDirectionBasis={formatProcessDirectionBasis}
      />
      <LoopProfileConstraintPanel
        loopFeatures={loopFeatures}
        monitoring={monitoring}
        formatNumber={formatNumber}
        formatPercentValue={formatPercentValue}
        statusColor={monitoringStatusColor}
        statusText={monitoringStatusText}
      />
      <LoopProfilePerformancePanel
        loopFeatures={loopFeatures}
        formatNumber={formatNumber}
        formatPercentValue={formatPercentValue}
        formatHarrisBasis={formatHarrisBasis}
        formatCpkBasis={formatCpkBasis}
      />
    </div>
  );
}
