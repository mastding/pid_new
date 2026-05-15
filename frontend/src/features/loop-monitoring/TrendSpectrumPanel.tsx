import type { Dispatch, ReactNode, SetStateAction } from 'react';
import type { Dayjs } from 'dayjs';
import { Button, DatePicker, Select, Space, Switch, Typography } from 'antd';
import { SyncOutlined } from '@ant-design/icons';

import type { HistoryLoop, HistoryLoopAssessment, HistoryLoopMonitoring, LoopSeriesResp } from '@/services/api';
import { SpectrumSummaryPanel } from '@/features/loop-monitoring/SpectrumSummaryPanel';
import { TrendChartPanel } from '@/features/loop-monitoring/TrendChartPanel';
import { TrendQueryDetails } from '@/features/loop-monitoring/TrendQueryDetails';

type SelectOption = {
  label: string;
  value: string;
  seconds?: number;
};

interface TrendSpectrumPanelProps {
  selectedLoopId?: string;
  selectedLoop?: HistoryLoop;
  scopedLoops: HistoryLoop[];
  series: LoopSeriesResp | null;
  seriesLoading: boolean;
  trendPreset: string;
  trendPointLimit: string;
  trendSplitYAxis: boolean;
  trendCustomRange: [Dayjs | null, Dayjs | null] | null;
  trendPresetOptions: SelectOption[];
  trendPointLimitOptions: SelectOption[];
  loopTypeLabel: Record<string, string>;
  assessment: HistoryLoopAssessment | null;
  monitoring?: HistoryLoopMonitoring['monitoring'];
  oscillationDetected: boolean;
  chart: ReactNode;
  onLoopChange: Dispatch<SetStateAction<string | undefined>>;
  onTrendPresetChange: Dispatch<SetStateAction<string>>;
  onTrendPointLimitChange: Dispatch<SetStateAction<string>>;
  onTrendSplitYAxisChange: Dispatch<SetStateAction<boolean>>;
  onTrendCustomRangeChange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  onRefresh: () => void;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  formatOscillationEvidence: (detected?: boolean, confidence?: number | null) => string;
  formatOscillationPhaseHint: (detected?: boolean, phaseHint?: string | null) => string;
}

export function TrendSpectrumPanel({
  selectedLoopId,
  selectedLoop,
  scopedLoops,
  series,
  seriesLoading,
  trendPreset,
  trendPointLimit,
  trendSplitYAxis,
  trendCustomRange,
  trendPresetOptions,
  trendPointLimitOptions,
  loopTypeLabel,
  assessment,
  monitoring,
  oscillationDetected,
  chart,
  onLoopChange,
  onTrendPresetChange,
  onTrendPointLimitChange,
  onTrendSplitYAxisChange,
  onTrendCustomRangeChange,
  onRefresh,
  formatNumber,
  formatPercentValue,
  formatOscillationEvidence,
  formatOscillationPhaseHint,
}: TrendSpectrumPanelProps) {
  const rangeLabel = trendPreset === 'custom'
    ? `${trendCustomRange?.[0]?.format('YYYY-MM-DD HH:mm:ss') ?? '-'} ~ ${trendCustomRange?.[1]?.format('YYYY-MM-DD HH:mm:ss') ?? '-'}`
    : trendPresetOptions.find((item) => item.value === trendPreset)?.label ?? '-';
  const pointLimitLabel = trendPointLimit === 'all'
    ? '全量点'
    : `${trendPointLimitOptions.find((item) => item.value === trendPointLimit)?.label ?? trendPointLimit}`;

  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">趋势查询</div>
            <Typography.Text type="secondary">
              选择回路和时间段后，趋势曲线会按后端时间过滤重新加载。
            </Typography.Text>
          </div>
          <Space wrap>
            <Select
              showSearch
              style={{ minWidth: 360 }}
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
              style={{ width: 150 }}
              value={trendPreset}
              onChange={onTrendPresetChange}
              options={trendPresetOptions.map((item) => ({ label: item.label, value: item.value }))}
            />
            <Select
              style={{ width: 170 }}
              value={trendPointLimit}
              onChange={onTrendPointLimitChange}
              options={trendPointLimitOptions}
            />
            <Space className="inline-switch">
              <Typography.Text type="secondary">PV/MV 分轴</Typography.Text>
              <Switch checked={trendSplitYAxis} onChange={onTrendSplitYAxisChange} />
            </Space>
            {trendPreset === 'custom' && (
              <DatePicker.RangePicker
                showTime
                value={trendCustomRange}
                onChange={onTrendCustomRangeChange}
              />
            )}
            <Button
              icon={<SyncOutlined />}
              loading={seriesLoading}
              onClick={onRefresh}
            >
              刷新趋势
            </Button>
          </Space>
        </div>
        <TrendQueryDetails
          selectedLoop={selectedLoop}
          loopTypeLabel={selectedLoop ? loopTypeLabel[selectedLoop.loop_type] ?? selectedLoop.loop_type : '-'}
          series={series}
          rangeLabel={rangeLabel}
          pointLimitLabel={pointLimitLabel}
        />
      </section>
      <TrendChartPanel
        selectedLoop={selectedLoop}
        series={series}
        loading={seriesLoading}
        chart={chart}
      />
      <SpectrumSummaryPanel
        assessment={assessment}
        monitoring={monitoring}
        oscillationDetected={oscillationDetected}
        formatNumber={formatNumber}
        formatPercentValue={formatPercentValue}
        formatOscillationEvidence={formatOscillationEvidence}
        formatOscillationPhaseHint={formatOscillationPhaseHint}
      />
    </div>
  );
}
