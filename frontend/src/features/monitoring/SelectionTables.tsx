import { Space, Table, Tag, Typography } from 'antd';

import type { HistoryLoop, HistoryWindow } from '@/services/api';

interface LoopSelectionTableProps {
  loops: HistoryLoop[];
  selectedLoopId?: string;
  loading: boolean;
  loopTypeLabels: Record<string, string>;
  onSelectLoop: (loopId: string) => void;
}

export function LoopSelectionTable({
  loops,
  selectedLoopId,
  loading,
  loopTypeLabels,
  onSelectLoop,
}: LoopSelectionTableProps) {
  return (
    <Table
      rowKey="loop_id"
      columns={[
        {
          title: '回路',
          dataIndex: 'loop_id',
          render: (value: string, row: HistoryLoop) => (
            <Space>
              <Typography.Text strong>{value}</Typography.Text>
              <Tag color="blue">{loopTypeLabels[row.loop_type] ?? row.loop_type}</Tag>
            </Space>
          ),
        },
        { title: '来源文件', dataIndex: 'source_filename', ellipsis: true },
        { title: '采样', dataIndex: 'sampling_time', render: (value: number) => `${value}s` },
        { title: '点数', dataIndex: 'rows' },
        {
          title: '时长',
          render: (_: unknown, row: HistoryLoop) => {
            const start = row.start_time ? new Date(row.start_time).getTime() : NaN;
            const end = row.end_time ? new Date(row.end_time).getTime() : NaN;
            if (Number.isNaN(start) || Number.isNaN(end) || end <= start) return '-';
            const h = (end - start) / 3_600_000;
            return h >= 1 ? `${h.toFixed(1)} h` : `${Math.round(h * 60)} min`;
          },
        },
        {
          title: 'PV 范围',
          render: (_: unknown, row: HistoryLoop) => {
            const min = row.pv_min;
            const max = row.pv_max;
            return (typeof min === 'number' && typeof max === 'number') ? `${min.toFixed(2)} ~ ${max.toFixed(2)}` : '-';
          },
        },
      ]}
      dataSource={loops}
      loading={loading}
      pagination={{ pageSize: 8 }}
      rowSelection={{
        type: 'radio',
        selectedRowKeys: selectedLoopId ? [selectedLoopId] : [],
        onChange: (keys) => onSelectLoop(String(keys[0])),
      }}
      onRow={(record) => ({ onClick: () => onSelectLoop(record.loop_id) })}
    />
  );
}

interface WindowSelectionTableProps {
  windows: HistoryWindow[];
  selectedWindowIndex?: number;
  scorePercent: (value?: number) => number;
  onSelectWindow: (index: number) => void;
}

export function WindowSelectionTable({
  windows,
  selectedWindowIndex,
  scorePercent,
  onSelectWindow,
}: WindowSelectionTableProps) {
  return (
    <Table
      size="small"
      rowKey="index"
      dataSource={windows}
      columns={[
        {
          title: '窗口',
          dataIndex: 'source',
          render: (value: string, row: HistoryWindow) => (
            <Space>
              <Tag color={row.usable ? 'green' : 'red'}>{row.usable ? '可用' : '风险'}</Tag>
              <Typography.Text strong>{value || `window_${row.index}`}</Typography.Text>
            </Space>
          ),
        },
        {
          title: '算法族',
          dataIndex: 'algorithm',
          render: (value: string, row: HistoryWindow) => row.algorithm_label || value || '-',
        },
        { title: '类型', dataIndex: 'type' },
        { title: '质量分', dataIndex: 'score', render: (value: number) => value.toFixed(3) },
        {
          title: '子分',
          dataIndex: 'score_breakdown',
          render: (_: unknown, row: HistoryWindow) => {
            const breakdown = row.score_breakdown ?? {};
            return `MV ${scorePercent(breakdown.mv_excitation)}/PV ${scorePercent(breakdown.pv_response)}/相关 ${scorePercent(breakdown.lag_correlation)}`;
          },
        },
        { title: '相关性', dataIndex: 'corr', render: (value: number) => value.toFixed(3) },
        { title: 'MV 幅度', dataIndex: 'mv_span' },
        { title: 'PV 幅度', dataIndex: 'pv_span' },
        { title: '点数', dataIndex: 'n_points' },
      ]}
      pagination={{ pageSize: 6 }}
      rowSelection={{
        type: 'radio',
        selectedRowKeys: selectedWindowIndex === undefined ? [] : [selectedWindowIndex],
        onChange: (keys) => onSelectWindow(Number(keys[0])),
      }}
      onRow={(record) => ({ onClick: () => onSelectWindow(record.index) })}
    />
  );
}
