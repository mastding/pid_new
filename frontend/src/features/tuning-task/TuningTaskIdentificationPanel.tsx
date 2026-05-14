import { Empty, Table, Tag, Typography } from 'antd';
import type { IdentificationAttempt, WindowAlgorithmFitSummary } from '@/types/tuning';
import { attemptFitKey } from './model';

interface TuningTaskIdentificationPanelProps {
  algorithmComparison: WindowAlgorithmFitSummary[];
  attempts: IdentificationAttempt[];
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  onSelectAttempt: (attemptKey: string) => void;
}

export function TuningTaskIdentificationPanel({
  algorithmComparison,
  attempts,
  formatNumber,
  formatPercentValue,
  onSelectAttempt,
}: TuningTaskIdentificationPanelProps) {
  return (
    <section className="agent-panel">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">辨识过程：所有轮次记录</div>
          <Typography.Text type="secondary">
            每行代表某一轮中“候选窗口 × 候选模型”的一次拟合尝试，按轮次和 fit_score 展示。
          </Typography.Text>
        </div>
        <Tag color="processing">{attempts.length} 次尝试</Tag>
      </div>
      {algorithmComparison.length ? (
        <Table<WindowAlgorithmFitSummary>
          className="detail-block"
          size="small"
          pagination={false}
          rowKey={(row) => `${row.algorithm}-${row.window_source}-${row.model_type}`}
          dataSource={algorithmComparison}
          columns={[
            { title: '窗口算法族', dataIndex: 'algorithm_label', render: (value, row) => String(value || row.algorithm || '-') },
            { title: '最佳窗口', dataIndex: 'window_source' },
            { title: '最佳模型', dataIndex: 'model_type', render: (value) => <Tag color="blue">{String(value || '-')}</Tag> },
            { title: 'fit_score', dataIndex: 'fit_score', render: (value) => formatNumber(value as number | undefined, 2) },
            { title: 'R²', dataIndex: 'r2_score', render: (value) => formatNumber(value as number | undefined, 3) },
            { title: 'NRMSE', dataIndex: 'normalized_rmse', render: (value) => formatPercentValue(value as number | undefined, 1) },
            { title: '置信度', dataIndex: 'confidence', render: (value) => formatPercentValue(value as number | undefined, 0) },
          ]}
        />
      ) : null}
      {attempts.length ? (
        <Table<IdentificationAttempt>
          size="small"
          rowKey={(row, index) => `${row.round ?? 0}-${row.model_type}-${row.window_source}-${index ?? 0}`}
          dataSource={attempts}
          pagination={{ pageSize: 8 }}
          onRow={(row) => ({
            onClick: () => {
              if (row.fit_preview?.points?.length) onSelectAttempt(attemptFitKey(row));
            },
          })}
          columns={[
            { title: 'Round', dataIndex: 'round', width: 80, render: (value) => `R${value ?? 0}` },
            { title: '模型', dataIndex: 'model_type', width: 100, render: (value) => <Tag color="blue">{value}</Tag> },
            { title: '算法族', dataIndex: 'window_algorithm', width: 150, render: (value, row) => row.window_algorithm_label || value || '-' },
            { title: '窗口', dataIndex: 'window_source', ellipsis: true },
            { title: '窗分', dataIndex: 'window_quality_score', render: (value) => formatNumber(value, 3) },
            { title: 'K', dataIndex: 'K', render: (value) => formatNumber(value, 3) },
            { title: 'T(s)', render: (_, row) => row.T1 && row.T2 ? `${formatNumber(row.T1, 1)}+${formatNumber(row.T2, 1)}` : formatNumber(row.T, 2) },
            { title: 'L(s)', dataIndex: 'L', render: (value) => formatNumber(value, 2) },
            { title: 'R²', dataIndex: 'r2_score', render: (value) => formatNumber(value, 3) },
            { title: 'NRMSE', dataIndex: 'normalized_rmse', render: (value) => formatPercentValue(value, 1) },
            { title: 'fit_score', dataIndex: 'fit_score', render: (value) => formatNumber(value, 2) },
            { title: '置信度', dataIndex: 'confidence', render: (value) => formatPercentValue(value, 0) },
            { title: '状态', dataIndex: 'success', render: (value, row) => value ? <Tag color="green">成功</Tag> : <Tag color="red">{row.error || '失败'}</Tag> },
          ]}
        />
      ) : (
        <Empty description="任务运行后会显示所有辨识尝试" />
      )}
    </section>
  );
}
