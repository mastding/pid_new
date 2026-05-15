import type { ReactNode } from 'react';
import { Line } from '@ant-design/charts';
import { Alert, Button, Descriptions, Empty, Progress, Select, Space, Table, Tag, Typography } from 'antd';

import type { HistoryWindow } from '@/services/api';
import type { IdentificationAttempt, IdentificationRefinementMeta, WindowAlgorithmFitSummary } from '@/types/tuning';
import { attemptFitKey } from '@/features/tuning-task/model';

interface ChartDatum {
  t: string | number | undefined;
  value: number | undefined;
  series: string;
}

interface ModelReliabilityPanelProps {
  windowAlgorithmSummary: Record<string, { total: number; usable: number }>;
  fitPreviewAttempts: IdentificationAttempt[];
  selectedFitAttempt?: IdentificationAttempt;
  fitPreviewChartData: ChartDatum[];
  selectedFitAttemptKey?: string;
  onSelectedFitAttemptKeyChange: (key: string) => void;
  taskAlgorithmComparison: WindowAlgorithmFitSummary[];
  deterministicRefinement?: IdentificationRefinementMeta;
  windows: HistoryWindow[];
  selectedWindow?: HistoryWindow;
  windowPreviewData: ChartDatum[];
  windowTable: ReactNode;
  chartLineTooltip: Record<string, unknown>;
  onOpenTuningTask: () => void;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  scorePercent: (value?: number) => number;
  scoreStatus: (value?: number) => 'exception' | 'success' | 'normal';
}

const fitPreviewColors = ['#35a7ff', '#28d7c5', '#ff9f43'];
const windowPreviewColors = ['#35a7ff', '#ff9f43', '#28d7c5'];

const fitPreviewAxis = {
  x: {
    title: 'X 轴：时间 / 采样点',
    titleFill: '#334155',
    titleFontSize: 12,
    titleFontWeight: 700,
    labelFill: '#475569',
    labelFontSize: 11,
    labelAutoHide: true,
    labelAutoRotate: true,
    lineStroke: '#cbd5e1',
    tickStroke: '#cbd5e1',
  },
  y: {
    title: 'Y 轴：PV / MV 数值',
    titleFill: '#334155',
    titleFontSize: 12,
    titleFontWeight: 700,
    labelFill: '#475569',
    labelFontSize: 12,
    lineStroke: '#cbd5e1',
    tickStroke: '#cbd5e1',
    gridStroke: '#d8e2ee',
    gridLineDash: [4, 4],
  },
};

const windowPreviewAxis = {
  x: {
    title: 'X 轴：窗口内相对时间 / 采样点',
    titleFill: '#334155',
    titleFontSize: 12,
    titleFontWeight: 700,
    labelFill: '#475569',
    labelFontSize: 11,
    labelAutoHide: true,
    labelAutoRotate: true,
    lineStroke: '#cbd5e1',
    tickStroke: '#cbd5e1',
  },
  y: {
    title: 'Y 轴：窗口 PV / MV 数值',
    titleFill: '#334155',
    titleFontSize: 12,
    titleFontWeight: 700,
    labelFill: '#475569',
    labelFontSize: 12,
    lineStroke: '#cbd5e1',
    tickStroke: '#cbd5e1',
    gridStroke: '#d8e2ee',
    gridLineDash: [4, 4],
  },
};

const lightLegend = {
  color: {
    position: 'top',
    itemLabelFill: '#334155',
    itemLabelFontSize: 13,
    itemLabelFontWeight: 600,
    markerSize: 10,
  },
};

const lightSlider = {
  height: 28,
  textStyle: { fill: '#64748b' },
  trendCfg: { lineStyle: { stroke: '#35a7ff' } },
  handlerStyle: { fill: '#ffffff', stroke: '#7fb8ff' },
};

export function ModelReliabilityPanel({
  windowAlgorithmSummary,
  fitPreviewAttempts,
  selectedFitAttempt,
  fitPreviewChartData,
  selectedFitAttemptKey,
  onSelectedFitAttemptKeyChange,
  taskAlgorithmComparison,
  deterministicRefinement,
  windows,
  selectedWindow,
  windowPreviewData,
  windowTable,
  chartLineTooltip,
  onOpenTuningTask,
  formatNumber,
  formatPercentValue,
  scorePercent,
  scoreStatus,
}: ModelReliabilityPanelProps) {
  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">窗口算法候选池</div>
            <Typography.Text type="secondary">同一份历史数据会尝试多类窗口，后续辨识按窗口质量分排序。</Typography.Text>
          </div>
        </div>
        <div className="kpi-grid">
          {Object.entries(windowAlgorithmSummary).length ? Object.entries(windowAlgorithmSummary).map(([name, item]) => (
            <div className="kpi-card" key={name}>
              <span>{name}</span>
              <strong>{item.usable}/{item.total}</strong>
              <em>可用/候选</em>
            </div>
          )) : (
            <div className="kpi-card">
              <span>候选池</span>
              <strong>-</strong>
              <em>等待窗口检测结果</em>
            </div>
          )}
        </div>
      </section>

      <section className="agent-panel chart-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">模型拟合曲线对比</div>
            <Typography.Text type="secondary">
              展示某一次“窗口 × 模型”的实测、仿真和阀位曲线；可切换查看不同辨识记录。
            </Typography.Text>
          </div>
          <Space wrap>
            <Tag color={fitPreviewAttempts.length ? 'processing' : 'default'}>
              {fitPreviewAttempts.length} 条曲线
            </Tag>
            <Select
              size="small"
              style={{ minWidth: 300 }}
              placeholder="选择拟合曲线"
              value={selectedFitAttemptKey}
              onChange={onSelectedFitAttemptKeyChange}
              options={fitPreviewAttempts.map((attempt) => ({
                value: attemptFitKey(attempt),
                label: `R${attempt.round ?? 0} · ${attempt.window_source || '-'} · ${attempt.model_type} · R²=${formatNumber(attempt.r2_score, 3)}`,
              }))}
            />
          </Space>
        </div>
        {selectedFitAttempt && fitPreviewChartData.length ? (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Descriptions bordered column={6} size="small" className="industrial-descriptions">
              <Descriptions.Item label="轮次">R{selectedFitAttempt.round ?? 0}</Descriptions.Item>
              <Descriptions.Item label="窗口">{selectedFitAttempt.window_source || '-'}</Descriptions.Item>
              <Descriptions.Item label="模型">{selectedFitAttempt.model_type}</Descriptions.Item>
              <Descriptions.Item label="R²">{formatNumber(selectedFitAttempt.r2_score, 3)}</Descriptions.Item>
              <Descriptions.Item label="NRMSE">{formatPercentValue(selectedFitAttempt.normalized_rmse, 1)}</Descriptions.Item>
              <Descriptions.Item label="置信度">{formatPercentValue(selectedFitAttempt.confidence, 0)}</Descriptions.Item>
              <Descriptions.Item label="K">{formatNumber(selectedFitAttempt.K, 4)}</Descriptions.Item>
              <Descriptions.Item label="T(s)" span={2}>
                {selectedFitAttempt.T1 && selectedFitAttempt.T2
                  ? `${formatNumber(selectedFitAttempt.T1, 2)} + ${formatNumber(selectedFitAttempt.T2, 2)}`
                  : formatNumber(selectedFitAttempt.T, 2)}
              </Descriptions.Item>
              <Descriptions.Item label="L(s)">{formatNumber(selectedFitAttempt.L, 2)}</Descriptions.Item>
              <Descriptions.Item label="fit_score">{formatNumber(selectedFitAttempt.fit_score, 2)}</Descriptions.Item>
              <Descriptions.Item label="算法族">{selectedFitAttempt.window_algorithm_label || selectedFitAttempt.window_algorithm || '-'}</Descriptions.Item>
            </Descriptions>
            {selectedFitAttempt.degenerate_T && (
              <Alert
                className="agent-alert"
                type="warning"
                showIcon
                message="该模型存在 T 塌缩惩罚"
                description="优化结果触碰或低于当前回路类型的时间常数合理下界，fit_score 已被惩罚，建议优先查看其它窗口或模型。"
              />
            )}
            <div className="chart-axis-note">
              <span>X 轴：时间 / 采样点</span>
              <span>Y 轴：PV 实测、PV 仿真、MV 数值</span>
            </div>
            <div className="chart-shell">
              <Line
                height={400}
                data={fitPreviewChartData}
                xField="t"
                yField="value"
                colorField="series"
                theme="classic"
                color={fitPreviewColors}
                scale={{ color: { range: fitPreviewColors } }}
                style={{ lineWidth: 2.2 }}
                padding={[34, 32, 88, 76]}
                axis={fitPreviewAxis}
                legend={lightLegend}
                xAxis={{
                  title: { text: 'X 轴：时间 / 采样点', style: { fill: '#334155', fontSize: 12, fontWeight: 700 } },
                  label: {
                    autoHide: true,
                    autoRotate: true,
                    style: { fill: '#475569', fontSize: 11 },
                    formatter: (text: string) => String(text).slice(5, 16),
                  },
                  line: { style: { stroke: '#cbd5e1' } },
                  tickLine: { style: { stroke: '#cbd5e1' } },
                }}
                yAxis={{
                  title: { text: 'Y 轴：PV / MV 数值', style: { fill: '#334155', fontSize: 12, fontWeight: 700 } },
                  label: { style: { fill: '#475569', fontSize: 12 } },
                  line: { style: { stroke: '#cbd5e1' } },
                  tickLine: { style: { stroke: '#cbd5e1' } },
                  grid: { line: { style: { stroke: '#d8e2ee', lineDash: [4, 4] } } },
                }}
                tooltip={chartLineTooltip}
                slider={lightSlider}
              />
            </div>
          </Space>
        ) : (
          <Empty description="暂无模型拟合曲线。请重新发起整定任务，后端会在成功辨识记录中返回拟合预览。" />
        )}
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">窗口算法族辨识效果对比</div>
            <Typography.Text type="secondary">整定任务运行后，按每类窗口算法族取最佳拟合结果，帮助判断下一轮应该优先换哪类窗口。</Typography.Text>
          </div>
          <Space wrap>
            <Tag color={taskAlgorithmComparison.length ? 'processing' : 'default'}>{taskAlgorithmComparison.length} 类算法族</Tag>
            <Button size="small" onClick={onOpenTuningTask}>去发起整定</Button>
          </Space>
        </div>
        {taskAlgorithmComparison.length ? (
          <Space direction="vertical" style={{ width: '100%' }}>
            {deterministicRefinement ? (
              <Alert
                className="agent-alert"
                type="info"
                showIcon
                message={`确定性策略推荐：${deterministicRefinement.recommended_algorithm_label || deterministicRefinement.recommended_algorithm || '-'}`}
                description={`${deterministicRefinement.rationale}${deterministicRefinement.recommended_window_source ? `；推荐窗口 ${deterministicRefinement.recommended_window_source}` : ''}`}
              />
            ) : null}
            <Table<WindowAlgorithmFitSummary>
              rowKey={(row) => `${row.algorithm}-${row.window_source}-${row.model_type}`}
              size="small"
              pagination={false}
              dataSource={taskAlgorithmComparison}
              columns={[
                { title: '窗口算法族', dataIndex: 'algorithm_label', render: (value, row) => value || row.algorithm || '-' },
                { title: '最佳窗口', dataIndex: 'window_source' },
                { title: '最佳模型', dataIndex: 'model_type', render: (value) => <Tag color="blue">{value || '-'}</Tag> },
                { title: '窗口质量分', dataIndex: 'window_quality_score', render: (value) => formatNumber(value, 3) },
                { title: '拟合分', dataIndex: 'fit_score', render: (value) => formatNumber(value, 2) },
                { title: 'R²', dataIndex: 'r2_score', render: (value) => formatNumber(value, 3) },
                { title: 'NRMSE', dataIndex: 'normalized_rmse', render: (value) => formatPercentValue(value, 1) },
                { title: '置信度', dataIndex: 'confidence', render: (value) => formatPercentValue(value, 0) },
              ]}
            />
          </Space>
        ) : (
          <Empty description="暂无算法族拟合对比。请先在“整定任务”中发起一次整定，完成模型辨识后这里会自动显示。" />
        )}
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">候选辨识窗口</div>
            <Typography.Text type="secondary">窗口表作为主入口，点击行后下方仅展示选中窗口预览。</Typography.Text>
          </div>
          <Tag color="blue">{windows.length} 个窗口</Tag>
        </div>
        {windowTable}
      </section>

      <section className="agent-panel chart-panel">
        <div className="panel-title">窗口 PV / MV 预览</div>
        {selectedWindow ? (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Descriptions bordered column={4} size="small" className="industrial-descriptions">
              <Descriptions.Item label="算法族">{selectedWindow.algorithm_label || selectedWindow.algorithm || '-'}</Descriptions.Item>
              <Descriptions.Item label="判据" span={3}>{selectedWindow.selection_basis || '-'}</Descriptions.Item>
              <Descriptions.Item label="窗口">{selectedWindow.source}</Descriptions.Item>
              <Descriptions.Item label="质量分">{selectedWindow.score}</Descriptions.Item>
              <Descriptions.Item label="相关性">{selectedWindow.corr}</Descriptions.Item>
              <Descriptions.Item label="激励幅度">{selectedWindow.amplitude}</Descriptions.Item>
            </Descriptions>
            <div className="score-grid compact-score-grid">
              {[
                ['MV激励', selectedWindow.score_breakdown?.mv_excitation],
                ['PV响应', selectedWindow.score_breakdown?.pv_response],
                ['滞后相关', selectedWindow.score_breakdown?.lag_correlation],
                ['饱和惩罚', selectedWindow.score_breakdown?.saturation_penalty],
                ['漂移惩罚', selectedWindow.score_breakdown?.drift_penalty],
              ].map(([label, value]) => (
                <div className="score-card" key={String(label)}>
                  <div className="score-title">{label}</div>
                  <Progress percent={scorePercent(value as number | undefined)} status={scoreStatus(value as number | undefined)} />
                </div>
              ))}
            </div>
            {selectedWindow.reasons?.length ? (
              <Alert
                className="agent-alert"
                type="warning"
                showIcon
                message="窗口风险原因"
                description={selectedWindow.reasons.join('；')}
              />
            ) : null}
            {windowPreviewData.length ? (
              <>
                <div className="chart-axis-note">
                  <span>X 轴：窗口内相对时间 / 采样点</span>
                  <span>Y 轴：窗口 PV / MV 数值</span>
                </div>
                <div className="chart-shell">
                  <Line
                    height={340}
                    data={windowPreviewData}
                    xField="t"
                    yField="value"
                    colorField="series"
                    theme="classic"
                    color={windowPreviewColors}
                    scale={{ color: { range: windowPreviewColors } }}
                    style={{ lineWidth: 2.1 }}
                    padding={[34, 32, 84, 76]}
                    axis={windowPreviewAxis}
                    legend={lightLegend}
                    slider={lightSlider}
                    xAxis={{
                      title: { text: 'X 轴：窗口内相对时间 / 采样点', style: { fill: '#334155', fontSize: 12, fontWeight: 700 } },
                      label: { autoHide: true, autoRotate: true, style: { fill: '#475569', fontSize: 11 } },
                      line: { style: { stroke: '#cbd5e1' } },
                      tickLine: { style: { stroke: '#cbd5e1' } },
                    }}
                    yAxis={{
                      title: { text: 'Y 轴：窗口 PV / MV 数值', style: { fill: '#334155', fontSize: 12, fontWeight: 700 } },
                      label: { style: { fill: '#475569', fontSize: 12 } },
                      line: { style: { stroke: '#cbd5e1' } },
                      tickLine: { style: { stroke: '#cbd5e1' } },
                      grid: { line: { style: { stroke: '#d8e2ee', lineDash: [4, 4] } } },
                    }}
                    tooltip={chartLineTooltip}
                  />
                </div>
              </>
            ) : <Empty description="暂无窗口预览" />}
          </Space>
        ) : <Empty description="请选择窗口" />}
      </section>
    </div>
  );
}
