import { Line } from '@ant-design/charts';
import { Alert, Button, Descriptions, Empty, Progress, Select, Space, Table, Tag, Typography } from 'antd';

import type { HistoryLoopAssessment, HistoryWindow } from '@/services/api';
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
  chartLineTooltip: Record<string, unknown>;
  onOpenTuningTask: () => void;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  scorePercent: (value?: number) => number;
  scoreStatus: (value?: number) => 'exception' | 'success' | 'normal';
  assessment: HistoryLoopAssessment | null;
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
  chartLineTooltip,
  onOpenTuningTask,
  formatNumber,
  formatPercentValue,
  scorePercent,
  scoreStatus,
  assessment,
}: ModelReliabilityPanelProps) {
  const usableWindows = windows.filter((item) => item.usable).length;
  const bestWindow = [...windows].sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0];
  const bestFit = [...fitPreviewAttempts].sort((a, b) => (b.fit_score ?? 0) - (a.fit_score ?? 0))[0];
  const dataScore = assessment?.data_quality?.score;
  const readinessScore = assessment?.tuning_readiness?.score ?? assessment?.readiness?.score;
  const identScore = assessment?.identification_suitability?.score ?? assessment?.identifiability?.score;
  const excitationScore = assessment?.identification_suitability?.excitation_score;
  const responseScore = assessment?.identification_suitability?.response_observability_score;
  const reliabilityInputs = [
    dataScore,
    identScore,
    excitationScore,
    responseScore,
    readinessScore,
    bestWindow?.score,
    bestFit?.confidence,
  ].filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  const reliabilityScore = reliabilityInputs.length
    ? reliabilityInputs.reduce((sum, value) => sum + Math.max(0, Math.min(1, value)), 0) / reliabilityInputs.length
    : undefined;
  const reliabilityLevel = reliabilityScore === undefined
    ? '待补充'
    : reliabilityScore >= 0.82
      ? '可信'
      : reliabilityScore >= 0.65
        ? '谨慎可用'
        : '暂不可信';
  const reliabilityColor = reliabilityLevel === '可信' ? 'green' : reliabilityLevel === '谨慎可用' ? 'orange' : reliabilityLevel === '暂不可信' ? 'red' : 'default';
  const gateRows = [
    {
      item: '数据质量',
      value: dataScore,
      status: dataScore === undefined ? '待补充' : dataScore >= 0.85 ? '通过' : dataScore >= 0.68 ? '谨慎' : '阻断',
      evidence: `缺失率 ${formatPercentValue(assessment?.data_quality?.missing_ratio, 2)}，连续性 ${formatPercentValue(assessment?.data_quality?.continuity_score, 0)}`,
      action: '先保证 PV/MV/SP 数据可信，模型可靠性不直接重复趋势细节。',
    },
    {
      item: '激励充分性',
      value: excitationScore,
      status: excitationScore === undefined || excitationScore === null ? '待补充' : excitationScore >= 0.7 ? '通过' : excitationScore >= 0.5 ? '谨慎' : '阻断',
      evidence: `可用窗口 ${usableWindows}/${windows.length || assessment?.identification_suitability?.window_count || '-'}`,
      action: 'MV 激励不足时，模型参数往往只是曲线拟合，不应直接用于整定。',
    },
    {
      item: '响应可观测性',
      value: responseScore,
      status: responseScore === undefined || responseScore === null ? '待补充' : responseScore >= 0.7 ? '通过' : responseScore >= 0.5 ? '谨慎' : '阻断',
      evidence: `方向置信 ${formatPercentValue(assessment?.identification_suitability?.direction_confidence, 0)}`,
      action: '确认 PV 对 MV 的方向、滞后和相关性，再解释模型 K/T/L。',
    },
    {
      item: '窗口代表性',
      value: bestWindow?.score ?? assessment?.identification_suitability?.best_window_score,
      status: bestWindow?.usable ? '通过' : bestWindow ? '谨慎' : '待补充',
      evidence: bestWindow ? `${bestWindow.source}，${bestWindow.selection_basis || '候选窗口'}` : '等待窗口检测',
      action: '这里只看窗口是否支撑建模，窗口筛选细节仍在“候选辨识窗口”。',
    },
    {
      item: '拟合可信度',
      value: bestFit?.confidence,
      status: bestFit ? (bestFit.confidence && bestFit.confidence >= 0.75 ? '通过' : '谨慎') : '待补充',
      evidence: bestFit ? `${bestFit.model_type}，R²=${formatNumber(bestFit.r2_score, 3)}，NRMSE=${formatPercentValue(bestFit.normalized_rmse, 1)}` : '尚未运行辨识拟合',
      action: '拟合结果只作为最后一层证据；没有拟合时仍可先判断数据和窗口是否具备建模条件。',
    },
  ];

  return (
    <div className="page-stack">
      <section className="agent-panel model-reliability-summary">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">模型可靠性</div>
            <Typography.Text type="secondary">判断“这个回路现在能不能信模型”，不重复整定任务执行过程，也不替代候选窗口菜单。</Typography.Text>
          </div>
          <Tag color={reliabilityColor}>{reliabilityLevel}</Tag>
        </div>
        <div className="model-reliability-grid">
          <div className="model-reliability-score">
            <span>综合可信度</span>
            <strong>{reliabilityScore === undefined ? '-' : formatPercentValue(reliabilityScore, 0)}</strong>
            <Progress percent={Math.round((reliabilityScore ?? 0) * 100)} status={reliabilityScore !== undefined && reliabilityScore < 0.65 ? 'exception' : 'normal'} />
          </div>
          <div className="model-reliability-card">
            <span>可用窗口</span>
            <strong>{usableWindows}/{windows.length || assessment?.identification_suitability?.window_count || '-'}</strong>
            <em>{bestWindow ? `最佳：${bestWindow.source}` : '等待窗口检测'}</em>
          </div>
          <div className="model-reliability-card">
            <span>最佳拟合</span>
            <strong>{bestFit ? formatNumber(bestFit.r2_score, 3) : '-'}</strong>
            <em>{bestFit ? `${bestFit.model_type} · R²` : '等待整定任务产生拟合'}</em>
          </div>
          <div className="model-reliability-card">
            <span>工程建议</span>
            <strong>{reliabilityLevel === '可信' ? '可进入评审' : reliabilityLevel === '谨慎可用' ? '保守使用' : '先补证据'}</strong>
            <em>{assessment?.summary?.recommended_next_action_text || '先确认数据、激励和响应方向'}</em>
          </div>
        </div>
      </section>

      <section className="agent-panel">
        <div className="panel-title">可靠性门控</div>
        <Table
          size="small"
          pagination={false}
          rowKey="item"
          dataSource={gateRows}
          columns={[
            { title: '门控项', dataIndex: 'item', width: 140 },
            { title: '状态', dataIndex: 'status', width: 100, render: (value: string) => <Tag color={value === '通过' ? 'green' : value === '谨慎' ? 'orange' : value === '阻断' ? 'red' : 'default'}>{value}</Tag> },
            { title: '得分', dataIndex: 'value', width: 150, render: (value?: number) => <Progress percent={scorePercent(value)} size="small" status={scoreStatus(value)} /> },
            { title: '证据', dataIndex: 'evidence', ellipsis: true },
            { title: '专家判断', dataIndex: 'action', ellipsis: true },
          ]}
        />
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">窗口算法候选池</div>
            <Typography.Text type="secondary">仅汇总各算法族能否提供建模证据；窗口逐项筛选留在“候选辨识窗口”。</Typography.Text>
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
          <Empty description="暂无拟合曲线。当前页仍可先依据数据质量、激励、响应可观测性判断是否值得建模。" />
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
            <div className="panel-title">建模证据窗口摘要</div>
            <Typography.Text type="secondary">只展示最能支撑模型可信度的窗口摘要，避免与候选窗口菜单重复。</Typography.Text>
          </div>
          <Tag color="blue">{windows.length} 个窗口</Tag>
        </div>
        {windows.length ? (
          <Table<HistoryWindow>
            size="small"
            pagination={false}
            rowKey={(row) => `${row.source}-${row.index}`}
            dataSource={[...windows].sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, 5)}
            columns={[
              { title: '窗口', dataIndex: 'source', width: 150 },
              { title: '算法族', dataIndex: 'algorithm_label', width: 150, render: (value, row) => value || row.algorithm || '-' },
              { title: '可用', dataIndex: 'usable', width: 80, render: (value: boolean) => <Tag color={value ? 'green' : 'orange'}>{value ? '可用' : '谨慎'}</Tag> },
              { title: '质量分', dataIndex: 'score', width: 150, render: (value?: number) => <Progress percent={scorePercent(value)} size="small" status={scoreStatus(value)} /> },
              { title: '建模意义', dataIndex: 'selection_basis', ellipsis: true, render: (value, row) => value || row.reasons?.join('；') || '-' },
            ]}
          />
        ) : <Empty description="暂无窗口摘要。请在候选辨识窗口页运行窗口检测。" />}
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
