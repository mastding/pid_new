import type { Dayjs } from 'dayjs';
import type { Dispatch, SetStateAction } from 'react';
import { Alert, DatePicker, Descriptions, Empty, Progress, Select, Space, Table, Tag, Typography } from 'antd';

import type { HistoryLoop, HistoryLoopAssessment, HistoryLoopMonitoringSnapshot } from '@/services/api';
import type { FeatureRangePreset } from '@/features/monitoring/pageConfig';

type DiagnosisPlanKey = 'pid_diagnosis' | 'valve_diagnosis' | 'measurement_noise_diagnosis' | 'process_disturbance_diagnosis';
type Severity = 'high' | 'medium' | 'low' | 'normal';

interface DiagnosisOverviewPanelProps {
  assessment: HistoryLoopAssessment | null;
  monitoring?: HistoryLoopMonitoringSnapshot;
  selectedLoopId?: string;
  scopedLoops: HistoryLoop[];
  featureRangePreset: string;
  featureCustomRange: [Dayjs | null, Dayjs | null] | null;
  featureRangeOptions: Array<{ label: string; value: string; seconds?: number }>;
  onLoopChange: Dispatch<SetStateAction<string | undefined>>;
  onRangePresetChange: Dispatch<SetStateAction<FeatureRangePreset>>;
  onCustomRangeChange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  loopTypeLabel: (loop: HistoryLoop) => string;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
}

interface OscillationDiagnosisPanelProps {
  assessment: HistoryLoopAssessment | null;
  monitoring?: HistoryLoopMonitoringSnapshot;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
}

interface DiagnosisPlanPanelProps {
  activeSub: DiagnosisPlanKey;
}

interface DiagnosisLane {
  key: DiagnosisPlanKey | 'data_quality';
  title: string;
  severity: Severity;
  confidence: number;
  evidence: string;
  action: string;
  nextMenu: string;
}

const diagnosisPlan: Record<DiagnosisPlanKey, { title: string; desc: string; rows: Array<{ item: string; evidence: string; action: string }> }> = {
  pid_diagnosis: {
    title: 'PID参数诊断',
    desc: '判断过激、过保守、积分过强/过弱、响应慢和参数导致的振荡风险。',
    rows: [
      { item: '过激/过保守', evidence: '超调、调节时间、MV动作量、PV衰减比', action: '调整Kp、Ti、Td或切换保守整定策略' },
      { item: '积分问题', evidence: '稳态误差、积分累积、低频偏差', action: '检查积分时间和抗积分饱和逻辑' },
      { item: '参数诱发振荡', evidence: 'PV/MV同频、相位关系、闭环衰减', action: '降低比例/积分作用并复核阀门状态' },
    ],
  },
  valve_diagnosis: {
    title: '阀门/执行机构',
    desc: '识别疑似死区、回差、卡滞、MV变化无PV响应和方向不对称。',
    rows: [
      { item: '死区', evidence: '小幅MV变化后PV低于噪声阈值', action: '检查阀门定位器、阀杆摩擦和执行机构气源' },
      { item: '回差', evidence: 'MV上升/下降路径对应PV响应不同', action: '做正反向小阶跃测试确认' },
      { item: '卡滞', evidence: 'MV突跳、PV滞后、动作呈锯齿', action: '优先检修执行机构，再考虑整定' },
    ],
  },
  measurement_noise_diagnosis: {
    title: '测量与噪声',
    desc: '识别传感器噪声、尖峰、漂移、平顶坏点和采样异常。',
    rows: [
      { item: '高频噪声', evidence: '高通残差、差分MAD、SNR', action: '检查仪表、滤波参数和采样配置' },
      { item: '尖峰/坏点', evidence: '离群点比例、单点突变、平顶时长', action: '清洗数据并排查采集链路' },
      { item: '测量漂移', evidence: '长期单向漂移且MV/SP无对应变化', action: '校验仪表零点和量程' },
    ],
  },
  process_disturbance_diagnosis: {
    title: '扰动与工艺',
    desc: '识别外扰、负荷变化、非线性、不同工况增益差异和过程约束。',
    rows: [
      { item: '外扰', evidence: 'PV变化领先MV动作，SP无变化', action: '追踪上游负荷或公用工程扰动' },
      { item: '非线性', evidence: '不同工作区间局部增益差异显著', action: '分工况建模或采用增益调度' },
      { item: '过程约束', evidence: 'MV饱和期间PV仍无法回归', action: '确认设备能力和工艺边界' },
    ],
  },
};

function BackendBadge({ implemented }: { implemented?: boolean }) {
  return implemented ? <Tag color="green">已接后端</Tag> : <Tag color="default">规则说明</Tag>;
}

function severityFromScore(score?: number | null): Severity {
  if (score === undefined || score === null) return 'normal';
  if (score < 0.45) return 'high';
  if (score < 0.68) return 'medium';
  if (score < 0.85) return 'low';
  return 'normal';
}

function severityColor(severity: Severity) {
  if (severity === 'high') return 'red';
  if (severity === 'medium') return 'orange';
  if (severity === 'low') return 'gold';
  return 'green';
}

function severityText(severity: Severity) {
  if (severity === 'high') return '优先排查';
  if (severity === 'medium') return '需要确认';
  if (severity === 'low') return '持续观察';
  return '暂未触发';
}

function confidenceBySeverity(severity: Severity, score?: number | null) {
  if (severity === 'normal') return Math.round((score ?? 0.9) * 100);
  return Math.max(35, Math.min(96, Math.round((1 - (score ?? 0.7)) * 100)));
}

function buildDiagnosisLanes(
  assessment: HistoryLoopAssessment | null,
  monitoring?: HistoryLoopMonitoringSnapshot,
): DiagnosisLane[] {
  const dataScore = monitoring?.data_health?.score ?? assessment?.data_quality?.score;
  const stabilityScore = monitoring?.stability?.score ?? assessment?.performance?.stability_score;
  const actuatorScore = monitoring?.pv_mv_behavior?.score ?? assessment?.performance?.pv_mv_behavior_score;
  const constraintScore = monitoring?.constraints?.score ?? assessment?.performance?.constraint_score;
  const responseScore = monitoring?.response_observability?.score ?? assessment?.identification_suitability?.response_observability_score;
  const readinessDecision = assessment?.tuning_readiness?.decision ?? assessment?.summary?.decision;
  const flags = assessment?.diagnostics.flags ?? [];
  const hasHighFlag = flags.some((flag) => ['high', 'critical', 'blocked'].includes(String(flag.severity).toLowerCase()));

  const dataSeverity = severityFromScore(dataScore);
  const pidSeverity = hasHighFlag ? 'high' : severityFromScore(stabilityScore);
  const valveSeverity = severityFromScore(Math.min(actuatorScore ?? 1, constraintScore ?? 1));
  const noiseSeverity = severityFromScore(dataScore);
  const processSeverity: Severity = readinessDecision === 'blocked'
    ? 'high'
    : readinessDecision === 'cautious' || monitoring?.operating_condition?.tuning_suitability === 'cautious'
      ? 'medium'
      : severityFromScore(responseScore);

  return [
    {
      key: 'pid_diagnosis',
      title: '控制器参数/闭环动态',
      severity: pidSeverity,
      confidence: confidenceBySeverity(pidSeverity, stabilityScore),
      evidence: `稳定性 ${Math.round((stabilityScore ?? 0) * 100) || '-'} 分；诊断标记 ${flags.length} 项`,
      action: pidSeverity === 'normal' ? '暂不优先怀疑 PID 参数，保留巡检。' : '先确认是否存在阀门/外扰问题，再评估比例与积分是否过强。',
      nextMenu: 'PID参数诊断',
    },
    {
      key: 'valve_diagnosis',
      title: '阀门/执行机构',
      severity: valveSeverity,
      confidence: confidenceBySeverity(valveSeverity, Math.min(actuatorScore ?? 1, constraintScore ?? 1)),
      evidence: `PV/MV 行为 ${Math.round((actuatorScore ?? 0) * 100) || '-'} 分；约束 ${Math.round((constraintScore ?? 0) * 100) || '-'} 分`,
      action: valveSeverity === 'normal' ? '未见明显执行机构证据。' : '优先查饱和、死区、卡滞和定位器状态，避免把机械问题误整定。',
      nextMenu: '阀门/执行机构',
    },
    {
      key: 'measurement_noise_diagnosis',
      title: '测量/采样质量',
      severity: noiseSeverity,
      confidence: confidenceBySeverity(noiseSeverity, dataScore),
      evidence: `数据健康 ${Math.round((dataScore ?? 0) * 100) || '-'} 分；SNR ${monitoring?.data_health?.pv_snr_db?.toFixed?.(1) ?? '-'}`,
      action: noiseSeverity === 'normal' ? '数据质量可支持诊断。' : '先处理缺失、尖峰、采样间隔和仪表噪声，再解释模型或整定结论。',
      nextMenu: '测量与噪声',
    },
    {
      key: 'process_disturbance_diagnosis',
      title: '工况/外扰/过程约束',
      severity: processSeverity,
      confidence: confidenceBySeverity(processSeverity, responseScore),
      evidence: `工况 ${monitoring?.operating_condition?.condition_label ?? '-'}；响应可观测 ${Math.round((responseScore ?? 0) * 100) || '-'} 分`,
      action: processSeverity === 'normal' ? '工况证据暂不阻断后续分析。' : '先确认负荷、上下游扰动和约束边界，再进入整定或模型解释。',
      nextMenu: '扰动与工艺',
    },
    {
      key: 'data_quality',
      title: '整定准入影响',
      severity: readinessDecision === 'blocked' ? 'high' : readinessDecision === 'cautious' ? 'medium' : dataSeverity,
      confidence: confidenceBySeverity(dataSeverity, assessment?.tuning_readiness?.score),
      evidence: `准入结论：${assessment?.summary?.decision_text || assessment?.tuning_readiness?.decision || '-'}`,
      action: assessment?.summary?.recommended_next_action_text || '按优先根因完成确认后，再进入整定任务。',
      nextMenu: '整定准备度',
    },
  ];
}

export function DiagnosisOverviewPanel({
  assessment,
  monitoring,
  selectedLoopId,
  scopedLoops,
  featureRangePreset,
  featureCustomRange,
  featureRangeOptions,
  onLoopChange,
  onRangePresetChange,
  onCustomRangeChange,
  loopTypeLabel,
  formatNumber,
  formatPercentValue,
}: DiagnosisOverviewPanelProps) {
  const lanes = buildDiagnosisLanes(assessment, monitoring);
  const flags = assessment?.diagnostics.flags ?? [];
  const primary = [...lanes].sort((a, b) => {
    const rank = { high: 3, medium: 2, low: 1, normal: 0 };
    return rank[b.severity] - rank[a.severity] || b.confidence - a.confidence;
  })[0];
  const overallScore = monitoring?.overall_score ?? assessment?.performance?.score;

  if (!assessment && !monitoring) {
    return (
      <section className="agent-panel">
        <div className="panel-title">诊断总览</div>
        <Empty description="请选择回路，系统会基于最近 8 小时监控快照生成根因分流。" />
      </section>
    );
  }

  return (
    <div className="page-stack diagnosis-overview-page">
      <section className="agent-panel diagnosis-summary-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">诊断总览</div>
            <Typography.Text type="secondary">这里仅做根因分流和排查顺序，不重复趋势频谱、控制性能或整定任务细节。</Typography.Text>
          </div>
          <Space wrap className="diagnosis-overview-controls">
            <Select
              showSearch
              size="small"
              className="performance-loop-select"
              placeholder="选择回路"
              value={selectedLoopId}
              onChange={onLoopChange}
              optionFilterProp="label"
              options={scopedLoops.map((loop) => ({
                value: loop.loop_id,
                label: `${loop.loop_id} · ${loopTypeLabel(loop)}`,
              }))}
            />
            <Select
              size="small"
              className="performance-range-select"
              value={featureRangePreset}
              onChange={(value) => onRangePresetChange(value as FeatureRangePreset)}
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
            <Tag color={severityColor(primary.severity)}>{severityText(primary.severity)}</Tag>
          </Space>
        </div>
        <div className="diagnosis-summary-grid">
          <div className="diagnosis-primary-card">
            <span>首要怀疑方向</span>
            <strong>{primary.title}</strong>
            <em>{primary.action}</em>
          </div>
          <div className="diagnosis-metric-card">
            <span>综合健康</span>
            <strong>{formatPercentValue(overallScore, 0)}</strong>
            <Progress percent={Math.round((overallScore ?? 0) * 100)} status={overallScore && overallScore < 0.65 ? 'exception' : 'normal'} />
          </div>
          <div className="diagnosis-metric-card">
            <span>风险标记</span>
            <strong>{flags.length}</strong>
            <em>{flags.length ? '需要人工确认' : '未见明显阻断项'}</em>
          </div>
          <div className="diagnosis-metric-card">
            <span>准入建议</span>
            <strong>{assessment?.summary?.decision_text || assessment?.tuning_readiness?.decision || '-'}</strong>
            <em>{assessment?.summary?.recommended_next_action_text || '保持巡检'}</em>
          </div>
        </div>
      </section>

      <section className="agent-panel">
        <div className="panel-title">根因分流矩阵</div>
        <Table<DiagnosisLane>
          size="small"
          pagination={false}
          rowKey="key"
          dataSource={lanes}
          columns={[
            { title: '方向', dataIndex: 'title', width: 180 },
            { title: '判断', dataIndex: 'severity', width: 110, render: (value: Severity) => <Tag color={severityColor(value)}>{severityText(value)}</Tag> },
            { title: '证据强度', dataIndex: 'confidence', width: 170, render: (value: number) => <Progress percent={value} size="small" /> },
            { title: '核心证据', dataIndex: 'evidence', ellipsis: true },
            { title: '下一步', dataIndex: 'action', ellipsis: true },
            { title: '进入菜单', dataIndex: 'nextMenu', width: 130, render: (value: string) => <Tag color="blue">{value}</Tag> },
          ]}
        />
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div className="panel-title">关键证据摘要</div>
          <Tag color={monitoring?.status === 'warning' ? 'orange' : monitoring?.status === 'alarm' ? 'red' : 'green'}>
            {monitoring?.status || 'assessment'}
          </Tag>
        </div>
        <Descriptions bordered column={4} size="small" className="industrial-descriptions">
          <Descriptions.Item label="数据健康">{formatPercentValue(monitoring?.data_health?.score ?? assessment?.data_quality?.score, 0)}</Descriptions.Item>
          <Descriptions.Item label="稳定性">{formatPercentValue(monitoring?.stability?.score ?? assessment?.performance?.stability_score, 0)}</Descriptions.Item>
          <Descriptions.Item label="PV/MV行为">{formatPercentValue(monitoring?.pv_mv_behavior?.score ?? assessment?.performance?.pv_mv_behavior_score, 0)}</Descriptions.Item>
          <Descriptions.Item label="约束健康">{formatPercentValue(monitoring?.constraints?.score ?? assessment?.performance?.constraint_score, 0)}</Descriptions.Item>
          <Descriptions.Item label="响应可观测">{formatPercentValue(monitoring?.response_observability?.score ?? assessment?.identification_suitability?.response_observability_score, 0)}</Descriptions.Item>
          <Descriptions.Item label="工况">{monitoring?.operating_condition?.condition_label ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="缺失率">{formatPercentValue(assessment?.data_quality?.missing_ratio ?? monitoring?.data_health?.missing_ratio, 2)}</Descriptions.Item>
          <Descriptions.Item label="窗口可用">{formatNumber(assessment?.identification_suitability?.usable_window_count ?? assessment?.identifiability?.usable_window_count, 0)}</Descriptions.Item>
        </Descriptions>
      </section>

      <section className="agent-panel">
        <div className="panel-title">诊断标记</div>
        {flags.length ? (
          <Table
            size="small"
            pagination={false}
            rowKey={(row) => `${row.type}-${row.message}`}
            dataSource={flags}
            columns={[
              { title: '类型', dataIndex: 'type', width: 160 },
              { title: '级别', dataIndex: 'severity', width: 100, render: (value: string) => <Tag color={value === 'high' || value === 'critical' ? 'red' : 'orange'}>{value}</Tag> },
              { title: '诊断信息', dataIndex: 'message', ellipsis: true },
            ]}
          />
        ) : <Alert type="success" showIcon message="未发现明显数据质量或可辨识性阻断风险。" />}
      </section>
    </div>
  );
}

export function OscillationDiagnosisPanel({
  assessment,
  monitoring,
  formatNumber,
  formatPercentValue,
}: OscillationDiagnosisPanelProps) {
  return (
    <section className="agent-panel">
      <div className="panel-title">振荡监测</div>
      {assessment || monitoring ? (
        <Descriptions column={4} bordered size="small" className="industrial-descriptions">
          <Descriptions.Item label="是否振荡">{monitoring?.stability?.oscillation_detected ?? assessment?.diagnostics.oscillation?.detected ? '检测到' : '未检测到'}</Descriptions.Item>
          <Descriptions.Item label="严重度">{monitoring?.stability?.oscillation_severity ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="置信度">{formatPercentValue(monitoring?.stability?.oscillation_confidence, 1)}</Descriptions.Item>
          <Descriptions.Item label="主周期">{String(monitoring?.stability?.pv_dominant_period_s ?? assessment?.diagnostics.oscillation?.period_sec ?? '-')}s</Descriptions.Item>
          <Descriptions.Item label="主频能量">{formatPercentValue(monitoring?.stability?.pv_dominant_power_ratio, 1)}</Descriptions.Item>
          <Descriptions.Item label="零交叉">{formatNumber(monitoring?.stability?.pv_zero_crossing_per_hour, 2)}/h</Descriptions.Item>
          <Descriptions.Item label="相位提示">{monitoring?.stability?.phase_hint ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="SNR">{formatNumber(monitoring?.data_health?.pv_snr_db, 2)} dB</Descriptions.Item>
        </Descriptions>
      ) : <Empty description="暂无诊断结果" />}
    </section>
  );
}

export function DiagnosisPlanPanel({ activeSub }: DiagnosisPlanPanelProps) {
  const plan = diagnosisPlan[activeSub];

  return (
    <section className="agent-panel">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">{plan.title}</div>
          <Typography.Text type="secondary">{plan.desc}</Typography.Text>
        </div>
        <BackendBadge implemented={false} />
      </div>
      <Table
        size="small"
        pagination={false}
        rowKey="item"
        dataSource={plan.rows}
        columns={[
          { title: '诊断项', dataIndex: 'item', width: 160 },
          { title: '证据来源', dataIndex: 'evidence' },
          { title: '建议动作', dataIndex: 'action' },
        ]}
      />
    </section>
  );
}

export type { DiagnosisPlanKey };
