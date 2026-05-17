import type { Dispatch, SetStateAction } from 'react';
import type { Dayjs } from 'dayjs';
import { AuditOutlined, SyncOutlined } from '@ant-design/icons';
import { Alert, Button, Collapse, DatePicker, Descriptions, Empty, Select, Space, Table, Tag, Typography } from 'antd';

import type { HistoryLoop, HistoryLoopFeatures } from '@/services/api';
import type {
  LlmThinkingEvent,
  WindowAlgorithmFamilySummary,
  WindowAlgorithmPlanItem,
  WindowPolicyFieldUsage,
  WindowPolicyResult,
  WindowSelectionMeta,
  WindowSelectionPolicy,
} from '@/types/tuning';
import type { TaskStageDataMap, TaskStageStatusMap, TaskStatus } from './model';

type WindowFlowStepStatus = 'waiting' | 'running' | 'done';

const WINDOW_ALGORITHM_FAMILY_LABELS: Record<string, string> = {
  sp_step: '设定值阶跃',
  sv_step: '设定值阶跃',
  step_up: '设定值上阶跃',
  step_down: '设定值下阶跃',
  mv_step: 'MV 阶跃',
  mv_ramp: 'MV 斜坡',
  steady_disturbance: '稳态扰动',
  steady_disturbance_scan: '稳态扰动扫描',
  rolling_scan: '滚动兜底扫描',
  mv_fallback: 'MV 兜底扫描',
  largest_mv_change: '最大 MV 变化',
  steady_segment: '稳态片段',
  load_change: '负荷变化',
  disturbance_recovery: '扰动恢复',
  oscillation_segment: '振荡片段',
};

const POLICY_FIELD_LABELS: Record<string, string> = {
  loop_id: '回路位号',
  loop_type: '回路类型',
  policy_version: '策略版本',
  confidence: '策略置信度',
  preferred_algorithm_families: '优先算法族',
  deprioritized_algorithm_families: '降权算法族',
  disabled_algorithm_families: '禁用算法族',
  algorithm_plan: '算法族执行计划',
  expected_gain_sign: '预期增益方向',
  min_mv_excitation: '最小 MV 激励',
  min_sp_excitation: '最小 SP 激励',
  min_pv_response: '最小 PV 响应',
  max_mv_saturation_ratio: '最大 MV 饱和比例',
  max_pv_noise_ratio: '最大 PV 噪声比例',
  max_drift_ratio: '最大漂移比例',
  expected_dead_time_range_s: '预期纯滞后范围',
  expected_time_constant_range_s: '预期时间常数范围',
  min_window_points: '最小窗口点数',
  min_window_duration_s: '最小窗口时长',
  max_window_points: '最大窗口点数',
  pre_window_s: '前置窗口',
  post_window_s: '后置窗口',
  steady_scan_window_s: '稳态扫描窗口',
  steady_scan_step_s: '稳态扫描步长',
  merge_gap_s: '合并间隔',
  max_candidates_per_family: '每族最大候选数',
  allowed_operating_states: '允许工况',
  avoid_operating_states: '规避工况',
  scoring_weights: '策略评分权重',
  hard_guards: '硬性准入规则',
  soft_penalties: '软性扣分规则',
  rationale: '策略说明',
  ontology_facts: '本体事实',
  source: '来源',
  name: '名称',
  action: '动作',
  process_direction: '过程方向',
  evidence: '证据',
  raw_answer: '原始回答',
  family: '算法族',
  state: '执行策略',
  reason: '原因',
  weight: '权重',
  code: '规则代码',
  threshold: '阈值',
  level: '级别',
};

const POLICY_FIELD_LABEL_ALIASES: Record<string, string> = {
  'preferred algorithm families': '优先算法族',
  'deprioritized algorithm families': '降权算法族',
  'disabled algorithm families': '禁用算法族',
  'algorithm family execution plan': '算法族执行计划',
  'minimum MV excitation': '最小 MV 激励',
  'minimum SP excitation': '最小 SP 激励',
  'minimum PV response': '最小 PV 响应',
  'maximum MV saturation ratio': '最大 MV 饱和比例',
  'maximum PV noise ratio': '最大 PV 噪声比例',
  'maximum PV drift ratio': '最大 PV 漂移比例',
  'expected dead-time range': '预期纯滞后范围',
  'expected time-constant range': '预期时间常数范围',
  'expected process gain sign': '预期过程增益方向',
  'minimum window points': '最小窗口点数',
  'minimum window duration': '最小窗口时长',
  'maximum window points': '最大窗口点数',
  'pre-event window length': '事件前窗口长度',
  'post-event window length': '事件后窗口长度',
  'steady scan window length': '稳态扫描窗口长度',
  'steady scan step': '稳态扫描步长',
  'event merge gap': '事件合并间隔',
  'maximum candidates per family': '每个算法族最大候选数',
  'allowed operating states': '允许工况',
  'avoided operating states': '规避工况',
  'policy scoring weights': '策略评分权重',
  'hard guards': '硬性准入规则',
  'soft penalties': '软性扣分规则',
  'policy rationale': '策略说明',
  'ontology facts': '本体事实',
};

const POLICY_NOTE_LABELS: Record<string, string> = {
  algorithm_plan: '控制每个算法族是否运行、降权或禁用。',
  preferred_algorithm_families: '参与算法族计划生成，并用于策略一致性评分。',
  deprioritized_algorithm_families: '参与算法族计划生成，并对对应候选施加软性扣分。',
  disabled_algorithm_families: '阻止禁用算法族运行，并硬性过滤对应候选。',
  min_mv_excitation: '提高 MV 阶跃/稳态扰动的激励阈值，过滤 MV 激励不足的窗口。',
  min_sp_excitation: '提高 SP 阶跃检测阈值。',
  min_pv_response: '过滤 PV 响应低于策略阈值的窗口。',
  max_mv_saturation_ratio: '过滤稳态扰动窗口，并对饱和候选进行扣分或提示。',
  max_drift_ratio: '对慢漂移主导的窗口进行扣分或过滤。',
  min_window_points: '控制稳态扫描最小长度，并过滤过短候选窗口。',
  max_window_points: '限制阶跃、斜坡和兜底窗口的事件后截取长度。',
  pre_window_s: '控制事件窗口的前置基线长度。',
  post_window_s: '控制事件窗口的后置时长，并影响 MV 斜坡检测窗口。',
  steady_scan_window_s: '控制稳态扰动滚动扫描窗口长度。',
  steady_scan_step_s: '控制稳态扰动滚动扫描步长。',
  merge_gap_s: '控制跨算法族事件合并距离。',
  max_candidates_per_family: '限制 MV 斜坡和稳态扰动算法族的候选数量。',
  expected_dead_time_range_s: '作为辨识/模型评审上下文下传，当前窗口生成器不直接消费。',
  expected_time_constant_range_s: '作为辨识/模型评审上下文下传，当前窗口生成器不直接消费。',
  expected_gain_sign: '作为模型合理性上下文下传，当前窗口生成器不直接消费。',
  max_pv_noise_ratio: '用于审计和页面展示，当前确定性窗口算法暂不直接消费。',
  min_window_duration_s: '用于审计和页面展示，具体窗口长度当前由前后置窗口和稳态扫描参数决定。',
  allowed_operating_states: '窗口评分会识别 operating_state，并过滤不在本体允许集合内的候选窗口。',
  avoid_operating_states: '窗口评分会识别 operating_state，并阻断或扣分本体规避工况中的候选窗口。',
};

const WINDOW_FLOW_STEPS = [
  { key: 'profile', title: '1 数据画像', desc: '读取回路原始特征' },
  { key: 'ontology', title: '2 本体检索', desc: '查询本体与回路上下文' },
  { key: 'policy', title: '3 策略生成', desc: '生成窗口算法族策略 JSON' },
  { key: 'algorithm', title: '4 算法族运行', desc: '按策略驱动算法模块产出候选窗口' },
  { key: 'llm', title: '5 大模型评审', desc: '结合画像、本体和候选窗口做解释性判断' },
  { key: 'gate', title: '6 准入结论', desc: '判断是否允许进入正式辨识' },
] as const;

type FeatureRangeOption = {
  label: string;
  value: string;
  seconds?: number;
};

interface WindowCandidatesPanelProps {
  selectedLoopId?: string;
  selectedLoop?: HistoryLoop;
  scopedLoops: HistoryLoop[];
  loopTypeLabels: Record<string, string>;
  featureRangeOptions: FeatureRangeOption[];
  windowRangePreset: string;
  windowCustomRange: [Dayjs | null, Dayjs | null] | null;
  running: boolean;
  taskStatus: TaskStatus;
  taskError?: string;
  taskCurrentStage?: string;
  taskStageStatus: TaskStageStatusMap;
  taskStageData: TaskStageDataMap;
  taskStageRunningData: TaskStageDataMap;
  taskWindowSelection: WindowSelectionMeta | null;
  taskThinking: LlmThinkingEvent[];
  onLoopChange: Dispatch<SetStateAction<string | undefined>>;
  onRangePresetChange: Dispatch<SetStateAction<string>>;
  onCustomRangeChange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  onPreviewWindows: () => void;
  onStartReview: () => void;
  onStop: () => void;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  formatRange: (min?: number | null, max?: number | null, digits?: number) => string;
  formatProcessDirection: (value?: string | null) => string;
  formatProcessDirectionBasis: (value?: string | null) => string;
}

function translateWindowAlgorithmFamily(value?: string) {
  if (!value) return '-';
  return WINDOW_ALGORITHM_FAMILY_LABELS[value] ?? value;
}

function translatePolicyState(value?: string) {
  if (value === 'preferred') return '优先';
  if (value === 'available') return '可用';
  if (value === 'deprioritized') return '降权';
  if (value === 'disabled') return '禁用';
  if (value === 'ran') return '已运行';
  if (value === 'skipped') return '跳过';
  if (value === 'blocked') return '已阻断';
  return value || '-';
}

function translatePolicyUsageStatus(value?: WindowPolicyFieldUsage['status']) {
  if (value === 'consumed') return '已消费';
  if (value === 'downstream_hint') return '下游提示';
  if (value === 'display_only') return '仅展示';
  return value || '-';
}

function policyUsageStatusColor(value?: WindowPolicyFieldUsage['status']) {
  if (value === 'consumed') return 'green';
  if (value === 'downstream_hint') return 'blue';
  return 'default';
}

function translatePolicyFieldName(field?: string, label?: string) {
  if (field && POLICY_FIELD_LABELS[field]) return POLICY_FIELD_LABELS[field];
  if (label) return POLICY_FIELD_LABEL_ALIASES[label] ?? label;
  return field || '-';
}

function translateOperatingState(value: string) {
  const map: Record<string, string> = {
    stable: '稳定工况',
    stable_production: '稳定生产',
    mild_load_change: '轻微负荷变化',
    load_change: '负荷变化',
    steady_disturbance: '稳态扰动',
    oscillatory: '振荡工况',
    constraint_limited: '约束受限',
    data_unreliable: '数据不可靠',
    hard_saturation: '严重饱和',
    manual_intervention: '人工干预',
    strong_oscillation: '强振荡',
    startup_shutdown: '开停工',
    excitation: '激励充分性',
    response: 'PV 响应',
    stability: '稳定性',
    ontology_consistency: '本体一致性',
    constraint: '约束状态',
    no_usable_window: '无可用窗口',
    block_formal_identification: '阻断正式辨识',
    outside_typical_time_scale: '超出典型时间尺度',
    decrease_confidence: '降低置信度',
    gain_sign_conflict: '增益方向冲突',
    positive: '正作用',
    negative: '反作用',
    unknown: '不确定',
  };
  return map[value] ?? value;
}

function translatePolicyScalar(value: string) {
  return translateWindowAlgorithmFamily(value) !== value
    ? translateWindowAlgorithmFamily(value)
    : translatePolicyState(value) !== value
      ? translatePolicyState(value)
      : translateOperatingState(value);
}

function formatWindowSource(value?: string) {
  if (!value) return '';
  const match = value.match(/^([a-zA-Z_]+)_(\d+)$/);
  if (!match) return translateWindowAlgorithmFamily(value);
  const [, family, index] = match;
  return `${translateWindowAlgorithmFamily(family)} #${index}`;
}

function formatPolicyObject(value: Record<string, unknown>): string {
  const entries = Object.entries(value).filter(([, item]) => item !== undefined && item !== null && item !== '');
  if (!entries.length) return '-';
  return entries
    .map(([key, item]) => `${translatePolicyFieldName(key)}：${formatPolicyValue(item)}`)
    .join('；');
}

function formatPolicyValue(value: unknown): string {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (Array.isArray(value)) {
    return value.length
      ? value.map((item) => {
        if (typeof item === 'string') return translatePolicyScalar(item);
        if (typeof item === 'number') return Number.isInteger(item) ? String(item) : item.toFixed(3);
        if (typeof item === 'boolean') return item ? '是' : '否';
        if (item && typeof item === 'object') return formatPolicyObject(item as Record<string, unknown>);
        return String(item);
      }).join('、')
      : '-';
  }
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (typeof value === 'string') return translatePolicyScalar(value);
  if (typeof value === 'object') return formatPolicyObject(value as Record<string, unknown>);
  return String(value);
}

function translatePolicyNote(usage: WindowPolicyFieldUsage) {
  return POLICY_NOTE_LABELS[usage.field] || usage.note || '-';
}

function translateSelectionMode(value?: string) {
  const map: Record<string, string> = {
    llm: '大模型评审',
    deterministic: '确定性算法',
    fallback_deterministic: '兜底确定性算法',
    user_override: '人工指定',
    blocked: '未准入',
  };
  return value ? map[value] ?? value : '-';
}

function translateOntologySource(value?: string) {
  const map: Record<string, string> = {
    mcp: 'MCP 本体服务',
    frontend: '前端上下文',
    none: '无本体上下文',
    skipped: '已跳过',
    default: '默认策略',
    llm: '大模型策略',
  };
  return value ? map[value] ?? value : '-';
}

function windowFlowStatusText(status: WindowFlowStepStatus) {
  if (status === 'done') return '完成';
  if (status === 'running') return '运行中';
  return '等待';
}

function windowFlowStatusColor(status: WindowFlowStepStatus) {
  if (status === 'done') return 'green';
  if (status === 'running') return 'processing';
  return 'default';
}

function renderWindowPolicyTables(
  policy: WindowSelectionPolicy | undefined,
  formatPercentValue: (value?: number | null, digits?: number) => string,
  formatProcessDirection: (value?: string | null) => string,
) {
  if (!policy) return <Empty description="等待策略生成结果" />;

  const policyRows = (policy.field_usage ?? []).map((usage) => ({
    key: usage.field,
    field: usage.field,
    label: translatePolicyFieldName(usage.field, usage.label),
    value: formatPolicyValue((policy as Record<string, unknown>)[usage.field]),
    usage,
  }));

  return (
    <Space direction="vertical" style={{ width: '100%' }}>
      <Descriptions bordered column={4} size="small" className="industrial-descriptions">
        <Descriptions.Item label="策略版本">{policy.policy_version ?? '-'}</Descriptions.Item>
        <Descriptions.Item label="策略置信度">{formatPercentValue(policy.confidence, 0)}</Descriptions.Item>
        <Descriptions.Item label="预期增益方向">{formatProcessDirection(policy.expected_gain_sign)}</Descriptions.Item>
        <Descriptions.Item label="策略来源">{translateOntologySource(policy.ontology_facts?.source)}</Descriptions.Item>
        <Descriptions.Item label="策略说明" span={4}>{policy.rationale ?? '-'}</Descriptions.Item>
      </Descriptions>
      <Table
        size="small"
        pagination={false}
        rowKey="key"
        dataSource={policyRows}
        columns={[
          { title: '策略字段', dataIndex: 'label', width: 190 },
          { title: '策略值', dataIndex: 'value', ellipsis: true },
          {
            title: '消费状态',
            dataIndex: 'usage',
            width: 160,
            render: (usage: WindowPolicyFieldUsage) => (
              <Tag color={policyUsageStatusColor(usage.status)}>
                {translatePolicyUsageStatus(usage.status)}
              </Tag>
            ),
          },
          {
            title: '消费算法族 / 说明',
            dataIndex: 'usage',
            render: (usage: WindowPolicyFieldUsage) => {
              const consumers = usage.consumed_by?.map(translateWindowAlgorithmFamily).join('、');
              return consumers || translatePolicyNote(usage);
            },
          },
        ]}
      />
      <Table<WindowAlgorithmPlanItem>
        size="small"
        pagination={false}
        rowKey={(row) => row.family || row.reason || Math.random()}
        dataSource={policy.algorithm_plan ?? []}
        columns={[
          { title: '算法族', dataIndex: 'family', width: 150, render: translateWindowAlgorithmFamily },
          { title: '执行策略', dataIndex: 'state', width: 120, render: (value) => <Tag color={value === 'preferred' ? 'green' : value === 'deprioritized' ? 'orange' : value === 'disabled' ? 'red' : 'blue'}>{translatePolicyState(value)}</Tag> },
          { title: '实际消费字段', dataIndex: 'consumed_policy_field_labels', render: (_value, row) => (row.consumed_policy_fields?.length ? row.consumed_policy_fields.map((field, index) => translatePolicyFieldName(field, row.consumed_policy_field_labels?.[index])).join('、') : '-') },
          { title: '原因', dataIndex: 'reason', ellipsis: true },
        ]}
      />
    </Space>
  );
}

export function WindowCandidatesPanel({
  selectedLoopId,
  selectedLoop,
  scopedLoops,
  loopTypeLabels,
  featureRangeOptions,
  windowRangePreset,
  windowCustomRange,
  running,
  taskStatus,
  taskError,
  taskCurrentStage,
  taskStageStatus,
  taskStageData,
  taskStageRunningData,
  taskWindowSelection,
  taskThinking,
  onLoopChange,
  onRangePresetChange,
  onCustomRangeChange,
  onPreviewWindows,
  onStartReview,
  onStop,
  formatNumber,
  formatPercentValue,
  formatRange,
  formatProcessDirection,
  formatProcessDirectionBasis,
}: WindowCandidatesPanelProps) {
  const ontologyPolicyDone = taskStageData.ontology_policy ?? null;
  const ontologyPolicyData = ontologyPolicyDone as { policy?: WindowSelectionPolicy } | null;
  const ontologyPolicyEarly = ontologyPolicyData?.policy ?? null;
  const windowPolicy = taskWindowSelection?.window_policy ?? ontologyPolicyEarly ?? undefined;
  const windowPolicyResults = taskWindowSelection?.window_policy_results ?? [];
  const familySummaries = taskWindowSelection?.window_algorithm_family_summaries ?? [];
  const windowThinking = taskThinking.filter((item) => item.stage === 'window_selection');
  const windowDataAnalysis = taskStageData.data_analysis ?? {};
  const windowProfileFeatures = (windowDataAnalysis.data_profile as HistoryLoopFeatures | undefined) ?? null;
  const windowProfileDataPoints =
    (windowDataAnalysis.data_points as number | undefined)
    ?? windowProfileFeatures?.data_profile?.row_count;
  const windowReviewStarted = taskStatus !== 'idle'
    || Boolean(taskStageStatus.data_analysis || taskStageStatus.ontology_policy || taskStageStatus.window_selection || taskWindowSelection);
  const dataProfileDone = Boolean(taskStageStatus.data_analysis === 'done' || taskStageData.data_analysis);
  const ontologyPolicyRunningPhase = (taskStageRunningData.ontology_policy?.phase as string | undefined) ?? '';
  const windowSelectionRunningPhase = (taskStageRunningData.window_selection?.phase as string | undefined) ?? '';
  const ontologyPolicyRunning = taskStageStatus.ontology_policy === 'running';
  const ontologyPolicyFinished = Boolean(taskStageStatus.ontology_policy === 'done' || ontologyPolicyDone);
  const windowSelectionDone = Boolean(taskStageStatus.window_selection === 'done' || taskWindowSelection);
  const ontologyStepDone = ontologyPolicyFinished
    || ontologyPolicyRunningPhase === 'building_policy'
    || windowSelectionDone;
  const policyStepRunning = !windowSelectionDone
    && ((ontologyPolicyRunning && ontologyPolicyRunningPhase === 'building_policy')
      || (ontologyPolicyFinished && !windowSelectionDone && !taskStageStatus.window_selection));
  const policyStepDone = Boolean(ontologyPolicyFinished || windowSelectionDone);
  const algorithmStepDone = windowSelectionDone
    || familySummaries.length > 0
    || windowPolicyResults.length > 0;
  const windowSelectionInFlight = taskCurrentStage === 'window_selection'
    || taskStageStatus.window_selection === 'running';
  const algorithmStepRunning = !algorithmStepDone
    && windowSelectionInFlight
    && (!windowSelectionRunningPhase || windowSelectionRunningPhase === 'algorithm');
  const llmStepRunning = !windowSelectionDone
    && windowSelectionInFlight
    && windowSelectionRunningPhase === 'llm';
  const gateStepRunning = !windowSelectionDone
    && windowSelectionInFlight
    && windowSelectionRunningPhase === 'gate';
  const stepStatus: Record<(typeof WINDOW_FLOW_STEPS)[number]['key'], WindowFlowStepStatus> = {
    profile: !windowReviewStarted ? 'waiting' : dataProfileDone ? 'done' : 'running',
    ontology: !dataProfileDone
      ? 'waiting'
      : ontologyStepDone
        ? 'done'
        : ontologyPolicyRunning ? 'running' : 'waiting',
    policy: !ontologyStepDone && !ontologyPolicyRunning
      ? 'waiting'
      : policyStepDone
        ? 'done'
        : policyStepRunning ? 'running' : 'waiting',
    algorithm: !policyStepDone
      ? 'waiting'
      : algorithmStepDone
        ? 'done'
        : algorithmStepRunning ? 'running' : 'waiting',
    llm: windowSelectionDone
      ? 'done'
      : llmStepRunning
        ? 'running'
        : 'waiting',
    gate: windowSelectionDone ? 'done' : gateStepRunning ? 'running' : 'waiting',
  };
  const allDone = WINDOW_FLOW_STEPS.every((item) => stepStatus[item.key] === 'done');
  const phaseText = !windowReviewStarted
    ? '等待开始'
    : (WINDOW_FLOW_STEPS.find((item) => stepStatus[item.key] === 'running')?.title
      ?? (allDone ? '流程已完成' : '等待后端事件…'));

  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar window-review-launch">
          <div>
            <div className="panel-title">窗口候选</div>
            <Typography.Text type="secondary">
              先选择需要评审的回路；点击开始后，系统按“数据画像 → 本体检索 → 策略生成 → 算法族运行 → 大模型评审 → 准入结论”的顺序执行。
            </Typography.Text>
          </div>
          <Space wrap>
            <Select
              showSearch
              size="small"
              style={{ minWidth: 360 }}
              placeholder="选择回路"
              value={selectedLoopId}
              onChange={onLoopChange}
              optionFilterProp="label"
              options={scopedLoops.map((loop) => ({
                value: loop.loop_id,
                label: `${loop.loop_id} · ${loopTypeLabels[loop.loop_type] ?? loop.loop_type}`,
              }))}
            />
            <Select
              size="small"
              style={{ width: 140 }}
              value={windowRangePreset}
              onChange={onRangePresetChange}
              options={featureRangeOptions.map((item) => ({ label: item.label, value: item.value }))}
            />
            {windowRangePreset === 'custom' && (
              <DatePicker.RangePicker
                size="small"
                showTime
                value={windowCustomRange}
                onChange={onCustomRangeChange}
              />
            )}
            <Button
              size="small"
              icon={<SyncOutlined />}
              disabled={!selectedLoop}
              onClick={onPreviewWindows}
            >
              预览该区间窗口
            </Button>
            <Button
              type="primary"
              icon={<AuditOutlined />}
              loading={running}
              disabled={!selectedLoop}
              onClick={onStartReview}
            >
              开始本体驱动窗口评审
            </Button>
            {running && <Button danger onClick={onStop}>停止</Button>}
          </Space>
        </div>
        {windowReviewStarted && (
          <Alert
            className="agent-alert"
            type={taskStatus === 'error' ? 'error' : taskStatus === 'done' ? 'success' : 'info'}
            showIcon
            message={`窗口评审${taskStatus === 'done' ? '已完成' : taskStatus === 'error' ? '异常' : '运行中'}：当前阶段：${phaseText}`}
            description={taskError || '页面按后端事件实时更新；当前后端将策略、本体和算法族结果汇总后返回。'}
          />
        )}
      </section>

      {windowReviewStarted && (
        <section className="agent-panel">
          <div className="panel-title">窗口候选全流程</div>
          <div className="window-flow-grid">
            {WINDOW_FLOW_STEPS.map((step) => {
              const status = stepStatus[step.key];
              return (
                <div key={step.key} className={`window-flow-card is-${status}`}>
                  <div className="window-flow-card-head">
                    <strong>{step.title}</strong>
                    <Tag color={windowFlowStatusColor(status)}>{windowFlowStatusText(status)}</Tag>
                  </div>
                  <p>{step.desc}</p>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {windowReviewStarted && (
        <>
          <section className="agent-panel">
            <div className="panel-toolbar">
              <div>
                <div className="panel-title">1 数据画像：原始指标</div>
                <Typography.Text type="secondary">展示基础画像和原始统计。</Typography.Text>
              </div>
              <Tag color={windowFlowStatusColor(stepStatus.profile)}>{windowFlowStatusText(stepStatus.profile)}</Tag>
            </div>
            {windowProfileFeatures ? (
              <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                <Descriptions.Item label="回路位号">{selectedLoop?.loop_id ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="回路类型">{loopTypeLabels[selectedLoop?.loop_type ?? ''] ?? selectedLoop?.loop_type ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="数据点">{windowProfileDataPoints ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="采样周期">{formatNumber(windowProfileFeatures.data_profile?.sample_time_median_s, 1)}s</Descriptions.Item>
                <Descriptions.Item label="PV范围">{formatRange(windowProfileFeatures.pv_stats?.min, windowProfileFeatures.pv_stats?.max, 3)}</Descriptions.Item>
                <Descriptions.Item label="MV范围">{formatRange(windowProfileFeatures.mv_stats?.min, windowProfileFeatures.mv_stats?.max, 3)}</Descriptions.Item>
                <Descriptions.Item label="MV活跃比例">{formatPercentValue(windowProfileFeatures.mv_stats?.active_step_ratio, 2)}</Descriptions.Item>
                <Descriptions.Item label="MV反向频次">{formatNumber(windowProfileFeatures.mv_stats?.direction_reversal_per_hour, 2)}/h</Descriptions.Item>
                <Descriptions.Item label="过程作用方向">
                  {formatProcessDirection(String(windowProfileFeatures.pv_mv_relation_raw?.process_direction ?? windowProfileFeatures.pv_mv_relation_raw?.estimated_direction_raw ?? ''))}
                </Descriptions.Item>
                <Descriptions.Item label="方向置信度">
                  {formatPercentValue(typeof windowProfileFeatures.pv_mv_relation_raw?.process_direction_confidence === 'number'
                    ? windowProfileFeatures.pv_mv_relation_raw.process_direction_confidence
                    : undefined, 1)}
                </Descriptions.Item>
                <Descriptions.Item label="方向证据">
                  {formatProcessDirectionBasis(String(windowProfileFeatures.pv_mv_relation_raw?.process_direction_basis ?? ''))}
                </Descriptions.Item>
                <Descriptions.Item label="MV饱和比例">{formatPercentValue(windowProfileFeatures.constraint_raw?.mv_saturation_ratio, 2)}</Descriptions.Item>
              </Descriptions>
            ) : (
              <Empty description={stepStatus.profile === 'running' ? '正在读取回路画像...' : '暂无回路画像'} />
            )}
          </section>

          <section className="agent-panel">
            <div className="panel-toolbar">
              <div>
                <div className="panel-title">2 本体查询与上下文</div>
                <Typography.Text type="secondary">展示后端向本体提出的问题、来源和返回内容。</Typography.Text>
              </div>
              <Tag color={windowFlowStatusColor(stepStatus.ontology)}>{windowFlowStatusText(stepStatus.ontology)}</Tag>
            </div>
            {taskWindowSelection ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                  <Descriptions.Item label="本体来源">{translateOntologySource(taskWindowSelection.ontology_context_source)}</Descriptions.Item>
                  <Descriptions.Item label="MCP 服务">{taskWindowSelection.ontology_mcp_server ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="MCP 工具">{taskWindowSelection.ontology_mcp_tool ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="返回字数">{taskWindowSelection.ontology_mcp_content_chars ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="查询问题" span={4}>{taskWindowSelection.ontology_mcp_query ?? '-'}</Descriptions.Item>
                </Descriptions>
                <Collapse
                  items={[{
                    key: 'ontology-raw',
                    label: '本体返回原文',
                    children: (
                      <Typography.Paragraph className="thinking-text">
                        {taskWindowSelection.ontology_mcp_content_raw || taskWindowSelection.ontology_mcp_content_preview || taskWindowSelection.ontology_mcp_error || '暂无本体返回内容'}
                      </Typography.Paragraph>
                    ),
                  }]}
                />
              </Space>
            ) : (
              <Empty description={stepStatus.ontology === 'running' ? '正在查询本体上下文...' : '等待本体检索'} />
            )}
          </section>

          <section className="agent-panel">
            <div className="panel-toolbar">
              <div>
                <div className="panel-title">3 策略生成</div>
                <Typography.Text type="secondary">展示策略字段及其使用方式。</Typography.Text>
              </div>
              <Tag color={windowFlowStatusColor(stepStatus.policy)}>{windowFlowStatusText(stepStatus.policy)}</Tag>
            </div>
            {renderWindowPolicyTables(windowPolicy, formatPercentValue, formatProcessDirection)}
          </section>

          <section className="agent-panel">
            <div className="panel-toolbar">
              <div>
                <div className="panel-title">4 算法族输出的候选窗口</div>
                <Typography.Text type="secondary">展示每个算法族执行状态、策略状态和窗口评分修正。</Typography.Text>
              </div>
              <Tag color={windowFlowStatusColor(stepStatus.algorithm)}>{windowFlowStatusText(stepStatus.algorithm)}</Tag>
            </div>
            {familySummaries.length || windowPolicyResults.length ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <Table<WindowAlgorithmFamilySummary>
                  size="small"
                  pagination={false}
                  rowKey={(row) => row.family || row.provider || Math.random()}
                  dataSource={familySummaries}
                  columns={[
                    { title: '算法族', dataIndex: 'family', render: translateWindowAlgorithmFamily },
                    { title: '执行状态', dataIndex: 'run_state', render: (value) => <Tag color={value === 'ran' ? 'green' : 'default'}>{translatePolicyState(value)}</Tag> },
                    { title: '策略状态', dataIndex: 'policy_state', render: (value) => <Tag color={value === 'preferred' ? 'green' : value === 'deprioritized' ? 'orange' : value === 'disabled' ? 'red' : 'blue'}>{translatePolicyState(value)}</Tag> },
                    { title: '候选/可用', render: (_value, row) => `${row.usable_count ?? 0}/${row.window_count ?? 0}` },
                    { title: '最佳分', dataIndex: 'best_score', render: (value) => formatNumber(value, 3) },
                    { title: '策略说明', dataIndex: 'policy_reason', ellipsis: true },
                  ]}
                />
                <Table<WindowPolicyResult>
                  size="small"
                  pagination={{ pageSize: 8 }}
                  rowKey={(row) => `${row.index}-${row.window_source}`}
                  dataSource={windowPolicyResults}
                  columns={[
                    { title: '窗口', render: (_value, row) => `#${row.index} ${formatWindowSource(row.window_source)}` },
                    { title: '算法族', dataIndex: 'algorithm_family', render: translateWindowAlgorithmFamily },
                    { title: '原始分', dataIndex: 'original_score', render: (value) => formatNumber(value, 3) },
                    { title: '策略分', dataIndex: 'policy_score', render: (value) => formatNumber(value, 3) },
                    { title: '本体一致性', dataIndex: 'ontology_consistency_score', render: (value) => formatNumber(value, 3) },
                    { title: '可用性', render: (_value, row) => <Tag color={row.usable_after_policy ? 'green' : 'red'}>{row.usable_after_policy ? '可用' : '过滤'}</Tag> },
                  ]}
                />
              </Space>
            ) : (
              <Empty description={stepStatus.algorithm === 'running' ? '算法族正在根据策略产出候选窗口...' : '等待算法族运行'} />
            )}
          </section>

          <section className="agent-panel">
            <div className="panel-toolbar">
              <div>
                <div className="panel-title">5 大模型评审</div>
                <Typography.Text type="secondary">大模型结合回路画像、本体证据、策略和候选窗口逐项判断，给出最终窗口建议。</Typography.Text>
              </div>
              <Tag color={windowFlowStatusColor(stepStatus.llm)}>{windowFlowStatusText(stepStatus.llm)}</Tag>
            </div>
            {taskWindowSelection ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                  <Descriptions.Item label="选择模式">{translateSelectionMode(taskWindowSelection.mode)}</Descriptions.Item>
                  <Descriptions.Item label="模型选中窗口">#{taskWindowSelection.chosen_index}</Descriptions.Item>
                  <Descriptions.Item label="算法确定性窗口">#{taskWindowSelection.deterministic_index}</Descriptions.Item>
                  <Descriptions.Item label="是否一致">{taskWindowSelection.agreed_with_deterministic === undefined ? '-' : taskWindowSelection.agreed_with_deterministic ? '一致' : '存在分歧'}</Descriptions.Item>
                  <Descriptions.Item label="评审说明" span={4}>{taskWindowSelection.reasoning || '-'}</Descriptions.Item>
                </Descriptions>
                {!!taskWindowSelection.ontology_evidence?.length && (
                  <Table
                    size="small"
                    pagination={false}
                    rowKey={(row, index) => `${row.fact}-${index}`}
                    dataSource={taskWindowSelection.ontology_evidence}
                    columns={[
                      { title: '模型引用的本体证据', dataIndex: 'fact' },
                      { title: '来源', dataIndex: 'source', width: 260 },
                    ]}
                  />
                )}
                {!!taskWindowSelection.window_judgements?.length && (
                  <Table
                    size="small"
                    pagination={false}
                    rowKey={(row) => `${row.index}-${row.verdict}`}
                    dataSource={taskWindowSelection.window_judgements}
                    columns={[
                      { title: '窗口', render: (_value, row) => `#${row.index} ${formatWindowSource(row.window_source)}` },
                      { title: '判断', dataIndex: 'verdict', render: (value) => <Tag color={value === 'preferred' ? 'green' : value === 'risk' ? 'orange' : 'blue'}>{value === 'preferred' ? '优先' : value === 'risk' ? '风险' : '可接受'}</Tag> },
                      { title: '质量分', dataIndex: 'window_quality_score', render: (value) => formatNumber(value, 3) },
                      { title: '判断依据', dataIndex: 'reason' },
                    ]}
                  />
                )}
                {!!windowThinking.length && (
                  <Collapse
                    items={windowThinking.map((item, index) => ({
                      key: `window-thinking-${index}`,
                      label: `模型分析摘要 · ${item.model} · ${(item.reasoning_content || item.raw_text || '').length} 字`,
                      children: <Typography.Paragraph className="thinking-text">{item.reasoning_content || item.raw_text}</Typography.Paragraph>,
                    }))}
                  />
                )}
              </Space>
            ) : (
              <Empty description={stepStatus.llm === 'running' ? '等待大模型评审窗口候选...' : '等待大模型评审'} />
            )}
          </section>

          <section className="agent-panel">
            <div className="panel-toolbar">
              <div>
                <div className="panel-title">6 准入结论</div>
                <Typography.Text type="secondary">没有适合正式辨识的窗口时，建议转为诊断性辨识或停止。</Typography.Text>
              </div>
              <Tag color={windowFlowStatusColor(stepStatus.gate)}>{windowFlowStatusText(stepStatus.gate)}</Tag>
            </div>
            {taskWindowSelection ? (
              <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                <Descriptions.Item label="正式辨识">{taskWindowSelection.formal_identification_allowed ? <Tag color="green">允许</Tag> : <Tag color="red">不建议</Tag>}</Descriptions.Item>
                <Descriptions.Item label="诊断辨识">{taskWindowSelection.diagnostic_identification_allowed ? <Tag color="green">允许</Tag> : <Tag color="orange">不建议</Tag>}</Descriptions.Item>
                <Descriptions.Item label="最终窗口">#{taskWindowSelection.chosen_index}</Descriptions.Item>
                <Descriptions.Item label="确定性分">{formatNumber(taskWindowSelection.deterministic_score, 3)}</Descriptions.Item>
                <Descriptions.Item label="准入原因" span={4}>{taskWindowSelection.window_candidate_decision?.primary_reason || taskWindowSelection.stop_reason || '存在可用于正式辨识的候选窗口。'}</Descriptions.Item>
              </Descriptions>
            ) : (
              <Empty description={stepStatus.gate === 'running' ? '正在生成准入结论...' : '等待准入结论'} />
            )}
          </section>
        </>
      )}
    </div>
  );
}
