import { Component, type ErrorInfo, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Line } from '@ant-design/charts';
import dayjs, { type Dayjs } from 'dayjs';
import {
  Alert,
  Button,
  Collapse,
  DatePicker,
  Divider,
  Descriptions,
  Dropdown,
  Empty,
  Form,
  Input,
  Modal,
  Progress,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
  Tree,
  message,
} from 'antd';
import type { UploadFile } from 'antd';
import type { DataNode } from 'antd/es/tree';
import {
  ApiOutlined,
  AppstoreOutlined,
  AuditOutlined,
  BellOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  DeploymentUnitOutlined,
  DownOutlined,
  EditOutlined,
  EllipsisOutlined,
  ExperimentOutlined,
  FileSearchOutlined,
  FundProjectionScreenOutlined,
  LineChartOutlined,
  MenuOutlined,
  RadarChartOutlined,
  PushpinOutlined,
  RobotOutlined,
  RocketOutlined,
  SendOutlined,
  SettingOutlined,
  SyncOutlined,
  ToolOutlined,
  UserOutlined,
  WarningOutlined,
  RightOutlined,
} from '@ant-design/icons';
import {
  fetchHistoryLoopFeatures,
  fetchHistoryLoopMonitoring,
  assistantSessionStream,
  createAssistantSession,
  deleteAssistantSession,
  getAssistantSession,
  getHistoryLoopAssessment,
  getHistoryLoopTuningPriorCore,
  getHistoryLoopTuningPriorOntology,
  getHistoryLoopSeries,
  getHistoryLoopWindows,
  importHistoryFiles,
  listHistoryLoops,
  listAssistantSessions,
  tuneHistoryLoopStream,
  fetchModelConfig,
  fetchPolicyConfig,
  fetchPromptConfig,
  getSession,
  listSessions,
  resetPromptConfig,
  reviewHistoryLoopTuningPrior,
  testModelConfig,
  updateAssistantSession,
  updateModelConfig,
  updatePromptConfig,
} from '@/services/api';
import McpConfigPage from '@/pages/settings/McpConfigPage';
import { DashboardAbnormalLoopsWidget } from '@/features/dashboard/DashboardAbnormalLoopsWidget';
import { DashboardAlertStatsWidget } from '@/features/dashboard/DashboardAlertStatsWidget';
import { DashboardBarsWidget, type DashboardBarRow } from '@/features/dashboard/DashboardBarsWidget';
import { DashboardConfigModal } from '@/features/dashboard/DashboardConfigModal';
import { DashboardDonutWidget } from '@/features/dashboard/DashboardDonutWidget';
import { DashboardHeader } from '@/features/dashboard/DashboardHeader';
import { DashboardKpiWidget } from '@/features/dashboard/DashboardKpiWidget';
import { DashboardQuickActionsWidget } from '@/features/dashboard/DashboardQuickActionsWidget';
import { DashboardSnapshotWidget } from '@/features/dashboard/DashboardSnapshotWidget';
import { DashboardTopLoopsWidget } from '@/features/dashboard/DashboardTopLoopsWidget';
import { DashboardTrendWidget } from '@/features/dashboard/DashboardTrendWidget';
import { DashboardWidgetGrid } from '@/features/dashboard/DashboardWidgetGrid';
import {
  DASHBOARD_WIDGET_STORAGE_KEY,
  DEFAULT_DASHBOARD_WIDGET_KEYS,
  buildDashboardRows,
  countByLabel,
  countDashboardAlertSeverities,
  getAbnormalDashboardRows,
  getRealDashboardRows,
  getTopHealthyDashboardRows,
  makeDashboardSlices,
  normalizeDashboardWidgetKeys,
  summarizeDashboardRows,
  type DashboardWidgetDefinition,
  type DashboardWidgetKey,
} from '@/features/dashboard/model';
import { ActuatorStatusPanel } from '@/features/loop-monitoring/ActuatorStatusPanel';
import { AlarmEventsPanel } from '@/features/loop-monitoring/AlarmEventsPanel';
import { ConstraintMonitorPanel } from '@/features/loop-monitoring/ConstraintMonitorPanel';
import {
  DiagnosisOverviewPanel,
  DiagnosisPlanPanel,
  OscillationDiagnosisPanel,
  type DiagnosisPlanKey,
} from '@/features/loop-monitoring/DiagnosisPanels';
import { LoopBoardPanel } from '@/features/loop-monitoring/LoopBoardPanel';
import { LoopProfilePanel } from '@/features/loop-monitoring/LoopProfilePanel';
import { OperatingConditionPanel } from '@/features/loop-monitoring/OperatingConditionPanel';
import { PerformanceScorePanel } from '@/features/loop-monitoring/PerformanceScorePanel';
import { SpectrumSummaryPanel } from '@/features/loop-monitoring/SpectrumSummaryPanel';
import { TrendChartPanel } from '@/features/loop-monitoring/TrendChartPanel';
import { TrendQueryDetails } from '@/features/loop-monitoring/TrendQueryDetails';
import { TuningReadinessPanel } from '@/features/loop-monitoring/TuningReadinessPanel';
import { ModelReliabilityPanel } from '@/features/model-reliability/ModelReliabilityPanel';
import type {
  HistoryLoop,
  AssistantSession,
  AssistantSessionSummary,
  HistoryLoopAssessment,
  HistoryLoopFeatures,
  HistoryLoopMonitoring,
  HistoryLoopTuningPrior,
  HistoryTimeRangeParams,
  HistoryWindow,
  LoopSeriesResp,
  ModelConfig,
  PolicyConfig,
  PromptConfig,
} from '@/services/api';
import type {
  IdentificationAttempt,
  IdentificationRefinementMeta,
  LlmThinkingEvent,
  ModelReviewMeta,
  PipelineEvent,
  TuningResult,
  WindowAlgorithmFamilySummary,
  WindowAlgorithmPlanItem,
  WindowAlgorithmFitSummary,
  WindowPolicyFieldUsage,
  WindowSelectionPolicy,
  WindowSelectionMeta,
} from '@/types/tuning';
import {
  attemptFitKey,
  clearRunningStageData,
  mergeDoneStageData,
  mergeIdentificationAttempts,
  mergeRunningStageData,
  prependTaskEventLog,
  type TaskEventLog,
  type TaskStageDataMap,
  type TaskStageStatusMap,
  type TaskStatus,
  upsertRefinement,
  upsertThinkingEvent,
} from '@/features/tuning-task/model';
import { TuningTaskDashboard } from '@/features/tuning-task/TuningTaskDashboard';
import { TuningTaskDetailDrawer } from '@/features/tuning-task/TuningTaskDetailDrawer';
import { TuningTaskPanel } from '@/features/tuning-task/TuningTaskPanel';
import { DataSourcesPanel } from '@/features/settings/DataSourcesPanel';
import { ModelConfigPanel } from '@/features/settings/ModelConfigPanel';
import { PromptConfigPanel } from '@/features/settings/PromptConfigPanel';
import { RuleConfigPanel } from '@/features/settings/RuleConfigPanel';
import { TuningPriorPanel } from '@/features/tuning-prior/TuningPriorPanel';
import './LoopMonitoringPage.css';

// ─── Error Boundary：兜住子树渲染异常，避免整个页面白屏 ──────────────────────────
//
// 没有 ErrorBoundary 时，任何渲染抛错都会导致 React 卸载整棵树，用户只能看到空白。
// 接到错误后这里把堆栈展示出来，并保留"重置"按钮让用户继续后续操作。
interface SectionBoundaryProps {
  /** 出现错误时显示的子树名称（中文，给用户看） */
  label: string;
  children: ReactNode;
}
interface SectionBoundaryState {
  error: Error | null;
  errorInfo: ErrorInfo | null;
}
class SectionErrorBoundary extends Component<SectionBoundaryProps, SectionBoundaryState> {
  state: SectionBoundaryState = { error: null, errorInfo: null };

  static getDerivedStateFromError(error: Error): SectionBoundaryState {
    return { error, errorInfo: null };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // eslint-disable-next-line no-console
    console.error(`[SectionErrorBoundary:${this.props.label}]`, error, errorInfo);
    this.setState({ error, errorInfo });
  }

  reset = () => this.setState({ error: null, errorInfo: null });

  render() {
    const { error, errorInfo } = this.state;
    if (error) {
      return (
        <Alert
          type="error"
          showIcon
          message={`${this.props.label} 渲染异常`}
          description={(
            <div style={{ maxHeight: 320, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
              <div style={{ fontWeight: 600 }}>{error.name}: {error.message}</div>
              {error.stack && <pre style={{ fontSize: 12, marginTop: 8 }}>{error.stack}</pre>}
              {errorInfo?.componentStack && (
                <pre style={{ fontSize: 12, marginTop: 8, color: '#888' }}>{errorInfo.componentStack}</pre>
              )}
              <div style={{ marginTop: 8 }}>
                <Button size="small" onClick={this.reset}>清除错误，重新渲染</Button>
              </div>
            </div>
          )}
        />
      );
    }
    return this.props.children as ReactNode;
  }
}

type ModuleKey = 'workspace' | 'monitor' | 'assessment' | 'diagnostics' | 'tuning' | 'experience' | 'settings';
type SubKey =
  | 'dashboard' | 'todo' | 'shift_tasks' | 'risk_alerts'
  | 'loop_board' | 'loop_profile' | 'trend_spectrum' | 'oscillation_diagnosis' | 'constraint_monitor' | 'alarm_events'
  | 'performance_score' | 'condition_recognition' | 'actuator_status' | 'tuning_readiness'
  | 'diagnosis_overview' | 'pid_diagnosis' | 'valve_diagnosis' | 'measurement_noise_diagnosis' | 'process_disturbance_diagnosis' | 'model_reliability'
  | 'tuning_task' | 'tuning_prior' | 'id_windows' | 'pid_candidates' | 'release_confirm'
  | 'case_library' | 'rule_library' | 'knowledge_graph' | 'model_versions'
  | 'data_sources' | 'asset_directory' | 'rule_config' | 'model_config' | 'prompt_config' | 'mcp_config';

const DIALOGUE_STARTER_PROMPTS = [
  {
    title: '分析当前回路是否适合整定',
    description: '结合监控评分、运行工况、约束和趋势给出判断。',
    prompt: '请分析当前回路是否适合进入整定，并说明主要依据和建议动作。',
  },
  {
    title: '解释波动或告警原因',
    description: '从 PV/MV 行为、工况变化、饱和和振荡风险入手。',
    prompt: '请分析当前回路最近波动或告警的可能原因，并列出需要优先确认的数据证据。',
  },
  {
    title: '生成整定前检查清单',
    description: '检查数据质量、阀位约束、激励充分性和窗口选择。',
    prompt: '请基于当前回路生成一份整定前检查清单，并指出哪些项需要人工确认。',
  },
  {
    title: '推荐下一步操作',
    description: '给出进入经典模式页面的建议路径。',
    prompt: '请根据当前回路状态推荐下一步操作，如果需要进入经典模式页面，请给出可执行的建议动作。',
  },
];

const LOOP_TYPE_LABEL: Record<string, string> = {
  flow: '流量',
  temperature: '温度',
  pressure: '压力',
  level: '液位',
  unknown: '未知',
};

const MODULES: Array<{
  key: ModuleKey;
  label: string;
  icon: React.ReactNode;
  subs: Array<{ key: SubKey; label: string; icon: React.ReactNode; implemented?: boolean }>;
}> = [
  {
    key: 'workspace',
    label: '综合看板',
    icon: <AppstoreOutlined />,
    subs: [
      { key: 'dashboard', label: '总览驾驶舱', icon: <FundProjectionScreenOutlined />, implemented: true },
      { key: 'risk_alerts', label: '风险预警', icon: <WarningOutlined />, implemented: true },
    ],
  },
  {
    key: 'monitor',
    label: '回路监控',
    icon: <RadarChartOutlined />,
    subs: [
      { key: 'loop_board', label: '全局回路看板', icon: <DatabaseOutlined />, implemented: true },
      { key: 'trend_spectrum', label: '趋势与频谱', icon: <LineChartOutlined />, implemented: true },
      { key: 'loop_profile', label: '单回路画像', icon: <FileSearchOutlined />, implemented: true },
    ],
  },
  {
    key: 'assessment',
    label: '回路评估',
    icon: <AuditOutlined />,
    subs: [
      { key: 'performance_score', label: '控制性能', icon: <FundProjectionScreenOutlined />, implemented: true },
      { key: 'actuator_status', label: '执行机构状态', icon: <ToolOutlined />, implemented: true },
    ],
  },
  {
    key: 'diagnostics',
    label: '根因诊断',
    icon: <DeploymentUnitOutlined />,
    subs: [
      { key: 'diagnosis_overview', label: '诊断总览', icon: <FileSearchOutlined />, implemented: true },
      { key: 'model_reliability', label: '模型可靠性', icon: <ExperimentOutlined />, implemented: true },
    ],
  },
  {
    key: 'tuning',
    label: '整定中心',
    icon: <RocketOutlined />,
    subs: [
      { key: 'tuning_prior', label: '整定先验', icon: <AuditOutlined />, implemented: true },
      { key: 'id_windows', label: '窗口候选', icon: <AuditOutlined />, implemented: true },
      { key: 'tuning_task', label: '整定任务', icon: <RocketOutlined />, implemented: true },
    ],
  },
  {
    key: 'settings',
    label: '系统设置',
    icon: <SettingOutlined />,
    subs: [
      { key: 'data_sources', label: '数据源配置', icon: <ApiOutlined />, implemented: true },
      { key: 'asset_directory', label: '装置资产目录', icon: <DeploymentUnitOutlined />, implemented: true },
      { key: 'rule_config', label: '规则配置', icon: <FileSearchOutlined />, implemented: true },
      { key: 'model_config', label: '模型配置', icon: <RobotOutlined />, implemented: true },
      { key: 'prompt_config', label: '提示词管理', icon: <FileSearchOutlined />, implemented: true },
      { key: 'mcp_config', label: '上下文服务配置', icon: <ApiOutlined />, implemented: true },
    ],
  },
];

const INITIAL_EXPANDED_MODULES: Record<ModuleKey, boolean> = {
  workspace: true,
  monitor: true,
  assessment: true,
  diagnostics: true,
  tuning: true,
  experience: false,
  settings: true,
};

type TrendPreset = 'all' | '1h' | '6h' | '24h' | '7d' | 'custom';
type TrendPointLimit = '6000' | '20000' | 'all';
type FeatureRangePreset = 'all' | '8h' | '1d' | '3d' | '7d' | 'custom';

const TREND_PRESET_OPTIONS: Array<{ label: string; value: TrendPreset; seconds?: number }> = [
  { label: '全部数据', value: 'all' },
  { label: '最近 1 小时', value: '1h', seconds: 3600 },
  { label: '最近 6 小时', value: '6h', seconds: 6 * 3600 },
  { label: '最近 24 小时', value: '24h', seconds: 24 * 3600 },
  { label: '最近 7 天', value: '7d', seconds: 7 * 24 * 3600 },
  { label: '自定义', value: 'custom' },
];

const TREND_POINT_LIMIT_OPTIONS: Array<{ label: string; value: TrendPointLimit }> = [
  { label: '快速抽样 6000 点', value: '6000' },
  { label: '高精度 20000 点', value: '20000' },
  { label: '全量点', value: 'all' },
];

type WindowFlowStepStatus = 'waiting' | 'running' | 'done';

const FEATURE_RANGE_OPTIONS: Array<{ label: string; value: FeatureRangePreset; seconds?: number }> = [
  { label: '全部历史', value: 'all' },
  { label: '最近 8 小时', value: '8h', seconds: 8 * 3600 },
  { label: '最近 1 天', value: '1d', seconds: 24 * 3600 },
  { label: '最近 3 天', value: '3d', seconds: 3 * 24 * 3600 },
  { label: '最近 7 天', value: '7d', seconds: 7 * 24 * 3600 },
  { label: '自定义', value: 'custom' },
];

const WINDOW_FLOW_STEPS = [
  { key: 'profile', title: '1 数据画像', desc: '读取回路原始特征' },
  { key: 'ontology', title: '2 本体检索', desc: '查询本体与回路上下文' },
  { key: 'policy', title: '3 策略生成', desc: '生成窗口算法族策略 JSON' },
  { key: 'algorithm', title: '4 算法族运行', desc: '按策略驱动算法模块产出候选窗口' },
  { key: 'llm', title: '5 大模型评审', desc: '结合画像、本体和候选窗口做解释性判断' },
  { key: 'gate', title: '6 准入结论', desc: '判断是否允许进入正式辨识' },
] as const;

const ASSESSMENT_DETAIL_SUBS = new Set<SubKey>([
  'tuning_task',
  'tuning_readiness',
  'performance_score',
  'condition_recognition',
]);

// 候选窗口只在用户点击“预览该区间窗口”或启动整定/窗口评审后按需计算。
// 进入页面或切换回路时不预加载，避免给出“窗口已经选好”的错觉。
const WINDOW_DETAIL_SUBS = new Set<SubKey>([]);

const FEATURE_DETAIL_SUBS = new Set<SubKey>([
  'loop_profile',
  'trend_spectrum',
  'performance_score',
  'condition_recognition',
  'actuator_status',
  'tuning_readiness',
  'diagnosis_overview',
  'pid_diagnosis',
  'valve_diagnosis',
  'measurement_noise_diagnosis',
  'process_disturbance_diagnosis',
  'model_reliability',
]);

const MONITORING_DETAIL_SUBS = new Set<SubKey>([
  'alarm_events',
  'tuning_readiness',
  'condition_recognition',
  'diagnosis_overview',
]);

interface AssistantAction {
  label: string;
  target: ModuleKey;
  sub: SubKey;
  loopId?: string;
}

interface AssistantEventItem {
  id: string;
  type: string;
  title: string;
  detail?: string;
}

interface AssistantMessage {
  id: number;
  role: 'user' | 'assistant';
  text: string;
  reasoning?: string;
  loading?: boolean;
  error?: string;
  actions?: AssistantAction[];
  eventLog?: AssistantEventItem[];
}

type PromptConfigField = Exclude<keyof PromptConfig, 'updated_at'>;

const PROMPT_CONFIG_ITEMS: Array<{
  key: PromptConfigField;
  label: string;
  group: string;
  help: string;
  placeholder: string;
  minRows: number;
  maxRows: number;
}> = [
  {
    key: 'assistant_system_prompt',
    label: '智能助手系统提示词',
    group: '智能助手',
    help: '定义智能助手身份、任务范围、回答风格和证据引用要求。',
    placeholder: '定义智能助手身份、任务范围、回答风格和证据引用要求',
    minRows: 10,
    maxRows: 20,
  },
  {
    key: 'assistant_developer_prompt',
    label: '智能助手安全与流程约束提示词',
    group: '智能助手',
    help: '定义禁止直接整定、禁止修改参数、高成本流程需确认等安全边界。',
    placeholder: '定义禁止直接整定、禁止修改参数、高成本流程需确认等边界',
    minRows: 8,
    maxRows: 16,
  },
  {
    key: 'assistant_response_schema',
    label: '智能助手响应格式说明',
    group: '智能助手',
    help: '定义答案、证据、风险级别和建议动作等结构化字段。',
    placeholder: '定义答案、证据、风险级别和建议动作等结构化字段',
    minRows: 8,
    maxRows: 16,
  },
  {
    key: 'window_policy_system_prompt',
    label: '窗口候选策略提示词',
    group: '窗口候选',
    help: '用于整定中心的窗口候选策略生成，指导模型根据画像、本体上下文输出窗口算法策略。',
    placeholder: '定义窗口策略生成器身份、算法族约束、输出字段和安全边界',
    minRows: 12,
    maxRows: 24,
  },
  {
    key: 'window_policy_user_prompt_template',
    label: '窗口候选用户提示词模板',
    group: '窗口候选',
    help: '运行时会替换 $base_policy_json、$profile_text、$pv_json、$mv_json、$raw_profile_json、$mcp_content、$frontend_text。',
    placeholder: '定义选窗模型接收实时画像和本体上下文的用户提示词模板',
    minRows: 10,
    maxRows: 20,
  },
  {
    key: 'identification_review_system_prompt',
    label: '辨识 / 模型评审提示词',
    group: '辨识评审',
    help: '用于辨识结束后的模型可信度评审，约束模型输出结论、理由和风险点。',
    placeholder: '定义模型评审专家身份、关键判据和结构化输出要求',
    minRows: 12,
    maxRows: 24,
  },
  {
    key: 'identification_review_user_prompt_template',
    label: '辨识 / 模型评审用户提示词模板',
    group: '辨识评审',
    help: '运行时会替换回路类型、数据画像、窗口来源、模型类型和辨识记录等变量。',
    placeholder: '定义模型评审接收辨识结果、窗口和尝试记录的用户提示词模板',
    minRows: 10,
    maxRows: 20,
  },
  {
    key: 'consultant_system_prompt',
    label: '整定顾问提示词',
    group: '整定顾问',
    help: '用于顾问式对话和工具调用流程，约束模型如何解释整定、调用工具和回答用户。',
    placeholder: '定义整定顾问角色、工具调用边界、回答风格和不编造参数等要求',
    minRows: 10,
    maxRows: 20,
  },
];

type AssetNodeType = 'factory' | 'department' | 'unit' | 'area' | 'equipment' | 'loop';

interface AssetNode {
  id: string;
  parentId?: string;
  name: string;
  type: AssetNodeType;
  code?: string;
  description?: string;
}

const ASSET_TYPE_LABEL: Record<AssetNodeType, string> = {
  factory: '厂区',
  department: '运行部',
  unit: '装置',
  area: '区域/系统',
  equipment: '设备',
  loop: '回路',
};

const DEFAULT_ASSET_NODES: AssetNode[] = [
  { id: 'factory', name: '石化工厂', type: 'factory', description: 'PID 智能整定资产根目录' },
  { id: 'dept_run_1', parentId: 'factory', name: '运行一部', type: 'department' },
  { id: 'unit_rongtuo', parentId: 'dept_run_1', name: '溶脱', type: 'unit' },
  { id: 'unit_ii_chang', parentId: 'dept_run_1', name: 'II常', type: 'unit' },
  { id: 'dept_run_2', parentId: 'factory', name: '运行二部', type: 'department' },
  { id: 'unit_1_jubingxi', parentId: 'dept_run_2', name: '1#聚丙烯', type: 'unit' },
  { id: 'unit_2_jubingxi', parentId: 'dept_run_2', name: '2#聚丙烯', type: 'unit' },
  { id: 'unit_3_jubingxi', parentId: 'dept_run_2', name: '3#聚丙烯', type: 'unit' },
  { id: 'dept_run_3', parentId: 'factory', name: '运行三部', type: 'department' },
  { id: 'unit_2_liuhuang', parentId: 'dept_run_3', name: '2#硫磺', type: 'unit' },
  { id: 'unit_2_dcc', parentId: 'dept_run_3', name: '2#DCC', type: 'unit' },
  { id: 'unit_xiqing', parentId: 'dept_run_3', name: '烯烃分离', type: 'unit' },
  { id: 'dept_run_45', parentId: 'factory', name: '运行四五部', type: 'department' },
  { id: 'unit_iii_chang', parentId: 'dept_run_45', name: 'III常', type: 'unit' },
  { id: 'unit_hangmei_hydrogen', parentId: 'dept_run_45', name: '航煤加氢', type: 'unit' },
  { id: 'unit_5203_hcu_2', parentId: 'dept_run_45', name: '2#石脑油加氢', type: 'unit', code: '5203' },
  { id: 'unit_solvent_recycle', parentId: 'dept_run_45', name: '2#溶剂再生/酸性水汽提', type: 'unit' },
  { id: 'unit_2_hydrocrack', parentId: 'dept_run_45', name: '2#加裂', type: 'unit', code: '5203' },
  { id: 'area_2_hydrocrack_fractionation', parentId: 'unit_2_hydrocrack', name: '分馏/回流系统', type: 'area' },
  { id: 'unit_light_recovery', parentId: 'dept_run_45', name: '轻烃回收', type: 'unit' },
  { id: 'unit_2_wax_hydrogen', parentId: 'dept_run_45', name: '2#蜡加', type: 'unit' },
  { id: 'unit_diesel_hydrogen', parentId: 'dept_run_45', name: '裂柴加氢', type: 'unit' },
  { id: 'dept_run_6', parentId: 'factory', name: '运行六部', type: 'department' },
  { id: 'unit_1_dingxi', parentId: 'dept_run_6', name: '1-丁烯', type: 'unit' },
  { id: 'unit_mtbe', parentId: 'dept_run_6', name: 'MTBE', type: 'unit' },
  { id: 'unit_1_dcc_regen', parentId: 'dept_run_6', name: '1#DCC反再', type: 'unit' },
  { id: 'unit_1_dcc_fraction', parentId: 'dept_run_6', name: '1#DCC分馏', type: 'unit' },
  { id: 'unit_product_gas', parentId: 'dept_run_6', name: '产品精制气分', type: 'unit' },
  { id: 'dept_run_7', parentId: 'factory', name: '运行七部', type: 'department' },
  { id: 'unit_1_wax_hydrogen', parentId: 'dept_run_7', name: '1#蜡加', type: 'unit' },
  { id: 'unit_1_hydrocrack', parentId: 'dept_run_7', name: '1#加裂', type: 'unit' },
  { id: 'dept_run_8', parentId: 'factory', name: '运行八部', type: 'department' },
  { id: 'unit_1_reforming', parentId: 'dept_run_8', name: '1#重整', type: 'unit' },
  { id: 'unit_hydrogen', parentId: 'dept_run_8', name: '制氢', type: 'unit' },
  { id: 'unit_1_naphtha_hydrogen', parentId: 'dept_run_8', name: '1#石脑油加氢', type: 'unit' },
  { id: 'unit_2_extraction', parentId: 'dept_run_8', name: '2#抽提', type: 'unit' },
  { id: 'unit_1_extraction', parentId: 'dept_run_8', name: '1#抽提', type: 'unit' },
  { id: 'unit_aromatics', parentId: 'dept_run_8', name: '芳构化', type: 'unit' },
  { id: 'dept_run_9', parentId: 'factory', name: '运行九部', type: 'department' },
  { id: 'unit_benzene_ethylene', parentId: 'dept_run_9', name: '苯乙烯', type: 'unit' },
  { id: 'unit_1_sulfur', parentId: 'dept_run_9', name: '1#硫磺', type: 'unit' },
  { id: 'unit_ethylbenzene', parentId: 'dept_run_9', name: '乙苯', type: 'unit' },
  { id: 'dept_run_10', parentId: 'factory', name: '运行十部', type: 'department' },
  { id: 'unit_aromatics_extract', parentId: 'dept_run_10', name: '芳烃', type: 'unit' },
  { id: 'unit_disproportionation', parentId: 'dept_run_10', name: '歧化', type: 'unit' },
  { id: 'dept_run_11', parentId: 'factory', name: '运行十一部', type: 'department' },
  { id: 'unit_2_reforming', parentId: 'dept_run_11', name: '2#重整', type: 'unit' },
  { id: 'unit_3_extraction', parentId: 'dept_run_11', name: '3#抽提', type: 'unit' },
  { id: 'unit_4_extraction', parentId: 'dept_run_11', name: '4#抽提', type: 'unit' },
  { id: 'dept_storage', parentId: 'factory', name: '储运部', type: 'department' },
  { id: 'unit_tank_area_1_2', parentId: 'dept_storage', name: '一期二期罐区', type: 'unit' },
  { id: 'unit_tank_area_3', parentId: 'dept_storage', name: '三期罐区', type: 'unit' },
  { id: 'unit_wharf', parentId: 'dept_storage', name: '码头', type: 'unit' },
  { id: 'dept_utility', parentId: 'factory', name: '公用工程一部', type: 'department' },
  { id: 'unit_wastewater', parentId: 'dept_utility', name: '污水处理场', type: 'unit' },
  { id: 'unit_desalt', parentId: 'dept_utility', name: '除盐水站', type: 'unit' },
  { id: 'unit_circulating_water', parentId: 'dept_utility', name: '循环水场', type: 'unit' },
];

function scorePercent(value?: number) {
  return Math.round((value ?? 0) * 100);
}

function scoreStatus(value?: number) {
  if ((value ?? 0) < 0.4) return 'exception';
  if ((value ?? 0) >= 0.75) return 'success';
  return 'normal';
}

function tagColor(level?: string) {
  if (level === 'excellent') return 'green';
  if (level === 'good') return 'blue';
  if (level === 'fair') return 'orange';
  return 'red';
}

function gateSeverityColor(severity?: string) {
  if (severity === 'critical' || severity === 'high' || severity === 'error' || severity === 'blocked') return 'red';
  if (severity === 'medium' || severity === 'warning') return 'orange';
  if (severity === 'ok' || severity === 'low') return 'green';
  return 'default';
}

function gateCheckLabel(value?: string) {
  if (value === 'data_quality') return '数据质量';
  if (value === 'operating_condition') return '运行工况';
  if (value === 'constraints') return '约束/饱和';
  if (value === 'oscillation') return '振荡状态';
  if (value === 'identification') return '可辨识性';
  return value || '-';
}

function gateImpact(check: { passed?: boolean; severity?: string; name?: string }, blockingReasons: Array<{ type: string; severity: string }>) {
  const severity = String(check.severity || '');
  if (!check.passed || ['critical', 'high', 'error', 'blocked'].includes(severity)) {
    return { text: '硬阻断', color: 'red' };
  }
  const hasSoftReason = blockingReasons.some((reason) => {
    const reasonType = reason.type === 'constraint' ? 'constraints' : reason.type;
    return reasonType === check.name && ['medium', 'warning', 'low', 'info'].includes(String(reason.severity));
  });
  if (hasSoftReason || ['medium', 'warning'].includes(severity)) {
    return { text: '软提醒', color: 'orange' };
  }
  return { text: '无影响', color: 'green' };
}

function gateCheckMessage(
  check: { name?: string; message?: string; evidence?: Record<string, unknown> },
  blockingReasons: Array<{ type: string; severity: string; message: string }>,
) {
  const softReason = blockingReasons.find((reason) => {
    const reasonType = reason.type === 'constraint' ? 'constraints' : reason.type;
    return reasonType === check.name && ['medium', 'warning', 'low', 'info'].includes(String(reason.severity));
  });
  if (check.name === 'operating_condition' && softReason) {
    return softReason.message;
  }
  return check.message || '-';
}

function gateDecisionText(decision?: string) {
  if (decision === 'ready') return '可发起整定';
  if (decision === 'caution') return '谨慎整定';
  if (decision === 'blocked') return '暂不建议整定';
  return decision || '待评估';
}

function monitoringStatusColor(status?: string) {
  if (status === 'normal') return 'green';
  if (status === 'warning') return 'orange';
  if (status === 'alarm' || status === 'critical') return 'red';
  if (status === 'unavailable') return 'default';
  return 'blue';
}

function monitoringStatusText(status?: string) {
  if (status === 'normal') return '正常';
  if (status === 'warning') return '关注';
  if (status === 'alarm') return '报警';
  if (status === 'critical') return '严重';
  if (status === 'unavailable') return '不可用';
  return status || '-';
}

function alertSeverityColor(severity?: string) {
  if (severity === 'critical' || severity === 'high' || severity === '高') return 'red';
  if (severity === 'warning' || severity === 'medium' || severity === '中') return 'orange';
  return 'blue';
}

function assetTagColor(type: AssetNodeType) {
  if (type === 'factory') return 'blue';
  if (type === 'department') return 'cyan';
  if (type === 'unit') return 'green';
  if (type === 'area') return 'orange';
  if (type === 'equipment') return 'purple';
  return 'default';
}

function inferLoopAssetId(loopId?: string) {
  if (!loopId) return 'factory';
  if (loopId.startsWith('5203_')) return 'area_2_hydrocrack_fractionation';
  return 'factory';
}

function buildAssetTreeData(nodes: AssetNode[]) {
  const childrenByParent = new Map<string | undefined, AssetNode[]>();
  nodes.forEach((node) => {
    const list = childrenByParent.get(node.parentId) ?? [];
    list.push(node);
    childrenByParent.set(node.parentId, list);
  });
  const build = (parentId?: string): DataNode[] =>
    (childrenByParent.get(parentId) ?? []).map((node) => {
      const children = build(node.id);
      return {
        key: node.id,
        title: (
          <Space size={6}>
            <span>{node.name}</span>
            <Tag color={assetTagColor(node.type)}>{ASSET_TYPE_LABEL[node.type]}</Tag>
            {node.code && <Tag color="blue">{node.code}</Tag>}
          </Space>
        ),
        children: children.length ? children : undefined,
      };
    });
  return build(undefined);
}

function getDescendantAssetIds(nodes: AssetNode[], rootId: string) {
  const ids = new Set<string>([rootId]);
  let changed = true;
  while (changed) {
    changed = false;
    nodes.forEach((node) => {
      if (node.parentId && ids.has(node.parentId) && !ids.has(node.id)) {
        ids.add(node.id);
        changed = true;
      }
    });
  }
  return ids;
}

function nextAssetType(parentType?: AssetNodeType): AssetNodeType {
  if (parentType === 'factory') return 'department';
  if (parentType === 'department') return 'unit';
  if (parentType === 'unit') return 'area';
  if (parentType === 'area') return 'equipment';
  return 'area';
}

function formatNumber(value?: number | null, digits = 2) {
  return value === null || value === undefined || Number.isNaN(value) ? '-' : value.toFixed(digits);
}

function formatChartTooltipValue(value: unknown, digits = 3) {
  if (typeof value === 'number') return formatNumber(value, digits);
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? formatNumber(parsed, digits) : value;
  }
  return '-';
}

function chartSeriesColor(series?: string) {
  if (!series) return '#35a7ff';
  if (series.includes('MV')) return '#ff9f43';
  if (series.includes('SV') || series.includes('SP')) return '#28d7c5';
  if (series.includes('仿真') || series.includes('拟合')) return '#28d7c5';
  return '#35a7ff';
}

const chartLineTooltip = {
  title: (datum: { t?: string | number }) => `X：${datum?.t ?? '-'}`,
  items: [
    (datum: { series?: string; value?: unknown }) => {
      const value = formatChartTooltipValue(datum.value);
      const series = datum.series || '数值';
      return {
        name: `${series}：${value}`,
        value: '',
        color: chartSeriesColor(series),
      };
    },
  ],
};

function formatRange(min?: number | null, max?: number | null, digits = 2) {
  return `${formatNumber(min, digits)} ~ ${formatNumber(max, digits)}`;
}

function formatPercentValue(value?: number | null, digits = 0) {
  return value === null || value === undefined || Number.isNaN(value) ? '-' : `${(value * 100).toFixed(digits)}%`;
}

function formatOscillationEvidence(detected?: boolean, confidence?: number | null) {
  if (!detected) return '无显著周期峰';
  return formatPercentValue(confidence, 1);
}

function formatOscillationPhaseHint(detected?: boolean, phaseHint?: string | null) {
  if (!detected) return '未判定';
  if (phaseHint === 'pv_mv_same_period') return 'PV/MV 同周期';
  if (phaseHint === 'pv_only_periodic') return 'PV 单侧周期';
  if (phaseHint === 'unknown' || !phaseHint) return '证据不足';
  return phaseHint;
}

function formatHarrisBasis(value?: string) {
  if (value === 'pv_minus_sp') return 'PV-SV 跟踪误差';
  if (value === 'pv_minus_constant_sp') return 'PV-固定SV偏差';
  if (value === 'detrended_pv') return '去趋势 PV 波动';
  return value || '-';
}

function formatCpkBasis(value?: string) {
  if (value === 'pv_spec_limits') return 'PV规格上下限';
  if (value === 'missing_pv_spec_limits') return '未配置PV规格上下限';
  return value || '-';
}

function formatProcessDirection(direction?: string | null) {
  if (direction === 'positive_gain' || direction === 'positive') return '正作用（MV↑ PV↑）';
  if (direction === 'negative_gain' || direction === 'negative') return '反作用（MV↑ PV↓）';
  return '不确定';
}

function formatProcessDirectionBasis(basis?: string | null) {
  if (basis === 'dmv_to_dpv_lag_corr') return 'MV/PV 变化量滞后相关';
  if (basis === 'mv_to_pv_lag_corr') return 'MV/PV 水平值滞后相关';
  return basis || '-';
}

function translateWindowAlgorithmFamily(value?: string) {
  if (value === 'sp_step') return 'SP 阶跃';
  if (value === 'mv_step') return 'MV 阶跃';
  if (value === 'mv_ramp') return 'MV 斜坡';
  if (value === 'steady_disturbance') return '稳态扰动';
  if (value === 'rolling_scan') return '滚动扫描';
  return value || '-';
}

function translatePolicyState(value?: string) {
  if (value === 'preferred') return '优先';
  if (value === 'available') return '可用';
  if (value === 'deprioritized') return '降级';
  if (value === 'disabled') return '禁用';
  if (value === 'ran') return '已运行';
  if (value === 'skipped') return '已跳过';
  return value || '-';
}

function translatePolicyUsageStatus(value?: WindowPolicyFieldUsage['status']) {
  if (value === 'consumed') return '已被算法消费';
  if (value === 'downstream_hint') return '下游提示';
  if (value === 'display_only') return '仅展示/审计';
  return '-';
}

function policyUsageStatusColor(value?: WindowPolicyFieldUsage['status']) {
  if (value === 'consumed') return 'green';
  if (value === 'downstream_hint') return 'blue';
  return 'default';
}

function translatePolicyFieldName(field?: string, label?: string) {
  const zh: Record<string, string> = {
    preferred_algorithm_families: '优先算法族',
    deprioritized_algorithm_families: '降级算法族',
    disabled_algorithm_families: '禁用算法族',
    algorithm_plan: '算法族执行计划',
    min_mv_excitation: '最小 MV 激励',
    min_sp_excitation: '最小 SP 激励',
    min_pv_response: '最小 PV 响应',
    max_mv_saturation_ratio: '最大 MV 饱和比例',
    max_pv_noise_ratio: '最大 PV 噪声比例',
    max_drift_ratio: '最大漂移比例',
    expected_dead_time_range_s: '预期死区范围',
    expected_time_constant_range_s: '预期时间常数范围',
    expected_gain_sign: '预期增益方向',
    min_window_points: '最小窗口点数',
    min_window_duration_s: '最小窗口时长',
    max_window_points: '最大窗口点数',
    pre_window_s: '事件前窗口',
    post_window_s: '事件后窗口',
    steady_scan_window_s: '稳态扫描窗口',
    steady_scan_step_s: '稳态扫描步长',
    merge_gap_s: '事件合并间隔',
    max_candidates_per_family: '单算法族候选上限',
    allowed_operating_states: '允许工况',
    avoid_operating_states: '规避工况',
    scoring_weights: '评分权重',
    hard_guards: '硬约束',
    soft_penalties: '软惩罚',
    rationale: '策略依据',
    ontology_facts: '本体事实',
  };
  return zh[field || ''] || label || field || '-';
}

function formatPolicyValue(value: unknown) {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : formatNumber(value, 3);
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.length ? value.map((item) => typeof item === 'string' ? translateWindowAlgorithmFamily(item) : String(item)).join('、') : '-';
  return JSON.stringify(value);
}

function windowFlowStatusText(status: WindowFlowStepStatus) {
  if (status === 'done') return '已完成';
  if (status === 'running') return '执行中';
  return '待执行';
}

function windowFlowStatusColor(status: WindowFlowStepStatus) {
  if (status === 'done') return 'green';
  if (status === 'running') return 'processing';
  return 'default';
}

function policyLoopImpact(loopType: string) {
  const label = LOOP_TYPE_LABEL[loopType] ?? loopType;
  return `${label}回路：辨识阶段按模型顺序扩大搜索；精修阶段在模型服务不可用时按备选模型池重试；时间常数下界约束优化器搜索空间，现实时间常数范围影响整定后仿真评分。`;
}

function yesNo(value?: boolean | null) {
  if (value === true) return '是';
  if (value === false) return '否';
  return '-';
}

function operatingConditionText(label?: string) {
  if (label === 'stable_production') return '稳定生产';
  if (label === 'load_change') return '负荷/工况切换';
  if (label === 'disturbance_recovery') return '扰动恢复';
  if (label === 'constraint_limited') return '约束受限';
  if (label === 'oscillatory') return '存在振荡';
  if (label === 'data_unreliable') return '数据不可靠';
  if (label === 'transition_or_load_change') return '过渡/负荷变化';
  if (label === 'data_quality_issue') return '数据质量问题';
  return '未判定';
}

function tuningSuitabilityText(value?: string) {
  if (value === 'suitable') return '适合整定';
  if (value === 'cautious') return '谨慎整定';
  if (value === 'not_recommended') return '不建议整定';
  return '未判定';
}

function tuningSuitabilityColor(value?: string) {
  if (value === 'suitable') return 'green';
  if (value === 'cautious') return 'orange';
  if (value === 'not_recommended') return 'red';
  return 'default';
}

function evidenceStatusText(value?: string) {
  if (value === 'normal') return '正常';
  if (value === 'warning') return '关注';
  if (value === 'alarm') return '异常';
  return value || '-';
}

function evidenceStatusColor(value?: string) {
  if (value === 'normal') return 'green';
  if (value === 'warning') return 'orange';
  if (value === 'alarm') return 'red';
  return 'default';
}

function conditionEvidenceName(name?: string) {
  if (name === 'data_quality') return '数据质量';
  if (name === 'mv_saturation') return 'MV 饱和/约束';
  if (name === 'oscillation') return '振荡证据';
  if (name === 'transition') return '均值漂移/过渡';
  if (name === 'excitation') return '激励充分性';
  return name || '-';
}

function conditionEvidenceDetail(detail?: string) {
  if (detail === 'missing_or_irregular_sample_ratio') return '缺失率或采样不规则比例';
  if (detail === 'mv_near_observed_or_percent_limits') return 'MV 接近观测上下限或百分比边界';
  if (detail === 'first_second_half_mean_shift_and_sp_activity') return '前后半段均值漂移与 SP 活跃度';
  if (detail === 'good') return '激励较充分';
  if (detail === 'fair') return '激励一般';
  if (detail === 'poor') return '激励不足';
  if (detail === 'pv_mv_same_period') return 'PV/MV 存在同周期迹象';
  if (detail === 'pv_only_periodic') return 'PV 单侧周期迹象';
  if (detail === 'unknown') return '证据不足';
  return detail || '-';
}

function conditionRecommendationText(value?: string) {
  if (value === 'fix_data_quality_before_assessment') return '先处理缺失、断点或采样异常，再做评估。';
  if (value === 'exclude_saturated_periods_or_check_valve_capacity') return '剔除饱和片段或先确认阀门/执行机构能力。';
  if (value === 'run_oscillation_diagnosis_before_tuning') return '先做振荡诊断，避免把振荡误当成可辨识激励。';
  if (value === 'prefer_steady_segments_for_identification') return '优先选择稳定片段做辨识，过渡段只作工况参考。';
  if (value === 'need_more_mv_excitation_for_identification') return '当前 MV 激励不足，建议补充可控小阶跃或等待更充分历史片段。';
  if (value === 'condition_is_acceptable_for_candidate_tuning') return '当前工况可进入候选整定评估。';
  return value || '-';
}

function LoopMonitoringPageInner() {
  const [activeModule, setActiveModule] = useState<ModuleKey>('workspace');
  const [activeSub, setActiveSub] = useState<SubKey>('dashboard');
  const [viewMode, setViewMode] = useState<'dialogue' | 'classic'>('dialogue');
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [loops, setLoops] = useState<HistoryLoop[]>([]);
  const [selectedLoopId, setSelectedLoopId] = useState<string>();
  const [assetNodes, setAssetNodes] = useState<AssetNode[]>(DEFAULT_ASSET_NODES);
  const [selectedAssetNodeId, setSelectedAssetNodeId] = useState<string>('unit_2_hydrocrack');
  const [assetDraftName, setAssetDraftName] = useState('');
  const [assetDraftType, setAssetDraftType] = useState<AssetNodeType>('area');
  const [assetRenameValue, setAssetRenameValue] = useState('');
  const [series, setSeries] = useState<LoopSeriesResp | null>(null);
  const [seriesLoading, setSeriesLoading] = useState(false);
  const [trendPreset, setTrendPreset] = useState<TrendPreset>('all');
  const [trendPointLimit, setTrendPointLimit] = useState<TrendPointLimit>('6000');
  const [trendSplitYAxis, setTrendSplitYAxis] = useState(false);
  const [trendCustomRange, setTrendCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const [featureRangePreset, setFeatureRangePreset] = useState<FeatureRangePreset>('all');
  const [featureCustomRange, setFeatureCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const [featureLoading, setFeatureLoading] = useState(false);
  const [windowRangePreset, setWindowRangePreset] = useState<FeatureRangePreset>('all');
  const [windowCustomRange, setWindowCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  // 整定任务页独立的时间窗 state；与窗口候选页的 windowRangePreset 解耦，
  // 这样两个页面分别发起整定时不会互相覆盖时间选择。
  const [tuningRangePreset, setTuningRangePreset] = useState<FeatureRangePreset>('8h');
  const [tuningCustomRange, setTuningCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const [tuningPriorRangePreset, setTuningPriorRangePreset] = useState<FeatureRangePreset>('8h');
  const [tuningPriorCustomRange, setTuningPriorCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const [tuningPriorCoreData, setTuningPriorCoreData] = useState<HistoryLoopTuningPrior | null>(null);
  const [tuningPriorOntologyData, setTuningPriorOntologyData] = useState<HistoryLoopTuningPrior | null>(null);
  const [tuningPriorReviewData, setTuningPriorReviewData] = useState<HistoryLoopTuningPrior | null>(null);
  const [tuningPriorCoreLoading, setTuningPriorCoreLoading] = useState(false);
  const [tuningPriorOntologyLoading, setTuningPriorOntologyLoading] = useState(false);
  const [tuningPriorReviewLoading, setTuningPriorReviewLoading] = useState(false);
  const [tuningPriorCoreError, setTuningPriorCoreError] = useState<string | null>(null);
  const [tuningPriorOntologyError, setTuningPriorOntologyError] = useState<string | null>(null);
  const [tuningPriorReviewError, setTuningPriorReviewError] = useState<string | null>(null);
  // 整定任务页是否启用 LLM 顾问；为了和窗口候选保持一致，默认 true。
  const [tuningUseLlm, setTuningUseLlm] = useState<boolean>(true);
  const [assessment, setAssessment] = useState<HistoryLoopAssessment | null>(null);
  const [assessmentLoading, setAssessmentLoading] = useState(false);
  const [assessmentError, setAssessmentError] = useState<string | null>(null);
  const [loopFeatures, setLoopFeatures] = useState<HistoryLoopFeatures | null>(null);
  const [loopMonitoring, setLoopMonitoring] = useState<HistoryLoopMonitoring | null>(null);
  const [monitoringByLoopId, setMonitoringByLoopId] = useState<Record<string, HistoryLoopMonitoring>>({});
  const monitoringBulkInFlightRef = useRef<Set<string>>(new Set());
  const [windows, setWindows] = useState<HistoryWindow[]>([]);
  const [windowAlgorithmSummary, setWindowAlgorithmSummary] = useState<Record<string, { total: number; usable: number }>>({});
  const [selectedWindowIndex, setSelectedWindowIndex] = useState<number>();
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [running, setRunning] = useState(false);
  const [taskId, setTaskId] = useState<string>();
  const [taskStatus, setTaskStatus] = useState<TaskStatus>('idle');
  const [taskStartedAt, setTaskStartedAt] = useState<string>();
  const [taskCurrentStage, setTaskCurrentStage] = useState<string>();
  const [taskStageStatus, setTaskStageStatus] = useState<TaskStageStatusMap>({});
  const [taskStageData, setTaskStageData] = useState<TaskStageDataMap>({});
  // running 事件里的 sub-phase 数据（如 ontology_policy 的 phase=fetching_mcp_context|building_policy）。
  // 不与 taskStageData（done payload）合并，避免 done 之后被 running 残留覆盖。
  const [taskStageRunningData, setTaskStageRunningData] = useState<TaskStageDataMap>({});
  const [taskWindowSelection, setTaskWindowSelection] = useState<WindowSelectionMeta | null>(null);
  const [taskModelReview, setTaskModelReview] = useState<ModelReviewMeta | null>(null);
  const [taskRefinements, setTaskRefinements] = useState<IdentificationRefinementMeta[]>([]);
  const [taskThinking, setTaskThinking] = useState<LlmThinkingEvent[]>([]);
  const [taskAttempts, setTaskAttempts] = useState<IdentificationAttempt[]>([]);
  const [selectedFitAttemptKey, setSelectedFitAttemptKey] = useState<string>();
  const [taskResult, setTaskResult] = useState<TuningResult | null>(null);
  const [taskError, setTaskError] = useState<string>();
  const [taskAbort, setTaskAbort] = useState<AbortController | null>(null);
  const [events, setEvents] = useState<TaskEventLog[]>([]);
  const [dataSourceType, setDataSourceType] = useState<string>('history_upload');
  const [taskDetailOpen, setTaskDetailOpen] = useState(false);
  const [rawLogExpanded, setRawLogExpanded] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [expandedModules, setExpandedModules] = useState<Record<ModuleKey, boolean>>(INITIAL_EXPANDED_MODULES);
  const [dashboardConfigOpen, setDashboardConfigOpen] = useState(false);
  const [dashboardWidgetKeys, setDashboardWidgetKeys] = useState<DashboardWidgetKey[]>(() => {
    if (typeof window === 'undefined') return DEFAULT_DASHBOARD_WIDGET_KEYS;
    try {
      const raw = window.localStorage.getItem(DASHBOARD_WIDGET_STORAGE_KEY);
      return normalizeDashboardWidgetKeys(raw ? JSON.parse(raw) : []);
    } catch {
      return DEFAULT_DASHBOARD_WIDGET_KEYS;
    }
  });
  const [draggedDashboardWidgetKey, setDraggedDashboardWidgetKey] = useState<DashboardWidgetKey | null>(null);
  const [assistantInput, setAssistantInput] = useState('');
  const [assistantMessages, setAssistantMessages] = useState<AssistantMessage[]>([]);
  const [assistantSessions, setAssistantSessions] = useState<AssistantSessionSummary[]>([]);
  const [activeAssistantSession, setActiveAssistantSession] = useState<AssistantSession | null>(null);
  const [assistantSessionsLoading, setAssistantSessionsLoading] = useState(false);
  const [assistantStreaming, setAssistantStreaming] = useState(false);
  const [pinnedAssistantSessionIds, setPinnedAssistantSessionIds] = useState<string[]>(() => {
    if (typeof window === 'undefined') return [];
    try {
      const raw = window.localStorage.getItem('pid_v2_pinned_assistant_sessions');
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === 'string') : [];
    } catch {
      return [];
    }
  });
  const assistantAbortRef = useRef<AbortController | null>(null);
  const dashboardWorstSelectionRef = useRef<string | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem('pid_v2_pinned_assistant_sessions', JSON.stringify(pinnedAssistantSessionIds));
    } catch {
      // Local pinning is an optional UI preference.
    }
  }, [pinnedAssistantSessionIds]);

  useEffect(() => {
    try {
      window.localStorage.setItem(DASHBOARD_WIDGET_STORAGE_KEY, JSON.stringify(dashboardWidgetKeys));
    } catch {
      // Dashboard layout is an optional UI preference.
    }
  }, [dashboardWidgetKeys]);

  const enabledDashboardWidgets = useMemo(
    () => new Set(dashboardWidgetKeys),
    [dashboardWidgetKeys],
  );

  const hideDashboardWidget = useCallback((key: DashboardWidgetKey) => {
    setDashboardWidgetKeys((prev) => {
      if (prev.length <= 1) {
        message.warning('至少保留一个看板模块');
        return prev;
      }
      return prev.filter((item) => item !== key);
    });
  }, []);

  const moveDashboardWidget = useCallback((source: DashboardWidgetKey, target: DashboardWidgetKey) => {
    if (source === target) return;
    setDashboardWidgetKeys((prev) => {
      const from = prev.indexOf(source);
      const to = prev.indexOf(target);
      if (from < 0 || to < 0) return prev;
      const next = [...prev];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
  }, []);

  const pinnedAssistantSessionIdSet = useMemo(
    () => new Set(pinnedAssistantSessionIds),
    [pinnedAssistantSessionIds],
  );

  const sortedAssistantSessions = useMemo(() => {
    const pinOrder = new Map(pinnedAssistantSessionIds.map((id, index) => [id, index]));
    return [...assistantSessions].sort((left, right) => {
      const leftPinned = pinOrder.has(left.id);
      const rightPinned = pinOrder.has(right.id);
      if (leftPinned && rightPinned) return (pinOrder.get(left.id) ?? 0) - (pinOrder.get(right.id) ?? 0);
      if (leftPinned) return -1;
      if (rightPinned) return 1;
      return 0;
    });
  }, [assistantSessions, pinnedAssistantSessionIds]);

  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null);
  const [modelConfigLoading, setModelConfigLoading] = useState(false);
  const [modelConfigSaving, setModelConfigSaving] = useState(false);
  const [modelConfigTesting, setModelConfigTesting] = useState(false);
  const [modelConfigForm] = Form.useForm();
  const [modelConfigTestResult, setModelConfigTestResult] = useState<{
    status: string;
    message: string;
  } | null>(null);
  const [policyConfig, setPolicyConfig] = useState<PolicyConfig | null>(null);
  const [policyConfigLoading, setPolicyConfigLoading] = useState(false);
  const [promptConfig, setPromptConfig] = useState<PromptConfig | null>(null);
  const [promptConfigLoading, setPromptConfigLoading] = useState(false);
  const [promptConfigSaving, setPromptConfigSaving] = useState(false);
  const [promptConfigForm] = Form.useForm();
  const [activePromptField, setActivePromptField] = useState<PromptConfigField>('assistant_system_prompt');

  const currentModule = MODULES.find((item) => item.key === activeModule) ?? MODULES[0];
  const currentSub = currentModule.subs.find((item) => item.key === activeSub) ?? currentModule.subs[0];
  const isSettingsView = activeModule === 'settings';
  const shouldLoadDashboardMonitoring = activeSub === 'dashboard' || activeSub === 'loop_board';
  const shouldRestoreLatestTask = activeSub === 'tuning_task' || activeSub === 'performance_score';
  const shouldLoadAssessmentDetail = ASSESSMENT_DETAIL_SUBS.has(activeSub);
  const shouldLoadWindowDetail = WINDOW_DETAIL_SUBS.has(activeSub);
  const shouldLoadFeatureDetail = FEATURE_DETAIL_SUBS.has(activeSub);
  const shouldLoadMonitoringDetail = MONITORING_DETAIL_SUBS.has(activeSub);

  const selectedLoop = useMemo(
    () => loops.find((item) => item.loop_id === selectedLoopId),
    [loops, selectedLoopId],
  );

  useEffect(() => {
    if (!shouldRestoreLatestTask || isSettingsView) return;
    if (!selectedLoop || running) return;
    if (taskResult?.loop_name === selectedLoop.loop_id) return;

    let cancelled = false;
    const restoreLatestTask = async () => {
      try {
        const sessions = await listSessions({ limit: 30, kind: 'tune' });
        const latest = sessions.items.find((item) => (
          item.loop_name === selectedLoop.loop_id
          || item.csv_name === `history:${selectedLoop.loop_id}`
        ));
        if (!latest) return;

        const detail = await getSession(latest.task_id);
        const resultEvent = [...detail.events].reverse().find((event) => event.type === 'result');
        const resultData = resultEvent?.data as TuningResult | undefined;
        if (cancelled || !resultData?.model) return;

        setTaskId(latest.task_id);
        setTaskStartedAt(latest.created_at ? new Date(latest.created_at).toLocaleString() : undefined);
        setTaskStatus(latest.status === 'error' ? 'error' : 'done');
        setTaskResult(resultData);
        setTaskAttempts((resultData.model.attempts ?? []).map((attempt) => ({ ...attempt })));
        setTaskError(latest.error);
      } catch {
        // 历史任务恢复只是体验增强，失败时保持当前页面状态即可。
      }
    };

    void restoreLatestTask();
    return () => {
      cancelled = true;
    };
  }, [isSettingsView, running, selectedLoop, shouldRestoreLatestTask, taskResult]);

  const selectedAssetNode = useMemo(
    () => assetNodes.find((item) => item.id === selectedAssetNodeId) ?? assetNodes[0],
    [assetNodes, selectedAssetNodeId],
  );

  const selectedAssetPath = useMemo(() => {
    const byId = new Map(assetNodes.map((item) => [item.id, item]));
    const path: AssetNode[] = [];
    let current: AssetNode | undefined = selectedAssetNode;
    while (current) {
      path.unshift(current);
      current = current.parentId ? byId.get(current.parentId) : undefined;
    }
    return path;
  }, [assetNodes, selectedAssetNode]);

  const scopedLoops = useMemo(() => {
    const scopeIds = getDescendantAssetIds(assetNodes, selectedAssetNodeId);
    return loops.filter((loop) => scopeIds.has(inferLoopAssetId(loop.loop_id)));
  }, [assetNodes, loops, selectedAssetNodeId]);

  const assetTreeData = useMemo(() => buildAssetTreeData(assetNodes), [assetNodes]);

  const scopedLoopStats = useMemo(() => {
    return {
      loopCount: scopedLoops.length,
    };
  }, [scopedLoops]);

  const dashboardRows = useMemo(
    () => buildDashboardRows(scopedLoops, monitoringByLoopId),
    [monitoringByLoopId, scopedLoops],
  );

  const dashboardWorstLoopId = useMemo(
    () => (dashboardRows.find((row) => row.snapshot) ?? dashboardRows[0])?.loop.loop_id,
    [dashboardRows],
  );

  const dashboardStats = useMemo(
    () => summarizeDashboardRows(dashboardRows, scopedLoops),
    [dashboardRows, scopedLoops],
  );

  const selectedWindow = useMemo(
    () => windows.find((item) => item.index === selectedWindowIndex),
    [selectedWindowIndex, windows],
  );

  const taskAlgorithmComparison = useMemo<WindowAlgorithmFitSummary[]>(() => {
    const identificationStage = taskStageData.identification ?? {};
    const stageComparison = identificationStage.algorithm_comparison;
    const resultComparison = taskResult?.model?.algorithm_comparison;
    const source = Array.isArray(stageComparison) ? stageComparison : resultComparison;
    return Array.isArray(source) ? source as WindowAlgorithmFitSummary[] : [];
  }, [taskResult, taskStageData]);

  const fitPreviewAttempts = useMemo(() => {
    const attemptsWithPreview = taskAttempts
      .filter((attempt) => attempt.success && !!attempt.fit_preview?.points?.length)
      .sort((a, b) => {
        const roundDiff = (b.round ?? 0) - (a.round ?? 0);
        if (roundDiff) return roundDiff;
        return (b.fit_score ?? -9999) - (a.fit_score ?? -9999);
      });
    if (attemptsWithPreview.length) return attemptsWithPreview;

    const model = taskResult?.model;
    if (!model?.fit_preview?.points?.length) return [];
    return [{
      success: true,
      round: 0,
      model_type: model.model_type,
      window_source: model.window_source,
      K: model.K,
      T: model.T,
      T1: model.T1,
      T2: model.T2,
      L: model.L,
      r2_score: model.r2_score,
      normalized_rmse: model.normalized_rmse,
      confidence: model.confidence,
      fit_preview: model.fit_preview,
    }];
  }, [taskAttempts, taskResult]);

  const selectedFitAttempt = useMemo(() => {
    if (!fitPreviewAttempts.length) return undefined;
    return fitPreviewAttempts.find((attempt) => attemptFitKey(attempt) === selectedFitAttemptKey)
      ?? fitPreviewAttempts[0];
  }, [fitPreviewAttempts, selectedFitAttemptKey]);

  const fitPreviewChartData = useMemo(() => {
    const points = selectedFitAttempt?.fit_preview?.points ?? [];
    return points.flatMap((point) => {
      const x = point.time ?? point.index;
      return [
        { t: x, value: point.pv, series: 'PV 实测' },
        { t: x, value: point.pv_fit, series: 'PV 仿真' },
        { t: x, value: point.mv, series: 'MV' },
      ];
    });
  }, [selectedFitAttempt]);

  const deterministicRefinement = useMemo(
    () => [...taskRefinements].reverse().find((item) => item.source === 'deterministic_algorithm_policy'),
    [taskRefinements],
  );

  const tuningGate = useMemo(() => {
    const readiness = assessment?.tuning_readiness;
    const decision = readiness?.decision ?? assessment?.summary?.decision;
    const gateChecks = readiness?.gate_checks ?? [];
    const failedChecks = gateChecks.filter((item) => !item.passed);
    const blockingReasons = readiness?.blocking_reasons ?? [];
    const hardBlocked = decision === 'blocked'
      || failedChecks.some((item) => ['critical', 'high', 'error', 'blocked'].includes(String(item.severity)));
    const caution = decision === 'caution' || blockingReasons.length > 0 || failedChecks.length > 0;
    return {
      decision,
      hardBlocked,
      caution,
      score: readiness?.score ?? assessment?.readiness?.score,
      level: readiness?.level ?? assessment?.readiness?.level,
      gateChecks,
      failedChecks,
      blockingReasons,
      nextAction: assessment?.summary?.recommended_next_action_text,
    };
  }, [assessment]);

  const railAlarms = useMemo(() => {
    const monitoringEvents = loopMonitoring?.monitoring.events ?? [];
    if (monitoringEvents.length) {
      return monitoringEvents.map((event, index) => ({
        key: `monitoring-event-${index}`,
        time: '当前',
        level: event.severity || '提示',
        name: event.name || event.type || 'monitoring',
        value: event.message,
        status: event.status || 'new',
        recommendation: event.recommendation || '',
        evidence: event.evidence ? JSON.stringify(event.evidence) : '',
      }));
    }
    const monitoringAlerts = loopMonitoring?.monitoring.alerts ?? [];
    if (monitoringAlerts.length) {
      return monitoringAlerts.map((alert, index) => ({
        key: `monitoring-${index}`,
        time: '当前',
        level: alert.severity || '提示',
        name: alert.type || 'monitoring',
        value: alert.message,
        status: monitoringStatusText(loopMonitoring?.monitoring.status),
        recommendation: '',
        evidence: '',
      }));
    }
    const flags = assessment?.diagnostics.flags ?? [];
    if (flags.length) {
      return flags.map((flag, index) => ({
        key: `flag-${index}`,
        time: '当前',
        level: flag.severity || '提示',
        name: flag.type,
        value: flag.message,
        status: '待确认',
        recommendation: '',
        evidence: '',
      }));
    }
    return [
      { key: 'monitoring', time: '当前', level: '中', name: '监控分', value: loopMonitoring?.monitoring?.overall_score === undefined ? '等待监控快照' : `${scorePercent(loopMonitoring.monitoring.overall_score)}%`, status: '跟踪', recommendation: '', evidence: '' },
      { key: 'source', time: '当前', level: '低', name: '数据源', value: dataSourceType === 'history' ? '历史文件导入' : '历史仓库/实时库', status: '正常', recommendation: '', evidence: '' },
      { key: 'task', time: taskStartedAt ? new Date(taskStartedAt).toLocaleTimeString() : '未启动', level: taskStatus === 'error' ? '高' : '低', name: '整定任务', value: taskId ? `任务 ${taskId}` : '暂无运行任务', status: taskStatus === 'done' ? '完成' : taskStatus === 'running' ? '运行' : '空闲', recommendation: '', evidence: '' },
    ];
  }, [assessment?.diagnostics.flags, dataSourceType, loopMonitoring, selectedLoop, taskId, taskStartedAt, taskStatus]);

  const switchTo = (moduleKey: ModuleKey, subKey: SubKey) => {
    setActiveModule(moduleKey);
    setActiveSub(subKey);
    setExpandedModules((prev) => ({ ...prev, [moduleKey]: true }));
  };

  const runAssistantAction = (action: AssistantAction) => {
    if (action.loopId) setSelectedLoopId(action.loopId);
    switchTo(action.target, action.sub);
    setViewMode('classic');
  };

  const buildDialogueActions = useCallback((loopId?: string | null): AssistantAction[] => [
    { label: '查看趋势与频谱', target: 'monitor', sub: 'trend_spectrum', loopId: loopId || undefined },
    { label: '生成整定先验', target: 'tuning', sub: 'tuning_prior', loopId: loopId || undefined },
    { label: '进入整定任务', target: 'tuning', sub: 'tuning_task', loopId: loopId || undefined },
  ], []);

  const normalizeAssistantAction = useCallback((value: string, loopId?: string | null): AssistantAction | null => {
    const text = value.trim().replace(/^[-•\d.\s]+/, '');
    if (!text) return null;
    const startsLikeAction = /^[-•\d.\s]*(进入|查看|打开|前往|跳转|发起|建议进入|建议查看)/.test(value.trim());
    if (!startsLikeAction || text.length > 32) return null;
    if (text.includes('趋势') || text.includes('频谱')) return { label: text, target: 'monitor', sub: 'trend_spectrum', loopId: loopId || undefined };
    if (text.includes('画像')) return { label: text, target: 'monitor', sub: 'loop_profile', loopId: loopId || undefined };
    if (text.includes('先验')) return { label: text, target: 'tuning', sub: 'tuning_prior', loopId: loopId || undefined };
    if (text.includes('窗口')) return { label: text, target: 'tuning', sub: 'id_windows', loopId: loopId || undefined };
    if (text.includes('整定任务') || text.includes('整定页面') || text.includes('发起整定') || text.includes('进入整定')) return { label: text, target: 'tuning', sub: 'tuning_task', loopId: loopId || undefined };
    return null;
  }, []);

  const formatAssistantEvent = useCallback((event: Record<string, unknown>): AssistantEventItem | null => {
    const type = String(event.type || '');
    if (!type || type === 'answer_delta' || type === 'done') return null;
    if (type === 'thinking_step' || type === 'reasoning_delta') {
      const content = String(event.content || '').trim();
      return {
        id: `${Date.now()}-${Math.random()}`,
        type,
        title: content || '模型正在结合会话历史、当前回路和可用监控指标生成判断。',
      };
    }
    if (type === 'tool_event') {
      return {
        id: `${Date.now()}-${Math.random()}`,
        type,
        title: String(event.name || 'tool'),
        detail: `状态：${String(event.status || 'ok')}`,
      };
    }
    if (type === 'error') {
      return {
        id: `${Date.now()}-${Math.random()}`,
        type,
        title: '调用异常',
        detail: String(event.message || ''),
      };
    }
    return {
      id: `${Date.now()}-${Math.random()}`,
      type,
      title: type,
      detail: JSON.stringify(event),
    };
  }, []);

  const buildAssistantContext = useCallback(() => {
    const riskRows = dashboardRows
      .filter((row) => row.alertCount > 0 || row.snapshot?.status === 'warning' || row.snapshot?.status === 'alarm' || row.snapshot?.status === 'critical')
      .slice(0, 8)
      .map((row) => ({
        loop_id: row.loop.loop_id,
        loop_type: row.loop.loop_type,
        status: row.snapshot?.status,
        overall_score: row.snapshot?.overall_score,
        alerts: row.snapshot?.alerts,
        events: row.snapshot?.events,
      }));
    return {
      loop_id: selectedLoop?.loop_id ?? selectedLoopId ?? null,
      start_time: selectedLoop?.start_time ?? null,
      end_time: selectedLoop?.end_time ?? null,
      page: { module: activeModule, sub: activeSub, title: currentSub.label },
      scope: {
        loop_count: scopedLoopStats.loopCount,
        avg_score: dashboardStats.avgScore,
        normal_count: dashboardStats.normalCount,
        warning_count: dashboardStats.warningCount,
        alarm_count: dashboardStats.alarmCount,
      },
      selected_loop: selectedLoop ? {
        loop_id: selectedLoop.loop_id,
        loop_type: selectedLoop.loop_type,
        start_time: selectedLoop.start_time,
        end_time: selectedLoop.end_time,
      } : null,
      selected_monitoring: loopMonitoring?.monitoring ?? (selectedLoopId ? monitoringByLoopId[selectedLoopId]?.monitoring : null),
      selected_features: loopFeatures,
      selected_assessment: assessment,
      risk_loops: riskRows,
      safety_rules: [
        '不要直接执行整定、窗口候选或 PID 参数修改。',
        '需要操作时只输出建议动作，由用户点击后进入对应页面确认。',
        '缺少上下文时必须明确说明。',
      ],
    };
  }, [
    activeModule,
    activeSub,
    assessment,
    currentSub.label,
    dashboardRows,
    dashboardStats.alarmCount,
    dashboardStats.avgScore,
    dashboardStats.normalCount,
    dashboardStats.warningCount,
    loopFeatures,
    loopMonitoring,
    monitoringByLoopId,
    scopedLoopStats.loopCount,
    selectedLoop,
    selectedLoopId,
  ]);

  const mapAssistantSessionMessages = useCallback((session: AssistantSession | null): AssistantMessage[] => {
    if (!session?.messages?.length) return [];
    return session.messages.map((item, index) => ({
      id: Number(`${index + 1}${String(item.created_at || '').replace(/\D/g, '').slice(-6)}`) || Date.now() + index,
      role: item.role,
      text: item.content,
      reasoning: item.reasoning_summary,
      loading: false,
      actions: item.role === 'assistant' ? buildDialogueActions(session.loop_id) : undefined,
      eventLog: item.role === 'assistant'
        ? (item.raw_events ?? []).map((event) => formatAssistantEvent(event)).filter(Boolean) as AssistantEventItem[]
        : undefined,
    }));
  }, [buildDialogueActions, formatAssistantEvent]);

  const loadAssistantSessions = useCallback(async () => {
    setAssistantSessionsLoading(true);
    try {
      const resp = await listAssistantSessions(100);
      setAssistantSessions(resp.items ?? []);
    } catch (error) {
      message.error(`加载对话列表失败：${String(error)}`);
    } finally {
      setAssistantSessionsLoading(false);
    }
  }, []);

  const openAssistantSession = useCallback(async (sessionId: string) => {
    try {
      const session = await getAssistantSession(sessionId);
      setActiveAssistantSession(session);
      setAssistantMessages(mapAssistantSessionMessages(session));
      if (session.loop_id) setSelectedLoopId(session.loop_id);
    } catch (error) {
      message.error(`加载对话失败：${String(error)}`);
    }
  }, [mapAssistantSessionMessages]);

  const createDialogueSession = useCallback(async () => {
    try {
      const session = await createAssistantSession({
        title: selectedLoop ? `${selectedLoop.loop_id} 对话` : '新对话',
        loop_id: selectedLoop?.loop_id ?? null,
      });
      setActiveAssistantSession(session);
      setAssistantMessages([]);
      await loadAssistantSessions();
    } catch (error) {
      message.error(`新建对话失败：${String(error)}`);
    }
  }, [loadAssistantSessions, selectedLoop]);

  const toggleAssistantSessionPin = useCallback((sessionId: string) => {
    setPinnedAssistantSessionIds((prev) => (
      prev.includes(sessionId)
        ? prev.filter((id) => id !== sessionId)
        : [sessionId, ...prev]
    ));
  }, []);

  const renameAssistantSession = useCallback((session: AssistantSessionSummary) => {
    let nextTitle = session.title || '';
    Modal.confirm({
      title: '重命名会话',
      icon: null,
      content: (
        <Input
          autoFocus
          defaultValue={nextTitle}
          maxLength={64}
          placeholder="请输入会话名称"
          onChange={(event) => {
            nextTitle = event.target.value;
          }}
        />
      ),
      okText: '保存',
      cancelText: '取消',
      async onOk() {
        const title = nextTitle.trim();
        if (!title) {
          message.warning('会话名称不能为空');
          return Promise.reject(new Error('empty title'));
        }
        const updated = await updateAssistantSession(session.id, { title });
        setAssistantSessions((prev) => prev.map((item) => (
          item.id === session.id ? { ...item, title: updated.title, updated_at: updated.updated_at } : item
        )));
        setActiveAssistantSession((prev) => (
          prev?.id === session.id ? { ...prev, title: updated.title, updated_at: updated.updated_at } : prev
        ));
      },
    });
  }, []);

  const deleteAssistantSessionWithConfirm = useCallback((session: AssistantSessionSummary) => {
    Modal.confirm({
      title: '删除会话',
      content: `确认删除“${session.title || '未命名对话'}”？删除后无法恢复。`,
      okText: '删除',
      okButtonProps: { danger: true },
      cancelText: '取消',
      async onOk() {
        await deleteAssistantSession(session.id);
        setPinnedAssistantSessionIds((prev) => prev.filter((id) => id !== session.id));
        if (activeAssistantSession?.id === session.id) {
          setActiveAssistantSession(null);
          setAssistantMessages([]);
        }
        await loadAssistantSessions();
      },
    });
  }, [activeAssistantSession, loadAssistantSessions]);

  const askAssistant = useCallback(async (preset?: string) => {
    const text = (preset ?? assistantInput).trim();
    if (!text || assistantStreaming) return;

    assistantAbortRef.current?.abort();
    let session = activeAssistantSession;
    if (!session) {
      try {
        session = await createAssistantSession({
          title: text.slice(0, 32) || '新对话',
          loop_id: selectedLoop?.loop_id ?? null,
        });
        setActiveAssistantSession(session);
        await loadAssistantSessions();
      } catch (error) {
        message.error(`新建对话失败：${String(error)}`);
        return;
      }
    }
    const userMessage: AssistantMessage = { id: Date.now() + Math.random(), role: 'user', text };
    const assistantId = Date.now() + Math.random() + 1;
    const assistantMessage: AssistantMessage = {
      id: assistantId,
      role: 'assistant',
      text: '',
      reasoning: '',
      eventLog: [],
      loading: true,
    };

    const nextMessages = [...assistantMessages, userMessage, assistantMessage].slice(-12);
    setAssistantMessages(nextMessages);
    setAssistantInput('');
    setAssistantStreaming(true);

    let answerBuffer = '';
    let reasoningBuffer = '';
    let flushTimer: number | undefined;
    const flushAssistantBuffers = () => {
      if (!answerBuffer && !reasoningBuffer) return;
      const answerChunk = answerBuffer;
      const reasoningChunk = reasoningBuffer;
      answerBuffer = '';
      reasoningBuffer = '';
      setAssistantMessages((prev) => prev.map((item) => {
        if (item.id !== assistantId) return item;
        return {
          ...item,
          text: `${item.text}${answerChunk}`.slice(-50000),
          reasoning: `${item.reasoning || ''}${reasoningChunk}`.slice(-20000),
        };
      }));
    };
    const scheduleAssistantFlush = () => {
      if (flushTimer !== undefined) return;
      flushTimer = window.setTimeout(() => {
        flushTimer = undefined;
        flushAssistantBuffers();
      }, 80);
    };

    assistantAbortRef.current = assistantSessionStream(
      session.id,
      {
        message: text,
        context: buildAssistantContext(),
      },
      (event) => {
        const type = String(event.type || '');
        const visibleEvent = formatAssistantEvent(event);
        if (visibleEvent) {
          setAssistantMessages((prev) => prev.map((item) => (
            item.id === assistantId
              ? { ...item, eventLog: [...(item.eventLog ?? []), visibleEvent].slice(-12) }
              : item
          )));
        }
        if (type === 'done') {
          if (flushTimer !== undefined) {
            window.clearTimeout(flushTimer);
            flushTimer = undefined;
          }
          setAssistantMessages((prev) => prev.map((item) => {
            if (item.id !== assistantId) return item;
            const combinedText = `${item.text}${answerBuffer}`.slice(-50000);
            const combinedReasoning = `${item.reasoning || ''}${reasoningBuffer}`.slice(-20000);
            answerBuffer = '';
            reasoningBuffer = '';
            const raw = combinedText.trim();
            const fallbackActions = buildDialogueActions(selectedLoop?.loop_id ?? selectedLoopId);
            if (!raw.startsWith('{')) {
              const inlineActions = raw.split('\n')
                .map((line) => normalizeAssistantAction(line, selectedLoop?.loop_id ?? selectedLoopId))
                .filter(Boolean) as AssistantAction[];
              const mergedActions = [...inlineActions, ...fallbackActions].filter((action, index, array) => (
                index === array.findIndex((next) => next.target === action.target && next.sub === action.sub && next.loopId === action.loopId)
              ));
              return { ...item, loading: false, actions: mergedActions };
            }
            try {
              const parsed = JSON.parse(raw) as {
                answer?: string;
                evidence?: string[];
                suggested_actions?: Array<{ label?: string; target_module?: string; target_sub?: string; loop_id?: string | null }>;
              };
              const evidence = Array.isArray(parsed.evidence) && parsed.evidence.length
                ? `\n\n证据：\n${parsed.evidence.map((line) => `- ${line}`).join('\n')}`
                : '';
              const actionsText = Array.isArray(parsed.suggested_actions) && parsed.suggested_actions.length
                ? `\n\n建议动作：\n${parsed.suggested_actions.map((action) => `- ${action.label || '查看详情'}`).join('\n')}`
                : '';
              const actions = Array.isArray(parsed.suggested_actions)
                ? parsed.suggested_actions
                    .map((action) => ({
                      label: action.label || '查看详情',
                      target: action.target_module as ModuleKey,
                      sub: action.target_sub as SubKey,
                      loopId: action.loop_id || undefined,
                    }))
                    .filter((action) => action.target && action.sub)
                : fallbackActions;
              const inlineActions = `${parsed.answer || ''}${actionsText}`.split('\n')
                .map((line) => normalizeAssistantAction(line, selectedLoop?.loop_id ?? selectedLoopId))
                .filter(Boolean) as AssistantAction[];
              const mergedActions = [...actions, ...inlineActions, ...fallbackActions].filter((action, index, array) => (
                index === array.findIndex((item) => item.target === action.target && item.sub === action.sub && item.loopId === action.loopId)
              ));
              return {
                ...item,
                loading: false,
                reasoning: combinedReasoning,
                text: `${parsed.answer || raw}${evidence}${actionsText}`,
                actions: mergedActions.length ? mergedActions : fallbackActions,
              };
            } catch {
              const inlineActions = combinedText.split('\n')
                .map((line) => normalizeAssistantAction(line, selectedLoop?.loop_id ?? selectedLoopId))
                .filter(Boolean) as AssistantAction[];
              const mergedActions = [...inlineActions, ...fallbackActions].filter((action, index, array) => (
                index === array.findIndex((item) => item.target === action.target && item.sub === action.sub && item.loopId === action.loopId)
              ));
              return { ...item, loading: false, text: combinedText, reasoning: combinedReasoning, actions: mergedActions };
            }
          }));
          setAssistantStreaming(false);
          assistantAbortRef.current = null;
          void openAssistantSession(session.id);
          void loadAssistantSessions();
          return;
        }
        if (type === 'error') {
          if (flushTimer !== undefined) {
            window.clearTimeout(flushTimer);
            flushTimer = undefined;
          }
          const error = String(event.message || 'AI 助手调用失败');
          setAssistantMessages((prev) => prev.map((item) => item.id === assistantId ? { ...item, loading: false, error, text: item.text || error } : item));
          setAssistantStreaming(false);
          assistantAbortRef.current = null;
          return;
        }
        const content = String(event.content || '');
        if (!content) return;
        if (type === 'thinking_step' || type === 'reasoning_delta') {
          reasoningBuffer += `${content}${type === 'thinking_step' ? '\n' : ''}`;
          scheduleAssistantFlush();
        } else if (type === 'tool_event') {
          reasoningBuffer += `已读取上下文：${String(event.name || 'tool')} (${String(event.status || 'ok')})\n`;
          scheduleAssistantFlush();
        } else if (type === 'answer_delta') {
          answerBuffer += content;
          scheduleAssistantFlush();
        }
      },
      (error) => {
        if (flushTimer !== undefined) {
          window.clearTimeout(flushTimer);
          flushTimer = undefined;
        }
        setAssistantMessages((prev) => prev.map((item) => item.id === assistantId ? { ...item, loading: false, error: error.message, text: item.text || error.message } : item));
        setAssistantStreaming(false);
        assistantAbortRef.current = null;
      },
    );
  }, [
    activeAssistantSession,
    assistantInput,
    assistantMessages,
    assistantStreaming,
    buildAssistantContext,
    buildDialogueActions,
    formatAssistantEvent,
    loadAssistantSessions,
    normalizeAssistantAction,
    openAssistantSession,
    selectedLoop,
    selectedLoopId,
  ]);

  const loadModelConfig = useCallback(async () => {
    setModelConfigLoading(true);
    try {
      const data = await fetchModelConfig();
      setModelConfig(data);
      setModelConfigTestResult(null);
    } catch {
      message.error('加载模型配置失败');
    } finally {
      setModelConfigLoading(false);
    }
  }, []);

  const saveModelConfig = useCallback(async (values: Record<string, unknown>) => {
    setModelConfigSaving(true);
    try {
      const body: Record<string, string | null> = {};
      const prevMaskedKey = modelConfig?.model_api_key || '';
      for (const k of ['model_api_url', 'model_api_key', 'model_name']) {
        const v = String(values[k] ?? '').trim();
        if (k === 'model_api_key' && v === prevMaskedKey) {
          body[k] = null;
        } else {
          body[k] = v || null;
        }
      }
      const resp = await updateModelConfig(body);
      setModelConfig(resp.config);
      modelConfigForm.setFieldsValue({
        model_api_url: resp.config.model_api_url || '',
        model_name: resp.config.model_name || '',
        model_api_key: resp.config.model_api_key || '',
      });
      setModelConfigTestResult(null);
      message.success('模型配置已保存并生效');
    } catch (e) {
      message.error(`保存失败: ${(e as Error).message}`);
    } finally {
      setModelConfigSaving(false);
    }
  }, [modelConfig, modelConfigForm]);

  const testModelConnection = useCallback(async () => {
    setModelConfigTesting(true);
    setModelConfigTestResult(null);
    try {
      const resp = await testModelConfig();
      setModelConfigTestResult(resp);
      if (resp.status === 'ok') {
        message.success('连接测试通过');
      } else {
        message.warning('连接测试失败，请检查配置');
      }
    } catch (e) {
      setModelConfigTestResult({ status: 'error', message: (e as Error).message });
      message.error('连接测试异常');
    } finally {
      setModelConfigTesting(false);
    }
  }, []);

  const loadPolicyConfig = useCallback(async () => {
    setPolicyConfigLoading(true);
    try {
      const data = await fetchPolicyConfig();
      setPolicyConfig(data);
    } catch (error) {
      message.error(`加载规则配置失败：${String(error)}`);
    } finally {
      setPolicyConfigLoading(false);
    }
  }, []);

  const applyPromptConfigToForm = useCallback((config: PromptConfig) => {
    promptConfigForm.setFieldsValue({
      assistant_system_prompt: config.assistant_system_prompt || '',
      assistant_developer_prompt: config.assistant_developer_prompt || '',
      assistant_response_schema: config.assistant_response_schema || '',
      window_policy_system_prompt: config.window_policy_system_prompt || '',
      window_policy_user_prompt_template: config.window_policy_user_prompt_template || '',
      identification_review_system_prompt: config.identification_review_system_prompt || '',
      identification_review_user_prompt_template: config.identification_review_user_prompt_template || '',
      consultant_system_prompt: config.consultant_system_prompt || '',
    });
  }, [promptConfigForm]);

  const loadPromptConfig = useCallback(async () => {
    setPromptConfigLoading(true);
    try {
      const data = await fetchPromptConfig();
      setPromptConfig(data);
      applyPromptConfigToForm(data);
    } catch (error) {
      message.error(`加载提示词配置失败：${String(error)}`);
    } finally {
      setPromptConfigLoading(false);
    }
  }, [applyPromptConfigToForm]);

  const savePromptConfig = useCallback(async () => {
    setPromptConfigSaving(true);
    try {
      const values = promptConfigForm.getFieldsValue(true) as Record<string, unknown>;
      const resp = await updatePromptConfig({
        assistant_system_prompt: String(values.assistant_system_prompt ?? '').trim(),
        assistant_developer_prompt: String(values.assistant_developer_prompt ?? '').trim(),
        assistant_response_schema: String(values.assistant_response_schema ?? '').trim(),
        window_policy_system_prompt: String(values.window_policy_system_prompt ?? '').trim(),
        window_policy_user_prompt_template: String(values.window_policy_user_prompt_template ?? '').trim(),
        identification_review_system_prompt: String(values.identification_review_system_prompt ?? '').trim(),
        identification_review_user_prompt_template: String(values.identification_review_user_prompt_template ?? '').trim(),
        consultant_system_prompt: String(values.consultant_system_prompt ?? '').trim(),
      });
      setPromptConfig(resp.config);
      applyPromptConfigToForm(resp.config);
      message.success('提示词配置已保存');
    } catch (error) {
      message.error(`保存提示词配置失败：${String(error)}`);
    } finally {
      setPromptConfigSaving(false);
    }
  }, [applyPromptConfigToForm, promptConfigForm]);

  const restoreDefaultPromptConfig = useCallback(async () => {
    setPromptConfigSaving(true);
    try {
      const resp = await resetPromptConfig();
      setPromptConfig(resp.config);
      applyPromptConfigToForm(resp.config);
      message.success('已恢复默认提示词');
    } catch (error) {
      message.error(`恢复默认提示词失败：${String(error)}`);
    } finally {
      setPromptConfigSaving(false);
    }
  }, [applyPromptConfigToForm]);

  const toggleModule = (moduleKey: ModuleKey) => {
    setExpandedModules((prev) => ({ ...prev, [moduleKey]: !prev[moduleKey] }));
  };

  const addAssetChild = () => {
    const parent = selectedAssetNode;
    const name = assetDraftName.trim();
    if (!parent || !name) {
      message.warning('请先选择父节点并输入节点名称');
      return;
    }
    const node: AssetNode = {
      id: `asset_${Date.now()}`,
      parentId: parent.id,
      name,
      type: assetDraftType || nextAssetType(parent.type),
    };
    setAssetNodes((prev) => [...prev, node]);
    setSelectedAssetNodeId(node.id);
    setAssetDraftName('');
    message.success(`已新增节点：${name}`);
  };

  const renameAssetNode = () => {
    const name = assetRenameValue.trim();
    if (!selectedAssetNode || !name) {
      message.warning('请输入新的节点名称');
      return;
    }
    setAssetNodes((prev) => prev.map((node) => (
      node.id === selectedAssetNode.id ? { ...node, name } : node
    )));
    setAssetRenameValue('');
    message.success('节点已重命名');
  };

  const deleteAssetNode = () => {
    if (!selectedAssetNode || selectedAssetNode.id === 'factory') {
      message.warning('根节点不能删除');
      return;
    }
    const hasChild = assetNodes.some((node) => node.parentId === selectedAssetNode.id);
    const hasLoop = loops.some((loop) => inferLoopAssetId(loop.loop_id) === selectedAssetNode.id);
    if (hasChild || hasLoop) {
      message.warning('该节点存在子节点或挂载回路，第一版请先清空后再删除');
      return;
    }
    setAssetNodes((prev) => prev.filter((node) => node.id !== selectedAssetNode.id));
    setSelectedAssetNodeId(selectedAssetNode.parentId ?? 'factory');
    message.success('节点已删除');
  };

  const loadLoops = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await listHistoryLoops();
      setLoops(resp.items);
      setSelectedLoopId((current) => current ?? resp.items[0]?.loop_id);
    } catch (error) {
      message.error(`加载历史回路失败：${String(error)}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const buildTrendSeriesParams = useCallback((loop?: HistoryLoop) => {
    const params: { start_time?: string; end_time?: string; max_points?: number } = {
      max_points: trendPointLimit === 'all' ? 0 : Number(trendPointLimit),
    };
    if (trendPreset === 'custom') {
      const [start, end] = trendCustomRange ?? [];
      if (start && end) {
        params.start_time = start.format('YYYY-MM-DD HH:mm:ss');
        params.end_time = end.format('YYYY-MM-DD HH:mm:ss');
      }
      return params;
    }
    if (trendPreset === 'all') return params;
    const preset = TREND_PRESET_OPTIONS.find((item) => item.value === trendPreset);
    if (!preset?.seconds) return params;
    const end = dayjs(loop?.end_time || undefined);
    const safeEnd = end.isValid() ? end : dayjs();
    params.start_time = safeEnd.subtract(preset.seconds, 'second').format('YYYY-MM-DD HH:mm:ss');
    params.end_time = safeEnd.format('YYYY-MM-DD HH:mm:ss');
    return params;
  }, [trendCustomRange, trendPointLimit, trendPreset]);

  const buildFeatureRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    const params: HistoryTimeRangeParams = {};
    if (featureRangePreset === 'custom') {
      const [start, end] = featureCustomRange ?? [];
      if (start && end) {
        params.start_time = start.format('YYYY-MM-DD HH:mm:ss');
        params.end_time = end.format('YYYY-MM-DD HH:mm:ss');
      }
      return params;
    }
    if (featureRangePreset === 'all') return params;
    const preset = FEATURE_RANGE_OPTIONS.find((item) => item.value === featureRangePreset);
    if (!preset?.seconds) return params;
    const end = dayjs(loop?.end_time || undefined);
    const safeEnd = end.isValid() ? end : dayjs();
    params.start_time = safeEnd.subtract(preset.seconds, 'second').format('YYYY-MM-DD HH:mm:ss');
    params.end_time = safeEnd.format('YYYY-MM-DD HH:mm:ss');
    return params;
  }, [featureCustomRange, featureRangePreset]);

  const buildWindowRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    const params: HistoryTimeRangeParams = {};
    if (windowRangePreset === 'custom') {
      const [start, end] = windowCustomRange ?? [];
      if (start && end) {
        params.start_time = start.format('YYYY-MM-DD HH:mm:ss');
        params.end_time = end.format('YYYY-MM-DD HH:mm:ss');
      }
      return params;
    }
    if (windowRangePreset === 'all') return params;
    const preset = FEATURE_RANGE_OPTIONS.find((item) => item.value === windowRangePreset);
    if (!preset?.seconds) return params;
    const end = dayjs(loop?.end_time || undefined);
    const safeEnd = end.isValid() ? end : dayjs();
    params.start_time = safeEnd.subtract(preset.seconds, 'second').format('YYYY-MM-DD HH:mm:ss');
    params.end_time = safeEnd.format('YYYY-MM-DD HH:mm:ss');
    return params;
  }, [windowCustomRange, windowRangePreset]);

  // 整定任务页独立的时间窗参数构造函数；逻辑与 buildWindowRangeParams 同模板，
  // 但读取 tuningRangePreset / tuningCustomRange，避免与窗口候选页耦合。
  const buildTuningRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    const params: HistoryTimeRangeParams = {};
    if (tuningRangePreset === 'custom') {
      const [start, end] = tuningCustomRange ?? [];
      if (start && end) {
        params.start_time = start.format('YYYY-MM-DD HH:mm:ss');
        params.end_time = end.format('YYYY-MM-DD HH:mm:ss');
      }
      return params;
    }
    if (tuningRangePreset === 'all') return params;
    const preset = FEATURE_RANGE_OPTIONS.find((item) => item.value === tuningRangePreset);
    if (!preset?.seconds) return params;
    const end = dayjs(loop?.end_time || undefined);
    const safeEnd = end.isValid() ? end : dayjs();
    params.start_time = safeEnd.subtract(preset.seconds, 'second').format('YYYY-MM-DD HH:mm:ss');
    params.end_time = safeEnd.format('YYYY-MM-DD HH:mm:ss');
    return params;
  }, [tuningCustomRange, tuningRangePreset]);

  const buildTuningPriorRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    const params: HistoryTimeRangeParams = {};
    if (tuningPriorRangePreset === 'custom') {
      const [start, end] = tuningPriorCustomRange ?? [];
      if (start && end) {
        params.start_time = start.format('YYYY-MM-DD HH:mm:ss');
        params.end_time = end.format('YYYY-MM-DD HH:mm:ss');
      }
      return params;
    }
    if (tuningPriorRangePreset === 'all') return params;
    const preset = FEATURE_RANGE_OPTIONS.find((item) => item.value === tuningPriorRangePreset);
    if (!preset?.seconds) return params;
    const end = dayjs(loop?.end_time || undefined);
    const safeEnd = end.isValid() ? end : dayjs();
    params.start_time = safeEnd.subtract(preset.seconds, 'second').format('YYYY-MM-DD HH:mm:ss');
    params.end_time = safeEnd.format('YYYY-MM-DD HH:mm:ss');
    return params;
  }, [tuningPriorCustomRange, tuningPriorRangePreset]);

  const loadSeries = useCallback(async (loopId: string, loop?: HistoryLoop) => {
    setSeries(null);
    setSeriesLoading(true);
    try {
      const resp = await getHistoryLoopSeries(loopId, buildTrendSeriesParams(loop));
      if (resp.error) message.warning(resp.error);
      setSeries(resp);
    } catch (error) {
      message.error(`加载趋势失败：${String(error)}`);
    } finally {
      setSeriesLoading(false);
    }
  }, [buildTrendSeriesParams]);

  const loadAssessment = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setAssessment(null);
    setAssessmentError(null);
    setAssessmentLoading(true);
    try {
      const resp = await getHistoryLoopAssessment(loopId, params);
      if (resp.error) message.warning(resp.error);
      setAssessment(resp);
    } catch (error) {
      setAssessmentError(String(error));
      message.error(`加载回路评估失败：${String(error)}`);
    } finally {
      setAssessmentLoading(false);
    }
  }, []);

  const loadTuningPriorCore = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setTuningPriorCoreData(null);
    setTuningPriorReviewData(null);
    setTuningPriorCoreError(null);
    setTuningPriorCoreLoading(true);
    try {
      const resp = await getHistoryLoopTuningPriorCore(loopId, params);
      if (resp.error) message.warning(resp.error);
      setTuningPriorCoreData(resp);
    } catch (error) {
      const text = String(error);
      setTuningPriorCoreError(text);
      message.error(`加载核心指标失败：${text}`);
    } finally {
      setTuningPriorCoreLoading(false);
    }
  }, []);

  const loadTuningPriorOntology = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setTuningPriorOntologyData(null);
    setTuningPriorReviewData(null);
    setTuningPriorOntologyError(null);
    setTuningPriorOntologyLoading(true);
    try {
      const resp = await getHistoryLoopTuningPriorOntology(loopId, params);
      if (resp.error) message.warning(resp.error);
      setTuningPriorOntologyData(resp);
    } catch (error) {
      const text = String(error);
      setTuningPriorOntologyError(text);
      message.error(`查询本体失败：${text}`);
    } finally {
      setTuningPriorOntologyLoading(false);
    }
  }, []);

  const loadTuningPriorReview = useCallback(async (loopId: string) => {
    if (!tuningPriorCoreData?.core_context) {
      message.warning('请先生成核心指标与评估诊断上下文');
      return;
    }
    setTuningPriorReviewData(null);
    setTuningPriorReviewError(null);
    setTuningPriorReviewLoading(true);
    try {
      const resp = await reviewHistoryLoopTuningPrior(loopId, {
        core_context: tuningPriorCoreData.core_context,
        ontology: tuningPriorOntologyData?.ontology ?? null,
      });
      if (resp.error) message.warning(resp.error);
      setTuningPriorReviewData(resp);
    } catch (error) {
      const text = String(error);
      setTuningPriorReviewError(text);
      message.error(`生成大模型先验评审失败：${text}`);
    } finally {
      setTuningPriorReviewLoading(false);
    }
  }, [tuningPriorCoreData, tuningPriorOntologyData]);

  const loadLoopFeatures = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setLoopFeatures(null);
    setFeatureLoading(true);
    try {
      const resp = await fetchHistoryLoopFeatures(loopId, params);
      if ((resp as { error?: string }).error) {
        message.warning((resp as { error?: string }).error);
        setLoopFeatures(null);
        return;
      }
      setLoopFeatures(resp);
    } catch (error) {
      message.error(`加载回路画像失败：${String(error)}`);
    } finally {
      setFeatureLoading(false);
    }
  }, []);

  const loadLoopMonitoring = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setLoopMonitoring(null);
    try {
      const resp = await fetchHistoryLoopMonitoring(loopId, params);
      if ((resp as { error?: string }).error) {
        message.warning((resp as { error?: string }).error);
        setLoopMonitoring(null);
        return;
      }
      setLoopMonitoring(resp);
      if (!params?.start_time && !params?.end_time) {
        setMonitoringByLoopId((prev) => ({ ...prev, [loopId]: resp }));
      }
      setLoopFeatures(resp.features ?? null);
    } catch (error) {
      message.error(`加载监控快照失败：${String(error)}`);
    }
  }, []);

  const loadWindows = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setWindows([]);
    setWindowAlgorithmSummary({});
    setSelectedWindowIndex(undefined);
    try {
      const resp = await getHistoryLoopWindows(loopId, params);
      if (resp.error) message.warning(resp.error);
      setWindows(resp.windows ?? []);
      setWindowAlgorithmSummary(resp.algorithm_summary ?? {});
      const firstUsable = resp.windows?.find((item) => item.usable) ?? resp.windows?.[0];
      setSelectedWindowIndex(firstUsable?.index);
    } catch (error) {
      message.error(`加载辨识窗口失败：${String(error)}`);
    }
  }, []);

  useEffect(() => {
    loadLoops();
  }, [loadLoops]);

  useEffect(() => {
    if (viewMode !== 'dialogue') return;
    loadAssistantSessions();
  }, [loadAssistantSessions, viewMode]);

  useEffect(() => {
    if (!selectedLoopId || isSettingsView) return;
    const shouldLoadDashboardTrend = activeSub === 'dashboard' && enabledDashboardWidgets.has('trend');
    if (activeSub !== 'trend_spectrum' && !shouldLoadDashboardTrend) return;
    loadSeries(selectedLoopId, selectedLoop);
  }, [activeSub, enabledDashboardWidgets, isSettingsView, loadSeries, selectedLoop, selectedLoopId]);

  useEffect(() => {
    if (!selectedLoopId || isSettingsView) return;
    if (!shouldLoadAssessmentDetail && !shouldLoadWindowDetail) return;
    if (shouldLoadAssessmentDetail) {
      const params = activeSub === 'tuning_task' ? buildTuningRangeParams(selectedLoop) : undefined;
      loadAssessment(selectedLoopId, params);
    }
    if (shouldLoadWindowDetail) loadWindows(selectedLoopId);
  }, [
    activeSub,
    buildTuningRangeParams,
    isSettingsView,
    loadAssessment,
    loadWindows,
    selectedLoopId,
    shouldLoadAssessmentDetail,
    shouldLoadWindowDetail,
  ]);

  useEffect(() => {
    if (!selectedLoopId || isSettingsView) return;
    if (!shouldLoadFeatureDetail && !shouldLoadMonitoringDetail) return;
    const featureParams = buildFeatureRangeParams(selectedLoop);
    if (shouldLoadMonitoringDetail) {
      loadLoopMonitoring(selectedLoopId, featureParams);
      return;
    }
    loadLoopFeatures(selectedLoopId, featureParams);
  }, [
    buildFeatureRangeParams,
    isSettingsView,
    loadLoopFeatures,
    loadLoopMonitoring,
    selectedLoop,
    selectedLoopId,
    shouldLoadFeatureDetail,
    shouldLoadMonitoringDetail,
  ]);

  useEffect(() => {
    if (activeSub !== 'tuning_prior') return;
    setTuningPriorCoreData(null);
    setTuningPriorOntologyData(null);
    setTuningPriorReviewData(null);
    setTuningPriorCoreError(null);
    setTuningPriorOntologyError(null);
    setTuningPriorReviewError(null);
  }, [activeSub, selectedLoopId, tuningPriorRangePreset, tuningPriorCustomRange]);

  useEffect(() => {
    if (!scopedLoops.length) return;
    if (!selectedLoopId || !scopedLoops.some((loop) => loop.loop_id === selectedLoopId)) {
      setSelectedLoopId(scopedLoops[0].loop_id);
    }
  }, [scopedLoops, selectedLoopId]);

  useEffect(() => {
    if (activeSub !== 'dashboard' || !dashboardWorstLoopId) return;
    const selectionKey = `${selectedAssetNodeId}:${dashboardWorstLoopId}`;
    if (dashboardWorstSelectionRef.current === selectionKey) return;
    dashboardWorstSelectionRef.current = selectionKey;
    setSelectedLoopId(dashboardWorstLoopId);
  }, [activeSub, dashboardWorstLoopId, selectedAssetNodeId]);

  useEffect(() => {
    if (!shouldLoadDashboardMonitoring) return undefined;
    let cancelled = false;
    const missing = scopedLoops
      .map((loop) => loop.loop_id)
      .filter((loopId) => !monitoringByLoopId[loopId] && !monitoringBulkInFlightRef.current.has(loopId));
    if (!missing.length) return undefined;

    missing.forEach((loopId) => monitoringBulkInFlightRef.current.add(loopId));

    const loadMissingMonitoring = async () => {
      for (let index = 0; index < missing.length && !cancelled; index += 2) {
        const batch = missing.slice(index, index + 2);
        const results = await Promise.allSettled(batch.map((loopId) => fetchHistoryLoopMonitoring(loopId)));
        if (cancelled) break;
        setMonitoringByLoopId((prev) => {
          const next = { ...prev };
          results.forEach((result) => {
            if (result.status === 'fulfilled') {
              next[result.value.loop_id] = result.value;
            }
          });
          return next;
        });
        batch.forEach((loopId) => monitoringBulkInFlightRef.current.delete(loopId));
      }
    };

    const timer = window.setTimeout(() => {
      void loadMissingMonitoring().finally(() => {
        missing.forEach((loopId) => monitoringBulkInFlightRef.current.delete(loopId));
      });
    }, 800);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      missing.forEach((loopId) => monitoringBulkInFlightRef.current.delete(loopId));
    };
  }, [monitoringByLoopId, scopedLoops, shouldLoadDashboardMonitoring]);

  useEffect(() => {
    if (activeSub === 'model_config' && !modelConfig) {
      loadModelConfig();
    }
  }, [activeSub, modelConfig, loadModelConfig]);

  useEffect(() => {
    if (activeSub === 'rule_config' && !policyConfig) {
      loadPolicyConfig();
    }
  }, [activeSub, policyConfig, loadPolicyConfig]);

  useEffect(() => {
    if (activeSub === 'prompt_config' && !promptConfig) {
      loadPromptConfig();
    }
  }, [activeSub, promptConfig, loadPromptConfig]);

  useEffect(() => {
    if (modelConfig) {
      modelConfigForm.setFieldsValue({
        model_api_url: modelConfig.model_api_url || '',
        model_name: modelConfig.model_name || '',
        model_api_key: modelConfig.model_api_key || '',
      });
    }
  }, [modelConfig, modelConfigForm]);

  const handleImport = async () => {
    const files = fileList.map((item) => item.originFileObj).filter(Boolean) as File[];
    if (!files.length) {
      message.warning('请先选择历史数据文件');
      return;
    }
    setImporting(true);
    try {
      const resp = await importHistoryFiles(files);
      message.success(`导入 ${resp.imported_count} 个回路`);
      if (resp.errors.length) message.warning(`${resp.errors.length} 个文件导入失败，请检查格式`);
      setFileList([]);
      await loadLoops();
      setSelectedLoopId(resp.loops[0]?.loop_id);
    } catch (error) {
      message.error(`导入失败：${String(error)}`);
    } finally {
      setImporting(false);
    }
  };

  const startTune = (options?: {
    /** 是否把当前 selectedWindowIndex 作为"工程师手动指定窗口"传给后端。
     *  传 true 时后端走 user_override 模式，会绕过本体策略 + LLM 选窗顾问。
     *  默认 false：让后端按本体策略 + LLM 顾问自主选窗。
     *  注意：如果设 true，必须确认 selectedWindowIndex 是用户在本页面"手动确认"
     *  过的，而不是某次 loadWindows 自动设的默认值，否则会污染整定决策。 */
    useSelectedWindow?: boolean;
    /** 是否启用 LLM 顾问（覆盖默认 true）。 */
    useLlmAdvisor?: boolean;
    /** 提前结束流水线。 */
    stopAfter?: 'window_selection' | 'identification';
    /** 时间范围参数；不传时由 activeSub 决定走哪套默认。 */
    timeRange?: HistoryTimeRangeParams;
  }) => {
    if (!selectedLoop) {
      message.warning('请先选择一个回路');
      return;
    }
    const timeScope = options?.timeRange
      ?? (activeSub === 'id_windows' ? buildWindowRangeParams(selectedLoop) : {});
    const useLlm = options?.useLlmAdvisor ?? true;
    const includeWindow = options?.useSelectedWindow === true;
    setRunning(true);
    setTaskStatus('running');
    setTaskStartedAt(new Date().toLocaleString());
    setTaskId(undefined);
    setTaskCurrentStage(undefined);
    setTaskStageStatus({});
    setTaskStageData({});
    setTaskStageRunningData({});
    setTaskWindowSelection(null);
    setTaskModelReview(null);
    setTaskRefinements([]);
    setTaskThinking([]);
    setTaskAttempts([]);
    setSelectedFitAttemptKey(undefined);
    setTaskResult(null);
    setTaskError(undefined);
    setEvents([]);
    setRawLogExpanded(false);
    taskAbort?.abort();
    const controller = tuneHistoryLoopStream(
      selectedLoop.loop_id,
      {
        loop_type: selectedLoop.loop_type === 'unknown' ? 'flow' : selectedLoop.loop_type,
        loop_name: selectedLoop.loop_id,
        selected_window_index: includeWindow ? selectedWindowIndex : undefined,
        use_llm_advisor: useLlm,
        stop_after: options?.stopAfter,
        start_time: timeScope.start_time,
        end_time: timeScope.end_time,
      },
      (event) => {
        const e = event as unknown as PipelineEvent;
        setEvents((prev) => prependTaskEventLog(prev, e));

        if (e.type === 'session_start') {
          setTaskId(e.task_id);
          return;
        }

        if (e.type === 'stage') {
          setTaskCurrentStage(e.stage);
          setTaskStageStatus((prev) => ({ ...prev, [e.stage]: e.status }));
          if (e.status === 'running') {
            setTaskStageRunningData((prev) => mergeRunningStageData(prev, e.stage, e.data));
          }
          if (e.status === 'done') {
            // done 之后清掉 running 子状态，让 stepStatus 不再误判为"运行中"。
            setTaskStageRunningData((prev) => clearRunningStageData(prev, e.stage));
          }
          if (e.status === 'done' && e.data) {
            setTaskStageData((prev) => mergeDoneStageData(prev, e.stage, e.data));
            if (e.stage === 'window_selection') {
              setTaskWindowSelection(e.data as unknown as WindowSelectionMeta);
            } else if (e.stage === 'model_review') {
              setTaskModelReview(e.data as unknown as ModelReviewMeta);
            } else if (e.stage === 'identification_refinement') {
              const nextRefinement = e.data as unknown as IdentificationRefinementMeta;
              setTaskRefinements((prev) => upsertRefinement(prev, nextRefinement));
            } else if (e.stage === 'identification') {
              setTaskAttempts((prev) => mergeIdentificationAttempts(prev, e.data));
            }
          }
          return;
        }

        if (e.type === 'llm_thinking') {
          setTaskThinking((prev) => upsertThinkingEvent(prev, e));
          return;
        }

        if (e.type === 'result') {
          setTaskResult(e.data);
          setTaskStatus('done');
          return;
        }

        if (e.type === 'error') {
          setRunning(false);
          setTaskStatus('error');
          setTaskError(e.message);
          return;
        }

        if (e.type === 'done') {
          setRunning(false);
          setTaskStatus((prev) => prev === 'error' ? 'error' : 'done');
          setTaskAbort(null);
        }
      },
    );
    setTaskAbort(controller);
  };

  const handleTune = () => {
    if (!selectedLoop) {
      message.warning('请先选择一个回路');
      return;
    }
    // 整定任务页的固定 startTune 选项：用本页的时间窗 + LLM 开关，
    // 同时禁止携带 selectedWindowIndex（避免误触发后端 user_override）。
    const tuningOptions = {
      timeRange: buildTuningRangeParams(selectedLoop),
      useLlmAdvisor: tuningUseLlm,
      useSelectedWindow: false as const,
    };
    if (tuningGate.hardBlocked) {
      Modal.warning({
        title: '当前回路暂不建议发起整定',
        content: (
          <Space direction="vertical" size={8}>
            <Typography.Text>准入校验存在阻断项，请先处理数据质量、工况或约束问题。</Typography.Text>
            {tuningGate.blockingReasons.slice(0, 3).map((item, index) => (
              <Typography.Text key={`${item.type}-${index}`} type="secondary">
                {index + 1}. {item.message}
              </Typography.Text>
            ))}
          </Space>
        ),
      });
      return;
    }
    if (!assessment || tuningGate.caution) {
      Modal.confirm({
        title: assessment ? '当前回路建议谨慎整定' : '尚未拿到整定准入评估',
        content: (
          <Space direction="vertical" size={8}>
            <Typography.Text>
              {assessment
                ? (tuningGate.nextAction || '建议确认当前数据片段代表目标工况后再发起整定。')
                : '系统还没有加载到准入评估结果，继续发起会直接进入辨识和整定流程。'}
            </Typography.Text>
            {tuningGate.blockingReasons.slice(0, 3).map((item, index) => (
              <Typography.Text key={`${item.type}-${index}`} type="secondary">
                {index + 1}. {item.message}
              </Typography.Text>
            ))}
          </Space>
        ),
        okText: '确认发起',
        cancelText: '先不发起',
        onOk: () => startTune(tuningOptions),
      });
      return;
    }
    startTune(tuningOptions);
  };

  const handleStopTune = () => {
    taskAbort?.abort();
    setTaskAbort(null);
    setRunning(false);
    setTaskStatus('error');
    setTaskError('任务已由用户停止');
  };

  const trendData = useMemo(() => {
    if (!series?.points?.length) return [];
    const rows: Array<{ t: string | number; value: number; series: string }> = [];
    series.points.forEach((point) => {
      rows.push({ t: point.t, value: point.pv, series: 'PV' });
      rows.push({ t: point.t, value: point.mv, series: 'MV' });
      if (point.sv !== null && point.sv !== undefined) rows.push({ t: point.t, value: point.sv, series: 'SV' });
    });
    return rows;
  }, [series]);

  const pvTrendData = useMemo(
    () => trendData.filter((item) => item.series === 'PV' || item.series === 'SV'),
    [trendData],
  );

  const mvTrendData = useMemo(
    () => trendData.filter((item) => item.series === 'MV'),
    [trendData],
  );

  const windowPreviewData = useMemo(() => {
    if (!selectedWindow?.preview?.length) return [];
    const rows: Array<{ t: string | number; value: number; series: string }> = [];
    selectedWindow.preview.forEach((point) => {
      rows.push({ t: point.t, value: point.pv, series: 'PV' });
      rows.push({ t: point.t, value: point.mv, series: 'MV' });
    });
    return rows;
  }, [selectedWindow]);

  const loopColumns = [
    {
      title: '回路',
      dataIndex: 'loop_id',
      render: (value: string, row: HistoryLoop) => (
        <Space>
          <Typography.Text strong>{value}</Typography.Text>
          <Tag color="blue">{LOOP_TYPE_LABEL[row.loop_type] ?? row.loop_type}</Tag>
        </Space>
      ),
    },
    { title: '来源文件', dataIndex: 'source_filename', ellipsis: true },
    { title: '采样', dataIndex: 'sampling_time', render: (value: number) => `${value}s` },
    { title: '点数', dataIndex: 'rows' },
    // 候选窗口 / 最佳窗口分需要在「窗口候选」页面主动评估后才有；这里展示
    // 数据时长和 PV 范围作为加载后立即可见的画像指标。
    { title: '时长', render: (_: unknown, row: HistoryLoop) => {
      const start = row.start_time ? new Date(row.start_time).getTime() : NaN;
      const end = row.end_time ? new Date(row.end_time).getTime() : NaN;
      if (Number.isNaN(start) || Number.isNaN(end) || end <= start) return '-';
      const h = (end - start) / 3_600_000;
      return h >= 1 ? `${h.toFixed(1)} h` : `${Math.round(h * 60)} min`;
    } },
    { title: 'PV 范围', render: (_: unknown, row: HistoryLoop) => {
      const a = row.pv_min, b = row.pv_max;
      return (typeof a === 'number' && typeof b === 'number') ? `${a.toFixed(2)} ~ ${b.toFixed(2)}` : '-';
    } },
  ];

  const windowColumns = [
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
        const b = row.score_breakdown ?? {};
        return `MV ${scorePercent(b.mv_excitation)}/PV ${scorePercent(b.pv_response)}/相关 ${scorePercent(b.lag_correlation)}`;
      },
    },
    { title: '相关性', dataIndex: 'corr', render: (value: number) => value.toFixed(3) },
    { title: 'MV 幅度', dataIndex: 'mv_span' },
    { title: 'PV 幅度', dataIndex: 'pv_span' },
    { title: '点数', dataIndex: 'n_points' },
  ];

  const renderLoopTable = () => (
    <Table
      rowKey="loop_id"
      columns={loopColumns}
      dataSource={scopedLoops}
      loading={loading}
      pagination={{ pageSize: 8 }}
      rowSelection={{
        type: 'radio',
        selectedRowKeys: selectedLoopId ? [selectedLoopId] : [],
        onChange: (keys) => setSelectedLoopId(String(keys[0])),
      }}
      onRow={(record) => ({ onClick: () => setSelectedLoopId(record.loop_id) })}
    />
  );

  const renderTrendLine = (
    data: Array<{ t: string | number; value: number; series: string }>,
    height: number,
    _yTitle: string,
    colors: string[],
  ) => (
    <div className="chart-shell">
      <Line
        height={height}
        data={data}
        xField="t"
        yField="value"
        colorField="series"
        theme="classic"
        color={colors}
        scale={{ color: { range: colors } }}
        style={{ lineWidth: 2.1 }}
        padding={[28, 28, 58, 58]}
        axis={{
          x: {
            title: '',
            titleFill: '#334155',
            titleFontSize: 12,
            titleFontWeight: 700,
            labelFill: '#334155',
            labelFontSize: 11,
            labelAutoHide: true,
            labelAutoRotate: true,
            lineStroke: '#cbd5e1',
            tickStroke: '#cbd5e1',
          },
          y: {
            title: '',
            titleFill: '#334155',
            titleFontSize: 12,
            titleFontWeight: 700,
            labelFill: '#334155',
            labelFontSize: 12,
            lineStroke: '#cbd5e1',
            tickStroke: '#cbd5e1',
            gridStroke: '#d8e2ee',
            gridLineDash: [4, 4],
          },
        }}
        legend={{
          color: {
            position: 'top',
            itemLabelFill: '#334155',
            itemLabelFontSize: 13,
            itemLabelFontWeight: 600,
            markerSize: 10,
          },
        }}
        slider={{
          height: 28,
          textStyle: { fill: '#64748b' },
          trendCfg: { lineStyle: { stroke: colors[0] ?? '#35a7ff' } },
          handlerStyle: { fill: '#ffffff', stroke: '#7fb8ff' },
        }}
        tooltip={chartLineTooltip}
      />
    </div>
  );

  const renderTrend = (height = 360) => (
    trendData.length ? (
      <>
        <div className="chart-axis-note">
          <span>X 轴：时间 / 采样点</span>
          <span>{trendSplitYAxis ? '分轴：上图 PV/SV，下图 MV，各自坐标' : 'Y 轴：PV / SV / MV 数值'}</span>
        </div>
        {trendSplitYAxis ? (
          <div className="split-trend-grid">
            <div className="split-trend-panel">
              <div className="split-trend-title">PV / SV 趋势</div>
              {renderTrendLine(pvTrendData, Math.max(260, Math.floor(height * 0.58)), 'Y 轴：PV / SV 数值', ['#35a7ff', '#ff9f43'])}
            </div>
            <div className="split-trend-panel">
              <div className="split-trend-title">MV 趋势</div>
              {renderTrendLine(mvTrendData, Math.max(220, Math.floor(height * 0.46)), 'Y 轴：MV 数值', ['#28d7c5'])}
            </div>
          </div>
        ) : (
        <div className="chart-shell">
          <Line
            height={height}
            data={trendData}
            xField="t"
            yField="value"
            colorField="series"
            theme="classic"
            color={['#35a7ff', '#28d7c5', '#ff9f43']}
            scale={{ color: { range: ['#35a7ff', '#28d7c5', '#ff9f43'] } }}
            style={{ lineWidth: 2.1 }}
            padding={[28, 28, 58, 58]}
            axis={{
              x: {
                title: '',
                titleFill: '#334155',
                titleFontSize: 12,
                titleFontWeight: 700,
                labelFill: '#334155',
                labelFontSize: 11,
                labelAutoHide: true,
                labelAutoRotate: true,
                lineStroke: '#cbd5e1',
                tickStroke: '#cbd5e1',
              },
              y: {
                title: '',
                titleFill: '#334155',
                titleFontSize: 12,
                titleFontWeight: 700,
                labelFill: '#334155',
                labelFontSize: 12,
                lineStroke: '#cbd5e1',
                tickStroke: '#cbd5e1',
                gridStroke: '#d8e2ee',
                gridLineDash: [4, 4],
              },
            }}
            legend={{
              color: {
                position: 'top',
                itemLabelFill: '#334155',
                itemLabelFontSize: 13,
                itemLabelFontWeight: 600,
                markerSize: 10,
              },
            }}
            slider={{
              height: 28,
              textStyle: { fill: '#64748b' },
              trendCfg: { lineStyle: { stroke: '#35a7ff' } },
              handlerStyle: { fill: '#ffffff', stroke: '#7fb8ff' },
            }}
            xAxis={{
              type: series?.x_axis === 'timestamp' ? 'timeCat' : 'linear',
              title: { text: '', style: { fill: '#334155', fontSize: 12, fontWeight: 700 } },
              label: { autoHide: true, autoRotate: true, style: { fill: '#334155', fontSize: 11, fontWeight: 600 } },
              line: { style: { stroke: '#cbd5e1' } },
              tickLine: { style: { stroke: '#cbd5e1' } },
            }}
            yAxis={{
              title: { text: '', style: { fill: '#334155', fontSize: 12, fontWeight: 700 } },
              label: { style: { fill: '#334155', fontSize: 12, fontWeight: 600 } },
              line: { style: { stroke: '#cbd5e1' } },
              tickLine: { style: { stroke: '#cbd5e1' } },
              grid: { line: { style: { stroke: '#d8e2ee', lineDash: [4, 4] } } },
            }}
            tooltip={chartLineTooltip}
          />
        </div>
        )}
      </>
    ) : <Empty description="暂无趋势数据" />
  );

  const renderAssessmentCards = () => (
    assessment ? (
      <>
      <div className="score-grid">
        <div className="score-card">
          <Tag color={tagColor(assessment.performance?.level ?? assessment.readiness.level)}>
            {assessment.summary?.decision_text ?? assessment.performance?.level ?? '-'}
          </Tag>
          <div className="score-title">综合评估</div>
          <Progress
            percent={scorePercent(assessment.performance?.score ?? assessment.readiness.score)}
            status={scoreStatus(assessment.performance?.score ?? assessment.readiness.score)}
          />
        </div>
        <div className="score-card">
          <Tag color={tagColor(assessment.data_quality.level)}>{assessment.data_quality.level}</Tag>
          <div className="score-title">数据质量</div>
          <Progress percent={scorePercent(assessment.data_quality.score)} status={scoreStatus(assessment.data_quality.score)} />
        </div>
        <div className="score-card">
          <Tag color={tagColor(assessment.identifiability.level)}>{assessment.identifiability.level}</Tag>
          <div className="score-title">可辨识性</div>
          <Progress percent={scorePercent(assessment.identifiability.score)} status={scoreStatus(assessment.identifiability.score)} />
        </div>
        <div className="score-card">
          <Tag color={tagColor(assessment.readiness.level)}>{assessment.readiness.level}</Tag>
          <div className="score-title">整定就绪度</div>
          <Progress percent={scorePercent(assessment.readiness.score)} status={scoreStatus(assessment.readiness.score)} />
        </div>
      </div>
      {assessment.summary && (
        <Alert
          className="agent-alert"
          type={assessment.summary.decision === 'blocked' ? 'error' : assessment.summary.decision === 'ready' ? 'success' : 'warning'}
          showIcon
          message={assessment.summary.decision_text}
          description={assessment.summary.recommended_next_action_text}
        />
      )}
      </>
    ) : <Empty description="暂无评估结果" />
  );

  const renderWindowTable = () => (
    <Table
      size="small"
      rowKey="index"
      dataSource={windows}
      columns={windowColumns}
      pagination={{ pageSize: 6 }}
      rowSelection={{
        type: 'radio',
        selectedRowKeys: selectedWindowIndex === undefined ? [] : [selectedWindowIndex],
        onChange: (keys) => setSelectedWindowIndex(Number(keys[0])),
      }}
      onRow={(record) => ({ onClick: () => setSelectedWindowIndex(record.index) })}
    />
  );

  const renderWindowPolicyTables = (policy?: WindowSelectionPolicy) => {
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
          <Descriptions.Item label="策略来源">{policy.ontology_facts?.source ?? '-'}</Descriptions.Item>
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
                return consumers || usage.note || '-';
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
  };

  const renderTaskDashboard = () => (
    <TuningTaskDashboard
      taskStatus={taskStatus}
      taskId={taskId}
      taskStartedAt={taskStartedAt}
      running={running}
      taskError={taskError}
      taskCurrentStage={taskCurrentStage}
      taskStageStatus={taskStageStatus}
      taskStageData={taskStageData}
      taskWindowSelection={taskWindowSelection}
      taskModelReview={taskModelReview}
      taskRefinements={taskRefinements}
      taskAlgorithmComparison={taskAlgorithmComparison}
      taskAttempts={taskAttempts}
      taskResult={taskResult}
      taskThinking={taskThinking}
      events={events}
      rawLogExpanded={rawLogExpanded}
      onStopTask={handleStopTune}
      onSelectAttempt={setSelectedFitAttemptKey}
      onToggleRawLogExpanded={() => setRawLogExpanded((prev) => !prev)}
      formatNumber={formatNumber}
      formatPercentValue={formatPercentValue}
    />
  );

  const renderPage = () => {
    const monitoring = loopMonitoring?.monitoring;
    const monitoringAlerts = monitoring?.alerts ?? [];
    const oscillationDetected = Boolean(monitoring?.stability?.oscillation_detected ?? assessment?.diagnostics.oscillation?.detected);

    if (activeSub === 'id_windows') {
      const ontologyPolicyDone = taskStageData.ontology_policy ?? null;
      const ontologyPolicyData = ontologyPolicyDone as { policy?: WindowSelectionPolicy } | null;
      const ontologyPolicyEarly = ontologyPolicyData?.policy ?? null;
      // 优先用 ontology_policy 阶段的策略；window_selection done 之后会再用更完整版本覆盖。
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
      // 子步骤拆分：
      //   ontology  ← ontology_policy 阶段的 fetching_mcp_context phase（MCP 检索）
      //   policy    ← ontology_policy 阶段的 building_policy phase（策略生成）
      //   algorithm ← detect_windows + apply_window_policy_to_candidates（在 window_selection done 中体现）
      //   llm       ← window_selection 阶段的 LLM 顾问执行
      //   gate      ← window_selection 阶段产出的准入结论
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
            : !policyStepDone ? 'waiting' : 'waiting',
        gate: windowSelectionDone ? 'done' : gateStepRunning ? 'running' : 'waiting',
      };
      const allDone = WINDOW_FLOW_STEPS.every((item) => stepStatus[item.key] === 'done');
      const phaseText = !windowReviewStarted
        ? '等待开始'
        : (WINDOW_FLOW_STEPS.find((item) => stepStatus[item.key] === 'running')?.title
          ?? (allDone ? '流程已完成' : '等待后端事件…'));

      return (
        <SectionErrorBoundary label="窗口候选页面">
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
                  onChange={setSelectedLoopId}
                  optionFilterProp="label"
                  options={scopedLoops.map((loop) => ({
                    value: loop.loop_id,
                    label: `${loop.loop_id} · ${LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type}`,
                  }))}
                />
                <Select
                  size="small"
                  style={{ width: 140 }}
                  value={windowRangePreset}
                  onChange={(value) => setWindowRangePreset(value)}
                  options={FEATURE_RANGE_OPTIONS.map((item) => ({ label: item.label, value: item.value }))}
                />
                {windowRangePreset === 'custom' && (
                  <DatePicker.RangePicker
                    size="small"
                    showTime
                    value={windowCustomRange}
                    onChange={(value) => setWindowCustomRange(value)}
                  />
                )}
                <Button
                  size="small"
                  icon={<SyncOutlined />}
                  disabled={!selectedLoop}
                  onClick={() => {
                    if (!selectedLoopId) return;
                    loadWindows(selectedLoopId, buildWindowRangeParams(selectedLoop));
                  }}
                >
                  预览该区间窗口
                </Button>
                <Button
                  type="primary"
                  icon={<AuditOutlined />}
                  loading={running}
                  disabled={!selectedLoop}
                  onClick={() => startTune({ useSelectedWindow: false, stopAfter: 'window_selection' })}
                >
                  开始本体驱动窗口评审
                </Button>
                {running && <Button danger onClick={handleStopTune}>停止</Button>}
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
                    <Descriptions.Item label="回路类型">{LOOP_TYPE_LABEL[selectedLoop?.loop_type ?? ''] ?? selectedLoop?.loop_type ?? '-'}</Descriptions.Item>
                    <Descriptions.Item label="数据点">{windowProfileDataPoints ?? '-'}</Descriptions.Item>
                    <Descriptions.Item label="采样周期">{formatNumber(windowProfileFeatures.data_profile?.sample_time_median_s, 1)}s</Descriptions.Item>
                    <Descriptions.Item label="PV范围">{formatRange(windowProfileFeatures.pv_stats?.min, windowProfileFeatures.pv_stats?.max, 3)}</Descriptions.Item>
                    <Descriptions.Item label="MV范围">{formatRange(windowProfileFeatures.mv_stats?.min, windowProfileFeatures.mv_stats?.max, 3)}</Descriptions.Item>
                    <Descriptions.Item label="MV活跃比例">{formatPercentValue(windowProfileFeatures.mv_stats?.active_step_ratio, 2)}</Descriptions.Item>
                    <Descriptions.Item label="MV反向频次">{formatNumber(windowProfileFeatures.mv_stats?.direction_reversal_per_hour, 2)}/h</Descriptions.Item>
                    <Descriptions.Item label="过程作用方向">
                      {formatProcessDirection(
                        String(
                          windowProfileFeatures.pv_mv_relation_raw?.process_direction
                            ?? windowProfileFeatures.pv_mv_relation_raw?.estimated_direction_raw
                            ?? '',
                        ),
                      )}
                    </Descriptions.Item>
                    <Descriptions.Item label="方向置信度">
                      {formatPercentValue(
                        typeof windowProfileFeatures.pv_mv_relation_raw?.process_direction_confidence === 'number'
                          ? windowProfileFeatures.pv_mv_relation_raw.process_direction_confidence
                          : undefined,
                        1,
                      )}
                    </Descriptions.Item>
                    <Descriptions.Item label="方向证据">
                      {formatProcessDirectionBasis(
                        String(windowProfileFeatures.pv_mv_relation_raw?.process_direction_basis ?? ''),
                      )}
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
                      <Descriptions.Item label="本体来源">{taskWindowSelection.ontology_context_source ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="上下文服务">{taskWindowSelection.ontology_mcp_server ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="上下文工具">{taskWindowSelection.ontology_mcp_tool ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="返回字数">{taskWindowSelection.ontology_mcp_content_chars ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="查询问题" span={4}>{taskWindowSelection.ontology_mcp_query ?? '-'}</Descriptions.Item>
                    </Descriptions>
                    <Collapse
                      items={[
                        {
                          key: 'ontology-raw',
                          label: '本体返回原文',
                          children: (
                            <Typography.Paragraph className="thinking-text">
                              {taskWindowSelection.ontology_mcp_content_raw || taskWindowSelection.ontology_mcp_content_preview || taskWindowSelection.ontology_mcp_error || '暂无本体返回内容'}
                            </Typography.Paragraph>
                          ),
                        },
                      ]}
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
                {renderWindowPolicyTables(windowPolicy)}
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
                    <Table
                      size="small"
                      pagination={{ pageSize: 8 }}
                      rowKey={(row) => `${row.index}-${row.window_source}`}
                      dataSource={windowPolicyResults}
                      columns={[
                        { title: '窗口', render: (_value, row) => `#${row.index} ${row.window_source || ''}` },
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
                      <Descriptions.Item label="选择模式">{taskWindowSelection.mode}</Descriptions.Item>
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
                          { title: '窗口', render: (_value, row) => `#${row.index} ${row.window_source || ''}` },
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
                          label: `模型推理过程 · ${item.model} · ${(item.reasoning_content || item.raw_text || '').length} 字`,
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
        </SectionErrorBoundary>
      );
    }

    switch (activeSub) {
      case 'dashboard': {
        const dashboardScore = dashboardStats.avgScore === undefined ? undefined : scorePercent(dashboardStats.avgScore);
        const loopCount = Math.max(scopedLoopStats.loopCount, 1);
        const loadedCount = dashboardRows.filter((row) => row.snapshot).length;
        const pendingCount = Math.max(scopedLoopStats.loopCount - loadedCount, 0);
        const warningTotal = dashboardStats.warningCount + dashboardStats.alarmCount;
        const realRows = getRealDashboardRows(dashboardRows);
        const compactTime = (value?: string | null) => value ? dayjs(value).format('MM-DD HH:mm') : '-';
        const compactDataRange = `${compactTime(dashboardStats.dataStart)} ~ ${compactTime(dashboardStats.dataEnd)}`;
        const typeCounts = countByLabel(scopedLoops, (loop) => LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type ?? '未知');
        const assetCounts = countByLabel(scopedLoops, (loop) => {
          const asset = assetNodes.find((node) => node.id === inferLoopAssetId(loop.loop_id));
          return asset?.name ?? '未归属';
        });
        const alertSeverityCounts = countDashboardAlertSeverities(realRows);
        const statusSlices = makeDashboardSlices([
          { label: '正常', value: dashboardStats.normalCount, color: '#22c55e' },
          { label: '关注', value: dashboardStats.warningCount, color: '#facc15' },
          { label: '告警', value: dashboardStats.alarmCount, color: '#ef4444' },
          { label: '待加载', value: pendingCount, color: '#64748b' },
        ]);
        const typePalette = ['#22c55e', '#facc15', '#60a5fa', '#a78bfa', '#fb923c', '#14b8a6'];
        const typeSlices = makeDashboardSlices(Object.entries(typeCounts).map(([label, value], index) => ({
          label,
          value,
          color: typePalette[index % typePalette.length],
        })));
        const assetRows = Object.entries(assetCounts)
          .map(([label, value]) => ({ label, value, percent: value / loopCount }))
          .sort((a, b) => b.value - a.value)
          .slice(0, 6);
        const indicatorRows = [
          { label: '数据健康', value: realRows.length ? realRows.reduce((sum, row) => sum + (row.snapshot?.data_health?.score ?? 0), 0) / realRows.length : undefined, color: '#38bdf8' },
          { label: '稳定性', value: realRows.length ? realRows.reduce((sum, row) => sum + (row.snapshot?.stability?.score ?? 0), 0) / realRows.length : undefined, color: '#22c55e' },
          { label: 'PV/MV 行为', value: realRows.length ? realRows.reduce((sum, row) => sum + (row.snapshot?.pv_mv_behavior?.score ?? 0), 0) / realRows.length : undefined, color: '#facc15' },
          { label: '约束', value: realRows.length ? realRows.reduce((sum, row) => sum + (row.snapshot?.constraints?.score ?? 0), 0) / realRows.length : undefined, color: '#fb923c' },
          { label: '响应可观测', value: realRows.length ? realRows.reduce((sum, row) => sum + (row.snapshot?.response_observability?.score ?? 0), 0) / realRows.length : undefined, color: '#a78bfa' },
        ];
        const topHealthyRows = getTopHealthyDashboardRows(realRows);
        const abnormalRows = getAbnormalDashboardRows(realRows);
        const alertRows = Object.entries(alertSeverityCounts)
          .map(([label, value], index) => ({ label, value, color: ['#ef4444', '#fb923c', '#facc15', '#60a5fa'][index % 4] }))
          .sort((a, b) => b.value - a.value);
        const assetBarRows: DashboardBarRow[] = assetRows.map((item) => ({
          label: item.label,
          percent: Math.max(4, item.percent * 100),
          trailing: `${item.value} (${formatPercentValue(item.percent, 1)})`,
        }));
        const indicatorBarRows: DashboardBarRow[] = indicatorRows.map((item) => {
          const pct = item.value === undefined ? 0 : scorePercent(item.value);
          return {
            label: item.label,
            percent: Math.max(0, Math.min(100, pct)),
            color: item.color,
            trailing: item.value === undefined ? '-' : `${pct}%`,
          };
        });
        const kpiItems: Array<{
          key: DashboardWidgetKey;
          label: string;
          value: number | string;
          suffix: string;
          color: string;
          sub: string;
        }> = [
          { key: 'kpi_total', label: '回路总数', value: scopedLoopStats.loopCount, suffix: '个', color: '#60a5fa', sub: `范围 ${compactDataRange}` },
          { key: 'kpi_loaded', label: '已监控回路', value: loadedCount, suffix: '个', color: '#22d3ee', sub: `覆盖率 ${formatPercentValue(scopedLoopStats.loopCount ? loadedCount / scopedLoopStats.loopCount : 0, 1)}` },
          { key: 'kpi_normal', label: '正常回路', value: dashboardStats.normalCount, suffix: '个', color: '#22c55e', sub: `占比 ${formatPercentValue(dashboardStats.normalCount / loopCount, 1)}` },
          { key: 'kpi_warning', label: '关注回路', value: dashboardStats.warningCount, suffix: '个', color: '#facc15', sub: `占比 ${formatPercentValue(dashboardStats.warningCount / loopCount, 1)}` },
          { key: 'kpi_alarm', label: '告警回路', value: dashboardStats.alarmCount, suffix: '个', color: '#ef4444', sub: `占比 ${formatPercentValue(dashboardStats.alarmCount / loopCount, 1)}` },
          { key: 'kpi_score', label: '监控均分', value: dashboardScore ?? '-', suffix: dashboardScore === undefined ? '' : '分', color: '#38bdf8', sub: dashboardScore === undefined ? '暂无快照' : '后端快照均值' },
          { key: 'kpi_alerts', label: '监控告警', value: dashboardStats.alertCount, suffix: '条', color: '#a78bfa', sub: warningTotal ? '需要处理' : '当前平稳' },
        ];
        const kpiWidgetMap = Object.fromEntries(kpiItems.map((item) => [
          item.key,
          {
            title: item.label,
            className: 'cockpit-kpi dashboard-widget-kpi',
            weight: 1,
            minWidth: 132,
            content: (
              <DashboardKpiWidget
                label={item.label}
                value={item.value}
                suffix={item.suffix}
                color={item.color}
                sub={item.sub}
              />
            ),
          },
        ])) as Partial<Record<DashboardWidgetKey, DashboardWidgetDefinition>>;
        const dashboardWidgetMap: Partial<Record<DashboardWidgetKey, DashboardWidgetDefinition>> = {
          ...kpiWidgetMap,
          health: {
            title: '回路健康分布',
            className: 'cockpit-card dashboard-widget-medium',
            weight: 2,
            minWidth: 320,
            content: (
              <DashboardDonutWidget
                title="回路健康分布"
                total={scopedLoopStats.loopCount}
                totalLabel="总回路数"
                slices={statusSlices}
                formatPercent={(value) => formatPercentValue(value, 1)}
              />
            ),
          },
          asset: {
            title: '回路按装置分布',
            className: 'cockpit-card dashboard-widget-medium',
            weight: 2,
            minWidth: 320,
            content: (
              <DashboardBarsWidget
                title="回路按装置分布"
                rows={assetBarRows}
                emptyDescription="暂无回路"
              />
            ),
          },
          type: {
            title: '回路类型分布',
            className: 'cockpit-card dashboard-widget-medium',
            weight: 2,
            minWidth: 320,
            content: (
              <DashboardDonutWidget
                title="回路类型分布"
                total={scopedLoopStats.loopCount}
                totalLabel="总回路数"
                slices={typeSlices}
                formatPercent={(value) => formatPercentValue(value, 1)}
              />
            ),
          },
          metrics: {
            title: '关键指标均值',
            className: 'cockpit-card dashboard-widget-medium',
            weight: 2,
            minWidth: 320,
            content: (
              <DashboardBarsWidget
                title="关键指标均值"
                rows={indicatorBarRows}
                variant="metric"
              />
            ),
          },
          top: {
            title: '性能评分 TOP5',
            className: 'cockpit-card wide dashboard-widget-wide',
            weight: 3,
            minWidth: 430,
            content: (
              <DashboardTopLoopsWidget
                rows={topHealthyRows}
                loopTypeLabels={LOOP_TYPE_LABEL}
                scorePercent={scorePercent}
                statusColor={monitoringStatusColor}
                statusText={monitoringStatusText}
                onSelectLoop={setSelectedLoopId}
                onViewLoop={(loopId) => { setSelectedLoopId(loopId); switchTo('monitor', 'loop_profile'); }}
              />
            ),
          },
          abnormal: {
            title: '异常回路列表',
            className: 'cockpit-card wide dashboard-widget-wide',
            weight: 3,
            minWidth: 430,
            content: (
              <DashboardAbnormalLoopsWidget
                rows={abnormalRows}
                statusText={monitoringStatusText}
                onViewLoop={(loopId) => { setSelectedLoopId(loopId); switchTo('monitor', 'loop_profile'); }}
              />
            ),
          },
          alerts: {
            title: '告警统计',
            className: 'cockpit-card alerts dashboard-widget-medium',
            weight: 2,
            minWidth: 300,
            content: (
              <DashboardAlertStatsWidget
                total={dashboardStats.alertCount}
                rows={alertRows}
              />
            ),
          },
          trend: {
            title: '选中回路真实趋势',
            className: 'cockpit-card trend dashboard-widget-wide',
            weight: 4,
            minWidth: 520,
            content: (
              <DashboardTrendWidget
                loopId={selectedLoop?.loop_id}
                trend={renderTrend(300)}
              />
            ),
          },
          snapshot: {
            title: '选中回路监控快照',
            className: 'cockpit-card snapshot dashboard-widget-medium',
            weight: 2,
            minWidth: 320,
            content: (
              <DashboardSnapshotWidget
                status={monitoring?.status}
                overallScore={monitoring?.overall_score}
                dataHealthScore={monitoring?.data_health?.score}
                stabilityScore={monitoring?.stability?.score}
                behaviorScore={monitoring?.pv_mv_behavior?.score}
                constraintScore={monitoring?.constraints?.score}
                scorePercent={scorePercent}
                statusColor={monitoringStatusColor}
                statusText={monitoringStatusText}
              />
            ),
          },
          quick: {
            title: '快捷操作',
            className: 'cockpit-card quick dashboard-widget-medium',
            weight: 2,
            minWidth: 300,
            content: (
              <DashboardQuickActionsWidget
                onCreateTuningTask={() => switchTo('tuning', 'tuning_task')}
                onViewLoopProfile={() => switchTo('monitor', 'loop_profile')}
                onViewTrendSpectrum={() => switchTo('monitor', 'trend_spectrum')}
                onOpenDiagnosis={() => switchTo('diagnostics', 'diagnosis_overview')}
              />
            ),
          },
        };
        return (
          <div className="dashboard-cockpit">
            <DashboardHeader
              assetTypeLabel={selectedAssetNode ? ASSET_TYPE_LABEL[selectedAssetNode.type] : '-'}
              assetTagColor={assetTagColor(selectedAssetNode?.type ?? 'factory')}
              pathLabel={selectedAssetPath.map((item) => item.name).join(' / ')}
              onOpenConfig={() => setDashboardConfigOpen(true)}
              onSwitchAsset={() => switchTo('settings', 'asset_directory')}
            />

            <DashboardWidgetGrid
              widgetKeys={dashboardWidgetKeys}
              widgetMap={dashboardWidgetMap}
              draggedWidgetKey={draggedDashboardWidgetKey}
              onDragStart={setDraggedDashboardWidgetKey}
              onDrop={moveDashboardWidget}
              onDragEnd={() => setDraggedDashboardWidgetKey(null)}
              onHide={hideDashboardWidget}
            />

            <DashboardConfigModal
              open={dashboardConfigOpen}
              widgetKeys={dashboardWidgetKeys}
              onChange={setDashboardWidgetKeys}
              onClose={() => setDashboardConfigOpen(false)}
            />
          </div>
        );
      }
      case 'loop_board':
        return (
          <LoopBoardPanel
            selectedLoop={selectedLoop}
            monitoring={monitoring}
            alertCount={monitoringAlerts.length}
            loading={loading}
            loopTable={renderLoopTable()}
            scorePercent={scorePercent}
            statusColor={monitoringStatusColor}
            statusText={monitoringStatusText}
            onRefresh={loadLoops}
          />
        );
      case 'asset_directory':
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">装置资产目录</div>
                  <Typography.Text type="secondary">
                    当前作用域：{selectedAssetPath.map((item) => item.name).join(' / ')}
                  </Typography.Text>
                </div>
                <Space wrap>
                  <Tag color={assetTagColor(selectedAssetNode?.type ?? 'factory')}>
                    {selectedAssetNode ? ASSET_TYPE_LABEL[selectedAssetNode.type] : '-'}
                  </Tag>
                  <Tag color="blue">{scopedLoopStats.loopCount} 个回路</Tag>
                </Space>
              </div>
              <div className="asset-directory-grid">
                <div className="asset-tree-panel">
                  <Tree
                    treeData={assetTreeData}
                    selectedKeys={[selectedAssetNodeId]}
                    defaultExpandedKeys={selectedAssetPath.map((item) => item.id)}
                    onSelect={(keys) => {
                      const next = String(keys[0] ?? selectedAssetNodeId);
                      setSelectedAssetNodeId(next);
                      setAssetRenameValue(assetNodes.find((item) => item.id === next)?.name ?? '');
                    }}
                  />
                </div>
                <div className="asset-editor-panel">
                  <Descriptions bordered column={2} size="small" className="industrial-descriptions">
                    <Descriptions.Item label="节点名称">{selectedAssetNode?.name ?? '-'}</Descriptions.Item>
                    <Descriptions.Item label="节点类型">
                      {selectedAssetNode ? ASSET_TYPE_LABEL[selectedAssetNode.type] : '-'}
                    </Descriptions.Item>
                    <Descriptions.Item label="节点编码">{selectedAssetNode?.code ?? '-'}</Descriptions.Item>
                    <Descriptions.Item label="挂载回路">{scopedLoopStats.loopCount} 个</Descriptions.Item>
                    <Descriptions.Item label="路径" span={2}>
                      {selectedAssetPath.map((item) => item.name).join(' / ')}
                    </Descriptions.Item>
                  </Descriptions>

                  <Divider orientation="left">新增子节点</Divider>
                  <div className="asset-edit-row">
                    <Input
                      value={assetDraftName}
                      placeholder="例如：反应系统、分馏系统、V-0202回流罐"
                      onChange={(event) => setAssetDraftName(event.target.value)}
                    />
                    <Select
                      value={assetDraftType}
                      onChange={setAssetDraftType}
                      options={(Object.keys(ASSET_TYPE_LABEL) as AssetNodeType[]).map((type) => ({
                        label: ASSET_TYPE_LABEL[type],
                        value: type,
                      }))}
                    />
                    <Button type="primary" onClick={addAssetChild}>新增</Button>
                  </div>

                  <Divider orientation="left">编辑当前节点</Divider>
                  <div className="asset-edit-row">
                    <Input
                      value={assetRenameValue}
                      placeholder={selectedAssetNode?.name ?? '节点名称'}
                      onChange={(event) => setAssetRenameValue(event.target.value)}
                    />
                    <Button onClick={renameAssetNode}>重命名</Button>
                    <Button danger onClick={deleteAssetNode}>删除空节点</Button>
                  </div>

                  <Alert
                    className="agent-alert"
                    type="info"
                    showIcon
                    message="第一版为前端本地目录"
                    description="当前目录结构仅保存在前端本地。"
                  />
                </div>
              </div>
            </section>

            <section className="agent-panel">
              <div className="panel-title">当前作用域回路</div>
              <Table
                size="small"
                pagination={false}
                rowKey="loop_id"
                dataSource={scopedLoops}
                columns={[
                  { title: '回路位号', dataIndex: 'loop_id' },
                  { title: '类型', dataIndex: 'loop_type', render: (value: string) => LOOP_TYPE_LABEL[value] ?? value },
                  { title: '归属节点', render: (_: unknown, row: HistoryLoop) => assetNodes.find((node) => node.id === inferLoopAssetId(row.loop_id))?.name ?? '-' },
                  { title: '数据点', dataIndex: 'rows' },
                  {
                    title: '操作',
                    render: (_: unknown, row: HistoryLoop) => (
                      <Space>
                        <Button size="small" onClick={() => {
                          setSelectedLoopId(row.loop_id);
                          switchTo('monitor', 'loop_profile');
                        }}>画像</Button>
                        <Button size="small" onClick={() => {
                          setSelectedLoopId(row.loop_id);
                          switchTo('tuning', 'tuning_task');
                        }}>整定</Button>
                      </Space>
                    ),
                  },
                ]}
              />
            </section>
          </div>
        );
      case 'loop_profile':
        return (
          <LoopProfilePanel
            selectedLoopId={selectedLoopId}
            selectedLoop={selectedLoop}
            scopedLoops={scopedLoops}
            loopFeatures={loopFeatures}
            assessment={assessment}
            monitoring={monitoring}
            featureRangePreset={featureRangePreset}
            featureCustomRange={featureCustomRange}
            featureRangeOptions={FEATURE_RANGE_OPTIONS}
            featureLoading={featureLoading}
            loopTypeLabel={LOOP_TYPE_LABEL}
            onLoopChange={setSelectedLoopId}
            onRangePresetChange={(value) => setFeatureRangePreset(value as FeatureRangePreset)}
            onCustomRangeChange={setFeatureCustomRange}
            onRefresh={() => {
              if (!selectedLoopId) return;
              const params = buildFeatureRangeParams(selectedLoop);
              loadLoopFeatures(selectedLoopId, params);
              loadLoopMonitoring(selectedLoopId, params);
            }}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
            formatRange={formatRange}
            scorePercent={scorePercent}
            tagColor={tagColor}
            formatProcessDirection={formatProcessDirection}
            formatProcessDirectionBasis={formatProcessDirectionBasis}
            formatHarrisBasis={formatHarrisBasis}
            formatCpkBasis={formatCpkBasis}
            monitoringStatusColor={monitoringStatusColor}
            monitoringStatusText={monitoringStatusText}
          />
        );
      case 'trend_spectrum':
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
                    onChange={setSelectedLoopId}
                    optionFilterProp="label"
                    options={scopedLoops.map((loop) => ({
                      value: loop.loop_id,
                      label: `${loop.loop_id} · ${LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type}`,
                    }))}
                  />
                  <Select
                    style={{ width: 150 }}
                    value={trendPreset}
                    onChange={(value) => setTrendPreset(value)}
                    options={TREND_PRESET_OPTIONS.map((item) => ({ label: item.label, value: item.value }))}
                  />
                  <Select
                    style={{ width: 170 }}
                    value={trendPointLimit}
                    onChange={(value) => setTrendPointLimit(value)}
                    options={TREND_POINT_LIMIT_OPTIONS}
                  />
                  <Space className="inline-switch">
                    <Typography.Text type="secondary">PV/MV 分轴</Typography.Text>
                    <Switch checked={trendSplitYAxis} onChange={setTrendSplitYAxis} />
                  </Space>
                  {trendPreset === 'custom' && (
                    <DatePicker.RangePicker
                      showTime
                      value={trendCustomRange}
                      onChange={(value) => setTrendCustomRange(value)}
                    />
                  )}
                  <Button
                    icon={<SyncOutlined />}
                    loading={seriesLoading}
                    onClick={() => selectedLoopId && loadSeries(selectedLoopId, selectedLoop)}
                  >
                    刷新趋势
                  </Button>
                </Space>
              </div>
              <TrendQueryDetails
                selectedLoop={selectedLoop}
                loopTypeLabel={selectedLoop ? LOOP_TYPE_LABEL[selectedLoop.loop_type] ?? selectedLoop.loop_type : '-'}
                series={series}
                rangeLabel={trendPreset === 'custom'
                  ? `${trendCustomRange?.[0]?.format('YYYY-MM-DD HH:mm:ss') ?? '-'} ~ ${trendCustomRange?.[1]?.format('YYYY-MM-DD HH:mm:ss') ?? '-'}`
                  : TREND_PRESET_OPTIONS.find((item) => item.value === trendPreset)?.label ?? '-'}
                pointLimitLabel={trendPointLimit === 'all'
                  ? '全量点'
                  : `${TREND_POINT_LIMIT_OPTIONS.find((item) => item.value === trendPointLimit)?.label ?? trendPointLimit}`}
              />
            </section>
            <TrendChartPanel
              selectedLoop={selectedLoop}
              series={series}
              loading={seriesLoading}
              chart={renderTrend(420)}
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
      case 'alarm_events':
      case 'risk_alerts':
        return (
          <AlarmEventsPanel
            railAlarms={railAlarms}
            monitoringStatus={monitoring?.status}
            monitoringEventCount={monitoring?.events?.length ?? monitoringAlerts.length}
            diagnosticFlagCount={assessment?.diagnostics.flags.length ?? 0}
            taskLabel={taskId ? taskStatus : '暂无任务'}
            pathLabel={selectedAssetPath.map((item) => item.name).join(' / ')}
            monitoringStatusText={monitoringStatusText}
            monitoringStatusColor={monitoringStatusColor}
            alertSeverityColor={alertSeverityColor}
          />
        );
      case 'tuning_readiness':
        return (
          <TuningReadinessPanel
            assessment={assessment}
            showDetails={activeSub === 'tuning_readiness'}
            assessmentCards={renderAssessmentCards()}
            tagColor={tagColor}
          />
        );
      case 'performance_score':
        return (
          <PerformanceScorePanel
            assessment={assessment}
            assessmentLoading={assessmentLoading}
            taskResult={taskResult}
            formatNumber={formatNumber}
            scorePercent={scorePercent}
            tagColor={tagColor}
          />
        );
      case 'condition_recognition':
        return (
          <OperatingConditionPanel
            conditionProfile={loopFeatures?.operating_condition_profile}
            selectedLoop={selectedLoop}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
            operatingConditionText={operatingConditionText}
            tuningSuitabilityText={tuningSuitabilityText}
            tuningSuitabilityColor={tuningSuitabilityColor}
            conditionEvidenceName={conditionEvidenceName}
            conditionEvidenceDetail={conditionEvidenceDetail}
            conditionRecommendationText={conditionRecommendationText}
            evidenceStatusText={evidenceStatusText}
            evidenceStatusColor={evidenceStatusColor}
          />
        );
      case 'actuator_status':
        return (
          <ActuatorStatusPanel
            selectedLoopId={selectedLoopId}
            scopedLoops={scopedLoops}
            loopFeatures={loopFeatures}
            onLoopChange={setSelectedLoopId}
            loopTypeLabel={(loop) => LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
            yesNo={yesNo}
          />
        );
      case 'constraint_monitor':
        return (
          <ConstraintMonitorPanel
            loopFeatures={loopFeatures}
            monitoring={monitoring}
            scorePercent={scorePercent}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
            monitoringStatusText={monitoringStatusText}
            monitoringStatusColor={monitoringStatusColor}
          />
        );
      case 'diagnosis_overview':
        return (
          <DiagnosisOverviewPanel assessment={assessment} />
        );
      case 'oscillation_diagnosis':
        return (
          <OscillationDiagnosisPanel
            assessment={assessment}
            monitoring={monitoring}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
          />
        );
      case 'pid_diagnosis':
      case 'valve_diagnosis':
      case 'measurement_noise_diagnosis':
      case 'process_disturbance_diagnosis':
        return <DiagnosisPlanPanel activeSub={activeSub as DiagnosisPlanKey} />;
      case 'tuning_prior':
        return (
          <TuningPriorPanel
            loops={loops}
            selectedLoopId={selectedLoopId}
            selectedLoop={selectedLoop}
            loopTypeLabel={LOOP_TYPE_LABEL}
            featureRangeOptions={FEATURE_RANGE_OPTIONS}
            tuningPriorRangePreset={tuningPriorRangePreset}
            tuningPriorCustomRange={tuningPriorCustomRange}
            tuningPriorCoreData={tuningPriorCoreData}
            tuningPriorOntologyData={tuningPriorOntologyData}
            tuningPriorReviewData={tuningPriorReviewData}
            tuningPriorCoreLoading={tuningPriorCoreLoading}
            tuningPriorOntologyLoading={tuningPriorOntologyLoading}
            tuningPriorReviewLoading={tuningPriorReviewLoading}
            tuningPriorCoreError={tuningPriorCoreError}
            tuningPriorOntologyError={tuningPriorOntologyError}
            tuningPriorReviewError={tuningPriorReviewError}
            onLoopChange={setSelectedLoopId}
            onRangePresetChange={(value) => setTuningPriorRangePreset(value as FeatureRangePreset)}
            onCustomRangeChange={setTuningPriorCustomRange}
            onLoadCore={() => {
              if (!selectedLoopId) return;
              loadTuningPriorCore(selectedLoopId, buildTuningPriorRangeParams(selectedLoop));
            }}
            onLoadOntology={() => {
              if (!selectedLoopId) return;
              loadTuningPriorOntology(selectedLoopId, buildTuningPriorRangeParams(selectedLoop));
            }}
            onLoadReview={() => {
              if (!selectedLoopId) return;
              loadTuningPriorReview(selectedLoopId);
            }}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
            formatRange={formatRange}
            formatProcessDirection={formatProcessDirection}
            operatingConditionText={operatingConditionText}
            monitoringStatusText={monitoringStatusText}
            gateDecisionText={gateDecisionText}
            gateCheckLabel={gateCheckLabel}
            tagColor={tagColor}
          />
        );
      case 'model_reliability':
        return (
          <ModelReliabilityPanel
            windowAlgorithmSummary={windowAlgorithmSummary}
            fitPreviewAttempts={fitPreviewAttempts}
            selectedFitAttempt={selectedFitAttempt}
            fitPreviewChartData={fitPreviewChartData}
            selectedFitAttemptKey={selectedFitAttempt ? attemptFitKey(selectedFitAttempt) : undefined}
            onSelectedFitAttemptKeyChange={setSelectedFitAttemptKey}
            taskAlgorithmComparison={taskAlgorithmComparison}
            deterministicRefinement={deterministicRefinement}
            windows={windows}
            selectedWindow={selectedWindow}
            windowPreviewData={windowPreviewData}
            windowTable={renderWindowTable()}
            chartLineTooltip={chartLineTooltip}
            onOpenTuningTask={() => setActiveSub('tuning_task')}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
            scorePercent={scorePercent}
            scoreStatus={scoreStatus}
          />
        );
      case 'tuning_task':
        return (
          <TuningTaskPanel
            selectedLoopId={selectedLoopId}
            selectedLoop={selectedLoop}
            scopedLoops={scopedLoops}
            loopTypeLabel={LOOP_TYPE_LABEL}
            featureRangeOptions={FEATURE_RANGE_OPTIONS}
            tuningRangePreset={tuningRangePreset}
            tuningCustomRange={tuningCustomRange}
            tuningUseLlm={tuningUseLlm}
            running={running}
            tuningGate={tuningGate}
            assessmentLoading={assessmentLoading}
            assessmentError={assessmentError}
            assessment={assessment}
            taskAttemptsCount={taskAttempts.length}
            taskStageStatus={taskStageStatus}
            taskStageData={taskStageData}
            taskStatus={taskStatus}
            taskId={taskId}
            taskCurrentStage={taskCurrentStage}
            taskResult={taskResult}
            onLoopChange={setSelectedLoopId}
            onRangePresetChange={(value) => setTuningRangePreset(value as FeatureRangePreset)}
            onCustomRangeChange={setTuningCustomRange}
            onUseLlmChange={setTuningUseLlm}
            onTune={handleTune}
            onStopTune={handleStopTune}
            onOpenTaskDetail={() => setTaskDetailOpen(true)}
            gateDecisionText={gateDecisionText}
            gateCheckLabel={gateCheckLabel}
            gateSeverityColor={gateSeverityColor}
            gateImpact={gateImpact}
            gateCheckMessage={gateCheckMessage}
            tagColor={tagColor}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
          />
        );
      case 'data_sources':
        return (
          <DataSourcesPanel
            dataSourceType={dataSourceType}
            fileList={fileList}
            importedLoopCount={loops.length}
            importing={importing}
            onDataSourceTypeChange={setDataSourceType}
            onFileListChange={setFileList}
            onImport={handleImport}
          />
        );
      case 'rule_config':
        return (
          <RuleConfigPanel
            policyConfig={policyConfig}
            loading={policyConfigLoading}
            onRefresh={loadPolicyConfig}
            loopTypeLabel={(loopType) => LOOP_TYPE_LABEL[loopType] ?? loopType}
            policyLoopImpact={policyLoopImpact}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
          />
        );
      case 'prompt_config':
        return (
          <PromptConfigPanel
            form={promptConfigForm}
            promptConfig={promptConfig}
            promptItems={PROMPT_CONFIG_ITEMS}
            activePromptField={activePromptField}
            loading={promptConfigLoading}
            saving={promptConfigSaving}
            onActivePromptFieldChange={(value) => setActivePromptField(value as PromptConfigField)}
            onSave={savePromptConfig}
            onRefresh={loadPromptConfig}
            onRestoreDefault={restoreDefaultPromptConfig}
          />
        );
      case 'model_config':
        return (
          <ModelConfigPanel
            form={modelConfigForm}
            modelConfig={modelConfig}
            testResult={modelConfigTestResult}
            loading={modelConfigLoading}
            saving={modelConfigSaving}
            testing={modelConfigTesting}
            onSave={saveModelConfig}
            onTest={testModelConnection}
            onRefresh={loadModelConfig}
          />
        );
      case 'mcp_config':
        return (
          <div className="page-stack embedded-settings-page">
            <McpConfigPage embedded />
          </div>
        );
      default:
        return (
          <section className="agent-panel">
            <div className="panel-title">{currentSub.label}</div>
            <Empty description="该页面暂未开放" />
          </section>
        );
    }
  };

  const renderModeSwitch = (className = '') => (
    <div className={`mode-switch ${className}`}>
      <button
        type="button"
        title="对话模式"
        aria-label="对话模式"
        className={viewMode === 'dialogue' ? 'active' : ''}
        onClick={() => setViewMode('dialogue')}
      >
        <RobotOutlined />
      </button>
      <button
        type="button"
        title="经典模式"
        aria-label="经典模式"
        className={viewMode === 'classic' ? 'active' : ''}
        onClick={() => setViewMode('classic')}
      >
        <AppstoreOutlined />
      </button>
    </div>
  );

  const renderAppTopbar = () => (
    <header className="pid-app-header">
      <div className="pid-app-topbar">
        <div className="pid-app-brand">
          <button
            type="button"
            className="menu-trigger"
            aria-label={sidebarCollapsed ? '展开导航菜单' : '折叠导航菜单'}
            onClick={() => setSidebarCollapsed((value) => !value)}
          >
            <MenuOutlined />
          </button>
          <div className="brand-mark">PID</div>
          <div>
            <h1>智能PID控制系统平台</h1>
          </div>
          {renderModeSwitch('classic-mode-switch')}
        </div>
        <div className="system-meta">
          <span style={{ color: '#1d4ed8', fontWeight: 800 }}>V1.0</span>
          <span><ClockCircleOutlined /> {new Date().toLocaleString()}</span>
          <span><UserOutlined /> admin</span>
          <span className="alarm-pill"><BellOutlined /> 6</span>
        </div>
      </div>
    </header>
  );

  const renderAssistantTextLine = (item: AssistantMessage, line: string, index: number) => {
    const action = normalizeAssistantAction(line, selectedLoop?.loop_id ?? selectedLoopId);
    if (action) {
      return (
        <p key={`${item.id}-${index}`} className="dialogue-action-line">
          <button type="button" className="dialogue-inline-action" onClick={() => runAssistantAction(action)}>
            {action.label}
          </button>
        </p>
      );
    }
    return <p key={`${item.id}-${index}`}>{line}</p>;
  };

  const renderDialogueMode = () => {
    return (
      <div className="dialogue-shell">
        {renderAppTopbar()}

        <main className={sidebarCollapsed ? 'dialogue-main history-collapsed' : 'dialogue-main'}>
          <aside className="dialogue-history">
            <div className="dialogue-history-head">
              <h2>历史对话</h2>
              <Button size="small" type="primary" ghost icon={<SyncOutlined />} loading={assistantSessionsLoading} onClick={createDialogueSession}>新建对话</Button>
            </div>
            <div className="history-list">
              {sortedAssistantSessions.length ? (
                <>
                  <div className="history-group">最近</div>
                  {sortedAssistantSessions.map((item) => {
                    const isActive = item.id === activeAssistantSession?.id;
                    const isPinned = pinnedAssistantSessionIdSet.has(item.id);
                    return (
                      <div
                      key={item.id}
                      className={isActive ? 'history-item active' : 'history-item'}
                    >
                      <button
                        type="button"
                        className="history-item-main"
                        onClick={() => openAssistantSession(item.id)}
                      >
                      <span>{item.title || '未命名对话'}</span>
                      <em>{item.updated_at ? dayjs(item.updated_at).format('HH:mm') : ''}</em>
                      </button>
                      <Dropdown
                        trigger={['click']}
                        placement="bottomRight"
                        menu={{
                          items: [
                            { key: 'pin', icon: <PushpinOutlined />, label: isPinned ? '取消置顶' : '置顶' },
                            { key: 'rename', icon: <EditOutlined />, label: '重命名' },
                            {
                              key: 'delete',
                              icon: <DeleteOutlined />,
                              label: <span className="history-danger-menu-item">删除</span>,
                              danger: true,
                            },
                          ],
                          onClick: ({ key, domEvent }) => {
                            domEvent.stopPropagation();
                            if (key === 'pin') toggleAssistantSessionPin(item.id);
                            if (key === 'rename') renameAssistantSession(item);
                            if (key === 'delete') deleteAssistantSessionWithConfirm(item);
                          },
                        }}
                      >
                        <Button
                          type="text"
                          size="small"
                          className="history-more-btn"
                          icon={<EllipsisOutlined />}
                          onClick={(event) => event.stopPropagation()}
                        />
                      </Dropdown>
                    </div>
                    );
                  })}
                </>
              ) : (
                <Empty description="暂无历史对话" />
              )}
            </div>
          </aside>

          <section className="dialogue-chat">
            <div className="dialogue-loop-select">
              <Select
                size="large"
                allowClear
                value={selectedLoopId}
                onChange={setSelectedLoopId}
                style={{ minWidth: 360 }}
                popupClassName="dialogue-loop-dropdown"
                placeholder="选择回路上下文"
                options={loops.map((loop) => ({
                  value: loop.loop_id,
                  label: `${loop.loop_id} · ${LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type}`,
                }))}
              />
              <Tag color={activeAssistantSession ? 'blue' : 'default'} style={{ marginLeft: 12 }}>
                {activeAssistantSession ? `当前对话：${activeAssistantSession.title}` : '未创建会话'}
              </Tag>
            </div>

            <div className="chat-thread">
              {assistantMessages.length ? (
                assistantMessages.map((item) => (
                  <div key={item.id} className={item.role === 'user' ? 'chat-question' : 'chat-answer-row'}>
                    {item.role === 'user' ? (
                      <>
                        {item.text}
                        <span>刚刚</span>
                      </>
                    ) : (
                      <>
                        <div className="bot-avatar"><RobotOutlined /></div>
                        <div className="chat-answer-card">
                          {!!item.eventLog?.length && (
                            <div className="dialogue-event-stream">
                              <div className="dialogue-event-title">事件流</div>
                              {item.eventLog.map((eventItem) => (
                                <div key={eventItem.id} className={`dialogue-event-item ${eventItem.type}`}>
                                  <span className="dialogue-event-dot" />
                                  <div>
                                    <strong>{eventItem.title}</strong>
                                    {eventItem.detail && <em>{eventItem.detail}</em>}
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                          {(item.reasoning || item.loading) && (
                            <div className="ai-reasoning-box">
                              <div className="ai-reasoning-title">分析过程</div>
                              <div className="ai-reasoning-text">
                                {(item.reasoning || '正在读取上下文并生成分析摘要...').split('\n').map((line, index) => (
                                  line ? <p key={`${item.id}-r-${index}`}>{line}</p> : null
                                ))}
                              </div>
                            </div>
                          )}
                          <div className="ai-message-text">
                            {(item.text || (item.loading ? '正在生成回答...' : '')).split('\n').map((line, index) => renderAssistantTextLine(item, line, index))}
                            {item.loading && <span className="ai-stream-cursor" />}
                          </div>
                          {item.error && <Alert type="error" showIcon message={item.error} />}
                          {!!item.actions?.length && (
                            <div className="dialogue-actions">
                              {item.actions.map((action) => (
                                <Button key={`${item.id}-${action.label}`} size="small" onClick={() => runAssistantAction(action)}>
                                  {action.label}
                                </Button>
                              ))}
                            </div>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                ))
              ) : (
                <div className="dialogue-starter">
                  <div className="dialogue-starter-mark"><RobotOutlined /></div>
                  <h2>{selectedLoop?.loop_id ? `${selectedLoop.loop_id} 智能分析` : 'PID 智能整定助手'}</h2>
                  <p>选择一个问题开始，或在下方直接输入你的问题。</p>
                  <div className="dialogue-starter-grid">
                    {DIALOGUE_STARTER_PROMPTS.map((item) => (
                      <button
                        type="button"
                        key={item.title}
                        className="dialogue-starter-card"
                        onClick={() => askAssistant(item.prompt)}
                        disabled={assistantStreaming}
                      >
                        <strong>{item.title}</strong>
                        <span>{item.description}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="dialogue-input-row">
              <Input.TextArea
                value={assistantInput}
                onChange={(event) => setAssistantInput(event.target.value)}
                autoSize={{ minRows: 2, maxRows: 4 }}
                placeholder="请输入您的问题，例如：这个回路为什么波动大？"
                onPressEnter={(event) => {
                  if (!event.shiftKey) {
                    event.preventDefault();
                    askAssistant();
                  }
                }}
              />
              <Button type="primary" icon={<SendOutlined />} loading={assistantStreaming} onClick={() => askAssistant()} />
            </div>
          </section>

        </main>
      </div>
    );
  };

  if (viewMode === 'dialogue') {
    return renderDialogueMode();
  }

  return (
    <div className="agent-console">
      {renderAppTopbar()}

        <main className={sidebarCollapsed ? 'agent-main industrial-main sidebar-collapsed' : 'agent-main industrial-main'}>
          <aside className={sidebarCollapsed ? 'side-menu industrial-tree collapsed' : 'side-menu industrial-tree'}>
            {MODULES.map((module) => {
            const expanded = expandedModules[module.key];
            return (
              <div className={expanded ? 'nav-group expanded' : 'nav-group'} key={module.key}>
                <button
                  className={module.key === activeModule ? 'nav-group-title active' : 'nav-group-title'}
                  title={module.label}
                  onClick={() => {
                    if (sidebarCollapsed) {
                      setSidebarCollapsed(false);
                      switchTo(module.key, module.subs[0].key);
                    } else {
                      toggleModule(module.key);
                    }
                  }}
                >
                  {module.icon}
                  <span>{module.label}</span>
                  <i className="nav-arrow">{expanded ? <DownOutlined /> : <RightOutlined />}</i>
                </button>
                {expanded && !sidebarCollapsed && (
                  <div className="nav-sub-list">
                    {module.subs.map((sub) => (
                      <button
                        key={sub.key}
                        className={sub.key === activeSub ? 'active' : ''}
                        title={sub.label}
                        onClick={() => switchTo(module.key, sub.key)}
                      >
                        {sub.icon}
                        <span>{sub.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </aside>

          <section className="content-area">
            <div className="industrial-content-shell no-context-rail">
            <div className="primary-workspace">
              {renderPage()}
            </div>
          </div>
          <TuningTaskDetailDrawer
            open={taskDetailOpen}
            onClose={() => setTaskDetailOpen(false)}
          >
            {renderTaskDashboard()}
          </TuningTaskDetailDrawer>
        </section>
      </main>
    </div>
  );
}

// 顶层包一层 ErrorBoundary：任何 hook 或 renderPage 的渲染异常都不会让整个页面白屏，
// 而是显示错误堆栈，便于现场定位问题。
export default function LoopMonitoringPage() {
  return (
    <SectionErrorBoundary label="回路监控页面">
      <LoopMonitoringPageInner />
    </SectionErrorBoundary>
  );
}


