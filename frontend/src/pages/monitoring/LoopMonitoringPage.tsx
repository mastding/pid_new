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
  Drawer,
  Dropdown,
  Empty,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Progress,
  Select,
  Space,
  Statistic,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  Tree,
  Upload,
  message,
} from 'antd';
import type { UploadFile } from 'antd';
import type { DataNode } from 'antd/es/tree';
import {
  ApiOutlined,
  AppstoreOutlined,
  AuditOutlined,
  BellOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloudUploadOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  DeploymentUnitOutlined,
  DownOutlined,
  EditOutlined,
  EllipsisOutlined,
  ExperimentOutlined,
  FileSearchOutlined,
  FundProjectionScreenOutlined,
  KeyOutlined,
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
  StrategyCandidate,
  TuningResult,
  WindowAlgorithmFamilySummary,
  WindowAlgorithmPlanItem,
  WindowAlgorithmFitSummary,
  WindowPolicyFieldUsage,
  WindowSelectionPolicy,
  WindowSelectionMeta,
} from '@/types/tuning';
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
      { key: 'mcp_config', label: 'MCP 服务配置', icon: <ApiOutlined />, implemented: true },
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

const TUNING_STAGE_KEYS = [
  'data_analysis',
  'ontology_policy',
  'window_selection',
  'identification',
  'model_review',
  'identification_refinement',
  'tuning',
  'evaluation',
];

const TUNING_STAGE_LABELS: Record<string, string> = {
  data_analysis: '数据分析',
  ontology_policy: '本体策略',
  window_selection: '窗口选择',
  identification: '模型辨识',
  model_review: 'LLM 评审',
  identification_refinement: '精修建议',
  tuning: 'PID 整定',
  evaluation: '性能评估',
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
  { key: 'profile', title: '1 数据画像', desc: '读取 LoopFeatures 原始特征' },
  { key: 'ontology', title: '2 本体检索', desc: '查询本体/MCP 回路上下文' },
  { key: 'policy', title: '3 策略生成', desc: '生成窗口算法族策略 JSON' },
  { key: 'algorithm', title: '4 算法族运行', desc: '按策略驱动 provider 产出候选窗口' },
  { key: 'llm', title: '5 LLM 评审', desc: '结合画像、本体和候选窗口做解释性判断' },
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

type TaskStatus = 'idle' | 'running' | 'done' | 'error';

interface TaskEventLog {
  id: number;
  label: string;
  detail?: string;
}

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
    label: 'AI 助手系统提示词',
    group: 'AI 助手',
    help: '定义 AI 助手身份、任务范围、回答风格和证据引用要求。',
    placeholder: '定义 AI 助手身份、任务范围、回答风格和证据引用要求',
    minRows: 10,
    maxRows: 20,
  },
  {
    key: 'assistant_developer_prompt',
    label: 'AI 助手安全 / 流程约束提示词',
    group: 'AI 助手',
    help: '定义禁止直接整定、禁止修改参数、高成本流程需确认等安全边界。',
    placeholder: '定义禁止直接整定、禁止修改参数、高成本流程需确认等边界',
    minRows: 8,
    maxRows: 16,
  },
  {
    key: 'assistant_response_schema',
    label: 'AI 助手响应格式说明 / JSON Schema',
    group: 'AI 助手',
    help: '定义 answer、evidence、risk_level、suggested_actions 等结构化字段。',
    placeholder: '定义 answer、evidence、risk_level、suggested_actions 等结构化字段',
    minRows: 8,
    maxRows: 16,
  },
  {
    key: 'window_policy_system_prompt',
    label: '窗口候选策略提示词',
    group: '窗口候选',
    help: '用于整定中心的窗口候选策略生成，指导 LLM 根据画像、本体/MCP 上下文输出窗口算法策略 JSON。',
    placeholder: '定义窗口策略生成器身份、算法族约束、输出 JSON 字段和安全边界',
    minRows: 12,
    maxRows: 24,
  },
  {
    key: 'window_policy_user_prompt_template',
    label: '窗口候选用户提示词模板',
    group: '窗口候选',
    help: '运行时会替换 $base_policy_json、$profile_text、$pv_json、$mv_json、$raw_profile_json、$mcp_content、$frontend_text。',
    placeholder: '定义选窗 LLM 接收实时画像和本体上下文的 user prompt 模板',
    minRows: 10,
    maxRows: 20,
  },
  {
    key: 'identification_review_system_prompt',
    label: '辨识 / 模型评审提示词',
    group: '辨识评审',
    help: '用于辨识结束后的模型可信度评审，约束 LLM 输出 accept / downgrade、理由和 concerns。',
    placeholder: '定义模型评审专家身份、K/T/L/R2/NRMSE/corr 等判据和 JSON 输出要求',
    minRows: 12,
    maxRows: 24,
  },
  {
    key: 'identification_review_user_prompt_template',
    label: '辨识 / 模型评审用户提示词模板',
    group: '辨识评审',
    help: '运行时会替换 $loop_type、$data_profile_text、$window_source、$model_type、$attempts_text 等变量。',
    placeholder: '定义模型评审 LLM 接收辨识结果、窗口和 attempts 的 user prompt 模板',
    minRows: 10,
    maxRows: 20,
  },
  {
    key: 'consultant_system_prompt',
    label: '整定顾问 Agent 提示词',
    group: '整定顾问',
    help: '用于顾问式对话和工具调用流程，约束 LLM 如何解释整定、调用工具和回答用户。',
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
  return `${label}回路：辨识阶段按模型顺序扩大搜索；精修阶段在 LLM 不可用时按备选模型池重试；T 下界约束优化器搜索空间，Reality T 范围影响整定后仿真评分。`;
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

function eventLabel(event: PipelineEvent) {
  if (event.type === 'stage') {
    const stageLabel = TUNING_STAGE_LABELS[event.stage] ?? event.stage;
    return `${stageLabel} ${event.status === 'running' ? '运行中' : '完成'}`;
  }
  if (event.type === 'llm_thinking') {
    return `${TUNING_STAGE_LABELS[event.stage] ?? event.stage} LLM 思考`;
  }
  if (event.type === 'session_start') return `任务已创建：${event.task_id}`;
  if (event.type === 'result') return '完整结果已返回';
  if (event.type === 'error') return `流程异常：${event.message}`;
  return '流程结束';
}

function formatEventDetail(detail?: string) {
  if (!detail) return '';
  try {
    return JSON.stringify(JSON.parse(detail), null, 2);
  } catch {
    return detail;
  }
}

function BackendBadge({ implemented }: { implemented?: boolean }) {
  return implemented ? <Tag color="green">已接后端</Tag> : <Tag color="default">未开放</Tag>;
}

function EmptyBackendHint({ title = '该能力后端尚未接入' }: { title?: string }) {
  return (
    <Alert
      className="agent-alert"
      type="warning"
      showIcon
      message={title}
      description="当前先按产品化界面占位展示字段、筛选项和操作入口，后续补齐后端接口后可直接替换为真实数据。"
    />
  );
}

function attemptFitKey(attempt: IdentificationAttempt) {
  return [
    attempt.round ?? 0,
    attempt.window_source ?? '',
    attempt.window_algorithm ?? '',
    attempt.model_type ?? '',
    attempt.fit_score ?? '',
  ].join('|');
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
  const [taskStageStatus, setTaskStageStatus] = useState<Record<string, 'running' | 'done'>>({});
  const [taskStageData, setTaskStageData] = useState<Record<string, Record<string, unknown>>>({});
  // running 事件里的 sub-phase 数据（如 ontology_policy 的 phase=fetching_mcp_context|building_policy）。
  // 不与 taskStageData（done payload）合并，避免 done 之后被 running 残留覆盖。
  const [taskStageRunningData, setTaskStageRunningData] = useState<Record<string, Record<string, unknown>>>({});
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

  useEffect(() => {
    try {
      window.localStorage.setItem('pid_v2_pinned_assistant_sessions', JSON.stringify(pinnedAssistantSessionIds));
    } catch {
      // Local pinning is an optional UI preference.
    }
  }, [pinnedAssistantSessionIds]);

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

  const dashboardRows = useMemo(() => scopedLoops.map((loop) => {
    const snapshot = monitoringByLoopId[loop.loop_id]?.monitoring;
    // best_window_score 字段也不再预算；用监控快照分作为唯一兜底。
    const overallScore = snapshot?.overall_score ?? 0;
    const alertCount = snapshot?.alerts?.length ?? 0;
    const statusRank = snapshot?.status === 'alarm' ? 3 : snapshot?.status === 'warning' ? 2 : alertCount ? 1 : 0;
    return {
      loop,
      snapshot,
      overallScore,
      alertCount,
      riskRank: statusRank * 1000 + (1 - overallScore) * 100 + alertCount * 10,
    };
  }).sort((a, b) => b.riskRank - a.riskRank), [monitoringByLoopId, scopedLoops]);

  const dashboardStats = useMemo(() => {
    const snapshots = dashboardRows.map((row) => row.snapshot).filter(Boolean);
    const avgScore = snapshots.length
      ? snapshots.reduce((sum, item) => sum + (item?.overall_score ?? 0), 0) / snapshots.length
      : undefined;
    const warningCount = snapshots.filter((item) => item?.status === 'warning').length;
    const alarmCount = snapshots.filter((item) => item?.status === 'alarm' || item?.status === 'critical').length;
    const normalCount = snapshots.filter((item) => !item?.status || item.status === 'normal' || item.status === 'ok').length;
    const alertCount = snapshots.reduce((sum, item) => sum + (item?.alerts?.length ?? 0), 0);
    const dataStart = scopedLoops
      .map((loop) => loop.start_time)
      .filter(Boolean)
      .sort()[0];
    const sortedEndTimes = scopedLoops
      .map((loop) => loop.end_time)
      .filter(Boolean)
      .sort();
    const dataEnd = sortedEndTimes[sortedEndTimes.length - 1];
    return {
      avgScore,
      warningCount,
      alarmCount,
      normalCount,
      alertCount,
      dataStart,
      dataEnd,
    };
  }, [dashboardRows, scopedLoops]);

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
      message.error(`加载 LoopFeatures 失败：${String(error)}`);
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
    if (activeSub !== 'trend_spectrum') return;
    loadSeries(selectedLoopId, selectedLoop);
  }, [activeSub, isSettingsView, loadSeries, selectedLoop, selectedLoopId]);

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
        setEvents((prev) => [
          {
            id: Date.now() + Math.random(),
            label: eventLabel(e),
            detail: e.type === 'stage' && e.data ? JSON.stringify(e.data) : undefined,
          },
          ...prev,
        ].slice(0, 30));

        if (e.type === 'session_start') {
          setTaskId(e.task_id);
          return;
        }

        if (e.type === 'stage') {
          setTaskCurrentStage(e.stage);
          setTaskStageStatus((prev) => ({ ...prev, [e.stage]: e.status }));
          if (e.status === 'running') {
            const runningPayload = e.data && typeof e.data === 'object' && !Array.isArray(e.data)
              ? e.data as Record<string, unknown>
              : {};
            setTaskStageRunningData((prev) => ({
              ...prev,
              [e.stage]: {
                ...(prev[e.stage] ?? {}),
                ...runningPayload,
              },
            }));
          }
          if (e.status === 'done') {
            // done 之后清掉 running 子状态，让 stepStatus 不再误判为"运行中"。
            setTaskStageRunningData((prev) => {
              if (!(e.stage in prev)) return prev;
              const next = { ...prev };
              delete next[e.stage];
              return next;
            });
          }
          if (e.status === 'done' && e.data) {
            const nextStageData = e.data && typeof e.data === 'object' && !Array.isArray(e.data)
              ? e.data as Record<string, unknown>
              : {};
            setTaskStageData((prev) => ({
              ...prev,
              [e.stage]: {
                ...(prev[e.stage] ?? {}),
                ...nextStageData,
              },
            }));
            if (e.stage === 'window_selection') {
              setTaskWindowSelection(e.data as unknown as WindowSelectionMeta);
            } else if (e.stage === 'model_review') {
              setTaskModelReview(e.data as unknown as ModelReviewMeta);
            } else if (e.stage === 'identification_refinement') {
              const nextRefinement = e.data as unknown as IdentificationRefinementMeta;
              setTaskRefinements((prev) => [
                ...prev.filter((item) => item.round !== nextRefinement.round),
                nextRefinement,
              ].sort((a, b) => a.round - b.round));
            } else if (e.stage === 'identification') {
              const round = typeof e.data.round === 'number' ? e.data.round : 0;
              const attempts = ((e.data.attempts as IdentificationAttempt[] | undefined) ?? []).map((attempt) => ({
                ...attempt,
                round: typeof attempt.round === 'number' ? attempt.round : round,
              }));
              setTaskAttempts((prev) => [
                ...prev.filter((attempt) => (attempt.round ?? 0) !== round),
                ...attempts,
              ].sort((a, b) => {
                const roundDiff = (a.round ?? 0) - (b.round ?? 0);
                if (roundDiff !== 0) return roundDiff;
                return (b.fit_score ?? -1e12) - (a.fit_score ?? -1e12);
              }));
            }
          }
          return;
        }

        if (e.type === 'llm_thinking') {
          setTaskThinking((prev) => [
            ...prev.filter((item) => !(item.stage === e.stage && item.round === e.round)),
            e,
          ]);
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
    yTitle: string,
    colors: string[],
  ) => (
    <div className="chart-shell">
      <Line
        height={height}
        data={data}
        xField="t"
        yField="value"
        colorField="series"
        theme="classicDark"
        color={colors}
        scale={{ color: { range: colors } }}
        style={{ lineWidth: 2.1 }}
        padding={[34, 32, 84, 76]}
        axis={{
          x: {
            title: 'X 轴：时间 / 采样点',
            titleFill: '#d8e8ff',
            titleFontSize: 13,
            titleFontWeight: 700,
            labelFill: '#a9c0de',
            labelFontSize: 11,
            labelAutoHide: true,
            labelAutoRotate: true,
            lineStroke: '#3b5068',
            tickStroke: '#3b5068',
          },
          y: {
            title: yTitle,
            titleFill: '#d8e8ff',
            titleFontSize: 13,
            titleFontWeight: 700,
            labelFill: '#a9c0de',
            labelFontSize: 12,
            lineStroke: '#3b5068',
            tickStroke: '#3b5068',
            gridStroke: '#223247',
            gridLineDash: [4, 4],
          },
        }}
        legend={{
          color: {
            position: 'top',
            itemLabelFill: '#d8e8ff',
            itemLabelFontSize: 13,
            itemLabelFontWeight: 600,
            markerSize: 10,
          },
        }}
        slider={{
          height: 28,
          textStyle: { fill: '#b8cbe5' },
          trendCfg: { lineStyle: { stroke: colors[0] ?? '#35a7ff' } },
          handlerStyle: { fill: '#16263a', stroke: '#7fb8ff' },
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
            theme="classicDark"
            color={['#35a7ff', '#28d7c5', '#ff9f43']}
            scale={{ color: { range: ['#35a7ff', '#28d7c5', '#ff9f43'] } }}
            style={{ lineWidth: 2.1 }}
            padding={[34, 32, 84, 76]}
            axis={{
              x: {
                title: 'X 轴：时间 / 采样点',
                titleFill: '#d8e8ff',
                titleFontSize: 13,
                titleFontWeight: 700,
                labelFill: '#a9c0de',
                labelFontSize: 11,
                labelAutoHide: true,
                labelAutoRotate: true,
                lineStroke: '#3b5068',
                tickStroke: '#3b5068',
              },
              y: {
                title: 'Y 轴：PV / SV / MV 数值',
                titleFill: '#d8e8ff',
                titleFontSize: 13,
                titleFontWeight: 700,
                labelFill: '#a9c0de',
                labelFontSize: 12,
                lineStroke: '#3b5068',
                tickStroke: '#3b5068',
                gridStroke: '#223247',
                gridLineDash: [4, 4],
              },
            }}
            legend={{
              color: {
                position: 'top',
                itemLabelFill: '#d8e8ff',
                itemLabelFontSize: 13,
                itemLabelFontWeight: 600,
                markerSize: 10,
              },
            }}
            slider={{
              height: 28,
              textStyle: { fill: '#b8cbe5' },
              trendCfg: { lineStyle: { stroke: '#35a7ff' } },
              handlerStyle: { fill: '#16263a', stroke: '#7fb8ff' },
            }}
            xAxis={{
              type: series?.x_axis === 'timestamp' ? 'timeCat' : 'linear',
              title: { text: 'X 轴：时间 / 采样点', style: { fill: '#d8e8ff', fontSize: 13, fontWeight: 700 } },
              label: { autoHide: true, autoRotate: true, style: { fill: '#9fb6d6', fontSize: 11 } },
              line: { style: { stroke: '#3b5068' } },
              tickLine: { style: { stroke: '#3b5068' } },
            }}
            yAxis={{
              title: { text: 'Y 轴：PV / SV / MV 数值', style: { fill: '#d8e8ff', fontSize: 13, fontWeight: 700 } },
              label: { style: { fill: '#9fb6d6', fontSize: 12 } },
              line: { style: { stroke: '#3b5068' } },
              tickLine: { style: { stroke: '#3b5068' } },
              grid: { line: { style: { stroke: '#223247', lineDash: [4, 4] } } },
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
            { title: '算法族 provider', dataIndex: 'family', width: 150, render: translateWindowAlgorithmFamily },
            { title: '执行策略', dataIndex: 'state', width: 120, render: (value) => <Tag color={value === 'preferred' ? 'green' : value === 'deprioritized' ? 'orange' : value === 'disabled' ? 'red' : 'blue'}>{translatePolicyState(value)}</Tag> },
            { title: '实际消费字段', dataIndex: 'consumed_policy_field_labels', render: (_value, row) => (row.consumed_policy_fields?.length ? row.consumed_policy_fields.map((field, index) => translatePolicyFieldName(field, row.consumed_policy_field_labels?.[index])).join('、') : '-') },
            { title: '原因', dataIndex: 'reason', ellipsis: true },
          ]}
        />
      </Space>
    );
  };

  const renderTaskStageSummary = (stage: string, data?: Record<string, unknown>) => {
    if (!data) return '等待执行';
    if (stage === 'data_analysis') {
      // data_analysis 只负责画像；候选窗口数搬到 window_selection 阶段。
      const dt = data.sampling_time as number | undefined;
      const dtTxt = typeof dt === 'number' ? `${dt}s` : '-';
      const profile = data.data_profile as { data_profile?: { duration_h?: number } } | undefined;
      const dur = profile?.data_profile?.duration_h;
      const durTxt = typeof dur === 'number' ? ` / ${dur.toFixed(1)} h` : '';
      return `${data.data_points ?? '-'} 点 / 采样 ${dtTxt}${durTxt}`;
    }
    if (stage === 'ontology_policy') {
      const src = data.source as string | undefined;
      const ontologySrc = data.ontology_source as string | undefined;
      const conf = typeof data.confidence === 'number' ? `${Math.round(data.confidence * 100)}%` : '-';
      const srcLabel = src === 'llm' ? 'LLM 改写策略' : src === 'default' ? '默认策略' : src ?? '-';
      const ontLabel = ontologySrc === 'mcp' ? 'MCP 本体已注入'
        : ontologySrc === 'frontend' ? '前端注入'
        : ontologySrc === 'default' ? '无本体上下文'
        : ontologySrc ?? '-';
      return `${srcLabel} · ${ontLabel} · 置信度 ${conf}`;
    }
    if (stage === 'window_selection') {
      const mode = data.mode as string | undefined;
      const modeLabel = mode === 'llm' ? 'LLM 选窗'
        : mode === 'fallback_deterministic' ? 'LLM 失败回退确定性'
        : mode === 'deterministic' ? '确定性选窗'
        : mode === 'user_override' ? '工程师手动指定'
        : mode === 'blocked' ? '阻断'
        : mode ?? '-';
      // 候选/可用窗口数现在是 window_selection 阶段的产物
      const candidates = (data.candidate_window_count as number | undefined)
        ?? (data.policy_adjusted_candidate_windows as number | undefined);
      const usable = data.policy_adjusted_usable_windows as number | undefined;
      const agreedRaw = data.agreed_with_deterministic;
      const agreed = typeof agreedRaw === 'boolean' ? (agreedRaw ? '与算法一致' : '与算法分歧') : null;
      const evidenceCount = Array.isArray(data.ontology_evidence) ? (data.ontology_evidence as unknown[]).length : 0;
      const judgementCount = Array.isArray(data.window_judgements) ? (data.window_judgements as unknown[]).length : 0;
      const parts: string[] = [];
      if (candidates !== undefined) parts.push(`候选 ${candidates} / 可用 ${usable ?? '-'}`);
      parts.push(`${modeLabel} #${data.chosen_index ?? '-'}（算法 #${data.deterministic_index ?? '-'}）`);
      if (agreed) parts.push(agreed);
      if (evidenceCount) parts.push(`本体证据 ${evidenceCount} 项`);
      if (judgementCount) parts.push(`窗口判断 ${judgementCount} 项`);
      return parts.join(' · ');
    }
    if (stage === 'identification') {
      const r2 = typeof data.r2_score === 'number' ? data.r2_score.toFixed(3) : '-';
      const confidence = typeof data.confidence === 'number' ? `${Math.round(data.confidence * 100)}%` : '-';
      return `R${data.round ?? 0} ${data.model_type ?? '-'} / R²=${r2} / 置信度 ${confidence}`;
    }
    if (stage === 'model_review') {
      return `${data.verdict ?? '-'}：${data.reason ?? ''}`;
    }
    if (stage === 'identification_refinement') {
      return `${data.retry ? '继续重试' : '停止重试'}：${data.rationale ?? ''}`;
    }
    if (stage === 'tuning') {
      return `${data.strategy ?? '-'} / Kp=${formatNumber(data.Kp as number | undefined, 3)}`;
    }
    if (stage === 'evaluation') {
      return `${data.passed ? '可上线' : '需优化'} / 综合 ${formatNumber(data.final_rating as number | undefined, 1)}`;
    }
    return '已完成';
  };

  const renderTaskDashboard = () => {
    const activeStep = taskStatus === 'done'
      ? TUNING_STAGE_KEYS.length
      : Math.max(TUNING_STAGE_KEYS.indexOf(taskCurrentStage ?? ''), 0);
    const idStage = taskStageData.identification;
    const tuningStage = taskStageData.tuning;
    const evaluationStage = taskStageData.evaluation;
    const result = taskResult;
    // result.evaluation 在 stop_after="window_selection" / "identification" 早停模式下会是 null，
    // 不能直接 result?.evaluation.passed —— optional chain 只挡 result 本身为空。
    const evaluationPassed = (evaluationStage?.passed as boolean | undefined) ?? result?.evaluation?.passed;
    const visibleEvents = rawLogExpanded ? events : events.slice(0, 8);
    const scoreColor = (score?: number) => {
      if ((score ?? 0) >= 8) return '#22a06b';
      if ((score ?? 0) >= 6) return '#f59e0b';
      return '#f04438';
    };
    const stageCards = TUNING_STAGE_KEYS.map((stage, index) => {
      const rawStatus = taskStageStatus[stage];
      const derivedStatus = taskStatus === 'done' || rawStatus === 'done' || taskStageData[stage]
        ? 'done'
        : rawStatus === 'running'
          ? 'running'
          : 'wait';
      return {
        stage,
        index,
        status: derivedStatus,
        label: TUNING_STAGE_LABELS[stage],
        summary: derivedStatus === 'running' ? '运行中...' : renderTaskStageSummary(stage, taskStageData[stage]),
        isCurrent: taskCurrentStage === stage && taskStatus === 'running',
      };
    });

    return (
      <div className="task-dashboard">
        <section className="agent-panel task-hero">
          <div>
            <div className="panel-title">整定任务驾驶舱</div>
            <Typography.Text type="secondary">
              把后端流式事件拆成可读过程：数据、窗口、辨识、LLM 评审、精修、整定和评估都会沉淀在这里。
            </Typography.Text>
          </div>
          <div className="task-hero-actions">
            <Tag color={taskStatus === 'running' ? 'processing' : taskStatus === 'done' ? 'green' : taskStatus === 'error' ? 'red' : 'default'}>
              {taskStatus === 'running' ? '运行中' : taskStatus === 'done' ? '已完成' : taskStatus === 'error' ? '异常/已停止' : '未开始'}
            </Tag>
            {taskId && <Tag color="blue">任务 ID：{taskId}</Tag>}
            {taskStartedAt && <Tag>开始：{taskStartedAt}</Tag>}
            {running && <Button danger onClick={handleStopTune}>停止任务</Button>}
          </div>
        </section>

        {taskError && (
          <Alert type="error" showIcon message="任务未正常完成" description={taskError} />
        )}

        <section className="agent-panel task-stage-panel">
          <div className="panel-toolbar">
            <div>
              <div className="panel-title">流程节点</div>
              <Typography.Text type="secondary">
                长摘要改为节点卡片展示，LLM 评审和精修建议会保留关键判断，不再被横向步骤截断。
              </Typography.Text>
            </div>
            <Tag color={taskStatus === 'running' ? 'processing' : taskStatus === 'done' ? 'green' : taskStatus === 'error' ? 'red' : 'default'}>
              {taskStatus === 'idle'
                ? `等待 0 / ${TUNING_STAGE_KEYS.length}`
                : `当前 ${Math.min(taskStatus === 'done' ? TUNING_STAGE_KEYS.length : activeStep + 1, TUNING_STAGE_KEYS.length)} / ${TUNING_STAGE_KEYS.length}`}
            </Tag>
          </div>
          <div className="task-stage-grid">
            {stageCards.map((item) => (
              <div
                key={item.stage}
                className={`task-stage-card is-${item.status}${item.isCurrent ? ' is-current' : ''}`}
              >
                <div className="stage-index">{item.status === 'done' ? '✓' : item.index + 1}</div>
                <div className="stage-body">
                  <div className="stage-title-row">
                    <strong>{item.label}</strong>
                    <Tag color={item.status === 'running' ? 'processing' : item.status === 'done' ? 'green' : 'default'}>
                      {item.status === 'running' ? '运行' : item.status === 'done' ? '完成' : '等待'}
                    </Tag>
                  </div>
                  <p title={item.summary}>{item.summary}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        <div className="task-kpi-grid">
          <div className="task-kpi-card">
            <span>候选窗口</span>
            {/* 候选/可用窗口数现在来自 window_selection 阶段，不再来自 data_analysis */}
            <strong>{(taskWindowSelection?.candidate_window_count
              ?? taskWindowSelection?.policy_adjusted_candidate_windows
              ?? '-') as number | string}</strong>
            <em>可用 {(taskWindowSelection?.policy_adjusted_usable_windows
              ?? '-') as number | string} 个</em>
          </div>
          <div className="task-kpi-card">
            <span>辨识模型</span>
            {/* result.model / pid_params / evaluation 在 stop_after 早停模式下都可能是 null，
                optional chain 必须穿到第二级。 */}
            <strong>{idStage?.model_type as string ?? result?.model?.model_type ?? '-'}</strong>
            <em>R² {formatNumber((idStage?.r2_score as number | undefined) ?? result?.model?.r2_score, 3)}</em>
          </div>
          <div className="task-kpi-card">
            <span>推荐策略</span>
            <strong>{tuningStage?.strategy as string ?? result?.pid_params?.strategy ?? '-'}</strong>
            <em>Kp {formatNumber((tuningStage?.Kp as number | undefined) ?? result?.pid_params?.Kp, 3)}</em>
          </div>
          <div className="task-kpi-card">
            <span>综合评分</span>
            <strong>{formatNumber((evaluationStage?.final_rating as number | undefined) ?? result?.evaluation?.final_rating, 1)}</strong>
            <em>{evaluationPassed === undefined ? '等待评估' : evaluationPassed ? '可以上线' : '需要优化'}</em>
          </div>
        </div>

        <section className="agent-panel">
          <div className="panel-toolbar">
            <div>
              <div className="panel-title">本体查询与上下文</div>
              <Typography.Text type="secondary">
                后端向本体 / MCP 提出的问题、来源以及返回内容；窗口策略和 LLM 选窗都基于此。
              </Typography.Text>
            </div>
            {taskWindowSelection ? (
              <Tag color={taskWindowSelection.ontology_mcp_error ? 'red' : taskWindowSelection.ontology_context_source === 'mcp' ? 'green' : 'default'}>
                {taskWindowSelection.ontology_mcp_error ? '本体查询失败' :
                 taskWindowSelection.ontology_context_source === 'mcp' ? `MCP 已注入 ${taskWindowSelection.ontology_mcp_content_chars ?? '-'} 字` :
                 '无本体上下文'}
              </Tag>
            ) : null}
          </div>
          {taskWindowSelection ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                <Descriptions.Item label="本体来源">{taskWindowSelection.ontology_context_source ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="MCP 服务">{taskWindowSelection.ontology_mcp_server ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="MCP 工具">{taskWindowSelection.ontology_mcp_tool ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="返回字数">{taskWindowSelection.ontology_mcp_content_chars ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="查询问题" span={4}>{taskWindowSelection.ontology_mcp_query ?? '-'}</Descriptions.Item>
                {taskWindowSelection.ontology_mcp_error ? (
                  <Descriptions.Item label="失败原因" span={4}>
                    <Typography.Text type="danger">{taskWindowSelection.ontology_mcp_error}</Typography.Text>
                  </Descriptions.Item>
                ) : null}
              </Descriptions>
              {taskWindowSelection.ontology_mcp_content_raw || taskWindowSelection.ontology_mcp_content_preview ? (
                <Collapse
                  items={[{
                    key: 'ontology-raw',
                    label: '本体 / MCP 返回原文',
                    children: (
                      <Typography.Paragraph className="thinking-text">
                        {taskWindowSelection.ontology_mcp_content_raw || taskWindowSelection.ontology_mcp_content_preview}
                      </Typography.Paragraph>
                    ),
                  }]}
                />
              ) : null}
            </Space>
          ) : <Empty description="等待本体检索结果" />}
        </section>

        <div className="panel-grid">
          <section className="agent-panel">
            <div className="panel-title">窗口选择</div>
            {taskWindowSelection ? (
              <Descriptions column={1} size="small">
                <Descriptions.Item label="选择模式">{taskWindowSelection.mode}</Descriptions.Item>
                <Descriptions.Item label="最终窗口">#{taskWindowSelection.chosen_index}</Descriptions.Item>
                <Descriptions.Item label="算法窗口">#{taskWindowSelection.deterministic_index}</Descriptions.Item>
                <Descriptions.Item label="算法分数">{formatNumber(taskWindowSelection.deterministic_score, 3)}</Descriptions.Item>
                <Descriptions.Item label="选择理由">{taskWindowSelection.reasoning || '-'}</Descriptions.Item>
              </Descriptions>
            ) : <Empty description="等待窗口选择结果" />}
          </section>

          <section className="agent-panel">
            <div className="panel-title">LLM 评审与精修</div>
            {taskModelReview ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <Alert
                  type={taskModelReview.verdict === 'accept' ? 'success' : 'warning'}
                  showIcon
                  message={`评审结论：${taskModelReview.verdict}`}
                  description={taskModelReview.reason}
                />
                {!!taskModelReview.concerns?.length && (
                  <List
                    size="small"
                    header="具体担忧"
                    bordered
                    dataSource={taskModelReview.concerns}
                    renderItem={(item) => <List.Item>{item}</List.Item>}
                  />
                )}
                {!!taskRefinements.length && (
                  <List
                    size="small"
                    header="精修建议"
                    bordered
                    dataSource={taskRefinements}
                    renderItem={(item) => (
                      <List.Item>
                        R{item.round}：{item.retry ? '继续重试' : '放弃重试'}
                        {item.source ? `；来源 ${item.source === 'deterministic_algorithm_policy' ? '确定性算法族策略' : item.source}` : ''}
                        ；{item.rationale}
                        {item.recommended_algorithm_label || item.recommended_algorithm ? `；推荐算法族 ${item.recommended_algorithm_label || item.recommended_algorithm}` : ''}
                        {item.recommended_window_source ? `；推荐窗口 ${item.recommended_window_source}` : ''}
                        {item.force_model_types?.length ? `；模型池 ${item.force_model_types.join(', ')}` : ''}
                        {item.force_window_index !== undefined && item.force_window_index !== null ? `；窗口 #${item.force_window_index}` : ''}
                      </List.Item>
                    )}
                  />
                )}
              </Space>
            ) : <Empty description="等待 LLM 评审" />}
          </section>
        </div>

        <section className="agent-panel">
          <div className="panel-toolbar">
            <div>
              <div className="panel-title">辨识过程：所有轮次 attempts</div>
              <Typography.Text type="secondary">每行代表某一轮中“候选窗口 × 候选模型”的一次拟合尝试，按轮次和 fit_score 展示。</Typography.Text>
            </div>
            <Tag color="processing">{taskAttempts.length} 次尝试</Tag>
          </div>
          {taskAlgorithmComparison.length ? (
            <Table
              className="detail-block"
              size="small"
              pagination={false}
              rowKey={(row) => `${row.algorithm}-${row.window_source}-${row.model_type}`}
              dataSource={taskAlgorithmComparison as unknown as Array<Record<string, unknown>>}
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
          {taskAttempts.length ? (
            <Table<IdentificationAttempt>
              size="small"
              rowKey={(row, index) => `${row.round ?? 0}-${row.model_type}-${row.window_source}-${index ?? 0}`}
              dataSource={taskAttempts}
              pagination={{ pageSize: 8 }}
              onRow={(row) => ({
                onClick: () => {
                  if (row.fit_preview?.points?.length) setSelectedFitAttemptKey(attemptFitKey(row));
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
          ) : <Empty description="任务运行后会显示所有辨识尝试" />}
        </section>

        {/* stop_after="window_selection" / "identification" 早停模式下 result.pid_params /
            result.evaluation 都是 null，必须分别 gate 防止 .xxx 访问爆 TypeError。 */}
        {result && result.pid_params && (
          <>
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">PID 参数结果</div>
                  <Typography.Text type="secondary">推荐策略与所有候选策略对比，PB=100/Kp。</Typography.Text>
                </div>
                <Tag color="cyan">{result.pid_params.strategy}</Tag>
              </div>
              <Descriptions column={6} size="small" className="detail-block">
                <Descriptions.Item label="Kp">{formatNumber(result.pid_params.Kp, 4)}</Descriptions.Item>
                <Descriptions.Item label="PB">{result.pid_params.Kp > 0 ? `${formatNumber(100 / result.pid_params.Kp, 2)}%` : '-'}</Descriptions.Item>
                <Descriptions.Item label="Ki">{formatNumber(result.pid_params.Ki, 6)}</Descriptions.Item>
                <Descriptions.Item label="Kd">{formatNumber(result.pid_params.Kd, 4)}</Descriptions.Item>
                <Descriptions.Item label="Ti">{formatNumber(result.pid_params.Ti, 2)}s</Descriptions.Item>
                <Descriptions.Item label="Td">{formatNumber(result.pid_params.Td, 2)}s</Descriptions.Item>
              </Descriptions>
              {!!result.pid_params.candidates?.length && (
                <Table<StrategyCandidate>
                  size="small"
                  rowKey="strategy"
                  dataSource={result.pid_params.candidates}
                  pagination={false}
                  columns={[
                    { title: '策略', dataIndex: 'strategy', render: (value, row) => <Space><Tag color={row.is_recommended ? 'green' : 'blue'}>{value}</Tag>{row.is_recommended && <Tag color="gold">推荐</Tag>}</Space> },
                    { title: 'Kp', dataIndex: 'Kp', render: (value) => formatNumber(value, 4) },
                    { title: 'PB(%)', render: (_, row) => row.Kp > 0 ? formatNumber(100 / row.Kp, 2) : '-' },
                    { title: 'Ki', dataIndex: 'Ki', render: (value) => formatNumber(value, 6) },
                    { title: 'Kd', dataIndex: 'Kd', render: (value) => formatNumber(value, 4) },
                    { title: 'Ti(s)', dataIndex: 'Ti', render: (value) => formatNumber(value, 2) },
                    { title: 'Td(s)', dataIndex: 'Td', render: (value) => formatNumber(value, 2) },
                    { title: '说明', dataIndex: 'description', ellipsis: true },
                  ]}
                />
              )}
            </section>

            {result.evaluation && (
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">性能评估</div>
                  <Typography.Text type="secondary">闭环仿真、自检封顶和上线建议都集中在这里。</Typography.Text>
                </div>
                <Tag color={result.evaluation.passed ? 'green' : 'red'}>{result.evaluation.passed ? '可以上线' : '需要优化'}</Tag>
              </div>
              <div className="task-score-grid">
                {[
                  ['性能评分', result.evaluation.performance_score],
                  ['综合评分', result.evaluation.final_rating],
                  ['就绪评分', result.evaluation.readiness_score],
                  ['鲁棒评分', result.evaluation.robustness_score],
                ].map(([label, value]) => (
                  <div key={label} className="task-score-card">
                    <Progress
                      type="circle"
                      percent={Number(value) * 10}
                      format={() => formatNumber(Number(value), 1)}
                      strokeColor={scoreColor(Number(value))}
                      size={72}
                    />
                    <span>{label}</span>
                  </div>
                ))}
              </div>
              <Descriptions column={4} size="small" className="detail-block">
                <Descriptions.Item label="稳定性">
                  <Tag color={result.evaluation.is_stable ? 'green' : 'red'}>{result.evaluation.is_stable ? '稳定' : '不稳定'}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="超调量">{formatNumber(result.evaluation.overshoot_percent, 1)}%</Descriptions.Item>
                <Descriptions.Item label="调节时间">{formatNumber(result.evaluation.settling_time_s, 1)}s</Descriptions.Item>
                <Descriptions.Item label="稳态误差">{formatNumber(result.evaluation.steady_state_error, 2)}%</Descriptions.Item>
                <Descriptions.Item label="振荡次数">{result.evaluation.oscillation_count}</Descriptions.Item>
                <Descriptions.Item label="MV 饱和">{formatNumber(result.evaluation.mv_saturation_pct, 1)}%</Descriptions.Item>
                <Descriptions.Item label="Reality Score">{formatNumber(result.evaluation.reality_check_score, 1)}</Descriptions.Item>
                <Descriptions.Item label="典型 T">{result.evaluation.reality_check_typical_T ? `${result.evaluation.reality_check_typical_T}s` : '-'}</Descriptions.Item>
              </Descriptions>
              {(result.evaluation.reality_check_diverged || !!result.evaluation.score_caps_applied?.length) && (
                <Alert
                  className="agent-alert"
                  type="error"
                  showIcon
                  message="评估自检触发"
                  description={[
                    result.evaluation.reality_check_diverged ? `Reality check 认为名义模型与典型回路仿真差异过大，评分 ${formatNumber(result.evaluation.reality_check_score, 1)}` : '',
                    ...(result.evaluation.score_caps_applied ?? []),
                  ].filter(Boolean).join('；')}
                />
              )}
              <Alert type={result.evaluation.passed ? 'success' : 'warning'} showIcon message={result.evaluation.recommendation} />
            </section>
            )}
          </>
        )}

        {!!taskThinking.length && (
          <section className="agent-panel">
            <div className="panel-title">LLM 判断依据</div>
            <Collapse
              items={taskThinking.map((item, index) => ({
                key: `${item.stage}-${item.round ?? 'x'}-${index}`,
                label: `${TUNING_STAGE_LABELS[item.stage] ?? item.stage}${item.round !== undefined ? ` R${item.round}` : ''} · ${item.model} · ${item.reasoning_content.length} 字`,
                children: (
                  <Typography.Paragraph className="thinking-text">
                    {item.reasoning_content || item.raw_text}
                  </Typography.Paragraph>
                ),
              }))}
            />
          </section>
        )}

        <section className="agent-panel raw-log-panel">
          <div className="panel-toolbar">
            <div>
              <div className="panel-title">原始事件日志</div>
              <Typography.Text type="secondary">
                {events.length
                  ? `共 ${events.length} 条，${rawLogExpanded ? '已展开全部' : '默认显示最近 8 条'}`
                  : '保留后端 SSE 原始事件，便于排查'}
              </Typography.Text>
            </div>
            {events.length > 8 && (
              <Button size="small" onClick={() => setRawLogExpanded((prev) => !prev)}>
                {rawLogExpanded ? '收起日志' : '展开全部'}
              </Button>
            )}
          </div>
          {events.length ? (
            <div className={`event-log-box${rawLogExpanded ? ' is-expanded' : ''}`}>
              {visibleEvents.map((item, index) => (
                <div className="event-log-item" key={item.id}>
                  <div className="event-log-head">
                    <Tag color="blue">#{events.length - index}</Tag>
                    <Typography.Text strong>{item.label}</Typography.Text>
                  </div>
                  {item.detail && (
                    <pre className="event-detail">{formatEventDetail(item.detail)}</pre>
                  )}
                </div>
              ))}
            </div>
          ) : <Alert type="info" showIcon message="点击发起整定后，这里会保留后端 SSE 原始事件，便于排查。" />}
        </section>
      </div>
    );
  };

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
                  先选择需要评审的回路；点击开始后，系统按“数据画像 → 本体检索 → 策略生成 → 算法族运行 → LLM 评审 → 准入结论”的顺序执行。
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
                description={taskError || '页面按后端事件实时更新；当前后端将策略、本体、算法族与 LLM 汇总在 window_selection 阶段返回。'}
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
                    <div className="panel-title">1 数据画像：LoopFeatures 原始指标</div>
                    <Typography.Text type="secondary">这里只展示基础画像和原始统计，不再混入窗口算法先验判断。</Typography.Text>
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
                    <Typography.Text type="secondary">展示后端向本体/MCP提出的问题、来源和返回内容，作为后续策略生成依据。</Typography.Text>
                  </div>
                  <Tag color={windowFlowStatusColor(stepStatus.ontology)}>{windowFlowStatusText(stepStatus.ontology)}</Tag>
                </div>
                {taskWindowSelection ? (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                      <Descriptions.Item label="本体来源">{taskWindowSelection.ontology_context_source ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="MCP服务">{taskWindowSelection.ontology_mcp_server ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="MCP工具">{taskWindowSelection.ontology_mcp_tool ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="返回字数">{taskWindowSelection.ontology_mcp_content_chars ?? '-'}</Descriptions.Item>
                      <Descriptions.Item label="查询问题" span={4}>{taskWindowSelection.ontology_mcp_query ?? '-'}</Descriptions.Item>
                    </Descriptions>
                    <Collapse
                      items={[
                        {
                          key: 'ontology-raw',
                          label: '本体/MCP返回原文',
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
                  <Empty description={stepStatus.ontology === 'running' ? '正在查询本体/MCP上下文...' : '等待本体检索'} />
                )}
              </section>

              <section className="agent-panel">
                <div className="panel-toolbar">
                  <div>
                    <div className="panel-title">3 策略生成</div>
                    <Typography.Text type="secondary">明确每个策略字段是被算法族消费、作为下游提示，还是仅用于展示/审计。</Typography.Text>
                  </div>
                  <Tag color={windowFlowStatusColor(stepStatus.policy)}>{windowFlowStatusText(stepStatus.policy)}</Tag>
                </div>
                {renderWindowPolicyTables(windowPolicy)}
              </section>

              <section className="agent-panel">
                <div className="panel-toolbar">
                  <div>
                    <div className="panel-title">4 算法族输出的候选窗口</div>
                    <Typography.Text type="secondary">按策略驱动各窗口 provider 后，展示每个算法族是否执行、策略状态和窗口评分修正。</Typography.Text>
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
                        { title: '算法族 provider', dataIndex: 'family', render: translateWindowAlgorithmFamily },
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
                    <div className="panel-title">5 LLM 评审</div>
                    <Typography.Text type="secondary">LLM 结合回路画像、本体证据、策略和候选窗口逐项判断，给出最终窗口建议。</Typography.Text>
                  </div>
                  <Tag color={windowFlowStatusColor(stepStatus.llm)}>{windowFlowStatusText(stepStatus.llm)}</Tag>
                </div>
                {taskWindowSelection ? (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                      <Descriptions.Item label="选择模式">{taskWindowSelection.mode}</Descriptions.Item>
                      <Descriptions.Item label="LLM选中窗口">#{taskWindowSelection.chosen_index}</Descriptions.Item>
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
                          { title: 'LLM引用的本体证据', dataIndex: 'fact' },
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
                          label: `LLM 思维链 · ${item.model} · ${(item.reasoning_content || item.raw_text || '').length} 字`,
                          children: <Typography.Paragraph className="thinking-text">{item.reasoning_content || item.raw_text}</Typography.Paragraph>,
                        }))}
                      />
                    )}
                  </Space>
                ) : (
                  <Empty description={stepStatus.llm === 'running' ? '等待 LLM 评审窗口候选...' : '等待 LLM 评审'} />
                )}
              </section>

              <section className="agent-panel">
                <div className="panel-toolbar">
                  <div>
                    <div className="panel-title">6 准入结论</div>
                    <Typography.Text type="secondary">如果没有适合正式辨识的窗口，后续系统辨识应转为诊断性辨识或停止。</Typography.Text>
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
        const realRows = dashboardRows.filter((row) => row.snapshot);
        const compactTime = (value?: string | null) => value ? dayjs(value).format('MM-DD HH:mm') : '-';
        const compactDataRange = `${compactTime(dashboardStats.dataStart)} ~ ${compactTime(dashboardStats.dataEnd)}`;
        const typeCounts = scopedLoops.reduce<Record<string, number>>((acc, loop) => {
          const label = LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type ?? '未知';
          acc[label] = (acc[label] ?? 0) + 1;
          return acc;
        }, {});
        const assetCounts = scopedLoops.reduce<Record<string, number>>((acc, loop) => {
          const asset = assetNodes.find((node) => node.id === inferLoopAssetId(loop.loop_id));
          const label = asset?.name ?? '未归属';
          acc[label] = (acc[label] ?? 0) + 1;
          return acc;
        }, {});
        const alertSeverityCounts = realRows.reduce<Record<string, number>>((acc, row) => {
          (row.snapshot?.alerts ?? []).forEach((alert) => {
            const label = alert.severity || 'unknown';
            acc[label] = (acc[label] ?? 0) + 1;
          });
          return acc;
        }, {});
        const makeSlices = (items: Array<{ label: string; value: number; color: string }>) => {
          const total = items.reduce((sum, item) => sum + item.value, 0);
          return items.map((item) => ({ ...item, percent: total > 0 ? item.value / total : 0 }));
        };
        const conic = (items: Array<{ value: number; color: string }>) => {
          const total = items.reduce((sum, item) => sum + item.value, 0);
          if (!total) return 'conic-gradient(#26364d 0 100%)';
          let cursor = 0;
          return `conic-gradient(${items.map((item) => {
            const start = cursor;
            cursor += (item.value / total) * 100;
            return `${item.color} ${start}% ${cursor}%`;
          }).join(', ')})`;
        };
        const statusSlices = makeSlices([
          { label: '正常', value: dashboardStats.normalCount, color: '#22c55e' },
          { label: '关注', value: dashboardStats.warningCount, color: '#facc15' },
          { label: '告警', value: dashboardStats.alarmCount, color: '#ef4444' },
          { label: '待加载', value: pendingCount, color: '#64748b' },
        ]);
        const typePalette = ['#22c55e', '#facc15', '#60a5fa', '#a78bfa', '#fb923c', '#14b8a6'];
        const typeSlices = makeSlices(Object.entries(typeCounts).map(([label, value], index) => ({
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
        const topHealthyRows = [...realRows]
          .sort((a, b) => (b.snapshot?.overall_score ?? -1) - (a.snapshot?.overall_score ?? -1))
          .slice(0, 5);
        const abnormalRows = realRows
          .filter((row) => row.alertCount > 0 || row.snapshot?.status === 'warning' || row.snapshot?.status === 'alarm' || row.snapshot?.status === 'critical')
          .slice(0, 5);
        const alertRows = Object.entries(alertSeverityCounts)
          .map(([label, value], index) => ({ label, value, color: ['#ef4444', '#fb923c', '#facc15', '#60a5fa'][index % 4] }))
          .sort((a, b) => b.value - a.value);
        return (
          <div className="dashboard-cockpit">
            <section className="cockpit-header">
              <div>
                <h2>首页驾驶舱</h2>
                <span>基于历史回路清单与后端监控快照聚合，未接入的指标显示为待加载或暂无。</span>
              </div>
              <Space wrap>
                <Tag color={assetTagColor(selectedAssetNode?.type ?? 'factory')}>
                  {selectedAssetNode ? ASSET_TYPE_LABEL[selectedAssetNode.type] : '-'}
                </Tag>
                <span>范围：{selectedAssetPath.map((item) => item.name).join(' / ')}</span>
                <Button size="small" onClick={() => switchTo('settings', 'asset_directory')}>切换装置</Button>
              </Space>
            </section>

            <section className="cockpit-kpis">
              {[
                { label: '回路总数', value: scopedLoopStats.loopCount, suffix: '个', color: '#60a5fa', sub: `范围 ${compactDataRange}` },
                { label: '已监控回路', value: loadedCount, suffix: '个', color: '#22d3ee', sub: `覆盖率 ${formatPercentValue(scopedLoopStats.loopCount ? loadedCount / scopedLoopStats.loopCount : 0, 1)}` },
                { label: '正常回路', value: dashboardStats.normalCount, suffix: '个', color: '#22c55e', sub: `占比 ${formatPercentValue(dashboardStats.normalCount / loopCount, 1)}` },
                { label: '关注回路', value: dashboardStats.warningCount, suffix: '个', color: '#facc15', sub: `占比 ${formatPercentValue(dashboardStats.warningCount / loopCount, 1)}` },
                { label: '告警回路', value: dashboardStats.alarmCount, suffix: '个', color: '#ef4444', sub: `占比 ${formatPercentValue(dashboardStats.alarmCount / loopCount, 1)}` },
                { label: '监控均分', value: dashboardScore ?? '-', suffix: dashboardScore === undefined ? '' : '分', color: '#38bdf8', sub: dashboardScore === undefined ? '暂无快照' : '后端快照均值' },
                { label: '监控告警', value: dashboardStats.alertCount, suffix: '条', color: '#a78bfa', sub: warningTotal ? '需要处理' : '当前平稳' },
              ].map((item) => (
                <div className="cockpit-kpi" key={item.label}>
                  <i style={{ background: item.color }} />
                  <div>
                    <span>{item.label}</span>
                    <strong>{item.value}<em>{item.suffix}</em></strong>
                    <small>{item.sub}</small>
                  </div>
                </div>
              ))}
            </section>

            <section className="cockpit-grid top">
              <div className="cockpit-card">
                <div className="cockpit-card-title">回路健康分布</div>
                <div className="cockpit-donut-row">
                  <div className="cockpit-donut" style={{ background: conic(statusSlices) }}>
                    <strong>{scopedLoopStats.loopCount}</strong>
                    <span>总回路数</span>
                  </div>
                  <div className="cockpit-legend">
                    {statusSlices.map((item) => (
                      <span key={item.label}><i style={{ background: item.color }} />{item.label}<b>{item.value}</b><em>{formatPercentValue(item.percent, 1)}</em></span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="cockpit-card">
                <div className="cockpit-card-title">回路按装置分布</div>
                <div className="cockpit-bars">
                  {assetRows.map((item) => (
                    <div className="cockpit-bar" key={item.label}>
                      <span>{item.label}</span>
                      <em><i style={{ width: `${Math.max(4, item.percent * 100)}%` }} /></em>
                      <b>{item.value} ({formatPercentValue(item.percent, 1)})</b>
                    </div>
                  ))}
                  {!assetRows.length && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无回路" />}
                </div>
              </div>

              <div className="cockpit-card">
                <div className="cockpit-card-title">回路类型分布</div>
                <div className="cockpit-donut-row">
                  <div className="cockpit-donut" style={{ background: conic(typeSlices) }}>
                    <strong>{scopedLoopStats.loopCount}</strong>
                    <span>总回路数</span>
                  </div>
                  <div className="cockpit-legend">
                    {typeSlices.map((item) => (
                      <span key={item.label}><i style={{ background: item.color }} />{item.label}<b>{item.value}</b><em>{formatPercentValue(item.percent, 1)}</em></span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="cockpit-card">
                <div className="cockpit-card-title">关键指标均值</div>
                <div className="cockpit-bars metric">
                  {indicatorRows.map((item) => {
                    const pct = item.value === undefined ? 0 : scorePercent(item.value);
                    return (
                      <div className="cockpit-bar" key={item.label}>
                        <span>{item.label}</span>
                        <em><i style={{ width: `${Math.max(0, Math.min(100, pct))}%`, background: item.color }} /></em>
                        <b>{item.value === undefined ? '-' : `${pct}%`}</b>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>

            <section className="cockpit-grid middle">
              <div className="cockpit-card wide">
                <div className="cockpit-card-title">性能评分 TOP5</div>
                <Table
                  className="cockpit-table"
                  size="small"
                  pagination={false}
                  rowKey={(row) => row.loop.loop_id}
                  dataSource={topHealthyRows}
                  columns={[
                    { title: '排名', width: 64, render: (_: unknown, __: unknown, index: number) => index + 1 },
                    {
                      title: '回路名称',
                      render: (_: unknown, row: (typeof dashboardRows)[number]) => (
                        <Button type="link" onClick={() => setSelectedLoopId(row.loop.loop_id)}>{row.loop.loop_id}</Button>
                      ),
                    },
                    { title: '类型', width: 100, render: (_: unknown, row: (typeof dashboardRows)[number]) => LOOP_TYPE_LABEL[row.loop.loop_type] ?? row.loop.loop_type },
                    { title: '健康评分', width: 110, render: (_: unknown, row: (typeof dashboardRows)[number]) => `${scorePercent(row.snapshot?.overall_score)}%` },
                    { title: '状态', width: 90, render: (_: unknown, row: (typeof dashboardRows)[number]) => <Tag color={monitoringStatusColor(row.snapshot?.status)}>{monitoringStatusText(row.snapshot?.status)}</Tag> },
                    { title: '操作', width: 90, render: (_: unknown, row: (typeof dashboardRows)[number]) => <Button size="small" onClick={() => { setSelectedLoopId(row.loop.loop_id); switchTo('monitor', 'loop_profile'); }}>查看</Button> },
                  ]}
                />
              </div>

              <div className="cockpit-card wide">
                <div className="cockpit-card-title">异常回路列表</div>
                <Table
                  className="cockpit-table"
                  size="small"
                  pagination={false}
                  rowKey={(row) => row.loop.loop_id}
                  dataSource={abnormalRows}
                  locale={{ emptyText: '当前监控快照未发现异常回路' }}
                  columns={[
                    { title: '回路名称', render: (_: unknown, row: (typeof dashboardRows)[number]) => row.loop.loop_id },
                    { title: '异常类型', width: 160, render: (_: unknown, row: (typeof dashboardRows)[number]) => row.snapshot?.alerts?.[0]?.type || monitoringStatusText(row.snapshot?.status) },
                    { title: '严重度', width: 100, render: (_: unknown, row: (typeof dashboardRows)[number]) => row.snapshot?.alerts?.[0]?.severity || row.snapshot?.status || '-' },
                    { title: '告警数', width: 90, render: (_: unknown, row: (typeof dashboardRows)[number]) => row.alertCount },
                    { title: '操作', width: 90, render: (_: unknown, row: (typeof dashboardRows)[number]) => <Button size="small" onClick={() => { setSelectedLoopId(row.loop.loop_id); switchTo('monitor', 'loop_profile'); }}>查看</Button> },
                  ]}
                />
              </div>

              <div className="cockpit-card alerts">
                <div className="cockpit-card-title">告警统计</div>
                <strong className="cockpit-alert-total">{dashboardStats.alertCount}</strong>
                <span>当前监控快照告警总数</span>
                <div className="cockpit-bars">
                  {alertRows.length ? alertRows.map((item) => (
                    <div className="cockpit-bar" key={item.label}>
                      <span>{item.label}</span>
                      <em><i style={{ width: `${Math.max(8, (item.value / Math.max(dashboardStats.alertCount, 1)) * 100)}%`, background: item.color }} /></em>
                      <b>{item.value}</b>
                    </div>
                  )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无告警" />}
                </div>
              </div>
            </section>

            <section className="cockpit-grid bottom">
              <div className="cockpit-card trend">
                <div className="cockpit-card-title">选中回路真实趋势</div>
                <Typography.Text type="secondary">{selectedLoop?.loop_id ?? '-'} · 来自后端 `/history/loops/{'{loop_id}'}/series`</Typography.Text>
                {renderTrend(300)}
              </div>
              <div className="cockpit-card">
                <div className="cockpit-card-title">选中回路监控快照</div>
                <Descriptions bordered size="small" column={2} className="industrial-descriptions cockpit-descriptions">
                  <Descriptions.Item label="监控状态"><Tag color={monitoringStatusColor(monitoring?.status)}>{monitoringStatusText(monitoring?.status)}</Tag></Descriptions.Item>
                  <Descriptions.Item label="综合分">{monitoring?.overall_score === undefined ? '-' : `${scorePercent(monitoring.overall_score)}%`}</Descriptions.Item>
                  <Descriptions.Item label="数据健康">{monitoring?.data_health?.score === undefined ? '-' : `${scorePercent(monitoring.data_health.score)}%`}</Descriptions.Item>
                  <Descriptions.Item label="稳定性">{monitoring?.stability?.score === undefined ? '-' : `${scorePercent(monitoring.stability.score)}%`}</Descriptions.Item>
                  <Descriptions.Item label="PV/MV行为">{monitoring?.pv_mv_behavior?.score === undefined ? '-' : `${scorePercent(monitoring.pv_mv_behavior.score)}%`}</Descriptions.Item>
                  <Descriptions.Item label="约束饱和">{monitoring?.constraints?.score === undefined ? '-' : `${scorePercent(monitoring.constraints.score)}%`}</Descriptions.Item>
                </Descriptions>
              </div>
              <div className="cockpit-card quick">
                <div className="cockpit-card-title">快捷操作</div>
                <button type="button" onClick={() => switchTo('tuning', 'tuning_task')}>新建整定任务</button>
                <button type="button" onClick={() => switchTo('monitor', 'loop_profile')}>查看回路画像</button>
                <button type="button" onClick={() => switchTo('monitor', 'trend_spectrum')}>趋势与频谱</button>
                <button type="button" onClick={() => switchTo('diagnostics', 'diagnosis_overview')}>进入诊断总览</button>
              </div>
            </section>
          </div>
        );
      }
      case 'loop_board':
        return (
          <section className="agent-panel">
            <div className="panel-toolbar">
              <div>
                <div className="panel-title">全局回路看板</div>
                <Typography.Text type="secondary">
                  当前选中：{selectedLoop?.loop_id ?? '-'} · 监控状态：
                  <Tag color={monitoringStatusColor(monitoring?.status)}>{monitoringStatusText(monitoring?.status)}</Tag>
                  综合分：{monitoring?.overall_score === undefined ? '-' : `${scorePercent(monitoring.overall_score)}%`}
                </Typography.Text>
              </div>
              <Space>
                <Tag color={monitoringStatusColor(monitoring?.status)}>
                  {monitoringStatusText(monitoring?.status)}
                </Tag>
                <Button icon={<SyncOutlined />} onClick={loadLoops} loading={loading}>刷新</Button>
              </Space>
            </div>
            <div className="kpi-grid compact-kpi">
              <Statistic
                title="监控综合分"
                value={monitoring?.overall_score === undefined ? '-' : scorePercent(monitoring.overall_score)}
                suffix={monitoring?.overall_score === undefined ? undefined : '%'}
              />
              <Statistic
                title="数据健康"
                value={monitoring?.data_health?.score === undefined ? '-' : scorePercent(monitoring.data_health.score)}
                suffix={monitoring?.data_health?.score === undefined ? undefined : '%'}
              />
              <Statistic
                title="PV/MV行为"
                value={monitoring?.pv_mv_behavior?.score === undefined ? '-' : scorePercent(monitoring.pv_mv_behavior.score)}
                suffix={monitoring?.pv_mv_behavior?.score === undefined ? undefined : '%'}
              />
              <Statistic title="监控告警" value={monitoringAlerts.length} suffix="条" />
            </div>
            {renderLoopTable()}
          </section>
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
                    description="当前先用于确认产品交互和层级结构；后续会补后端持久化、拖拽移动、批量导入和位号自动映射规则。"
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
          <div className="page-stack">
            <section className="agent-panel profile-panel compact-profile">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">单回路画像</div>
                  <Typography.Text type="secondary">集中展示资产信息、量程、采样、原始统计与约束饱和摘要；趋势曲线统一放到“趋势与频谱”。</Typography.Text>
                </div>
                <Space wrap>
                  <Select
                    showSearch
                    size="small"
                    style={{ minWidth: 300 }}
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
                    value={featureRangePreset}
                    onChange={(value) => setFeatureRangePreset(value)}
                    options={FEATURE_RANGE_OPTIONS.map((item) => ({ label: item.label, value: item.value }))}
                  />
                  {featureRangePreset === 'custom' && (
                    <DatePicker.RangePicker
                      size="small"
                      showTime
                      value={featureCustomRange}
                      onChange={(value) => setFeatureCustomRange(value)}
                    />
                  )}
                  <Button
                    size="small"
                    icon={<SyncOutlined />}
                    loading={featureLoading}
                    onClick={() => {
                      if (!selectedLoopId) return;
                      const params = buildFeatureRangeParams(selectedLoop);
                      loadLoopFeatures(selectedLoopId, params);
                      loadLoopMonitoring(selectedLoopId, params);
                    }}
                  >
                    刷新区间指标
                  </Button>
                  <Tag color="blue">{selectedLoop ? LOOP_TYPE_LABEL[selectedLoop.loop_type] ?? selectedLoop.loop_type : '-'}</Tag>
                  <Tag color={loopFeatures ? 'cyan' : 'default'}>{loopFeatures ? 'LoopFeatures 已加载' : 'LoopFeatures 待加载'}</Tag>
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
            <section className="agent-panel compact-facts">
              <div className="panel-title">LoopFeatures 原始统计</div>
              {loopFeatures ? (
                <Descriptions bordered size="small" column={4} className="industrial-descriptions">
                  <Descriptions.Item label="行数">{loopFeatures.data_profile?.row_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="有效行">{loopFeatures.data_profile?.valid_row_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="中位采样">{formatNumber(loopFeatures.data_profile?.sample_time_median_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="P95 间隔">{formatNumber(loopFeatures.data_profile?.sample_interval_p95_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="P99 间隔">{formatNumber(loopFeatures.data_profile?.sample_interval_p99_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="不规则采样">{formatPercentValue(loopFeatures.data_profile?.irregular_sample_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="长间隔数">{loopFeatures.data_profile?.long_gap_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="重复时间戳">{loopFeatures.data_profile?.duplicate_timestamp_count ?? '-'}</Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无 LoopFeatures 数据" />}
            </section>
            <section className="agent-panel compact-facts">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">数据质量指标</div>
                  <Typography.Text type="secondary">缺失、连续性、采样异常、噪声和离群点统一放在单回路画像中查看。</Typography.Text>
                </div>
                {assessment?.data_quality ? (
                  <Tag color={tagColor(assessment.data_quality.level)}>{assessment.data_quality.level}</Tag>
                ) : null}
              </div>
              {assessment ? (
                <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                  <Descriptions.Item label="缺失比例">{(assessment.data_quality.missing_ratio * 100).toFixed(2)}%</Descriptions.Item>
                  <Descriptions.Item label="连续性">{scorePercent(assessment.data_quality.continuity_score)}%</Descriptions.Item>
                  <Descriptions.Item label="噪声得分">{scorePercent(assessment.data_quality.noise_score)}%</Descriptions.Item>
                  <Descriptions.Item label="采样不规则">{formatPercentValue(monitoring?.data_health?.irregular_sample_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="长间隔">{monitoring?.data_health?.long_gap_count ?? 0} 个</Descriptions.Item>
                  <Descriptions.Item label="重复时间戳">{formatPercentValue(monitoring?.data_health?.duplicate_timestamp_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="PV噪声比">{formatPercentValue(monitoring?.data_health?.pv_noise_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="PV SNR">{formatNumber(monitoring?.data_health?.pv_snr_db, 2)} dB</Descriptions.Item>
                  <Descriptions.Item label="PV尖峰">{monitoring?.data_health?.pv_spike_count ?? 0} 个</Descriptions.Item>
                  <Descriptions.Item label="PV离群">{monitoring?.data_health?.pv_outlier_count ?? 0} 个</Descriptions.Item>
                  <Descriptions.Item label="MV离群">{monitoring?.data_health?.mv_outlier_count ?? 0} 个</Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无数据质量指标" />}
            </section>
            <section className="agent-panel compact-facts">
              <div className="panel-title">PV / MV 原始分布</div>
              {loopFeatures ? (
                <Descriptions bordered size="small" column={4} className="industrial-descriptions">
                  <Descriptions.Item label="PV 均值">{formatNumber(loopFeatures.pv_stats?.mean, 3)}</Descriptions.Item>
                  <Descriptions.Item label="PV 标准差">{formatNumber(loopFeatures.pv_stats?.std, 3)}</Descriptions.Item>
                  <Descriptions.Item label="PV 跨度">{formatNumber(loopFeatures.pv_stats?.span, 3)}</Descriptions.Item>
                  <Descriptions.Item label="PV P95跳变">{formatNumber(loopFeatures.pv_stats?.p95_abs_step, 3)}</Descriptions.Item>
                  <Descriptions.Item label="MV 均值">{formatNumber(loopFeatures.mv_stats?.mean, 3)}</Descriptions.Item>
                  <Descriptions.Item label="MV 标准差">{formatNumber(loopFeatures.mv_stats?.std, 3)}</Descriptions.Item>
                  <Descriptions.Item label="MV 跨度">{formatNumber(loopFeatures.mv_stats?.span, 3)}</Descriptions.Item>
                  <Descriptions.Item label="MV P95跳变">{formatNumber(loopFeatures.mv_stats?.p95_abs_step, 3)}</Descriptions.Item>
                  <Descriptions.Item label="MV 活跃比例">{formatPercentValue(loopFeatures.mv_stats?.active_step_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="MV 平坦比例">{formatPercentValue(loopFeatures.mv_stats?.flat_step_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="MV 反向频次">{formatNumber(loopFeatures.mv_stats?.direction_reversal_per_hour, 2)}/h</Descriptions.Item>
                  <Descriptions.Item label="MV 总行程">{formatNumber(loopFeatures.mv_stats?.total_travel, 2)}</Descriptions.Item>
                  <Descriptions.Item label="过程作用方向">
                    {formatProcessDirection(
                      monitoring?.response_observability?.process_direction
                        ?? String(loopFeatures.pv_mv_relation_raw?.process_direction ?? loopFeatures.pv_mv_relation_raw?.estimated_direction_raw ?? ''),
                    )}
                  </Descriptions.Item>
                  <Descriptions.Item label="方向置信度">
                    {formatPercentValue(
                      monitoring?.response_observability?.process_direction_confidence
                        ?? (typeof loopFeatures.pv_mv_relation_raw?.process_direction_confidence === 'number'
                          ? loopFeatures.pv_mv_relation_raw.process_direction_confidence
                          : undefined),
                      1,
                    )}
                  </Descriptions.Item>
                  <Descriptions.Item label="方向证据">
                    {formatProcessDirectionBasis(
                      monitoring?.response_observability?.process_direction_basis
                        ?? String(loopFeatures.pv_mv_relation_raw?.process_direction_basis ?? ''),
                    )}
                  </Descriptions.Item>
                  <Descriptions.Item label="滞后相关峰值">{formatNumber(loopFeatures.pv_mv_relation_raw?.cross_correlation_peak_abs as number | undefined, 3)}</Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无 PV / MV 统计" />}
            </section>
            <section className="agent-panel compact-facts">
              <div className="panel-title">约束与饱和 constraint_raw</div>
              {loopFeatures ? (
                <Descriptions bordered size="small" column={4} className="industrial-descriptions">
                  <Descriptions.Item label="MV饱和比例">{formatPercentValue(loopFeatures.constraint_raw?.mv_saturation_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="高限饱和">{formatPercentValue(loopFeatures.constraint_raw?.mv_high_saturation_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="低限饱和">{formatPercentValue(loopFeatures.constraint_raw?.mv_low_saturation_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="最长饱和">{formatNumber(loopFeatures.constraint_raw?.longest_mv_saturation_duration_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="饱和段数">{loopFeatures.constraint_raw?.mv_saturation_segment_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="PV近低限">{formatPercentValue(loopFeatures.constraint_raw?.pv_near_observed_min_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="PV近高限">{formatPercentValue(loopFeatures.constraint_raw?.pv_near_observed_max_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="监控状态">
                    <Tag color={monitoringStatusColor(monitoring?.constraints?.status)}>
                      {monitoringStatusText(monitoring?.constraints?.status)}
                    </Tag>
                  </Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无约束统计" />}
            </section>
            <section className="agent-panel compact-facts">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">控制性能指标</div>
                  <Typography.Text type="secondary">
                    基于历史 PV/MV/SV 计算 Harris 指数、Cpk 过程能力和震荡指数；Cpk 只有配置 PV 规格上下限后才给出标准值。
                  </Typography.Text>
                </div>
              </div>
              {loopFeatures ? (
                <>
                  <Descriptions bordered size="small" column={4} className="industrial-descriptions">
                    <Descriptions.Item label="Harris指数(1优0差)">
                      {formatNumber(loopFeatures.performance_raw?.harris_index, 4)}
                    </Descriptions.Item>
                    <Descriptions.Item label="Harris劣化指数">
                      {formatNumber(loopFeatures.performance_raw?.harris_degradation_index, 4)}
                    </Descriptions.Item>
                    <Descriptions.Item label="误差口径">
                      {formatHarrisBasis(loopFeatures.performance_raw?.harris_error_basis)}
                    </Descriptions.Item>
                    <Descriptions.Item label="实际误差方差">
                      {formatNumber(loopFeatures.performance_raw?.harris_actual_variance, 6)}
                    </Descriptions.Item>
                    <Descriptions.Item label="Cpk">
                      {loopFeatures.performance_raw?.cpk === null || loopFeatures.performance_raw?.cpk === undefined
                        ? '未计算'
                        : formatNumber(loopFeatures.performance_raw.cpk, 4)}
                    </Descriptions.Item>
                    <Descriptions.Item label="Cpk依据">
                      {formatCpkBasis(loopFeatures.performance_raw?.cpk_basis)}
                    </Descriptions.Item>
                    <Descriptions.Item label="规格下限/上限">
                      {loopFeatures.performance_raw?.cpk_lsl === null || loopFeatures.performance_raw?.cpk_usl === null
                        ? '-'
                        : `${formatNumber(loopFeatures.performance_raw?.cpk_lsl, 3)} / ${formatNumber(loopFeatures.performance_raw?.cpk_usl, 3)}`}
                    </Descriptions.Item>
                    <Descriptions.Item label="震荡指数">
                      {formatPercentValue(loopFeatures.performance_raw?.oscillation_index ?? loopFeatures.oscillation_raw?.confidence, 1)}
                    </Descriptions.Item>
                    <Descriptions.Item label="震荡周期">
                      {formatNumber(loopFeatures.performance_raw?.oscillation_period_s ?? loopFeatures.oscillation_raw?.pv_dominant_period_s, 1)}s
                    </Descriptions.Item>
                    <Descriptions.Item label="主频能量">
                      {formatPercentValue(loopFeatures.performance_raw?.oscillation_power_ratio ?? loopFeatures.oscillation_raw?.pv_dominant_power_ratio, 1)}
                    </Descriptions.Item>
                    <Descriptions.Item label="零交叉频次">
                      {formatNumber(loopFeatures.performance_raw?.oscillation_zero_crossing_per_hour ?? loopFeatures.oscillation_raw?.pv_zero_crossing_per_hour, 2)}/h
                    </Descriptions.Item>
                  </Descriptions>
                  <Table
                    className="formula-table"
                    size="small"
                    pagination={false}
                    rowKey="name"
                    dataSource={[
                      {
                        name: 'Harris指数',
                        formula: 'HI = σ²_min / σ²_actual',
                        explain: 'σ²_min 使用 PV 高频残差方差近似最小理论方差；σ²_actual 使用 PV-SV 跟踪误差方差，若无有效SV则使用去趋势PV方差。HI越接近1表示越接近最小方差控制，越接近0表示波动越大、改善空间越大。',
                      },
                      {
                        name: 'Harris劣化指数',
                        formula: 'DI = 1 - HI',
                        explain: '为了便于报警和排序额外展示的辅助指标，越接近1表示偏离最小方差基准越多；它不是标准 Harris Index 本体。',
                      },
                      {
                        name: 'Cpk过程能力',
                        formula: 'Cpk = min((USL-μ)/(3σ), (μ-LSL)/(3σ))',
                        explain: '必须有PV规格上限USL和下限LSL；当前历史导入若没有规格限字段，系统不伪造Cpk，只显示未计算。',
                      },
                      {
                        name: '震荡指数/周期',
                        formula: '指数由主频能量占比、PV零交叉频次综合得到；周期 = 1 / 主频',
                        explain: '先用约30分钟滚动中位数去趋势，再做FFT找PV主频；主频能量和零交叉都足够时才认为有显著震荡。',
                      },
                    ]}
                    columns={[
                      { title: '指标', dataIndex: 'name', width: 180 },
                      { title: '公式', dataIndex: 'formula', width: 300 },
                      { title: '说明', dataIndex: 'explain' },
                    ]}
                  />
                </>
              ) : <Empty description="暂无控制性能指标" />}
            </section>
          </div>
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
              <Descriptions bordered size="small" column={4} className="industrial-descriptions">
                <Descriptions.Item label="当前回路">{selectedLoop?.loop_id ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="回路类型">{selectedLoop ? LOOP_TYPE_LABEL[selectedLoop.loop_type] ?? selectedLoop.loop_type : '-'}</Descriptions.Item>
                <Descriptions.Item label="原始时间范围" span={2}>
                  {selectedLoop?.start_time || '-'} ~ {selectedLoop?.end_time || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="当前显示点数">{series?.sampled_points ?? 0}/{series?.total_points ?? 0}</Descriptions.Item>
                <Descriptions.Item label="采样周期">{selectedLoop?.sampling_time ?? '-'}s</Descriptions.Item>
                <Descriptions.Item label="时间筛选" span={2}>
                  {trendPreset === 'custom'
                    ? `${trendCustomRange?.[0]?.format('YYYY-MM-DD HH:mm:ss') ?? '-'} ~ ${trendCustomRange?.[1]?.format('YYYY-MM-DD HH:mm:ss') ?? '-'}`
                    : TREND_PRESET_OPTIONS.find((item) => item.value === trendPreset)?.label ?? '-'}
                </Descriptions.Item>
                <Descriptions.Item label="点数模式">
                  {trendPointLimit === 'all'
                    ? '全量点'
                    : `${TREND_POINT_LIMIT_OPTIONS.find((item) => item.value === trendPointLimit)?.label ?? trendPointLimit}`}
                </Descriptions.Item>
                <Descriptions.Item label="显示说明" span={3}>
                  {series && series.sampled_points < series.total_points
                    ? `当前为抽样趋势，后端从 ${series.total_points} 点中返回 ${series.sampled_points} 点。`
                    : '当前时间范围内为全量点显示。'}
                </Descriptions.Item>
              </Descriptions>
            </section>
            <section className="agent-panel chart-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">趋势曲线</div>
                  <Typography.Text type="secondary">PV/MV 长周期趋势，后续可叠加 SP、模式切换、报警与候选窗口标记。</Typography.Text>
                </div>
                <Space wrap>
                  <Tag color="blue">{series?.sampled_points ?? 0} 点</Tag>
                  <Tag color="cyan">{selectedLoop?.sampling_time ?? '-'}s</Tag>
                </Space>
              </div>
              {seriesLoading ? <Empty description="正在加载趋势数据..." /> : renderTrend(420)}
            </section>
            <section className="agent-panel compact-facts">
              <div className="panel-title">频谱与振荡监测</div>
              {assessment || monitoring ? (
                <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                  <Descriptions.Item label="是否振荡">
                    {monitoring?.stability?.oscillation_detected ?? assessment?.diagnostics.oscillation?.detected ? '检测到' : '未检测到'}
                  </Descriptions.Item>
                  <Descriptions.Item label="严重度">{monitoring?.stability?.oscillation_severity ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="振荡证据">{formatOscillationEvidence(oscillationDetected, monitoring?.stability?.oscillation_confidence)}</Descriptions.Item>
                  <Descriptions.Item label="主周期">
                    {String(monitoring?.stability?.pv_dominant_period_s ?? assessment?.diagnostics.oscillation?.period_sec ?? '-')}s
                  </Descriptions.Item>
                  <Descriptions.Item label="主频能量">{formatPercentValue(monitoring?.stability?.pv_dominant_power_ratio, 1)}</Descriptions.Item>
                  <Descriptions.Item label="零交叉">{formatNumber(monitoring?.stability?.pv_zero_crossing_per_hour, 2)}/h</Descriptions.Item>
                  <Descriptions.Item label="相位关系">{formatOscillationPhaseHint(oscillationDetected, monitoring?.stability?.phase_hint)}</Descriptions.Item>
                  <Descriptions.Item label="PV SNR">
                    {formatNumber(
                      monitoring?.data_health?.pv_snr_db
                        ?? (typeof assessment?.diagnostics.noise?.snr_db === 'number' ? assessment.diagnostics.noise.snr_db : undefined),
                      2,
                    )} dB
                  </Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无频谱分析" />}
            </section>
          </div>
        );
      case 'alarm_events':
      case 'risk_alerts':
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">报警事件 / Agent 事件</div>
                  <Typography.Text type="secondary">
                    仅在本菜单集中展示监控告警、数据质量提示和整定任务事件，不再作为全局底栏常驻。
                  </Typography.Text>
                </div>
                <Space wrap>
                  <Tag color={railAlarms.length ? 'orange' : 'green'}>{railAlarms.length} 条事件</Tag>
                  <Tag color={monitoringStatusColor(monitoring?.status)}>{monitoringStatusText(monitoring?.status)}</Tag>
                </Space>
              </div>
              <Table
                size="small"
                pagination={{ pageSize: 10 }}
                rowKey="key"
                dataSource={railAlarms}
                columns={[
                  { title: '时间', dataIndex: 'time', width: 140 },
                  { title: '级别', dataIndex: 'level', width: 90, render: (value: string) => <Tag color={alertSeverityColor(value)}>{value}</Tag> },
                  { title: '名称', dataIndex: 'name', width: 180 },
                  { title: '描述', dataIndex: 'value', ellipsis: true },
                  { title: '建议动作', dataIndex: 'recommendation', ellipsis: true },
                  { title: '状态', dataIndex: 'status', width: 120 },
                ]}
              />
            </section>

            <section className="agent-panel">
              <div className="panel-title">事件来源说明</div>
              <Descriptions bordered size="small" column={3} className="industrial-descriptions">
                <Descriptions.Item label="监控事件">{monitoring?.events?.length ?? monitoringAlerts.length} 条</Descriptions.Item>
                <Descriptions.Item label="诊断标记">{assessment?.diagnostics.flags.length ?? 0} 条</Descriptions.Item>
                <Descriptions.Item label="整定任务">{taskId ? taskStatus : '暂无任务'}</Descriptions.Item>
                <Descriptions.Item label="当前作用域" span={3}>
                  {selectedAssetPath.map((item) => item.name).join(' / ')}
                </Descriptions.Item>
              </Descriptions>
            </section>
          </div>
        );
      case 'tuning_readiness':
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">整定准备度</div>
                  <Typography.Text type="secondary">把评分、原始指标和建议动作拆开，避免大卡片堆叠。</Typography.Text>
                </div>
                <Tag color={tagColor(assessment?.readiness.level)}>
                  {assessment?.readiness.level ?? '-'}
                </Tag>
              </div>
              {renderAssessmentCards()}
            </section>
            <section className="agent-panel">
              <div className="panel-title">建议动作</div>
              <Table
                size="small"
                pagination={false}
                rowKey="item"
                dataSource={(assessment?.readiness.recommendations ?? ['暂无建议']).map((item, index) => ({ index: index + 1, item }))}
                columns={[
                  { title: '#', dataIndex: 'index', width: 64 },
                  { title: '建议', dataIndex: 'item' },
                ]}
              />
            </section>
            {activeSub === 'tuning_readiness' && assessment?.summary && (
              <section className="agent-panel">
                <div className="panel-title">整定准入结论</div>
                <Alert
                  className="agent-alert"
                  type={assessment.summary.decision === 'blocked' ? 'error' : assessment.summary.decision === 'ready' ? 'success' : 'warning'}
                  showIcon
                  message={assessment.summary.decision_text}
                  description={assessment.summary.recommended_next_action_text}
                />
              </section>
            )}
            {activeSub === 'tuning_readiness' && assessment?.tuning_readiness && (
              <section className="panel-grid">
                <div className="agent-panel">
                  <div className="panel-title">准入检查</div>
                  <Table
                    size="small"
                    pagination={false}
                    rowKey="name"
                    dataSource={assessment.tuning_readiness.gate_checks ?? []}
                    columns={[
                      { title: '检查项', dataIndex: 'name', width: 150 },
                      {
                        title: '状态',
                        dataIndex: 'passed',
                        width: 100,
                        render: (value: boolean) => <Tag color={value ? 'green' : 'orange'}>{value ? '通过' : '需处理'}</Tag>,
                      },
                      { title: '说明', dataIndex: 'message' },
                    ]}
                  />
                </div>
                <div className="agent-panel">
                  <div className="panel-title">阻断/关注原因</div>
                  <Table
                    size="small"
                    pagination={false}
                    rowKey={(row) => `${row.type}-${row.message}`}
                    dataSource={assessment.tuning_readiness.blocking_reasons ?? []}
                    columns={[
                      { title: '类型', dataIndex: 'type', width: 140 },
                      {
                        title: '等级',
                        dataIndex: 'severity',
                        width: 90,
                        render: (value: string) => <Tag color={value === 'high' ? 'red' : 'orange'}>{value}</Tag>,
                      },
                      { title: '原因', dataIndex: 'message' },
                    ]}
                  />
                </div>
              </section>
            )}
          </div>
        );
      case 'performance_score':
        {
          const historicalPerformance = assessment?.performance;
          const hasHistoricalPerformance = Boolean(
            historicalPerformance && (
              historicalPerformance.score !== undefined
              || historicalPerformance.monitoring_score !== undefined
              || historicalPerformance.stability_score !== undefined
              || historicalPerformance.constraint_score !== undefined
              || historicalPerformance.pv_mv_behavior_score !== undefined
            ),
          );
          return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">控制性能</div>
                  <Typography.Text type="secondary">
                    评估当前回路控制效果，重点关注偏差、响应速度、MV动作量和约束影响。
                  </Typography.Text>
                </div>
                <Tag color={taskResult?.evaluation?.passed ? 'green' : 'orange'}>
                  {taskResult
                    ? (taskResult.evaluation?.passed ? '可接受' : '需要优化')
                    : assessmentLoading
                      ? '加载评估'
                      : historicalPerformance?.level || '等待评估'}
                </Tag>
              </div>
              {assessmentLoading ? (
                <Empty description="正在加载历史控制性能评估..." />
              ) : taskResult && taskResult.evaluation ? (
                <>
                  <div className="task-score-grid">
                    {[
                      ['性能评分', taskResult.evaluation.performance_score],
                      ['综合评分', taskResult.evaluation.final_rating],
                      ['就绪评分', taskResult.evaluation.readiness_score],
                      ['鲁棒评分', taskResult.evaluation.robustness_score],
                    ].map(([label, value]) => (
                      <div key={label} className="task-score-card">
                        <Progress
                          type="circle"
                          percent={Number(value) * 10}
                          format={() => formatNumber(Number(value), 1)}
                          strokeColor={Number(value) >= 8 ? '#22a06b' : Number(value) >= 6 ? '#f59e0b' : '#ef4444'}
                          size={72}
                        />
                        <span>{label}</span>
                      </div>
                    ))}
                  </div>
                  <Descriptions column={4} bordered size="small" className="detail-block industrial-descriptions">
                    <Descriptions.Item label="稳定性">
                      <Tag color={taskResult.evaluation.is_stable ? 'green' : 'red'}>{taskResult.evaluation.is_stable ? '稳定' : '不稳定'}</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="超调量">{formatNumber(taskResult.evaluation.overshoot_percent, 1)}%</Descriptions.Item>
                    <Descriptions.Item label="调节时间">{formatNumber(taskResult.evaluation.settling_time_s, 1)}s</Descriptions.Item>
                    <Descriptions.Item label="稳态误差">{formatNumber(taskResult.evaluation.steady_state_error, 2)}%</Descriptions.Item>
                    <Descriptions.Item label="振荡次数">{taskResult.evaluation.oscillation_count}</Descriptions.Item>
                    <Descriptions.Item label="MV 饱和">{formatNumber(taskResult.evaluation.mv_saturation_pct, 1)}%</Descriptions.Item>
                  </Descriptions>
                </>
              ) : hasHistoricalPerformance ? (
                <>
                  <div className="task-score-grid">
                    {[
                      ['历史性能评分', historicalPerformance?.score],
                      ['监控评分', historicalPerformance?.monitoring_score],
                      ['稳定性', historicalPerformance?.stability_score],
                      ['PV/MV 行为', historicalPerformance?.pv_mv_behavior_score],
                      ['约束健康', historicalPerformance?.constraint_score],
                    ].map(([label, value]) => (
                      <div key={label as string} className="task-score-card">
                        <Progress
                          type="circle"
                          percent={value === undefined || value === null ? 0 : scorePercent(Number(value))}
                          format={() => value === undefined || value === null ? '-' : `${scorePercent(Number(value))}%`}
                          strokeColor={Number(value ?? 0) >= 0.8 ? '#22a06b' : Number(value ?? 0) >= 0.6 ? '#f59e0b' : '#ef4444'}
                          size={72}
                        />
                        <span>{label}</span>
                      </div>
                    ))}
                  </div>
                  <Descriptions column={3} bordered size="small" className="detail-block industrial-descriptions">
                    <Descriptions.Item label="综合结论">
                      {assessment?.summary?.decision_text ?? historicalPerformance?.level ?? '-'}
                    </Descriptions.Item>
                    <Descriptions.Item label="建议动作" span={2}>
                      {assessment?.summary?.recommended_next_action_text ?? assessment?.summary?.recommended_next_action ?? '-'}
                    </Descriptions.Item>
                    <Descriptions.Item label="数据质量">
                      <Tag color={tagColor(assessment?.data_quality?.level)}>{assessment?.data_quality?.level ?? '-'}</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="可辨识性">
                      <Tag color={tagColor(assessment?.identifiability?.level)}>{assessment?.identifiability?.level ?? '-'}</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="整定准备度">
                      <Tag color={tagColor(assessment?.readiness?.level)}>{assessment?.readiness?.level ?? '-'}</Tag>
                    </Descriptions.Item>
                  </Descriptions>
                </>
              ) : (
                <Alert
                  type="info"
                  showIcon
                  message="当前暂无整定仿真性能结果"
                  description="后续会接入在线/历史控制性能评估：IAE、ISE、偏差统计、超调、调节时间、稳态误差和MV动作量。"
                />
              )}
            </section>
          </div>
          );
        }
      case 'condition_recognition':
        {
          const conditionProfile = loopFeatures?.operating_condition_profile;
          return (
            <div className="page-stack">
              <section className="agent-panel">
                <div className="panel-toolbar">
                  <div>
                    <div className="panel-title">运行工况</div>
                    <Typography.Text type="secondary">
                      基于历史数据质量、饱和、振荡、SP/MV 活跃度和均值漂移，判断当前片段是否适合进入整定。
                    </Typography.Text>
                  </div>
                  <Space>
                    <BackendBadge implemented={Boolean(conditionProfile)} />
                    <Tag color={tuningSuitabilityColor(conditionProfile?.tuning_suitability)}>
                      {tuningSuitabilityText(conditionProfile?.tuning_suitability)}
                    </Tag>
                  </Space>
                </div>
                {conditionProfile ? (
                  <>
                    <div className="industrial-kpi-grid compact">
                      <div className="industrial-kpi-card">
                        <span>工况判断</span>
                        <strong>{operatingConditionText(conditionProfile.condition_label)}</strong>
                      </div>
                      <div className="industrial-kpi-card">
                        <span>整定适宜性</span>
                        <strong>{tuningSuitabilityText(conditionProfile.tuning_suitability)}</strong>
                      </div>
                      <div className="industrial-kpi-card">
                        <span>判断置信度</span>
                        <strong>{formatPercentValue(conditionProfile.confidence, 1)}</strong>
                      </div>
                      <div className="industrial-kpi-card">
                        <span>本体知识库</span>
                        <strong>{conditionProfile.ontology_context?.status === 'not_connected' ? '未接入' : conditionProfile.ontology_context?.status || '-'}</strong>
                      </div>
                    </div>
                    <Table
                      size="small"
                      pagination={false}
                      rowKey={(row) => String(row.name)}
                      dataSource={conditionProfile.evidence ?? []}
                      columns={[
                        { title: '证据项', dataIndex: 'name', width: 180, render: (value: string) => conditionEvidenceName(value) },
                        {
                          title: '状态',
                          dataIndex: 'status',
                          width: 110,
                          render: (value: string) => <Tag color={evidenceStatusColor(value)}>{evidenceStatusText(value)}</Tag>,
                        },
                        {
                          title: '数值',
                          dataIndex: 'value',
                          width: 120,
                          render: (value: unknown, row: { name?: string }) => {
                            if (typeof value !== 'number') return String(value ?? '-');
                            if (row.name === 'data_quality' || row.name === 'mv_saturation' || row.name === 'oscillation' || row.name === 'transition' || row.name === 'excitation') {
                              return formatPercentValue(value, 1);
                            }
                            return formatNumber(value, 3);
                          },
                        },
                        { title: '判断依据', dataIndex: 'detail', render: (value: string) => conditionEvidenceDetail(value) },
                      ]}
                    />
                  </>
                ) : (
                  <EmptyBackendHint title="运行工况评估暂无后端数据" />
                )}
              </section>
              {conditionProfile && (
                <section className="two-column-grid">
                  <div className="agent-panel">
                    <div className="panel-title">工况片段估算</div>
                    <Table
                      size="small"
                      pagination={false}
                      rowKey={(row) => String(row.label)}
                      dataSource={conditionProfile.segment_summary ?? []}
                      columns={[
                        { title: '片段类型', dataIndex: 'label', render: (value: string) => operatingConditionText(value) },
                        { title: '占比', dataIndex: 'ratio', width: 100, render: (value: number) => formatPercentValue(value, 1) },
                        { title: '时长', dataIndex: 'duration_s', width: 120, render: (value: number) => `${formatNumber(value, 0)}s` },
                        {
                          title: '可用于整定',
                          dataIndex: 'tuning_usable',
                          width: 120,
                          render: (value: boolean) => <Tag color={value ? 'green' : 'default'}>{value ? '可用' : '不建议'}</Tag>,
                        },
                      ]}
                    />
                  </div>
                  <div className="agent-panel">
                    <div className="panel-title">本体与建议</div>
                    <Descriptions bordered column={1} size="small" className="industrial-descriptions">
                      <Descriptions.Item label="本体状态">
                        {conditionProfile.ontology_context?.status === 'not_connected' ? '未接入' : conditionProfile.ontology_context?.status || '-'}
                      </Descriptions.Item>
                      <Descriptions.Item label="回路类型">{conditionProfile.ontology_context?.loop_type_hint || selectedLoop?.loop_type || '-'}</Descriptions.Item>
                      <Descriptions.Item label="后续需要字段">
                        {(conditionProfile.ontology_context?.requires_fields ?? []).join('、') || '-'}
                      </Descriptions.Item>
                    </Descriptions>
                    <div className="compact-list">
                      {(conditionProfile.recommendations ?? []).map((item) => (
                        <div className="compact-list-row" key={item}>
                          <span>{conditionRecommendationText(item)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
              )}
            </div>
          );
        }
      case 'actuator_status':
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">执行机构状态</div>
                  <Typography.Text type="secondary">
                    基于历史 MV 动作、分辨率、贴边、疑似死区和长时间不动片段，判断整定前是否需要先处理阀门/执行机构问题。
                  </Typography.Text>
                </div>
                <Space wrap>
                  <Select
                    showSearch
                    style={{ minWidth: 320 }}
                    placeholder="选择回路"
                    value={selectedLoopId}
                    onChange={setSelectedLoopId}
                    optionFilterProp="label"
                    options={scopedLoops.map((loop) => ({
                      value: loop.loop_id,
                      label: `${loop.loop_id} · ${LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type}`,
                    }))}
                  />
                  <Tag color={(loopFeatures?.actuator_profile?.mv_saturation_ratio ?? 0) > 0.05 ? 'orange' : 'green'}>
                    {(loopFeatures?.actuator_profile?.mv_saturation_ratio ?? 0) > 0.05 ? '需关注' : '正常'}
                  </Tag>
                </Space>
              </div>
              {loopFeatures ? (
                <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                  <Descriptions.Item label="MV分辨率">{formatNumber(loopFeatures.actuator_profile?.mv_resolution_hint, 5)}</Descriptions.Item>
                  <Descriptions.Item label="死区迹象(滞后窗)">{formatPercentValue(loopFeatures.actuator_profile?.mv_deadband_lagged_ratio, 1)}</Descriptions.Item>
                  <Descriptions.Item label="死区事件">{loopFeatures.actuator_profile?.mv_deadband_event_count ?? 0}/{loopFeatures.actuator_profile?.mv_deadband_events_total ?? 0}</Descriptions.Item>
                  <Descriptions.Item label="估计死区宽度">{formatNumber(loopFeatures.actuator_profile?.mv_deadband_estimated_width, 4)}</Descriptions.Item>
                  <Descriptions.Item label="死区观察窗">{formatNumber(loopFeatures.actuator_profile?.mv_deadband_lag_used_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="回差迹象">{yesNo(loopFeatures.actuator_profile?.mv_hysteresis_hint)}</Descriptions.Item>
                  <Descriptions.Item label="回差指数">{formatPercentValue(loopFeatures.actuator_profile?.mv_hysteresis_ratio, 1)}</Descriptions.Item>
                  <Descriptions.Item label="正/反向增益">{formatNumber(loopFeatures.actuator_profile?.mv_hysteresis_up_gain, 3)} / {formatNumber(loopFeatures.actuator_profile?.mv_hysteresis_down_gain, 3)}</Descriptions.Item>
                  <Descriptions.Item label="黏滞迹象">{yesNo(loopFeatures.actuator_profile?.mv_stiction_hint)}</Descriptions.Item>
                  <Descriptions.Item label="黏滞指数">{formatPercentValue(loopFeatures.actuator_profile?.mv_stiction_score, 1)}</Descriptions.Item>
                  <Descriptions.Item label="卡涩迹象">{yesNo(loopFeatures.actuator_profile?.mv_stuck_hint)}</Descriptions.Item>
                  <Descriptions.Item label="最长不动作">{formatNumber(loopFeatures.actuator_profile?.longest_mv_stuck_duration_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="速率限制迹象">{yesNo(loopFeatures.actuator_profile?.mv_rate_limit_hint)}</Descriptions.Item>
                  <Descriptions.Item label="低限余量">{formatNumber(loopFeatures.actuator_profile?.mv_saturation_margin_low, 3)}</Descriptions.Item>
                  <Descriptions.Item label="高限余量">{formatNumber(loopFeatures.actuator_profile?.mv_saturation_margin_high, 3)}</Descriptions.Item>
                  <Descriptions.Item label="MV饱和比例">{formatPercentValue(loopFeatures.actuator_profile?.mv_saturation_ratio, 2)}</Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无执行机构特征" />}
            </section>
            <section className="agent-panel">
              <div className="panel-title">后端算法与公式说明</div>
              <Table
                className="formula-table"
                size="small"
                pagination={false}
                rowKey="item"
                dataSource={[
                  {
                    item: '死区',
                    formula: 'ratio = N(MV有效变化且PV响应≤4σ_noise) / N(MV有效变化)',
                    backend: '已有后端计算：actuator_profile.mv_deadband_lagged_ratio',
                    note: '按回路类型给PV响应观察窗：流量10s、压力60s、温度300s、液位600s；它是历史数据迹象，不等同于阀门离线死区试验。',
                  },
                  {
                    item: '回差',
                    formula: 'hysteresis = |K_up - K_down| / median(|K|)',
                    backend: '已新增后端计算：正向MV段与反向MV段分别估计PV/MV增益',
                    note: '需要正反向动作样本都足够；样本不足时显示未判定。',
                  },
                  {
                    item: '粘滞',
                    formula: 'stiction_score = longest_stuck_s / max(3600s, 120×采样周期)',
                    backend: '已有并增强后端计算：longest_mv_stuck_duration_s、mv_stiction_score',
                    note: '长时间不动作但历史上存在动作时，提示疑似粘滞或控制器输出保持。',
                  },
                  {
                    item: '卡涩',
                    formula: 'stuck_hint = longest_stuck_s ≥ max(3600s, 120×采样周期)',
                    backend: '已新增后端字段：mv_stuck_hint',
                    note: '卡涩需要结合阀门反馈、定位器和现场动作试验最终确认；当前只是历史趋势迹象。',
                  },
                ]}
                columns={[
                  { title: '项目', dataIndex: 'item', width: 120 },
                  { title: '计算公式/判据', dataIndex: 'formula', width: 360 },
                  { title: '后端字段', dataIndex: 'backend', width: 320 },
                  { title: '说明', dataIndex: 'note' },
                ]}
              />
            </section>
            <section className="agent-panel">
              <div className="panel-title">整定影响说明</div>
              <Table
                size="small"
                pagination={false}
                rowKey="item"
                dataSource={[
                  { item: '死区/黏滞', effect: '会让小幅 PID 输出无效，辨识 K/T 容易偏差', action: '先做执行机构检查或剔除死区片段' },
                  { item: '饱和/贴边', effect: '会截断过程响应，整定后仿真偏乐观或偏悲观', action: '优先确认工况和阀门能力' },
                  { item: '分辨率/速率限制', effect: '限制闭环可达到的响应速度', action: '整定时提高保守度并限制 Kp' },
                ]}
                columns={[
                  { title: '问题', dataIndex: 'item', width: 160 },
                  { title: '对整定影响', dataIndex: 'effect' },
                  { title: '建议动作', dataIndex: 'action' },
                ]}
              />
            </section>
          </div>
        );
      case 'constraint_monitor':
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">约束与饱和</div>
                  <Typography.Text type="secondary">
                    监测 MV 上下限触碰、连续饱和和饱和期间 PV 是否仍无法回到目标。
                  </Typography.Text>
                </div>
                <Tag color={monitoringStatusColor(monitoring?.constraints?.status)}>
                  {monitoringStatusText(monitoring?.constraints?.status)}
                </Tag>
              </div>
              <Descriptions bordered column={3} size="small" className="industrial-descriptions">
                <Descriptions.Item label="约束评分">
                  {monitoring?.constraints?.score === undefined ? '-' : `${scorePercent(monitoring.constraints.score)}%`}
                </Descriptions.Item>
                <Descriptions.Item label="MV饱和比例">
                  {formatPercentValue(loopFeatures?.constraint_raw?.mv_saturation_ratio, 2)}
                </Descriptions.Item>
                <Descriptions.Item label="MV范围">
                  {loopFeatures?.mv_stats ? `${formatNumber(loopFeatures.mv_stats.min, 3)} ~ ${formatNumber(loopFeatures.mv_stats.max, 3)}` : '-'}
                </Descriptions.Item>
                <Descriptions.Item label="PV范围">
                  {loopFeatures?.pv_stats ? `${formatNumber(loopFeatures.pv_stats.min, 3)} ~ ${formatNumber(loopFeatures.pv_stats.max, 3)}` : '-'}
                </Descriptions.Item>
                <Descriptions.Item label="约束影响">
                  {monitoring?.constraints?.status === 'normal' ? '暂无明显约束风险' : '存在约束风险或后端尚未返回详细原因'}
                </Descriptions.Item>
              </Descriptions>
            </section>
          </div>
        );
      case 'diagnosis_overview':
        return (
          <section className="agent-panel">
            <div className="panel-toolbar">
              <div className="panel-title">诊断总览</div>
              <Tag color={(assessment?.diagnostics.flags.length ?? 0) > 0 ? 'orange' : 'green'}>
                {(assessment?.diagnostics.flags.length ?? 0) > 0 ? `${assessment?.diagnostics.flags.length} 项风险` : '无明显风险'}
              </Tag>
            </div>
            {assessment?.diagnostics.flags.length ? (
              <Table
                size="small"
                pagination={false}
                rowKey={(row) => `${row.type}-${row.message}`}
                dataSource={assessment.diagnostics.flags}
                columns={[
                  { title: '类型', dataIndex: 'type', width: 160 },
                  { title: '级别', dataIndex: 'severity', width: 100, render: (value: string) => <Tag color={value === 'high' ? 'red' : 'orange'}>{value}</Tag> },
                  { title: '诊断信息', dataIndex: 'message', ellipsis: true },
                ]}
              />
            ) : <Alert type="success" showIcon message="未发现明显数据质量或可辨识性风险。" />}
          </section>
        );
      case 'oscillation_diagnosis':
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
      case 'pid_diagnosis':
      case 'valve_diagnosis':
      case 'measurement_noise_diagnosis':
      case 'process_disturbance_diagnosis': {
        const diagnosisPlan: Record<string, { title: string; desc: string; rows: Array<{ item: string; evidence: string; action: string }> }> = {
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
      case 'tuning_prior': {
        const priorFeatures = tuningPriorCoreData?.features;
        const priorMonitoring = tuningPriorCoreData?.monitoring?.monitoring;
        const priorAssessment = tuningPriorCoreData?.assessment;
        const priorOntology = tuningPriorOntologyData?.ontology;
        const readiness = priorAssessment?.tuning_readiness;
        const gateRows = readiness?.gate_checks ?? [];
        const diagnosisFlags = priorAssessment?.diagnostics?.flags ?? [];
        const ontologyContent = priorOntology?.content || priorOntology?.error || '';
        type PriorEvidenceRow = {
          kind: string;
          name?: string;
          type?: string;
          passed?: boolean;
          severity?: string;
          message?: string;
        };
        const priorEvidenceRows: PriorEvidenceRow[] = [
          ...gateRows.map((item) => ({ ...item, kind: '准入校验' })),
          ...diagnosisFlags.map((item) => ({ ...item, kind: '诊断标记', passed: false })),
        ];
        const priorRangeLabel = tuningPriorRangePreset === 'custom'
          ? '自定义区间'
          : FEATURE_RANGE_OPTIONS.find((item) => item.value === tuningPriorRangePreset)?.label ?? tuningPriorRangePreset;
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">整定先验</div>
                  <Typography.Text type="secondary">
                    先选择回路和时间范围，再按需分别生成核心上下文、本体上下文和大模型先验评审；先验只作为建议，不作为强约束拦截。
                  </Typography.Text>
                </div>
                <Space wrap>
                  <Select
                    size="small"
                    style={{ minWidth: 320 }}
                    value={selectedLoopId}
                    placeholder="选择回路"
                    onChange={setSelectedLoopId}
                    options={loops.map((loop) => ({
                      value: loop.loop_id,
                      label: loop.loop_id + ' · ' + (LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type),
                    }))}
                  />
                  <Select
                    size="small"
                    style={{ width: 150 }}
                    value={tuningPriorRangePreset}
                    onChange={setTuningPriorRangePreset}
                    options={FEATURE_RANGE_OPTIONS.map((item) => ({ label: item.label, value: item.value }))}
                  />
                  {tuningPriorRangePreset === 'custom' && (
                    <DatePicker.RangePicker
                      size="small"
                      showTime
                      value={tuningPriorCustomRange}
                      onChange={(value) => setTuningPriorCustomRange(value)}
                    />
                  )}
                </Space>
              </div>
              {selectedLoop ? (
                <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                  <Descriptions.Item label="当前回路">{selectedLoop.loop_id}</Descriptions.Item>
                  <Descriptions.Item label="回路类型">{LOOP_TYPE_LABEL[selectedLoop.loop_type] ?? selectedLoop.loop_type}</Descriptions.Item>
                  <Descriptions.Item label="时间范围">{priorRangeLabel}</Descriptions.Item>
                  <Descriptions.Item label="数据点数">{priorFeatures?.data_profile?.row_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="区间开始">{priorFeatures?.data_profile?.time_start || '-'}</Descriptions.Item>
                  <Descriptions.Item label="区间结束">{priorFeatures?.data_profile?.time_end || '-'}</Descriptions.Item>
                  <Descriptions.Item label="采样周期">{formatNumber(priorFeatures?.data_profile?.sample_time_median_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="本体状态">
                    <Tag color={priorOntology?.content ? 'green' : priorOntology?.error ? 'orange' : 'default'}>
                      {priorOntology?.content ? '已返回' : priorOntology?.error ? '异常/降级' : '待查询'}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="评审状态">
                    <Tag color={tuningPriorReviewData?.review ? 'green' : tuningPriorReviewError ? 'orange' : 'default'}>
                      {tuningPriorReviewData?.review ? '已生成' : tuningPriorReviewError ? '异常/降级' : '待评审'}
                    </Tag>
                  </Descriptions.Item>
                </Descriptions>
              ) : <Empty description="请先选择回路" />}
            </section>

            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">1 核心指标与评估诊断上下文</div>
                  <Typography.Text type="secondary">来自 LoopFeatures、监控快照、准入评估和诊断 flags，作为整定先验的第一类上下文。</Typography.Text>
                </div>
                <Space wrap>
                  <Button
                    size="small"
                    icon={<SyncOutlined />}
                    loading={tuningPriorCoreLoading}
                    disabled={!selectedLoop}
                    onClick={() => {
                      if (!selectedLoopId) return;
                      loadTuningPriorCore(selectedLoopId, buildTuningPriorRangeParams(selectedLoop));
                    }}
                  >
                    生成核心上下文
                  </Button>
                  <Tag color={priorMonitoring?.status === 'normal' ? 'green' : priorMonitoring?.status ? 'orange' : 'default'}>
                    监控状态：{monitoringStatusText(priorMonitoring?.status)}
                  </Tag>
                  <Tag color={tagColor(readiness?.level)}>{formatPercentValue(readiness?.score, 0)}</Tag>
                </Space>
              </div>
              {tuningPriorCoreLoading ? (
                <Alert className="agent-alert" type="info" showIcon message="正在生成核心上下文" description="正在按所选时间范围聚合 LoopFeatures、监控、评估和诊断结果。" />
              ) : tuningPriorCoreError ? (
                <Alert className="agent-alert" type="error" showIcon message="核心上下文加载失败" description={tuningPriorCoreError} />
              ) : tuningPriorCoreData ? (
                <div className="page-stack compact-stack">
                  <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                    <Descriptions.Item label="控制性能分">{formatPercentValue(priorAssessment?.performance?.score, 0)}</Descriptions.Item>
                    <Descriptions.Item label="整定准备度">{gateDecisionText(readiness?.decision)}</Descriptions.Item>
                    <Descriptions.Item label="可辨识性">{formatPercentValue(priorAssessment?.identification_suitability?.score, 0)}</Descriptions.Item>
                    <Descriptions.Item label="数据质量">{formatPercentValue(priorAssessment?.data_quality?.score, 0)}</Descriptions.Item>
                    <Descriptions.Item label="PV 范围">{formatRange(priorFeatures?.pv_stats?.min, priorFeatures?.pv_stats?.max, 3)}</Descriptions.Item>
                    <Descriptions.Item label="MV 范围">{formatRange(priorFeatures?.mv_stats?.min, priorFeatures?.mv_stats?.max, 3)}</Descriptions.Item>
                    <Descriptions.Item label="MV 饱和比例">{formatPercentValue(priorFeatures?.constraint_raw?.mv_saturation_ratio, 2)}</Descriptions.Item>
                    <Descriptions.Item label="过程方向">{formatProcessDirection(priorFeatures?.pv_mv_relation_raw?.process_direction)}</Descriptions.Item>
                    <Descriptions.Item label="运行工况">{operatingConditionText(priorFeatures?.operating_condition_profile?.condition_label)}</Descriptions.Item>
                    <Descriptions.Item label="振荡状态">{priorFeatures?.oscillation_raw?.detected ? '疑似振荡' : '未检测到明显振荡'}</Descriptions.Item>
                    <Descriptions.Item label="噪声水平">{priorMonitoring?.data_health?.pv_snr_db === undefined ? '-' : formatNumber(priorMonitoring.data_health.pv_snr_db, 2) + ' dB'}</Descriptions.Item>
                    <Descriptions.Item label="报警数量">{priorMonitoring?.alerts?.length ?? 0}</Descriptions.Item>
                  </Descriptions>
                  <Table<PriorEvidenceRow>
                    size="small"
                    pagination={false}
                    rowKey={(row, index) => (row.name || row.type || 'row') + '-' + index}
                    dataSource={priorEvidenceRows}
                    columns={[
                      { title: '类别', dataIndex: 'kind', width: 120 },
                      { title: '项目', dataIndex: 'name', width: 180, render: (value: string, row: PriorEvidenceRow) => value ? gateCheckLabel(value) : row.type ?? '-' },
                      { title: '结果/级别', dataIndex: 'severity', width: 140, render: (value: string, row: PriorEvidenceRow) => row.passed === undefined ? <Tag color={tagColor(value)}>{value || '-'}</Tag> : <Tag color={row.passed ? 'green' : 'orange'}>{row.passed ? '通过' : '提醒'}</Tag> },
                      { title: '说明', dataIndex: 'message' },
                    ]}
                  />
                </div>
              ) : <Empty description="暂无核心上下文，请点击“生成核心上下文”" />}
            </section>

            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">2 本体查询与返回结果</div>
                  <Typography.Text type="secondary">本体结果单独展示，便于核对 LLM 后续解释是否真正引用了装置、变量、增益方向和时间尺度知识。</Typography.Text>
                </div>
                <Space wrap>
                  <Button
                    size="small"
                    icon={<SyncOutlined />}
                    loading={tuningPriorOntologyLoading}
                    disabled={!selectedLoop}
                    onClick={() => {
                      if (!selectedLoopId) return;
                      loadTuningPriorOntology(selectedLoopId, buildTuningPriorRangeParams(selectedLoop));
                    }}
                  >
                    查询本体上下文
                  </Button>
                  <Tag color={priorOntology?.source ? 'blue' : 'default'}>{priorOntology?.server_name || priorOntology?.source || '未查询'}</Tag>
                  <Tag color={ontologyContent ? 'green' : 'default'}>{ontologyContent ? ontologyContent.length + ' 字' : '无内容'}</Tag>
                </Space>
              </div>
              {tuningPriorOntologyLoading ? (
                <Alert className="agent-alert" type="info" showIcon message="正在查询本体/MCP" description="本体查询可能需要数十秒，完成后会在下方展示返回原文。" />
              ) : tuningPriorOntologyError ? (
                <Alert className="agent-alert" type="error" showIcon message="本体查询失败" description={tuningPriorOntologyError} />
              ) : tuningPriorOntologyData ? (
                <div className="page-stack compact-stack">
                  <Descriptions bordered column={2} size="small" className="industrial-descriptions">
                    <Descriptions.Item label="查询问题" span={2}>{priorOntology?.query || '-'}</Descriptions.Item>
                    <Descriptions.Item label="来源服务">{priorOntology?.server_name || priorOntology?.source || '-'}</Descriptions.Item>
                    <Descriptions.Item label="调用工具">{priorOntology?.tool || '-'}</Descriptions.Item>
                  </Descriptions>
                  <Collapse
                    defaultActiveKey={['ontology']}
                    items={[{
                      key: 'ontology',
                      label: '本体 / MCP 返回原文',
                      children: (
                        <Typography.Paragraph className="thinking-text">
                          {ontologyContent || '暂无本体返回内容。'}
                        </Typography.Paragraph>
                      ),
                    }]}
                  />
                </div>
              ) : <Empty description="暂无本体结果，请点击“查询本体上下文”" />}
            </section>

            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">3 大模型整定先验解释</div>
                  <Typography.Text type="secondary">基于前两步返回的核心上下文与本体原文，加上提示词，请大模型给出仅供参考的先验评审结果。</Typography.Text>
                </div>
                <Space wrap>
                  <Tag color="blue">仅建议，不硬拦截</Tag>
                  <Button
                    size="small"
                    type="primary"
                    icon={<RobotOutlined />}
                    loading={tuningPriorReviewLoading}
                    disabled={!selectedLoop || !tuningPriorCoreData?.core_context}
                    onClick={() => {
                      if (!selectedLoopId) return;
                      loadTuningPriorReview(selectedLoopId);
                    }}
                  >
                    生成大模型先验评审
                  </Button>
                </Space>
              </div>
              {tuningPriorReviewLoading ? (
                <Alert className="agent-alert" type="info" showIcon message="正在生成大模型先验评审" description="模型会综合核心指标与本体结果输出建议；该建议不作为整定硬约束。" />
              ) : tuningPriorReviewError ? (
                <Alert className="agent-alert" type="error" showIcon message="大模型先验评审失败" description={tuningPriorReviewError} />
              ) : tuningPriorReviewData ? (
                <Space direction="vertical" style={{ width: '100%' }}>
                  {tuningPriorReviewData.error && (
                    <Alert className="agent-alert" type="warning" showIcon message="评审未完成" description={tuningPriorReviewData.error} />
                  )}
                  {tuningPriorReviewData.review ? (
                    <Typography.Paragraph className="thinking-text">
                      {tuningPriorReviewData.review}
                    </Typography.Paragraph>
                  ) : (
                    <Empty description="大模型未返回可展示的评审说明，请检查模型配置或稍后重试。" />
                  )}
                </Space>
              ) : <Empty description="请先生成核心上下文，再点击“生成大模型先验评审”；本体上下文可选但建议先查询。" />}
            </section>
          </div>
        );
      }
      case 'model_reliability':
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">窗口算法候选池</div>
                  <Typography.Text type="secondary">同一份历史数据会同时尝试 MV 阶跃、MV 斜坡、SP 阶跃和稳态扰动扫描，后续辨识按窗口质量分排序。</Typography.Text>
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
                    <span>candidate_pool</span>
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
                    展示某一次“窗口 × 模型”的 PV 实测、PV 仿真和 MV 曲线；切换下拉框或点击全流程详情里的 attempts 行可查看不同模型。
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
                    value={selectedFitAttempt ? attemptFitKey(selectedFitAttempt) : undefined}
                    onChange={setSelectedFitAttemptKey}
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
                      theme="classicDark"
                      color={['#35a7ff', '#28d7c5', '#ff9f43']}
                      scale={{ color: { range: ['#35a7ff', '#28d7c5', '#ff9f43'] } }}
                      style={{ lineWidth: 2.2 }}
                      padding={[34, 32, 88, 76]}
                      axis={{
                        x: {
                          title: 'X 轴：时间 / 采样点',
                          titleFill: '#d8e8ff',
                          titleFontSize: 13,
                          titleFontWeight: 700,
                          labelFill: '#a9c0de',
                          labelFontSize: 11,
                          labelAutoHide: true,
                          labelAutoRotate: true,
                          lineStroke: '#3b5068',
                          tickStroke: '#3b5068',
                        },
                        y: {
                          title: 'Y 轴：PV / MV 数值',
                          titleFill: '#d8e8ff',
                          titleFontSize: 13,
                          titleFontWeight: 700,
                          labelFill: '#a9c0de',
                          labelFontSize: 12,
                          lineStroke: '#3b5068',
                          tickStroke: '#3b5068',
                          gridStroke: '#223247',
                          gridLineDash: [4, 4],
                        },
                      }}
                      legend={{
                        color: {
                          position: 'top',
                          itemLabelFill: '#d8e8ff',
                          itemLabelFontSize: 13,
                          itemLabelFontWeight: 600,
                          markerSize: 10,
                        },
                      }}
                      xAxis={{
                        title: {
                          text: 'X 轴：时间 / 采样点',
                          style: { fill: '#d8e8ff', fontSize: 13, fontWeight: 700 },
                        },
                        label: {
                          autoHide: true,
                          autoRotate: true,
                          style: { fill: '#9fb6d6', fontSize: 11 },
                          formatter: (text: string) => String(text).slice(5, 16),
                        },
                        line: { style: { stroke: '#3b5068' } },
                        tickLine: { style: { stroke: '#3b5068' } },
                      }}
                      yAxis={{
                        title: {
                          text: 'Y 轴：PV / MV 数值',
                          style: { fill: '#d8e8ff', fontSize: 13, fontWeight: 700 },
                        },
                        label: { style: { fill: '#9fb6d6', fontSize: 12 } },
                        line: { style: { stroke: '#3b5068' } },
                        tickLine: { style: { stroke: '#3b5068' } },
                        grid: { line: { style: { stroke: '#223247', lineDash: [4, 4] } } },
                      }}
                      tooltip={chartLineTooltip}
                      slider={{
                        height: 28,
                        textStyle: { fill: '#b8cbe5' },
                        trendCfg: { lineStyle: { stroke: '#35a7ff' } },
                        handlerStyle: { fill: '#16263a', stroke: '#7fb8ff' },
                      }}
                    />
                  </div>
                </Space>
              ) : (
                <Empty description="暂无模型拟合曲线。请重新发起一次整定任务，后端会在每个成功 attempt 中返回 fit_preview。" />
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
                  <Button size="small" onClick={() => setActiveSub('tuning_task')}>去发起整定</Button>
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
                      { title: 'fit_score', dataIndex: 'fit_score', render: (value) => formatNumber(value, 2) },
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
              {renderWindowTable()}
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
                          theme="classicDark"
                          color={['#35a7ff', '#ff9f43', '#28d7c5']}
                          scale={{ color: { range: ['#35a7ff', '#ff9f43', '#28d7c5'] } }}
                          style={{ lineWidth: 2.1 }}
                          padding={[34, 32, 84, 76]}
                          axis={{
                            x: {
                              title: 'X 轴：窗口内相对时间 / 采样点',
                              titleFill: '#d8e8ff',
                              titleFontSize: 13,
                              titleFontWeight: 700,
                              labelFill: '#a9c0de',
                              labelFontSize: 11,
                              labelAutoHide: true,
                              labelAutoRotate: true,
                              lineStroke: '#3b5068',
                              tickStroke: '#3b5068',
                            },
                            y: {
                              title: 'Y 轴：窗口 PV / MV 数值',
                              titleFill: '#d8e8ff',
                              titleFontSize: 13,
                              titleFontWeight: 700,
                              labelFill: '#a9c0de',
                              labelFontSize: 12,
                              lineStroke: '#3b5068',
                              tickStroke: '#3b5068',
                              gridStroke: '#223247',
                              gridLineDash: [4, 4],
                            },
                          }}
                          legend={{
                            color: {
                              position: 'top',
                              itemLabelFill: '#d8e8ff',
                              itemLabelFontSize: 13,
                              itemLabelFontWeight: 600,
                              markerSize: 10,
                            },
                          }}
                          slider={{
                            height: 28,
                            textStyle: { fill: '#b8cbe5' },
                            trendCfg: { lineStyle: { stroke: '#35a7ff' } },
                            handlerStyle: { fill: '#16263a', stroke: '#7fb8ff' },
                          }}
                          xAxis={{
                            title: { text: 'X 轴：窗口内相对时间 / 采样点', style: { fill: '#d8e8ff', fontSize: 13, fontWeight: 700 } },
                            label: { autoHide: true, autoRotate: true, style: { fill: '#9fb6d6', fontSize: 11 } },
                            line: { style: { stroke: '#3b5068' } },
                            tickLine: { style: { stroke: '#3b5068' } },
                          }}
                          yAxis={{
                            title: { text: 'Y 轴：窗口 PV / MV 数值', style: { fill: '#d8e8ff', fontSize: 13, fontWeight: 700 } },
                            label: { style: { fill: '#9fb6d6', fontSize: 12 } },
                            line: { style: { stroke: '#3b5068' } },
                            tickLine: { style: { stroke: '#3b5068' } },
                            grid: { line: { style: { stroke: '#223247', lineDash: [4, 4] } } },
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
      case 'tuning_task':
        return (
          <div className="page-stack">
            <section className="agent-panel tuning-launch-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">发起整定任务</div>
                  <Typography.Text type="secondary">
                    选择需要整定的回路与时间区间；整定流水线按"数据画像 → 本体策略 → 窗口候选与选择 → 辨识 → 整定 → 评估"顺序执行，与窗口候选页面共用同一套本体驱动逻辑。
                  </Typography.Text>
                </div>
                <Space wrap>
                  <Select
                    showSearch
                    size="small"
                    style={{ minWidth: 280 }}
                    placeholder="选择整定回路"
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
                    value={tuningRangePreset}
                    onChange={(value) => setTuningRangePreset(value)}
                    options={FEATURE_RANGE_OPTIONS.map((item) => ({ label: item.label, value: item.value }))}
                  />
                  {tuningRangePreset === 'custom' && (
                    <DatePicker.RangePicker
                      size="small"
                      showTime
                      value={tuningCustomRange}
                      onChange={(value) => setTuningCustomRange(value)}
                    />
                  )}
                  <Tooltip title="关闭后流水线全程走确定性算法（本体策略与窗口选择不再调用 LLM）">
                    <Space size={4}>
                      <span style={{ fontSize: 12, color: 'rgba(0,0,0,0.55)' }}>LLM 顾问</span>
                      <Switch size="small" checked={tuningUseLlm} onChange={setTuningUseLlm} />
                    </Space>
                  </Tooltip>
                  <Button type="primary" icon={<RocketOutlined />} loading={running} disabled={!selectedLoop} onClick={handleTune}>
                    发起整定
                  </Button>
                  {running && <Button danger onClick={handleStopTune}>停止</Button>}
                </Space>
              </div>

              {selectedLoop ? (
                <div className="tuning-launch-summary">
                  <Statistic title="当前整定回路" value={selectedLoop.loop_id} />
                  <Descriptions bordered column={3} size="small" className="industrial-descriptions">
                      <Descriptions.Item label="类型">{LOOP_TYPE_LABEL[selectedLoop.loop_type] ?? selectedLoop.loop_type}</Descriptions.Item>
                      <Descriptions.Item label="时间区间">
                        {tuningRangePreset === 'all' ? '全部历史' :
                         tuningRangePreset === 'custom' ? '自定义区间' :
                         FEATURE_RANGE_OPTIONS.find((it) => it.value === tuningRangePreset)?.label ?? tuningRangePreset}
                      </Descriptions.Item>
                      <Descriptions.Item label="选窗策略">
                        <Tag color={tuningUseLlm ? 'blue' : 'default'}>
                          {tuningUseLlm ? '本体策略 + LLM 顾问' : '确定性算法（LLM 已关闭）'}
                        </Tag>
                      </Descriptions.Item>
                      <Descriptions.Item label="当前准入" span={3}>
                        <Tag color={tuningGate.hardBlocked ? 'red' : tuningGate.caution ? 'orange' : 'green'}>
                          {gateDecisionText(tuningGate.decision)}
                        </Tag>
                      </Descriptions.Item>
                  </Descriptions>
                </div>
              ) : <Empty description="请先选择回路" />}
            </section>

            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">整定准入校验</div>
                  <Typography.Text type="secondary">发起整定前先查看数据质量、工况、约束、振荡和可辨识性门槛。</Typography.Text>
                </div>
                <Space wrap>
                  <BackendBadge implemented />
                  <Tag color={tuningGate.hardBlocked ? 'red' : tuningGate.caution ? 'orange' : 'green'}>
                    {gateDecisionText(tuningGate.decision)}
                  </Tag>
                  <Tag color={tagColor(tuningGate.level)}>{formatPercentValue(tuningGate.score, 0)}</Tag>
                </Space>
              </div>
              {assessmentLoading ? null : assessmentError ? (
                <Alert className="agent-alert" type="error" showIcon message="整定准入后端接口调用失败" description={assessmentError} />
              ) : assessment ? (
                <div className="page-stack compact-stack">
                  <Table
                    size="small"
                    pagination={false}
                    rowKey={(row) => row.name}
                    dataSource={tuningGate.gateChecks}
                    columns={[
                      { title: '校验项', dataIndex: 'name', width: 180, render: (value: string) => gateCheckLabel(value) },
                      {
                        title: '结果',
                        dataIndex: 'passed',
                        width: 100,
                        render: (value: boolean) => <Tag color={value ? 'green' : 'red'}>{value ? '通过' : '未通过'}</Tag>,
                      },
                      {
                        title: '级别',
                        dataIndex: 'severity',
                        width: 100,
                        render: (value: string) => <Tag color={gateSeverityColor(value)}>{value || '-'}</Tag>,
                      },
                      {
                        title: '准入影响',
                        width: 110,
                        render: (_, row) => {
                          const impact = gateImpact(row, tuningGate.blockingReasons);
                          return <Tag color={impact.color}>{impact.text}</Tag>;
                        },
                      },
                      {
                        title: '说明',
                        dataIndex: 'message',
                        render: (_, row) => gateCheckMessage(row, tuningGate.blockingReasons),
                      },
                    ]}
                  />
                  {tuningGate.blockingReasons.length ? (
                    <Alert
                      className="agent-alert gate-alert"
                      type={tuningGate.hardBlocked ? 'error' : 'warning'}
                      showIcon
                      message="准入提醒"
                      description={(
                        <Space direction="vertical" size={4}>
                          {tuningGate.blockingReasons.map((reason, index) => (
                            <Typography.Text className="gate-alert-text" key={`${reason.type}-${index}`}>
                              {index + 1}. {gateCheckLabel(reason.type)}：{reason.message}
                            </Typography.Text>
                          ))}
                        </Space>
                      )}
                    />
                  ) : null}
                </div>
              ) : (
                <Alert className="agent-alert" type="warning" showIcon message="暂无整定准入校验结果" description="请选择回路或刷新数据。该区域已接入后端 assessment 接口，不再使用模拟数据。" />
              )}
            </section>

            <section className="agent-panel task-process-summary">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">整定流程总览</div>
                  <Typography.Text type="secondary">主界面保留阶段态势，详细 attempts、LLM 判断和候选参数进入抽屉查看。</Typography.Text>
                </div>
                <Space wrap>
                  <Tag color={taskAttempts.length ? 'processing' : 'default'}>{taskAttempts.length} 次辨识尝试</Tag>
                  <Button onClick={() => setTaskDetailOpen(true)}>查看全流程详情</Button>
                </Space>
              </div>
              <Table
                size="small"
                pagination={false}
                rowKey="stage"
                dataSource={TUNING_STAGE_KEYS.map((stage) => ({
                  stage,
                  label: TUNING_STAGE_LABELS[stage],
                  state: taskStageStatus[stage] ?? (taskStageData[stage] ? 'done' : 'wait'),
                  summary: renderTaskStageSummary(stage, taskStageData[stage]),
                }))}
                columns={[
                  { title: '阶段', dataIndex: 'label', width: 160 },
                  {
                    title: '状态',
                    dataIndex: 'state',
                    width: 110,
                    render: (value: string) => (
                      <Tag color={value === 'running' ? 'processing' : value === 'done' ? 'green' : 'default'}>
                        {value === 'running' ? '运行中' : value === 'done' ? '完成' : '等待'}
                      </Tag>
                    ),
                  },
                  { title: '摘要', dataIndex: 'summary', ellipsis: true },
                ]}
              />
            </section>

            <section className="agent-panel">
              <div className="panel-title">任务结果摘要</div>
              <Descriptions bordered column={3} size="small" className="industrial-descriptions">
                <Descriptions.Item label="任务状态">
                  <Tag color={taskStatus === 'running' ? 'processing' : taskStatus === 'done' ? 'green' : taskStatus === 'error' ? 'red' : 'default'}>
                    {taskStatus === 'running' ? '运行中' : taskStatus === 'done' ? '已完成' : taskStatus === 'error' ? '异常/停止' : '未开始'}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="任务 ID">{taskId || '-'}</Descriptions.Item>
                <Descriptions.Item label="当前阶段">{taskCurrentStage ? TUNING_STAGE_LABELS[taskCurrentStage] ?? taskCurrentStage : '-'}</Descriptions.Item>
                <Descriptions.Item label="推荐模型">{taskResult?.model?.model_type ?? (taskStageData.identification?.model_type as string | undefined) ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="推荐策略">{taskResult?.pid_params?.strategy ?? (taskStageData.tuning?.strategy as string | undefined) ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="综合评分">{formatNumber(taskResult?.evaluation?.final_rating ?? (taskStageData.evaluation?.final_rating as number | undefined), 1)}</Descriptions.Item>
              </Descriptions>
            </section>
          </div>
        );
      case 'data_sources':
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">数据源接入配置</div>
                  <Typography.Text type="secondary">
                    先选择数据源类型，再配置连接参数；当前仅“历史文件导入”已接真实后端。
                  </Typography.Text>
                </div>
                <BackendBadge implemented={dataSourceType === 'history_upload'} />
              </div>

              <Form layout="vertical" className="datasource-form" initialValues={{
                source_name: '5203历史回路数据源',
                source_type: dataSourceType,
                host: '127.0.0.1',
                port: dataSourceType === 'opcua' ? 4840 : 8080,
                database: 'pid_history',
                username: 'pid_reader',
                polling_interval_s: 30,
              }}>
                <div className="form-grid">
                  <Form.Item label="数据源名称" name="source_name">
                    <Input placeholder="例如：5203装置实时历史库" />
                  </Form.Item>
                  <Form.Item label="数据源类型" name="source_type">
                    <Select
                      value={dataSourceType}
                      onChange={setDataSourceType}
                      options={[
                        { label: '历史文件导入', value: 'history_upload' },
                      ]}
                    />
                  </Form.Item>
                  {dataSourceType !== 'history_upload' && (
                    <>
                      <Form.Item label="服务 IP / 域名" name="host">
                        <Input placeholder="例如：10.18.2.35 或 historian.company.local" />
                      </Form.Item>
                      <Form.Item label="端口" name="port">
                        <InputNumber min={1} max={65535} style={{ width: '100%' }} />
                      </Form.Item>
                      <Form.Item label={dataSourceType === 'opcua' ? 'Endpoint Path' : '库名 / Namespace'} name="database">
                        <Input placeholder={dataSourceType === 'opcua' ? '例如：/UA/PIDData' : '例如：PID_HISTORY'} />
                      </Form.Item>
                      <Form.Item label="用户名" name="username">
                        <Input placeholder="只读账号" />
                      </Form.Item>
                      <Form.Item label="密码 / Token" name="password">
                        <Input.Password placeholder="后端待接入，当前不保存" />
                      </Form.Item>
                      <Form.Item label="采集/刷新周期 (s)" name="polling_interval_s">
                        <InputNumber min={1} max={3600} style={{ width: '100%' }} />
                      </Form.Item>
                    </>
                  )}
                </div>

                {dataSourceType !== 'history_upload' && (
                  <Alert
                    className="agent-alert"
                    type="warning"
                    showIcon
                    message="该数据源类型后端待接入"
                    description="当前仅完成前端配置形态设计。后续需要新增连接测试、点表同步、实时趋势读取、报警事件读取等后端 provider。"
                  />
                )}

                {dataSourceType === 'history_upload' && (
                  <>
                    <Divider orientation="left">历史数据导入</Divider>
                    <div className="upload-config-grid">
                      <div>
                        <Upload.Dragger
                          multiple
                          fileList={fileList}
                          beforeUpload={() => false}
                          onChange={({ fileList: next }) => setFileList(next)}
                        >
                          <p className="ant-upload-drag-icon"><DatabaseOutlined /></p>
                          <p className="ant-upload-text">拖拽或选择多个回路历史文件</p>
                          <p className="ant-upload-hint">支持：时间戳 + 位号.PV + 位号.MV，也兼容当前 Excel 样例格式</p>
                        </Upload.Dragger>
                      </div>
                      <Alert
                        type="info"
                        showIcon
                        message="历史文件模式已接后端"
                        description="上传后系统会保存原始文件、生成标准化 CSV、识别回路、计算采样周期和候选辨识窗口。"
                      />
                    </div>
                  </>
                )}

                <Space className="datasource-actions" wrap>
                  <Button type="primary" disabled={dataSourceType !== 'history_upload'}>
                    保存配置
                  </Button>
                  {dataSourceType === 'history_upload' && (
                    <Button
                      type="primary"
                      icon={<CloudUploadOutlined />}
                      loading={importing}
                      onClick={handleImport}
                    >
                      导入并生成回路资产
                    </Button>
                  )}
                </Space>
              </Form>
            </section>

            <section className="agent-panel">
              <div className="panel-title">接入状态</div>
              <Descriptions column={1} bordered size="small">
                <Descriptions.Item label="当前模式">
                  离线历史文件导入
                </Descriptions.Item>
                <Descriptions.Item label="已接接口">
                  /api/history/import、/api/history/loops、/api/history/loops/:id/series
                </Descriptions.Item>
                <Descriptions.Item label="已导入回路">{loops.length} 个</Descriptions.Item>
              </Descriptions>
            </section>
          </div>
        );
      case 'rule_config': {
        const loopTypes = Array.from(new Set([
          ...Object.keys(policyConfig?.loop_priors.model_order ?? {}),
          ...Object.keys(policyConfig?.refinement.model_fallbacks ?? {}),
        ]));
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">规则配置</div>
                  <Typography.Text type="secondary">
                    只读展示当前后端实际生效的回路先验、辨识精修阈值和算法族备选模型池。
                  </Typography.Text>
                </div>
                <Space wrap>
                  <BackendBadge implemented />
                  <Button icon={<SyncOutlined />} loading={policyConfigLoading} onClick={loadPolicyConfig}>刷新</Button>
                </Space>
              </div>
              {policyConfig ? (
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Descriptions bordered size="small" column={4} className="industrial-descriptions">
                    <Descriptions.Item label="最低置信度">{formatPercentValue(policyConfig.refinement.fallback_rule.min_confidence, 0)}</Descriptions.Item>
                    <Descriptions.Item label="最低 R²">{formatNumber(policyConfig.refinement.fallback_rule.min_r2, 2)}</Descriptions.Item>
                    <Descriptions.Item label="最低窗口质量">{formatPercentValue(policyConfig.refinement.fallback_rule.min_window_quality, 0)}</Descriptions.Item>
                    <Descriptions.Item label="最大模型池">{policyConfig.refinement.fallback_rule.max_model_pool_size} 个</Descriptions.Item>
                  </Descriptions>
                  <Table
                    size="small"
                    pagination={false}
                    rowKey="key"
                    dataSource={[
                      {
                        key: 'min_confidence',
                        item: '最低置信度',
                        value: formatPercentValue(policyConfig.refinement.fallback_rule.min_confidence, 0),
                        stage: '辨识精修',
                        impact: 'LLM 精修不可用/放弃时，候选算法族最佳结果达到该置信度，才允许确定性策略继续重试。',
                      },
                      {
                        key: 'min_r2',
                        item: '最低 R²',
                        value: formatNumber(policyConfig.refinement.fallback_rule.min_r2, 2),
                        stage: '辨识精修',
                        impact: '过滤明显不可解释的候选窗口算法族，避免低拟合质量结果驱动下一轮。',
                      },
                      {
                        key: 'min_window_quality',
                        item: '最低窗口质量',
                        value: formatPercentValue(policyConfig.refinement.fallback_rule.min_window_quality, 0),
                        stage: '窗口/辨识',
                        impact: '当 R² 或置信度还不充分时，允许高质量窗口作为探索性备选进入下一轮辨识。',
                      },
                      {
                        key: 'max_model_pool_size',
                        item: '最大模型池',
                        value: `${policyConfig.refinement.fallback_rule.max_model_pool_size} 个`,
                        stage: '辨识精修',
                        impact: '限制下一轮强制模型数量，避免模型池过宽导致结果不可解释。',
                      },
                    ]}
                    columns={[
                      { title: '规则项', dataIndex: 'item', width: 160 },
                      { title: '当前值', dataIndex: 'value', width: 120 },
                      { title: '影响链路', dataIndex: 'stage', width: 140, render: (value: string) => <Tag color="blue">{value}</Tag> },
                      { title: '规则用途', dataIndex: 'impact' },
                    ]}
                  />
                </Space>
              ) : (
                <Empty description={policyConfigLoading ? '正在加载规则配置' : '暂无规则配置'} />
              )}
            </section>

            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">按回路类型的模型与时间常数先验</div>
                  <Typography.Text type="secondary">
                    辨识阶段按默认模型顺序尝试；精修阶段在 LLM 不可用时使用备选模型池做保守重试。
                  </Typography.Text>
                </div>
                <Tag color="blue">{loopTypes.length} 类回路</Tag>
              </div>
              <Table
                size="small"
                loading={policyConfigLoading}
                pagination={false}
                rowKey="loop_type"
                dataSource={loopTypes.map((loopType) => ({
                  loop_type: loopType,
                  label: LOOP_TYPE_LABEL[loopType] ?? loopType,
                  model_order: policyConfig?.loop_priors.model_order?.[loopType] ?? [],
                  refinement_models: policyConfig?.refinement.model_fallbacks?.[loopType] ?? [],
                  min_t: policyConfig?.loop_priors.min_reasonable_t?.[loopType],
                  reality_range: policyConfig?.loop_priors.reality_t_ranges?.[loopType],
                  impact: policyLoopImpact(loopType),
                }))}
                columns={[
                  { title: '回路类型', dataIndex: 'label', width: 120 },
                  { title: '辨识模型顺序', dataIndex: 'model_order', render: (models: string[]) => <Space wrap>{models.map((item) => <Tag color="blue" key={item}>{item}</Tag>)}</Space> },
                  { title: '精修备选模型池', dataIndex: 'refinement_models', render: (models: string[]) => <Space wrap>{models.map((item) => <Tag color="cyan" key={item}>{item}</Tag>)}</Space> },
                  { title: 'T下界(s)', dataIndex: 'min_t', width: 110, render: (value: number | undefined) => formatNumber(value, 0) },
                  {
                    title: 'Reality T范围(s)',
                    dataIndex: 'reality_range',
                    width: 160,
                    render: (value?: { min: number; max: number }) => value ? `${formatNumber(value.min, 0)} ~ ${formatNumber(value.max, 0)}` : '-',
                  },
                  { title: '规则用途/影响链路', dataIndex: 'impact', width: 360 },
                ]}
              />
            </section>

            <section className="agent-panel">
              <div className="panel-title">配置说明</div>
              <Alert
                className="agent-alert"
                type="info"
                showIcon
                message="当前为只读版本"
                description="这页读取 /api/policy-config 返回的运行时配置。后续如果要可编辑，需要增加后端持久化、参数校验、审计日志和灰度生效机制。"
              />
            </section>
          </div>
        );
      }
      case 'prompt_config': {
        const activePromptItem = PROMPT_CONFIG_ITEMS.find((item) => item.key === activePromptField) ?? PROMPT_CONFIG_ITEMS[0];
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">提示词管理</div>
                  <Typography.Text type="secondary">
                    统一维护 AI 助手、窗口候选、辨识评审和整定顾问 Agent 的 LLM 提示词。
                  </Typography.Text>
                </div>
                <Space>
                  {promptConfig?.updated_at && (
                    <Tag color="blue">更新于 {dayjs(promptConfig.updated_at).format('YYYY-MM-DD HH:mm')}</Tag>
                  )}
                </Space>
              </div>
              <Alert
                className="agent-alert"
                type="info"
                showIcon
                message="按提示词类型选择后编辑"
                description="未选中的提示词不会显示在页面上，但保存时会随完整配置一起保留；整定、窗口候选、参数修改仍需用户确认后才能执行。"
              />
              <Form
                form={promptConfigForm}
                layout="vertical"
                onFinish={savePromptConfig}
              >
                <Form.Item label="提示词类型">
                  <Select
                    value={activePromptField}
                    onChange={(value) => setActivePromptField(value as PromptConfigField)}
                    options={PROMPT_CONFIG_ITEMS.map((item) => ({
                      label: `${item.group} / ${item.label}`,
                      value: item.key,
                    }))}
                  />
                </Form.Item>
                <Form.Item
                  label={activePromptItem.label}
                  name={activePromptItem.key}
                  rules={[{ required: true, message: `请输入${activePromptItem.label}` }]}
                  help={activePromptItem.help}
                >
                  <Input.TextArea
                    autoSize={{ minRows: activePromptItem.minRows, maxRows: activePromptItem.maxRows }}
                    placeholder={activePromptItem.placeholder}
                  />
                </Form.Item>
                <Space className="datasource-actions">
                  <Button
                    type="primary"
                    icon={<SettingOutlined />}
                    loading={promptConfigSaving}
                    htmlType="submit"
                  >
                    保存提示词
                  </Button>
                  <Button
                    icon={<SyncOutlined />}
                    loading={promptConfigLoading}
                    onClick={loadPromptConfig}
                  >
                    刷新
                  </Button>
                  <Button
                    icon={<RobotOutlined />}
                    loading={promptConfigSaving}
                    onClick={restoreDefaultPromptConfig}
                  >
                    恢复默认
                  </Button>
                </Space>
              </Form>
            </section>
            <section className="agent-panel">
              <div className="panel-title">调用流程建议</div>
              <Descriptions column={1} bordered size="small">
                <Descriptions.Item label="上下文输入">
                  前端传入当前页面、装置范围、选中回路、监控快照、画像指标、整定历史等 context JSON。
                </Descriptions.Item>
                <Descriptions.Item label="模型输出">
                  模型返回结构化 answer、evidence、risk_level 和 suggested_actions，前端只渲染白名单动作。
                </Descriptions.Item>
                <Descriptions.Item label="高风险操作">
                  整定、窗口候选、参数修改等操作只允许用户点击按钮后进入对应页面确认，不由模型直接执行。
                </Descriptions.Item>
                <Descriptions.Item label="持久化位置">
                  配置保存到 <Typography.Text code>backend/var/config/prompt_config.json</Typography.Text>。
                </Descriptions.Item>
              </Descriptions>
            </section>
          </div>
        );
      }
      case 'model_config':
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">LLM 模型配置</div>
                  <Typography.Text type="secondary">
                    配置大模型连接参数（API 地址、Key、模型名称），保存后即时生效，无需重启。
                  </Typography.Text>
                </div>
                <Space>
                  {modelConfig && modelConfig.model_api_key && (
                    <Tag color="success" icon={<CheckCircleOutlined />}>已配置</Tag>
                  )}
                  {modelConfigTestResult?.status === 'ok' && (
                    <Tag color="success">连接正常</Tag>
                  )}
                  {modelConfigTestResult?.status === 'error' && (
                    <Tag color="error">连接失败</Tag>
                  )}
                  {!modelConfig?.model_api_key && (
                    <Tag>未配置</Tag>
                  )}
                </Space>
              </div>
              <Form
                form={modelConfigForm}
                layout="vertical"
                onFinish={saveModelConfig}
              >
                <div className="form-grid">
                  <Form.Item
                    label="模型 API 地址"
                    name="model_api_url"
                    rules={[
                      { required: true, message: '请输入 API 地址' },
                      { pattern: /^https?:\/\/.+/, message: '以 http:// 或 https:// 开头' },
                    ]}
                  >
                    <Input prefix={<ApiOutlined />} placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1" />
                  </Form.Item>
                  <Form.Item
                    label="模型名称"
                    name="model_name"
                    rules={[{ required: true, message: '请输入模型名称' }]}
                  >
                    <Input prefix={<RobotOutlined />} placeholder="qwen-plus / qwen-max / deepseek-chat" />
                  </Form.Item>
                  <Form.Item
                    label="API Key"
                    name="model_api_key"
                    rules={[{ required: true, message: '请输入 API Key' }]}
                    help="Key 存储在服务器本地文件（已加入 .gitignore），前端展示时脱敏"
                  >
                    <Input.Password prefix={<KeyOutlined />} placeholder="sk-..." />
                  </Form.Item>
                </div>
                <Space className="datasource-actions">
                  <Button
                    type="primary"
                    icon={<SettingOutlined />}
                    loading={modelConfigSaving}
                    htmlType="submit"
                  >
                    保存配置
                  </Button>
                  <Button
                    icon={<ApiOutlined />}
                    loading={modelConfigTesting}
                    onClick={testModelConnection}
                    disabled={!modelConfig?.model_api_key}
                  >
                    测试连接
                  </Button>
                  <Button
                    icon={<SyncOutlined />}
                    loading={modelConfigLoading}
                    onClick={loadModelConfig}
                  >
                    刷新
                  </Button>
                </Space>
              </Form>
              {modelConfigTestResult && (
                <Alert
                  type={modelConfigTestResult.status === 'ok' ? 'success' : 'error'}
                  showIcon
                  className="agent-alert"
                  message={modelConfigTestResult.status === 'ok' ? '连接成功' : '连接失败'}
                  description={
                    <Typography.Text
                      type={modelConfigTestResult.status === 'ok' ? 'success' : 'danger'}
                      style={{ whiteSpace: 'pre-wrap' }}
                    >
                      {modelConfigTestResult.message}
                    </Typography.Text>
                  }
                />
              )}
            </section>
            <section className="agent-panel">
              <div className="panel-title">配置说明</div>
              <Descriptions column={1} bordered size="small">
                <Descriptions.Item label="API 地址">
                  兼容 OpenAI 接口规范。阿里云 DashScope：
                  <Typography.Text code>https://dashscope.aliyuncs.com/compatible-mode/v1</Typography.Text>
                </Descriptions.Item>
                <Descriptions.Item label="API Key">
                  DashScope Key 可在 <Typography.Text code>https://bailian.console.aliyun.com</Typography.Text> 生成。
                </Descriptions.Item>
                <Descriptions.Item label="模型名称">
                  常用：<Tag>qwen-plus</Tag> <Tag>qwen-max</Tag> <Tag>deepseek-chat</Tag> <Tag>deepseek-reasoner</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="持久化">
                  配置保存在 <Typography.Text code>backend/var/config/model.json</Typography.Text>，已在 .gitignore 中。
                </Descriptions.Item>
              </Descriptions>
            </section>
          </div>
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

        <main className="dialogue-main">
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
                <Empty
                  description={activeAssistantSession ? '当前会话暂无消息，请输入问题开始分析' : '请新建对话，或直接输入问题自动创建会话'}
                />
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
          <Drawer
            title="整定任务全流程详情"
            width="min(1180px, 92vw)"
            open={taskDetailOpen}
            onClose={() => setTaskDetailOpen(false)}
            className="industrial-drawer"
          >
            {renderTaskDashboard()}
          </Drawer>
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
