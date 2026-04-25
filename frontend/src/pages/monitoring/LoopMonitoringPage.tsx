import { useCallback, useEffect, useMemo, useState } from 'react';
import { Line } from '@ant-design/charts';
import {
  Alert,
  Button,
  Collapse,
  Divider,
  Descriptions,
  Drawer,
  Empty,
  Form,
  Input,
  InputNumber,
  List,
  Progress,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  Tree,
  Upload,
  message,
} from 'antd';
import type { UploadFile } from 'antd';
import type { DataNode } from 'antd/es/tree';
import {
  AlertOutlined,
  ApiOutlined,
  AppstoreOutlined,
  AuditOutlined,
  BellOutlined,
  BranchesOutlined,
  ClockCircleOutlined,
  CloudUploadOutlined,
  DatabaseOutlined,
  DeploymentUnitOutlined,
  DownOutlined,
  DesktopOutlined,
  ExperimentOutlined,
  FileSearchOutlined,
  FundProjectionScreenOutlined,
  HistoryOutlined,
  LineChartOutlined,
  MenuOutlined,
  RadarChartOutlined,
  RocketOutlined,
  SafetyCertificateOutlined,
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
  getHistoryLoopAssessment,
  getHistoryLoopSeries,
  getHistoryLoopWindows,
  importHistoryFiles,
  listHistoryLoops,
  tuneHistoryLoopStream,
} from '@/services/api';
import type {
  HistoryLoop,
  HistoryLoopAssessment,
  HistoryLoopFeatures,
  HistoryLoopMonitoring,
  HistoryWindow,
  LoopSeriesResp,
} from '@/services/api';
import type {
  IdentificationAttempt,
  IdentificationRefinementMeta,
  LlmThinkingEvent,
  ModelReviewMeta,
  PipelineEvent,
  StrategyCandidate,
  TuningResult,
  WindowSelectionMeta,
} from '@/types/tuning';
import './LoopMonitoringPage.css';

type ModuleKey = 'workspace' | 'monitor' | 'assessment' | 'diagnostics' | 'tuning' | 'experience' | 'settings';
type SubKey =
  | 'dashboard' | 'todo' | 'shift_tasks' | 'risk_alerts'
  | 'loop_board' | 'loop_profile' | 'trend_spectrum' | 'alarm_events'
  | 'performance_score' | 'data_quality' | 'condition_recognition' | 'tuning_readiness'
  | 'diagnosis_overview' | 'valve_diagnosis' | 'oscillation_diagnosis' | 'model_reliability'
  | 'tuning_task' | 'id_windows' | 'pid_candidates' | 'release_confirm'
  | 'case_library' | 'rule_library' | 'knowledge_graph' | 'model_versions'
  | 'data_sources' | 'asset_directory' | 'loop_master_data' | 'rule_config' | 'roles';

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
    label: '工作台',
    icon: <AppstoreOutlined />,
    subs: [
      { key: 'dashboard', label: '总览驾驶舱', icon: <FundProjectionScreenOutlined /> },
      { key: 'todo', label: '待处理回路', icon: <AlertOutlined /> },
      { key: 'shift_tasks', label: '本班任务', icon: <AuditOutlined /> },
      { key: 'risk_alerts', label: '风险预警', icon: <WarningOutlined /> },
    ],
  },
  {
    key: 'monitor',
    label: '回路监控',
    icon: <RadarChartOutlined />,
    subs: [
      { key: 'loop_board', label: '全局回路看板', icon: <DatabaseOutlined />, implemented: true },
      { key: 'loop_profile', label: '单回路画像', icon: <FileSearchOutlined />, implemented: true },
      { key: 'trend_spectrum', label: '趋势与频谱', icon: <LineChartOutlined />, implemented: true },
      { key: 'alarm_events', label: '报警与事件', icon: <AlertOutlined /> },
    ],
  },
  {
    key: 'assessment',
    label: '回路评估',
    icon: <AuditOutlined />,
    subs: [
      { key: 'performance_score', label: '性能评分', icon: <FundProjectionScreenOutlined /> },
      { key: 'data_quality', label: '数据质量', icon: <SafetyCertificateOutlined />, implemented: true },
      { key: 'condition_recognition', label: '工况识别', icon: <BranchesOutlined /> },
      { key: 'tuning_readiness', label: '整定准备度', icon: <RocketOutlined />, implemented: true },
    ],
  },
  {
    key: 'diagnostics',
    label: '根因诊断',
    icon: <DeploymentUnitOutlined />,
    subs: [
      { key: 'diagnosis_overview', label: '诊断总览', icon: <FileSearchOutlined />, implemented: true },
      { key: 'valve_diagnosis', label: '阀门诊断', icon: <ToolOutlined /> },
      { key: 'oscillation_diagnosis', label: '振荡诊断', icon: <LineChartOutlined />, implemented: true },
      { key: 'model_reliability', label: '模型可靠性', icon: <ExperimentOutlined />, implemented: true },
    ],
  },
  {
    key: 'tuning',
    label: '整定中心',
    icon: <RocketOutlined />,
    subs: [
      { key: 'tuning_task', label: '整定任务', icon: <RocketOutlined />, implemented: true },
      { key: 'id_windows', label: '窗口与辨识', icon: <AuditOutlined />, implemented: true },
      { key: 'pid_candidates', label: '参数候选', icon: <ExperimentOutlined /> },
      { key: 'release_confirm', label: '下发确认', icon: <SafetyCertificateOutlined /> },
    ],
  },
  {
    key: 'experience',
    label: '经验中心',
    icon: <HistoryOutlined />,
    subs: [
      { key: 'case_library', label: '整定案例库', icon: <HistoryOutlined /> },
      { key: 'rule_library', label: '规则库', icon: <FileSearchOutlined /> },
      { key: 'knowledge_graph', label: '知识图谱', icon: <BranchesOutlined /> },
      { key: 'model_versions', label: '模型版本', icon: <DeploymentUnitOutlined /> },
    ],
  },
  {
    key: 'settings',
    label: '系统设置',
    icon: <SettingOutlined />,
    subs: [
      { key: 'data_sources', label: '数据源配置', icon: <ApiOutlined />, implemented: true },
      { key: 'asset_directory', label: '装置资产目录', icon: <DeploymentUnitOutlined />, implemented: true },
      { key: 'loop_master_data', label: '回路主数据', icon: <DatabaseOutlined />, implemented: true },
      { key: 'rule_config', label: '规则配置', icon: <SettingOutlined /> },
      { key: 'roles', label: '角色权限', icon: <SafetyCertificateOutlined /> },
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
  'window_selection',
  'identification',
  'model_review',
  'identification_refinement',
  'tuning',
  'evaluation',
];

const TUNING_STAGE_LABELS: Record<string, string> = {
  data_analysis: '数据分析',
  window_selection: '窗口选择',
  identification: '模型辨识',
  model_review: 'LLM 评审',
  identification_refinement: '精修建议',
  tuning: 'PID 整定',
  evaluation: '性能评估',
};

type TaskStatus = 'idle' | 'running' | 'done' | 'error';

interface TaskEventLog {
  id: number;
  label: string;
  detail?: string;
}

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

function formatRange(min?: number | null, max?: number | null, digits = 2) {
  return `${formatNumber(min, digits)} ~ ${formatNumber(max, digits)}`;
}

function formatPercentValue(value?: number | null, digits = 0) {
  return value === null || value === undefined || Number.isNaN(value) ? '-' : `${(value * 100).toFixed(digits)}%`;
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
  return implemented ? <Tag color="green">已接后端</Tag> : <Tag color="orange">后端待接入 · 模拟展示</Tag>;
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

export default function LoopMonitoringPage() {
  const [activeModule, setActiveModule] = useState<ModuleKey>('workspace');
  const [activeSub, setActiveSub] = useState<SubKey>('dashboard');
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [loops, setLoops] = useState<HistoryLoop[]>([]);
  const [selectedLoopId, setSelectedLoopId] = useState<string>();
  const [assetNodes, setAssetNodes] = useState<AssetNode[]>(DEFAULT_ASSET_NODES);
  const [selectedAssetNodeId, setSelectedAssetNodeId] = useState<string>('unit_2_hydrocrack');
  const [assetDraftName, setAssetDraftName] = useState('');
  const [assetDraftType, setAssetDraftType] = useState<AssetNodeType>('area');
  const [assetRenameValue, setAssetRenameValue] = useState('');
  const [series, setSeries] = useState<LoopSeriesResp | null>(null);
  const [assessment, setAssessment] = useState<HistoryLoopAssessment | null>(null);
  const [loopFeatures, setLoopFeatures] = useState<HistoryLoopFeatures | null>(null);
  const [loopMonitoring, setLoopMonitoring] = useState<HistoryLoopMonitoring | null>(null);
  const [windows, setWindows] = useState<HistoryWindow[]>([]);
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
  const [taskWindowSelection, setTaskWindowSelection] = useState<WindowSelectionMeta | null>(null);
  const [taskModelReview, setTaskModelReview] = useState<ModelReviewMeta | null>(null);
  const [taskRefinements, setTaskRefinements] = useState<IdentificationRefinementMeta[]>([]);
  const [taskThinking, setTaskThinking] = useState<LlmThinkingEvent[]>([]);
  const [taskAttempts, setTaskAttempts] = useState<IdentificationAttempt[]>([]);
  const [taskResult, setTaskResult] = useState<TuningResult | null>(null);
  const [taskError, setTaskError] = useState<string>();
  const [taskAbort, setTaskAbort] = useState<AbortController | null>(null);
  const [events, setEvents] = useState<TaskEventLog[]>([]);
  const [dataSourceType, setDataSourceType] = useState<string>('historian');
  const [taskDetailOpen, setTaskDetailOpen] = useState(false);
  const [rawLogExpanded, setRawLogExpanded] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [expandedModules, setExpandedModules] = useState<Record<ModuleKey, boolean>>(INITIAL_EXPANDED_MODULES);

  const currentModule = MODULES.find((item) => item.key === activeModule) ?? MODULES[0];
  const currentSub = currentModule.subs.find((item) => item.key === activeSub) ?? currentModule.subs[0];

  const selectedLoop = useMemo(
    () => loops.find((item) => item.loop_id === selectedLoopId),
    [loops, selectedLoopId],
  );

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
    const totalWindows = scopedLoops.reduce((sum, item) => sum + (item.window_count ?? 0), 0);
    const usableWindows = scopedLoops.reduce((sum, item) => sum + (item.usable_window_count ?? 0), 0);
    return {
      loopCount: scopedLoops.length,
      usableWindows,
      totalWindows,
      windowRatio: totalWindows ? Math.round((usableWindows / totalWindows) * 100) : 0,
    };
  }, [scopedLoops]);

  const showContextRail = ![
    'asset_directory',
    'data_sources',
    'loop_master_data',
    'rule_config',
    'roles',
    'case_library',
    'rule_library',
    'knowledge_graph',
    'model_versions',
  ].includes(activeSub);

  const selectedWindow = useMemo(
    () => windows.find((item) => item.index === selectedWindowIndex),
    [selectedWindowIndex, windows],
  );

  const railAlarms = useMemo(() => {
    const monitoringAlerts = loopMonitoring?.monitoring.alerts ?? [];
    if (monitoringAlerts.length) {
      return monitoringAlerts.slice(0, 4).map((alert, index) => ({
        key: `monitoring-${index}`,
        time: '当前',
        level: alert.severity || '提示',
        name: alert.type || 'monitoring',
        value: alert.message,
        status: monitoringStatusText(loopMonitoring?.monitoring.status),
      }));
    }
    const flags = assessment?.diagnostics.flags ?? [];
    if (flags.length) {
      return flags.slice(0, 4).map((flag, index) => ({
        key: `flag-${index}`,
        time: '当前',
        level: flag.severity || '提示',
        name: flag.type,
        value: flag.message,
        status: '待确认',
      }));
    }
    return [
      { key: 'stable', time: '当前', level: '中', name: '窗口质量', value: `可用窗口 ${selectedLoop?.usable_window_count ?? 0}/${selectedLoop?.window_count ?? 0}`, status: '跟踪' },
      { key: 'source', time: '当前', level: '低', name: '数据源', value: dataSourceType === 'history' ? '历史文件导入' : '历史仓库/实时库', status: '正常' },
      { key: 'task', time: taskStartedAt ? new Date(taskStartedAt).toLocaleTimeString() : '未启动', level: taskStatus === 'error' ? '高' : '低', name: '整定任务', value: taskId ? `任务 ${taskId}` : '暂无运行任务', status: taskStatus === 'done' ? '完成' : taskStatus === 'running' ? '运行' : '空闲' },
    ];
  }, [assessment?.diagnostics.flags, dataSourceType, loopMonitoring, selectedLoop, taskId, taskStartedAt, taskStatus]);

  const switchTo = (moduleKey: ModuleKey, subKey: SubKey) => {
    setActiveModule(moduleKey);
    setActiveSub(subKey);
    setExpandedModules((prev) => ({ ...prev, [moduleKey]: true }));
  };

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

  const loadSeries = useCallback(async (loopId: string) => {
    setSeries(null);
    try {
      const resp = await getHistoryLoopSeries(loopId, { max_points: 6000 });
      if (resp.error) message.warning(resp.error);
      setSeries(resp);
    } catch (error) {
      message.error(`加载趋势失败：${String(error)}`);
    }
  }, []);

  const loadAssessment = useCallback(async (loopId: string) => {
    setAssessment(null);
    try {
      const resp = await getHistoryLoopAssessment(loopId);
      if (resp.error) message.warning(resp.error);
      setAssessment(resp);
    } catch (error) {
      message.error(`加载回路评估失败：${String(error)}`);
    }
  }, []);

  const loadLoopFeatures = useCallback(async (loopId: string) => {
    setLoopFeatures(null);
    try {
      const resp = await fetchHistoryLoopFeatures(loopId);
      setLoopFeatures(resp);
    } catch (error) {
      message.error(`加载 LoopFeatures 失败：${String(error)}`);
    }
  }, []);

  const loadLoopMonitoring = useCallback(async (loopId: string) => {
    setLoopMonitoring(null);
    try {
      const resp = await fetchHistoryLoopMonitoring(loopId);
      setLoopMonitoring(resp);
      setLoopFeatures((current) => current ?? resp.features ?? null);
    } catch (error) {
      message.error(`加载监控快照失败：${String(error)}`);
    }
  }, []);

  const loadWindows = useCallback(async (loopId: string) => {
    setWindows([]);
    setSelectedWindowIndex(undefined);
    try {
      const resp = await getHistoryLoopWindows(loopId);
      if (resp.error) message.warning(resp.error);
      setWindows(resp.windows ?? []);
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
    if (!selectedLoopId) return;
    loadSeries(selectedLoopId);
    loadAssessment(selectedLoopId);
    loadLoopFeatures(selectedLoopId);
    loadLoopMonitoring(selectedLoopId);
    loadWindows(selectedLoopId);
  }, [loadAssessment, loadLoopFeatures, loadLoopMonitoring, loadSeries, loadWindows, selectedLoopId]);

  useEffect(() => {
    if (!scopedLoops.length) return;
    if (!selectedLoopId || !scopedLoops.some((loop) => loop.loop_id === selectedLoopId)) {
      setSelectedLoopId(scopedLoops[0].loop_id);
    }
  }, [scopedLoops, selectedLoopId]);

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

  const handleTune = () => {
    if (!selectedLoop) {
      message.warning('请先选择一个回路');
      return;
    }
    setRunning(true);
    setTaskStatus('running');
    setTaskStartedAt(new Date().toLocaleString());
    setTaskId(undefined);
    setTaskCurrentStage(undefined);
    setTaskStageStatus({});
    setTaskStageData({});
    setTaskWindowSelection(null);
    setTaskModelReview(null);
    setTaskRefinements([]);
    setTaskThinking([]);
    setTaskAttempts([]);
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
        selected_window_index: selectedWindowIndex,
        use_llm_advisor: true,
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
          if (e.status === 'done' && e.data) {
            setTaskStageData((prev) => ({ ...prev, [e.stage]: e.data ?? {} }));
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
    { title: '候选窗口', render: (_: unknown, row: HistoryLoop) => `${row.usable_window_count ?? 0}/${row.window_count ?? 0}` },
    { title: '最佳窗口分', dataIndex: 'best_window_score', render: (value: number | null) => value == null ? '-' : value.toFixed(3) },
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
    { title: '类型', dataIndex: 'type' },
    { title: '质量分', dataIndex: 'score', render: (value: number) => value.toFixed(3) },
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

  const renderTrend = (height = 360) => (
    trendData.length ? (
      <div className="chart-shell">
        <Line
          height={height}
          data={trendData}
          xField="t"
          yField="value"
          colorField="series"
          legend={{ position: 'top-right' }}
          slider={{}}
          xAxis={{
            type: series?.x_axis === 'timestamp' ? 'timeCat' : 'linear',
            title: { text: '时间' },
            label: { autoHide: true, autoRotate: false },
          }}
          yAxis={{
            title: { text: 'PV / MV' },
            grid: { line: { style: { stroke: '#e5edf6', lineDash: [4, 4] } } },
          }}
        />
      </div>
    ) : <Empty description="暂无趋势数据" />
  );

  const renderAssessmentCards = () => (
    assessment ? (
      <div className="score-grid">
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

  const renderTaskStageSummary = (stage: string, data?: Record<string, unknown>) => {
    if (!data) return '等待执行';
    if (stage === 'data_analysis') {
      return `${data.data_points ?? '-'} 点 / ${data.usable_windows ?? '-'} 可用窗口`;
    }
    if (stage === 'window_selection') {
      return `${data.mode ?? '-'} 选择 #${data.chosen_index ?? '-'}`;
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
    const evaluationPassed = (evaluationStage?.passed as boolean | undefined) ?? result?.evaluation.passed;
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
            <strong>{taskStageData.data_analysis?.candidate_windows as number ?? selectedLoop?.window_count ?? '-'}</strong>
            <em>可用 {taskStageData.data_analysis?.usable_windows as number ?? selectedLoop?.usable_window_count ?? '-'} 个</em>
          </div>
          <div className="task-kpi-card">
            <span>辨识模型</span>
            <strong>{idStage?.model_type as string ?? result?.model.model_type ?? '-'}</strong>
            <em>R² {formatNumber((idStage?.r2_score as number | undefined) ?? result?.model.r2_score, 3)}</em>
          </div>
          <div className="task-kpi-card">
            <span>推荐策略</span>
            <strong>{tuningStage?.strategy as string ?? result?.pid_params.strategy ?? '-'}</strong>
            <em>Kp {formatNumber((tuningStage?.Kp as number | undefined) ?? result?.pid_params.Kp, 3)}</em>
          </div>
          <div className="task-kpi-card">
            <span>综合评分</span>
            <strong>{formatNumber((evaluationStage?.final_rating as number | undefined) ?? result?.evaluation.final_rating, 1)}</strong>
            <em>{evaluationPassed === undefined ? '等待评估' : evaluationPassed ? '可以上线' : '需要优化'}</em>
          </div>
        </div>

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
                        R{item.round}：{item.retry ? '继续重试' : '放弃重试'}；{item.rationale}
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
          {taskAttempts.length ? (
            <Table<IdentificationAttempt>
              size="small"
              rowKey={(row, index) => `${row.round ?? 0}-${row.model_type}-${row.window_source}-${index ?? 0}`}
              dataSource={taskAttempts}
              pagination={{ pageSize: 8 }}
              columns={[
                { title: 'Round', dataIndex: 'round', width: 80, render: (value) => `R${value ?? 0}` },
                { title: '模型', dataIndex: 'model_type', width: 100, render: (value) => <Tag color="blue">{value}</Tag> },
                { title: '窗口', dataIndex: 'window_source', ellipsis: true },
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

        {result && (
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

    switch (activeSub) {
      case 'dashboard':
        return (
          <>
            <div className="kpi-grid">
              <Statistic title="当前范围回路" value={scopedLoopStats.loopCount} suffix="个" />
              <Statistic title="建议整定" value={scopedLoops.filter((item) => (item.usable_window_count ?? 0) > 0).length} suffix="个" />
              <Statistic title="窗口可用率" value={scopedLoopStats.windowRatio} suffix="%" />
              <Statistic
                title="监控综合分"
                value={monitoring?.overall_score === undefined ? '-' : scorePercent(monitoring.overall_score)}
                suffix={monitoring?.overall_score === undefined ? undefined : '%'}
              />
            </div>
            <div className="panel-grid two">
              <section className="agent-panel">
                <div className="panel-title">TOP 待关注回路 · {selectedAssetNode?.name}</div>
                <div className="loop-card-list">
                  {scopedLoops.slice(0, 5).map((loop) => (
                    <button
                      key={loop.loop_id}
                      className={loop.loop_id === selectedLoopId ? 'loop-card active' : 'loop-card'}
                      onClick={() => setSelectedLoopId(loop.loop_id)}
                    >
                      <span>
                        <strong>{loop.loop_id}</strong>
                        <em>{LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type}</em>
                      </span>
                      <span>
                        <Tag color={(loop.usable_window_count ?? 0) > 0 ? 'green' : 'red'}>
                          窗口 {loop.usable_window_count ?? 0}/{loop.window_count ?? 0}
                        </Tag>
                        <Tag color="blue">分 {loop.best_window_score ?? '-'}</Tag>
                      </span>
                    </button>
                  ))}
                  {!scopedLoops.length && <Empty description="当前资产范围暂无回路" />}
                </div>
              </section>
              <section className="agent-panel">
                <div className="panel-toolbar">
                  <div className="panel-title">Agent 本班建议</div>
                  <Tag color={monitoringStatusColor(monitoring?.status)}>
                    {monitoringStatusText(monitoring?.status)}
                  </Tag>
                </div>
                <Descriptions bordered column={2} size="small" className="industrial-descriptions detail-block">
                  <Descriptions.Item label="监控状态">
                    <Tag color={monitoringStatusColor(monitoring?.status)}>{monitoringStatusText(monitoring?.status)}</Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="综合分">
                    {monitoring?.overall_score === undefined ? '-' : `${scorePercent(monitoring.overall_score)}%`}
                  </Descriptions.Item>
                  <Descriptions.Item label="数据健康">
                    <Tag color={monitoringStatusColor(monitoring?.data_health?.status)}>
                      {monitoringStatusText(monitoring?.data_health?.status)}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="约束饱和">
                    <Tag color={monitoringStatusColor(monitoring?.constraints?.status)}>
                      {monitoringStatusText(monitoring?.constraints?.status)}
                    </Tag>
                  </Descriptions.Item>
                </Descriptions>
                <List
                  dataSource={monitoringAlerts.length
                    ? monitoringAlerts.map((alert) => `${alert.severity} · ${alert.message}`)
                    : [
                      `当前作用域：${selectedAssetPath.map((item) => item.name).join(' / ')}`,
                      '当前选中回路暂无监控告警。',
                      '对未接实时库的页面，先使用历史数据离线评估。',
                      '后续补实时数据源后，可把当前历史仓库替换为 historian provider。',
                    ]}
                  renderItem={(item) => <List.Item>{item}</List.Item>}
                />
              </section>
            </div>
          </>
        );
      case 'loop_board':
      case 'loop_master_data':
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
                  { title: '候选窗口', render: (_: unknown, row: HistoryLoop) => `${row.usable_window_count ?? 0}/${row.window_count ?? 0}` },
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
                  <Typography.Text type="secondary">资产信息、量程、采样与窗口摘要保持等宽对齐，详细趋势放在下方工作区。</Typography.Text>
                </div>
                <Space wrap>
                  <Tag color="blue">{selectedLoop ? LOOP_TYPE_LABEL[selectedLoop.loop_type] ?? selectedLoop.loop_type : '-'}</Tag>
                  <Tag color={loopFeatures ? 'cyan' : 'default'}>{loopFeatures ? 'LoopFeatures 已加载' : 'LoopFeatures 待加载'}</Tag>
                  <Tag color={(selectedLoop?.usable_window_count ?? 0) > 0 ? 'green' : 'red'}>
                    可用窗口 {selectedLoop?.usable_window_count ?? 0}/{selectedLoop?.window_count ?? 0}
                  </Tag>
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
                    <Descriptions.Item label="采样周期">{formatNumber(loopFeatures?.data_profile.sample_time_median_s ?? selectedLoop.sampling_time, 0)}s</Descriptions.Item>
                    <Descriptions.Item label="数据点数">{loopFeatures?.data_profile.row_count ?? selectedLoop.rows}</Descriptions.Item>
                    <Descriptions.Item label="有效点数">{loopFeatures?.data_profile.valid_row_count ?? '-'}</Descriptions.Item>
                    <Descriptions.Item label="总时长">{loopFeatures?.data_profile.duration_h === undefined ? '-' : `${formatNumber(loopFeatures.data_profile.duration_h, 1)}h`}</Descriptions.Item>
                    <Descriptions.Item label="PV 范围">{formatRange(loopFeatures?.pv_stats?.min ?? selectedLoop.pv_min, loopFeatures?.pv_stats?.max ?? selectedLoop.pv_max, 2)}</Descriptions.Item>
                    <Descriptions.Item label="MV 范围">{formatRange(loopFeatures?.mv_stats?.min ?? selectedLoop.mv_min, loopFeatures?.mv_stats?.max ?? selectedLoop.mv_max, 2)}</Descriptions.Item>
                    <Descriptions.Item label="开始时间">{loopFeatures?.data_profile.time_start || selectedLoop.start_time || '-'}</Descriptions.Item>
                    <Descriptions.Item label="结束时间">{loopFeatures?.data_profile.time_end || selectedLoop.end_time || '-'}</Descriptions.Item>
                  </Descriptions>
                </div>
              ) : <Empty description="暂无选中回路" />}
            </section>
            <section className="agent-panel compact-facts">
              <div className="panel-title">LoopFeatures 原始统计</div>
              {loopFeatures ? (
                <Descriptions bordered size="small" column={4} className="industrial-descriptions">
                  <Descriptions.Item label="行数">{loopFeatures.data_profile.row_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="有效行">{loopFeatures.data_profile.valid_row_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="中位采样">{formatNumber(loopFeatures.data_profile.sample_time_median_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="P95 间隔">{formatNumber(loopFeatures.data_profile.sample_interval_p95_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="P99 间隔">{formatNumber(loopFeatures.data_profile.sample_interval_p99_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="不规则采样">{formatPercentValue(loopFeatures.data_profile.irregular_sample_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="长间隔数">{loopFeatures.data_profile.long_gap_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="重复时间戳">{loopFeatures.data_profile.duplicate_timestamp_count ?? '-'}</Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无 LoopFeatures 数据" />}
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
            <section className="agent-panel chart-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">PV / MV 趋势</div>
                  <Typography.Text type="secondary">用于观察当前回路长期波动、激励区间和可能的异常片段。</Typography.Text>
                </div>
                <Space wrap>
                  <Tag color="blue">展示 {series?.sampled_points ?? 0}/{series?.total_points ?? 0} 点</Tag>
                  <Tag color="cyan">采样 {selectedLoop?.sampling_time ?? '-'}s</Tag>
                </Space>
              </div>
              {renderTrend(430)}
            </section>
          </div>
        );
      case 'trend_spectrum':
        return (
          <div className="page-stack">
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
              {renderTrend(420)}
            </section>
            <section className="agent-panel compact-facts">
              <div className="panel-title">频谱与滞后特征</div>
              {assessment ? (
                <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                  <Descriptions.Item label="振荡">{assessment.diagnostics.oscillation?.detected ? '检测到' : '未检测到'}</Descriptions.Item>
                  <Descriptions.Item label="主周期">{String(assessment.diagnostics.oscillation?.period_sec ?? '-')}s</Descriptions.Item>
                  <Descriptions.Item label="SNR">{String(assessment.diagnostics.noise?.snr_db ?? '-')} dB</Descriptions.Item>
                  <Descriptions.Item label="死区证据">{String(assessment.diagnostics.deadzone?.evidence_ratio ?? '-')}</Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无频谱分析" />}
            </section>
          </div>
        );
      case 'data_quality':
      case 'tuning_readiness':
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">{activeSub === 'data_quality' ? '数据质量' : '整定准备度'}</div>
                  <Typography.Text type="secondary">把评分、原始指标和建议动作拆开，避免大卡片堆叠。</Typography.Text>
                </div>
                <Tag color={tagColor(activeSub === 'data_quality' ? assessment?.data_quality.level : assessment?.readiness.level)}>
                  {activeSub === 'data_quality' ? assessment?.data_quality.level ?? '-' : assessment?.readiness.level ?? '-'}
                </Tag>
              </div>
              {renderAssessmentCards()}
              {assessment && (
                <Descriptions bordered column={3} size="small" className="detail-block industrial-descriptions">
                  <Descriptions.Item label="缺失比例">{(assessment.data_quality.missing_ratio * 100).toFixed(2)}%</Descriptions.Item>
                  <Descriptions.Item label="连续性">{scorePercent(assessment.data_quality.continuity_score)}%</Descriptions.Item>
                  <Descriptions.Item label="噪声得分">{scorePercent(assessment.data_quality.noise_score)}%</Descriptions.Item>
                  <Descriptions.Item label="可用窗口">{assessment.identifiability.usable_window_count}/{assessment.identifiability.window_count}</Descriptions.Item>
                  <Descriptions.Item label="最佳窗口">{assessment.identifiability.best_window_source || '-'}</Descriptions.Item>
                  <Descriptions.Item label="最佳窗口分">{assessment.identifiability.best_window_score ?? '-'}</Descriptions.Item>
                </Descriptions>
              )}
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
            <div className="panel-title">振荡诊断</div>
            {assessment ? (
              <Descriptions column={4} bordered size="small" className="industrial-descriptions">
                <Descriptions.Item label="是否振荡">{assessment.diagnostics.oscillation?.detected ? '检测到' : '未检测到'}</Descriptions.Item>
                <Descriptions.Item label="主周期">{String(assessment.diagnostics.oscillation?.period_sec ?? '-')}s</Descriptions.Item>
                <Descriptions.Item label="噪声等级">{String(assessment.diagnostics.noise?.noise_level ?? '-')}</Descriptions.Item>
                <Descriptions.Item label="SNR">{String(assessment.diagnostics.noise?.snr_db ?? '-')} dB</Descriptions.Item>
              </Descriptions>
            ) : <Empty description="暂无诊断结果" />}
          </section>
        );
      case 'model_reliability':
      case 'id_windows':
        return (
          <div className="page-stack">
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
                    <Descriptions.Item label="窗口">{selectedWindow.source}</Descriptions.Item>
                    <Descriptions.Item label="质量分">{selectedWindow.score}</Descriptions.Item>
                    <Descriptions.Item label="相关性">{selectedWindow.corr}</Descriptions.Item>
                    <Descriptions.Item label="激励幅度">{selectedWindow.amplitude}</Descriptions.Item>
                  </Descriptions>
                  {windowPreviewData.length ? (
                    <div className="chart-shell">
                      <Line height={320} data={windowPreviewData} xField="t" yField="value" colorField="series" legend={{ position: 'top-right' }} slider={{}} />
                    </div>
                  ) : <Empty description="暂无窗口预览" />}
                </Space>
              ) : <Empty description="请选择窗口" />}
            </section>
          </div>
        );
      case 'tuning_task':
        return (
          <div className="page-stack">
            <div className="panel-grid">
              <section className="agent-panel">
                <div className="panel-title">发起整定任务</div>
                {selectedLoop ? (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Statistic title="回路位号" value={selectedLoop.loop_id} />
                    <Descriptions column={1} size="small">
                      <Descriptions.Item label="类型">{LOOP_TYPE_LABEL[selectedLoop.loop_type] ?? selectedLoop.loop_type}</Descriptions.Item>
                      <Descriptions.Item label="候选窗口">{selectedLoop.usable_window_count}/{selectedLoop.window_count}</Descriptions.Item>
                      <Descriptions.Item label="指定窗口">{selectedWindow ? `${selectedWindow.source} (#${selectedWindow.index})` : '自动选择'}</Descriptions.Item>
                    </Descriptions>
                    <Space wrap>
                      <Button type="primary" icon={<RocketOutlined />} loading={running} onClick={handleTune}>基于该回路发起整定</Button>
                      {running && <Button danger onClick={handleStopTune}>停止</Button>}
                    </Space>
                  </Space>
                ) : <Empty description="请先选择回路" />}
              </section>
              <section className="agent-panel">
                <div className="panel-title">任务结果摘要</div>
                <Descriptions column={1} size="small">
                  <Descriptions.Item label="任务状态">
                    <Tag color={taskStatus === 'running' ? 'processing' : taskStatus === 'done' ? 'green' : taskStatus === 'error' ? 'red' : 'default'}>
                      {taskStatus === 'running' ? '运行中' : taskStatus === 'done' ? '已完成' : taskStatus === 'error' ? '异常/停止' : '未开始'}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="任务 ID">{taskId || '-'}</Descriptions.Item>
                  <Descriptions.Item label="当前阶段">{taskCurrentStage ? TUNING_STAGE_LABELS[taskCurrentStage] ?? taskCurrentStage : '-'}</Descriptions.Item>
                  <Descriptions.Item label="推荐模型">{taskResult?.model.model_type ?? (taskStageData.identification?.model_type as string | undefined) ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="推荐策略">{taskResult?.pid_params.strategy ?? (taskStageData.tuning?.strategy as string | undefined) ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="综合评分">{formatNumber(taskResult?.evaluation.final_rating ?? (taskStageData.evaluation?.final_rating as number | undefined), 1)}</Descriptions.Item>
                </Descriptions>
              </section>
            </div>
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
                        { label: '实时 Historian / 时序库', value: 'historian' },
                        { label: 'PI / AF Server', value: 'pi_af' },
                        { label: 'OPC UA 实时数据', value: 'opcua' },
                        { label: '关系数据库', value: 'database' },
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
                  <Button disabled={dataSourceType === 'history_upload'}>
                    测试连接
                  </Button>
                  {dataSourceType !== 'history_upload' && <Tag color="orange">按钮已占位，等待后端接口</Tag>}
                </Space>
              </Form>
            </section>

            <section className="agent-panel">
              <div className="panel-title">接入状态</div>
              <Descriptions column={1} bordered size="small">
                <Descriptions.Item label="当前模式">
                  {dataSourceType === 'history_upload' ? '离线历史文件导入' : '实时数据源配置占位'}
                </Descriptions.Item>
                <Descriptions.Item label="已接接口">
                  /api/history/import、/api/history/loops、/api/history/loops/:id/series
                </Descriptions.Item>
                <Descriptions.Item label="待接接口">
                  连接测试、点表同步、实时趋势、报警事件、PID 参数库、设备主数据
                </Descriptions.Item>
                <Descriptions.Item label="已导入回路">{loops.length} 个</Descriptions.Item>
              </Descriptions>
            </section>
          </div>
        );
      default:
        return (
          <section className="agent-panel">
            <div className="panel-title">{currentSub.label}</div>
            <EmptyBackendHint />
            <div className="mock-grid">
              <Statistic title="模拟评分" value={76} suffix="分" />
              <Statistic title="待确认事项" value={3} suffix="项" />
              <Statistic title="建议动作" value="人工确认" />
            </div>
            <Table
              size="small"
              pagination={false}
              dataSource={[
                { key: 1, item: '字段规划', value: '已在前端占位', status: '待后端' },
                { key: 2, item: '数据源', value: '实时库/报警库/PID 参数库', status: '待接入' },
                { key: 3, item: '操作入口', value: '保留按钮与表格结构', status: '可扩展' },
              ]}
              columns={[
                { title: '项目', dataIndex: 'item' },
                { title: '内容', dataIndex: 'value' },
                { title: '状态', dataIndex: 'status', render: (value: string) => <Tag color="orange">{value}</Tag> },
              ]}
            />
          </section>
        );
    }
  };

  return (
    <div className="agent-console">
      <header className="agent-header">
        <div className="industrial-topbar">
          <div className="agent-brand">
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
              <p>方案一：经典监控总览</p>
            </div>
          </div>
          <div className="system-meta">
            <span className="status-dot" />
            <span>通信正常</span>
            <span><DesktopOutlined /> SCADA服务器01</span>
            <span><ClockCircleOutlined /> {new Date().toLocaleString()}</span>
            <span><UserOutlined /> admin</span>
            <span className="alarm-pill"><BellOutlined /> 6</span>
          </div>
        </div>
      </header>

      <main className={sidebarCollapsed ? 'agent-main industrial-main sidebar-collapsed' : 'agent-main industrial-main'}>
        <aside className={sidebarCollapsed ? 'side-menu industrial-tree collapsed' : 'side-menu industrial-tree'}>
          <div className="side-title">导航菜单</div>
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
                        {!sub.implemented && <em>待接</em>}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </aside>

        <section className="content-area">
          <div className="workspace-tabs">
            <button className="active">总览</button>
            <button>{currentSub.label}</button>
          </div>
          <div className={showContextRail ? 'industrial-content-shell' : 'industrial-content-shell no-context-rail'}>
            <div className="primary-workspace">
              <div className="content-head">
                <div>
                  <Typography.Title level={2}>{currentSub.label}</Typography.Title>
                  <Typography.Text type="secondary">
                    作用域：{selectedAssetPath.map((item) => item.name).join(' / ')} ·
                    当前选中：{selectedLoop?.loop_id ?? '暂无回路'} · 数据模式：历史导入
                  </Typography.Text>
                </div>
                <Space>
                  <Button onClick={() => switchTo('settings', 'asset_directory')}>装置目录</Button>
                  <BackendBadge implemented={currentSub.implemented} />
                  <Button icon={<SyncOutlined />} onClick={loadLoops} loading={loading}>刷新数据</Button>
                </Space>
              </div>
              {renderPage()}
            </div>

            {showContextRail && (
              <>
                <aside className="right-rail">
                  <section className="rail-panel">
                    <div className="rail-title">关键参数</div>
                    <div className="rail-metrics">
                      <div>
                        <span>回路位号</span>
                        <strong>{selectedLoop?.loop_id ?? '-'}</strong>
                      </div>
                      <div>
                        <span>类型</span>
                        <strong>{selectedLoop ? LOOP_TYPE_LABEL[selectedLoop.loop_type] ?? selectedLoop.loop_type : '-'}</strong>
                      </div>
                      <div>
                        <span>采样周期</span>
                        <strong>{selectedLoop ? `${formatNumber(loopFeatures?.data_profile.sample_time_median_s ?? selectedLoop.sampling_time, 0)}s` : '-'}</strong>
                      </div>
                      <div>
                        <span>监控状态</span>
                        <strong>{monitoringStatusText(loopMonitoring?.monitoring.status)}</strong>
                      </div>
                      <div>
                        <span>监控综合分</span>
                        <strong>{loopMonitoring?.monitoring.overall_score === undefined ? '-' : `${scorePercent(loopMonitoring.monitoring.overall_score)}%`}</strong>
                      </div>
                      <div>
                        <span>MV饱和比例</span>
                        <strong>{formatPercentValue(loopFeatures?.constraint_raw?.mv_saturation_ratio, 2)}</strong>
                      </div>
                    </div>
                  </section>

                  <section className="rail-panel">
                    <div className="rail-title">当前整定态势</div>
                    <div className="rail-state">
                      <Tag color={taskStatus === 'running' ? 'processing' : taskStatus === 'error' ? 'red' : taskStatus === 'done' ? 'green' : 'default'}>
                        {taskStatus === 'running' ? '运行中' : taskStatus === 'done' ? '已完成' : taskStatus === 'error' ? '异常' : '空闲'}
                      </Tag>
                      <strong>{taskResult?.pid_params.strategy ?? (taskStageData.tuning?.strategy as string | undefined) ?? '等待任务'}</strong>
                      <span>阶段：{taskCurrentStage ? TUNING_STAGE_LABELS[taskCurrentStage] ?? taskCurrentStage : '-'}</span>
                      <span>评分：{formatNumber(taskResult?.evaluation.final_rating ?? (taskStageData.evaluation?.final_rating as number | undefined), 1)}</span>
                    </div>
                  </section>

                  <section className="rail-panel">
                    <div className="rail-title">快捷操作</div>
                    <div className="rail-actions">
                      <Button size="small" onClick={() => switchTo('monitor', 'loop_profile')}>单回路画像</Button>
                      <Button size="small" onClick={() => switchTo('tuning', 'tuning_task')}>发起整定</Button>
                      <Button size="small" onClick={() => switchTo('tuning', 'id_windows')}>窗口与辨识</Button>
                      <Button size="small" onClick={() => switchTo('assessment', 'data_quality')}>质量评估</Button>
                    </div>
                  </section>
                </aside>

                <section className="alarm-strip">
                  <div className="rail-title">报警事件 / Agent 事件</div>
                  <Table
                    size="small"
                    pagination={false}
                    rowKey="key"
                    dataSource={railAlarms}
                    columns={[
                      { title: '时间', dataIndex: 'time', width: 120 },
                      { title: '级别', dataIndex: 'level', width: 80, render: (value: string) => <Tag color={alertSeverityColor(value)}>{value}</Tag> },
                      { title: '名称', dataIndex: 'name', width: 150 },
                      { title: '描述', dataIndex: 'value', ellipsis: true },
                      { title: '状态', dataIndex: 'status', width: 100 },
                    ]}
                  />
                </section>
              </>
            )}
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
