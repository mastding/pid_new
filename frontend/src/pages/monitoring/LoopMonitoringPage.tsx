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
  Modal,
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
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloudUploadOutlined,
  DatabaseOutlined,
  DeploymentUnitOutlined,
  DownOutlined,
  ExperimentOutlined,
  FileSearchOutlined,
  FundProjectionScreenOutlined,
  HistoryOutlined,
  KeyOutlined,
  LineChartOutlined,
  MenuOutlined,
  RadarChartOutlined,
  RobotOutlined,
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
  fetchModelConfig,
  fetchPolicyConfig,
  getSession,
  listSessions,
  testModelConfig,
  updateModelConfig,
} from '@/services/api';
import type {
  HistoryLoop,
  HistoryLoopAssessment,
  HistoryLoopFeatures,
  HistoryLoopMonitoring,
  HistoryWindow,
  LoopSeriesResp,
  ModelConfig,
  PolicyConfig,
} from '@/services/api';
import type {
  IdentificationAttempt,
  IdentificationRefinementMeta,
  LlmThinkingEvent,
  ModelReviewMeta,
  PipelineEvent,
  StrategyCandidate,
  TuningResult,
  WindowAlgorithmFitSummary,
  WindowSelectionMeta,
} from '@/types/tuning';
import './LoopMonitoringPage.css';

type ModuleKey = 'workspace' | 'monitor' | 'assessment' | 'diagnostics' | 'tuning' | 'experience' | 'settings';
type SubKey =
  | 'dashboard' | 'todo' | 'shift_tasks' | 'risk_alerts'
  | 'loop_board' | 'loop_profile' | 'trend_spectrum' | 'data_quality' | 'oscillation_diagnosis' | 'constraint_monitor' | 'alarm_events'
  | 'performance_score' | 'condition_recognition' | 'actuator_status' | 'tuning_readiness'
  | 'diagnosis_overview' | 'pid_diagnosis' | 'valve_diagnosis' | 'measurement_noise_diagnosis' | 'process_disturbance_diagnosis' | 'model_reliability'
  | 'tuning_task' | 'tuning_prior' | 'id_windows' | 'pid_candidates' | 'release_confirm'
  | 'case_library' | 'rule_library' | 'knowledge_graph' | 'model_versions'
  | 'data_sources' | 'asset_directory' | 'rule_config' | 'model_config';

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
      { key: 'dashboard', label: '总览驾驶舱', icon: <FundProjectionScreenOutlined />, implemented: true },
      { key: 'todo', label: '待处理回路', icon: <AlertOutlined /> },
      { key: 'shift_tasks', label: '本班任务', icon: <AuditOutlined /> },
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
      { key: 'data_quality', label: '数据质量', icon: <SafetyCertificateOutlined />, implemented: true },
      { key: 'loop_profile', label: '单回路画像', icon: <FileSearchOutlined />, implemented: true },
    ],
  },
  {
    key: 'assessment',
    label: '回路评估',
    icon: <AuditOutlined />,
    subs: [
      { key: 'performance_score', label: '控制性能', icon: <FundProjectionScreenOutlined />, implemented: true },
      { key: 'condition_recognition', label: '运行工况', icon: <BranchesOutlined />, implemented: true },
      { key: 'actuator_status', label: '执行机构状态', icon: <ToolOutlined />, implemented: true },
      { key: 'tuning_readiness', label: '整定准备度', icon: <RocketOutlined />, implemented: true },
    ],
  },
  {
    key: 'diagnostics',
    label: '根因诊断',
    icon: <DeploymentUnitOutlined />,
    subs: [
      { key: 'diagnosis_overview', label: '诊断总览', icon: <FileSearchOutlined />, implemented: true },
      { key: 'pid_diagnosis', label: 'PID参数诊断', icon: <SettingOutlined />, implemented: true },
      { key: 'valve_diagnosis', label: '阀门/执行机构', icon: <ToolOutlined />, implemented: true },
      { key: 'measurement_noise_diagnosis', label: '测量与噪声', icon: <SafetyCertificateOutlined />, implemented: true },
      { key: 'process_disturbance_diagnosis', label: '扰动与工艺', icon: <BranchesOutlined />, implemented: true },
      { key: 'model_reliability', label: '模型可靠性', icon: <ExperimentOutlined />, implemented: true },
    ],
  },
  {
    key: 'tuning',
    label: '整定中心',
    icon: <RocketOutlined />,
    subs: [
      { key: 'tuning_task', label: '整定任务', icon: <RocketOutlined />, implemented: true },
      { key: 'tuning_prior', label: '整定先验', icon: <AuditOutlined />, implemented: true },
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
      { key: 'rule_config', label: '规则配置', icon: <FileSearchOutlined />, implemented: true },
      { key: 'model_config', label: '模型配置', icon: <RobotOutlined />, implemented: true },
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

function attemptFitKey(attempt: IdentificationAttempt) {
  return [
    attempt.round ?? 0,
    attempt.window_source ?? '',
    attempt.window_algorithm ?? '',
    attempt.model_type ?? '',
    attempt.fit_score ?? '',
  ].join('|');
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
  const [assessmentLoading, setAssessmentLoading] = useState(false);
  const [assessmentError, setAssessmentError] = useState<string | null>(null);
  const [loopFeatures, setLoopFeatures] = useState<HistoryLoopFeatures | null>(null);
  const [loopMonitoring, setLoopMonitoring] = useState<HistoryLoopMonitoring | null>(null);
  const [monitoringByLoopId, setMonitoringByLoopId] = useState<Record<string, HistoryLoopMonitoring>>({});
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
  const [dataSourceType, setDataSourceType] = useState<string>('historian');
  const [taskDetailOpen, setTaskDetailOpen] = useState(false);
  const [rawLogExpanded, setRawLogExpanded] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [expandedModules, setExpandedModules] = useState<Record<ModuleKey, boolean>>(INITIAL_EXPANDED_MODULES);

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

  const currentModule = MODULES.find((item) => item.key === activeModule) ?? MODULES[0];
  const currentSub = currentModule.subs.find((item) => item.key === activeSub) ?? currentModule.subs[0];

  const selectedLoop = useMemo(
    () => loops.find((item) => item.loop_id === selectedLoopId),
    [loops, selectedLoopId],
  );

  useEffect(() => {
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
  }, [running, selectedLoop, taskResult]);

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

  const dashboardRows = useMemo(() => scopedLoops.map((loop) => {
    const snapshot = monitoringByLoopId[loop.loop_id]?.monitoring;
    const fallbackScore = loop.best_window_score ?? 0;
    const overallScore = snapshot?.overall_score ?? fallbackScore;
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
    const normalCount = Math.max(scopedLoops.length - warningCount - alarmCount, 0);
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
    const stageComparison = taskStageData.identification?.algorithm_comparison;
    const resultComparison = taskResult?.model.algorithm_comparison;
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
      { key: 'stable', time: '当前', level: '中', name: '窗口质量', value: `可用窗口 ${selectedLoop?.usable_window_count ?? 0}/${selectedLoop?.window_count ?? 0}`, status: '跟踪', recommendation: '', evidence: '' },
      { key: 'source', time: '当前', level: '低', name: '数据源', value: dataSourceType === 'history' ? '历史文件导入' : '历史仓库/实时库', status: '正常', recommendation: '', evidence: '' },
      { key: 'task', time: taskStartedAt ? new Date(taskStartedAt).toLocaleTimeString() : '未启动', level: taskStatus === 'error' ? '高' : '低', name: '整定任务', value: taskId ? `任务 ${taskId}` : '暂无运行任务', status: taskStatus === 'done' ? '完成' : taskStatus === 'running' ? '运行' : '空闲', recommendation: '', evidence: '' },
    ];
  }, [assessment?.diagnostics.flags, dataSourceType, loopMonitoring, selectedLoop, taskId, taskStartedAt, taskStatus]);

  const switchTo = (moduleKey: ModuleKey, subKey: SubKey) => {
    setActiveModule(moduleKey);
    setActiveSub(subKey);
    setExpandedModules((prev) => ({ ...prev, [moduleKey]: true }));
  };

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
    setAssessmentError(null);
    setAssessmentLoading(true);
    try {
      const resp = await getHistoryLoopAssessment(loopId);
      if (resp.error) message.warning(resp.error);
      setAssessment(resp);
    } catch (error) {
      setAssessmentError(String(error));
      message.error(`加载回路评估失败：${String(error)}`);
    } finally {
      setAssessmentLoading(false);
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
      setMonitoringByLoopId((prev) => ({ ...prev, [loopId]: resp }));
      setLoopFeatures((current) => current ?? resp.features ?? null);
    } catch (error) {
      message.error(`加载监控快照失败：${String(error)}`);
    }
  }, []);

  const loadWindows = useCallback(async (loopId: string) => {
    setWindows([]);
    setWindowAlgorithmSummary({});
    setSelectedWindowIndex(undefined);
    try {
      const resp = await getHistoryLoopWindows(loopId);
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

  useEffect(() => {
    let cancelled = false;
    const missing = scopedLoops
      .map((loop) => loop.loop_id)
      .filter((loopId) => !monitoringByLoopId[loopId]);
    if (!missing.length) return undefined;
    Promise.allSettled(missing.map((loopId) => fetchHistoryLoopMonitoring(loopId))).then((results) => {
      if (cancelled) return;
      setMonitoringByLoopId((prev) => {
        const next = { ...prev };
        results.forEach((result) => {
          if (result.status === 'fulfilled') {
            next[result.value.loop_id] = result.value;
          }
        });
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [monitoringByLoopId, scopedLoops]);

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

  const startTune = () => {
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

  const handleTune = () => {
    if (!selectedLoop) {
      message.warning('请先选择一个回路');
      return;
    }
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
        onOk: startTune,
      });
      return;
    }
    startTune();
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

  const renderTrend = (height = 360) => (
    trendData.length ? (
      <>
        <div className="chart-axis-note">
          <span>X 轴：时间 / 采样点</span>
          <span>Y 轴：PV / SV / MV 数值</span>
        </div>
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
          {Array.isArray(taskStageData.identification?.algorithm_comparison) && taskStageData.identification.algorithm_comparison.length ? (
            <Table
              className="detail-block"
              size="small"
              pagination={false}
              rowKey={(row) => `${row.algorithm}-${row.window_source}-${row.model_type}`}
              dataSource={taskStageData.identification.algorithm_comparison as Array<Record<string, unknown>>}
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
    const oscillationDetected = Boolean(monitoring?.stability?.oscillation_detected ?? assessment?.diagnostics.oscillation?.detected);

    switch (activeSub) {
      case 'dashboard':
        return (
          <div className="page-stack dashboard-page">
            <section className="agent-panel dashboard-scope-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">装置运行总览</div>
                  <Typography.Text type="secondary">
                    当前作用域：{selectedAssetPath.map((item) => item.name).join(' / ')}
                  </Typography.Text>
                </div>
                <Space wrap>
                  <Tag color={assetTagColor(selectedAssetNode?.type ?? 'factory')}>
                    {selectedAssetNode ? ASSET_TYPE_LABEL[selectedAssetNode.type] : '-'}
                  </Tag>
                  <Tag color="blue">历史导入</Tag>
                  <Button size="small" onClick={() => switchTo('settings', 'asset_directory')}>切换装置</Button>
                </Space>
              </div>
              <Descriptions bordered size="small" column={4} className="industrial-descriptions">
                <Descriptions.Item label="数据范围" span={2}>{dashboardStats.dataStart || '-'} ~ {dashboardStats.dataEnd || '-'}</Descriptions.Item>
                <Descriptions.Item label="采样周期">{selectedLoop ? `${formatNumber(selectedLoop.sampling_time, 0)}s` : '-'}</Descriptions.Item>
                <Descriptions.Item label="接入回路">{scopedLoopStats.loopCount} 个</Descriptions.Item>
              </Descriptions>
            </section>

            <div className="kpi-grid dashboard-kpi-grid">
              <Statistic title="当前范围回路" value={scopedLoopStats.loopCount} suffix="个" />
              <Statistic title="正常回路" value={dashboardStats.normalCount} suffix="个" />
              <Statistic title="关注/告警" value={dashboardStats.warningCount + dashboardStats.alarmCount} suffix="个" />
              <Statistic
                title="平均监控分"
                value={dashboardStats.avgScore === undefined ? '-' : scorePercent(dashboardStats.avgScore)}
                suffix={dashboardStats.avgScore === undefined ? undefined : '%'}
              />
              <Statistic title="可整定回路" value={scopedLoops.filter((item) => (item.usable_window_count ?? 0) > 0).length} suffix="个" />
              <Statistic title="窗口可用率" value={scopedLoopStats.windowRatio} suffix="%" />
            </div>

            <div className="panel-grid two dashboard-main-grid">
              <section className="agent-panel">
                <div className="panel-toolbar">
                  <div>
                    <div className="panel-title">TOP 待处理回路</div>
                    <Typography.Text type="secondary">按状态、综合分、告警数和窗口可用性排序。</Typography.Text>
                  </div>
                  <Tag color={dashboardStats.alertCount ? 'orange' : 'green'}>{dashboardStats.alertCount} 条告警</Tag>
                </div>
                <Table
                  size="small"
                  pagination={false}
                  rowKey={(row) => row.loop.loop_id}
                  dataSource={dashboardRows.slice(0, 6)}
                  columns={[
                    {
                      title: '回路',
                      render: (_: unknown, row: (typeof dashboardRows)[number]) => (
                        <Button type="link" onClick={() => setSelectedLoopId(row.loop.loop_id)}>
                          {row.loop.loop_id}
                        </Button>
                      ),
                    },
                    {
                      title: '状态',
                      width: 90,
                      render: (_: unknown, row: (typeof dashboardRows)[number]) => (
                        <Tag color={monitoringStatusColor(row.snapshot?.status)}>
                          {monitoringStatusText(row.snapshot?.status)}
                        </Tag>
                      ),
                    },
                    { title: '综合分', width: 90, render: (_: unknown, row: (typeof dashboardRows)[number]) => row.snapshot?.overall_score === undefined ? '-' : `${scorePercent(row.snapshot.overall_score)}%` },
                    { title: 'PV/MV行为', width: 110, render: (_: unknown, row: (typeof dashboardRows)[number]) => row.snapshot?.pv_mv_behavior?.score === undefined ? '-' : `${scorePercent(row.snapshot.pv_mv_behavior.score)}%` },
                    { title: '约束', width: 90, render: (_: unknown, row: (typeof dashboardRows)[number]) => row.snapshot?.constraints?.score === undefined ? '-' : `${scorePercent(row.snapshot.constraints.score)}%` },
                    { title: '窗口', width: 90, render: (_: unknown, row: (typeof dashboardRows)[number]) => `${row.loop.usable_window_count ?? 0}/${row.loop.window_count ?? 0}` },
                    {
                      title: '操作',
                      width: 150,
                      render: (_: unknown, row: (typeof dashboardRows)[number]) => (
                        <Space>
                          <Button size="small" onClick={() => {
                            setSelectedLoopId(row.loop.loop_id);
                            switchTo('monitor', 'loop_profile');
                          }}>画像</Button>
                          <Button size="small" onClick={() => {
                            setSelectedLoopId(row.loop.loop_id);
                            switchTo('tuning', 'tuning_task');
                          }}>整定</Button>
                        </Space>
                      ),
                    },
                  ]}
                />
              </section>

              <section className="agent-panel">
                <div className="panel-toolbar">
                  <div className="panel-title">Agent 本班建议</div>
                  <Tag color={dashboardStats.warningCount || dashboardStats.alarmCount ? 'orange' : 'green'}>
                    {dashboardStats.warningCount || dashboardStats.alarmCount ? '需要关注' : '运行平稳'}
                  </Tag>
                </div>
                <List
                  dataSource={[
                    `当前装置范围 ${scopedLoopStats.loopCount} 个回路，平均监控分 ${dashboardStats.avgScore === undefined ? '-' : `${scorePercent(dashboardStats.avgScore)}%`}。`,
                    dashboardStats.warningCount || dashboardStats.alarmCount
                      ? `优先处理 ${dashboardRows[0]?.loop.loop_id ?? '-'}，该回路综合分最低或存在告警。`
                      : '当前没有严重告警，建议按窗口质量和PV/MV行为评分安排巡检。',
                    scopedLoopStats.windowRatio >= 60
                      ? `当前窗口可用率 ${scopedLoopStats.windowRatio}%，已有较多历史激励可用于离线整定筛选。`
                      : `当前窗口可用率 ${scopedLoopStats.windowRatio}%，建议先补充可辨识激励或延长历史数据。`,
                    'SP/SV 字段暂未接入时，设定值跟踪和偏差类指标会标记为不可用，后续接实时库后再纳入总分。',
                  ]}
                  renderItem={(item) => <List.Item>{item}</List.Item>}
                />
              </section>
            </div>

            <div className="panel-grid two dashboard-main-grid">
              <section className="agent-panel chart-panel">
                <div className="panel-toolbar">
                  <div>
                    <div className="panel-title">选中回路趋势预览</div>
                    <Typography.Text type="secondary">
                      {selectedLoop?.loop_id ?? '-'} · 用于快速判断 PV/MV 波动、饱和和激励片段。
                    </Typography.Text>
                  </div>
                  <Space wrap>
                    <Button size="small" onClick={() => switchTo('monitor', 'loop_profile')}>查看画像</Button>
                    <Button size="small" onClick={() => switchTo('monitor', 'trend_spectrum')}>趋势与频谱</Button>
                  </Space>
                </div>
                {renderTrend(320)}
              </section>

              <section className="agent-panel">
                <div className="panel-title">选中回路监控快照</div>
                <Descriptions bordered size="small" column={2} className="industrial-descriptions">
                  <Descriptions.Item label="监控状态">
                    <Tag color={monitoringStatusColor(monitoring?.status)}>{monitoringStatusText(monitoring?.status)}</Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="综合分">{monitoring?.overall_score === undefined ? '-' : `${scorePercent(monitoring.overall_score)}%`}</Descriptions.Item>
                  <Descriptions.Item label="数据健康">{monitoring?.data_health?.score === undefined ? '-' : `${scorePercent(monitoring.data_health.score)}%`}</Descriptions.Item>
                  <Descriptions.Item label="稳定性">{monitoring?.stability?.score === undefined ? '-' : `${scorePercent(monitoring.stability.score)}%`}</Descriptions.Item>
                  <Descriptions.Item label="PV/MV行为">{monitoring?.pv_mv_behavior?.score === undefined ? '-' : `${scorePercent(monitoring.pv_mv_behavior.score)}%`}</Descriptions.Item>
                  <Descriptions.Item label="约束饱和">{monitoring?.constraints?.score === undefined ? '-' : `${scorePercent(monitoring.constraints.score)}%`}</Descriptions.Item>
                  <Descriptions.Item label="MV饱和比例">{formatPercentValue(loopFeatures?.constraint_raw?.mv_saturation_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="可用窗口">{selectedLoop ? `${selectedLoop.usable_window_count ?? 0}/${selectedLoop.window_count ?? 0}` : '-'}</Descriptions.Item>
                </Descriptions>
              </section>
            </div>
          </div>
        );
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
                  <Typography.Text type="secondary">集中展示资产信息、量程、采样、窗口、原始统计与约束饱和摘要；趋势曲线统一放到“趋势与频谱”。</Typography.Text>
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
                  <Descriptions.Item label="采样不规则">{formatPercentValue(monitoring?.data_health?.irregular_sample_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="长间隔">{monitoring?.data_health?.long_gap_count ?? 0} 个</Descriptions.Item>
                  <Descriptions.Item label="重复时间戳">{formatPercentValue(monitoring?.data_health?.duplicate_timestamp_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="PV噪声比">{formatPercentValue(monitoring?.data_health?.pv_noise_ratio, 2)}</Descriptions.Item>
                  <Descriptions.Item label="PV SNR">{formatNumber(monitoring?.data_health?.pv_snr_db, 2)} dB</Descriptions.Item>
                  <Descriptions.Item label="PV尖峰">{monitoring?.data_health?.pv_spike_count ?? 0} 个</Descriptions.Item>
                  <Descriptions.Item label="PV离群">{monitoring?.data_health?.pv_outlier_count ?? 0} 个</Descriptions.Item>
                  <Descriptions.Item label="MV离群">{monitoring?.data_health?.mv_outlier_count ?? 0} 个</Descriptions.Item>
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
                <Tag color={taskResult?.evaluation.passed ? 'green' : 'orange'}>
                  {taskResult ? (taskResult.evaluation.passed ? '可接受' : '需要优化') : '等待评估'}
                </Tag>
              </div>
              {taskResult ? (
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
                <Tag color={(loopFeatures?.actuator_profile?.mv_saturation_ratio ?? 0) > 0.05 ? 'orange' : 'green'}>
                  {(loopFeatures?.actuator_profile?.mv_saturation_ratio ?? 0) > 0.05 ? '需关注' : '正常'}
                </Tag>
              </div>
              {loopFeatures ? (
                <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                  <Descriptions.Item label="MV分辨率">{formatNumber(loopFeatures.actuator_profile?.mv_resolution_hint, 5)}</Descriptions.Item>
                  <Descriptions.Item label="死区迹象">{formatPercentValue(loopFeatures.actuator_profile?.mv_deadband_hint_ratio, 1)}</Descriptions.Item>
                  <Descriptions.Item label="黏滞迹象">{yesNo(loopFeatures.actuator_profile?.mv_stiction_hint)}</Descriptions.Item>
                  <Descriptions.Item label="最长不动作">{formatNumber(loopFeatures.actuator_profile?.longest_mv_stuck_duration_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="速率限制迹象">{yesNo(loopFeatures.actuator_profile?.mv_rate_limit_hint)}</Descriptions.Item>
                  <Descriptions.Item label="低限余量">{formatNumber(loopFeatures.actuator_profile?.mv_saturation_margin_low, 3)}</Descriptions.Item>
                  <Descriptions.Item label="高限余量">{formatNumber(loopFeatures.actuator_profile?.mv_saturation_margin_high, 3)}</Descriptions.Item>
                  <Descriptions.Item label="MV饱和比例">{formatPercentValue(loopFeatures.actuator_profile?.mv_saturation_ratio, 2)}</Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无执行机构特征" />}
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
      case 'tuning_prior':
        return (
          <div className="page-stack">
            <section className="agent-panel">
              <div className="panel-toolbar">
                <div>
                  <div className="panel-title">整定先验</div>
                  <Typography.Text type="secondary">
                    汇总量程、过程方向、粗增益、滞后和 T 搜索范围，供后续辨识约束和 PID 整定策略使用。
                  </Typography.Text>
                </div>
                <Tag color={loopFeatures?.process_prior?.k_sign_constraint === 'unknown' ? 'orange' : 'green'}>
                  K符号：{loopFeatures?.process_prior?.k_sign_constraint ?? '-'}
                </Tag>
              </div>
              {loopFeatures ? (
                <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                  <Descriptions.Item label="PV有效量程">
                    {formatRange(loopFeatures.scale_profile?.pv?.effective_min, loopFeatures.scale_profile?.pv?.effective_max, 3)}
                  </Descriptions.Item>
                  <Descriptions.Item label="MV有效量程">
                    {formatRange(loopFeatures.scale_profile?.mv?.effective_min, loopFeatures.scale_profile?.mv?.effective_max, 3)}
                  </Descriptions.Item>
                  <Descriptions.Item label="MV量纲">{loopFeatures.scale_profile?.mv_scale_type ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="PV类型">{loopFeatures.scale_profile?.pv_range_type ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="过程方向">{formatProcessDirection(loopFeatures.process_prior?.process_direction)}</Descriptions.Item>
                  <Descriptions.Item label="方向置信度">{formatPercentValue(loopFeatures.process_prior?.process_direction_confidence, 1)}</Descriptions.Item>
                  <Descriptions.Item label="粗增益">{formatNumber(loopFeatures.process_prior?.static_gain_hint, 5)}</Descriptions.Item>
                  <Descriptions.Item label="增益样本数">{loopFeatures.process_prior?.gain_sample_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="响应滞后">{formatNumber(loopFeatures.process_prior?.response_lag_hint_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="T先验下界">{formatNumber(loopFeatures.process_prior?.time_constant_prior_min_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="T先验上界">{formatNumber(loopFeatures.process_prior?.time_constant_prior_max_s, 1)}s</Descriptions.Item>
                  <Descriptions.Item label="先验依据">{loopFeatures.process_prior?.time_constant_prior_basis ?? '-'}</Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无整定先验" />}
            </section>
            <section className="agent-panel">
              <div className="panel-title">激励质量</div>
              {loopFeatures ? (
                <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                  <Descriptions.Item label="激励等级">{loopFeatures.excitation_profile?.excitation_level ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="MV激励跨度">{formatNumber(loopFeatures.excitation_profile?.mv_excitation_span, 3)}</Descriptions.Item>
                  <Descriptions.Item label="MV有效激励跨度">{formatNumber(loopFeatures.excitation_profile?.mv_effective_excitation_span, 3)}</Descriptions.Item>
                  <Descriptions.Item label="MV激励事件">{loopFeatures.excitation_profile?.mv_excitation_event_count ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="PV响应比例">{formatPercentValue(loopFeatures.excitation_profile?.pv_response_after_mv_ratio, 1)}</Descriptions.Item>
                  <Descriptions.Item label="非饱和比例">{formatPercentValue(loopFeatures.excitation_profile?.saturation_free_ratio, 1)}</Descriptions.Item>
                  <Descriptions.Item label="可用激励比例">{formatPercentValue(loopFeatures.excitation_profile?.usable_excitation_ratio, 1)}</Descriptions.Item>
                  <Descriptions.Item label="MV斜坡事件">{loopFeatures.excitation_profile?.mv_ramp_event_count ?? '-'}</Descriptions.Item>
                </Descriptions>
              ) : <Empty description="暂无激励质量特征" />}
            </section>
          </div>
        );
      case 'model_reliability':
      case 'id_windows':
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
                  <Typography.Text type="secondary">先选择具体回路，再基于当前数据、候选窗口和准入校验发起整定。</Typography.Text>
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
                  <Button type="primary" icon={<RocketOutlined />} loading={running} disabled={!selectedLoop} onClick={handleTune}>
                    发起整定
                  </Button>
                  {running && <Button danger onClick={handleStopTune}>停止</Button>}
                </Space>
              </div>

              {selectedLoop ? (
                <div className="tuning-launch-summary">
                  <Statistic title="当前整定回路" value={selectedLoop.loop_id} />
                  <Descriptions bordered column={4} size="small" className="industrial-descriptions">
                      <Descriptions.Item label="类型">{LOOP_TYPE_LABEL[selectedLoop.loop_type] ?? selectedLoop.loop_type}</Descriptions.Item>
                      <Descriptions.Item label="候选窗口">{selectedLoop.usable_window_count}/{selectedLoop.window_count}</Descriptions.Item>
                      <Descriptions.Item label="指定窗口">{selectedWindow ? `${selectedWindow.source} (#${selectedWindow.index})` : '自动选择'}</Descriptions.Item>
                      <Descriptions.Item label="当前准入">
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
              {assessmentLoading ? (
                <Alert className="agent-alert" type="info" showIcon message="正在从后端加载整定准入校验" description="调用 /api/history/loops/{loop_id}/assessment，返回后会展示 tuning_readiness.gate_checks。" />
              ) : assessmentError ? (
                <Alert className="agent-alert" type="error" showIcon message="整定准入后端接口调用失败" description={assessmentError} />
              ) : assessment ? (
                <div className="page-stack compact-stack">
                  {tuningGate.nextAction && (
                    <Alert
                      className="agent-alert"
                      type={tuningGate.hardBlocked ? 'error' : tuningGate.caution ? 'warning' : 'success'}
                      showIcon
                      message={assessment.summary?.decision_text ?? gateDecisionText(tuningGate.decision)}
                      description={tuningGate.nextAction}
                    />
                  )}
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
                      { title: '说明', dataIndex: 'message' },
                    ]}
                  />
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
                <Descriptions.Item label="推荐模型">{taskResult?.model.model_type ?? (taskStageData.identification?.model_type as string | undefined) ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="推荐策略">{taskResult?.pid_params.strategy ?? (taskStageData.tuning?.strategy as string | undefined) ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="综合评分">{formatNumber(taskResult?.evaluation.final_rating ?? (taskStageData.evaluation?.final_rating as number | undefined), 1)}</Descriptions.Item>
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
            </div>
          </div>
          <div className="system-meta">
            <span style={{ color: '#51a7ff', fontWeight: 700 }}>V1.0</span>
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
