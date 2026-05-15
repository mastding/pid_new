import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Dayjs } from 'dayjs';
import {
  Empty,
  Form,
  Input,
  Modal,
  Space,
  Typography,
  message,
} from 'antd';
import type { UploadFile } from 'antd';
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
import { ClassicSideMenu } from '@/features/app-shell/ClassicSideMenu';
import { INITIAL_EXPANDED_MODULES, LOOP_TYPE_LABEL, MODULES, type ModuleKey, type SubKey } from '@/features/app-shell/navigation';
import { PidAppTopbar } from '@/features/app-shell/PidAppTopbar';
import { SectionErrorBoundary } from '@/features/app-shell/SectionErrorBoundary';
import { chartLineTooltip, LoopTrendChart } from '@/features/charts/LoopTrendChart';
import { DashboardCockpitPanel } from '@/features/dashboard/DashboardCockpitPanel';
import {
  DASHBOARD_WIDGET_STORAGE_KEY,
  DEFAULT_DASHBOARD_WIDGET_KEYS,
  buildDashboardRows,
  normalizeDashboardWidgetKeys,
  summarizeDashboardRows,
  type DashboardWidgetKey,
} from '@/features/dashboard/model';
import { DialogueChatPanel } from '@/features/dialogue/DialogueChatPanel';
import { DialogueHistoryPanel } from '@/features/dialogue/DialogueHistoryPanel';
import {
  buildDialogueActions,
  formatAssistantEvent,
  normalizeAssistantAction,
  type AssistantAction,
  type AssistantEventItem,
  type AssistantMessage,
} from '@/features/dialogue/model';
import { DIALOGUE_STARTER_PROMPTS } from '@/features/dialogue/prompts';
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
import { TrendSpectrumPanel } from '@/features/loop-monitoring/TrendSpectrumPanel';
import { TuningReadinessPanel } from '@/features/loop-monitoring/TuningReadinessPanel';
import { AssessmentCards } from '@/features/monitoring/AssessmentCards';
import { LoopSelectionTable, WindowSelectionTable } from '@/features/monitoring/SelectionTables';
import {
  alertSeverityColor,
  conditionEvidenceDetail,
  conditionEvidenceName,
  conditionRecommendationText,
  evidenceStatusColor,
  evidenceStatusText,
  formatCpkBasis,
  formatHarrisBasis,
  formatNumber,
  formatOscillationEvidence,
  formatOscillationPhaseHint,
  formatPercentValue,
  formatProcessDirection,
  formatProcessDirectionBasis,
  formatRange,
  gateCheckLabel,
  gateCheckMessage,
  gateDecisionText,
  gateImpact,
  gateSeverityColor,
  monitoringStatusColor,
  monitoringStatusText,
  operatingConditionText,
  policyLoopImpact,
  scorePercent,
  scoreStatus,
  tagColor,
  tuningSuitabilityColor,
  tuningSuitabilityText,
  yesNo,
} from '@/features/monitoring/formatters';
import {
  ASSESSMENT_DETAIL_SUBS,
  FEATURE_DETAIL_SUBS,
  FEATURE_RANGE_OPTIONS,
  MONITORING_DETAIL_SUBS,
  TREND_POINT_LIMIT_OPTIONS,
  TREND_PRESET_OPTIONS,
  WINDOW_DETAIL_SUBS,
  buildFeatureRangeParams as buildFeatureRangeQueryParams,
  buildTrendSeriesParams as buildTrendSeriesQueryParams,
  type FeatureRangePreset,
  type TrendPointLimit,
  type TrendPreset,
} from '@/features/monitoring/pageConfig';
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
  WindowAlgorithmFitSummary,
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
import { WindowCandidatesPanel } from '@/features/tuning-task/WindowCandidatesPanel';
import { AssetDirectoryPanel } from '@/features/settings/AssetDirectoryPanel';
import {
  ASSET_TYPE_LABEL,
  DEFAULT_ASSET_NODES,
  assetTagColor,
  buildAssetTreeData,
  getDescendantAssetIds,
  inferLoopAssetId,
  nextAssetType,
  type AssetNode,
  type AssetNodeType,
} from '@/features/settings/assetModel';
import { DataSourcesPanel } from '@/features/settings/DataSourcesPanel';
import { ModelConfigPanel } from '@/features/settings/ModelConfigPanel';
import { PromptConfigPanel } from '@/features/settings/PromptConfigPanel';
import { PROMPT_CONFIG_ITEMS, type PromptConfigField } from '@/features/settings/promptConfigItems';
import { RuleConfigPanel } from '@/features/settings/RuleConfigPanel';
import { TuningPriorPanel } from '@/features/tuning-prior/TuningPriorPanel';
import './LoopMonitoringPage.css';

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

  const assetTreeData = useMemo(() => buildAssetTreeData(assetNodes, ASSET_TYPE_LABEL), [assetNodes]);

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
    return buildTrendSeriesQueryParams(trendPreset, trendCustomRange, trendPointLimit, loop);
  }, [trendCustomRange, trendPointLimit, trendPreset]);

  const buildFeatureRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    return buildFeatureRangeQueryParams(featureRangePreset, featureCustomRange, loop);
  }, [featureCustomRange, featureRangePreset]);

  const buildWindowRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    return buildFeatureRangeQueryParams(windowRangePreset, windowCustomRange, loop);
  }, [windowCustomRange, windowRangePreset]);

  const buildTuningRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    return buildFeatureRangeQueryParams(tuningRangePreset, tuningCustomRange, loop);
  }, [tuningCustomRange, tuningRangePreset]);

  const buildTuningPriorRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    return buildFeatureRangeQueryParams(tuningPriorRangePreset, tuningPriorCustomRange, loop);
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

  const windowPreviewData = useMemo(() => {
    if (!selectedWindow?.preview?.length) return [];
    const rows: Array<{ t: string | number; value: number; series: string }> = [];
    selectedWindow.preview.forEach((point) => {
      rows.push({ t: point.t, value: point.pv, series: 'PV' });
      rows.push({ t: point.t, value: point.mv, series: 'MV' });
    });
    return rows;
  }, [selectedWindow]);

  const renderLoopTable = () => (
    <LoopSelectionTable
      loops={scopedLoops}
      selectedLoopId={selectedLoopId}
      loading={loading}
      loopTypeLabels={LOOP_TYPE_LABEL}
      onSelectLoop={setSelectedLoopId}
    />
  );

  const renderTrend = (height = 360) => (
    <LoopTrendChart
      data={trendData}
      height={height}
      splitYAxis={trendSplitYAxis}
      xAxisMode={series?.x_axis}
    />
  );

  const renderAssessmentCards = () => (
    <AssessmentCards
      assessment={assessment}
      scorePercent={scorePercent}
      scoreStatus={scoreStatus}
      tagColor={tagColor}
    />
  );

  const renderWindowTable = () => (
    <WindowSelectionTable
      windows={windows}
      selectedWindowIndex={selectedWindowIndex}
      scorePercent={scorePercent}
      onSelectWindow={setSelectedWindowIndex}
    />
  );

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
      return (
        <SectionErrorBoundary label="窗口候选页面">
          <WindowCandidatesPanel
            selectedLoopId={selectedLoopId}
            selectedLoop={selectedLoop}
            scopedLoops={scopedLoops}
            loopTypeLabels={LOOP_TYPE_LABEL}
            featureRangeOptions={FEATURE_RANGE_OPTIONS}
            windowRangePreset={windowRangePreset}
            windowCustomRange={windowCustomRange}
            running={running}
            taskStatus={taskStatus}
            taskError={taskError}
            taskCurrentStage={taskCurrentStage}
            taskStageStatus={taskStageStatus}
            taskStageData={taskStageData}
            taskStageRunningData={taskStageRunningData}
            taskWindowSelection={taskWindowSelection}
            taskThinking={taskThinking}
            onLoopChange={setSelectedLoopId}
            onRangePresetChange={(value) => setWindowRangePreset(value as FeatureRangePreset)}
            onCustomRangeChange={setWindowCustomRange}
            onPreviewWindows={() => {
              if (!selectedLoopId) return;
              loadWindows(selectedLoopId, buildWindowRangeParams(selectedLoop));
            }}
            onStartReview={() => startTune({ useSelectedWindow: false, stopAfter: 'window_selection' })}
            onStop={handleStopTune}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
            formatRange={formatRange}
            formatProcessDirection={formatProcessDirection}
            formatProcessDirectionBasis={formatProcessDirectionBasis}
          />
        </SectionErrorBoundary>
      );
    }
    switch (activeSub) {
      case 'dashboard':
        return (
          <DashboardCockpitPanel
            scopedLoops={scopedLoops}
            scopedLoopCount={scopedLoopStats.loopCount}
            dashboardRows={dashboardRows}
            dashboardStats={dashboardStats}
            selectedLoopId={selectedLoopId}
            selectedLoop={selectedLoop}
            monitoring={monitoring}
            assetTypeLabel={selectedAssetNode ? ASSET_TYPE_LABEL[selectedAssetNode.type] : '-'}
            assetTagColor={assetTagColor(selectedAssetNode?.type ?? 'factory')}
            pathLabel={selectedAssetPath.map((item) => item.name).join(' / ')}
            loopTypeLabels={LOOP_TYPE_LABEL}
            widgetKeys={dashboardWidgetKeys}
            draggedWidgetKey={draggedDashboardWidgetKey}
            configOpen={dashboardConfigOpen}
            trend={renderTrend(300)}
            assetNameForLoop={(loop) => assetNodes.find((node) => node.id === inferLoopAssetId(loop.loop_id))?.name ?? '未归属'}
            scorePercent={scorePercent}
            formatPercentValue={formatPercentValue}
            statusColor={monitoringStatusColor}
            statusText={monitoringStatusText}
            onOpenConfig={() => setDashboardConfigOpen(true)}
            onCloseConfig={() => setDashboardConfigOpen(false)}
            onWidgetKeysChange={setDashboardWidgetKeys}
            onDragStart={setDraggedDashboardWidgetKey}
            onDrop={moveDashboardWidget}
            onDragEnd={() => setDraggedDashboardWidgetKey(null)}
            onHide={hideDashboardWidget}
            onSwitchAsset={() => switchTo('settings', 'asset_directory')}
            onSelectLoop={setSelectedLoopId}
            onViewLoop={(loopId) => {
              setSelectedLoopId(loopId);
              switchTo('monitor', 'loop_profile');
            }}
            onCreateTuningTask={() => switchTo('tuning', 'tuning_task')}
            onViewLoopProfile={() => switchTo('monitor', 'loop_profile')}
            onViewTrendSpectrum={() => switchTo('monitor', 'trend_spectrum')}
            onOpenDiagnosis={() => switchTo('diagnostics', 'diagnosis_overview')}
          />
        );
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
          <AssetDirectoryPanel
            pathLabel={selectedAssetPath.map((item) => item.name).join(' / ')}
            selectedAssetTypeLabel={selectedAssetNode ? ASSET_TYPE_LABEL[selectedAssetNode.type] : '-'}
            selectedAssetTagColor={assetTagColor(selectedAssetNode?.type ?? 'factory')}
            scopedLoopCount={scopedLoopStats.loopCount}
            assetTreeData={assetTreeData}
            selectedAssetNodeId={selectedAssetNodeId}
            selectedAssetPathIds={selectedAssetPath.map((item) => item.id)}
            selectedAssetName={selectedAssetNode?.name}
            selectedAssetCode={selectedAssetNode?.code}
            assetDraftName={assetDraftName}
            assetDraftType={assetDraftType}
            assetTypeOptions={(Object.keys(ASSET_TYPE_LABEL) as AssetNodeType[]).map((type) => ({
              label: ASSET_TYPE_LABEL[type],
              value: type,
            }))}
            assetRenameValue={assetRenameValue}
            scopedLoops={scopedLoops}
            onAssetSelect={(nodeId) => {
              setSelectedAssetNodeId(nodeId);
              setAssetRenameValue(assetNodes.find((item) => item.id === nodeId)?.name ?? '');
            }}
            onAssetDraftNameChange={setAssetDraftName}
            onAssetDraftTypeChange={(value) => setAssetDraftType(value as AssetNodeType)}
            onAssetRenameValueChange={setAssetRenameValue}
            onAddAssetChild={addAssetChild}
            onRenameAssetNode={renameAssetNode}
            onDeleteAssetNode={deleteAssetNode}
            loopTypeLabel={(loopType) => LOOP_TYPE_LABEL[loopType] ?? loopType}
            assetNameForLoop={(loop) => assetNodes.find((node) => node.id === inferLoopAssetId(loop.loop_id))?.name ?? '-'}
            onViewLoop={(loopId) => {
              setSelectedLoopId(loopId);
              switchTo('monitor', 'loop_profile');
            }}
            onTuneLoop={(loopId) => {
              setSelectedLoopId(loopId);
              switchTo('tuning', 'tuning_task');
            }}
          />
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
          <TrendSpectrumPanel
            selectedLoopId={selectedLoopId}
            selectedLoop={selectedLoop}
            scopedLoops={scopedLoops}
            series={series}
            seriesLoading={seriesLoading}
            trendPreset={trendPreset}
            trendPointLimit={trendPointLimit}
            trendSplitYAxis={trendSplitYAxis}
            trendCustomRange={trendCustomRange}
            trendPresetOptions={TREND_PRESET_OPTIONS}
            trendPointLimitOptions={TREND_POINT_LIMIT_OPTIONS}
            loopTypeLabel={LOOP_TYPE_LABEL}
            assessment={assessment}
            monitoring={monitoring}
            oscillationDetected={oscillationDetected}
            chart={renderTrend(420)}
            onLoopChange={setSelectedLoopId}
            onTrendPresetChange={(value) => setTrendPreset(value as TrendPreset)}
            onTrendPointLimitChange={(value) => setTrendPointLimit(value as TrendPointLimit)}
            onTrendSplitYAxisChange={setTrendSplitYAxis}
            onTrendCustomRangeChange={setTrendCustomRange}
            onRefresh={() => selectedLoopId && loadSeries(selectedLoopId, selectedLoop)}
            formatNumber={formatNumber}
            formatPercentValue={formatPercentValue}
            formatOscillationEvidence={formatOscillationEvidence}
            formatOscillationPhaseHint={formatOscillationPhaseHint}
          />
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

  const renderAppTopbar = () => (
    <PidAppTopbar
      sidebarCollapsed={sidebarCollapsed}
      viewMode={viewMode}
      onSidebarToggle={() => setSidebarCollapsed((value) => !value)}
      onViewModeChange={setViewMode}
    />
  );

  const renderDialogueMode = () => {
    return (
      <div className="dialogue-shell">
        {renderAppTopbar()}

        <main className={sidebarCollapsed ? 'dialogue-main history-collapsed' : 'dialogue-main'}>
          <DialogueHistoryPanel
            sessions={sortedAssistantSessions}
            activeSessionId={activeAssistantSession?.id}
            pinnedSessionIds={pinnedAssistantSessionIdSet}
            loading={assistantSessionsLoading}
            onCreateSession={createDialogueSession}
            onOpenSession={openAssistantSession}
            onTogglePin={toggleAssistantSessionPin}
            onRename={renameAssistantSession}
            onDelete={deleteAssistantSessionWithConfirm}
          />

          <DialogueChatPanel
            loops={loops}
            loopTypeLabels={LOOP_TYPE_LABEL}
            selectedLoopId={selectedLoopId}
            selectedLoopLabel={selectedLoop?.loop_id}
            activeSessionTitle={activeAssistantSession?.title}
            messages={assistantMessages}
            inputValue={assistantInput}
            streaming={assistantStreaming}
            starterPrompts={DIALOGUE_STARTER_PROMPTS}
            onLoopChange={setSelectedLoopId}
            onInputChange={setAssistantInput}
            onAsk={askAssistant}
            normalizeAction={normalizeAssistantAction}
            onRunAction={(action) => runAssistantAction(action as AssistantAction)}
          />

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
          <ClassicSideMenu
            modules={MODULES}
            collapsed={sidebarCollapsed}
            activeModule={activeModule}
            activeSub={activeSub}
            expandedModules={expandedModules}
            onToggleModule={(moduleKey) => toggleModule(moduleKey as ModuleKey)}
            onSelect={(moduleKey, subKey) => switchTo(moduleKey as ModuleKey, subKey as SubKey)}
            onExpandFromCollapsed={(moduleKey, firstSubKey) => {
              setSidebarCollapsed(false);
              switchTo(moduleKey as ModuleKey, firstSubKey as SubKey);
            }}
          />

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


