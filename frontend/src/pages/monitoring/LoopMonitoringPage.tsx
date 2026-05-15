import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Dayjs } from 'dayjs';
import {
  Empty,
  Input,
  Modal,
  Space,
  Typography,
  message,
} from 'antd';
import {
  assistantSessionStream,
  createAssistantSession,
  deleteAssistantSession,
  getAssistantSession,
  listAssistantSessions,
  tuneHistoryLoopStream,
  getSession,
  listSessions,
  updateAssistantSession,
} from '@/services/api';
import McpConfigPage from '@/pages/settings/McpConfigPage';
import { ClassicModePage } from '@/features/app-shell/ClassicModePage';
import { LOOP_TYPE_LABEL, MODULES, type ModuleKey, type SubKey } from '@/features/app-shell/navigation';
import { SectionErrorBoundary } from '@/features/app-shell/SectionErrorBoundary';
import { useAppShellState } from '@/features/app-shell/useAppShellState';
import { chartLineTooltip, LoopTrendChart } from '@/features/charts/LoopTrendChart';
import { DashboardCockpitPanel } from '@/features/dashboard/DashboardCockpitPanel';
import {
  buildDashboardRows,
  summarizeDashboardRows,
} from '@/features/dashboard/model';
import { useDashboardWidgetLayout } from '@/features/dashboard/useDashboardWidgetLayout';
import { DialogueModePage } from '@/features/dialogue/DialogueModePage';
import {
  buildDialogueActions,
  buildAssistantContext as buildDialogueContext,
  formatAssistantEvent,
  mapAssistantSessionMessages,
  normalizeAssistantAction,
  type AssistantAction,
  type AssistantMessage,
} from '@/features/dialogue/model';
import { DIALOGUE_STARTER_PROMPTS } from '@/features/dialogue/prompts';
import { usePinnedAssistantSessions } from '@/features/dialogue/usePinnedAssistantSessions';
import { ActuatorStatusPanel } from '@/features/loop-monitoring/ActuatorStatusPanel';
import { AlarmEventsPanel } from '@/features/loop-monitoring/AlarmEventsPanel';
import { buildRailAlarms } from '@/features/loop-monitoring/alarmModel';
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
  type FeatureRangePreset,
  type TrendPointLimit,
  type TrendPreset,
} from '@/features/monitoring/pageConfig';
import { useHistoryImport } from '@/features/monitoring/useHistoryImport';
import { useHistoryLoops } from '@/features/monitoring/useHistoryLoops';
import { useLoopAssessment } from '@/features/monitoring/useLoopAssessment';
import { useLoopMonitoringData } from '@/features/monitoring/useLoopMonitoringData';
import { useLoopWindows } from '@/features/monitoring/useLoopWindows';
import { useTrendSeries } from '@/features/monitoring/useTrendSeries';
import { ModelReliabilityPanel } from '@/features/model-reliability/ModelReliabilityPanel';
import type {
  HistoryLoop,
  AssistantSession,
  AssistantSessionSummary,
  HistoryTimeRangeParams,
} from '@/services/api';
import type {
  IdentificationAttempt,
  IdentificationRefinementMeta,
  LlmThinkingEvent,
  ModelReviewMeta,
  PipelineEvent,
  TuningResult,
  WindowSelectionMeta,
} from '@/types/tuning';
import {
  attemptFitKey,
  buildTuningGate,
  buildFitPreviewChartData,
  clearRunningStageData,
  getDeterministicRefinement,
  getFitPreviewAttempts,
  getSelectedFitAttempt,
  getTaskAlgorithmComparison,
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
import { TuningTaskPanel } from '@/features/tuning-task/TuningTaskPanel';
import { WindowCandidatesPanel } from '@/features/tuning-task/WindowCandidatesPanel';
import { AssetDirectoryPanel } from '@/features/settings/AssetDirectoryPanel';
import { DataSourcesPanel } from '@/features/settings/DataSourcesPanel';
import { ModelConfigPanel } from '@/features/settings/ModelConfigPanel';
import { PromptConfigPanel } from '@/features/settings/PromptConfigPanel';
import { PROMPT_CONFIG_ITEMS, type PromptConfigField } from '@/features/settings/promptConfigItems';
import { RuleConfigPanel } from '@/features/settings/RuleConfigPanel';
import { useAssetDirectory } from '@/features/settings/useAssetDirectory';
import { useSettingsConfigs } from '@/features/settings/useSettingsConfigs';
import { TuningPriorPanel } from '@/features/tuning-prior/TuningPriorPanel';
import { useTuningPrior } from '@/features/tuning-prior/useTuningPrior';
import './LoopMonitoringPage.css';

function LoopMonitoringPageInner() {
  const {
    activeModule,
    activeSub,
    viewMode,
    setViewMode,
    sidebarCollapsed,
    setSidebarCollapsed,
    expandedModules,
    currentSub,
    switchTo,
    toggleModule,
    toggleSidebar,
  } = useAppShellState();
  const {
    loops,
    selectedLoopId,
    selectedLoop,
    loading,
    loadLoops,
    setSelectedLoopId,
  } = useHistoryLoops();
  const {
    dataSourceType,
    fileList,
    importing,
    handleImport,
    setDataSourceType,
    setFileList,
  } = useHistoryImport({ loadLoops, selectLoop: setSelectedLoopId });
  const {
    series,
    seriesLoading,
    trendPreset,
    trendPointLimit,
    trendSplitYAxis,
    trendCustomRange,
    loadSeries,
    setTrendPreset,
    setTrendPointLimit,
    setTrendSplitYAxis,
    setTrendCustomRange,
  } = useTrendSeries();
  const {
    windowRangePreset,
    windowCustomRange,
    windows,
    windowAlgorithmSummary,
    selectedWindowIndex,
    selectedWindow,
    buildWindowRangeParams,
    loadWindows,
    setWindowRangePreset,
    setWindowCustomRange,
    setSelectedWindowIndex,
  } = useLoopWindows();
  // 整定任务页独立的时间窗 state；与窗口候选页的 windowRangePreset 解耦，
  // 这样两个页面分别发起整定时不会互相覆盖时间选择。
  const [tuningRangePreset, setTuningRangePreset] = useState<FeatureRangePreset>('8h');
  const [tuningCustomRange, setTuningCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const {
    tuningPriorRangePreset,
    tuningPriorCustomRange,
    tuningPriorCoreData,
    tuningPriorOntologyData,
    tuningPriorReviewData,
    tuningPriorCoreLoading,
    tuningPriorOntologyLoading,
    tuningPriorReviewLoading,
    tuningPriorCoreError,
    tuningPriorOntologyError,
    tuningPriorReviewError,
    buildTuningPriorRangeParams,
    loadTuningPriorCore,
    loadTuningPriorOntology,
    loadTuningPriorReview,
    resetTuningPrior,
    setTuningPriorRangePreset,
    setTuningPriorCustomRange,
  } = useTuningPrior();
  // 整定任务页是否启用 LLM 顾问；为了和窗口候选保持一致，默认 true。
  const [tuningUseLlm, setTuningUseLlm] = useState<boolean>(true);
  const {
    assessment,
    assessmentLoading,
    assessmentError,
    loadAssessment,
  } = useLoopAssessment();
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
  const [taskDetailOpen, setTaskDetailOpen] = useState(false);
  const [rawLogExpanded, setRawLogExpanded] = useState(false);
  const [dashboardConfigOpen, setDashboardConfigOpen] = useState(false);
  const {
    selectedAssetNode,
    selectedAssetNodeId,
    selectedAssetPathIds,
    pathLabel,
    selectedAssetTypeLabel,
    selectedAssetTagColor,
    scopedLoops,
    scopedLoopStats,
    assetTreeData,
    assetDraftName,
    assetDraftType,
    assetTypeOptions,
    assetRenameValue,
    setAssetDraftName,
    setAssetDraftType,
    setAssetRenameValue,
    selectAssetNode,
    addAssetChild,
    renameAssetNode,
    deleteAssetNode,
    assetNameForLoop,
  } = useAssetDirectory(loops);

  const {
    dashboardWidgetKeys,
    setDashboardWidgetKeys,
    enabledDashboardWidgets,
    draggedDashboardWidgetKey,
    setDraggedDashboardWidgetKey,
    hideDashboardWidget,
    moveDashboardWidget,
  } = useDashboardWidgetLayout();
  const [assistantInput, setAssistantInput] = useState('');
  const [assistantMessages, setAssistantMessages] = useState<AssistantMessage[]>([]);
  const [assistantSessions, setAssistantSessions] = useState<AssistantSessionSummary[]>([]);
  const [activeAssistantSession, setActiveAssistantSession] = useState<AssistantSession | null>(null);
  const [assistantSessionsLoading, setAssistantSessionsLoading] = useState(false);
  const [assistantStreaming, setAssistantStreaming] = useState(false);
  const assistantAbortRef = useRef<AbortController | null>(null);
  const dashboardWorstSelectionRef = useRef<string | null>(null);

  const {
    pinnedSessionIdSet: pinnedAssistantSessionIdSet,
    sortedSessions: sortedAssistantSessions,
    togglePin: toggleAssistantSessionPin,
    unpin: unpinAssistantSession,
  } = usePinnedAssistantSessions(assistantSessions);

  const {
    modelConfig,
    modelConfigLoading,
    modelConfigSaving,
    modelConfigTesting,
    modelConfigForm,
    modelConfigTestResult,
    loadModelConfig,
    saveModelConfig,
    testModelConnection,
    fillModelConfigForm,
    policyConfig,
    policyConfigLoading,
    loadPolicyConfig,
    promptConfig,
    promptConfigLoading,
    promptConfigSaving,
    promptConfigForm,
    activePromptField,
    setActivePromptField,
    loadPromptConfig,
    savePromptConfig,
    restoreDefaultPromptConfig,
  } = useSettingsConfigs();

  const isSettingsView = activeModule === 'settings';
  const shouldLoadDashboardMonitoring = activeSub === 'dashboard' || activeSub === 'loop_board';
  const shouldRestoreLatestTask = activeSub === 'tuning_task' || activeSub === 'performance_score';
  const shouldLoadAssessmentDetail = ASSESSMENT_DETAIL_SUBS.has(activeSub);
  const shouldLoadWindowDetail = WINDOW_DETAIL_SUBS.has(activeSub);
  const shouldLoadFeatureDetail = FEATURE_DETAIL_SUBS.has(activeSub);
  const shouldLoadMonitoringDetail = MONITORING_DETAIL_SUBS.has(activeSub);
  const {
    featureRangePreset,
    featureCustomRange,
    featureLoading,
    loopFeatures,
    loopMonitoring,
    monitoringByLoopId,
    buildFeatureRangeParams,
    loadLoopFeatures,
    loadLoopMonitoring,
    setFeatureRangePreset,
    setFeatureCustomRange,
  } = useLoopMonitoringData({ scopedLoops, shouldLoadDashboardMonitoring });

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

  const taskAlgorithmComparison = useMemo(
    () => getTaskAlgorithmComparison(taskStageData, taskResult),
    [taskResult, taskStageData],
  );

  const fitPreviewAttempts = useMemo(() => {
    return getFitPreviewAttempts(taskAttempts, taskResult);
  }, [taskAttempts, taskResult]);

  const selectedFitAttempt = useMemo(() => {
    return getSelectedFitAttempt(fitPreviewAttempts, selectedFitAttemptKey);
  }, [fitPreviewAttempts, selectedFitAttemptKey]);

  const fitPreviewChartData = useMemo(() => {
    return buildFitPreviewChartData(selectedFitAttempt);
  }, [selectedFitAttempt]);

  const deterministicRefinement = useMemo(
    () => getDeterministicRefinement(taskRefinements),
    [taskRefinements],
  );

  const tuningGate = useMemo(() => {
    return buildTuningGate(assessment);
  }, [assessment]);

  const railAlarms = useMemo(() => {
    return buildRailAlarms({
      assessment,
      dataSourceType,
      loopMonitoring,
      taskId,
      taskStartedAt,
      taskStatus,
      monitoringStatusText,
      scorePercent,
    });
  }, [assessment, dataSourceType, loopMonitoring, taskId, taskStartedAt, taskStatus]);

  const runAssistantAction = (action: AssistantAction) => {
    if (action.loopId) setSelectedLoopId(action.loopId);
    switchTo(action.target, action.sub);
    setViewMode('classic');
  };

  const buildAssistantContext = useCallback(() => {
    return buildDialogueContext({
      activeModule,
      activeSub,
      assessment,
      currentSubLabel: currentSub.label,
      dashboardRows,
      dashboardStats,
      loopFeatures,
      loopMonitoring,
      monitoringByLoopId,
      scopedLoopCount: scopedLoopStats.loopCount,
      selectedLoop,
      selectedLoopId,
    });
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
        unpinAssistantSession(session.id);
        if (activeAssistantSession?.id === session.id) {
          setActiveAssistantSession(null);
          setAssistantMessages([]);
        }
        await loadAssistantSessions();
      },
    });
  }, [activeAssistantSession, loadAssistantSessions, unpinAssistantSession]);

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

  const buildTuningRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    return buildFeatureRangeQueryParams(tuningRangePreset, tuningCustomRange, loop);
  }, [tuningCustomRange, tuningRangePreset]);

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
    resetTuningPrior();
  }, [activeSub, resetTuningPrior, selectedLoopId, tuningPriorRangePreset, tuningPriorCustomRange]);

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
      fillModelConfigForm();
    }
  }, [fillModelConfigForm, modelConfig]);

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
            assetTypeLabel={selectedAssetTypeLabel}
            assetTagColor={selectedAssetTagColor}
            pathLabel={pathLabel}
            loopTypeLabels={LOOP_TYPE_LABEL}
            widgetKeys={dashboardWidgetKeys}
            draggedWidgetKey={draggedDashboardWidgetKey}
            configOpen={dashboardConfigOpen}
            trend={renderTrend(300)}
            assetNameForLoop={(loop) => assetNameForLoop(loop, '未归属')}
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
            pathLabel={pathLabel}
            selectedAssetTypeLabel={selectedAssetTypeLabel}
            selectedAssetTagColor={selectedAssetTagColor}
            scopedLoopCount={scopedLoopStats.loopCount}
            assetTreeData={assetTreeData}
            selectedAssetNodeId={selectedAssetNodeId}
            selectedAssetPathIds={selectedAssetPathIds}
            selectedAssetName={selectedAssetNode?.name}
            selectedAssetCode={selectedAssetNode?.code}
            assetDraftName={assetDraftName}
            assetDraftType={assetDraftType}
            assetTypeOptions={assetTypeOptions}
            assetRenameValue={assetRenameValue}
            scopedLoops={scopedLoops}
            onAssetSelect={selectAssetNode}
            onAssetDraftNameChange={setAssetDraftName}
            onAssetDraftTypeChange={setAssetDraftType}
            onAssetRenameValueChange={setAssetRenameValue}
            onAddAssetChild={addAssetChild}
            onRenameAssetNode={renameAssetNode}
            onDeleteAssetNode={deleteAssetNode}
            loopTypeLabel={(loopType) => LOOP_TYPE_LABEL[loopType] ?? loopType}
            assetNameForLoop={assetNameForLoop}
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
            pathLabel={pathLabel}
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
            onOpenTuningTask={() => switchTo('tuning', 'tuning_task')}
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

  const renderDialogueMode = () => {
    return (
      <DialogueModePage
        sidebarCollapsed={sidebarCollapsed}
        viewMode={viewMode}
        sessions={sortedAssistantSessions}
        activeSessionId={activeAssistantSession?.id}
        pinnedSessionIds={pinnedAssistantSessionIdSet}
        sessionsLoading={assistantSessionsLoading}
        loops={loops}
        loopTypeLabels={LOOP_TYPE_LABEL}
        selectedLoopId={selectedLoopId}
        selectedLoopLabel={selectedLoop?.loop_id}
        activeSessionTitle={activeAssistantSession?.title}
        messages={assistantMessages}
        inputValue={assistantInput}
        streaming={assistantStreaming}
        starterPrompts={DIALOGUE_STARTER_PROMPTS}
        onSidebarToggle={toggleSidebar}
        onViewModeChange={setViewMode}
        onCreateSession={createDialogueSession}
        onOpenSession={openAssistantSession}
        onTogglePin={toggleAssistantSessionPin}
        onRename={renameAssistantSession}
        onDelete={deleteAssistantSessionWithConfirm}
        onLoopChange={setSelectedLoopId}
        onInputChange={setAssistantInput}
        onAsk={askAssistant}
        normalizeAction={normalizeAssistantAction}
        onRunAction={runAssistantAction}
      />
    );
  };

  if (viewMode === 'dialogue') {
    return renderDialogueMode();
  }

  return (
    <ClassicModePage
      sidebarCollapsed={sidebarCollapsed}
      viewMode={viewMode}
      modules={MODULES}
      activeModule={activeModule}
      activeSub={activeSub}
      expandedModules={expandedModules}
      taskDetailOpen={taskDetailOpen}
      taskDashboard={renderTaskDashboard()}
      onSidebarToggle={toggleSidebar}
      onViewModeChange={setViewMode}
      onToggleModule={(moduleKey) => toggleModule(moduleKey)}
      onSelect={(moduleKey, subKey) => switchTo(moduleKey, subKey)}
      onExpandFromCollapsed={(moduleKey, firstSubKey) => {
        setSidebarCollapsed(false);
        switchTo(moduleKey, firstSubKey);
      }}
      onTaskDetailClose={() => setTaskDetailOpen(false)}
    >
      {renderPage()}
    </ClassicModePage>
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


