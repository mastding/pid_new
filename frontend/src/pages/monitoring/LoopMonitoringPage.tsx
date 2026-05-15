import { useMemo, useState } from 'react';
import { Empty } from 'antd';
import { ClassicModePage } from '@/features/app-shell/ClassicModePage';
import { LOOP_TYPE_LABEL, MODULES } from '@/features/app-shell/navigation';
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
import { normalizeAssistantAction } from '@/features/dialogue/model';
import { DIALOGUE_STARTER_PROMPTS } from '@/features/dialogue/prompts';
import { useMonitoringAssistant } from '@/features/dialogue/useMonitoringAssistant';
import { ActuatorStatusPanel } from '@/features/loop-monitoring/ActuatorStatusPanel';
import { AlarmEventsPanel } from '@/features/loop-monitoring/AlarmEventsPanel';
import { buildRailAlarms } from '@/features/loop-monitoring/alarmModel';
import { ConstraintMonitorPanel } from '@/features/loop-monitoring/ConstraintMonitorPanel';
import { DiagnosticsModulePage } from '@/features/loop-monitoring/DiagnosticsModulePage';
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
  type FeatureRangePreset,
  type TrendPointLimit,
  type TrendPreset,
} from '@/features/monitoring/pageConfig';
import { useHistoryImport } from '@/features/monitoring/useHistoryImport';
import { useHistoryLoops } from '@/features/monitoring/useHistoryLoops';
import { useLoopAssessment } from '@/features/monitoring/useLoopAssessment';
import { useLoopChartRows } from '@/features/monitoring/useLoopChartRows';
import { useLoopMonitoringData } from '@/features/monitoring/useLoopMonitoringData';
import { useLoopSelectionSync } from '@/features/monitoring/useLoopSelectionSync';
import { useLoopWindows } from '@/features/monitoring/useLoopWindows';
import { useMonitoringPageEffects } from '@/features/monitoring/useMonitoringPageEffects';
import { useTrendSeries } from '@/features/monitoring/useTrendSeries';
import { ModelReliabilityPanel } from '@/features/model-reliability/ModelReliabilityPanel';
import {
  attemptFitKey,
  buildTuningGate,
  buildFitPreviewChartData,
  getDeterministicRefinement,
  getFitPreviewAttempts,
  getSelectedFitAttempt,
  getTaskAlgorithmComparison,
} from '@/features/tuning-task/model';
import { TuningTaskDashboard } from '@/features/tuning-task/TuningTaskDashboard';
import { TuningTaskPanel } from '@/features/tuning-task/TuningTaskPanel';
import { useTuningTaskCommand } from '@/features/tuning-task/useTuningTaskCommand';
import { useTuningTaskOptions } from '@/features/tuning-task/useTuningTaskOptions';
import { useTuningTaskRuntime } from '@/features/tuning-task/useTuningTaskRuntime';
import { WindowCandidatesPanel } from '@/features/tuning-task/WindowCandidatesPanel';
import { SettingsModulePage } from '@/features/settings/SettingsModulePage';
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
  const {
    tuningRangePreset,
    tuningCustomRange,
    tuningUseLlm,
    buildTuningRangeParams,
    setTuningRangePreset,
    setTuningCustomRange,
    setTuningUseLlm,
  } = useTuningTaskOptions();
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
  const {
    assessment,
    assessmentLoading,
    assessmentError,
    loadAssessment,
  } = useLoopAssessment();
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
  } = useSettingsConfigs(activeSub);

  const isSettingsView = activeModule === 'settings';
  const shouldLoadDashboardMonitoring = activeSub === 'dashboard' || activeSub === 'loop_board';
  const shouldRestoreLatestTask = activeSub === 'tuning_task' || activeSub === 'performance_score';
  const shouldLoadAssessmentDetail = ASSESSMENT_DETAIL_SUBS.has(activeSub);
  const shouldLoadWindowDetail = WINDOW_DETAIL_SUBS.has(activeSub);
  const shouldLoadFeatureDetail = FEATURE_DETAIL_SUBS.has(activeSub);
  const shouldLoadMonitoringDetail = MONITORING_DETAIL_SUBS.has(activeSub);
  const {
    events,
    running,
    selectedFitAttemptKey,
    taskAttempts,
    taskCurrentStage,
    taskError,
    taskId,
    taskModelReview,
    taskRefinements,
    taskResult,
    taskStageData,
    taskStageRunningData,
    taskStageStatus,
    taskStartedAt,
    taskStatus,
    taskThinking,
    taskWindowSelection,
    handleStopTune,
    setSelectedFitAttemptKey,
    startTune,
  } = useTuningTaskRuntime({
    activeSub,
    buildWindowRangeParams,
    isSettingsView,
    onRunStart: () => setRawLogExpanded(false),
    selectedLoop,
    selectedWindowIndex,
    shouldRestoreLatestTask,
  });
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

  const handleTune = useTuningTaskCommand({
    assessment,
    buildTuningRangeParams,
    selectedLoop,
    startTune,
    tuningGate,
    tuningUseLlm,
  });

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

  const {
    activeAssistantSession,
    assistantInput,
    assistantMessages,
    assistantSessionsLoading,
    assistantStreaming,
    pinnedAssistantSessionIdSet,
    sortedAssistantSessions,
    askAssistant,
    createDialogueSession,
    deleteAssistantSessionWithConfirm,
    openAssistantSession,
    renameAssistantSession,
    runAssistantAction,
    setAssistantInput,
    toggleAssistantSessionPin,
  } = useMonitoringAssistant({
    activeModule,
    activeSub,
    assessment,
    currentSubLabel: currentSub.label,
    dashboardRows,
    dashboardStats,
    enabled: viewMode === 'dialogue',
    loopFeatures,
    loopMonitoring,
    monitoringByLoopId,
    scopedLoopCount: scopedLoopStats.loopCount,
    selectedLoop,
    selectedLoopId,
    setViewMode,
    switchTo,
    onSelectLoop: setSelectedLoopId,
  });

  useMonitoringPageEffects({
    activeSub,
    buildFeatureRangeParams,
    buildTuningRangeParams,
    enabledDashboardWidgets,
    isSettingsView,
    loadAssessment,
    loadLoopFeatures,
    loadLoopMonitoring,
    loadSeries,
    loadWindows,
    resetTuningPrior,
    selectedLoop,
    selectedLoopId,
    shouldLoadAssessmentDetail,
    shouldLoadFeatureDetail,
    shouldLoadMonitoringDetail,
    shouldLoadWindowDetail,
    tuningPriorCustomRange,
    tuningPriorRangePreset,
  });

  useLoopSelectionSync({
    activeSub,
    dashboardWorstLoopId,
    scopedLoops,
    selectedAssetNodeId,
    selectedLoopId,
    setSelectedLoopId,
  });

  const { trendData, windowPreviewData } = useLoopChartRows({ selectedWindow, series });

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

    if (activeModule === 'settings') {
      return (
        <SettingsModulePage
          activeSub={activeSub}
          activePromptField={activePromptField}
          assetDraftName={assetDraftName}
          assetDraftType={assetDraftType}
          assetNameForLoop={assetNameForLoop}
          assetRenameValue={assetRenameValue}
          assetTreeData={assetTreeData}
          assetTypeOptions={assetTypeOptions}
          currentSubLabel={currentSub.label}
          dataSourceType={dataSourceType}
          fileList={fileList}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
          importedLoopCount={loops.length}
          importing={importing}
          modelConfig={modelConfig}
          modelConfigForm={modelConfigForm}
          modelConfigLoading={modelConfigLoading}
          modelConfigSaving={modelConfigSaving}
          modelConfigTestResult={modelConfigTestResult}
          modelConfigTesting={modelConfigTesting}
          pathLabel={pathLabel}
          policyConfig={policyConfig}
          policyConfigLoading={policyConfigLoading}
          policyLoopImpact={policyLoopImpact}
          promptConfig={promptConfig}
          promptConfigForm={promptConfigForm}
          promptConfigLoading={promptConfigLoading}
          promptConfigSaving={promptConfigSaving}
          scopedLoopCount={scopedLoopStats.loopCount}
          scopedLoops={scopedLoops}
          selectedAssetCode={selectedAssetNode?.code}
          selectedAssetName={selectedAssetNode?.name}
          selectedAssetNodeId={selectedAssetNodeId}
          selectedAssetPathIds={selectedAssetPathIds}
          selectedAssetTagColor={selectedAssetTagColor}
          selectedAssetTypeLabel={selectedAssetTypeLabel}
          loopTypeLabel={(loopType) => LOOP_TYPE_LABEL[loopType] ?? loopType}
          onAddAssetChild={addAssetChild}
          onAssetDraftNameChange={setAssetDraftName}
          onAssetDraftTypeChange={setAssetDraftType}
          onAssetRenameValueChange={setAssetRenameValue}
          onAssetSelect={selectAssetNode}
          onDataSourceTypeChange={setDataSourceType}
          onDeleteAssetNode={deleteAssetNode}
          onFileListChange={setFileList}
          onImport={handleImport}
          onLoadModelConfig={loadModelConfig}
          onLoadPolicyConfig={loadPolicyConfig}
          onLoadPromptConfig={loadPromptConfig}
          onRenameAssetNode={renameAssetNode}
          onRestoreDefaultPromptConfig={restoreDefaultPromptConfig}
          onSaveModelConfig={saveModelConfig}
          onSavePromptConfig={savePromptConfig}
          onSetActivePromptField={setActivePromptField}
          onTestModelConnection={testModelConnection}
          onTuneLoop={(loopId) => {
            setSelectedLoopId(loopId);
            switchTo('tuning', 'tuning_task');
          }}
          onViewLoop={(loopId) => {
            setSelectedLoopId(loopId);
            switchTo('monitor', 'loop_profile');
          }}
        />
      );
    }

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

    if (activeModule === 'diagnostics') {
      return (
        <DiagnosticsModulePage
          activeSub={activeSub}
          assessment={assessment}
          monitoring={monitoring}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
        />
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


