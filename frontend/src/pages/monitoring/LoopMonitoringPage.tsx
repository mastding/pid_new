import { useState } from 'react';
import { Empty } from 'antd';
import { ClassicModePage } from '@/features/app-shell/ClassicModePage';
import { LOOP_TYPE_LABEL, MODULES } from '@/features/app-shell/navigation';
import { SectionErrorBoundary } from '@/features/app-shell/SectionErrorBoundary';
import { useAppShellState } from '@/features/app-shell/useAppShellState';
import { LoopTrendChart } from '@/features/charts/LoopTrendChart';
import { useDashboardWidgetLayout } from '@/features/dashboard/useDashboardWidgetLayout';
import { DialogueModePage } from '@/features/dialogue/DialogueModePage';
import { normalizeAssistantAction } from '@/features/dialogue/model';
import { DIALOGUE_STARTER_PROMPTS } from '@/features/dialogue/prompts';
import { useMonitoringAssistant } from '@/features/dialogue/useMonitoringAssistant';
import { AssessmentModulePage } from '@/features/loop-monitoring/AssessmentModulePage';
import { DiagnosticsModulePage } from '@/features/loop-monitoring/DiagnosticsModulePage';
import { MonitoringModulePage } from '@/features/loop-monitoring/MonitoringModulePage';
import { AssessmentCards } from '@/features/monitoring/AssessmentCards';
import { LoopSelectionTable } from '@/features/monitoring/SelectionTables';
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
} from '@/features/monitoring/pageConfig';
import { useHistoryImport } from '@/features/monitoring/useHistoryImport';
import { useHistoryLoops } from '@/features/monitoring/useHistoryLoops';
import { useLoopAssessment } from '@/features/monitoring/useLoopAssessment';
import { useLoopChartRows } from '@/features/monitoring/useLoopChartRows';
import { useLoopMonitoringData } from '@/features/monitoring/useLoopMonitoringData';
import { useLoopSelectionSync } from '@/features/monitoring/useLoopSelectionSync';
import { useLoopWindows } from '@/features/monitoring/useLoopWindows';
import { useMonitoringDerivedState } from '@/features/monitoring/useMonitoringDerivedState';
import { useMonitoringPageEffects } from '@/features/monitoring/useMonitoringPageEffects';
import { useTrendSeries } from '@/features/monitoring/useTrendSeries';
import { TuningTaskDashboard } from '@/features/tuning-task/TuningTaskDashboard';
import { useTuningTaskCommand } from '@/features/tuning-task/useTuningTaskCommand';
import { TuningModulePage } from '@/features/tuning-task/TuningModulePage';
import { useTuningTaskOptions } from '@/features/tuning-task/useTuningTaskOptions';
import { useTuningTaskRuntime } from '@/features/tuning-task/useTuningTaskRuntime';
import { SettingsModulePage } from '@/features/settings/SettingsModulePage';
import { useAssetDirectory } from '@/features/settings/useAssetDirectory';
import { useSettingsConfigs } from '@/features/settings/useSettingsConfigs';
import { useTuningPrior } from '@/features/tuning-prior/useTuningPrior';
import type { PreparedAutoTuningTask } from '@/services/api';
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
  const [pendingAutoTuningTask, setPendingAutoTuningTask] = useState<{
    taskId: string;
    loopId?: string;
    ontologyContext?: Record<string, unknown>;
  } | null>(null);
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

  const {
    dashboardRows,
    dashboardStats,
    dashboardWorstLoopId,
    deterministicRefinement,
    fitPreviewAttempts,
    fitPreviewChartData,
    railAlarms,
    selectedFitAttempt,
    taskAlgorithmComparison,
    tuningGate,
  } = useMonitoringDerivedState({
    assessment,
    dataSourceType,
    loopMonitoring,
    monitoringByLoopId,
    scopedLoops,
    selectedFitAttemptKey,
    taskAttempts,
    taskId,
    taskRefinements,
    taskResult,
    taskStageData,
    taskStartedAt,
    taskStatus,
    monitoringStatusText,
    scorePercent,
  });

  const handleTune = useTuningTaskCommand({
    assessment,
    autoTuningTask: pendingAutoTuningTask?.loopId === selectedLoop?.loop_id ? pendingAutoTuningTask : null,
    buildTuningRangeParams,
    onAutoTuningTaskConsumed: () => setPendingAutoTuningTask(null),
    selectedLoop,
    startTune,
    tuningGate,
    tuningUseLlm,
  });

  const handleAutoTaskPrepared = (prepared: PreparedAutoTuningTask) => {
    setPendingAutoTuningTask({
      taskId: prepared.task.task_id,
      loopId: prepared.tuning_request.loop_id,
      ontologyContext: prepared.tuning_request.ontology_context,
    });
  };

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

    if (activeModule === 'tuning' || activeSub === 'model_reliability') {
      return (
        <TuningModulePage
          activeSub={activeSub}
          assessment={assessment}
          assessmentError={assessmentError}
          assessmentLoading={assessmentLoading}
          buildTuningPriorRangeParams={buildTuningPriorRangeParams}
          buildWindowRangeParams={buildWindowRangeParams}
          deterministicRefinement={deterministicRefinement}
          featureRangeOptions={FEATURE_RANGE_OPTIONS}
          fitPreviewAttempts={fitPreviewAttempts}
          fitPreviewChartData={fitPreviewChartData}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
          formatProcessDirection={formatProcessDirection}
          formatProcessDirectionBasis={formatProcessDirectionBasis}
          formatRange={formatRange}
          gateCheckLabel={gateCheckLabel}
          gateCheckMessage={gateCheckMessage}
          gateDecisionText={gateDecisionText}
          gateImpact={gateImpact}
          gateSeverityColor={gateSeverityColor}
          handleStopTune={handleStopTune}
          handleTune={handleTune}
          loadTuningPriorCore={loadTuningPriorCore}
          loadTuningPriorOntology={loadTuningPriorOntology}
          loadTuningPriorReview={loadTuningPriorReview}
          loadWindows={loadWindows}
          loopTypeLabel={LOOP_TYPE_LABEL}
          loops={loops}
          operatingConditionText={operatingConditionText}
          monitoringStatusText={monitoringStatusText}
          running={running}
          scorePercent={scorePercent}
          scoreStatus={scoreStatus}
          scopedLoops={scopedLoops}
          selectedFitAttempt={selectedFitAttempt}
          selectedFitAttemptKey={selectedFitAttemptKey}
          selectedLoop={selectedLoop}
          selectedLoopId={selectedLoopId}
          selectedWindow={selectedWindow}
          setSelectedFitAttemptKey={setSelectedFitAttemptKey}
          setSelectedLoopId={setSelectedLoopId}
          setTaskDetailOpen={setTaskDetailOpen}
          setTuningCustomRange={setTuningCustomRange}
          setTuningPriorCustomRange={setTuningPriorCustomRange}
          setTuningPriorRangePreset={setTuningPriorRangePreset}
          setTuningRangePreset={setTuningRangePreset}
          setTuningUseLlm={setTuningUseLlm}
          setWindowCustomRange={setWindowCustomRange}
          setWindowRangePreset={setWindowRangePreset}
          onAutoTaskPrepared={handleAutoTaskPrepared}
          startTune={startTune}
          switchToTuningTask={() => switchTo('tuning', 'tuning_task')}
          tagColor={tagColor}
          taskAlgorithmComparison={taskAlgorithmComparison}
          taskAttempts={taskAttempts}
          taskCurrentStage={taskCurrentStage}
          taskError={taskError}
          taskId={taskId}
          taskResult={taskResult}
          taskStageData={taskStageData}
          taskStageRunningData={taskStageRunningData}
          taskStageStatus={taskStageStatus}
          taskStatus={taskStatus}
          taskThinking={taskThinking}
          taskWindowSelection={taskWindowSelection}
          tuningCustomRange={tuningCustomRange}
          tuningGate={tuningGate}
          tuningPriorCoreData={tuningPriorCoreData}
          tuningPriorCoreError={tuningPriorCoreError}
          tuningPriorCoreLoading={tuningPriorCoreLoading}
          tuningPriorCustomRange={tuningPriorCustomRange}
          tuningPriorOntologyData={tuningPriorOntologyData}
          tuningPriorOntologyError={tuningPriorOntologyError}
          tuningPriorOntologyLoading={tuningPriorOntologyLoading}
          tuningPriorRangePreset={tuningPriorRangePreset}
          tuningPriorReviewData={tuningPriorReviewData}
          tuningPriorReviewError={tuningPriorReviewError}
          tuningPriorReviewLoading={tuningPriorReviewLoading}
          tuningRangePreset={tuningRangePreset}
          tuningUseLlm={tuningUseLlm}
          windowAlgorithmSummary={windowAlgorithmSummary}
          windowCustomRange={windowCustomRange}
          windowPreviewData={windowPreviewData}
          windowRangePreset={windowRangePreset}
          windows={windows}
        />
      );
    }

    if (activeModule === 'diagnostics') {
      return (
        <DiagnosticsModulePage
          activeSub={activeSub}
          assessment={assessment}
          monitoring={monitoring}
          selectedLoopId={selectedLoopId}
          scopedLoops={scopedLoops}
          featureRangePreset={featureRangePreset}
          featureCustomRange={featureCustomRange}
          featureRangeOptions={FEATURE_RANGE_OPTIONS}
          onLoopChange={setSelectedLoopId}
          onRangePresetChange={setFeatureRangePreset}
          onCustomRangeChange={setFeatureCustomRange}
          loopTypeLabel={(loop) => LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
        />
      );
    }

    if (activeModule === 'assessment') {
      return (
        <AssessmentModulePage
          activeSub={activeSub}
          assessment={assessment}
          assessmentCards={renderAssessmentCards()}
          buildFeatureRangeParams={buildFeatureRangeParams}
          featureCustomRange={featureCustomRange}
          featureRangeOptions={FEATURE_RANGE_OPTIONS}
          featureRangePreset={featureRangePreset}
          loopFeatures={loopFeatures}
          monitoring={monitoring}
          scopedLoops={scopedLoops}
          selectedLoop={selectedLoop}
          selectedLoopId={selectedLoopId}
          onCustomRangeChange={setFeatureCustomRange}
          onLoopChange={setSelectedLoopId}
          onRangePresetChange={setFeatureRangePreset}
          conditionEvidenceDetail={conditionEvidenceDetail}
          conditionEvidenceName={conditionEvidenceName}
          conditionRecommendationText={conditionRecommendationText}
          evidenceStatusColor={evidenceStatusColor}
          evidenceStatusText={evidenceStatusText}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
          loopTypeLabel={(loop) => LOOP_TYPE_LABEL[loop.loop_type] ?? loop.loop_type}
          monitoringStatusColor={monitoringStatusColor}
          monitoringStatusText={monitoringStatusText}
          operatingConditionText={operatingConditionText}
          scorePercent={scorePercent}
          tagColor={tagColor}
          tuningSuitabilityColor={tuningSuitabilityColor}
          tuningSuitabilityText={tuningSuitabilityText}
          yesNo={yesNo}
        />
      );
    }

    if (activeModule === 'workspace' || activeModule === 'monitor') {
      return (
        <MonitoringModulePage
          activeSub={activeSub}
          alertSeverityColor={alertSeverityColor}
          assessment={assessment}
          assetNameForLoop={assetNameForLoop}
          buildFeatureRangeParams={buildFeatureRangeParams}
          configOpen={dashboardConfigOpen}
          dashboardRows={dashboardRows}
          dashboardStats={dashboardStats}
          draggedWidgetKey={draggedDashboardWidgetKey}
          featureCustomRange={featureCustomRange}
          featureLoading={featureLoading}
          featureRangeOptions={FEATURE_RANGE_OPTIONS}
          featureRangePreset={featureRangePreset}
          formatCpkBasis={formatCpkBasis}
          formatHarrisBasis={formatHarrisBasis}
          formatNumber={formatNumber}
          formatOscillationEvidence={formatOscillationEvidence}
          formatOscillationPhaseHint={formatOscillationPhaseHint}
          formatPercentValue={formatPercentValue}
          formatProcessDirection={formatProcessDirection}
          formatProcessDirectionBasis={formatProcessDirectionBasis}
          formatRange={formatRange}
          hideDashboardWidget={hideDashboardWidget}
          loading={loading}
          loadLoopFeatures={loadLoopFeatures}
          loadLoopMonitoring={loadLoopMonitoring}
          loadLoops={loadLoops}
          loadSeries={loadSeries}
          loopFeatures={loopFeatures}
          loopTable={renderLoopTable()}
          loopTypeLabels={LOOP_TYPE_LABEL}
          monitoring={monitoring}
          monitoringStatusColor={monitoringStatusColor}
          monitoringStatusText={monitoringStatusText}
          moveDashboardWidget={moveDashboardWidget}
          onCreateTuningTask={() => switchTo('tuning', 'tuning_task')}
          onOpenConfig={() => setDashboardConfigOpen(true)}
          onOpenDiagnosis={() => switchTo('diagnostics', 'diagnosis_overview')}
          onSwitchAsset={() => switchTo('settings', 'asset_directory')}
          onViewLoop={(loopId) => {
            setSelectedLoopId(loopId);
            switchTo('monitor', 'loop_profile');
          }}
          onViewLoopProfile={() => switchTo('monitor', 'loop_profile')}
          onViewTrendSpectrum={() => switchTo('monitor', 'trend_spectrum')}
          oscillationDetected={oscillationDetected}
          pathLabel={pathLabel}
          railAlarms={railAlarms}
          scopedLoopCount={scopedLoopStats.loopCount}
          scopedLoops={scopedLoops}
          scorePercent={scorePercent}
          selectedAssetTagColor={selectedAssetTagColor}
          selectedAssetTypeLabel={selectedAssetTypeLabel}
          selectedLoop={selectedLoop}
          selectedLoopId={selectedLoopId}
          series={series}
          seriesLoading={seriesLoading}
          setConfigOpen={setDashboardConfigOpen}
          setDraggedWidgetKey={setDraggedDashboardWidgetKey}
          setFeatureCustomRange={setFeatureCustomRange}
          setFeatureRangePreset={setFeatureRangePreset}
          setSelectedLoopId={setSelectedLoopId}
          setTrendCustomRange={setTrendCustomRange}
          setTrendPointLimit={setTrendPointLimit}
          setTrendPreset={setTrendPreset}
          setTrendSplitYAxis={setTrendSplitYAxis}
          setWidgetKeys={setDashboardWidgetKeys}
          tagColor={tagColor}
          taskId={taskId}
          taskStatus={taskStatus}
          trend={renderTrend(300)}
          trendChart={renderTrend(420)}
          trendCustomRange={trendCustomRange}
          trendPointLimit={trendPointLimit}
          trendPointLimitOptions={TREND_POINT_LIMIT_OPTIONS}
          trendPreset={trendPreset}
          trendPresetOptions={TREND_PRESET_OPTIONS}
          trendSplitYAxis={trendSplitYAxis}
          widgetKeys={dashboardWidgetKeys}
        />
      );
    }

    return (
      <section className="agent-panel">
        <div className="panel-title">{currentSub.label}</div>
        <Empty description="???????" />
      </section>
    );
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
        modelConfig={modelConfig}
        modelConfigForm={modelConfigForm}
        modelConfigLoading={modelConfigLoading}
        modelConfigSaving={modelConfigSaving}
        modelConfigTesting={modelConfigTesting}
        modelConfigTestResult={modelConfigTestResult}
        promptConfig={promptConfig}
        promptConfigForm={promptConfigForm}
        promptConfigLoading={promptConfigLoading}
        promptConfigSaving={promptConfigSaving}
        activePromptField={activePromptField}
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
        onLoadModelConfig={loadModelConfig}
        onSaveModelConfig={saveModelConfig}
        onTestModelConnection={testModelConnection}
        onLoadPromptConfig={loadPromptConfig}
        onRestoreDefaultPromptConfig={restoreDefaultPromptConfig}
        onSavePromptConfig={savePromptConfig}
        onSetActivePromptField={setActivePromptField}
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


