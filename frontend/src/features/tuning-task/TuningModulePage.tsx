import type { Dispatch, SetStateAction } from 'react';
import type { Dayjs } from 'dayjs';
import type {
  HistoryLoop,
  HistoryLoopAssessment,
  HistoryLoopTuningPrior,
  HistoryTimeRangeParams,
  HistoryWindow,
} from '@/services/api';
import type {
  IdentificationAttempt,
  IdentificationRefinementMeta,
  LlmThinkingEvent,
  TuningResult,
  WindowAlgorithmFitSummary,
  WindowSelectionMeta,
} from '@/types/tuning';
import type { SubKey } from '@/features/app-shell/navigation';
import { SectionErrorBoundary } from '@/features/app-shell/SectionErrorBoundary';
import { chartLineTooltip } from '@/features/charts/LoopTrendChart';
import type { LoopTrendPoint } from '@/features/charts/LoopTrendChart';
import { ModelReliabilityPanel } from '@/features/model-reliability/ModelReliabilityPanel';
import type { FeatureRangePreset } from '@/features/monitoring/pageConfig';
import { TuningPriorPanel } from '@/features/tuning-prior/TuningPriorPanel';
import {
  attemptFitKey,
  buildTuningGate,
  type TaskStageDataMap,
  type TaskStageStatusMap,
  type TaskStatus,
} from '@/features/tuning-task/model';
import type { StartTuneOptions } from '@/features/tuning-task/useTuningTaskRuntime';
import { TuningTaskPanel } from '@/features/tuning-task/TuningTaskPanel';
import { WindowCandidatesPanel } from '@/features/tuning-task/WindowCandidatesPanel';

interface GateCheck {
  name?: string;
  passed?: boolean;
  severity?: string;
  message?: string;
  evidence?: Record<string, unknown>;
}

interface BlockingReason {
  type: string;
  severity: string;
  message: string;
}

interface TuningModulePageProps {
  activeSub: SubKey;
  assessment: HistoryLoopAssessment | null;
  assessmentError?: string | null;
  assessmentLoading: boolean;
  buildTuningPriorRangeParams: (loop?: HistoryLoop) => HistoryTimeRangeParams;
  buildWindowRangeParams: (loop?: HistoryLoop) => HistoryTimeRangeParams;
  deterministicRefinement?: IdentificationRefinementMeta;
  featureRangeOptions: Array<{ label: string; value: string; seconds?: number }>;
  fitPreviewAttempts: IdentificationAttempt[];
  fitPreviewChartData: LoopTrendPoint[];
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  formatProcessDirection: (value?: string | null) => string;
  formatProcessDirectionBasis: (value?: string | null) => string;
  formatRange: (min?: number | null, max?: number | null, digits?: number) => string;
  gateCheckLabel: (value?: string) => string;
  gateCheckMessage: (check: GateCheck, blockingReasons: BlockingReason[]) => string;
  gateDecisionText: (decision?: string) => string;
  gateImpact: (check: GateCheck, blockingReasons: BlockingReason[]) => { text: string; color: string };
  gateSeverityColor: (severity?: string) => string;
  handleStopTune: () => void;
  handleTune: () => void;
  loadTuningPriorCore: (loopId: string, params?: HistoryTimeRangeParams) => void;
  loadTuningPriorOntology: (loopId: string, params?: HistoryTimeRangeParams) => void;
  loadTuningPriorReview: (loopId: string) => void;
  loadWindows: (loopId: string, params?: HistoryTimeRangeParams) => void;
  loopTypeLabel: Record<string, string>;
  loops: HistoryLoop[];
  operatingConditionText: (value?: string) => string;
  monitoringStatusText: (status?: string) => string;
  running: boolean;
  scorePercent: (value?: number) => number;
  scoreStatus: (value?: number) => 'exception' | 'success' | 'normal';
  scopedLoops: HistoryLoop[];
  selectedFitAttempt?: IdentificationAttempt;
  selectedFitAttemptKey?: string;
  selectedLoop?: HistoryLoop;
  selectedLoopId?: string;
  selectedWindow?: HistoryWindow;
  setSelectedFitAttemptKey: (key: string) => void;
  setSelectedLoopId: Dispatch<SetStateAction<string | undefined>>;
  setTaskDetailOpen: (open: boolean) => void;
  setTuningCustomRange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  setTuningPriorCustomRange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  setTuningPriorRangePreset: Dispatch<SetStateAction<FeatureRangePreset>>;
  setTuningRangePreset: Dispatch<SetStateAction<FeatureRangePreset>>;
  setTuningUseLlm: Dispatch<SetStateAction<boolean>>;
  setWindowCustomRange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  setWindowRangePreset: Dispatch<SetStateAction<FeatureRangePreset>>;
  startTune: (options?: StartTuneOptions) => void;
  switchToTuningTask: () => void;
  tagColor: (level?: string) => string;
  taskAlgorithmComparison: WindowAlgorithmFitSummary[];
  taskAttempts: IdentificationAttempt[];
  taskCurrentStage?: string;
  taskError?: string;
  taskId?: string;
  taskResult: TuningResult | null;
  taskStageData: TaskStageDataMap;
  taskStageRunningData: TaskStageDataMap;
  taskStageStatus: TaskStageStatusMap;
  taskStatus: TaskStatus;
  taskThinking: LlmThinkingEvent[];
  taskWindowSelection: WindowSelectionMeta | null;
  tuningCustomRange: [Dayjs | null, Dayjs | null] | null;
  tuningGate: ReturnType<typeof buildTuningGate>;
  tuningPriorCoreData?: HistoryLoopTuningPrior | null;
  tuningPriorCoreError?: string | null;
  tuningPriorCoreLoading: boolean;
  tuningPriorCustomRange: [Dayjs | null, Dayjs | null] | null;
  tuningPriorOntologyData?: HistoryLoopTuningPrior | null;
  tuningPriorOntologyError?: string | null;
  tuningPriorOntologyLoading: boolean;
  tuningPriorRangePreset: string;
  tuningPriorReviewData?: HistoryLoopTuningPrior | null;
  tuningPriorReviewError?: string | null;
  tuningPriorReviewLoading: boolean;
  tuningRangePreset: string;
  tuningUseLlm: boolean;
  windowAlgorithmSummary: Record<string, { total: number; usable: number }>;
  windowCustomRange: [Dayjs | null, Dayjs | null] | null;
  windowPreviewData: LoopTrendPoint[];
  windowRangePreset: string;
  windows: HistoryWindow[];
}

export function TuningModulePage({
  activeSub,
  assessment,
  assessmentError,
  assessmentLoading,
  buildTuningPriorRangeParams,
  buildWindowRangeParams,
  deterministicRefinement,
  featureRangeOptions,
  fitPreviewAttempts,
  fitPreviewChartData,
  formatNumber,
  formatPercentValue,
  formatProcessDirection,
  formatProcessDirectionBasis,
  formatRange,
  gateCheckLabel,
  gateCheckMessage,
  gateDecisionText,
  gateImpact,
  gateSeverityColor,
  handleStopTune,
  handleTune,
  loadTuningPriorCore,
  loadTuningPriorOntology,
  loadTuningPriorReview,
  loadWindows,
  loopTypeLabel,
  loops,
  operatingConditionText,
  monitoringStatusText,
  running,
  scorePercent,
  scoreStatus,
  scopedLoops,
  selectedFitAttempt,
  selectedLoop,
  selectedLoopId,
  selectedWindow,
  setSelectedFitAttemptKey,
  setSelectedLoopId,
  setTaskDetailOpen,
  setTuningCustomRange,
  setTuningPriorCustomRange,
  setTuningPriorRangePreset,
  setTuningRangePreset,
  setTuningUseLlm,
  setWindowCustomRange,
  setWindowRangePreset,
  startTune,
  switchToTuningTask,
  tagColor,
  taskAlgorithmComparison,
  taskAttempts,
  taskCurrentStage,
  taskError,
  taskId,
  taskResult,
  taskStageData,
  taskStageRunningData,
  taskStageStatus,
  taskStatus,
  taskThinking,
  taskWindowSelection,
  tuningCustomRange,
  tuningGate,
  tuningPriorCoreData,
  tuningPriorCoreError,
  tuningPriorCoreLoading,
  tuningPriorCustomRange,
  tuningPriorOntologyData,
  tuningPriorOntologyError,
  tuningPriorOntologyLoading,
  tuningPriorRangePreset,
  tuningPriorReviewData,
  tuningPriorReviewError,
  tuningPriorReviewLoading,
  tuningRangePreset,
  tuningUseLlm,
  windowAlgorithmSummary,
  windowCustomRange,
  windowPreviewData,
  windowRangePreset,
  windows,
}: TuningModulePageProps) {
  switch (activeSub) {
    case 'id_windows':
      return (
        <SectionErrorBoundary label="窗口候选页面">
          <WindowCandidatesPanel
            selectedLoopId={selectedLoopId}
            selectedLoop={selectedLoop}
            scopedLoops={scopedLoops}
            loopTypeLabels={loopTypeLabel}
            featureRangeOptions={featureRangeOptions}
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
    case 'tuning_prior':
      return (
        <TuningPriorPanel
          loops={loops}
          selectedLoopId={selectedLoopId}
          selectedLoop={selectedLoop}
          loopTypeLabel={loopTypeLabel}
          featureRangeOptions={featureRangeOptions}
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
          chartLineTooltip={chartLineTooltip}
          onOpenTuningTask={switchToTuningTask}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
          scorePercent={scorePercent}
          scoreStatus={scoreStatus}
          assessment={assessment}
          selectedLoopId={selectedLoopId}
        />
      );
    case 'tuning_task':
      return (
        <TuningTaskPanel
          selectedLoopId={selectedLoopId}
          selectedLoop={selectedLoop}
          scopedLoops={scopedLoops}
          loopTypeLabel={loopTypeLabel}
          featureRangeOptions={featureRangeOptions}
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
      return null;
  }
}
