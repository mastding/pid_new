import {
  TuningTaskEventLogPanel,
} from '@/features/tuning-task/TuningTaskEventLogPanel';
import { TuningTaskHero } from '@/features/tuning-task/TuningTaskHero';
import { TuningTaskIdentificationPanel } from '@/features/tuning-task/TuningTaskIdentificationPanel';
import { TuningTaskKpiGrid } from '@/features/tuning-task/TuningTaskKpiGrid';
import { TuningTaskOntologyPanel } from '@/features/tuning-task/TuningTaskOntologyPanel';
import { TuningTaskResultPanels } from '@/features/tuning-task/TuningTaskResultPanels';
import { TuningTaskStagePanel } from '@/features/tuning-task/TuningTaskStagePanel';
import { TuningTaskThinkingPanel } from '@/features/tuning-task/TuningTaskThinkingPanel';
import { TuningTaskWindowReviewGrid } from '@/features/tuning-task/TuningTaskWindowReviewGrid';
import type {
  IdentificationAttempt,
  IdentificationRefinementMeta,
  LlmThinkingEvent,
  ModelReviewMeta,
  TuningResult,
  WindowAlgorithmFitSummary,
  WindowSelectionMeta,
} from '@/types/tuning';
import {
  TUNING_STAGE_KEYS,
  buildTaskStageCards,
  type TaskEventLog,
  type TaskStageDataMap,
  type TaskStageStatusMap,
  type TaskStatus,
} from './model';

interface TuningTaskDashboardProps {
  taskStatus: TaskStatus;
  taskId?: string;
  taskStartedAt?: string;
  running: boolean;
  taskError?: string;
  taskCurrentStage?: string;
  taskStageStatus: TaskStageStatusMap;
  taskStageData: TaskStageDataMap;
  taskWindowSelection: WindowSelectionMeta | null;
  taskModelReview: ModelReviewMeta | null;
  taskRefinements: IdentificationRefinementMeta[];
  taskAlgorithmComparison: WindowAlgorithmFitSummary[];
  taskAttempts: IdentificationAttempt[];
  taskResult: TuningResult | null;
  taskThinking: LlmThinkingEvent[];
  events: TaskEventLog[];
  rawLogExpanded: boolean;
  onStopTask: () => void;
  onSelectAttempt: (attemptKey: string) => void;
  onToggleRawLogExpanded: () => void;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
}

function scoreColor(score?: number) {
  if ((score ?? 0) >= 8) return '#22a06b';
  if ((score ?? 0) >= 6) return '#f59e0b';
  return '#f04438';
}

export function TuningTaskDashboard({
  taskStatus,
  taskId,
  taskStartedAt,
  running,
  taskError,
  taskCurrentStage,
  taskStageStatus,
  taskStageData,
  taskWindowSelection,
  taskModelReview,
  taskRefinements,
  taskAlgorithmComparison,
  taskAttempts,
  taskResult,
  taskThinking,
  events,
  rawLogExpanded,
  onStopTask,
  onSelectAttempt,
  onToggleRawLogExpanded,
  formatNumber,
  formatPercentValue,
}: TuningTaskDashboardProps) {
  const activeStep = taskStatus === 'done'
    ? TUNING_STAGE_KEYS.length
    : Math.max(TUNING_STAGE_KEYS.indexOf(taskCurrentStage ?? ''), 0);
  const idStage = taskStageData.identification;
  const tuningStage = taskStageData.tuning;
  const evaluationStage = taskStageData.evaluation;
  const evaluationPassed = (evaluationStage?.passed as boolean | undefined) ?? taskResult?.evaluation?.passed;
  const stageCards = buildTaskStageCards({
    taskStatus,
    taskCurrentStage,
    taskStageStatus,
    taskStageData,
  });

  return (
    <div className="task-dashboard">
      <TuningTaskHero
        taskStatus={taskStatus}
        taskId={taskId}
        taskStartedAt={taskStartedAt}
        running={running}
        taskError={taskError}
        onStopTask={onStopTask}
      />

      <TuningTaskStagePanel
        stageCards={stageCards}
        taskStatus={taskStatus}
        activeStep={activeStep}
      />

      <TuningTaskKpiGrid
        candidateWindowCount={(taskWindowSelection?.candidate_window_count
          ?? taskWindowSelection?.policy_adjusted_candidate_windows
          ?? '-') as number | string}
        usableWindowCount={(taskWindowSelection?.policy_adjusted_usable_windows
          ?? '-') as number | string}
        modelType={idStage?.model_type as string ?? taskResult?.model?.model_type ?? '-'}
        r2Score={formatNumber((idStage?.r2_score as number | undefined) ?? taskResult?.model?.r2_score, 3)}
        strategy={tuningStage?.strategy as string ?? taskResult?.pid_params?.strategy ?? '-'}
        kp={formatNumber((tuningStage?.Kp as number | undefined) ?? taskResult?.pid_params?.Kp, 3)}
        finalScore={formatNumber((evaluationStage?.final_rating as number | undefined) ?? taskResult?.evaluation?.final_rating, 1)}
        evaluationText={evaluationPassed === undefined ? '等待评估' : evaluationPassed ? '可以上线' : '需要优化'}
      />

      <TuningTaskOntologyPanel windowSelection={taskWindowSelection} />

      <TuningTaskWindowReviewGrid
        windowSelection={taskWindowSelection}
        modelReview={taskModelReview}
        refinements={taskRefinements}
        formatNumber={formatNumber}
      />

      <TuningTaskIdentificationPanel
        algorithmComparison={taskAlgorithmComparison}
        attempts={taskAttempts}
        formatNumber={formatNumber}
        formatPercentValue={formatPercentValue}
        onSelectAttempt={onSelectAttempt}
      />

      <TuningTaskResultPanels
        result={taskResult}
        formatNumber={formatNumber}
        scoreColor={scoreColor}
      />

      <TuningTaskThinkingPanel thinking={taskThinking} />

      <TuningTaskEventLogPanel
        events={events}
        rawLogExpanded={rawLogExpanded}
        onToggleExpanded={onToggleRawLogExpanded}
      />
    </div>
  );
}
