import { useCallback, useEffect, useState } from 'react';
import { message } from 'antd';
import { getSession, listSessions, tuneHistoryLoopStream } from '@/services/api';
import type { HistoryLoop, HistoryTimeRangeParams } from '@/services/api';
import type { SubKey } from '@/features/app-shell/navigation';
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

export interface StartTuneOptions {
  useSelectedWindow?: boolean;
  useLlmAdvisor?: boolean;
  stopAfter?: 'window_selection' | 'identification';
  timeRange?: HistoryTimeRangeParams;
}

interface UseTuningTaskRuntimeOptions {
  activeSub: SubKey;
  buildWindowRangeParams: (loop?: HistoryLoop) => HistoryTimeRangeParams;
  isSettingsView: boolean;
  onRunStart?: () => void;
  selectedLoop?: HistoryLoop;
  selectedWindowIndex?: number;
  shouldRestoreLatestTask: boolean;
}

export function useTuningTaskRuntime({
  activeSub,
  buildWindowRangeParams,
  isSettingsView,
  onRunStart,
  selectedLoop,
  selectedWindowIndex,
  shouldRestoreLatestTask,
}: UseTuningTaskRuntimeOptions) {
  const [running, setRunning] = useState(false);
  const [taskId, setTaskId] = useState<string>();
  const [taskStatus, setTaskStatus] = useState<TaskStatus>('idle');
  const [taskStartedAt, setTaskStartedAt] = useState<string>();
  const [taskCurrentStage, setTaskCurrentStage] = useState<string>();
  const [taskStageStatus, setTaskStageStatus] = useState<TaskStageStatusMap>({});
  const [taskStageData, setTaskStageData] = useState<TaskStageDataMap>({});
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
        // 恢复历史任务只是体验增强，失败时保持当前页面状态即可。
      }
    };

    void restoreLatestTask();
    return () => {
      cancelled = true;
    };
  }, [isSettingsView, running, selectedLoop, shouldRestoreLatestTask, taskResult]);

  const startTune = useCallback((options?: StartTuneOptions) => {
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
    onRunStart?.();
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
  }, [activeSub, buildWindowRangeParams, onRunStart, selectedLoop, selectedWindowIndex, taskAbort]);

  const handleStopTune = useCallback(() => {
    taskAbort?.abort();
    setTaskAbort(null);
    setRunning(false);
    setTaskStatus('error');
    setTaskError('任务已由用户停止');
  }, [taskAbort]);

  return {
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
  };
}
