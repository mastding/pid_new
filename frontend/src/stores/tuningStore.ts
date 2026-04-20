/** Module-level store for tuning page state.
 *
 * Survives route changes (TuningPage unmount/remount) so the user can
 * navigate away and come back without losing results.
 *
 * Uses React 18's useSyncExternalStore — no Provider, no extra deps.
 */
import { useSyncExternalStore } from 'react';
import type { UploadFile } from 'antd';
import type { TuningResult, WindowSelectionMeta, ModelReviewMeta } from '@/types/tuning';

export interface LlmThinkingPayload {
  stage: string;
  model: string;
  reasoning_content: string;
  raw_text: string;
}

export interface TuningState {
  fileList: UploadFile[];
  loopType: string;
  useLlmAdvisor: boolean;
  running: boolean;
  currentStage: number;
  stageData: Record<string, unknown>;
  windowSelection: WindowSelectionMeta | null;
  modelReview: ModelReviewMeta | null;
  llmThinking: LlmThinkingPayload | null;
  llmThinkingByStage: Record<string, LlmThinkingPayload>;
  taskId: string | null;
  result: TuningResult | null;
  error: string | null;
}

const initial: TuningState = {
  fileList: [],
  loopType: 'flow',
  useLlmAdvisor: true,
  running: false,
  currentStage: -1,
  stageData: {},
  windowSelection: null,
  modelReview: null,
  llmThinking: null,
  llmThinkingByStage: {},
  taskId: null,
  result: null,
  error: null,
};

let state: TuningState = initial;
const listeners = new Set<() => void>();

export function setTuningState(updater: (prev: TuningState) => TuningState): void {
  state = updater(state);
  listeners.forEach((l) => l());
}

export function resetTuningState(): void {
  state = { ...initial };
  listeners.forEach((l) => l());
}

export function useTuningStore(): TuningState {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => state,
    () => initial,
  );
}
