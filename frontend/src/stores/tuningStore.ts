/** Module-level store for tuning page state.
 *
 * Survives route changes (TuningPage unmount/remount) so the user can
 * navigate away and come back without losing results.
 *
 * Uses React 18's useSyncExternalStore — no Provider, no extra deps.
 */
import { useSyncExternalStore } from 'react';
import type { UploadFile } from 'antd';
import type { TuningResult, WindowSelectionMeta, ModelReviewMeta, IdentificationAttempt } from '@/types/tuning';

const STORAGE_KEY = 'pid_v2:tuning-state:v1';

export interface LlmThinkingPayload {
  stage: string;
  round?: number;
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
  identificationAttemptsHistory: IdentificationAttempt[];
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
  identificationAttemptsHistory: [],
  taskId: null,
  result: null,
  error: null,
};

function persistState(next: TuningState): void {
  if (typeof window === 'undefined') return;
  try {
    const serializable = {
      ...next,
      fileList: [] as UploadFile[],
    };
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
  } catch {
    // Ignore persistence errors and keep the in-memory store usable.
  }
}

function loadPersistedState(): TuningState {
  if (typeof window === 'undefined') return initial;
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return initial;
    const parsed = JSON.parse(raw) as Partial<TuningState>;
    return {
      ...initial,
      ...parsed,
      fileList: [],
      llmThinkingByStage: parsed.llmThinkingByStage ?? {},
      identificationAttemptsHistory: parsed.identificationAttemptsHistory ?? [],
      stageData: parsed.stageData ?? {},
    };
  } catch {
    return initial;
  }
}

let state: TuningState = loadPersistedState();
const listeners = new Set<() => void>();

export function setTuningState(updater: (prev: TuningState) => TuningState): void {
  state = updater(state);
  persistState(state);
  listeners.forEach((l) => l());
}

export function resetTuningState(): void {
  state = { ...initial };
  persistState(state);
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
