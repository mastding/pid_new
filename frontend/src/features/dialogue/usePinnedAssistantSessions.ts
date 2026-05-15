import { useCallback, useEffect, useMemo, useState } from 'react';

import type { AssistantSessionSummary } from '@/services/api';

const PINNED_ASSISTANT_SESSIONS_KEY = 'pid_v2_pinned_assistant_sessions';

function loadPinnedSessionIds() {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(PINNED_ASSISTANT_SESSIONS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === 'string') : [];
  } catch {
    return [];
  }
}

export function usePinnedAssistantSessions(sessions: AssistantSessionSummary[]) {
  const [pinnedSessionIds, setPinnedSessionIds] = useState<string[]>(loadPinnedSessionIds);

  useEffect(() => {
    try {
      window.localStorage.setItem(PINNED_ASSISTANT_SESSIONS_KEY, JSON.stringify(pinnedSessionIds));
    } catch {
      // Local pinning is an optional UI preference.
    }
  }, [pinnedSessionIds]);

  const pinnedSessionIdSet = useMemo(
    () => new Set(pinnedSessionIds),
    [pinnedSessionIds],
  );

  const sortedSessions = useMemo(() => {
    const pinOrder = new Map(pinnedSessionIds.map((id, index) => [id, index]));
    return [...sessions].sort((left, right) => {
      const leftPinned = pinOrder.has(left.id);
      const rightPinned = pinOrder.has(right.id);
      if (leftPinned && rightPinned) return (pinOrder.get(left.id) ?? 0) - (pinOrder.get(right.id) ?? 0);
      if (leftPinned) return -1;
      if (rightPinned) return 1;
      return 0;
    });
  }, [pinnedSessionIds, sessions]);

  const togglePin = useCallback((sessionId: string) => {
    setPinnedSessionIds((prev) => (
      prev.includes(sessionId)
        ? prev.filter((id) => id !== sessionId)
        : [sessionId, ...prev]
    ));
  }, []);

  const unpin = useCallback((sessionId: string) => {
    setPinnedSessionIds((prev) => prev.filter((id) => id !== sessionId));
  }, []);

  return {
    pinnedSessionIdSet,
    sortedSessions,
    togglePin,
    unpin,
  };
}
