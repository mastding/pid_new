import { useEffect, useRef } from 'react';
import type { SubKey } from '@/features/app-shell/navigation';
import type { HistoryLoop } from '@/services/api';

interface UseLoopSelectionSyncOptions {
  activeSub: SubKey;
  dashboardWorstLoopId?: string;
  scopedLoops: HistoryLoop[];
  selectedAssetNodeId: string;
  selectedLoopId?: string;
  setSelectedLoopId: (loopId: string) => void;
}

export function useLoopSelectionSync({
  activeSub,
  dashboardWorstLoopId,
  scopedLoops,
  selectedAssetNodeId,
  selectedLoopId,
  setSelectedLoopId,
}: UseLoopSelectionSyncOptions) {
  const dashboardWorstSelectionRef = useRef<string | null>(null);

  useEffect(() => {
    if (!scopedLoops.length) return;
    if (!selectedLoopId || !scopedLoops.some((loop) => loop.loop_id === selectedLoopId)) {
      setSelectedLoopId(scopedLoops[0].loop_id);
    }
  }, [scopedLoops, selectedLoopId, setSelectedLoopId]);

  useEffect(() => {
    if (activeSub !== 'dashboard' || !dashboardWorstLoopId) return;
    const selectionKey = `${selectedAssetNodeId}:${dashboardWorstLoopId}`;
    if (dashboardWorstSelectionRef.current === selectionKey) return;
    dashboardWorstSelectionRef.current = selectionKey;
    setSelectedLoopId(dashboardWorstLoopId);
  }, [activeSub, dashboardWorstLoopId, selectedAssetNodeId, setSelectedLoopId]);
}
