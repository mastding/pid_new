import { useCallback, useEffect, useMemo, useState } from 'react';
import { message } from 'antd';
import { listHistoryLoops } from '@/services/api';
import type { HistoryLoop } from '@/services/api';

export function useHistoryLoops() {
  const [loops, setLoops] = useState<HistoryLoop[]>([]);
  const [selectedLoopId, setSelectedLoopId] = useState<string>();
  const [loading, setLoading] = useState(false);

  const selectedLoop = useMemo(
    () => loops.find((item) => item.loop_id === selectedLoopId),
    [loops, selectedLoopId],
  );

  const loadLoops = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await listHistoryLoops();
      setLoops(resp.items);
      setSelectedLoopId((current) => current ?? resp.items[0]?.loop_id);
    } catch (error) {
      message.error(`加载历史回路失败：${String(error)}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadLoops();
  }, [loadLoops]);

  return {
    loops,
    selectedLoopId,
    selectedLoop,
    loading,
    loadLoops,
    setSelectedLoopId,
  };
}
