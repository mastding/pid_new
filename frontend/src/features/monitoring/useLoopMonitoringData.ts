import { useCallback, useEffect, useRef, useState } from 'react';
import type { Dayjs } from 'dayjs';
import { message } from 'antd';
import { fetchHistoryLoopFeatures, fetchHistoryLoopMonitoring } from '@/services/api';
import type {
  HistoryLoop,
  HistoryLoopFeatures,
  HistoryLoopMonitoring,
  HistoryTimeRangeParams,
} from '@/services/api';
import {
  buildFeatureRangeParams as buildFeatureRangeQueryParams,
  DEFAULT_TIME_RANGE_PRESET,
  type FeatureRangePreset,
} from '@/features/monitoring/pageConfig';

interface UseLoopMonitoringDataOptions {
  scopedLoops: HistoryLoop[];
  shouldLoadDashboardMonitoring: boolean;
}

export function useLoopMonitoringData({
  scopedLoops,
  shouldLoadDashboardMonitoring,
}: UseLoopMonitoringDataOptions) {
  const [featureRangePreset, setFeatureRangePreset] = useState<FeatureRangePreset>(DEFAULT_TIME_RANGE_PRESET);
  const [featureCustomRange, setFeatureCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const [featureLoading, setFeatureLoading] = useState(false);
  const [loopFeatures, setLoopFeatures] = useState<HistoryLoopFeatures | null>(null);
  const [loopMonitoring, setLoopMonitoring] = useState<HistoryLoopMonitoring | null>(null);
  const [monitoringByLoopId, setMonitoringByLoopId] = useState<Record<string, HistoryLoopMonitoring>>({});
  const monitoringBulkInFlightRef = useRef<Set<string>>(new Set());

  const buildFeatureRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    return buildFeatureRangeQueryParams(featureRangePreset, featureCustomRange, loop);
  }, [featureCustomRange, featureRangePreset]);

  const loadLoopFeatures = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setLoopFeatures(null);
    setFeatureLoading(true);
    try {
      const resp = await fetchHistoryLoopFeatures(loopId, params);
      if ((resp as { error?: string }).error) {
        message.warning((resp as { error?: string }).error);
        setLoopFeatures(null);
        return;
      }
      setLoopFeatures(resp);
    } catch (error) {
      message.error(`加载回路画像失败：${String(error)}`);
    } finally {
      setFeatureLoading(false);
    }
  }, []);

  const loadLoopMonitoring = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setLoopMonitoring(null);
    try {
      const resp = await fetchHistoryLoopMonitoring(loopId, params);
      if ((resp as { error?: string }).error) {
        message.warning((resp as { error?: string }).error);
        setLoopMonitoring(null);
        return;
      }
      setLoopMonitoring(resp);
      if (!params?.start_time && !params?.end_time) {
        setMonitoringByLoopId((prev) => ({ ...prev, [loopId]: resp }));
      }
      setLoopFeatures(resp.features ?? null);
    } catch (error) {
      message.error(`加载监控快照失败：${String(error)}`);
    }
  }, []);

  useEffect(() => {
    if (!shouldLoadDashboardMonitoring) return undefined;
    let cancelled = false;
    const missing = scopedLoops
      .map((loop) => loop.loop_id)
      .filter((loopId) => !monitoringByLoopId[loopId] && !monitoringBulkInFlightRef.current.has(loopId));
    if (!missing.length) return undefined;

    missing.forEach((loopId) => monitoringBulkInFlightRef.current.add(loopId));

    const loadMissingMonitoring = async () => {
      for (let index = 0; index < missing.length && !cancelled; index += 2) {
        const batch = missing.slice(index, index + 2);
        const results = await Promise.allSettled(batch.map((loopId) => fetchHistoryLoopMonitoring(loopId)));
        if (cancelled) break;
        setMonitoringByLoopId((prev) => {
          const next = { ...prev };
          results.forEach((result) => {
            if (result.status === 'fulfilled') {
              next[result.value.loop_id] = result.value;
            }
          });
          return next;
        });
        batch.forEach((loopId) => monitoringBulkInFlightRef.current.delete(loopId));
      }
    };

    const timer = window.setTimeout(() => {
      void loadMissingMonitoring().finally(() => {
        missing.forEach((loopId) => monitoringBulkInFlightRef.current.delete(loopId));
      });
    }, 800);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      missing.forEach((loopId) => monitoringBulkInFlightRef.current.delete(loopId));
    };
  }, [monitoringByLoopId, scopedLoops, shouldLoadDashboardMonitoring]);

  return {
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
  };
}
