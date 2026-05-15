import { useCallback, useMemo, useState } from 'react';
import type { Dayjs } from 'dayjs';
import { message } from 'antd';
import { getHistoryLoopWindows } from '@/services/api';
import type { HistoryLoop, HistoryTimeRangeParams, HistoryWindow } from '@/services/api';
import {
  buildFeatureRangeParams as buildFeatureRangeQueryParams,
  type FeatureRangePreset,
} from '@/features/monitoring/pageConfig';

export function useLoopWindows() {
  const [windowRangePreset, setWindowRangePreset] = useState<FeatureRangePreset>('all');
  const [windowCustomRange, setWindowCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const [windows, setWindows] = useState<HistoryWindow[]>([]);
  const [windowAlgorithmSummary, setWindowAlgorithmSummary] = useState<Record<string, { total: number; usable: number }>>({});
  const [selectedWindowIndex, setSelectedWindowIndex] = useState<number>();

  const selectedWindow = useMemo(
    () => windows.find((item) => item.index === selectedWindowIndex),
    [selectedWindowIndex, windows],
  );

  const buildWindowRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    return buildFeatureRangeQueryParams(windowRangePreset, windowCustomRange, loop);
  }, [windowCustomRange, windowRangePreset]);

  const loadWindows = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setWindows([]);
    setWindowAlgorithmSummary({});
    setSelectedWindowIndex(undefined);
    try {
      const resp = await getHistoryLoopWindows(loopId, params);
      if (resp.error) message.warning(resp.error);
      setWindows(resp.windows ?? []);
      setWindowAlgorithmSummary(resp.algorithm_summary ?? {});
      const firstUsable = resp.windows?.find((item) => item.usable) ?? resp.windows?.[0];
      setSelectedWindowIndex(firstUsable?.index);
    } catch (error) {
      message.error(`加载辨识窗口失败：${String(error)}`);
    }
  }, []);

  return {
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
    setSelectedWindowIndex,
  };
}
