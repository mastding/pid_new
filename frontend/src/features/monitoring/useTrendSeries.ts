import { useCallback, useState } from 'react';
import type { Dayjs } from 'dayjs';
import { message } from 'antd';
import { getHistoryLoopSeries } from '@/services/api';
import type { HistoryLoop, LoopSeriesResp } from '@/services/api';
import {
  buildTrendSeriesParams as buildTrendSeriesQueryParams,
  type TrendPointLimit,
  type TrendPreset,
} from '@/features/monitoring/pageConfig';

export function useTrendSeries() {
  const [series, setSeries] = useState<LoopSeriesResp | null>(null);
  const [seriesLoading, setSeriesLoading] = useState(false);
  const [trendPreset, setTrendPreset] = useState<TrendPreset>('all');
  const [trendPointLimit, setTrendPointLimit] = useState<TrendPointLimit>('6000');
  const [trendSplitYAxis, setTrendSplitYAxis] = useState(false);
  const [trendCustomRange, setTrendCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);

  const buildTrendSeriesParams = useCallback((loop?: HistoryLoop) => {
    return buildTrendSeriesQueryParams(trendPreset, trendCustomRange, trendPointLimit, loop);
  }, [trendCustomRange, trendPointLimit, trendPreset]);

  const loadSeries = useCallback(async (loopId: string, loop?: HistoryLoop) => {
    setSeries(null);
    setSeriesLoading(true);
    try {
      const resp = await getHistoryLoopSeries(loopId, buildTrendSeriesParams(loop));
      if (resp.error) message.warning(resp.error);
      setSeries(resp);
    } catch (error) {
      message.error(`加载趋势失败：${String(error)}`);
    } finally {
      setSeriesLoading(false);
    }
  }, [buildTrendSeriesParams]);

  return {
    series,
    seriesLoading,
    trendPreset,
    trendPointLimit,
    trendSplitYAxis,
    trendCustomRange,
    loadSeries,
    setTrendPreset,
    setTrendPointLimit,
    setTrendSplitYAxis,
    setTrendCustomRange,
  };
}
