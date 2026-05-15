import { useMemo } from 'react';
import type { LoopTrendPoint } from '@/features/charts/LoopTrendChart';
import type { HistoryWindow, LoopSeriesResp } from '@/services/api';

function seriesToTrendRows(series?: Pick<LoopSeriesResp, 'points'> | null): LoopTrendPoint[] {
  if (!series?.points?.length) return [];

  const rows: LoopTrendPoint[] = [];
  series.points.forEach((point) => {
    rows.push({ t: point.t, value: point.pv, series: 'PV' });
    rows.push({ t: point.t, value: point.mv, series: 'MV' });
    if (point.sv !== null && point.sv !== undefined) {
      rows.push({ t: point.t, value: point.sv, series: 'SV' });
    }
  });
  return rows;
}

function windowToPreviewRows(window?: Pick<HistoryWindow, 'preview'> | null): LoopTrendPoint[] {
  if (!window?.preview?.length) return [];

  const rows: LoopTrendPoint[] = [];
  window.preview.forEach((point) => {
    rows.push({ t: point.t, value: point.pv, series: 'PV' });
    rows.push({ t: point.t, value: point.mv, series: 'MV' });
  });
  return rows;
}

export function useLoopChartRows({
  selectedWindow,
  series,
}: {
  selectedWindow?: Pick<HistoryWindow, 'preview'> | null;
  series?: Pick<LoopSeriesResp, 'points'> | null;
}) {
  const trendData = useMemo(() => seriesToTrendRows(series), [series]);
  const windowPreviewData = useMemo(() => windowToPreviewRows(selectedWindow), [selectedWindow]);

  return {
    trendData,
    windowPreviewData,
  };
}
