import dayjs, { type Dayjs } from 'dayjs';

import type { SubKey } from '@/features/app-shell/navigation';
import type { HistoryLoop, HistoryTimeRangeParams } from '@/services/api';

export type TrendPreset = 'all' | '1h' | '6h' | '24h' | '7d' | 'custom';
export type TrendPointLimit = '6000' | '20000' | 'all';
export type FeatureRangePreset = 'all' | '8h' | '1d' | '3d' | '7d' | 'custom';

export const TREND_PRESET_OPTIONS: Array<{ label: string; value: TrendPreset; seconds?: number }> = [
  { label: '全部数据', value: 'all' },
  { label: '最近 1 小时', value: '1h', seconds: 3600 },
  { label: '最近 6 小时', value: '6h', seconds: 6 * 3600 },
  { label: '最近 24 小时', value: '24h', seconds: 24 * 3600 },
  { label: '最近 7 天', value: '7d', seconds: 7 * 24 * 3600 },
  { label: '自定义', value: 'custom' },
];

export const TREND_POINT_LIMIT_OPTIONS: Array<{ label: string; value: TrendPointLimit }> = [
  { label: '快速抽样 6000 点', value: '6000' },
  { label: '高精度 20000 点', value: '20000' },
  { label: '全量点', value: 'all' },
];

export const FEATURE_RANGE_OPTIONS: Array<{ label: string; value: FeatureRangePreset; seconds?: number }> = [
  { label: '全部历史', value: 'all' },
  { label: '最近 8 小时', value: '8h', seconds: 8 * 3600 },
  { label: '最近 1 天', value: '1d', seconds: 24 * 3600 },
  { label: '最近 3 天', value: '3d', seconds: 3 * 24 * 3600 },
  { label: '最近 7 天', value: '7d', seconds: 7 * 24 * 3600 },
  { label: '自定义', value: 'custom' },
];

export const ASSESSMENT_DETAIL_SUBS = new Set<SubKey>([
  'tuning_task',
  'tuning_readiness',
  'performance_score',
  'condition_recognition',
]);

export const WINDOW_DETAIL_SUBS = new Set<SubKey>([]);

export const FEATURE_DETAIL_SUBS = new Set<SubKey>([
  'loop_profile',
  'trend_spectrum',
  'performance_score',
  'condition_recognition',
  'actuator_status',
  'tuning_readiness',
  'diagnosis_overview',
  'pid_diagnosis',
  'valve_diagnosis',
  'measurement_noise_diagnosis',
  'process_disturbance_diagnosis',
  'model_reliability',
]);

export const MONITORING_DETAIL_SUBS = new Set<SubKey>([
  'alarm_events',
  'tuning_readiness',
  'condition_recognition',
  'diagnosis_overview',
]);

type CustomRange = [Dayjs | null, Dayjs | null] | null;

export function buildTrendSeriesParams(
  presetValue: TrendPreset,
  customRange: CustomRange,
  pointLimit: TrendPointLimit,
  loop?: HistoryLoop,
) {
  const params: { start_time?: string; end_time?: string; max_points?: number } = {
    max_points: pointLimit === 'all' ? 0 : Number(pointLimit),
  };
  if (presetValue === 'custom') {
    const [start, end] = customRange ?? [];
    if (start && end) {
      params.start_time = start.format('YYYY-MM-DD HH:mm:ss');
      params.end_time = end.format('YYYY-MM-DD HH:mm:ss');
    }
    return params;
  }
  if (presetValue === 'all') return params;
  const preset = TREND_PRESET_OPTIONS.find((item) => item.value === presetValue);
  if (!preset?.seconds) return params;
  const end = dayjs(loop?.end_time || undefined);
  const safeEnd = end.isValid() ? end : dayjs();
  params.start_time = safeEnd.subtract(preset.seconds, 'second').format('YYYY-MM-DD HH:mm:ss');
  params.end_time = safeEnd.format('YYYY-MM-DD HH:mm:ss');
  return params;
}

export function buildFeatureRangeParams(
  presetValue: FeatureRangePreset,
  customRange: CustomRange,
  loop?: HistoryLoop,
): HistoryTimeRangeParams {
  const params: HistoryTimeRangeParams = {};
  if (presetValue === 'custom') {
    const [start, end] = customRange ?? [];
    if (start && end) {
      params.start_time = start.format('YYYY-MM-DD HH:mm:ss');
      params.end_time = end.format('YYYY-MM-DD HH:mm:ss');
    }
    return params;
  }
  if (presetValue === 'all') return params;
  const preset = FEATURE_RANGE_OPTIONS.find((item) => item.value === presetValue);
  if (!preset?.seconds) return params;
  const end = dayjs(loop?.end_time || undefined);
  const safeEnd = end.isValid() ? end : dayjs();
  params.start_time = safeEnd.subtract(preset.seconds, 'second').format('YYYY-MM-DD HH:mm:ss');
  params.end_time = safeEnd.format('YYYY-MM-DD HH:mm:ss');
  return params;
}
