import type { SubKey } from '@/features/app-shell/navigation';

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
