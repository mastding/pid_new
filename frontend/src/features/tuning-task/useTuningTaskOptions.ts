import { useCallback, useState } from 'react';
import type { Dayjs } from 'dayjs';
import type { HistoryLoop, HistoryTimeRangeParams } from '@/services/api';
import {
  buildFeatureRangeParams as buildFeatureRangeQueryParams,
  DEFAULT_TIME_RANGE_PRESET,
  type FeatureRangePreset,
} from '@/features/monitoring/pageConfig';

export function useTuningTaskOptions() {
  const [tuningRangePreset, setTuningRangePreset] = useState<FeatureRangePreset>(DEFAULT_TIME_RANGE_PRESET);
  const [tuningCustomRange, setTuningCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const [tuningUseLlm, setTuningUseLlm] = useState<boolean>(true);

  const buildTuningRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    return buildFeatureRangeQueryParams(tuningRangePreset, tuningCustomRange, loop);
  }, [tuningCustomRange, tuningRangePreset]);

  return {
    tuningRangePreset,
    tuningCustomRange,
    tuningUseLlm,
    buildTuningRangeParams,
    setTuningRangePreset,
    setTuningCustomRange,
    setTuningUseLlm,
  };
}
