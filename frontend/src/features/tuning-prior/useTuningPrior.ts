import { useCallback, useState } from 'react';
import type { Dayjs } from 'dayjs';
import { message } from 'antd';
import {
  getHistoryLoopTuningPriorCore,
  getHistoryLoopTuningPriorOntology,
  reviewHistoryLoopTuningPrior,
} from '@/services/api';
import type { HistoryLoop, HistoryLoopTuningPrior, HistoryTimeRangeParams } from '@/services/api';
import {
  buildFeatureRangeParams as buildFeatureRangeQueryParams,
  type FeatureRangePreset,
} from '@/features/monitoring/pageConfig';

export function useTuningPrior() {
  const [tuningPriorRangePreset, setTuningPriorRangePreset] = useState<FeatureRangePreset>('8h');
  const [tuningPriorCustomRange, setTuningPriorCustomRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const [tuningPriorCoreData, setTuningPriorCoreData] = useState<HistoryLoopTuningPrior | null>(null);
  const [tuningPriorOntologyData, setTuningPriorOntologyData] = useState<HistoryLoopTuningPrior | null>(null);
  const [tuningPriorReviewData, setTuningPriorReviewData] = useState<HistoryLoopTuningPrior | null>(null);
  const [tuningPriorCoreLoading, setTuningPriorCoreLoading] = useState(false);
  const [tuningPriorOntologyLoading, setTuningPriorOntologyLoading] = useState(false);
  const [tuningPriorReviewLoading, setTuningPriorReviewLoading] = useState(false);
  const [tuningPriorCoreError, setTuningPriorCoreError] = useState<string | null>(null);
  const [tuningPriorOntologyError, setTuningPriorOntologyError] = useState<string | null>(null);
  const [tuningPriorReviewError, setTuningPriorReviewError] = useState<string | null>(null);

  const buildTuningPriorRangeParams = useCallback((loop?: HistoryLoop): HistoryTimeRangeParams => {
    return buildFeatureRangeQueryParams(tuningPriorRangePreset, tuningPriorCustomRange, loop);
  }, [tuningPriorCustomRange, tuningPriorRangePreset]);

  const resetTuningPrior = useCallback(() => {
    setTuningPriorCoreData(null);
    setTuningPriorOntologyData(null);
    setTuningPriorReviewData(null);
    setTuningPriorCoreError(null);
    setTuningPriorOntologyError(null);
    setTuningPriorReviewError(null);
  }, []);

  const loadTuningPriorCore = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setTuningPriorCoreData(null);
    setTuningPriorReviewData(null);
    setTuningPriorCoreError(null);
    setTuningPriorCoreLoading(true);
    try {
      const resp = await getHistoryLoopTuningPriorCore(loopId, params);
      if (resp.error) message.warning(resp.error);
      setTuningPriorCoreData(resp);
    } catch (error) {
      const text = String(error);
      setTuningPriorCoreError(text);
      message.error(`加载核心指标失败：${text}`);
    } finally {
      setTuningPriorCoreLoading(false);
    }
  }, []);

  const loadTuningPriorOntology = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setTuningPriorOntologyData(null);
    setTuningPriorReviewData(null);
    setTuningPriorOntologyError(null);
    setTuningPriorOntologyLoading(true);
    try {
      const resp = await getHistoryLoopTuningPriorOntology(loopId, params);
      if (resp.error) message.warning(resp.error);
      setTuningPriorOntologyData(resp);
    } catch (error) {
      const text = String(error);
      setTuningPriorOntologyError(text);
      message.error(`查询本体失败：${text}`);
    } finally {
      setTuningPriorOntologyLoading(false);
    }
  }, []);

  const loadTuningPriorReview = useCallback(async (loopId: string) => {
    if (!tuningPriorCoreData?.core_context) {
      message.warning('请先生成核心指标与评估诊断上下文');
      return;
    }

    setTuningPriorReviewData(null);
    setTuningPriorReviewError(null);
    setTuningPriorReviewLoading(true);
    try {
      const resp = await reviewHistoryLoopTuningPrior(loopId, {
        core_context: tuningPriorCoreData.core_context,
        ontology: tuningPriorOntologyData?.ontology ?? null,
      });
      if (resp.error) message.warning(resp.error);
      setTuningPriorReviewData(resp);
    } catch (error) {
      const text = String(error);
      setTuningPriorReviewError(text);
      message.error(`生成大模型先验评审失败：${text}`);
    } finally {
      setTuningPriorReviewLoading(false);
    }
  }, [tuningPriorCoreData, tuningPriorOntologyData]);

  return {
    tuningPriorRangePreset,
    tuningPriorCustomRange,
    tuningPriorCoreData,
    tuningPriorOntologyData,
    tuningPriorReviewData,
    tuningPriorCoreLoading,
    tuningPriorOntologyLoading,
    tuningPriorReviewLoading,
    tuningPriorCoreError,
    tuningPriorOntologyError,
    tuningPriorReviewError,
    buildTuningPriorRangeParams,
    loadTuningPriorCore,
    loadTuningPriorOntology,
    loadTuningPriorReview,
    resetTuningPrior,
    setTuningPriorRangePreset,
    setTuningPriorCustomRange,
  };
}
