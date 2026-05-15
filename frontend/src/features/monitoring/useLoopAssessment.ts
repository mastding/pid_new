import { useCallback, useState } from 'react';
import { message } from 'antd';
import { getHistoryLoopAssessment } from '@/services/api';
import type { HistoryLoopAssessment, HistoryTimeRangeParams } from '@/services/api';

export function useLoopAssessment() {
  const [assessment, setAssessment] = useState<HistoryLoopAssessment | null>(null);
  const [assessmentLoading, setAssessmentLoading] = useState(false);
  const [assessmentError, setAssessmentError] = useState<string | null>(null);

  const loadAssessment = useCallback(async (loopId: string, params?: HistoryTimeRangeParams) => {
    setAssessment(null);
    setAssessmentError(null);
    setAssessmentLoading(true);
    try {
      const resp = await getHistoryLoopAssessment(loopId, params);
      if (resp.error) message.warning(resp.error);
      setAssessment(resp);
    } catch (error) {
      setAssessmentError(String(error));
      message.error(`加载回路评估失败：${String(error)}`);
    } finally {
      setAssessmentLoading(false);
    }
  }, []);

  return {
    assessment,
    assessmentLoading,
    assessmentError,
    loadAssessment,
  };
}
