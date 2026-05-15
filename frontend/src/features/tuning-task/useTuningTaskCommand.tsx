import { useCallback } from 'react';
import { Modal, Space, Typography, message } from 'antd';
import type { HistoryLoop, HistoryLoopAssessment, HistoryTimeRangeParams } from '@/services/api';
import type { StartTuneOptions } from '@/features/tuning-task/useTuningTaskRuntime';

interface TuningGateState {
  hardBlocked: boolean;
  caution: boolean;
  nextAction?: string;
  blockingReasons: Array<{
    type: string;
    message: string;
  }>;
}

interface UseTuningTaskCommandOptions {
  assessment: HistoryLoopAssessment | null;
  buildTuningRangeParams: (loop?: HistoryLoop) => HistoryTimeRangeParams;
  selectedLoop?: HistoryLoop;
  startTune: (options?: StartTuneOptions) => void;
  tuningGate: TuningGateState;
  tuningUseLlm: boolean;
}

export function useTuningTaskCommand({
  assessment,
  buildTuningRangeParams,
  selectedLoop,
  startTune,
  tuningGate,
  tuningUseLlm,
}: UseTuningTaskCommandOptions) {
  return useCallback(() => {
    if (!selectedLoop) {
      message.warning('请先选择一个回路');
      return;
    }

    const tuningOptions: StartTuneOptions = {
      timeRange: buildTuningRangeParams(selectedLoop),
      useLlmAdvisor: tuningUseLlm,
      useSelectedWindow: false,
    };

    if (tuningGate.hardBlocked) {
      Modal.warning({
        title: '当前回路暂不建议发起整定',
        content: (
          <Space direction="vertical" size={8}>
            <Typography.Text>准入校验存在阻断项，请先处理数据质量、工况或约束问题。</Typography.Text>
            {tuningGate.blockingReasons.slice(0, 3).map((item, index) => (
              <Typography.Text key={`${item.type}-${index}`} type="secondary">
                {index + 1}. {item.message}
              </Typography.Text>
            ))}
          </Space>
        ),
      });
      return;
    }

    if (!assessment || tuningGate.caution) {
      Modal.confirm({
        title: assessment ? '当前回路建议谨慎整定' : '尚未拿到整定准入评估',
        content: (
          <Space direction="vertical" size={8}>
            <Typography.Text>
              {assessment
                ? (tuningGate.nextAction || '建议确认当前数据片段代表目标工况后再发起整定。')
                : '系统还没有加载到准入评估结果，继续发起会直接进入辨识和整定流程。'}
            </Typography.Text>
            {tuningGate.blockingReasons.slice(0, 3).map((item, index) => (
              <Typography.Text key={`${item.type}-${index}`} type="secondary">
                {index + 1}. {item.message}
              </Typography.Text>
            ))}
          </Space>
        ),
        okText: '确认发起',
        cancelText: '先不发起',
        onOk: () => startTune(tuningOptions),
      });
      return;
    }

    startTune(tuningOptions);
  }, [assessment, buildTuningRangeParams, selectedLoop, startTune, tuningGate, tuningUseLlm]);
}
