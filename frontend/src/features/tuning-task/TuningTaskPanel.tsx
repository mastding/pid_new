import type { Dispatch, SetStateAction } from 'react';
import { useCallback, useEffect, useState } from 'react';
import dayjs, { type Dayjs } from 'dayjs';
import {
  Alert,
  Button,
  DatePicker,
  Descriptions,
  Empty,
  Select,
  Space,
  Statistic,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd';
import { RocketOutlined } from '@ant-design/icons';

import {
  listAutoTuningTasks,
  prepareAutoTuningTask,
  type AutoTuningTask,
  type HistoryLoop,
  type PreparedAutoTuningTask,
} from '@/services/api';
import type { TuningResult } from '@/types/tuning';
import {
  TUNING_STAGE_KEYS,
  TUNING_STAGE_LABELS,
  summarizeTaskStage,
  type TaskStageDataMap,
  type TaskStageStatusMap,
  type TaskStatus,
} from '@/features/tuning-task/model';

interface BackendBadgeProps {
  implemented?: boolean;
}

function BackendBadge({ implemented }: BackendBadgeProps) {
  return <Tag color={implemented ? 'green' : 'default'}>{implemented ? '后端已接入' : '待接入'}</Tag>;
}

type FeatureRangeOption = {
  label: string;
  value: string;
  seconds?: number;
};

interface GateCheck {
  name?: string;
  passed?: boolean;
  severity?: string;
  message?: string;
  evidence?: Record<string, unknown>;
}

interface BlockingReason {
  type: string;
  severity: string;
  message: string;
}

interface TuningGate {
  decision?: string;
  hardBlocked: boolean;
  caution: boolean;
  score?: number;
  level?: string;
  gateChecks: GateCheck[];
  blockingReasons: BlockingReason[];
}

interface TuningTaskPanelProps {
  selectedLoopId?: string;
  selectedLoop?: HistoryLoop;
  scopedLoops: HistoryLoop[];
  loopTypeLabel: Record<string, string>;
  featureRangeOptions: FeatureRangeOption[];
  tuningRangePreset: string;
  tuningCustomRange: [Dayjs | null, Dayjs | null] | null;
  tuningUseLlm: boolean;
  running: boolean;
  tuningGate: TuningGate;
  assessmentLoading: boolean;
  assessmentError?: string | null;
  assessment?: unknown;
  taskAttemptsCount: number;
  taskStageStatus: TaskStageStatusMap;
  taskStageData: TaskStageDataMap;
  taskStatus: TaskStatus;
  taskId?: string | null;
  taskCurrentStage?: string | null;
  taskResult?: TuningResult | null;
  onLoopChange: Dispatch<SetStateAction<string | undefined>>;
  onRangePresetChange: Dispatch<SetStateAction<string>>;
  onCustomRangeChange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  onUseLlmChange: Dispatch<SetStateAction<boolean>>;
  onAutoTaskPrepared?: (prepared: PreparedAutoTuningTask) => void;
  onTune: () => void;
  onStopTune: () => void;
  onOpenTaskDetail: () => void;
  gateDecisionText: (decision?: string) => string;
  gateCheckLabel: (value?: string) => string;
  gateSeverityColor: (severity?: string) => string;
  gateImpact: (check: GateCheck, blockingReasons: BlockingReason[]) => { text: string; color: string };
  gateCheckMessage: (check: GateCheck, blockingReasons: BlockingReason[]) => string;
  tagColor: (level?: string) => string;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
}

export function TuningTaskPanel({
  selectedLoopId,
  selectedLoop,
  scopedLoops,
  loopTypeLabel,
  featureRangeOptions,
  tuningRangePreset,
  tuningCustomRange,
  tuningUseLlm,
  running,
  tuningGate,
  assessmentLoading,
  assessmentError,
  assessment,
  taskAttemptsCount,
  taskStageStatus,
  taskStageData,
  taskStatus,
  taskId,
  taskCurrentStage,
  taskResult,
  onLoopChange,
  onRangePresetChange,
  onCustomRangeChange,
  onUseLlmChange,
  onAutoTaskPrepared,
  onTune,
  onStopTune,
  onOpenTaskDetail,
  gateDecisionText,
  gateCheckLabel,
  gateSeverityColor,
  gateImpact,
  gateCheckMessage,
  tagColor,
  formatNumber,
  formatPercentValue,
}: TuningTaskPanelProps) {
  const [autoTasks, setAutoTasks] = useState<AutoTuningTask[]>([]);
  const [autoTasksLoading, setAutoTasksLoading] = useState(false);
  const [preparingTaskId, setPreparingTaskId] = useState<string | null>(null);
  const loadAutoTasks = useCallback(async () => {
    setAutoTasksLoading(true);
    try {
      const resp = await listAutoTuningTasks({ loop_id: selectedLoopId, limit: 8 });
      setAutoTasks(resp.items ?? []);
    } catch {
      setAutoTasks([]);
    } finally {
      setAutoTasksLoading(false);
    }
  }, [selectedLoopId]);

  useEffect(() => {
    void loadAutoTasks();
  }, [loadAutoTasks]);

  const handlePrepareAutoTask = useCallback(async (task: AutoTuningTask) => {
    setPreparingTaskId(task.task_id);
    try {
      const prepared = await prepareAutoTuningTask(task.task_id, {
        confirm: true,
        use_llm_advisor: tuningUseLlm,
      });
      if (!prepared.guard.allowed) {
        message.warning(prepared.guard.reason || '\u5019\u9009\u4efb\u52a1\u672a\u901a\u8fc7\u6574\u5b9a\u95e8\u7981');
        await loadAutoTasks();
        return;
      }
      const request = prepared.tuning_request;
      if (request.loop_id) {
        onLoopChange(request.loop_id);
      }
      if (request.start_time && request.end_time) {
        onRangePresetChange('custom');
        onCustomRangeChange([dayjs(request.start_time), dayjs(request.end_time)]);
      }
      if (typeof request.use_llm_advisor === 'boolean') {
        onUseLlmChange(request.use_llm_advisor);
      }
      onAutoTaskPrepared?.(prepared);
      message.success('\u5df2\u8f7d\u5165\u81ea\u52a8\u5019\u9009\u4efb\u52a1\uff0c\u8bf7\u590d\u6838\u540e\u53d1\u8d77\u6574\u5b9a');
      await loadAutoTasks();
    } catch (error) {
      message.error(error instanceof Error ? error.message : '\u8f7d\u5165\u81ea\u52a8\u5019\u9009\u4efb\u52a1\u5931\u8d25');
    } finally {
      setPreparingTaskId(null);
    }
  }, [
    loadAutoTasks,
    onCustomRangeChange,
    onLoopChange,
    onRangePresetChange,
    onAutoTaskPrepared,
    onUseLlmChange,
    tuningUseLlm,
  ]);

  const selectedRangeLabel = tuningRangePreset === 'all'
    ? '全部历史'
    : tuningRangePreset === 'custom'
      ? '自定义区间'
      : featureRangeOptions.find((item) => item.value === tuningRangePreset)?.label ?? tuningRangePreset;

  return (
    <div className="page-stack">
      <section className="agent-panel tuning-launch-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">发起整定任务</div>
            <Typography.Text type="secondary">
              选择需要整定的回路与时间区间；整定流水线按"数据画像 → 本体策略 → 窗口候选与选择 → 辨识 → 整定 → 评估"顺序执行，与窗口候选页面共用同一套本体驱动逻辑。
            </Typography.Text>
          </div>
          <Space wrap>
            <Select
              showSearch
              size="small"
              style={{ minWidth: 280 }}
              placeholder="选择整定回路"
              value={selectedLoopId}
              onChange={onLoopChange}
              optionFilterProp="label"
              options={scopedLoops.map((loop) => ({
                value: loop.loop_id,
                label: `${loop.loop_id} · ${loopTypeLabel[loop.loop_type] ?? loop.loop_type}`,
              }))}
            />
            <Select
              size="small"
              style={{ width: 140 }}
              value={tuningRangePreset}
              onChange={onRangePresetChange}
              options={featureRangeOptions.map((item) => ({ label: item.label, value: item.value }))}
            />
            {tuningRangePreset === 'custom' && (
              <DatePicker.RangePicker
                size="small"
                showTime
                value={tuningCustomRange}
                onChange={onCustomRangeChange}
              />
            )}
            <Tooltip title="关闭后流水线全程走确定性算法（本体策略与窗口选择不再调用大模型）">
              <Space size={4}>
                <span style={{ fontSize: 12, color: 'rgba(0,0,0,0.55)' }}>大模型顾问</span>
                <Switch size="small" checked={tuningUseLlm} onChange={onUseLlmChange} />
              </Space>
            </Tooltip>
            <Button type="primary" icon={<RocketOutlined />} loading={running} disabled={!selectedLoop} onClick={onTune}>
              发起整定
            </Button>
            {running && <Button danger onClick={onStopTune}>停止</Button>}
          </Space>
        </div>

        {selectedLoop ? (
          <div className="tuning-launch-summary">
            <Statistic title="当前整定回路" value={selectedLoop.loop_id} />
            <Descriptions bordered column={3} size="small" className="industrial-descriptions">
              <Descriptions.Item label="类型">{loopTypeLabel[selectedLoop.loop_type] ?? selectedLoop.loop_type}</Descriptions.Item>
              <Descriptions.Item label="时间区间">{selectedRangeLabel}</Descriptions.Item>
              <Descriptions.Item label="选窗策略">
                <Tag color={tuningUseLlm ? 'blue' : 'default'}>
                  {tuningUseLlm ? '本体策略 + 大模型顾问' : '确定性算法（大模型已关闭）'}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="当前准入" span={3}>
                <Tag color={tuningGate.hardBlocked ? 'red' : tuningGate.caution ? 'orange' : 'green'}>
                  {gateDecisionText(tuningGate.decision)}
                </Tag>
              </Descriptions.Item>
            </Descriptions>
          </div>
        ) : <Empty description="请先选择回路" />}
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">自动整定候选队列</div>
            <Typography.Text type="secondary">
              来自实时评估快照的候选任务默认停留在待复核/待执行状态，不会绕过工程师确认直接下发参数。
            </Typography.Text>
          </div>
          <Button size="small" onClick={loadAutoTasks} loading={autoTasksLoading}>刷新队列</Button>
        </div>
        <Table<AutoTuningTask>
          size="small"
          pagination={false}
          rowKey="task_id"
          dataSource={autoTasks}
          locale={{ emptyText: '暂无自动评估生成的整定候选任务' }}
          columns={[
            { title: '任务ID', dataIndex: 'task_id', ellipsis: true },
            { title: '回路', dataIndex: 'loop_id', width: 150 },
            { title: '状态', dataIndex: 'status', width: 120, render: (value: string) => <Tag color={value === 'blocked' ? 'red' : value === 'pending' ? 'blue' : 'orange'}>{value}</Tag> },
            { title: '触发方式', dataIndex: 'trigger_mode', width: 120 },
            { title: '触发原因', dataIndex: 'trigger_reason', ellipsis: true },
            { title: '创建时间', dataIndex: 'created_at', width: 180 },
            {
              title: '\u64cd\u4f5c',
              width: 120,
              render: (_, row) => (
                <Button
                  size="small"
                  type={row.status === 'blocked' ? 'default' : 'link'}
                  disabled={row.status === 'blocked' || running}
                  loading={preparingTaskId === row.task_id}
                  onClick={() => void handlePrepareAutoTask(row)}
                >
                  {'\u786e\u8ba4\u8f7d\u5165'}
                </Button>
              ),
            },
          ]}
        />
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">整定准入校验</div>
            <Typography.Text type="secondary">发起整定前先查看数据质量、工况、约束、振荡和可辨识性门槛。</Typography.Text>
          </div>
          <Space wrap>
            <BackendBadge implemented />
            <Tag color={tuningGate.hardBlocked ? 'red' : tuningGate.caution ? 'orange' : 'green'}>
              {gateDecisionText(tuningGate.decision)}
            </Tag>
            <Tag color={tagColor(tuningGate.level)}>{formatPercentValue(tuningGate.score, 0)}</Tag>
          </Space>
        </div>
        {assessmentLoading ? null : assessmentError ? (
          <Alert className="agent-alert" type="error" showIcon message="整定准入后端接口调用失败" description={assessmentError} />
        ) : assessment ? (
          <div className="page-stack compact-stack">
            <Table<GateCheck>
              size="small"
              pagination={false}
              rowKey={(row) => row.name ?? row.message ?? 'gate-check'}
              dataSource={tuningGate.gateChecks}
              columns={[
                { title: '校验项', dataIndex: 'name', width: 180, render: (value: string) => gateCheckLabel(value) },
                {
                  title: '结果',
                  dataIndex: 'passed',
                  width: 100,
                  render: (value: boolean) => <Tag color={value ? 'green' : 'red'}>{value ? '通过' : '未通过'}</Tag>,
                },
                {
                  title: '级别',
                  dataIndex: 'severity',
                  width: 100,
                  render: (value: string) => <Tag color={gateSeverityColor(value)}>{value || '-'}</Tag>,
                },
                {
                  title: '准入影响',
                  width: 110,
                  render: (_, row) => {
                    const impact = gateImpact(row, tuningGate.blockingReasons);
                    return <Tag color={impact.color}>{impact.text}</Tag>;
                  },
                },
                {
                  title: '说明',
                  dataIndex: 'message',
                  render: (_, row) => gateCheckMessage(row, tuningGate.blockingReasons),
                },
              ]}
            />
            {tuningGate.blockingReasons.length ? (
              <Alert
                className="agent-alert gate-alert"
                type={tuningGate.hardBlocked ? 'error' : 'warning'}
                showIcon
                message="准入提醒"
                description={(
                  <Space direction="vertical" size={4}>
                    {tuningGate.blockingReasons.map((reason, index) => (
                      <Typography.Text className="gate-alert-text" key={`${reason.type}-${index}`}>
                        {index + 1}. {gateCheckLabel(reason.type)}：{reason.message}
                      </Typography.Text>
                    ))}
                  </Space>
                )}
              />
            ) : null}
          </div>
        ) : (
          <Alert className="agent-alert" type="warning" showIcon message="暂无整定准入校验结果" description="请选择回路或刷新数据。该区域已接入后端 assessment 接口，不再使用模拟数据。" />
        )}
      </section>

      <section className="agent-panel task-process-summary">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">整定流程总览</div>
            <Typography.Text type="secondary">主界面保留阶段态势，详细辨识记录、模型判断和候选参数进入抽屉查看。</Typography.Text>
          </div>
          <Space wrap>
            <Tag color={taskAttemptsCount ? 'processing' : 'default'}>{taskAttemptsCount} 次辨识尝试</Tag>
            <Button onClick={onOpenTaskDetail}>查看全流程详情</Button>
          </Space>
        </div>
        <Table
          size="small"
          pagination={false}
          rowKey="stage"
          dataSource={TUNING_STAGE_KEYS.map((stage) => ({
            stage,
            label: TUNING_STAGE_LABELS[stage],
            state: taskStageStatus[stage] ?? (taskStageData[stage] ? 'done' : 'wait'),
            summary: summarizeTaskStage(stage, taskStageData[stage]),
          }))}
          columns={[
            { title: '阶段', dataIndex: 'label', width: 160 },
            {
              title: '状态',
              dataIndex: 'state',
              width: 110,
              render: (value: string) => (
                <Tag color={value === 'running' ? 'processing' : value === 'done' ? 'green' : 'default'}>
                  {value === 'running' ? '运行中' : value === 'done' ? '完成' : '等待'}
                </Tag>
              ),
            },
            { title: '摘要', dataIndex: 'summary', ellipsis: true },
          ]}
        />
      </section>

      <section className="agent-panel">
        <div className="panel-title">任务结果摘要</div>
        <Descriptions bordered column={3} size="small" className="industrial-descriptions">
          <Descriptions.Item label="任务状态">
            <Tag color={taskStatus === 'running' ? 'processing' : taskStatus === 'done' ? 'green' : taskStatus === 'error' ? 'red' : 'default'}>
              {taskStatus === 'running' ? '运行中' : taskStatus === 'done' ? '已完成' : taskStatus === 'error' ? '异常/停止' : '未开始'}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="任务 ID">{taskId || '-'}</Descriptions.Item>
          <Descriptions.Item label="当前阶段">{taskCurrentStage ? TUNING_STAGE_LABELS[taskCurrentStage] ?? taskCurrentStage : '-'}</Descriptions.Item>
          <Descriptions.Item label="推荐模型">{taskResult?.model?.model_type ?? (taskStageData.identification?.model_type as string | undefined) ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="推荐策略">{taskResult?.pid_params?.strategy ?? (taskStageData.tuning?.strategy as string | undefined) ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="综合评分">{formatNumber(taskResult?.evaluation?.final_rating ?? (taskStageData.evaluation?.final_rating as number | undefined), 1)}</Descriptions.Item>
        </Descriptions>
      </section>
    </div>
  );
}
