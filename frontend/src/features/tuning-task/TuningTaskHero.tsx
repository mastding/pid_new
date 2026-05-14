import { Alert, Button, Tag, Typography } from 'antd';
import type { TaskStatus } from './model';

interface TuningTaskHeroProps {
  taskStatus: TaskStatus;
  taskId?: string;
  taskStartedAt?: string;
  running: boolean;
  taskError?: string;
  onStopTask: () => void;
}

function taskStatusText(status: TaskStatus) {
  if (status === 'running') return '运行中';
  if (status === 'done') return '已完成';
  if (status === 'error') return '异常/已停止';
  return '未开始';
}

function taskStatusColor(status: TaskStatus) {
  if (status === 'running') return 'processing';
  if (status === 'done') return 'green';
  if (status === 'error') return 'red';
  return 'default';
}

export function TuningTaskHero({
  taskStatus,
  taskId,
  taskStartedAt,
  running,
  taskError,
  onStopTask,
}: TuningTaskHeroProps) {
  return (
    <>
      <section className="agent-panel task-hero">
        <div>
          <div className="panel-title">整定任务驾驶舱</div>
          <Typography.Text type="secondary">
            把后端流式事件拆成可读过程：数据、窗口、辨识、大模型评审、精修、整定和评估都会沉淀在这里。
          </Typography.Text>
        </div>
        <div className="task-hero-actions">
          <Tag color={taskStatusColor(taskStatus)}>{taskStatusText(taskStatus)}</Tag>
          {taskId && <Tag color="blue">任务 ID：{taskId}</Tag>}
          {taskStartedAt && <Tag>开始：{taskStartedAt}</Tag>}
          {running && <Button danger onClick={onStopTask}>停止任务</Button>}
        </div>
      </section>

      {taskError && (
        <Alert type="error" showIcon message="任务未正常完成" description={taskError} />
      )}
    </>
  );
}
