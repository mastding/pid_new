import { Tag, Typography } from 'antd';
import { TUNING_STAGE_KEYS, type TaskStageCardModel, type TaskStatus } from './model';

interface TuningTaskStagePanelProps {
  stageCards: TaskStageCardModel[];
  taskStatus: TaskStatus;
  activeStep: number;
}

export function TuningTaskStagePanel({ stageCards, taskStatus, activeStep }: TuningTaskStagePanelProps) {
  return (
    <section className="agent-panel task-stage-panel">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">流程节点</div>
          <Typography.Text type="secondary">
            长摘要改为节点卡片展示，大模型评审和精修建议会保留关键判断，不再被横向步骤截断。
          </Typography.Text>
        </div>
        <Tag color={taskStatus === 'running' ? 'processing' : taskStatus === 'done' ? 'green' : taskStatus === 'error' ? 'red' : 'default'}>
          {taskStatus === 'idle'
            ? `等待 0 / ${TUNING_STAGE_KEYS.length}`
            : `当前 ${Math.min(taskStatus === 'done' ? TUNING_STAGE_KEYS.length : activeStep + 1, TUNING_STAGE_KEYS.length)} / ${TUNING_STAGE_KEYS.length}`}
        </Tag>
      </div>
      <div className="task-stage-grid">
        {stageCards.map((item) => (
          <div
            key={item.stage}
            className={`task-stage-card is-${item.status}${item.isCurrent ? ' is-current' : ''}`}
          >
            <div className="stage-index">{item.status === 'done' ? '✓' : item.index + 1}</div>
            <div className="stage-body">
              <div className="stage-title-row">
                <strong>{item.label}</strong>
                <Tag color={item.status === 'running' ? 'processing' : item.status === 'done' ? 'green' : 'default'}>
                  {item.status === 'running' ? '运行' : item.status === 'done' ? '完成' : '等待'}
                </Tag>
              </div>
              <p title={item.summary}>{item.summary}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
