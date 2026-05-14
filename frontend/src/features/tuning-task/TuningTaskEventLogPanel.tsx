import { Alert, Button, Tag, Typography } from 'antd';
import { formatEventDetail, type TaskEventLog } from './model';

interface TuningTaskEventLogPanelProps {
  events: TaskEventLog[];
  rawLogExpanded: boolean;
  onToggleExpanded: () => void;
}

export function TuningTaskEventLogPanel({
  events,
  rawLogExpanded,
  onToggleExpanded,
}: TuningTaskEventLogPanelProps) {
  const visibleEvents = rawLogExpanded ? events : events.slice(0, 8);

  return (
    <section className="agent-panel raw-log-panel">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">原始事件日志</div>
          <Typography.Text type="secondary">
            {events.length
              ? `共 ${events.length} 条，${rawLogExpanded ? '已展开全部' : '默认显示最近 8 条'}`
              : '保留后端 SSE 原始事件，便于排查'}
          </Typography.Text>
        </div>
        {events.length > 8 && (
          <Button size="small" onClick={onToggleExpanded}>
            {rawLogExpanded ? '收起日志' : '展开全部'}
          </Button>
        )}
      </div>
      {events.length ? (
        <div className={`event-log-box${rawLogExpanded ? ' is-expanded' : ''}`}>
          {visibleEvents.map((item, index) => (
            <div className="event-log-item" key={item.id}>
              <div className="event-log-head">
                <Tag color="blue">#{events.length - index}</Tag>
                <Typography.Text strong>{item.label}</Typography.Text>
              </div>
              {item.detail && (
                <pre className="event-detail">{formatEventDetail(item.detail)}</pre>
              )}
            </div>
          ))}
        </div>
      ) : (
        <Alert type="info" showIcon message="点击发起整定后，这里会保留后端 SSE 原始事件，便于排查。" />
      )}
    </section>
  );
}
