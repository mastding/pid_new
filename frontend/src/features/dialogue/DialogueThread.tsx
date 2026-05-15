import { RobotOutlined } from '@ant-design/icons';
import { Alert, Button, Empty } from 'antd';

export interface AssistantActionLike {
  label: string;
  target: string;
  sub: string;
  loopId?: string;
}

export interface AssistantEventLike {
  id: string;
  type: string;
  title: string;
  detail?: string;
}

export interface DialogueMessageLike {
  id: number;
  role: 'user' | 'assistant';
  text: string;
  reasoning?: string;
  loading?: boolean;
  error?: string;
  actions?: AssistantActionLike[];
  eventLog?: AssistantEventLike[];
}

interface DialogueStarterPrompt {
  title: string;
  description: string;
  prompt: string;
}

interface DialogueThreadProps {
  messages: DialogueMessageLike[];
  selectedLoopLabel?: string;
  actionLoopId?: string;
  streaming: boolean;
  starterPrompts: DialogueStarterPrompt[];
  normalizeAction: (line: string, loopId?: string | null) => AssistantActionLike | null;
  onRunAction: (action: AssistantActionLike) => void;
  onAskPrompt: (prompt: string) => void;
}

export function DialogueThread({
  messages,
  selectedLoopLabel,
  actionLoopId,
  streaming,
  starterPrompts,
  normalizeAction,
  onRunAction,
  onAskPrompt,
}: DialogueThreadProps) {
  const renderAssistantTextLine = (item: DialogueMessageLike, line: string, index: number) => {
    const action = normalizeAction(line, actionLoopId);
    if (action) {
      return (
        <p key={`${item.id}-${index}`} className="dialogue-action-line">
          <button type="button" className="dialogue-inline-action" onClick={() => onRunAction(action)}>
            {action.label}
          </button>
        </p>
      );
    }
    return <p key={`${item.id}-${index}`}>{line}</p>;
  };

  return (
    <div className="chat-thread">
      {messages.length ? (
        messages.map((item) => (
          <div key={item.id} className={item.role === 'user' ? 'chat-question' : 'chat-answer-row'}>
            {item.role === 'user' ? (
              <>
                {item.text}
                <span>刚刚</span>
              </>
            ) : (
              <>
                <div className="bot-avatar"><RobotOutlined /></div>
                <div className="chat-answer-card">
                  {!!item.eventLog?.length && (
                    <div className="dialogue-event-stream">
                      <div className="dialogue-event-title">事件流</div>
                      {item.eventLog.map((eventItem) => (
                        <div key={eventItem.id} className={`dialogue-event-item ${eventItem.type}`}>
                          <span className="dialogue-event-dot" />
                          <div>
                            <strong>{eventItem.title}</strong>
                            {eventItem.detail && <em>{eventItem.detail}</em>}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {(item.reasoning || item.loading) && (
                    <div className="ai-reasoning-box">
                      <div className="ai-reasoning-title">分析过程</div>
                      <div className="ai-reasoning-text">
                        {(item.reasoning || '正在读取上下文并生成分析摘要...').split('\n').map((line, index) => (
                          line ? <p key={`${item.id}-r-${index}`}>{line}</p> : null
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="ai-message-text">
                    {(item.text || (item.loading ? '正在生成回答...' : '')).split('\n').map((line, index) => renderAssistantTextLine(item, line, index))}
                    {item.loading && <span className="ai-stream-cursor" />}
                  </div>
                  {item.error && <Alert type="error" showIcon message={item.error} />}
                  {!!item.actions?.length && (
                    <div className="dialogue-actions">
                      {item.actions.map((action) => (
                        <Button key={`${item.id}-${action.label}`} size="small" onClick={() => onRunAction(action)}>
                          {action.label}
                        </Button>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        ))
      ) : (
        <div className="dialogue-starter">
          <div className="dialogue-starter-mark"><RobotOutlined /></div>
          <h2>{selectedLoopLabel ? `${selectedLoopLabel} 智能分析` : 'PID 智能整定助手'}</h2>
          <p>选择一个问题开始，或在下方直接输入你的问题。</p>
          <div className="dialogue-starter-grid">
            {starterPrompts.map((item) => (
              <button
                type="button"
                key={item.title}
                className="dialogue-starter-card"
                onClick={() => onAskPrompt(item.prompt)}
                disabled={streaming}
              >
                <strong>{item.title}</strong>
                <span>{item.description}</span>
              </button>
            ))}
          </div>
          {!starterPrompts.length && <Empty description="暂无快捷问题" />}
        </div>
      )}
    </div>
  );
}
