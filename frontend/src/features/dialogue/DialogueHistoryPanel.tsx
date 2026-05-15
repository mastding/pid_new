import dayjs from 'dayjs';
import { DeleteOutlined, EditOutlined, EllipsisOutlined, PushpinOutlined, SyncOutlined } from '@ant-design/icons';
import { Button, Dropdown, Empty } from 'antd';

import type { AssistantSessionSummary } from '@/services/api';

interface DialogueHistoryPanelProps {
  sessions: AssistantSessionSummary[];
  activeSessionId?: string;
  pinnedSessionIds: Set<string>;
  loading: boolean;
  onCreateSession: () => void;
  onOpenSession: (sessionId: string) => void;
  onTogglePin: (sessionId: string) => void;
  onRename: (session: AssistantSessionSummary) => void;
  onDelete: (session: AssistantSessionSummary) => void;
}

export function DialogueHistoryPanel({
  sessions,
  activeSessionId,
  pinnedSessionIds,
  loading,
  onCreateSession,
  onOpenSession,
  onTogglePin,
  onRename,
  onDelete,
}: DialogueHistoryPanelProps) {
  return (
    <aside className="dialogue-history">
      <div className="dialogue-history-head">
        <h2>历史对话</h2>
        <Button size="small" type="primary" ghost icon={<SyncOutlined />} loading={loading} onClick={onCreateSession}>
          新建对话
        </Button>
      </div>
      <div className="history-list">
        {sessions.length ? (
          <>
            <div className="history-group">最近</div>
            {sessions.map((item) => {
              const isActive = item.id === activeSessionId;
              const isPinned = pinnedSessionIds.has(item.id);
              return (
                <div key={item.id} className={isActive ? 'history-item active' : 'history-item'}>
                  <button
                    type="button"
                    className="history-item-main"
                    onClick={() => onOpenSession(item.id)}
                  >
                    <span>{item.title || '未命名对话'}</span>
                    <em>{item.updated_at ? dayjs(item.updated_at).format('HH:mm') : ''}</em>
                  </button>
                  <Dropdown
                    trigger={['click']}
                    placement="bottomRight"
                    menu={{
                      items: [
                        { key: 'pin', icon: <PushpinOutlined />, label: isPinned ? '取消置顶' : '置顶' },
                        { key: 'rename', icon: <EditOutlined />, label: '重命名' },
                        {
                          key: 'delete',
                          icon: <DeleteOutlined />,
                          label: <span className="history-danger-menu-item">删除</span>,
                          danger: true,
                        },
                      ],
                      onClick: ({ key, domEvent }) => {
                        domEvent.stopPropagation();
                        if (key === 'pin') onTogglePin(item.id);
                        if (key === 'rename') onRename(item);
                        if (key === 'delete') onDelete(item);
                      },
                    }}
                  >
                    <Button
                      type="text"
                      size="small"
                      className="history-more-btn"
                      icon={<EllipsisOutlined />}
                      onClick={(event) => event.stopPropagation()}
                    />
                  </Dropdown>
                </div>
              );
            })}
          </>
        ) : (
          <Empty description="暂无历史对话" />
        )}
      </div>
    </aside>
  );
}
