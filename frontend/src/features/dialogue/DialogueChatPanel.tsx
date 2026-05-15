import { SendOutlined } from '@ant-design/icons';
import { Button, Input, Select, Tag } from 'antd';

import type { HistoryLoop } from '@/services/api';
import {
  DialogueThread,
  type AssistantActionLike,
  type DialogueMessageLike,
  type DialogueStarterPrompt,
} from './DialogueThread';

interface DialogueChatPanelProps {
  loops: HistoryLoop[];
  loopTypeLabels: Record<string, string>;
  selectedLoopId?: string;
  selectedLoopLabel?: string;
  activeSessionTitle?: string;
  messages: DialogueMessageLike[];
  inputValue: string;
  streaming: boolean;
  starterPrompts: DialogueStarterPrompt[];
  onLoopChange: (loopId?: string) => void;
  onInputChange: (value: string) => void;
  onAsk: (preset?: string) => void;
  normalizeAction: (line: string, loopId?: string | null) => AssistantActionLike | null;
  onRunAction: (action: AssistantActionLike) => void;
}

export function DialogueChatPanel({
  loops,
  loopTypeLabels,
  selectedLoopId,
  selectedLoopLabel,
  activeSessionTitle,
  messages,
  inputValue,
  streaming,
  starterPrompts,
  onLoopChange,
  onInputChange,
  onAsk,
  normalizeAction,
  onRunAction,
}: DialogueChatPanelProps) {
  return (
    <section className="dialogue-chat">
      <div className="dialogue-loop-select">
        <Select
          size="large"
          allowClear
          value={selectedLoopId}
          onChange={onLoopChange}
          style={{ minWidth: 360 }}
          popupClassName="dialogue-loop-dropdown"
          placeholder="选择回路上下文"
          options={loops.map((loop) => ({
            value: loop.loop_id,
            label: `${loop.loop_id} · ${loopTypeLabels[loop.loop_type] ?? loop.loop_type}`,
          }))}
        />
        <Tag color={activeSessionTitle ? 'blue' : 'default'} style={{ marginLeft: 12 }}>
          {activeSessionTitle ? `当前对话：${activeSessionTitle}` : '未创建会话'}
        </Tag>
      </div>

      <DialogueThread
        messages={messages}
        selectedLoopLabel={selectedLoopLabel}
        actionLoopId={selectedLoopId}
        streaming={streaming}
        starterPrompts={starterPrompts}
        normalizeAction={normalizeAction}
        onRunAction={onRunAction}
        onAskPrompt={onAsk}
      />

      <div className="dialogue-input-row">
        <Input.TextArea
          value={inputValue}
          onChange={(event) => onInputChange(event.target.value)}
          autoSize={{ minRows: 2, maxRows: 4 }}
          placeholder="请输入您的问题，例如：这个回路为什么波动大？"
          onPressEnter={(event) => {
            if (!event.shiftKey) {
              event.preventDefault();
              onAsk();
            }
          }}
        />
        <Button type="primary" icon={<SendOutlined />} loading={streaming} onClick={() => onAsk()} />
      </div>
    </section>
  );
}
