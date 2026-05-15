import { PidAppTopbar } from '@/features/app-shell/PidAppTopbar';
import type { AssistantSessionSummary, HistoryLoop } from '@/services/api';
import { DialogueChatPanel } from './DialogueChatPanel';
import { DialogueHistoryPanel } from './DialogueHistoryPanel';
import type { AssistantAction, AssistantMessage } from './model';
import type { DialogueStarterPrompt } from './DialogueThread';

type ViewMode = 'dialogue' | 'classic';

interface DialogueModePageProps {
  sidebarCollapsed: boolean;
  viewMode: ViewMode;
  sessions: AssistantSessionSummary[];
  activeSessionId?: string;
  pinnedSessionIds: Set<string>;
  sessionsLoading: boolean;
  loops: HistoryLoop[];
  loopTypeLabels: Record<string, string>;
  selectedLoopId?: string;
  selectedLoopLabel?: string;
  activeSessionTitle?: string;
  messages: AssistantMessage[];
  inputValue: string;
  streaming: boolean;
  starterPrompts: DialogueStarterPrompt[];
  onSidebarToggle: () => void;
  onViewModeChange: (mode: ViewMode) => void;
  onCreateSession: () => void;
  onOpenSession: (sessionId: string) => void;
  onTogglePin: (sessionId: string) => void;
  onRename: (session: AssistantSessionSummary) => void;
  onDelete: (session: AssistantSessionSummary) => void;
  onLoopChange: (loopId?: string) => void;
  onInputChange: (value: string) => void;
  onAsk: (preset?: string) => void;
  normalizeAction: (line: string, loopId?: string | null) => AssistantAction | null;
  onRunAction: (action: AssistantAction) => void;
}

export function DialogueModePage({
  sidebarCollapsed,
  viewMode,
  sessions,
  activeSessionId,
  pinnedSessionIds,
  sessionsLoading,
  loops,
  loopTypeLabels,
  selectedLoopId,
  selectedLoopLabel,
  activeSessionTitle,
  messages,
  inputValue,
  streaming,
  starterPrompts,
  onSidebarToggle,
  onViewModeChange,
  onCreateSession,
  onOpenSession,
  onTogglePin,
  onRename,
  onDelete,
  onLoopChange,
  onInputChange,
  onAsk,
  normalizeAction,
  onRunAction,
}: DialogueModePageProps) {
  return (
    <div className="dialogue-shell">
      <PidAppTopbar
        sidebarCollapsed={sidebarCollapsed}
        viewMode={viewMode}
        onSidebarToggle={onSidebarToggle}
        onViewModeChange={onViewModeChange}
      />

      <main className={sidebarCollapsed ? 'dialogue-main history-collapsed' : 'dialogue-main'}>
        <DialogueHistoryPanel
          sessions={sessions}
          activeSessionId={activeSessionId}
          pinnedSessionIds={pinnedSessionIds}
          loading={sessionsLoading}
          onCreateSession={onCreateSession}
          onOpenSession={onOpenSession}
          onTogglePin={onTogglePin}
          onRename={onRename}
          onDelete={onDelete}
        />

        <DialogueChatPanel
          loops={loops}
          loopTypeLabels={loopTypeLabels}
          selectedLoopId={selectedLoopId}
          selectedLoopLabel={selectedLoopLabel}
          activeSessionTitle={activeSessionTitle}
          messages={messages}
          inputValue={inputValue}
          streaming={streaming}
          starterPrompts={starterPrompts}
          onLoopChange={onLoopChange}
          onInputChange={onInputChange}
          onAsk={onAsk}
          normalizeAction={normalizeAction}
          onRunAction={(action) => onRunAction(action as AssistantAction)}
        />
      </main>
    </div>
  );
}
