import { useState } from 'react';
import type { FormInstance } from 'antd';

import { PidAppTopbar } from '@/features/app-shell/PidAppTopbar';
import type { AssistantSessionSummary, HistoryLoop, ModelConfig, PromptConfig } from '@/services/api';
import type { ModelConfigTestResult } from '@/features/settings/ModelConfigPanel';
import type { PromptConfigField } from '@/features/settings/promptConfigItems';
import { DialogueChatPanel } from './DialogueChatPanel';
import { DialogueHistoryPanel } from './DialogueHistoryPanel';
import { DialogueSettingsModal } from './DialogueSettingsModal';
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
  modelConfig: ModelConfig | null;
  modelConfigForm: FormInstance;
  modelConfigLoading: boolean;
  modelConfigSaving: boolean;
  modelConfigTesting: boolean;
  modelConfigTestResult: ModelConfigTestResult | null;
  promptConfig: PromptConfig | null;
  promptConfigForm: FormInstance;
  promptConfigLoading: boolean;
  promptConfigSaving: boolean;
  activePromptField: PromptConfigField;
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
  onLoadModelConfig: () => void;
  onSaveModelConfig: (values: Record<string, unknown>) => void;
  onTestModelConnection: () => void;
  onLoadPromptConfig: () => void;
  onRestoreDefaultPromptConfig: () => void;
  onSavePromptConfig: () => void;
  onSetActivePromptField: (value: PromptConfigField) => void;
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
  modelConfig,
  modelConfigForm,
  modelConfigLoading,
  modelConfigSaving,
  modelConfigTesting,
  modelConfigTestResult,
  promptConfig,
  promptConfigForm,
  promptConfigLoading,
  promptConfigSaving,
  activePromptField,
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
  onLoadModelConfig,
  onSaveModelConfig,
  onTestModelConnection,
  onLoadPromptConfig,
  onRestoreDefaultPromptConfig,
  onSavePromptConfig,
  onSetActivePromptField,
  normalizeAction,
  onRunAction,
}: DialogueModePageProps) {
  const [settingsOpen, setSettingsOpen] = useState(false);

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
          onOpenSettings={() => setSettingsOpen(true)}
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

      <DialogueSettingsModal
        open={settingsOpen}
        modelConfig={modelConfig}
        modelConfigForm={modelConfigForm}
        modelConfigLoading={modelConfigLoading}
        modelConfigSaving={modelConfigSaving}
        modelConfigTesting={modelConfigTesting}
        modelConfigTestResult={modelConfigTestResult}
        promptConfig={promptConfig}
        promptConfigForm={promptConfigForm}
        promptConfigLoading={promptConfigLoading}
        promptConfigSaving={promptConfigSaving}
        activePromptField={activePromptField}
        onClose={() => setSettingsOpen(false)}
        onLoadModelConfig={onLoadModelConfig}
        onSaveModelConfig={onSaveModelConfig}
        onTestModelConnection={onTestModelConnection}
        onLoadPromptConfig={onLoadPromptConfig}
        onRestoreDefaultPromptConfig={onRestoreDefaultPromptConfig}
        onSavePromptConfig={onSavePromptConfig}
        onSetActivePromptField={onSetActivePromptField}
      />
    </div>
  );
}
