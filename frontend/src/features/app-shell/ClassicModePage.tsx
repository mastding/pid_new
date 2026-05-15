import type { ReactNode } from 'react';

import { TuningTaskDetailDrawer } from '@/features/tuning-task/TuningTaskDetailDrawer';
import { ClassicSideMenu } from './ClassicSideMenu';
import type { ModuleKey, SubKey } from './navigation';
import { PidAppTopbar } from './PidAppTopbar';

type ViewMode = 'dialogue' | 'classic';

interface ClassicModePageProps {
  sidebarCollapsed: boolean;
  viewMode: ViewMode;
  modules: Array<{
    key: ModuleKey;
    label: string;
    icon: ReactNode;
    subs: Array<{ key: SubKey; label: string; icon: ReactNode; implemented?: boolean }>;
  }>;
  activeModule: ModuleKey;
  activeSub: SubKey;
  expandedModules: Record<ModuleKey, boolean>;
  taskDetailOpen: boolean;
  children: ReactNode;
  taskDashboard: ReactNode;
  onSidebarToggle: () => void;
  onViewModeChange: (mode: ViewMode) => void;
  onToggleModule: (moduleKey: ModuleKey) => void;
  onSelect: (moduleKey: ModuleKey, subKey: SubKey) => void;
  onExpandFromCollapsed: (moduleKey: ModuleKey, firstSubKey: SubKey) => void;
  onTaskDetailClose: () => void;
}

export function ClassicModePage({
  sidebarCollapsed,
  viewMode,
  modules,
  activeModule,
  activeSub,
  expandedModules,
  taskDetailOpen,
  children,
  taskDashboard,
  onSidebarToggle,
  onViewModeChange,
  onToggleModule,
  onSelect,
  onExpandFromCollapsed,
  onTaskDetailClose,
}: ClassicModePageProps) {
  return (
    <div className="agent-console">
      <PidAppTopbar
        sidebarCollapsed={sidebarCollapsed}
        viewMode={viewMode}
        onSidebarToggle={onSidebarToggle}
        onViewModeChange={onViewModeChange}
      />

      <main className={sidebarCollapsed ? 'agent-main industrial-main sidebar-collapsed' : 'agent-main industrial-main'}>
        <ClassicSideMenu
          modules={modules}
          collapsed={sidebarCollapsed}
          activeModule={activeModule}
          activeSub={activeSub}
          expandedModules={expandedModules}
          onToggleModule={(moduleKey) => onToggleModule(moduleKey as ModuleKey)}
          onSelect={(moduleKey, subKey) => onSelect(moduleKey as ModuleKey, subKey as SubKey)}
          onExpandFromCollapsed={(moduleKey, firstSubKey) => onExpandFromCollapsed(moduleKey as ModuleKey, firstSubKey as SubKey)}
        />

        <section className="content-area">
          <div className="industrial-content-shell no-context-rail">
            <div className="primary-workspace">
              {children}
            </div>
          </div>
          <TuningTaskDetailDrawer
            open={taskDetailOpen}
            onClose={onTaskDetailClose}
          >
            {taskDashboard}
          </TuningTaskDetailDrawer>
        </section>
      </main>
    </div>
  );
}
