import { useCallback, useMemo, useState } from 'react';

import { INITIAL_EXPANDED_MODULES, MODULES, type ModuleKey, type SubKey } from './navigation';

export type ViewMode = 'dialogue' | 'classic';

export function useAppShellState() {
  const [activeModule, setActiveModule] = useState<ModuleKey>('workspace');
  const [activeSub, setActiveSub] = useState<SubKey>('dashboard');
  const [viewMode, setViewMode] = useState<ViewMode>('dialogue');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [expandedModules, setExpandedModules] = useState<Record<ModuleKey, boolean>>(INITIAL_EXPANDED_MODULES);

  const currentModule = useMemo(
    () => MODULES.find((item) => item.key === activeModule) ?? MODULES[0],
    [activeModule],
  );

  const currentSub = useMemo(
    () => currentModule.subs.find((item) => item.key === activeSub) ?? currentModule.subs[0],
    [activeSub, currentModule],
  );

  const switchTo = useCallback((moduleKey: ModuleKey, subKey: SubKey) => {
    setActiveModule(moduleKey);
    setActiveSub(subKey);
    setExpandedModules((prev) => ({ ...prev, [moduleKey]: true }));
  }, []);

  const toggleModule = useCallback((moduleKey: ModuleKey) => {
    setExpandedModules((prev) => ({ ...prev, [moduleKey]: !prev[moduleKey] }));
  }, []);

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((value) => !value);
  }, []);

  return {
    activeModule,
    activeSub,
    viewMode,
    setViewMode,
    sidebarCollapsed,
    setSidebarCollapsed,
    expandedModules,
    currentModule,
    currentSub,
    switchTo,
    toggleModule,
    toggleSidebar,
  };
}
