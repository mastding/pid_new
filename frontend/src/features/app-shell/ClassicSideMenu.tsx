import type { ReactNode } from 'react';
import { DownOutlined, RightOutlined } from '@ant-design/icons';

export interface ClassicSubMenuItem {
  key: string;
  label: string;
  icon: ReactNode;
  implemented?: boolean;
}

export interface ClassicMenuModule {
  key: string;
  label: string;
  icon: ReactNode;
  subs: ClassicSubMenuItem[];
}

interface ClassicSideMenuProps {
  modules: ClassicMenuModule[];
  collapsed: boolean;
  activeModule: string;
  activeSub: string;
  expandedModules: Record<string, boolean>;
  onToggleModule: (moduleKey: string) => void;
  onSelect: (moduleKey: string, subKey: string) => void;
  onExpandFromCollapsed: (moduleKey: string, firstSubKey: string) => void;
}

export function ClassicSideMenu({
  modules,
  collapsed,
  activeModule,
  activeSub,
  expandedModules,
  onToggleModule,
  onSelect,
  onExpandFromCollapsed,
}: ClassicSideMenuProps) {
  return (
    <aside className={collapsed ? 'side-menu industrial-tree collapsed' : 'side-menu industrial-tree'}>
      {modules.map((module) => {
        const expanded = expandedModules[module.key];
        return (
          <div className={expanded ? 'nav-group expanded' : 'nav-group'} key={module.key}>
            <button
              className={module.key === activeModule ? 'nav-group-title active' : 'nav-group-title'}
              title={module.label}
              onClick={() => {
                if (collapsed) {
                  onExpandFromCollapsed(module.key, module.subs[0].key);
                } else {
                  onToggleModule(module.key);
                }
              }}
            >
              {module.icon}
              <span>{module.label}</span>
              <i className="nav-arrow">{expanded ? <DownOutlined /> : <RightOutlined />}</i>
            </button>
            {expanded && !collapsed && (
              <div className="nav-sub-list">
                {module.subs.map((sub) => (
                  <button
                    key={sub.key}
                    type="button"
                    className={sub.key === activeSub ? 'active' : ''}
                    onClick={() => onSelect(module.key, sub.key)}
                  >
                    {sub.icon}
                    <span>{sub.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </aside>
  );
}
