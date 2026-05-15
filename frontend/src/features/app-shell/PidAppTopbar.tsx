import {
  AppstoreOutlined,
  BellOutlined,
  ClockCircleOutlined,
  MenuOutlined,
  RobotOutlined,
  UserOutlined,
} from '@ant-design/icons';

export type PidViewMode = 'dialogue' | 'classic';

interface PidAppTopbarProps {
  sidebarCollapsed: boolean;
  viewMode: PidViewMode;
  onSidebarToggle: () => void;
  onViewModeChange: (mode: PidViewMode) => void;
}

function ModeSwitch({
  viewMode,
  className = '',
  onViewModeChange,
}: {
  viewMode: PidViewMode;
  className?: string;
  onViewModeChange: (mode: PidViewMode) => void;
}) {
  return (
    <div className={`mode-switch ${className}`}>
      <button
        type="button"
        title="对话模式"
        aria-label="对话模式"
        className={viewMode === 'dialogue' ? 'active' : ''}
        onClick={() => onViewModeChange('dialogue')}
      >
        <RobotOutlined />
      </button>
      <button
        type="button"
        title="经典模式"
        aria-label="经典模式"
        className={viewMode === 'classic' ? 'active' : ''}
        onClick={() => onViewModeChange('classic')}
      >
        <AppstoreOutlined />
      </button>
    </div>
  );
}

export function PidAppTopbar({
  sidebarCollapsed,
  viewMode,
  onSidebarToggle,
  onViewModeChange,
}: PidAppTopbarProps) {
  return (
    <header className="pid-app-header">
      <div className="pid-app-topbar">
        <div className="pid-app-brand">
          <button
            type="button"
            className="menu-trigger"
            aria-label={sidebarCollapsed ? '展开导航菜单' : '折叠导航菜单'}
            onClick={onSidebarToggle}
          >
            <MenuOutlined />
          </button>
          <div className="brand-mark">PID</div>
          <div>
            <h1>智能PID控制系统平台</h1>
          </div>
          <ModeSwitch
            viewMode={viewMode}
            className="classic-mode-switch"
            onViewModeChange={onViewModeChange}
          />
        </div>
        <div className="system-meta">
          <span style={{ color: '#1d4ed8', fontWeight: 800 }}>V1.0</span>
          <span><ClockCircleOutlined /> {new Date().toLocaleString()}</span>
          <span><UserOutlined /> admin</span>
          <span className="alarm-pill"><BellOutlined /> 6</span>
        </div>
      </div>
    </header>
  );
}
