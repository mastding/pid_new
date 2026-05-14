import type { ReactNode } from 'react';
import { Drawer } from 'antd';

interface TuningTaskDetailDrawerProps {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
}

export function TuningTaskDetailDrawer({ open, onClose, children }: TuningTaskDetailDrawerProps) {
  return (
    <Drawer
      title="整定任务全流程详情"
      width="min(1180px, 92vw)"
      open={open}
      onClose={onClose}
      className="industrial-drawer"
    >
      {children}
    </Drawer>
  );
}
