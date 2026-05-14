import type { DragEvent, ReactNode } from 'react';
import { Button, Tooltip } from 'antd';
import { DeleteOutlined, MenuOutlined } from '@ant-design/icons';

interface DashboardWidgetFrameProps {
  widgetKey: string;
  title: string;
  className: string;
  isDragging: boolean;
  children: ReactNode;
  onDragStart: (widgetKey: string) => void;
  onDrop: (sourceKey: string, targetKey: string) => void;
  onDragEnd: () => void;
  onHide: (widgetKey: string) => void;
}

export function DashboardWidgetFrame({
  widgetKey,
  title,
  className,
  isDragging,
  children,
  onDragStart,
  onDrop,
  onDragEnd,
  onHide,
}: DashboardWidgetFrameProps) {
  const handleDragStart = (event: DragEvent<HTMLDivElement>) => {
    onDragStart(widgetKey);
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('text/plain', widgetKey);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const source = event.dataTransfer.getData('text/plain');
    if (source) onDrop(source, widgetKey);
    onDragEnd();
  };

  return (
    <div
      className={`dashboard-widget ${className}${isDragging ? ' dashboard-widget-dragging' : ''}`}
      draggable
      onDragStart={handleDragStart}
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      onDragEnd={onDragEnd}
    >
      <div className="dashboard-widget-actions" draggable={false}>
        <Tooltip title="拖动调整位置">
          <span className="dashboard-drag-handle"><MenuOutlined /></span>
        </Tooltip>
        <Tooltip title={`移除${title}`}>
          <Button
            type="text"
            size="small"
            icon={<DeleteOutlined />}
            onClick={(event) => {
              event.stopPropagation();
              onHide(widgetKey);
            }}
          />
        </Tooltip>
      </div>
      {children}
    </div>
  );
}
