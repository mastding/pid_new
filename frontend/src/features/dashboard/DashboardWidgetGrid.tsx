import { DashboardWidgetFrame } from './DashboardWidgetFrame';
import type { DashboardWidgetDefinition, DashboardWidgetKey } from './model';

interface DashboardWidgetGridProps {
  widgetKeys: DashboardWidgetKey[];
  widgetMap: Partial<Record<DashboardWidgetKey, DashboardWidgetDefinition>>;
  draggedWidgetKey: DashboardWidgetKey | null;
  onDragStart: (widgetKey: DashboardWidgetKey) => void;
  onDrop: (sourceKey: DashboardWidgetKey, targetKey: DashboardWidgetKey) => void;
  onDragEnd: () => void;
  onHide: (widgetKey: DashboardWidgetKey) => void;
}

export function DashboardWidgetGrid({
  widgetKeys,
  widgetMap,
  draggedWidgetKey,
  onDragStart,
  onDrop,
  onDragEnd,
  onHide,
}: DashboardWidgetGridProps) {
  return (
    <section className="cockpit-widget-grid cockpit-widget-grid-adaptive">
      {widgetKeys.map((key) => {
        const widget = widgetMap[key];
        if (!widget) return null;

        return (
          <DashboardWidgetFrame
            key={key}
            widgetKey={key}
            title={widget.title}
            className={widget.className}
            isDragging={draggedWidgetKey === key}
            onDragStart={(widgetKey) => onDragStart(widgetKey as DashboardWidgetKey)}
            onDrop={(source, target) => onDrop(source as DashboardWidgetKey, target as DashboardWidgetKey)}
            onDragEnd={onDragEnd}
            onHide={(widgetKey) => onHide(widgetKey as DashboardWidgetKey)}
          >
            {widget.content}
          </DashboardWidgetFrame>
        );
      })}
    </section>
  );
}
