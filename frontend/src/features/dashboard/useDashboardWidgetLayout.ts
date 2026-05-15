import { useCallback, useEffect, useMemo, useState } from 'react';
import { message } from 'antd';

import {
  DASHBOARD_WIDGET_STORAGE_KEY,
  DEFAULT_DASHBOARD_WIDGET_KEYS,
  normalizeDashboardWidgetKeys,
  type DashboardWidgetKey,
} from './model';

function loadDashboardWidgetKeys() {
  if (typeof window === 'undefined') return DEFAULT_DASHBOARD_WIDGET_KEYS;
  try {
    const raw = window.localStorage.getItem(DASHBOARD_WIDGET_STORAGE_KEY);
    return normalizeDashboardWidgetKeys(raw ? JSON.parse(raw) : []);
  } catch {
    return DEFAULT_DASHBOARD_WIDGET_KEYS;
  }
}

export function useDashboardWidgetLayout() {
  const [dashboardWidgetKeys, setDashboardWidgetKeys] = useState<DashboardWidgetKey[]>(loadDashboardWidgetKeys);
  const [draggedDashboardWidgetKey, setDraggedDashboardWidgetKey] = useState<DashboardWidgetKey | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(DASHBOARD_WIDGET_STORAGE_KEY, JSON.stringify(dashboardWidgetKeys));
    } catch {
      // Dashboard layout is an optional UI preference.
    }
  }, [dashboardWidgetKeys]);

  const enabledDashboardWidgets = useMemo(
    () => new Set(dashboardWidgetKeys),
    [dashboardWidgetKeys],
  );

  const hideDashboardWidget = useCallback((key: DashboardWidgetKey) => {
    setDashboardWidgetKeys((prev) => {
      if (prev.length <= 1) {
        message.warning('至少保留一个看板模块');
        return prev;
      }
      return prev.filter((item) => item !== key);
    });
  }, []);

  const moveDashboardWidget = useCallback((source: DashboardWidgetKey, target: DashboardWidgetKey) => {
    if (source === target) return;
    setDashboardWidgetKeys((prev) => {
      const from = prev.indexOf(source);
      const to = prev.indexOf(target);
      if (from < 0 || to < 0) return prev;
      const next = [...prev];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
  }, []);

  return {
    dashboardWidgetKeys,
    setDashboardWidgetKeys,
    enabledDashboardWidgets,
    draggedDashboardWidgetKey,
    setDraggedDashboardWidgetKey,
    hideDashboardWidget,
    moveDashboardWidget,
  };
}
