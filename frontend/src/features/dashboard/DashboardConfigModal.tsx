import { Button, Checkbox, Modal, Space, Typography, message } from 'antd';
import {
  DASHBOARD_WIDGET_KEY_SET,
  DASHBOARD_WIDGET_OPTIONS,
  DEFAULT_DASHBOARD_WIDGET_KEYS,
  type DashboardWidgetKey,
} from './model';

interface DashboardConfigModalProps {
  open: boolean;
  widgetKeys: DashboardWidgetKey[];
  onChange: (widgetKeys: DashboardWidgetKey[]) => void;
  onClose: () => void;
}

export function DashboardConfigModal({
  open,
  widgetKeys,
  onChange,
  onClose,
}: DashboardConfigModalProps) {
  return (
    <Modal
      title="自定义看板"
      open={open}
      onCancel={onClose}
      footer={(
        <Space>
          <Button onClick={() => onChange(DEFAULT_DASHBOARD_WIDGET_KEYS)}>恢复默认</Button>
          <Button type="primary" onClick={onClose}>完成</Button>
        </Space>
      )}
    >
      <Typography.Paragraph type="secondary">
        选择首页驾驶舱需要展示的模块，系统会记住本机偏好。
      </Typography.Paragraph>
      <Checkbox.Group
        className="dashboard-widget-picker"
        value={widgetKeys}
        options={DASHBOARD_WIDGET_OPTIONS}
        onChange={(values) => {
          const next = values.filter((item): item is DashboardWidgetKey => DASHBOARD_WIDGET_KEY_SET.has(item as DashboardWidgetKey));
          if (!next.length) {
            message.warning('至少保留一个看板模块');
            return;
          }
          onChange(next);
        }}
      />
    </Modal>
  );
}
