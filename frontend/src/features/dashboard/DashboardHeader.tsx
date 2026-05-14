import { Button, Space, Tag } from 'antd';

interface DashboardHeaderProps {
  assetTypeLabel: string;
  assetTagColor: string;
  pathLabel: string;
  onOpenConfig: () => void;
  onSwitchAsset: () => void;
}

export function DashboardHeader({
  assetTypeLabel,
  assetTagColor,
  pathLabel,
  onOpenConfig,
  onSwitchAsset,
}: DashboardHeaderProps) {
  return (
    <section className="cockpit-header">
      <div>
        <h2>首页驾驶舱</h2>
      </div>
      <Space wrap>
        <Tag color={assetTagColor}>{assetTypeLabel}</Tag>
        <span>范围：{pathLabel}</span>
        <Button size="small" onClick={onOpenConfig}>自定义看板</Button>
        <Button size="small" onClick={onSwitchAsset}>切换装置</Button>
      </Space>
    </section>
  );
}
