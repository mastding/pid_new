import type { DataNode } from 'antd/es/tree';
import { Alert, Button, Descriptions, Divider, Input, Select, Space, Table, Tag, Tree, Typography } from 'antd';

import type { HistoryLoop } from '@/services/api';

interface SelectOption {
  label: string;
  value: string;
}

interface AssetDirectoryPanelProps {
  pathLabel: string;
  selectedAssetTypeLabel: string;
  selectedAssetTagColor: string;
  scopedLoopCount: number;
  assetTreeData: DataNode[];
  selectedAssetNodeId: string;
  selectedAssetPathIds: string[];
  selectedAssetName?: string;
  selectedAssetCode?: string;
  assetDraftName: string;
  assetDraftType: string;
  assetTypeOptions: SelectOption[];
  assetRenameValue: string;
  scopedLoops: HistoryLoop[];
  onAssetSelect: (nodeId: string) => void;
  onAssetDraftNameChange: (value: string) => void;
  onAssetDraftTypeChange: (value: string) => void;
  onAssetRenameValueChange: (value: string) => void;
  onAddAssetChild: () => void;
  onRenameAssetNode: () => void;
  onDeleteAssetNode: () => void;
  loopTypeLabel: (loopType: string) => string;
  assetNameForLoop: (loop: HistoryLoop) => string;
  onViewLoop: (loopId: string) => void;
  onTuneLoop: (loopId: string) => void;
}

export function AssetDirectoryPanel({
  pathLabel,
  selectedAssetTypeLabel,
  selectedAssetTagColor,
  scopedLoopCount,
  assetTreeData,
  selectedAssetNodeId,
  selectedAssetPathIds,
  selectedAssetName,
  selectedAssetCode,
  assetDraftName,
  assetDraftType,
  assetTypeOptions,
  assetRenameValue,
  scopedLoops,
  onAssetSelect,
  onAssetDraftNameChange,
  onAssetDraftTypeChange,
  onAssetRenameValueChange,
  onAddAssetChild,
  onRenameAssetNode,
  onDeleteAssetNode,
  loopTypeLabel,
  assetNameForLoop,
  onViewLoop,
  onTuneLoop,
}: AssetDirectoryPanelProps) {
  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">装置资产目录</div>
            <Typography.Text type="secondary">
              当前作用域：{pathLabel}
            </Typography.Text>
          </div>
          <Space wrap>
            <Tag color={selectedAssetTagColor}>
              {selectedAssetTypeLabel}
            </Tag>
            <Tag color="blue">{scopedLoopCount} 个回路</Tag>
          </Space>
        </div>
        <div className="asset-directory-grid">
          <div className="asset-tree-panel">
            <Tree
              treeData={assetTreeData}
              selectedKeys={[selectedAssetNodeId]}
              defaultExpandedKeys={selectedAssetPathIds}
              onSelect={(keys) => onAssetSelect(String(keys[0] ?? selectedAssetNodeId))}
            />
          </div>
          <div className="asset-editor-panel">
            <Descriptions bordered column={2} size="small" className="industrial-descriptions">
              <Descriptions.Item label="节点名称">{selectedAssetName ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="节点类型">{selectedAssetTypeLabel}</Descriptions.Item>
              <Descriptions.Item label="节点编码">{selectedAssetCode ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="挂载回路">{scopedLoopCount} 个</Descriptions.Item>
              <Descriptions.Item label="路径" span={2}>{pathLabel}</Descriptions.Item>
            </Descriptions>

            <Divider orientation="left">新增子节点</Divider>
            <div className="asset-edit-row">
              <Input
                value={assetDraftName}
                placeholder="例如：反应系统、分馏系统、V-0202回流罐"
                onChange={(event) => onAssetDraftNameChange(event.target.value)}
              />
              <Select
                value={assetDraftType}
                onChange={onAssetDraftTypeChange}
                options={assetTypeOptions}
              />
              <Button type="primary" onClick={onAddAssetChild}>新增</Button>
            </div>

            <Divider orientation="left">编辑当前节点</Divider>
            <div className="asset-edit-row">
              <Input
                value={assetRenameValue}
                placeholder={selectedAssetName ?? '节点名称'}
                onChange={(event) => onAssetRenameValueChange(event.target.value)}
              />
              <Button onClick={onRenameAssetNode}>重命名</Button>
              <Button danger onClick={onDeleteAssetNode}>删除空节点</Button>
            </div>

            <Alert
              className="agent-alert"
              type="info"
              showIcon
              message="第一版为前端本地目录"
              description="当前目录结构仅保存在前端本地。"
            />
          </div>
        </div>
      </section>

      <section className="agent-panel">
        <div className="panel-title">当前作用域回路</div>
        <Table<HistoryLoop>
          size="small"
          pagination={false}
          rowKey="loop_id"
          dataSource={scopedLoops}
          columns={[
            { title: '回路位号', dataIndex: 'loop_id' },
            { title: '类型', dataIndex: 'loop_type', render: (value: string) => loopTypeLabel(value) },
            { title: '归属节点', render: (_: unknown, row) => assetNameForLoop(row) },
            { title: '数据点', dataIndex: 'rows' },
            {
              title: '操作',
              render: (_: unknown, row) => (
                <Space>
                  <Button size="small" onClick={() => onViewLoop(row.loop_id)}>画像</Button>
                  <Button size="small" onClick={() => onTuneLoop(row.loop_id)}>整定</Button>
                </Space>
              ),
            },
          ]}
        />
      </section>
    </div>
  );
}
