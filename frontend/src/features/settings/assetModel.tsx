import { Space, Tag } from 'antd';
import type { DataNode } from 'antd/es/tree';

export type AssetNodeType = 'factory' | 'department' | 'unit' | 'area' | 'equipment' | 'loop';

export interface AssetNode {
  id: string;
  parentId?: string;
  name: string;
  type: AssetNodeType;
  code?: string;
  description?: string;
}

export function assetTagColor(type: AssetNodeType) {
  if (type === 'factory') return 'blue';
  if (type === 'department') return 'cyan';
  if (type === 'unit') return 'green';
  if (type === 'area') return 'orange';
  if (type === 'equipment') return 'purple';
  return 'default';
}

export function inferLoopAssetId(loopId?: string) {
  if (!loopId) return 'factory';
  if (loopId.startsWith('5203_')) return 'area_2_hydrocrack_fractionation';
  return 'factory';
}

export function buildAssetTreeData(
  nodes: AssetNode[],
  labels: Record<AssetNodeType, string>,
) {
  const childrenByParent = new Map<string | undefined, AssetNode[]>();
  nodes.forEach((node) => {
    const list = childrenByParent.get(node.parentId) ?? [];
    list.push(node);
    childrenByParent.set(node.parentId, list);
  });

  const build = (parentId?: string): DataNode[] =>
    (childrenByParent.get(parentId) ?? []).map((node) => {
      const children = build(node.id);
      return {
        key: node.id,
        title: (
          <Space size={6}>
            <span>{node.name}</span>
            <Tag color={assetTagColor(node.type)}>{labels[node.type]}</Tag>
            {node.code && <Tag color="blue">{node.code}</Tag>}
          </Space>
        ),
        children: children.length ? children : undefined,
      };
    });

  return build(undefined);
}

export function getDescendantAssetIds(nodes: AssetNode[], rootId: string) {
  const ids = new Set<string>([rootId]);
  let changed = true;
  while (changed) {
    changed = false;
    nodes.forEach((node) => {
      if (node.parentId && ids.has(node.parentId) && !ids.has(node.id)) {
        ids.add(node.id);
        changed = true;
      }
    });
  }
  return ids;
}

export function nextAssetType(parentType?: AssetNodeType): AssetNodeType {
  if (parentType === 'factory') return 'department';
  if (parentType === 'department') return 'unit';
  if (parentType === 'unit') return 'area';
  if (parentType === 'area') return 'equipment';
  return 'area';
}
