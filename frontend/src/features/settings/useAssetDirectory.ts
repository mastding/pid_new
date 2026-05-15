import { useCallback, useMemo, useState } from 'react';
import { message } from 'antd';

import type { HistoryLoop } from '@/services/api';
import {
  ASSET_TYPE_LABEL,
  DEFAULT_ASSET_NODES,
  assetTagColor,
  buildAssetTreeData,
  getAssetPath,
  getScopedLoops,
  inferLoopAssetId,
  nextAssetType,
  type AssetNode,
  type AssetNodeType,
} from './assetModel';

export function useAssetDirectory(loops: HistoryLoop[]) {
  const [assetNodes, setAssetNodes] = useState<AssetNode[]>(DEFAULT_ASSET_NODES);
  const [selectedAssetNodeId, setSelectedAssetNodeId] = useState<string>('unit_2_hydrocrack');
  const [assetDraftName, setAssetDraftName] = useState('');
  const [assetDraftType, setAssetDraftType] = useState<AssetNodeType>('area');
  const [assetRenameValue, setAssetRenameValue] = useState('');

  const selectedAssetNode = useMemo(
    () => assetNodes.find((item) => item.id === selectedAssetNodeId) ?? assetNodes[0],
    [assetNodes, selectedAssetNodeId],
  );

  const selectedAssetPath = useMemo(() => {
    return getAssetPath(assetNodes, selectedAssetNode);
  }, [assetNodes, selectedAssetNode]);

  const scopedLoops = useMemo(() => {
    return getScopedLoops(assetNodes, loops, selectedAssetNodeId);
  }, [assetNodes, loops, selectedAssetNodeId]);

  const assetTreeData = useMemo(() => buildAssetTreeData(assetNodes, ASSET_TYPE_LABEL), [assetNodes]);

  const scopedLoopStats = useMemo(() => ({
    loopCount: scopedLoops.length,
  }), [scopedLoops]);

  const pathLabel = useMemo(
    () => selectedAssetPath.map((item) => item.name).join(' / '),
    [selectedAssetPath],
  );

  const assetTypeOptions = useMemo(
    () => (Object.keys(ASSET_TYPE_LABEL) as AssetNodeType[]).map((type) => ({
      label: ASSET_TYPE_LABEL[type],
      value: type,
    })),
    [],
  );

  const assetNameForLoop = useCallback((loop: HistoryLoop, fallback = '-') => {
    return assetNodes.find((node) => node.id === inferLoopAssetId(loop.loop_id))?.name ?? fallback;
  }, [assetNodes]);

  const selectAssetNode = useCallback((nodeId: string) => {
    setSelectedAssetNodeId(nodeId);
    setAssetRenameValue(assetNodes.find((item) => item.id === nodeId)?.name ?? '');
  }, [assetNodes]);

  const changeAssetDraftType = useCallback((value: string) => {
    setAssetDraftType(value as AssetNodeType);
  }, []);

  const addAssetChild = useCallback(() => {
    const parent = selectedAssetNode;
    const name = assetDraftName.trim();
    if (!parent || !name) {
      message.warning('请先选择父节点并输入节点名称');
      return;
    }
    const node: AssetNode = {
      id: `asset_${Date.now()}`,
      parentId: parent.id,
      name,
      type: assetDraftType || nextAssetType(parent.type),
    };
    setAssetNodes((prev) => [...prev, node]);
    setSelectedAssetNodeId(node.id);
    setAssetDraftName('');
    message.success(`已新增节点：${name}`);
  }, [assetDraftName, assetDraftType, selectedAssetNode]);

  const renameAssetNode = useCallback(() => {
    const name = assetRenameValue.trim();
    if (!selectedAssetNode || !name) {
      message.warning('请输入新的节点名称');
      return;
    }
    setAssetNodes((prev) => prev.map((node) => (
      node.id === selectedAssetNode.id ? { ...node, name } : node
    )));
    setAssetRenameValue('');
    message.success('节点已重命名');
  }, [assetRenameValue, selectedAssetNode]);

  const deleteAssetNode = useCallback(() => {
    if (!selectedAssetNode || selectedAssetNode.id === 'factory') {
      message.warning('根节点不能删除');
      return;
    }
    const hasChild = assetNodes.some((node) => node.parentId === selectedAssetNode.id);
    const hasLoop = loops.some((loop) => inferLoopAssetId(loop.loop_id) === selectedAssetNode.id);
    if (hasChild || hasLoop) {
      message.warning('该节点存在子节点或挂载回路，第一版请先清空后再删除');
      return;
    }
    setAssetNodes((prev) => prev.filter((node) => node.id !== selectedAssetNode.id));
    setSelectedAssetNodeId(selectedAssetNode.parentId ?? 'factory');
    message.success('节点已删除');
  }, [assetNodes, loops, selectedAssetNode]);

  return {
    assetNodes,
    selectedAssetNode,
    selectedAssetNodeId,
    selectedAssetPath,
    selectedAssetPathIds: selectedAssetPath.map((item) => item.id),
    pathLabel,
    selectedAssetTypeLabel: selectedAssetNode ? ASSET_TYPE_LABEL[selectedAssetNode.type] : '-',
    selectedAssetTagColor: assetTagColor(selectedAssetNode?.type ?? 'factory'),
    scopedLoops,
    scopedLoopStats,
    assetTreeData,
    assetDraftName,
    assetDraftType,
    assetTypeOptions,
    assetRenameValue,
    setAssetDraftName,
    setAssetDraftType: changeAssetDraftType,
    setAssetRenameValue,
    selectAssetNode,
    addAssetChild,
    renameAssetNode,
    deleteAssetNode,
    assetNameForLoop,
  };
}
