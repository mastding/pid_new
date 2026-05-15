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

export const ASSET_TYPE_LABEL: Record<AssetNodeType, string> = {
  factory: '厂区',
  department: '运行部',
  unit: '装置',
  area: '区域/系统',
  equipment: '设备',
  loop: '回路',
};

export const DEFAULT_ASSET_NODES: AssetNode[] = [
  { id: 'factory', name: '石化工厂', type: 'factory', description: 'PID 智能整定资产根目录' },
  { id: 'dept_run_1', parentId: 'factory', name: '运行一部', type: 'department' },
  { id: 'unit_rongtuo', parentId: 'dept_run_1', name: '溶脱', type: 'unit' },
  { id: 'unit_ii_chang', parentId: 'dept_run_1', name: 'II常', type: 'unit' },
  { id: 'dept_run_2', parentId: 'factory', name: '运行二部', type: 'department' },
  { id: 'unit_1_jubingxi', parentId: 'dept_run_2', name: '1#聚丙烯', type: 'unit' },
  { id: 'unit_2_jubingxi', parentId: 'dept_run_2', name: '2#聚丙烯', type: 'unit' },
  { id: 'unit_3_jubingxi', parentId: 'dept_run_2', name: '3#聚丙烯', type: 'unit' },
  { id: 'dept_run_3', parentId: 'factory', name: '运行三部', type: 'department' },
  { id: 'unit_2_liuhuang', parentId: 'dept_run_3', name: '2#硫磺', type: 'unit' },
  { id: 'unit_2_dcc', parentId: 'dept_run_3', name: '2#DCC', type: 'unit' },
  { id: 'unit_xiqing', parentId: 'dept_run_3', name: '烯烃分离', type: 'unit' },
  { id: 'dept_run_45', parentId: 'factory', name: '运行四五部', type: 'department' },
  { id: 'unit_iii_chang', parentId: 'dept_run_45', name: 'III常', type: 'unit' },
  { id: 'unit_hangmei_hydrogen', parentId: 'dept_run_45', name: '航煤加氢', type: 'unit' },
  { id: 'unit_5203_hcu_2', parentId: 'dept_run_45', name: '2#石脑油加氢', type: 'unit', code: '5203' },
  { id: 'unit_solvent_recycle', parentId: 'dept_run_45', name: '2#溶剂再生/酸性水汽提', type: 'unit' },
  { id: 'unit_2_hydrocrack', parentId: 'dept_run_45', name: '2#加裂', type: 'unit', code: '5203' },
  { id: 'area_2_hydrocrack_fractionation', parentId: 'unit_2_hydrocrack', name: '分馏/回流系统', type: 'area' },
  { id: 'unit_light_recovery', parentId: 'dept_run_45', name: '轻烃回收', type: 'unit' },
  { id: 'unit_2_wax_hydrogen', parentId: 'dept_run_45', name: '2#蜡加', type: 'unit' },
  { id: 'unit_diesel_hydrogen', parentId: 'dept_run_45', name: '裂柴加氢', type: 'unit' },
  { id: 'dept_run_6', parentId: 'factory', name: '运行六部', type: 'department' },
  { id: 'unit_1_dingxi', parentId: 'dept_run_6', name: '1-丁烯', type: 'unit' },
  { id: 'unit_mtbe', parentId: 'dept_run_6', name: 'MTBE', type: 'unit' },
  { id: 'unit_1_dcc_regen', parentId: 'dept_run_6', name: '1#DCC反再', type: 'unit' },
  { id: 'unit_1_dcc_fraction', parentId: 'dept_run_6', name: '1#DCC分馏', type: 'unit' },
  { id: 'unit_product_gas', parentId: 'dept_run_6', name: '产品精制气分', type: 'unit' },
  { id: 'dept_run_7', parentId: 'factory', name: '运行七部', type: 'department' },
  { id: 'unit_1_wax_hydrogen', parentId: 'dept_run_7', name: '1#蜡加', type: 'unit' },
  { id: 'unit_1_hydrocrack', parentId: 'dept_run_7', name: '1#加裂', type: 'unit' },
  { id: 'dept_run_8', parentId: 'factory', name: '运行八部', type: 'department' },
  { id: 'unit_1_reforming', parentId: 'dept_run_8', name: '1#重整', type: 'unit' },
  { id: 'unit_hydrogen', parentId: 'dept_run_8', name: '制氢', type: 'unit' },
  { id: 'unit_1_naphtha_hydrogen', parentId: 'dept_run_8', name: '1#石脑油加氢', type: 'unit' },
  { id: 'unit_2_extraction', parentId: 'dept_run_8', name: '2#抽提', type: 'unit' },
  { id: 'unit_1_extraction', parentId: 'dept_run_8', name: '1#抽提', type: 'unit' },
  { id: 'unit_aromatics', parentId: 'dept_run_8', name: '芳构化', type: 'unit' },
  { id: 'dept_run_9', parentId: 'factory', name: '运行九部', type: 'department' },
  { id: 'unit_benzene_ethylene', parentId: 'dept_run_9', name: '苯乙烯', type: 'unit' },
  { id: 'unit_1_sulfur', parentId: 'dept_run_9', name: '1#硫磺', type: 'unit' },
  { id: 'unit_ethylbenzene', parentId: 'dept_run_9', name: '乙苯', type: 'unit' },
  { id: 'dept_run_10', parentId: 'factory', name: '运行十部', type: 'department' },
  { id: 'unit_aromatics_extract', parentId: 'dept_run_10', name: '芳烃', type: 'unit' },
  { id: 'unit_disproportionation', parentId: 'dept_run_10', name: '歧化', type: 'unit' },
  { id: 'dept_run_11', parentId: 'factory', name: '运行十一部', type: 'department' },
  { id: 'unit_2_reforming', parentId: 'dept_run_11', name: '2#重整', type: 'unit' },
  { id: 'unit_3_extraction', parentId: 'dept_run_11', name: '3#抽提', type: 'unit' },
  { id: 'unit_4_extraction', parentId: 'dept_run_11', name: '4#抽提', type: 'unit' },
  { id: 'dept_storage', parentId: 'factory', name: '储运部', type: 'department' },
  { id: 'unit_tank_area_1_2', parentId: 'dept_storage', name: '一期二期罐区', type: 'unit' },
  { id: 'unit_tank_area_3', parentId: 'dept_storage', name: '三期罐区', type: 'unit' },
  { id: 'unit_wharf', parentId: 'dept_storage', name: '码头', type: 'unit' },
  { id: 'dept_utility', parentId: 'factory', name: '公用工程一部', type: 'department' },
  { id: 'unit_wastewater', parentId: 'dept_utility', name: '污水处理场', type: 'unit' },
  { id: 'unit_desalt', parentId: 'dept_utility', name: '除盐水站', type: 'unit' },
  { id: 'unit_circulating_water', parentId: 'dept_utility', name: '循环水场', type: 'unit' },
];

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
