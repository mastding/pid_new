import { Empty } from 'antd';
import type { FormInstance, UploadFile } from 'antd';
import type { DataNode } from 'antd/es/tree';
import McpConfigPage from '@/pages/settings/McpConfigPage';
import type { HistoryLoop, ModelConfig, PolicyConfig, PromptConfig } from '@/services/api';
import { AssetDirectoryPanel } from '@/features/settings/AssetDirectoryPanel';
import { DataSourcesPanel } from '@/features/settings/DataSourcesPanel';
import { ModelConfigPanel, type ModelConfigTestResult } from '@/features/settings/ModelConfigPanel';
import { PromptConfigPanel } from '@/features/settings/PromptConfigPanel';
import { PROMPT_CONFIG_ITEMS, type PromptConfigField } from '@/features/settings/promptConfigItems';
import { RuleConfigPanel } from '@/features/settings/RuleConfigPanel';
import type { SubKey } from '@/features/app-shell/navigation';

interface SelectOption {
  label: string;
  value: string;
}

interface SettingsModulePageProps {
  activeSub: SubKey;
  assetDraftName: string;
  assetDraftType: string;
  assetNameForLoop: (loop: HistoryLoop) => string;
  assetRenameValue: string;
  assetTreeData: DataNode[];
  assetTypeOptions: SelectOption[];
  currentSubLabel: string;
  dataSourceType: string;
  fileList: UploadFile[];
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  importedLoopCount: number;
  importing: boolean;
  modelConfig: ModelConfig | null;
  modelConfigForm: FormInstance;
  modelConfigLoading: boolean;
  modelConfigSaving: boolean;
  modelConfigTestResult: ModelConfigTestResult | null;
  modelConfigTesting: boolean;
  pathLabel: string;
  policyConfig: PolicyConfig | null;
  policyConfigLoading: boolean;
  promptConfig: PromptConfig | null;
  promptConfigForm: FormInstance;
  promptConfigLoading: boolean;
  promptConfigSaving: boolean;
  activePromptField: PromptConfigField;
  scopedLoopCount: number;
  scopedLoops: HistoryLoop[];
  selectedAssetCode?: string;
  selectedAssetName?: string;
  selectedAssetNodeId: string;
  selectedAssetPathIds: string[];
  selectedAssetTagColor: string;
  selectedAssetTypeLabel: string;
  loopTypeLabel: (loopType: string) => string;
  policyLoopImpact: (loopType: string) => string;
  onAddAssetChild: () => void;
  onAssetDraftNameChange: (value: string) => void;
  onAssetDraftTypeChange: (value: string) => void;
  onAssetRenameValueChange: (value: string) => void;
  onAssetSelect: (nodeId: string) => void;
  onDataSourceTypeChange: (value: string) => void;
  onDeleteAssetNode: () => void;
  onFileListChange: (next: UploadFile[]) => void;
  onImport: () => void;
  onLoadModelConfig: () => void;
  onLoadPolicyConfig: () => void;
  onLoadPromptConfig: () => void;
  onRenameAssetNode: () => void;
  onRestoreDefaultPromptConfig: () => void;
  onSaveModelConfig: (values: Record<string, unknown>) => void;
  onSavePromptConfig: () => void;
  onSetActivePromptField: (value: PromptConfigField) => void;
  onTestModelConnection: () => void;
  onTuneLoop: (loopId: string) => void;
  onViewLoop: (loopId: string) => void;
}

export function SettingsModulePage({
  activeSub,
  activePromptField,
  assetDraftName,
  assetDraftType,
  assetNameForLoop,
  assetRenameValue,
  assetTreeData,
  assetTypeOptions,
  currentSubLabel,
  dataSourceType,
  fileList,
  formatNumber,
  formatPercentValue,
  importedLoopCount,
  importing,
  modelConfig,
  modelConfigForm,
  modelConfigLoading,
  modelConfigSaving,
  modelConfigTestResult,
  modelConfigTesting,
  pathLabel,
  policyConfig,
  policyConfigLoading,
  policyLoopImpact,
  promptConfig,
  promptConfigForm,
  promptConfigLoading,
  promptConfigSaving,
  scopedLoopCount,
  scopedLoops,
  selectedAssetCode,
  selectedAssetName,
  selectedAssetNodeId,
  selectedAssetPathIds,
  selectedAssetTagColor,
  selectedAssetTypeLabel,
  loopTypeLabel,
  onAddAssetChild,
  onAssetDraftNameChange,
  onAssetDraftTypeChange,
  onAssetRenameValueChange,
  onAssetSelect,
  onDataSourceTypeChange,
  onDeleteAssetNode,
  onFileListChange,
  onImport,
  onLoadModelConfig,
  onLoadPolicyConfig,
  onLoadPromptConfig,
  onRenameAssetNode,
  onRestoreDefaultPromptConfig,
  onSaveModelConfig,
  onSavePromptConfig,
  onSetActivePromptField,
  onTestModelConnection,
  onTuneLoop,
  onViewLoop,
}: SettingsModulePageProps) {
  switch (activeSub) {
    case 'asset_directory':
      return (
        <AssetDirectoryPanel
          pathLabel={pathLabel}
          selectedAssetTypeLabel={selectedAssetTypeLabel}
          selectedAssetTagColor={selectedAssetTagColor}
          scopedLoopCount={scopedLoopCount}
          assetTreeData={assetTreeData}
          selectedAssetNodeId={selectedAssetNodeId}
          selectedAssetPathIds={selectedAssetPathIds}
          selectedAssetName={selectedAssetName}
          selectedAssetCode={selectedAssetCode}
          assetDraftName={assetDraftName}
          assetDraftType={assetDraftType}
          assetTypeOptions={assetTypeOptions}
          assetRenameValue={assetRenameValue}
          scopedLoops={scopedLoops}
          onAssetSelect={onAssetSelect}
          onAssetDraftNameChange={onAssetDraftNameChange}
          onAssetDraftTypeChange={onAssetDraftTypeChange}
          onAssetRenameValueChange={onAssetRenameValueChange}
          onAddAssetChild={onAddAssetChild}
          onRenameAssetNode={onRenameAssetNode}
          onDeleteAssetNode={onDeleteAssetNode}
          loopTypeLabel={loopTypeLabel}
          assetNameForLoop={assetNameForLoop}
          onViewLoop={onViewLoop}
          onTuneLoop={onTuneLoop}
        />
      );
    case 'data_sources':
      return (
        <DataSourcesPanel
          dataSourceType={dataSourceType}
          fileList={fileList}
          importedLoopCount={importedLoopCount}
          importing={importing}
          onDataSourceTypeChange={onDataSourceTypeChange}
          onFileListChange={onFileListChange}
          onImport={onImport}
        />
      );
    case 'rule_config':
      return (
        <RuleConfigPanel
          policyConfig={policyConfig}
          loading={policyConfigLoading}
          onRefresh={onLoadPolicyConfig}
          loopTypeLabel={loopTypeLabel}
          policyLoopImpact={policyLoopImpact}
          formatNumber={formatNumber}
          formatPercentValue={formatPercentValue}
        />
      );
    case 'prompt_config':
      return (
        <PromptConfigPanel
          form={promptConfigForm}
          promptConfig={promptConfig}
          promptItems={PROMPT_CONFIG_ITEMS}
          activePromptField={activePromptField}
          loading={promptConfigLoading}
          saving={promptConfigSaving}
          onActivePromptFieldChange={(value) => onSetActivePromptField(value as PromptConfigField)}
          onSave={onSavePromptConfig}
          onRefresh={onLoadPromptConfig}
          onRestoreDefault={onRestoreDefaultPromptConfig}
        />
      );
    case 'model_config':
      return (
        <ModelConfigPanel
          form={modelConfigForm}
          modelConfig={modelConfig}
          testResult={modelConfigTestResult}
          loading={modelConfigLoading}
          saving={modelConfigSaving}
          testing={modelConfigTesting}
          onSave={onSaveModelConfig}
          onTest={onTestModelConnection}
          onRefresh={onLoadModelConfig}
        />
      );
    case 'mcp_config':
      return (
        <div className="page-stack embedded-settings-page">
          <McpConfigPage embedded />
        </div>
      );
    default:
      return (
        <section className="agent-panel">
          <div className="panel-title">{currentSubLabel}</div>
          <Empty description="该页面暂未开放" />
        </section>
      );
  }
}
