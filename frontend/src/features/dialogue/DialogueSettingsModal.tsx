import { useEffect, useMemo, useState } from 'react';
import {
  ApiOutlined,
  FunctionOutlined,
  ReloadOutlined,
  RobotOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { Button, Empty, Modal, Spin, Tabs, Tag, Tooltip, message } from 'antd';
import type { FormInstance } from 'antd';

import McpConfigPage from '@/pages/settings/McpConfigPage';
import { listSkills, type ModelConfig, type PromptConfig, type SkillMetadataInfo } from '@/services/api';
import { ModelConfigPanel, type ModelConfigTestResult } from '@/features/settings/ModelConfigPanel';
import { PromptConfigPanel } from '@/features/settings/PromptConfigPanel';
import { PROMPT_CONFIG_ITEMS, type PromptConfigField } from '@/features/settings/promptConfigItems';

interface DialogueSettingsModalProps {
  open: boolean;
  modelConfig: ModelConfig | null;
  modelConfigForm: FormInstance;
  modelConfigLoading: boolean;
  modelConfigSaving: boolean;
  modelConfigTesting: boolean;
  modelConfigTestResult: ModelConfigTestResult | null;
  promptConfig: PromptConfig | null;
  promptConfigForm: FormInstance;
  promptConfigLoading: boolean;
  promptConfigSaving: boolean;
  activePromptField: PromptConfigField;
  onClose: () => void;
  onLoadModelConfig: () => void;
  onSaveModelConfig: (values: Record<string, unknown>) => void;
  onTestModelConnection: () => void;
  onLoadPromptConfig: () => void;
  onRestoreDefaultPromptConfig: () => void;
  onSavePromptConfig: () => void;
  onSetActivePromptField: (value: PromptConfigField) => void;
}

const riskColor: Record<string, string> = {
  low: 'green',
  medium: 'gold',
  high: 'red',
};

const riskLabel: Record<string, string> = {
  low: '低风险',
  medium: '中风险',
  high: '高风险',
};

const stageLabel: Record<string, string> = {
  data_analysis: '数据理解',
  window_selection: '窗口选择',
  identification: '模型辨识',
  model_review: '模型评审',
  tuning: 'PID 整定',
  evaluation: '仿真评估',
  performance_assessment: '性能评估',
  monitoring: '监控评估',
  assessment: '综合评估',
  ontology_policy: '本体策略',
  general: '通用',
};

function renderEffectText(skill: SkillMetadataInfo) {
  if (!skill.effects?.length) return '无显式状态写入';
  return skill.effects
    .map((item) => item.key || item.description)
    .filter(Boolean)
    .join('、');
}

function SkillCatalogPanel({ open }: { open: boolean }) {
  const [skills, setSkills] = useState<SkillMetadataInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const data = await listSkills();
      setSkills(data.items);
    } catch (error) {
      message.error(`加载 Skill 列表失败：${String(error)}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open && !skills.length) load();
  }, [open]);

  const grouped = useMemo(() => {
    const bucket = new Map<string, SkillMetadataInfo[]>();
    skills.forEach((skill) => {
      const stage = skill.stage || 'general';
      bucket.set(stage, [...(bucket.get(stage) ?? []), skill]);
    });
    return Array.from(bucket.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [skills]);

  return (
    <div className="dialogue-settings-skills">
      <div className="dialogue-settings-toolbar">
        <div>
          <strong>当前项目 Skills</strong>
          <span>{skills.length} 个已注册能力，包含核心流程与外部扩展。</span>
        </div>
        <Tooltip title="刷新 Skill 元数据">
          <Button icon={<ReloadOutlined />} loading={loading} onClick={load} />
        </Tooltip>
      </div>

      {loading && !skills.length ? (
        <div className="dialogue-settings-loading"><Spin /></div>
      ) : grouped.length ? (
        <div className="skill-stage-list">
          {grouped.map(([stage, items]) => (
            <section className="skill-stage-section" key={stage}>
              <div className="skill-stage-title">
                <span>{stageLabel[stage] ?? stage}</span>
                <Tag>{items.length}</Tag>
              </div>
              <div className="skill-card-grid">
                {items.map((skill) => (
                  <article className="skill-card" key={skill.name}>
                    <div className="skill-card-head">
                      <strong>{skill.name}</strong>
                      <Tag color={riskColor[skill.risk_level] ?? 'default'}>
                        {riskLabel[skill.risk_level] ?? skill.risk_level}
                      </Tag>
                    </div>
                    <p>{skill.description || '暂无描述'}</p>
                    <div className="skill-card-meta">
                      <span>前置：{skill.preconditions?.length ? skill.preconditions.join('、') : '无'}</span>
                      <span>写入：{renderEffectText(skill)}</span>
                    </div>
                    {skill.deterministic_gate && <Tag color="blue">确定性门控</Tag>}
                  </article>
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <Empty description="暂无 Skill 元数据" />
      )}
    </div>
  );
}

export function DialogueSettingsModal({
  open,
  modelConfig,
  modelConfigForm,
  modelConfigLoading,
  modelConfigSaving,
  modelConfigTesting,
  modelConfigTestResult,
  promptConfig,
  promptConfigForm,
  promptConfigLoading,
  promptConfigSaving,
  activePromptField,
  onClose,
  onLoadModelConfig,
  onSaveModelConfig,
  onTestModelConnection,
  onLoadPromptConfig,
  onRestoreDefaultPromptConfig,
  onSavePromptConfig,
  onSetActivePromptField,
}: DialogueSettingsModalProps) {
  useEffect(() => {
    if (open && !promptConfig) onLoadPromptConfig();
    if (open && !modelConfig) onLoadModelConfig();
  }, [open, promptConfig, modelConfig, onLoadPromptConfig, onLoadModelConfig]);

  return (
    <Modal
      title={
        <span className="dialogue-settings-title">
          <SettingOutlined />
          设置
        </span>
      }
      open={open}
      onCancel={onClose}
      footer={null}
      width={1040}
      className="dialogue-settings-modal"
      destroyOnClose={false}
    >
      <Tabs
        className="dialogue-settings-tabs"
        defaultActiveKey="skills"
        items={[
          {
            key: 'skills',
            label: <span><FunctionOutlined /> Skills</span>,
            children: <SkillCatalogPanel open={open} />,
          },
          {
            key: 'prompt',
            label: <span><SettingOutlined /> 提示词</span>,
            children: (
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
            ),
          },
          {
            key: 'model',
            label: <span><RobotOutlined /> 模型</span>,
            children: (
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
            ),
          },
          {
            key: 'mcp',
            label: <span><ApiOutlined /> MCP 服务</span>,
            children: <McpConfigPage embedded embeddedTone="dialogue" />,
          },
        ]}
      />
    </Modal>
  );
}
