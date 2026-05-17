import { RobotOutlined, SettingOutlined, SyncOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import { Alert, Button, Descriptions, Form, Input, Select, Space, Tag, Typography } from 'antd';
import type { FormInstance } from 'antd';

import type { PromptConfig } from '@/services/api';

export interface PromptConfigItem {
  key: string;
  label: string;
  group: string;
  help: string;
  placeholder: string;
  minRows: number;
  maxRows: number;
}

interface PromptConfigPanelProps {
  form: FormInstance;
  promptConfig: PromptConfig | null;
  promptItems: PromptConfigItem[];
  activePromptField: string;
  loading: boolean;
  saving: boolean;
  onActivePromptFieldChange: (value: string) => void;
  onSave: () => void;
  onRefresh: () => void;
  onRestoreDefault: () => void;
}

export function PromptConfigPanel({
  form,
  promptConfig,
  promptItems,
  activePromptField,
  loading,
  saving,
  onActivePromptFieldChange,
  onSave,
  onRefresh,
  onRestoreDefault,
}: PromptConfigPanelProps) {
  const activePromptItem = promptItems.find((item) => item.key === activePromptField) ?? promptItems[0];

  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">提示词管理</div>
            <Typography.Text type="secondary">
              统一维护智能助手、窗口候选和辨识评审提示词。
            </Typography.Text>
          </div>
          <Space>
            {promptConfig?.updated_at && (
              <Tag color="blue">更新于 {dayjs(promptConfig.updated_at).format('YYYY-MM-DD HH:mm')}</Tag>
            )}
          </Space>
        </div>
        <Alert
          className="agent-alert"
          type="info"
          showIcon
          message="选择类型后编辑，保存即生效"
          description="整定、窗口候选和参数修改仍需用户在对应页面确认。"
        />
        <Form
          form={form}
          layout="vertical"
          onFinish={onSave}
        >
          <Form.Item label="提示词类型">
            <Select
              value={activePromptField}
              onChange={onActivePromptFieldChange}
              options={promptItems.map((item) => ({
                label: `${item.group} / ${item.label}`,
                value: item.key,
              }))}
            />
          </Form.Item>
          <Form.Item
            label={activePromptItem.label}
            name={activePromptItem.key}
            rules={[{ required: true, message: `请输入${activePromptItem.label}` }]}
            help={activePromptItem.help}
          >
            <Input.TextArea
              autoSize={{ minRows: activePromptItem.minRows, maxRows: activePromptItem.maxRows }}
              placeholder={activePromptItem.placeholder}
            />
          </Form.Item>
          <Space className="datasource-actions">
            <Button
              type="primary"
              icon={<SettingOutlined />}
              loading={saving}
              htmlType="submit"
            >
              保存提示词
            </Button>
            <Button
              icon={<SyncOutlined />}
              loading={loading}
              onClick={onRefresh}
            >
              刷新
            </Button>
            <Button
              icon={<RobotOutlined />}
              loading={saving}
              onClick={onRestoreDefault}
            >
              恢复默认
            </Button>
          </Space>
        </Form>
      </section>
      <section className="agent-panel">
        <div className="panel-title">调用流程建议</div>
        <Descriptions column={1} bordered size="small">
          <Descriptions.Item label="上下文输入">
            前端传入当前页面、装置范围、选中回路、监控快照、画像指标和整定历史等上下文数据。
          </Descriptions.Item>
          <Descriptions.Item label="模型输出">
            模型返回答案、证据、风险级别和建议动作，前端只渲染白名单动作。
          </Descriptions.Item>
          <Descriptions.Item label="高风险操作">
            整定、窗口候选、参数修改等操作只允许用户点击按钮后进入对应页面确认，不由模型直接执行。
          </Descriptions.Item>
          <Descriptions.Item label="持久化位置">
            配置保存在后端本地配置文件。
          </Descriptions.Item>
        </Descriptions>
      </section>
    </div>
  );
}
