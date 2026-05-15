import { ApiOutlined, CheckCircleOutlined, KeyOutlined, RobotOutlined, SettingOutlined, SyncOutlined } from '@ant-design/icons';
import { Alert, Button, Descriptions, Form, Input, Space, Tag, Typography } from 'antd';
import type { FormInstance } from 'antd';

import type { ModelConfig } from '@/services/api';

export interface ModelConfigTestResult {
  status: string;
  message: string;
}

interface ModelConfigPanelProps {
  form: FormInstance;
  modelConfig: ModelConfig | null;
  testResult: ModelConfigTestResult | null;
  loading: boolean;
  saving: boolean;
  testing: boolean;
  onSave: (values: Record<string, unknown>) => void;
  onTest: () => void;
  onRefresh: () => void;
}

export function ModelConfigPanel({
  form,
  modelConfig,
  testResult,
  loading,
  saving,
  testing,
  onSave,
  onTest,
  onRefresh,
}: ModelConfigPanelProps) {
  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">大模型配置</div>
            <Typography.Text type="secondary">
              配置大模型连接参数（API 地址、Key、模型名称），保存后即时生效，无需重启。
            </Typography.Text>
          </div>
          <Space>
            {modelConfig && modelConfig.model_api_key && (
              <Tag color="success" icon={<CheckCircleOutlined />}>已配置</Tag>
            )}
            {testResult?.status === 'ok' && (
              <Tag color="success">连接正常</Tag>
            )}
            {testResult?.status === 'error' && (
              <Tag color="error">连接失败</Tag>
            )}
            {!modelConfig?.model_api_key && (
              <Tag>未配置</Tag>
            )}
          </Space>
        </div>
        <Form
          form={form}
          layout="vertical"
          onFinish={onSave}
        >
          <div className="form-grid">
            <Form.Item
              label="模型 API 地址"
              name="model_api_url"
              rules={[
                { required: true, message: '请输入 API 地址' },
                { pattern: /^https?:\/\/.+/, message: '以 http:// 或 https:// 开头' },
              ]}
            >
              <Input prefix={<ApiOutlined />} placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1" />
            </Form.Item>
            <Form.Item
              label="模型名称"
              name="model_name"
              rules={[{ required: true, message: '请输入模型名称' }]}
            >
              <Input prefix={<RobotOutlined />} placeholder="qwen-plus / qwen-max / deepseek-chat" />
            </Form.Item>
            <Form.Item
              label="API Key"
              name="model_api_key"
              rules={[{ required: true, message: '请输入 API Key' }]}
              help="Key 存储在服务器本地文件（已加入 .gitignore），前端展示时脱敏"
            >
              <Input.Password prefix={<KeyOutlined />} placeholder="sk-..." />
            </Form.Item>
          </div>
          <Space className="datasource-actions">
            <Button
              type="primary"
              icon={<SettingOutlined />}
              loading={saving}
              htmlType="submit"
            >
              保存配置
            </Button>
            <Button
              icon={<ApiOutlined />}
              loading={testing}
              onClick={onTest}
              disabled={!modelConfig?.model_api_key}
            >
              测试连接
            </Button>
            <Button
              icon={<SyncOutlined />}
              loading={loading}
              onClick={onRefresh}
            >
              刷新
            </Button>
          </Space>
        </Form>
        {testResult && (
          <Alert
            type={testResult.status === 'ok' ? 'success' : 'error'}
            showIcon
            className="agent-alert"
            message={testResult.status === 'ok' ? '连接成功' : '连接失败'}
            description={(
              <Typography.Text
                type={testResult.status === 'ok' ? 'success' : 'danger'}
                style={{ whiteSpace: 'pre-wrap' }}
              >
                {testResult.message}
              </Typography.Text>
            )}
          />
        )}
      </section>
      <section className="agent-panel">
        <div className="panel-title">配置说明</div>
        <Descriptions column={1} bordered size="small">
          <Descriptions.Item label="API 地址">
            兼容 OpenAI 接口规范。阿里云 DashScope：
            <Typography.Text code>https://dashscope.aliyuncs.com/compatible-mode/v1</Typography.Text>
          </Descriptions.Item>
          <Descriptions.Item label="API Key">
            DashScope Key 可在 <Typography.Text code>https://bailian.console.aliyun.com</Typography.Text> 生成。
          </Descriptions.Item>
          <Descriptions.Item label="模型名称">
            常用：<Tag>qwen-plus</Tag> <Tag>qwen-max</Tag> <Tag>deepseek-chat</Tag> <Tag>deepseek-reasoner</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="持久化">
            配置保存在 <Typography.Text code>backend/var/config/model.json</Typography.Text>，已在 .gitignore 中。
          </Descriptions.Item>
        </Descriptions>
      </section>
    </div>
  );
}
