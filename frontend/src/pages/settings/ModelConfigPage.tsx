import { useCallback, useEffect, useState } from 'react';
import { PageContainer, ProCard } from '@ant-design/pro-components';
import {
  Alert,
  Button,
  Descriptions,
  Form,
  Input,
  message,
  Space,
  Tabs,
  Tag,
  Typography,
} from 'antd';
import {
  ApiOutlined,
  CheckCircleOutlined,
  KeyOutlined,
  RobotOutlined,
  SendOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import {
  fetchModelConfig,
  testModelConfig,
  updateModelConfig,
  type ModelConfig,
} from '@/services/api';
import { useNavigate } from 'react-router-dom';

const { Text } = Typography;

export default function ModelConfigPage() {
  const navigate = useNavigate();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [config, setConfig] = useState<ModelConfig | null>(null);
  const [testResult, setTestResult] = useState<{
    status: string;
    message: string;
  } | null>(null);

  const loadConfig = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchModelConfig();
      setConfig(data);
      form.setFieldsValue(data);
      setTestResult(null);
    } catch {
      message.error('加载模型配置失败');
    } finally {
      setLoading(false);
    }
  }, [form]);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  const handleSave = useCallback(async () => {
    try {
      await form.validateFields();
    } catch {
      return;
    }
    setSaving(true);
    try {
      const values = form.getFieldsValue();
      const body: Record<string, string | null> = {};
      for (const k of ['model_api_url', 'model_api_key', 'model_name']) {
        const v = (values[k] ?? '').trim();
        body[k] = v || null;
      }
      const resp = await updateModelConfig(body);
      setConfig(resp.config);
      setTestResult(null);
      message.success('模型配置已保存并生效');
    } catch (e) {
      message.error(`保存失败: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  }, [form]);

  const handleTest = useCallback(async () => {
    try {
      await form.validateFields();
    } catch {
      return;
    }
    setTesting(true);
    setTestResult(null);
    try {
      const resp = await testModelConfig();
      setTestResult(resp);
      if (resp.status === 'ok') {
        message.success('连接测试通过');
      } else {
        message.warning('连接测试失败，请检查配置');
      }
    } catch (e) {
      setTestResult({ status: 'error', message: (e as Error).message });
      message.error('连接测试异常');
    } finally {
      setTesting(false);
    }
  }, [form]);

  const isConfigured = config?.model_api_url && config?.model_api_key;

  return (
    <PageContainer
      title="系统配置"
      subTitle="管理 LLM 模型连接参数，保存后即时生效，无需重启服务"
    >
      <Tabs
        activeKey="llm"
        onChange={(key) => {
          if (key === 'mcp') navigate('/settings/mcp');
        }}
        items={[
          { key: 'llm', label: 'LLM 模型配置' },
          { key: 'mcp', label: 'MCP 服务配置' },
        ]}
      />

      <ProCard
        title={
          <Space>
            <RobotOutlined />
            <span>LLM 模型配置</span>
          </Space>
        }
        style={{ marginBottom: 16 }}
        loading={loading}
        extra={
          config && (
            <Space>
              <Tag
                color={isConfigured ? 'success' : 'default'}
                icon={isConfigured ? <CheckCircleOutlined /> : undefined}
              >
                {isConfigured ? '已配置' : '未配置'}
              </Tag>
              {testResult?.status === 'ok' && (
                <Tag color="success">连接正常</Tag>
              )}
              {testResult?.status === 'error' && (
                <Tag color="error">连接失败</Tag>
              )}
            </Space>
          )
        }
      >
        <Form
          form={form}
          layout="vertical"
          style={{ maxWidth: 640 }}
          initialValues={{
            model_api_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            model_name: 'qwen-plus',
          }}
        >
          <Form.Item
            label="模型 API 地址"
            name="model_api_url"
            rules={[
              { required: true, message: '请输入 API 地址' },
              {
                pattern: /^https?:\/\/.+/,
                message: '必须以 http:// 或 https:// 开头',
              },
            ]}
          >
            <Input
              prefix={<ApiOutlined />}
              placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1"
            />
          </Form.Item>

          <Form.Item label="模型名称" name="model_name" rules={[{ required: true, message: '请输入模型名称' }]}>
            <Input
              prefix={<RobotOutlined />}
              placeholder="qwen-plus / qwen-max / deepseek-chat"
            />
          </Form.Item>

          <Form.Item
            label="API Key"
            name="model_api_key"
            rules={[{ required: true, message: '请输入 API Key' }]}
            help="API Key 以加密形式存储在前端和后端之间，GET 返回时脱敏展示"
          >
            <Input.Password
              prefix={<KeyOutlined />}
              placeholder="sk-..."
              visibilityToggle
            />
          </Form.Item>

          <Space>
            <Button
              type="default"
              icon={<SendOutlined />}
              loading={testing}
              onClick={handleTest}
              disabled={!isConfigured}
            >
              测试连接
            </Button>
            <Button
              type="primary"
              icon={<SettingOutlined />}
              loading={saving}
              onClick={handleSave}
            >
              保存配置
            </Button>
          </Space>
        </Form>

        {testResult && (
          <Alert
            type={testResult.status === 'ok' ? 'success' : 'error'}
            showIcon
            style={{ marginTop: 16, maxWidth: 640 }}
            message={testResult.status === 'ok' ? '连接成功' : '连接失败'}
            description={
              <Text
                type={testResult.status === 'ok' ? 'success' : 'danger'}
                style={{ whiteSpace: 'pre-wrap' }}
              >
                {testResult.message}
              </Text>
            }
          />
        )}
      </ProCard>

      <ProCard title="配置说明" size="small">
        <Descriptions column={1} size="small">
          <Descriptions.Item label="API 地址">
            兼容 OpenAI 接口规范的 API 端点。推荐使用阿里云 DashScope：
            <Text code>https://dashscope.aliyuncs.com/compatible-mode/v1</Text>
          </Descriptions.Item>
          <Descriptions.Item label="API Key">
            在对应平台获取。DashScope Key 可在
            <Text code>https://bailian.console.aliyun.com</Text> 生成。
          </Descriptions.Item>
          <Descriptions.Item label="模型名称">
            常用选项：
            <Tag>qwen-plus</Tag>
            <Tag>qwen-max</Tag>
            <Tag>deepseek-chat</Tag>
            <Tag>deepseek-reasoner</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="生效方式">
            保存后即时生效，无需重启服务。配置持久化在
            <Text code>backend/var/config/model.json</Text> 文件中。
          </Descriptions.Item>
          <Descriptions.Item label="安全提示">
            API Key 以明文存储在服务器本地文件中（该目录已加入 .gitignore，不会提交到版本控制）。
            前端 GET 请求返回时自动脱敏（仅显示首尾各 4 位）。
          </Descriptions.Item>
        </Descriptions>
      </ProCard>
    </PageContainer>
  );
}
