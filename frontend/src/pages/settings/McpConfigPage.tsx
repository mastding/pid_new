import { useCallback, useEffect, useMemo, useState } from 'react';
import { PageContainer, ProCard } from '@ant-design/pro-components';
import {
  Button,
  Form,
  Input,
  Modal,
  Popconfirm,
  Select,
  Space,
  Switch,
  Table,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import {
  ApiOutlined,
  CheckCircleOutlined,
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import {
  createMcpServer,
  deleteMcpServer,
  listMcpServers,
  testMcpServer,
  updateMcpServer,
  type McpServerConfig,
  type McpServerPayload,
  type McpTransport,
} from '@/services/api';
import { useNavigate } from 'react-router-dom';

const { Text } = Typography;
const { TextArea } = Input;

const transportOptions: Array<{ label: string; value: McpTransport }> = [
  { label: 'SSE', value: 'sse' },
  { label: 'Streamable HTTP', value: 'streamable-http' },
  { label: 'STDIO', value: 'stdio' },
];

interface FormValues {
  name: string;
  url?: string;
  transport: McpTransport;
  raw_json?: string;
  enabled?: boolean;
  description?: string;
}

interface McpConfigPageProps {
  embedded?: boolean;
}

const MCP_LIST_CACHE_TTL_MS = 10_000;
let mcpListCache: { items: McpServerConfig[]; ts: number } | null = null;
let mcpListInFlight: Promise<McpServerConfig[]> | null = null;

function formatJson(raw: string) {
  const trimmed = raw.trim();
  if (!trimmed) return '';
  return JSON.stringify(JSON.parse(trimmed), null, 2);
}

export default function McpConfigPage({ embedded = false }: McpConfigPageProps) {
  const navigate = useNavigate();
  const [form] = Form.useForm<FormValues>();
  const [items, setItems] = useState<McpServerConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState<McpServerConfig | null>(null);
  const transport = Form.useWatch('transport', form);

  const loadItems = useCallback(async (force = false) => {
    const now = Date.now();
    if (!force && mcpListCache && now - mcpListCache.ts < MCP_LIST_CACHE_TTL_MS) {
      setItems(mcpListCache.items);
      return;
    }

    setLoading(true);
    try {
      if (!force && mcpListInFlight) {
        setItems(await mcpListInFlight);
        return;
      }

      mcpListInFlight = listMcpServers()
        .then((data) => {
          mcpListCache = { items: data.items, ts: Date.now() };
          return data.items;
        })
        .finally(() => {
          mcpListInFlight = null;
        });

      setItems(await mcpListInFlight);
    } catch (error) {
      message.error(`加载 MCP 服务配置失败: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadItems();
  }, [loadItems]);

  const openCreate = useCallback(() => {
    setEditing(null);
    form.setFieldsValue({
      name: '',
      url: '',
      transport: 'sse',
      raw_json: '',
      enabled: true,
      description: '',
    });
    setModalOpen(true);
  }, [form]);

  const openEdit = useCallback((record: McpServerConfig) => {
    setEditing(record);
    form.setFieldsValue(record);
    setModalOpen(true);
  }, [form]);

  const handleFormatJson = useCallback(() => {
    const raw = form.getFieldValue('raw_json') || '';
    try {
      form.setFieldValue('raw_json', formatJson(raw));
      message.success('JSON 已格式化');
    } catch (error) {
      message.error(`JSON 格式不正确: ${(error as Error).message}`);
    }
  }, [form]);

  const handleSave = useCallback(async () => {
    let values: FormValues;
    try {
      values = await form.validateFields();
      if (values.raw_json?.trim()) {
        JSON.parse(values.raw_json);
      }
    } catch (error) {
      if (error instanceof SyntaxError) {
        message.error(`通用 MCP JSON 格式不正确: ${error.message}`);
      }
      return;
    }

    const body: McpServerPayload = {
      name: values.name.trim(),
      url: values.url?.trim() || '',
      transport: values.transport,
      raw_json: values.raw_json?.trim() || '',
      enabled: values.enabled ?? true,
      description: values.description?.trim() || '',
    };

    setSaving(true);
    try {
      if (editing) {
        await updateMcpServer(editing.id, body);
        message.success('MCP 服务配置已更新');
      } else {
        await createMcpServer(body);
        message.success('MCP 服务配置已新增');
      }
      setModalOpen(false);
      await loadItems(true);
    } catch (error) {
      message.error(`保存失败: ${(error as Error).message}`);
    } finally {
      setSaving(false);
    }
  }, [editing, form, loadItems]);

  const handleDelete = useCallback(async (id: string) => {
    try {
      await deleteMcpServer(id);
      message.success('MCP 服务配置已删除');
      await loadItems(true);
    } catch (error) {
      message.error(`删除失败: ${(error as Error).message}`);
    }
  }, [loadItems]);

  const handleTest = useCallback(async (record: McpServerConfig) => {
    setTestingId(record.id);
    try {
      const resp = await testMcpServer(record.id);
      if (resp.status === 'ok') {
        message.success(resp.message);
      } else {
        message.warning(resp.message);
      }
    } catch (error) {
      message.error(`连通性自检失败: ${(error as Error).message}`);
    } finally {
      setTestingId(null);
    }
  }, []);

  const columns = useMemo<ColumnsType<McpServerConfig>>(
    () => [
      {
        title: '服务名称',
        dataIndex: 'name',
        key: 'name',
        render: (name: string, record) => (
          <Space direction="vertical" size={0}>
            <Space>
              <Text strong>{name}</Text>
              <Tag color={record.enabled ? 'success' : 'default'}>
                {record.enabled ? '启用' : '停用'}
              </Tag>
            </Space>
            {record.description && <Text type="secondary">{record.description}</Text>}
          </Space>
        ),
      },
      {
        title: 'Transport',
        dataIndex: 'transport',
        key: 'transport',
        width: 170,
        render: (value: McpTransport) => <Tag>{value}</Tag>,
      },
      {
        title: '服务地址',
        dataIndex: 'url',
        key: 'url',
        ellipsis: true,
        render: (url: string, record) => (
          url ? <Text copyable>{url}</Text> : <Text type="secondary">{record.transport === 'stdio' ? '由 JSON 定义' : '-'}</Text>
        ),
      },
      {
        title: '通用 MCP JSON',
        dataIndex: 'raw_json',
        key: 'raw_json',
        width: 150,
        render: (raw: string) => (
          raw?.trim() ? <Tag color="processing">已配置</Tag> : <Tag>未配置</Tag>
        ),
      },
      {
        title: '操作',
        key: 'actions',
        width: 240,
        render: (_, record) => (
          <Space>
            <Button
              icon={<ThunderboltOutlined />}
              loading={testingId === record.id}
              onClick={() => handleTest(record)}
            >
              自检
            </Button>
            <Button icon={<EditOutlined />} onClick={() => openEdit(record)}>
              编辑
            </Button>
            <Popconfirm
              title="删除 MCP 服务配置？"
              okText="删除"
              cancelText="取消"
              onConfirm={() => handleDelete(record.id)}
            >
              <Button danger icon={<DeleteOutlined />} />
            </Popconfirm>
          </Space>
        ),
      },
    ],
    [handleDelete, handleTest, openEdit, testingId],
  );

  const enabledCount = items.filter((item) => item.enabled).length;
  const transportCount = new Set(items.map((item) => item.transport)).size;

  const tableNode = (
    <Table
      rowKey="id"
      loading={loading}
      columns={columns}
      dataSource={items}
      pagination={false}
      scroll={{ x: 980 }}
      className={embedded ? 'mcp-config-table' : undefined}
    />
  );

  const modalNode = (
    <Modal
      title={editing ? '编辑 MCP 服务' : '新增 MCP 服务'}
      open={modalOpen}
      onCancel={() => setModalOpen(false)}
      onOk={handleSave}
      confirmLoading={saving}
      width={760}
      destroyOnHidden
      className={embedded ? 'industrial-modal mcp-config-modal' : undefined}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{ transport: 'sse', enabled: true }}
      >
        <Form.Item
          label="服务名称"
          name="name"
          rules={[{ required: true, message: '请输入服务名称' }]}
        >
          <Input placeholder="例如：知识库检索 MCP" />
        </Form.Item>

        <Form.Item
          label="Transport"
          name="transport"
          rules={[{ required: true, message: '请选择 Transport' }]}
        >
          <Select options={transportOptions} />
        </Form.Item>

        <Form.Item
          label="MCP 服务地址"
          name="url"
          rules={[
            {
              validator: async (_, value: string | undefined) => {
                if (transport === 'stdio') return;
                if (!value?.trim()) throw new Error('请输入 MCP 服务地址');
                if (!/^https?:\/\/.+/.test(value.trim())) {
                  throw new Error('服务地址必须以 http:// 或 https:// 开头');
                }
              },
            },
          ]}
        >
          <Input
            prefix={<ApiOutlined />}
            placeholder={transport === 'stdio' ? 'stdio 可留空' : 'https://example.com/mcp'}
          />
        </Form.Item>

        <Form.Item label="启用" name="enabled" valuePropName="checked">
          <Switch checkedChildren={<CheckCircleOutlined />} unCheckedChildren="停用" />
        </Form.Item>

        <Form.Item label="描述" name="description">
          <Input.TextArea rows={2} placeholder="可选，说明服务用途或凭证来源" />
        </Form.Item>

        <Form.Item
          label={
            <Space>
              <span>通用 MCP JSON</span>
              <Button size="small" onClick={handleFormatJson}>
                格式化
              </Button>
            </Space>
          }
          name="raw_json"
          extra="可填写 command、args、env、headers 等通用 MCP 配置；为空时仅保存上方字段。"
        >
          <TextArea
            rows={10}
            spellCheck={false}
            placeholder={'{\n  "headers": {\n    "Authorization": "Bearer ..."\n  }\n}'}
          />
        </Form.Item>
      </Form>
    </Modal>
  );

  if (embedded) {
    return (
      <div className="mcp-config-page mcp-config-page--embedded">
        <section className="agent-panel mcp-config-panel">
          <div className="panel-toolbar mcp-config-toolbar">
            <div>
              <h3>MCP 服务配置</h3>
              <p>管理可对接的 Model Context Protocol 服务，保存后立即写入后端配置文件。</p>
            </div>
            <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
              新增 MCP 服务
            </Button>
          </div>

          <div className="mcp-config-summary">
            <div className="metric-cell">
              <span>已配置服务</span>
              <strong>{items.length}</strong>
            </div>
            <div className="metric-cell">
              <span>启用服务</span>
              <strong>{enabledCount}</strong>
            </div>
            <div className="metric-cell">
              <span>Transport 类型</span>
              <strong>{transportCount}</strong>
            </div>
          </div>

          <div className="mcp-config-table-shell">{tableNode}</div>
        </section>
        {modalNode}
      </div>
    );
  }

  return (
    <PageContainer
      title="MCP 服务配置"
      subTitle="管理可对接的 Model Context Protocol 服务，保存后立即写入后端配置文件"
    >
      <Tabs
        activeKey="mcp"
        onChange={(key) => {
          if (key === 'llm') navigate('/settings');
        }}
        items={[
          { key: 'llm', label: 'LLM 模型配置' },
          { key: 'mcp', label: 'MCP 服务配置' },
        ]}
      />

      <ProCard
        title={
          <Space>
            <ApiOutlined />
            <span>MCP 服务列表</span>
          </Space>
        }
        extra={
          <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
            新增 MCP 服务
          </Button>
        }
      >
        {tableNode}
      </ProCard>

      {modalNode}
    </PageContainer>
  );
}
