import { useCallback, useEffect, useMemo, useState } from 'react';
import { PageContainer, ProCard } from '@ant-design/pro-components';
import {
  Alert,
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
import { useNavigate } from 'react-router-dom';
import {
  createMcpServer,
  deleteMcpServer,
  fetchMcpOntologyQueryConfig,
  listMcpServers,
  resetMcpOntologyQueryConfig,
  testMcpServer,
  updateMcpOntologyQueryConfig,
  updateMcpServer,
  type McpServerConfig,
  type McpServerPayload,
  type McpTransport,
} from '@/services/api';

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

interface OntologyQueryFormValues {
  query_template: string;
}

interface McpConfigPageProps {
  embedded?: boolean;
  embeddedTone?: 'industrial' | 'dialogue';
}

const MCP_LIST_CACHE_TTL_MS = 10_000;
let mcpListCache: { items: McpServerConfig[]; ts: number } | null = null;
let mcpListInFlight: Promise<McpServerConfig[]> | null = null;

function formatJson(raw: string) {
  const trimmed = raw.trim();
  if (!trimmed) return '';
  return JSON.stringify(JSON.parse(trimmed), null, 2);
}

export default function McpConfigPage({ embedded = false, embeddedTone = 'industrial' }: McpConfigPageProps) {
  const navigate = useNavigate();
  const [form] = Form.useForm<FormValues>();
  const [ontologyForm] = Form.useForm<OntologyQueryFormValues>();
  const [items, setItems] = useState<McpServerConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [ontologyLoading, setOntologyLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [ontologySaving, setOntologySaving] = useState(false);
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

  const loadOntologyConfig = useCallback(async () => {
    setOntologyLoading(true);
    try {
      const config = await fetchMcpOntologyQueryConfig();
      ontologyForm.setFieldsValue({ query_template: config.query_template });
    } catch (error) {
      message.error(`加载本体查询问题配置失败: ${(error as Error).message}`);
    } finally {
      setOntologyLoading(false);
    }
  }, [ontologyForm]);

  useEffect(() => {
    loadItems();
    loadOntologyConfig();
  }, [loadItems, loadOntologyConfig]);

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
        message.error(`通用配置格式不正确: ${error.message}`);
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

  const handleSaveOntologyConfig = useCallback(async () => {
    let values: OntologyQueryFormValues;
    try {
      values = await ontologyForm.validateFields();
    } catch {
      return;
    }
    setOntologySaving(true);
    try {
      const resp = await updateMcpOntologyQueryConfig({
        query_template: values.query_template.trim(),
      });
      ontologyForm.setFieldsValue({ query_template: resp.config.query_template });
      message.success('本体查询问题配置已保存');
    } catch (error) {
      message.error(`保存本体查询问题失败: ${(error as Error).message}`);
    } finally {
      setOntologySaving(false);
    }
  }, [ontologyForm]);

  const handleResetOntologyConfig = useCallback(async () => {
    setOntologySaving(true);
    try {
      const resp = await resetMcpOntologyQueryConfig();
      ontologyForm.setFieldsValue({ query_template: resp.config.query_template });
      message.success('本体查询问题已恢复默认');
    } catch (error) {
      message.error(`恢复默认失败: ${(error as Error).message}`);
    } finally {
      setOntologySaving(false);
    }
  }, [ontologyForm]);

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
        title: '传输方式',
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
          url ? <Text copyable>{url}</Text> : <Text type="secondary">{record.transport === 'stdio' ? '由通用配置定义' : '-'}</Text>
        ),
      },
      {
        title: '通用配置',
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

  const ontologyNode = (
    <div className="mcp-ontology-query-panel">
      <div className="mcp-ontology-config-head">
        <div>
          <h3>本体问题查询配置</h3>
          <p>整定任务和先验评审调用 MCP 本体服务时，会用这里的问题模板生成实际查询问题。</p>
        </div>
        <Space>
          <Button loading={ontologyLoading} onClick={loadOntologyConfig}>
            刷新
          </Button>
          <Popconfirm
            title="恢复默认本体查询问题？"
            okText="恢复"
            cancelText="取消"
            onConfirm={handleResetOntologyConfig}
          >
            <Button loading={ontologySaving}>恢复默认</Button>
          </Popconfirm>
          <Button type="primary" loading={ontologySaving} onClick={handleSaveOntologyConfig}>
            保存配置
          </Button>
        </Space>
      </div>
      <Form form={ontologyForm} layout="vertical" className="mcp-ontology-form">
        <Form.Item
          label="当前本体查询问题模板"
          name="query_template"
          rules={[{ required: true, message: '请输入本体查询问题模板' }]}
          extra="支持占位符 $loop_name 和 $loop_type，后端会在实际查询时替换为当前回路。"
        >
          <TextArea rows={8} spellCheck={false} />
        </Form.Item>
      </Form>
    </div>
  );

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
      className={
        embedded
          ? embeddedTone === 'dialogue'
            ? 'mcp-config-modal mcp-config-modal--dialogue'
            : 'industrial-modal mcp-config-modal'
          : undefined
      }
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
          <Input placeholder="例如：本体知识库 MCP" />
        </Form.Item>

        <Form.Item
          label="传输方式"
          name="transport"
          rules={[{ required: true, message: '请选择传输方式' }]}
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
          <Input.TextArea rows={2} placeholder="可选，说明服务用途、凭证来源或工具能力" />
        </Form.Item>

        <Form.Item
          label={(
            <Space>
              <span>通用 MCP JSON 配置</span>
              <Button size="small" onClick={handleFormatJson}>
                格式化
              </Button>
            </Space>
          )}
          name="raw_json"
          extra="可填写启动命令、参数、环境变量或请求头；为空时仅保存上方字段。"
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
      <div className={`mcp-config-page mcp-config-page--embedded mcp-config-page--${embeddedTone}`}>
        <section className="agent-panel mcp-config-panel">
          <div className="panel-toolbar mcp-config-toolbar">
            <div>
              <h3>MCP 服务配置</h3>
              <p>管理可接入的 MCP 服务，保存后后端会用于本体查询和工具调用。</p>
            </div>
            <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
              新增 MCP 服务
            </Button>
          </div>

          {ontologyNode}

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
              <span>传输类型</span>
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
      subTitle="管理模型上下文协议服务，并配置后端查询本体知识库时使用的问题模板。"
    >
      <Tabs
        activeKey="mcp"
        onChange={(key) => {
          if (key === 'llm') navigate('/settings');
        }}
        items={[
          { key: 'llm', label: '大模型配置' },
          { key: 'mcp', label: 'MCP 服务配置' },
        ]}
      />

      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <ProCard
          title={(
            <Space>
              <ApiOutlined />
              <span>MCP 服务列表</span>
            </Space>
          )}
          extra={(
            <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
              新增 MCP 服务
            </Button>
          )}
        >
          {ontologyNode}
          <Alert
            type="info"
            showIcon
            style={{ marginBottom: 12 }}
            message="启用的 MCP 服务会被后端按顺序尝试，当前本体检索默认调用名为 chat 的工具。"
          />
          {tableNode}
        </ProCard>
      </Space>

      {modalNode}
    </PageContainer>
  );
}
