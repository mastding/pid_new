import { CloudUploadOutlined, DatabaseOutlined } from '@ant-design/icons';
import {
  Alert,
  Button,
  Descriptions,
  Divider,
  Form,
  Input,
  InputNumber,
  Select,
  Space,
  Switch,
  Tag,
  Typography,
  Upload,
  message,
} from 'antd';
import type { UploadFile } from 'antd';
import { useEffect, useState } from 'react';

import {
  fetchDataSourcesConfig,
  saveDataSourcesConfig,
  type DataSourceConfigItem,
} from '@/services/api';

interface DataSourcesPanelProps {
  dataSourceType: string;
  fileList: UploadFile[];
  importedLoopCount: number;
  importing: boolean;
  onDataSourceTypeChange: (value: string) => void;
  onFileListChange: (next: UploadFile[]) => void;
  onImport: () => void;
}

const sourceTypeOptions = [
  { label: '历史文件导入', value: 'history_upload' },
  { label: '实时历史库 / Historian', value: 'historian' },
  { label: 'OPC UA', value: 'opcua' },
];

function BackendBadge() {
  return <Tag color="green">后端已接入</Tag>;
}

export function DataSourcesPanel({
  dataSourceType,
  fileList,
  importedLoopCount,
  importing,
  onDataSourceTypeChange,
  onFileListChange,
  onImport,
}: DataSourcesPanelProps) {
  const [form] = Form.useForm<DataSourceConfigItem>();
  const [sources, setSources] = useState<DataSourceConfigItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const selectedType = Form.useWatch('source_type', form) ?? dataSourceType;

  const loadConfig = async () => {
    setLoading(true);
    try {
      const resp = await fetchDataSourcesConfig();
      const first = resp.items?.[0] ?? {
        id: 'history_upload_default',
        source_name: '历史文件导入',
        source_type: dataSourceType,
        enabled: true,
      };
      setSources(resp.items ?? []);
      form.setFieldsValue(first);
      onDataSourceTypeChange(first.source_type || dataSourceType);
    } catch (error) {
      message.error(error instanceof Error ? error.message : '数据源配置加载失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      setSaving(true);
      const saved = await saveDataSourcesConfig([
        { ...values, id: values.id || sources[0]?.id || 'history_upload_default' },
      ]);
      setSources(saved.items ?? []);
      if (saved.items?.[0]) {
        form.setFieldsValue(saved.items[0]);
        onDataSourceTypeChange(saved.items[0].source_type);
      }
      message.success('数据源配置已保存');
    } catch (error) {
      if (error instanceof Error) {
        message.error(error.message);
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">数据源接入配置</div>
            <Typography.Text type="secondary">
              配置会持久化到后端，密钥只保存状态并脱敏回显；历史文件导入仍可直接生成回路资产。
            </Typography.Text>
          </div>
          <BackendBadge />
        </div>

        <Form
          form={form}
          layout="vertical"
          className="datasource-form"
          initialValues={{
            source_name: '历史文件导入',
            source_type: dataSourceType,
            enabled: true,
            host: '127.0.0.1',
            port: dataSourceType === 'opcua' ? 4840 : 8080,
            database: 'pid_history',
            username: 'pid_reader',
            polling_interval_s: 30,
          }}
        >
          <div className="form-grid">
            <Form.Item name="id" hidden>
              <Input />
            </Form.Item>
            <Form.Item label="数据源名称" name="source_name" rules={[{ required: true, message: '请输入数据源名称' }]}>
              <Input placeholder="例如：5203装置实时历史库" />
            </Form.Item>
            <Form.Item label="数据源类型" name="source_type" rules={[{ required: true, message: '请选择数据源类型' }]}>
              <Select options={sourceTypeOptions} onChange={onDataSourceTypeChange} />
            </Form.Item>
            <Form.Item label="启用" name="enabled" valuePropName="checked">
              <Switch />
            </Form.Item>

            {selectedType !== 'history_upload' && (
              <>
                <Form.Item label="服务 IP / 域名" name="host">
                  <Input placeholder="例如：10.18.2.35 或 historian.company.local" />
                </Form.Item>
                <Form.Item label="端口" name="port">
                  <InputNumber min={1} max={65535} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item label={selectedType === 'opcua' ? '服务路径' : '库名 / 命名空间'} name="database">
                  <Input placeholder={selectedType === 'opcua' ? '例如：/UA/PIDData' : '例如：PID_HISTORY'} />
                </Form.Item>
                <Form.Item label="用户名" name="username">
                  <Input placeholder="只读账号" />
                </Form.Item>
                <Form.Item label="密码 / 访问令牌" name="password">
                  <Input.Password placeholder="留空保留原密钥；输入新值会更新后端密钥" />
                </Form.Item>
                <Form.Item label="采集 / 刷新周期 (s)" name="polling_interval_s">
                  <InputNumber min={1} max={86400} style={{ width: '100%' }} />
                </Form.Item>
              </>
            )}
          </div>

          {selectedType !== 'history_upload' && (
            <Alert
              className="agent-alert"
              type="info"
              showIcon
              message="数据源配置已接后端持久化"
              description="当前保存连接元数据与密钥状态；真实连接测试、点表同步和实时采集会继续接入。"
            />
          )}

          {selectedType === 'history_upload' && (
            <>
              <Divider orientation="left">历史数据导入</Divider>
              <div className="upload-config-grid">
                <div>
                  <Upload.Dragger
                    multiple
                    fileList={fileList}
                    beforeUpload={() => false}
                    onChange={({ fileList: next }) => onFileListChange(next)}
                  >
                    <p className="ant-upload-drag-icon"><DatabaseOutlined /></p>
                    <p className="ant-upload-text">拖拽或选择多个回路历史文件</p>
                    <p className="ant-upload-hint">支持历史数据表格文件，字段需包含时间戳、过程变量和阀位变量。</p>
                  </Upload.Dragger>
                </div>
                <Alert
                  type="info"
                  showIcon
                  message="历史文件模式已接后端"
                  description="上传后系统会保存原始文件、生成标准化 CSV、识别回路、计算采样周期和候选辨识窗口。"
                />
              </div>
            </>
          )}

          <Space className="datasource-actions" wrap>
            <Button type="primary" loading={saving || loading} onClick={() => void handleSave()}>
              保存配置
            </Button>
            {selectedType === 'history_upload' && (
              <Button
                type="primary"
                icon={<CloudUploadOutlined />}
                loading={importing}
                onClick={onImport}
              >
                导入并生成回路资产
              </Button>
            )}
            <Button onClick={() => void loadConfig()} loading={loading}>
              刷新
            </Button>
          </Space>
        </Form>
      </section>

      <section className="agent-panel">
        <div className="panel-title">接入状态</div>
        <Descriptions column={1} bordered size="small">
          <Descriptions.Item label="当前类型">
            {sourceTypeOptions.find((item) => item.value === selectedType)?.label ?? selectedType}
          </Descriptions.Item>
          <Descriptions.Item label="已接接口">
            配置持久化、历史导入、回路列表、单回路趋势查询
          </Descriptions.Item>
          <Descriptions.Item label="已保存配置">{sources.length} 个</Descriptions.Item>
          <Descriptions.Item label="已导入回路">{importedLoopCount} 个</Descriptions.Item>
        </Descriptions>
      </section>
    </div>
  );
}
