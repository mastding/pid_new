import { CloudUploadOutlined, DatabaseOutlined } from '@ant-design/icons';
import { Alert, Button, Descriptions, Divider, Form, Input, InputNumber, Select, Space, Tag, Typography, Upload } from 'antd';
import type { UploadFile } from 'antd';

interface DataSourcesPanelProps {
  dataSourceType: string;
  fileList: UploadFile[];
  importedLoopCount: number;
  importing: boolean;
  onDataSourceTypeChange: (value: string) => void;
  onFileListChange: (next: UploadFile[]) => void;
  onImport: () => void;
}

function BackendBadge({ implemented }: { implemented?: boolean }) {
  return implemented ? <Tag color="green">已接后端</Tag> : <Tag color="default">未开放</Tag>;
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
  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">数据源接入配置</div>
            <Typography.Text type="secondary">
              先选择数据源类型，再配置连接参数；当前仅“历史文件导入”已接真实后端。
            </Typography.Text>
          </div>
          <BackendBadge implemented={dataSourceType === 'history_upload'} />
        </div>

        <Form layout="vertical" className="datasource-form" initialValues={{
          source_name: '5203历史回路数据源',
          source_type: dataSourceType,
          host: '127.0.0.1',
          port: dataSourceType === 'opcua' ? 4840 : 8080,
          database: 'pid_history',
          username: 'pid_reader',
          polling_interval_s: 30,
        }}>
          <div className="form-grid">
            <Form.Item label="数据源名称" name="source_name">
              <Input placeholder="例如：5203装置实时历史库" />
            </Form.Item>
            <Form.Item label="数据源类型" name="source_type">
              <Select
                value={dataSourceType}
                onChange={onDataSourceTypeChange}
                options={[
                  { label: '历史文件导入', value: 'history_upload' },
                ]}
              />
            </Form.Item>
            {dataSourceType !== 'history_upload' && (
              <>
                <Form.Item label="服务 IP / 域名" name="host">
                  <Input placeholder="例如：10.18.2.35 或 historian.company.local" />
                </Form.Item>
                <Form.Item label="端口" name="port">
                  <InputNumber min={1} max={65535} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item label={dataSourceType === 'opcua' ? '服务路径' : '库名 / 命名空间'} name="database">
                  <Input placeholder={dataSourceType === 'opcua' ? '例如：/UA/PIDData' : '例如：PID_HISTORY'} />
                </Form.Item>
                <Form.Item label="用户名" name="username">
                  <Input placeholder="只读账号" />
                </Form.Item>
                <Form.Item label="密码 / 访问令牌" name="password">
                  <Input.Password placeholder="后端待接入，当前不保存" />
                </Form.Item>
                <Form.Item label="采集/刷新周期 (s)" name="polling_interval_s">
                  <InputNumber min={1} max={3600} style={{ width: '100%' }} />
                </Form.Item>
              </>
            )}
          </div>

          {dataSourceType !== 'history_upload' && (
            <Alert
              className="agent-alert"
              type="warning"
              showIcon
              message="该数据源类型后端待接入"
              description="该类型需要后端补充连接测试、点表同步、实时趋势和报警读取能力。"
            />
          )}

          {dataSourceType === 'history_upload' && (
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
            <Button type="primary" disabled={dataSourceType !== 'history_upload'}>
              保存配置
            </Button>
            {dataSourceType === 'history_upload' && (
              <Button
                type="primary"
                icon={<CloudUploadOutlined />}
                loading={importing}
                onClick={onImport}
              >
                导入并生成回路资产
              </Button>
            )}
          </Space>
        </Form>
      </section>

      <section className="agent-panel">
        <div className="panel-title">接入状态</div>
        <Descriptions column={1} bordered size="small">
          <Descriptions.Item label="当前模式">
            离线历史文件导入
          </Descriptions.Item>
          <Descriptions.Item label="已接接口">
            历史导入、回路列表、单回路趋势查询
          </Descriptions.Item>
          <Descriptions.Item label="已导入回路">{importedLoopCount} 个</Descriptions.Item>
        </Descriptions>
      </section>
    </div>
  );
}
