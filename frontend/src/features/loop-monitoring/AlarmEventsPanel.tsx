import { Descriptions, Space, Table, Tag, Typography } from 'antd';

export interface RailAlarmRow {
  key: string;
  time: string;
  level: string;
  name: string;
  value: string;
  status: string;
  recommendation: string;
  evidence: string;
}

interface AlarmEventsPanelProps {
  railAlarms: RailAlarmRow[];
  monitoringStatus?: string;
  monitoringEventCount: number;
  diagnosticFlagCount: number;
  taskLabel: string;
  pathLabel: string;
  monitoringStatusText: (status?: string) => string;
  monitoringStatusColor: (status?: string) => string;
  alertSeverityColor: (severity?: string) => string;
}

export function AlarmEventsPanel({
  railAlarms,
  monitoringStatus,
  monitoringEventCount,
  diagnosticFlagCount,
  taskLabel,
  pathLabel,
  monitoringStatusText,
  monitoringStatusColor,
  alertSeverityColor,
}: AlarmEventsPanelProps) {
  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">报警与智能体事件</div>
            <Typography.Text type="secondary">
              集中展示监控告警、数据质量提示和整定任务事件。
            </Typography.Text>
          </div>
          <Space wrap>
            <Tag color={railAlarms.length ? 'orange' : 'green'}>{railAlarms.length} 条事件</Tag>
            <Tag color={monitoringStatusColor(monitoringStatus)}>{monitoringStatusText(monitoringStatus)}</Tag>
          </Space>
        </div>
        <Table
          size="small"
          pagination={{ pageSize: 10 }}
          rowKey="key"
          dataSource={railAlarms}
          columns={[
            { title: '时间', dataIndex: 'time', width: 140 },
            { title: '级别', dataIndex: 'level', width: 90, render: (value: string) => <Tag color={alertSeverityColor(value)}>{value}</Tag> },
            { title: '名称', dataIndex: 'name', width: 180 },
            { title: '描述', dataIndex: 'value', ellipsis: true },
            { title: '建议动作', dataIndex: 'recommendation', ellipsis: true },
            { title: '状态', dataIndex: 'status', width: 120 },
          ]}
        />
      </section>

      <section className="agent-panel">
        <div className="panel-title">事件来源说明</div>
        <Descriptions bordered size="small" column={3} className="industrial-descriptions">
          <Descriptions.Item label="监控事件">{monitoringEventCount} 条</Descriptions.Item>
          <Descriptions.Item label="诊断标记">{diagnosticFlagCount} 条</Descriptions.Item>
          <Descriptions.Item label="整定任务">{taskLabel}</Descriptions.Item>
          <Descriptions.Item label="当前作用域" span={3}>
            {pathLabel}
          </Descriptions.Item>
        </Descriptions>
      </section>
    </div>
  );
}
