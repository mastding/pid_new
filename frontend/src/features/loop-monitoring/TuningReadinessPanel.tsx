import type { ReactNode } from 'react';
import { Alert, Table, Tag, Typography } from 'antd';

import type { HistoryLoopAssessment } from '@/services/api';

interface TuningReadinessPanelProps {
  assessment: HistoryLoopAssessment | null;
  showDetails: boolean;
  assessmentCards: ReactNode;
  tagColor: (level?: string) => string;
}

export function TuningReadinessPanel({
  assessment,
  showDetails,
  assessmentCards,
  tagColor,
}: TuningReadinessPanelProps) {
  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">整定准备度</div>
            <Typography.Text type="secondary">把评分、原始指标和建议动作拆开，避免大卡片堆叠。</Typography.Text>
          </div>
          <Tag color={tagColor(assessment?.readiness.level)}>
            {assessment?.readiness.level ?? '-'}
          </Tag>
        </div>
        {assessmentCards}
      </section>
      <section className="agent-panel">
        <div className="panel-title">建议动作</div>
        <Table
          size="small"
          pagination={false}
          rowKey="item"
          dataSource={(assessment?.readiness.recommendations ?? ['暂无建议']).map((item, index) => ({ index: index + 1, item }))}
          columns={[
            { title: '#', dataIndex: 'index', width: 64 },
            { title: '建议', dataIndex: 'item' },
          ]}
        />
      </section>
      {showDetails && assessment?.summary && (
        <section className="agent-panel">
          <div className="panel-title">整定准入结论</div>
          <Alert
            className="agent-alert"
            type={assessment.summary.decision === 'blocked' ? 'error' : assessment.summary.decision === 'ready' ? 'success' : 'warning'}
            showIcon
            message={assessment.summary.decision_text}
            description={assessment.summary.recommended_next_action_text}
          />
        </section>
      )}
      {showDetails && assessment?.tuning_readiness && (
        <section className="panel-grid">
          <div className="agent-panel">
            <div className="panel-title">准入检查</div>
            <Table
              size="small"
              pagination={false}
              rowKey="name"
              dataSource={assessment.tuning_readiness.gate_checks ?? []}
              columns={[
                { title: '检查项', dataIndex: 'name', width: 150 },
                {
                  title: '状态',
                  dataIndex: 'passed',
                  width: 100,
                  render: (value: boolean) => <Tag color={value ? 'green' : 'orange'}>{value ? '通过' : '需处理'}</Tag>,
                },
                { title: '说明', dataIndex: 'message' },
              ]}
            />
          </div>
          <div className="agent-panel">
            <div className="panel-title">阻断/关注原因</div>
            <Table
              size="small"
              pagination={false}
              rowKey={(row) => `${row.type}-${row.message}`}
              dataSource={assessment.tuning_readiness.blocking_reasons ?? []}
              columns={[
                { title: '类型', dataIndex: 'type', width: 140 },
                {
                  title: '等级',
                  dataIndex: 'severity',
                  width: 90,
                  render: (value: string) => <Tag color={value === 'high' ? 'red' : 'orange'}>{value}</Tag>,
                },
                { title: '原因', dataIndex: 'message' },
              ]}
            />
          </div>
        </section>
      )}
    </div>
  );
}
