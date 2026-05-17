import { Alert, Descriptions, Empty, Space, Table, Tag, Typography } from 'antd';

import type { HistoryLoop, HistoryLoopFeatures } from '@/services/api';

type OperatingConditionProfile = NonNullable<HistoryLoopFeatures['operating_condition_profile']>;

interface OperatingConditionPanelProps {
  conditionProfile?: OperatingConditionProfile;
  selectedLoop?: HistoryLoop;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  operatingConditionText: (label?: string) => string;
  tuningSuitabilityText: (value?: string) => string;
  tuningSuitabilityColor: (value?: string) => string;
  conditionEvidenceName: (value?: string) => string;
  conditionEvidenceDetail: (value?: string) => string;
  conditionRecommendationText: (value?: string) => string;
  evidenceStatusText: (value?: string) => string;
  evidenceStatusColor: (value?: string) => string;
}

function ontologyStatusText(value?: string) {
  if (!value || value === 'not_connected') return '使用本地画像规则';
  if (value === 'connected') return '本体已接入';
  return value;
}

export function OperatingConditionPanel({
  conditionProfile,
  selectedLoop,
  formatNumber,
  formatPercentValue,
  operatingConditionText,
  tuningSuitabilityText,
  tuningSuitabilityColor,
  conditionEvidenceName,
  conditionEvidenceDetail,
  conditionRecommendationText,
  evidenceStatusText,
  evidenceStatusColor,
}: OperatingConditionPanelProps) {
  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">运行工况</div>
            <Typography.Text type="secondary">
              基于历史画像中的数据质量、饱和、振荡、SP/MV 活跃度和漂移，判断当前片段是否适合进入整定。
            </Typography.Text>
          </div>
          <Space>
            <Tag color="green">后端已接入</Tag>
            <Tag color={tuningSuitabilityColor(conditionProfile?.tuning_suitability)}>
              {tuningSuitabilityText(conditionProfile?.tuning_suitability)}
            </Tag>
          </Space>
        </div>
        {conditionProfile ? (
          <>
            <div className="industrial-kpi-grid compact">
              <div className="industrial-kpi-card">
                <span>工况判断</span>
                <strong>{operatingConditionText(conditionProfile.condition_label)}</strong>
              </div>
              <div className="industrial-kpi-card">
                <span>整定适宜性</span>
                <strong>{tuningSuitabilityText(conditionProfile.tuning_suitability)}</strong>
              </div>
              <div className="industrial-kpi-card">
                <span>判断置信度</span>
                <strong>{formatPercentValue(conditionProfile.confidence, 1)}</strong>
              </div>
              <div className="industrial-kpi-card">
                <span>知识来源</span>
                <strong>{ontologyStatusText(conditionProfile.ontology_context?.status)}</strong>
              </div>
            </div>
            <Table
              size="small"
              pagination={false}
              rowKey={(row) => String(row.name)}
              dataSource={conditionProfile.evidence ?? []}
              columns={[
                { title: '证据项', dataIndex: 'name', width: 180, render: (value: string) => conditionEvidenceName(value) },
                {
                  title: '状态',
                  dataIndex: 'status',
                  width: 110,
                  render: (value: string) => <Tag color={evidenceStatusColor(value)}>{evidenceStatusText(value)}</Tag>,
                },
                {
                  title: '数值',
                  dataIndex: 'value',
                  width: 120,
                  render: (value: unknown, row: { name?: string }) => {
                    if (typeof value !== 'number') return String(value ?? '-');
                    if (['data_quality', 'mv_saturation', 'oscillation', 'transition', 'excitation'].includes(String(row.name))) {
                      return formatPercentValue(value, 1);
                    }
                    return formatNumber(value, 3);
                  },
                },
                { title: '判断依据', dataIndex: 'detail', render: (value: string) => conditionEvidenceDetail(value) },
              ]}
            />
          </>
        ) : (
          <Empty description="暂无运行工况画像，请先选择回路或刷新历史画像。" />
        )}
      </section>
      {conditionProfile && (
        <section className="two-column-grid">
          <div className="agent-panel">
            <div className="panel-title">工况片段估算</div>
            <Table
              size="small"
              pagination={false}
              rowKey={(row) => String(row.label)}
              dataSource={conditionProfile.segment_summary ?? []}
              columns={[
                { title: '片段类型', dataIndex: 'label', render: (value: string) => operatingConditionText(value) },
                { title: '占比', dataIndex: 'ratio', width: 100, render: (value: number) => formatPercentValue(value, 1) },
                { title: '时长', dataIndex: 'duration_s', width: 120, render: (value: number) => `${formatNumber(value, 0)}s` },
                {
                  title: '可用于整定',
                  dataIndex: 'tuning_usable',
                  width: 120,
                  render: (value: boolean) => <Tag color={value ? 'green' : 'default'}>{value ? '可用' : '不建议'}</Tag>,
                },
              ]}
            />
          </div>
          <div className="agent-panel">
            <div className="panel-title">知识与建议</div>
            <Descriptions bordered column={1} size="small" className="industrial-descriptions">
              <Descriptions.Item label="知识来源">{ontologyStatusText(conditionProfile.ontology_context?.status)}</Descriptions.Item>
              <Descriptions.Item label="回路类型">{conditionProfile.ontology_context?.loop_type_hint || selectedLoop?.loop_type || '-'}</Descriptions.Item>
              <Descriptions.Item label="后续建议字段">
                {(conditionProfile.ontology_context?.requires_fields ?? []).join('、') || '-'}
              </Descriptions.Item>
            </Descriptions>
            {(conditionProfile.recommendations ?? []).length ? (
              <div className="compact-list">
                {(conditionProfile.recommendations ?? []).map((item) => (
                  <div className="compact-list-row" key={item}>
                    <span>{conditionRecommendationText(item)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <Alert type="info" showIcon message="当前工况暂无额外建议" />
            )}
          </div>
        </section>
      )}
    </div>
  );
}
