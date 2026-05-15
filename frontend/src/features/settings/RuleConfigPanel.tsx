import { SyncOutlined } from '@ant-design/icons';
import { Alert, Button, Descriptions, Empty, Space, Table, Tag, Typography } from 'antd';

import type { PolicyConfig } from '@/services/api';

interface RuleConfigPanelProps {
  policyConfig: PolicyConfig | null;
  loading: boolean;
  onRefresh: () => void;
  loopTypeLabel: (loopType: string) => string;
  policyLoopImpact: (loopType: string) => string;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
}

function BackendBadge({ implemented }: { implemented?: boolean }) {
  return implemented ? <Tag color="green">已接后端</Tag> : <Tag color="default">未开放</Tag>;
}

export function RuleConfigPanel({
  policyConfig,
  loading,
  onRefresh,
  loopTypeLabel,
  policyLoopImpact,
  formatNumber,
  formatPercentValue,
}: RuleConfigPanelProps) {
  const loopTypes = Array.from(new Set([
    ...Object.keys(policyConfig?.loop_priors.model_order ?? {}),
    ...Object.keys(policyConfig?.refinement.model_fallbacks ?? {}),
  ]));

  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">规则配置</div>
            <Typography.Text type="secondary">
              只读展示当前后端实际生效的回路先验、辨识精修阈值和算法族备选模型池。
            </Typography.Text>
          </div>
          <Space wrap>
            <BackendBadge implemented />
            <Button icon={<SyncOutlined />} loading={loading} onClick={onRefresh}>刷新</Button>
          </Space>
        </div>
        {policyConfig ? (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Descriptions bordered size="small" column={4} className="industrial-descriptions">
              <Descriptions.Item label="最低置信度">{formatPercentValue(policyConfig.refinement.fallback_rule.min_confidence, 0)}</Descriptions.Item>
              <Descriptions.Item label="最低 R²">{formatNumber(policyConfig.refinement.fallback_rule.min_r2, 2)}</Descriptions.Item>
              <Descriptions.Item label="最低窗口质量">{formatPercentValue(policyConfig.refinement.fallback_rule.min_window_quality, 0)}</Descriptions.Item>
              <Descriptions.Item label="最大模型池">{policyConfig.refinement.fallback_rule.max_model_pool_size} 个</Descriptions.Item>
            </Descriptions>
            <Table
              size="small"
              pagination={false}
              rowKey="key"
              dataSource={[
                {
                  key: 'min_confidence',
                  item: '最低置信度',
                  value: formatPercentValue(policyConfig.refinement.fallback_rule.min_confidence, 0),
                  stage: '辨识精修',
                  impact: '大模型精修不可用或放弃时，候选算法族最佳结果达到该置信度，才允许确定性策略继续重试。',
                },
                {
                  key: 'min_r2',
                  item: '最低 R²',
                  value: formatNumber(policyConfig.refinement.fallback_rule.min_r2, 2),
                  stage: '辨识精修',
                  impact: '过滤明显不可解释的候选窗口算法族，避免低拟合质量结果驱动下一轮。',
                },
                {
                  key: 'min_window_quality',
                  item: '最低窗口质量',
                  value: formatPercentValue(policyConfig.refinement.fallback_rule.min_window_quality, 0),
                  stage: '窗口/辨识',
                  impact: '当 R² 或置信度还不充分时，允许高质量窗口作为探索性备选进入下一轮辨识。',
                },
                {
                  key: 'max_model_pool_size',
                  item: '最大模型池',
                  value: `${policyConfig.refinement.fallback_rule.max_model_pool_size} 个`,
                  stage: '辨识精修',
                  impact: '限制下一轮强制模型数量，避免模型池过宽导致结果不可解释。',
                },
              ]}
              columns={[
                { title: '规则项', dataIndex: 'item', width: 160 },
                { title: '当前值', dataIndex: 'value', width: 120 },
                { title: '影响链路', dataIndex: 'stage', width: 140, render: (value: string) => <Tag color="blue">{value}</Tag> },
                { title: '规则用途', dataIndex: 'impact' },
              ]}
            />
          </Space>
        ) : (
          <Empty description={loading ? '正在加载规则配置' : '暂无规则配置'} />
        )}
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">按回路类型的模型与时间常数先验</div>
            <Typography.Text type="secondary">
              辨识阶段按默认模型顺序尝试；精修阶段在大模型不可用时使用备选模型池做保守重试。
            </Typography.Text>
          </div>
          <Tag color="blue">{loopTypes.length} 类回路</Tag>
        </div>
        <Table
          size="small"
          loading={loading}
          pagination={false}
          rowKey="loop_type"
          dataSource={loopTypes.map((loopType) => ({
            loop_type: loopType,
            label: loopTypeLabel(loopType),
            model_order: policyConfig?.loop_priors.model_order?.[loopType] ?? [],
            refinement_models: policyConfig?.refinement.model_fallbacks?.[loopType] ?? [],
            min_t: policyConfig?.loop_priors.min_reasonable_t?.[loopType],
            reality_range: policyConfig?.loop_priors.reality_t_ranges?.[loopType],
            impact: policyLoopImpact(loopType),
          }))}
          columns={[
            { title: '回路类型', dataIndex: 'label', width: 120 },
            { title: '辨识模型顺序', dataIndex: 'model_order', render: (models: string[]) => <Space wrap>{models.map((item) => <Tag color="blue" key={item}>{item}</Tag>)}</Space> },
            { title: '精修备选模型池', dataIndex: 'refinement_models', render: (models: string[]) => <Space wrap>{models.map((item) => <Tag color="cyan" key={item}>{item}</Tag>)}</Space> },
            { title: 'T下界(s)', dataIndex: 'min_t', width: 110, render: (value: number | undefined) => formatNumber(value, 0) },
            {
              title: '现实时间常数范围(s)',
              dataIndex: 'reality_range',
              width: 160,
              render: (value?: { min: number; max: number }) => value ? `${formatNumber(value.min, 0)} ~ ${formatNumber(value.max, 0)}` : '-',
            },
            { title: '规则用途/影响链路', dataIndex: 'impact', width: 360 },
          ]}
        />
      </section>

      <section className="agent-panel">
        <div className="panel-title">配置说明</div>
        <Alert
          className="agent-alert"
          type="info"
          showIcon
          message="当前为只读版本"
          description="当前展示后端运行时策略配置；编辑能力需补充持久化、校验、审计和灰度生效流程。"
        />
      </section>
    </div>
  );
}
