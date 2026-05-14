import { Alert, Descriptions, Progress, Space, Table, Tag, Typography } from 'antd';
import type { StrategyCandidate, TuningResult } from '@/types/tuning';

interface TuningTaskResultPanelsProps {
  result: TuningResult | null;
  formatNumber: (value?: number | null, digits?: number) => string;
  scoreColor: (score?: number) => string;
}

export function TuningTaskResultPanels({
  result,
  formatNumber,
  scoreColor,
}: TuningTaskResultPanelsProps) {
  if (!result?.pid_params) return null;

  const { pid_params: pidParams, evaluation } = result;

  return (
    <>
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">PID 参数结果</div>
            <Typography.Text type="secondary">推荐策略与所有候选策略对比，PB=100/Kp。</Typography.Text>
          </div>
          <Tag color="cyan">{pidParams.strategy}</Tag>
        </div>
        <Descriptions column={6} size="small" className="detail-block">
          <Descriptions.Item label="Kp">{formatNumber(pidParams.Kp, 4)}</Descriptions.Item>
          <Descriptions.Item label="PB">{pidParams.Kp > 0 ? `${formatNumber(100 / pidParams.Kp, 2)}%` : '-'}</Descriptions.Item>
          <Descriptions.Item label="Ki">{formatNumber(pidParams.Ki, 6)}</Descriptions.Item>
          <Descriptions.Item label="Kd">{formatNumber(pidParams.Kd, 4)}</Descriptions.Item>
          <Descriptions.Item label="Ti">{formatNumber(pidParams.Ti, 2)}s</Descriptions.Item>
          <Descriptions.Item label="Td">{formatNumber(pidParams.Td, 2)}s</Descriptions.Item>
        </Descriptions>
        {!!pidParams.candidates?.length && (
          <Table<StrategyCandidate>
            size="small"
            rowKey="strategy"
            dataSource={pidParams.candidates}
            pagination={false}
            columns={[
              { title: '策略', dataIndex: 'strategy', render: (value, row) => <Space><Tag color={row.is_recommended ? 'green' : 'blue'}>{value}</Tag>{row.is_recommended && <Tag color="gold">推荐</Tag>}</Space> },
              { title: 'Kp', dataIndex: 'Kp', render: (value) => formatNumber(value, 4) },
              { title: 'PB(%)', render: (_, row) => row.Kp > 0 ? formatNumber(100 / row.Kp, 2) : '-' },
              { title: 'Ki', dataIndex: 'Ki', render: (value) => formatNumber(value, 6) },
              { title: 'Kd', dataIndex: 'Kd', render: (value) => formatNumber(value, 4) },
              { title: 'Ti(s)', dataIndex: 'Ti', render: (value) => formatNumber(value, 2) },
              { title: 'Td(s)', dataIndex: 'Td', render: (value) => formatNumber(value, 2) },
              { title: '说明', dataIndex: 'description', ellipsis: true },
            ]}
          />
        )}
      </section>

      {evaluation && (
        <section className="agent-panel">
          <div className="panel-toolbar">
            <div>
              <div className="panel-title">性能评估</div>
              <Typography.Text type="secondary">闭环仿真、自检封顶和上线建议都集中在这里。</Typography.Text>
            </div>
            <Tag color={evaluation.passed ? 'green' : 'red'}>{evaluation.passed ? '可以上线' : '需要优化'}</Tag>
          </div>
          <div className="task-score-grid">
            {[
              ['性能评分', evaluation.performance_score],
              ['综合评分', evaluation.final_rating],
              ['就绪评分', evaluation.readiness_score],
              ['鲁棒评分', evaluation.robustness_score],
            ].map(([label, value]) => (
              <div key={label} className="task-score-card">
                <Progress
                  type="circle"
                  percent={Number(value) * 10}
                  format={() => formatNumber(Number(value), 1)}
                  strokeColor={scoreColor(Number(value))}
                  size={72}
                />
                <span>{label}</span>
              </div>
            ))}
          </div>
          <Descriptions column={4} size="small" className="detail-block">
            <Descriptions.Item label="稳定性">
              <Tag color={evaluation.is_stable ? 'green' : 'red'}>{evaluation.is_stable ? '稳定' : '不稳定'}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="超调量">{formatNumber(evaluation.overshoot_percent, 1)}%</Descriptions.Item>
            <Descriptions.Item label="调节时间">{formatNumber(evaluation.settling_time_s, 1)}s</Descriptions.Item>
            <Descriptions.Item label="稳态误差">{formatNumber(evaluation.steady_state_error, 2)}%</Descriptions.Item>
            <Descriptions.Item label="振荡次数">{evaluation.oscillation_count}</Descriptions.Item>
            <Descriptions.Item label="MV 饱和">{formatNumber(evaluation.mv_saturation_pct, 1)}%</Descriptions.Item>
            <Descriptions.Item label="仿真一致性评分">{formatNumber(evaluation.reality_check_score, 1)}</Descriptions.Item>
            <Descriptions.Item label="典型 T">{evaluation.reality_check_typical_T ? `${evaluation.reality_check_typical_T}s` : '-'}</Descriptions.Item>
          </Descriptions>
          {(evaluation.reality_check_diverged || !!evaluation.score_caps_applied?.length) && (
            <Alert
              className="agent-alert"
              type="error"
              showIcon
              message="评估自检触发"
              description={[
                evaluation.reality_check_diverged ? `仿真一致性检查认为名义模型与典型回路差异过大，评分 ${formatNumber(evaluation.reality_check_score, 1)}` : '',
                ...(evaluation.score_caps_applied ?? []),
              ].filter(Boolean).join('；')}
            />
          )}
          <Alert type={evaluation.passed ? 'success' : 'warning'} showIcon message={evaluation.recommendation} />
        </section>
      )}
    </>
  );
}
