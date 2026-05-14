import { Alert, Descriptions, Empty, Progress, Tag, Typography } from 'antd';

import type { HistoryLoopAssessment } from '@/services/api';
import type { TuningResult } from '@/types/tuning';

interface PerformanceScorePanelProps {
  assessment: HistoryLoopAssessment | null;
  assessmentLoading: boolean;
  taskResult: TuningResult | null;
  formatNumber: (value?: number | null, digits?: number) => string;
  scorePercent: (value?: number) => number;
  tagColor: (level?: string) => string;
}

export function PerformanceScorePanel({
  assessment,
  assessmentLoading,
  taskResult,
  formatNumber,
  scorePercent,
  tagColor,
}: PerformanceScorePanelProps) {
  const historicalPerformance = assessment?.performance;
  const hasHistoricalPerformance = Boolean(
    historicalPerformance && (
      historicalPerformance.score !== undefined
      || historicalPerformance.monitoring_score !== undefined
      || historicalPerformance.stability_score !== undefined
      || historicalPerformance.constraint_score !== undefined
      || historicalPerformance.pv_mv_behavior_score !== undefined
    ),
  );

  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">控制性能</div>
            <Typography.Text type="secondary">
              评估当前回路控制效果，重点关注偏差、响应速度、MV动作量和约束影响。
            </Typography.Text>
          </div>
          <Tag color={taskResult?.evaluation?.passed ? 'green' : 'orange'}>
            {taskResult
              ? (taskResult.evaluation?.passed ? '可接受' : '需要优化')
              : assessmentLoading
                ? '加载评估'
                : historicalPerformance?.level || '等待评估'}
          </Tag>
        </div>
        {assessmentLoading ? (
          <Empty description="正在加载历史控制性能评估..." />
        ) : taskResult && taskResult.evaluation ? (
          <>
            <div className="task-score-grid">
              {[
                ['性能评分', taskResult.evaluation.performance_score],
                ['综合评分', taskResult.evaluation.final_rating],
                ['就绪评分', taskResult.evaluation.readiness_score],
                ['鲁棒评分', taskResult.evaluation.robustness_score],
              ].map(([label, value]) => (
                <div key={label} className="task-score-card">
                  <Progress
                    type="circle"
                    percent={Number(value) * 10}
                    format={() => formatNumber(Number(value), 1)}
                    strokeColor={Number(value) >= 8 ? '#22a06b' : Number(value) >= 6 ? '#f59e0b' : '#ef4444'}
                    size={72}
                  />
                  <span>{label}</span>
                </div>
              ))}
            </div>
            <Descriptions column={4} bordered size="small" className="detail-block industrial-descriptions">
              <Descriptions.Item label="稳定性">
                <Tag color={taskResult.evaluation.is_stable ? 'green' : 'red'}>{taskResult.evaluation.is_stable ? '稳定' : '不稳定'}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="超调量">{formatNumber(taskResult.evaluation.overshoot_percent, 1)}%</Descriptions.Item>
              <Descriptions.Item label="调节时间">{formatNumber(taskResult.evaluation.settling_time_s, 1)}s</Descriptions.Item>
              <Descriptions.Item label="稳态误差">{formatNumber(taskResult.evaluation.steady_state_error, 2)}%</Descriptions.Item>
              <Descriptions.Item label="振荡次数">{taskResult.evaluation.oscillation_count}</Descriptions.Item>
              <Descriptions.Item label="MV 饱和">{formatNumber(taskResult.evaluation.mv_saturation_pct, 1)}%</Descriptions.Item>
            </Descriptions>
          </>
        ) : hasHistoricalPerformance ? (
          <>
            <div className="task-score-grid">
              {[
                ['历史性能评分', historicalPerformance?.score],
                ['监控评分', historicalPerformance?.monitoring_score],
                ['稳定性', historicalPerformance?.stability_score],
                ['PV/MV 行为', historicalPerformance?.pv_mv_behavior_score],
                ['约束健康', historicalPerformance?.constraint_score],
              ].map(([label, value]) => (
                <div key={label as string} className="task-score-card">
                  <Progress
                    type="circle"
                    percent={value === undefined || value === null ? 0 : scorePercent(Number(value))}
                    format={() => value === undefined || value === null ? '-' : `${scorePercent(Number(value))}%`}
                    strokeColor={Number(value ?? 0) >= 0.8 ? '#22a06b' : Number(value ?? 0) >= 0.6 ? '#f59e0b' : '#ef4444'}
                    size={72}
                  />
                  <span>{label}</span>
                </div>
              ))}
            </div>
            <Descriptions column={3} bordered size="small" className="detail-block industrial-descriptions">
              <Descriptions.Item label="综合结论">
                {assessment?.summary?.decision_text ?? historicalPerformance?.level ?? '-'}
              </Descriptions.Item>
              <Descriptions.Item label="建议动作" span={2}>
                {assessment?.summary?.recommended_next_action_text ?? assessment?.summary?.recommended_next_action ?? '-'}
              </Descriptions.Item>
              <Descriptions.Item label="数据质量">
                <Tag color={tagColor(assessment?.data_quality?.level)}>{assessment?.data_quality?.level ?? '-'}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="可辨识性">
                <Tag color={tagColor(assessment?.identifiability?.level)}>{assessment?.identifiability?.level ?? '-'}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="整定准备度">
                <Tag color={tagColor(assessment?.readiness?.level)}>{assessment?.readiness?.level ?? '-'}</Tag>
              </Descriptions.Item>
            </Descriptions>
          </>
        ) : (
          <Alert
            type="info"
            showIcon
            message="当前暂无整定仿真性能结果"
            description="等待后端返回在线或历史控制性能评估。"
          />
        )}
      </section>
    </div>
  );
}
