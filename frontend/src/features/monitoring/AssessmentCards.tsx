import { Alert, Empty, Progress, Tag } from 'antd';

import type { HistoryLoopAssessment } from '@/services/api';

interface AssessmentCardsProps {
  assessment: HistoryLoopAssessment | null;
  scorePercent: (value?: number) => number;
  scoreStatus: (value?: number) => 'success' | 'normal' | 'exception';
  tagColor: (level?: string) => string;
}

export function AssessmentCards({
  assessment,
  scorePercent,
  scoreStatus,
  tagColor,
}: AssessmentCardsProps) {
  if (!assessment) return <Empty description="暂无评估结果" />;

  return (
    <>
      <div className="score-grid">
        <div className="score-card">
          <Tag color={tagColor(assessment.performance?.level ?? assessment.readiness.level)}>
            {assessment.summary?.decision_text ?? assessment.performance?.level ?? '-'}
          </Tag>
          <div className="score-title">综合评估</div>
          <Progress
            percent={scorePercent(assessment.performance?.score ?? assessment.readiness.score)}
            status={scoreStatus(assessment.performance?.score ?? assessment.readiness.score)}
          />
        </div>
        <div className="score-card">
          <Tag color={tagColor(assessment.data_quality.level)}>{assessment.data_quality.level}</Tag>
          <div className="score-title">数据质量</div>
          <Progress percent={scorePercent(assessment.data_quality.score)} status={scoreStatus(assessment.data_quality.score)} />
        </div>
        <div className="score-card">
          <Tag color={tagColor(assessment.identifiability.level)}>{assessment.identifiability.level}</Tag>
          <div className="score-title">可辨识性</div>
          <Progress percent={scorePercent(assessment.identifiability.score)} status={scoreStatus(assessment.identifiability.score)} />
        </div>
        <div className="score-card">
          <Tag color={tagColor(assessment.readiness.level)}>{assessment.readiness.level}</Tag>
          <div className="score-title">整定就绪度</div>
          <Progress percent={scorePercent(assessment.readiness.score)} status={scoreStatus(assessment.readiness.score)} />
        </div>
      </div>
      {assessment.summary && (
        <Alert
          className="agent-alert"
          type={assessment.summary.decision === 'blocked' ? 'error' : assessment.summary.decision === 'ready' ? 'success' : 'warning'}
          showIcon
          message={assessment.summary.decision_text}
          description={assessment.summary.recommended_next_action_text}
        />
      )}
    </>
  );
}
