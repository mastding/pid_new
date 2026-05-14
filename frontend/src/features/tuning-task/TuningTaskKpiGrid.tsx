import type { ReactNode } from 'react';

interface TuningTaskKpiGridProps {
  candidateWindowCount: ReactNode;
  usableWindowCount: ReactNode;
  modelType: ReactNode;
  r2Score: ReactNode;
  strategy: ReactNode;
  kp: ReactNode;
  finalScore: ReactNode;
  evaluationText: ReactNode;
}

export function TuningTaskKpiGrid({
  candidateWindowCount,
  usableWindowCount,
  modelType,
  r2Score,
  strategy,
  kp,
  finalScore,
  evaluationText,
}: TuningTaskKpiGridProps) {
  return (
    <div className="task-kpi-grid">
      <div className="task-kpi-card">
        <span>候选窗口</span>
        <strong>{candidateWindowCount}</strong>
        <em>可用 {usableWindowCount} 个</em>
      </div>
      <div className="task-kpi-card">
        <span>辨识模型</span>
        <strong>{modelType}</strong>
        <em>R² {r2Score}</em>
      </div>
      <div className="task-kpi-card">
        <span>推荐策略</span>
        <strong>{strategy}</strong>
        <em>Kp {kp}</em>
      </div>
      <div className="task-kpi-card">
        <span>综合评分</span>
        <strong>{finalScore}</strong>
        <em>{evaluationText}</em>
      </div>
    </div>
  );
}
