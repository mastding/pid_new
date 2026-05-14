interface DashboardQuickActionsWidgetProps {
  onCreateTuningTask: () => void;
  onViewLoopProfile: () => void;
  onViewTrendSpectrum: () => void;
  onOpenDiagnosis: () => void;
}

export function DashboardQuickActionsWidget({
  onCreateTuningTask,
  onViewLoopProfile,
  onViewTrendSpectrum,
  onOpenDiagnosis,
}: DashboardQuickActionsWidgetProps) {
  return (
    <>
      <div className="cockpit-card-title">快捷操作</div>
      <button type="button" onClick={onCreateTuningTask}>新建整定任务</button>
      <button type="button" onClick={onViewLoopProfile}>查看回路画像</button>
      <button type="button" onClick={onViewTrendSpectrum}>趋势与频谱</button>
      <button type="button" onClick={onOpenDiagnosis}>进入诊断总览</button>
    </>
  );
}
