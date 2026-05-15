export function scorePercent(value?: number) {
  return Math.round((value ?? 0) * 100);
}

export function scoreStatus(value?: number) {
  if ((value ?? 0) < 0.4) return 'exception';
  if ((value ?? 0) >= 0.75) return 'success';
  return 'normal';
}

export function tagColor(level?: string) {
  if (level === 'excellent') return 'green';
  if (level === 'good') return 'blue';
  if (level === 'fair') return 'orange';
  return 'red';
}

export function gateSeverityColor(severity?: string) {
  if (severity === 'critical' || severity === 'high' || severity === 'error' || severity === 'blocked') return 'red';
  if (severity === 'medium' || severity === 'warning') return 'orange';
  if (severity === 'ok' || severity === 'low') return 'green';
  return 'default';
}

export function gateCheckLabel(value?: string) {
  if (value === 'data_quality') return '数据质量';
  if (value === 'operating_condition') return '运行工况';
  if (value === 'constraints') return '约束/饱和';
  if (value === 'oscillation') return '振荡状态';
  if (value === 'identification') return '可辨识性';
  return value || '-';
}

export function gateImpact(
  check: { passed?: boolean; severity?: string; name?: string },
  blockingReasons: Array<{ type: string; severity: string }>,
) {
  const severity = String(check.severity || '');
  if (!check.passed || ['critical', 'high', 'error', 'blocked'].includes(severity)) {
    return { text: '硬阻断', color: 'red' };
  }
  const hasSoftReason = blockingReasons.some((reason) => {
    const reasonType = reason.type === 'constraint' ? 'constraints' : reason.type;
    return reasonType === check.name && ['medium', 'warning', 'low', 'info'].includes(String(reason.severity));
  });
  if (hasSoftReason || ['medium', 'warning'].includes(severity)) {
    return { text: '软提醒', color: 'orange' };
  }
  return { text: '无影响', color: 'green' };
}

export function gateCheckMessage(
  check: { name?: string; message?: string; evidence?: Record<string, unknown> },
  blockingReasons: Array<{ type: string; severity: string; message: string }>,
) {
  const softReason = blockingReasons.find((reason) => {
    const reasonType = reason.type === 'constraint' ? 'constraints' : reason.type;
    return reasonType === check.name && ['medium', 'warning', 'low', 'info'].includes(String(reason.severity));
  });
  if (check.name === 'operating_condition' && softReason) {
    return softReason.message;
  }
  return check.message || '-';
}

export function gateDecisionText(decision?: string) {
  if (decision === 'ready') return '可发起整定';
  if (decision === 'caution') return '谨慎整定';
  if (decision === 'blocked') return '暂不建议整定';
  return decision || '待评估';
}

export function monitoringStatusColor(status?: string) {
  if (status === 'normal') return 'green';
  if (status === 'warning') return 'orange';
  if (status === 'alarm' || status === 'critical') return 'red';
  if (status === 'unavailable') return 'default';
  return 'blue';
}

export function monitoringStatusText(status?: string) {
  if (status === 'normal') return '正常';
  if (status === 'warning') return '关注';
  if (status === 'alarm') return '报警';
  if (status === 'critical') return '严重';
  if (status === 'unavailable') return '不可用';
  return status || '-';
}

export function alertSeverityColor(severity?: string) {
  if (severity === 'critical' || severity === 'high' || severity === '高') return 'red';
  if (severity === 'warning' || severity === 'medium' || severity === '中') return 'orange';
  return 'blue';
}

export function formatNumber(value?: number | null, digits = 2) {
  return value === null || value === undefined || Number.isNaN(value) ? '-' : value.toFixed(digits);
}

export function formatRange(min?: number | null, max?: number | null, digits = 2) {
  return `${formatNumber(min, digits)} ~ ${formatNumber(max, digits)}`;
}

export function formatPercentValue(value?: number | null, digits = 0) {
  return value === null || value === undefined || Number.isNaN(value) ? '-' : `${(value * 100).toFixed(digits)}%`;
}

export function formatOscillationEvidence(detected?: boolean, confidence?: number | null) {
  if (!detected) return '无显著周期峰';
  return formatPercentValue(confidence, 1);
}

export function formatOscillationPhaseHint(detected?: boolean, phaseHint?: string | null) {
  if (!detected) return '未判定';
  if (phaseHint === 'pv_mv_same_period') return 'PV/MV 同周期';
  if (phaseHint === 'pv_only_periodic') return 'PV 单侧周期';
  if (phaseHint === 'unknown' || !phaseHint) return '证据不足';
  return phaseHint;
}

export function formatHarrisBasis(value?: string) {
  if (value === 'pv_minus_sp') return 'PV-SV 跟踪误差';
  if (value === 'pv_minus_constant_sp') return 'PV-固定SV偏差';
  if (value === 'detrended_pv') return '去趋势 PV 波动';
  return value || '-';
}

export function formatCpkBasis(value?: string) {
  if (value === 'pv_spec_limits') return 'PV规格上下限';
  if (value === 'missing_pv_spec_limits') return '未配置PV规格上下限';
  return value || '-';
}

export function formatProcessDirection(direction?: string | null) {
  if (direction === 'positive_gain' || direction === 'positive') return '正作用（MV↑ PV↑）';
  if (direction === 'negative_gain' || direction === 'negative') return '反作用（MV↑ PV↓）';
  return '不确定';
}

export function formatProcessDirectionBasis(basis?: string | null) {
  if (basis === 'dmv_to_dpv_lag_corr') return 'MV/PV 变化量滞后相关';
  if (basis === 'mv_to_pv_lag_corr') return 'MV/PV 水平值滞后相关';
  return basis || '-';
}

export function policyLoopImpact(loopType: string) {
  const labelMap: Record<string, string> = {
    flow: '流量',
    temperature: '温度',
    pressure: '压力',
    level: '液位',
    unknown: '未知',
  };
  const label = labelMap[loopType] ?? loopType;
  return `${label}回路：辨识阶段按模型顺序扩大搜索；精修阶段在模型服务不可用时按备选模型池重试；时间常数下界约束优化器搜索空间，现实时间常数范围影响整定后仿真评分。`;
}

export function yesNo(value?: boolean | null) {
  if (value === true) return '是';
  if (value === false) return '否';
  return '-';
}

export function operatingConditionText(label?: string) {
  if (label === 'stable_production') return '稳定生产';
  if (label === 'load_change') return '负荷/工况切换';
  if (label === 'disturbance_recovery') return '扰动恢复';
  if (label === 'constraint_limited') return '约束受限';
  if (label === 'oscillatory') return '存在振荡';
  if (label === 'data_unreliable') return '数据不可靠';
  if (label === 'transition_or_load_change') return '过渡/负荷变化';
  if (label === 'data_quality_issue') return '数据质量问题';
  return '未判定';
}

export function tuningSuitabilityText(value?: string) {
  if (value === 'suitable') return '适合整定';
  if (value === 'cautious') return '谨慎整定';
  if (value === 'not_recommended') return '不建议整定';
  return '未判定';
}

export function tuningSuitabilityColor(value?: string) {
  if (value === 'suitable') return 'green';
  if (value === 'cautious') return 'orange';
  if (value === 'not_recommended') return 'red';
  return 'default';
}

export function evidenceStatusText(value?: string) {
  if (value === 'normal') return '正常';
  if (value === 'warning') return '关注';
  if (value === 'alarm') return '异常';
  return value || '-';
}

export function evidenceStatusColor(value?: string) {
  if (value === 'normal') return 'green';
  if (value === 'warning') return 'orange';
  if (value === 'alarm') return 'red';
  return 'default';
}

export function conditionEvidenceName(name?: string) {
  if (name === 'data_quality') return '数据质量';
  if (name === 'mv_saturation') return 'MV 饱和/约束';
  if (name === 'oscillation') return '振荡证据';
  if (name === 'transition') return '均值漂移/过渡';
  if (name === 'excitation') return '激励充分性';
  return name || '-';
}

export function conditionEvidenceDetail(detail?: string) {
  if (detail === 'missing_or_irregular_sample_ratio') return '缺失率或采样不规则比例';
  if (detail === 'mv_near_observed_or_percent_limits') return 'MV 接近观测上下限或百分比边界';
  if (detail === 'first_second_half_mean_shift_and_sp_activity') return '前后半段均值漂移与 SP 活跃度';
  if (detail === 'good') return '激励较充分';
  if (detail === 'fair') return '激励一般';
  if (detail === 'poor') return '激励不足';
  if (detail === 'pv_mv_same_period') return 'PV/MV 存在同周期迹象';
  if (detail === 'pv_only_periodic') return 'PV 单侧周期迹象';
  if (detail === 'unknown') return '证据不足';
  return detail || '-';
}

export function conditionRecommendationText(value?: string) {
  if (value === 'fix_data_quality_before_assessment') return '先处理缺失、断点或采样异常，再做评估。';
  if (value === 'exclude_saturated_periods_or_check_valve_capacity') return '剔除饱和片段或先确认阀门/执行机构能力。';
  if (value === 'run_oscillation_diagnosis_before_tuning') return '先做振荡诊断，避免把振荡误当成可辨识激励。';
  if (value === 'prefer_steady_segments_for_identification') return '优先选择稳定片段做辨识，过渡段只作工况参考。';
  if (value === 'need_more_mv_excitation_for_identification') return '当前 MV 激励不足，建议补充可控小阶跃或等待更充分历史片段。';
  if (value === 'condition_is_acceptable_for_candidate_tuning') return '当前工况可进入候选整定评估。';
  return value || '-';
}
