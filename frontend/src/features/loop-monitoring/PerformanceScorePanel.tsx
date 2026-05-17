import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Dayjs } from 'dayjs';
import {
  Alert,
  Button,
  DatePicker,
  Empty,
  Progress,
  Select,
  Space,
  Steps,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd';
import { ReloadOutlined } from '@ant-design/icons';

import {
  fetchHistoryLoopCpk,
  fetchHistoryLoopHarris,
  fetchHistoryLoopOntologyRules,
  type HistoryLoop,
  type HistoryLoopCpk,
  type HistoryLoopHarris,
  type HistoryLoopOntologyRules,
  type HistoryTimeRangeParams,
} from '@/services/api';

type FeatureRangeOption = {
  label: string;
  value: string;
  seconds?: number;
};

type PerformanceMetric = 'harris' | 'cpk';

interface PerformanceScorePanelProps {
  selectedLoop?: HistoryLoop;
  selectedLoopId?: string;
  scopedLoops: HistoryLoop[];
  featureRangePreset: string;
  featureCustomRange: [Dayjs | null, Dayjs | null] | null;
  featureRangeOptions: FeatureRangeOption[];
  buildFeatureRangeParams: (loop?: HistoryLoop) => HistoryTimeRangeParams;
  onLoopChange: (loopId: string) => void;
  onRangePresetChange: (value: string) => void;
  onCustomRangeChange: (value: [Dayjs | null, Dayjs | null] | null) => void;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  loopTypeLabel: (loop: HistoryLoop) => string;
  tagColor: (level?: string) => string;
}

const levelText: Record<string, string> = {
  excellent: '优秀',
  good: '良好',
  fair: '一般',
  poor: '较差',
  weak: '偏弱',
  unavailable: '不可用',
};

const errorBasisText: Record<string, string> = {
  auto: '自动选择',
  deviation_from_sp: 'PV-SP 跟踪误差',
  detrended_pv: '去趋势 PV 波动',
  differenced_pv: 'PV 差分波动',
};

const deadtimeSourceText: Record<string, string> = {
  identified_model: '已辨识模型',
  cross_correlation: 'PV/MV 互相关',
  pv_acf: 'PV 自相关',
  user_override: '人工指定',
};

const riskFlagText: Record<string, string> = {
  deadtime_uncertain: '死时间不确定',
  non_white_residual: 'AR 残差非白',
  sp_change_dominates: '设定值变化主导',
  noise_elevated: '噪声偏高',
  noise_dominated: '噪声主导',
  integrating_loop_handled: '积分对象已差分处理',
  ar_order_limit_bumped: 'AR 阶数触及上限',
  data_too_short: '样本不足',
};

const cpkReasonText: Record<string, string> = {
  ok: '计算完成',
  insufficient_pv_samples: 'PV 有效样本不足',
  missing_pv_spec_limits: '缺少 PV 规格上下限',
  invalid_pv_spec_limits: 'PV 规格上下限无效',
  zero_pv_variance: 'PV 波动接近 0',
};

function harrisColor(level?: string, eta?: number | null) {
  if (level === 'excellent' || Number(eta ?? 0) >= 0.8) return '#22c55e';
  if (level === 'good' || Number(eta ?? 0) >= 0.6) return '#3b82f6';
  if (level === 'fair' || Number(eta ?? 0) >= 0.4) return '#f59e0b';
  return '#ef4444';
}

function cpkColor(level?: string, value?: number | null) {
  if (level === 'unavailable' || value === null || value === undefined) return '#94a3b8';
  if (level === 'excellent' || value >= 1.67) return '#22c55e';
  if (level === 'good' || value >= 1.33) return '#3b82f6';
  if (level === 'fair' || value >= 1.0) return '#f59e0b';
  return '#ef4444';
}

function boolText(value?: boolean) {
  if (value === true) return '建议复核整定';
  if (value === false) return '暂不建议';
  return '-';
}

function levelLabel(level?: string) {
  return level ? levelText[level] || level : '-';
}

function ruleStatusColor(status?: string) {
  if (status === 'pass') return 'green';
  if (status === 'blocked') return 'red';
  if (status === 'warn') return 'orange';
  return 'blue';
}

function ruleStatusLabel(status?: string) {
  if (status === 'pass') return '通过';
  if (status === 'blocked') return '阻断风险';
  if (status === 'warn') return '需复核';
  if (status === 'unknown') return '缺少事实';
  return status || '待评估';
}

function decisionLabel(decision?: string) {
  if (decision === 'blocked') return '存在阻断风险';
  if (decision === 'review_required') return '需要工程复核';
  if (decision === 'pass') return '通过';
  return decision || '待评估';
}

function compactEvidence(value: unknown) {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function PerformanceScorePanel({
  selectedLoop,
  selectedLoopId,
  scopedLoops,
  featureRangePreset,
  featureCustomRange,
  featureRangeOptions,
  buildFeatureRangeParams,
  onLoopChange,
  onRangePresetChange,
  onCustomRangeChange,
  formatNumber,
  formatPercentValue,
  loopTypeLabel,
  tagColor,
}: PerformanceScorePanelProps) {
  const [activeMetric, setActiveMetric] = useState<PerformanceMetric>('harris');
  const [harrisResult, setHarrisResult] = useState<HistoryLoopHarris | null>(null);
  const [harrisLoading, setHarrisLoading] = useState(false);
  const [harrisError, setHarrisError] = useState<string | null>(null);
  const [cpkResult, setCpkResult] = useState<HistoryLoopCpk | null>(null);
  const [cpkLoading, setCpkLoading] = useState(false);
  const [cpkError, setCpkError] = useState<string | null>(null);
  const [ruleResult, setRuleResult] = useState<HistoryLoopOntologyRules | null>(null);
  const [ruleLoading, setRuleLoading] = useState(false);

  const rangeParams = useMemo(
    () => buildFeatureRangeParams(selectedLoop),
    [buildFeatureRangeParams, selectedLoop, featureRangePreset, featureCustomRange],
  );

  const rangeLabel = useMemo(() => {
    if (rangeParams.start_time || rangeParams.end_time) {
      return `${rangeParams.start_time ?? '-'} ~ ${rangeParams.end_time ?? '-'}`;
    }
    return '全部历史数据';
  }, [rangeParams.end_time, rangeParams.start_time]);

  const loadHarris = useCallback(async () => {
    if (!selectedLoopId) {
      setHarrisResult(null);
      setHarrisError(null);
      return;
    }
    setHarrisLoading(true);
    setHarrisError(null);
    try {
      const resp = await fetchHistoryLoopHarris(selectedLoopId, rangeParams);
      if (resp.error) {
        setHarrisError(resp.error);
        message.warning(resp.error);
      }
      setHarrisResult(resp);
    } catch (error) {
      const text = String(error);
      setHarrisError(text);
      message.error(`加载 Harris 性能评估失败：${text}`);
    } finally {
      setHarrisLoading(false);
    }
  }, [rangeParams, selectedLoopId]);

  const loadCpk = useCallback(async (forceRefreshOntology = false) => {
    if (!selectedLoopId) {
      setCpkResult(null);
      setCpkError(null);
      return;
    }
    setCpkLoading(true);
    setCpkError(null);
    try {
      const resp = await fetchHistoryLoopCpk(selectedLoopId, {
        ...rangeParams,
        refresh_ontology: forceRefreshOntology,
      });
      if (resp.error) {
        setCpkError(resp.error);
        message.warning(resp.error);
      }
      setCpkResult(resp);
    } catch (error) {
      const text = String(error);
      setCpkError(text);
      message.error(`加载 CPK 指标失败：${text}`);
    } finally {
      setCpkLoading(false);
    }
  }, [rangeParams, selectedLoopId]);

  const loadOntologyRules = useCallback(async (forceRefreshOntology = false) => {
    if (!selectedLoopId) {
      setRuleResult(null);
      return;
    }
    setRuleLoading(true);
    try {
      const resp = await fetchHistoryLoopOntologyRules(selectedLoopId, {
        ...rangeParams,
        refresh_ontology: forceRefreshOntology,
      });
      setRuleResult(resp);
    } catch {
      setRuleResult(null);
    } finally {
      setRuleLoading(false);
    }
  }, [rangeParams, selectedLoopId]);

  const loadMetrics = useCallback((forceRefreshOntology = false) => {
    void loadHarris();
    void loadCpk(forceRefreshOntology);
    void loadOntologyRules(forceRefreshOntology);
  }, [loadCpk, loadHarris, loadOntologyRules]);

  useEffect(() => {
    loadMetrics(false);
  }, [loadMetrics]);

  const formalHarris = harrisResult?.harris;
  const eta = formalHarris?.eta ?? null;
  const confidence = formalHarris?.confidence ?? null;
  const harrisLevel = formalHarris?.level;
  const riskFlags = formalHarris?.risk_flags ?? [];
  const harrisPercent = eta === null || eta === undefined ? 0 : Math.round(Math.max(0, Math.min(1, eta)) * 100);
  const confidencePercent = confidence === null || confidence === undefined
    ? 0
    : Math.round(Math.max(0, Math.min(1, confidence)) * 100);

  const cpk = cpkResult?.cpk;
  const cpkValue = cpk?.value ?? null;
  const cpkLevel = cpk?.level;
  const cpkPercent = cpkValue === null || cpkValue === undefined
    ? 0
    : Math.round(Math.max(0, Math.min(1, cpkValue / 1.67)) * 100);
  const activeLoading = activeMetric === 'harris' ? harrisLoading : cpkLoading;
  const activeError = activeMetric === 'harris' ? harrisError : cpkError;

  const harrisProcessSteps = useMemo(() => [
    {
      title: '数据窗口',
      description: `${formalHarris?.data_window?.n_samples ?? '-'} 点，${formatNumber(formalHarris?.data_window?.duration_s, 1)} s`,
    },
    {
      title: '误差信号',
      description: errorBasisText[formalHarris?.error_basis || ''] || formalHarris?.error_basis || '-',
    },
    {
      title: '死时间估计',
      description: `${formatNumber(formalHarris?.deadtime_used?.value_s, 1)} s / ${formalHarris?.deadtime_used?.samples ?? '-'} 点`,
    },
    {
      title: 'AR 残差模型',
      description: `阶数 ${formalHarris?.ar_model?.order ?? '-'}，残差 ACF ${formatNumber(formalHarris?.ar_model?.residual_acf_lag1, 3)}`,
    },
    {
      title: '方差分解',
      description: `最小方差 ${formatNumber(formalHarris?.variance_decomp?.sigma2_mv, 6)} / 实际方差 ${formatNumber(formalHarris?.variance_decomp?.sigma2_y, 6)}`,
    },
  ], [formalHarris, formatNumber]);

  const cpkProcessSteps = useMemo(() => [
    {
      title: '数据窗口',
      description: `${cpkResult?.data_window?.n_samples ?? '-'} 点，${formatNumber(cpkResult?.data_window?.duration_s, 1)} s`,
    },
    {
      title: 'PV 统计量',
      description: `均值 ${formatNumber(cpkResult?.statistics?.mean, 4)}，样本标准差 ${formatNumber(cpkResult?.statistics?.std, 4)}`,
    },
    {
      title: '规格上下限',
      description: `LSL ${formatNumber(cpkResult?.limits?.lsl, 4)}，USL ${formatNumber(cpkResult?.limits?.usl, 4)}`,
    },
    {
      title: '单边能力',
      description: `CPL ${formatNumber(cpk?.cpl, 4)}，CPU ${formatNumber(cpk?.cpu, 4)}`,
    },
    {
      title: 'CPK 结果',
      description: cpkValue === null || cpkValue === undefined
        ? cpkReasonText[cpk?.reason || ''] || cpk?.reason || '暂不可用'
        : `CPK ${formatNumber(cpkValue, 4)}，等级 ${levelLabel(cpkLevel)}`,
    },
  ], [cpk, cpkLevel, cpkResult, cpkValue, formatNumber]);

  const calculationSteps = useMemo(() => {
    if (activeMetric === 'cpk') {
      return [
        {
          title: '1. 选定数据窗口',
          description: `回路 ${selectedLoopId || '-'}，类型 ${selectedLoop ? loopTypeLabel(selectedLoop) : '-'}，CPK 只使用该时间范围内的 PV 序列：${rangeLabel}。`,
        },
        {
          title: '2. 计算 PV 统计量',
          description: `有效样本 ${cpkResult?.data_window?.n_samples ?? '-'} 个，均值 μ=${formatNumber(cpkResult?.statistics?.mean, 6)}，样本标准差 σ=${formatNumber(cpkResult?.statistics?.std, 6)}。`,
        },
        {
          title: '3. 读取 PV 规格上下限',
          description: `后端从 PV_LSL/PV_LL/PV_LOW/LSL 与 PV_USL/PV_HH/PV_HIGH/USL 等规格列读取：LSL=${formatNumber(cpkResult?.limits?.lsl, 6)}，USL=${formatNumber(cpkResult?.limits?.usl, 6)}。`,
        },
        {
          title: '4. 计算 CPL',
          description: `CPL=(μ-LSL)/(3σ)=${formatNumber(cpk?.cpl, 6)}，表示过程均值距离下规格限的能力裕量。`,
        },
        {
          title: '5. 计算 CPU',
          description: `CPU=(USL-μ)/(3σ)=${formatNumber(cpk?.cpu, 6)}，表示过程均值距离上规格限的能力裕量。`,
        },
        {
          title: '6. 得到 CPK',
          description: cpkValue === null || cpkValue === undefined
            ? `CPK=min(CPL,CPU)，当前未输出正式结果：${cpkReasonText[cpk?.reason || ''] || cpk?.reason || '原因未知'}。`
            : `CPK=min(CPL,CPU)=${formatNumber(cpkValue, 6)}，等级 ${levelLabel(cpkLevel)}。`,
        },
      ];
    }

    return [
      {
        title: '1. 选定数据窗口',
        description: `回路 ${selectedLoopId || '-'}，类型 ${selectedLoop ? loopTypeLabel(selectedLoop) : '-'}，Harris 只使用该时间范围内的 PV/SP/MV 序列：${rangeLabel}。`,
      },
      {
        title: '2. 构造误差信号',
        description: `后端按 error_basis=${errorBasisText[formalHarris?.error_basis || ''] || formalHarris?.error_basis || 'auto'} 选择 PV-SP、去趋势 PV 或差分 PV 作为闭环误差信号。`,
      },
      {
        title: '3. 估计死时间',
        description: `从辨识模型、PV/MV 互相关或 PV 自相关中确定死时间：${formatNumber(formalHarris?.deadtime_used?.value_s, 1)} s / ${formalHarris?.deadtime_used?.samples ?? '-'} 点。`,
      },
      {
        title: '4. 建立 AR 残差模型',
        description: `对误差信号建立 AR 模型，当前阶数 ${formalHarris?.ar_model?.order ?? '-'}，残差 ACF ${formatNumber(formalHarris?.ar_model?.residual_acf_lag1, 3)}。`,
      },
      {
        title: '5. 方差分解',
        description: `计算实际误差方差 σ²_y=${formatNumber(formalHarris?.variance_decomp?.sigma2_y, 6)}，最小可达方差 σ²_mv=${formatNumber(formalHarris?.variance_decomp?.sigma2_mv, 6)}。`,
      },
      {
        title: '6. 得到 Harris η',
        description: `η=σ²_mv/σ²_y=${eta === null || eta === undefined ? '-' : formatNumber(eta, 4)}，置信度 ${confidence === null || confidence === undefined ? '-' : `${confidencePercent}%`}，等级 ${levelLabel(harrisLevel)}。`,
      },
    ];
  }, [
    activeMetric,
    confidence,
    confidencePercent,
    cpk,
    cpkLevel,
    cpkResult,
    cpkValue,
    eta,
    formalHarris,
    formatNumber,
    harrisLevel,
    loopTypeLabel,
    rangeLabel,
    selectedLoop,
    selectedLoopId,
  ]);

  const metricCards = [
    {
      key: 'harris' as const,
      title: 'Harris 指标',
      value: eta === null || eta === undefined ? '-' : formatNumber(eta, 4),
      meta: harrisLevel ? levelLabel(harrisLevel) : harrisLoading ? '计算中' : '待计算',
      color: harrisColor(harrisLevel, eta),
    },
    {
      key: 'cpk' as const,
      title: 'CPK 指标',
      value: cpkValue === null || cpkValue === undefined ? '-' : formatNumber(cpkValue, 4),
      meta: cpkLevel ? levelLabel(cpkLevel) : cpkLoading ? '计算中' : '待计算',
      color: cpkColor(cpkLevel, cpkValue),
    },
  ];

  return (
    <div className="page-stack performance-page">
      <section className="agent-panel performance-hero-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">控制性能</div>
            <Typography.Text type="secondary">
              本页只展示 Harris 最小方差性能指标和 CPK 过程能力指标，选中指标后查看对应公式、输入数据和计算过程。
            </Typography.Text>
          </div>
          <Space>
            {selectedLoopId && <Tag color="blue">{selectedLoopId}</Tag>}
            <Tag color={activeMetric === 'harris' ? tagColor(harrisLevel) : tagColor(cpkLevel)}>
              {activeMetric === 'harris' ? `Harris ${levelLabel(harrisLevel)}` : `CPK ${levelLabel(cpkLevel)}`}
            </Tag>
          </Space>
        </div>

        <div className="performance-controls">
          <Select
            showSearch
            size="small"
            className="performance-loop-select"
            placeholder="选择回路"
            value={selectedLoopId}
            onChange={onLoopChange}
            optionFilterProp="label"
            options={scopedLoops.map((loop) => ({
              value: loop.loop_id,
              label: `${loop.loop_id} · ${loopTypeLabel(loop)}`,
            }))}
          />
          <Select
            size="small"
            className="performance-range-select"
            value={featureRangePreset}
            onChange={onRangePresetChange}
            options={featureRangeOptions.map((item) => ({ label: item.label, value: item.value }))}
          />
          {featureRangePreset === 'custom' && (
            <DatePicker.RangePicker
              size="small"
              showTime
              value={featureCustomRange}
              onChange={onCustomRangeChange}
            />
          )}
          <Button
            size="small"
            icon={<ReloadOutlined />}
            loading={harrisLoading || cpkLoading}
            onClick={() => loadMetrics(true)}
          >
            重新计算
          </Button>
          <Tag color="blue">{rangeLabel}</Tag>
        </div>

        <div className="performance-metric-switch">
          {metricCards.map((item) => (
            <button
              key={item.key}
              type="button"
              className={`metric-switch-card${activeMetric === item.key ? ' active' : ''}`}
              onClick={() => setActiveMetric(item.key)}
            >
              <span className="metric-switch-dot" style={{ background: item.color }} />
              <span>
                <strong>{item.title}</strong>
                <em>{item.meta}</em>
              </span>
              <b>{item.value}</b>
            </button>
          ))}
        </div>

        {ruleLoading ? (
          <Alert type="info" showIcon message="正在读取本体规则评估..." />
        ) : ruleResult?.rules?.length ? (
          <div className="performance-rule-strip">
            <div className="performance-rule-strip-title">
              <strong>本体规则评估</strong>
              <Tag color={ruleResult.summary?.decision === 'blocked' ? 'red' : 'blue'}>
                {decisionLabel(ruleResult.summary?.decision)}
              </Tag>
              {ruleResult.summary?.advisory_only && <Tag color="default">仅作评审提示</Tag>}
              {ruleResult.ontology_facts && (
                <Tag color="default">
                  {((ruleResult.ontology_facts.cache as { hit?: boolean } | undefined)?.hit) ? '本体缓存' : '本体新查'}
                </Tag>
              )}
            </div>
            <Space wrap>
              {ruleResult.rules.slice(0, 6).map((rule) => (
                <Tooltip key={rule.rule_id} title={rule.action || rule.title}>
                  <Tag color={ruleStatusColor(rule.status)}>
                    {rule.title}：{rule.status}
                  </Tag>
                </Tooltip>
              ))}
            </Space>
            <div className="performance-rule-grid">
              {ruleResult.rules.map((rule) => (
                <details key={rule.rule_id} className={`performance-rule-item status-${rule.status}`}>
                  <summary>
                    <span>
                      <b>{rule.title}</b>
                      <em>{rule.rule_id}</em>
                    </span>
                    <Tag color={ruleStatusColor(rule.status)}>{ruleStatusLabel(rule.status)}</Tag>
                  </summary>
                  <div className="performance-rule-detail">
                    {rule.action && <p>{rule.action}</p>}
                    {rule.missing_fields?.length ? (
                      <div>
                        <strong>缺失事实</strong>
                        <span>{rule.missing_fields.join('、')}</span>
                      </div>
                    ) : null}
                    {rule.evidence?.length ? (
                      <div>
                        <strong>证据</strong>
                        <ul>
                          {rule.evidence.map((item, index) => (
                            <li key={`${rule.rule_id}-evidence-${index}`}>{compactEvidence(item)}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                </details>
              ))}
            </div>
            {ruleResult.summary?.message && (
              <div className="performance-rule-note">{ruleResult.summary.message}</div>
            )}
          </div>
        ) : null}

        <div className="performance-calculation-panel">
          <div className="calculation-title">
            <span>{activeMetric === 'harris' ? 'Harris 计算过程' : 'CPK 计算过程'}</span>
            <Tag color={activeMetric === 'harris' ? (formalHarris?.recommend_tuning ? 'orange' : formalHarris ? 'green' : 'default') : tagColor(cpkLevel)}>
              {activeMetric === 'harris'
                ? (formalHarris ? boolText(formalHarris.recommend_tuning) : '等待计算')
                : levelLabel(cpkLevel)}
            </Tag>
          </div>
          <div className="calculation-formula">
            {activeMetric === 'harris'
              ? 'Harris η = 最小可达误差方差 σ²_mv / 实际误差方差 σ²_y。η 越接近 1，表示闭环越接近最小方差性能。'
              : 'CPK = min((USL-μ)/(3σ), (μ-LSL)/(3σ))。μ 为 PV 均值，σ 为 PV 样本标准差，USL/LSL 为 PV 规格上下限。'}
          </div>
          <div className="calculation-step-grid">
            {calculationSteps.map((step) => (
              <div key={step.title} className="calculation-step">
                <strong>{step.title}</strong>
                <span>{step.description}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="agent-panel harris-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">
              {activeMetric === 'harris' ? 'Harris 最小方差性能评估' : 'CPK 过程能力评估'}
            </div>
            <Typography.Text type="secondary">
              {activeMetric === 'harris'
                ? '后端调用 compute_harris_closed_loop skill，按误差信号、死时间、AR 模型和方差分解计算 Harris 指标。'
                : '后端只基于 PV 序列和 PV 规格上下限计算正式 CPK，不使用综合健康评分或整定准备度。'}
            </Typography.Text>
          </div>
          <Button icon={<ReloadOutlined />} loading={activeLoading} onClick={() => {
            if (activeMetric === 'harris') {
              void loadHarris();
            } else {
              void loadCpk(true);
            }
          }}>
            重新评估
          </Button>
        </div>

        {activeLoading ? (
          <Empty description={activeMetric === 'harris' ? '正在计算 Harris 性能评估...' : '正在计算 CPK 指标...'} />
        ) : activeError ? (
          <Alert type="warning" showIcon message="指标暂不可用" description={activeError} />
        ) : activeMetric === 'harris' ? (
          formalHarris?.abort_reason ? (
            <Alert
              type="warning"
              showIcon
              message="Harris 评估已终止"
              description={`原因：${riskFlagText[formalHarris.abort_reason] || formalHarris.abort_reason}`}
            />
          ) : formalHarris ? (
            <>
              <div className="harris-layout">
                <div className="harris-score-card">
                  <Progress
                    type="dashboard"
                    percent={harrisPercent}
                    format={() => eta === null || eta === undefined ? '-' : formatNumber(eta, 4)}
                    strokeColor={harrisColor(harrisLevel, eta)}
                    size={150}
                  />
                  <div className="harris-score-meta">
                    <strong>Harris 指标 η</strong>
                    <span>越接近 1 表示越接近最小方差控制；越低说明当前闭环波动仍有优化空间。</span>
                    <Tag color={formalHarris.recommend_tuning ? 'orange' : 'green'}>
                      {boolText(formalHarris.recommend_tuning)}
                    </Tag>
                  </div>
                </div>

                <div className="harris-process-card">
                  <Steps
                    size="small"
                    direction="vertical"
                    current={harrisProcessSteps.length}
                    items={harrisProcessSteps.map((item) => ({
                      title: item.title,
                      description: item.description,
                    }))}
                  />
                </div>
              </div>

              <div className="harris-metric-grid">
                <div className="harris-metric">
                  <span>置信度</span>
                  <strong>{confidence === null || confidence === undefined ? '-' : `${confidencePercent}%`}</strong>
                  <Progress percent={confidencePercent} showInfo={false} strokeColor="#3b82f6" />
                </div>
                <div className="harris-metric">
                  <span>死时间来源</span>
                  <strong>{deadtimeSourceText[formalHarris.deadtime_used?.source || ''] || formalHarris.deadtime_used?.source || '-'}</strong>
                  <em>{formatNumber(formalHarris.deadtime_used?.value_s, 1)} s</em>
                </div>
                <div className="harris-metric">
                  <span>AR 模型</span>
                  <strong>{formalHarris.ar_model?.provider || '-'}</strong>
                  <em>阶数 {formalHarris.ar_model?.order ?? '-'}</em>
                </div>
                <div className="harris-metric">
                  <span>实际误差方差</span>
                  <strong>{formatNumber(formalHarris.variance_decomp?.sigma2_y, 6)}</strong>
                  <em>用于 Harris 分母</em>
                </div>
                <div className="harris-metric">
                  <span>最小方差估计</span>
                  <strong>{formatNumber(formalHarris.variance_decomp?.sigma2_mv, 6)}</strong>
                  <em>用于 Harris 分子</em>
                </div>
                <div className="harris-metric">
                  <span>设定值变化占比</span>
                  <strong>{formatPercentValue(formalHarris.data_window?.sp_change_ratio, 1)}</strong>
                  <em>过高时会影响评价可信度</em>
                </div>
              </div>

              <div className="harris-explain-card harris-risk-card">
                <div className="harris-explain-title">Harris 计算风险标记</div>
                <Space wrap>
                  {riskFlags.length ? riskFlags.map((flag) => (
                    <Tooltip key={flag} title={flag}>
                      <Tag color="orange">{riskFlagText[flag] || flag}</Tag>
                    </Tooltip>
                  )) : <Tag color="green">未发现明显风险</Tag>}
                </Space>
              </div>

              {harrisResult?.reasoning && (
                <Alert type="info" showIcon message="后端评估说明" description={harrisResult.reasoning} />
              )}
            </>
          ) : (
            <Empty description="请选择回路后计算 Harris 性能评估" />
          )
        ) : cpkResult ? (
          <>
            <div className="harris-layout">
              <div className="harris-score-card">
                <Progress
                  type="dashboard"
                  percent={cpkPercent}
                  format={() => cpkValue === null || cpkValue === undefined ? '-' : formatNumber(cpkValue, 4)}
                  strokeColor={cpkColor(cpkLevel, cpkValue)}
                  size={150}
                />
                <div className="harris-score-meta">
                  <strong>CPK 指标</strong>
                  <span>CPK 衡量 PV 均值相对规格上下限的过程能力，正式计算必须有 PV LSL/USL。</span>
                  <Tag color={tagColor(cpkLevel)}>{levelLabel(cpkLevel)}</Tag>
                </div>
              </div>

              <div className="harris-process-card">
                <Steps
                  size="small"
                  direction="vertical"
                  current={cpkProcessSteps.length}
                  items={cpkProcessSteps.map((item) => ({
                    title: item.title,
                    description: item.description,
                  }))}
                />
              </div>
            </div>

            <div className="harris-metric-grid">
              <div className="harris-metric">
                <span>PV 均值 μ</span>
                <strong>{formatNumber(cpkResult.statistics?.mean, 6)}</strong>
                <em>当前窗口 PV 平均值</em>
              </div>
              <div className="harris-metric">
                <span>PV 样本标准差 σ</span>
                <strong>{formatNumber(cpkResult.statistics?.std, 6)}</strong>
                <em>CPK 分母使用 3σ</em>
              </div>
              <div className="harris-metric">
                <span>PV 下规格限 LSL</span>
                <strong>{formatNumber(cpkResult.limits?.lsl, 6)}</strong>
                <em>{cpkResult.limits?.lsl_column || '未配置'}</em>
              </div>
              <div className="harris-metric">
                <span>PV 上规格限 USL</span>
                <strong>{formatNumber(cpkResult.limits?.usl, 6)}</strong>
                <em>{cpkResult.limits?.usl_column || '未配置'}</em>
              </div>
              <div className="harris-metric">
                <span>CPL</span>
                <strong>{formatNumber(cpk?.cpl, 6)}</strong>
                <em>(μ-LSL)/(3σ)</em>
              </div>
              <div className="harris-metric">
                <span>CPU</span>
                <strong>{formatNumber(cpk?.cpu, 6)}</strong>
                <em>(USL-μ)/(3σ)</em>
              </div>
            </div>

            {cpkResult.warnings?.length ? (
              <Alert
                type="warning"
                showIcon
                message="CPK 计算提示"
                description={cpkResult.warnings.join('；')}
              />
            ) : null}
            {cpkResult.reasoning && (
              <Alert type="info" showIcon message="后端计算说明" description={cpkResult.reasoning} />
            )}
          </>
        ) : (
          <Empty description="请选择回路后计算 CPK 指标" />
        )}
      </section>
    </div>
  );
}
