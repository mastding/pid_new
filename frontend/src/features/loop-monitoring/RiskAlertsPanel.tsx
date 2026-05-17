import { Alert, Button, Empty, Progress, Select, Space, Spin, Table, Tag, Typography } from 'antd';
import {
  CheckCircleFilled,
  ExclamationCircleFilled,
  EyeOutlined,
  FireFilled,
  ReloadOutlined,
  WarningFilled,
} from '@ant-design/icons';
import type { CSSProperties, ReactNode } from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import { dashboardConicGradient, makeDashboardSlices, type DashboardLoopRow } from '@/features/dashboard/model';
import {
  fetchHistoryRiskAlerts,
  type HistoryLoop,
  type HistoryRiskAlertsResp,
  type HistoryRiskLevel,
  type HistoryRiskTrendRow,
} from '@/services/api';

type RiskLevel = HistoryRiskLevel;

interface RiskAlertsPanelProps {
  dashboardRows: DashboardLoopRow[];
  scopedLoops: HistoryLoop[];
  pathLabel: string;
  loading: boolean;
  loopTypeLabels: Record<string, string>;
  assetNameForLoop: (loop: HistoryLoop) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  scorePercent: (value?: number) => number;
  onRefresh: () => void;
  onViewLoop: (loopId: string) => void;
  onViewTrendSpectrum: (loopId: string) => void;
  onOpenDiagnosis: () => void;
  onCreateTuningTask: () => void;
}

interface RiskRow {
  key: string;
  level: RiskLevel;
  levelText: string;
  type: string;
  loopId: string;
  loopType: string;
  asset: string;
  description: string;
  trigger: string;
  duration: string;
  riskScore: number;
  foundAt: string;
}

const riskMeta: Record<RiskLevel, { label: string; color: string; icon: ReactNode; softBg: string; tag: string }> = {
  high: { label: '高风险', color: '#dc2626', icon: <FireFilled />, softBg: '#fff1f2', tag: 'red' },
  medium: { label: '中风险', color: '#f97316', icon: <ExclamationCircleFilled />, softBg: '#fff7ed', tag: 'orange' },
  low: { label: '低风险', color: '#eab308', icon: <WarningFilled />, softBg: '#fefce8', tag: 'gold' },
  potential: { label: '潜在风险', color: '#2563eb', icon: <WarningFilled />, softBg: '#eff6ff', tag: 'blue' },
  handled: { label: '已处理', color: '#16a34a', icon: <CheckCircleFilled />, softBg: '#f0fdf4', tag: 'green' },
};

const alertTypeText: Record<string, string> = {
  data_quality: '数据质量风险',
  data_health: '数据质量风险',
  stability: '稳定性风险',
  oscillation: '振荡风险',
  pv_mv_behavior: '阀门/执行机构风险',
  actuator: '阀门/执行机构风险',
  constraint: '约束/饱和风险',
  constraints: '约束/饱和风险',
  tracking: '设定值跟踪风险',
  response_observability: '模型/响应风险',
  operating_condition: '工况波动风险',
  noise: '测量噪声风险',
  monitoring: '综合监控风险',
};

const timeRangeOptions = [
  { label: '最近 8 小时', value: '8h' },
  { label: '最近 24 小时', value: '24h' },
  { label: '最近 7 天', value: '7d' },
];

function normalizeSeverity(value?: string): RiskLevel {
  if (value === 'critical' || value === 'high' || value === 'alarm') return 'high';
  if (value === 'warning' || value === 'medium' || value === 'warn') return 'medium';
  if (value === 'low' || value === 'info') return 'low';
  if (value === 'handled') return 'handled';
  return 'potential';
}

function levelBySnapshot(row: DashboardLoopRow): RiskLevel {
  const status = row.snapshot?.status;
  const alertLevel = row.snapshot?.events?.[0]?.severity || row.snapshot?.alerts?.[0]?.severity;
  if (alertLevel) return normalizeSeverity(alertLevel);
  if (status === 'critical' || status === 'alarm') return 'high';
  if (status === 'warning') return 'medium';
  const score = row.snapshot?.overall_score;
  if (score === undefined) return 'potential';
  if (score < 0.45) return 'high';
  if (score < 0.65) return 'medium';
  if (score < 0.82) return 'low';
  return 'potential';
}

function scoreRisk(row: DashboardLoopRow, level: RiskLevel) {
  const base = Math.round((1 - (row.snapshot?.overall_score ?? 0.7)) * 100);
  const levelBonus = level === 'high' ? 42 : level === 'medium' ? 28 : level === 'low' ? 16 : 8;
  return Math.max(1, Math.min(100, base + levelBonus + row.alertCount * 8));
}

function compactTime(value?: string | null) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return `${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
}

function indicatorFallback(row: DashboardLoopRow) {
  const snapshot = row.snapshot;
  const candidates = [
    { key: 'data_health', score: snapshot?.data_health?.score },
    { key: 'stability', score: snapshot?.stability?.score },
    { key: 'pv_mv_behavior', score: snapshot?.pv_mv_behavior?.score },
    { key: 'constraints', score: snapshot?.constraints?.score },
    { key: 'response_observability', score: snapshot?.response_observability?.score },
  ].filter((item) => item.score !== undefined);
  return candidates.sort((a, b) => Number(a.score) - Number(b.score))[0]?.key || 'monitoring';
}

function buildFallbackRiskRows(
  rows: DashboardLoopRow[],
  loopTypeLabels: Record<string, string>,
  assetNameForLoop: (loop: HistoryLoop) => string,
): RiskRow[] {
  return rows
    .filter((row) => row.snapshot)
    .map((row) => {
      const event = row.snapshot?.events?.[0];
      const alert = row.snapshot?.alerts?.[0];
      const level = levelBySnapshot(row);
      const typeKey = event?.type || alert?.type || indicatorFallback(row);
      return {
        key: row.loop.loop_id,
        level,
        levelText: riskMeta[level].label,
        type: alertTypeText[typeKey] || typeKey || '综合监控风险',
        loopId: row.loop.loop_id,
        loopType: loopTypeLabels[row.loop.loop_type] ?? row.loop.loop_type ?? '-',
        asset: assetNameForLoop(row.loop),
        description: event?.message || alert?.message || `监控评分 ${Math.round((row.snapshot?.overall_score ?? 0) * 100)} 分，建议关注趋势变化。`,
        trigger: event?.name || event?.type || alert?.type || alertTypeText[typeKey] || '监控快照',
        duration: row.loop.start_time && row.loop.end_time ? `${compactTime(row.loop.start_time)} ~ ${compactTime(row.loop.end_time)}` : '-',
        riskScore: scoreRisk(row, level),
        foundAt: compactTime(row.loop.end_time),
      };
    })
    .sort((a, b) => b.riskScore - a.riskScore);
}

function buildApiRiskRows(data: HistoryRiskAlertsResp | null, loopTypeLabels: Record<string, string>): RiskRow[] {
  return (data?.items ?? []).map((item) => {
    const level = normalizeSeverity(item.level);
    return {
      key: item.key || item.loop_id,
      level,
      levelText: riskMeta[level].label,
      type: item.type_label || alertTypeText[item.type] || item.type || '综合监控风险',
      loopId: item.loop_id,
      loopType: loopTypeLabels[item.loop_type ?? ''] ?? item.loop_type ?? '-',
      asset: item.asset || '未归属',
      description: item.description || '监控指标触发风险提示。',
      trigger: item.trigger || '监控快照',
      duration: item.duration || '-',
      riskScore: item.risk_score ?? 0,
      foundAt: compactTime(item.found_at),
    };
  }).sort((a, b) => b.riskScore - a.riskScore);
}

function RiskMiniTrend({ rows, apiTrend }: { rows: RiskRow[]; apiTrend?: HistoryRiskTrendRow[] }) {
  const levels = ['high', 'medium', 'low', 'potential'] as const;
  const trendRows = apiTrend?.length
    ? apiTrend.map((item) => ({
      date: item.date,
      high: item.high ?? 0,
      medium: item.medium ?? 0,
      low: item.low ?? 0,
      potential: item.potential ?? 0,
    }))
    : Array.from(new Set(rows.map((item) => item.foundAt === '-' ? '未标记' : item.foundAt.slice(0, 5)))).slice(-7).map((date) => ({
      date,
      high: rows.filter((row) => (row.foundAt === '-' ? '未标记' : row.foundAt.slice(0, 5)) === date && row.level === 'high').length,
      medium: rows.filter((row) => (row.foundAt === '-' ? '未标记' : row.foundAt.slice(0, 5)) === date && row.level === 'medium').length,
      low: rows.filter((row) => (row.foundAt === '-' ? '未标记' : row.foundAt.slice(0, 5)) === date && row.level === 'low').length,
      potential: rows.filter((row) => (row.foundAt === '-' ? '未标记' : row.foundAt.slice(0, 5)) === date && row.level === 'potential').length,
    }));
  const width = 560;
  const height = 190;
  const padding = 28;
  const maxValue = Math.max(1, ...trendRows.flatMap((row) => levels.map((level) => row[level] ?? 0)));

  if (!rows.length) {
    return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无风险趋势" />;
  }

  return (
    <div className="risk-trend-chart">
      <div className="risk-chart-legend">
        {levels.map((level) => (
          <span key={level}><i style={{ background: riskMeta[level].color }} />{riskMeta[level].label}</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="风险趋势">
        {[0, 0.5, 1].map((tick) => (
          <line
            key={tick}
            x1={padding}
            x2={width - padding}
            y1={height - padding - tick * (height - padding * 2)}
            y2={height - padding - tick * (height - padding * 2)}
            className="risk-grid-line"
          />
        ))}
        {levels.map((level) => {
          const coordinates = trendRows.map((item, index) => {
            const x = padding + (trendRows.length <= 1 ? (width - padding * 2) / 2 : index * ((width - padding * 2) / (trendRows.length - 1)));
            const y = height - padding - ((item[level] ?? 0) / maxValue) * (height - padding * 2);
            return { x, y };
          });
          const points = coordinates.map((point) => `${point.x},${point.y}`).join(' ');
          return (
            <g key={level}>
              {trendRows.length > 1 && (
                <polyline
                  points={points}
                  fill="none"
                  stroke={riskMeta[level].color}
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}
              {coordinates.map((point, index) => (
                <circle key={`${level}-${index}`} cx={point.x} cy={point.y} r="4" fill={riskMeta[level].color} />
              ))}
            </g>
          );
        })}
        {trendRows.map((item, index) => {
          const x = padding + (trendRows.length <= 1 ? (width - padding * 2) / 2 : index * ((width - padding * 2) / (trendRows.length - 1)));
          return <text key={item.date} x={x} y={height - 6} textAnchor="middle">{item.date}</text>;
        })}
      </svg>
    </div>
  );
}

export function RiskAlertsPanel({
  dashboardRows,
  scopedLoops,
  pathLabel,
  loading,
  loopTypeLabels,
  assetNameForLoop,
  formatPercentValue,
  onRefresh,
  onViewLoop,
  onViewTrendSpectrum,
  onOpenDiagnosis,
  onCreateTuningTask,
}: RiskAlertsPanelProps) {
  const [assetFilter, setAssetFilter] = useState('all');
  const [timeRange, setTimeRange] = useState('8h');
  const [riskData, setRiskData] = useState<HistoryRiskAlertsResp | null>(null);
  const [riskLoading, setRiskLoading] = useState(false);
  const [riskError, setRiskError] = useState<string | null>(null);

  const loadRiskAlerts = useCallback(async () => {
    setRiskLoading(true);
    setRiskError(null);
    try {
      const data = await fetchHistoryRiskAlerts({ asset: assetFilter, time_range: timeRange });
      setRiskData(data);
    } catch (error) {
      setRiskError(error instanceof Error ? error.message : String(error));
      setRiskData(null);
    } finally {
      setRiskLoading(false);
    }
  }, [assetFilter, timeRange]);

  useEffect(() => {
    void loadRiskAlerts();
  }, [loadRiskAlerts]);

  const fallbackRows = useMemo(
    () => buildFallbackRiskRows(dashboardRows, loopTypeLabels, assetNameForLoop),
    [assetNameForLoop, dashboardRows, loopTypeLabels],
  );
  const apiRows = useMemo(() => buildApiRiskRows(riskData, loopTypeLabels), [loopTypeLabels, riskData]);
  const riskRows = apiRows.length || riskData ? apiRows : fallbackRows;
  const loadedCount = riskData?.loaded_count ?? dashboardRows.filter((row) => row.snapshot).length;
  const totalLoopCount = riskData?.total_loops ?? scopedLoops.length;
  const totalRisk = riskRows.length;
  const counts = {
    high: riskData?.counts?.high ?? riskRows.filter((row) => row.level === 'high').length,
    medium: riskData?.counts?.medium ?? riskRows.filter((row) => row.level === 'medium').length,
    low: riskData?.counts?.low ?? riskRows.filter((row) => row.level === 'low').length,
    potential: riskData?.counts?.potential ?? riskRows.filter((row) => row.level === 'potential').length,
    handled: riskData?.counts?.handled ?? riskRows.filter((row) => row.level === 'handled').length,
  };
  const riskSlices = makeDashboardSlices([
    { label: riskMeta.high.label, value: counts.high, color: riskMeta.high.color },
    { label: riskMeta.medium.label, value: counts.medium, color: riskMeta.medium.color },
    { label: riskMeta.low.label, value: counts.low, color: riskMeta.low.color },
    { label: riskMeta.potential.label, value: counts.potential, color: riskMeta.potential.color },
  ]);
  const localAssetOptions = Array.from(new Set(scopedLoops.map(assetNameForLoop))).filter(Boolean).map((name) => ({ label: name, value: name }));
  const assetOptions = [
    { label: '全部装置', value: 'all' },
    ...(riskData?.asset_options?.length ? riskData.asset_options : localAssetOptions),
  ];
  const typeRows = riskData?.type_distribution?.length
    ? riskData.type_distribution.map((item) => [item.label, item.value] as [string, number])
    : Object.entries(riskRows.reduce<Record<string, number>>((acc, row) => {
      acc[row.type] = (acc[row.type] ?? 0) + 1;
      return acc;
    }, {})).sort((a, b) => b[1] - a[1]).slice(0, 7);
  const assetRows = riskData?.asset_distribution?.length
    ? riskData.asset_distribution.map((item) => [item.label, item.value] as [string, number])
    : Object.entries(riskRows.reduce<Record<string, number>>((acc, row) => {
      acc[row.asset] = (acc[row.asset] ?? 0) + 1;
      return acc;
    }, {})).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const maxType = Math.max(1, ...typeRows.map(([, value]) => value));
  const maxAsset = Math.max(1, ...assetRows.map(([, value]) => value));
  const topRows = riskRows.slice(0, 8);
  const dominantType = typeRows[0]?.[0] || '暂无明显风险';
  const highOrMedium = counts.high + counts.medium;
  const refreshing = loading || riskLoading;

  return (
    <div className="risk-alerts-page">
      <section className="risk-alerts-header">
        <div>
          <h2>风险预警</h2>
          <Typography.Text type="secondary">按装置和时间范围汇总回路风险，优先暴露需要本班确认的异常。</Typography.Text>
        </div>
        <Space wrap>
          <span className="risk-filter-label">所属装置</span>
          <Select value={assetFilter} style={{ width: 180 }} options={assetOptions} onChange={setAssetFilter} />
          <span className="risk-filter-label">时间范围</span>
          <Select value={timeRange} style={{ width: 150 }} options={timeRangeOptions} onChange={setTimeRange} />
          <Button
            icon={<ReloadOutlined />}
            loading={refreshing}
            onClick={() => {
              onRefresh();
              void loadRiskAlerts();
            }}
          >
            刷新
          </Button>
        </Space>
      </section>

      {riskError && (
        <Alert
          type="warning"
          showIcon
          message="后端风险汇总暂不可用"
          description="页面已回退到当前驾驶仓快照数据，刷新或切换时间范围后会再次尝试加载后端汇总。"
        />
      )}

      <Spin spinning={riskLoading}>
        <section className="risk-kpi-grid">
          {(['high', 'medium', 'low', 'potential', 'handled'] as RiskLevel[]).map((level) => (
            <div className={`risk-kpi-card risk-kpi-card--${level}`} key={level}>
              <i style={{ color: riskMeta[level].color, background: riskMeta[level].softBg }}>{riskMeta[level].icon}</i>
              <div>
                <span>{riskMeta[level].label}</span>
                <strong>{counts[level]}</strong>
                <small>{level === 'handled' ? '来自已闭环事件' : `占比 ${formatPercentValue(totalRisk ? counts[level] / totalRisk : 0, 1)}`}</small>
              </div>
            </div>
          ))}
          <div className="risk-distribution-card">
            <div className="risk-card-title">风险等级分布</div>
            <div className="risk-donut-row">
              <div className="risk-donut" style={{ background: dashboardConicGradient(riskSlices) }}>
                <strong>{totalRisk}</strong>
                <span>总风险</span>
              </div>
              <div className="risk-legend">
                {riskSlices.map((item) => (
                  <span key={item.label}>
                    <i style={{ background: item.color }} />
                    {item.label}
                    <b>{item.value}</b>
                    <em>{formatPercentValue(item.percent, 1)}</em>
                  </span>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="risk-main-grid">
          <div className="risk-panel risk-trend-panel">
            <div className="risk-card-title">风险趋势</div>
            <RiskMiniTrend rows={riskRows} apiTrend={riskData?.trend} />
          </div>
          <div className="risk-panel">
            <div className="risk-card-title">风险类型分布</div>
            <div className="risk-bars">
              {typeRows.length ? typeRows.map(([label, value], index) => (
                <div className="risk-bar" key={label}>
                  <span title={label}>{label}</span>
                  <em><i style={{ width: `${Math.max(5, value / maxType * 100)}%`, background: ['#dc2626', '#f97316', '#eab308', '#2563eb', '#7c3aed'][index % 5] }} /></em>
                  <b>{value}</b>
                </div>
              )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无风险类型" />}
            </div>
          </div>
          <div className="risk-panel">
            <div className="risk-card-title">风险热力图（装置）</div>
            <div className="risk-heatmap">
              {assetRows.length ? assetRows.map(([label, value]) => (
                <button
                  key={label}
                  type="button"
                  style={{ '--risk-heat': `${0.18 + value / maxAsset * 0.72}` } as CSSProperties}
                  onClick={() => setAssetFilter(label)}
                >
                  <span title={label}>{label}</span>
                  <strong>{value}</strong>
                </button>
              )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无装置风险" />}
            </div>
            <div className="risk-heat-scale"><span>低</span><i /><span>高</span></div>
          </div>
        </section>

        <section className="risk-bottom-grid">
          <div className="risk-panel risk-table-panel">
            <div className="risk-card-title">风险预警列表</div>
            <Table
              size="small"
              rowKey="key"
              dataSource={topRows}
              pagination={{ pageSize: 8 }}
              columns={[
                {
                  title: '风险等级',
                  dataIndex: 'level',
                  width: 110,
                  render: (value: RiskLevel) => <Tag color={riskMeta[value]?.tag ?? 'blue'}>{riskMeta[value]?.label ?? value}</Tag>,
                },
                { title: '风险类型', dataIndex: 'type', width: 150 },
                { title: '回路名称', dataIndex: 'loopId', width: 160 },
                { title: '所属装置', dataIndex: 'asset', width: 140 },
                { title: '风险描述', dataIndex: 'description', width: 220, ellipsis: true },
                { title: '触发原因', dataIndex: 'trigger', width: 150, ellipsis: true },
                { title: '风险分', dataIndex: 'riskScore', width: 90, render: (value: number) => <strong className="risk-score">{value}</strong> },
                {
                  title: '操作',
                  width: 120,
                  render: (_value, row: RiskRow) => (
                    <Space>
                      <Button size="small" icon={<EyeOutlined />} onClick={() => onViewLoop(row.loopId)}>查看</Button>
                      <Button size="small" onClick={() => onViewTrendSpectrum(row.loopId)}>趋势</Button>
                    </Space>
                  ),
                },
              ]}
            />
            <Typography.Text type="secondary">已接入 {loadedCount} / {totalLoopCount} 个回路监控快照，当前范围：{assetFilter === 'all' ? pathLabel : assetFilter} · {timeRangeOptions.find((item) => item.value === timeRange)?.label}</Typography.Text>
          </div>

          <div className="risk-panel risk-advice-panel">
            <div className="risk-card-title">预警建议</div>
            <div className="risk-advice-list">
              <Alert
                type={counts.high ? 'error' : 'success'}
                showIcon
                message={counts.high ? '优先处理高风险回路' : '当前没有高风险回路'}
                description={counts.high ? `当前存在 ${counts.high} 个高风险回路，建议优先查看趋势与控制性能页。` : '保持巡检，关注中低风险变化。'}
              />
              <Alert
                type={highOrMedium ? 'warning' : 'info'}
                showIcon
                message="检查主要风险类型"
                description={`当前风险主要集中在：${dominantType}。建议结合趋势、频谱和控制性能评估核对原因。`}
              />
              <Alert
                type="info"
                showIcon
                message="联动诊断与整定"
                description="确认风险由振荡、约束或执行机构异常造成时，先诊断再进入整定任务，避免把非控制器问题误当成参数问题。"
              />
            </div>
            <div className="risk-action-grid">
              <Button type="primary" onClick={onCreateTuningTask}>新建整定任务</Button>
              <Button onClick={onOpenDiagnosis}>进入根因诊断</Button>
            </div>
            <div className="risk-progress-card">
              <span>快照覆盖率</span>
              <Progress percent={Math.round((loadedCount / Math.max(totalLoopCount, 1)) * 100)} />
              <small>范围：{assetFilter === 'all' ? pathLabel : assetFilter}</small>
            </div>
          </div>
        </section>
      </Spin>
    </div>
  );
}
