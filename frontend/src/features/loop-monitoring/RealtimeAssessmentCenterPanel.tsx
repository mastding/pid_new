import { useCallback, useEffect, useMemo, useState } from 'react';
import { Alert, Button, Descriptions, Drawer, Empty, Select, Space, Spin, Table, Tag, Typography } from 'antd';
import { EyeOutlined, ReloadOutlined } from '@ant-design/icons';

import {
  fetchLatestRealtimeAssessments,
  listAutoTuningTasks,
  runRealtimeMonitorTick,
  type AutoTuningTask,
  type HistoryLoop,
  type RealtimeAssessmentSnapshot,
} from '@/services/api';

interface RealtimeAssessmentCenterPanelProps {
  scopedLoops: HistoryLoop[];
  loopTypeLabels: Record<string, string>;
  onSelectLoop: (loopId: string) => void;
  onOpenTuningTask: () => void;
  formatNumber: (value?: number | null, digits?: number) => string;
}

const riskColor: Record<string, string> = {
  high: 'red',
  medium: 'orange',
  low: 'gold',
  potential: 'blue',
  normal: 'green',
};

const timeRangeOptions = [
  { label: '最近 8 小时', value: '8h' },
  { label: '最近 24 小时', value: '24h' },
  { label: '最近 7 天', value: '7d' },
];

function assetIdOf(loop: HistoryLoop) {
  const loopId = loop.loop_id || '';
  return loopId.includes('_') ? loopId.split('_')[0] : loop.dataset_id || 'default';
}

function metricValue(snapshot: RealtimeAssessmentSnapshot, name: string) {
  return snapshot.metrics?.find((item) => item.name === name);
}

function compactTime(value?: string | null) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return `${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
}

function JsonBlock({ value }: { value: unknown }) {
  if (value === undefined || value === null) return <Typography.Text type="secondary">-</Typography.Text>;
  return (
    <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12 }}>
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export function RealtimeAssessmentCenterPanel({
  scopedLoops,
  loopTypeLabels,
  onSelectLoop,
  onOpenTuningTask,
  formatNumber,
}: RealtimeAssessmentCenterPanelProps) {
  const [assetFilter, setAssetFilter] = useState('all');
  const [timeRange, setTimeRange] = useState('8h');
  const [snapshots, setSnapshots] = useState<RealtimeAssessmentSnapshot[]>([]);
  const [tasks, setTasks] = useState<AutoTuningTask[]>([]);
  const [selectedSnapshot, setSelectedSnapshot] = useState<RealtimeAssessmentSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const assetOptions = useMemo(() => {
    const assets = Array.from(new Set(scopedLoops.map(assetIdOf))).filter(Boolean);
    return [{ label: '全部装置', value: 'all' }, ...assets.map((asset) => ({ label: asset, value: asset }))];
  }, [scopedLoops]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [snapshotResp, taskResp] = await Promise.all([
        fetchLatestRealtimeAssessments({
          asset_id: assetFilter === 'all' ? undefined : assetFilter,
          limit: 300,
        }),
        listAutoTuningTasks({
          asset_id: assetFilter === 'all' ? undefined : assetFilter,
          limit: 100,
        }),
      ]);
      setSnapshots(snapshotResp.items ?? []);
      setTasks(taskResp.items ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSnapshots([]);
      setTasks([]);
    } finally {
      setLoading(false);
    }
  }, [assetFilter]);

  useEffect(() => {
    void load();
  }, [load]);

  const runNow = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      await runRealtimeMonitorTick({ force: true });
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }, [load]);

  const latestByLoop = useMemo(() => {
    const map = new Map<string, RealtimeAssessmentSnapshot>();
    for (const item of snapshots) {
      if (!map.has(item.loop_id)) {
        map.set(item.loop_id, item);
      }
    }
    return Array.from(map.values());
  }, [snapshots]);

  const loopCount = assetFilter === 'all'
    ? scopedLoops.length
    : scopedLoops.filter((loop) => assetIdOf(loop) === assetFilter).length;
  const needTuningCount = latestByLoop.filter((item) => item.decision?.need_tuning).length;
  const highRiskCount = latestByLoop.filter((item) => item.risk_level === 'high').length;
  const mediumRiskCount = latestByLoop.filter((item) => item.risk_level === 'medium').length;
  const ontologyMissingCount = latestByLoop.filter((item) => (item.ontology?.missing_fields ?? []).length > 0).length;

  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">实时评估中心</div>
            <Typography.Text type="secondary">
              汇总最近评估快照，统一查看 Harris/CPK、诊断主因、本体缺失和整定候选任务。
            </Typography.Text>
          </div>
          <Space wrap>
            <Select value={assetFilter} style={{ width: 160 }} options={assetOptions} onChange={setAssetFilter} />
            <Select value={timeRange} style={{ width: 140 }} options={timeRangeOptions} onChange={setTimeRange} />
            <Button icon={<ReloadOutlined />} loading={loading} onClick={load}>刷新</Button>
            <Button type="primary" loading={running} onClick={runNow}>立即评估</Button>
          </Space>
        </div>
        {error && <Alert className="agent-alert" type="warning" showIcon message="实时评估中心加载失败" description={error} />}
      </section>

      <section className="risk-kpi-grid">
        <div className="risk-kpi-card risk-kpi-card--potential">
          <div><span>评估覆盖率</span><strong>{latestByLoop.length}/{loopCount}</strong><small>按回路最新快照去重</small></div>
        </div>
        <div className="risk-kpi-card risk-kpi-card--high">
          <div><span>高风险</span><strong>{highRiskCount}</strong><small>需优先复核</small></div>
        </div>
        <div className="risk-kpi-card risk-kpi-card--medium">
          <div><span>中风险</span><strong>{mediumRiskCount}</strong><small>建议班内处理</small></div>
        </div>
        <div className="risk-kpi-card risk-kpi-card--low">
          <div><span>需要整定</span><strong>{needTuningCount}</strong><small>已通过诊断规则筛选</small></div>
        </div>
        <div className="risk-kpi-card risk-kpi-card--handled">
          <div><span>自动候选</span><strong>{tasks.length}</strong><small>待复核/待执行/已完成</small></div>
        </div>
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">回路实时评估表</div>
            <Typography.Text type="secondary">
              点击详情查看指标、本体证据和 skill trace；需要整定的回路可跳转到整定任务队列。
            </Typography.Text>
          </div>
          <Space>
            <Tag color={ontologyMissingCount ? 'orange' : 'green'}>本体缺失 {ontologyMissingCount}</Tag>
            <Button onClick={onOpenTuningTask}>整定候选队列</Button>
          </Space>
        </div>
        <Spin spinning={loading || running}>
          <Table<RealtimeAssessmentSnapshot>
            size="small"
            rowKey="snapshot_id"
            dataSource={latestByLoop}
            locale={{ emptyText: <Empty description="暂无实时评估快照，可点击立即评估生成" /> }}
            columns={[
              { title: '回路', dataIndex: 'loop_id', width: 170 },
              {
                title: '类型',
                dataIndex: 'loop_type',
                width: 90,
                render: (value: string) => loopTypeLabels[value] ?? value,
              },
              {
                title: '风险',
                dataIndex: 'risk_level',
                width: 90,
                render: (value: string) => <Tag color={riskColor[value] ?? 'default'}>{value || '-'}</Tag>,
              },
              {
                title: 'Harris',
                width: 100,
                render: (_, row) => {
                  const harris = metricValue(row, 'harris');
                  return harris?.success ? formatNumber(harris.value, 3) : <Tag>缺失</Tag>;
                },
              },
              {
                title: 'CPK',
                width: 100,
                render: (_, row) => {
                  const cpk = metricValue(row, 'cpk');
                  return cpk?.success ? formatNumber(cpk.value, 3) : <Tag>缺失</Tag>;
                },
              },
              {
                title: '诊断主因',
                render: (_, row) => row.diagnosis?.[0]?.root_cause ?? '-',
              },
              {
                title: '决策',
                width: 140,
                render: (_, row) => (
                  <Tag color={row.decision?.need_tuning ? 'orange' : row.decision?.blocked ? 'red' : 'blue'}>
                    {row.decision?.decision ?? '-'}
                  </Tag>
                ),
              },
              {
                title: '评估时间',
                width: 140,
                render: (_, row) => compactTime(row.created_at),
              },
              {
                title: '操作',
                width: 150,
                render: (_, row) => (
                  <Space>
                    <Button size="small" icon={<EyeOutlined />} onClick={() => setSelectedSnapshot(row)} />
                    <Button size="small" onClick={() => onSelectLoop(row.loop_id)}>定位</Button>
                  </Space>
                ),
              },
            ]}
          />
        </Spin>
      </section>

      <Drawer
        title={selectedSnapshot ? `${selectedSnapshot.loop_id} 实时评估详情` : '实时评估详情'}
        width={780}
        open={Boolean(selectedSnapshot)}
        onClose={() => setSelectedSnapshot(null)}
      >
        {selectedSnapshot && (
          <div className="page-stack compact-stack">
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="风险等级">
                <Tag color={riskColor[selectedSnapshot.risk_level] ?? 'default'}>{selectedSnapshot.risk_level}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="决策">{selectedSnapshot.decision?.decision ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="窗口">
                {selectedSnapshot.time_window?.start_time || '-'} ~ {selectedSnapshot.time_window?.end_time || '-'}
              </Descriptions.Item>
              <Descriptions.Item label="本体来源">{selectedSnapshot.ontology?.source ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="本体缺失" span={2}>
                {(selectedSnapshot.ontology?.missing_fields ?? []).length
                  ? selectedSnapshot.ontology?.missing_fields?.map((field) => <Tag key={field}>{field}</Tag>)
                  : <Tag color="green">完整</Tag>}
              </Descriptions.Item>
            </Descriptions>
            {selectedSnapshot.decision?.summary && (
              <Alert type={selectedSnapshot.decision.need_tuning ? 'warning' : 'info'} showIcon message="评估结论" description={selectedSnapshot.decision.summary} />
            )}
            <Table
              size="small"
              pagination={false}
              rowKey="name"
              dataSource={selectedSnapshot.metrics ?? []}
              expandable={{
                expandedRowRender: (row) => <JsonBlock value={row.raw} />,
                rowExpandable: (row) => Boolean(row.raw),
              }}
              columns={[
                { title: '指标', dataIndex: 'name' },
                { title: '值', dataIndex: 'value', render: (value) => formatNumber(Number(value), 4) },
                { title: '等级', dataIndex: 'level', render: (value) => <Tag>{value || '-'}</Tag> },
                { title: '置信度', dataIndex: 'confidence', render: (value) => formatNumber(Number(value), 3) },
                { title: '状态', dataIndex: 'success', render: (value) => <Tag color={value ? 'green' : 'orange'}>{value ? '成功' : '不可用'}</Tag> },
              ]}
            />
            <Table
              size="small"
              pagination={false}
              rowKey={(row) => row.diagnosis_id ?? row.root_cause}
              dataSource={selectedSnapshot.diagnosis ?? []}
              expandable={{
                expandedRowRender: (row) => <JsonBlock value={row.evidence} />,
                rowExpandable: (row) => Boolean(row.evidence?.length),
              }}
              columns={[
                { title: '根因', dataIndex: 'root_cause' },
                { title: '严重度', dataIndex: 'severity', render: (value) => <Tag color={riskColor[value] ?? 'default'}>{value || '-'}</Tag> },
                { title: '置信度', dataIndex: 'confidence', render: (value) => formatNumber(Number(value), 3) },
                { title: '建议动作', dataIndex: 'action' },
              ]}
            />
            <Table
              size="small"
              pagination={false}
              rowKey={(row) => row.trace_id ?? row.skill_name}
              dataSource={selectedSnapshot.skill_trace ?? []}
              expandable={{
                expandedRowRender: (row) => (
                  <Descriptions bordered size="small" column={1}>
                    <Descriptions.Item label="输入摘要"><JsonBlock value={row.inputs_summary} /></Descriptions.Item>
                    <Descriptions.Item label="输出摘要"><JsonBlock value={row.outputs_summary} /></Descriptions.Item>
                    <Descriptions.Item label="Guard"><JsonBlock value={row.guard} /></Descriptions.Item>
                  </Descriptions>
                ),
              }}
              columns={[
                { title: 'Skill', dataIndex: 'skill_name' },
                { title: '风险', dataIndex: 'risk_level', render: (value) => <Tag color={riskColor[value] ?? 'default'}>{value}</Tag> },
                { title: '状态', dataIndex: 'status' },
                { title: '耗时(ms)', dataIndex: 'duration_ms' },
              ]}
            />
          </div>
        )}
      </Drawer>
    </div>
  );
}
