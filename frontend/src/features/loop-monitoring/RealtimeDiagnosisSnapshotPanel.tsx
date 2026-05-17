import { useCallback, useEffect, useMemo, useState } from 'react';
import { Alert, Button, Descriptions, Empty, Progress, Space, Table, Tag, Typography } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';

import {
  fetchLatestRealtimeAssessments,
  runRealtimeAssessment,
  type RealtimeAssessmentSnapshot,
} from '@/services/api';

interface RealtimeDiagnosisSnapshotPanelProps {
  selectedLoopId?: string;
  timeRange?: string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
}

function riskColor(level?: string) {
  if (level === 'high') return 'red';
  if (level === 'medium') return 'orange';
  if (level === 'low') return 'gold';
  if (level === 'normal') return 'green';
  return 'blue';
}

export function RealtimeDiagnosisSnapshotPanel({
  selectedLoopId,
  timeRange = '8h',
  formatPercentValue,
}: RealtimeDiagnosisSnapshotPanelProps) {
  const [snapshot, setSnapshot] = useState<RealtimeAssessmentSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!selectedLoopId) {
      setSnapshot(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const latest = await fetchLatestRealtimeAssessments({ loop_id: selectedLoopId, limit: 1 });
      if (latest.items.length) {
        setSnapshot(latest.items[0]);
      } else {
        const created = await runRealtimeAssessment({
          loop_ids: [selectedLoopId],
          time_range: timeRange || '8h',
          include_formal_metrics: true,
        });
        setSnapshot(created.items[0] ?? null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSnapshot(null);
    } finally {
      setLoading(false);
    }
  }, [selectedLoopId, timeRange]);

  useEffect(() => {
    void load();
  }, [load]);

  const primaryDiagnosis = snapshot?.diagnosis?.[0];
  const harris = useMemo(() => snapshot?.metrics?.find((item) => item.name === 'harris'), [snapshot]);
  const cpk = useMemo(() => snapshot?.metrics?.find((item) => item.name === 'cpk'), [snapshot]);

  if (!selectedLoopId) {
    return (
      <section className="agent-panel">
        <div className="panel-title">实时评估快照</div>
        <Empty description="请选择回路后查看实时评估诊断证据" />
      </section>
    );
  }

  return (
    <section className="agent-panel">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">实时评估快照</div>
          <Typography.Text type="secondary">
            这里展示 SQLite 持久化的最近评估结果，只做诊断总览证据汇总，不替代各专项诊断菜单。
          </Typography.Text>
        </div>
        <Space wrap>
          {snapshot && <Tag color={riskColor(snapshot.risk_level)}>{snapshot.risk_level}</Tag>}
          {snapshot?.decision?.decision && <Tag color={snapshot.decision.need_tuning ? 'orange' : 'green'}>{snapshot.decision.decision}</Tag>}
          <Button size="small" icon={<ReloadOutlined />} loading={loading} onClick={load}>刷新</Button>
        </Space>
      </div>

      {error ? (
        <Alert type="warning" showIcon message="实时评估快照暂不可用" description={error} />
      ) : !snapshot ? (
        <Empty description={loading ? '正在加载实时评估快照...' : '暂无实时评估快照'} />
      ) : (
        <div className="page-stack compact-stack">
          <Descriptions bordered column={4} size="small" className="industrial-descriptions">
            <Descriptions.Item label="快照ID" span={2}>{snapshot.snapshot_id}</Descriptions.Item>
            <Descriptions.Item label="时间范围">{snapshot.time_window?.range || timeRange}</Descriptions.Item>
            <Descriptions.Item label="综合评分">{formatPercentValue(snapshot.score, 0)}</Descriptions.Item>
            <Descriptions.Item label="主根因" span={2}>{primaryDiagnosis?.root_cause || '-'}</Descriptions.Item>
            <Descriptions.Item label="根因置信度">{formatPercentValue(primaryDiagnosis?.confidence, 0)}</Descriptions.Item>
            <Descriptions.Item label="是否建议整定">{snapshot.decision?.need_tuning ? '是' : '否'}</Descriptions.Item>
            <Descriptions.Item label="Harris">{harris ? `${harris.value ?? '-'} / ${harris.level ?? '-'}` : '-'}</Descriptions.Item>
            <Descriptions.Item label="CPK">{cpk ? `${cpk.value ?? '-'} / ${cpk.level ?? '-'}` : '-'}</Descriptions.Item>
            <Descriptions.Item label="本体缺失字段" span={2}>
              {snapshot.ontology?.missing_fields?.length ? snapshot.ontology.missing_fields.join(', ') : '无'}
            </Descriptions.Item>
          </Descriptions>

          {snapshot.decision?.summary && (
            <Alert type={snapshot.decision.need_tuning ? 'warning' : 'info'} showIcon message="实时评估结论" description={snapshot.decision.summary} />
          )}

          <Table
            size="small"
            pagination={false}
            rowKey={(row) => row.diagnosis_id || row.root_cause}
            dataSource={snapshot.diagnosis ?? []}
            columns={[
              { title: '根因', dataIndex: 'root_cause', width: 180 },
              { title: '等级', dataIndex: 'severity', width: 100, render: (value: string) => <Tag color={riskColor(value)}>{value || '-'}</Tag> },
              { title: '置信度', dataIndex: 'confidence', width: 160, render: (value?: number) => <Progress percent={Math.round((value ?? 0) * 100)} size="small" /> },
              { title: '建议动作', dataIndex: 'action', ellipsis: true },
            ]}
          />

          <Table
            size="small"
            pagination={false}
            rowKey={(row) => row.trace_id || row.skill_name}
            dataSource={snapshot.skill_trace ?? []}
            columns={[
              { title: 'Skill', dataIndex: 'skill_name', width: 220 },
              { title: '风险', dataIndex: 'risk_level', width: 90, render: (value: string) => <Tag color={riskColor(value)}>{value}</Tag> },
              { title: '状态', dataIndex: 'status', width: 110 },
              { title: '耗时', dataIndex: 'duration_ms', width: 100, render: (value?: number) => value === undefined ? '-' : `${value} ms` },
              {
                title: '输出摘要',
                dataIndex: 'outputs_summary',
                ellipsis: true,
                render: (value?: Record<string, unknown>) => value ? JSON.stringify(value) : '-',
              },
            ]}
          />
        </div>
      )}
    </section>
  );
}
