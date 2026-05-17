import { PageContainer, ProCard } from '@ant-design/pro-components';
import { Alert, Button, Descriptions, Empty, Input, Modal, Select, Space, Statistic, Switch, Table, Tag, Typography, message } from 'antd';
import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  attachExperienceOutcome,
  fetchExperienceSkills,
  fetchExperienceSnapshots,
  type ExperienceSkillSummary,
  type ExperienceSnapshot,
} from '@/services/api';

function formatTime(ts?: number | null) {
  if (!ts) return '-';
  const date = new Date(ts * 1000);
  if (Number.isNaN(date.getTime())) return '-';
  return date.toLocaleString();
}

function JsonBlock({ value }: { value: unknown }) {
  if (value === undefined || value === null) return <Typography.Text type="secondary">-</Typography.Text>;
  return (
    <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12 }}>
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export default function ExperiencePage() {
  const [skills, setSkills] = useState<ExperienceSkillSummary[]>([]);
  const [snapshots, setSnapshots] = useState<ExperienceSnapshot[]>([]);
  const [skillFilter, setSkillFilter] = useState<string | undefined>();
  const [onlyWithOutcome, setOnlyWithOutcome] = useState(false);
  const [loading, setLoading] = useState(false);
  const [savingOutcome, setSavingOutcome] = useState(false);
  const [outcomeTarget, setOutcomeTarget] = useState<ExperienceSnapshot | null>(null);
  const [outcomeText, setOutcomeText] = useState('{\n  "human_label": "good"\n}');
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [skillResp, snapshotResp] = await Promise.all([
        fetchExperienceSkills(),
        fetchExperienceSnapshots({
          skill_name: skillFilter,
          only_with_outcome: onlyWithOutcome,
          limit: 200,
        }),
      ]);
      setSkills(skillResp.items ?? []);
      setSnapshots(snapshotResp.items ?? []);
    } catch (err) {
      const messageText = err instanceof Error ? err.message : String(err);
      setError(messageText);
      message.error(messageText);
    } finally {
      setLoading(false);
    }
  }, [onlyWithOutcome, skillFilter]);

  useEffect(() => {
    void load();
  }, [load]);

  const skillOptions = useMemo(() => [
    { label: '全部 Skill', value: '' },
    ...skills.map((item) => ({
      label: `${item.skill_name} (${item.snapshot_count})`,
      value: item.skill_name,
    })),
  ], [skills]);
  const totalSnapshots = skills.reduce((sum, item) => sum + item.snapshot_count, 0);
  const totalOutcomes = skills.reduce((sum, item) => sum + (item.outcome_count ?? 0), 0);

  const openOutcomeModal = (row: ExperienceSnapshot) => {
    setOutcomeTarget(row);
    setOutcomeText(JSON.stringify(row.observable_outcomes || { human_label: 'good' }, null, 2));
  };

  const saveOutcome = async () => {
    if (!outcomeTarget) return;
    let outcome: Record<string, unknown>;
    try {
      const parsed = JSON.parse(outcomeText);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('Outcome 必须是 JSON 对象');
      }
      outcome = parsed as Record<string, unknown>;
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Outcome JSON 格式错误');
      return;
    }
    setSavingOutcome(true);
    try {
      await attachExperienceOutcome({
        skill_name: outcomeTarget.skill_name,
        snapshot_id: outcomeTarget.snapshot_id,
        outcome,
      });
      message.success('Outcome 已回填');
      setOutcomeTarget(null);
      await load();
    } catch (err) {
      message.error(err instanceof Error ? err.message : 'Outcome 回填失败');
    } finally {
      setSavingOutcome(false);
    }
  };

  return (
    <PageContainer title="整定经验" subTitle="学习样本、工程复核 outcome 与相似回路经验沉淀">
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        {error && <Alert type="warning" showIcon message="经验库加载失败" description={error} />}

        <ProCard split="vertical">
          <ProCard>
            <Statistic title="Skill 数" value={skills.length} />
          </ProCard>
          <ProCard>
            <Statistic title="学习样本" value={totalSnapshots} />
          </ProCard>
          <ProCard>
            <Statistic title="已回填 Outcome" value={totalOutcomes} />
          </ProCard>
          <ProCard>
            <Statistic title="当前列表" value={snapshots.length} />
          </ProCard>
        </ProCard>

        <ProCard>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
            <Space wrap>
              <Select
                style={{ minWidth: 260 }}
                value={skillFilter ?? ''}
                options={skillOptions}
                onChange={(value) => setSkillFilter(value || undefined)}
              />
              <Space size={6}>
                <Typography.Text type="secondary">仅看已回填 outcome</Typography.Text>
                <Switch checked={onlyWithOutcome} onChange={setOnlyWithOutcome} />
              </Space>
            </Space>
            <Button loading={loading} onClick={load}>刷新</Button>
          </div>

          <Table<ExperienceSnapshot>
            rowKey={(row) => `${row.skill_name}:${row.snapshot_id}`}
            loading={loading}
            dataSource={snapshots}
            pagination={{ pageSize: 12 }}
            locale={{ emptyText: <Empty description="暂无经验样本。运行整定、辨识或评估流程后会自动生成 learning snapshot。" /> }}
            expandable={{
              expandedRowRender: (row) => (
                <Descriptions bordered size="small" column={1}>
                  <Descriptions.Item label="Payload Preview"><JsonBlock value={row.payload_preview} /></Descriptions.Item>
                  <Descriptions.Item label="Observable Outcomes"><JsonBlock value={row.observable_outcomes} /></Descriptions.Item>
                </Descriptions>
              ),
            }}
            columns={[
              { title: 'Skill', dataIndex: 'skill_name', width: 220, ellipsis: true },
              { title: 'Snapshot ID', dataIndex: 'snapshot_id', width: 180, ellipsis: true },
              { title: '来源', dataIndex: 'code_origin', width: 130, render: (value) => <Tag>{value || '-'}</Tag> },
              {
                title: 'Outcome',
                dataIndex: 'has_outcome',
                width: 110,
                render: (value: boolean) => <Tag color={value ? 'green' : 'default'}>{value ? '已回填' : '待回填'}</Tag>,
              },
              {
                title: 'Payload Keys',
                dataIndex: 'payload_keys',
                ellipsis: true,
                render: (value?: string[]) => value?.slice(0, 6).join(', ') || '-',
              },
              { title: '时间', dataIndex: 'ts', width: 180, render: (value?: number) => formatTime(value) },
              {
                title: '操作',
                width: 120,
                render: (_, row) => <Button size="small" type="link" onClick={() => openOutcomeModal(row)}>回填</Button>,
              },
            ]}
          />
        </ProCard>

        <Modal
          title={outcomeTarget ? `回填 Outcome：${outcomeTarget.skill_name}` : '回填 Outcome'}
          open={Boolean(outcomeTarget)}
          confirmLoading={savingOutcome}
          onOk={saveOutcome}
          onCancel={() => setOutcomeTarget(null)}
          okText="保存"
          cancelText="取消"
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            <Typography.Text type="secondary">
              用 JSON 对象记录工程复核或上线后的可观测结果，例如 human_label、delta_harris、delta_cpk。
            </Typography.Text>
            <Input.TextArea
              rows={10}
              value={outcomeText}
              onChange={(event) => setOutcomeText(event.target.value)}
            />
          </Space>
        </Modal>
      </Space>
    </PageContainer>
  );
}
