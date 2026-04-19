/** 会话历史页：列出所有 /api/tune/stream 与 /api/consult/stream 的运行记录，
 * 点击行展开详情：元数据 + 事件时间线 + LLM 思维链。
 */
import { useEffect, useMemo, useState } from 'react';
import {
  PageContainer,
  ProCard,
  ProTable,
  type ProColumns,
} from '@ant-design/pro-components';
import {
  Button,
  Drawer,
  Tag,
  Space,
  Typography,
  Descriptions,
  Timeline,
  Collapse,
  Empty,
  Popconfirm,
  message,
} from 'antd';
import {
  ReloadOutlined,
  DeleteOutlined,
  RobotOutlined,
  BulbOutlined,
} from '@ant-design/icons';
import {
  deleteSession,
  getSession,
  listSessions,
  type SessionDetail,
  type SessionMeta,
} from '@/services/api';

const { Text, Paragraph } = Typography;

function fmtDuration(s?: number): string {
  if (s == null) return '-';
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s - m * 60);
  return `${m}m${r}s`;
}

function statusTag(meta: SessionMeta) {
  if (meta.status === 'error') return <Tag color="error">异常</Tag>;
  if (meta.kind === 'consult') {
    return <Tag color={meta.status === 'ok' ? 'processing' : 'default'}>顾问会话</Tag>;
  }
  const passed = meta.summary?.passed;
  if (passed === true) return <Tag color="success">通过</Tag>;
  if (passed === false) return <Tag color="warning">未通过</Tag>;
  return <Tag>{meta.status ?? '-'}</Tag>;
}

function modeTag(mode?: string) {
  if (!mode) return null;
  const map: Record<string, { color: string; label: string }> = {
    llm: { color: 'purple', label: 'LLM 选窗' },
    deterministic: { color: 'default', label: '确定性' },
    fallback_deterministic: { color: 'orange', label: 'LLM 回退' },
    user_override: { color: 'blue', label: '人工指定' },
  };
  const m = map[mode] ?? { color: 'default', label: mode };
  return <Tag color={m.color}>{m.label}</Tag>;
}

export default function SessionsPage() {
  const [items, setItems] = useState<SessionMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const r = await listSessions({ limit: 200 });
      setItems(r.items);
    } catch (e) {
      message.error(`加载会话列表失败: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function openDetail(task_id: string) {
    setDrawerOpen(true);
    setDetailLoading(true);
    try {
      const d = await getSession(task_id);
      setDetail(d);
    } catch (e) {
      message.error(`加载会话详情失败: ${(e as Error).message}`);
      setDetail(null);
    } finally {
      setDetailLoading(false);
    }
  }

  async function handleDelete(task_id: string) {
    await deleteSession(task_id);
    message.success('已删除');
    refresh();
  }

  const columns: ProColumns<SessionMeta>[] = [
    {
      title: '时间',
      dataIndex: 'created_at',
      width: 170,
      render: (_, r) => (r.created_at || '').replace('T', ' ').slice(0, 19),
    },
    {
      title: '类型',
      dataIndex: 'kind',
      width: 90,
      filters: [
        { text: 'tune', value: 'tune' },
        { text: 'consult', value: 'consult' },
      ],
      onFilter: (v, r) => r.kind === v,
      render: (_, r) =>
        r.kind === 'tune' ? <Tag color="geekblue">整定</Tag> : <Tag color="cyan">顾问</Tag>,
    },
    {
      title: '状态',
      width: 90,
      render: (_, r) => statusTag(r),
    },
    {
      title: 'CSV',
      dataIndex: 'csv_name',
      ellipsis: true,
      render: (v) => v || '-',
    },
    {
      title: '回路',
      dataIndex: 'loop_type',
      width: 80,
      render: (v) => v || '-',
    },
    {
      title: '选窗',
      width: 130,
      render: (_, r) => {
        const s = r.summary;
        if (!s?.selection_mode) return '-';
        return (
          <Space size={4}>
            {modeTag(s.selection_mode)}
            <Text code>#{s.chosen_index}</Text>
          </Space>
        );
      },
    },
    {
      title: '评分',
      width: 100,
      render: (_, r) => {
        const s = r.summary;
        if (s?.final_rating == null) return '-';
        return (
          <Space size={4}>
            <Text strong>{s.final_rating.toFixed(2)}</Text>
            <Text type="secondary" style={{ fontSize: 12 }}>
              perf {s.performance_score?.toFixed(1)}
            </Text>
          </Space>
        );
      },
    },
    {
      title: '辨识 R²',
      width: 90,
      render: (_, r) =>
        r.summary?.r2_score == null ? '-' : r.summary.r2_score.toFixed(3),
    },
    {
      title: '用时',
      dataIndex: 'duration_s',
      width: 80,
      render: (v) => fmtDuration(v as number | undefined),
    },
    {
      title: '操作',
      width: 140,
      fixed: 'right',
      render: (_, r) => (
        <Space size={4}>
          <Button size="small" type="link" onClick={() => openDetail(r.task_id)}>
            详情
          </Button>
          <Popconfirm title="删除此会话？" onConfirm={() => handleDelete(r.task_id)}>
            <Button size="small" type="link" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <PageContainer
      title="会话历史"
      subTitle="所有 /api/tune/stream 与 /api/consult/stream 的运行记录"
      extra={[
        <Button key="r" icon={<ReloadOutlined />} onClick={refresh} loading={loading}>
          刷新
        </Button>,
      ]}
    >
      <ProCard>
        <ProTable<SessionMeta>
          rowKey="task_id"
          dataSource={items}
          columns={columns}
          loading={loading}
          search={false}
          pagination={{ pageSize: 20 }}
          options={{ density: false, fullScreen: true, reload: false, setting: false }}
          scroll={{ x: 1100 }}
        />
      </ProCard>

      <Drawer
        title={detail?.meta.task_id ? `会话 ${detail.meta.task_id}` : '会话详情'}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        width={760}
        loading={detailLoading}
      >
        {detail ? <SessionDetailView detail={detail} /> : <Empty />}
      </Drawer>
    </PageContainer>
  );
}

function SessionDetailView({ detail }: { detail: SessionDetail }) {
  const { meta, events } = detail;

  // 提炼事件子集供时间线
  const timelineItems = useMemo(() => {
    return events
      .filter((e) => {
        const t = e.type;
        return (
          t === 'session_start' ||
          t === 'stage' ||
          t === 'llm_thinking' ||
          t === 'llm_tool_call' ||
          t === 'llm_tool_result' ||
          t === 'llm_message' ||
          t === 'result' ||
          t === 'error'
        );
      })
      .map((e, idx) => {
        const t = String(e.type);
        const stage = (e.stage as string) || '';
        const status = (e.status as string) || '';
        const at = typeof e._t === 'number' ? `t+${e._t.toFixed(1)}s` : '';
        let color: string = 'blue';
        let label = `${t} ${stage} ${status}`.trim();
        let body: React.ReactNode = null;

        if (t === 'session_start') {
          color = 'gray';
          label = `会话开始 (${(e.kind as string) || '-'})`;
        } else if (t === 'stage') {
          color = status === 'done' ? 'green' : 'blue';
          label = `${stage} · ${status}`;
          const data = e.data as Record<string, unknown> | undefined;
          if (data && status === 'done') {
            body = (
              <pre style={{ fontSize: 12, margin: 0, color: '#666' }}>
                {JSON.stringify(data, null, 2).slice(0, 600)}
              </pre>
            );
          }
        } else if (t === 'llm_thinking') {
          color = 'purple';
          label = `🧠 LLM 思维链 (${(e.model as string) || ''}, ${stage})`;
          const rc = (e.reasoning_content as string) || '';
          const raw = (e.raw_text as string) || '';
          body = (
            <Collapse
              ghost
              items={[
                {
                  key: 'rc',
                  label: <Text type="secondary">reasoning_content ({rc.length} 字)</Text>,
                  children: (
                    <Paragraph
                      style={{
                        whiteSpace: 'pre-wrap',
                        fontSize: 12,
                        background: '#f6f5fb',
                        padding: 8,
                        borderRadius: 4,
                        maxHeight: 400,
                        overflow: 'auto',
                      }}
                    >
                      {rc || '(空)'}
                    </Paragraph>
                  ),
                },
                ...(raw
                  ? [
                      {
                        key: 'raw',
                        label: <Text type="secondary">最终 JSON 输出</Text>,
                        children: (
                          <pre style={{ fontSize: 12, margin: 0 }}>{raw}</pre>
                        ),
                      },
                    ]
                  : []),
              ]}
            />
          );
        } else if (t === 'llm_tool_call') {
          color = 'gold';
          label = `🔧 调用 ${e.tool_name as string}`;
          body = (
            <pre style={{ fontSize: 12, margin: 0 }}>
              {JSON.stringify(e.args, null, 2)}
            </pre>
          );
        } else if (t === 'llm_tool_result') {
          color = 'cyan';
          label = `✓ ${e.tool_name as string} 返回`;
          const summary = e.summary ?? e.result;
          body = (
            <pre style={{ fontSize: 12, margin: 0, maxHeight: 200, overflow: 'auto' }}>
              {JSON.stringify(summary, null, 2)}
            </pre>
          );
        } else if (t === 'llm_message') {
          color = 'blue';
          label = `💬 LLM 回复 (${(e.role as string) || 'assistant'})`;
          body = (
            <Paragraph
              style={{ whiteSpace: 'pre-wrap', margin: 0, fontSize: 13 }}
            >
              {(e.content as string) || ''}
            </Paragraph>
          );
        } else if (t === 'result') {
          color = 'green';
          label = '最终结果';
        } else if (t === 'error') {
          color = 'red';
          label = `错误: ${(e.error_code as string) || ''}`;
          body = <Text type="danger">{(e.message as string) || ''}</Text>;
        }

        return {
          key: idx,
          color,
          children: (
            <div>
              <div>
                <Text strong>{label}</Text>{' '}
                {at && <Text type="secondary" style={{ fontSize: 12 }}>{at}</Text>}
              </div>
              {body}
            </div>
          ),
        };
      });
  }, [events]);

  const s = meta.summary;
  return (
    <Space direction="vertical" style={{ width: '100%' }} size={16}>
      <Descriptions size="small" column={2} bordered>
        <Descriptions.Item label="task_id" span={2}>
          <Text code>{meta.task_id}</Text>
        </Descriptions.Item>
        <Descriptions.Item label="类型">{meta.kind}</Descriptions.Item>
        <Descriptions.Item label="状态">{statusTag(meta)}</Descriptions.Item>
        <Descriptions.Item label="开始">{(meta.created_at || '').replace('T', ' ').slice(0, 19)}</Descriptions.Item>
        <Descriptions.Item label="用时">{fmtDuration(meta.duration_s)}</Descriptions.Item>
        <Descriptions.Item label="CSV" span={2}>{meta.csv_name || '-'}</Descriptions.Item>
        <Descriptions.Item label="回路">{meta.loop_type || '-'}</Descriptions.Item>
        <Descriptions.Item label="LLM 顾问">
          {meta.use_llm_advisor ? <Tag color="purple" icon={<RobotOutlined />}>开启</Tag> : <Tag>关闭</Tag>}
        </Descriptions.Item>
        {s && (
          <>
            <Descriptions.Item label="选窗" span={2}>
              <Space>
                {modeTag(s.selection_mode)}
                <Text>窗口 #{s.chosen_index}</Text>
                {s.deterministic_index != null && (
                  <Text type="secondary">（确定性会选 #{s.deterministic_index}）</Text>
                )}
                {s.agreed_with_deterministic === false && <Tag color="orange">分歧</Tag>}
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="模型">{s.model_type ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="R²">{s.r2_score?.toFixed(4) ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="性能">{s.performance_score?.toFixed(2) ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="终评">{s.final_rating?.toFixed(2) ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="衰减比">{s.decay_ratio?.toFixed(3) ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="超调%">{s.overshoot_percent?.toFixed(2) ?? '-'}</Descriptions.Item>
          </>
        )}
        {meta.error && (
          <Descriptions.Item label="错误" span={2}>
            <Text type="danger">{meta.error}</Text>
          </Descriptions.Item>
        )}
      </Descriptions>

      <ProCard
        title={
          <Space>
            <BulbOutlined />
            事件时间线 ({events.length} 条)
          </Space>
        }
        size="small"
        bodyStyle={{ maxHeight: 600, overflow: 'auto' }}
      >
        <Timeline items={timelineItems} />
      </ProCard>
    </Space>
  );
}
