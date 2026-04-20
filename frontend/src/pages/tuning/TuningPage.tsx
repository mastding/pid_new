import { useCallback } from 'react';
import {
  PageContainer,
  ProCard,
} from '@ant-design/pro-components';
import {
  Upload,
  Button,
  Select,
  Steps,
  Result,
  Descriptions,
  Space,
  Tag,
  Table,
  Typography,
  message,
  Progress,
  Alert,
  Switch,
  Tooltip,
  Collapse,
} from 'antd';
import { UploadOutlined, RocketOutlined, ReloadOutlined, RobotOutlined, BulbOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import { tunePidStream } from '@/services/api';
import type { TuningResult, PipelineEvent, StrategyCandidate, WindowSelectionMeta, ModelReviewMeta, IdentificationAttempt } from '@/types/tuning';
import SimulationChart from '@/components/charts/SimulationChart';
import FitPreviewChart from '@/components/charts/FitPreviewChart';
import { useTuningStore, setTuningState, resetTuningState } from '@/stores/tuningStore';

const LOOP_TYPES = [
  { label: '流量', value: 'flow' },
  { label: '温度', value: 'temperature' },
  { label: '压力', value: 'pressure' },
  { label: '液位', value: 'level' },
];

const STAGE_LABELS: Record<string, string> = {
  data_analysis: '数据分析',
  identification: '系统辨识',
  tuning: 'PID 整定',
  evaluation: '性能评估',
};

const STAGE_KEYS = ['data_analysis', 'identification', 'tuning', 'evaluation'];

const STRATEGY_COLORS: Record<string, string> = {
  IMC: 'blue',
  LAMBDA: 'cyan',
  ZN: 'orange',
  CHR: 'purple',
};

const MODEL_TYPE_COLORS: Record<string, string> = {
  FO: 'default',
  FOPDT: 'blue',
  SOPDT: 'geekblue',
  SOPDT_UNDER: 'magenta',
  IPDT: 'volcano',
  IFOPDT: 'gold',
};

export default function TuningPage() {
  const { fileList, loopType, useLlmAdvisor, running, currentStage, stageData, windowSelection, modelReview, llmThinkingByStage, taskId, result, error } =
    useTuningStore();

  const handleRun = useCallback(() => {
    const file = fileList[0]?.originFileObj;
    if (!file) {
      message.warning('请先上传 CSV 文件');
      return;
    }

    setTuningState((s) => ({
      ...s,
      running: true,
      currentStage: 0,
      stageData: {},
      windowSelection: null,
      modelReview: null,
      llmThinking: null,
      llmThinkingByStage: {},
      taskId: null,
      result: null,
      error: null,
    }));

    tunePidStream(file, { loop_type: loopType, use_llm_advisor: useLlmAdvisor }, (event: Record<string, unknown>) => {
      const e = event as unknown as PipelineEvent;
      if (e.type === 'stage') {
        const se = e as { stage: string; status: string; data?: Record<string, unknown> };
        const idx = STAGE_KEYS.indexOf(se.stage);
        if (se.status === 'running' && idx >= 0) {
          setTuningState((s) => ({ ...s, currentStage: idx }));
        }
        if (se.status === 'done' && se.data) {
          if (se.stage === 'window_selection') {
            setTuningState((s) => ({
              ...s,
              windowSelection: se.data as unknown as WindowSelectionMeta,
            }));
          } else if (se.stage === 'model_review') {
            setTuningState((s) => ({
              ...s,
              modelReview: se.data as unknown as ModelReviewMeta,
            }));
          } else {
            setTuningState((s) => ({
              ...s,
              stageData: { ...s.stageData, [se.stage]: se.data as Record<string, unknown> },
            }));
          }
        }
      } else if ((e as { type: string }).type === 'session_start') {
        const ss = e as unknown as { task_id: string };
        setTuningState((s) => ({ ...s, taskId: ss.task_id }));
      } else if ((e as { type: string }).type === 'llm_thinking') {
        const lt = e as unknown as { stage: string; model: string; reasoning_content: string; raw_text: string };
        const payload = {
          stage: lt.stage,
          model: lt.model,
          reasoning_content: lt.reasoning_content || '',
          raw_text: lt.raw_text || '',
        };
        setTuningState((s) => ({
          ...s,
          llmThinking: payload,
          llmThinkingByStage: { ...s.llmThinkingByStage, [lt.stage]: payload },
        }));
      } else if (e.type === 'result') {
        setTuningState((s) => ({ ...s, result: (e as { data: TuningResult }).data }));
      } else if (e.type === 'error') {
        setTuningState((s) => ({
          ...s,
          error: (e as { message: string }).message,
          running: false,
        }));
      } else if (e.type === 'done') {
        setTuningState((s) => ({ ...s, running: false, currentStage: STAGE_KEYS.length }));
      }
    });
  }, [fileList, loopType, useLlmAdvisor]);

  const scoreColor = (score: number) =>
    score >= 8 ? '#52c41a' : score >= 6 ? '#faad14' : '#ff4d4f';

  return (
    <PageContainer title="PID 智能整定" subTitle="上传数据 → 自动辨识 → 参数整定 → 性能评估">
      {/* Upload & Config */}
      <ProCard style={{ marginBottom: 16 }}>
        <Space size="large" align="center">
          <Upload
            accept=".csv"
            maxCount={1}
            fileList={fileList}
            beforeUpload={() => false}
            onChange={({ fileList: fl }) => setTuningState((s) => ({ ...s, fileList: fl }))}
          >
            <Button icon={<UploadOutlined />}>选择 CSV 文件</Button>
          </Upload>
          <Select
            value={loopType}
            onChange={(v) => setTuningState((s) => ({ ...s, loopType: v }))}
            options={LOOP_TYPES}
            style={{ width: 120 }}
          />
          <Tooltip title="开启后由 deepseek-reasoner 在多个候选窗口里挑选最优窗口；关闭则按 quality_score 最高自动选">
            <Space size={4}>
              <RobotOutlined style={{ color: useLlmAdvisor ? '#1677ff' : '#bbb' }} />
              <Switch
                checked={useLlmAdvisor}
                onChange={(v) => setTuningState((s) => ({ ...s, useLlmAdvisor: v }))}
                checkedChildren="LLM 顾问"
                unCheckedChildren="确定性"
              />
            </Space>
          </Tooltip>
          <Button
            type="primary"
            icon={<RocketOutlined />}
            loading={running}
            onClick={handleRun}
            disabled={fileList.length === 0}
          >
            开始整定
          </Button>
          {(result || error) && (
            <Button icon={<ReloadOutlined />} onClick={resetTuningState}>
              重置
            </Button>
          )}
        </Space>
      </ProCard>

      {/* Progress Steps */}
      {currentStage >= 0 && (
        <ProCard style={{ marginBottom: 16 }}>
          <Steps
            current={currentStage}
            status={error ? 'error' : running ? 'process' : 'finish'}
            items={STAGE_KEYS.map((key) => {
              const d = stageData[key] as Record<string, unknown> | undefined;
              return {
                title: STAGE_LABELS[key],
                description: d
                  ? key === 'data_analysis'
                    ? `${d.data_points} 点 / ${d.usable_windows ?? d.candidate_windows} 窗口`
                    : key === 'identification'
                    ? `${d.model_type} R²=${typeof d.r2_score === 'number' ? d.r2_score.toFixed(3) : '-'}`
                    : key === 'tuning'
                    ? `${d.strategy} Kp=${typeof d.Kp === 'number' ? d.Kp.toFixed(3) : '-'}`
                    : key === 'evaluation'
                    ? `${d.passed ? '✓ 通过' : '✗ 未通过'} 评分 ${typeof d.final_rating === 'number' ? d.final_rating.toFixed(1) : '-'}`
                    : undefined
                  : undefined,
              };
            })}
          />
        </ProCard>
      )}

      {/* Window Selection (LLM Advisor) */}
      {windowSelection && (
        <ProCard
          title={
            <Space>
              <RobotOutlined />
              <span>窗口选择</span>
              {windowSelection.mode === 'llm' && <Tag color="processing">LLM 顾问</Tag>}
              {windowSelection.mode === 'fallback_deterministic' && <Tag color="warning">LLM 失败 → 回退</Tag>}
              {windowSelection.mode === 'deterministic' && <Tag>按分数选窗</Tag>}
              {windowSelection.mode === 'user_override' && <Tag color="purple">手动指定</Tag>}
            </Space>
          }
          style={{ marginBottom: 16 }}
          extra={
            windowSelection.mode === 'llm' && windowSelection.agreed_with_deterministic !== undefined && (
              <Tag color={windowSelection.agreed_with_deterministic ? 'success' : 'orange'}>
                {windowSelection.agreed_with_deterministic
                  ? '与 baseline 一致'
                  : '与 baseline 不同'}
              </Tag>
            )
          }
        >
          <Descriptions column={3} size="small" style={{ marginBottom: 12 }}>
            <Descriptions.Item label="选中窗口 #">
              <Tag color="blue">{windowSelection.chosen_index}</Tag>
              {windowSelection.chosen_window_summary?.source && (
                <span style={{ color: '#888', fontSize: 12 }}>
                  {' '}({windowSelection.chosen_window_summary.source})
                </span>
              )}
            </Descriptions.Item>
            <Descriptions.Item label="确定性 baseline #">
              <Tag>{windowSelection.deterministic_index}</Tag>
              <span style={{ color: '#888', fontSize: 12 }}>
                {' '}score={windowSelection.deterministic_score.toFixed(3)}
              </span>
            </Descriptions.Item>
            {windowSelection.llm_reasoning_chain_len !== undefined && (
              <Descriptions.Item label="LLM 思考链">
                {windowSelection.llm_reasoning_chain_len} 字
              </Descriptions.Item>
            )}
          </Descriptions>
          <Alert
            type={windowSelection.mode === 'llm' ? 'info' : 'warning'}
            message="选择理由"
            description={windowSelection.reasoning}
            showIcon
          />
          {llmThinkingByStage['window_selection']?.reasoning_content && (
            <Collapse
              ghost
              style={{ marginTop: 12 }}
              items={[
                {
                  key: 'rc',
                  label: (
                    <Space>
                      <BulbOutlined style={{ color: '#722ed1' }} />
                      <Typography.Text strong>LLM 思维链</Typography.Text>
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        {llmThinkingByStage['window_selection'].model} · {llmThinkingByStage['window_selection'].reasoning_content.length} 字
                      </Typography.Text>
                    </Space>
                  ),
                  children: (
                    <Typography.Paragraph
                      style={{
                        whiteSpace: 'pre-wrap',
                        fontSize: 12,
                        background: '#f6f5fb',
                        padding: 12,
                        borderRadius: 4,
                        maxHeight: 360,
                        overflow: 'auto',
                        margin: 0,
                      }}
                    >
                      {llmThinkingByStage['window_selection'].reasoning_content}
                    </Typography.Paragraph>
                  ),
                },
              ]}
            />
          )}
          {taskId && (
            <div style={{ marginTop: 8, textAlign: 'right' }}>
              <Link to="/sessions">
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  会话 ID: <Typography.Text code>{taskId}</Typography.Text> · 进入会话历史 →
                </Typography.Text>
              </Link>
            </div>
          )}
        </ProCard>
      )}

      {/* Identification Attempts: 各模型 × 窗口的拟合对比（评审前的原始证据） */}
      {(() => {
        const idStage = stageData['identification'] as Record<string, unknown> | undefined;
        const attempts = (idStage?.attempts as IdentificationAttempt[] | undefined) ?? [];
        if (!attempts.length) return null;
        const bestWindow = (idStage?.best_window_source as string) || '';
        const bestType = (idStage?.model_type as string) || '';
        return (
          <ProCard
            title={
              <Space>
                <span>辨识结果对比</span>
                <Tag>{attempts.length} 次尝试</Tag>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  各候选模型 × 窗口的拟合表现，按 fit_score 降序
                </Typography.Text>
              </Space>
            }
            style={{ marginBottom: 16 }}
          >
            <Table
              size="small"
              rowKey={(r, i) => `${r.model_type}-${r.window_source}-${i ?? 0}`}
              pagination={false}
              dataSource={attempts}
              onRow={(r) => ({
                style:
                  r.success && r.model_type === bestType && r.window_source === bestWindow
                    ? { background: '#fffbe6' }
                    : !r.success
                    ? { background: '#fff1f0', opacity: 0.7 }
                    : {},
              })}
              columns={[
                {
                  title: '模型',
                  dataIndex: 'model_type',
                  width: 140,
                  render: (v: string, r: IdentificationAttempt) => (
                    <Space size={4}>
                      <Tag color={MODEL_TYPE_COLORS[v] ?? 'default'}>{v}</Tag>
                      {r.success && r.model_type === bestType && r.window_source === bestWindow && (
                        <Tag color="gold">★ 选中</Tag>
                      )}
                      {r.degenerate_T && <Tag color="error">T 塌缩</Tag>}
                    </Space>
                  ),
                },
                {
                  title: '窗口',
                  dataIndex: 'window_source',
                  width: 120,
                  render: (v: string) => <Typography.Text code style={{ fontSize: 12 }}>{v || '-'}</Typography.Text>,
                },
                {
                  title: 'K',
                  dataIndex: 'K',
                  width: 90,
                  align: 'right',
                  render: (v: number | undefined) => (typeof v === 'number' ? v.toFixed(3) : '-'),
                },
                {
                  title: 'T (s)',
                  dataIndex: 'T',
                  width: 80,
                  align: 'right',
                  render: (v: number | undefined, r: IdentificationAttempt) => {
                    if (r.model_type === 'SOPDT' && r.T1 && r.T2)
                      return `${r.T1.toFixed(1)}+${r.T2.toFixed(1)}`;
                    return typeof v === 'number' ? v.toFixed(2) : '-';
                  },
                },
                {
                  title: 'ζ',
                  dataIndex: 'zeta',
                  width: 70,
                  align: 'right',
                  render: (v: number | undefined, r: IdentificationAttempt) =>
                    r.model_type === 'SOPDT_UNDER' && typeof v === 'number' ? v.toFixed(3) : '-',
                },
                {
                  title: 'L (s)',
                  dataIndex: 'L',
                  width: 80,
                  align: 'right',
                  render: (v: number | undefined) => (typeof v === 'number' ? v.toFixed(2) : '-'),
                },
                {
                  title: 'R²',
                  dataIndex: 'r2_score',
                  width: 80,
                  align: 'right',
                  render: (v: number | undefined) => {
                    if (typeof v !== 'number') return '-';
                    const color = v >= 0.8 ? '#389e0d' : v >= 0.5 ? '#d48806' : '#cf1322';
                    return <span style={{ color }}>{v.toFixed(3)}</span>;
                  },
                },
                {
                  title: 'NRMSE',
                  dataIndex: 'normalized_rmse',
                  width: 80,
                  align: 'right',
                  render: (v: number | undefined) =>
                    typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '-',
                },
                {
                  title: 'fit_score',
                  dataIndex: 'fit_score',
                  width: 90,
                  align: 'right',
                  render: (v: number | undefined) => (typeof v === 'number' ? v.toFixed(2) : '-'),
                },
                {
                  title: '置信度',
                  dataIndex: 'confidence',
                  width: 90,
                  align: 'right',
                  render: (v: number | undefined) =>
                    typeof v === 'number' ? `${(v * 100).toFixed(0)}%` : '-',
                },
                {
                  title: '状态',
                  dataIndex: 'success',
                  width: 120,
                  render: (ok: boolean, r: IdentificationAttempt) =>
                    ok ? (
                      <Tag color="success">成功</Tag>
                    ) : (
                      <Tooltip title={r.error || '拟合失败'}>
                        <Tag color="error">失败</Tag>
                      </Tooltip>
                    ),
                },
              ]}
            />
            <Typography.Paragraph type="secondary" style={{ fontSize: 12, margin: '8px 0 0' }}>
              说明：fit_score = R² - AIC 惩罚（参数越多惩罚越大）。★ 为算法按此分数选中，交给 LLM 评审的候选。
            </Typography.Paragraph>
          </ProCard>
        );
      })()}

      {/* Model Review (LLM verdict on identification) */}
      {modelReview && (
        <ProCard
          title={
            <Space>
              <RobotOutlined />
              <span>辨识结果评审</span>
              {modelReview.verdict === 'accept' && <Tag color="success">采纳</Tag>}
              {modelReview.verdict === 'downgrade' && <Tag color="warning">降级（限制评分）</Tag>}
              {modelReview.verdict === 'reject' && <Tag color="error">拒绝</Tag>}
            </Space>
          }
          style={{ marginBottom: 16 }}
        >
          <Alert
            type={
              modelReview.verdict === 'accept' ? 'success'
              : modelReview.verdict === 'downgrade' ? 'warning'
              : 'error'
            }
            message="评审结论"
            description={modelReview.reason}
            showIcon
            style={{ marginBottom: 12 }}
          />
          {modelReview.concerns && modelReview.concerns.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <Typography.Text strong>具体担忧：</Typography.Text>
              <ul style={{ marginTop: 4, marginBottom: 0, paddingLeft: 20 }}>
                {modelReview.concerns.map((c, i) => (
                  <li key={i} style={{ fontSize: 13 }}>{c}</li>
                ))}
              </ul>
            </div>
          )}
          {llmThinkingByStage['model_review']?.reasoning_content && (
            <Collapse
              ghost
              items={[
                {
                  key: 'rc',
                  label: (
                    <Space>
                      <BulbOutlined style={{ color: '#722ed1' }} />
                      <Typography.Text strong>LLM 思维链</Typography.Text>
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        {llmThinkingByStage['model_review'].model} · {llmThinkingByStage['model_review'].reasoning_content.length} 字
                      </Typography.Text>
                    </Space>
                  ),
                  children: (
                    <Typography.Paragraph
                      style={{
                        whiteSpace: 'pre-wrap',
                        fontSize: 12,
                        background: '#f6f5fb',
                        padding: 12,
                        borderRadius: 4,
                        maxHeight: 360,
                        overflow: 'auto',
                        margin: 0,
                      }}
                    >
                      {llmThinkingByStage['model_review'].reasoning_content}
                    </Typography.Paragraph>
                  ),
                },
              ]}
            />
          )}
        </ProCard>
      )}

      {/* Error */}
      {error && (
        <ProCard style={{ marginBottom: 16 }}>
          <Result status="error" title="整定失败" subTitle={error} />
        </ProCard>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Model card */}
          <ProCard title="辨识模型" style={{ marginBottom: 16 }} extra={
            <Space>
              <Tag color={MODEL_TYPE_COLORS[result.model.model_type] ?? 'default'}>
                {result.model.model_type}
              </Tag>
              <Tag color={
                result.model.confidence >= 0.85 ? 'success' :
                result.model.confidence >= 0.6 ? 'warning' : 'error'
              }>
                {result.model.confidence_quality}
              </Tag>
            </Space>
          }>
            <Descriptions column={4}>
              <Descriptions.Item label="增益 K">{result.model.K.toFixed(4)}</Descriptions.Item>
              <Descriptions.Item label="时间常数 T">{result.model.T.toFixed(2)} s</Descriptions.Item>
              {result.model.model_type === 'SOPDT' && (
                <>
                  <Descriptions.Item label="T₁">{result.model.T1.toFixed(2)} s</Descriptions.Item>
                  <Descriptions.Item label="T₂">{result.model.T2.toFixed(2)} s</Descriptions.Item>
                </>
              )}
              <Descriptions.Item label="死区时间 L">{result.model.L.toFixed(2)} s</Descriptions.Item>
              <Descriptions.Item label="R²">{result.model.r2_score.toFixed(4)}</Descriptions.Item>
              <Descriptions.Item label="NRMSE">{(result.model.normalized_rmse * 100).toFixed(2)}%</Descriptions.Item>
              <Descriptions.Item label="置信度">{(result.model.confidence * 100).toFixed(0)}%</Descriptions.Item>
            </Descriptions>

            {/* T 塌缩 / 低置信度 警告 */}
            {(() => {
              const lt = (result.loop_type || '').toLowerCase();
              const minT: Record<string, number> = { flow: 1, pressure: 5, temperature: 30, level: 60 };
              const minVal = minT[lt] ?? 1;
              const tEff =
                result.model.model_type === 'SOPDT'
                  ? (result.model.T1 || 0) + (result.model.T2 || 0)
                  : result.model.T || 0;
              const degenerate = result.model.model_type !== 'IPDT' && tEff < minVal;
              const lowConf = result.model.confidence < 0.5;
              if (!degenerate && !lowConf) return null;
              const msgs: string[] = [];
              if (degenerate)
                msgs.push(`模型时间常数 T=${tEff.toFixed(2)}s 低于 ${result.loop_type} 回路最小合理值 ${minVal}s，模型可能塌缩为纯比例环节，整定参数不可信`);
              if (lowConf)
                msgs.push(`模型置信度 ${(result.model.confidence * 100).toFixed(0)}% < 50%，整定结论需人工复核`);
              return (
                <Alert
                  type="error"
                  showIcon
                  style={{ marginTop: 12 }}
                  message="模型可信度警告"
                  description={<ul style={{ margin: 0, paddingLeft: 20 }}>{msgs.map((m, i) => <li key={i}>{m}</li>)}</ul>}
                />
              );
            })()}

            {/* Fit preview chart */}
            {result.model.fit_preview?.pv_actual && result.model.fit_preview.pv_actual.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div style={{ color: '#888', fontSize: 12, marginBottom: 8 }}>模型拟合预览</div>
                <FitPreviewChart
                  fitPreview={result.model.fit_preview}
                  dt={result.data_analysis.sampling_time}
                />
              </div>
            )}
          </ProCard>

          {/* PID params card */}
          <ProCard title="PID 参数" style={{ marginBottom: 16 }} extra={
            <Tag color={STRATEGY_COLORS[result.pid_params.strategy] ?? 'default'}>
              {result.pid_params.strategy}
            </Tag>
          }>
            <Descriptions column={4} style={{ marginBottom: 16 }}>
              <Descriptions.Item label="Kp">{result.pid_params.Kp.toFixed(4)}</Descriptions.Item>
              <Descriptions.Item label={<Tooltip title="比例带 PB = 100 / Kp，DCS 常用表达">PB</Tooltip>}>
                {result.pid_params.Kp > 0 ? `${(100 / result.pid_params.Kp).toFixed(2)} %` : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="Ki">{result.pid_params.Ki.toFixed(6)}</Descriptions.Item>
              <Descriptions.Item label="Kd">{result.pid_params.Kd.toFixed(4)}</Descriptions.Item>
              <Descriptions.Item label="Ti">{result.pid_params.Ti.toFixed(2)} s</Descriptions.Item>
              <Descriptions.Item label="Td">{result.pid_params.Td.toFixed(2)} s</Descriptions.Item>
            </Descriptions>

            {/* Strategy candidates table */}
            {result.pid_params.candidates?.length > 0 && (
              <Table<StrategyCandidate>
                size="small"
                dataSource={result.pid_params.candidates}
                rowKey="strategy"
                pagination={false}
                rowClassName={(r) => r.is_recommended ? 'ant-table-row-selected' : ''}
                columns={[
                  {
                    title: '策略', dataIndex: 'strategy', width: 80,
                    render: (v: string, r: StrategyCandidate) => (
                      <Space>
                        <Tag color={STRATEGY_COLORS[v] ?? 'default'}>{v}</Tag>
                        {r.is_recommended && <Tag color="success">推荐</Tag>}
                      </Space>
                    ),
                  },
                  { title: 'Kp', dataIndex: 'Kp', render: (v: number) => v.toFixed(4) },
                  {
                    title: 'PB (%)', dataIndex: 'Kp',
                    render: (v: number) => v > 0 ? (100 / v).toFixed(2) : '-',
                  },
                  { title: 'Ki', dataIndex: 'Ki', render: (v: number) => v.toFixed(6) },
                  { title: 'Kd', dataIndex: 'Kd', render: (v: number) => v.toFixed(4) },
                  { title: 'Ti (s)', dataIndex: 'Ti', render: (v: number) => v.toFixed(2) },
                  { title: 'Td (s)', dataIndex: 'Td', render: (v: number) => v.toFixed(2) },
                  { title: '说明', dataIndex: 'description', ellipsis: true },
                ]}
              />
            )}
          </ProCard>

          {/* Evaluation card */}
          <ProCard title="性能评估" style={{ marginBottom: 16 }} extra={
            <Tag color={result.evaluation.passed ? 'success' : 'error'} style={{ fontSize: 14 }}>
              {result.evaluation.passed ? '✓ 可以上线' : '✗ 需要优化'}
            </Tag>
          }>
            {/* Scores row */}
            <Space size={40} style={{ marginBottom: 20 }}>
              {[
                { label: '性能评分', value: result.evaluation.performance_score },
                { label: '综合评分', value: result.evaluation.final_rating },
                { label: '就绪评分', value: result.evaluation.readiness_score },
                { label: '鲁棒评分', value: result.evaluation.robustness_score },
              ].map(({ label, value }) => (
                <div key={label} style={{ textAlign: 'center' }}>
                  <Progress
                    type="circle"
                    percent={value * 10}
                    format={() => value.toFixed(1)}
                    strokeColor={scoreColor(value)}
                    size={72}
                  />
                  <div style={{ marginTop: 4, fontSize: 12, color: '#888' }}>{label}</div>
                </div>
              ))}
            </Space>

            <Descriptions column={4} style={{ marginBottom: 16 }}>
              <Descriptions.Item label="超调量">{result.evaluation.overshoot_percent.toFixed(1)}%</Descriptions.Item>
              <Descriptions.Item label="调节时间">{result.evaluation.settling_time_s.toFixed(1)} s</Descriptions.Item>
              <Descriptions.Item label="稳态误差">{result.evaluation.steady_state_error.toFixed(2)}%</Descriptions.Item>
              <Descriptions.Item label="振荡次数">{result.evaluation.oscillation_count}</Descriptions.Item>
              <Descriptions.Item label="MV 饱和">{result.evaluation.mv_saturation_pct.toFixed(1)}%</Descriptions.Item>
              <Descriptions.Item label="稳定性">
                <Tag color={result.evaluation.is_stable ? 'success' : 'error'}>
                  {result.evaluation.is_stable ? '稳定' : '不稳定'}
                </Tag>
              </Descriptions.Item>
            </Descriptions>

            {/* Reality check 与评分封顶提示 */}
            {(result.evaluation.reality_check_diverged ||
              (result.evaluation.score_caps_applied && result.evaluation.score_caps_applied.length > 0)) && (
              <Alert
                type="error"
                showIcon
                style={{ marginBottom: 12 }}
                message="评估自检触发"
                description={
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {result.evaluation.reality_check_diverged && (
                      <li>
                        Reality check：用 {result.loop_type} 典型时间常数（
                        {result.evaluation.reality_check_typical_T} s）仿真评分仅
                        {result.evaluation.reality_check_score?.toFixed(1)}，与名义评分差距过大，
                        提示辨识模型可能塌缩
                      </li>
                    )}
                    {(result.evaluation.score_caps_applied || []).map((r: string, i: number) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                }
              />
            )}

            <Alert
              type={result.evaluation.passed ? 'success' : 'warning'}
              message={result.evaluation.recommendation}
              style={{ marginBottom: 16 }}
            />

            {/* Closed-loop simulation chart */}
            {result.evaluation.simulation?.pv_history?.length > 0 && (
              <>
                <div style={{ color: '#888', fontSize: 12, marginBottom: 8 }}>
                  闭环仿真曲线（SP 50→60 阶跃）
                </div>
                <SimulationChart simulation={result.evaluation.simulation} />
              </>
            )}
          </ProCard>
        </>
      )}
    </PageContainer>
  );
}
