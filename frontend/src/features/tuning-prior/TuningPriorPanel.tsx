import type { Dispatch, SetStateAction } from 'react';
import type { Dayjs } from 'dayjs';
import {
  Alert,
  Button,
  Collapse,
  DatePicker,
  Descriptions,
  Empty,
  Select,
  Space,
  Table,
  Tag,
  Typography,
} from 'antd';
import { RobotOutlined, SyncOutlined } from '@ant-design/icons';

import type { HistoryLoop, HistoryLoopTuningPrior } from '@/services/api';

type FeatureRangeOption = {
  label: string;
  value: string;
  seconds?: number;
};

type PriorEvidenceRow = {
  kind: string;
  name?: string;
  type?: string;
  passed?: boolean;
  severity?: string;
  message?: string;
};

interface TuningPriorPanelProps {
  loops: HistoryLoop[];
  selectedLoopId?: string;
  selectedLoop?: HistoryLoop;
  loopTypeLabel: Record<string, string>;
  featureRangeOptions: FeatureRangeOption[];
  tuningPriorRangePreset: string;
  tuningPriorCustomRange: [Dayjs | null, Dayjs | null] | null;
  tuningPriorCoreData?: HistoryLoopTuningPrior | null;
  tuningPriorOntologyData?: HistoryLoopTuningPrior | null;
  tuningPriorReviewData?: HistoryLoopTuningPrior | null;
  tuningPriorCoreLoading: boolean;
  tuningPriorOntologyLoading: boolean;
  tuningPriorReviewLoading: boolean;
  tuningPriorCoreError?: string | null;
  tuningPriorOntologyError?: string | null;
  tuningPriorReviewError?: string | null;
  onLoopChange: Dispatch<SetStateAction<string | undefined>>;
  onRangePresetChange: Dispatch<SetStateAction<string>>;
  onCustomRangeChange: Dispatch<SetStateAction<[Dayjs | null, Dayjs | null] | null>>;
  onLoadCore: () => void;
  onLoadOntology: () => void;
  onLoadReview: () => void;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  formatRange: (min?: number | null, max?: number | null, digits?: number) => string;
  formatProcessDirection: (value?: string | null) => string;
  operatingConditionText: (value?: string) => string;
  monitoringStatusText: (status?: string) => string;
  gateDecisionText: (decision?: string) => string;
  gateCheckLabel: (value?: string) => string;
  tagColor: (level?: string) => string;
}

export function TuningPriorPanel({
  loops,
  selectedLoopId,
  selectedLoop,
  loopTypeLabel,
  featureRangeOptions,
  tuningPriorRangePreset,
  tuningPriorCustomRange,
  tuningPriorCoreData,
  tuningPriorOntologyData,
  tuningPriorReviewData,
  tuningPriorCoreLoading,
  tuningPriorOntologyLoading,
  tuningPriorReviewLoading,
  tuningPriorCoreError,
  tuningPriorOntologyError,
  tuningPriorReviewError,
  onLoopChange,
  onRangePresetChange,
  onCustomRangeChange,
  onLoadCore,
  onLoadOntology,
  onLoadReview,
  formatNumber,
  formatPercentValue,
  formatRange,
  formatProcessDirection,
  operatingConditionText,
  monitoringStatusText,
  gateDecisionText,
  gateCheckLabel,
  tagColor,
}: TuningPriorPanelProps) {
  const priorFeatures = tuningPriorCoreData?.features;
  const priorMonitoring = tuningPriorCoreData?.monitoring?.monitoring;
  const priorAssessment = tuningPriorCoreData?.assessment;
  const priorOntology = tuningPriorOntologyData?.ontology;
  const readiness = priorAssessment?.tuning_readiness;
  const gateRows = readiness?.gate_checks ?? [];
  const diagnosisFlags = priorAssessment?.diagnostics?.flags ?? [];
  const ontologyContent = priorOntology?.content || priorOntology?.error || '';
  const priorEvidenceRows: PriorEvidenceRow[] = [
    ...gateRows.map((item) => ({ ...item, kind: '准入校验' })),
    ...diagnosisFlags.map((item) => ({ ...item, kind: '诊断标记', passed: false })),
  ];
  const priorRangeLabel = tuningPriorRangePreset === 'custom'
    ? '自定义区间'
    : featureRangeOptions.find((item) => item.value === tuningPriorRangePreset)?.label ?? tuningPriorRangePreset;

  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">整定先验</div>
            <Typography.Text type="secondary">
              先选择回路和时间范围，再按需分别生成核心上下文、本体上下文和大模型先验评审；先验只作为建议，不作为强约束拦截。
            </Typography.Text>
          </div>
          <Space wrap>
            <Select
              size="small"
              style={{ minWidth: 320 }}
              value={selectedLoopId}
              placeholder="选择回路"
              onChange={onLoopChange}
              options={loops.map((loop) => ({
                value: loop.loop_id,
                label: loop.loop_id + ' · ' + (loopTypeLabel[loop.loop_type] ?? loop.loop_type),
              }))}
            />
            <Select
              size="small"
              style={{ width: 150 }}
              value={tuningPriorRangePreset}
              onChange={onRangePresetChange}
              options={featureRangeOptions.map((item) => ({ label: item.label, value: item.value }))}
            />
            {tuningPriorRangePreset === 'custom' && (
              <DatePicker.RangePicker
                size="small"
                showTime
                value={tuningPriorCustomRange}
                onChange={onCustomRangeChange}
              />
            )}
          </Space>
        </div>
        {selectedLoop ? (
          <Descriptions bordered column={4} size="small" className="industrial-descriptions">
            <Descriptions.Item label="当前回路">{selectedLoop.loop_id}</Descriptions.Item>
            <Descriptions.Item label="回路类型">{loopTypeLabel[selectedLoop.loop_type] ?? selectedLoop.loop_type}</Descriptions.Item>
            <Descriptions.Item label="时间范围">{priorRangeLabel}</Descriptions.Item>
            <Descriptions.Item label="数据点数">{priorFeatures?.data_profile?.row_count ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="区间开始">{priorFeatures?.data_profile?.time_start || '-'}</Descriptions.Item>
            <Descriptions.Item label="区间结束">{priorFeatures?.data_profile?.time_end || '-'}</Descriptions.Item>
            <Descriptions.Item label="采样周期">{formatNumber(priorFeatures?.data_profile?.sample_time_median_s, 1)}s</Descriptions.Item>
            <Descriptions.Item label="本体状态">
              <Tag color={priorOntology?.content ? 'green' : priorOntology?.error ? 'orange' : 'default'}>
                {priorOntology?.content ? '已返回' : priorOntology?.error ? '异常/降级' : '待查询'}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="评审状态">
              <Tag color={tuningPriorReviewData?.review ? 'green' : tuningPriorReviewError ? 'orange' : 'default'}>
                {tuningPriorReviewData?.review ? '已生成' : tuningPriorReviewError ? '异常/降级' : '待评审'}
              </Tag>
            </Descriptions.Item>
          </Descriptions>
        ) : <Empty description="请先选择回路" />}
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">1 核心指标与评估诊断上下文</div>
            <Typography.Text type="secondary">来自画像、监控快照、准入评估和诊断标记。</Typography.Text>
          </div>
          <Space wrap>
            <Button
              size="small"
              icon={<SyncOutlined />}
              loading={tuningPriorCoreLoading}
              disabled={!selectedLoop}
              onClick={onLoadCore}
            >
              生成核心上下文
            </Button>
            <Tag color={priorMonitoring?.status === 'normal' ? 'green' : priorMonitoring?.status ? 'orange' : 'default'}>
              监控状态：{monitoringStatusText(priorMonitoring?.status)}
            </Tag>
            <Tag color={tagColor(readiness?.level)}>{formatPercentValue(readiness?.score, 0)}</Tag>
          </Space>
        </div>
        {tuningPriorCoreLoading ? (
          <Alert className="agent-alert" type="info" showIcon message="正在生成核心上下文" description="正在按所选时间范围聚合画像、监控、评估和诊断结果。" />
        ) : tuningPriorCoreError ? (
          <Alert className="agent-alert" type="error" showIcon message="核心上下文加载失败" description={tuningPriorCoreError} />
        ) : tuningPriorCoreData ? (
          <div className="page-stack compact-stack">
            <Descriptions bordered column={4} size="small" className="industrial-descriptions">
              <Descriptions.Item label="控制性能分">{formatPercentValue(priorAssessment?.performance?.score, 0)}</Descriptions.Item>
              <Descriptions.Item label="整定准备度">{gateDecisionText(readiness?.decision)}</Descriptions.Item>
              <Descriptions.Item label="可辨识性">{formatPercentValue(priorAssessment?.identification_suitability?.score, 0)}</Descriptions.Item>
              <Descriptions.Item label="数据质量">{formatPercentValue(priorAssessment?.data_quality?.score, 0)}</Descriptions.Item>
              <Descriptions.Item label="PV 范围">{formatRange(priorFeatures?.pv_stats?.min, priorFeatures?.pv_stats?.max, 3)}</Descriptions.Item>
              <Descriptions.Item label="MV 范围">{formatRange(priorFeatures?.mv_stats?.min, priorFeatures?.mv_stats?.max, 3)}</Descriptions.Item>
              <Descriptions.Item label="MV 饱和比例">{formatPercentValue(priorFeatures?.constraint_raw?.mv_saturation_ratio, 2)}</Descriptions.Item>
              <Descriptions.Item label="过程方向">{formatProcessDirection(priorFeatures?.pv_mv_relation_raw?.process_direction)}</Descriptions.Item>
              <Descriptions.Item label="运行工况">{operatingConditionText(priorFeatures?.operating_condition_profile?.condition_label)}</Descriptions.Item>
              <Descriptions.Item label="振荡状态">{priorFeatures?.oscillation_raw?.detected ? '疑似振荡' : '未检测到明显振荡'}</Descriptions.Item>
              <Descriptions.Item label="噪声水平">{priorMonitoring?.data_health?.pv_snr_db === undefined ? '-' : formatNumber(priorMonitoring.data_health.pv_snr_db, 2) + ' dB'}</Descriptions.Item>
              <Descriptions.Item label="报警数量">{priorMonitoring?.alerts?.length ?? 0}</Descriptions.Item>
            </Descriptions>
            <Table<PriorEvidenceRow>
              size="small"
              pagination={false}
              rowKey={(row, index) => (row.name || row.type || 'row') + '-' + index}
              dataSource={priorEvidenceRows}
              columns={[
                { title: '类别', dataIndex: 'kind', width: 120 },
                { title: '项目', dataIndex: 'name', width: 180, render: (value: string, row: PriorEvidenceRow) => value ? gateCheckLabel(value) : row.type ?? '-' },
                { title: '结果/级别', dataIndex: 'severity', width: 140, render: (value: string, row: PriorEvidenceRow) => row.passed === undefined ? <Tag color={tagColor(value)}>{value || '-'}</Tag> : <Tag color={row.passed ? 'green' : 'orange'}>{row.passed ? '通过' : '提醒'}</Tag> },
                { title: '说明', dataIndex: 'message' },
              ]}
            />
          </div>
        ) : <Empty description="暂无核心上下文，请点击“生成核心上下文”" />}
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">2 本体查询与返回结果</div>
            <Typography.Text type="secondary">本体结果单独展示，便于核对后续解释是否真正引用了装置、变量、增益方向和时间尺度知识。</Typography.Text>
          </div>
          <Space wrap>
            <Button
              size="small"
              icon={<SyncOutlined />}
              loading={tuningPriorOntologyLoading}
              disabled={!selectedLoop}
              onClick={onLoadOntology}
            >
              查询本体上下文
            </Button>
            <Tag color={priorOntology?.source ? 'blue' : 'default'}>{priorOntology?.server_name || priorOntology?.source || '未查询'}</Tag>
            <Tag color={ontologyContent ? 'green' : 'default'}>{ontologyContent ? ontologyContent.length + ' 字' : '无内容'}</Tag>
          </Space>
        </div>
        {tuningPriorOntologyLoading ? (
          <Alert className="agent-alert" type="info" showIcon message="正在查询本体上下文" description="本体查询可能需要数十秒，完成后会在下方展示返回原文。" />
        ) : tuningPriorOntologyError ? (
          <Alert className="agent-alert" type="error" showIcon message="本体查询失败" description={tuningPriorOntologyError} />
        ) : tuningPriorOntologyData ? (
          <div className="page-stack compact-stack">
            <Descriptions bordered column={2} size="small" className="industrial-descriptions">
              <Descriptions.Item label="查询问题" span={2}>{priorOntology?.query || '-'}</Descriptions.Item>
              <Descriptions.Item label="来源服务">{priorOntology?.server_name || priorOntology?.source || '-'}</Descriptions.Item>
              <Descriptions.Item label="调用工具">{priorOntology?.tool || '-'}</Descriptions.Item>
            </Descriptions>
            <Collapse
              defaultActiveKey={['ontology']}
              items={[{
                key: 'ontology',
                label: '本体返回原文',
                children: (
                  <Typography.Paragraph className="thinking-text">
                    {ontologyContent || '暂无本体返回内容。'}
                  </Typography.Paragraph>
                ),
              }]}
            />
          </div>
        ) : <Empty description="暂无本体结果，请点击“查询本体上下文”" />}
      </section>

      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">3 大模型整定先验解释</div>
            <Typography.Text type="secondary">基于前两步返回的核心上下文与本体原文，加上提示词，请大模型给出仅供参考的先验评审结果。</Typography.Text>
          </div>
          <Space wrap>
            <Tag color="blue">仅建议，不硬拦截</Tag>
            <Button
              size="small"
              type="primary"
              icon={<RobotOutlined />}
              loading={tuningPriorReviewLoading}
              disabled={!selectedLoop || !tuningPriorCoreData?.core_context}
              onClick={onLoadReview}
            >
              生成大模型先验评审
            </Button>
          </Space>
        </div>
        {tuningPriorReviewLoading ? (
          <Alert className="agent-alert" type="info" showIcon message="正在生成大模型先验评审" description="模型会综合核心指标与本体结果输出建议；该建议不作为整定硬约束。" />
        ) : tuningPriorReviewError ? (
          <Alert className="agent-alert" type="error" showIcon message="大模型先验评审失败" description={tuningPriorReviewError} />
        ) : tuningPriorReviewData ? (
          <Space direction="vertical" style={{ width: '100%' }}>
            {tuningPriorReviewData.error && (
              <Alert className="agent-alert" type="warning" showIcon message="评审未完成" description={tuningPriorReviewData.error} />
            )}
            {tuningPriorReviewData.review ? (
              <Typography.Paragraph className="thinking-text">
                {tuningPriorReviewData.review}
              </Typography.Paragraph>
            ) : (
              <Empty description="大模型未返回可展示的评审说明，请检查模型配置或稍后重试。" />
            )}
          </Space>
        ) : <Empty description="请先生成核心上下文，再点击“生成大模型先验评审”；本体上下文可选但建议先查询。" />}
      </section>
    </div>
  );
}
