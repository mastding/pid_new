import { Alert, Descriptions, Empty, Table, Tag, Typography } from 'antd';

import type { HistoryLoopAssessment, HistoryLoopMonitoringSnapshot } from '@/services/api';

type DiagnosisPlanKey = 'pid_diagnosis' | 'valve_diagnosis' | 'measurement_noise_diagnosis' | 'process_disturbance_diagnosis';

interface DiagnosisOverviewPanelProps {
  assessment: HistoryLoopAssessment | null;
}

interface OscillationDiagnosisPanelProps {
  assessment: HistoryLoopAssessment | null;
  monitoring?: HistoryLoopMonitoringSnapshot;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
}

interface DiagnosisPlanPanelProps {
  activeSub: DiagnosisPlanKey;
}

const diagnosisPlan: Record<DiagnosisPlanKey, { title: string; desc: string; rows: Array<{ item: string; evidence: string; action: string }> }> = {
  pid_diagnosis: {
    title: 'PID参数诊断',
    desc: '判断过激、过保守、积分过强/过弱、响应慢和参数导致的振荡风险。',
    rows: [
      { item: '过激/过保守', evidence: '超调、调节时间、MV动作量、PV衰减比', action: '调整Kp、Ti、Td或切换保守整定策略' },
      { item: '积分问题', evidence: '稳态误差、积分累积、低频偏差', action: '检查积分时间和抗积分饱和逻辑' },
      { item: '参数诱发振荡', evidence: 'PV/MV同频、相位关系、闭环衰减', action: '降低比例/积分作用并复核阀门状态' },
    ],
  },
  valve_diagnosis: {
    title: '阀门/执行机构',
    desc: '识别疑似死区、回差、卡滞、MV变化无PV响应和方向不对称。',
    rows: [
      { item: '死区', evidence: '小幅MV变化后PV低于噪声阈值', action: '检查阀门定位器、阀杆摩擦和执行机构气源' },
      { item: '回差', evidence: 'MV上升/下降路径对应PV响应不同', action: '做正反向小阶跃测试确认' },
      { item: '卡滞', evidence: 'MV突跳、PV滞后、动作呈锯齿', action: '优先检修执行机构，再考虑整定' },
    ],
  },
  measurement_noise_diagnosis: {
    title: '测量与噪声',
    desc: '识别传感器噪声、尖峰、漂移、平顶坏点和采样异常。',
    rows: [
      { item: '高频噪声', evidence: '高通残差、差分MAD、SNR', action: '检查仪表、滤波参数和采样配置' },
      { item: '尖峰/坏点', evidence: '离群点比例、单点突变、平顶时长', action: '清洗数据并排查采集链路' },
      { item: '测量漂移', evidence: '长期单向漂移且MV/SP无对应变化', action: '校验仪表零点和量程' },
    ],
  },
  process_disturbance_diagnosis: {
    title: '扰动与工艺',
    desc: '识别外扰、负荷变化、非线性、不同工况增益差异和过程约束。',
    rows: [
      { item: '外扰', evidence: 'PV变化领先MV动作，SP无变化', action: '追踪上游负荷或公用工程扰动' },
      { item: '非线性', evidence: '不同工作区间局部增益差异显著', action: '分工况建模或采用增益调度' },
      { item: '过程约束', evidence: 'MV饱和期间PV仍无法回归', action: '确认设备能力和工艺边界' },
    ],
  },
};

function BackendBadge({ implemented }: { implemented?: boolean }) {
  return implemented ? <Tag color="green">已接后端</Tag> : <Tag color="default">未开放</Tag>;
}

export function DiagnosisOverviewPanel({ assessment }: DiagnosisOverviewPanelProps) {
  const flags = assessment?.diagnostics.flags ?? [];

  return (
    <section className="agent-panel">
      <div className="panel-toolbar">
        <div className="panel-title">诊断总览</div>
        <Tag color={flags.length > 0 ? 'orange' : 'green'}>
          {flags.length > 0 ? `${flags.length} 项风险` : '无明显风险'}
        </Tag>
      </div>
      {flags.length ? (
        <Table
          size="small"
          pagination={false}
          rowKey={(row) => `${row.type}-${row.message}`}
          dataSource={flags}
          columns={[
            { title: '类型', dataIndex: 'type', width: 160 },
            { title: '级别', dataIndex: 'severity', width: 100, render: (value: string) => <Tag color={value === 'high' ? 'red' : 'orange'}>{value}</Tag> },
            { title: '诊断信息', dataIndex: 'message', ellipsis: true },
          ]}
        />
      ) : <Alert type="success" showIcon message="未发现明显数据质量或可辨识性风险。" />}
    </section>
  );
}

export function OscillationDiagnosisPanel({
  assessment,
  monitoring,
  formatNumber,
  formatPercentValue,
}: OscillationDiagnosisPanelProps) {
  return (
    <section className="agent-panel">
      <div className="panel-title">振荡监测</div>
      {assessment || monitoring ? (
        <Descriptions column={4} bordered size="small" className="industrial-descriptions">
          <Descriptions.Item label="是否振荡">{monitoring?.stability?.oscillation_detected ?? assessment?.diagnostics.oscillation?.detected ? '检测到' : '未检测到'}</Descriptions.Item>
          <Descriptions.Item label="严重度">{monitoring?.stability?.oscillation_severity ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="置信度">{formatPercentValue(monitoring?.stability?.oscillation_confidence, 1)}</Descriptions.Item>
          <Descriptions.Item label="主周期">{String(monitoring?.stability?.pv_dominant_period_s ?? assessment?.diagnostics.oscillation?.period_sec ?? '-')}s</Descriptions.Item>
          <Descriptions.Item label="主频能量">{formatPercentValue(monitoring?.stability?.pv_dominant_power_ratio, 1)}</Descriptions.Item>
          <Descriptions.Item label="零交叉">{formatNumber(monitoring?.stability?.pv_zero_crossing_per_hour, 2)}/h</Descriptions.Item>
          <Descriptions.Item label="相位提示">{monitoring?.stability?.phase_hint ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="SNR">{formatNumber(monitoring?.data_health?.pv_snr_db, 2)} dB</Descriptions.Item>
        </Descriptions>
      ) : <Empty description="暂无诊断结果" />}
    </section>
  );
}

export function DiagnosisPlanPanel({ activeSub }: DiagnosisPlanPanelProps) {
  const plan = diagnosisPlan[activeSub];

  return (
    <section className="agent-panel">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">{plan.title}</div>
          <Typography.Text type="secondary">{plan.desc}</Typography.Text>
        </div>
        <BackendBadge implemented={false} />
      </div>
      <Table
        size="small"
        pagination={false}
        rowKey="item"
        dataSource={plan.rows}
        columns={[
          { title: '诊断项', dataIndex: 'item', width: 160 },
          { title: '证据来源', dataIndex: 'evidence' },
          { title: '建议动作', dataIndex: 'action' },
        ]}
      />
    </section>
  );
}

export type { DiagnosisPlanKey };
