import { Descriptions, Empty, Select, Space, Table, Tag, Typography } from 'antd';

import type { HistoryLoop, HistoryLoopFeatures } from '@/services/api';

interface ActuatorStatusPanelProps {
  selectedLoopId?: string;
  scopedLoops: HistoryLoop[];
  loopFeatures: HistoryLoopFeatures | null;
  onLoopChange: (loopId: string) => void;
  loopTypeLabel: (loop: HistoryLoop) => string;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  yesNo: (value?: boolean | null) => string;
}

const actuatorCriteriaRows = [
  {
    item: '死区',
    formula: '统计有效阀位变化后，过程变量响应仍接近噪声范围的比例。',
    backend: '死区滞后窗比例',
    note: '按回路类型给PV响应观察窗：流量10s、压力60s、温度300s、液位600s；它是历史数据迹象，不等同于阀门离线死区试验。',
  },
  {
    item: '回差',
    formula: '分别比较正向与反向阀位动作下的过程响应差异。',
    backend: '正反向响应增益差异',
    note: '需要正反向动作样本都足够；样本不足时显示未判定。',
  },
  {
    item: '粘滞',
    formula: '按最长不动作时长与采样周期综合估计粘滞程度。',
    backend: '最长不动作时长、粘滞指数',
    note: '长时间不动作但历史上存在动作时，提示疑似粘滞或控制器输出保持。',
  },
  {
    item: '卡涩',
    formula: '当长时间不动作超过回路采样尺度时标记为卡涩迹象。',
    backend: '卡涩迹象',
    note: '卡涩需要结合阀门反馈、定位器和现场动作试验最终确认；当前只是历史趋势迹象。',
  },
];

const actuatorImpactRows = [
  { item: '死区/黏滞', effect: '会让小幅 PID 输出无效，辨识 K/T 容易偏差', action: '先做执行机构检查或剔除死区片段' },
  { item: '饱和/贴边', effect: '会截断过程响应，整定后仿真偏乐观或偏悲观', action: '优先确认工况和阀门能力' },
  { item: '分辨率/速率限制', effect: '限制闭环可达到的响应速度', action: '整定时提高保守度并限制 Kp' },
];

export function ActuatorStatusPanel({
  selectedLoopId,
  scopedLoops,
  loopFeatures,
  onLoopChange,
  loopTypeLabel,
  formatNumber,
  formatPercentValue,
  yesNo,
}: ActuatorStatusPanelProps) {
  const actuatorProfile = loopFeatures?.actuator_profile;
  const needsAttention = (actuatorProfile?.mv_saturation_ratio ?? 0) > 0.05;

  return (
    <div className="page-stack">
      <section className="agent-panel">
        <div className="panel-toolbar">
          <div>
            <div className="panel-title">执行机构状态</div>
            <Typography.Text type="secondary">
              基于历史 MV 动作、分辨率、贴边、疑似死区和长时间不动片段，判断整定前是否需要先处理阀门/执行机构问题。
            </Typography.Text>
          </div>
          <Space wrap>
            <Select
              showSearch
              style={{ minWidth: 320 }}
              placeholder="选择回路"
              value={selectedLoopId}
              onChange={onLoopChange}
              optionFilterProp="label"
              options={scopedLoops.map((loop) => ({
                value: loop.loop_id,
                label: `${loop.loop_id} · ${loopTypeLabel(loop)}`,
              }))}
            />
            <Tag color={needsAttention ? 'orange' : 'green'}>
              {needsAttention ? '需关注' : '正常'}
            </Tag>
          </Space>
        </div>
        {loopFeatures ? (
          <Descriptions bordered column={4} size="small" className="industrial-descriptions">
            <Descriptions.Item label="MV分辨率">{formatNumber(actuatorProfile?.mv_resolution_hint, 5)}</Descriptions.Item>
            <Descriptions.Item label="死区迹象(滞后窗)">{formatPercentValue(actuatorProfile?.mv_deadband_lagged_ratio, 1)}</Descriptions.Item>
            <Descriptions.Item label="死区事件">{actuatorProfile?.mv_deadband_event_count ?? 0}/{actuatorProfile?.mv_deadband_events_total ?? 0}</Descriptions.Item>
            <Descriptions.Item label="估计死区宽度">{formatNumber(actuatorProfile?.mv_deadband_estimated_width, 4)}</Descriptions.Item>
            <Descriptions.Item label="死区观察窗">{formatNumber(actuatorProfile?.mv_deadband_lag_used_s, 1)}s</Descriptions.Item>
            <Descriptions.Item label="回差迹象">{yesNo(actuatorProfile?.mv_hysteresis_hint)}</Descriptions.Item>
            <Descriptions.Item label="回差指数">{formatPercentValue(actuatorProfile?.mv_hysteresis_ratio, 1)}</Descriptions.Item>
            <Descriptions.Item label="正/反向增益">{formatNumber(actuatorProfile?.mv_hysteresis_up_gain, 3)} / {formatNumber(actuatorProfile?.mv_hysteresis_down_gain, 3)}</Descriptions.Item>
            <Descriptions.Item label="黏滞迹象">{yesNo(actuatorProfile?.mv_stiction_hint)}</Descriptions.Item>
            <Descriptions.Item label="黏滞指数">{formatPercentValue(actuatorProfile?.mv_stiction_score, 1)}</Descriptions.Item>
            <Descriptions.Item label="卡涩迹象">{yesNo(actuatorProfile?.mv_stuck_hint)}</Descriptions.Item>
            <Descriptions.Item label="最长不动作">{formatNumber(actuatorProfile?.longest_mv_stuck_duration_s, 1)}s</Descriptions.Item>
            <Descriptions.Item label="速率限制迹象">{yesNo(actuatorProfile?.mv_rate_limit_hint)}</Descriptions.Item>
            <Descriptions.Item label="低限余量">{formatNumber(actuatorProfile?.mv_saturation_margin_low, 3)}</Descriptions.Item>
            <Descriptions.Item label="高限余量">{formatNumber(actuatorProfile?.mv_saturation_margin_high, 3)}</Descriptions.Item>
            <Descriptions.Item label="MV饱和比例">{formatPercentValue(actuatorProfile?.mv_saturation_ratio, 2)}</Descriptions.Item>
          </Descriptions>
        ) : <Empty description="暂无执行机构特征" />}
      </section>
      <section className="agent-panel">
        <div className="panel-title">后端判据说明</div>
        <Table
          className="formula-table"
          size="small"
          pagination={false}
          rowKey="item"
          dataSource={actuatorCriteriaRows}
          columns={[
            { title: '项目', dataIndex: 'item', width: 120 },
            { title: '判据', dataIndex: 'formula', width: 360 },
            { title: '后端指标', dataIndex: 'backend', width: 240 },
            { title: '说明', dataIndex: 'note' },
          ]}
        />
      </section>
      <section className="agent-panel">
        <div className="panel-title">整定影响说明</div>
        <Table
          size="small"
          pagination={false}
          rowKey="item"
          dataSource={actuatorImpactRows}
          columns={[
            { title: '问题', dataIndex: 'item', width: 160 },
            { title: '对整定影响', dataIndex: 'effect' },
            { title: '建议动作', dataIndex: 'action' },
          ]}
        />
      </section>
    </div>
  );
}
