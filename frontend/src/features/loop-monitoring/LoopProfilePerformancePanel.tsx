import { Descriptions, Empty, Table, Typography } from 'antd';

import type { HistoryLoopFeatures } from '@/services/api';

interface LoopProfilePerformancePanelProps {
  loopFeatures: HistoryLoopFeatures | null;
  formatNumber: (value?: number | null, digits?: number) => string;
  formatPercentValue: (value?: number | null, digits?: number) => string;
  formatHarrisBasis: (value?: string) => string;
  formatCpkBasis: (value?: string) => string;
}

const formulaRows = [
  {
    name: 'Harris指数',
    formula: 'HI = σ²_min / σ²_actual',
    explain:
      'σ²_min 使用 PV 高频残差方差近似最小理论方差；σ²_actual 使用 PV-SV 跟踪误差方差，若无有效SV则使用去趋势PV方差。HI越接近1表示越接近最小方差控制，越接近0表示波动越大、改善空间越大。',
  },
  {
    name: 'Harris劣化指数',
    formula: 'DI = 1 - HI',
    explain: '为了便于报警和排序额外展示的辅助指标，越接近1表示偏离最小方差基准越多；它不是标准 Harris Index 本体。',
  },
  {
    name: 'Cpk过程能力',
    formula: 'Cpk = min((USL-μ)/(3σ), (μ-LSL)/(3σ))',
    explain: '必须有PV规格上限USL和下限LSL；当前历史导入若没有规格限字段，系统不伪造Cpk，只显示未计算。',
  },
  {
    name: '震荡指数/周期',
    formula: '指数由主频能量占比、PV零交叉频次综合得到；周期 = 1 / 主频',
    explain: '先用约30分钟滚动中位数去趋势，再做FFT找PV主频；主频能量和零交叉都足够时才认为有显著震荡。',
  },
];

export function LoopProfilePerformancePanel({
  loopFeatures,
  formatNumber,
  formatPercentValue,
  formatHarrisBasis,
  formatCpkBasis,
}: LoopProfilePerformancePanelProps) {
  return (
    <section className="agent-panel compact-facts">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">控制性能指标</div>
          <Typography.Text type="secondary">
            基于历史 PV/MV/SV 计算 Harris 指数、Cpk 过程能力和震荡指数；Cpk 只有配置 PV 规格上下限后才给出标准值。
          </Typography.Text>
        </div>
      </div>
      {loopFeatures ? (
        <>
          <Descriptions bordered size="small" column={4} className="industrial-descriptions">
            <Descriptions.Item label="Harris指数(1优0差)">
              {formatNumber(loopFeatures.performance_raw?.harris_index, 4)}
            </Descriptions.Item>
            <Descriptions.Item label="Harris劣化指数">
              {formatNumber(loopFeatures.performance_raw?.harris_degradation_index, 4)}
            </Descriptions.Item>
            <Descriptions.Item label="误差口径">
              {formatHarrisBasis(loopFeatures.performance_raw?.harris_error_basis)}
            </Descriptions.Item>
            <Descriptions.Item label="实际误差方差">
              {formatNumber(loopFeatures.performance_raw?.harris_actual_variance, 6)}
            </Descriptions.Item>
            <Descriptions.Item label="Cpk">
              {loopFeatures.performance_raw?.cpk === null || loopFeatures.performance_raw?.cpk === undefined
                ? '未计算'
                : formatNumber(loopFeatures.performance_raw.cpk, 4)}
            </Descriptions.Item>
            <Descriptions.Item label="Cpk依据">
              {formatCpkBasis(loopFeatures.performance_raw?.cpk_basis)}
            </Descriptions.Item>
            <Descriptions.Item label="规格下限/上限">
              {loopFeatures.performance_raw?.cpk_lsl === null || loopFeatures.performance_raw?.cpk_usl === null
                ? '-'
                : `${formatNumber(loopFeatures.performance_raw?.cpk_lsl, 3)} / ${formatNumber(loopFeatures.performance_raw?.cpk_usl, 3)}`}
            </Descriptions.Item>
            <Descriptions.Item label="震荡指数">
              {formatPercentValue(loopFeatures.performance_raw?.oscillation_index ?? loopFeatures.oscillation_raw?.confidence, 1)}
            </Descriptions.Item>
            <Descriptions.Item label="震荡周期">
              {formatNumber(loopFeatures.performance_raw?.oscillation_period_s ?? loopFeatures.oscillation_raw?.pv_dominant_period_s, 1)}s
            </Descriptions.Item>
            <Descriptions.Item label="主频能量">
              {formatPercentValue(loopFeatures.performance_raw?.oscillation_power_ratio ?? loopFeatures.oscillation_raw?.pv_dominant_power_ratio, 1)}
            </Descriptions.Item>
            <Descriptions.Item label="零交叉频次">
              {formatNumber(loopFeatures.performance_raw?.oscillation_zero_crossing_per_hour ?? loopFeatures.oscillation_raw?.pv_zero_crossing_per_hour, 2)}/h
            </Descriptions.Item>
          </Descriptions>
          <Table
            className="formula-table"
            size="small"
            pagination={false}
            rowKey="name"
            dataSource={formulaRows}
            columns={[
              { title: '指标', dataIndex: 'name', width: 180 },
              { title: '公式', dataIndex: 'formula', width: 300 },
              { title: '说明', dataIndex: 'explain' },
            ]}
          />
        </>
      ) : <Empty description="暂无控制性能指标" />}
    </section>
  );
}
