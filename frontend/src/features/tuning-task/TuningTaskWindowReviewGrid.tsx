import { Alert, Descriptions, Empty, List, Space } from 'antd';
import type {
  IdentificationRefinementMeta,
  ModelReviewMeta,
  WindowSelectionMeta,
} from '@/types/tuning';

interface TuningTaskWindowReviewGridProps {
  windowSelection: WindowSelectionMeta | null;
  modelReview: ModelReviewMeta | null;
  refinements: IdentificationRefinementMeta[];
  formatNumber: (value?: number | null, digits?: number) => string;
}

function formatRefinement(item: IdentificationRefinementMeta) {
  return [
    `R${item.round}：${item.retry ? '继续重试' : '放弃重试'}`,
    item.source ? `来源 ${item.source === 'deterministic_algorithm_policy' ? '确定性算法族策略' : item.source}` : '',
    item.rationale,
    item.recommended_algorithm_label || item.recommended_algorithm
      ? `推荐算法族 ${item.recommended_algorithm_label || item.recommended_algorithm}`
      : '',
    item.recommended_window_source ? `推荐窗口 ${item.recommended_window_source}` : '',
    item.force_model_types?.length ? `模型池 ${item.force_model_types.join(', ')}` : '',
    item.force_window_index !== undefined && item.force_window_index !== null ? `窗口 #${item.force_window_index}` : '',
  ].filter(Boolean).join('；');
}

export function TuningTaskWindowReviewGrid({
  windowSelection,
  modelReview,
  refinements,
  formatNumber,
}: TuningTaskWindowReviewGridProps) {
  return (
    <div className="panel-grid">
      <section className="agent-panel">
        <div className="panel-title">窗口选择</div>
        {windowSelection ? (
          <Descriptions column={1} size="small">
            <Descriptions.Item label="选择模式">{windowSelection.mode}</Descriptions.Item>
            <Descriptions.Item label="最终窗口">#{windowSelection.chosen_index}</Descriptions.Item>
            <Descriptions.Item label="算法窗口">#{windowSelection.deterministic_index}</Descriptions.Item>
            <Descriptions.Item label="算法分数">{formatNumber(windowSelection.deterministic_score, 3)}</Descriptions.Item>
            <Descriptions.Item label="选择理由">{windowSelection.reasoning || '-'}</Descriptions.Item>
          </Descriptions>
        ) : (
          <Empty description="等待窗口选择结果" />
        )}
      </section>

      <section className="agent-panel">
        <div className="panel-title">大模型评审与精修</div>
        {modelReview ? (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Alert
              type={modelReview.verdict === 'accept' ? 'success' : 'warning'}
              showIcon
              message={`评审结论：${modelReview.verdict}`}
              description={modelReview.reason}
            />
            {!!modelReview.concerns?.length && (
              <List
                size="small"
                header="具体担忧"
                bordered
                dataSource={modelReview.concerns}
                renderItem={(item) => <List.Item>{item}</List.Item>}
              />
            )}
            {!!refinements.length && (
              <List
                size="small"
                header="精修建议"
                bordered
                dataSource={refinements}
                renderItem={(item) => <List.Item>{formatRefinement(item)}</List.Item>}
              />
            )}
          </Space>
        ) : (
          <Empty description="等待大模型评审" />
        )}
      </section>
    </div>
  );
}
