import { Collapse, Typography } from 'antd';
import type { LlmThinkingEvent } from '@/types/tuning';
import { TUNING_STAGE_LABELS } from './model';

interface TuningTaskThinkingPanelProps {
  thinking: LlmThinkingEvent[];
}

export function TuningTaskThinkingPanel({ thinking }: TuningTaskThinkingPanelProps) {
  if (!thinking.length) return null;

  return (
    <section className="agent-panel">
      <div className="panel-title">大模型判断依据</div>
      <Collapse
        items={thinking.map((item, index) => ({
          key: `${item.stage}-${item.round ?? 'x'}-${index}`,
          label: `${TUNING_STAGE_LABELS[item.stage] ?? item.stage}${item.round !== undefined ? ` R${item.round}` : ''} · ${item.model} · ${item.reasoning_content.length} 字`,
          children: (
            <Typography.Paragraph className="thinking-text">
              {item.reasoning_content || item.raw_text}
            </Typography.Paragraph>
          ),
        }))}
      />
    </section>
  );
}
