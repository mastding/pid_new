import type { PromptConfig } from '@/services/api';

export type PromptConfigField = Exclude<keyof PromptConfig, 'updated_at'>;

export const PROMPT_CONFIG_ITEMS: Array<{
  key: PromptConfigField;
  label: string;
  group: string;
  help: string;
  placeholder: string;
  minRows: number;
  maxRows: number;
}> = [
  {
    key: 'assistant_system_prompt',
    label: '智能助手系统提示词',
    group: '智能助手',
    help: '定义智能助手身份、任务范围、回答风格和证据引用要求。',
    placeholder: '定义智能助手身份、任务范围、回答风格和证据引用要求',
    minRows: 10,
    maxRows: 20,
  },
  {
    key: 'assistant_developer_prompt',
    label: '智能助手安全与流程约束提示词',
    group: '智能助手',
    help: '定义禁止直接整定、禁止修改参数、高成本流程需确认等安全边界。',
    placeholder: '定义禁止直接整定、禁止修改参数、高成本流程需确认等边界',
    minRows: 8,
    maxRows: 16,
  },
  {
    key: 'assistant_response_schema',
    label: '智能助手响应格式说明',
    group: '智能助手',
    help: '定义答案、证据、风险级别和建议动作等结构化字段。',
    placeholder: '定义答案、证据、风险级别和建议动作等结构化字段',
    minRows: 8,
    maxRows: 16,
  },
  {
    key: 'window_policy_system_prompt',
    label: '窗口候选策略提示词',
    group: '窗口候选',
    help: '用于整定中心的窗口候选策略生成，指导模型根据画像、本体上下文输出窗口算法策略。',
    placeholder: '定义窗口策略生成器身份、算法族约束、输出字段和安全边界',
    minRows: 12,
    maxRows: 24,
  },
  {
    key: 'window_policy_user_prompt_template',
    label: '窗口候选用户提示词模板',
    group: '窗口候选',
    help: '运行时会替换 $base_policy_json、$profile_text、$pv_json、$mv_json、$raw_profile_json、$mcp_content、$frontend_text。',
    placeholder: '定义选窗模型接收实时画像和本体上下文的用户提示词模板',
    minRows: 10,
    maxRows: 20,
  },
  {
    key: 'identification_review_system_prompt',
    label: '辨识 / 模型评审提示词',
    group: '辨识评审',
    help: '用于辨识结束后的模型可信度评审，约束模型输出结论、理由和风险点。',
    placeholder: '定义模型评审专家身份、关键判据和结构化输出要求',
    minRows: 12,
    maxRows: 24,
  },
  {
    key: 'identification_review_user_prompt_template',
    label: '辨识 / 模型评审用户提示词模板',
    group: '辨识评审',
    help: '运行时会替换回路类型、数据画像、窗口来源、模型类型和辨识记录等变量。',
    placeholder: '定义模型评审接收辨识结果、窗口和尝试记录的用户提示词模板',
    minRows: 10,
    maxRows: 20,
  },
];
