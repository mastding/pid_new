import { Collapse, Descriptions, Empty, Space, Tag, Typography } from 'antd';
import type { WindowSelectionMeta } from '@/types/tuning';

interface TuningTaskOntologyPanelProps {
  windowSelection: WindowSelectionMeta | null;
}

export function TuningTaskOntologyPanel({ windowSelection }: TuningTaskOntologyPanelProps) {
  return (
    <section className="agent-panel">
      <div className="panel-toolbar">
        <div>
          <div className="panel-title">本体查询与上下文</div>
          <Typography.Text type="secondary">
            后端向本体提出的问题、来源以及返回内容；窗口策略和大模型选窗都基于此。
          </Typography.Text>
        </div>
        {windowSelection ? (
          <Tag color={windowSelection.ontology_mcp_error ? 'red' : windowSelection.ontology_context_source === 'mcp' ? 'green' : 'default'}>
            {windowSelection.ontology_mcp_error ? '本体查询失败'
              : windowSelection.ontology_context_source === 'mcp'
                ? `本体上下文已注入 ${windowSelection.ontology_mcp_content_chars ?? '-'} 字`
                : '无本体上下文'}
          </Tag>
        ) : null}
      </div>
      {windowSelection ? (
        <Space direction="vertical" style={{ width: '100%' }}>
          <Descriptions bordered column={4} size="small" className="industrial-descriptions">
            <Descriptions.Item label="本体来源">{windowSelection.ontology_context_source ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="MCP 服务">{windowSelection.ontology_mcp_server ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="MCP 工具">{windowSelection.ontology_mcp_tool ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="返回字数">{windowSelection.ontology_mcp_content_chars ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="查询问题" span={4}>{windowSelection.ontology_mcp_query ?? '-'}</Descriptions.Item>
            {windowSelection.ontology_mcp_error ? (
              <Descriptions.Item label="失败原因" span={4}>
                <Typography.Text type="danger">{windowSelection.ontology_mcp_error}</Typography.Text>
              </Descriptions.Item>
            ) : null}
          </Descriptions>
          {windowSelection.ontology_mcp_content_raw || windowSelection.ontology_mcp_content_preview ? (
            <Collapse
              items={[{
                key: 'ontology-raw',
                label: '本体返回原文',
                children: (
                  <Typography.Paragraph className="thinking-text">
                    {windowSelection.ontology_mcp_content_raw || windowSelection.ontology_mcp_content_preview}
                  </Typography.Paragraph>
                ),
              }]}
            />
          ) : null}
        </Space>
      ) : (
        <Empty description="等待本体检索结果" />
      )}
    </section>
  );
}
