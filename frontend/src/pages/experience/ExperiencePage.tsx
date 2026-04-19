import { PageContainer, ProCard } from '@ant-design/pro-components';
import { Empty } from 'antd';

export default function ExperiencePage() {
  return (
    <PageContainer title="整定经验" subTitle="历史整定记录与相似回路检索">
      <ProCard>
        <Empty
          description={
            <span style={{ color: '#888' }}>
              经验库功能开发中
              <br />
              <small>后端经验存储模块（backend/memory/）尚未实现</small>
            </span>
          }
          style={{ padding: '60px 0' }}
        />
      </ProCard>
    </PageContainer>
  );
}
