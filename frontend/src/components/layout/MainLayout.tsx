import { ProLayout } from '@ant-design/pro-components';
import {
  ApiOutlined,
  DashboardOutlined,
  ExperimentOutlined,
  HistoryOutlined,
  ProfileOutlined,
  RadarChartOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { useLocation, useNavigate } from 'react-router-dom';

const menuRoutes = {
  routes: [
    { path: '/', name: 'PID 整定', icon: <ExperimentOutlined /> },
    { path: '/analysis', name: '回路分析', icon: <DashboardOutlined /> },
    { path: '/monitoring', name: '智能监控驾驶舱', icon: <RadarChartOutlined /> },
    { path: '/experience', name: '整定经验', icon: <HistoryOutlined /> },
    { path: '/sessions', name: '会话历史', icon: <ProfileOutlined /> },
    {
      path: '/settings',
      name: '系统配置',
      icon: <SettingOutlined />,
      routes: [
        { path: '/settings', name: 'LLM 模型配置', icon: <SettingOutlined /> },
        { path: '/settings/mcp', name: 'MCP 服务配置', icon: <ApiOutlined /> },
      ],
    },
  ],
};

export default function MainLayout({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <ProLayout
      title="PID 智能整定"
      logo={false}
      layout="side"
      fixSiderbar
      route={menuRoutes}
      location={{ pathname: location.pathname }}
      menuItemRender={(item, dom) => (
        <div onClick={() => item.path && navigate(item.path)}>{dom}</div>
      )}
    >
      {children}
    </ProLayout>
  );
}
