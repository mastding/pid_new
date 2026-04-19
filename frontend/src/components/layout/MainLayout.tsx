import { ProLayout } from '@ant-design/pro-components';
import { DashboardOutlined, ExperimentOutlined, HistoryOutlined, ProfileOutlined } from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';

const menuRoutes = {
  routes: [
    { path: '/', name: 'PID 整定', icon: <ExperimentOutlined /> },
    { path: '/analysis', name: '回路分析', icon: <DashboardOutlined /> },
    { path: '/experience', name: '整定经验', icon: <HistoryOutlined /> },
    { path: '/sessions', name: '会话历史', icon: <ProfileOutlined /> },
  ],
};

export default function MainLayout({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <ProLayout
      title="PID 智能整定"
      logo={false}
      layout="mix"
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
