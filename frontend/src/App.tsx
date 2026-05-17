import { lazy, Suspense } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { Spin } from 'antd';
import MainLayout from '@/components/layout/MainLayout';

const TuningPage = lazy(() => import('@/pages/tuning/TuningPage'));
const AnalysisPage = lazy(() => import('@/pages/analysis/AnalysisPage'));
const ExperiencePage = lazy(() => import('@/pages/experience/ExperiencePage'));
const LoopMonitoringPage = lazy(() => import('@/pages/monitoring/LoopMonitoringPage'));
const SessionsPage = lazy(() => import('@/pages/sessions/SessionsPage'));
const ModelConfigPage = lazy(() => import('@/pages/settings/ModelConfigPage'));
const McpConfigPage = lazy(() => import('@/pages/settings/McpConfigPage'));

function PageFallback() {
  return (
    <div style={{ display: 'grid', minHeight: '100vh', placeItems: 'center' }}>
      <Spin tip="页面加载中..." />
    </div>
  );
}

function AppRoutes() {
  const location = useLocation();

  if (location.pathname.startsWith('/monitoring')) {
    return (
      <Routes>
        <Route path="/monitoring" element={<LoopMonitoringPage />} />
      </Routes>
    );
  }

  return (
    <MainLayout>
      <Routes>
        <Route path="/" element={<Navigate to="/monitoring" replace />} />
        <Route path="/tuning" element={<TuningPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
        <Route path="/experience" element={<ExperiencePage />} />
        <Route path="/sessions" element={<SessionsPage />} />
        <Route path="/settings" element={<ModelConfigPage />} />
        <Route path="/settings/mcp" element={<McpConfigPage />} />
        <Route path="*" element={<Navigate to="/monitoring" replace />} />
      </Routes>
    </MainLayout>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageFallback />}>
        <AppRoutes />
      </Suspense>
    </BrowserRouter>
  );
}
