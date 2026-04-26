import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import MainLayout from '@/components/layout/MainLayout';
import TuningPage from '@/pages/tuning/TuningPage';
import AnalysisPage from '@/pages/analysis/AnalysisPage';
import ExperiencePage from '@/pages/experience/ExperiencePage';
import LoopMonitoringPage from '@/pages/monitoring/LoopMonitoringPage';
import SessionsPage from '@/pages/sessions/SessionsPage';
import ModelConfigPage from '@/pages/settings/ModelConfigPage';

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
        <Route path="*" element={<Navigate to="/monitoring" replace />} />
      </Routes>
    </MainLayout>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
