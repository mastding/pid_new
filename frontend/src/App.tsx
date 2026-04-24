import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import MainLayout from '@/components/layout/MainLayout';
import TuningPage from '@/pages/tuning/TuningPage';
import AnalysisPage from '@/pages/analysis/AnalysisPage';
import ExperiencePage from '@/pages/experience/ExperiencePage';
import LoopMonitoringPage from '@/pages/monitoring/LoopMonitoringPage';
import SessionsPage from '@/pages/sessions/SessionsPage';

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
        <Route path="/" element={<TuningPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
        <Route path="/experience" element={<ExperiencePage />} />
        <Route path="/sessions" element={<SessionsPage />} />
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
