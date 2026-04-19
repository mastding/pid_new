import { BrowserRouter, Route, Routes } from 'react-router-dom';
import MainLayout from '@/components/layout/MainLayout';
import TuningPage from '@/pages/tuning/TuningPage';
import AnalysisPage from '@/pages/analysis/AnalysisPage';
import ExperiencePage from '@/pages/experience/ExperiencePage';
import SessionsPage from '@/pages/sessions/SessionsPage';

export default function App() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<TuningPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/experience" element={<ExperiencePage />} />
          <Route path="/sessions" element={<SessionsPage />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  );
}
