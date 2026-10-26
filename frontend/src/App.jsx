import { HashRouter, Navigate, Route, Routes } from "react-router-dom";
import Home from "./pages/Home.jsx";
import { NewJob, CurrentJob, AllJobs } from "./pages/Researcher.jsx";
import { VolunteerStart, VolunteerSafety, VolunteerHistory } from "./pages/Volunteer.jsx";
import { AdminLogin, AdminDashboard } from "./pages/Admin.jsx";

// HashRouter keeps page addresses after a "#" (e.g. /#/admin), so they can
// never collide with API paths like /jobs or /admin/status on the server.
export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/researcher" element={<Navigate to="/researcher/new" replace />} />
        <Route path="/researcher/new" element={<NewJob />} />
        <Route path="/researcher/current" element={<CurrentJob />} />
        <Route path="/researcher/jobs" element={<AllJobs />} />
        <Route path="/volunteer" element={<VolunteerStart />} />
        <Route path="/volunteer/safety" element={<VolunteerSafety />} />
        <Route path="/volunteer/history" element={<VolunteerHistory />} />
        <Route path="/admin/login" element={<AdminLogin />} />
        <Route path="/admin" element={<AdminDashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HashRouter>
  );
}
