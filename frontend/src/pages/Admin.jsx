import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { api, ApiError, getKey, setKey, clearKey, usePolling, fmtPct, fmtDuration, SAMPLE_ADMIN } from "../api.js";
import { Shell, PageHead, PreviewPill, LogoMark, Icon } from "../components/ui.jsx";

export function AdminLogin() {
  const navigate = useNavigate();
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function signIn(e) {
    e.preventDefault();
    if (!value.trim()) { setError("Enter the admin key."); return; }
    setBusy(true); setError("");
    setKey("admin", value.trim());
    try {
      await api.adminStatus();
      navigate("/admin");
    } catch (err) {
      if (err instanceof ApiError && err.notBuilt) navigate("/admin"); // admin endpoints not built yet: preview
      else {
        clearKey("admin");
        setError(err instanceof ApiError && err.badKey ? "That key doesn't match the server's admin key." : err.message);
      }
    } finally { setBusy(false); }
  }

  return (
    <div style={{ minHeight: "100vh", padding: 16, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ width: "100%", maxWidth: 820, display: "flex", flexWrap: "wrap", borderRadius: 32, overflow: "hidden" }}>
        <div style={{ flex: "1 1 320px", minHeight: 360, background: "var(--side)", color: "#fff", padding: 40, display: "flex", flexDirection: "column", justifyContent: "space-between", gap: 24 }}>
          <Link to="/" className="logo" style={{ fontSize: 36, color: "#fff" }}>
            <span className="logo-tile" style={{ width: 41, height: 41, borderRadius: 12 }}><LogoMark size={26} /></span>lattice
          </Link>
          <div>
            <span className="pill lilac">Admin</span>
            <p style={{ marginTop: 14, fontSize: 18, lineHeight: 1.5, color: "#c9ccd3", maxWidth: 280 }}>
              Server health, every node, and every event in one place.
            </p>
          </div>
        </div>
        <form onSubmit={signIn} style={{ flex: "1 1 320px", background: "var(--surface)", padding: 40, display: "flex", flexDirection: "column", justifyContent: "center", gap: 18 }}>
          <div>
            <h1 style={{ fontSize: 34, fontWeight: 600, letterSpacing: "-0.02em" }}>Sign in</h1>
            <p className="muted" style={{ marginTop: 8, fontSize: 15 }}>Visible to administrators only.</p>
          </div>
          <label className="field" style={{ fontSize: 14 }}>
            Admin key
            <input type="password" autoComplete="current-password" value={value} onChange={(e) => setValue(e.target.value)} />
          </label>
          {error && <div className="error" role="alert">{error}</div>}
          <button className="btn block" type="submit" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
          <p className="muted small">Later: sign in with GitHub, limited to listed admin accounts.</p>
        </form>
      </div>
    </div>
  );
}

async function fetchAdmin() {
  const [status, nodes, events] = await Promise.all([api.adminStatus(), api.adminNodes(), api.adminEvents()]);
  return { status, nodes, events };
}

const NODE_STATUS = {
  training: { cls: "dark", label: "Training", row: "sky" },
  available: { cls: "outline", label: "Available", row: "" },
  standby: { cls: "outline", label: "Standby", row: "" },
  dropped: { cls: "warn", label: "Dropped", row: "" },
};

export function AdminDashboard() {
  const navigate = useNavigate();
  const signedIn = Boolean(getKey("admin"));
  const { data, preview, error, loading } = usePolling(fetchAdmin, 2000, SAMPLE_ADMIN);

  if (!signedIn) return <Navigate to="/admin/login" replace />;
  if (error instanceof ApiError && error.badKey) { clearKey("admin"); return <Navigate to="/admin/login" replace />; }

  const signOut = () => { clearKey("admin"); navigate("/admin/login"); };
  const s = data?.status || {};
  const c = s.counts || {};
  const st = s.settings || {};
  const job = s.current_job;

  return (
    <Shell area="admin" onSignOut={signOut}>
      <PageHead
        title="Server overview"
        sub={`Online for ${fmtDuration(s.uptime_seconds)} · version ${s.version || "–"} · refreshes every 2 s`}
        right={<div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>{preview && <PreviewPill />}<span className="pill dashed">Scheduler: {st.scheduler || "–"}</span></div>}
      />
      {loading && <p className="muted">Loading…</p>}
      {error && <div className="error">{error.message}</div>}

      {data && (
        <>
          <div className="grid-cards">
            <div className="stat lime"><span className="stat-title">Server:</span><span className="stat-big">Online</span><span className="sub small">up {fmtDuration(s.uptime_seconds)}</span></div>
            <div className="stat lilac"><span className="stat-title">Nodes online:</span><span className="stat-big">{c.online ?? 0} / {c.total ?? 0}</span><span className="sub small">{c.standby ?? 0} standby · {c.dropped ?? 0} dropped</span></div>
            <div className="stat mint"><span className="stat-title">Training now:</span><span className="stat-big">{c.training ?? 0}</span><span className="sub small">{job ? `in round ${job.current_round}` : "no job running"}</span></div>
            <div className="stat sky"><span className="stat-title">Pooled CPUs:</span><span className="stat-big">{c.cpus ?? 0}</span><span className="sub small">across online nodes</span></div>
          </div>

          <div className="row-wrap">
            <section className="list" style={{ flex: "999 1 460px", minWidth: 0 }}>
              <div className="page-head" style={{ alignItems: "center", padding: "0 4px" }}>
                <h2 className="section-title">Nodes</h2>
                <span className="muted" style={{ fontSize: 14 }}>Dropped after {st.heartbeat_timeout ?? 20} s without a heartbeat</span>
              </div>
              {data.nodes.map((n) => {
                const ns = NODE_STATUS[n.status] || NODE_STATUS.standby;
                return (
                  <div key={n.node_id} className={`list-row ${ns.row}`}>
                    <span className={`list-icon${n.status === "dropped" ? " dashed" : ""}`}>{n.status === "dropped" ? Icon.cross : Icon.laptop}</span>
                    <div className="list-body">
                      <p className="list-title">{n.node_id}</p>
                      <p className="list-meta">{n.device} · {n.cpu_count} cores · reliability {Math.round((n.reliability ?? 1) * 100)}%</p>
                    </div>
                    <span className={`pill ${ns.cls}`}>{ns.label}</span>
                    <span className="pill light">{Math.round(n.seconds_since_seen)}s ago</span>
                  </div>
                );
              })}
              {!data.nodes.length && <p className="muted" style={{ padding: "0 4px" }}>No nodes have registered yet.</p>}
            </section>

            <section style={{ flex: "1 1 300px", display: "flex", flexDirection: "column", gap: 16 }}>
              <div className="card" style={{ background: "var(--side)", color: "#fff", gap: 10, borderRadius: 26 }}>
                <div className="page-head" style={{ alignItems: "center" }}>
                  <h2 style={{ fontSize: 18, fontWeight: 600 }}>Current job</h2>
                  {job && <span className="pill lilac">Training</span>}
                </div>
                {job ? (
                  <>
                    <p style={{ fontSize: 20, fontWeight: 600 }}>{job.name}</p>
                    <p style={{ fontSize: 14, color: "#c9ccd3" }}>Round {job.current_round} of {job.max_rounds} · {fmtPct(job.accuracy)} test accuracy</p>
                  </>
                ) : <p style={{ fontSize: 14, color: "#c9ccd3" }}>No job is running.</p>}
              </div>
              <div className="card" style={{ gap: 10, borderRadius: 26 }}>
                <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 4 }}>Settings in effect</h2>
                {[["Heartbeat timeout", `${st.heartbeat_timeout ?? "–"} s`], ["Round deadline", `${st.round_deadline ?? "–"} s`], ["Min nodes per round", st.min_nodes ?? "–"]].map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 14 }}><span className="muted">{k}</span><span style={{ fontWeight: 500 }}>{v}</span></div>
                ))}
              </div>
            </section>
          </div>

          <section className="card" style={{ borderRadius: 26 }}>
            <h2 className="section-title">Recent events</h2>
            <div className="list">
              {data.events.map((ev, i) => (
                <div key={i} style={{ display: "flex", gap: 16, alignItems: "center" }}>
                  <span className="muted small" style={{ minWidth: 76 }}>{ev.time}</span>
                  <span className={ev.kind === "dropped" ? "pill warn" : ev.kind === "round" ? "pill mint" : "pill"} style={{ flex: "1 1 auto", whiteSpace: "normal", padding: "12px 16px", borderRadius: 16, fontSize: 14, background: ev.kind === "info" ? "var(--row)" : undefined }}>
                    {ev.text}
                  </span>
                </div>
              ))}
              {!data.events.length && <p className="muted">Nothing has happened yet.</p>}
            </div>
          </section>
        </>
      )}
    </Shell>
  );
}
