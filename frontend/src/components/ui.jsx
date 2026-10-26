import { useState } from "react";
import { Link, NavLink } from "react-router-dom";
import { getKey, setKey, pct } from "../api.js";

/* ---------- Logo: a star topology, one server with six volunteers ---------- */
const OUTER = [0, 1, 2, 3, 4, 5].map((k) => {
  const a = ((-90 + 60 * k) * Math.PI) / 180;
  return [12 + 8.5 * Math.cos(a), 12 + 8.5 * Math.sin(a)];
});

export function LogoMark({ size = 26, color = "#16181D" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <g stroke={color} strokeWidth="1.6" strokeLinecap="round">
        {OUTER.map(([x, y], i) => <line key={i} x1="12" y1="12" x2={x} y2={y} />)}
      </g>
      <g fill={color}>
        {OUTER.map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2.1" />)}
        <circle cx="12" cy="12" r="3.6" />
      </g>
    </svg>
  );
}

export function Logo({ size = 32, onDark = false, to = "/" }) {
  const mark = onDark ? (
    <span className="logo-tile" style={{ width: size * 1.15, height: size * 1.15, borderRadius: size * 0.35 }}>
      <LogoMark size={size * 0.72} />
    </span>
  ) : (
    <LogoMark size={size * 0.8} />
  );
  return (
    <Link to={to} className="logo" style={{ fontSize: size }} aria-label="Lattice home">
      {mark}lattice
    </Link>
  );
}

/* ---------- Icons ---------- */
const P = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" };
export const Icon = {
  plus: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path {...P} d="M12 5v14M5 12h14" /></svg>,
  bars: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path {...P} d="M4 19V9M10 19V5M16 19v-7M22 19H2" /></svg>,
  list: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path {...P} d="M8 6h13M8 12h13M8 18h13M4 6h.01M4 12h.01M4 18h.01" /></svg>,
  laptop: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><g {...P}><rect x="4" y="5" width="16" height="11" rx="1.5" /><path d="M2 19h20" /></g></svg>,
  flask: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path {...P} d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3" /></svg>,
  download: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path {...P} d="M12 4v11M7 10l5 5 5-5M4 20h16" /></svg>,
  shield: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path {...P} d="M12 3 4 6v6c0 4.5 3.4 8 8 9 4.6-1 8-4.5 8-9V6z" /></svg>,
  clock: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><g {...P}><circle cx="12" cy="12" r="8" /><path d="M12 8v4l3 2" /></g></svg>,
  grid: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><g {...P}><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></g></svg>,
  logout: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path {...P} d="M10 4H5v16h5M14 8l4 4-4 4M18 12H9" /></svg>,
  check: <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path {...P} strokeWidth="2" d="m5 12 5 5 9-10" /></svg>,
  cross: <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path {...P} strokeWidth="2" d="M8 8l8 8M16 8l-8 8" /></svg>,
};

/* ---------- App shell with sidebar ---------- */
const NAV = {
  research: {
    label: "Research",
    items: [
      { to: "/researcher/new", label: "New job", icon: Icon.plus },
      { to: "/researcher/current", label: "Current job", icon: Icon.bars },
      { to: "/researcher/jobs", label: "All jobs", icon: Icon.list },
    ],
    other: { to: "/volunteer", label: "Volunteer", icon: Icon.laptop },
  },
  volunteer: {
    label: "Volunteer",
    items: [
      { to: "/volunteer", label: "Get started", icon: Icon.download, end: true },
      { to: "/volunteer/safety", label: "Safety & stopping", icon: Icon.shield },
      { to: "/volunteer/history", label: "Your history", icon: Icon.clock },
    ],
    other: { to: "/researcher/new", label: "Researcher", icon: Icon.flask },
  },
  admin: {
    label: "Server",
    items: [{ to: "/admin", label: "Overview", icon: Icon.grid, end: true }],
  },
};

export function Shell({ area, className = "", onSignOut, children }) {
  const nav = NAV[area];
  return (
    <div className={`shell ${className}`}>
      <aside className="sidebar">
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <Logo onDark />
          {area === "admin" && <span className="pill lilac">admin</span>}
        </div>
        <nav aria-label={nav.label}>
          <span className="side-label">{nav.label}</span>
          {nav.items.map((it) => (
            <NavLink key={it.to} to={it.to} end={it.end} className={({ isActive }) => `side-link${isActive ? " active" : ""}`}>
              {it.icon}{it.label}
            </NavLink>
          ))}
        </nav>
        {nav.other && (
          <nav aria-label="Elsewhere">
            <span className="side-label">Elsewhere</span>
            <NavLink to={nav.other.to} className="side-link">{nav.other.icon}{nav.other.label}</NavLink>
          </nav>
        )}
        {onSignOut && (
          <button type="button" className="side-link side-bottom" onClick={onSignOut} style={{ background: "none", border: "none", cursor: "pointer", width: "100%" }}>
            {Icon.logout}Sign out
          </button>
        )}
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}

export function PageHead({ title, sub, right }) {
  return (
    <div className="page-head">
      <div>
        <h1 className="page-title">{title}</h1>
        {sub && <p className="page-sub">{sub}</p>}
      </div>
      {right}
    </div>
  );
}

export function PreviewPill() {
  return (
    <span className="pill dashed" title="This part of the server isn't built yet, so sample data is shown.">
      Preview data
    </span>
  );
}

/* ---------- Key gate: asks for a key once per tab ---------- */
export function KeyGate({ kind, title, help, children, onSaved }) {
  const [has, setHas] = useState(() => Boolean(getKey(kind)));
  const [value, setValue] = useState("");
  if (has) return children;
  return (
    <section className="card" style={{ maxWidth: 520 }}>
      <h2 className="section-title">{title}</h2>
      <p className="muted" style={{ lineHeight: 1.5 }}>{help}</p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!value.trim()) return;
          setKey(kind, value.trim());
          setHas(true);
          onSaved?.();
        }}
        style={{ display: "flex", flexDirection: "column", gap: 14 }}
      >
        <label className="field">
          Key
          <input type="password" autoComplete="off" value={value} onChange={(e) => setValue(e.target.value)} />
        </label>
        <button className="btn" type="submit">Continue</button>
      </form>
    </section>
  );
}

/* ---------- Accuracy by round ---------- */
export function AccuracyChart({ history = [], maxRounds = 30, target = 0.6 }) {
  const W = 760, H = 300, L = 48, R = 740, T = 20, B = 260;
  const x = (r) => L + ((r - 1) * (R - L)) / Math.max(1, maxRounds - 1);
  const y = (p) => B - ((B - T) * p) / 100;
  const pts = history.map((h) => [x(h.round), y(pct(h.accuracy)), h]);
  const last = history[history.length - 1];
  const ticks = [1, ...[5, 10, 15, 20, 25, 30, 40, 50].filter((t) => t < maxRounds), maxRounds];
  const t = pct(target);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" style={{ width: "100%", height: "auto", display: "block" }}
      aria-label={last ? `Test accuracy over ${history.length} rounds, latest ${pct(last.accuracy).toFixed(1)} percent, target ${t} percent` : "No rounds finished yet"}>
      {[0, 20, 40, 60, 80, 100].map((p) => (
        <g key={p}>
          <line x1={L} y1={y(p)} x2={R} y2={y(p)} stroke="#E3E3E3" />
          <text x={L - 8} y={y(p) + 4} textAnchor="end" fontSize="12" fill="#555">{p}%</text>
        </g>
      ))}
      <line x1={L} y1={y(t)} x2={R} y2={y(t)} stroke="#555" strokeWidth="1.5" strokeDasharray="6 5" />
      <text x={R - 4} y={y(t) - 8} textAnchor="end" fontSize="12" fill="#555">target {t}%</text>
      {pts.length > 1 && (
        <polyline points={pts.map(([px, py]) => `${px},${py}`).join(" ")} fill="none" stroke="#161616" strokeWidth="2.5" strokeLinejoin="round" />
      )}
      {pts.map(([px, py, h]) => (
        <g key={h.round}>
          <title>{`Round ${h.round}: ${pct(h.accuracy).toFixed(1)}%, ${h.reported} reported${h.dropped ? `, ${h.dropped} dropped` : ""}`}</title>
          {h.dropped ? (
            <circle cx={px} cy={py} r="6" fill="#fff" stroke="#161616" strokeWidth="2" strokeDasharray="3 2" />
          ) : (
            <circle cx={px} cy={py} r="4" fill="#161616" />
          )}
        </g>
      ))}
      {last && (
        <g>
          <line x1={x(last.round)} y1={T} x2={x(last.round)} y2={B} stroke="#161616" strokeDasharray="3 4" />
          <text x={x(last.round) + 6} y={T + 14} fontSize="12" fill="#161616">now · round {last.round}</text>
        </g>
      )}
      {ticks.map((r) => (
        <text key={r} x={x(r)} y={B + 22} textAnchor="middle" fontSize="12" fill="#555">{r}</text>
      ))}
      <text x={(L + R) / 2} y={H - 2} textAnchor="middle" fontSize="12" fill="#555">Round</text>
      {!history.length && <text x={(L + R) / 2} y={(T + B) / 2} textAnchor="middle" fontSize="14" fill="#555">The first round's accuracy appears here</text>}
    </svg>
  );
}
