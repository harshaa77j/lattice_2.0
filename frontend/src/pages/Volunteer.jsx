import { useState } from "react";
import { Link } from "react-router-dom";
import { Shell, PageHead, Icon } from "../components/ui.jsx";

async function downloadPackage(platform, code) {
  const res = await fetch(`/download/${platform}?code=${encodeURIComponent(code)}`);
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail || ""; } catch { /* not JSON */ }
    throw new Error(res.status === 403
      ? (detail === "invalid invite code" ? "That invite code doesn't match. Check it with the person who invited you."
                                          : "Downloads aren't enabled on this server yet.")
      : `The download failed (${res.status}). Try again in a minute.`);
  }
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = `lattice-volunteer-${platform}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

const STEPS = {
  windows: ["Extract the zip (right-click \u2192 Extract All).",
            "Double-click install.bat. If Windows warns you, choose More info \u2192 Run anyway.",
            "Double-click start.bat to volunteer. Close its window to stop."],
  mac: ["Double-click the zip to unzip it.",
        "Right-click install.command \u2192 Open \u2192 Open (first time only).",
        "Open start.command the same way to volunteer. Close its window to stop."],
};

export function VolunteerStart() {
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [done, setDone] = useState(null);

  const [copied, setCopied] = useState(false);
  const command = `irm "${window.location.origin}/install.ps1?code=${encodeURIComponent(code.trim())}" | iex`;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("Couldn't copy automatically. Select the line and copy it with Ctrl+C.");
    }
  };

  const get = async (platform) => {
    setError(null); setDone(null); setBusy(platform);
    try { await downloadPackage(platform, code.trim()); setDone(platform); }
    catch (e) { setError(e.message); }
    finally { setBusy(null); }
  };

  return (
    <Shell area="volunteer">
      <PageHead
        title="Lend your laptop"
        sub="Your idle computer trains someone else's model. Download the volunteer package, install it once, and start it whenever you want to help."
      />
      <section className="card" style={{ padding: 30, gap: 20 }}>
        <h2 style={{ fontSize: 26, fontWeight: 600 }}>Get the volunteer package</h2>
        <label className="field" style={{ maxWidth: 420 }}>
          <span>Invite code</span>
          <input value={code} onChange={(e) => setCode(e.target.value)} placeholder="From the person who invited you"
                 autoComplete="off" spellCheck={false} />
        </label>
        {code.trim() && (
          <div className="note" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <strong>Windows: install in one step (recommended)</strong>
            <span className="small">
              Open the Start menu, type <strong>PowerShell</strong> and press Enter. Paste this line and press Enter.
              It installs everything Lattice needs (including Python, if missing), adds a <strong>Lattice</strong> icon
              to your Desktop, and opens the app. About 10 minutes the first time.
            </span>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <code style={{ flex: "1 1 320px", padding: "10px 12px", background: "#FFFFFF", border: "1px solid #E3E3DE",
                             borderRadius: 10, fontSize: 13, overflowX: "auto", whiteSpace: "nowrap" }}>{command}</code>
              <button type="button" className="btn secondary" onClick={copy}>{copied ? "Copied \u2713" : "Copy"}</button>
            </div>
          </div>
        )}
        <span className="muted small">Or download the package and install it yourself:</span>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button type="button" className="btn" disabled={!code.trim() || busy} onClick={() => get("windows")}>
            {Icon.download}{busy === "windows" ? "Preparing\u2026" : "Download for Windows"}
          </button>
          <button type="button" className="btn secondary" disabled={!code.trim() || busy} onClick={() => get("mac")}>
            {Icon.download}{busy === "mac" ? "Preparing\u2026" : "Download for Mac"}
          </button>
        </div>
        {error && <p className="error" role="alert">{error}</p>}
        {done && (
          <div className="note" role="status" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <strong>Downloaded. Next steps on {done === "mac" ? "your Mac" : "Windows"}:</strong>
            <ol style={{ margin: 0, paddingLeft: 20, display: "flex", flexDirection: "column", gap: 4 }}>
              {STEPS[done].map((s, i) => <li key={i}>{s}</li>)}
            </ol>
          </div>
        )}
        <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
          {[
            "Download. The server address and access are already inside.",
            "Run the installer once. It sets everything up in its own folder.",
            "Start it whenever you want to help. Close the window to stop.",
          ].map((text, i) => (
            <li key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
              <span className="pill dark" style={{ width: 28, height: 28, padding: 0, justifyContent: "center", flex: "0 0 28px" }}>{i + 1}</span>
              <span style={{ fontSize: 15, lineHeight: 1.4 }}>{text}</span>
            </li>
          ))}
        </ol>
        <p className="muted" style={{ fontSize: 14, margin: 0 }}>
          The one-step installer sets up Python by itself. The downloaded package needs Python 3.10 or newer;
          its installer tells you if it's missing.
        </p>
      </section>
      <p className="muted" style={{ fontSize: 15 }}>
        Wondering what Lattice does on your laptop? <Link to="/volunteer/safety">Read how it keeps you safe</Link>.
      </p>
    </Shell>
  );
}

function Point({ title, text }) {
  return (
    <div>
      <p style={{ fontSize: 16, fontWeight: 600 }}>{title}</p>
      <p className="muted" style={{ marginTop: 2, fontSize: 14, lineHeight: 1.45 }}>{text}</p>
    </div>
  );
}

export function VolunteerSafety() {
  return (
    <Shell area="volunteer">
      <PageHead title="Safe by design" sub="What Lattice does on your laptop, what it never does, and how to stop it." />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
        <section className="card">
          <h2 style={{ fontSize: 22, fontWeight: 600 }}>How Lattice keeps you safe:</h2>
          <Point title="No one else's code runs here" text="Jobs send settings, never programs." />
          <Point title="Model files can't execute" text="Weights load in a safe mode that only reads numbers." />
          <Point title="Nobody can connect in" text="Lattice only connects out and opens no ports." />
          <Point title="Your files stay yours" text="It reads only the dataset it downloads." />
          <p className="muted small">It does use your CPU, so your fan may spin up. Pause when you need full speed.</p>
        </section>
        <section className="card">
          <h2 style={{ fontSize: 22, fontWeight: 600 }}>Stop any time:</h2>
          <Point title="Take a break" text="Click Pause in the app. You stay connected but take no work." />
          <Point title="Stop completely" text="Click Stop Lattice, or close its window. The job carries on without you." />
          <Point title="Remove Lattice" text="Double-click Uninstall in the Lattice folder." />
          <p className="muted small">Stopping mid-round is safe. Nothing is left running.</p>
        </section>
      </div>
    </Shell>
  );
}

export function VolunteerHistory() {
  return (
    <Shell area="volunteer">
      <PageHead title="Your history" sub="Everything your laptops have trained, and when." />
      <section className="card" style={{ maxWidth: 640 }}>
        <h2 className="section-title">Coming with volunteer accounts</h2>
        <p className="muted" style={{ lineHeight: 1.55 }}>
          To show only your laptops, Lattice needs to know who you are, so this page arrives with volunteer
          sign-in. Until then, the Lattice app on each laptop shows its own rounds and compute for today.
        </p>
        <Link className="btn secondary" to="/volunteer" style={{ alignSelf: "flex-start" }}>Get the Lattice app</Link>
      </section>
    </Shell>
  );
}
