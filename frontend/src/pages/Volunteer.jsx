import { useState } from "react";
import { Link } from "react-router-dom";
import { Shell, PageHead, Icon } from "../components/ui.jsx";

const origin = typeof window !== "undefined" ? window.location.origin : "[server-url]";

export function VolunteerStart() {
  const [asked, setAsked] = useState(null);
  return (
    <Shell area="volunteer">
      <PageHead
        title="Lend your laptop"
        sub="Your idle CPU trains someone else's model. Everything happens in the small Lattice app; this page is for downloads and your history."
      />
      <section className="card" style={{ padding: 30, gap: 20 }}>
        <h2 style={{ fontSize: 26, fontWeight: 600 }}>Get the Lattice app</h2>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button type="button" className="btn" onClick={() => setAsked("Windows")}>{Icon.download}Download for Windows</button>
          <button type="button" className="btn secondary" onClick={() => setAsked("Mac")}>{Icon.download}Download for Mac</button>
        </div>
        {asked && (
          <p className="note" role="status">
            The {asked} installer isn't published yet. Until it is, use the terminal option below, or ask the
            person running this server for the client folder.
          </p>
        )}
        <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
          {[
            "Download. Your server and token are already inside.",
            "Double-click it. Lattice installs and opens its window.",
            "Click Volunteer now in the app.",
          ].map((text, i) => (
            <li key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
              <span className="pill dark" style={{ width: 28, height: 28, padding: 0, justifyContent: "center", flex: "0 0 28px" }}>{i + 1}</span>
              <span style={{ fontSize: 15, lineHeight: 1.4 }}>{text}</span>
            </li>
          ))}
        </ol>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: "14px 16px", border: "1.5px dashed var(--dash)", borderRadius: 18 }}>
          <span style={{ fontSize: 14, fontWeight: 500 }}>Prefer the terminal?</span>
          <code style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12, lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
            {`Mac / Linux:  curl -fsSL ${origin}/install.sh | sh\nWindows:      irm ${origin}/install.ps1 | iex`}
          </code>
        </div>
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
