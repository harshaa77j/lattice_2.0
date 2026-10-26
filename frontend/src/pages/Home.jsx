import { Link } from "react-router-dom";
import { Logo } from "../components/ui.jsx";

const STEPS = [
  ["Pick nodes", "The scheduler chooses available laptops."],
  ["Train slices", "Each laptop trains on its own slice of data."],
  ["Average", "FedAvg combines every update that arrived."],
  ["Evaluate", "Accuracy is logged; dropped laptops are skipped."],
];

export default function Home() {
  return (
    <div className="home">
      <div className="home-inner">
        <header className="home-head">
          <Logo size={32} />
          <nav style={{ display: "flex", gap: 10, flexWrap: "wrap" }} aria-label="Main">
            <Link className="btn small" to="/researcher/new">Researcher</Link>
            <Link className="btn small secondary" to="/volunteer">Volunteer</Link>
          </nav>
        </header>

        <main style={{ paddingBottom: 48 }}>
          <h1 className="home-title">Train models on laptops you already have.</h1>
          <p className="page-sub" style={{ fontSize: 19, maxWidth: 660 }}>
            Lattice pools volunteer machines into one training cluster. Researchers submit a job, volunteers lend
            idle compute, and training keeps going when a laptop drops out.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16, marginTop: 48 }}>
            <Link to="/researcher/new" className="home-cta lime">
              <span style={{ fontSize: 30, fontWeight: 600, letterSpacing: "-0.02em" }}>I'm a researcher</span>
              <span className="sub" style={{ fontSize: 17, lineHeight: 1.5, maxWidth: 340 }}>
                Submit a training job and watch accuracy climb round by round.
              </span>
              <span className="btn small" style={{ marginTop: "auto", alignSelf: "flex-start" }}>Start a job</span>
            </Link>
            <Link to="/volunteer" className="home-cta sky">
              <span style={{ fontSize: 30, fontWeight: 600, letterSpacing: "-0.02em" }}>I'm a volunteer</span>
              <span className="sub" style={{ fontSize: 17, lineHeight: 1.5, maxWidth: 340 }}>
                Lend your laptop's idle CPU to someone else's model. Stop any time.
              </span>
              <span className="btn small" style={{ marginTop: "auto", alignSelf: "flex-start" }}>Join the cluster</span>
            </Link>
          </div>

          <section style={{ marginTop: 56 }}>
            <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 16 }}>How a round works</h2>
            <ol style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16 }}>
              {STEPS.map(([title, text], i) => (
                <li key={title} className="card" style={{ padding: 22, gap: 8, borderRadius: 22 }}>
                  <span className="pill dark" style={{ width: 32, height: 32, padding: 0, justifyContent: "center" }}>{i + 1}</span>
                  <span style={{ fontSize: 18, fontWeight: 600 }}>{title}</span>
                  <span className="muted" style={{ fontSize: 15, lineHeight: 1.5 }}>{text}</span>
                </li>
              ))}
            </ol>
          </section>
        </main>
      </div>
    </div>
  );
}
