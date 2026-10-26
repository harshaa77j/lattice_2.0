import { useCallback, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  api, ApiError, clearKey, usePolling, pct, fmtPct, fmtDuration, SAMPLE_JOB, SAMPLE_JOBS,
} from "../api.js";
import { Shell, PageHead, PreviewPill, KeyGate, AccuracyChart, Icon } from "../components/ui.jsx";

const STATUS = {
  waiting_for_nodes: "Waiting for volunteers",
  training: "Training",
  done: "Done",
  stopped: "Stopped",
};

function ResearcherGate({ children }) {
  const [gen, setGen] = useState(0);
  return (
    <KeyGate key={gen} kind="researcher" title="Enter your researcher key"
      help="Lattice uses a researcher key to stop strangers from submitting jobs to your volunteers. Ask whoever runs your server for it."
      onSaved={() => setGen((g) => g)}>
      {typeof children === "function" ? children(() => { clearKey("researcher"); setGen((g) => g + 1); }) : children}
    </KeyGate>
  );
}

function ApiErrorBox({ error, onBadKey }) {
  if (!error) return null;
  if (error instanceof ApiError && error.badKey) {
    return (
      <div className="error">
        The server didn't accept your researcher key.{" "}
        <button type="button" className="btn small secondary" onClick={onBadKey} style={{ marginLeft: 8 }}>Enter it again</button>
      </div>
    );
  }
  return <div className="error">{error.message}</div>;
}

/* ================= New job ================= */
export function NewJob() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: "cifar10-run-1", target: 60, maxRounds: 30, epochs: 1 });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const problems = [];
  if (!form.name.trim()) problems.push("Give the job a name.");
  if (!(form.target >= 1 && form.target <= 99)) problems.push("Target must be between 1 and 99%.");
  if (!(form.maxRounds >= 1 && form.maxRounds <= 200)) problems.push("Max rounds must be between 1 and 200.");
  if (!(form.epochs >= 1 && form.epochs <= 10)) problems.push("Epochs per round must be between 1 and 10.");

  return (
    <Shell area="research">
      <PageHead title="Start a new job" sub="Pick a dataset and a model, set when to stop, and Lattice handles the rest." />
      <ResearcherGate>
        {(resetKey) => (
          <form
            className="card"
            style={{ padding: 28, gap: 22 }}
            onSubmit={async (e) => {
              e.preventDefault();
              if (problems.length) return;
              setBusy(true); setError(null);
              try {
                await api.createJob({
                  name: form.name.trim(), dataset: "cifar10", model: "cnn_small",
                  target_accuracy: Number(form.target) / 100,
                  max_rounds: Number(form.maxRounds), local_epochs: Number(form.epochs),
                });
                navigate("/researcher/current");
              } catch (err) {
                setError(err instanceof ApiError && err.notBuilt
                  ? new Error("Job submission isn't live on the server yet. It arrives with the job endpoints.")
                  : err);
              } finally { setBusy(false); }
            }}
          >
            <div>
              <h2 style={{ fontSize: 30, fontWeight: 600, letterSpacing: "-0.02em" }}>Job settings</h2>
              <p className="muted" style={{ marginTop: 6, fontSize: 15 }}>Jobs send settings only, never code.</p>
            </div>
            <div className="row-wrap" style={{ gap: 28 }}>
              <div style={{ flex: "1 1 320px", display: "flex", flexDirection: "column", gap: 18 }}>
                <label className="field">Job name<input value={form.name} onChange={set("name")} maxLength={64} /></label>
                <div className="field-group">
                  <span>Dataset</span>
                  <div className="chips">
                    <button type="button" className="chip on" aria-pressed="true">CIFAR-10 · 50,000 images · 10 classes</button>
                    <button type="button" className="chip" disabled>Your data (soon)</button>
                  </div>
                </div>
                <div className="field-group">
                  <span>Model</span>
                  <div className="chips">
                    <button type="button" className="chip on" aria-pressed="true">cnn_small · 620k params · 2.5 MB</button>
                    <button type="button" className="chip" disabled>Model library (soon)</button>
                  </div>
                </div>
              </div>
              <div style={{ flex: "1 1 280px", display: "flex", flexDirection: "column", gap: 18 }}>
                <div className="grid-3">
                  <label className="field">Target %<input type="number" min="1" max="99" value={form.target} onChange={set("target")} /></label>
                  <label className="field">Max rounds<input type="number" min="1" max="200" value={form.maxRounds} onChange={set("maxRounds")} /></label>
                  <label className="field">Epochs / round<input type="number" min="1" max="10" value={form.epochs} onChange={set("epochs")} /></label>
                </div>
                <p className="note">
                  Stops when test accuracy reaches {form.target || "?"}% or after {form.maxRounds || "?"} rounds,
                  whichever comes first. You can download the trained model when it's done.
                </p>
                {problems.length > 0 && <div className="error">{problems[0]}</div>}
                <ApiErrorBox error={error} onBadKey={resetKey} />
                <button className="btn block" type="submit" disabled={busy || problems.length > 0}>
                  {busy ? "Submitting…" : "Submit job"}
                </button>
              </div>
            </div>
          </form>
        )}
      </ResearcherGate>
    </Shell>
  );
}

/* ================= Current job ================= */
async function fetchCurrentJob() {
  const jobs = await api.listJobs();
  if (!jobs.length) return null;
  const pick = jobs.find((j) => j.status === "training" || j.status === "waiting_for_nodes") || jobs[0];
  return api.getJob(pick.job_id);
}

function elapsedSeconds(job) {
  if (!job.started_at) return job.history?.reduce((s, h) => s + (h.seconds || 0), 0) ?? null;
  const end = job.finished_at ? new Date(job.finished_at) : new Date();
  return (end - new Date(job.started_at)) / 1000;
}

function CurrentJobBody({ resetKey }) {
  const { data: job, preview, error, loading } = usePolling(fetchCurrentJob, 2000, SAMPLE_JOB);

  if (loading) return <p className="muted">Loading…</p>;
  if (error && !job) return <ApiErrorBox error={error} onBadKey={resetKey} />;
  if (!job) {
    return (
      <section className="card" style={{ maxWidth: 560 }}>
        <h2 className="section-title">No jobs yet</h2>
        <p className="muted">Start a job and its progress shows up here, round by round.</p>
        <Link className="btn" to="/researcher/new" style={{ alignSelf: "flex-start" }}>Start a job</Link>
      </section>
    );
  }

  const history = job.history || [];
  const last = history[history.length - 1];
  const prev = history[history.length - 2];
  const delta = last && prev ? pct(last.accuracy) - pct(prev.accuracy) : null;
  const avg = history.length ? history.reduce((s, h) => s + (h.seconds || 0), 0) / history.length : null;
  const left = avg != null ? Math.max(0, job.max_rounds - job.current_round) * avg : null;
  const v = job.volunteers || {};
  const target = job.target_accuracy ?? 0.6;
  const live = job.status === "training" || job.status === "waiting_for_nodes";

  return (
    <>
      <div className="page-head" style={{ alignItems: "center" }}>
        <h2 style={{ fontSize: 26, fontWeight: 600 }}>{job.name}</h2>
        <div style={{ display: "flex", gap: 8 }}>
          {preview && <PreviewPill />}
          <span className={`pill ${live ? "dark" : "outline"}`}>{STATUS[job.status] || job.status}</span>
        </div>
      </div>
      {error && <ApiErrorBox error={error} onBadKey={resetKey} />}

      <div className="grid-cards">
        <div className="stat white">
          <span className="stat-title">Round:</span>
          <span className="stat-big">{job.current_round} / {job.max_rounds}</span>
          <div className="track" style={{ background: "var(--row)" }}>
            <div style={{ width: `${Math.min(100, (100 * job.current_round) / job.max_rounds)}%` }} />
          </div>
          <span className="sub small">Stops at {pct(target)}% or round {job.max_rounds}</span>
        </div>
        <div className="stat white">
          <span className="stat-title">Test accuracy:</span>
          <span className="stat-big">{last ? fmtPct(last.accuracy) : "–"}</span>
          <span className="sub small">
            best {fmtPct(job.best_accuracy)}
            {delta != null && ` · ${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts last round`}
            {` · target ${pct(target)}%`}
          </span>
        </div>
        <div className="stat white">
          <span className="stat-title">Volunteers:</span>
          <div className="stat-nums">
            {[["Training", v.training], ["Reported", v.reported], ["Dropped", v.dropped]].map(([label, n]) => (
              <div key={label}><p className="stat-num">{n ?? 0}</p><p className="stat-label sub">{label}</p></div>
            ))}
          </div>
        </div>
        <div className="stat white">
          <span className="stat-title">Time:</span>
          <div className="stat-nums">
            <div><p className="stat-num">{fmtDuration(elapsedSeconds(job))}</p><p className="stat-label sub">Elapsed</p></div>
            <div><p className="stat-num">{avg != null ? `${Math.round(avg)} s` : "–"}</p><p className="stat-label sub">Per round</p></div>
          </div>
          {live && left != null && <span className="sub small">at most {fmtDuration(left)} left at this pace</span>}
        </div>
      </div>

      <section className="card">
        <div className="page-head" style={{ alignItems: "center" }}>
          <h2 style={{ fontSize: 22, fontWeight: 600 }}>Test accuracy by round</h2>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 13, color: "var(--ink-2)" }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span style={{ width: 18, height: 3, background: "#161616" }} />Accuracy after each round</span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span style={{ width: 18, borderTop: "2px dashed #555" }} />Target</span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span style={{ width: 12, height: 12, borderRadius: "50%", border: "2px dashed #161616" }} />A volunteer dropped out</span>
          </div>
        </div>
        <AccuracyChart history={history} maxRounds={job.max_rounds} target={target} />
        <p className="muted small">Hover a point to see its round, accuracy, and how many volunteers reported.</p>
      </section>

      <section className="list">
        <h2 className="section-title" style={{ padding: "0 4px" }}>Rounds in this job</h2>
        {live && (
          <div className="list-row highlight">
            <span className="list-icon" style={{ background: "var(--row)" }}>{job.current_round}</span>
            <div className="list-body"><p className="list-title">Round {job.current_round} · in progress</p><p className="list-meta">{v.training ?? 0} volunteers training</p></div>
          </div>
        )}
        {[...history].reverse().map((h) => (
          <div className="list-row" key={h.round}>
            <span className={`list-icon${h.dropped ? " dashed" : ""}`}>{h.round}</span>
            <div className="list-body">
              <p className="list-title">{fmtPct(h.accuracy)} accuracy</p>
              <p className="list-meta">{h.reported} reported · {h.dropped || 0} dropped{h.dropped ? ", round still finished" : ""}</p>
            </div>
            <span className="pill light">{h.seconds != null ? `${Math.round(h.seconds)} s` : "–"}</span>
          </div>
        ))}
        {!history.length && <p className="muted" style={{ padding: "0 4px" }}>No rounds have finished yet.</p>}
      </section>
    </>
  );
}

export function CurrentJob() {
  return (
    <Shell area="research" className="gray">
      <PageHead title="Current job" sub="Your job's progress, refreshed every 2 seconds." />
      <ResearcherGate>{(resetKey) => <CurrentJobBody resetKey={resetKey} />}</ResearcherGate>
    </Shell>
  );
}

/* ================= All jobs ================= */
function JobsBody({ resetKey }) {
  const { data: jobs, preview, error, loading } = usePolling(api.listJobs, 5000, SAMPLE_JOBS);
  const [filter, setFilter] = useState("all");
  const [dlError, setDlError] = useState(null);

  const download = useCallback(async (job) => {
    setDlError(null);
    try {
      const res = await api.downloadModel(job.job_id);
      const url = URL.createObjectURL(await res.blob());
      const a = document.createElement("a");
      a.href = url; a.download = `${job.name}.pt`; a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setDlError(e instanceof ApiError && e.notBuilt ? new Error("Model downloads aren't live on the server yet.") : e);
    }
  }, []);

  if (loading) return <p className="muted">Loading…</p>;
  const list = (jobs || []).filter((j) =>
    filter === "all" ? true : filter === "training" ? ["training", "waiting_for_nodes"].includes(j.status) : ["done", "stopped"].includes(j.status));

  return (
    <>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        {[["all", "All"], ["training", "Training"], ["done", "Finished"]].map(([k, label]) => (
          <button key={k} type="button" className={`chip${filter === k ? " on" : ""}`} aria-pressed={filter === k} onClick={() => setFilter(k)}>{label}</button>
        ))}
        {preview && <PreviewPill />}
        <Link className="btn small" to="/researcher/new" style={{ marginLeft: "auto" }}>New job</Link>
      </div>
      <ApiErrorBox error={error || dlError} onBadKey={resetKey} />
      <section className="list">
        {list.map((j) => {
          const live = ["training", "waiting_for_nodes"].includes(j.status);
          return (
            <div key={j.job_id} className={`list-row${live ? " lilac" : ""}`} style={{ padding: "14px 16px", borderRadius: 22 }}>
              <span className={`list-icon ${j.status === "done" ? "mint" : j.status === "stopped" ? "lime" : ""}`}>{Icon.bars}</span>
              <div className="list-body">
                <p className="list-title" style={{ fontSize: 17 }}>{j.name}</p>
                <p className="list-meta">CIFAR-10 · {j.model}{j.finished_at ? ` · finished ${new Date(j.finished_at).toLocaleString()}` : ""}</p>
              </div>
              {live && <span className="pill dark">{STATUS[j.status]} · round {j.current_round}/{j.max_rounds}</span>}
              {j.status === "done" && <span className="pill mint">Done · {fmtPct(j.best_accuracy)}</span>}
              {j.status === "stopped" && <span className="pill dashed">Stopped at round {j.current_round} · {fmtPct(j.best_accuracy)}</span>}
              {live
                ? <Link className="btn small secondary" to="/researcher/current">Open</Link>
                : <button type="button" className="btn small" onClick={() => download(j)}>Download model</button>}
            </div>
          );
        })}
        {!list.length && (
          <div className="card" style={{ maxWidth: 560 }}>
            <p className="muted">{filter === "all" ? "You haven't submitted any jobs yet." : "No jobs match this filter."}</p>
            {filter === "all" && <Link className="btn" to="/researcher/new" style={{ alignSelf: "flex-start" }}>Start a job</Link>}
          </div>
        )}
      </section>
    </>
  );
}

export function AllJobs() {
  return (
    <Shell area="research">
      <PageHead title="Your jobs" sub="Every job you've submitted. Finished models stay available to download." />
      <ResearcherGate>{(resetKey) => <JobsBody resetKey={resetKey} />}</ResearcherGate>
    </Shell>
  );
}
