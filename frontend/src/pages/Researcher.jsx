import { useCallback, useEffect, useRef, useState } from "react";
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

/* ================= Model assistant (chat) ================= */
const PRIORITIES = [["fastest", "Fastest"], ["balanced", "Balanced"], ["accurate", "Most accurate"]];
const MAX_QUESTION = 500;

function AssistantMessage({ m, titleOf, isReady, currentModel, onUse }) {
  const mine = m.role === "user";
  return (
    <div style={{ display: "flex", justifyContent: mine ? "flex-end" : "flex-start" }}>
      <div style={{
        maxWidth: "88%", padding: "10px 12px", borderRadius: 14, whiteSpace: "pre-wrap", lineHeight: 1.45,
        background: mine ? "#17181C" : "#FFFFFF", color: mine ? "#FFFFFF" : "inherit",
        border: mine ? "none" : "1px solid #E3E3DE", display: "flex", flexDirection: "column", gap: 8,
      }}>
        <span className="small">{m.content}</span>
        {m.ranking && (
          <table className="small" style={{ borderCollapse: "collapse" }}>
            <tbody>
              {m.ranking.map((r) => (
                <tr key={r.model} style={{ fontWeight: r.model === m.suggested_model ? 600 : 400 }}>
                  <td style={{ paddingRight: 14 }}>{titleOf(r.model)}</td>
                  <td style={{ paddingRight: 14 }}>~{r.est_round_s} s / round</td>
                  <td>~{r.est_total_min} min total</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!mine && m.suggested_model && isReady(m.suggested_model) && (
          <div>
            <button type="button" className="btn small" disabled={currentModel === m.suggested_model}
              onClick={() => onUse(m.suggested_model)}>
              {currentModel === m.suggested_model ? "Selected \u2713" : `Use ${titleOf(m.suggested_model)}`}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function ModelAssistant({ form, setForm, models }) {
  const [priority, setPriority] = useState("balanced");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const logRef = useRef(null);

  const titleOf = (name) => models?.find((m) => m.name === name)?.title || name;
  const isReady = (name) => !!models?.find((m) => m.name === name && m.ready);
  const context = () => ({
    dataset: form.dataset || "cifar10", priority,
    max_rounds: Number(form.maxRounds) || 20, local_epochs: Number(form.epochs) || 1,
  });

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [messages, busy]);

  const recommend = async () => {
    setBusy(true); setError(null);
    try {
      const data = await api.recommend(context());
      setMessages((ms) => [...ms, { role: "assistant", content: data.explanation,
        suggested_model: data.model, ranking: data.ranking }]);
    } catch (e) {
      setError(e.message);
    } finally { setBusy(false); }
  };

  const ask = async () => {
    const q = input.trim().slice(0, MAX_QUESTION);
    if (!q || busy) return;
    const history = [...messages, { role: "user", content: q }];
    setMessages(history); setInput(""); setBusy(true); setError(null);
    try {
      const res = await api.assistant({
        ...context(),
        messages: history.slice(-10).map(({ role, content }) => ({ role, content })),
      });
      setMessages((ms) => [...ms, { role: "assistant", content: res.reply, suggested_model: res.suggested_model || null }]);
    } catch (e) {
      setError(e instanceof ApiError && e.notBuilt
        ? "Follow-up questions aren't available on this server yet. The recommendation above still works."
        : e.message);
    } finally { setBusy(false); }
  };

  return (
    <div className="note" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <strong>Model assistant</strong>
      <span className="muted small">
        Get a recommendation for the selected dataset and the volunteers online now, then ask follow-up questions.
      </span>
      <div className="chips">
        {PRIORITIES.map(([key, label]) => (
          <button key={key} type="button" className={`chip${priority === key ? " on" : ""}`}
            aria-pressed={priority === key} onClick={() => setPriority(key)}>{label}</button>
        ))}
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button type="button" className="btn secondary" disabled={busy} onClick={recommend}>
          {busy && !messages.length ? "Thinking\u2026" : "Recommend a model"}
        </button>
        {messages.length > 0 && (
          <button type="button" className="btn small secondary" disabled={busy}
            onClick={() => { setMessages([]); setError(null); }}>New conversation</button>
        )}
      </div>

      {(messages.length > 0 || busy) && (
        <div ref={logRef} style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 340, overflowY: "auto", paddingRight: 4 }}>
          {messages.map((m, i) => (
            <AssistantMessage key={i} m={m} titleOf={titleOf} isReady={isReady} currentModel={form.model}
              onUse={(name) => setForm((f) => ({ ...f, model: name }))} />
          ))}
          {busy && <span className="muted small">Thinking\u2026</span>}
        </div>
      )}

      {messages.length > 0 && (
        <div style={{ display: "flex", gap: 8 }}>
          <input value={input} maxLength={MAX_QUESTION} placeholder="Ask about models, time, or your data\u2026"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); ask(); } }}
            style={{ flex: 1, padding: "10px 12px", borderRadius: 10, border: "1px solid #E3E3DE", font: "inherit" }} />
          <button type="button" className="btn" disabled={busy || !input.trim()} onClick={ask}>Send</button>
        </div>
      )}
      {error && <p className="error" role="alert">{error}</p>}
      <span className="muted small">Estimates use measured model speeds and the volunteers online right now. Accuracies were measured on CIFAR-10 and are a guide only.</span>
    </div>
  );
}

/* ================= New job ================= */
export function NewJob() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: "cifar10-run-1", model: "cnn_bn", target: 80, maxRounds: 40, epochs: 1 });
  const [models, setModels] = useState(null);
  useEffect(() => {
    api.listModels().then(setModels).catch(() => setModels([]));
  }, []);

  const [datasets, setDatasets] = useState(null);
  const [upload, setUpload] = useState({ open: false, file: null, name: "", consent: false, busy: false,
    progress: null, error: null, done: null });
  const loadDatasets = (select) =>
    api.listDatasets().then((list) => {
      setDatasets(list);
      setForm((f) => {
        if (select) return { ...f, dataset: select };
        if (f.dataset || !list.length) return f;
        return { ...f, dataset: (list.find((d) => d.name === "CIFAR-10") || list[0]).id };
      });
    }).catch(() => setDatasets([]));
  useEffect(() => { loadDatasets(); }, []);

  const doUpload = async () => {
    setUpload((u) => ({ ...u, busy: true, progress: 0, error: null, done: null }));
    try {
      const meta = await api.uploadDataset(upload.name.trim(), upload.file,
        (fraction) => setUpload((u) => ({ ...u, progress: fraction })));
      await loadDatasets(meta.id);
      setUpload({ open: false, file: null, name: "", consent: false, busy: false, progress: null, error: null,
        done: `Uploaded "${meta.name}": ${meta.train_count} training and ${meta.test_count} test images in ${meta.num_classes} classes`
          + (meta.skipped_files ? ` (${meta.skipped_files} files skipped)` : "") + "." });
    } catch (e) {
      setUpload((u) => ({ ...u, busy: false, progress: null, error: e.message }));
    }
  };

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const problems = [];
  if (!form.name.trim()) problems.push("Give the job a name.");
  if (!(form.target >= 1 && form.target <= 99)) problems.push("Target must be between 1 and 99%.");
  if (!(form.maxRounds >= 1 && form.maxRounds <= 200)) problems.push("Max rounds must be between 1 and 200.");
  if (!(form.epochs >= 1 && form.epochs <= 10)) problems.push("Epochs per round must be between 1 and 10.");

  const uploadLabel = upload.progress == null ? null
    : upload.progress < 1 ? `Uploading\u2026 ${Math.round(upload.progress * 100)}%`
    : "Preparing your images\u2026 this can take a minute for large datasets.";

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
                  name: form.name.trim(), dataset: form.dataset || "cifar10", model: form.model,
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
                    {datasets === null && <span className="muted small">Loading datasets…</span>}
                    {datasets?.length === 0 && (
                      <button type="button" className="chip on" aria-pressed="true">CIFAR-10 (built-in)</button>
                    )}
                    {datasets?.map((d) => (
                      <button key={d.id} type="button" className={`chip${form.dataset === d.id ? " on" : ""}`}
                        aria-pressed={form.dataset === d.id}
                        onClick={() => setForm((f) => ({ ...f, dataset: d.id }))}>
                        {d.name} · {(d.train_count + d.test_count).toLocaleString()} images · {d.num_classes} classes
                      </button>
                    ))}
                    <button type="button" className={`chip${upload.open ? " on" : ""}`}
                      onClick={() => setUpload((u) => ({ ...u, open: !u.open, error: null }))}>
                      + Upload your data
                    </button>
                  </div>
                  {upload.done && <span className="muted small">{upload.done}</span>}
                  {upload.open && (
                    <div className="note" style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 8 }}>
                      <span className="small">
                        A <strong>.zip with one folder per class</strong> (for example <code>cats/</code> and <code>dogs/</code>),
                        at least 2 classes with 5 or more images each, up to 200 MB. Images are resized to 32×32 pixels,
                        and 10% of each class is kept aside for testing.
                      </span>
                      <label className="field">Dataset name
                        <input value={upload.name} maxLength={64} placeholder="e.g. Leaf diseases" disabled={upload.busy}
                          onChange={(e) => setUpload((u) => ({ ...u, name: e.target.value }))} />
                      </label>
                      <input type="file" accept=".zip,application/zip" disabled={upload.busy}
                        onChange={(e) => setUpload((u) => ({ ...u, file: e.target.files?.[0] || null,
                          name: u.name || (e.target.files?.[0]?.name || "").replace(/\.zip$/i, "") }))} />
                      <label className="small" style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                        <input type="checkbox" checked={upload.consent} disabled={upload.busy}
                          onChange={(e) => setUpload((u) => ({ ...u, consent: e.target.checked }))} />
                        <span>I understand that volunteers' computers will receive and process these images while training.
                          Don't upload private or sensitive images.</span>
                      </label>
                      {uploadLabel && (
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }} role="status" aria-live="polite">
                          <div className="track" style={{ background: "var(--row)" }}>
                            <div style={{ width: `${Math.round((upload.progress || 0) * 100)}%`, transition: "width 0.2s" }} />
                          </div>
                          <span className="muted small">{uploadLabel}</span>
                        </div>
                      )}
                      {upload.error && <p className="error" role="alert">{upload.error}</p>}
                      <div>
                        <button type="button" className="btn"
                          disabled={!upload.file || !upload.name.trim() || !upload.consent || upload.busy}
                          onClick={doUpload}>
                          {upload.busy ? "Uploading\u2026" : "Upload dataset"}
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                <div className="field-group">
                  <span>Model</span>
                  <div className="chips">
                    {models === null && <span className="muted small">Loading the model library…</span>}
                    {models?.length === 0 && <span className="muted small">Couldn't load the model library.</span>}
                    {models?.map((m) => (
                      <button key={m.name} type="button" disabled={!m.ready}
                        className={`chip${form.model === m.name ? " on" : ""}`}
                        aria-pressed={form.model === m.name}
                        title={m.description}
                        onClick={() => setForm((f) => ({ ...f, model: m.name }))}>
                        {m.title}{m.ready ? ` · ${Math.round(m.parameters / 1000)}k params · ${m.size_mb} MB` : " (soon)"}
                      </button>
                    ))}
                  </div>
                  {models?.find((m) => m.name === form.model) && (
                    <span className="muted small">{models.find((m) => m.name === form.model).description}</span>
                  )}
                </div>

                <ModelAssistant form={form} setForm={setForm} models={models} />
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
        <div>
          <h2 style={{ fontSize: 26, fontWeight: 600 }}>{job.name}</h2>
          <p className="muted small" style={{ marginTop: 4 }}>{job.dataset_name || "CIFAR-10"} · {job.model}</p>
        </div>
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
                <p className="list-meta">{j.dataset_name || "CIFAR-10"} · {j.model}{j.finished_at ? ` · finished ${new Date(j.finished_at).toLocaleString()}` : ""}</p>
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