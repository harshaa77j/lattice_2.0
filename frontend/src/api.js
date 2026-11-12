import { useEffect, useRef, useState } from "react";

// Must match shared/protocol.py on the server.
export const RESEARCHER_KEY_HEADER = "X-Researcher-Key";
export const ADMIN_KEY_HEADER = "X-Admin-Key";

const STORE = { researcher: "lattice.researcherKey", admin: "lattice.adminKey" };

// Keys live only for this browser tab (sessionStorage), never in code.
export function getKey(kind) {
  try { return sessionStorage.getItem(STORE[kind]) || ""; } catch { return ""; }
}
export function setKey(kind, value) {
  try { sessionStorage.setItem(STORE[kind], value); } catch { /* storage blocked */ }
}
export function clearKey(kind) {
  try { sessionStorage.removeItem(STORE[kind]); } catch { /* storage blocked */ }
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
  // 404/405: the server doesn't have this endpoint yet.
  get notBuilt() { return this.status === 404 || this.status === 405; }
  get badKey() { return this.status === 401 || this.status === 403; }
}

async function request(path, { kind, method = "GET", body, file, raw = false } = {}) {
  const headers = {};
  if (kind === "researcher") headers[RESEARCHER_KEY_HEADER] = getKey("researcher");
  if (kind === "admin") headers[ADMIN_KEY_HEADER] = getKey("admin");
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (file !== undefined) headers["Content-Type"] = "application/octet-stream";
  let res;
  try {
    res = await fetch(path, { method, headers, body: file !== undefined ? file : body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    throw new ApiError(0, "Can't reach the Lattice server. Check your connection and try again.");
  }
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail; } catch { /* not JSON */ }
    const fallback = res.status === 502 || res.status === 503
      ? "The server is restarting or unavailable. Try again in a minute."
      : res.statusText || `The server returned an error (${res.status}).`;
    throw new ApiError(res.status, typeof detail === "string" && detail ? detail : fallback);
  }
  return raw ? res : res.json();
}

// Uploads with progress. fetch can't report upload progress; XMLHttpRequest can.
// onProgress receives a fraction from 0 to 1; 1 means "all sent, the server is preparing".
function uploadWithProgress(path, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.setRequestHeader(RESEARCHER_KEY_HEADER, getKey("researcher"));
    xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
    xhr.upload.onload = () => { if (onProgress) onProgress(1); };
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new ApiError(xhr.status, (data && typeof data.detail === "string" && data.detail)
        || `The upload failed (${xhr.status}).`));
    };
    xhr.onerror = () => reject(new ApiError(0, "The upload failed: check your connection and try again."));
    xhr.send(file);
  });
}

// D8 contract (DESIGN.md)
export const api = {
  listModels: () => request("/models"),
  listDatasets: () => request("/datasets", { kind: "researcher" }),
  uploadDataset: (name, file, onProgress) =>
    uploadWithProgress(`/datasets?name=${encodeURIComponent(name)}&consent=true`, file, onProgress),
  deleteDataset: (id) => request(`/datasets/${encodeURIComponent(id)}`, { kind: "researcher", method: "DELETE" }),
  uploadStatus: (id) => request(`/datasets/uploads/${encodeURIComponent(id)}`, { kind: "researcher" }),
  recommend: (req) => request("/recommend", { kind: "researcher", method: "POST", body: req }),
  assistant: (req) => request("/assistant", { kind: "researcher", method: "POST", body: req }),
  listJobs: () => request("/jobs", { kind: "researcher" }),
  getJob: (id) => request(`/jobs/${encodeURIComponent(id)}`, { kind: "researcher" }),
  createJob: (job) => request("/jobs", { kind: "researcher", method: "POST", body: job }),
  downloadModel: (id) => request(`/jobs/${encodeURIComponent(id)}/model`, { kind: "researcher", raw: true }),
  adminStatus: () => request("/admin/status", { kind: "admin" }),
  adminNodes: () => request("/admin/nodes", { kind: "admin" }),
  adminEvents: () => request("/admin/events", { kind: "admin" }),
  adminDecisions: () => request("/admin/decisions?limit=5", { kind: "admin" }),
};

// Accuracy may arrive as a fraction (0.54) or a percentage (54).
export const pct = (v) => (v == null ? null : v <= 1 ? v * 100 : v);
export const fmtPct = (v) => (v == null ? "–" : `${pct(v).toFixed(1)}%`);
export function fmtDuration(seconds) {
  if (seconds == null) return "–";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}:${String(r).padStart(2, "0")}`;
}

/**
 * Calls `fetcher` now and every `intervalMs` while the tab is visible.
 * If the endpoint doesn't exist on the server yet (404/405), shows `sample`
 * and sets preview=true, so pages are usable before the backend catches up.
 */
export function usePolling(fetcher, intervalMs, sample) {
  const [state, setState] = useState({ data: null, preview: false, error: null, loading: true });
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    let alive = true;
    let timer;
    const tick = async () => {
      if (document.hidden) { timer = setTimeout(tick, intervalMs); return; }
      try {
        const data = await fetcherRef.current();
        if (alive) setState({ data, preview: false, error: null, loading: false });
      } catch (e) {
        if (!alive) return;
        if (e instanceof ApiError && e.notBuilt && sample !== undefined) {
          setState({ data: sample, preview: true, error: null, loading: false });
        } else {
          setState((s) => ({ ...s, error: e, loading: false }));
        }
      }
      if (alive && intervalMs) timer = setTimeout(tick, intervalMs);
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [intervalMs, sample]);

  return state;
}

/* ---------- Preview data (shown only while an endpoint isn't built yet) ---------- */
const sampleHistory = [18, 27, 34, 39, 43, 46, 45, 49, 51, 53, 55, 56].map((a, i) => ({
  round: i + 1,
  accuracy: a / 100,
  reported: i === 6 ? 4 : 5,
  dropped: i === 6 ? 1 : 0,
  seconds: 48 + (i % 3) * 4,
}));

export const SAMPLE_JOB = {
  job_id: "preview",
  name: "cifar10-run-1",
  dataset: "cifar10",
  dataset_name: "CIFAR-10",
  model: "cnn_small",
  status: "training",
  current_round: 12,
  max_rounds: 30,
  target_accuracy: 0.6,
  best_accuracy: 0.56,
  started_at: null,
  finished_at: null,
  volunteers: { training: 5, reported: 0, dropped: 0 },
  history: sampleHistory,
};

export const SAMPLE_JOBS = [
  { ...SAMPLE_JOB },
  { job_id: "p2", name: "cifar10-baseline", dataset: "cifar10", dataset_name: "CIFAR-10", model: "cnn_small", status: "done", current_round: 19, max_rounds: 30, best_accuracy: 0.61 },
  { job_id: "p3", name: "cifar10-short", dataset: "cifar10", dataset_name: "CIFAR-10", model: "cnn_small", status: "stopped", current_round: 10, max_rounds: 10, best_accuracy: 0.52 },
];

export const SAMPLE_ADMIN = {
  status: {
    uptime_seconds: 5400,
    version: "1.0.0",
    settings: { heartbeat_timeout: 20, round_deadline: 150, min_nodes: 2, scheduler: "heuristic" },
    counts: { online: 4, total: 5, standby: 1, dropped: 1, training: 3, cpus: 28 },
    current_job: { name: "cifar10-run-1", current_round: 12, max_rounds: 30, accuracy: 0.56 },
  },
  nodes: [
    { node_id: "windows-1", device: "cpu", cpu_count: 8, status: "training", seconds_since_seen: 2, reliability: 0.96 },
    { node_id: "mac", device: "mps", cpu_count: 8, status: "standby", seconds_since_seen: 4, reliability: 1 },
    { node_id: "lab-pc-3", device: "cpu", cpu_count: 4, status: "dropped", seconds_since_seen: 41, reliability: 0.72 },
  ],
  events: [
    { time: "12:04:51", kind: "round", text: "Round 12 closed: 4 of 5 reported, 56.0%" },
    { time: "12:04:10", kind: "dropped", text: "lab-pc-3 dropped: no heartbeat for 20 s" },
    { time: "11:58:02", kind: "info", text: "windows-1 re-registered after server restart" },
    { time: "11:57:54", kind: "info", text: "mac registered: mps, 8 cores" },
  ],
};