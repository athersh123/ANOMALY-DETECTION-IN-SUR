// ============================================================
//  SentinelAI – API Client  v2.0  (api.js)
//  All fetch() calls to Flask backend in one place.
//  Shared by dashboard.html, wildlife.html, anomaly.html,
//  history.html
// ============================================================

const API = (() => {
  'use strict';

  const BASE = window.location.protocol === 'file:' ? 'http://localhost:5000' : `${window.location.protocol}//${window.location.hostname}:5000`;

  // ── Generic fetch helpers ────────────────────────────────
  async function get(path) {
    try {
      const res = await fetch(BASE + path, {
        method:  'GET',
        headers: { 'Accept': 'application/json' },
        signal:  AbortSignal.timeout(8000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return null;
    }
  }

  async function post(path, body) {
    try {
      const res = await fetch(BASE + path, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
        signal:  AbortSignal.timeout(8000),
      });
      return await res.json();
    } catch (e) {
      return null;
    }
  }

  // ── Video feed URLs ──────────────────────────────────────
  const VIDEO_WILDLIFE = BASE + '/video_feed/wildlife';
  const VIDEO_ANOMALY  = BASE + '/video_feed/anomaly';

  // ── Standard polling endpoints ───────────────────────────
  const getStatus          = () => get('/api/status');
  const getWildlife        = () => get('/api/wildlife');
  const getAnomaly         = () => get('/api/anomaly');
  const getWildlifeHistory = () => get('/api/wildlife/history');
  const getAnomalyHistory  = () => get('/api/anomaly/history');
  const getAlerts          = (limit = 20, type = '') =>
    get(`/api/alerts?limit=${limit}${type ? '&type=' + type : ''}`);
  const postSettings       = (body) => post('/api/settings', body);

  // ── Backend reachability check ───────────────────────────
  async function checkBackend() {
    const data = await get('/');
    return data !== null;
  }

  // ── Upload Image – Wildlife ──────────────────────────────
  async function uploadWildlifeImage(file) {
    console.log("Upload started", file.name);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch(BASE + '/api/wildlife/upload-image', {
        method: 'POST',
        body:   fd,
        signal: AbortSignal.timeout(30000),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      console.log("API response", data);
      return data;
    } catch (e) {
      console.error("Upload failed", e);
      throw e;
    }
  }

  // ── Upload Image – Anomaly ───────────────────────────────
  async function uploadAnomalyImage(file) {
    console.log("Upload started", file.name);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch(BASE + '/api/anomaly/upload-image', {
        method: 'POST',
        body:   fd,
        signal: AbortSignal.timeout(30000),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      console.log("API response", data);
      return data;
    } catch (e) {
      console.error("Upload failed", e);
      throw e;
    }
  }

  // ── Upload Video with XHR progress ──────────────────────
  function _uploadVideo(endpoint, file, onProgress) {
    return new Promise((resolve, reject) => {
      console.log("Upload started", file.name, "to", endpoint);
      const fd  = new FormData();
      fd.append('file', file);
      const xhr = new XMLHttpRequest();
      xhr.open('POST', BASE + endpoint, true);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress)
          onProgress(Math.round((e.loaded / e.total) * 50)); // upload = first 50%
      };
      xhr.onload = () => {
        if (xhr.status === 202) {
          try {
            const data = JSON.parse(xhr.responseText);
            console.log("API response", data);
            resolve(data);
          }
          catch (_) { reject(new Error('Invalid server response')); }
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.error || `HTTP ${xhr.status}`));
          } catch (_) {
            reject(new Error(`HTTP ${xhr.status}`));
          }
        }
      };
      xhr.onerror = () => reject(new Error('Network error'));
      xhr.send(fd);
    });
  }

  const uploadWildlifeVideo = (file, onProgress) =>
    _uploadVideo('/api/wildlife/upload-video', file, onProgress);
  const uploadAnomalyVideo  = (file, onProgress) =>
    _uploadVideo('/api/anomaly/upload-video',  file, onProgress);

  // ── Job polling ──────────────────────────────────────────
  const pollJob = (jobId) => get(`/api/job/${jobId}`);

  /**
   * Poll a video job until done/error.
   * onProgress(pct 0-100), onDone(result), onError(msg)
   */
  function watchJob(jobId, { onProgress, onDone, onError } = {}) {
    const interval = setInterval(async () => {
      const job = await pollJob(jobId);
      console.log('[API WATCHJOB] polled job ->', jobId, job);
      try { if (window.wlDebugLog) window.wlDebugLog('[API WATCHJOB] polled job -> ' + jobId + ' ' + JSON.stringify(job)); } catch(_){}
      if (!job) return;
      const uploadPct   = 50;  // upload phase was 0-50%
      const processPct  = Math.round(job.progress / 2) + uploadPct;  // processing = 50-100%
      if (onProgress) onProgress(Math.min(processPct, 99));
        // Mirror job state into on-page job output panel for visibility
        try {
          const out = document.getElementById && document.getElementById('wl-job-output-entries');
          if (out) {
            out.textContent = job && job.result ? JSON.stringify(job.result, null, 2) : JSON.stringify(job, null, 2);
          }
        } catch (_) {}
      if (job.status === 'done') {
        console.log('[API WATCHJOB] job done ->', jobId, job.result);
        try { if (window.wlDebugLog) window.wlDebugLog('[API WATCHJOB] job done -> ' + jobId + ' ' + JSON.stringify(job.result)); } catch(_){}
        clearInterval(interval);
        if (onProgress) onProgress(100);
          if (onDone)     onDone(job.result);
          // Also call page renderer if available (makes the UI update even if caller didn't)
          try { if (window && typeof window.renderWlVideoResult === 'function') window.renderWlVideoResult(job.result); } catch(_) {}
      } else if (job.status === 'error') {
        console.error('[API WATCHJOB] job error ->', jobId, job.result);
        try { if (window.wlDebugLog) window.wlDebugLog('[API WATCHJOB] job error -> ' + jobId + ' ' + JSON.stringify(job.result)); } catch(_){}
        clearInterval(interval);
        if (onError) onError(job.result?.error || 'Processing failed');
      }
    }, 1500);
    return interval; // caller can clearInterval to cancel
  }

  // ── Detection History ────────────────────────────────────
  function getHistory({ days = 7, category = '', label = '', camera = '',
                        source = '', limit = 100, offset = 0 } = {}) {
    const p = new URLSearchParams({ days, limit, offset });
    if (category) p.append('category', category);
    if (label)    p.append('label',    label);
    if (camera)   p.append('camera',   camera);
    if (source)   p.append('source',   source);
    return get(`/api/history?${p}`);
  }

  const getHistoryStats   = () => get('/api/history/stats');
  const getHistoryFilters = (category = '') =>
    get(`/api/history/filters?category=${category}`);

  // ── CSV / PDF downloads (trigger via browser) ────────────
  function downloadCSV(days = 7, category = '') {
    const p = new URLSearchParams({ days });
    if (category) p.append('category', category);
    window.open(`${BASE}/api/history/export-csv?${p}`, '_blank');
  }

  function downloadReport() {
    window.open(`${BASE}/api/report/download`, '_blank');
  }

  // ── Snapshot URL helper ──────────────────────────────────
  const snapshotURL = (path) => path ? `${BASE}/${path}` : null;

  // ── Monitoring On/Off ────────────────────────────────────
  const startWildlife = () => post('/api/start-wildlife', {});
  const stopWildlife  = () => post('/api/stop-wildlife',  {});
  const startAnomaly  = () => post('/api/start-anomaly',  {});
  const stopAnomaly   = () => post('/api/stop-anomaly',   {});

  return {
    BASE,
    VIDEO_WILDLIFE,
    VIDEO_ANOMALY,
    // Live polling
    getStatus,
    getWildlife,
    getAnomaly,
    getWildlifeHistory,
    getAnomalyHistory,
    getAlerts,
    postSettings,
    checkBackend,
    // Monitoring control (on-demand start/stop)
    startWildlife,
    stopWildlife,
    startAnomaly,
    stopAnomaly,
    // Uploads
    uploadWildlifeImage,
    uploadAnomalyImage,
    uploadWildlifeVideo,
    uploadAnomalyVideo,
    // Jobs
    pollJob,
    watchJob,
    // History
    getHistory,
    getHistoryStats,
    getHistoryFilters,
    // Exports
    downloadCSV,
    downloadReport,
    // Utility
    snapshotURL,
  };
})();


// ── Shared helpers ────────────────────────────────────────

/** Show a backend-offline banner if the server is unreachable */
async function checkAndShowBackendStatus() {
  const ok = await API.checkBackend();
  const banner = document.getElementById('backend-banner');
  if (banner) banner.style.display = ok ? 'none' : 'flex';
  return ok;
}

/** Alert sound – plays an 800 Hz beep for 200 ms on critical events */
function playAlertBeep() {
  try {
    const ctx  = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type      = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.4);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.4);
  } catch (_) { /* browser may block before user interaction */ }
}

/** Show a toast notification (requires .toast-container in the page) */
function showToast(title, message, type = 'alert-info') {
  const container = document.getElementById('toast-container') ||
                    document.querySelector('.toast-container');
  if (!container) return;

  if (type === 'alert-critical') playAlertBeep();

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <div class="toast-icon">${type === 'alert-critical' ? '🚨' : type === 'alert-high' ? '⚠️' : 'ℹ️'}</div>
    <div class="toast-body">
      <div class="toast-title">${title}</div>
      <div class="toast-msg">${message}</div>
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>`;
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 400);
  }, 6000);
}

/** Format bytes to human-readable size */
function formatBytes(bytes) {
  if (bytes < 1024)        return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/** Format seconds to MM:SS */
function formatDuration(secs) {
  const m = Math.floor(secs / 60).toString().padStart(2, '0');
  const s = (secs % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}
