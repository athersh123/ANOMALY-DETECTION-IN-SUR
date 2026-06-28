// ============================================================
//  SENTINELAI – Master JavaScript
//  Shared utilities, charts, animations, live data simulation
// ============================================================

'use strict';

/* ─── Constants & Config ──────────────────────────────────── */
const CONFIG = {
  dataRefresh:     1000,    // ms between API polls (1 second as required)
  chartUpdate:     2000,    // ms between chart updates
  toastDuration:   5000,    // ms toast stays visible
};

const ANIMALS = [
  { name: 'Tiger',    emoji: '🐅', color: '#ff6b35', threat: 'CRITICAL' },
  { name: 'Leopard',  emoji: '🐆', color: '#ff9500', threat: 'HIGH'     },
  { name: 'Lion',     emoji: '🦁', color: '#ffcc00', threat: 'CRITICAL' },
  { name: 'Bear',     emoji: '🐻', color: '#8b4513', threat: 'HIGH'     },
  { name: 'Elephant', emoji: '🐘', color: '#7ab0d4', threat: 'MEDIUM'   },
  { name: 'Cheetah',  emoji: '🐾', color: '#ff8c00', threat: 'HIGH'     },
];

const ANOMALIES = [
  { name: 'Running',             emoji: '🏃', color: '#ffcc00', threat: 'MEDIUM'   },
  { name: 'Fighting',            emoji: '🥊', color: '#ff2d55', threat: 'CRITICAL' },
  { name: 'Crowd Formation',     emoji: '👥', color: '#ff8c00', threat: 'HIGH'     },
  { name: 'Vehicle Intrusion',   emoji: '🚗', color: '#ff2d55', threat: 'CRITICAL' },
  { name: 'Suspicious Activity', emoji: '👤', color: '#ff9500', threat: 'HIGH'     },
];

const CAMERA_LOCATIONS = [
  'North Perimeter', 'East Gate', 'South Forest', 'West Boundary',
  'Village Entry', 'Water Hole', 'Forest Edge', 'Road Crossing',
  'Hill Top', 'River Bank', 'Patrol Route A', 'Patrol Route B',
];

/* ─── State ───────────────────────────────────────────────── */
const state = {
  alerts:       [],
  cameras:      [],
  chartInstances: {},
};

/* ─── DOM Utilities ───────────────────────────────────────── */
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function el(tag, cls = '', html = '') {
  const e = document.createElement(tag);
  if (cls)  e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

/* ─── Clock ───────────────────────────────────────────────── */
function initClock() {
  const clockEls = $$('.topbar-time');
  function tick() {
    const now = new Date();
    const str = now.toLocaleTimeString('en-GB', { hour12: false }) +
                ' | ' + now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    clockEls.forEach(el => el.textContent = str);
  }
  tick();
  setInterval(tick, 1000);
}

/* ─── Active Nav Highlight ────────────────────────────────── */
function initNav() {
  const current = window.location.pathname.split('/').pop() || 'dashboard.html';
  $$('.nav-item').forEach(item => {
    const href = item.getAttribute('href') || '';
    if (href === current || (current === '' && href === 'dashboard.html')) {
      item.classList.add('active');
    }
  });
}

/* ─── Toast Notifications ────────────────────────────────────*/
let toastContainer;

function initToasts() {
  toastContainer = el('div', 'toast-container');
  document.body.appendChild(toastContainer);
}

function showToast(title, message, type = 'alert-info') {
  if (!toastContainer) initToasts();
  const icons = { 'alert-critical': '🚨', 'alert-info': 'ℹ️', 'alert-warn': '⚠️', 'alert-success': '✅' };
  const toast = el('div', `toast ${type}`);
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || '🔔'}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      <div class="toast-message">${message}</div>
    </div>
  `;
  toastContainer.appendChild(toast);
  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add('show'));
  });
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 400);
  }, CONFIG.toastDuration);
}

/* ─── Particles Background ────────────────────────────────── */
function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  for (let i = 0; i < 60; i++) {
    particles.push({
      x:  Math.random() * 1920,
      y:  Math.random() * 1080,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r:  Math.random() * 1.5 + 0.5,
      a:  Math.random() * 0.6 + 0.1,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 212, 255, ${p.a})`;
      ctx.fill();
    });
    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0, 212, 255, ${0.08 * (1 - dist/120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
}

/* ─── Animated Counters ───────────────────────────────────── */
function animateCounter(el, target, duration = 1200, suffix = '') {
  const start = parseInt(el.textContent) || 0;
  const step = (target - start) / (duration / 16);
  let current = start;
  const timer = setInterval(() => {
    current += step;
    if ((step > 0 && current >= target) || (step < 0 && current <= target)) {
      current = target;
      clearInterval(timer);
    }
    el.textContent = Math.round(current) + suffix;
  }, 16);
}

/* ─── Confidence Bar Animation ────────────────────────────── */
function animateConfBars() {
  $$('.conf-bar-fill').forEach(bar => {
    const target = bar.getAttribute('data-width') || '0';
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = target + '%'; }, 200);
  });
}

/* ─── Chart: Line Chart ───────────────────────────────────── */
function createLineChart(canvasId, labels, datasets, opts = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  if (state.chartInstances[canvasId]) {
    state.chartInstances[canvasId].destroy();
  }
  const chart = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 800, easing: 'easeInOutQuart' },
      plugins: {
        legend: {
          labels: { color: '#7ab0d4', font: { family: 'Rajdhani', size: 12 } }
        },
        tooltip: {
          backgroundColor: 'rgba(4, 14, 30, 0.95)',
          borderColor: 'rgba(0, 212, 255, 0.3)',
          borderWidth: 1,
          titleColor: '#00d4ff',
          bodyColor: '#7ab0d4',
        }
      },
      scales: {
        x: {
          ticks: { color: '#3d6b8f', font: { size: 11 } },
          grid:  { color: 'rgba(0, 212, 255, 0.04)', borderColor: 'rgba(0, 212, 255, 0.1)' },
        },
        y: {
          ticks: { color: '#3d6b8f', font: { size: 11 } },
          grid:  { color: 'rgba(0, 212, 255, 0.04)', borderColor: 'rgba(0, 212, 255, 0.1)' },
          ...( opts.yMin !== undefined ? { min: opts.yMin } : {} ),
          ...( opts.yMax !== undefined ? { max: opts.yMax } : {} ),
        },
      },
      elements: {
        point: { radius: 4, hoverRadius: 6 },
        line:  { borderWidth: 2 },
      },
      ...opts,
    }
  });
  state.chartInstances[canvasId] = chart;
  return chart;
}

/* ─── Chart: Bar Chart ────────────────────────────────────── */
function createBarChart(canvasId, labels, datasets, opts = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  if (state.chartInstances[canvasId]) state.chartInstances[canvasId].destroy();
  const chart = new Chart(canvas, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 800 },
      plugins: {
        legend: { labels: { color: '#7ab0d4', font: { family: 'Rajdhani', size: 12 } } },
        tooltip: {
          backgroundColor: 'rgba(4,14,30,0.95)',
          borderColor: 'rgba(0,212,255,0.3)',
          borderWidth: 1,
          titleColor: '#00d4ff',
          bodyColor: '#7ab0d4',
        }
      },
      scales: {
        x: { ticks: { color: '#3d6b8f' }, grid: { color: 'rgba(0,212,255,0.04)' } },
        y: { ticks: { color: '#3d6b8f' }, grid: { color: 'rgba(0,212,255,0.04)' } },
      },
      borderRadius: 4,
      ...opts,
    }
  });
  state.chartInstances[canvasId] = chart;
  return chart;
}

/* ─── Chart: Doughnut ─────────────────────────────────────── */
function createDoughnutChart(canvasId, labels, data, colors) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  if (state.chartInstances[canvasId]) state.chartInstances[canvasId].destroy();
  const chart = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors.map(c => c + '33'),
        borderColor: colors,
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: '70%',
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#7ab0d4', font: { family: 'Rajdhani', size: 12 }, padding: 16 }
        },
        tooltip: {
          backgroundColor: 'rgba(4,14,30,0.95)',
          borderColor: 'rgba(0,212,255,0.3)',
          borderWidth: 1,
          titleColor: '#00d4ff',
          bodyColor: '#7ab0d4',
        }
      }
    }
  });
  state.chartInstances[canvasId] = chart;
  return chart;
}

/* ─── Radar Chart ─────────────────────────────────────────── */
function createRadarChart(canvasId, labels, datasets) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  if (state.chartInstances[canvasId]) state.chartInstances[canvasId].destroy();
  const chart = new Chart(canvas, {
    type: 'radar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { labels: { color: '#7ab0d4', font: { family: 'Rajdhani', size: 12 } } },
        tooltip: {
          backgroundColor: 'rgba(4,14,30,0.95)',
          borderColor: 'rgba(0,212,255,0.3)',
          borderWidth: 1,
          titleColor: '#00d4ff',
          bodyColor: '#7ab0d4',
        }
      },
      scales: {
        r: {
          ticks: { color: '#3d6b8f', backdropColor: 'transparent' },
          grid:  { color: 'rgba(0,212,255,0.08)' },
          pointLabels: { color: '#7ab0d4', font: { size: 11 } },
          angleLines: { color: 'rgba(0,212,255,0.08)' },
        }
      }
    }
  });
  state.chartInstances[canvasId] = chart;
  return chart;
}

/* ─── Simulated Live Feed Timestamp ──────────────────────────*/
function initFeedTimestamps() {
  const tss = $$('.feed-timestamp');
  function update() {
    const now = new Date();
    const str = now.toLocaleString('en-GB', { hour12: false }).replace(',', '');
    tss.forEach(el => el.textContent = str);
  }
  update();
  setInterval(update, 1000);
}

/* ─── Alert Rendering ────────────────────────────────────────*/
const levelIcons = {
  critical: { bg: 'rgba(255,45,85,0.15)',  border: 'rgba(255,45,85,0.3)',  icon: '🚨' },
  high:     { bg: 'rgba(255,140,0,0.15)',  border: 'rgba(255,140,0,0.3)',  icon: '⚠️' },
  medium:   { bg: 'rgba(255,204,0,0.15)',  border: 'rgba(255,204,0,0.3)',  icon: '🔶' },
  low:      { bg: 'rgba(0,255,136,0.12)',  border: 'rgba(0,255,136,0.2)',  icon: 'ℹ️' },
};

function renderAlertItem(alert) {
  const info = levelIcons[alert.level] || levelIcons.medium;
  const div = el('div', `alert-item ${alert.level}`);
  div.innerHTML = `
    <div class="alert-icon-wrap" style="background:${info.bg}; border:1px solid ${info.border}">
      ${info.icon}
    </div>
    <div class="alert-content">
      <div class="alert-title">${alert.title}</div>
      <div class="alert-meta"><span>${alert.loc}</span></div>
    </div>
    <div class="alert-time">${alert.time}</div>
  `;
  return div;
}

/* ─── Sidebar Toggle (Mobile) ────────────────────────────────*/
function initSidebarToggle() {
  const btn = $('#sidebar-toggle');
  const sidebar = $('.sidebar');
  if (!btn || !sidebar) return;
  btn.addEventListener('click', () => sidebar.classList.toggle('open'));
  document.addEventListener('click', e => {
    if (sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target !== btn) {
      sidebar.classList.remove('open');
    }
  });
}

/* ─── Confidence Ring SVG ─────────────────────────────────── */
function buildRingSVG(pct, color) {
  const r = 64, cx = 80, cy = 80;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  return `
    <svg viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="10"/>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none"
              stroke="${color}" stroke-width="10"
              stroke-linecap="round"
              stroke-dasharray="${dash} ${circ - dash}"
              style="filter:drop-shadow(0 0 8px ${color})"/>
    </svg>
  `;
}

/* ─── Generate Heatmap on Canvas ─────────────────────────────*/
function drawHeatmap(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#030910';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(0,212,255,0.05)';
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 30) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y += 30) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

  // Hotspots
  const spots = [
    { x: W*0.2, y: H*0.3, r: 80, color: [255, 45, 85], intensity: 0.8 },
    { x: W*0.6, y: H*0.5, r: 60, color: [255, 140, 0], intensity: 0.6 },
    { x: W*0.8, y: H*0.2, r: 50, color: [255, 45, 85], intensity: 0.7 },
    { x: W*0.3, y: H*0.7, r: 70, color: [255, 204, 0], intensity: 0.5 },
    { x: W*0.7, y: H*0.75,r: 40, color: [0, 255, 136], intensity: 0.4 },
    { x: W*0.5, y: H*0.2, r: 45, color: [255, 140, 0], intensity: 0.5 },
  ];

  spots.forEach(s => {
    const gradient = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r);
    gradient.addColorStop(0,   `rgba(${s.color.join(',')}, ${s.intensity})`);
    gradient.addColorStop(0.5, `rgba(${s.color.join(',')}, ${s.intensity * 0.4})`);
    gradient.addColorStop(1,   `rgba(${s.color.join(',')}, 0)`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, W, H);
  });

  // Location dots
  spots.forEach(s => {
    ctx.beginPath();
    ctx.arc(s.x, s.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = `rgb(${s.color.join(',')})`;
    ctx.fill();
    ctx.shadowColor = `rgb(${s.color.join(',')})`;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  });
}

/* ─── Reconstuction Error Simulation ─────────────────────── */
function generateErrorData(n = 60, anomalyPct = 0.15) {
  const labels = [], data = [], colors = [], borderColors = [];
  const now = Date.now();
  for (let i = n - 1; i >= 0; i--) {
    const t = new Date(now - i * 3000);
    labels.push(t.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    const isAnomaly = Math.random() < anomalyPct;
    const val = isAnomaly
      ? 0.6 + Math.random() * 0.4
      : 0.05 + Math.random() * 0.15;
    data.push(parseFloat(val.toFixed(3)));
    colors.push(isAnomaly ? 'rgba(255,45,85,0.4)' : 'rgba(0,255,136,0.4)');
    borderColors.push(isAnomaly ? '#ff2d55' : '#00ff88');
  }
  return { labels, data, colors, borderColors };
}

/* ─── Utility: Random int in range ───────────────────────── */
function randInt(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }
function randFloat(min, max, dec = 2) { return parseFloat((Math.random() * (max - min) + min).toFixed(dec)); }
function randItem(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

/* ─── Initialize on DOM ready ────────────────────────────────*/
document.addEventListener('DOMContentLoaded', () => {
  // Defensive: remove any third-party injected "enable_copy" script (browser extension/user script)
  // Some browsers/extensions inject enable_copy.js which logs "E.C.P is not enabled" and
  // can clutter console or interfere with app behavior. Remove script tags whose src contains
  // 'enable_copy' and clear a possible global to silence it.
  try {
    document.querySelectorAll('script[src*="enable_copy"]').forEach(s => s.remove());
    if (window.hasOwnProperty('enable_copy')) {
      try { delete window.enable_copy; } catch (_) { window.enable_copy = undefined; }
    }
  } catch (_) {}
  // Some extensions/userscripts inject after DOMContentLoaded; run a short-lived remover
  try {
    const _ecRem = setInterval(() => {
      try {
        document.querySelectorAll('script[src*="enable_copy"]').forEach(s => s.remove());
        if (window.hasOwnProperty('enable_copy')) {
          try { delete window.enable_copy; } catch (_) { window.enable_copy = undefined; }
        }
      } catch (_) {}
    }, 500);
    setTimeout(() => clearInterval(_ecRem), 10000); // run for 10s
  } catch (_) {}

  // Filter noisy E.C.P console messages as an extra temporary safeguard
  try {
    ['log','info','warn','error'].forEach(level => {
      const orig = console[level].bind(console);
      console[level] = (...args) => {
        try {
          if (args.some(a => typeof a === 'string' && /E\.C\.P|enable_copy/i.test(a))) return;
        } catch (_) {}
        orig(...args);
      };
    });
  } catch (_) {}
  initClock();
  initNav();
  initToasts();
  initParticles();
  initFeedTimestamps();
  initSidebarToggle();
  animateConfBars();

  // Animate stat counters
  $$('[data-count]').forEach(el => {
    const target = parseInt(el.getAttribute('data-count'));
    animateCounter(el, target);
  });

  console.log('%c🛡️ SENTINELAI INITIALIZED', 'color:#00d4ff; font-size:16px; font-weight:bold; font-family:monospace;');
  // Navigation guard: prevent navigation while an upload result panel is explicitly open
  document.addEventListener('click', (e) => {
    try {
      if (!window.__uploadResultOpen) return;
      let el = e.target;
      while (el && el !== document) {
        if (el.tagName === 'A' && el.getAttribute('href')) {
          e.preventDefault(); e.stopPropagation();
          showToast('Action blocked', 'Close the upload result before navigating', 'alert-warn');
          return;
        }
        const oc = el.getAttribute && el.getAttribute('onclick');
        if (oc && /location\.|window\.location|location\.href|location\.reload/.test(oc)) {
          e.preventDefault(); e.stopPropagation();
          showToast('Action blocked', 'Close the upload result before navigating', 'alert-warn');
          return;
        }
        el = el.parentElement;
      }
    } catch (_) {}
  }, true);

  // Programmatic navigation guard: block location.assign/replace and history state changes
  try {
    const _origAssign = window.location.assign.bind(window.location);
    const _origReplace = window.location.replace.bind(window.location);
    const _origPush = history.pushState.bind(history);
    const _origReplaceState = history.replaceState.bind(history);

    window.location.assign = function (url) {
      if (window.__uploadResultOpen) {
        showToast('Action blocked', 'Close the upload result before navigating', 'alert-warn');
        return;
      }
      return _origAssign(url);
    };

    window.location.replace = function (url) {
      if (window.__uploadResultOpen) {
        showToast('Action blocked', 'Close the upload result before navigating', 'alert-warn');
        return;
      }
      return _origReplace(url);
    };

    history.pushState = function (state, title, url) {
      if (window.__uploadResultOpen) {
        showToast('Action blocked', 'Close the upload result before navigating', 'alert-warn');
        return;
      }
      return _origPush(state, title, url);
    };

    history.replaceState = function (state, title, url) {
      if (window.__uploadResultOpen) {
        showToast('Action blocked', 'Close the upload result before navigating', 'alert-warn');
        return;
      }
      return _origReplaceState(state, title, url);
    };
  } catch (_) {}

  // Instead of using beforeunload (which forces the browser confirmation dialog),
  // override location.reload to silently block reloads while a result panel is open.
  try {
    const _origReload = window.location.reload.bind(window.location);
    window.location.reload = function () {
      if (window.__uploadResultOpen) {
        showToast('Action blocked', 'Close the upload result before reloading', 'alert-warn');
        return;
      }
      return _origReload();
    };
  } catch (_) {}

  // Best-effort: intercept direct assignments to window.location.href
  try {
    const loc = window.location;
    const hrefDesc = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(loc), 'href') || Object.getOwnPropertyDescriptor(Location.prototype, 'href');
    if (hrefDesc && hrefDesc.set) {
      const origHrefSet = hrefDesc.set.bind(loc);
      Object.defineProperty(loc, 'href', {
        configurable: true,
        enumerable: true,
        get: hrefDesc.get ? hrefDesc.get.bind(loc) : function () { return String(loc); },
        set: function (v) {
          if (window.__uploadResultOpen) {
            showToast('Action blocked', 'Close the upload result before navigating', 'alert-warn');
            return;
          }
          return origHrefSet(v);
        }
      });
    }
  } catch (_) {}
});