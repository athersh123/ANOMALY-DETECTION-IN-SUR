// nav.js – Injects shared sidebar navigation into every page
(function () {
  const NAV_HTML = `
    <canvas id="particles-canvas"></canvas>
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-logo">
        <div class="logo-icon">🛡️</div>
        <div class="logo-text">
          <div class="logo-title">SENTINEL<span style="color:#fff;">AI</span></div>
          <div class="logo-sub">SOC Platform v2.4</div>
        </div>
      </div>

      <nav class="sidebar-nav">
        <div class="nav-section-label">Main</div>
        <a class="nav-item" href="dashboard.html">
          <span class="nav-icon">🏠</span> Dashboard
        </a>
        <a class="nav-item" href="wildlife.html">
          <span class="nav-icon">🐾</span> Wildlife Surveillance
          <span class="nav-badge" id="nav-wildlife-badge">0</span>
        </a>
        <a class="nav-item" href="anomaly.html">
          <span class="nav-icon">🔍</span> Human Anomaly
          <span class="nav-badge" id="nav-anomaly-badge">0</span>
        </a>

        <div class="nav-section-label" style="margin-top:8px;">Operations</div>
        <a class="nav-item" href="alerts.html">
          <span class="nav-icon">🚨</span> Alert Center
          <span class="nav-badge" id="alert-nav-badge">0</span>
        </a>
        <a class="nav-item" href="analytics.html">
          <span class="nav-icon">📊</span> Analytics
        </a>
        <a class="nav-item" href="history.html">
          <span class="nav-icon">📋</span> Incident History
        </a>

        <div class="nav-section-label" style="margin-top:8px;">System</div>
        <a class="nav-item" href="cameras.html">
          <span class="nav-icon">📷</span> Camera Management
        </a>
        <a class="nav-item" href="settings.html">
          <span class="nav-icon">⚙️</span> Settings
        </a>
        <a class="nav-item" href="index.html">
          <span class="nav-icon">🏠</span> Home
        </a>
      </nav>

      <div class="sidebar-footer">
        <div class="system-status">
          <div class="status-dot" id="sys-status-dot"></div>
          <div class="status-text">
            <strong id="sys-status-label">ALL SYSTEMS ONLINE</strong>
            <span>AI Models Active</span>
          </div>
        </div>
      </div>
    </aside>
  `;

  const TOPBAR_HTML = `
    <header class="topbar">
      <button class="icon-btn" id="sidebar-toggle" style="display:none;">☰</button>
      <div>
        <div class="topbar-title" id="topbar-page-title">DASHBOARD</div>
        <div class="topbar-breadcrumb">SentinelAI <span>›</span> <span id="topbar-breadcrumb-page">Overview</span></div>
      </div>
      <div class="topbar-spacer"></div>
      <div class="topbar-time" id="topbar-clock">Loading…</div>
      <div class="topbar-actions">
        <button class="icon-btn" title="Notifications" id="notif-btn">
          🔔
          <span class="badge" id="notif-badge">5</span>
        </button>
        <button class="icon-btn" title="Search">🔍</button>
        <button class="icon-btn" title="Fullscreen" id="fullscreen-btn">⛶</button>
        <div class="user-avatar" title="Admin User">A</div>
      </div>
    </header>
  `;

  // Inject shell
  const appShell = document.getElementById('app-shell');
  if (!appShell) return;

  const mainContent = appShell.querySelector('.main-content');
  const pageContent = mainContent ? mainContent.innerHTML : '';
  appShell.innerHTML = NAV_HTML + `<div class="main-content">${TOPBAR_HTML}<div class="page-content" id="page-content">${pageContent}</div></div>`;

  // Set active nav
  const current = window.location.pathname.split('/').pop() || 'dashboard.html';
  document.querySelectorAll('.nav-item').forEach(item => {
    if (item.getAttribute('href') === current) item.classList.add('active');
  });

  // Page title mapping
  const titles = {
    'dashboard.html': ['DASHBOARD', 'Overview'],
    'wildlife.html':  ['WILDLIFE SURVEILLANCE', 'Wildlife Detection'],
    'anomaly.html':   ['HUMAN ANOMALY', 'Anomaly Detection'],
    'alerts.html':    ['ALERT CENTER', 'Active Alerts'],
    'analytics.html': ['ANALYTICS', 'Performance Metrics'],
    'history.html':   ['INCIDENT HISTORY', 'Past Incidents'],
    'cameras.html':   ['CAMERA MANAGEMENT', 'CCTV Network'],
    'settings.html':  ['SETTINGS', 'Configuration'],
  };
  const info = titles[current] || ['SENTINELAI', 'Overview'];
  const titleEl = document.getElementById('topbar-page-title');
  const bcEl    = document.getElementById('topbar-breadcrumb-page');
  if (titleEl) titleEl.textContent = info[0];
  if (bcEl)    bcEl.textContent    = info[1];

  // Fullscreen
  const fsBtn = document.getElementById('fullscreen-btn');
  if (fsBtn) fsBtn.addEventListener('click', () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
    else document.exitFullscreen();
  });

  // Alert badge – updated by live API, not by fake events
  // Poll /api/status every 5s and update sidebar badges
  (async function pollNavBadges() {
    try {
      const r = await fetch('http://127.0.0.1:5000/api/status', { signal: AbortSignal.timeout(3000) });
      if (!r.ok) return;
      const d = await r.json();
      const total = (d.active_alerts || 0);
      const wl    = (d.wildlife_alerts  || 0);
      const hu    = (d.human_anomalies  || 0);
      ['notif-badge','alert-nav-badge'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = total;
      });
      const wlBadge = document.getElementById('nav-wildlife-badge');
      if (wlBadge) wlBadge.textContent = wl;
      const anBadge = document.getElementById('nav-anomaly-badge');
      if (anBadge) anBadge.textContent = hu;
    } catch (_) {}
    setTimeout(pollNavBadges, 5000);
  })();
})();
