/* Shared helpers for every page: data loading, wording, overall status, alert list, picture viewer. */
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const App = { state: null, ppe: null, render: null };

const WORK_LABEL = { HIGH_RISK: 'High risk', VIOLATION: 'PPE violation', CAUTION: 'Caution (stress)', SAFE: 'Safe' };
const WORK_COLOR = { HIGH_RISK: '#b52a2a', VIOLATION: '#a85f00', CAUTION: '#2a6fb0' };
const TRANSPORT_LABEL = { ROUTE_DEVIATION: 'Route deviation', UNEXPECTED_STOP: 'Unexpected stop', SIGNAL_LOST: 'Signal lost', STOP_MISSED: 'Stop missed' };
const RANK = { SAFE: 0, CAUTION: 1, VIOLATION: 2, HIGH_RISK: 3 };

function clock(d) {
  const x = d instanceof Date ? d : new Date(d);
  return isNaN(x) ? '' : x.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}
function ago(d) {
  const sec = Math.max(0, (Date.now() - (d instanceof Date ? d : new Date(d)).getTime()) / 1000);
  if (isNaN(sec)) return '';
  if (sec < 60) return 'just now';
  if (sec < 3600) return Math.round(sec / 60) + ' min ago';
  if (sec < 86400) return Math.round(sec / 3600) + ' h ago';
  return new Date(Date.now() - sec * 1000).toLocaleDateString([], { day: 'numeric', month: 'short' });
}

/* "missing helmet,vest (in_zone=True) stress=HIGH; missing none (in_zone=False)" -> plain sentences */
function describe(details) {
  const parts = String(details || '').split(';').map(s => s.trim()).filter(Boolean);
  const sentences = parts.map(seg => {
    const m = seg.match(/missing (.*?) \(in_zone=(True|False)\)(?: stress=(HIGH|LOW))?/);
    if (!m) return seg;
    const miss = m[1] === 'none' ? [] : m[1].split(',').map(x => x.trim()).filter(Boolean);
    const out = [];
    out.push(miss.length ? 'Missing ' + (miss.length > 1 ? miss.slice(0, -1).join(', ') + ' and ' + miss[miss.length - 1] : miss[0]) : 'PPE complete');
    if (m[2] === 'True') out.push('in the danger zone');
    if (m[3]) out.push('stress ' + m[3].toLowerCase());
    return out.join(', ');
  });
  return sentences.length > 1 ? sentences.map((t, i) => 'Person ' + (i + 1) + ': ' + t).join('. ') : (sentences[0] || '');
}

function workItem(x) {
  return '<div class="item">' + (x.snapshot ? '<img class="pic" src="' + esc(x.snapshot) + '" data-big="' + esc(x.snapshot) + '" alt="Snapshot of the alert">' : '<span class="tag ' + esc(x.status) + '"></span>') +
    '<div class="txt"><div class="head"><span class="pill source">Workplace</span><b>' + esc(WORK_LABEL[x.status] || x.status) + '</b>' +
    (x.stress_high ? '<span class="pill stress">Stress high</span>' : '') + '<span class="when">' + esc(ago(x.time)) + ' (' + esc(clock(x.time)) + ')</span></div>' +
    '<div class="detail">' + esc(describe(x.details)) + '</div></div></div>';
}
function transportItem(a) {
  const active = a.resolved == null;
  return '<div class="item" style="' + (active ? '' : 'opacity:.6') + '"><span class="tag ' + esc(a.severity) + '"></span><div class="txt"><div class="head"><span class="pill source">Transport</span><b>' +
    esc(TRANSPORT_LABEL[a.type] || a.type) + '</b><span class="pill ' + (active ? esc(a.severity) : 'wait') + '">' + (active ? 'Active' : 'Resolved') + '</span>' +
    '<span class="when">' + esc(ago(a.created * 1000)) + ' (' + esc(clock(a.created * 1000)) + ')</span></div><div class="detail">' + esc(a.vehicle_id) + ': ' + esc(a.message) + '</div></div></div>';
}
function mergedAlerts(limit) {
  const items = [];
  if (App.ppe && App.ppe.available) for (const x of App.ppe.rows) items.push({ t: Date.parse(x.time) || 0, html: workItem(x) });
  if (App.state) for (const a of App.state.alerts) items.push({ t: a.created * 1000, html: transportItem(a) });
  items.sort((p, q) => q.t - p.t);
  return items.slice(0, limit).map(i => i.html);
}

function activeTransport() {
  return App.state ? App.state.alerts.filter(a => a.resolved == null) : [];
}
function overallStatus() {
  const cur = App.ppe && App.ppe.available ? App.ppe.current : 'SAFE';
  const act = activeTransport();
  const mins = App.ppe ? App.ppe.window_min : 5;
  let level = 'normal', title = 'All clear';
  if (cur === 'HIGH_RISK' || act.some(a => a.severity === 'high')) { level = 'high'; title = 'High risk'; }
  else if (cur !== 'SAFE' || act.length) { level = 'elevated'; title = 'Needs attention'; }
  const w = cur === 'SAFE' ? 'No workplace alerts in the last ' + mins + ' minutes.' : (WORK_LABEL[cur] + ' in the workplace in the last ' + mins + ' minutes.');
  const t = act.length ? act.length + ' active transport alert' + (act.length > 1 ? 's' : '') + '.' : 'All buses are following their plan.';
  return { level, title, text: w + ' ' + t, cur, active: act.length };
}

/* stacked bars: alerts per hour, last 12 hours */
function hourlyChart(hourly) {
  const W = 520, H = 190, L = 28, B = 24, T = 8, n = hourly.length || 12;
  const tot = h => h.CAUTION + h.VIOLATION + h.HIGH_RISK;
  const mx = Math.max(4, ...hourly.map(tot));
  const bw = (W - L) / n, plotH = H - B - T;
  let g = '';
  for (let k = 0; k <= 2; k++) {
    const v = Math.round(mx * k / 2), y = T + plotH - plotH * k / 2;
    g += '<line x1="' + L + '" x2="' + W + '" y1="' + y + '" y2="' + y + '" stroke="#e4e8f1"/><text x="' + (L - 6) + '" y="' + (y + 4) + '" text-anchor="end" font-size="11" fill="#5a647d">' + v + '</text>';
  }
  hourly.forEach((h, i) => {
    let y = T + plotH;
    for (const k of ['CAUTION', 'VIOLATION', 'HIGH_RISK']) {
      const hh = plotH * h[k] / mx; if (!hh) continue;
      y -= hh; g += '<rect x="' + (L + i * bw + bw * .18) + '" y="' + y + '" width="' + (bw * .64) + '" height="' + hh + '" fill="' + WORK_COLOR[k] + '"><title>' + esc(h.hour + ': ' + h[k] + ' ' + WORK_LABEL[k]) + '</title></rect>';
    }
    if (i % 2 === 0 || n <= 6) g += '<text x="' + (L + i * bw + bw / 2) + '" y="' + (H - 6) + '" text-anchor="middle" font-size="11" fill="#5a647d">' + esc(h.hour) + '</text>';
  });
  const legend = ['HIGH_RISK', 'VIOLATION', 'CAUTION'].map(k => '<span><i style="background:' + WORK_COLOR[k] + '"></i>' + WORK_LABEL[k] + '</span>').join('');
  return '<div class="chart"><svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Alerts per hour for the last 12 hours">' + g + '</svg><div class="legend">' + legend + '</div></div>';
}

/* ---------- loading ---------- */
async function getJSON(url) {
  const r = await fetch(url); if (!r.ok) throw new Error(r.status); return r.json();
}
async function refreshState() {
  try {
    App.state = await getJSON('/api/state');
    $('#banner').style.display = 'none'; $('#dot').classList.remove('off'); $('#liveText').textContent = 'Live';
  } catch (e) {
    $('#banner').style.display = 'block'; $('#dot').classList.add('off'); $('#liveText').textContent = 'Offline';
  }
  if (App.render) App.render();
}
async function refreshPpe() {
  try { App.ppe = await getJSON('/api/ppe' + (window.PPE_QUERY || '')); } catch (e) { return; }
  if (App.render) App.render();
}
function start(render) {
  App.render = () => { render(); sideSummary(); };
  refreshState(); refreshPpe();
  setInterval(refreshState, 2000); setInterval(refreshPpe, 3000);
}

document.addEventListener('click', e => {
  const big = e.target.dataset && e.target.dataset.big;
  if (big) { $('#lightbox img').src = big; $('#lightbox').style.display = 'flex'; }
  else if (e.target.closest('#lightbox')) $('#lightbox').style.display = 'none';
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') $('#lightbox').style.display = 'none'; });

/* dashboard side menu: small live summary under the links */
function sideSummary() {
  const el = $('#sideStatus'); if (!el) return;
  const o = overallStatus();
  el.innerHTML = '<span class="pill ' + (o.level === 'high' ? 'HIGH_RISK' : o.level === 'elevated' ? 'VIOLATION' : 'SAFE') + '">' + esc(o.title) + '</span>';
}

/* full screen button (also works with the F11 key of the browser) */
(function () {
  const btn = document.getElementById('fs'); if (!btn) return;
  const label = () => { btn.textContent = document.fullscreenElement ? 'Exit full screen' : 'Full screen'; };
  btn.onclick = () => {
    if (!document.fullscreenEnabled) { alert('Your browser does not allow full screen here. Press F11 instead.'); return; }
    if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen();
  };
  document.addEventListener('fullscreenchange', label);
})();

/* close (exit full screen) button: shown in the top-right corner only while in full screen */
(function () {
  const x = document.createElement('button');
  x.type = 'button'; x.id = 'fsclose'; x.textContent = '✕'; x.title = 'Exit full screen (Esc)';
  x.setAttribute('aria-label', 'Exit full screen');
  x.onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); };
  document.body.appendChild(x);
  const sync = () => { x.style.display = document.fullscreenElement ? 'flex' : 'none'; };
  document.addEventListener('fullscreenchange', sync); sync();
})();
