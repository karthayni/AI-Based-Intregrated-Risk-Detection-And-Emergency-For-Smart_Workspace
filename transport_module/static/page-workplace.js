let filter = 'ALL';

function renderCards() {
  const p = App.ppe;
  if (!p) return;
  if (!p.available) {
    $('#cards').innerHTML = '<div class="empty">The server cannot find the workplace log. Start it with <code>--ppe-log C:\\path\\to\\events.csv</code>, using the file that <code>ppe_detect.py</code> creates in its <code>logs</code> folder.</div>';
    return;
  }
  const rows = p.rows.filter(r => filter === 'ALL' || (filter === 'STRESS' ? r.stress_high : r.status === filter));
  $('#cards').innerHTML = rows.length ? rows.map(r =>
    '<article class="card ' + esc(r.status) + '"><div class="shot">' +
      (r.snapshot ? '<img src="' + esc(r.snapshot) + '" data-big="' + esc(r.snapshot) + '" alt="Snapshot of the alert">' : 'No picture saved') + '</div>' +
    '<div class="body"><div class="row"><span class="pill ' + esc(r.status) + '">' + esc(WORK_LABEL[r.status] || r.status) + '</span>' +
      (r.stress_high ? '<span class="pill stress">Stress high</span>' : '') + '<span class="when">' + esc(ago(r.time)) + ', ' + esc(clock(r.time)) + '</span></div>' +
    '<p>' + esc(describe(r.details)) + '</p></div></article>').join('')
    : '<div class="empty">' + (p.rows.length ? 'No alerts match this filter.' : 'No alerts yet. Run <code>ppe_detect.py</code> with a camera. When someone is missing PPE, or looks stressed, the alert and its picture appear here within a few seconds.') + '</div>';
}

start(() => {
  const p = App.ppe;
  if (!p || !p.available) { renderCards(); return; }
  $('#cur').textContent = WORK_LABEL[p.current] || p.current;
  $('#cur').className = 'big ' + p.current;
  $('#curNote').textContent = p.current === 'SAFE' ? 'No alerts in the last ' + p.window_min + ' minutes.' : 'Worst alert in the last ' + p.window_min + ' minutes.';
  $('#counts').innerHTML = [['HIGH_RISK', 'High risk'], ['VIOLATION', 'PPE violations'], ['CAUTION', 'Caution']].map(([k, l]) =>
    '<div class="fig ' + ({ HIGH_RISK: 'high', VIOLATION: 'viol', CAUTION: 'caut' })[k] + '"><b>' + (p.counts[k] || 0) + '</b><span>' + l + ' (all time)</span></div>').join('');
  $('#chart').innerHTML = hourlyChart(p.hourly);
  renderCards();
});

$$('.chip').forEach(b => b.onclick = () => {
  filter = b.dataset.f;
  $$('.chip').forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
  renderCards();
});
