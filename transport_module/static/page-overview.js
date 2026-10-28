const BUS_CLASS = { 'ON ROUTE': 'ok', 'COMPLETED': 'done', 'WAITING': 'wait', 'OFF ROUTE': 'bad', 'STOPPED': 'warn', 'NO SIGNAL': 'warn' };

start(() => {
  const o = overallStatus();
  $('#board').className = 'board ' + (o.level === 'normal' ? '' : o.level);
  $('#bTitle').textContent = o.title;
  $('#bText').textContent = App.ppe && !App.ppe.available
    ? 'Workplace log not found yet. ' + (activeTransport().length ? o.text : 'Transport is being tracked.') : o.text;

  const p = App.ppe && App.ppe.available ? App.ppe : null;
  $('#bHour').textContent = p ? p.last_hour : '-';
  $('#fHigh').textContent = p ? (p.today.HIGH_RISK || 0) : '-';
  $('#fViol').textContent = p ? (p.today.VIOLATION || 0) : '-';
  $('#fCaut').textContent = p ? (p.today.CAUTION || 0) : '-';
  $('#fStress').textContent = p ? p.stress_today : '-';

  if (App.state) {
    const v = App.state.vehicles;
    const on = v.filter(x => x.status === 'ON ROUTE').length;
    $('#bTrans').textContent = o.active; $('#fAlert').textContent = o.active;
    $('#bBus').textContent = on; $('#fOn').textContent = on;
    $('#fDone').textContent = v.filter(x => x.trip === 'completed').length;
    $('#fleet').innerHTML = v.map(x => {
      const done = x.stops.filter(s => s.status === 'done' || s.status === 'missed').length;
      return '<div class="bus"><div class="top2"><b>' + esc(x.name) + '</b><span class="pill ' + (BUS_CLASS[x.status] || 'wait') + '">' + esc(x.status.toLowerCase()) + '</span></div>' +
        '<small>' + esc(x.route_name) + '</small><div class="bar"><div style="width:' + Math.round(100 * done / x.stops.length) + '%"></div></div></div>';
    }).join('');
  }

  $('#chart').innerHTML = p ? hourlyChart(p.hourly) : '<div class="empty">The workplace chart appears once <code>events.csv</code> is found.</div>';
  const items = mergedAlerts(8);
  $('#alerts').innerHTML = items.length ? items.join('') : '<div class="empty">No alerts yet. Run <code>ppe_detect.py</code> with the camera, or <code>simulator.py</code> for the buses, and alerts will appear here.</div>';
});
