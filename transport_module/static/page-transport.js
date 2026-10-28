const BUS_CLASS = { 'ON ROUTE': 'ok', 'COMPLETED': 'done', 'WAITING': 'wait', 'OFF ROUTE': 'bad', 'STOPPED': 'warn', 'NO SIGNAL': 'warn' };
const STOP_COLOR = { pending: '#7d869b', arrived: '#a85f00', done: '#1f8a4c', missed: '#b52a2a' };
const layers = {};
let selected = null, firstFit = false;

const map = L.map('map').setView([12.97, 77.6], 11);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }).addTo(map);
map.on('click', e => L.popup().setLatLng(e.latlng).setContent('<b>Coordinates</b><br><code>' + e.latlng.lat.toFixed(5) + ', ' + e.latlng.lng.toFixed(5) + '</code>').openOn(map));

function select(id, fit) {
  selected = id; styleLayers(); renderVehicles(); renderStops();
  if (fit && layers[id]) map.fitBounds(layers[id].route.getBounds(), { padding: [30, 30] });
}
function styleLayers() {
  for (const id in layers) {
    const on = id === selected;
    layers[id].route.setStyle({ color: on ? '#213163' : '#9aa5bf', weight: on ? 5 : 3, opacity: on ? .85 : .6, dashArray: on ? null : '6 8' });
  }
}
function ensureLayers(v) {
  if (layers[v.id]) return;
  const route = L.polyline(v.points, { color: '#9aa5bf', weight: 3 }).addTo(map);
  const trail = L.polyline([], { color: '#1f8a4c', weight: 4 }).addTo(map);
  const stops = v.stops.map(s => L.circleMarker([s.lat, s.lon], { radius: 8, color: '#fff', weight: 2, fillOpacity: 1, fillColor: STOP_COLOR.pending }).bindTooltip(esc(s.name)).addTo(map));
  layers[v.id] = { route, trail, stops, marker: null };
}
function updateMap(s) {
  for (const v of s.vehicles) {
    ensureLayers(v);
    const l = layers[v.id];
    l.trail.setLatLngs(v.trail);
    l.trail.setStyle({ color: v.active_alerts.length ? '#b52a2a' : '#1f8a4c' });
    v.stops.forEach((st, i) => { l.stops[i].setStyle({ fillColor: STOP_COLOR[st.status] }); l.stops[i].setTooltipContent(esc(st.name) + ' (' + st.status + ')'); });
    if (v.position) {
      const icon = L.divIcon({ className: '', html: '<div class="veh-pin ' + (BUS_CLASS[v.status] || 'wait') + '">' + esc(v.id) + '</div>', iconSize: [66, 24], iconAnchor: [33, 12] });
      if (!l.marker) { l.marker = L.marker(v.position, { icon, zIndexOffset: 1000 }).addTo(map); l.marker.on('click', () => select(v.id, false)); }
      else { l.marker.setLatLng(v.position); l.marker.setIcon(icon); }
    }
  }
  if (!firstFit && s.vehicles.length) { map.fitBounds(L.featureGroup(Object.values(layers).map(l => l.route)).getBounds(), { padding: [30, 30] }); firstFit = true; }
  const sel = s.vehicles.find(v => v.id === selected);
  if (sel && sel.position && $('#follow').checked) map.panTo(sel.position);
}
function renderVehicles() {
  $('#vehicles').innerHTML = App.state.vehicles.map(v => {
    const done = v.stops.filter(s => s.status === 'done' || s.status === 'missed').length;
    return '<div class="bus ' + (v.id === selected ? 'sel' : '') + '" data-id="' + esc(v.id) + '" tabindex="0"><div class="top2"><b>' + esc(v.name) + '</b><span class="pill ' + (BUS_CLASS[v.status] || 'wait') + '">' + esc(v.status.toLowerCase()) + '</span></div>' +
      '<small>' + esc(v.id) + (v.driver ? ', driver ' + esc(v.driver) : '') + '</small><small>' + esc(v.route_name) + '</small>' +
      '<small>' + (v.speed_kmh == null ? 'speed unknown' : v.speed_kmh + ' km/h') + '; ' + (v.off_route_m == null ? 'distance from route unknown' : v.off_route_m + ' m from route') + '; ' +
      (v.seconds_since_update == null ? 'no data yet' : 'updated ' + v.seconds_since_update + ' s ago') + '</small>' +
      '<div class="bar"><div style="width:' + Math.round(100 * done / v.stops.length) + '%"></div></div></div>';
  }).join('');
  $$('#vehicles .bus').forEach(el => { el.onclick = () => select(el.dataset.id, true); el.onkeydown = e => { if (e.key === 'Enter') select(el.dataset.id, true); }; });
}
function renderStops() {
  const v = App.state.vehicles.find(x => x.id === selected);
  if (!v) { $('#stops').innerHTML = '<div class="empty">Select a bus to see its stops.</div>'; return; }
  $('#stopsTitle').textContent = v.name + ': stops';
  $('#stops').innerHTML = '<table class="stops"><tr><th>Stop</th><th>Status</th><th>Time</th></tr>' + v.stops.map(s =>
    '<tr><td>' + esc(s.name) + '<small>' + esc(s.kind) + '</small></td><td class="st-' + s.status + '">' + esc(s.status) + '</td><td>' + (s.arrived_at ? esc(clock(s.arrived_at * 1000)) : '-') + '</td></tr>').join('') + '</table>';
}
function renderAlerts() {
  const rows = App.state.alerts;
  $('#alerts').innerHTML = rows.length ? rows.map(transportItem).join('') : '<div class="empty">No transport alerts. Every bus is following its plan.</div>';
}

start(() => {
  if (!App.state) return;
  if (!selected && App.state.vehicles.length) selected = App.state.vehicles[0].id;
  updateMap(App.state); styleLayers(); renderVehicles(); renderStops(); renderAlerts();
});
