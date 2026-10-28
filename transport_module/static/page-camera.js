let st = null, feedOn = false;
const feed = $('#feed');
const streamBase = () => 'http://' + (location.hostname || '127.0.0.1') + ':' + (st ? st.port : 5001);

function say(text, bad) { const m = $('#msg'); m.textContent = text || ''; m.className = 'msg' + (bad ? ' bad' : ''); }

async function post(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ('Request failed (' + r.status + ')'));
  return j;
}

function showFeed(on) {
  if (on && !feedOn) { feed.src = streamBase() + '/video?t=' + Date.now(); feed.hidden = false; $('#placeholder').hidden = true; feedOn = true; }
  if (!on && feedOn) { feed.removeAttribute('src'); feed.hidden = true; $('#placeholder').hidden = false; feedOn = false; }
}

async function pollCamera() {
  try { st = await getJSON('/api/camera/status'); } catch (e) { return; }
  $('#start').disabled = st.running || !st.ready;
  $('#stop').disabled = !st.running;
  $('#shot').disabled = !st.running;
  $('#stress').disabled = st.running;
  $('#source').disabled = st.running; $('#file').disabled = st.running;
  if (!st.stress_ok && !st.running) { $('#stress').checked = false; $('#stress').disabled = true; }
  const crashed = !st.running && !st.stopped_by_user && st.exit_code != null && st.exit_code !== 0;
  showFeed(st.running);
  if (st.running && $('#msg').textContent.startsWith('Starting') && feedOn && feed.naturalWidth) say('');
  if (!st.ready) { $('#placeholder').textContent = st.problem; say(st.problem, true); }
  else if (!st.running) {
    $('#placeholder').textContent = crashed ? 'The detection stopped with an error. See the message under the buttons.' : 'The camera is off. Press Start detection.';
    if (crashed) say('Detection stopped: ' + (st.log.slice(-3).join(' ') || 'unknown error'), true);
    else if (!$('#msg').textContent || $('#msg').className.includes('bad')) say(st.stress_ok ? '' : 'Stress files not found in the PPE folder, so only PPE detection is available.');
  }
  if (st.running) {
    try {
      const s = await (await fetch(streamBase() + '/status')).json();
      $('#readout').innerHTML = '<span class="pill ' + esc(s.status) + '">' + esc(WORK_LABEL[s.status] || s.status) + '</span> ' +
        s.people + ' ' + (s.people === 1 ? 'person' : 'people') + ' in view' + (st.stress ? ', ' + s.stress_high + ' with high stress' : '') + ', ' + s.fps + ' frames per second' +
        (st.zone ? '' : '. No danger zone set, so zone checks are off');
    } catch (e) { $('#readout').textContent = 'Starting up. The first frame can take a few seconds while the models load.'; }
  } else $('#readout').textContent = '';
}

$('#source').onchange = () => { $('#fileField').hidden = $('#source').value !== 'file'; };
$('#start').onclick = async () => {
  const src = $('#source').value === 'file' ? $('#file').value.trim() : $('#source').value;
  if (!src) { say('Type the path of the video file first.', true); return; }
  say('Starting. Loading the models can take up to 20 seconds.');
  try { await post('/api/camera/start', { source: src, stress: $('#stress').checked }); } catch (e) { say(e.message, true); }
  pollCamera();
};
$('#stop').onclick = async () => { try { await post('/api/camera/stop'); say('Detection stopped.'); } catch (e) { say(e.message, true); } pollCamera(); };
$('#shot').onclick = async () => {
  try { const j = await (await fetch(streamBase() + '/shot')).json(); say(j.saved ? 'Screenshot saved: ' + j.saved + ' in your PPE folder.' : (j.error || 'Could not save.'), !j.saved); }
  catch (e) { say('Could not save the screenshot.', true); }
};
feed.onerror = () => { if (feedOn) { feedOn = false; feed.hidden = true; $('#placeholder').hidden = false; $('#placeholder').textContent = 'Waiting for the first frame...'; setTimeout(pollCamera, 1500); } };

start(() => {
  const p = App.ppe;
  $('#alerts').innerHTML = p && p.available && p.rows.length ? p.rows.slice(0, 5).map(workItem).join('')
    : '<div class="empty">No alerts yet. When someone is missing PPE or looks stressed, the alert and its picture appear here.</div>';
});
pollCamera(); setInterval(pollCamera, 2000);
