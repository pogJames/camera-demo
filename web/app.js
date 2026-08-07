function render(s) {
  const stepsEl = document.getElementById('steps');
  stepsEl.innerHTML = '';
  s.steps.forEach((st, i) => {
    const li = document.createElement('li');
    li.className = 'step ' + st.state;
    const proof = st.state === 'done'
      ? '<a class="proof" href="/log/' + i + '" target="_blank">clip</a>'
      : '';
    li.innerHTML = '<span class="dot"></span><span class="name">' +
      st.title + '</span>' + proof + '<span class="tick">&#10003;</span>';
    stepsEl.appendChild(li);
  });

  const b = document.getElementById('banner');
  const err = s.error || {};
  if (err.active) {
    b.className = 'banner error';
    b.textContent = 'WRONG ITEM: expected ' + (err.expected || '?') +
      ', saw ' + (err.got || '?');
  } else if (s.misplaced) {
    b.className = 'banner warn';
    b.textContent = s.misplaced + ' is not inside the box';
  } else if (s.complete) {
    b.className = 'banner done';
    b.textContent = 'Sequence complete — press Reset';
  } else {
    b.className = 'banner hidden';
  }

  const lamps = document.getElementById('lamps');
  lamps.innerHTML = '';
  (s.lamps || []).forEach(l => lamps.appendChild(lamp(l.name, l.on)));
}

function lamp(label, on) {
  const d = document.createElement('div');
  d.className = 'lamp' + (on ? ' on' : '');
  d.innerHTML = '<span class="bulb"></span>' + label;
  return d;
}

function doReset() { fetch('/reset', { method: 'POST' }); }

fetch('/state').then(r => r.json()).then(s => { if (s && s.steps) render(s); });

const es = new EventSource('/events');
es.onmessage = e => render(JSON.parse(e.data));
