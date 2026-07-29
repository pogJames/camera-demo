function render(s) {
  const stepsEl = document.getElementById('steps');
  stepsEl.innerHTML = '';
  s.steps.forEach((st, i) => {
    const li = document.createElement('li');
    li.className = 'step ' + st.state;
    const proof = st.state === 'done'
      ? '<a class="proof" href="/log/' + i + '" target="_blank">view</a>'
      : '';
    li.innerHTML = '<span class="dot"></span><span class="name">' +
      st.label + '</span>' + proof + '<span class="tick">&#10003;</span>';
    stepsEl.appendChild(li);
  });

  const b = document.getElementById('banner');
  if (s.fault && s.fault.active) {
    b.className = 'banner fault';
    b.textContent = 'FAULT: expected ' + (s.fault.expected || '?') +
      ', saw ' + (s.fault.got || '?');
  } else if (s.complete) {
    b.className = 'banner done';
    b.textContent = 'Sequence complete';
  } else {
    b.className = 'banner hidden';
  }

  const lamps = document.getElementById('lamps');
  lamps.innerHTML = '';
  s.steps.forEach((st, i) => lamps.appendChild(lamp('L' + (i + 1), st.state === 'done', false)));
  lamps.appendChild(lamp('FAULT', s.fault && s.fault.active, true));
}

function lamp(label, on, isFault) {
  const d = document.createElement('div');
  d.className = 'lamp' + (on ? ' on' : '') + (isFault ? ' fault' : '');
  d.innerHTML = '<span class="bulb"></span>' + label;
  return d;
}

function doReset() { fetch('/reset', { method: 'POST' }); }

fetch('/state').then(r => r.json()).then(s => { if (s && s.steps) render(s); });

const es = new EventSource('/events');
es.onmessage = e => render(JSON.parse(e.data));
