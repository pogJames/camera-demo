function render(s) {
  document.body.classList.toggle('scanned', !!s.scan);
  document.body.classList.toggle('running', !!s.loaded);
  renderPart(s.scan);

  const stepsEl = document.getElementById('steps');
  stepsEl.innerHTML = '';
  s.steps.forEach((st, i) => {
    const li = document.createElement('li');
    li.className = 'step ' + st.state;
    const proof = st.state === 'done'
      ? '<a class="proof" href="/log/' + i + '" target="_blank">clip</a>'
      : '';
    const at = st.at ? '<span class="at">' + st.at + '</span>' : '';
    li.innerHTML = '<span class="dot"></span><span class="text"><span class="name">' +
      st.title + '</span>' + at + '</span>' + proof +
      '<span class="tick">&#10003;</span>';
    stepsEl.appendChild(li);
  });

  const b = document.getElementById('banner');
  const err = s.error || {};
  if (err.active) {
    b.className = 'banner error';
    b.textContent = 'WRONG ITEM: expected ' + (err.expected || '?') +
      ', saw ' + (err.got || '?');
  } else if (!s.scan) {
    b.className = 'banner warn';
    b.textContent = 'Present a closed box to scan…';
  } else if (!s.loaded) {
    b.className = 'banner warn';
    b.textContent = 'Unknown product';
  } else if (s.misplaced) {
    b.className = 'banner warn';
    b.textContent = s.misplaced + ' is not inside the box';
  } else if (s.complete) {
    b.className = 'banner done';
    b.textContent = 'Sequence complete';
  } else if (s.current === 0) {
    b.className = 'banner done';
    b.textContent = 'Product scanned, start packing!';
  } else {
    b.className = 'banner hidden';
  }

  const lamps = document.getElementById('lamps');
  lamps.innerHTML = '';
  (s.lamps || []).forEach(l => lamps.appendChild(lamp(l.name, l.on)));
}

let partOpen = false;

function renderPart(scan) {
  const el = document.getElementById('part');
  if (!scan) {
    partOpen = false;
    el.innerHTML = '<div class="part empty">Waiting for barcode…</div>';
    return;
  }
  const specs = scan.specs || {};
  const hidden = ['sku', 'steps'];          // sku is the heading, steps drive the panel
  const rest = Object.keys(specs).filter(k => hidden.indexOf(k) < 0);
  const rows = rest.length
    ? rest.map(k => row(titleCase(k), specs[k])).join('')
    : row('Code', scan.code);
  el.innerHTML =
    '<div class="part' + (specs.sku ? '' : ' unknown') + (partOpen ? ' open' : '') + '">' +
      '<div class="sku">' + esc(specs.sku || 'Not in catalog') +
        '<span class="chev">' + (partOpen ? '&#10005;' : '&#8250;') + '</span></div>' +
      '<div class="details">' + rows + '</div>' +
    '</div>';
  el.firstChild.onclick = () => { partOpen = !partOpen; renderPart(scan); };
}

function row(k, v) {
  const value = Array.isArray(v)
    ? '<ul class="list">' + v.map(i => '<li>' + esc(i) + '</li>').join('') + '</ul>'
    : esc(v);
  return '<div class="row"><span class="k">' + esc(k) + '</span>' +
    '<span class="v">' + value + '</span></div>';
}

function titleCase(k) {
  return k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function esc(v) {
  const d = document.createElement('div');
  d.textContent = v == null ? '' : v;
  return d.innerHTML;
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
