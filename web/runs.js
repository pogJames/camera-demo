const HUE = '#22c55e';
const HUE_DIM = 'rgba(34, 197, 94, 0.35)';
const MUTED = '#a1a1aa';
const BORDER = '#27272a';
const CARD = '#0c0c0f';
// HUE_DIM's green at descending opacity: strongest = biggest slice
const RAMP = [0.80, 0.68, 0.56, 0.46, 0.38].map(a => `rgba(34, 197, 94, ${a})`);
const LINE = '#fafafa';

Chart.defaults.color = MUTED;
Chart.defaults.borderColor = BORDER;
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.animation = false;

const charts = {};

function secs(ms) { return Math.round(ms / 100) / 10; }

function clock(ms) {
  const s = Math.round(ms / 1000);
  return s < 60 ? s + 's' : Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
}

function hhmm(epoch) {
  const d = new Date(epoch * 1000);
  return String(d.getHours()).padStart(2, '0') + ':' +
         String(d.getMinutes()).padStart(2, '0');
}

// dashed 80% rule on the cumulative axis: bars left of the crossing are the vital few
const vitalFew = {
  id: 'vitalFew',
  beforeDatasetsDraw(chart, args, opts) {
    const scale = chart.scales.y1;
    if (!scale) return;
    const y = scale.getPixelForValue((opts && opts.at) || 80);
    const { left, right } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = MUTED;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = MUTED;
    ctx.font = '10px ' + Chart.defaults.font.family;
    ctx.textAlign = 'right';
    ctx.fillText('80%', right - 4, y - 4);
    ctx.restore();
  },
};

function swap(key, canvasId, emptyId, rows, config) {
  if (charts[key]) { charts[key].destroy(); delete charts[key]; }
  const canvas = document.getElementById(canvasId);
  const empty = document.getElementById(emptyId);
  const has = rows.length > 0;
  canvas.parentElement.hidden = !has;
  empty.hidden = has;
  if (has) charts[key] = new Chart(canvas, config());
}

function renderKpis(k) {
  const tiles = [
    { label: 'Runs', value: k.runs, note: k.complete + ' complete' },
    { label: 'First pass', value: k.runs ? k.fpy + '%' : '—',
      note: k.clean + ' clean of ' + k.runs },
    { label: 'Median cycle', value: k.complete ? clock(k.median_ms) : '—',
      note: 'completed runs' },
    { label: 'Abandoned', value: k.abandoned, note: 'never finished',
      bad: k.abandoned > 0 },
  ];
  document.getElementById('kpis').innerHTML = tiles.map(t =>
    '<div class="kpi"><div class="kpi-label">' + t.label + '</div>' +
    '<div class="kpi-value' + (t.bad ? ' bad' : '') + '">' + t.value + '</div>' +
    '<div class="kpi-note">' + t.note + '</div></div>').join('');
}

function renderTrend(rows) {
  swap('trend', 'c-trend', 'e-trend', rows, () => ({
    type: 'line',
    data: {
      labels: rows.map(r => hhmm(r.started_at)),
      datasets: [{
        label: 'Cycle time',
        data: rows.map(r => secs(r.duration_ms)),
        borderColor: LINE,
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: rows.map(r => r.events ? LINE : HUE),
        pointBorderColor: rows.map(r => r.events ? LINE : HUE),
      }],
    },
    options: {
      scales: {
        y: { beginAtZero: true, title: { display: true, text: 'seconds' },
             grid: { color: BORDER } },
        x: { grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (it) => 'Run #' + rows[it[0].dataIndex].id,
            label: (it) => {
              const r = rows[it.dataIndex];
              return clock(r.duration_ms) +
                (r.events ? '  ·  ' + r.events + ' error(s)' : '  ·  clean');
            },
          },
        },
      },
    },
  }));
}

function renderSteps(rows) {
  const sorted = rows.slice().sort((a, b) => b.median_ms - a.median_ms);
  const total = sorted.reduce((s, r) => s + r.median_ms, 0) || 1;
  swap('steps', 'c-steps', 'e-steps', sorted, () => ({
    type: 'pie',
    data: {
      labels: sorted.map(r => r.title),
      datasets: [{
        data: sorted.map(r => secs(r.median_ms)),
        backgroundColor: sorted.map((_, i) => RAMP[i % RAMP.length]),
        borderColor: CARD,
        borderWidth: 2,
      }],
    },
    options: {
      plugins: {
        legend: { position: 'right', labels: { boxWidth: 12, boxHeight: 12 } },
        tooltip: {
          callbacks: {
            label: (it) => {
              const r = sorted[it.dataIndex];
              return secs(r.median_ms) + 's  ·  ' +
                     Math.round(r.median_ms * 100 / total) + '%  ·  ' +
                     r.n + ' samples';
            },
          },
        },
      },
    },
  }));
}

function renderPareto(rows) {
  swap('pareto', 'c-pareto', 'e-pareto', rows, () => ({
    data: {
      labels: rows.map(r => [r.title, r.kind.replace('_', ' ')]),
      datasets: [
        {
          type: 'bar',
          label: 'Times seen',
          data: rows.map(r => r.n),
          yAxisID: 'y',
          backgroundColor: HUE_DIM,
          borderColor: HUE,
          borderWidth: 2,
          borderRadius: 4,
          maxBarThickness: 56,
          order: 2,
        },
        {
          type: 'line',
          label: 'Cumulative',
          data: rows.map(r => r.cum),
          yAxisID: 'y1',
          borderColor: LINE,
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: LINE,
          tension: 0,
          order: 1,
          clip: { top: 10, left: 0, right: 0, bottom: 0 },  // 100% dot sits on the edge
        },
      ],
    },
    options: {
      layout: { padding: { top: 10 } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 },
             title: { display: true, text: 'times seen' },
             grid: { color: BORDER } },
        y1: { position: 'right', min: 0, max: 100,
              ticks: { callback: (v) => v + '%', stepSize: 25 },
              grid: { display: false } },
        x: { grid: { display: false } },
      },
      plugins: {
        legend: { display: true, labels: { boxWidth: 10, boxHeight: 10 } },
        vitalFew: { at: 80 },
        tooltip: {
          callbacks: {
            label: (it) => {
              const r = rows[it.dataIndex];
              return it.datasetIndex === 0
                ? r.n + ' times  ·  ' + r.secs + 's lost'
                : r.cum + '% of all errors up to here';
            },
          },
        },
      },
    },
    plugins: [vitalFew],
  }));
}

function renderAbandon(rows) {
  swap('abandon', 'c-abandon', 'e-abandon', rows, () => ({
    type: 'bar',
    data: {
      labels: rows.map(r => r.title),
      datasets: [{
        data: rows.map(r => r.n),
        backgroundColor: 'rgba(125, 125, 125, 0.35)',
        borderColor: MUTED,
        borderWidth: 1,
        borderRadius: 4,
        barThickness: 56,
      }],
    },
    options: {
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: BORDER } },
        x: { grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (it) => rows[it.dataIndex].n + ' run(s) stopped here',
          },
        },
      },
    },
  }));
}

function fillFilter(el, values, current, allLabel) {
  el.innerHTML = '<option value="">' + allLabel + '</option>' +
    values.map(v => '<option value="' + v + '"' +
      (v === current ? ' selected' : '') + '>' + v + '</option>').join('');
}

let day = '';
let sku = '';

function load() {
  const q = new URLSearchParams();
  if (day) q.set('day', day);
  if (sku) q.set('sku', sku);
  fetch('/api/runs?' + q.toString())
    .then(r => r.json())
    .then(d => {
      fillFilter(document.getElementById('f-day'), d.days, day, 'All days');
      fillFilter(document.getElementById('f-sku'), d.skus, sku, 'All products');
      renderKpis(d.kpi);
      renderTrend(d.trend);
      renderSteps(d.steps);
      renderPareto(d.pareto);
      renderAbandon(d.abandon);
    });
}

document.getElementById('f-day').addEventListener('change', e => {
  day = e.target.value; load();
});
document.getElementById('f-sku').addEventListener('change', e => {
  sku = e.target.value; load();
});

load();
