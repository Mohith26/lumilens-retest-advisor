import {
  simulateLine, buildContext, POLICIES, runBayes, score, posterior, calibration, biasAt, stationBiasAt,
  trueFail, DEFAULT_COSTS, LANES,
} from './core/disposition.mjs';

const $ = (s) => document.querySelector(s);
const fmt = (n, d = 0) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const money = (n) => '$' + fmt(Math.round(n));
const DAY0 = Date.UTC(2026, 9, 27); // Oct 27 2026 = production day 0
const dateOf = (day) => new Date(DAY0 + day * 86400000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
const pct = (p) => {
  if (p < 1e-5) return '< 0.001%';
  if (p < 1e-3) return (p * 100).toFixed(3) + '%';
  if (p < 0.1) return (p * 100).toFixed(2) + '%';
  return (p * 100).toFixed(1) + '%';
};
const tip = document.createElement('div');
tip.className = 'tip';
document.body.appendChild(tip);
const showTip = (e, html) => { tip.innerHTML = html; tip.style.display = 'block'; tip.style.left = e.clientX + 14 + 'px'; tip.style.top = e.clientY + 14 + 'px'; };
const hideTip = () => { tip.style.display = 'none'; };

// ------------------------------------------------------------------ data + context
const sim = simulateLine();
const SPEC = 3.4;
const state = { costs: { ...DEFAULT_COSTS }, res: null, filter: 'prevented', page: 0, sel: null, station: sim.cfg.drift.station, runId: 0 };
let ctx = buildContext(sim, { spec: SPEC, costs: state.costs });
const S = sim.cfg.stations;
const evalStart = sim.units[ctx.evalIdx[0]].day, evalEnd = sim.units[ctx.evalIdx[ctx.evalIdx.length - 1]].day;

$('#meta').innerHTML = [
  ['Product', '800G DR8 OSFP, 8 × 100G PAM4'],
  ['Limit', `TDECQ ≤ ${SPEC} dB, every lane`],
  ['Replay', `${dateOf(evalStart)} – ${dateOf(evalEnd)} · ${fmt(ctx.evalIdx.length)} modules`],
  ['Priors', `fit on ${dateOf(0)} – ${dateOf(ctx.nTrain / sim.cfg.unitsPerDay - 1)} (${fmt(ctx.nTrain)} modules)`],
  ['Stations', `${S.length} TX test stations`],
].map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join('');

// ------------------------------------------------------------------ cost controls
const sliders = [['escape', '#c-escape', '#v-escape'], ['falseHold', '#c-hold', '#v-hold'], ['retest', '#c-retest', '#v-retest']];
for (const [k, inp, out] of sliders) {
  $(inp).value = state.costs[k];
  $(out).textContent = money(state.costs[k]);
  $(inp).addEventListener('input', () => { $(out).textContent = money(+$(inp).value); });
  $(inp).addEventListener('change', () => { state.costs[k] = +$(inp).value; ctx = { ...ctx, costs: { ...state.costs } }; run(); });
}

// ------------------------------------------------------------------ replay (chunked so the page stays responsive)
function run() {
  const id = ++state.runId;
  const prog = $('#progress');
  prog.classList.remove('done');
  const setP = (p, label) => { prog.querySelector('.bar').style.setProperty('--p', (p * 100).toFixed(1) + '%'); prog.querySelector('span').textContent = label; };
  const res = {};
  for (const [key, p] of Object.entries(POLICIES)) {
    const dec = [];
    for (const i of ctx.evalIdx) dec[i] = p.run(sim.units[i], SPEC);
    res[key] = { label: p.label, decisions: dec, score: score(dec, sim.truth, SPEC, ctx.costs, ctx.evalIdx) };
  }
  const dec = [];
  let c = 0;
  const N = ctx.evalIdx.length;
  setP(0, 'Replaying line…');
  const step = () => {
    if (id !== state.runId) return;
    const t0 = performance.now();
    while (c < N && performance.now() - t0 < 40) { const i = ctx.evalIdx[c++]; dec[i] = runBayes(sim.units[i], ctx); }
    setP(c / N, `Replaying line… ${fmt(c)} / ${fmt(N)}`);
    if (c < N) { setTimeout(step, 0); return; }
    res.bayes = { label: 'Retest Advisor', decisions: dec, score: score(dec, sim.truth, SPEC, ctx.costs, ctx.evalIdx) };
    state.res = res;
    prog.classList.add('done');
    setP(1, `Replayed ${fmt(N)} modules with live inference`);
    renderAll();
  };
  setTimeout(step, 0);
}

// ------------------------------------------------------------------ policy table + insights
function renderPolicy() {
  const r = state.res;
  const rows = [
    ['single', 'Test once', 'Ship if every lane measures ≤ limit'],
    ['retestUntilPass', 'Retest until pass', 'Up to 3 attempts, ship on the first pass'],
    ['guardBand', 'Guard band 0.15 dB', 'Ship on an attempt ≤ 3.25 dB, up to 3 attempts'],
    ['bayes', 'Retest Advisor', 'Posterior risk + value-of-information retests'],
  ];
  const best = (m) => Math.min(...rows.map(([k]) => r[k].score[m]));
  $('#policy-body').innerHTML = rows.map(([k, name, sub]) => {
    const s = r[k].score;
    const ours = k === 'bayes';
    const cell = (v, m, txt) => `<td class="${ours && v === best(m) ? 'good-num' : ''}">${txt ?? fmt(v)}</td>`;
    return `<tr class="${ours ? 'ours' : ''}"><td><div class="name">${name}${ours ? '<span class="tag">New</span>' : ''}</div><span class="sub">${sub}</span></td>
      <td>${fmt(s.shipped)}</td>${cell(s.escapes, 'escapes')}<td>${fmt(s.dppm)}</td>${cell(s.falseHolds, 'falseHolds')}
      <td>${fmt(s.retests)}</td><td>${fmt(s.shippedOnRetest)}</td>${cell(s.cost, 'cost', money(s.cost))}</tr>`;
  }).join('');
  $('#replay-sub').textContent = `Same ${fmt(ctx.evalIdx.length)} modules and the same measured attempts for every rule. Escapes and held good modules are scored against the simulator's ground truth.`;

  const rup = r.retestUntilPass.score, b = r.bayes.score, one = r.single.score;
  const trackS = sim.cfg.drift.station;
  const tr = ctx.gauge.tracks[trackS];
  const peak = tr.reduce((m, p) => (p.x > m.x ? p : m), tr[0]);
  const base = tr.filter((p) => p.day >= 0 && p.day < ctx.nTrain / sim.cfg.unitsPerDay).reduce((a, p, _, arr) => a + p.x / arr.length, 0);
  const lessRetest = 1 - b.retests / rup.retests;
  $('#insights').innerHTML = `
    <div class="insight"><div class="big"><span class="mag">${fmt(rup.escapesOnRetest)}</span> of ${fmt(rup.shippedOnRetest)}</div>
      <p>modules that retest-until-pass shipped on a second or third attempt were truly out of spec. Retesting until pass shipped ${fmt(rup.escapes)} bad modules; testing once shipped ${fmt(one.escapes)}.</p></div>
    <div class="insight"><div class="big">${fmt(b.escapes)} <span style="font-size:22px">escapes vs</span> ${fmt(rup.escapes)}</div>
      <p>with ${Math.round(lessRetest * 100)}% fewer retests (${fmt(b.retests)} vs ${fmt(rup.retests)}). It retests only where one more measurement can change the decision.</p></div>
    <div class="insight"><div class="big">${S[trackS]} <span class="mag">+${(peak.x - base).toFixed(2)} dB</span></div>
      <p>of drift by ${dateOf(peak.day)}, followed shift by shift from golden units and removed from every attempt measured there. A plain limit check reads it as lost margin on good modules.</p></div>`;
}

// ------------------------------------------------------------------ queue
function categories() {
  const r = state.res;
  const cats = { prevented: [], recovered: [], skipped: [], all: [] };
  for (const i of ctx.evalIdx) {
    const a = r.bayes.decisions[i], q = r.retestUntilPass.decisions[i];
    const bad = trueFail(sim.truth[i], SPEC);
    const differs = a.action !== q.action || a.used !== q.used;
    if (!differs) continue;
    cats.all.push(i);
    if (q.action === 'ship' && bad && a.action === 'hold') cats.prevented.push(i);
    if (a.action === 'ship' && !bad && r.guardBand.decisions[i].action === 'hold') cats.recovered.push(i);
    if (a.used < q.used) cats.skipped.push(i);
  }
  const key = (i) => -r.bayes.decisions[i].pFail;
  for (const k of Object.keys(cats)) cats[k].sort((x, y) => key(x) - key(y));
  return cats;
}

function renderQueue() {
  const cats = categories();
  const labels = { prevented: 'Escapes prevented', recovered: 'Good modules the guard band holds', skipped: 'Retests skipped', all: 'All differences' };
  $('#queue-chips').innerHTML = Object.keys(labels).map((k) => `<button class="chip ${state.filter === k ? 'on' : ''}" data-k="${k}">${labels[k]}<span class="n">${fmt(cats[k].length)}</span></button>`).join('');
  $('#queue-chips').querySelectorAll('.chip').forEach((b) => b.addEventListener('click', () => { state.filter = b.dataset.k; state.page = 0; renderQueue(); }));
  const list = cats[state.filter];
  const per = 9;
  const pages = Math.max(1, Math.ceil(list.length / per));
  state.page = Math.min(state.page, pages - 1);
  const rows = list.slice(state.page * per, state.page * per + per);
  if (state.sel === null || !cats.all.includes(state.sel)) {
    // default: the prevented escape where retest-until-pass needed the most failing readings to ship
    const r0 = state.res;
    const pick = [...cats.prevented].sort((x, y) => (r0.retestUntilPass.decisions[y].used - r0.retestUntilPass.decisions[x].used) || (r0.bayes.decisions[y].used - r0.bayes.decisions[x].used))[0];
    state.sel = pick ?? cats.all[0];
  }
  $('#queue-sub').textContent = `${fmt(cats.all.length)} of ${fmt(ctx.evalIdx.length)} modules got a different action or number of attempts than retest-until-pass. Click one to see why.`;
  const r = state.res;
  $('#queue-body').innerHTML = rows.map((i) => {
    const u = sim.units[i], a = r.bayes.decisions[i], q = r.retestUntilPass.decisions[i];
    const last = u.attempts[a.used - 1];
    const worst = Math.max(...last.y);
    const bad = trueFail(sim.truth[i], SPEC);
    return `<tr data-i="${i}" class="${i === state.sel ? 'sel' : ''}"><td class="mono">${u.sn}</td><td>${S[u.attempts[0].station]}</td>
      <td class="${worst > SPEC ? 'bad-num' : ''}">${worst.toFixed(2)} dB</td><td>${pct(a.pFail)}</td>
      <td>${badge(a.action)}<span class="verdict">${a.used} attempt${a.used > 1 ? 's' : ''}</span></td>
      <td>${badge(q.action)}<span class="verdict">${q.used} attempt${q.used > 1 ? 's' : ''}</span></td>
      <td>${bad ? '<span class="bad-num">Out of spec</span>' : '<span class="good-num">Good</span>'}<span class="verdict">worst ${Math.max(...sim.truth[i].lanes).toFixed(2)} dB</span></td></tr>`;
  }).join('') || `<tr><td colspan="7" style="text-align:left;color:var(--ink-2)">No modules in this view at these costs.</td></tr>`;
  $('#queue-body').querySelectorAll('tr[data-i]').forEach((tr) => tr.addEventListener('click', () => { state.sel = +tr.dataset.i; renderQueue(); renderUnit(); }));
  $('#pager').innerHTML = `<span>${list.length ? `${fmt(state.page * per + 1)}–${fmt(Math.min(list.length, state.page * per + per))} of ${fmt(list.length)}` : ''}</span>
    <span><button id="pp" ${state.page === 0 ? 'disabled' : ''}>Previous</button> <button id="pn" ${state.page >= pages - 1 ? 'disabled' : ''}>Next</button></span>`;
  $('#pp').onclick = () => { state.page--; renderQueue(); };
  $('#pn').onclick = () => { state.page++; renderQueue(); };
}

const badge = (a) => `<span class="badge b-${a}">${a === 'ship' ? 'Ship' : a === 'hold' ? 'Hold for FA' : 'Retest'}</span>`;

// ------------------------------------------------------------------ unit detail
function renderUnit() {
  const i = state.sel;
  if (i === undefined || i === null) { $('#unit').innerHTML = '<p class="muted">No module selected.</p>'; return; }
  const r = state.res;
  const u = sim.units[i], a = r.bayes.decisions[i], q = r.retestUntilPass.decisions[i];
  const att = u.attempts.slice(0, Math.max(a.used, q.used));
  const post = posterior(u.attempts.slice(0, a.used), ctx.gauge, ctx.pop, SPEC);
  const truth = sim.truth[i].lanes;
  const bad = trueFail(sim.truth[i], SPEC);
  const worstLane = post.lanes.reduce((m, l, j) => (l.mean > post.lanes[m].mean ? j : m), 0);
  const W = 560, H = 270, pl = 46, pr = 14, pt = 16, pb = 34;
  const vals = [...att.flatMap((x) => x.y), ...post.lanes.map((l) => l.mean + 2 * l.sd), SPEC + 0.15];
  const lo = Math.floor((Math.min(...att.flatMap((x) => x.y), ...post.lanes.map((l) => l.mean - 2 * l.sd)) - 0.1) * 10) / 10;
  const hi = Math.ceil((Math.max(...vals) + 0.05) * 10) / 10;
  const y = (v) => pt + (H - pt - pb) * (1 - (v - lo) / (hi - lo));
  const x = (j) => pl + (W - pl - pr) * ((j + 0.5) / LANES);
  const ticks = [];
  for (let t = Math.ceil(lo * 5) / 5; t <= hi + 1e-9; t += 0.2) ticks.push(+t.toFixed(2));
  const colors = ['#0f47d6', '#7a8cff', '#b4c2ff', '#cdd6ff'];
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Lane TDECQ">`;
  for (const t of ticks) svg += `<line x1="${pl}" x2="${W - pr}" y1="${y(t)}" y2="${y(t)}" stroke="#eef1fb"/><text x="${pl - 8}" y="${y(t) + 4}" font-size="11" fill="#6b6690" text-anchor="end">${t.toFixed(1)}</text>`;
  svg += `<line x1="${pl}" x2="${W - pr}" y1="${y(SPEC)}" y2="${y(SPEC)}" stroke="#d332d9" stroke-dasharray="5 4" stroke-width="1.5"/><text x="${W - pr}" y="${y(SPEC) - 6}" font-size="11" fill="#d332d9" text-anchor="end">limit ${SPEC} dB</text>`;
  for (let j = 0; j < LANES; j++) {
    const L = post.lanes[j];
    svg += `<rect x="${x(j) - 15}" y="${y(L.mean + 2 * L.sd)}" width="30" height="${Math.max(2, y(L.mean - 2 * L.sd) - y(L.mean + 2 * L.sd))}" rx="8" fill="${j === worstLane ? 'rgba(15,71,214,.16)' : 'rgba(15,71,214,.08)'}" data-tip="Lane ${j + 1}: true TDECQ ${L.mean.toFixed(2)} ± ${(2 * L.sd).toFixed(2)} dB (95%)<br>P(defective lane) ${pct(L.defectProb)}"/>`;
    svg += `<line x1="${x(j) - 15}" x2="${x(j) + 15}" y1="${y(L.mean)}" y2="${y(L.mean)}" stroke="#18075c" stroke-width="2"/>`;
    att.forEach((at, k) => {
      const cx = x(j) - 9 + k * (18 / Math.max(1, att.length - 1 || 1));
      const used = k < a.used;
      const dirty = used && post.pContam[k] > 0.5;
      svg += `<circle cx="${att.length === 1 ? x(j) : cx}" cy="${y(at.y[j])}" r="4" fill="${dirty ? '#fff' : colors[k]}" stroke="${colors[k]}" stroke-width="1.6" opacity="${used ? 1 : 0.45}" data-tip="Attempt ${k + 1} on ${S[at.station]}: ${at.y[j].toFixed(2)} dB measured${dirty ? '<br>likely dirty connector' : ''}${used ? '' : '<br>(only retest-until-pass took this attempt)'}"/>`;
    });
    svg += `<path d="M${x(j) + 19} ${y(truth[j]) - 4} l4 4 l-4 4 l-4 -4 z" fill="#a49fc4" data-tip="Sim. truth lane ${j + 1}: ${truth[j].toFixed(2)} dB"/>`;
    svg += `<text x="${x(j)}" y="${H - 12}" font-size="11.5" fill="#46397f" text-anchor="middle">L${j + 1}</text>`;
  }
  svg += `<text x="12" y="${pt + 4}" font-size="11" fill="#6b6690" transform="rotate(-90 12 ${pt + 4})" text-anchor="end">TDECQ (dB)</text></svg>`;

  const traceRows = a.trace.map((t) => `<tr><td>${t.k}</td><td>${S[u.attempts[t.k - 1].station]}</td><td>${pct(t.pFail)}</td><td>${money(t.ship)}</td><td>${money(t.hold)}</td>
      <td>${t.retestCost === null ? (t.k >= 4 ? 'last attempt' : 'cannot pay off') : money(t.retestCost)}</td><td>${badge(t.action)}</td></tr>`).join('');
  const notes = [];
  post.pContam.forEach((p, k) => { if (p > 0.5) notes.push(`Attempt ${k + 1} looks like a dirty connector (P = ${p.toFixed(2)}): all 8 lanes moved together, so it is discounted rather than trusted or rerolled.`); });
  const lw = post.lanes[worstLane];
  notes.push(`Worst lane L${worstLane + 1}: estimated true TDECQ ${lw.mean.toFixed(2)} dB, after removing the ${S[u.attempts[0].station]} offset of ${biasAt(ctx.gauge, u.attempts[0].station, u.attempts[0].day).b >= 0 ? '+' : ''}${biasAt(ctx.gauge, u.attempts[0].station, u.attempts[0].day).b.toFixed(2)} dB.`);
  if (q.action === 'ship' && q.used > 1) notes.push(`Retest-until-pass shipped it on attempt ${q.used} after ${q.used - 1} failing reading${q.used > 2 ? 's' : ''}; every failing reading is still evidence.`);

  $('#unit').innerHTML = `
    <div class="unit-top"><div><h3><span class="mono" style="font-size:17px">${u.sn}</span> ${badge(a.action)}</h3>
      <p class="muted">Tested ${dateOf(u.day)} on ${S[u.attempts[0].station]} · Sim. truth: ${bad ? `<b style="color:var(--bad);font-weight:500">out of spec</b> (max lane ${Math.max(...truth).toFixed(2)} dB)` : `<b style="color:var(--ok);font-weight:500">good</b> (max lane ${Math.max(...truth).toFixed(2)} dB)`}</p></div></div>
    <div class="unit-kpis">
      <div class="kpi"><div class="l">P(any lane out of spec)</div><div class="v">${pct(a.pFail)}</div></div>
      <div class="kpi"><div class="l">Advisor</div><div class="v">${a.action === 'hold' ? 'Hold' : a.action === 'ship' ? 'Ship' : 'Retest'} · ${a.used} att.</div></div>
      <div class="kpi"><div class="l">Retest-until-pass</div><div class="v">${q.action === 'hold' ? 'Hold' : 'Ship'} · ${q.used} att.</div></div>
    </div>
    <div class="chart" id="lane-chart">${svg}</div>
    <div class="legend"><span><i style="background:#0f47d6"></i>attempt 1</span><span><i style="background:#7a8cff"></i>2</span><span><i style="background:#b4c2ff"></i>3</span><span><i style="background:rgba(15,71,214,.25);height:10px"></i>true-TDECQ estimate ±2σ</span><span><i style="background:#a49fc4"></i>sim. truth</span></div>
    <table class="trace"><thead><tr><th>Attempt</th><th>Station</th><th>P(fail)</th><th>Ship now</th><th>Hold now</th><th>Retest (expected)</th><th>Decision</th></tr></thead><tbody>${traceRows}</tbody></table>
    <p class="note">${notes.join(' ')}</p>
    ${mathBox(a, ctx.costs)}`;
  bindTips($('#lane-chart'));
}

function mathBox(a, c) {
  const t = a.trace[a.trace.length - 1];
  return `<div class="math"><div class="card-kicker">How the decision at attempt ${t.k} was made</div>
    <div class="eq"><span>Ship now</span><code>P(fail) × escape = ${pct(t.pFail)} × ${money(c.escape)}</code><b>${money(t.ship)}</b></div>
    <div class="eq"><span>Hold now</span><code>(1 − P(fail)) × held good module = ${pct(1 - t.pFail)} × ${money(c.falseHold)}</code><b>${money(t.hold)}</b></div>
    <div class="eq"><span>Retest</span><code>${money(c.retest)} + E[min(ship, hold) after one more attempt]</code><b>${t.retestCost !== null ? money(t.retestCost) : t.k >= 4 ? 'no attempts left' : 'cannot pay off'}</b></div>
    <p>The expectation is over 40 draws from the posterior predictive of the next attempt (same station offset, mating noise and dirty-connector risk). A retest is skipped outright whenever the cheaper action already costs less than ${money(c.retest)}, which is exact because a retest can never cost less than that.</p></div>`;
}

function bindTips(root) {
  root.querySelectorAll('[data-tip]').forEach((el) => {
    el.addEventListener('mousemove', (e) => showTip(e, el.dataset.tip));
    el.addEventListener('mouseleave', hideTip);
  });
}

// ------------------------------------------------------------------ station chart
function renderStations() {
  $('#station-chips').innerHTML = S.map((n, s) => `<button class="chip ${state.station === s ? 'on' : ''}" data-s="${s}">${n}</button>`).join('');
  $('#station-chips').querySelectorAll('.chip').forEach((b) => b.addEventListener('click', () => { state.station = +b.dataset.s; renderStations(); }));
  const s = state.station;
  const tr = ctx.gauge.tracks[s].filter((p) => p.day >= -2);
  const W = 640, H = 250, pl = 48, pr = 14, pt = 14, pb = 30;
  const x0 = -2, x1 = sim.cfg.days;
  const ys = tr.flatMap((p) => [p.z, p.x + 2 * Math.sqrt(p.P), p.x - 2 * Math.sqrt(p.P)]);
  const lo = Math.min(-0.15, Math.floor(Math.min(...ys) * 10) / 10), hi = Math.max(0.2, Math.ceil(Math.max(...ys) * 10) / 10);
  const X = (d) => pl + (W - pl - pr) * (d - x0) / (x1 - x0);
  const Y = (v) => pt + (H - pt - pb) * (1 - (v - lo) / (hi - lo));
  let svg = `<svg viewBox="0 0 ${W} ${H}">`;
  for (let t = Math.ceil(lo * 10) / 10; t <= hi + 1e-9; t += 0.1) svg += `<line x1="${pl}" x2="${W - pr}" y1="${Y(t)}" y2="${Y(t)}" stroke="#eef1fb"/><text x="${pl - 8}" y="${Y(t) + 4}" font-size="11" fill="#6b6690" text-anchor="end">${t >= 0 ? '+' : ''}${t.toFixed(1)}</text>`;
  for (let d = 0; d < x1; d += 2) svg += `<text x="${X(d)}" y="${H - 9}" font-size="11" fill="#6b6690" text-anchor="middle">${dateOf(d)}</text>`;
  svg += `<rect x="${X(x0)}" y="${pt}" width="${X(ctx.nTrain / sim.cfg.unitsPerDay) - X(x0)}" height="${H - pt - pb}" fill="rgba(243,236,252,.6)"/><text x="${X(0.1)}" y="${pt + 13}" font-size="11" fill="#46397f">prior fit</text>`;
  const band = tr.map((p) => `${X(p.day)},${Y(p.x + 2 * Math.sqrt(p.P))}`).join(' ') + ' ' + [...tr].reverse().map((p) => `${X(p.day)},${Y(p.x - 2 * Math.sqrt(p.P))}`).join(' ');
  svg += `<polygon points="${band}" fill="rgba(15,71,214,.12)"/>`;
  const truthPts = [];
  for (let d = x0; d <= x1; d += 0.05) truthPts.push(`${X(d)},${Y(stationBiasAt(sim.cfg, s, d))}`);
  svg += `<polyline points="${truthPts.join(' ')}" fill="none" stroke="#a49fc4" stroke-width="1.5" stroke-dasharray="4 4"/>`;
  svg += `<polyline points="${tr.map((p) => `${X(p.day)},${Y(p.x)}`).join(' ')}" fill="none" stroke="#0f47d6" stroke-width="2"/>`;
  for (const p of tr) svg += `<circle cx="${X(p.day)}" cy="${Y(p.z)}" r="2.6" fill="#18075c" opacity=".55" data-tip="${S[s]} golden check ${dateOf(p.day)}<br>median offset ${p.z >= 0 ? '+' : ''}${p.z.toFixed(3)} dB<br>filtered ${p.x >= 0 ? '+' : ''}${p.x.toFixed(3)} ± ${(2 * Math.sqrt(p.P)).toFixed(3)}"/>`;
  svg += `<line x1="${pl}" x2="${W - pr}" y1="${Y(0)}" y2="${Y(0)}" stroke="#cfd8f6"/></svg>`;
  $('#station-chart').innerHTML = svg;
  $('#station-legend').innerHTML = `<span><i style="background:#18075c;opacity:.55;height:6px;width:6px;border-radius:50%"></i>golden check (median of 3)</span><span><i style="background:#0f47d6"></i>filtered offset ±2σ</span><span><i style="background:#a49fc4"></i>sim. truth</span>`;
  bindTips($('#station-chart'));
}

// ------------------------------------------------------------------ calibration
function renderCalibration() {
  const bins = calibration(sim, ctx, state.res.bayes.decisions);
  const names = ['< 0.01%', '0.01–0.1%', '0.1–1%', '1–10%', '10–50%', '50–90%', '> 90%'];
  const W = 640, H = 220, pl = 48, pr = 14, pt = 18, pb = 44;
  const exp = bins.map((b) => b.pred), obs = bins.map((b) => b.obs);
  const mx = Math.max(...exp, ...obs, 1);
  const Y = (v) => pt + (H - pt - pb) * (1 - Math.sqrt(v) / Math.sqrt(mx));
  const bw = (W - pl - pr) / bins.length;
  let svg = `<svg viewBox="0 0 ${W} ${H}">`;
  for (const t of [0, 1, 5, 20, 50, 100, 200, 400].filter((t) => t <= mx)) svg += `<line x1="${pl}" x2="${W - pr}" y1="${Y(t)}" y2="${Y(t)}" stroke="#eef1fb"/><text x="${pl - 8}" y="${Y(t) + 4}" font-size="11" fill="#6b6690" text-anchor="end">${t}</text>`;
  bins.forEach((b, k) => {
    const cx = pl + bw * k + bw / 2;
    svg += `<rect x="${cx - 17}" y="${Y(b.pred)}" width="16" height="${Y(0) - Y(b.pred)}" rx="4" fill="#b4c2ff" data-tip="${names[k]} bucket: ${fmt(b.n)} modules<br>expected out-of-spec ${b.pred.toFixed(1)}"/>`;
    svg += `<rect x="${cx + 1}" y="${Y(b.obs)}" width="16" height="${Y(0) - Y(b.obs)}" rx="4" fill="#18075c" data-tip="${names[k]} bucket: ${fmt(b.n)} modules<br>actually out of spec ${b.obs}"/>`;
    svg += `<text x="${cx}" y="${H - 26}" font-size="11" fill="#46397f" text-anchor="middle">${names[k]}</text><text x="${cx}" y="${H - 12}" font-size="10.5" fill="#6b6690" text-anchor="middle">n=${fmt(b.n)}</text>`;
  });
  svg += `<text x="12" y="${pt}" font-size="11" fill="#6b6690" transform="rotate(-90 12 ${pt})" text-anchor="end">modules (√ scale)</text></svg>`;
  $('#calib-chart').innerHTML = svg + `<div class="legend"><span><i style="background:#b4c2ff;height:10px"></i>expected out of spec (sum of P)</span><span><i style="background:#18075c;height:10px"></i>actually out of spec (sim. truth)</span></div>`;
  bindTips($('#calib-chart'));
  const g = ctx.gauge, p = ctx.pop, c = sim.cfg;
  const rows = [
    ['Lane repeatability σ', Math.sqrt(g.laneVar).toFixed(3) + ' dB', c.laneNoiseSd.toFixed(3) + ' dB (Student-t, 6 df)'],
    ['Re-mate σ (all lanes)', Math.sqrt(g.mateVar).toFixed(3) + ' dB', c.mateSd.toFixed(3) + ' dB'],
    ['Dirty-connector rate · shift', `${(g.contamRate * 100).toFixed(1)}% · +${g.contamMean.toFixed(2)} dB`, `${(c.contamRate * 100).toFixed(1)}% · +${c.contamMean.toFixed(2)} dB`],
    ['Module level', `${p.modMean.toFixed(2)} ± ${Math.sqrt(p.modVar).toFixed(2)} dB`, `${c.modMean.toFixed(2)} ± ${c.modSd.toFixed(2)} dB`],
    ['Defective-lane rate', `${(p.lane.pi * 100).toFixed(2)}%`, `${(c.defectRate * 100).toFixed(2)}%`],
  ];
  $('#model-table').innerHTML = `<thead><tr><th>Measurement + population model</th><th>Estimated from data</th><th>Sim. truth</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('')}</tbody>`;
}

function renderAll() {
  renderPolicy();
  renderQueue();
  renderUnit();
  renderStations();
  renderCalibration();
}

renderStations();
run();
