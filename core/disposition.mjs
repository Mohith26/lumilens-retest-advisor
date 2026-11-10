// Retest Advisor core: Bayesian per-module disposition for multi-lane transceiver TX test.
// Pure ES module (browser + node). No dependencies.
//
// Pieces
//   1. simulateLine(): seeded synthetic 800G DR8 TX-test line (truth kept separate from observations)
//   2. estimateGauge(): measurement-system model from golden-unit checks (station bias track,
//      mating noise, contamination mixture, lane repeatability)
//   3. fitPopulation(): empirical-Bayes priors for module level and lane offsets from first-pass data
//   4. posteriorPass(): P(all 8 lanes truly <= spec | every attempt so far)
//   5. decide(): ship / retest / hold by expected cost with a one-step value-of-information lookahead
//   6. replay(): run any policy over the line and score it against truth

// ---------------------------------------------------------------- numerics
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussFactory(rand) {
  let spare = null;
  return function () {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0;
    while (u <= 1e-300) u = rand();
    v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

// erfc with fractional error < 1.2e-7 everywhere (Numerical Recipes erfcc)
export function erfc(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
// standard normal CDF and upper tail
export const Phi = (z) => 0.5 * erfc(-z / Math.SQRT2);
export const PhiUpper = (z) => 0.5 * erfc(z / Math.SQRT2);
const LOG2PI = Math.log(2 * Math.PI);
export const logNormPdf = (x, m, v) => -0.5 * (LOG2PI + Math.log(v) + (x - m) * (x - m) / v);

function logSumExp(arr) {
  let m = -Infinity;
  for (const a of arr) if (a > m) m = a;
  if (m === -Infinity) return m;
  let s = 0;
  for (const a of arr) s += Math.exp(a - m);
  return m + Math.log(s);
}

export function median(a) {
  const s = [...a].sort((x, y) => x - y);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

// ---------------------------------------------------------------- line simulator
export const LANES = 8;
export const DEFAULT_SIM = {
  seed: 1009,
  days: 14,
  unitsPerDay: 720,
  stations: ['TX-01', 'TX-02', 'TX-03', 'TX-04', 'TX-05', 'TX-06'],
  stationBias: [-0.04, 0.02, 0.0, 0.05, -0.02, 0.03], // dB, reference-receiver / patch-cord offsets
  drift: { station: 3, startDay: 5.5, endDay: 10.25, slope: 0.065 }, // dB/day until the cord is cleaned
  attempts: 4,
  // true lane TDECQ = module level + lane offset (dB)
  modMean: 2.32, modSd: 0.27,
  laneSd: 0.15, defectRate: 0.012, defectMean: 0.85, defectSd: 0.35,
  // measurement system
  laneNoiseSd: 0.07, laneNoiseDf: 6, // per-lane DCA/TDECQ repeatability (Student-t, heavier than Gaussian)
  mateSd: 0.055,                     // shared per re-mate effect (fiber/MPO seating)
  contamRate: 0.035, contamMean: 0.42, contamSd: 0.14, // dirty-connector events, all lanes shifted
  goldenPerShift: 3, shiftsPerDay: 3, goldenHistoryDays: 21,
};

function studentT(gauss, rand, df) {
  // t = z / sqrt(chi2/df), chi2 via sum of squares (df small integer)
  let c = 0;
  for (let i = 0; i < df; i++) { const g = gauss(); c += g * g; }
  return gauss() / Math.sqrt(c / df);
}

export function stationBiasAt(cfg, s, day) {
  let b = cfg.stationBias[s];
  const d = cfg.drift;
  if (s === d.station && day >= d.startDay && day < d.endDay) b += d.slope * (day - d.startDay);
  return b;
}

// Returns { units, golden, truth } ; units/golden carry observations only.
export function simulateLine(over = {}) {
  const cfg = { ...DEFAULT_SIM, ...over };
  const rand = mulberry32(cfg.seed);
  const gauss = gaussFactory(rand);
  const tScale = cfg.laneNoiseSd / Math.sqrt(cfg.laneNoiseDf / (cfg.laneNoiseDf - 2));
  const measureAttempt = (trueLanes, s, day) => {
    const b = stationBiasAt(cfg, s, day);
    let shared = cfg.mateSd * gauss();
    const contaminated = rand() < cfg.contamRate;
    if (contaminated) shared += cfg.contamMean + cfg.contamSd * gauss();
    const y = trueLanes.map((t) => +(t + b + shared + tScale * studentT(gauss, rand, cfg.laneNoiseDf)).toFixed(3));
    return { y, contaminated };
  };
  // golden module: characterised on the reference bench, values known to the MES
  const goldenTrue = [2.18, 2.31, 2.25, 2.42, 2.36, 2.21, 2.29, 2.47];
  const golden = [];
  const S = cfg.stations.length;
  for (let d = -cfg.goldenHistoryDays; d < cfg.days; d++) {
    for (let sh = 0; sh < cfg.shiftsPerDay; sh++) {
      const day = d + sh / cfg.shiftsPerDay + 0.01;
      for (let s = 0; s < S; s++) {
        const attempts = [];
        for (let g = 0; g < cfg.goldenPerShift; g++) attempts.push(measureAttempt(goldenTrue, s, day).y);
        golden.push({ station: s, day, attempts });
      }
    }
  }
  const units = [];
  const truth = [];
  const N = cfg.days * cfg.unitsPerDay;
  for (let i = 0; i < N; i++) {
    const day = (i + 0.5) / cfg.unitsPerDay;
    const mod = cfg.modMean + cfg.modSd * gauss();
    const lanes = [];
    for (let j = 0; j < LANES; j++) {
      let off = cfg.laneSd * gauss();
      if (rand() < cfg.defectRate) off += cfg.defectMean + cfg.defectSd * Math.abs(gauss());
      lanes.push(mod + off);
    }
    const st = Math.floor(rand() * S);
    const attempts = [];
    const contam = [];
    for (let k = 0; k < cfg.attempts; k++) {
      // retests go back to the same station 70% of the time, otherwise any free station
      const s = k === 0 || rand() < 0.7 ? st : Math.floor(rand() * S);
      const dd = day + k * 0.004;
      const m = measureAttempt(lanes, s, dd);
      attempts.push({ station: s, day: +dd.toFixed(4), y: m.y });
      contam.push(m.contaminated);
    }
    const sn = `LM8D${String(260000 + i).padStart(7, '0')}`;
    units.push({ id: i, sn, day: +day.toFixed(4), attempts });
    truth.push({ lanes: lanes.map((v) => +v.toFixed(4)), contam });
  }
  return { cfg, units, golden, goldenTrue, truth };
}

// ---------------------------------------------------------------- gauge model from golden units
// Two passes over golden-unit checks:
//   pass 1: local-level Kalman filter per station on the median of each check's attempt means
//           (median of 3 re-mates is robust to one dirty-connector attempt)
//   pass 2: 2-component EM on attempt-mean residuals around the filtered level -> mating noise and
//           contamination rate/size; within-attempt lane residuals -> lane repeatability
export function estimateGauge(golden, goldenTrue, nStations, opts = {}) {
  const q = opts.q ?? 0.035 * 0.035; // bias random-walk variance per shift
  const lanesN = goldenTrue.length;
  const attMean = (y) => y.reduce((a, v, j) => a + v - goldenTrue[j], 0) / lanesN;
  // lane repeatability from within-attempt residual spread
  let wSS = 0, wN = 0;
  for (const g of golden) if (g.day < (opts.fitUntilDay ?? Infinity)) for (const y of g.attempts) {
    const m = attMean(y);
    y.forEach((v, j) => { const e = v - goldenTrue[j] - m; wSS += e * e; wN++; });
  }
  const laneVar = wSS / (wN * (lanesN - 1) / lanesN);
  // noise parameters from within-check pairwise differences of attempt means (station bias cancels):
  // clean-clean ~ N(0, 2 v0); one dirty attempt ~ N(+-d, 2 v0 + v1). Only checks before `fitUntilDay`.
  const fitUntil = opts.fitUntilDay ?? Infinity;
  const diffs = [];
  for (const g of golden) {
    if (g.day >= fitUntil) continue;
    const ms = g.attempts.map(attMean);
    for (let i = 0; i < ms.length; i++) for (let j = i + 1; j < ms.length; j++) diffs.push(ms[i] - ms[j]);
  }
  let pi = 0.03, d = 0.3, v0 = 0.003, v1 = 0.02;
  for (let it = 0; it < 400; it++) {
    let sw = 0, swax = 0, swxx1 = 0, s0 = 0, s0xx = 0;
    for (const x of diffs) {
      const l0 = Math.log(Math.max(1e-9, 1 - 2 * pi)) + logNormPdf(x, 0, 2 * v0);
      const lp = Math.log(pi) + logNormPdf(x, d, 2 * v0 + v1);
      const ln = Math.log(pi) + logNormPdf(x, -d, 2 * v0 + v1);
      const z = logSumExp([l0, lp, ln]);
      const w0 = Math.exp(l0 - z), wp = Math.exp(lp - z), wn = Math.exp(ln - z);
      s0 += w0; s0xx += w0 * x * x;
      sw += wp + wn; swax += wp * x - wn * x;
      swxx1 += wp * (x - d) * (x - d) + wn * (x + d) * (x + d);
    }
    pi = Math.min(0.24, Math.max(1e-4, sw / (2 * diffs.length)));
    d = Math.max(0.05, swax / Math.max(sw, 1e-9));
    v0 = Math.max(1e-6, s0xx / Math.max(s0, 1e-9) / 2);
    v1 = Math.max(1e-4, swxx1 / Math.max(sw, 1e-9) - 2 * v0);
  }
  // each signed side has weight pi, so P(exactly one dirty in a pair) = 2 rho (1 - rho) = 2 pi
  const rho = (1 - Math.sqrt(Math.max(0, 1 - 4 * pi))) / 2;
  const runKF = (Rcheck) => {
    const tracks = Array.from({ length: nStations }, () => []);
    const byStation = Array.from({ length: nStations }, () => []);
    golden.forEach((g) => byStation[g.station].push({ day: g.day, ms: g.attempts.map(attMean) }));
    for (let s = 0; s < nStations; s++) {
      let x = 0, P = 0.05;
      for (const o of byStation[s].sort((a, b) => a.day - b.day)) {
        P += q;
        const z = median(o.ms);
        const K = P / (P + Rcheck);
        x += K * (z - x); P = (1 - K) * P;
        tracks[s].push({ day: o.day, z, x, P });
      }
    }
    return tracks;
  };
  const sharedVar = v0;
  const mateVar = Math.max(1e-6, sharedVar - laneVar / lanesN);
  const tracks = runKF(0.449 * sharedVar + 1e-6);
  return { laneVar, mateVar, sharedVar, contamRate: rho, contamMean: d, contamVar: v1, tracks, q, nPairs: diffs.length };
}

// causal bias lookup: latest filtered estimate at or before `day`, plus random-walk growth since
export function biasAt(gauge, s, day) {
  const tr = gauge.tracks[s];
  let lo = 0, hi = tr.length - 1, idx = -1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (tr[mid].day <= day) { idx = mid; lo = mid + 1; } else hi = mid - 1; }
  if (idx < 0) return { b: 0, v: 0.05 };
  const shiftsSince = (day - tr[idx].day) * 3;
  return { b: tr[idx].x, v: tr[idx].P + gauge.q * shiftsSince };
}

// ---------------------------------------------------------------- population priors (empirical Bayes)
export function fitPopulation(units, gauge, nUse = units.length) {
  const L = LANES;
  const means = [];
  const offs = [];
  for (let i = 0; i < nUse; i++) {
    const a = units[i].attempts[0];
    const { b } = biasAt(gauge, a.station, a.day);
    const m = a.y.reduce((s, v) => s + v, 0) / L;
    means.push(m - b);
    for (const v of a.y) offs.push(v - m);
  }
  // module level: robust location/scale, then remove measurement + contamination variance
  const med = median(means);
  const mad = median(means.map((x) => Math.abs(x - med))) * 1.4826;
  const contamShift = gauge.contamRate * gauge.contamMean;
  const modMean = med - contamShift;
  const modVar = Math.max(0.01, mad * mad - gauge.sharedVar);
  // lane offsets: 2-component EM with known measurement variance added to both components
  const noise = gauge.laneVar * (L - 1) / L;
  let pi = 0.02, m0 = 0, v0 = 0.02, m1 = 0.7, v1 = 0.12;
  for (let it = 0; it < 150; it++) {
    let a0 = 0, a0x = 0, a0xx = 0, a1 = 0, a1x = 0, a1xx = 0;
    const V0 = v0 + noise, V1 = v1 + noise;
    for (const x of offs) {
      const l0 = Math.log(1 - pi) + logNormPdf(x, m0, V0);
      const l1 = Math.log(pi) + logNormPdf(x, m1, V1);
      const w = 1 / (1 + Math.exp(l0 - l1));
      a1 += w; a1x += w * x; a1xx += w * x * x; a0 += 1 - w; a0x += (1 - w) * x; a0xx += (1 - w) * x * x;
    }
    pi = Math.min(0.3, Math.max(1e-4, a1 / offs.length));
    m0 = a0x / a0; m1 = a1x / Math.max(a1, 1e-9);
    v0 = Math.max(1e-4, a0xx / a0 - m0 * m0 - noise);
    v1 = Math.max(1e-3, a1xx / Math.max(a1, 1e-9) - m1 * m1 - noise);
  }
  return { modMean, modVar, lane: { pi, m0, v0, m1, v1 }, nUsed: nUse };
}

// ---------------------------------------------------------------- posterior
// attempts: [{station, day, y[8]}]
export function posterior(attempts, gauge, pop, spec) {
  const L = LANES;
  const k = attempts.length;
  const ybar = [], vbar = [];
  for (const a of attempts) {
    const { b, v } = biasAt(gauge, a.station, a.day);
    ybar.push(a.y.reduce((s, x) => s + x, 0) / L - b);
    vbar.push(gauge.mateVar + gauge.laneVar / L + v);
  }
  // module level L: mixture over contamination patterns z in {0,1}^k
  const comps = [];
  for (let mask = 0; mask < 1 << k; mask++) {
    let prec = 1 / pop.modVar, num = pop.modMean / pop.modVar;
    let logPrior = 0;
    const shifts = [], vars = [];
    for (let t = 0; t < k; t++) {
      const z = (mask >> t) & 1;
      const sh = z ? gauge.contamMean : 0;
      const vv = vbar[t] + (z ? gauge.contamVar : 0);
      logPrior += Math.log(z ? gauge.contamRate : 1 - gauge.contamRate);
      shifts.push(sh); vars.push(vv);
      prec += 1 / vv; num += (ybar[t] - sh) / vv;
    }
    const mPost = num / prec, vPost = 1 / prec;
    // marginal likelihood of ybar under this pattern (Gaussian with shared L): compute via
    // log p(y|z) = sum log N(y_t; mPost+sh, v_t) + log N(mPost; prior) - log N(mPost; post)
    let ll = 0;
    for (let t = 0; t < k; t++) ll += logNormPdf(ybar[t], mPost + shifts[t], vars[t]);
    ll += logNormPdf(mPost, pop.modMean, pop.modVar) - logNormPdf(mPost, mPost, vPost);
    comps.push({ mask, logw: logPrior + ll, m: mPost, v: vPost });
  }
  const lz = logSumExp(comps.map((c) => c.logw));
  for (const c of comps) c.w = Math.exp(c.logw - lz);
  // lane offsets: average within-attempt deviations, 2-component conjugate update per lane
  const nv = gauge.laneVar * (L - 1) / L / k;
  const lanes = [];
  const lp = pop.lane;
  for (let j = 0; j < L; j++) {
    let r = 0;
    for (let t = 0; t < k; t++) {
      const a = attempts[t];
      r += a.y[j] - a.y.reduce((s, x) => s + x, 0) / L;
    }
    r /= k;
    const comp = [[1 - lp.pi, lp.m0, lp.v0], [lp.pi, lp.m1, lp.v1]].map(([w, m, v]) => {
      const lw = Math.log(w) + logNormPdf(r, m, v + nv);
      const vp = 1 / (1 / v + 1 / nv);
      const mp = vp * (m / v + r / nv);
      return { lw, m: mp, v: vp };
    });
    const z = logSumExp(comp.map((c) => c.lw));
    lanes.push({ r, comps: comp.map((c) => ({ w: Math.exp(c.lw - z), m: c.m, sd: Math.sqrt(c.v) })) });
  }
  // P(all lanes <= spec) = E_L[ prod_j P(o_j <= spec - L) ], 1-D quadrature over the L mixture
  let lo = Infinity, hi = -Infinity;
  for (const c of comps) if (c.w > 1e-9) { const s = Math.sqrt(c.v); lo = Math.min(lo, c.m - 7 * s); hi = Math.max(hi, c.m + 7 * s); }
  const G = 241;
  const h = (hi - lo) / (G - 1);
  let pPass = 0, mass = 0;
  for (let g = 0; g < G; g++) {
    const x = lo + g * h;
    let dens = 0;
    for (const c of comps) if (c.w > 1e-12) dens += c.w * Math.exp(logNormPdf(x, c.m, c.v));
    const wq = (g === 0 || g === G - 1 ? 0.5 : 1) * h * dens;
    if (wq === 0) continue;
    let prod = 1;
    for (const ln of lanes) {
      let p = 0;
      for (const c of ln.comps) p += c.w * Phi((spec - x - c.m) / c.sd);
      prod *= p;
    }
    pPass += wq * prod; mass += wq;
  }
  pPass /= mass;
  const pFail = Math.min(1, Math.max(0, 1 - pPass));
  // per-lane marginal probabilities and posterior means for display
  const Lmean = comps.reduce((s, c) => s + c.w * c.m, 0);
  const Lvar = comps.reduce((s, c) => s + c.w * (c.v + c.m * c.m), 0) - Lmean * Lmean;
  const laneSummary = lanes.map((ln) => {
    const om = ln.comps.reduce((s, c) => s + c.w * c.m, 0);
    const ov = ln.comps.reduce((s, c) => s + c.w * (c.sd * c.sd + c.m * c.m), 0) - om * om;
    return { mean: Lmean + om, sd: Math.sqrt(Math.max(0, ov + Lvar)), defectProb: ln.comps[1].w };
  });
  const pContam = comps.reduce((acc, c) => acc.map((v, t) => v + (((c.mask >> t) & 1) ? c.w : 0)), new Array(k).fill(0));
  return { pFail, pPass, level: { mean: Lmean, sd: Math.sqrt(Math.max(0, Lvar)) }, lanes: laneSummary, pContam, comps, laneComps: lanes };
}

// ---------------------------------------------------------------- decision rule
export const DEFAULT_COSTS = { escape: 5000, falseHold: 180, retest: 4 };

// sample a hypothetical next attempt from the posterior predictive
function samplePredictive(post, nextStation, nextDay, gauge, rand, gauss) {
  // draw module level from mixture
  let u = rand(), c = post.comps[post.comps.length - 1];
  for (const cc of post.comps) { u -= cc.w; if (u <= 0) { c = cc; break; } }
  const Lv = c.m + Math.sqrt(c.v) * gauss();
  const { b, v } = biasAt(gauge, nextStation, nextDay);
  let shared = b + Math.sqrt(v) * gauss() + Math.sqrt(gauge.mateVar) * gauss();
  if (rand() < gauge.contamRate) shared += gauge.contamMean + Math.sqrt(gauge.contamVar) * gauss();
  const sdl = Math.sqrt(gauge.laneVar);
  const y = post.laneComps.map((ln) => {
    let uu = rand(), cc = ln.comps[1];
    if (uu < ln.comps[0].w) cc = ln.comps[0];
    const o = cc.m + cc.sd * gauss();
    return Lv + o + shared + sdl * gauss();
  });
  return { station: nextStation, day: nextDay, y };
}

export function actionCosts(pFail, costs) {
  return { ship: pFail * costs.escape, hold: (1 - pFail) * costs.falseHold };
}

// Decide after the attempts so far. Returns {action, pFail, ship, hold, retestValue?}
export function decide(attempts, ctx) {
  const { gauge, pop, spec, costs, maxAttempts = 4, voiDraws = 40, nextStation, nextDay, seed = 1 } = ctx;
  const post = posterior(attempts, gauge, pop, spec);
  const ac = actionCosts(post.pFail, costs);
  const best = Math.min(ac.ship, ac.hold);
  const now = ac.ship <= ac.hold ? 'ship' : 'hold';
  const out = { post, pFail: post.pFail, ship: ac.ship, hold: ac.hold, action: now, retestCost: null };
  // exact pruning: a retest costs at least costs.retest, so it can never beat an action already cheaper
  if (attempts.length >= maxAttempts || best <= costs.retest) return out;
  const rand = mulberry32(seed);
  const gauss = gaussFactory(rand);
  let exp = 0;
  for (let d = 0; d < voiDraws; d++) {
    const nxt = samplePredictive(post, nextStation, nextDay, gauge, rand, gauss);
    const p2 = posterior([...attempts, nxt], gauge, pop, spec);
    const a2 = actionCosts(p2.pFail, costs);
    exp += Math.min(a2.ship, a2.hold);
  }
  exp /= voiDraws;
  out.retestCost = costs.retest + exp;
  if (out.retestCost < best) out.action = 'retest';
  return out;
}

// ---------------------------------------------------------------- policies + replay
export const passes = (y, spec, guard = 0) => Math.max(...y) <= spec - guard;

export const POLICIES = {
  single: { label: 'Single test', run: (u, spec) => ({ action: passes(u.attempts[0].y, spec) ? 'ship' : 'hold', used: 1 }) },
  retestUntilPass: {
    label: 'Retest until pass (3 tries)',
    run: (u, spec) => {
      for (let k = 0; k < 3; k++) if (passes(u.attempts[k].y, spec)) return { action: 'ship', used: k + 1 };
      return { action: 'hold', used: 3 };
    },
  },
  guardBand: {
    label: 'Guard band 0.15 dB (3 tries)',
    run: (u, spec) => {
      for (let k = 0; k < 3; k++) if (passes(u.attempts[k].y, spec, 0.15)) return { action: 'ship', used: k + 1 };
      return { action: 'hold', used: 3 };
    },
  },
};

export function runBayes(u, ctx) {
  const trace = [];
  for (let k = 1; k <= u.attempts.length; k++) {
    const nxt = u.attempts[k] ?? u.attempts[k - 1];
    const d = decide(u.attempts.slice(0, k), { ...ctx, nextStation: nxt.station, nextDay: nxt.day, seed: (u.id + 1) * 7919 + k });
    trace.push({ k, pFail: d.pFail, ship: d.ship, hold: d.hold, retestCost: d.retestCost, action: d.action });
    if (d.action !== 'retest') return { action: d.action, used: k, pFail: d.pFail, trace };
  }
  const last = trace[trace.length - 1];
  return { action: last.ship <= last.hold ? 'ship' : 'hold', used: u.attempts.length, pFail: last.pFail, trace };
}

export function trueFail(truthRow, spec) { return Math.max(...truthRow.lanes) > spec; }

export function score(decisions, truth, spec, costs, idx) {
  const s = { n: 0, shipped: 0, held: 0, escapes: 0, falseHolds: 0, trueFails: 0, tests: 0, retests: 0, shippedOnRetest: 0, escapesOnRetest: 0 };
  for (const i of idx) {
    const d = decisions[i];
    const bad = trueFail(truth[i], spec);
    s.n++; s.tests += d.used; s.retests += d.used - 1;
    if (bad) s.trueFails++;
    if (d.action === 'ship') {
      s.shipped++;
      if (d.used > 1) s.shippedOnRetest++;
      if (bad) { s.escapes++; if (d.used > 1) s.escapesOnRetest++; }
    } else { s.held++; if (!bad) s.falseHolds++; }
  }
  s.dppm = s.shipped ? (s.escapes / s.shipped) * 1e6 : 0;
  s.cost = s.escapes * costs.escape + s.falseHolds * costs.falseHold + s.retests * costs.retest;
  return s;
}

// full pipeline: estimate on days < trainDays, evaluate on the rest (causal: bias track is causal anyway)
export function buildContext(sim, { spec = 3.4, costs = DEFAULT_COSTS, trainDays = 3 } = {}) {
  const gauge = estimateGauge(sim.golden, sim.goldenTrue, sim.cfg.stations.length, { fitUntilDay: trainDays });
  const nTrain = sim.units.filter((u) => u.day < trainDays).length;
  const pop = fitPopulation(sim.units, gauge, nTrain);
  const evalIdx = [];
  for (let i = nTrain; i < sim.units.length; i++) evalIdx.push(i);
  return { gauge, pop, spec, costs, evalIdx, nTrain };
}

export function replayAll(sim, ctx, onProgress) {
  const out = {};
  for (const [key, p] of Object.entries(POLICIES)) {
    const dec = [];
    for (const i of ctx.evalIdx) dec[i] = p.run(sim.units[i], ctx.spec);
    out[key] = { label: p.label, decisions: dec, score: score(dec, sim.truth, ctx.spec, ctx.costs, ctx.evalIdx) };
  }
  const dec = [];
  let c = 0;
  for (const i of ctx.evalIdx) {
    dec[i] = runBayes(sim.units[i], ctx);
    if (onProgress && ++c % 1000 === 0) onProgress(c / ctx.evalIdx.length);
  }
  out.bayes = { label: 'Retest Advisor', decisions: dec, score: score(dec, sim.truth, ctx.spec, ctx.costs, ctx.evalIdx) };
  return out;
}

// reliability of the posterior at the moment of the final decision (shipped + held units)
export function calibration(sim, ctx, bayesDecisions, edges = [0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0000001]) {
  const bins = edges.slice(0, -1).map((lo, i) => ({ lo, hi: edges[i + 1], n: 0, pred: 0, obs: 0 }));
  for (const i of ctx.evalIdx) {
    const d = bayesDecisions[i];
    const p = d.pFail;
    const b = bins.find((x) => p >= x.lo && p < x.hi);
    b.n++; b.pred += p; b.obs += trueFail(sim.truth[i], ctx.spec) ? 1 : 0;
  }
  return bins.map((b) => ({ ...b, predMean: b.n ? b.pred / b.n : 0, obsRate: b.n ? b.obs / b.n : 0 }));
}
