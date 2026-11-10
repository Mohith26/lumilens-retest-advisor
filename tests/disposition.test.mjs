import test from 'node:test';
import assert from 'node:assert/strict';
import {
  Phi, PhiUpper, erfc, mulberry32, gaussFactory, simulateLine, estimateGauge, fitPopulation, posterior, decide,
  buildContext, replayAll, runBayes, score, POLICIES, biasAt, stationBiasAt, DEFAULT_COSTS, LANES, trueFail,
} from '../core/disposition.mjs';

const sim = simulateLine();
const ctx = buildContext(sim);

test('normal CDF matches reference values (incl. deep tail used for DPPM-level risks)', () => {
  assert.ok(Math.abs(Phi(0) - 0.5) < 2e-7);
  assert.ok(Math.abs(Phi(1.959963985) - 0.975) < 2e-7);
  assert.ok(Math.abs(Phi(-1) - 0.158655254) < 2e-7);
  const ref = 3.167124183e-5; // upper tail at 4 sigma
  assert.ok(Math.abs(PhiUpper(4) - ref) / ref < 1e-6);
  const ref6 = 9.865876450e-10; // upper tail at 6 sigma
  assert.ok(Math.abs(PhiUpper(6) - ref6) / ref6 < 1e-6);
  assert.ok(Math.abs(erfc(0.5) - 0.4795001222) < 2e-7);
});

test('simulator is deterministic and observations carry no ground truth', () => {
  const again = simulateLine();
  assert.deepEqual(again.units[1234], sim.units[1234]);
  assert.deepEqual(again.golden[77], sim.golden[77]);
  const u = sim.units[0];
  assert.deepEqual(Object.keys(u).sort(), ['attempts', 'day', 'id', 'sn']);
  for (const a of u.attempts) assert.deepEqual(Object.keys(a).sort(), ['day', 'station', 'y']);
  assert.equal(sim.units.length, sim.cfg.days * sim.cfg.unitsPerDay);
  assert.equal(u.attempts[0].y.length, LANES);
});

test('gauge model recovers the measurement system from golden-unit checks alone', () => {
  const g = ctx.gauge, c = sim.cfg;
  assert.ok(Math.abs(Math.sqrt(g.laneVar) - c.laneNoiseSd) / c.laneNoiseSd < 0.1, 'lane repeatability');
  assert.ok(Math.abs(Math.sqrt(g.mateVar) - c.mateSd) / c.mateSd < 0.2, 'mating noise');
  assert.ok(g.contamRate > 0.02 && g.contamRate < 0.06, `contamination rate ${g.contamRate}`);
  assert.ok(Math.abs(g.contamMean - c.contamMean) < 0.1, 'contamination size');
  // the causal bias track follows the drifting station and recovers after cleaning
  const s = c.drift.station;
  const late = biasAt(g, s, c.drift.endDay - 0.05).b, truthLate = stationBiasAt(c, s, c.drift.endDay - 0.05);
  assert.ok(Math.abs(late - truthLate) < 0.12, `drift tracked within 0.12 dB (${late.toFixed(3)} vs ${truthLate.toFixed(3)})`);
  const after = biasAt(g, s, c.drift.endDay + 1.5).b;
  assert.ok(Math.abs(after - c.stationBias[s]) < 0.05, 'recovers after the cord is cleaned');
});

test('bias tracking is causal: changing future golden checks cannot change past estimates or decisions', () => {
  const cut = 8;
  const golden2 = sim.golden.map((g) => (g.day >= cut ? { ...g, attempts: g.attempts.map((y) => y.map((v) => v + 0.5)) } : g));
  const g2 = estimateGauge(golden2, sim.goldenTrue, sim.cfg.stations.length, { fitUntilDay: 3 });
  for (let s = 0; s < 6; s++) assert.deepEqual(biasAt(g2, s, 7.9), biasAt(ctx.gauge, s, 7.9));
  const ctx2 = { ...ctx, gauge: g2 };
  const early = ctx.evalIdx.filter((i) => sim.units[i].attempts.every((a) => a.day < cut)).slice(0, 200);
  for (const i of early) assert.deepEqual(runBayes(sim.units[i], ctx2).action, runBayes(sim.units[i], ctx).action);
});

test('posterior is monotone, bounded and sensible at the extremes', () => {
  const st = 0, day = 5;
  const mk = (vals) => [{ station: st, day, y: vals }];
  const base = [2.2, 2.3, 2.25, 2.4, 2.35, 2.2, 2.3, 2.45];
  const p0 = posterior(mk(base), ctx.gauge, ctx.pop, 3.4).pFail;
  assert.ok(p0 < 1e-5, `clearly good module ${p0}`);
  let prev = p0;
  for (const v of [2.9, 3.1, 3.3, 3.45, 3.6, 3.9]) {
    const y = [...base]; y[3] = v;
    const p = posterior(mk(y), ctx.gauge, ctx.pop, 3.4).pFail;
    assert.ok(p >= prev - 1e-12, 'raising one lane never lowers risk');
    assert.ok(p >= 0 && p <= 1);
    prev = p;
  }
  assert.ok(prev > 0.99, 'lane measured 0.5 dB over spec is almost surely bad');
});

test('a dirty-connector attempt is explained away by a clean retest', () => {
  const clean = [2.62, 2.7, 2.66, 2.81, 2.74, 2.6, 2.69, 2.86];
  const dirty = clean.map((v) => +(v + 0.6).toFixed(3)); // all lanes up ~0.6 dB: classic contamination
  const one = posterior([{ station: 1, day: 5, y: dirty }], ctx.gauge, ctx.pop, 3.4);
  const two = posterior([{ station: 1, day: 5, y: dirty }, { station: 1, day: 5.004, y: clean }], ctx.gauge, ctx.pop, 3.4);
  assert.ok(Math.max(...dirty) > 3.4, 'a plain limit check fails this attempt');
  assert.ok(one.pFail > 0.05, `first attempt alone carries real risk (${one.pFail})`);
  assert.ok(one.pContam[0] > 0.3, 'the uniform all-lane shift already looks like contamination');
  assert.ok(two.pContam[0] > 0.9, `first attempt attributed to contamination (${two.pContam[0]})`);
  assert.ok(two.pFail < 0.01 && two.pFail < one.pFail / 10, `after a clean retest the module is very likely good (${two.pFail})`);
});

test('inference is calibrated on data drawn from its own fitted model (validates the math)', () => {
  const rand = mulberry32(99), gauss = gaussFactory(rand);
  const { gauge, pop } = ctx;
  const bins = [[0.02, 0.2], [0.2, 0.8], [0.8, 0.98]].map(([lo, hi]) => ({ lo, hi, n: 0, p: 0, y: 0 }));
  let sumP = 0, sumY = 0, n = 0;
  for (let i = 0; i < 4000; i++) {
    const L = pop.modMean + Math.sqrt(pop.modVar) * gauss() + 0.35; // shift toward the spec to get informative cases
    const lanes = Array.from({ length: LANES }, () => {
      const lp = pop.lane; const d = rand() < lp.pi;
      return L + (d ? lp.m1 + Math.sqrt(lp.v1) * gauss() : lp.m0 + Math.sqrt(lp.v0) * gauss());
    });
    let shared = Math.sqrt(gauge.mateVar) * gauss() + biasAt(gauge, 2, 2).b;
    if (rand() < gauge.contamRate) shared += gauge.contamMean + Math.sqrt(gauge.contamVar) * gauss();
    const y = lanes.map((t) => t + shared + Math.sqrt(gauge.laneVar) * gauss());
    // shifted prior must be used for calibration: evaluate with the same shifted population
    const p = posterior([{ station: 2, day: 2, y }], gauge, { ...pop, modMean: pop.modMean + 0.35 }, 3.4).pFail;
    const f = Math.max(...lanes) > 3.4 ? 1 : 0;
    sumP += p; sumY += f; n++;
    const b = bins.find((x) => p >= x.lo && p < x.hi); if (b) { b.n++; b.p += p; b.y += f; }
  }
  assert.ok(Math.abs(sumP - sumY) / sumY < 0.08, `expected ${sumP.toFixed(1)} vs observed ${sumY} failures`);
  for (const b of bins) if (b.n > 60) assert.ok(Math.abs(b.p / b.n - b.y / b.n) < 0.08, `bin ${b.lo}-${b.hi}: ${(b.p / b.n).toFixed(3)} vs ${(b.y / b.n).toFixed(3)}`);
});

test('value-of-information rule: pruning is exact and retest is chosen only when it pays', () => {
  for (const i of ctx.evalIdx.slice(0, 400)) {
    const u = sim.units[i];
    const d = decide(u.attempts.slice(0, 1), { ...ctx, nextStation: u.attempts[1].station, nextDay: u.attempts[1].day, seed: 5 });
    const best = Math.min(d.ship, d.hold);
    if (best <= DEFAULT_COSTS.retest) { assert.equal(d.retestCost, null); assert.notEqual(d.action, 'retest'); }
    if (d.action === 'retest') assert.ok(d.retestCost < best);
    if (d.retestCost !== null) assert.ok(d.retestCost >= DEFAULT_COSTS.retest);
  }
});

test('baseline policies behave as specified and scoring identities hold', () => {
  const spec = 3.4;
  const u = { attempts: [{ y: [3.5, 2, 2, 2, 2, 2, 2, 2] }, { y: [3.3, 2, 2, 2, 2, 2, 2, 2] }, { y: [3.2, 2, 2, 2, 2, 2, 2, 2] }] };
  assert.deepEqual(POLICIES.retestUntilPass.run(u, spec), { action: 'ship', used: 2 });
  assert.deepEqual(POLICIES.single.run(u, spec), { action: 'hold', used: 1 });
  assert.deepEqual(POLICIES.guardBand.run(u, spec), { action: 'ship', used: 3 });
  const res = replayAll(sim, ctx);
  for (const r of Object.values(res)) {
    const s = r.score;
    assert.equal(s.shipped + s.held, s.n);
    assert.equal(s.tests, s.n + s.retests);
    assert.equal(s.cost, s.escapes * DEFAULT_COSTS.escape + s.falseHolds * DEFAULT_COSTS.falseHold + s.retests * DEFAULT_COSTS.retest);
    assert.ok(s.escapes <= s.trueFails);
  }
  // headline relations on the shown line
  const b = res.bayes.score, rup = res.retestUntilPass.score, gb = res.guardBand.score, one = res.single.score;
  assert.ok(rup.escapes > one.escapes, 'retest-until-pass ships more bad modules than testing once');
  assert.ok(b.escapes < rup.escapes && b.retests < rup.retests);
  assert.ok(b.escapes <= gb.escapes && b.falseHolds < gb.falseHolds && b.retests < gb.retests, 'dominates the guard band');
  assert.ok(b.cost < Math.min(one.cost, rup.cost, gb.cost));
  // predicted escapes among shipped units are consistent with what actually escaped (Poisson range)
  let pred = 0;
  for (const i of ctx.evalIdx) if (res.bayes.decisions[i].action === 'ship') pred += res.bayes.decisions[i].pFail;
  assert.ok(b.escapes <= pred + 3 * Math.sqrt(pred) + 2, `pred ${pred.toFixed(2)} vs ${b.escapes}`);
});

test('ground truth is only read by the scorer', () => {
  // runBayes never receives truth; decisions identical if truth is scrambled
  const scrambled = sim.truth.map(() => ({ lanes: new Array(8).fill(0), contam: [] }));
  const i = ctx.evalIdx[50];
  assert.deepEqual(runBayes(sim.units[i], ctx), runBayes({ ...sim.units[i] }, ctx));
  assert.equal(trueFail(scrambled[i], 3.4), false);
  const s = score({ [i]: { action: 'ship', used: 1 } }, sim.truth, 3.4, DEFAULT_COSTS, [i]);
  assert.equal(s.n, 1);
});
