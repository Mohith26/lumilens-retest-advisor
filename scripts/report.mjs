// Prints every headline number shown on the page and writes data/results.json.
import { simulateLine, buildContext, replayAll, calibration } from '../core/disposition.mjs';
import { writeFileSync, mkdirSync } from 'node:fs';
const sim = simulateLine();
const ctx = buildContext(sim);
const g = ctx.gauge;
const res = replayAll(sim, ctx);
const scores = Object.fromEntries(Object.entries(res).map(([k, r]) => [k, r.score]));
for (const [k, s] of Object.entries(scores)) console.log(k.padEnd(16), `shipped ${s.shipped}  escapes ${s.escapes}  dppm ${Math.round(s.dppm)}  falseHolds ${s.falseHolds}  retests ${s.retests}  shippedOnRetest ${s.shippedOnRetest} (bad ${s.escapesOnRetest})  cost $${s.cost}`);
const calib = calibration(sim, ctx, res.bayes.decisions).map((b) => ({ lo: b.lo, hi: b.hi, n: b.n, expected: +b.pred.toFixed(2), observed: b.obs }));
console.table(calib);
const gauge = { laneSd: Math.sqrt(g.laneVar), mateSd: Math.sqrt(g.mateVar), contamRate: g.contamRate, contamMean: g.contamMean, contamSd: Math.sqrt(g.contamVar) };
console.log('gauge', JSON.stringify(gauge));
console.log('retests: advisor vs retest-until-pass', (1 - scores.bayes.retests / scores.retestUntilPass.retests).toFixed(3));
mkdirSync(new URL('../data/', import.meta.url), { recursive: true });
writeFileSync(new URL('../data/results.json', import.meta.url), JSON.stringify({ generatedBy: 'node scripts/report.mjs', seed: sim.cfg.seed, evalModules: ctx.evalIdx.length, trainModules: ctx.nTrain, scores, calibration: calib, gauge, pop: ctx.pop }, null, 1));
