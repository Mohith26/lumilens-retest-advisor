// Robustness: rerun the full pipeline on independently seeded lines. Writes data/seeds.json.
import { simulateLine, buildContext, replayAll } from '../core/disposition.mjs';
import { writeFileSync, mkdirSync } from 'node:fs';
const seeds = Array.from({ length: Number(process.argv[2] ?? 12) }, (_, i) => 1000 + i);
const rows = [];
for (const seed of seeds) {
  const sim = simulateLine({ seed });
  const ctx = buildContext(sim);
  const r = replayAll(sim, ctx);
  const row = { seed };
  for (const k of Object.keys(r)) { const s = r[k].score; row[k] = { escapes: s.escapes, falseHolds: s.falseHolds, retests: s.retests, cost: s.cost, shipped: s.shipped, dppm: Math.round(s.dppm) }; }
  rows.push(row);
  console.log(seed, ['single','retestUntilPass','guardBand','bayes'].map(k => `${k}:${row[k].escapes}/${row[k].falseHolds}/${row[k].retests}/$${row[k].cost}`).join('  '));
}
const med = (a) => { const s = [...a].sort((x, y) => x - y); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const summary = {};
for (const k of ['single', 'retestUntilPass', 'guardBand', 'bayes']) summary[k] = Object.fromEntries(['escapes', 'falseHolds', 'retests', 'cost', 'dppm'].map((m) => [m, med(rows.map((r) => r[k][m]))]));
summary.bayesBeatsAllOnCost = rows.filter((r) => r.bayes.cost < Math.min(r.single.cost, r.retestUntilPass.cost, r.guardBand.cost)).length;
summary.bayesFewerEscapesThanRUP = rows.filter((r) => r.bayes.escapes < r.retestUntilPass.escapes).length;
summary.bayesFewerRetestsThanRUP = rows.filter((r) => r.bayes.retests < r.retestUntilPass.retests).length;
summary.rupMoreEscapesThanSingle = rows.filter((r) => r.retestUntilPass.escapes > r.single.escapes).length;
summary.n = rows.length;
console.log(JSON.stringify(summary, null, 1));
mkdirSync(new URL('../data/', import.meta.url), { recursive: true });
writeFileSync(new URL('../data/seeds.json', import.meta.url), JSON.stringify({ generatedBy: 'node scripts/seeds.mjs', seeds, rows, summary }, null, 1));
