// Writes the first 200 modules of the seeded line as CSV (observations only) so the data format is inspectable.
import { simulateLine } from '../core/disposition.mjs';
import { writeFileSync } from 'node:fs';
const sim = simulateLine();
const rows = ['serial,day,attempt,station,' + Array.from({ length: 8 }, (_, j) => `tdecq_l${j + 1}_db`).join(',')];
for (const u of sim.units.slice(0, 200)) u.attempts.forEach((a, k) => rows.push([u.sn, a.day, k + 1, sim.cfg.stations[a.station], ...a.y].join(',')));
writeFileSync(new URL('../data/sample_attempts.csv', import.meta.url), rows.join('\n') + '\n');
console.log('wrote', rows.length - 1, 'attempt rows');
