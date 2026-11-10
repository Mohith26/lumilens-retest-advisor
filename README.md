# Retest Advisor for Lumilens (concept)

**Live demo:** https://mohith26.github.io/lumilens-retest-advisor/

An unofficial concept feature for the manufacturing test floor at [Lumilens](https://lumilens.com). It answers the question every TX test line answers thousands of times a day: **this 800G module just failed (or barely passed) TDECQ on one lane. Ship it, retest it, or hold it?**

Instead of "retest until it passes" or a fixed guard band, the advisor computes the probability that any lane's *true* TDECQ is over the limit given *every* reading taken so far, corrects each reading for the station's current offset (tracked from golden-unit checks), recognises readings that look like a dirty connector, and orders a retest only when one more measurement is expected to be worth more than it costs.

Synthetic data. Not affiliated with Lumilens.

## Why

- **Retest-until-pass is a multiple-testing trap.** TDECQ repeatability is on the order of 0.1 dB (Keysight's TDECQ measurement notes), so a module whose worst lane is truly 3.45 dB will eventually read 3.39 dB if you remate and remeasure enough times. On the simulated line, retest-until-pass shipped 110 modules on a second or third attempt and **36 of them were truly out of spec**. It shipped 63 bad modules in total, more than twice as many as testing once (27).
- **A fixed guard band pays for that safety with good modules and station time:** 121 good modules held and 1,344 retests.
- **Stations drift.** A reference receiver or patch cord that ages adds a few tenths of a dB to every reading on that station; a plain limit check reads it as lost margin on good modules. Golden-unit checks already measure this, but the number is rarely fed back into disposition.

## How it works (`core/disposition.mjs`)

1. **Measurement-system model from golden units only.** Every station measures a characterised golden module 3 times per shift with a re-mate in between. Within-attempt lane residuals give lane repeatability. Within-check *pairwise differences* of attempt means cancel the station offset entirely, and a 3-component EM on those differences (clean pair, ± one dirty attempt) recovers the re-mate noise, the dirty-connector rate and its size. Fit on golden checks before the replay starts.
2. **Station offsets, causally.** A local-level Kalman filter per station on the median of each golden check. Every production attempt is corrected with the latest estimate *before* that attempt, and the filter variance (plus random-walk growth since the check) is carried into the posterior.
3. **Empirical-Bayes priors** from the first 3 days of first-pass data: module-level TDECQ (robust scale minus measurement variance) and a two-component lane-offset mixture (normal lanes vs defective lanes) fitted by EM with the known measurement variance added to both components.
4. **Posterior per module.** A reading is `true lane TDECQ + station offset + per-attempt shared effect (re-mate, or a dirty connector shifting all lanes) + lane noise`. The model splits each attempt into its lane mean and within-attempt lane deviations: the shared effects cancel in the deviations, so lane offsets get a closed-form two-component conjugate update, and the module level gets an exact Gaussian mixture over all 2^k dirty/clean patterns of the k attempts. `P(all 8 lanes ≤ 3.4 dB)` is then a 1-D integral over the module level (241-point quadrature) of a product of per-lane mixture CDFs.
5. **Decision.** Ship costs `P(fail) × escape cost`; hold costs `(1 − P(fail)) × cost of holding a good module`; retest costs the retest plus the expected cost of the best action after one more attempt, estimated with 40 posterior-predictive draws (same station offset, re-mate noise and dirty-connector risk). Retest is skipped outright when the cheaper action already costs less than a retest, which is exact (a retest can never cost less than its own price). Costs are sliders on the page; changing one re-runs the replay with live inference in the browser.

## Results (all from `node scripts/report.mjs`; the page computes the same numbers live)

Simulated 800G DR8 TX test line: 6 stations, 14 days, 10,080 modules, up to 4 recorded attempts per module, one station (TX-04) drifting about +0.3 dB over 5 days until its cord is cleaned. Priors fit on days 1-3 (2,160 modules); every rule is replayed on the remaining 7,920 modules using the same recorded attempts. Costs: escape $5,000, holding a good module $180, retest $4.

| Rule | Shipped | Escapes | Escape DPPM | Good modules held | Retests | Total cost |
|---|---|---|---|---|---|---|
| Test once | 7,400 | 27 | 3,649 | 91 | 0 | $151,380 |
| Retest until pass (3 tries) | 7,510 | 63 | 8,389 | 17 | 953 | $321,872 |
| Guard band 0.15 dB (3 tries) | 7,346 | 3 | 408 | 121 | 1,344 | $42,156 |
| **Retest Advisor** | **7,379** | **2** | **271** | **87** | **627** | **$28,168** |

Across 12 independently seeded lines (`node scripts/seeds.mjs`, seeds 1000-1011), medians: Retest Advisor 2 escapes / 85 good modules held / 679 retests, versus retest-until-pass 57 / 14.5 / 872 and the guard band 4 / 106 / 1,225. The advisor had the lowest cost on 12/12 lines and fewer escapes and fewer retests than retest-until-pass on 12/12. Retest-until-pass shipped more bad modules than testing once on 12/12. The shown line (seed 1009) was picked because its advisor result (2 escapes) is the 12-seed median.

Calibration of P(fail) at the final decision (shown line): expected vs actual out-of-spec modules per bucket 19.9 vs 18 (10-50%), 91.7 vs 97 (50-90%), 334.2 vs 338 (>90%). The low-risk tail is slightly over-confident: the 0.1-1% bucket expected 0.35 and saw 2 (see limitations).

## Tests

```
node --test tests/disposition.test.mjs   # 10 tests
```
Normal-tail accuracy down to 6σ; simulator determinism and that observations carry no ground truth; the gauge model recovers lane repeatability, re-mate noise, dirty-connector rate and size from golden checks, and the bias track follows the drifting station and recovers after cleaning; causality (rewriting all golden checks after day 8 changes no earlier estimate or decision); posterior monotonicity and extremes; a dirty first attempt is explained away by a clean retest; calibration on 4,000 modules drawn from the fitted model (validates the inference math, including the lane/level factorisation); the value-of-information pruning is exact; baseline rule semantics, scoring identities and the headline relations on the shown line; truth is read only by the scorer.

## Honest limitations

- **Synthetic line.** TDECQ levels, lane defects, Student-t lane noise, re-mate noise, dirty-connector events and the station drift are plausible stand-ins, not Lumilens data. The ground truth exists only because it is a simulation; on a real line, escape rates would come from field returns or a reference-station audit sample.
- **One metric.** Only TX TDECQ per lane. A real disposition would combine outer OMA, extinction ratio, TDECQ over temperature and RX sensitivity/BER; the same posterior machinery extends to a vector of limits but is not built here.
- **Approximations.** Lane offsets are treated as independent given the module level (the within-attempt deviations actually sum to zero), attempts on the same station share a bias uncertainty that is treated as independent, and the model assumes Gaussian lane noise while the simulator uses heavier-tailed noise. This is why the 0.1-1% risk bucket is slightly over-confident.
- **One-step lookahead.** The value-of-information rule looks one retest ahead, not a full dynamic programme over the remaining attempts.
- **Costs are inputs.** The policy is only as good as the escape / held-module / retest costs; that is why they are on the page.
- **UI.** Lumilens's MES and test-floor software are internal and no screenshots are public. The page uses the lumilens.com design system (header, Helvetica Neue type, indigo #18075c and blue #0f47d6, soft blue panels, 24 px radii); the MES navigation items are my assumption.

## Novelty note

Retest-rate and false-failure analytics exist (Keysight PathWave Manufacturing Analytics, yieldHUB's "target retest at bins with a real recovery rate"), guard-banding and retest schemes are studied for IC test, and decision rules under measurement uncertainty are standardised in metrology (JCGM 106). I found nothing public from Lumilens about how retest disposition is done, and no product that computes a per-module posterior across all attempts of a multi-lane optical module, separates dirty-connector readings from real margin, corrects with golden-unit station tracks, and decides retests by value of information. That combination for transceiver TX test is the contribution here.

## Run it

Static site, no build step. Open `index.html` through any static server. `node scripts/report.mjs` regenerates `data/results.json`, `node scripts/seeds.mjs` regenerates `data/seeds.json`, and `node scripts/sample.mjs` writes `data/sample_attempts.csv` (the first 200 modules' readings).
