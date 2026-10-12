# Ladder loop, L5m (the best model, measured in full): THE GAIN HOLDS ON THE HELD-ASIDE SET AND ON FRESH ATTEMPTS. The model that read 144.77 successes per 1,000 attempts on the selected-on sampling reads 77.5 against 61.35 on 600 held-aside problems and 145.62 against 108.23 on fresh attempts at the goal problems, each against the model it was trained from. It is a gain in how often problems are solved, not in how many

Labelled **pretrained on published proofs**, as everything built on that file is.

Spec: `docs/spec/ladder-loop.spec.md`, "L5m: the best model, measured in full" (fixed 2026-10-12, before L5c's
read and before any run of this stage), its order amended at `034ee05b7`, and its "Made exact by the build". Run
from `a2bab6705`. NO TRAINING. One task on the GPU box at Lean v4.27 through the
`oeis` pool (16 workers by day): `rlvr_lean/ladder_l5_measure_seed3_r1` (task c8e2546c), 12:18 to 14:17 UTC
2026-10-12, `done`. Its smoke run passed first (`ladder_l5_measure_smoke_r2`, task f791c4a4; the first smoke task was
refused at its first step for a gap of the smoke world alone, fixed at `a2bab6705` with the real stage unchanged).
**All checks pass; the run is not INCONCLUSIVE.** Lean answered: at most 0.0001 of a set without a verdict.

**Why this stage.** L5f's `published` (the first 800 published rows of seed 3's order, trained from the
twice-pretrained model `second_full` in four orders, the averaged adapter) read 144.77 per 1,000 on G's first
sampling, the one set of 12,544 episodes more than a dozen models have been compared on. The highest of many figures
on one set carries the selection. L5m measures that model where nothing was selected.

## What was run

THE CANDIDATE: L5f's stored averaged `published` of seed 3, only read. THE BAR: `second_full`, the model it was
trained from (110.25 on G's first sampling). The candidate was measured as L5's models are: 8 episodes on each of
712 rung problems; 32 attempts at each of the 392 goal problems (its first sampling MEASURED AGAIN, sampling seed
1001) and 61 fresh attempts (seed 1020); 16 attempts at each of the 600 held-aside problems (seed 1051). The bar's
rungs and goal samplings are L4p's stored attempts at the same seeds; on the held-aside set it was measured here from
its stored adapter. Every difference is the candidate minus `second_full`, successes per attempt, paired by problem,
with a 95% bootstrap over problems.

## The reads fixed before the run

| read | result | per 1,000 attempts | difference [95% interval] |
|---|---|---|---|
| **THE HELD-ASIDE SET** (the primary; 600 problems, 16 attempts) | **THE GAIN HOLDS ON THE HELD-ASIDE SET** | 77.50 against 61.35 | +16.15 [+7.71, +24.69] |
| **FRESH ATTEMPTS** (the 392 goal problems, 61 fresh attempts) | **THE GAIN HOLDS ON FRESH ATTEMPTS** | 145.62 against 108.23 | +37.39 [+26.43, +48.39] |
| The measurement again (its first sampling here minus its stored one) | names nothing | 146.44 against 144.77 | +1.67 [-3.03, +6.54] |

**What follows, by the rule fixed in advance:** THE CANDIDATE IS THE BEST MODEL MEASURED IN FULL: THE NEXT CYCLE IS
PROPOSED FROM IT. It is said without the caution of selection.

## Beside, naming nothing

| | the candidate | `second_full` | difference |
|---|---|---|---|
| All of G over the 93 attempts, per 1,000 | 145.90 | 108.93 | +36.98 [+26.41, +47.48] |
| G' (the 192 goal problems the start model `pre_r64` does not solve), fresh attempts | 42.09 | 26.90 | +15.20 [+6.32, +24.68] |
| The 74 long goal problems (8 lines or more), per 1,000 | 62.48 | 46.06 | +16.42 [-0.44, +33.42] |
| Goal problems solved in 93 attempts | 264 | 267 | gained 23, lost 26 (sign test p = 0.78) |
| Goal problems solved reliably (half the attempts or more) | 32 | 16 | gained 20, lost 4 |
| Of the 147 goal problems `pre_r64` never solved in 93 | 41 | 42 | gained 12, lost 13 |
| Long goal problems solved | 32 | 31 | gained 4, lost 3 |
| Held-aside problems solved at least once in 16 | 264 | 247 | gained 96, lost 79 (sign test p = 0.23) |
| Rung below, per 1,000 (166 problems) | 332.83 | 281.63 | +51.20 [+13.55, +88.10] |
| Rung in (155 problems) | 472.58 | 445.97 | +26.61 [-16.94, +69.35] |
| Rung above (391 problems) | 845.91 | 803.39 | +42.52 [+22.38, +62.98] |
| Share of distinct attempts on G | 0.6995 | 0.7981 | |
| Share of distinct attempts on the rungs | 0.7058 | 0.8165 | |
| Attempts timed out in Lean, G / held-aside | 2.55% / 0.77% | 2.25% / 0.63% | under a point apart |

## What it says

- **The gain is not an artefact of the selection.** On problems no model was chosen on, the candidate is 26% above
  the model it was trained from per attempt (77.5 against 61.35); on fresh attempts at the goal problems it is 35%
  above (145.6 against 108.2). Measured again at the same sampling seed, its first figure comes back (146.44 against
  144.77).
- **It is reliability, not reach.** Over 93 attempts the candidate solves 264 goal problems against 267: no more
  problems, and a different 23 for a lost 26. What rises is how often a problem it can solve is solved: problems
  solved in half the attempts or more go from 16 to 32. On the held-aside set it solves 264 problems against 247 in
  16 attempts, which the sign test does not separate (p = 0.23).
- **It repeats itself more.** The share of distinct attempts falls from 0.80 to 0.70 on G and from 0.82 to 0.71 on
  the rungs: a sharper model, with less variety between attempts.
- **No rung pays for it.** All three rungs read above the bar (the middle one with an interval that holds zero); the
  easy rung, where the larger adapter once cost, is +42.5.

## An offline look after the run: reach by the number of attempts (stored attempts only; names nothing)

Expected goal problems solved when k of a model's 93 stored attempts at each problem are drawn without replacement
(exact, from the stored counts of verified attempts; nothing fitted, nothing sent to Lean):

| attempts a problem | 8 | 16 | 32 | 46 | 64 | 93 |
|---|---|---|---|---|---|---|
| the candidate | 170.0 | 205.4 | 234.3 | 246.4 | 255.7 | 264 |
| `second_full` | 147.3 | 187.1 | 222.8 | 239.2 | 252.8 | 267 |

- **The candidate reaches the same problems sooner, not more of them.** It is 23 problems ahead at 8 attempts, 18 at
  16, 11.5 at 32, 3 at 64, and 3 behind at 93. From 64 to 93 attempts `second_full` still adds 14.2 problems and the
  candidate 8.3: at budgets past 93 the less sharp model may reach more. That is an extrapolation and is not measured.
- **A budget split between the two models reaches a little more at a large budget, nothing at a small one.** 93
  attempts split 46 and 47 are expected to solve 272.2, against 264 and 267 for one model alone; 64 split evenly
  261.0 against 255.7 and 252.8; 32 split evenly 235.5 against 234.3 for the candidate alone (the earlier look at 32
  attempts read the same: no lever there).
- By at least one of the two in its 93 attempts (186 in all): 290 goal problems; by both 241; by neither 102. The 23
  only the candidate solves and the 26 only `second_full` solves are nearly all problems solved once to six times in
  93.

## What it cannot say

- One candidate, one seed, one draw of 800 rows, and `second_full` is one training. The spread between such models
  over seeds is not measured here.
- The bar is the model the candidate was trained from. Nothing here compares it with a model trained on the model's
  own proofs on the held-aside set (L5f's `shallow` and `deep` were measured on G's first sampling alone, where they
  solve 245 and 236 problems in 32 attempts against the candidate's 235).
- What made the gain (the 800 more rows or the average of four trainings) is L5c's question; whether proofs the
  candidate itself finds add to it is L6b's.
- The timed-out attempts were not checked a second time: no two sets of one kind are more than a point apart, so
  the stage's rule asks for none.
