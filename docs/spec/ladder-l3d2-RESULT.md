# Ladder loop, L3d Step 2 (the loop with assembly in the round, six rounds, seed 0): not shown, with the power to have seen it; 156 assembled proofs in the training set change nothing on long-proof problems, because once minimised they are as short as the model's own; the six rounds themselves take the loop further than three did, on short-proof problems only

Spec: `docs/spec/ladder-loop.spec.md`, "L3d: train on what the episode reaches" (Step 2 and its "Made
exact by the build"). Why: `ladder-l3c-RESULT.md` and its addendum, `ladder-ceiling-RESULT.md`.
Run 2026-10-08 (UTC) at Lean v4.27 through the `oeis` pool, on the GPU box:
tasks `rlvr_lean/ladder_l3d2_smoke_r1` (4 minutes) and `rlvr_lean/ladder_l3d2_seed0_r1` (229 minutes),
from da578c7f8. One seed, a first run by the seed rule.

**What was run.** The loop at t = 1/10 for six rounds of 1,000 problems, with the Lean-only assembly after
every batch (a problem only assembly resolved counted as k = 1 for the challenger), the harvest H0 in the
training set from round 1, and after round six a twin: `with` is the round-six model (3,362 rows), `without`
is trained from the base on its 3,206 one-shot rows alone, in the same order with the others left out. The
two differ by 156 assembled proofs: 98 the rounds assembled and 58 of H0's 92. (The challenger proposed 72
of H0's problems once the solver had trained on their proofs, and the rounds resolved 34 of them themselves:
26 by an attempt, 16 of those once in 8, and 8 by assembly again.)

**Verdict by the read fixed before the run: NOT SHOWN. Assembled proofs teach less than published ones,
each.**

- **The four checks pass.** 156 assembled proofs between the two (150 asked); both trainings took (0.179
  against 0.229 and 0.182 against 0.238, last tenth against first); neither model stopped writing proofs (0
  and 1 of 5,832 rung attempts without an answer); every assembled row once in `with`'s record and none in
  `without`'s.
- **Primary (the 230 goal problems whose shortest published proof is 4 lines or more, successes per attempt
  over 93 one-shot attempts, `with` minus `without`): −0.00019 [−0.00122, +0.00061].** 2.66 per 1,000 against
  2.85 (57 successes against 61 in 21,390 attempts each).
- **Could it have seen a gain?** Yes. At the ceiling's figure for a published proof (+0.000013 per attempt
  each) 156 proofs would give +0.0020; this run resolves about ±0.0009. The interval's upper end, +0.0006,
  puts an assembled proof at under a third of a published one.
- **One seed, and no more are run for this read:** it is not positive, and it is bounded where it matters.

## `with` against `without`

| | `without` | `with` | Difference |
|---|---|---|---|
| Goal problems of 4 lines or more, per 1,000 attempts | 2.85 | 2.66 | −0.00019 [−0.00122, +0.00061] |
| All of G (392), per 1,000 attempts | 13.0 | 13.6 | +0.00058 [−0.00069, +0.00181] |
| 1 line (37) | 38.4 | 39.5 | +0.0012 [−0.0055, +0.0081] |
| 2 to 3 lines (125) | 24.2 | 26.0 | +0.0018 [−0.0010, +0.0048] |
| 4 to 7 lines (156) | 4.1 | 3.9 | −0.0003 [−0.0018, +0.0008] |
| 8 lines or more (74) | 0.15 | 0.15 | 0 |
| Goal problems solved at 93 attempts | 68 | 72 | gained 13, lost 9 (p = 0.52) |
| Solved reliably (half of 11 episodes of 8) | 19 | 23 | |
| Rung below the band / in it / above it | 0.115 / 0.431 / 0.870 | 0.111 / 0.440 / 0.877 | −0.005, +0.008, +0.007 (each interval holds zero) |
| Attempts on G that open with a `have` | 18.3% | 23.8% | |

Nothing separates the two. `with` does write differently (it opens with a `have` more often), and that does
not turn into more long-proof problems solved.

## Why: what an assembled proof is, once it is minimised

| Training rows of the round-six model | Rows | Proof lines: median, nine tenths within, longest | Mean loss when trained on |
|---|---|---|---|
| The model's own one-shot proofs | 3,206 | 3, 6, 16 | 0.189 |
| Assembled in the rounds | 98 | 3, 6, 38 | 0.336 |
| H0 (assembled from stored attempts) | 58 | 4, 9, 12 | 0.299 |
| (The ceiling's published proofs, for scale) | 8,000 | 8, 18, 98 | 0.265 at the start |

- **They are not longer proofs.** As assembled the rounds' proofs have a median of 7 lines; with the pool
  blocks the closing step does not need taken out, 3. Assembly solves problems that have a short proof the
  model's 8 attempts did not happen to write whole: one attempt had the fact, another the closing step.
- **They are newer to the model than its own proofs** (a loss of 0.34 against 0.19), about as new as a
  published proof is. But new in the way a missed combination is new, not in the way a longer argument is.
- So the ceiling's lesson holds and is narrower than "proofs beyond one-shot reach": what teaches is proofs
  that carry steps the model does not take, and the loop's search, in this form, does not make those.

## The six rounds, by round

| Round | Solved by an attempt | Only assembly solved | Share of the unsolved | Picks' mean pass rate | Refutations among its training rows |
|---|---|---|---|---|---|
| 1 | 585 | 12 | 2.9% | 0.25 | 27% |
| 2 | 655 | 19 | 5.5% | 0.35 | 8% |
| 3 | 472 | 17 | 3.2% | 0.24 | 5% |
| 4 | 427 | 24 | 4.2% | 0.19 | 6% |
| 5 | 567 | 8 | 1.8% | 0.29 | 11% |
| 6 | 500 | 18 | 3.6% | 0.24 | 10% |

- **The yield does not grow with training:** 8 to 24 a round, 98 in six rounds (3.5% of the 2,794 problems
  the attempts left unsolved; a pool stood on 1,470 of them).
- **The challenger stays on target for six rounds** (mean pass rate of its picks 0.19 to 0.35), as it did
  for three at this target.
- **Assembly's cost:** 4,132 pool checks, 10,345 closer checks and 639 minimisation checks over the six
  rounds, beside about 49,000 attempt checks, and no check made again for an error's position (the round's
  own answers are kept). 900 closer checks ran to the 30 s limit. A round took a quarter to half an hour
  with it; the whole run 229 minutes.

## Six rounds against three (not this stage's question; read from the same measurement)

Both six-round models, with or without the assembled proofs, stand well above the stored three-round model
of this seed on the goal set, and all of it is on short proofs:

| Goal set, seed 0 | Base | Three rounds (stored) | Six rounds (`without`) | Six rounds (`with`) |
|---|---|---|---|---|
| Successes per 1,000 attempts, all of G | 4.0 | 7.2 | 13.0 | 13.6 |
| 1 line | 9.9 | 23.0 | 38.4 | 39.5 |
| 2 to 3 lines | 4.7 | 11.4 | 24.2 | 26.0 |
| 4 to 7 lines | 3.5 | 3.4 | 4.1 | 3.9 |
| 8 lines or more | 0.9 | 0.4 | 0.15 | 0.15 |
| Solved at least once / reliably (11 episodes) | 57 / 3 | 59 / 15 | 68 / 19 | 72 / 23 |

The three-round model was trained on 1,685 proofs and these on 3,206: the picks and the order differ too, so
this is not a paired read of "more rounds". It says the loop had not stopped at three rounds where a short
proof exists, and that at 4 lines or more six rounds do what three did, which is nothing.

## With assembly on top (Lean only, after the run)

Each model's stored goal attempts in 11 episodes of 8, attempts alone and then with the episode's Lean-only
assembly (`tools/ladder_goal_assembly.py`):

| Goal set, seed 0 | Episodes resolved (of 4,312) | By an assembled proof | Solved at least once | Reliably | 4 lines or more, solved at least once |
|---|---|---|---|---|---|
| The base | 117, then 164 | 48 | 57, then 77 | 3, then 3 | 22, then 29 |
| Three rounds (stored) | 193, then 232 | 45 | 59, then 77 | 15, then 15 | 21, then 26 |
| Six rounds, `without` | 281, then 333 | 58 | 68, then 84 | 19, then 21 | 18, then 23 |
| Six rounds, `with` | 302, then 366 | 69 | 72, then 92 | 23, then 26 | 20, then 29 |

The best system the loop has under the rule as it stood is the six-round model with assembly at inference:
92 goal problems solved at least once and 26 reliably, where the base alone had 57 and 3. Assembly adds about
twenty problems of reach to any of these models and two or three reliable ones to the six-round models. (The
ceiling's 8,000-proof model, from its stored attempts: 242 then 262, and 106 then 119.)

## What this says and does not say

- **It says the loop's own search, as assembly, does not supply what published proofs supply.** The test
  could see a gain of the size in question and saw none; the reason is visible in the proofs themselves.
- **It does not take back L3c.** Assembly still reaches problems at no cost in generations. It is a search
  step worth keeping in an episode, and not a source of training text that moves one-shot ability.
- **It does not say no search could.** A search that builds longer arguments (lemmas proposed for a goal,
  not recombined from failed attempts) is another thing, and is not built.
- **One seed.** A bounded null at one seed; the rounds' own numbers (the table by round, six against three)
  are one run's.

## For the owner

Under the rule as it stands the loop now has three measured parts: its own proofs make short-proof problems
reliable, more with more rounds; assembly adds reach at inference and nothing to training; and nothing it
makes of its own moves problems that need 4 lines or more. The one thing measured that does is training on
proofs written by stronger provers (`ladder-ceiling-RESULT.md`: 106 goal problems reliably, against 19 to 23
here). The decision put there was answered the same day: the owner chose both arms. This run is the pure arm
already made, six rounds of the loop from the base. The other arm, the same six rounds from a model
pretrained on published proofs, is L4 in the spec.

## Files

`experiments/rlvr_lean/ladder_l3d2_seed0_r1/steps/` (the report `report_ladder_l3d2.json`, each batch's
`assembly_r<r>_b<b>.jsonl`, each round's `training_examples_r<r>.jsonl` with its rows' origins, the loss of
every row of every training); the arm `gpu/ladder_assembly.py` and `gpu/ladder_l2.py`, the stage
`gpu/ladder_l3d2.py`, the report `reporting/ladder_l3d2.py`; the harvest `data/ladder_l3d/harvest_h0.jsonl`
(not shipped in this copy: `tools/ladder_harvest.py` writes it).
Both adapters are kept on the box (`adapters/m6`, `adapters/without`).
