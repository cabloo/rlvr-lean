# Ladder loop, L4t2 (two more one-change checks on the stored rounds, seed 0, from the larger pretrained model): THE GAIN OF TRAINING ON OWN PROOFS MIXED WITH PUBLISHED ONES IS A SECOND PASS OVER THE PUBLISHED ONES. The published rows alone do as well on the start model's failures (16.2 against 15.7 successes per 1,000, against 8.4 for the start model) and better beside: 252 goal problems solved, 124 reliably, 11 of the 81 never solved. Reward-weighted rows with rehearsal pass every read and are the first model above the pretrained one on all goal problems per attempt (+0.0104 [+0.0014, +0.0194]), but whose gain that is, the rounds' rows or the published ones, is not separated. By the rule fixed beforehand, no six-round arm with rehearsal rows is run.

Labelled **pretrained on published proofs**, as everything built on that file is.

Spec: `docs/spec/ladder-loop.spec.md`, "L4t2: two more one-change checks, on what L4t left open", the reads,
the branches and the rule for the arm fixed before the run. Why: `ladder-l4t-RESULT.md` (the old rule narrows at
rank 64 too; two rules give the breadth back; what it does not say). Run
2026-10-09 (UTC) at Lean v4.27 through the `oeis` pool, on the GPU box: tasks
`rlvr_lean/ladder_l4_rows2_r64_smoke_r1`, `rlvr_lean/ladder_l4_rows2_r64_seed0_r1` (75 minutes; failed in its
second training, below) and `rlvr_lean/ladder_l4_rows2_r64_seed0_r2` (64 minutes; resumed there), from 388316745
and f0faa6624. One seed.

**What was run. No new round.** Two models, each trained FROM `pre_r64` at rank 64, one pass, and measured as
L4t's models were (three rungs; the goal set G, 392 problems, 32 and 61 attempts a problem):
- `rehearse_only`: the 3,422 published proofs of L4t's `rehearse` set and nothing else. `pre_r64` was pretrained
  on them once, with 21,444 others; this is a second pass over a seventh of that file. 16 minutes of training.
- `reward_rehearse`: L4t's `reward_rows` set (1,349 rows of the rounds, weighted to the hard problems) and 1,262
  published proofs (the first of the same draw), 2,611 rows.
All four checks pass for both (the training ran; it still writes proofs; at most 0.008% of a set without an
answer; no barred row).

## The reads fixed before the run

Three reads of a model against `pre_r64`: it ADDS (G′, the 192 goal problems `pre_r64` did not solve in its first
32 attempts, over the 61 fresh attempts: interval above zero), it is NOT NARROWER (goal problems solved in 93
attempts: not lost more than gained under 0.05), it HOLDS THE LONG PROOFS (the 230 problems of 4 lines or more,
per attempt: interval not wholly below zero).

| | `pre_r64` (start) | `old_rule` | `reward_rows` | `rehearse` | `rehearse_only` | `reward_rehearse` |
|---|---|---|---|---|---|---|
| Rows of the rounds / published rows | | 3,422 / 0 | 1,349 / 0 | 3,422 / 3,422 | 0 / 3,422 | 1,349 / 1,262 |
| G′, successes per 1,000 (61 fresh) | 8.4 | 13.8 | 14.5 | 15.7 | 16.2 | 19.1 |
| ADDS: minus `pre_r64` | | +0.0054 [+0.0000, +0.0127] | +0.0062 [+0.0017, +0.0110] | +0.0073 [+0.0021, +0.0136] | +0.0079 [+0.0037, +0.0126] | +0.0108 [+0.0047, +0.0182] |
| Goal problems solved in 93 | 245 | 217 | 241 | 234 | 252 | 247 |
| NOT NARROWER: gained, lost | | 25, 53 | 32, 36 | 27, 38 | 35, 28 | 32, 30 |
| HOLDS: 4 lines or more, minus `pre_r64` | | −0.0064 [−0.0179, +0.0062] | −0.0101 [−0.0190, −0.0012] | +0.0081 [−0.0030, +0.0200] | +0.0093 [−0.0011, +0.0198] | +0.0075 [−0.0029, +0.0182] |
| The three reads | | fails two | fails the third | all three | all three | all three |
| All of G per 1,000 (93 attempts) | 80.0 | 78.9 | 84.0 | 85.0 | 86.8 | 90.4 |
| minus `pre_r64`, per attempt (the last two: my bootstrap, not a fixed read) | | −0.0011 [−0.0114, +0.0096] | +0.0040 [−0.0046, +0.0127] | +0.0050 [−0.0045, +0.0148] | +0.0068 [−0.0023, +0.0155] | +0.0104 [+0.0014, +0.0194] |
| Solved reliably | 107 | 97 | 108 | 114 | 124 | 118 |
| Of the 81 never solved before | 7 | 4 | 8 | 4 | 11 | 7 |
| Share of distinct attempts on G | 0.818 | 0.723 | 0.772 | 0.737 | 0.758 | 0.791 |

- **Question 1: THE GAIN IS A SECOND PASS OVER PUBLISHED PROOFS.** On G′ over the 61 fresh attempts `rehearse`
  minus `rehearse_only` is −0.00051 [−0.00606, +0.00512], not above zero, and `rehearse_only` itself adds
  (+0.00786 [+0.00367, +0.01264]). The mix's gain is not shown to be the loop's.
- **Question 2: `reward_rehearse` ADDS, KEEPS THE BREADTH AND HOLDS THE LONG PROOFS.**
- **The arm's rule: NO ARM WITH REHEARSAL ROWS**, because Question 1 reads a second pass: such an arm could not
  show that the loop adds. The arm (L4c) is built and is not run.

## What it says

- **More training on published proofs still helps this model, and quickly.** Sixteen minutes over a seventh of
  the file it was pretrained on made the broadest and most reliable model measured here: 7 more goal problems
  solved, 17 more solved reliably, the long proofs kept (4 to 7 lines: 107 solved against 100; 8 or more: 28
  against 25), and 11 of the 81 never-solved problems against 7. It is not narrower by any read. With L4r (the
  adapter's size capped the pretrained model) this is the second time the PRETRAINING, not the loop, turned out
  to have room.
- **Adding the rounds' rows the old way to that rehearsal made it worse, not better:** 234 problems solved
  against 252, 114 reliably against 124. The old rule costs breadth even when diluted one to one.
- **The reward-weighted rows with rehearsal are the best per attempt** (90.4 per 1,000; 19.1 on the start model's
  failures, 2.3 times the start model) and keep the breadth and the variety of attempts (0.79 distinct against
  0.82). This is the first trained model above `pre_r64` on all goal problems with an interval clear of zero,
  which is the bar L4t's two rules missed.

## What it does not say

- **Whether the loop's rows add anything.** `reward_rehearse` against `rehearse_only`: +0.0029 [−0.0029, +0.0091]
  on G′ and +0.0036 [−0.0050, +0.0126] on all of G (my bootstrap, not one of the fixed reads): not separated, and
  the two differ in two things at once (rows of the rounds, and 1,262 published rows against 3,422). Its own
  control, 1,262 published rows alone, was not run.
- **How far a second pass goes.** `rehearse_only` is 14% of one. More may help more, or the gain may be that of
  any short further training of a model whose one pass ended at a decayed learning rate. L4p measures the dose.
- **Anything at more than one seed.** The differences among the three models that pass every read are inside
  each other's intervals.

## What follows

No arm, by the rule. The outcome is with the owner. Meanwhile L4p runs (spec, fixed before any run): the dose of
a second pass, at 50% and 100% of the pretraining file along the order `rehearse_only`'s rows begin, read on all
of G against `rehearse_only`, with the overfitting reads on every branch. Every way on needs it: a pretraining
recipe needs to know where more passes stop helping, and a loop arm cannot be read against a start model that
more of its own pretraining still improves.

## A fault of the first run

The second training of the first task ran out of GPU memory at the cap: 10.65 GiB allocated, 0.9 GiB reserved
and not allocated, 0.55 GiB asked. The room was there and was in pieces; the trainings before it had fitted the
same rows in another order with nothing to spare. VOID for that model; the first model's stored steps were kept.
Fixed in f0faa6624 without changing any number a training computes (the allocator's segments may grow; its cache
is handed back between rows, 2,547 times in the training that had failed). The task queued again resumed at that
training, which fitted at 12.25 GB reserved.
