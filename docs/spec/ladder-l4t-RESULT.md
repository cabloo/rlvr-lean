# Ladder loop, L4t (what should a round train on? seed 0, from the larger pretrained model): THE OLD RULE NARROWS AT RANK 64 TOO, and neither new rule clears the bar set for it. Trained on the stored rounds' proofs, the model solves 217 goal problems against 245 (gained 25, lost 53) and gains nothing per attempt; reward-weighted rows and rows mixed with published proofs both give the breadth back (241 and 234) without a clear gain per attempt on all goal problems. By its own rule the six-round arm (L4b) is not run as specified. On the start model's own failures all three trained models succeed about 1.7 times as often, on no more problems.

Labelled **pretrained on published proofs**, as everything built on that file is.

Spec: `docs/spec/ladder-loop.spec.md`, "L4t: what should a round train on?" with its amendment "L4r's first
branch was taken", read fixed before the run; the rule that names L4b's training rule was fixed in "L4b" before
this run was read. Why: `ladder-l4-RESULT.md` (six rounds from the pretrained model: more reliable, narrower) and
`ladder-l4r-RESULT.md` (the adapter's size capped the pretrained model). Run
2026-10-09 (UTC) at Lean v4.27 through the `oeis` pool, on the GPU box: tasks
`rlvr_lean/ladder_l4_rows_r64_smoke_r1` and `rlvr_lean/ladder_l4_rows_r64_seed0_r1` (290 minutes), from
7956166e1. One seed, a first run by the seed rule.

**What was run. No new round.** Three models, each trained FROM `pre_r64` (the pretrained model at rank 64) at
its rank, one pass, the arm's recipe, on the rows the rank-16 arm's six rounds stored, and measured as the arm's
models were (three rungs; the goal set G, 392 problems, twice: 32 and 61 attempts a problem):
- `old_rule`: the twin's 3,422 one-shot rows, one verified proof for every solved problem (what every arm so far
  trains on);
- `reward_rows`: 1,349 of the rounds' 3,509 rows, each kept with the probability of its problem's reward at the
  target rate 1/10 (1,009 rows of problems solved by 1 solver of 8, 273 by 2, 62 by 3; all 87 assembled proofs);
- `rehearse`: the 3,422 rows and as many published proofs of the pretraining file, 6,844 rows.
Then `hot`: `old_rule` and `pre_r64` sampled again on G at temperature 1.2. Every check passes for every model
(each training took; each model still writes proofs; at most 0.008% of any set was left without an answer; no
barred row). The trainings fitted the card: reserved memory peaked at 12.5, 11.2 and 12.8 GB.

## The reads fixed before the run

| | `pre_r64` (start) | `old_rule` | `reward_rows` | `rehearse` |
|---|---|---|---|---|
| Goal problems solved in 93 attempts | 245 | 217 | 241 | 234 |
| against `pre_r64`: gained, lost (sign test) | | 25, 53 (p = 0.002) | 32, 36 (p = 0.72) | 27, 38 (p = 0.21) |
| against `old_rule`: gained, lost (sign test) | | | 48, 24 (p = 0.006) | 41, 24 (p = 0.046) |
| All of G, successes per 1,000 attempts | 80.0 | 78.9 | 84.0 | 85.0 |
| minus `pre_r64`, per attempt | | −0.00107 [−0.01141, +0.00960] | +0.00395 [−0.00461, +0.01267] | +0.00502 [−0.00453, +0.01478] |
| Solved reliably (half of its episodes of 8) | 107 | 97 | 108 | 114 |
| Share of distinct attempts on G | 0.818 | 0.723 | 0.772 | 0.737 |
| Verified proofs of 8 lines or more, share | 0.254 | 0.125 | 0.119 | 0.180 |

- **`old_rule`, read by itself against `pre_r64`: THE OLD RULE NARROWS AT RANK 64 TOO.** It loses more goal
  problems than it gains with the sign test under 0.05. The cap was not the cause of the narrowing; the training
  rule is. At rank 16 the same rows gave +0.011 per attempt on all of G; at rank 64 they give nothing (−0.001).
- **`reward_rows`: IT GIVES THE BREADTH BACK BY UNDOING THE TRAINING.** More gained than lost against `old_rule`
  under 0.05, and the interval for all of G against `pre_r64` is not above zero.
- **`rehearse`: the same branch**, for the same two reasons.
- **`hot`, read and not branched.** At 1.2 `old_rule` solves 215 goal problems against 217 at 1.0 and has 67.5
  successes per 1,000 against 78.9; `pre_r64` solves 240 against 245 and has 62.4 against 80.0. Temperature costs
  successes and gives no breadth. It is not a lever.
- **L4b, by its own rule:** no rule reads "gives the breadth back and keeps the gain", and `old_rule` narrows.
  That is its third case: **THE ARM IS NOT RUN AS SPECIFIED.**

**A fault in the bar.** "Keeps the gain" asked for more successes per attempt on all of G than `pre_r64`, with
an interval above zero. I set it from rank 16, where the old rule had such a gain. At rank 64 the old rule has
none, so the branch's name, "by undoing the training", says more than was measured: on all of G there was no
gain to undo. The verdicts above stand as written. What follows reads the same rows the way the ARM is read, a
read that was fixed (in L4b) before this run was read; it is secondary here and decides nothing by itself.

## What the same rows say by the arm's read

**On the start model's own failures.** G′ is the 192 goal problems `pre_r64` did not solve in its first 32
attempts; the read is over the 61 fresh attempts.

| | `pre_r64` | `old_rule` | `reward_rows` | `rehearse` |
|---|---|---|---|---|
| G′, successes per 1,000 | 8.4 | 13.8 | 14.5 | 15.7 |
| minus `pre_r64`, per attempt | | +0.00538 [+0.00000, +0.01272] | +0.00615 [+0.00171, +0.01101] | +0.00734 [+0.00205, +0.01358] |
| G′ problems solved in the 61 attempts | 45 | 48 | 48 | 46 |
| The other 200, per 1,000 | 151.2 | 140.1 | 151.3 | 149.3 |
| The other 200, solved in the 61 attempts | 180 | 158 | 168 | 170 |

All three succeed about 1.7 times as often where the start model fails, and two intervals are clear of zero.
`reward_rows` and `rehearse` pay nothing per attempt on the other 200 problems; `old_rule` pays 11 per 1,000.

**The selection favours them, and a set that does not says the same.** G′ is chosen by `pre_r64`'s own first
sampling, so any model that differs from it gains there by the choice alone. On the 154 goal problems the
rank-16 `pre` does not solve in 93 attempts (no rank-64 model chose them) the four models have 4.9, 8.5, 7.5 and
7.6 successes per 1,000 over 93 attempts, and solve 38, 28, 37 and 31 problems. No interval was computed for
this set. **More successes, on no more problems:** the training concentrates successes on the hard problems a
model can already reach now and then.

**Long proofs are what the old rule loses.** By the length of the shortest published proof, successes per 1,000
attempts and problems solved in 93:

| | 1 line (37) | 2 to 3 (125) | 4 to 7 (156) | 8 or more (74) |
|---|---|---|---|---|
| `pre_r64` | 116.2, 26 | 104.9, 94 | 78.8, 100 | 22.4, 25 |
| `old_rule` | 128.5, 25 | 109.7, 93 | 71.3, 82 | 18.5, 17 |
| `reward_rows` | 147.3, 26 | 126.8, 96 | 67.6, 100 | 14.4, 19 |
| `rehearse` | 130.2, 26 | 101.7, 92 | 89.2, 97 | 25.6, 19 |

On the 230 problems of 4 lines or more, per attempt against `pre_r64`: `old_rule` −0.00636 [−0.01786, +0.00622]
and 99 solved against 125 (gained 12, lost 38, p = 0.0003); `reward_rows` −0.01014 [−0.01903, −0.00117];
`rehearse` +0.00809 [−0.00295, +0.01996]. The rounds' own proofs are short (median 3 lines, 11% of 8 or more),
and a model trained on them writes shorter proofs: the share of its verified proofs with 8 lines or more halves.
`reward_rows` shifts the same way and more (it gains clearly on 1 to 3 lines: +0.031 and +0.022). `rehearse`,
whose other half is published proofs (median 5 lines over the set), holds the long proofs.

**The rungs** (8 episodes a problem, minus `pre_r64`): `old_rule` in the band +0.071 [+0.026, +0.118];
`reward_rows` in the band +0.100 [+0.062, +0.140] and above it +0.035 [+0.015, +0.055]; `rehearse` none clear.

**Never solved by anything stored.** Of the 392 goal problems 81 had been solved by nothing before the rank-16
arm. In 93 attempts `pre_r64` solves 7 of them, `old_rule` 4, `reward_rows` 8, `rehearse` 4, the two samplings at
1.2 solve 6 and 7; with the rank-16 arm's two models (4 and 3) that is 23 of the 81, and 58 are left. Eleven
are new with this run. Each is solved once or twice in 93 attempts: reached, not reliable.

**A second model is worth a little more than more attempts of the first.** Goal problems solved in 186 attempts:
`pre_r64` with its own sampling at 1.2, 263; `pre_r64` with `old_rule`, `reward_rows` or `rehearse`, 270, 277 and
272. A mixture at 93 attempts (32 of one model and 61 of the other) solves 241 to 252 against `pre_r64`'s 245.
The four models' 372 attempts together solve 293.

## What it says

- **The loop's own proofs, as every arm has used them, cost the pretrained model its long proofs,** at either
  adapter size. That is what "narrower" was.
- **What they buy is real and narrow:** about 1.7 times the successes on problems the start model rarely solves,
  seen on a set the start model did not choose, and no more problems solved.
- **Two changes of the training rows remove most of the cost.** A third of the rows, weighted to the hard
  problems, keeps the breadth and loses the long proofs per attempt. Mixing in published proofs keeps the long
  proofs and has the most problems solved reliably (114 against 107).
- **Neither is shown to beat the pretrained model on all goal problems per attempt** (+0.004 and +0.005, both
  intervals through zero; they resolve about 0.009).

## What it does not say

- **Whether `rehearse`'s gain is the loop's.** Half its rows are published proofs `pre_r64` has already seen
  once; a second pass over those alone might show the same. This is L4t2's first question.
- **Whether `reward_rows`' breadth is from the weighting or from training on fewer rows.** Not separated.
- **What fresh rounds would give.** The rows were written by rank-16 models built on `pre`, at their frontier,
  and each row's k is theirs. The arm from `pre_r64` would make its own.
- **Anything at more than one seed.** The choice between the two rules that follows is made on reads of one goal
  set at one seed, and is exploratory.

## What follows

L4t2 (spec, fixed before any run): two more one-pass models on the stored rounds, about an hour each:
`rehearse_only` (the published rows of `rehearse` alone) and `reward_rehearse` (the reward-weighted rows with as
many published rows as it has one-shot rows). Three reads of a model against `pre_r64`, the arm's own: it adds on
G′, it is not narrower, it holds the long proofs. The arm is then run from `pre_r64` with a rule that passes all
three, and carries its own separation of the loop's rows from the rehearsal; or, if the rehearsal gain is a
second pass over published proofs, no arm with rehearsal rows is run and that goes to the owner.

## A fault of the pool during the run, and what it cost

Two of the pool's four Lean servers had not been restarted for a day and leak processes as they work (a fix is
installed on one server and waits on the others). One was paused
by another session at 17:10 UTC after it began to drop under load; the other reached its process limit at about
18:20 UTC and answered every check with an error until I paused it a minute after my guard reported it. The
last measurement (`pre_r64` at 1.2) ran through both. No attempt of any set was left without an answer (the
largest share is 0.008%), so nothing was read from a damaged set; the two samplings at 1.2 took 44 and 64
minutes where a measurement at 1.0 took 39 to 54.
