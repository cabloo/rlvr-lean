# Ladder loop, L4p part 1 (the matched control, seed 0): THE MODEL'S OWN PROOFS OF HARD PROBLEMS ARE NOT SEPARATED FROM THE SAME PUBLISHED PROOFS ALONE, and cost breadth. On the start model's failures the mix has 19.1 successes per 1,000 against 20.2 for the published rows alone (−0.0011 [−0.0071, +0.0049]); it solves 247 goal problems against 264. The control itself, six minutes of training on 1,262 published proofs the model had seen once, is the best model measured (99.2 per 1,000 against 80.0), and beats a longer second pass: the dose is not monotone and is not understood.

Labelled **pretrained on published proofs**, as everything built on that file is.

Spec: `docs/spec/ladder-loop.spec.md`, "L4p: how much more pretraining? The dose of a second pass" and its
paragraph "Added the same day, after the owner's question": the read fixed before the run. Why:
`ladder-l4t2-RESULT.md` left `reward_rehearse` without its own control, and the owner asked (2026-10-09) whether
the loop "is actually creating a feedback cycle that is able to push the model beyond capabilities that we're
able to get through pre-training alone". Run 2026-10-09 and 10 (UTC) at Lean
v4.27 through the `oeis` pool, on the GPU box: tasks `rlvr_lean/ladder_l4_dose_r64_smoke_r1` and
`rlvr_lean/ladder_l4_dose_r64_matched_seed0_r1` (71 minutes), from 4c1892f95. One seed. Part 2 (the dose at 50%
and 100%) has not run.

**What was run.** One model, `matched_1262`: FROM `pre_r64` at rank 64, one pass (158 optimizer steps) over the
first 1,262 rows of the hash order the rehearsal draw uses, which are exactly the published rows of L4t2's
`reward_rehearse` (held by id and place). Measured as every model of L4t (three rungs; the goal set G, 392
problems, 32 and 61 attempts a problem). Its four checks pass.

## The owner's question, at the size of one pass

`reward_rehearse` is `pre_r64` trained on 1,349 rows of the stored rounds (the model's own proofs, weighted to
problems one solver in eight solved) and those 1,262 published rows. `matched_1262` is `pre_r64` trained on the
1,262 published rows and nothing else. The two differ by the rounds' rows alone.

| | `pre_r64` (start) | `matched_1262` | `reward_rehearse` | `reward_rehearse` minus `matched_1262` |
|---|---|---|---|---|
| G′ (192), the 61 fresh attempts, per 1,000 | 8.4 | 20.2 | 19.1 | −0.0011 [−0.0071, +0.0049] |
| All of G, 93 attempts, per 1,000 | 80.0 | 99.2 | 90.4 | −0.0088 [−0.0178, +0.0003] |
| Goal problems solved in 93 | 245 | 264 | 247 | gained 22, lost 39 (p = 0.04) |
| Solved reliably | 107 | 134 | 118 | |

**The read: NOT SEPARATED AT THIS SIZE** (the fixed read is the first row; it resolves 0.006). Beside it, and
deciding nothing: the mix is lower on all of G, with an interval that ends at zero, and solves fewer problems
with the sign test under 0.05.

## The control itself

`matched_1262` against `pre_r64`: all of G +0.0192 [+0.0104, +0.0279]; goal problems solved 264 against 245
(gained 44, lost 25, p = 0.03); 4 lines or more +0.0227 [+0.0116, +0.0343] (83.4 against 60.6 per 1,000); 8 lines
or more 36.6 against 22.4; the rung in the band +0.081 [+0.043, +0.120], above it +0.059 [+0.039, +0.079]; the
share of distinct attempts on G 0.806 against 0.818. Its verified proofs are longer than the start model's
(median 5 lines against 4; 72% of 4 lines or more against 62%).

**The dose so far is not monotone.** All of G against `pre_r64`: 1,262 rows +0.0192; 3,422 rows (`rehearse_only`)
+0.0068 [−0.0023, +0.0157]; and `rehearse_only` minus `matched_1262` −0.0123 [−0.0206, −0.0040].

## What it says

- **On the stored rounds, the loop's rows are not what helped.** Every gain L4t and L4t2 saw from a set that
  held published rows is matched or beaten by the published rows alone, and the rows of the rounds beside them
  cost breadth.
- **A very small further training on published proofs moves this model a long way.** Why 1,262 rows do more
  than 3,422 is not known.

## What it does not say

- **Anything about fresh rounds or compounding.** The rows were written by rank-16 models built on `pre`, one
  pass, never a second round on top of the first. The six-round arm from `pre_r64` with its own
  `rehearsal_alone` control (L4c) is that test and is running.
- **That any two of these single models differ by more than two trainings of one recipe would.** Every interval
  here is over problems. The spread between trainings is not measured at this rank, and the non-monotone dose
  is a reason to think it may be large. No difference between two trained models here should be read as an
  effect until it is.
- **Whether more published DATA would do what a short second pass does.** Not asked.

## A measurement check made on this result

`pre_r64`'s stored goal attempts hold more Lean timeouts (4.3%) than any model measured since (2.3% to 3.5%), and
a proof that times out is a failed attempt. One pool server answers many times slower when other jobs load its
machine, and `pre_r64` was measured on such a morning. So every timed-out goal attempt of `pre_r64` (1,583) and
of `matched_1262` (1,278) was checked again, Lean only, at background priority, with no measurement running:

| | timed out, checked again | answered now | verified now | per 1,000 stored | with the second check |
|---|---|---|---|---|---|
| `pre_r64` | 1,583 | 407 (26%) | 109 (6.9%) | 80.0 | 83.0 |
| `matched_1262` | 1,278 | 249 (19%) | 71 (5.6%) | 99.2 | 101.1 |

With the second check counted on both sides the difference is +0.0181 where the stored rows give +0.0192, and
the problems solved are 266 against 247. The stored baseline is low by about 0.001 against a later model, and no
more: the result does not rest on it. The same check is owed to any two models measured hours apart before a
difference of a few per 1,000 between them is read.

---

# Part 2 (the dose at 50% and 100%, seed 0): INCONCLUSIVE BY ITS OWN CHECK; read outside it, A WHOLE SECOND PASS OVER THE PUBLISHED PROOFS IS THE BEST MODEL MEASURED (108.9 successes per 1,000 against 80.0, 267 goal problems solved against 245, 139 reliably against 107, 13 of the 81 never-solved), and the spread between two trainings of one kind is large: about 6 per 1,000 and 5 problems.

Run 2026-10-10 (UTC), task `rlvr_lean/ladder_l4_dose_r64_seed0_r1` (333 minutes, the pool at 15 workers on
two servers throughout), from 1d15320ef. `second_half`: `pre_r64` trained again on the first 12,433 rows of the
order (1,555 optimizer steps, 59 minutes); `second_full`: on all 24,866 (3,109 steps, 119 minutes).

**The check that failed.** `second_full` fails "its training ran": the mean loss over the last tenth of its rows
is 0.1385 and over the first tenth 0.1271, a rise of 0.0114 where two standard errors allow 0.0046. By the rule
fixed before the run the model is INCONCLUSIVE and the primary (`second_full` minus `rehearse_only`) names no
branch. The check was the wrong one for this model, and I wrote it: it asks that a loss not rise, which is right
for rows a model has not seen. On a second pass the loss on already-seen rows rises across the pass (`second_half`
0.1291 to 0.1335, inside its allowance) while proof success rises. That is the third time here that the loss on
proof text and proof success part ways (the capacity check; the first pretraining). The verdict stands; what
follows is read outside it.

## The five published-only models from `pre_r64`

| | optimizer steps | all of G, per 1,000 | G′, 61 fresh, per 1,000 | goal problems solved | solved reliably | of the 81 never solved |
|---|---|---|---|---|---|---|
| `pre_r64` (start) | | 80.0 | 8.4 | 245 | 107 | 7 |
| `matched_1262` (5% of a pass) | 158 | 99.2 | 20.2 | 264 | 134 | 11 |
| `rehearsal_alone` (L4c's control, 6%) | 189 | 95.4 | 19.3 | 258 | 135 | 10 |
| `rehearse_only` (14%) | 428 | 86.8 | 16.2 | 252 | 124 | 11 |
| `second_half` (50%) | 1,555 | 99.3 | 17.6 | 255 | 124 | 10 |
| `second_full` (100%) | 3,109 | 108.9 | 26.9 | 267 | 139 | 13 |

`second_half` against `pre_r64` (its checks pass): all of G +0.0193 [+0.0083, +0.0307]; G′ +0.0092 [+0.0042,
+0.0153]; 4 lines or more +0.0258 [+0.0133, +0.0396]; goal problems gained 34, lost 24; no rung below zero.

## What it says

- **A second pass over the same published proofs helps, and the whole pass helps most.** Outside the rule:
  `second_full` minus `rehearse_only` on all of G is +0.0221, and on G′ it has three times the start model's
  successes. The pretraining-only bar for any loop is therefore not 80.0 and 245: it is at least 108.9 and 267,
  and nothing shows it stops there.
- **The dose is not in order, and that is the spread between trainings.** 5%, 6%, 14% and 50% of a pass give
  99.2, 95.4, 86.8 and 99.3. Four trainings that differ in which rows and how many, and in nothing else, have a
  standard deviation of about 6 per 1,000 on all of G, 2 on G′ and 5 goal problems solved. Part 1 called the dip
  at 14% "not understood": it is this.
- **What the spread does to earlier reads.** No difference between two single trained models in L4t, L4t2 or
  part 1 smaller than about twice that spread is an effect: which training rule gives the breadth back, and
  whether reward-weighted rows beat the old rule, are not settled by one training each. What is far outside it:
  every published-only model solves more goal problems (252 to 267) than every model that holds rows of the
  rounds (217 to 247), and the whole second pass on G′ (26.9 against 16.2 to 20.2).

## What it does not say

- **Where pretraining stops.** A third pass was not run, nor the other half of the pool's published proofs.
- **By a rule fixed beforehand, anything about `second_full`:** its figures are read outside its check.
