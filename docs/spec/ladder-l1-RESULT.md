# Ladder loop, L1 (one round, three seeds): the challenger aims well and its picks beat a random draw on every rung; the rung below the band rose 0.9 points with an interval that touches zero; reach did not clearly move

Spec: `docs/spec/ladder-loop.spec.md`, milestone L1 ("L1's read, fixed now"). Run 2026-10-04 at Lean v4.27
through the `oeis` pool (37 workers), on the
GPU box. Tasks: `rlvr_lean/ladder_l1_seed0_r1`, `ladder_l1_seed1_r2` (its per-problem files:
`ladder_l1_seed1_r3`), `ladder_l1_seed2_r1`. Data: `src/rlvr_lean/data/ladder_l0`, `ladder_l1` (not shipped in
this copy).

**Verdict.** Every seed selected the pre-registered branch ESCALATE (a primary at or above zero), and no VOID
condition held. At three seeds the primary is **+0.0090 [−0.0003, +0.0183]**: positive in every seed, small,
and not separated from zero. By the spec the next step is L2 (three rounds). **The run did not show that one
round takes the model past what the base could do:** reach on the goal set is within the base's own luck.

## What ran

One round at target rate t = 1/4, two arms that differ only in who chooses the round's 1,000 problems from
the same 20,000 candidates: the **challenger** (a predictor of the base's pass rate, fitted on L0's base map,
choosing by expected reward) or a **random draw**. Each problem got 8 episodes from the base; the arm then
trained one pass, native format, on one randomly chosen verified proof of every problem with k ≥ 1; the trained
model and the base afresh were measured on the held-out rungs (8 episodes), on the goal set G (32) and again on
the round's own problems (8). Until L3 an episode attempts only the side a certificate does not rule out; 2%
of problems kept both sides and no soundness alarm was raised. About 104,500 attempts were checked per seed,
with 1 or 0 left unanswered.

## The pre-registered read, three seeds

Every difference is paired by problem and averaged over the seeds; intervals are a bootstrap over problems.

| Held-out rung (base's placing rate) | Challenger arm − base | Random arm − base | Challenger − random |
|---|---|---|---|
| **Below the band** (166 problems; base about 0.067) | **+0.0090 [−0.0003, +0.0183]** | −0.0035 [−0.0105, +0.0038] | +0.0126 [+0.0033, +0.0221] |
| In the band (155; about 0.270) | +0.0435 [+0.0261, +0.0608] | +0.0159 [+0.0024, +0.0296] | +0.0277 [+0.0137, +0.0419] |
| Above the band (391; about 0.767) | +0.0514 [+0.0436, +0.0595] | +0.0214 [+0.0148, +0.0282] | +0.0299 [+0.0228, +0.0373] |

By seed, challenger arm: below +0.0053, +0.0120, +0.0098; in +0.0379, +0.0371, +0.0556; above +0.0566,
+0.0435, +0.0540. Random arm: below +0.0023, −0.0030, −0.0098.

## Reach on the goal set (392 problems the base never resolved in its 32 placing episodes)

| | Base afresh | Challenger-trained | Random-trained |
|---|---|---|---|
| Problems resolved, by seed (32 episodes each) | 37, 22, 26 | 27, 31, 35 | 39, 28, 35 |
| Successful episodes of 37,632 | 133 | 166 | 155 |
| Problems resolved in at least one seed | 54 | 54 | 62 |

- Share of G resolved, trained minus base: challenger arm +0.0068 [−0.0085, +0.0221]; random arm +0.0145
  [+0.0026, +0.0272]. Success per episode: +0.0009 [−0.0001, +0.0019] and +0.0006 [−0.0001, +0.0013].
- **The base's own count swings from 22 to 37 between seeds.** That is the luck every reach number here must
  be read against, and it is as large as any difference in the table.
- Of the 338 problems the base resolved in no seed, the challenger-trained models resolved 15 in some seed and
  the random-trained 16. The base resolved 54 that it had "never" resolved at placing: G is defined by one
  sample of 32 episodes and is porous.
- The challenger arm wrote more successful attempts (166 against 133) on the same number of distinct problems
  (54). The random arm reached 8 more distinct problems; its interval for the share excludes zero, the
  per-episode one does not. One round gives no arm a reach result.

## Gain by k: how hard should a training problem be (the owner's question of 2026-10-04)

Pass rate of the round's own problems, trained model minus the base afresh, pooled over the seeds:

| k of 8 in the round | Challenger arm | Random arm |
|---|---|---|
| 0 (no proof to train on) | +0.018 [+0.011, +0.024] (777 problems) | +0.000 [−0.001, +0.001] (2,113) |
| 1 | +0.093 [+0.075, +0.112] | +0.055 [+0.036, +0.074] |
| 2 | **+0.136** [+0.113, +0.158] | +0.055 [+0.023, +0.089] |
| 3 | +0.110 [+0.089, +0.130] | +0.062 [+0.017, +0.106] |
| 4 | +0.119 [+0.097, +0.143] | +0.083 [+0.040, +0.127] |
| 5 | +0.097 [+0.074, +0.119] | +0.068 [+0.031, +0.106] |
| 6 | +0.079 [+0.058, +0.099] | +0.025 [+0.000, +0.051] |
| 7 | +0.066 [+0.047, +0.084] | +0.011 [−0.006, +0.029] |
| 8 | +0.035 [+0.016, +0.053] | +0.013 [+0.000, +0.028] |

- **k = 1 to 5 gain about equally and more than k = 6 to 8.** The largest single value is at k = 2, which is
  the target t = 1/4; it is not separated from k = 3, 4 or 1. A target below one half costs nothing; nothing
  here says 1/4 beats 1/2. The spec's rule stands: t changes only by a spec edit made from this table, and
  the table gives no reason to change it.
- **A small transfer upward exists.** The challenger's own unsolved picks (k = 0: problems it expected near the
  band, base afresh 0.028) rose 1.8 points with an interval clear of zero; the random draw's unsolved problems
  (mostly STP conjectures the base cannot touch, base afresh 0.005) did not move.

## What the challenger did

| | Challenger | Random draw |
|---|---|---|
| Mean reward of its 1,000 problems | 0.40, 0.40, 0.40 | 0.13, 0.13, 0.13 |
| Share in the band | 0.23, 0.22, 0.19 | 0.06, 0.07, 0.06 |
| Share at k = 0 | 0.26 | 0.70 to 0.71 |
| Problems with a training proof | 741, 742, 740 | 290, 300, 297 |
| Training proofs that prove a negation | 311, 305, 312 (42%) | 37, 37, 33 (12%) |

- **It aims.** On base-map problems held back from its fit, its best tenth by expected reward scored a mean
  reward of 0.36 to 0.40 against 0.11 to 0.13 for all, and 19% to 23% in the band against 5% to 6%.
- **It found a shortcut the spec did not foresee (seed 0, read from its proposals):** 378 of its 1,000 picks
  are known-false Lean Workbook problems, which are 3.7% of the candidates (735 of 20,000); 543 are known-true
  Lean Workbook problems and 79 are STP conjectures (56% of the candidates). The base proves negations often
  enough for them to sit in the band. So 42% of the challenger arm's training proofs are negation proofs,
  while G is 99% known true. The reward as written allows this. Whether it is why the challenger arm's reach
  is no better than the random arm's is **not tested**.
- Attempts stayed diverse: 89% to 94% of a trained model's 8 attempts at a problem are distinct, against 95%
  to 96% for the base.

## Where the time went

A seed took 70 minutes (the job's estimate, made before the rolling pipeline and the larger pool, was 270). In
the round's attempt step the generator never waited on Lean (0.0 s), the pool answered 31 to 37 checks a
second, and the model wrote about 2,300 tokens a second: **the GPU is now the limit, not the pool.** Embedding
24,000 statements took 8 minutes; a training pass 100 seconds.

## What went wrong, and what was changed

- **Seed 1's first task ran out of GPU memory in training** (`ladder_l1_seed1_r1`). The model was never put in
  training mode, so gradient checkpointing was inactive and memory grew with an example's length: 12.35 GB on
  seed 0's longest, over the desktop's reserve on seed 1's. Fixed (`fe0f4cf83`): peak 8.3 GB. `lora.dropout`
  is 0, so the mathematics is the same and seed 0 stays comparable. **Seed 0 trained without checkpointing,
  seeds 1 and 2 with it.** The same omission is in Phase A's and B's training path, untouched.
- **A rerun after a failed task did not deliver the finished steps' per-problem files** (`d4829d872`).
- **The pool's proxy marked one of its Lean servers down once for a moment** under full load (its health
  check is too quick); refused checks were asked again.

## Departures from the spec

- **The standard evaluation of Phase A was not run:** its statements are not built at v4.27. Proposed: drop it
  from L1's measures.
- **Reach on the problems of G with no near neighbour in the training sets was not built:** the spec does not
  define a near neighbour.
- **Base-map problems are barred from the candidates** (the solver has attempted them); both arms choose from
  one exported, seeded draw of 20,000.
- **L1 trains on its one round's set.** "From the base on all rounds so far" needs the union in L2.
- **The branch for a primary below zero whose interval contains zero** was named before seed 0 was queued
  (escalate); it was not needed.
- **Seeds 1 and 2 were queued on the pre-registered branch without asking** (three seeds are within the repo's
  seed rule).

## What this does and does not show

- **Shown, three seeds:** choosing problems near a 1/4 pass rate yields 2.5 times the training proofs of a
  random draw and a larger gain on every held-out rung; training on the solver's own proofs makes problems it
  already solves sometimes more reliable (+4 to +5 points in and above the band).
- **Shown:** the gain on a trained problem is about +0.09 to +0.14 for k = 1 to 5 and falls above that.
- **Not shown:** that a round lifts the rung below the band. +0.009 [−0.0003, +0.018] on a base rate of 0.067
  is the size one round gives, and three rounds are what L2 is for.
- **Not shown:** that the loop reaches problems the base cannot. This is the same pattern as Phase A, Phase B
  and 13b: more reliable where it could already succeed.
- **Not judged:** the idea. One round was never expected to move reach; nothing here is a kill.

## Next (for the owner)

1. **The known-false share of the challenger's picks.** Leave it; hold each side to its share of the candidates
   (about 4%); or hold it to its share of G (about 1%). Recommended: its share of the candidates, from L2 on.
2. **L2 as specified:** three rounds, the challenger refitted each round on the current solver, the model
   trained from the base on all rounds' sets, then the base given the loop's whole sampling budget on G. Its
   code is not built (the union of rounds, the refit). A round is 70 minutes.
