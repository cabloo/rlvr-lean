# Ladder loop, L2 (three rounds, three seeds): every held-out rung rises, the hardest by 2.0 points; the third round adds little; the goal set is not shown to move, and the base given the same attempts solves more of it; the challenger's picks get easier because the pool has no middle left

Spec: `docs/spec/ladder-loop.spec.md`, "L2: three rounds (the owner's go-ahead, 2026-10-04)". Run 2026-10-05
(UTC) at Lean v4.27 through the `oeis` pool (37 workers), on the
GPU box: 98 minutes for seed 0 with the embedding, 75 and 77 for seeds 1 and 2. Tasks:
`rlvr_lean/ladder_l2_smoke_r1`, `ladder_l2_seed0_r1`, `ladder_l2_seed1_r1`, `ladder_l2_seed2_r1` (and `_r2`
of each, which added the trained model's extra attempts on the goal set).

**Verdict by the read fixed before the run.**

- **Primary (below-band rung, M(3) minus the base): +0.0196 [+0.0065, +0.0341]. Positive, interval above
  zero.** By seed +0.023, +0.028, +0.008. Every seed read ESCALATE; no round fired the stop rule; no seed is
  VOID; no soundness alarm.
- **The climb (M(3) minus M(1)):** below +0.006 [−0.003, +0.015] (not separated from zero), in the band
  +0.029 [+0.009, +0.048], above +0.019 [+0.011, +0.028]. The second round adds; the third adds little on any
  rung (M(3) minus M(2): +0.006, +0.007, +0.003, every interval containing zero).
- **Reach on G, 32 attempts each: not shown.** M(3) against the base afresh, 54 gained and 43 lost (p = 0.31);
  against M(1), 41 and 35 (p = 0.57).
- **The equal-compute control: plain sampling wins.** The base with the loop's 24,000 attempts (93 on each
  problem) solves 158 goal problems over the three seeds; M(3) with 32 solves 96. Gained 30, lost 92
  (p < 0.0001).
- **At 93 attempts each (added before it ran):** not shown problem by problem. M(3) solves 170 goal problems over the three seeds against
  the base's 158: 64 gained, 52 lost (p = 0.31). Its successes per attempt are 1.5 times the base's: 5.4 per
  1,000 against 3.6, paired by problem +1.8 [+0.6, +3.2].
- **The challenger's trajectory.** The known-false share of the scored picks falls at every seed (45%, 40%,
  31% by round, mean of the seeds), as stated before the run. The picks' mean pass rate moves AWAY from the
  target at every seed (0.37, 0.47, 0.47), the opposite of what was stated before the run.

So the loop does what the rungs measure and has not yet done what it is for. Three rounds make the solver
better at held-out problems of every difficulty the base could already sometimes solve, the hardest by about
three tenths of its base rate, and half again as reliable per attempt on the goal problems. They are not shown
to take it to problems the base could not solve, and by the loop's own accounting the same attempts spent on
sampling reach more of those. It is the result Phase A and L1 gave, one level harder: reliability, not reach.

## What ran

Three rounds at each of three seeds. A round: 1,000 problems proposed in four batches of 250 (225 scored by
expected reward at t = 1/4, 25 at random), the challenger refit after each batch on the base map and on
every finished batch (the current round at full weight, earlier ones halved per round); 8 episodes of the
round's model on each; the next model trained from the base, one pass, on one verified proof of every solved
problem of every round so far (about 720, 1,470 and 2,190 proofs). Candidates: the whole pool, 51,631
problems. After each training: 8 episodes on the 712 held-out rung problems and 32 on the 392 goal problems,
with L1's sampling seeds, so each model pairs by problem with the base's fresh results. After round 3: the
control (61 more base episodes on each goal problem). The reward was unchanged and the same for either side;
no quota, no filter on what is trained on. About 108,000 Lean checks a seed.

## The held-out rungs (model minus base; per problem the mean over seeds; 95% bootstrap over problems)

| After | Below the band (166; base 0.067) | In the band (155; base 0.270) | Above the band (391; base 0.767) | Distinct attempts |
|---|---|---|---|---|
| 1 round | +0.014 [+0.003, +0.026] | +0.058 [+0.039, +0.077] | +0.057 [+0.048, +0.066] | 90% |
| 2 rounds | +0.014 [+0.001, +0.027] | +0.080 [+0.056, +0.104] | +0.073 [+0.063, +0.083] | 87% |
| **3 rounds** | **+0.020 [+0.007, +0.034]** | +0.087 [+0.059, +0.115] | +0.076 [+0.065, +0.088] | 87% |
| 3 rounds minus 1 | +0.006 [−0.003, +0.015] | +0.029 [+0.009, +0.048] | +0.019 [+0.011, +0.028] | |
| 3 rounds minus 2 | +0.006 [−0.002, +0.014] | +0.007 [−0.009, +0.023] | +0.003 [−0.004, +0.010] | |

By seed after three rounds: below +0.023, +0.028, +0.008; in +0.087, +0.099, +0.074; above +0.070, +0.092,
+0.066. The base's distinct-attempt share is 95%. L1's one round (20,000 candidates, one batch) gave +0.009
[−0.000, +0.018], +0.044 and +0.051; L2's first round (the whole pool, four batches) gives +0.014, +0.058
and +0.057.

By what the problem is, after three rounds: known true +0.063 [+0.053, +0.072] (660 problems), known false
+0.101 [+0.060, +0.144] (52), Lean Workbook +0.074 [+0.063, +0.085] (608), STP conjectures +0.015 [−0.005,
+0.036] (104). STP problems still gain nothing that separates from zero.

## The goal set (392 problems the base never resolved while being placed)

| Problems solved | Base, 32 attempts | M(1), 32 | M(2), 32 | M(3), 32 | Base, 93 attempts | M(3), 93 attempts |
|---|---|---|---|---|---|---|
| Seed 0 | 37 | 28 (+11 −20) | 33 (+15 −19) | 37 (+18 −18) | 59 | 61 |
| Seed 1 | 22 | 34 (+16 −4) | 37 (+20 −5) | 36 (+23 −9) | 43 | 56 |
| Seed 2 | 26 | 28 (+15 −13) | 29 (+18 −15) | 23 (+13 −16) | 56 | 53 |
| All seeds | 85 | 90 | 99 | 96 | 158 | 170 |
| Successful attempts per 1,000 | 3.5 | 4.1 | 5.0 | 5.0 | 3.6 | 5.4 |

| Problem by problem, summed over the seeds | Gained | Lost | Sign test |
|---|---|---|---|
| M(1) against the base afresh | 42 | 37 | p = 0.65 |
| M(2) against the base afresh | 53 | 39 | p = 0.18 |
| M(3) against the base afresh | 54 | 43 | p = 0.31 |
| M(3) against M(1) | 41 | 35 | p = 0.57 |
| M(3) in 32 attempts against the base in 93 | 30 | 92 | p < 0.0001 |
| M(3) against the base, 93 attempts each | 64 | 52 | p = 0.31 |

- At 32 attempts the trained models solve a few more goal problems than the base's fresh draw and succeed on
  more attempts (5.0 per 1,000 against 3.5), by seed 5.5 against 4.4, 6.0 against 3.1, 3.5 against 3.1. None
  of it separates from zero problem by problem.
- The control is the hard reading of "equal compute": the loop is charged for its training attempts. Three
  times the attempts on the goal problems themselves reach 158 of them; three rounds of training on other
  problems reach 96.
- **At 93 attempts each** M(3) solves 61, 56 and 53 goal problems against the base's 59, 43 and 56, and succeeds
  on 594 attempts against 397 (5.4 per 1,000 against 3.6; paired by problem, the mean over seeds, +1.8
  [+0.6, +3.2]). The extra successes fall mostly on problems both models solve (462 against 317). Problems only
  one of them solves: 64 for M(3), 52 for the base. Over the three seeds the base solves 82 distinct goal problems
  and M(3) 87, 65 of them the same. The loop made the solver half again as reliable on the goal set; it is not
  shown to have changed which problems it can reach.

## The challenger (the owner's points 2 and 3)

| Scored picks, mean of the seeds | Known false | Mean pass rate | k = 0 | k = 1 to 3 | k = 4 or more | Mean reward | Training proofs from k ≥ 4 | Training proofs that are refutations |
|---|---|---|---|---|---|---|---|---|
| Round 1 | 45% | 0.37 | 22% | 39% | 39% | 0.45 | 51% | 45% |
| Round 2 | 40% | 0.47 | 19% | 30% | 51% | 0.36 | 64% | 41% |
| Round 3 | 31% | 0.47 | 22% | 26% | 52% | 0.32 | 67% | 34% |

Known-false share of the scored picks by batch (mean of the seeds): round 1: 47%, 45%, 46%, 41%; round 2:
55%, 44%, 31%, 29%; round 3: 40%, 53%, 20%, 11%.

- **The refutation share falls, and it falls inside a round.** The first two batches of rounds 2 and 3 are
  chosen before the challenger has seen much of the new model and stay at 40% to 55%; the last two fall to
  29% and 11%. The reward does move the challenger off refutations once it sees that the current model finds
  them easy. It does so half a round late, each round.
- **The picks get easier, not harder.** Mean pass rate 0.37, 0.47, 0.47; picks solved 4 or more times of 8
  go from 39% to 52%; the mean reward falls from 0.45 to 0.32.
- **The reason is supply.** Candidates the challenger predicted in the band at the start of each round:

| Predicted pass rate | 0.05 to 0.13 | 0.13 to 0.25 | 0.25 to 0.41 | 0.41 to 0.60 | 0.60 and above |
|---|---|---|---|---|---|
| Start of round 1 (51,631 candidates) | 6,424 | 1,837 | 718 | 1,826 | 5,550 |
| Start of round 2, seeds 0, 1, 2 | 6,475, 6,482, 6,362 | 1,674, 1,700, 1,580 | 72, 53, 77 | 1,655, 1,679, 1,712 | 5,425, 5,413, 5,358 |
| Start of round 3 | 7,286, 6,461, 7,127 | 1,487, 2,124, 1,696 | 0, 6, 1 | 787, 623, 790 | 5,704, 5,837, 5,749 |

  The whole pool held 718 problems predicted between 0.25 and 0.41 and round 1 took them. From round 2 the
  challenger chooses between easier problems (0.41 to 0.60) and harder ones (0.13 to 0.25), and at t = 1/4
  its expected reward is higher on the easy side (0.43 at a rate of 0.50 against 0.33 at 0.15).
- **The estimate also runs behind the model.** In rounds 2 and 3 the picks are solved more often than
  predicted (0.47 against 0.40, and 0.47 against 0.42).

## What a lower target would do (computed from stored data, no run)

The reward's one setting is the target rate t. The challenger's estimate of one problem is loose, so at
t = 1/4 its expected reward peaks at a true pass rate of 0.35, not 0.25. The 900 scored picks of a first
round, chosen again at other targets from the whole pool; outcomes expected from the 6,000 L1 picks whose k
was measured:

| t | Expected reward peaks at | Expected mean pass rate of the picks | Never solved | k = 1 to 3 | k = 4 or more | Proofs per round | Known false |
|---|---|---|---|---|---|---|---|
| 1/4 (as run) | 0.35 | 0.38 | 22% | 36% | 41% | 699 | 46% |
| 1/5 | 0.35 | 0.36 | 25% | 37% | 38% | 676 | 42% |
| 1/6 | 0.30 | 0.33 | 28% | 38% | 35% | 650 | 38% |
| 1/8 | 0.30 | 0.30 | 32% | 37% | 31% | 612 | 30% |
| 1/10 | 0.25 | 0.27 | 37% | 36% | 28% | 571 | 25% |

A lower target centres the picks on a quarter and cuts the refutation share, through the reward alone. It
does not raise the share in the band (36% to 38% at every target: the estimate's ceiling); it trades picks
that are too easy for picks that are never solved, and costs about a fifth of the proofs. It does not
create supply either: the pool has 1,837 candidates predicted between 0.13 and 0.25 and 718 between 0.25
and 0.41, about three rounds' worth.

## What is not settled

- The below-band rung's climb from round 1 to round 3: +0.006 [−0.003, +0.015].
- Whether more rounds would help: the third round's gains are inside their intervals on every rung, and the
  picks are getting easier.
- Whether training on harder picks (a lower t) or on more of the same kind (a larger pool) moves the goal
  set. Nothing here tests either.
