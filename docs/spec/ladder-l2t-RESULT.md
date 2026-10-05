# Ladder loop, L2t (the lower target, three rounds, three seeds): not shown on the hardest rung; the challenger stays on target and the model is a third more reliable on the goal set than at t = 1/4; more goal problems solved in one seed of three

Spec: `docs/spec/ladder-loop.spec.md`, "L2t: the lower target". Run
2026-10-05 (UTC) at Lean v4.27 through the `oeis` pool (37 workers), on the GPU box: 91, 97 and 90
minutes. Tasks `rlvr_lean/ladder_l2_t010_seed0_r1`, `ladder_l2_t010_seed1_r1`, `ladder_l2_t010_seed2_r1`,
each paired with its control `ladder_l2_seed<N>_r2` (`ladder-l2-RESULT.md`): the same seed, candidates,
batches, sampling seeds, box and code, and ONE setting changed, the challenger's target rate, 0.10 in place of
0.25.

**Verdict by the read fixed before the run: NOT SHOWN on the primary.** Not a win of the lower target and not
a loss.

- **The setting acted at every seed:** round 1's scored picks have a mean pass rate of 0.274, 0.262 and 0.283
  against the control's 0.368, 0.363 and 0.364 (the bar was 0.05 lower). No seed is VOID; no soundness alarm.
- **Primary (the below-band rung, M(3) at t = 1/10 minus M(3) at t = 1/4): +0.0058 [−0.0040, +0.0156].** By
  seed +0.0075, +0.0038, +0.0060: the same sign three times, inside an interval that contains zero. Against the
  base: +0.0254 [+0.0123, +0.0392] at t = 1/10 and +0.0196 [+0.0065, +0.0339] at t = 1/4.
- **The other rungs:** in the band +0.0022 [−0.0183, +0.0226]; above it −0.0092 [−0.0183, +0.0005], and
  clearly lower after rounds 1 and 2 (−0.0113 [−0.0189, −0.0037], −0.0178 [−0.0260, −0.0098]). Fewer easy
  proofs are trained on, so a smaller gain there was expected and is not a failure.
- **The goal set at 93 attempts each, successes per attempt: +0.0017 [+0.0008, +0.0028]** over the model at
  t = 1/4, at every seed (263, 282, 237 successes against 202, 225, 167; the base 146, 123, 129). Per 1,000
  attempts: 7.2 at t = 1/10, 5.4 at t = 1/4, 3.6 for the base.
- **The goal set, problems solved: more in one seed of three.** 189 against 170 at t = 1/4 (gained 58, lost
  39, p = 0.067) and against the base's 158 (gained 79, lost 48, p = 0.008). By seed against the base: 20
  gained and 20 lost; 39 and 8; 20 and 20. Two ties and one large gain is not a consistent effect, and it is
  not read as reach.

So the lower target does what it was built to do to the challenger, and the model it trains is as good on the
hard rungs, more reliable on the problems the base never solved in its 32 placing attempts, and about a point
behind on the easy rung, from a quarter fewer proofs. Whether it solves problems the old target's model does
not is not settled by three seeds.

## The held-out rungs (per problem the mean over seeds; 95% bootstrap over problems)

| After | Below the band (166) | In the band (155) | Above the band (391) |
|---|---|---|---|
| 1 round, t = 1/10 minus t = 1/4 | −0.005 [−0.015, +0.005] | −0.009 [−0.025, +0.007] | −0.011 [−0.019, −0.004] |
| 2 rounds | +0.009 [−0.002, +0.020] | −0.008 [−0.027, +0.011] | −0.018 [−0.026, −0.010] |
| **3 rounds** | **+0.006 [−0.004, +0.016]** | +0.002 [−0.018, +0.023] | −0.009 [−0.018, +0.001] |
| 3 rounds, t = 1/10 minus the base | +0.025 [+0.012, +0.039] | +0.089 [+0.064, +0.115] | +0.067 [+0.055, +0.079] |
| 3 rounds, t = 1/4 minus the base | +0.020 [+0.007, +0.034] | +0.087 [+0.059, +0.115] | +0.076 [+0.064, +0.088] |

By seed after three rounds, t = 1/10 minus t = 1/4: below +0.0075, +0.0038, +0.0060; in +0.0081, −0.0137,
+0.0121; above −0.0026, −0.0288, +0.0038. Distinct attempts on the rungs after three rounds: 89%, 91%, 90% at
t = 1/10 against 88%, 86%, 86% at t = 1/4 (the base: 95%).

## The goal set (392 problems a seed)

| | Base | 3 rounds, t = 1/4 | 3 rounds, t = 1/10 |
|---|---|---|---|
| Problems solved, 32 attempts, by seed | 37, 22, 26 (85) | 37, 36, 23 (96) | 41, 46, 39 (126) |
| Problems solved, 93 attempts, by seed | 59, 43, 56 (158) | 61, 56, 53 (170) | 59, 74, 56 (189) |
| Successes, 93 attempts, by seed | 146, 123, 129 (398) | 202, 225, 167 (594) | 263, 282, 237 (782) |
| Successes per 1,000 attempts | 3.6 | 5.4 | 7.2 |

Gained against lost at 93 attempts each, by seed:

| | Seed 0 | Seed 1 | Seed 2 | Together |
|---|---|---|---|---|
| t = 1/10 against the base | 20 / 20 | 39 / 8 | 20 / 20 | 79 / 48 (p = 0.008) |
| t = 1/4 against the base | 22 / 20 | 28 / 15 | 14 / 17 | 64 / 52 (p = 0.31) |
| t = 1/10 against t = 1/4 | 14 / 16 | 27 / 9 | 17 / 14 | 58 / 39 (p = 0.067) |

Seed 1 is where the base's own draw was poorest (43 problems, against 59 and 56 from the same model with other
sampling seeds) and the lower target's model best. The base's results in an arm's run and in its control's run
are the same to within one success on one problem.

**A reading made after seeing the above, not fixed before the run:** against one reference, the problems the
base did not solve in its OTHER two seeds' 186 attempts, a seed's 93 attempts find 17, 7 and 8 new problems
with the base, 22, 15 and 11 with the model at t = 1/4, and 21, 25 and 14 at t = 1/10. Over all three seeds
the base's 279 attempts solve 82 goal problems, the three models at t = 1/4 solve 87 between them (22 the base
never solved; 17 of the base's missed) and the three at t = 1/10 solve 94 (30 and 18; sign test p = 0.11). The
direction is the same in every seed and the size is small. It is a candidate for the next read, not a result.

## The challenger (means of the three seeds, by round)

| | t = 1/10: round 1, 2, 3 | t = 1/4: round 1, 2, 3 |
|---|---|---|
| Mean pass rate of the scored picks | 0.27, 0.34, 0.24 | 0.37, 0.47, 0.47 |
| Never solved (k = 0) | 36%, 35%, 51% | 22%, 19%, 22% |
| Known-false picks | 24%, 10%, 5% | 45%, 40%, 31% |
| Proofs trained on (per seed) | 598, 607, 465 | 720, 747, 723 |

- **It stays on target.** The picks' mean pass rate does not drift to a half. Candidates predicted between
  0.25 and 0.41 still run out (718, then about 190, then about 25), and the lower target does not need them:
  1,837 are predicted between 0.13 and 0.25 at the start and about 2,200 by round 3, as the solver improves.
- **Refutations fall to a twentieth of the picks.** The reward was the same for either side, with no quota.
- **Half of round 3's picks are never solved.** A lower target reaches into problems nobody solves, and those
  give no proof: 465 proofs in round 3 against 723.

## Why the model differs: the same hard proofs, fewer easy ones

Over the three seeds and rounds, one proof per solved problem:

| | From problems 1 to 3 of 8 solvers solved | From problems 4 or more solved | Picks never solved |
|---|---|---|---|
| t = 1/10 | 2,486 | 2,524 | 3,990 |
| t = 1/4 | 2,601 | 3,968 | 2,431 |

The hard side of the training set is the same size. The lower target leaves out about 1,450 easy proofs. The
model trained without them keeps more distinct attempts and succeeds more often on the goal problems, and
gives up about a point on the easy rung. That is the dose curve's finding from the other side
(`ladder-l1b-RESULT.md`): proofs of problems the model already solves narrow it.

## For the owner: which target for further rounds

1. **Take t = 1/10 as the target for further rounds.** Not shown better on the primary; as good on the hard
   and middle rungs, a third more successes on the goal set at every seed, the challenger on target, a quarter
   fewer proofs to train on. Cost: about a point on the easy rung, and half of round 3's picks wasted.
2. **Keep t = 1/4,** the default the primary did not displace.
3. **Six seeds on the question the three leave open** (does the lower target's model solve goal problems the
   other does not?), with the read fixed first: the "new problems against one reference" count above, or
   problems solved at 93 attempts by seed. Three more seeds of each arm are about 9 hours of the GPU.

Recommended: 1 for what is built next, and 3 only if "reach" at this target is the claim to be made. The
config's default stays 0.25 until the owner says otherwise: the arm did not win the read it was given.

## Files

`experiments/rlvr_lean/ladder_l2_t010_seed<N>_r1/steps/` and each control's
`experiments/rlvr_lean/ladder_l2_seed<N>_r2/steps/`: `report_ladder_l2.json`, `episodes_rungs_m<r>_problems.jsonl`,
`episodes_reach_m<r>_problems.jsonl`, `episodes_control_m3_problems.jsonl`, `episodes_control_problems.jsonl`,
`base_reach.jsonl`, `episodes_round_r<r>_b<b>_problems.jsonl`, `candidate_scores_r<r>.jsonl`.
