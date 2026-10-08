# Ladder loop, L3c (an episode that keeps what verified, no training, three seeds): it wins its read at every seed; at equal generations it resolves more hard episodes (+1.0 point) and solves 92 goal problems to blind sampling's 65, twenty of them never solved by any model here; the new problems are reached, not yet reliable

Spec: `docs/spec/ladder-loop.spec.md`, "L3c: an episode that keeps what verified (no training)", approved by
the owner 2026-10-07. Why it was built: `reach-diagnosis-RESULT.md`. Run
2026-10-08 (UTC) at Lean v4.27 through the `oeis` pool (37 workers), on the GPU box, the base model,
nothing trained: under 30 minutes a seed. Tasks `rlvr_lean/ladder_l3c_smoke_r1`,
`ladder_l3c_pilot_seed0_r1` (80 hard problems, a seed of its own, not read as a result) and
`ladder_l3c_seed0_r1`, `ladder_l3c_seed1_r1`, `ladder_l3c_seed2_r1`. Seed 0 read a win, so by the seed rule it
went to three; a seed here is a sampling seed (1032, 1132, 1232), every generation a new sample.

**Verdict by the read fixed before the run: THE ACCUMULATING EPISODE IS THE SOLVER'S EPISODE FROM HERE, at
each of the three seeds.**

- **All three "can this run see a win" checks pass at every seed.** A pool in 61% of hard episodes by their
  last generation; a state for every continuing generation; the pool stands in 14,054 of 14,060 checks.
- **The rows were checked before any number was read, each seed** (18 checks): every continuing proof is the
  pool it was shown and then what the model wrote; a fresh generation is the blind arm's own sample wherever
  both arms had the episode open (55,706 of 56,169; the others were drawn after the blind arm had resolved);
  a pool only grows; the report rebuilds from the stored rows.
- **Primary (G and the below-band rung, 558 problems; episodes resolved within 8 generations; accumulate
  minus blind):** +0.0114 [+0.0039, +0.0191], +0.0114 [+0.0045, +0.0185], +0.0084 [+0.0015, +0.0152] by
  seed. **Over the three seeds: +0.0104 [+0.0053, +0.0157]** (1,294 of 10,044 episodes against 1,190).
- **Reach, the goal set by problem:** by seed 54 against 42 (gained 16, lost 4), 46 against 35 (15 and 4),
  59 against 40 (25 and 6). **Over a problem's 18 episodes: 92 against 65; gained 28, lost 1.**
- **Where the shortest published proof is 4 lines or more (230 goal problems): 36 against 28; gained 8, lost
  0** (sign test p = 0.008). By seed 21 against 17, 16 against 13, 23 against 16.
- **Budget on the primary's problems, three seeds:** the same generations (65,279 against 65,620), 91% of
  the generated tokens (5.26 M against 5.77 M), 1.39 times the prompt tokens, and 1.5 times the Lean checks
  (96,455 against 64,271: 13,641 pool checks and 21,927 closer checks). The 10% rule does not fire.

So keeping what verified does what the diagnosis said it might. With the same number of generations from the
same untrained model, an episode that pools its verified lemmas resolves more hard episodes and solves
problems that sampling does not.

## Running totals, hard problems, three seeds (10,044 episodes)

| Within | Accumulate | Blind | Lead | Only one arm | Paired difference |
|---|---|---|---|---|---|
| 1 generation (shared) | 200 | 200 | 0 | | |
| 2 | 393 | 390 | +3 | 39 / 36 | +0.0003 [−0.0018, +0.0024] |
| 3 | 577 | 546 | +31 | 85 / 54 | +0.0031 [+0.0003, +0.0061] |
| 4 | 740 | 690 | +50 | 113 / 63 | +0.0050 [+0.0017, +0.0084] |
| 5 | 889 | 818 | +71 | 144 / 73 | +0.0071 [+0.0033, +0.0112] |
| 6 | 1,052 | 964 | +88 | 172 / 84 | +0.0088 [+0.0045, +0.0132] |
| 7 | 1,178 | 1,080 | +98 | 192 / 94 | +0.0098 [+0.0050, +0.0148] |
| 8 | 1,294 | 1,190 | +104 | 215 / 111 | +0.0104 [+0.0053, +0.0157] |

The lead grows with every generation. (Repair's lead was made at one step and held: `ladder-l3a2-RESULT.md`.)
By set within 8 generations: G +0.0106 [+0.0058, +0.0157] (260 episodes against 185, 41% more); below the band
+0.0097 [−0.0033, +0.0234]; in the band −0.0075 [−0.0204, +0.0054]; above it −0.0004 [−0.0013, 0.0000].
Where a fresh attempt usually works the episode neither helps nor is shown to cost.

## The goal set, by problem (392 problems, 18 episodes of up to 8 generations each)

| | Accumulate | Blind |
|---|---|---|
| Problems solved | 92 | 65 |
| Never solved by the base in 279 attempts | 33 | 13 |
| Never solved by the base nor by the models after three rounds (558 attempts) | 20 | 5 |
| Won in at least 2 of the 18 episodes | 52 | 41 |
| Won in at least 5 | 21 | 13 |
| Won in at least 9 (half) | 2 | 2 |

For scale: the base with 279 attempts a problem solved 82 of these, and the models after three rounds at
t = 1/10, with 279, solved 94. The accumulating episode solves 92 with 144 generations of the untrained model.

By the length of the shortest published proof:

| Shortest published proof | Goal problems | Accumulate | Blind | Gained / lost | Hard episodes, accumulate minus blind |
|---|---|---|---|---|---|
| 1 line | 37 | 19 | 12 | 7 / 0 | +0.0216 [−0.0041, +0.0489] |
| 2 to 3 lines | 125 | 37 | 25 | 13 / 1 | +0.0176 [+0.0078, +0.0279] |
| 4 to 7 lines | 156 | 30 | 24 | 6 / 0 | +0.0049 [0.0000, +0.0101] |
| 8 lines or more | 74 | 6 | 4 | 2 / 0 | −0.0006 [−0.0082, +0.0064] |

(The last column is over the hard problems of that length, 67, 193, 203 and 95 of them.)

- **Most of the gain is on problems whose proof is short.** These are problems the model's one-shot attempts
  kept nearly solving and never finished: one attempt had the facts, another the closing step.
- **The long ones move too, by problems and not yet by episodes:** 8 gained and none lost at 4 lines or more,
  with the episode count there at +0.5 points and an interval that touches zero.
- **Reached is not reliable.** Of the 28 gained problems, 19 were won in one episode of 18. A problem solved
  once in 144 generations is found, not learned.

## How the accumulate arm resolves, and what the model does with a pool

Hard problems, three seeds, resolved after the first generation: 824 by a fresh proof, 119 by a continuing
generation, 151 by an assembled proof (a kept closing step checked after a larger pool). The blind arm: 990.
So the arm gives up about 170 fresh resolutions (its continuing generations take the place of fresh ones) and
gets 270 back from the two things it adds. Of the 28 gained goal problems, 23 were won by an assembled proof
and 8 by a continuing generation.

What the untrained model writes when it is shown its pooled lemmas (12,476 continuing generations):

- **It closes.** Three quarters begin with a closing tactic, not a new `have`; the median continuation is 33
  tokens.
- **It rarely adds a lemma:** 1,607 of the 8,915 continuations that Lean rejected with an error left a new
  one.
- **It repeats itself:** 27% are proofs Lean had already rejected in the episode (3,319), which are not sent.
- **It sometimes finishes:** 185 verified.

The proofs are longer. On the goal set the accumulate arm's verified proofs have a median of 6 lines and the
longest is 21; the blind arm's, 4 and 11 (seed 0).

## What this says and does not say

- **Reach is shown, for the first time in this project,** on a read fixed before the run and at three seeds:
  problems solved that the same model's sampling does not solve, including 20 that no model here had solved
  in 558 attempts.
- **It is not reliability.** The owner's aim is to do reliably what could not be done before. This is the
  first half. The second is what training on these proofs is for.
- **It is at equal generations, not equal Lean checks.** The accumulate arm makes half again as many Lean
  checks. With 37 Lean workers the GPU is what limits a round, so generations are the budget; a closer check
  is a candidate proof that cost no generation.
- **The untrained model does not build on a pool.** Most of what the episode gains comes from recombining
  what it wrote. A model trained on assembled proofs has seen lemmas followed by more lemmas.
- **One base model, one pool of problems.** Nothing here is trained, so a seed is a sampling seed.

## For the owner: what follows

1. **L3d: put the episode in the round and train on what it assembles.** A round whose solver episodes
   accumulate, and a training set that holds the assembled and continued proofs (long, with lemmas) beside
   the one-shot ones. Its read, to be fixed first: held-out goal problems by proof length, and whether the
   trained model then solves the reached problems reliably and in one shot. This is the step the branch
   names; it needs a spec and the owner's approval.
2. **The ceiling** (still not approved): one fine-tune on published proofs of pool problems the base cannot
   solve, as a labelled diagnostic. It says how far training can take this model on long proofs at all.
3. **Target for the rounds:** 1/10 (recommended; `ladder-l2t-RESULT.md`).

Recommended: 1, with the target at 1/10; 2 beside it if the owner allows it, since it bounds what 1 can
reach.

## Points the build made exact

In the spec ("Made exact before the build", "Made exact by the build"). Two to keep in mind when reading: a
continuing proof re-proves the pool and a closer with more facts above it is slower, under the same 30 s
limit (242 closer checks and 31 continuing proofs timed out over the three seeds, counted as failures);
and the token cap covers only the continuation, so an accumulate proof can be longer than any blind one.

## Files

`experiments/rlvr_lean/ladder_l3c_seed<N>_r1/steps/`: `report_ladder_l3c.json` (each seed's read),
`repair_attempts_<loop>.jsonl`, `repair_pools_<loop>.jsonl`, `repair_closers_<loop>.jsonl`,
`repair_episodes.jsonl`. The three-seed tables are each problem's mean over its episodes of the three seeds,
95% bootstrap over problems, from `repair_episodes.jsonl`.
