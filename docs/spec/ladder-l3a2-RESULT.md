# Ladder loop, L3a2 (the second repair check, no training, seed 0): not shown at five attempts, at a size that resolves 0.7 points; one repair step resolves hard problems sooner, and no problem that blind sampling does not

Spec: `docs/spec/ladder-loop.spec.md`, "L3a2: one repair step after each fresh failure, then start over (no
training)". Run 2026-10-05 (UTC) at Lean v4.27 through the `oeis` pool (37
workers), on the GPU box, the base model, nothing trained: 33 minutes. Tasks
`rlvr_lean/ladder_l3a2_smoke_r1` and `rlvr_lean/ladder_l3a2_seed0_r1`. It follows `ladder-l3a-RESULT.md`.

**Verdict by the read fixed before the run: NOT SHOWN AT THIS SIZE.** Not a kill of repair: it is the
untrained model's repair step, as a search step, that is not shown to pay.

- **Both "can this run see a win" checks pass.** A state for 97.4% of the failed attempts a repair step starts
  from (13,075 of 13,424). On the hard problems the repair step at position 2 verifies 177 of 6,562 (2.70%)
  against the blind arm's 127 (1.93%).
- **The rows, checked before any number was read:** every resumed repair step starts from the fresh attempt
  before it (attempt 2 from attempt 1: 6,799 of 6,799; attempt 4 from attempt 3: 6,276 of 6,276; none from a
  repair step's own output), and attempts 3 and 5 are the same samples in both arms (6,439 of 6,439; 6,070 of
  6,070).
- **Primary (G and the below-band rung, 558 problems, 6,696 episodes; resolved within 5 attempts; alternate
  minus blind): +0.0048 [−0.0022, +0.0120]** (0.0899 against 0.0851). Without the trimmed steps (kept lines
  that were already a whole proof): +0.0013 [−0.0054, +0.0081].
- **The gain is made within 2 attempts and then kept, not added to:** +0.0075 [+0.0015, +0.0137] within 2,
  +0.0058 [0.0000, +0.0118] within 3, +0.0048 [−0.0025, +0.0123] within 4. (Corrected 2026-10-06: this line
  said the gain "is caught up". It is not; see "Read as running totals" below.)
- **G by problem (resolved in any of 12 episodes):** 49 with the alternate episode against 47 blind; gained
  12, lost 10 (p = 0.83). G alone within 5 attempts: +0.0038 [−0.0006, +0.0091]. Below the band:
  +0.0070 [−0.0141, +0.0291].
- **Where a fresh attempt is strong, nothing is shown either way:** in the band −0.036 [−0.081, +0.013],
  above it −0.003 [−0.012, +0.006].
- **Budget on the primary's problems:** the alternate arm generated 83% of the blind arm's tokens (1.86 M
  against 2.25 M) and sent 1.43 times the Lean checks (35,998 against 25,139). The 10% rule does not fire.

By the branch fixed before the run: a repair step is not worth its extra Lean check as a search step for the
untrained model, and what is left of repair is training on it, which is the owner's call.

## What ran

12 first attempts on each of the 392 goal problems and the 166 below-band problems, 2 on each of the 546 other
rung problems: 7,788 episodes, 815 verified at once. Each of the 6,973 failed first attempts was continued two
ways to at most 5 attempts: blind (whole proofs from the plain prompt), and alternate (repair from attempt 1, a
whole proof, repair from attempt 3, a whole proof). In both arms a proof Lean had already rejected in the
episode was not sent again. Sampling seed 1031; no generation shares a seed with L3a's. 60,901 attempts,
56,412 Lean checks, 3.74 M generated tokens. No soundness alarm (112 audit attempts on the side a certificate
rules out, none verified).

The size was set for this question: L3a's primary had a half-width of 0.016; this one has 0.007, against the
+0.013 that the reading composed from L3a's rows had suggested. That reading had been flagged as leaning on an
unusually poor fifth blind attempt, and it did not repeat.

## Read as running totals (corrected 2026-10-06, after the owner's review)

The first version of this note read the arms attempt by attempt, by the share of the episodes still open that
verify at each attempt (the table further down). The owner pointed out what is wrong with that: after attempt
2 the two arms no longer hold the same episodes. The arm that resolved more early is left with the harder
ones, so its later rates are compared on a worse population. The fair reads are the running total, and a
comparison on episodes that are open in both arms. The primary above is a running total and does not change;
three sentences of the first version do, and they are marked below.

Hard episodes (6,696) resolved within k attempts:

| Within | Alternate | Blind | Lead | Only the alternate arm / only the blind arm | Sign test |
|---|---|---|---|---|---|
| 1 attempt (shared) | 134 | 134 | 0 | | |
| 2 attempts | 311 (4.64%) | 261 (3.90%) | +50 | 163 / 113 | p = 0.003 |
| 3 attempts | 410 (6.12%) | 371 (5.54%) | +39 | 146 / 107 | p = 0.017 |
| 4 attempts | 516 (7.71%) | 484 (7.23%) | +32 | 216 / 184 | p = 0.12 |
| 5 attempts | 602 (8.99%) | 570 (8.51%) | +32 | 205 / 173 | p = 0.11 |

- **The alternate arm is ahead at every attempt count.** The first repair step makes a lead of 50 episodes
  and 32 of it is still there at five attempts. The blind arm does not catch up; the lead stops being
  separable from zero because the totals grow and the lead does not.
- **The first repair step, like for like (every failed first attempt, 6,562):** 177 verified against a fresh
  attempt's 127; 163 episodes only the repair step resolved, 113 only the fresh attempt (p = 0.003). It is
  worth 1.4 blind attempts (L3a: 1.8), and 18 of its 177 are trimmed steps. Of the 177, the blind arm
  resolved 14 at the same attempt, 39 more by attempt 5, and never 124.
- **The second repair step, like for like:** on the 6,179 episodes still open in BOTH arms after attempt 3
  (the same failed attempts 1 and 3 behind them), the repair step at attempt 4 verifies 94 times and a fresh
  attempt 98; 85 only the repair step, 89 only the fresh attempt; paired +0.0000 [−0.0066, +0.0068]. A second
  repair step in the same episode is worth one blind attempt. These are episodes where a first repair step
  has already failed: what a repair can fix, the first one takes.
- **Why the lead narrows from 50 to 32:** attempts 3 and 5 are the same samples in both arms, and some of
  them land on episodes the repair step had already resolved.

## By attempt position, hard problems (verified / attempts made at that position)

Shares of the episodes still open at that attempt. After attempt 2 the two rows are different episodes, so
they are not a comparison of the arms.

| | Attempt 2 | Attempt 3 | Attempt 4 | Attempt 5 |
|---|---|---|---|---|
| Blind | 127 / 6,562 (1.93%) | 110 / 6,435 (1.71%) | 113 / 6,325 (1.79%) | 86 / 6,212 (1.38%) |
| Alternate | **177 / 6,562 (2.70%)** repair | 99 / 6,385 (1.55%) | **106 / 6,286 (1.69%)** repair | 86 / 6,180 (1.39%) |

- **Withdrawn (2026-10-06):** "the second repair step is worth about one, 1.69% against 1.79%" compared
  different survivors. The like-for-like figure is above (94 against 98) and happens to say the same.
- **Withdrawn (2026-10-06):** "the blind arm catches up; the episodes a repair step resolves are mostly ones
  a later blind attempt resolves too". The blind arm resolved 53 of those 177 within five attempts and never
  124, and the alternate arm ends 32 episodes ahead.
- **Known copies:** at a repair step the model writes a proof Lean has already rejected in 15.7% of the hard
  failures at attempt 2 (1,028 of 6,562) and 15.5% at attempt 4 (977 of 6,286); the blind arm, 0.7% to 1.9%.
  Where a repair step had a state and wrote something new at attempt 2, it verified about 2.9% of the time.

## What the two checks say together

1. **Lean's state is information the model uses** (L3a: +0.026 over resuming without it), and one repair
   step straight after a failure verifies more often than a blind attempt, twice measured on separate
   attempts (L3a +0.0125 [+0.0027, +0.0228] within 2 attempts; here +0.0075 [+0.0015, +0.0137]).
2. **That is a small, kept lead, and not reach.** By five attempts the alternate arm has resolved 602 hard
   episodes to the blind arm's 570 (+0.5 points, not separable from zero), and the same number of goal
   problems (49 against 47). It is the result training gave, one more time: the hard problems that can be
   solved are solved a little more often, and no problem is shown to be solved that plain sampling does not
   solve. (Corrected 2026-10-06: this said the blind arm "has resolved nearly the same episodes"; the two
   arms resolve 397 of the same episodes, 205 only one and 173 only the other.)
3. **The untrained model wastes the step about one time in six** by rewriting the rejected proof, and the
   steps it does not waste are worth about one and a half blind attempts. Neither is a property of repair;
   both are what training on repaired proofs would be meant to change.
4. **Cost.** A repair step generates fewer tokens and costs one more Lean check. With 37 Lean workers the GPU
   is the limit, so the alternate episode is slightly cheaper for the same result, not a reason to adopt it.

## For the owner: what to do with repair

1. **Leave the untrained repair step out of the round** (the branch's reading). At five attempts it is ahead
   by half a point with an interval through zero: nothing shown gained, and nothing lost. If it were put in,
   the totals say one repair step an episode is all that is worth having (corrected 2026-10-06).
2. **Train on repair (L3b), as its own experiment with its own read:** rounds whose episodes take one repair
   step after each fresh failure, and training on the failed-then-repaired proofs. This run has 478 verified
   repair steps on held-out problems (80 of them trimmed, with nothing generated); a round of 1,000 problems
   would give a few hundred to train on. The
   question it asks is whether a trained repair step reaches problems blind sampling does not.
3. **Put the GPU on the lower target first.** At two of three seeds it is the first change that moves the
   goal set against the base (`ladder-loop.spec.md`, "L2t"); its third seed is running. (Added when it
   finished, the same day: at three seeds the gain in goal problems solved is in one seed of three and is not
   read as reach; `ladder-l2t-RESULT.md`.)

Recommended: 3 now, and 2 only after the lower target is read at three seeds, built inside whichever target
wins.

## Points the build made exact, and what it noticed

- Shared draws, known copies, the two checks' counting and the seed are in the spec ("Made exact when the
  stage was built").
- The smoke run did not reach attempts 3 to 5 (13 of its 14 first attempts verified), so attempt 4 was first
  seen in this run; it was checked from the rows before the read (above).
- Per-arm Lean checks are an upper bound for the alternate arm: a file both arms need is sent once.
- A resumed proof may be longer than any blind one (the cap is on new tokens); 34 alternate attempts hit the
  cap against the blind arm's 48, so it did not bind.

## Files

`experiments/rlvr_lean/ladder_l3a2_seed0_r1/steps/`: `report_ladder_l3a2.json` (every number above),
`repair_attempts_<loop>.jsonl`, `repair_states_<loop>.jsonl`, `repair_episodes.jsonl`.
