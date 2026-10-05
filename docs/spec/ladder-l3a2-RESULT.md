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
- **Within 2 attempts the gain is there and then it is caught up:** +0.0075 [+0.0015, +0.0137] within 2,
  +0.0058 [0.0000, +0.0118] within 3, +0.0048 [−0.0025, +0.0123] within 4.
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

## By attempt position, hard problems (verified / attempts made at that position)

| | Attempt 2 | Attempt 3 | Attempt 4 | Attempt 5 |
|---|---|---|---|---|
| Blind | 127 / 6,562 (1.93%) | 110 / 6,435 (1.71%) | 113 / 6,325 (1.79%) | 86 / 6,212 (1.38%) |
| Alternate | **177 / 6,562 (2.70%)** repair | 99 / 6,385 (1.55%) | **106 / 6,286 (1.69%)** repair | 86 / 6,180 (1.39%) |

- **The first repair step is worth 1.4 blind attempts** (L3a: 1.8), and 18 of its 177 are trimmed steps.
- **The second is worth about one.** After the first repair step has taken the episodes it can, a repair
  from the next fresh failure verifies no more often than a blind attempt in the other arm (1.69% against
  1.79%), and a little more often than the blind attempts beside it in its own arm (1.55%, 1.39%).
- **The blind arm catches up.** The episodes a repair step resolves are mostly ones a later blind attempt
  resolves too: the lead of 50 episodes at attempt 2 is 32 by attempt 5, and G by problem is 49 against 47.
- **Known copies:** at a repair step the model writes a proof Lean has already rejected in 15.7% of the hard
  failures at attempt 2 (1,028 of 6,562) and 15.5% at attempt 4 (977 of 6,286); the blind arm, 0.7% to 1.9%.
  Where a repair step had a state and wrote something new at attempt 2, it verified about 2.9% of the time.

## What the two checks say together

1. **Lean's state is information the model uses** (L3a: +0.026 over resuming without it), and one repair
   step straight after a failure verifies more often than a blind attempt, twice measured on separate
   attempts (L3a +0.0125 [+0.0027, +0.0228] within 2 attempts; here +0.0075 [+0.0015, +0.0137]).
2. **That is speed, not reach.** By five attempts the blind arm has resolved nearly the same episodes, and
   the same number of goal problems. It is the result training gave, one more time: the easier part of the
   hard problems is solved more reliably, and no problem is solved that plain sampling does not solve.
3. **The untrained model wastes the step about one time in six** by rewriting the rejected proof, and the
   steps it does not waste are worth about one and a half blind attempts. Neither is a property of repair;
   both are what training on repaired proofs would be meant to change.
4. **Cost.** A repair step generates fewer tokens and costs one more Lean check. With 37 Lean workers the GPU
   is the limit, so the alternate episode is slightly cheaper for the same result, not a reason to adopt it.

## For the owner: what to do with repair

1. **Leave the untrained repair step out of the round** (the branch's reading). Nothing is lost at five
   attempts and nothing gained.
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
