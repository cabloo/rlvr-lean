# Ladder loop, L4r (is the pretrained model capped by the size of its adapter? seed 0): THE ADAPTER'S SIZE CAPPED THE PRETRAINED MODEL. The same pretraining at rank 64 in the place of 16 gives +0.0133 successes per attempt on the goal set [+0.0077, +0.0192] (80.0 against 66.7 per 1,000), most of it on proofs of 4 lines or more, 107 goal problems solved reliably against 94, and 7 of the 81 goal problems nothing here had ever solved; it is 3.9 points worse on the easy rung, and its training loss was no better

Labelled **pretrained on published proofs**, as everything built on that file is.

Spec: `docs/spec/ladder-loop.spec.md`, "L4r: is the pretrained model capped by the size of its adapter?",
read fixed before the run. Why: `ladder-l4-RESULT.md` (three times the ceiling's proofs bought no more reach;
the rounds narrowed the model). Run 2026-10-09 (UTC) at Lean v4.27 through the
`oeis` pool, on the GPU box: tasks `rlvr_lean/ladder_l4_rank_smoke_r1` (1 minute: a rank-64 adapter
trained, saved, served and measured once) and `rlvr_lean/ladder_l4_rank_seed0_r1` (193 minutes: the training
118, the measurement 74), from 172e65d6a. One seed, a first run by the seed rule.

**What was run. One change.** `pre`'s pass again: from the base, one pass over the same 24,866 published
proofs in the same order, the same seed, learning rate, batch, sequence limit and target modules; the
adapter's rank is 64 and its alpha 128 where `pre` has 16 and 32 (149.9 million trained numbers against 37.5).
The model is `pre_r64`. It was measured as `pre` was, with `pre`'s sampling seeds, and read against `pre`'s
stored rows. It fitted the card with nothing to spare: the training's reserved memory peaked at the cap.

**The three checks pass.** The training took (mean loss 0.238 over the first tenth of its rows, 0.197 over
the last). It still writes proofs (2 of 5,832 rung attempts at the token cap, none without a verdict). Lean
answered (at most 1.05% of any set read without an answer, and that is `pre`'s first sampling).

## The primary

**All of G (392 problems), successes per attempt over 93 attempts, `pre_r64` minus `pre`, paired by problem:
+0.01333 [+0.00774, +0.01915]**, 80.0 per 1,000 against 66.7 (2,917 successes against 2,431 in 36,456 attempts
each). The interval is clear of zero and above: **THE ADAPTER'S SIZE CAPPED THE PRETRAINED MODEL.**

| Shortest published proof | Problems | `pre` (rank 16), per 1,000 | `pre_r64` | `pre_r64` minus `pre` |
|---|---|---|---|---|
| 1 line | 37 | 109.3 | 116.3 | +0.0070 [−0.0134, +0.0279] |
| 2-3 lines | 125 | 91.6 | 105.0 | +0.0133 [+0.0005, +0.0272] |
| 4-7 lines | 156 | 61.2 | 78.8 | +0.0176 [+0.0106, +0.0252] |
| 8 lines or more | 74 | 14.8 | 22.4 | +0.0076 [+0.0019, +0.0150] |
| 4 lines or more | 230 | 46.3 | 60.6 | +0.0144 [+0.0090, +0.0201] |
| **All of G** | 392 | 66.7 | 80.0 | **+0.0133 [+0.0077, +0.0192]** |

| | `pre` | `pre_r64` |
|---|---|---|
| Goal problems solved in 93 attempts | 238 | 245 (gained 38, lost 31; sign test p = 0.47) |
| The same, 4 lines or more | 115 | 125 (gained 25, lost 15; p = 0.15) |
| Solved in at least one episode of 8 / reliably | 234 / 94 | 241 / 107 |
| The same, 4 lines or more | 111 / 39 | 123 / 51 |
| Of the 81 goal problems nothing stored had solved | 0 | 7 (one of 1 line, one of 2-3, one of 4-7, four of 8 or more; one of them 5 times in 93) |
| Rung below the band, pass rate | 0.264 | 0.255 (−0.009 [−0.038, +0.020]) |
| Rung in the band | 0.414 | 0.398 (−0.016 [−0.050, +0.018]) |
| Rung above the band | 0.819 | 0.780 (**−0.039 [−0.056, −0.023]**) |
| Attempts that time out in Lean, on the rungs / on G | 0.65% / 3.6% | 1.06% / 4.3% |
| Share of distinct attempts, on the rungs / on G | 0.870 / 0.817 | 0.875 / 0.818 |

## What it says

- **More room, more reach, where the proofs are long.** A fifth more successes per attempt on the goal set
  and half again as many on its 8-line problems (22.4 against 14.8); 13 more goal problems solved reliably
  (107 against 94, and 51 against 39 where the proof needs 4 lines or more). The count of distinct problems
  solved at all moves less (245 against 238) and is not clear of chance: the gain is mostly depth on what was
  within reach, with a first hold on some that were not (the 7 never-solved problems; the arm's two models
  reached 4 and 3 of them, each once).
- **It is a trade.** On the easy rung the larger adapter is 3.9 points worse, clear of zero, and it is not
  better on the other two. One pass, one learning rate: an adapter four times the size may be less finished on
  the common easy patterns than the small one after the same pass.
- **The training loss did not show it, and pointed the other way.** The loss on rows not yet trained on, by
  twentieth of the pass: `pre` 0.266, 0.225, 0.218 ... 0.192, 0.189; `pre_r64` 0.255, 0.221, 0.217 ... 0.199,
  0.196: the larger adapter is lower over the first twentieth, the same within 0.002 from the third to the
  sixth, and higher by 0.003 to 0.008 in every twentieth from the seventh on. A proof's token loss is not its chance of checking: read by the loss, this check would
  have said "not the cap". The loss stays a check that a training ran, and is not used here to compare models.
- **What changes.** Every read of this project that says "not shown" was made at rank 16, under this cap:
  the dose curve, the three and six rounds from the base, the pretrained arm. The arm's narrowing in
  particular has a new suspect: an adapter with no room can only learn the rounds' proofs by giving something
  up. By the spec, the arm is run again from `pre_r64` before any more seeds are bought at rank 16, and the
  training-rule checks (L4t) start from `pre_r64` with every model at rank 64; their first model, the old
  rule at rank 64, is the direct test of that suspect.

## What it does not say

- **Not that 64 is enough, nor that an adapter is.** Rank 64 is what fits this card at this sequence limit
  (the training's reserved memory reached the 11.94 GiB cap). A null was written into the spec as "does not
  show that full training would not help"; a positive leaves the same question open one step up: the larger
  run the owner has named may want the whole model, or a larger adapter than this card can train.
- **Not a dose curve in rank.** Two points, one seed, one learning rate. The easy rung's loss may be a
  learning-rate or a one-pass effect and not a property of the size.
- **One seed.** By the spec it is confirmed by what is built on it, not by a second pretraining.

## A fault of mine during the run

For about 30 minutes of the measurement (12:02 to 12:34 UTC) a Lean-only replay of mine, at background
priority, ran on the pool beside it: I had taken its checks to be answered from the pool's cache, and the pool
does not cache a check that timed out, so several hundred were run again. Each check has its own 30-second
limit and a server runs a fixed number at once, so a verdict is not changed by the queue; if the contention
cost anything it is counted against `pre_r64` (a timeout is a failure, and its share on G is 4.3% against
`pre`'s 3.6%). The measurement took 74 minutes where the arm's two took 49 and 35. The replay was stopped when I saw
it; such replays run only while no measurement does.
