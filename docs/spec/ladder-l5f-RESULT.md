# Ladder loop, L5f (the stack): THEY COST AFTER A SECOND PASS AT THIS SEED. Trained from the twice-pretrained model, the two kinds with the model's own proofs read 126.35 and 133.45 successes per 1,000 attempts on the held-out goal set; the published kind reads 144.77, the highest figure of the project. All three are above the model they start from (110.25)

Labelled **pretrained on published proofs**, as everything built on that file is.

Spec: `docs/spec/ladder-loop.spec.md`, "L5f: the three kinds again, from the twice-pretrained model" (fixed
2026-10-12, before any run of this stage) and its "Made exact by the build". Built at
`2220ba9a7`. ONE seed (3). Run 2026-10-12 at Lean v4.27 through the `oeis` pool (30 workers), on the GPU box, as one
task: `rlvr_lean/ladder_l5_full_seed3_r1` (task 04d5fa6f), 07:20 to 09:04 UTC, `done`. Its smoke run
(`ladder_l5_full_smoke_r1`, task d7fc53e0) passed first. **All checks pass; the run is not INCONCLUSIVE.**

**Why this stage.** Two routes from the start model `pre_r64` (77.73) read the same on G's first sampling: a whole
second pass over the 24,866 published rows (`second_full`, 110.25) and 800 rows of which 400 are the model's own deep
solves (`deep`: 110.17 at seed 3, 111.53 at seed 4). Nothing had trained the own proofs FROM the twice-pretrained
model. L5f asks whether they still add there.

## What was run

L5d's stage with one change, the start adapter: `second_full`, L4p's stored one (only read; checked to be there, at
rank 64, and not `pre_r64`'s). The three sets are L5d's of seed 3, held id for id and order for order to the ones
L5d's seed-3 run stored and trained: `published` (800 published rows), `shallow` (400 own ordinary solves and 400
published rows), `deep` (400 own deep solves and 400 published rows). Each trained in four orders, one pass, constant
learning rate; the model measured is the adapter that is the mean of the four. Each measured on G's first sampling
alone: 32 attempts at each of 392 held-out goal problems, sampling seed 1001 (12,544 episodes). The own solves are
`pre_r64`'s; the published rows are ones `second_full` has trained on twice.

## The read fixed before the run

THE STACK: the mean of `shallow` and `deep`, minus `published`, all three from `second_full`. At or above zero
ESCALATE to seeds 4 and 5; below zero by more than 8.5 per 1,000 THEY COST AFTER A SECOND PASS AT THIS SEED; between
NOT SHOWN AT ONE SEED.

**Result: -14.87 per 1,000: THEY COST AFTER A SECOND PASS AT THIS SEED.** By the stage's own rule no further seed
runs and the outcome goes to the owner.

## The figures

Successes per 1,000 attempts, with the binomial sampling noise of one measurement in brackets:

| model | all of G | on the 200 the start model `pre_r64` solves | on G' (192) | goal problems solved in 32 | reliably (16+ of 32) | long (74) solved | distinct attempts | timed out |
|---|---|---|---|---|---|---|---|---|
| `pre_r64` | 77.73 | 152.34 | 0.00 | 200 | 9 | 18 | 0.82 | |
| `second_full` (the start here) | 110.25 | 191.25 | 25.88 | 224 | 19 | 22 | 0.80 | 2.25% |
| `published` | 144.77 (2.56) | 238.44 | 47.20 | 235 (+39, -28) | 36 | 28 (+11, -5) | 0.82 | 2.56% |
| `shallow` | 126.35 (2.53) | 216.56 | 32.39 | 245 (+43, -22) | 17 | 25 (+7, -4) | 0.89 | 3.35% |
| `deep` | 133.45 (2.53) | 229.38 | 33.53 | 236 (+41, -29) | 26 | 23 (+7, -6) | 0.90 | 4.73% |

Gained and lost are against `second_full`. Each kind minus the same kind trained from `pre_r64` at this seed (L5d:
90.88, 99.41, 110.17): +53.89, +26.95, +23.28.

**The timed-out attempts, checked again** (Lean only, 09:06 to 09:33 UTC, no measurement running; the shares are
more than a point apart, so the rule calls for it): 5 of 327 (`published`), 14 of 428 (`shallow`) and 7 of 604
(`deep`) verify on a second check. Counting each as a resolved episode: 145.17, 127.47, 134.01; **the stack -14.43.
The read stands both ways.**

## Where the gain sits (stored attempts, counted after the run; names nothing)

Against `second_full`, by how often `second_full` solved a problem in its 32 attempts:

| problems `second_full` solved | problems | `published` | `shallow` | `deep` |
|---|---|---|---|---|
| 0 of 32 | 168 | +113 successes, 39 now solved | +84, 43 now solved | +107, 41 now solved |
| 1 to 3 | 97 | +125 | +98 | +105 |
| 4 to 15 | 108 | +227, 20 now reliable | +87, 5 now reliable | +137, 13 now reliable |
| 16 to 32 | 19 | -32 | -67 | -58 |

On problems the start model never or rarely solved, the three kinds gain about alike. The published kind's lead is
in the middle: it makes problems solved sometimes into problems solved reliably. Goal problems solved at least once
by one of the three: 277; by all three 190; by `published` alone 8; by an own-proof kind and not by `published` 42.

## What it says, and what it cannot

- **After a whole second pass over published proofs, 400 of the model's own proofs in a set of 800 read lower than
  800 published rows, at this seed.** From `pre_r64` the same sets read the other way at two seeds (+13.91, +13.07).
  The own proofs here were found by `pre_r64`, the weaker model: what a search by the twice-pretrained model would
  find is not tested.
- **The published kind's +34.52 over `second_full` is the largest step of the night, and its cause is not known.**
  Two things changed at once: 800 more rows, and the average of four trainings. `second_full` is ONE training at a
  constant learning rate. L5c asks which it is, and whether more of it goes higher.
- The own-proof kinds solve as many or more goal problems at least once (245 and 236 against 235) at a lower rate:
  breadth without reliability, as L4c read it.
- ONE seed, one draw of rows, one `second_full`. A scout's read: it stops this stage and does not conclude over
  seeds. The level against `second_full` carries that one training's order (about 5 per 1,000).
- Measured on G's first sampling alone. The held-aside set and G's second sampling are not measured here.

## What follows

No further seed of this stage. The outcome goes to the owner beside L5s's (the dose: not shown) and L5d's (own
proofs lead from `pre_r64` at seeds 3 and 4; seed 5 run, unread until the owner's answer on six seeds). L5c runs
next: what made the published kind's step, and whether a wider average, a second averaged step or more rows goes
higher.
