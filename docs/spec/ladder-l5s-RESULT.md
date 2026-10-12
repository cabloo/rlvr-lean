# Ladder loop, L5s (the dose): NOT SHOWN AT ONE SEED. Four times as many of the model's own proofs (1,634 against 400) did not raise the solve rate on the held-out goal set: `stock` 105.55 successes per 1,000 attempts against 110.17 for that seed's `deep`. The same number of rows, all published, reads 111.85. No further seed runs

Labelled **pretrained on published proofs**, as everything built on that file is.

Spec: `docs/spec/ladder-loop.spec.md`, "L5s: the dose" (fixed 2026-10-12, before any run of this stage) and its
"Made exact by the build". Built at `6d5c1c4fd`. ONE seed (3, the first fresh seed of
L5d). Run 2026-10-12 at Lean v4.27 through the `oeis` pool (30 workers), on the GPU box, as one task:
`rlvr_lean/ladder_l5_stock_seed3_r1` (task 52d006e0), 04:33 to 07:11 UTC, `done`. Its smoke run
(`ladder_l5_stock_smoke_r1`, task 9b0229cd) passed first. **All checks pass; the run is not INCONCLUSIVE.**

**Why this stage.** Every model trained with the model's own proofs so far had 400 of them. In stock, found and
checked in Lean already, were 1,634 own solves of problems the start model `pre_r64` solves less than half the time:
L5's 520 deep solves and the 1,114 base-map problems solved in 1 to 3 of 8 attempts. L5s asks whether using all of
them raises the solve rate on the held-out goal set G.

## What was run

From `pre_r64` (rank 64, one pass), two sets, each trained in four orders with the recipe L5n named (constant
learning rate, the average of the four adapters is the model measured):

| set | rows | of which the model's own | published |
|---|---|---|---|
| `stock` | 3,268 | 1,634 (520 deep, 1,114 ordinary) | the first 1,634 rows of the seed's order of the pretraining file |
| `published_stock` | 3,268 | 0 | the first 3,268 rows of that order |

Each model was measured on G's first sampling alone: 32 attempts at each of 392 held-out goal problems, sampling
seed 1001 (12,544 episodes). No held-out problem, no problem of G and no held-aside problem is in any training row
(the training loop's own record: 0 of each, for all eight trainings).

## The read fixed before the run

`stock` minus `deep` of L5d's run of the same seed (110.17 per 1,000): at or above zero ESCALATE to seeds 4 and 5;
below zero by more than 10 per 1,000 MORE OF ITS OWN PROOFS COSTS AT THIS SEED; between NOT SHOWN AT ONE SEED.

**Result: -4.62 per 1,000 (105.55 against 110.17): NOT SHOWN AT ONE SEED.** By the stage's own rule no further seed
is queued: the dose is not the lever at this size.

## The figures

Successes per 1,000 attempts on all of G, with the binomial sampling noise of one measurement in brackets:

| model | rows | per 1,000 | on the 200 the start model solves | on G' (192) | goal problems solved in 32 | long (74) solved | timed out |
|---|---|---|---|---|---|---|---|
| `pre_r64` (start) | | 77.73 | 152.34 | 0.00 | 200 | 18 | |
| `stock` | 3,268 | 105.55 (2.35) | 191.41 | 16.11 | 225 (+47, -22) | 21 (+7, -4) | 3.35% |
| `published_stock` | 3,268 | 111.85 (2.40) | 200.62 | 19.37 | 224 (+48, -24) | 18 (+5, -5) | 2.81% |
| L5d seed 3 `published` | 800 | 90.88 | | | | | |
| L5d seed 3 `shallow` | 800 | 99.41 | | | | | |
| L5d seed 3 `deep` | 800 | 110.17 | | | | | |

Beside the read, naming nothing at one seed:

- **Its own proofs at this dose.** `stock` minus `published_stock`: -6.30 per 1,000. At 800 rows the two own-proof
  kinds LED the published kind by +13.91 at this seed. With four times the rows the lead is gone.
- **The steps without its own proofs.** `published_stock` minus L5d's `published` of this seed: +20.97 per 1,000
  (3,268 rows against 800 of the same order). Most of what a larger set adds is added by published rows alone.
- **Overfitting.** Neither model narrows against the start model (lost 22 and 24 goal problems beside 47 and 48
  gained); the share of distinct attempts on G rises (0.911 and 0.891 against 0.818).
- **Timeouts.** The two shares are not more than a point apart: no second check of timed-out attempts is owed.

## What it says, and what it cannot

- Read with the night's other figures, every route from `pre_r64` lands in one band: 800 rows with 400 own deep
  solves (110.17 at seed 3, 111.53 at seed 4), 3,268 published rows (111.85), a whole second pass over 24,866
  published rows (`second_full`, 110.25), 3,268 rows with 1,634 own solves (105.55). Goal problems solved in 32
  attempts: 219 to 225. **The model's own deep solves reached that band with a quarter of the rows; nothing here went
  past it.**
- `stock` is not `deep` at a higher dose alone: 1,114 of its 1,634 own rows are ORDINARY solves (the kind that read
  lower than deep ones at 800 rows), and it has four times the steps. The stage was built to answer "does using all
  of the stock help", and the answer at one seed is no.
- ONE seed. The spread between two averaged models of different draws is not measured at this size, and the binomial
  noise of a difference of two such measurements alone is about 3.4 per 1,000: -4.62 and -6.30 are 1.4 and 1.9 of it.
  Nothing here says more own proofs COST.
- The own proofs are `pre_r64`'s. What a second round's proofs (found by a better model) would add is L6's and the
  full-size cycle's question, not this stage's.

## What follows

No further seed of this stage. The read goes into the owner's decision on the full-size cycle beside L5f's (the same
three kinds trained from the twice-pretrained model), which asks whether own proofs go past the band from a better
start.
