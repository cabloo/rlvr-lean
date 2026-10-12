# Ladder loop, L5d (fresh draws): THE MODEL'S OWN PROOFS BESIDE PUBLISHED ONES BEAT PUBLISHED ONES ALONE ON THE GOAL SET. Over three fresh seeds, trained from the once-pretrained model, the two kinds with the model's own proofs read +12.69 successes per 1,000 attempts above the published kind (interval over seeds +9.08 to +16.30; positive at 3 seeds of 3)

Labelled **pretrained on published proofs**, as everything built on that file is.

Spec: `docs/spec/ladder-loop.spec.md`, "L5d: the three kinds again, over fresh draws, measured on the goal set's
first sampling" (fixed 2026-10-11, before L5n's read and before any fresh seed ran) and its "Made exact by the build".
Run from `28f015f6a` with the recipe L5n named (`mean_constant`: each set trained in
four orders, one pass, constant learning rate; the adapter measured is the mean of the four). THREE fresh seeds, one
task each on the GPU box, at Lean v4.27 through the `oeis` pool: `rlvr_lean/ladder_l5_draws_seed3_r1`,
`ladder_l5_draws_seed4_r1`, `ladder_l5_draws_seed5_r1` (task 43b9968c, done 10:59 UTC 2026-10-12). **All checks pass
at each seed; no run is INCONCLUSIVE.**

**The number of seeds.** The section fixed it before any fresh seed ran: six when the owner's approval came before
seed 5's figures were read, three otherwise. Seed 5 finished unread. The owner answered on 2026-10-12 (about 11:06
UTC): "let's do 3 seeds. I don't need super high confidence to keep pushing in this direction". Seed 5 was read after
that answer, and the read is at three. Seeds 6 to 8 were not run.

**Why this stage.** L5's three seeds left one sign for the owner's bet: on the goal set, the two models trained on
the model's own proofs beside published ones were above the model trained on published ones alone at 3 seeds of 3.
It was found on those seeds and not fixed in advance, so it was not a result. L5d asks it again on fresh draws, with
the read fixed first and the recipe that lowers the spread between trainings.

## What was run

At each fresh seed, three sets of 800 rows built by L5's own rule for the seed, from L5's ONE finished search
(520 deep solves): `published` (800 published rows), `shallow` (400 of the model's own ordinary solves and 400
published rows), `deep` (400 of its own deep solves and 400 published rows). Each trained from `pre_r64` (the
once-pretrained model, 77.73 per 1,000) at rank 64, in four orders, and averaged. Each measured on G's first sampling
alone: 32 attempts at each of 392 held-out goal problems, sampling seed 1001 (12,544 episodes). No rungs, no second
sampling, no held-aside set.

## The read fixed before any fresh seed ran

PRIMARY: the mean of `shallow` and `deep`, minus `published`, at each fresh seed. Every interval is over SEEDS
(trainings), never over problems: the mean of the per-seed differences plus and minus Student's t (4.303 at three)
times their standard deviation over the root of three.

**Result: +12.69 per 1,000, interval over the three seeds [+9.08, +16.30]: above zero.** Seed 3 +13.91, seed 4
+13.07, seed 5 +11.08. The standard deviation of the three differences is 1.45 per 1,000.

## The figures

Successes per 1,000 attempts on all of G (392 problems, 32 attempts), with the binomial sampling noise of one
measurement in brackets; the start model `pre_r64` reads 77.73:

| seed | `published` | `shallow` | `deep` | mean of the two, minus `published` |
|---|---|---|---|---|
| 3 | 90.88 (2.19) | 99.41 (2.29) | 110.17 (2.38) | +13.91 |
| 4 | 95.34 (2.26) | 105.31 (2.29) | 111.53 (2.40) | +13.07 |
| 5 | 97.58 (2.26) | 102.84 (2.26) | 114.48 (2.43) | +11.08 |
| standard deviation over the seeds | 3.41 | 2.96 | 2.20 | 1.45 |

On the 200 goal problems the start model solves in its own first sampling, and on G' (the 192 it does not):

| seed | the 200: `published` | `shallow` | `deep` | G': `published` | `shallow` | `deep` |
|---|---|---|---|---|---|---|
| 3 | 167.50 | 181.25 | 202.03 | 11.07 | 14.16 | 14.49 |
| 4 | 177.97 | 196.09 | 203.44 | 9.28 | 10.74 | 15.79 |
| 5 | 181.41 | 188.28 | 205.31 | 10.25 | 13.84 | 19.86 |

Goal problems solved at least once in the 32 attempts (gained, lost against that seed's `published`), and of the 74
long problems (8 lines or more):

| seed | `published` | `shallow` | `deep` | long: `published` | `shallow` | `deep` |
|---|---|---|---|---|---|---|
| 3 | 206 | 220 (+44, −30) | 226 (+45, −25) | 20 | 18 | 22 |
| 4 | 204 | 199 (+25, −30) | 219 (+42, −27) | 16 | 16 | 18 |
| 5 | 203 | 219 (+39, −23) | 226 (+45, −22) | 14 | 17 | 18 |

## Beside the primary, naming nothing

Each with its interval over the three seeds, per 1,000 attempts:

- `deep` minus `published`: **+17.46 [+13.41, +21.50]**.
- `shallow` minus `published`: +7.92 [+1.93, +13.91].
- `deep` minus `shallow` (the depth): +9.54 [+2.31, +16.77]. In L5's three seeds this read +0.4 with one training a
  model; with the averaged recipe the deep solves are above the ordinary ones at 3 fresh seeds of 3.
- The primary's difference is positive at 3 of the 3 fresh seeds.

**Where it sits: REACH TOO.** The primary's difference on the 200 problems the start model already solves is
[+9.19, +31.70] per 1,000; on G' it is [+0.26, +8.97]: above zero, so the section's name is REACH TOO. The interval on
G' ends a quarter of a point above zero and is over three seeds.

**The spread.** With four trainings averaged, a kind's figure moves by 2.2 to 3.4 per 1,000 between fresh seeds (rows
drawn and orders together), about the sampling noise of one measurement (2.2 to 2.4). With one training a model, L5n read about 15 between draws. `published` is the least steady
of the three (90.88, 95.34, 97.58).

## Timeouts, and the second check

Each set's share of attempts that timed out in Lean: seed 3 `published` 2.30%, `shallow` 3.20%, `deep` 3.96%; seed 4
2.70%, 2.80%, 3.58%; seed 5 2.59%, 3.26%, 4.61%. Where two models of a seed differ by more than a point, that seed's
timed-out attempts are checked again (Lean only, no measurement running) and the reads given both ways:

- Seed 3: the primary's difference is +13.91 as stored and +13.08 with the second check.
- Seed 4: no two models are more than a point apart; no second check is owed.
- Seed 5 (11:06 to 11:43 UTC 2026-10-12, with no measurement running): 10 of 330, 9 of 416 and 9 of 588 timed-out
  attempts verify (`published`, `shallow`, `deep`); the primary's difference is +11.08 as stored and +11.00 with the
  second check.

With the second check at the two seeds that owe one (+13.08, +13.07, +11.00) the mean is +12.38 and the interval
over the seeds [+9.41, +15.36]: above zero both ways.

## What follows, by the rule fixed in advance

THE MODEL'S OWN PROOFS BEAT: "the bet's first condition holds on the goal set at this size. L6 (a second round) runs
with this recipe and with as many seeds an arm as this stage's measured standard deviation asks for its own
difference; the cycle at full size has its evidence to put to the owner."

## What it cannot say

- **The bar is a second pass.** `published` trains on rows the start model was already trained on; the model's own
  proofs are of problems it never trained on. The result says: its own proofs of new problems are worth more than
  reading published proofs again. It does not say they are worth more than PUBLISHED proofs of new problems (the
  fourth kind, not built without the owner's word).
- **It is the once-pretrained model's result.** L5f ran the same three sets of seed 3 from the twice-pretrained model
  (`second_full`, 110.25): there `published` reads 144.77, `shallow` 126.35 and `deep` 133.45, and the same difference
  is −14.87 (ONE seed; `docs/spec/ladder-l5f-RESULT.md`). So the own proofs add from the weaker start and are
  not shown to add from the stronger one; the own proofs in both are `pre_r64`'s, found before the second pass.
- **The level is one a second pass also reaches.** `deep` reads 110 to 114; a whole second pass over the 24,866
  published rows reads 110.25, and 3,268 published rows read 111.85 (L5s, seed 3). 400 own deep solves beside 400
  published rows reach in 800 rows what published rows alone reach in several thousand; they do not go past it here.
- **G alone, its first sampling, 32 attempts.** It is the one set more than a dozen models have now been compared
  on. Nothing of the held-aside set, where reach on OTHER hard problems was not shown at three seeds in L5.
- **One search.** The 520 deep solves and the ordinary ones come from seed 0's search and map; two seeds' deep rows
  share about 308 of 400 problems. The spread over seeds is over draws and trainings, not over searches.
- One start model, one size of set, one pass, three seeds: the interval's half-width is 3.61 per 1,000 with a
  standard deviation known from three figures.
