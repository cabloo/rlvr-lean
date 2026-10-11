# Ladder loop, L5n (what is one training worth?): THE ROWS DRAWN MAKE MOST OF THE SPREAD. Trained on one fixed set of 800 published rows in four orders, a model moves by 5.0 successes per 1,000 attempts (standard deviation); between three draws of 800 rows it moved by 15.2. Decay of the learning rate is NOT SHOWN to lower it (5.6). The average of four adapters GIVES AT LEAST THE MEAN OF ITS MODELS (110.0 against 106.0). By the rule fixed before the run, a later stage trains each model four times and measures the averaged adapter.

Labelled **pretrained on published proofs**, as everything built on that file is.

Spec: `docs/spec/ladder-loop.spec.md`, "L5n: what is one training worth? The spread between trainings, and a
recipe that lowers it" (fixed 2026-10-11, after L5's seed 1) and its "Made exact by the build". One seed (nothing here is drawn by a seed). Run 2026-10-11 at Lean v4.27 through the `oeis`
pool (15 workers throughout: the main server 12, the GPU box 3), on the GPU box, in two tasks:

- `rlvr_lean/ladder_l5_noise_seed0_r1` (task e034e858): ended `task_failed` at 18:10 UTC with three models of ten
  measured (`constant_1`, `decay_1`, `constant_2`). **VOID for infrastructure, not a result:** the sampling engine did
  not start a fourth time (its cache came to 1.87 GiB against the 1.88 it starts with, while another user of the card
  held memory). Fixed at ea0a74813: an engine that did not start is started again, twice at most.
- `rlvr_lean/ladder_l5_noise_seed0_r2` (task b017787b, from ea0a74813): 20:25 to 23:11 UTC, `done`. It found the first
  task's steps done and went on from the fourth model. **All checks pass; the run is not INCONCLUSIVE.**

**Why this stage.** L5's seeds moved the same kind of model by up to 24 successes per 1,000 attempts, so no difference
between two single trained models could be read. The recipe has a constant learning rate and no decay, so the model
saved might be the last point of a walk. L5n asks how much of the spread is the ORDER of the rows alone, and whether
one of two cheap changes (a learning rate that falls to zero; the average of four adapters) lowers it.

## What was run

One fixed set of rows: the 800 published rows L5's `published` of seed 0 was trained on. From `pre_r64` at rank 64,
one pass each, in the orders of order seeds 1 to 4: `constant_k` (the recipe as it is) and `decay_k` (one change: after
the five warm-up steps the learning rate falls in a straight line to zero). Then two models that nothing trained:
`mean_constant` and `mean_decay`, the adapter whose every number is the mean of its four adapters'. Each of the ten is
measured on the goal set's first sampling alone: 32 attempts at each of the 392 goal problems, sampling seed 1001, so
every model here pairs by problem with every stored first sampling.

## The checks (all PASS)

- Each trained adapter is not the start adapter's (149,913,545 to 149,913,583 of its 149,913,600 numbers differ) and
  is read back at rank 64 and alpha 128.
- Each model was trained on the 800 rows, each once, and on nothing else (the training loop's own record).
- Each averaged adapter IS the mean of its four (0 of its 149,913,600 numbers are not), is none of them, and is the
  one its set was sampled from.
- Lean answered: at most 0.00016 of a set without an answer (0.02 allowed).

## The figures

Successes per 1,000 attempts on all of G over the 32 attempts; the sampling noise of one figure is 2.3.

| order | `constant` | `decay` | `decay` minus `constant` |
|---|---|---|---|
| 1 | 112.3 | 103.9 | −8.5 |
| 2 | 107.9 | 109.9 | +2.0 |
| 3 | 102.2 | 106.5 | +4.3 |
| 4 | 101.7 | 96.7 | −5.0 |
| mean of the four | 106.0 | 104.2 | −1.8 |
| standard deviation | **5.0** | **5.6** | |
| the averaged adapter | **110.0** | **107.4** | |
| best of the four | 112.3 | 109.9 | |

Beside them, from stored runs, on the same 32 attempts:

| model | per 1,000 | goal problems solved |
|---|---|---|
| `pre_r64`, the start model | 77.7 | |
| L5 `published` seed 0 (the SAME 800 rows, a fifth order) | 103.8 | 220 |
| L5 `published` seed 1 (another draw of 800 rows) | 78.6 | 177 |
| L5 `published` seed 2 (another draw of 800 rows) | 76.5 | 182 |

Goal problems solved in the 32 attempts: `constant` 215, 221, 209, 202 (standard deviation 8.1); `decay` 214, 212,
217, 210 (3.0); `mean_constant` 220; `mean_decay` 218.

## The reads, as fixed before the run

- **THE SPREAD OF THE RECIPE AS IT IS: THE ROWS DRAWN MAKE MOST OF IT.** The four `constant` figures have a standard
  deviation of 0.00504 per attempt; L5's `published` over its three seeds (rows and order both drawn) has 0.01520. The
  first is less than half of the second.
- **DOES DECAY LOWER IT: NOT SHOWN.** 0.00559 against 0.00504. With four against four only a ratio of about three
  between two standard deviations is beyond chance.
- **AND ITS LEVEL: NAMED NOTHING.** `decay` minus `constant` over the four orders: −0.00179 per attempt,
  [−0.01126, +0.00768].
- **AVERAGING: AVERAGING GIVES AT LEAST THE MEAN OF ITS MODELS,** for both. `mean_constant` 0.11001 against 0.10603
  (+0.00399; the best of its four 0.11232). `mean_decay` 0.10738 against 0.10423 (+0.00315; the best 0.10985).
- **WHAT FOLLOWS: A LATER STAGE TRAINS EACH MODEL FOUR TIMES AND MEASURES THE AVERAGED ADAPTER.**

## What it says

**One training of this recipe is worth about ±5 per 1,000 from the order of its rows alone.** That is twice the
sampling noise of the measurement (2.3), so it is a property of the training and not of the 32 attempts.

**Seed 0's `published` was not a lucky order: its level belongs to its rows.** The same 800 rows in five orders give
101.7 to 112.3 (mean 105.6, standard deviation 4.5), every one at least 24 above the start model. The two other draws
of 800 rows, one order each, give 78.6 and 76.5, the start model's level. So WHICH 800 published rows a model is
trained on moved it by about 14 per 1,000 (the spread between draws with the order's part taken off:
√(15.2² − 5.0²)), roughly three times what the order does. This rests on three draws, and on ONE order for each of
the two weak ones: whether they are weak by their rows or by that order is not told apart here (see "The reading that
is left open" below). L5d's fresh seeds add three draws, each in four orders.

**Averaging four orders buys level, and little spread between draws.** The averaged adapter is 4.0 above the mean of
its four and inside their range. It removes the order's part only: if the weak draws are weak by their rows, the
spread between draws goes by arithmetic from 15.2 to about 14.6, and for a comparison between KINDS of rows more
draws (seeds) is what narrows the read, not four orders a model. If they were weak by their order, averaging removes
most of that spread.

**The decay did not help.** The "last point of a walk" hypothesis predicted a narrower spread with a learning rate
that ends at zero. The spread is the same (5.6 against 5.0) and the level is not higher. At four against four this
test could only have seen a spread cut to a third, so a smaller effect is not ruled out.

## What it does not say

One set of rows, published rows only: nothing here says how much a draw of the model's OWN proofs moves a model. Four
orders. G alone at 32 attempts; nothing of the held-aside set. From `pre_r64` and no other start. The spread between
draws of rows is a standard deviation of three values (L5's three seeds), each in one order.

## Looked at after the result (reads of stored attempts; no run, nothing fitted)

**Were the two weak models of L5 really trained models, or the start adapter served by mistake?** Trained models.
Every model was sampled with one seed, so a model that IS the start model would write the start model's text at most
places. The share of the 12,768 stored attempts whose text is word for word the start model's at the same place
(problem, side, attempt): `published` seed 1 0.068 and seed 2 0.069, as every other trained model here (0.061 to
0.110). Two models trained on the SAME rows in different orders share more with each other (0.145 and 0.160) than two
trained on other rows (0.073 to 0.091), and the averaged model shares 0.243 with one of its four.

**Is the gain one tactic the high models picked up? No.** Every tactic word of every stored attempt was counted for
the three models at the start level (the start model, `published` of seeds 1 and 2) and the twelve single models
above it (six on seed 0's published rows, six with the model's own proofs). Two things put every high model on one
side of every low one: fewer one-line attempts (90 to 133 per 1,000 attempts against 148 to 178) and more attempts
that hand `nlinarith` a product hint, `mul_nonneg` (47 to 96 against 24 to 45). The hint is not what the rows differ
by: 9.0% to 11.1% of the rows of every draw hold one, own rows and published rows alike. And it does not carry the
gain: of a high model's 17 to 32 per 1,000 over the start model, attempts with a product hint carry 6 to 16 and
attempts without one 8 to 23; attempts of two lines or more verify at 98 to 114 per 1,000 against 81 to 84. The high
models are better across the board.

**The reading that is left open.** This run's first read ("the rows drawn make most of it") sets four orders of ONE
draw against three draws in one order each. It cannot tell a draw that adds all or nothing by its rows from a
training that sometimes fails by its order and did not in seed 0's five orders (were one training in three to fail,
five in a row would all succeed about one time in eight). One earlier figure leans to the second: on the 93 attempts
of all of G, published-only sets of 1,262, 12,433 and 24,866 rows read 99.2, 99.3 and 108.9 per 1,000, and the set of
3,422 rows 86.8, though the 1,262 rows are among its rows. L5d measures three more published-only draws, each
averaged over four orders: draws at both levels there say the rows; all three high say the order.

**Lean timeouts.** Between 1.3% and 2.6% of a model's attempts timed out in Lean and count as failures (`constant_1`
1.3%, the others 1.7% to 2.6%). On L5's seed 2, attempts that timed out and were checked again verified at 2% to 3%
(13 of 676; 44 of 1,510): under 1 per 1,000 on a model's figure, and less on a difference between two models, against
a spread of 5. The second check of this run's timed-out attempts is running (Lean only, at background priority) for
the seven models whose stored attempts have been pulled; the three measured by the first task (`constant_1`,
`decay_1`, `constant_2`) are stored on the GPU box only. Its counts are added below when it ends.

## What follows

- **L5d (the three kinds again over fresh draws) runs with `mean_constant`:** each set trained in four orders, the
  averaged adapter measured. The setting was changed to it at 28f015f6a; the stage's first step holds the setting to
  this run's report on the box. Its smoke run first, then seeds 3, 4 and 5.
- **More seeds is the stronger lever for L5d, and it is the owner's to grant.** L5's primary there (the mean of
  `shallow` and `deep`, minus `published`) had a standard deviation of 11.1 over three seeds. With three fresh seeds
  the 95% interval's half-width is 2.48 times that, about 27, against an effect found of 16.5; with six it is 1.05
  times, about 12. Averaging cuts the order's part only (about 11 to about 10). Seeds 6 to 8 were asked for on
  2026-10-11 and are not approved; they are queued only if the answer comes before seed 5's figures are read.
- **L6 stays on hold** until L5d gives the spread of each kind over fresh draws.
- **Why one draw of 800 published rows adds 25 per 1,000 and another adds nothing is not explained.** Nothing on the
  surface of the rows tells them apart (`ladder-l5-RESULT.md`, "Looked at after the three seeds"). L5d trains three
  more draws of `published`, each in four orders: whether a draw's effect is all or nothing is read there first.
