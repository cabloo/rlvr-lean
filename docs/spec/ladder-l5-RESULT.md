# Ladder loop, L5 (do DEEP solves teach what ordinary solves do not? three seeds): NOT SHOWN AT THREE SEEDS on the primary (held-aside hard problems, `deep` minus `published` +0.0015 [−0.0029, +0.0057]), and DEPTH IS NOT SHOWN TO MATTER (`deep` and `shallow` are level on every set). On the goal set THE LEAD IS CONFIRMED by its rule on the two new seeds (`deep` minus `published` +0.0203 [+0.0144, +0.0266]); read over trainings it is +16.7 successes per 1,000 at three seeds of three (p = 0.05): the model's own proofs beside published ones beat published ones alone, hard-won or ordinary alike (103.7 and 103.2 per 1,000 against 87.0). Reach on the long problems NOT CONFIRMED. Part 1 below is seed 0 as first read; Part 2 is the read over three seeds.

**Correction, 2026-10-11 (after L5's seeds 0 and 1).** The differences this note reports between two single trained
models do not stand as effects. L5 trained the same three kinds of model at two seeds (another draw of the rows and
of their order; nothing else changes) and their figures on all of G moved between the seeds by up to 24 successes
per 1,000 attempts and 45 goal problems (`published` 103.1 and 79.5, with 262 and 217 goal problems solved; `deep`
112.5 and 97.2; `shallow` 104.5 and 110.0). Every interval in this note is over problems, with one training a model:
it says nothing of what another training of the same kind would give. Here: at seed 1 `shallow` is the best model on
every read, and `deep`'s lead over `shallow` and its "one sign of reach" did not repeat. I first wrote here that
seed 0's lead of `deep` "did not repeat": as the lead was defined before the run (`deep` minus `published` on all of
G) it did, at seed 1 and at seed 2. The primary on the held-aside set holds zero at every seed. The read over three
seeds is Part 2 of this note. The fault is in the read, which is mine, and not in the run; the figures themselves
stand as measured. See `ladder-l5-RESULT.md` and the spec's "L5n: what is one training worth?".

Labelled **pretrained on published proofs**, as everything built on that file is.

# Part 1: seed 0, as first read (2026-10-11)

Spec: `docs/spec/ladder-loop.spec.md`, "L5: do DEEP solves teach what ordinary solves do not?" with its
amendment (the primary moved to the held-aside set BEFORE any run) and "L5, further seeds". Run 2026-10-10 17:50 UTC to 2026-10-11 01:57 UTC (487 minutes) at Lean v4.27 through
the `oeis` pool, on the GPU box: tasks `rlvr_lean/ladder_l5_deep_smoke_r1` and
`rlvr_lean/ladder_l5_deep_seed0_r1`, from c0d8cbc25. One seed. All checks pass.

**Why this stage.** L4c and L4p said that proofs the model finds within 8 attempts sharpen it and narrow it, and
that published proofs beside them do the adding. The owner's bet (2026-10-09) is that the loop can push a model
"beyond the reach of things we even have pre-training data for". A proof found within 8 attempts is one the
model could already write. L5 asks whether a proof it could NOT write in 8 attempts, found by searching deeper,
teaches what an ordinary one does not, against the bar the owner set: the same number of published proofs.

**What was run.**
- **The deep search.** `pre_r64` made 56 more attempts (64 with its map's 8) at 1,000 of the 1,985 pool problems
  it had failed 8 of 8. It solved 520 (first success at attempt 9 to 16: 202; 17 to 32: 183; 33 to 64: 135).
  None of these problems' published proofs has been seen by any model here.
- **The held-aside set (HA).** 600 others of those 1,985, searched by nothing and trained on by no model.
- **Three models, each trained from `pre_r64` on 800 rows for 100 optimizer steps:** `published` (800 published
  proofs, the bar), `shallow` (400 ordinary solves, problems `pre_r64` solved in 1 to 3 of its 8 map attempts,
  and 400 published proofs), `deep` (400 deep solves and the SAME 400 published proofs).
- **Measured,** each the same way: the three rungs, the 392 goal problems (93 attempts each), and HA (16
  attempts each; `pre_r64` too).

## The read fixed before the run: the held-aside set

| 600 held-aside problems, 16 attempts each | `pre_r64` (start) | `published` (bar) | `shallow` | `deep` |
|---|---|---|---|---|
| Successes per 1,000 attempts | 36.2 | 50.8 | 48.9 | 51.5 |
| Problems solved | 187 | 214 | 204 | 217 |

- **Primary, `deep` minus `published`: +0.00063 [−0.00604, +0.00729]: NOT SEPARATED AT THIS SIZE.** Problems
  gained 74, lost 71. The read resolves 0.0067 per attempt (half its interval): a gap under about an eighth of
  `published`'s own rate could not have shown.
- **The mechanism, `deep` minus `shallow`: +0.00260 [−0.00365, +0.00885]: NOT SHOWN.** Problems gained 67, lost
  54.
- **The replication of L4c, `shallow` minus `published`: −0.00198 [−0.00792, +0.00407]: HOLDS ZERO.** Problems
  gained 60, lost 70.
- All three trained models are above `pre_r64` here (+0.013 to +0.015 per attempt, each interval clear of zero).
  That comparison is not fair to `pre_r64` (the set is its own failures); the three among themselves are.

## Beside it, naming no branch: the held-out goal set

| 392 goal problems | `pre_r64` (start) | `published` (bar) | `shallow` | `deep` |
|---|---|---|---|---|
| All of G, 93 attempts, per 1,000 | 80.0 | 103.1 | 104.5 | **112.5** |
| G′ (192 `pre_r64` fails first), 61 fresh attempts, per 1,000 | 8.4 | 17.7 | 14.3 | **21.2** |
| Goal problems solved in 93 | 245 | 262 | 257 | **275** |
| of 8 lines or more (74) | 25 | 25 | 22 | **36** |
| Solved reliably (half of their episodes of 8) | 107 | 146 | 136 | **151** |
| of 4 lines or more (230), reliably | 51 | 69 | 65 | **74** |
| Of the 81 never solved before the first pretrained arm | 7 | 10 | 8 | **16** |
| Share of its verified proofs with 8 lines or more | 25% | 30% | 25% | 31% |

- **`deep` minus `published` on all of G: +0.00941 [+0.00063, +0.01816].** On G′ +0.00350 [−0.00162, +0.00854].
  Goal problems gained 35, lost 22 (p = 0.11). By the length of the shortest published proof, per 1,000: 1 line
  137 against 130; 2 to 3 lines 136 against 127; 4 to 7 lines 123 against 107 (+0.0161 [+0.0032, +0.0292]); 8
  lines or more 38 against 43 (−0.0042 [−0.0169, +0.0070]).
- **`deep` minus `shallow`:** all of G +0.00795 [−0.00038, +0.01649]; G′ +0.00683 [+0.00307, +0.01093]; goal
  problems gained 39, lost 21 (p = 0.03); 4 to 7 lines +0.0177 [+0.0038, +0.0325]; 8 lines or more +0.0115
  [+0.0015, +0.0219].
- **`shallow` minus `published`:** all of G +0.0015 [−0.0075, +0.0103]; 8 lines or more −0.0157 [−0.0323,
  −0.0017] (27 against 43 per 1,000). Ordinary solves cost on the long problems again, as in L4c.
- **Against every model measured before** (all from `pre_r64`, one training each): `deep` has the most goal
  problems solved (275; the best published-only model, a whole second pass of 3,109 steps, 267), the most solved
  reliably (151; 139), the most per attempt (112.5; 108.9) and the most of the 81 never-solved (16; 13). It is
  the first model trained on its own proofs that is not behind its published-only control in breadth. Of the 81,
  two were solved for the first time by `deep` and one by `published`: 40 are now solved by something, 41 by
  nothing.
- **The easy rungs,** against `pre_r64` (below, in, above the band): `published` +0.041, +0.090, +0.046;
  `shallow` +0.051, +0.062, +0.049; `deep` +0.049, +0.041, +0.023. `deep` gains least on what the model already
  did.

## The rows

| The proofs trained on | deep rows (400) | shallow rows (400) | the published rows beside them (400) |
|---|---|---|---|
| Median lines | 9.5 | 6 | 7 |
| Share with 8 lines or more | 64% | 44% | 50% |

The training loss over the first and last tenth of the rows: `published` 0.142 to 0.136; `shallow` 0.148 to
0.168; `deep` 0.156 to 0.153. It does not order the models (the fourth time).

## Lean timeouts

A proof that times out in Lean is a failed attempt. The three models were measured one after another, and the
pool changed between them (15 workers until 23:49 UTC; 37 until 00:40, one of the added servers slow under other
load; 24 after).

| Share of attempts that timed out | `published` | `shallow` | `deep` |
|---|---|---|---|
| HA | 1.18% | 1.34% | 1.25% |
| Goal set, first 32 attempts | 2.49% | 3.13% | 4.69% |
| Goal set, the 61 more | 2.54% | 2.72% | 3.99% |
| Rungs | 0.36% | 0.63% | 0.98% |

On HA, the primary's set, the shares are level (`pre_r64` 1.37%): the primary is not touched. On the goal set
`deep` times out most, also in its second sampling, which ran after the slow server was paused: its proofs are
heavier, not only the pool slower. A timeout counts AGAINST `deep`, so the goal set's lead is not made by it.

**Checked again** (Lean only, after the run, with no measurement running): of the goal attempts that had timed
out, 18 of 934 (`published`, 1.9%), 14 of 1,058 (`shallow`, 1.3%) and 43 of 1,565 (`deep`, 2.7%) verify on a
second check; nine in ten time out again. The timeouts are heavy proofs, not a slow pool. With those counted as
successes: all of G per 1,000 103.6, 104.9, 113.6; `deep` minus `published` +0.0101 [+0.0012, +0.0192] (as
stored +0.0094 [+0.0006, +0.0182]); goal problems solved 262, 258, 275; long problems solved 25, 22, 36.
Every read is the same both ways.

## What it says, and what it does not

- **The pre-set read shows nothing.** On hard problems of the kind the search trained on, 400 deep solves and
  400 published proofs are level. By the rule this stage is NOT SEPARATED AT THIS SIZE, and no figure from the
  goal set changes that name.
- **On the goal set deep solves are the first rows of the model's own that do not narrow it.** Ordinary solves
  did (L4c: 229 goal problems against the control's 258; here `shallow` 257 against 262 and behind on the long
  problems). Deep solves did not (275 against 262). This is what the owner's bet needs and it is the first sign
  of it.
- **Why it is a lead and not a result.** (1) The primary was moved to HA before the run, for its power, and it
  holds zero. (2) Each model is one training: two trainings of one kind differ by about 0.006 per attempt on all
  of G and 5 goal problems (one standard deviation, L4p), and the lead is +0.009 and 13 problems. (3) The
  interval on all of G is over problems, not over trainings. (4) The goal set's reads were fixed before the run
  as "beside, naming no branch".
- **Depth is not yet apart from length.** The deep rows are longer proofs than the shallow ones (64% against 44%
  of 8 lines or more). `deep`'s edge over `shallow` is on the problems of 4 lines or more. "A proof the model
  could not sample" and "a longer proof" are one variable in this run. If the lead is confirmed, the next check
  is ordinary solves matched in length.
- **HA and the goal set do not disagree; the lead per attempt is reliability** (the stored attempts looked at
  again after the read; exploratory). 84% of `deep`'s per-attempt lead over `published` on all of G sits on the
  200 goal problems `pre_r64` already solves in its first 32 attempts (200.9 against 185.4 per 1,000, +0.0155
  [−0.0007, +0.0320]). On G′, the 192 it fails, over all 93 attempts the difference is +0.0031 [−0.0020,
  +0.0083]: level, as on HA. On problems the start model cannot solve, deep solves and published proofs teach
  the same amount per attempt.
- **The one sign of reach is a count on the long problems.** Goal problems of 8 lines or more solved at least
  once: `deep` 36, `published` 25 (gained 14, lost 3, p = 0.013), `shallow` 22 (gained 17, lost 3). Per attempt
  on those 74 `deep` is not ahead (38 against 43 per 1,000): it reaches more long problems, each of them rarely.
  One slice of several looked at, one training each. The spec was amended before seeds 1 and 2 run so that this
  read is fixed in advance for them.

## What follows

By the standing seed rule (a scout whose primary is not negative goes to three seeds): seeds 1 and 2, fixed in
the spec before they run. The same search and held-aside set; another draw of the 400 deep solves from the 520,
of the ordinary solves, of the published rows and of the order. The primary is pooled over three seeds on HA.
The goal set's lead is CONFIRMED only by seeds 1 and 2 alone (seed 0 found it, so it cannot confirm it), and it
is then named RELIABILITY or REACH TOO by its split between the 200 and G′. Reach on the long problems is read
on the two new seeds alone as well. About five hours a seed.

After the read over three seeds, on every branch but THEY COST: L6, a second round (the model trained on round
1's deep solves searches again; two rounds against the published-only bar at equal steps, with ordinary solves
matched in length beside), fixed in the spec before that read.

# Part 2: the read over three seeds (2026-10-11)

**What ran.** Seeds 1 and 2, as fixed in the spec's "L5, further seeds" before they ran: the same search, the same
held-aside set and the same stored measurement of `pre_r64` on it; by each seed another draw of the 400 deep solves
(of the 520), of the ordinary solves, of the published rows and of the order. Tasks
`rlvr_lean/ladder_l5_deep_seed1_r1` (03:44 to 10:53 UTC) and `rlvr_lean/ladder_l5_deep_seed2_r1` (11:02 to 16:35 UTC),
both from cd8372bfc. All checks pass at every seed. Read with `python -m rlvr_lean.tools.ladder_l5_seeds` on the three
pulled task directories.

| | `pre_r64` (start) | `published` (bar) | `shallow` | `deep` |
|---|---|---|---|---|
| **All of G, per 1,000 attempts:** seeds 0, 1, 2 | 80.0 | 103.1, 79.5, 78.5 | 104.5, 110.0, 95.0 | 112.5, 97.2, 101.5 |
| mean (standard deviation over the seeds) | | 87.0 (13.9) | 103.2 (7.6) | 103.7 (7.9) |
| **Goal problems solved in 93:** seeds 0, 1, 2 | 245 | 262, 217, 229 | 257, 261, 249 | 275, 250, 262 |
| mean (standard deviation) | | 236.0 (23.3) | 255.7 (6.1) | 262.3 (12.5) |
| **G′, the 61 fresh attempts, per 1,000** | 8.4 | 17.7, 9.9, 12.2 | 14.3, 18.0, 13.4 | 21.2, 14.2, 14.9 |
| **Long problems (8+ lines, 74) solved** | 25 | 25, 19, 21 | 22, 33, 22 | 36, 24, 24 |
| **Held-aside set, per 1,000 attempts** | 36.2 | 50.8, 42.7, 47.1 | 48.9, 53.9, 44.0 | 51.5, 43.4, 50.1 |
| mean (standard deviation) | | 46.9 (4.1) | 48.9 (4.9) | 48.3 (4.3) |

## The reads fixed before the seeds ran

- **The primary, over the three seeds, on the held-aside set: NOT SHOWN AT THREE SEEDS.** `deep` minus `published`,
  each problem's mean over the seeds: +0.00146 [−0.00292, +0.00573] (48.3 against 46.9 per 1,000); at the seeds
  +0.0006, +0.0007, +0.0030.
- **The goal set's lead, on seeds 1 and 2 alone: THE LEAD IS CONFIRMED.** `deep` minus `published` on all of G:
  +0.02034 [+0.01436, +0.02658] (99.4 against 79.0 per 1,000), +0.0177 at seed 1 and +0.0230 at seed 2. Goal problems
  gained and lost: 52 and 19, 48 and 15.
- **Its split: REACH TOO.** On the 200 goal problems `pre_r64` solves in its first 32 attempts +0.0366 [+0.0255,
  +0.0477] (181.4 against 144.8 per 1,000); on G′ over the 93 attempts +0.00339 [+0.00036, +0.00661].
- **Reach on the long problems: NOT CONFIRMED.** The share of the two new seeds in which a long problem is solved,
  `deep` minus `published`: +0.054 [−0.007, +0.122]; `deep` solves more of them at both (24 against 19, 24 against 21).
- **The branch: "otherwise"** (a lead confirmed as reach too, with reach on the long problems not confirmed, has no
  branch of its own in the section): the depth or the dose is the thing to raise.
- **Beside, naming nothing.** `shallow` minus `published` on the new seeds +0.0235 [+0.0175, +0.0297]. `deep` minus
  `shallow` on the new seeds −0.0032 [−0.0089, +0.0026] and over the three seeds +0.0006 [−0.0049, +0.0061].

## The same figures read over trainings

The rule above was fixed before seed 1 showed what one training is worth, and its intervals are over problems: the
fault this note's correction names. Read with each seed's model as ONE draw of its kind (paired by seed: the seeds
share the published rows beside and the order's seed), three differences and a t test on two degrees of freedom:

| `deep` minus `published` | seed 0 | seed 1 | seed 2 | mean | p |
|---|---|---|---|---|---|
| All of G, per 1,000 | +9.4 | +17.7 | +23.0 | +16.7 | 0.052 |
| the 200 the start model reaches, per 1,000 | +15.5 | +31.0 | +42.2 | +29.6 | 0.062 |
| G′ over the 93 attempts, per 1,000 | +3.1 | +3.9 | +2.9 | +3.3 | 0.008 |
| Goal problems solved | +13 | +33 | +33 | +26.3 | 0.059 |
| Long problems solved | +11 | +5 | +3 | +6.3 | 0.12 |
| Held-aside set, per 1,000 | +0.6 | +0.7 | +3.0 | +1.5 | 0.20 |

- `shallow` minus `published` on all of G: +1.5, +30.5, +16.5 (mean +16.1, p = 0.19); goal problems solved −5, +44,
  +20. The same size as `deep`'s, with twice the scatter.
- `deep` minus `shallow` on all of G: +8.0, −12.8, +6.5 (mean +0.5, p = 0.94). On the held-aside set −0.6.
- Seven measures and three pairs were looked at; one p of 0.008 among them is not by itself a finding, though G′
  for `deep` minus `published` is the read the split had fixed.

## What it says, and what it does not

- **The question the stage asked has its answer: depth is NOT SHOWN to matter.** Over three seeds a model trained on
  hard-won solves and one trained on solves found within 8 attempts are level on every set (103.7 and 103.2 per
  1,000 on all of G; 48.3 and 48.9 on the held-aside set). Seed 0's lead of `deep` over `shallow` was one draw.
- **What the seeds do show, and the stage did not ask: the model's own proofs beside published ones beat published
  ones alone on the goal set, at three seeds of three.** +16.7 per 1,000 for `deep` (about a fifth more), more goal
  problems solved at every seed, and on the start model's own failures too. By the rule fixed in advance this is
  CONFIRMED; over trainings it stands at p = 0.05, which is the edge of what three seeds resolve. It is the first
  replicated sign in the direction of the owner's bet, and it is not yet beyond doubt.
- **It is not there on the held-aside hard problems,** where the three kinds are level at three seeds, and where the
  primary was put. Why the goal set and not that set is not explained (the goal problems are all of one source,
  the held-aside ones mostly of another).
- **The model's own proofs did not narrow it here.** Goal problems solved: `deep` 275, 250, 262 and `shallow` 257,
  261, 249, against the start model's 245; `published` 262, 217, 229. The narrowing earlier notes blamed on the
  model's own proofs is, in this stage, on the side of the published rows alone.
- **The bar is the noisy arm.** `published` reads 103.1, then 79.5 and 78.5: at two seeds of three it gains nothing
  over the start model. The earlier published-only models from `pre_r64` (five of them, L4p) read 86.8 to 108.9. So
  whether 79 or 103 is the bar's usual level is open, and L5n's four `constant` models (four more trainings on
  seed 0's own 800 published rows, in other orders) say which.
- **What the bar is.** Its 800 rows are a second pass over proofs `pre_r64` has trained on; the own rows are
  problems it has never trained on. So the finding, if it holds, is that new solves of the model's own teach more
  than a second pass over old published proofs: the comparison that matters when the published proofs have run
  out, and not a comparison with fresh published proofs, which was not run.
- **A statement of mine corrected.** On 2026-10-11, after seed 1, I wrote that seed 0's lead "did not repeat". As
  the lead was defined before the run (`deep` minus `published` on all of G) it was +0.0177 at seed 1 and it did
  repeat; what did not repeat was `deep` being the best of the three models, and its lead on the long problems.
- **Lean timeouts.** Shares of goal attempts that timed out, `published`, `shallow`, `deep`: seed 0 2.5%, 2.9%,
  4.2%; seed 1 3.0%, 4.9%, 3.3%; seed 2 1.6%, 3.6%, 3.6%. At every seed the bar has the fewest, and a timeout is a
  failed attempt, so the timeouts do not make the lead. Seed 1 and most of seed 2 were measured while the pool's
  main server shared its machine with other work (slow checks; two ten-minute pauses fell inside seed 1's `shallow`).
  Their timed-out attempts are NOT yet checked again: no window without a running measurement has opened. Seed 0's
  were, and changed no read.

## What the models write (seed 0's stored goal attempts, read after the result; no run)

- **One style.** All four models call `nlinarith` in 77 to 79% of their goal attempts, and 91 to 98% of every model's
  timeouts are attempts with it. Among those attempts the share that times out: `pre_r64` 5.3%, `published` 3.0%,
  `shallow` 3.6%, `deep` 5.2%. `deep` keeps the start model's heavy calls (its longest hint list has 6 terms or more
  in 50% of them, `published` 45%); training on published proofs alone makes them lighter.
- **The long problems are solved by long proofs.** On the 74 goal problems of 8 lines or more every model's verified
  proofs have a median of 10 to 11 lines and four in five have 8 or more: `deep`'s reach there is not one-line
  automation.
- **The reach is made of single hits.** Of the 14 long problems `deep` solves and `published` does not, 11 are solved
  once in 93 attempts, two twice, one three times. `pre_r64` had solved 6 of the 14. "Solved at least once" is a
  noisy count at these rates, which is why the read for seeds 1 and 2 asks for both seeds.

## Looked at after the three seeds (2026-10-11; reads of the stored attempts and rows, no run, nothing fitted)

Why `published` is at the start model's level at seeds 1 and 2 and 23 per 1,000 above it at seed 0 was looked for
on the surface. Nothing found there explains it.

- **The first sampling alone carries the figures.** On G's first 32 attempts, successes per 1,000 at seeds 0, 1, 2:
  `published` 103.8, 78.6, 76.5; `shallow` 105.0, 107.6, 95.1; `deep` 110.2, 98.1, 100.8: within about 2 of the 93
  attempts'. The mean of `shallow` and `deep`, minus `published`: +3.8, +24.2, +21.5 (mean +16.5, standard
  deviation 11.1, t 2.58 on 2 degrees, p 0.12). `deep` minus `shallow`: +5.2, −9.6, +5.7. This is what L5d measures
  on fresh seeds.
- **The weak models are not broken.** What each of the nine models and `pre_r64` wrote on those 32 attempts: the
  tokens written (means 103 to 131), the lines (4.7 to 5.6), the share that ran to the token limit (2 to 4 per
  1,000), the first tactic (`nlinarith` 19% to 28%, `have` 20% to 27%) and Lean's first error (`linarith failed` on
  55% to 65% of attempts) are alike across all ten. `published` at seeds 1 and 2 writes a little less (109 and 103
  tokens against 125 at seed 0) and at seed 2 with less variety (0.830 of a problem's attempts distinct against
  0.89 to 0.93 elsewhere). No tactic, error or length marks them.
- **They are lower on each band that holds enough problems to read.** By the start model's own successes of 32 (0:
  192 problems; 1 to 3: 109; 4 to 15: 82), successes per 1,000 stored attempts: `published` 15.9, 99.7, 275.5 at
  seed 0; 8.2, 61.9, 210.4 at seed 1; 10.1, 68.2, 202.0 at seed 2; the six models with the model's own proofs 11.2
  to 18.2, 86.1 to 105.4, 250.8 to 291.2. The 9 problems the start model solves 16 times or more are too few to
  read (`published` 441, 569, 434; the six 479 to 569). (The bands are chosen by the start model's own sampling, so
  every model is compared with the others in a band and not with the start model. 7 of the 392 problems are tried
  on both sides, so a model stores 12,768 attempts for the 12,544 the rate above counts.)
- **The rows look alike.** Of each seed's 800 published rows: 254, 266 and 267 from the workbook set; 9.2, 8.8 and
  8.9 lines on average; 399, 367 and 388 of 8 lines or more. Each draw is the whole file in small: by the prover the proof came from (73% to 74% of one
  source in each, 73% in the file), by first tactic (`have` 40%, 40%, 43%; 41% in the file) and by the share of
  one-line proofs (9.6%, 9.1%, 7.8%; 8.5% in the file). Two seeds' draws share 23 to 26 rows. No training's
  loss jumps (the mean over each fifth of the pass is 0.12 to 0.18 in all nine trainings, 0.12 to 0.16 in the three
  `published` ones).
- **Timeouts, seed 2, checked again in Lean with no measurement running** (two of its three models so far): of
  `published`'s 676 timed-out attempts 13 verify (12 on G), of `deep`'s 1,510, 44 (38 on G). On all of G that moves
  `deep` minus `published` at seed 2 by under 1 per 1,000, towards `deep`. `shallow` of seed 2 and seed 1 are still
  to be checked.

- **Seed 0's rows are not nearer the goal problems.** For each goal problem, the nearest statement among a seed's
  800 rows (the leak check's measure: the share of 5-character pieces two statements have in common): mean 0.337,
  0.340, 0.347 at seeds 0, 1, 2; goal problems with a row at 0.4 or over: 145, 150, 150. And `published` of seed 0
  gains over the start model on the 168 goal problems with no row over 0.3 (+21 per 1,000) as on the 145 with a
  row at 0.4 or over (+30 to +37), while at seeds 1 and 2 it gains in no band (seed 2: −21 on the 68 problems with
  a row at 0.5 or over).
- **The models that succeed more write longer attempts, and the length is a marker more than the cause.** Across
  the ten models the mean lines written and the successes per 1,000 correlate at +0.85. Inside every model a
  verified attempt is 0.7 to 1.5 lines longer than a failed one on the same problem; one-line attempts verify at 34
  to 81 per 1,000 and attempts of 4 lines or more at 86 to 136. The three models at the start model's level write
  the most one-line attempts (15% to 18% of attempts at the statement, against 9% to 13%). But the mix explains
  little: the start model with seed 0 `published`'s mix of lengths and its own rate inside each band of length would
  have 82.7 per 1,000 against its 78.3, where that model has 104.3. Inside every band the stronger models verify
  more.

So what is left is the rows themselves: which published proofs a draw holds, by something other than their source,
their length or how near they are to the goal problems. A second pass over 800 rows the start model was already
trained on is expected to add little (L4t2's `rehearse_only`: 3,422 such rows added 7 per 1,000 on all of G),
which is what seeds 1 and 2 show; seed 0's +23 on all of G is the figure to explain. L5n's first read (the spread from the
order alone, on seed 0's rows) says how much of the spread that is, and L5d's `published` at each fresh seed is a
further draw.

## What follows

L5n (the spread between trainings on fixed rows, and whether a decaying learning rate or averaged adapters lower
it) runs now; its four `constant` models are also four more draws of the order on seed 0's rows. After it: L5d,
the three kinds again at fresh seeds with the recipe L5n names, measured on the goal set's first sampling (built,
not run; three seeds, six with the owner's approval, asked 2026-10-11). The cycle at full size that the owner
proposed on 2026-10-11 (every problem of the unseen half searched, training on what is found, a second round) is
specified for review in `ladder-cycle.spec.md` and is not built: it waits for the owner's go.
