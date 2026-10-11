# Ladder loop, L5 (do DEEP solves teach what ordinary solves do not? seed 0): NOT SEPARATED AT THIS SIZE on the primary, and a lead on the goal set. On 600 held-aside hard problems a model trained on 400 of its own deep solves is level with one trained on the same number of published proofs (51.5 against 50.8 successes per 1,000, +0.0006 [−0.0060, +0.0073]). On the held-out goal set it is the best model measured so far (112.5 per 1,000 against 103.1; 275 goal problems solved against 262; 151 reliably; 16 of the 81 never solved before), mostly by solving more often what the start model already reached; the one sign of reach is the long problems solved at least once, 36 against 25. One training each: a lead, not a result. Two further seeds decide it.

Labelled **pretrained on published proofs**, as everything built on that file is.

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
