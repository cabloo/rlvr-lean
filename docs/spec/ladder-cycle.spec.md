# Feature: the deep-search cycle at full size (ladder loop L7)

> **Spec-driven.** This file is the source of truth for behavior. Implement STRICTLY to it — no
> behavior that isn't specified. If anything here is ambiguous or underspecified, STOP and record
> it under **Open questions** rather than guessing. Iterate by editing this spec, then implement
> the diff. If code and spec disagree, the spec wins (or we change the spec).

- **Owning module:** `rlvr_lean`
- **Module path:** `src/rlvr_lean/` (stage code under `gpu/`, rules under `domain/ladder_round/`, reports under `reporting/`)
- **Status:** draft (PROPOSED 2026-10-11; NOT APPROVED: it needs the owner's go; nothing of it is built)
- **Spec file:** `docs/spec/ladder-cycle.spec.md`
- **Continues:** `docs/spec/ladder-loop.spec.md`, whose words are used as defined there: `pre_r64` (the start
  model, pretrained on published proofs, rank 64), G (the 392 goal problems), the base map (4,000 problems), the
  `pretrain` and `loop` halves of the pool, the pretraining file, L5 (the deep search of 1,000 problems and its three
  kinds), L5n (the noise stage), L5d (the three kinds again at fresh seeds), L6 (a second round at L5's size).

## Purpose

Round 1 of the cycle the owner described, at the size where it can show a difference: the start model searches
every problem of the half of the pool it never trained on, three kinds of model are trained on what it finds, and
the result is what a model SOLVES WITH SEARCH on hard problems it never trained on. Round 2, the test of whether
the cycle compounds, is named here and specified after round 1's read.

## L7: round 1 at full size. The whole unseen half searched, and the three kinds at seventeen times the dose

**Under the owner's words.** 2026-10-09: "this loop could be used to push it beyond the reach of things we even have
pre-training data for ... I just want to make sure that this is actually creating a feedback cycle". 2026-10-11: "so
how do we get to the point where the model is actually solving significantly more problems than it could previously?
... select one intelligently that's within reach but difficult for it, and then give the solver enough attempts to
actually solve it, train ... on the resulting proof". Labelled **pretrained on published proofs** (it starts from
`pre_r64`).

**Why at this size.** What is measured so far, each with what it is worth:
- **Enough attempts is the strongest lever measured.** Of 1,000 problems `pre_r64` failed 8 of 8, it solved 520
  within 56 more attempts (first success at attempt 9 to 16: 202; 17 to 32: 183; 33 to 64: 135).
- **The dose tested is 2% of the model's training.** L5 and L5d train on 400 of the model's own proofs, beside
  24,866 published ones it was built from. On the goal set the kinds with its own proofs are above the published-only
  kind at 3 seeds of 3 (L5; L5d is its confirmation); on the 600 held-aside hard problems nothing is shown (`deep`
  minus `published` +0.7, +0.7, +3.0 per 1,000 attempts at the three seeds).
- **The loop half holds about seventeen times as many hard-won proofs.** 26,029 candidates no model here has trained
  on. By L5's yields (1,985 of 4,000 failed 8 of 8; 520 of 1,000 of those solved): about 12,900 fail 8 of 8 and about
  6,700 of them are solved within 64 attempts. An extrapolation from one search of 1,000 problems.
- **Who orders the search is decided.** The challenger test (2026-10-11, another session; its note,
  `challenger-test-RESULT.md`, is on another branch and not in this copy): ranking the 1,000 searched problems and the
  600 held aside by whether the search solved them, today's challenger has AUC 0.706 and 0.738, the surface of the
  failed attempts 0.635 and 0.635, a head on the solver's states 0.626 and 0.628, the model asked in text 0.584 and
  0.558. Today's challenger orders the search here.

**What runs.** One stage, `pre_r64` the solver throughout, the round's episode in every respect not named (the prompt,
the limits, one side by the audit rule, Lean's verdict, the soundness alarm).
1. **The map.** 8 attempts at each of the loop half's 26,029 candidates (about 208,000 attempts), as `pre_r64`'s own
   map was made on the base map's 4,000. A problem failed 8 of 8 is a CANDIDATE OF THE SEARCH; a problem solved in 1
   to 3 of its 8 is an ORDINARY solve.
2. **The order.** Today's challenger, fitted once on the map's outcomes (each problem's successes of 8), scores every
   candidate of the search. It is the scorer the challenger test named "today's challenger", and before it is used
   it is held to give that test's AUC again on the stored search of 1,000 (0.706): a scorer that does not is not the
   one that was chosen. It is not fitted again during the search.
3. **The deep search, in three passes with a stop at the first verified proof.** Pass one: 8 more attempts at every
   candidate. Pass two: 16 more at each one still unsolved. Pass three: 32 more at each one still unsolved. 64
   attempts at most, the map's 8 among them. Within a pass the problems go in the challenger's order, the highest
   score first, so that a run cut short has searched the likeliest first and every candidate has had its sixteen
   before any has its sixty-four. By L5's yields: about 40 attempts a candidate, about 520,000 in all. A problem
   solved here is a HARD-WON solve; its training row is one verified attempt of the pass that solved it, drawn by the
   seed.
4. **The models: three kinds, each trained at THREE seeds from `pre_r64` at its rank, one pass, the recipe L5n
   names.** D is the number of hard-won solves, all of them (no cap).
   - `hard_won`: the D hard-won rows and the first D rows of the seed's hash order of the pretraining file;
   - `easy`: D rows of ordinary solves (a seeded draw, one verified map attempt each) and the same D published rows;
     when the map has fewer than D ordinary solves, the problems solved in 4 of 8 are added, then 5, and so on, and
     it is said;
   - `published`: the first 2D rows of that order and nothing else. The bar.
   A seed draws the published rows, which ordinary solves, which verified attempt of each solve, and the order. All
   D hard-won problems are in `hard_won` at every seed.
5. **The measurements, each model the same.** (a) The goal set's first sampling: 32 attempts a problem with the
   sampling seed `pre_r64`'s run recorded (L5n's and L5d's measurement). (b) The 600 held-aside hard problems of L5
   (base-map problems `pre_r64` failed 8 of 8; no model here trains on them or has seen their published proofs), in
   three passes with a stop at the first verified proof: 16 attempts, 16 more at each one unsolved, 32 more at each
   one still unsolved. The first pass is L5's own measurement of that set (its 16 attempts, its sampling seed), so
   that every stored model pairs with it by problem; the second and the third have a sampling seed each, shared by
   every model. `pre_r64` itself is measured on (b) once, with its 16 stored attempts as its first pass.

**The reads, fixed before the run.** Every interval is over SEEDS (trainings): the mean of the per-seed differences
± t × their standard deviation over the root of three (t 4.303).
- **Checks first** (any FAIL: INCONCLUSIVE, nothing else said). The search found at least 2,000 hard-won solves.
  Each training ran; each adapter is not the start adapter's and is read back at its rank. Lean answered (at most
  0.02 of a set, of the map and of each pass of the search without an answer). No barred row: no held-out problem,
  no held-aside problem, no base-map problem, published rows of the `pretrain` half alone, no row with text.
- **Primary: reach with search.** The number of the 600 held-aside problems a model solves within its 64 attempts,
  `hard_won` minus `published`, at each seed. Interval above zero: THE CYCLE'S PROOFS ADD REACH THAT THE SAME NUMBER
  OF PUBLISHED PROOFS DOES NOT. Below zero: THEY COST. Holds zero: UNDETECTABLE AT THREE SEEDS when its half-width
  is above 30 problems, with the interval; otherwise NOT SHOWN. Thirty is 5 of 100 held-aside problems: the smallest
  difference between two kinds that would be called a gain in reach here, fixed now and not from a result.
- **Against the start model, beside the primary on every branch:** each kind's count minus `pre_r64`'s own within
  64 attempts. This is "solving more problems than it could before", and it is said for each kind, the bar too.
- **The goal set:** the mean of `hard_won` and `easy`, minus `published`, successes per attempt on the first
  sampling: L5d's primary at this dose, the same names.
- **Depth:** `hard_won` minus `easy`, on the primary's count and on the goal set. DEPTH MATTERS when an interval is
  above zero; NOT SHOWN otherwise.
- **Per attempt on the held-aside set:** the first pass (16 attempts, where no stop has acted), each pair.
- **Overfitting, for each model against `pre_r64`:** goal problems lost and gained in the 32 attempts; the share of
  distinct attempts; the long goal problems (8 lines or more) solved.
- **The search itself:** the candidates, the solved by pass, the attempts spent; the challenger's AUC on this
  search's outcomes (its first test out of the sample it was chosen on), and the yield of its top quarter.
- **Timeouts:** each set's share; where two models of one seed differ by more than a point on a set, that set's
  timed-out attempts are checked again (Lean only, no measurement running) and the reads are given both ways.

**What follows, fixed now.**
- THE CYCLE'S PROOFS ADD REACH: round 2 (below) runs. This is the first half of the gate for the larger model.
- UNDETECTABLE or NOT SHOWN, with every kind above `pre_r64`: the cycle's proofs are worth about what published
  proofs are. Round 2 still runs: whether the gain compounds does not wait on this. Said with it: the bar is a second
  pass over proofs the model has seen.
- No kind above `pre_r64` on the primary's count: the ceiling at this adapter's size. That is the evidence on the
  owner's question about the model's size, and it goes to the owner before round 2.
- THEY COST: stop, and it goes to the owner.

**Round 2, named here and specified after round 1's read.** `hard_won` of seed 0 (fixed now, whatever its figure) is
M(1). It searches what round 1 left unsolved, the same three passes. A new model is trained from `pre_r64` on both
rounds' hard-won rows, with a published-only bar at equal steps, at three seeds, and measured the same way. Round 2
against round 1 on the primary's count is the test of whether the cycle compounds.

**What it cannot say.**
- **The bar is a second pass.** As in L5d: `published` reads again proofs `pre_r64` was trained on, and the model's
  own proofs are of problems it never trained on. The kind that would separate the two (each hard-won problem's
  PUBLISHED proof in place of the model's own) trains on published proofs of the loop half: not built without the
  owner's word, asked 2026-10-11. With the word it is a fourth kind here, `published_new`, at the same three seeds.
- **The loop half is used up.** After this run its problems have been trained on; fresh problems would come from
  the published statements our Lean has not yet checked.
- One start model, one adapter size, one pass. A search of 64 attempts.
- Near variants of a problem can fall on both sides of every split here, as everywhere in this arm.

**Left out on purpose.** The challenger fitted again between passes; a training step that learns from failed
attempts; the accumulating episode in the search (L3c's, which reached longer proofs); the challenger writing
problems (the owner's rule keeps problems to those with a published answer).

**Budget, on the one GPU with 15 pool workers.** The map about 7 hours (L5's map: 32,464 attempts an hour). The
search about 22 hours (about 400 attempts a minute). Nine trainings of 2D rows (13,400 at the expected yield: about
70 minutes each at 800 rows in 4 minutes): 10 hours. Nine measurements of about two hours (25 minutes for the goal
set; the held-aside passes at most 38,400 attempts, fewer with the stop) and `pre_r64`'s own passes: 19 hours. About
two days and a half. The fourth kind, with the owner's word: 10 hours more. The map, the search and each seed are
tasks of their own, queued one after another, each resumable.

**Before it runs.** The owner's go. L5n's read (the recipe). A smoke stage on the fixtures. L5d does not gate it:
this run holds L5d's comparison at seventeen times the dose.

## Input contract, output contract, public API, dependencies

Module-internal only. It reads stored runs of this arm (`pre_r64`'s run, its map, L5's finished search and its
held-aside set, L5n's report) and the snapshot's data files; it writes run directories of its own and nothing else.
The stage's exact files, steps and refusals are fixed by its build, in a "Made exact by the build" block here.

## Fixtures

The smoke stage's fixtures are the arm's own (`data/ladder_l4_fixture`, the smoke runs of L5). The tests are fixed
by the build.

## Open questions

1. **The owner's go** on round 1 at this size (about two days and a half of the one GPU), or another size.
2. **The fourth kind, `published_new`** (each hard-won problem's published proof in place of the model's own): it
   trains on published proofs of the loop half, so it needs the owner's word. Asked 2026-10-11.
3. **The scorer.** "Today's challenger" as the challenger test scored it lives on another branch
   (not in this copy). The build either brings that scorer over or shows this branch's challenger gives the
   same ranking; it is held to the test's AUC on the stored search either way.
