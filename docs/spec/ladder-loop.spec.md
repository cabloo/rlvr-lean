# Feature: the ladder loop — a challenger and a solver that take the model past what it could do

- **Status:** approved 2026-10-04, no open question. Revised that day after the owner's corrections and the
  change of the reward target; the owner's answers to the five questions are the approval.
  <!-- draft → approved → built -->
- **Code:** `src/rlvr_lean/` (new stages; nothing built yet)
- **Builds on:** the first experiment's spec (Phase A, B, 13a, 13b), the OEIS Open spec (O2a), the
  results `phase-b-selection-RESULT.md`, `native-format-RESULT.md`, `ladder-first-rung-RESULT.md`,
  `o2a-version-tax-RESULT.md`, and the diagnosis ledger of the held-out loss and the learning-progress score.

## Goal

The solver resolves problems the base model cannot. **"Beyond" is measured on one fixed, held-out set of
problems whose answer is already published and which the base model does not resolve under a fixed budget.**
Pass@1 is reported and is not the goal: Phase A and B showed it moves by making already-solved problems more
reliable.

## The owner's rules (2026-10-04)

1. **Either side counts.** A problem is resolved by a proof of the statement or a proof of its negation, and
   the solver is rewarded for either.
2. **Train only on problems whose answer is known**, decided before training, from published data and not by
   our own model. Statements nobody has settled are a later, novel stage.
3. **No difficulty screen.** Difficulty falls out of the challenger's reward for a problem's pass rate.
4. **The pass rate is per problem:** k of the n solvers that attempted ONE problem resolved it. **The reward
   peaks below one half:** k = 1 to 3 of 8 should earn more, possibly more than k = 4 (the owner, after the
   first revision, which had the peak at n/2).
5. **A ladder that moves:** when a problem at the target rate becomes easier than it, the challenger looks
   further up.
6. **Agent-like acts with caps:** Lean's error fed back for a repair attempt, targets broken into pieces; about
   5 loops and about 5 minutes per episode.
7. **A control with equal compute** is wanted and is not the top goal.
8. **The loop must not depend on the model:** a much larger prover may be swapped in later.

## What we have today, measured 2026-10-04

**Rule 2 did not hold.** Every conjecture in Phase A, Phase B, 13a and 13b was written by our model and had no
outside answer; we trained on whatever our own solver proved. The Lean Workbook statements we seed and evaluate
on mostly have no published answer either.

| Set | Size | Published proved | Published disproved | No published answer |
|---|---|---|---|---|
| Lean Workbook, all | 140,214 | 37,325 (26.6%) | 3,083 (2.2%) | 100,012 |
| Our seed statements | 2,000 | 615 | 36 | 1,355 |
| Our reward statements, gradient half | 987 | 294 | 18 | 676 |
| Workbook holdout | 1,013 | 301 | 21 | 693 |
| of which the base proved (8 + 64 samples) | 135 | 135 | 0 | 0 |
| of which unsolved by the base (the 878) | 878 | 166 | 21 | 693 |

A few problems are published both ways, so the two middle columns overlap (206 in all, 2 in the holdout).
Sources, all matched to our statements by name with identical text (0 mismatches): the Lean Workbook file's own
`proof` field (12,823 proved); InternLM's `wkbk_1009.parquet` (10,434 proved, 3,083 disproved);
`Goedel-LM/Lean-workbook-proofs` (29,750 proved); `kfdong/STP_Lean_0320` (25,415 Lean Workbook proved, plus
**703,996 distinct conjectures written by STP's model, each with a published proof**, and 79,828 Mathlib
statements).

**A published label is not a certificate.** 206 Lean Workbook problems are published as both proved and
disproved. The cause, read from the rows: InternLM's "disproved" keeps the hypotheses and negates only the
conclusion. When the hypotheses contradict each other the statement is provable both ways (it says nothing).
Of the other 2,877 published disproofs, 1,876 have no hypotheses (1,775 have no variables at all), so the
published disproof is a proof of the exact negation; 958 have hypotheses, and for those the exact negation
follows only once the hypotheses are shown able to hold; 43 our parser did not split. The split is a heuristic
(a binder is a hypothesis when its type holds a relation or a logic symbol); two independent readings agree on
all 2,834. Every published disproof is one chain of tactics ending in "no goals", so it can be written back as
proof text and checked.

Our own negation builder (`domain/conjecturing/negation.py`, the 13b census) already builds the exact negation,
`¬ (∀ binders, type)`, and checks with Lean that it is exact.

## What the evidence so far constrains

| Finding | Source | Consequence here |
|---|---|---|
| At least 28% of our own conjectures were false, 65% were never proved, and none had an outside answer | 13b census | The solver works only on problems with a published, re-checked answer |
| The prompt does not steer difficulty | ledger section 9 | Difficulty is aimed by a chooser trained on the pass-rate reward |
| Training on mid and easy successes raised pass@1 and left pass@32 flat: solved problems got more reliable, reach did not move | Phase A and B, three seeds | The reward's peak sits below one half |
| On 9 untrained hard conjectures (108 samples each; base 14 stored, 18 fresh) only the adapters trained on hard conjectures beat the base: 29 and 35. Trained on the 50% band: 21, 19, 14 and 10; random: 15. One conjecture carries much of it (without it 17 and 23 against 13 and 17) | 13b diagnosis | A lead, not a finding. L1 measures the gain by k |
| The arm trained on hard conjectures was the lowest on overall pass@1, by about 0.002 to 0.004 | Phase B, three seeds | Pass@1 is reported, and is not what the target is set by |
| The learning-progress score ranked a formatting token | diagnosis | The challenger learns from measured k of n and nothing else |
| Three passes over 137 examples overfit | 13a, three seeds | One pass |
| Targets must be in the model's native format | diagnosis, 13a | Native format everywhere |
| Each training target was the shortest proof, and trained models wrote shorter attempts (46 → 34 tokens) | 13b diagnosis | The target is a randomly chosen verified proof |
| One pass on the 50% band did not lift harder, untrained conjectures: −0.034 [−0.063, −0.004] | 13b, one seed, 46 conjectures | "Learning builds upward" is still unproven; it is measured every round |
| Lean v4.27 verifies 98.8% of what v4.9 does on identical proofs | O2a | The loop runs at v4.27 through the pool |

## Behavior

### A problem, and the pool

A **problem** is a statement with a **certificate**: a published proof of the statement (known true) or of its
exact negation (known false) that Lean accepts **under our pin**. Without a re-checked certificate a statement
is not a problem and never reaches the solver.

- **The negation is the exact one:** `¬ (∀ binders, type)`. Exactly one side of a statement is true, and it is
  the form a counterexample to an OEIS conjecture takes.
- **The pool** is: Lean Workbook problems published proved and not also published disproved (about 37,100);
  Lean Workbook problems published disproved with no hypotheses (about 1,900); STP's 703,996 proved
  conjectures. Nothing our model wrote is in it.
- **Left out:** the 206 published both ways (they say nothing); the 958 disproofs with hypotheses, until their
  exact negation has a certificate; Mathlib statements; anything whose certificate fails under our pin.
- **A published proof is a certificate only. It is never a training target.** Every target is a proof this
  run's solver wrote and Lean verified, so any gain is the loop's and not a copy of a stronger prover. (The one
  exception the owner has named is the later pretraining option below, outside L0 to L2.)
- **A published proof is brought to our pin mechanically, and Lean still judges it.** The published proofs
  were written for a 2024 Mathlib. Two fixed rewrites are applied before a certificate is checked, each marked
  on what it changed: the notation rewrite (`∑ x in s` to `∑ x ∈ s`, also applied to statements, decided
  below), and **the renaming of library names Mathlib has since renamed or removed** (a table in the config,
  old name to new name; applied to certificates only, never to a statement). A wrong entry costs a
  certificate; it cannot admit a false statement, because Lean accepts or refuses the result. Measured on 489
  published proofs at v4.27 (2026-10-04): 388 verified; of the 96 whose proof failed, 72 name a library name
  that is gone, and three names (`div_le_div_iff`, `le_div_iff`, `div_le_iff`) are 60% of those. STP's
  conjectures lose the most (44 of 80 verify). The other 5 are negations Lean did not confirm as exact.
- **Order of work:** certificates are tried in the order `ladder_loop.certificates.source_order`, at most two
  per source, the shortest first; a problem is settled by the first that verifies. The renaming pass runs on
  what the first pass left unsettled, with the table built from the names that pass actually reports missing.
  **The held-out sets are drawn only after both passes**, so that problems whose published proofs use the
  renamed lemmas (inequalities, mostly) are not missing from them.
- **The held-out sets do not wait for the whole pool** (owner, 2026-10-04: a round should not wait hours on
  the check). A seeded random **slice** of candidates is taken through both passes first: the present
  holdout's candidates; a seeded sample of Lean Workbook candidates; one of STP candidates; one of all
  candidates. H is the present holdout's verified problems plus the first verified ones of the Lean Workbook
  sample, in its seeded order, up to 1,000, and the first 1,000 verified of the STP sample. The base-map
  sample is the first 4,000 verified of the all-candidates sample that are neither in H nor duplicates of a
  member of H. A random sample of candidates, kept where verified, is a random sample of the verified: it is
  the same draw, made early. The rest of the pool is checked afterwards, at lower priority than the solver's
  own checks, and only grows the pool; a later duplicate of a member of H never enters it.

### The reward and the band

A problem's pass rate is p = k/n: k of the n solvers that attempted it resolved it. The challenger's reward for
that problem has ONE setting, `challenger.target_rate` t (default 1/4):

    r(k) = p (1 − p)^a / [ t (1 − t)^a ],   a = 1/t − 1

It is 1 at p = t and 0 at k = 0 and at k = n. At t = 1/2 it is the symmetric p(1 − p), scaled.

| k of 8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Reward at t = 1/4 (default) | 0 | 0.79 | 1.00 | 0.87 | 0.59 | 0.31 | 0.11 | 0.02 | 0 |
| Reward at t = 1/3 | 0 | 0.65 | 0.95 | 0.99 | 0.84 | 0.59 | 0.32 | 0.09 | 0 |
| Reward at t = 1/2 | 0 | 0.44 | 0.75 | 0.94 | 1.00 | 0.94 | 0.75 | 0.44 | 0 |
| A problem of that true rate yields any proof in 8 attempts | 0% | 66% | 90% | 98% | 100% | 100% | 100% | 100% | 100% |
| What a found proof teaches, −ln p (nats) | | 2.08 | 1.39 | 0.98 | 0.69 | 0.47 | 0.29 | 0.13 | 0 |
| Rise in the problem's own pass rate from one training step, p(1 − p)², relative | 0 | 0.65 | 0.96 | 1.00 | 0.85 | 0.60 | 0.32 | 0.09 | 0 |

The last two rows are models, not measurements. Both put k = 2 and k = 3 above k = 4; they disagree on k = 1
(the rarest proof teaches the most, and is the one most often not found at all). The measurement is L1's gain
by k.

- **Where the challenger aims.** k is a noisy count, so a challenger that maximises its expected reward aims at
  a true pass rate of 0.29 at t = 1/4, where 7% of its proposals yield no proof (0.36 and 3% at t = 1/3; 0.50
  and 0.4% at t = 1/2).
- **The band** is the pass rates the reward scores at least 0.8: **0.13 to 0.41** at t = 1/4 (k = 2 or 3 of 8;
  5 to 13 of 32). Every use of "the band" below is this interval, computed from t.

### Held-out sets

Fixed before any training and excluded from the pool, from every training set and from the challenger's data,
together with their duplicates by the existing duplicate check:

- **H:** 1,000 Lean Workbook problems (the usable ones of our present holdout, 309, plus a random draw) and
  1,000 STP conjectures. The base model gets `goal.base_episodes` (32) episodes on each.
- **G, the goal set:** the Lean Workbook problems of H the base resolved in none of its 32 episodes. From the
  present holdout that is at most 174 (164 known true, 10 known false) of 309, so about 560 are expected in
  all. The base was never asked for a negation before, so the known-false ones may fall.
- **The rungs:** the problems of H with at least one success, placed by the base's 32 episodes against the
  band: below it (1 to 4 successes), in it (5 to 13), above it (14 or more). The rung below the band is four
  counts wide; L0 must show that each rung has at least 137 problems, and if the lower rung cannot be filled
  that is L0's finding. An STP conjecture of H with no success is in no rung and not in G; it is counted on a
  line of its own.
- **Set aside:** the 693 holdout problems with no published answer. Never trained on; the first target of the
  novel stage.

### An episode

One solver's work on one problem. It **resolves** the problem when Lean accepts a proof of the statement or of
its exact negation. Until L3 an episode is one native-format attempt at each side (at most 1,024 new tokens and
30 s of Lean checking each). From L3 it is at most `episode.loops` (5) attempts over both sides, the model
choosing the side and seeing its last failed proof and Lean's first error; about 5 minutes in all. An episode
that reaches a cap has failed.

A verified proof on the side the certificate contradicts is a soundness alarm: the round stops.

**Until L3 the side the certificate rules out is not attempted** (owner, 2026-10-04: yes). An attempt at the
negation of a statement known to be true cannot verify, so sampling it and sending it to Lean changes no
count and costs half of every episode stage (at L0's sizes about 94,000 of 198,000 Lean checks, and about 8
of 20 million tokens). The episode's result is unchanged: a known-true problem is resolved by a proof of the
statement, a known-false one by a proof of its negation. For a seeded 2% of problems
(`episode.contradicted_side_audit_share`) both sides are still sampled and checked, which keeps the
soundness alarm. Settings: `episode.contradicted_side: audit`, `episode.skip_generating_contradicted_side:
true`. From L3 the model chooses its side and this rule ends.

### A round

Round r starts from model M(r); M(0) is the base model.

1. **Propose.** The challenger chooses `round.problems` (1,000) pool problems not proposed before. A chosen
   problem whose certificate has not yet been checked under the pin is checked now and dropped if it fails.
2. **Attempt.** Each problem gets n = `round.solvers` (8) episodes from M(r); k of them resolve it. Its pass
   rate is k/n.
3. **Reward the challenger** with r(k) of "The reward and the band" for that problem: at the default t = 1/4
   and n = 8 it is 0.79, 1.00, 0.87, 0.59 for k = 1 to 4, then 0.31, 0.11, 0.02, and 0 at k = 0 and k = 8. The
   challenger is refit on every (problem, k, n) seen so far, recent rounds weighted most.
4. **Train the solver.** Native format, one pass, on every problem of the round with k ≥ 1. The target is one
   of its verified proofs chosen at random with the round's seed, on whichever side was proved. M(r+1) is
   trained from the base model on all rounds' training sets so far.
5. **Measure:**
   - **Reach:** how many problems of G the model resolves at `measure.reach_episodes` (32) each, against the
     base's count under the same fresh budget (its luck, measured once).
   - **The rungs:** pass rates of the three held-out rungs under M(r+1) at `measure.rung_episodes` (8) each,
     each minus the base's. The base's side is a fresh measurement of its own (8 episodes, made once), never
     the 32 episodes that placed the problem: a problem placed low by a noisy count reads higher the next time
     with no training at all.
   - **Gain by k:** every problem of the round gets n fresh episodes from M(r+1) and n fresh from M(r), the
     model the round started from. Reported by the problem's k in the round: the change in pass rate, M(r+1)
     minus M(r), paired by problem, for the round's own problems; and the same change for the held-out rungs,
     by the base's pass rate. M(r)'s fresh episodes take out the same placing effect. This is the measured
     answer to "how strongly does a problem at each k push toward mastery", and it is what sets t: changing t
     is an edit of this spec made from that table.
   - **The challenger:** mean reward; share of its problems in the band, at k = 0 and at k = n; how hard the
     base found the problems it now chooses.
   - Share of resolutions that were negations; the standard evaluation of Phase A; distinct attempts.
6. **Stop rule.** A round whose interval for the below-band rung lies entirely below zero stops the loop for
   diagnosis. No more than three rounds run without a review.

There is no truth screen and no difficulty screen. The n episodes are the only solver sampling in a round
outside the measurements: they give the solver its training proofs and the challenger its reward.

### The challenger in this stage

Under rule 2 a freshly written statement has no published answer, so in this stage **the challenger chooses
from the pool; it does not write** (decided by the owner). Its simplest form: a small predictor of a problem's pass rate under the
current solver. It reads the statement (the base model's embedding of it, computed once) and published facts
(source, length of the published proof, STP's round number). Each round it draws 20,000 candidates at random,
takes the `round.problems` with the highest expected reward, and gives 10% of the places to random candidates
so it keeps learning. It starts from the base map of L0. Writing new statements returns in the novel stage,
trained by the same reward.

### The equal-compute control

L1's control arm draws its problems from the pool at random and spends the same number of episodes. At the end
of L2, and whenever reach moves, the base model is given the loop's whole sampling budget on G.

### A later option: pretraining on published proofs (not part of L0 to L2)

The owner, on certificates only: "though we might want to pretrain on some of these later if we can't make
progress, see if it helps against a holdout." If the loop does not move reach on G, one further experiment is
allowed: train the base on the published proofs of part of the pool, then run the loop from that model and
compare with the loop from the base. H and the 693 are excluded from that pretraining as from everything else,
so H is the holdout it is judged on. It gets its own pre-registered read before it runs. Until then no
published proof is a training target.

### Independence from the model and the Lean version

The model, its prompt builders and its native format are named in the config and nowhere else. The Lean pin is
`lean.pin` (O2a). The loop runs at v4.27.

## Milestones (each ends with a result file and a stop for review)

| | What | Decides | Rough cost |
|---|---|---|---|
| **L0** | The data, no training. The three sources pinned by revision. Certificates checked under v4.27: all Lean Workbook ones, and a first 50,000 STP ones. H held out; the base's 32 episodes on H give G and the rungs. The base map: 8 episodes on 4,000 random pool problems | How many certificates survive v4.27; the size of G; how much of the pool sits in the band for the base; whether each rung has at least 137 problems (the rungs are cut from L0's counts once t is fixed, so L0 itself does not depend on t) | About 281,000 Lean checks (9 hours on the one v4.27 server at 9 checks/s, about 2 on four); about 2 GPU hours |
| **L1** | One round, two arms that differ only in who chooses the problems: the challenger, or a random draw. Seed 0, then three seeds per the seed rule | Whether a round lifts the rung below the band, whether aiming at the target rate beats a random draw, and the gain by k that sets t for L2 | About 107,000 Lean checks (3.3 hours on one server at 9 checks/s, 49 minutes on four) and about 2 GPU hours per arm and seed: 16,000 for the round, 36,000 for reach, 23,000 for the rungs, 32,000 for the gain by k. About 23,000 more once, for the base's fresh rates on the rungs |
| **L2** | Three rounds of the challenger arm, then the equal-compute control | Whether the ladder climbs, and whether reach on G passes the base's luck | About 107,000 Lean checks per round |
| **L3** | Episodes with loops: Lean's error fed back, the model choosing the side. First a check without training against blind resampling at equal tokens, then training the repair from our own failed-then-solved attempts | Whether repair is a stronger search step than resampling | One scout, then one training run |
| **L4** | The novel stage: the challenger writes new statements; the 693 are its first target | Whether the loop settles statements nobody has | Its own spec |

### L1's read, fixed now

- **Primary:** fresh pass rate on the held-out below-band rung, the challenger arm minus the base, paired by
  problem, 95% interval by bootstrap over problems.
- **Also:** the same for the control arm, and challenger minus control; reach on G against the base's luck; the
  in-band rung; distinct attempts; each arm's mean reward and the share of its problems in the band.
- **Gain by k, reported and not a branch:** the table of "A round" step 5 for both arms. The random draw
  covers every k, so its table is the one that compares k = 1, 2, 3 and 4 on equal terms. A change of t for L2
  is made from it, as a spec edit.
- **Branches:** primary at or above zero: escalate to three seeds, then L2. Primary below zero with an interval
  that contains zero: escalate to three seeds too (one seed cannot tell a small loss from none; named
  2026-10-04, before the first seed was queued). Interval entirely below zero: stop
  and diagnose; the run failed, and the idea is not judged until the verdict gates pass. **VOID**, fix and
  re-run one seed, when the challenger's mean reward is no higher than the random draw's (the arm did not test
  aiming), when the below-band rung has fewer than 137 problems, or on a soundness alarm.
- **Pairing:** both arms use the same base, seed and box (the GPU box, as the owner authorized).

### L1b: the dose curve (owner, 2026-10-04, after L1's three seeds)

The owner's questions: *"plot the loss curves over time, both on the training set and on the held-out set. Are
we overfitting, and did we measure at our peak, and would training longer help with the held-out set?"* L1
cannot answer them: it recorded the training loss only, and it measured once, at the end of its one pass.

**What runs.** Seed 0's challenger arm again, from the base, on the same 741 training proofs, for **three
passes** in place of one (same optimizer settings, seed 0, the order reshuffled each pass). Nothing else of the
round changes; the first pass repeats L1's training.

1. **Loss over time, three curves, by part** (first proof token, body, final newline, fence):
   - the training loss of each step, read before its update. In the first pass every proof is new to the
     model when it is read, so this is a held-out reading for the round's own problems; from the second pass on
     it is not;
   - at fixed points (every step up to 12, then every 6): the loss on a fixed sample of 150 of the training
     proofs, and on **150 held-out proofs**: verified proofs the base wrote for held-out rung problems in L1's
     seed-0 measurement, 50 per rung, drawn with the seed. No held-out proof and no held-out problem is trained on.
2. **Held-out pass rate at checkpoints:** the adapter is saved after 0.5, 1, 1.5, 2 and 3 passes. Each
   checkpoint gets 8 episodes on every problem of the three held-out rungs, with L1 seed 0's sampling seed for
   the rungs, so that each is paired by problem with the base's stored result and with the others. Reach on G
   (32 episodes) is read at 1 and at 3 passes. Distinct attempts are read at each checkpoint.
3. **The 1-pass checkpoint must reproduce L1 seed 0** (in-band +0.038, above +0.057, within their intervals),
   or the run is VOID: it would not be the same training.

**The read, fixed before the run.** Paired by problem, 95% interval by bootstrap over problems.

| Question | Answered by |
|---|---|
| Are we overfitting? | Yes at a checkpoint where the held-out body loss is above its lowest point by more than the spread of its last four readings while the training-sample loss still falls, AND a held-out rung's pass rate is below its 1-pass value with an interval below zero. Loss alone is not a verdict: 13a's lesson is that the loss moved by a token while the pass rate did not. |
| Did L1 measure at the peak? | The checkpoint with the highest pass rate on the below-band and in-band rungs together, against the 1-pass checkpoint. "At the peak" if no other checkpoint beats 1 pass with an interval above zero. |
| Would training longer help? | The 2-pass and 3-pass checkpoints minus the 1-pass one, on the below-band rung (the primary) and the in-band rung. |

- **Branches.** A later checkpoint beats 1 pass on the below-band rung, interval above zero: escalate that
  number of passes to three seeds, and L2 uses it if it holds. No checkpoint differs from 1 pass: one pass
  stands, and the half-pass checkpoint says whether less would do. Later checkpoints are worse: one pass stands
  and the overfitting point is recorded. One seed is a scout: it can stop us training longer, it cannot prove
  a gain.
- **Cost.** About 5 minutes of training, about 15 of loss readings, five rung measurements of 5,832 attempts
  and two reach measurements of 12,768: about 55,000 Lean checks and about an hour on the GPU box.

### L1b at three seeds (the escalation; fixed 2026-10-04 after seed 0, before seeds 1 and 2 run)

Seed 0 read ESCALATE (`ladder-l1b-RESULT.md`): two passes beat one pass on the below-band rung, +0.023
[+0.003, +0.044]; its one-pass checkpoint reproduced L1; no checkpoint is overfitting by the rule above. One
seed cannot prove a gain, so the same thing runs at seeds 1 and 2.

**What runs.** The same stage for seeds 1 and 2, each on the training proofs its own L1 round stored (742 and
740), with the same three passes, readings and five checkpoints. One addition: reach on G is read at 2 passes
too (`dose.reach_at: [1, 2, 3]`), because two passes is the dose in question. Seed 0's two-pass adapter is on
the box; its reach at 2 passes is added once the stage can add a measurement to a finished run.

**The read.** For each problem, the mean over the three seeds of the paired difference; 95% interval by
bootstrap over problems (the pooling of L1's three-seed result).

- **Primary: the below-band rung, 2 passes minus 1 pass.**
  - Interval above zero: later rounds (L2) train two passes, unless the guard fails.
  - Interval contains zero: UNDETECTABLE at three seeds on that rung. The in-band rung then decides, and this
    is a secondary named after seeing seed 0's in-band figure (+0.072 [+0.040, +0.104]), to be labelled so:
    interval above zero, L2 trains two passes; otherwise one pass stands.
  - Interval below zero: one pass stands.
- **Guard: reach on G at 2 passes.** Problems the two-pass model resolves and the base afresh does not
  (gained), against the reverse (lost), summed over the seeds, two-sided sign test. Lost above gained at
  p < 0.05: two passes cost reach. Both numbers then go to the owner as a trade, and L2 does not switch by
  default.
- **VOID, per seed:** its one-pass checkpoint must reproduce that seed's L1 result on the in-band and above
  rungs (within L1's intervals), or the seed is fixed and run again.
- Three passes are reported with the same pooled read; no branch depends on them.

**Outcome (2026-10-04, `ladder-l1b-RESULT.md`).** Primary +0.011 [−0.001, +0.023]: UNDETECTABLE at three
seeds. Secondary (in-band) +0.040 [+0.021, +0.061]. The guard as written is not triggered (against the base
afresh two passes gain 41 goal problems and lose 44). The guard compared against the wrong thing: against ONE
PASS, which is the choice at hand, two passes gain 32 and lose 63 (p = 0.002; not pre-registered). So the
letter of the branch says two passes and the purpose of the guard says a trade for the owner. **One pass per
round stays the default until the owner decides.** A guard on a choice between two doses compares the two
doses; a later spec fixes that before its runs.

### L2: three rounds (the owner's go-ahead, 2026-10-04)

**Decided by the owner after L1b.**

1. **One pass per round.**
2. **No quota on known-false problems, and no hack in its place:** "the model should be rewarded the right
   way, not us put in hacks to avoid the model getting the right incentive on its own. Can we fix it through
   the reward function? Perhaps negations are worth less reward?"
3. **No rule that trains only on in-band problems:** "it seems like the reward function should resolve this
   especially over time."
4. **L2 goes ahead**, and the challenger scores the whole pool.

**What the stored data say about the reward and the two sides** (no new run: L1's candidate scores, the 6,000
picks whose k was measured, L1b's held-out changes; `ladder-l1b-RESULT.md`, "The reward and the two sides").

- **Refutations are where the middle of the pool is.** 89% of the known-false candidates are predicted at a
  pass rate of 0.25 or more, against 18% of the known-true ones (74% of those are predicted below 0.10). Of the
  693 candidates predicted between 0.25 and 0.50, 317 are known false. The challenger takes them because the
  reward says to.
- **A lower reward for a refutation is a switch, not a dial.** With a known-false problem's reward multiplied
  by v, round 1's 900 scored picks hold 375 known-false problems at v = 1, 258 at 0.9, 51 at 0.8 and none at
  0.7 or below (none at 0.9 once the whole pool is scored). The best candidates' expected rewards lie within
  12% of one another, because the estimate is loose: by the challenger's own rule the expected reward is 0.44
  at a true pass rate of 0.25, 0.47 at 0.35 and 0.43 at 0.50. Any real discount is the quota again, as a ban.
- **The reward as written moves off them when the solver learns them.** After one pass, unseen known-false
  problems above the band rise 9.5 points, against 5.1 for known-true ones (in the band 3.1 and 3.5). Moving
  each candidate's predicted pass rate by that measured change and choosing again among the candidates not yet
  proposed gives 99 known-false problems in the next 900 (11%); supply alone gives 198 (22%); round 1 had 375
  (42%). The picks' mean pass rate falls from 0.39 to 0.33. An estimate from stored scores, not a run.
- **That needs the challenger to see the current solver.** As "A round" is written it learns M(r)'s pass rates
  only from round r's own results, after round r's proposals are made: always one model late.

So the reward is not changed, and the challenger is made to learn from it sooner.

**What L2 changes in a round.**

- **Candidates: the whole pool.** Every pool problem not yet proposed, less H and the base map. L1 drew
  20,000 and used every one predicted between 0.25 and 0.50. Statements are embedded once, by the base model.
- **Proposals in batches.** A round's 1,000 proposals are made in `round.batches` (4) equal batches, each with
  its `challenger.random_share` of random places. After each batch's episodes the challenger is refit on
  everything seen so far: the current round's results at full weight, each earlier round's at
  `challenger.recency_decay` per round. Three quarters of a round's picks are then chosen knowing how the
  current model does on problems like them. No sampling is added; every proposed problem gets its n episodes;
  none is dropped or kept by a measured rate (fixture 5 stands). The challenger learns from its reward as the
  reward arrives.
- **The reward is unchanged:** r(k) at t = 1/4, the same for either side (rule 1). No quota. Nothing filters
  what is trained on.
- **Training:** M(r+1) from the base, one pass over every round's training set so far (one verified proof,
  chosen at random with the round's seed, for every problem with k ≥ 1).
- **Not measured in L2: the gain by k** (n fresh episodes of two models on every problem of the round, 16,000
  episodes a round, the largest block). L1 measured it at three seeds and t stays 1/4.

**Measured after each round's training:** the three rungs under M(r+1) (8 episodes) and G (32 episodes), each
paired with the base's fresh results of that seed; distinct attempts; and the challenger's table by batch and
by round: the share of the scored picks that are known false, the picks' mean pass rate, the shares at k = 0,
in the band and above it, the mean reward, the predictor's calibration, and the share of the training set that
comes from problems with k ≥ 4 and from refutations.

**The read, fixed before the run** (seed 0, then three seeds by the seed rule). Paired by problem, 95%
interval by bootstrap over problems; at three seeds, each problem's mean over the seeds.

- **Primary: the below-band rung, M(3) minus the base.**
- **The climb: each rung, M(3) minus M(1).** A ladder that climbs gains with rounds.
- **Reach on G as gained against lost** (two-sided sign test), for M(3) against the base afresh AND against
  M(1). A comparison of two models compares those two (L1b's lesson).
- **The equal-compute control**, after round 3: the base gets the loop's attempt episodes on G, 3 × 8,000 =
  24,000, which is 61 more on each of the 392 problems on top of its 32. Read: problems M(3) solves in 32
  episodes against the base in 93, as counts and as gained against lost.
- **The challenger's trajectory (the owner's 2 and 3).** Stated now: from round 1 to round 3 the known-false
  share of the scored picks falls and the picks' mean pass rate moves toward the target. If round 3's
  known-false share is at or above round 1's, the reward is not moving off them, and that goes to the owner
  with the switch table above.
- **Branches (seed 0).** Primary at or above zero, or below zero with an interval that contains zero:
  escalate to three seeds. An interval entirely below zero for any round's M(r+1) minus the base on the
  below-band rung: the loop stops for diagnosis (the stop rule).
- **VOID.** Round 1 with both the in-band and the above-band rung at or below zero (one round measured +0.044
  and +0.051 at three seeds in L1): the round did not train; fix and run again. A soundness alarm stops the
  run.
- **Cost.** A round: 8,000 attempt episodes in four batches, the training, 5,832 rung attempts and 12,768
  reach attempts, about 27,000 Lean checks and about half an hour. Once: embedding about 50,000 statements
  (about 20 minutes) and the control's 24,000 episodes (about 15 minutes). About two hours a seed.

**Made exact when the stage was built (2026-10-04, before any L2 run; none changes a read).**

- **Names.** The code and the report number the rounds 1 to 3: round r starts from M(r − 1) and trains M(r);
  M(0) is the base. "A round" above counts the same rounds from 0; the models' names agree (M(1) after one
  round, M(3) after three).
- **The candidates are 51,631:** the pool's 55,631 (H is not in the pool) less the base map's 4,000.
- **The stop rule stops the run.** It is read after each round's measurement; when it fires the later rounds
  and the control do not run and the report says where the loop stopped. Three looks at a 95% interval give
  about a 7% chance of a false stop in a seed.
- **The control's episodes are derived:** the rounds' attempt episodes over G's problems, rounded down (61).
- **The trajectory's words.** "Falls": round 3's known-false share of the scored picks is strictly below
  round 1's. "Moves toward the target": the scored picks' mean pass rate is closer to t in round 3 than in
  round 1.
- **A round's first batch is still chosen one model late;** the three batches after it are not.
- **Seeds.** The rungs and G use L1's sampling seeds of that seed (they pair with L1's stored base results);
  the rounds' attempts and the control have seeds of their own. Round 1 of L2 is not L1's round: other
  candidates, batches, another sampling seed.

**Added 2026-10-05, after seeds 0 and 1 and before it runs: the same extra attempts for the trained model.**
The equal-compute control charges the loop for its training attempts: M(3) with 32 attempts on each goal
problem against the base with 93. At seeds 0 and 1 the base wins that (26 gained, 55 lost) while M(3) succeeds
on 5.7 attempts per 1,000 against the base's 3.7. So one more measurement, which does not replace the control:
**M(3) gets the same 61 extra attempts on each problem of G, with the control's sampling seed**, so the two
models are compared at 93 attempts each.

- **Read:** problems of G solved in 93 attempts, M(3) against the base, as gained against lost summed over the
  three seeds, two-sided sign test; and the successes per attempt of each.
- Gained above lost at p < 0.05: three rounds moved the goal set at equal sampling. Otherwise: not shown.
- A run that has finished gets this measurement alone when its task is queued again; nothing else is sampled.
- Cost: 23,912 attempts and about 15 minutes a seed.

**Outcome at three seeds (2026-10-05, `ladder-l2-RESULT.md`).** Primary +0.020 [+0.007, +0.034]: positive,
interval above zero, every seed ESCALATE, no stop, no VOID. The climb: in the band +0.029 [+0.009, +0.048] and
above +0.019 [+0.011, +0.028]; below +0.006 [−0.003, +0.015]; the third round adds nothing that separates from
zero on any rung. Reach on G at 32 attempts: 54 gained, 43 lost, not shown. The equal-compute control: the
base wins, 92 against 30. At 93 attempts each: 64 gained, 52 lost, not shown, with M(3) succeeding on 1.5
times as many attempts (+1.8 per 1,000 [+0.6, +3.2]). The challenger: the known-false share falls (45%, 40%,
31%), as stated; the picks' mean pass rate moves away from the target (0.37, 0.47, 0.47), against what was
stated, because the pool's 718 candidates predicted between 0.25 and 0.41 (the part of the band its expected reward favours; 2,555 are predicted in the band as a whole) are used up in round 1. **No further rounds
run without a review** ("A round", stop rule): what to change is the owner's decision.

### L2t: the lower target (the owner, 2026-10-05: "lower reward target and add repair in parallel")

**Why.** At t = 1/4 the challenger's expected reward peaks at a true pass rate of 0.35 (its estimate of one
problem is loose), its picks average 0.37 and drift to 0.47, and the pool's middle is used up in a round. The
pool is deep on the hard side: 6,424 candidates predicted between 0.05 and 0.13 and 1,837 between 0.13 and
0.25. From stored scores, t = 1/10 puts the expected reward's peak at 0.25, the picks' mean pass rate at 0.27
and the known-false share at 25%, for about 18% fewer proofs (`ladder-l2-RESULT.md`, "What a lower target
would do").

**What runs.** L2 again at seed 0 with ONE setting changed: `challenger.target_rate` 0.10 in place of 0.25.
Same candidates, batches, seeds, sampling seeds, box and code; a run directory of its own
(`ladder_l2_t010_seed0`). The held-out rungs stay L1's (cut at t = 1/4): only the challenger's aim moves.
Three rounds, the control and the trained model's extra attempts on G, as in L2. The control arm is
`ladder_l2_seed0` (t = 1/4), already run: a paired two-recipe delta at one seed.

**The read, fixed before the run.**

- **Did the setting act (else VOID):** the scored picks of round 1 have a mean pass rate at least 0.05 below
  the control's 0.368.
- **Primary: the below-band rung, M(3) at t = 1/10 minus M(3) at t = 1/4**, paired by problem, 95% bootstrap
  over problems.
- **Secondary:** the goal set at 93 attempts each, M(3) at t = 1/10 against M(3) at t = 1/4, gained against
  lost and successes per attempt; the in-band and above-band rungs (fewer easy proofs are trained on, so a
  smaller gain there is expected and is not a failure).
- **The challenger, by round:** the picks' mean pass rate; the shares at k = 0, k = 1 to 3 and k ≥ 4 (fixed
  classes: the band moves with t); the known-false share; proofs per round; the candidates left by predicted
  rate.
- **Branches (the seed rule).** The control's own spread on this rung, M(3) minus the base over its three
  seeds, is 0.010 (+0.023, +0.028, +0.008). Primary at or above zero, or below zero by less than 0.010:
  escalate to three seeds. Below zero by more than 0.010: not escalated, and the verdict gates of the repo's
  methodology are run before anything is called a kill. A win cannot be shown at one seed.
- **Cost.** About 95 minutes on the GPU box.

**The scout's read (seed 0, 2026-10-05; `ladder_l2_t010_seed0` against `ladder_l2_seed0`): ESCALATE.** Recorded
before seeds 1 and 2 run; the read above is not changed.

- **The setting acted:** round 1's scored picks have a mean pass rate of 0.274 against the control's 0.368
  (0.093 lower; the bar was 0.05). By round: 0.274, 0.334, 0.273 against 0.368, 0.460, 0.468. The picks stay
  near a quarter where the control's drift to a half.
- **Primary: +0.0075 [−0.0068, +0.0226]** (M(3) at t = 1/10 minus M(3) at t = 1/4 on the below-band rung).
  At or above zero: escalate. One seed shows no win and no loss.
- **Secondary, for the record (one seed):** in-band +0.008 [−0.023, +0.039], above-band −0.003
  [−0.018, +0.013]. The goal set at 93 attempts each: 59 problems against 61 (gained 14, lost 16), and 263
  successes against 202 (per attempt +0.0017 [+0.0003, +0.0033]).
- **The challenger:** known-false share 25%, 8%, 10% (control 45%, 38%, 36%); never-solved picks (k = 0) 37%,
  35%, 47% (control 23%, 19%, 22%); proofs per round 590, 595, 500 (control 710, 744, 728). Candidates left
  with a predicted rate of 0.25 to 0.41: 718, 176, 31; of 0.13 to 0.25: 1,837, 1,503, 2,081.
- **What follows:** seeds 1 and 2 of the arm (`ladder_l2_t010_seed1.json`, `ladder_l2_t010_seed2.json`: job configs,
  not shipped in this copy),
  each paired with its own control `ladder_l2_seed<N>`. The read at three seeds: the same primary over the
  three seeds' problems together and by seed.

**Outcome at three seeds (2026-10-05; `ladder-l2t-RESULT.md`): NOT SHOWN on the primary.** The setting acted
at every seed (round 1's picks 0.27 against 0.37). Primary +0.0058 [−0.0040, +0.0156], by seed +0.0075,
+0.0038, +0.0060. In the band +0.002 [−0.018, +0.023]; above it −0.009 [−0.018, +0.001]. The goal set at 93
attempts each: successes per attempt +0.0017 [+0.0008, +0.0028] over the model at t = 1/4, at every seed (7.2
per 1,000 against 5.4; the base 3.6); problems solved 189 against 170 and the base's 158, with the gain in
one seed of three (against the base 20 gained and 20 lost, 39 and 8, 20 and 20), so it is not read as reach.
The challenger stays on target (0.27, 0.34, 0.24 against 0.37, 0.47, 0.47) and half of round 3's picks are
never solved. The hard side of the training set is the same size (2,486 proofs against 2,601); about 1,450
easy proofs are left out. The config's default stays 0.25: the arm did not win its read; which target further
rounds use is the owner's decision.

### L3a: the repair check, no training (the owner, 2026-10-05: "add repair in parallel")

Milestone L3's first step: "a check without training against blind resampling at equal tokens". Every
measurement so far is blind resampling: an attempt is one whole proof, checked once, and a failure tells the
model nothing (the owner's question of 2026-10-05). L2's result is reliability, not reach; repair is the
change that could move what the solver reaches. This step asks one thing: **given the same failed attempt, is
the next attempt more likely to verify when it starts from what Lean said, than when it starts from nothing?**

**How Lean's answer is fed back: cut at the first error and resume from the proof state.** Measured on the
pool, 2026-10-05: a failed attempt's messages carry the position of each error; the same proof cut before the
failing step, with `sorry` in place of the rest, comes back with the exact proof state there (hypotheses and
goal). The prover was trained by its authors to continue a proof from a prefix followed by the state in a
comment (`/- tactic state: ... -/`, the "truncate and resume" of DeepSeek-Prover-V1.5), so this is its own
format and needs no instruction-following it was never taught.

1. **The cut.** The first error by position. If it lies in the proof body: keep every proof line before the
   line it is on. If it is the theorem's own "unsolved goals" (the proof ran out with goals open): keep the
   whole proof. A cut that leaves nothing kept is allowed (the state is then the theorem's own goal).
2. **The state.** The kept prefix, then `all_goals sorry` at the cut line's indentation, checked by Lean. The
   state is every goal Lean reports there, provided no error lies before it. (Made exact 2026-10-05 from the
   build's dry run on 600 stored failed attempts: a plain `sorry` reports only the goal it closes, which left
   other open goals unshown in 38 of 592 states, and the prover's own format shows them all; `all_goals
   sorry` returned them in 31 of the 38.) If no state can be had (a cut inside a structure the prefix does
   not close, a timeout), that loop is a blind attempt and is counted as "no state": never dropped.
2a. **A proof that was already complete.** If Lean reports that no goals are left at the cut, the kept lines
   are a whole proof with something extra after it (3% of the rungs' failed attempts in the dry run). They are
   then checked on their own, and if they verify, that loop resolves the episode with nothing generated. Both
   resuming arms get this, since it comes from the cut and not from the state. It is counted as "trimmed" and
   reported apart, and the primary is given with and without it.
3. **The prompt.** The blind prompt, the kept proof lines, the state in the model's own comment format, and
   the model continues. The proof checked is the prefix plus the continuation.
4. **An episode** is one blind attempt and then up to `episode.loops` − 1 = 4 repair loops, each from the
   latest failed proof, stopping at the first verified one. Caps as now: 1,024 new tokens and 30 s of Lean
   per attempt. The attempt is on the side the certificate allows, as in L0 to L2: **the model choosing its
   side is not part of this check** and stays with the step that trains repair.

**Three arms, the same first attempts.** For each problem, `repair.episodes` first attempts are sampled blind
from the base model. Every first attempt that failed is continued three ways, each to at most 4 more attempts:

- **Blind:** 4 more whole proofs from the plain prompt (what every measurement so far does).
- **Resume with the state:** the repair loop above.
- **Resume without the state:** the same cut and kept prefix, no state comment. It separates what keeping
  the valid prefix buys from what Lean's state buys.

**Problems.** The goal set G (392) and the three held-out rungs (712), the base model. 4 episodes a problem on
G, 2 on the rungs. About 36,000 attempts and about 60,000 Lean checks (each repair loop costs one extra check
for the state); about 40 minutes on the GPU box.

**The read, fixed before the run.** Paired by episode (the same first attempt), each problem's mean over its
episodes, 95% bootstrap over problems.

- **Primary: on the hard problems (G and the below-band rung, 558 problems), episodes resolved within 5
  attempts, resume-with-state minus blind.**
- **The single step:** given a failed first attempt, the share whose next attempt verifies, by arm.
- **Problems of G resolved in any episode,** resume-with-state against blind, gained against lost.
- **Budget, so the comparison is at equal tokens:** generated tokens and Lean checks by arm. If the resume
  arm generates more than 10% more tokens than the blind arm, the blind arm is read at the number of attempts
  that matches.
- **What the model did with it:** the share of repair loops with a state; the share whose first step repeats
  the step that had just failed; how far along the proof the cut moves from loop to loop.
- **Branches.** Primary above zero with an interval clear of zero: repair is a stronger search step; build it
  into the round (episodes with loops) and train on failed-then-repaired proofs (L3b, its own read). Interval
  contains zero: not shown at this size; the two diagnostics above say whether the model ignores the state
  or repeats itself, which is what training on repair would be for, and that goes to the owner. Below zero:
  resuming is worse than resampling FOR THIS MODEL UNTRAINED; that is a finding about this format, not a
  verdict on repair.
- **Can this run see a win (else INCONCLUSIVE, not a verdict).** On the above-band rung, where a blind next
  attempt verifies about half the time, the resume-with-state arm's next attempt must verify at least half as
  often as the blind one; less means the prompt or the cut is broken. A state for fewer than half of the
  failed first attempts means the cut is broken. Either is fixed and run again.
- A verified proof on the side a certificate rules out stops the run (the soundness alarm), as everywhere.

**Outcome (seed 0, 2026-10-05; `ladder-l3a-RESULT.md`): NOT SHOWN AT THIS SIZE.** Both "can this run see a
win" checks pass (a state for 97.5% of the failed first attempts; above the band the resumed next attempt
verifies 55.5% against blind's 65.9%). Primary +0.0063 [−0.0099, +0.0220]; without the trimmed loops 0.0000.
The diagnostics, which the branch sends to the owner: the model uses the state (with it minus without it,
+0.026 [+0.013, +0.039] on the hard problems) and it repeats itself (the first step repeats the failed one in
35% of repair loops, and by the fifth attempt 46% of resumed proofs are exact copies of an earlier one in the
episode). The single step is where the gain is: the first resumed attempt verifies on 36 of 1,880 hard
failures against a blind attempt's 20 (within 2 attempts, +0.0125 [+0.0027, +0.0228]). On the in-band and
above-band rungs starting over beats resuming (−0.136, −0.055).

### L3a2: one repair step after each fresh failure, then start over (no training)

**Why.** L3a's repair loop resumed four times in a row from where it had just failed. Its first step paid (36
verified of 1,880 hard failures against a blind attempt's 20) and its later ones did not (it rewrote what Lean
had rejected). A reading composed from L3a's stored rows after seeing them, one repair step and then blind
attempts, gave +0.0134 [+0.0049, +0.0224] on the hard problems within 5 attempts. That reading chose its own
shape from the data it is read on, so it is a candidate. This step tests the policy it points to, on first
attempts and sampling seeds L3a never used.

**The episode under test ("alternate").** Up to 5 attempts, stopping at the first verified one:

1. a whole proof from the plain prompt (blind);
2. one repair step from attempt 1 (L3a's items 1 to 3 and 2a: the cut, the state by `all_goals sorry`, the
   prover's own comment format; the trimmed case resolves the episode and is reported apart);
3. a whole proof from the plain prompt;
4. one repair step from attempt 3 (the latest fresh failure, never from a repair step's own output);
5. a whole proof from the plain prompt.

A repair step with no state to give is a blind attempt and is counted so (as in L3a). **In both arms, a proof
whose text equals one already rejected in the same episode is not sent to Lean:** it is a failed attempt, its
tokens are counted, and it is counted as a known copy.

**Two arms, the same first attempts.** Blind: 4 more whole proofs from the plain prompt. Alternate: the episode
above. The base model, nothing trained. The attempt is on the side the certificate allows; the 2% audit of the
other side and the soundness alarm are as everywhere.

**Problems and size.** The hard problems, where the primary is read: G (392) and the below-band rung (166),
**12 episodes each** (6,696 episodes). The in-band and above-band rungs (546) at 2 episodes each, for the
question whether the policy loses where a fresh attempt is strong. A sampling seed of its own (not L3a's
1030). About 62,000 attempts, about 45 minutes on the GPU box. The size comes from L3a's spread: its
primary had a half-width of 0.016 at 1,900 hard episodes; 12 episodes a problem brings that to about 0.008,
against the +0.013 the composed reading suggests. One sampling seed: nothing is trained, so there is no
training seed to vary, and the episodes are the sample.

**The read, fixed before the run.** Paired by episode (the same first attempt), each problem's mean over its
episodes, 95% bootstrap over problems.

- **Primary: on the hard problems (558), episodes resolved within 5 attempts, alternate minus blind.**
- **Secondary:** the same within 2, 3 and 4 attempts; G alone and the below-band rung alone; G by problem
  (resolved in any of its 12 episodes), gained against lost; the in-band and the above-band rung within 5
  attempts; the verified share by attempt position in each arm (positions 2 and 4 are the repair steps);
  the trimmed resolutions, and the primary without them.
- **Budget:** generated tokens, prompt tokens and Lean checks by arm, and the known copies by arm. If the
  alternate arm generates more than 10% more tokens than the blind arm, the blind arm is read at the number
  of attempts that matches.
- **Branches.** Primary above zero with an interval clear of zero: this is the episode a round should use;
  how it enters the round (and what training on repaired proofs adds, L3b) goes to the owner with the
  numbers. Interval contains zero: not shown at a size that resolves about ±0.008; a repair step is then not
  worth its extra Lean check as a search step for the untrained model, and what is left of repair is training
  on it, the owner's call. Below zero with an interval clear of zero: L3a's single step did not repeat on
  fresh attempts; reported as that.
- **Can this run see a win (else INCONCLUSIVE, not a verdict).** A state for at least half of the failed
  attempts a repair step starts from; and on the hard problems the repair step at position 2 verifies at
  least half as often as the blind arm's attempt at position 2 (L3a: 1.9% against 1.1%). Less means this
  build broke the prompt or the cut.

**Made exact when the stage was built (2026-10-05, before any run).**

- **The arms share their blind draws.** A sampling seed does not depend on the arm (as in L3a), and the same
  prompt with the same seed is one generation. So attempts 3 and 5 are the same samples in both arms while
  both are open, and the arms differ only by what attempts 2 and 4 were: a repair step or a blind draw. Each
  arm's own distribution is unchanged; the pairing is tighter than the size above assumed.
- **A known copy:** "rejected" is an attempt that was sent and that Lean failed (not one never sent, not one
  Lean gave no answer for); "text" is the proof as it stands in the checked file; "the same episode" is the
  shared first attempt and that arm's own later attempts. A repair step that would start from a known copy is
  a blind attempt, counted with that reason.
- **The checks:** the state check counts both repair positions; the position-2 check compares counts.
- **Sampling seed** 1031 (L3a: 1030); no generation shares a seed with L3a's. Sizes: 7,788 first attempts
  (4,704 on G, 1,992 below the band, 310 in it, 782 above), about 61,700 attempts and 58,000 Lean checks.
- **Not shown by any dry run:** what the model writes at attempt 4 (nothing stored starts from a third
  attempt). The smoke run is the first place it is seen.

**Outcome (seed 0, 2026-10-05; `ladder-l3a2-RESULT.md`): NOT SHOWN AT THIS SIZE.** Both checks pass (a state
for 97.4% of the failed attempts a repair step starts from; at position 2 on the hard problems the repair step
verifies 177 of 6,562 against blind's 127). Primary +0.0048 [−0.0022, +0.0120]; without the trimmed steps
+0.0013 [−0.0054, +0.0081]. Within 2 attempts +0.0075 [+0.0015, +0.0137], and the lead is then kept and not
added to (hard episodes resolved within 2, 3, 4, 5 attempts: 311, 410, 516, 602 against 261, 371, 484, 570);
G by problem 49 against 47 (gained 12, lost 10). The second repair step is worth one blind attempt: on the
6,179 episodes still open in both arms after attempt 3 it verifies 94 times, a fresh attempt 98. (Corrected
2026-10-06 after the owner's review: this note first said "the blind arm then catches up" and compared the
second repair step with a blind attempt on different survivors, 106 of 6,286 against 113 of 6,325. Arms are
compared by running totals or on matched episodes, never by the rate among the episodes each has left.) The
smoke run did not reach attempt 3, so attempt 4 was first seen in the run
and was checked from its rows before the read (6,276 of 6,276 resumed steps start from attempt 3). By the
branch: an untrained repair step is not worth its extra Lean check as a search step; training on repair (L3b)
is the owner's call.

### L3c: an episode that keeps what verified (no training) — APPROVED by the owner 2026-10-07 ("yes on building")

That yes was for building and running the episode. The ceiling at the end of this section was approved
separately, on 2026-10-08, and has its own heading there.

**Why** (`reach-diagnosis-RESULT.md`). The problems the model never solves need longer proofs (median 4 lines
against 1 on the easy rung), and three rounds of training help where a short proof exists (2.3 times at 1 to 3
lines) and not where a long one is needed (0.7 times at 8 or more). In 58% of failed attempts on hard problems
every step verified but the last, and the next attempt starts from nothing. Pooling those verified steps
across a problem's stored attempts, with no generation, closed 15 hard problems that no whole attempt of that
run had, 9 of them never solved by any model here in 558 attempts. L3a and L3a2 resumed from ONE failed
attempt, which keeps that attempt's plan; this step keeps the verified facts of every attempt.

**The episode ("accumulate").** One problem, up to `accumulate.generations` = 8 generations (the round's
attempts a problem), stopping at the first verified proof. It holds a **pool**: an ordered list of verified
lemmas, empty at the start.

1. **A fresh generation:** a whole proof from the plain prompt, as now. Checked by Lean.
2. **Harvest.** From a failed proof, its leading top-level `have` steps that end before its first error are
   verified in place (L3a's cut gives the line). They join the pool with the names they bind made unique to
   that generation, unless the same statement is already there. A step that binds a pattern, or anything other
   than a `have`, ends the harvest: what follows it may depend on a goal it changed. The pool holds at most
   `accumulate.pool_blocks` = 12 blocks.
3. **Try the kept closers again.** A failed proof that was lemmas and then ONE closing step leaves that step
   as a kept closer (at most 8). Whenever the pool has grown, each kept closer is checked after the pool. This
   costs Lean checks and no generation. A proof that verifies this way resolves the episode and is counted
   apart ("assembled").
4. **A continuing generation:** the plain prompt, the pool's lemmas as the proof so far, and Lean's state
   after them in the prover's own comment format (L3a's items 2 and 3: `all_goals sorry`, every goal, so the
   pooled facts appear as hypotheses); the model continues. Checked as the pool plus the continuation;
   harvested like a fresh one (its new lemmas were verified with the pool above them, and join it).
5. **Which kind:** generation 1 is fresh. After that a generation continues when the pool has changed since
   the last continuing generation, and is fresh otherwise (an unchanged pool would be shown the same prompt
   again, which is how L3a's loop came to repeat itself). With an empty pool a generation is fresh.
6. A proof whose text Lean has already rejected in the episode is not sent again (L3a2's rule).

**Two arms, the same draws where they can be.** Blind: 8 whole proofs from the plain prompt. Accumulate: the
episode above. A sampling seed does not depend on the arm, so every fresh generation in the accumulate arm is
the same sample as the blind arm's at that position, and the arms differ only where the accumulate arm
continued. The base model, nothing trained. The side a certificate allows, the 2% audit and the soundness
alarm as everywhere.

**Problems and size.** The hard problems: G (392) and the below-band rung (166), 6 episodes each (3,348
episodes of up to 8 generations); the in-band and above-band rungs at 2 episodes each for the question whether
it costs anything where a fresh attempt is strong. A sampling seed of its own. About 45,000 generations and
about 110,000 Lean checks; about 45 minutes on the GPU box.

**The read, fixed before the run.** Paired by episode, each problem's mean over its episodes, 95% bootstrap
over problems. Arms are compared by RUNNING TOTALS only (resolved within k generations), never by the rate
among the episodes each arm has left (the owner's correction of 2026-10-06).

- **Primary: on the hard problems, episodes resolved within 8 generations, accumulate minus blind.**
- **Reach, the second number that decides:** problems of G resolved in any of their 6 episodes, accumulate
  against blind, gained against lost; and the same restricted to the problems whose shortest published proof
  is 4 lines or more (230 of G by the shipped line counts; this said 248, from an earlier count that took the
  proof with the fewest characters).
- **Secondary:** the running total at 2, 4 and 6 generations; the primary by the length of the shortest
  published proof (1, 2 to 3, 4 to 7, 8 or more lines); how the accumulate arm's resolutions came (a fresh
  proof, a continuing generation, an assembled proof); the lengths of the proofs each arm verified; the
  in-band and above-band rungs.
- **Budget:** generations, generated tokens, prompt tokens and Lean checks by arm. If the accumulate arm
  generates more than 10% more tokens, the blind arm is read at the number of generations that matches.
- **Can this run see a win (else INCONCLUSIVE):** a non-empty pool in at least 30% of hard episodes by their
  last generation (the stored attempts gave a verified run in 22% of failed attempts, unevenly over problems,
  so about half is expected after four fresh ones); a state for at least half of the continuing generations;
  and the pool as
  harvested stands with `sorry` in at least 95% of the checks made of it.
- **Branches.** Primary above zero with an interval clear of zero: the accumulating episode is the solver's
  episode from here, and L3d trains on the proofs it assembles (its own read; the question there is whether
  the model then writes the longer proofs itself). Interval contains zero and reach is not ahead: not shown;
  the probe's gain came from more attempts' pieces than an episode has, and the size of the pool is the next
  thing to vary, on the owner's word. Reach ahead (gained more than lost, sign test under 0.05) with the
  primary's interval through zero: the episode reaches problems without resolving more episodes; reported as
  that, and L3d is still the next step.

**Made exact before the build (2026-10-07).**

- **The pool only grows.** A lemma that came later may use one that came earlier (a generation's own harvest
  is ordered, and a continuing generation's lemmas were verified with the whole pool above them), so nothing
  is ever dropped from the middle or the front. A harvest that would take the pool past 12 blocks is cut at
  its tail, and a full pool takes no more. (The first wording, "when it is full the oldest go", would have
  broken lemmas that use them.)
- **The same statement twice.** A harvested `have` whose statement is already in the pool is not added; the
  name it bound is read as the pooled lemma's name in what follows it (its own later lemmas, its closer).
- **The pool is checked whenever it changes** (the pool, then `all_goals sorry`): that check gives the state
  for the next continuing generation, and if Lean reports an error in it the newest harvest is taken back out.
- **Closers are the model's own.** A kept closer is a closing step the model wrote. No fixed list of tactics
  is tried: the diagnosis used six plain ones to probe, and here a continuing generation, which sees the pool,
  has that job. A kept closer that names a lemma the pool does not hold is not tried. A (pool, closer) pair is
  checked once.
- **Names.** A harvested lemma's name gets a suffix for its generation; anonymous ones stay anonymous.
- **A pilot before the run:** the same stage on 40 hard problems at 2 episodes (`ladder_l3c_pilot`), because
  the fixture problems are too easy for a smoke run to reach a continuing generation (L3a2's smoke resolved 13
  of 14 episodes at the first attempt). The pilot's episodes are not read as a result.
- **Proof lengths** for the read by length: the line count of each held-out problem's shortest published
  proof, shipped as a small data file made from the pool build's candidates
  (`data/ladder_l0/heldout_proof_lines.jsonl`). It is used by the report only; no prompt and no rule sees it.
- **Sampling seed** 1032 at seed 0 (L3a 1030, L3a2 1031).

**Made exact by the build (2026-10-08, before any run).**

- **A pool stands** only when Lean reports no error at all and exactly one goal at the `sorry`. A timeout, no
  answer, no goal left or several goals also take the newest harvest back out.
- **A lemma** is a top-level `have` that binds one name or none and carries its own proof. A `have` with no
  proof, a pattern, or one followed by a combinator line at step indentation ends the harvest.
- **A closer and its own proof.** The pool right after a proof's own harvest, with that proof's own closer, is
  the proof Lean just rejected under pooled names: it is not sent. A one-step proof with no lemma also leaves
  a closer.
- **Nothing is harvested** from a known copy, from a proof with no error position (the token cap, a timeout),
  or from a continuing proof whose first error lies in the pool's own lines.
- **The pilot has a sampling seed of its own** (1033), so no generation of the run is seen beforehand.
- **A fourth branch,** which the read above did not name: the primary's interval entirely below zero. It is
  reported as "the accumulating episode resolves fewer episodes than blind attempts".
- **Extra Lean time falls on the accumulate arm:** a continuing proof re-proves the pool, and a closer with
  more facts above it is slower, under the same 30 s limit. Timeouts are counted as failures and reported by
  kind.

**Outcome at three seeds (2026-10-08; `ladder-l3c-RESULT.md`): THE ACCUMULATING EPISODE IS THE SOLVER'S EPISODE
FROM HERE, at each seed.** All three checks pass; the rows were checked before the read. Primary by seed
+0.0114 [+0.0039, +0.0191], +0.0114 [+0.0045, +0.0185], +0.0084 [+0.0015, +0.0152]; over the three,
+0.0104 [+0.0053, +0.0157] (1,294 of 10,044 hard episodes against 1,190), with the lead growing at every
generation (3, 31, 50, 71, 88, 98, 104 episodes within 2 to 8). Reach: the goal set by problem over 18
episodes, 92 against 65, gained 28 and lost 1; where the shortest published proof is 4 lines or more, 36
against 28, gained 8 and lost 0. Twenty of the 92 were never solved by the base or the trained models in 558
attempts (5 of the blind arm's 65). The same generations, 91% of the generated tokens, 1.5 times the Lean
checks. What it does not show: reliability (19 of the 28 gained problems were won in one episode of 18), and
a gain in episodes where 8 lines or more are needed (−0.0006 [−0.0082, +0.0064]). The untrained model mostly
answers a pool with a closing step (75%) and repeats a rejected proof in 27% of continuing generations; 151
of the arm's resolutions on hard problems are assembled proofs and 119 continuing generations. What follows
is L3d (the episode in the round, training on what it assembles), which needs its own spec and the owner's
approval.

**After it, on the owner's word: the ceiling.** One fine-tune on published proofs of pool problems the base
cannot solve (never held-out ones), the same recipe and number of proofs as a three-round model, measured on
the rungs and on G by proof length. Published proofs are certificates, not training text, by the owner's rule;
this is a labelled diagnostic and nothing trained this way is kept. It bounds what training on assembled
proofs can reach with this model.

### The ceiling: a labelled diagnostic — APPROVED by the owner 2026-10-08 ("Yes to all, keep pushing")

**What it is for.** L3c reaches problems and does not make them reliable; the step that would is training on
longer proofs (L3d). Before that is built, one question bounds it: **shown longer proofs of problems it cannot
solve, does this model learn to write such proofs in one shot?** If it does not even from published proofs,
training on a few hundred assembled ones will not do it, and reach has to stay in the search. If it does, the
size of the gain against the number of proofs says how many assembled proofs L3d needs.

**It is an exception, not a change of rule.** The owner's rule stands: published proofs are certificates and
not training text. This trains once on them, on a branch, as a diagnostic. No model trained this way is used
in a round or kept, its training file is not published, and its result is always labelled "ceiling".

**The training file** (`data/ladder_ceiling/training.jsonl`, built on the dev machine with Lean checks only, by
`python -m rlvr_lean.tools.ladder_ceiling_set`, which holds every rule below; its `--dry-run` selects and
counts and sends nothing):

- **Eligible:** pool candidates (never a held-out problem, never one of the base map's) on the side `true`,
  whose pass rate the challenger predicted for the base below 0.05 (round 1 of L2's seed 0): 4,190 Lean
  Workbook problems and 31,083 STP ones.
- **Chosen:** every Lean Workbook one, and a random 4,800 of the STP ones.
- **The proof:** of a problem's published proofs (as published, and as the pool build rewrote them for
  Mathlib's lemma renames), the one with the fewest lines that verifies under v4.27 as a solver's attempt is
  checked. A published proof is tried only if `len(statement) + len(proof) <= 2400` characters, so that no
  row can be refused for its length on the box: stored model proofs of 150 tokens or more measure at least
  1.25 characters a token (6,178 of them; 1% at 1.52, the median 1.96), so such a row is at most 1,920 tokens
  and the prompt's fixed part, under the 2,048 a training example is cut at. Of the proofs that pass, at most
  the three shortest by lines are sent to Lean.
- **Written:** 8,000 rows in a shuffled order: every chosen Lean Workbook problem with a verified proof, and
  STP ones to make up the number. The first 2,000 rows are the smaller dose.

**The run.** From the base, the round's training recipe unchanged, ONE pass over the 8,000 rows in the file's
order, with a checkpoint after 2,000 rows (about the size of a three-round model's training set) and at the
end. Seed 0. At each checkpoint: 8 episodes on the three held-out rungs and 93 attempts on each goal problem
(32 and 61, with the sampling seeds L2 used for its models), so each checkpoint pairs by problem with the
base's stored 93 attempts and with the three-round model at t = 1/10.

**The read, fixed before the run.** One-shot attempts only. Paired by problem, 95% bootstrap over problems;
problems solved as gained against lost.

- **Primary: the goal problems whose shortest published proof is 4 lines or more (230), successes per attempt
  over 93 attempts, the 8,000-proof model minus the base.** It is read against **the loop's own gain** there,
  ONE number, computed on 2026-10-08 from the stored three-seed rows, before the run: the model after three
  rounds at t = 1/10 minus the base on the same 230 problems, the three seeds pooled (279 attempts a
  problem), paired by problem: +0.00064 per attempt [−0.00025, +0.00162] (176 successes against the base's
  135 in 64,170 attempts, 2.74 against 2.10 per 1,000; by seed −0.00019, +0.00192, +0.00019). It is what the
  loop's own training reaches. Beside the primary the report also shows this seed's three-round model minus
  the base, from its stored rows: information that decides nothing.
- **Secondary:** the same by length group (1, 2 to 3, 4 to 7, 8 or more) and for the 2,000-proof model; goal
  problems solved at 93 attempts, gained against lost, by length group; the three rungs; the line counts of
  the proofs each model verifies; the share of distinct attempts.
- **Branches.** The primary above zero with an interval clear of zero, and at least twice the loop's own gain
  (a point of +0.00128 or more): the model can learn longer proofs from examples; L3d is worth building, and
  the two doses say how the gain scales with proofs. Above zero with an interval clear of zero but not twice
  the loop's own (a point under +0.00128): longer proofs help about as the loop's own do; L3d's case rests on
  assembled proofs being better aimed than published ones, which this cannot tell. Interval through zero at
  8,000 proofs (and a point not above the loop's own gain): for this model and recipe, one-shot writing
  of longer proofs is not shown to be learnable at this size; L3d as plain training is unlikely to pay, and
  reach stays in the episode (a larger pool, more generations, or training the continuing step).
- **Can this run see a win (else INCONCLUSIVE).** The training took: the mean training loss over the last 500
  rows is below the mean over the first 500. The model still writes proofs: its share of attempts without an
  answer on the rungs stays under 5%. And training on other provers' proofs has not broken it: its pass rate
  on the above-band rung is at least half the base's.
- **One seed:** a diagnostic that sizes the next step. If the primary's interval holds zero and its point is
  above the loop's own gain (+0.00064), two more seeds are run before anything is concluded.

**Made exact by the build (2026-10-08, before any run).**

- **The training set and its order.** The first 8,000 rows of the file (the last checkpoint is the end of the
  pass), in the file's order. Nothing is shuffled: a round shuffles its examples, and here the file's order is
  the order. A file with fewer rows is refused; rows past the last checkpoint would not be read.
- **A row as a training example.** The round's own builder: the theorem is `statement`, the completion is
  `proof`, the target is the proof and the closing fence, in the native format (the sequence-start token
  between prompt and proof).
- **The recipe is the round's.** Batch 8, learning rate 1e-4 after a warm-up of 5 steps and constant from
  there, the LoRA settings, gradient checkpointing, the adapter drawn from the task's seed. No loss is read
  on held-out proofs during training (L1b's readings): the read asks for the training loss only.
- **Checkpoints by rows.** A checkpoint is saved after the first optimizer step at or after its number of
  rows, and the rows seen by then are recorded (2,000 and 8,000 fall on steps 250 and 1,000). They are named
  `small` and `full`.
- **What the prepare step refuses,** with nothing written: a row of a held-out problem (the snapshot's H, and
  the groups of the L1 run it reads) or of the base map, anywhere in the file; a row not on the side `true`;
  a problem that appears twice; a file with fewer rows than the last checkpoint; a row longer than the 2,048
  tokens the round's recipe cuts an example at (`training.max_sequence_tokens`, counted with the model's
  tokenizer as the training step encodes the row), which would be trained on without its end. The training
  file's character rule keeps every row under that; the count on the box stays, as the check of it.
- **The training loss.** Each row's mean loss per target token, read before the update of the step it was in,
  stored for every row. The first check compares the mean over the last 500 rows with the mean over the first
  500. A run of fewer than 1,000 rows (the smoke run) compares its two halves.
- **The 93 attempts are two samplings of G:** 32 episodes with L1's sampling seed for G (1001 at seed 0) and
  61 with the seed of L2's control (1020), the number read from what that control stored. The base's are L1
  seed 0's `reach_base` and `ladder_l2_seed0`'s `control`; the three-round model's are
  `ladder_l2_t010_seed0`'s `reach_m3` and `control_m3`. One base for every comparison: the arm's run sampled
  the control again (it differs by one success on one problem), and that copy is not read. The rungs get 8
  episodes with L1's seed for them (1002). Stored rows sampled with another seed, another number of episodes
  or on other problems are refused. An attempt is one episode, as in L2's read at equal attempts.
- **The loop's own gain is ONE fixed number,** the three seeds pooled, and not this seed's: +0.00064 per
  attempt (`ladder_loop.ceiling.loops_own_gain`; the Primary bullet says where it comes from). This seed's
  three-round model would not do as the reference: at seed 0 it has 52 successes against the base's 56 in
  21,390 attempts on the 230 problems, −0.00019 [−0.00140, +0.00112], so twice its gain would lie below zero
  and every primary clear of zero would pass. The branches are arithmetic on the primary and the fixed
  number: an interval clear of zero and a point of at least +0.00128; an interval clear of zero and a point
  under +0.00128; an interval that holds zero and a point above +0.00064 (two more seeds); an interval that
  holds zero and a point at or under +0.00064 (not shown). The report shows this seed's three-round model
  beside the primary as information, says that it decides nothing, and reads its branch just the same when
  no stored three-round model was read.
- **A case the branches did not name:** the primary's interval entirely below zero. It is reported as "the
  model trained on the published proofs solves fewer of these problems per attempt than the base". An interval
  that ends at zero holds zero.
- **The second check's "without an answer":** an attempt that reached the token cap (it is never sent to
  Lean) or on which Lean gave no verdict, as a share of the model's attempts on the three rungs. The two
  counts are printed apart.
- **The second and third checks are read on the 8,000-proof model,** the one the primary is read on. The
  2,000-proof model's numbers stand beside them and decide nothing. The third check's pass rate is the mean
  over the above-band problems.
- **When a check fails** the report prints the three checks and INCONCLUSIVE, and nothing else. What was
  measured stays in the report file under `measured_and_not_read`, for whoever repairs the run.
- **The secondary reads.** The three-round model is read like the two ceiling models, from its stored rows:
  by length group, problems solved, the rungs, lines, distinct attempts. The lines of a verified proof are
  counted as a published proof's are, on G over all 93 attempts and on the three rungs, with the shares of 4
  lines or more and of 8 or more. Distinct attempts are counted as in L1 and L2, on the rungs and on G.
- **Nothing is kept, and the adapters go only after a report that can be read.** The report step deletes
  both adapters once a report that can be read is written, and records that it did. A report that is not to
  be read (Lean gave no verdict on more than 2% of a set's attempts, the rule of every report here) leaves
  the adapters where they are, says that they were kept and why, and fails its step as before. Queued again,
  the task measures again the sets of its own that Lean did not answer, and no other, from the kept adapters,
  with no new training; the first report that can be read deletes them. A STORED set (the base's, the
  three-round model's) that Lean did not answer is refused by the prepare step, before anything is trained:
  no report could be read against it. A run whose report is written is not trained again. The run directory
  holds no statement and no proof of the training file (ids, kinds, line counts and token counts only); the
  training step reads the file again and refuses one whose hash the prepare step did not record.
- **Names.** Stage `ladder_ceiling`, run directory `ladder_ceiling_seed<N>`. Every file it writes has
  `ceiling` in its name but `problems.jsonl`, which the shared episode step reads by that name (its sets are
  `ceiling_...`).
- **The smoke stage** (`ladder_ceiling_smoke`) reads `ladder_l1_smoke` alone: a 24-row fixture
  (`data/ladder_ceiling_fixture`, rows of the trial sample) in the training file's place, checkpoints after 12
  and 24 rows (saved after steps 2 and 3). No L2 smoke run holds the stored attempts the real run reads, so
  none are read: G gets its first sampling only and no three-round model stands beside. The fixture's goal
  problem has no shipped proof length, so the primary has no problem and no branch is named.
- **The file as built** (2026-10-08, by the tool, before the run; SHA-256 `b8d7444c…adf98c`). 19,421 checks:
  15,405 verified, 3,974 rejected, 42 timed out at 60 s; 5 published proofs were not tried for their length.
  8,000 rows: 4,123 Lean Workbook problems and 3,877 STP ones. Proof lines: median 8, mean 9.4, nine tenths
  within 18; by length group 264 of 1 line, 942 of 2 to 3, 2,603 of 4 to 7 and 4,191 of 8 or more. The first
  2,000 rows: 1,025 Lean Workbook; 68, 234, 642 and 1,056 by group. The longest row is 2,310 characters. A
  rebuild can differ in the few problems whose checks time out (three builds this day left 63, 47 and 42
  timeouts, under a pool that other work was using): the committed file is the one trained on.
- **The smoke run** (`ladder_ceiling_smoke_r1`, 3 minutes, the real model, from c456e161c). The prepare step
  counted the fixture with the model's tokenizer (the longest row 869 tokens for 1,746 characters; 9,518
  tokens for 16,221 characters with the prompt's fixed part, 1.7 characters a token); three steps trained
  (peak 9.2 GB); both adapters were measured through Lean; the three checks passed and no branch was named.

**Outcome (seed 0, 2026-10-08; `ladder-ceiling-RESULT.md`): THE MODEL CAN LEARN LONGER PROOFS FROM EXAMPLES.**
The three checks pass. Primary +0.0478 [+0.0380, +0.0584]: 50.4 successes per 1,000 attempts against the
base's 2.62 on the 230 goal problems of 4 lines or more, 75 times the loop's own gain there. Goal problems
solved at 93 attempts: 245 against the base's 59 (gained 187, lost 1), 209 with 2,000 proofs; the loop's
three-round model at this seed 59 (gained 20, lost 20). Solved reliably (at least half of 11 episodes of 8
attempts): 106, against 3 for the base and 15 for the loop's model. No goal problem's statement is in the
training file, and on the 123 problems of the primary with no near statement there the gain is +0.0438
[+0.0296, +0.0596]. Both adapters were deleted by the report step. What it puts to the owner (the named
fallback: start the loop from a model pretrained on published proofs) is in the note; L3d's first step runs
in any case.

### L3d: train on what the episode reaches — direction APPROVED by the owner 2026-10-08 ("Yes to all, keep pushing"); the read fixed here before any run

**The aim, as a count.** The owner's aim is to do reliably what could not be done before. On the goal set
that can be counted: **a goal problem is solved reliably when it is resolved in at least half of its
episodes**, an episode being 8 one-shot attempts (with assembly after them where the system has it). From
stored one-shot attempts, each model on its own fresh samples (11, 7 and 11 episodes a problem at seeds 0, 1
and 2):

| | Solved at least once | Reliably (at least half of its episodes) |
|---|---|---|
| The base | 57, 37, 55 | 3, 1, 3 |
| Three rounds at t = 1/4 | 61, 47, 52 | 7, 12, 5 |
| Three rounds at t = 1/10 | 59, 62, 55 | 15, 13, 8 |

Training makes problems reliable and does not reach more of them. The episode's assembly reaches (94 goal
problems against 65 over 18 episodes, base model) and makes none reliable (2 against 2 at half). L3d asks
whether training on what the assembly finds turns its reach into reliability.

**Measured before designing it (2026-10-08; Lean only, no generation).**

1. **The form of the episode** (`ladder-l3c-RESULT.md`, addendum). The blind arm's own 8 attempts with the
   episode's Lean-only part resolve 1,373 hard episodes of 10,044, against 1,294 for the accumulate arm as
   run and 1,190 blind. So an episode here is 8 one-shot attempts with assembly after them, and no
   generation continues from a pool.
2. **What a round yields.** The same replay on the stored rounds at t = 1/10 (seeds 0 and 1): of the 375 to
   565 problems a round leaves with no verified attempt, assembly resolves 10, 12, 14 and 11, 19, 27 (2.4%
   to 5.1%, rising with the rounds as the solver is trained), with proofs of median 7 lines (longest 16).
   That is about 15 new proofs a round beside about 600: a round as it stands holds too few to show whether
   they teach anything.
3. **What is stored.** The training-side sets of L1, L2 and L2 at t = 1/10 hold attempts at 8,141 pool
   problems. No stored attempt verified 3,516 of them, and 91,956 failed attempts at those are kept (a
   problem's median 24).

**Step 1: do assembled proofs teach? One seed, about 70 GPU-minutes.**

- **The harvest H0** (Lean only; `tools/ladder_harvest.py`). For every pool problem that no stored
  training-side attempt verified, each side's stored failed attempts (a text once; in the order run, set,
  attempt; at most 48) are checked again for Lean's error positions and put through the episode's Lean-only
  part, with L3c's rules and sizes. Held-out problems are never read.
- **A harvested proof is minimised before it is trained on.** An assembled proof holds every pooled lemma,
  those of unrelated failed attempts with it. Its pool blocks are taken out one at a time, from the last to
  the first, and a block stays out when the proof still verifies. What is left is checked once more as a
  solver's attempt is, and that text is the training text; if that check fails, the proof as it was
  assembled is used. Names are left as the pool made them.
- **Two models from the base, the round's recipe, one pass, seed 0.** `without`: the training examples of
  seed 0's three-round model at t = 1/10 (its `training_examples_r1` to `_r3`), in a seeded order. `with`:
  the same rows in the same relative order, with H0's proofs at seeded places among them. Both are trained
  in this run by the same code, so that what differs between them is H0 and not the run.
- **Measured** as the ceiling's models are: 8 episodes on the three rungs and 93 one-shot attempts on each
  goal problem with L2's sampling seeds, so each pairs by problem with the stored base and the stored
  three-round model. After the run, Lean only: each model's stored goal attempts cut into episodes of 8 in
  the order drawn, with assembly (item 1's replay), for the reliable count.

**The read, fixed before the run.** Paired by problem, 95% bootstrap over problems; totals over the same
problems and the same number of attempts.

- **Primary: the goal problems whose shortest published proof is 4 lines or more (230), successes per
  attempt over 93 one-shot attempts, `with` minus `without`.** It is the ceiling's primary, so the two can
  be set side by side per proof trained on.
- **The noise floor:** `without` minus the stored three-round model on the same quantity. They have the same
  training examples and differ by the run and the order: this is what two trainings on the same data differ
  by, and no primary within it is read as an effect.
- **Secondary:** the same by length group and on all of G; the three rungs; goal problems solved at least
  once and reliably, by attempts alone and with assembly, for `with`, `without`, the stored three-round
  model and the base; the lines of the proofs each model verifies and the share of its attempts that open
  with a `have` (whether it writes differently).
- **Branches.** Interval clear of zero and outside the noise floor: assembled proofs teach; a matched
  control is trained first (`without` and as many more ONE-SHOT proofs as H0 holds, from verified attempts
  stored in other runs at pool problems outside the training set), to tell assembled proofs from more
  proofs, and then Step 2 goes to three seeds. Interval holds zero: not shown at H0's size; with the
  ceiling's two doses beside it the note says whether a gain of the size the ceiling shows for each proof
  could have been seen here, and if it could not, the verdict is UNDETECTABLE AT THIS SIZE and what follows
  (a larger harvest by generation, or none) is decided on the ceiling's read. Interval below zero: the
  assembled proofs cost one-shot attempts; Step 2 is not run as designed.
- **Can this run see a win (else INCONCLUSIVE).** H0 holds at least 100 proofs (with fewer, Step 1 is not
  run on it and the harvest is enlarged by generation first). Both trainings took: the mean training loss
  over the last tenth of the rows is below the mean over the first tenth. Both models still write proofs:
  under 5% of attempts on the rungs without an answer. And `with` was trained on H0: every one of its rows
  is counted in the training step's record.

**Step 2: the loop with assembly in the round (three seeds; built and sized only on Step 1's and the
ceiling's read).** The L2 stage's arm at t = 1/10 with one more setting: after a batch's attempts, the
Lean-only assembly over the 8 attempts of each problem none of them resolved. A problem an assembled proof
resolves is resolved, and its minimised proof is its training example. For the challenger's reward k stays
the number of verified attempts, and a problem with k = 0 that assembly resolves counts as k = 1: the
episode found one proof. (Behaviour is shaped through the reward: this pays the challenger for problems at
the edge of what the episode can do, which is where assembled proofs come from.) The size of a round (1,000
problems, or more by the same ranking) is fixed before the run from Step 1 and the ceiling. Its read: the
goal set by proof length, by attempts alone and by episode with assembly, and the reliable count, against
the stored arm at t = 1/10 of the same seed.

**Made exact by the build (2026-10-08, before any run; Step 1 only, Step 2 is not built).**

*The harvest (`tools/ladder_harvest.py`; the episode's Lean-only part is `domain/repair/replay.py`).*

- **What is read.** Under `--runs-root`, the task directories whose name starts `ladder_l` and holds neither
  `smoke` nor `pilot`, in the order of their names; their `steps/episodes_*_attempts_*.jsonl`. A run pulled
  more than once (`<run>_r1`, `<run>_r2`) gives each file once. A file is left at its first row that is a
  held-out problem or is not a pool candidate (`data/ladder_l2/candidates.jsonl`: the whole pool), so a
  held-out problem is never read; no base-map problem is a candidate either. "In the order run, set,
  attempt" is the order (run, file name, row).
- **What is replayed.** Each SIDE of a problem is an episode of its own (the negation's statement is the
  exact negation, built as the pool builds it). Its attempts are the stored ones with status `lean_error`
  and a completion: a text once (as it stands in the checked file), none with a forbidden token, at most 48
  (`ladder_loop.l3d.harvest.attempts`). Sizes are L3c's (12 pool blocks, 8 kept closers); a proof's Lean
  limit is an episode's (30 s); every check is background work; a file is named by a hash of its text.
- **A stored failure that verifies when it is checked again** is counted apart and is no assembled proof: it
  is not written to H0.
- **One proof a problem.** A proof Lean verifies on the side a problem's published answer rules out is the
  soundness alarm: the tool stops with exit code 3, and what it held is kept in `<out>.partial.json`. So no
  problem can resolve on both sides, and the rule "if both sides resolve, the first resolved" has no case;
  two rows of one problem would be refused as a proof of both sides.
- **Minimisation.** Every pool block is tried, the ones the closing step names too: one check a block, from
  the last block to the first, each time on what is left. The proofs go together, one place from the end a
  batch. The text that is left is checked once more, read as a solver's attempt is at the loop's pin. A row
  records the blocks and lines before and after, the checks, and `minimised: false` when that last check
  failed and the proof as assembled is the training text.
- **What is written.** One row a problem: `problem_id`, `side`, `theorem`, `completion` (a round's training
  example), and `kind`, `published_side`, `assembled: true`, `resolved_after_attempt`, `attempts_replayed`,
  `stored_attempts`, `pooled_attempts`, `runs`, `blocks_before`, `blocks_after`, `lines_before`,
  `lines_after`, `minimise_checks`, `minimised`. Beside it `harvest_h0.summary.json` (counts, the task
  directories read, every check status, timeouts). While it runs, what is resolved so far is written to
  `<out>.partial.json` every 4 attempts and removed at the end. The tool keeps nothing else between runs: a
  rerun sends the same files and the pool answers them from its cache.

*The goal set's replay (`tools/ladder_goal_assembly.py`; the count is `domain/ladder_round/reliable.py`).*

- **An episode** is one side's 8 attempts of one stored sampling, in the order drawn (a row's `episode`);
  each sampling is cut on its own and the attempts left over are no episode (32 give 4, 61 give 7: 11 a
  problem, 5 attempts left over). It ends at its first verified attempt; the failed attempts before that are
  checked again and replayed. A stored failure that verifies on the re-check is not replayed and is no
  success. A PROBLEM's episode is resolved when either side's is.
- **The counts:** goal problems resolved in at least one, a quarter, half ("reliably": 6 of 11) and nine
  tenths of their episodes, by attempts alone and with assembly, on all of G, by the length group of the
  shortest published proof and on the problems of 4 lines or more. A problem with no episode is counted in
  none. `report` prints several models' outputs side by side. The alarm is the harvest's.

*The stage (`ladder_l3d1`, and `ladder_l3d1_smoke`; `gpu/ladder_l3d1.py`).*

- **Names.** Run directory `ladder_l3d1_seed<N>`. Steps `ladder_l3d1_prepare`, `_train_without`,
  `_train_with`, `_measure_without`, `_measure_with`, `_report`. Sets `l3d1_rungs_<model>`,
  `l3d1_reach_<model>`, `l3d1_more_<model>`. Every file it writes has `l3d1` in its name but
  `problems.jsonl`, which the shared episode step reads by that name.
- **It is the ceiling's code, not a copy.** The reading of the stored runs, the one pass and the measurement
  are functions of `gpu/ladder_ceiling.py` that both stages call (`read_stored_runs`, `one_pass`,
  `measure_model`). The ceiling stage does what it did: its tests pass unchanged, and its prepare and
  stand-in training write the same 13 files byte for byte on the stored rows of seed 0.
- **What is read on the box,** never written: what the ceiling reads (L1's run of the seed; L2's run, for
  the base's control attempts; the run of the arm `l3d.step_1.loop_arm`, t010, for its three-round model),
  and from that last run its `training_examples_r1.jsonl`, `_r2` and `_r3`, which are `without`'s rows. At
  seed 0 they hold 590, 595 and 500 rows: 1,685, one a problem. The report alone also reads, if it is there,
  the ceiling's report of the seed (`ladder_ceiling_seed<N>/report_ladder_ceiling.json`).
- **H0 travels in the snapshot** (`data/ladder_l3d/harvest_h0.jsonl`). The prepare step refuses, before
  anything is trained: a snapshot without it; fewer than 100 rows (`minimum_h0`); a row that is not an
  assembled proof with its statement; a problem twice; a held-out problem (the snapshot's set, and the held-
  out groups of the L1 run read); a base-map problem; a problem the rounds' training examples hold already
  (a problem in both); a row longer than the 2,048 tokens an example is cut at.
- **The two orders, by the TASK's seed** (0 at seed 0), each a content hash, so that a rerun gives the same
  order whatever order the files are read in. `without`: the rounds' rows sorted by
  SHA-256 of `<seed>:l3d1_order:<attempt id>`. `with`: those rows in that order, none moved, and each row of
  H0 put in one of the rows + 1 gaps (before the first row, between two, after the last): the gap is
  SHA-256 of `<seed>:l3d1_place:<problem id>#h0` taken modulo rows + 1, and rows of H0 in one gap stand in
  the order of their hashes. Both orders are stored with their texts (`l3d1_training_without.jsonl`,
  `l3d1_training_with.jsonl`); a training step reads its own run's file and no other.
- **The two trainings** are two steps, each a process of its own: from the base, one pass in the prepared
  order with nothing shuffled, the round's recipe, the SAME adapter seed (the task's), one adapter saved at
  the end. Each stores every row's loss, read before its step's update, and the training loop's own record of
  the rows each optimizer step was made on (`rows_trained` in `l3d1_loss_<model>.json`).
- **Both adapters are kept** (`adapters/without`, `adapters/with`): they are the loop's own models.
- **The measurements** are the ceiling's: 8 episodes on the rungs with L1's seed for them, 32 attempts on
  each goal problem with L1's seed for G and as many more as L2's control gave the base with the control's
  seed (61: 93 a goal problem, 42,817 generations a model). A report that is not to be read (Lean gave no
  verdict on more than 2% of a set's attempts) fails its step; queued again, the task measures that set,
  and no other, again from the kept adapter. A stored set Lean did not answer is refused by the prepare step.
- **The four checks.** (1) H0's rows against `minimum_h0`: the prepare step has refused fewer already, and
  the report prints the count. (2) For EACH training, the mean loss over its last tenth of rows below the
  mean over its first tenth; a tenth is the rows times 0.1 rounded down, at least one (168 of 1,685). (3) For
  EACH model, the ceiling's second check: under 5% of its attempts on the three rungs reached the token cap
  or got no verdict. (4) Every row of H0 stands exactly once in the record of the rows `with` was trained
  on, none stands in `without`'s, and each record is the prepared order and nothing else. Any FAIL: the
  report prints the four checks and INCONCLUSIVE and nothing else; what was measured stays in the file under
  `measured_and_not_read`.
- **The noise floor is read as a size.** `without` minus the stored three-round model has no natural sign
  (which training is subtracted from which is arbitrary), so the floor is ONE number: the end of its 95%
  interval farthest from zero. The primary is read as an effect only when its interval is clear of zero AND
  its point is farther from zero than that. (A point outside the floor's own interval but inside its size,
  say +0.0004 against a floor of −0.0005 [−0.0012, +0.0002], is not read as an effect.)
- **The branches,** in this order. A check failed: INCONCLUSIVE. Interval above zero and the point outside
  the noise floor: ASSEMBLED PROOFS TEACH. Interval below zero and the point outside it: THE ASSEMBLED PROOFS
  COST ONE-SHOT ATTEMPTS. Interval clear of zero and the point within the noise floor, a case the Branches
  bullet does not name: WITHIN WHAT TWO TRAININGS ON THE SAME DATA DIFFER BY, and no effect is read. Interval
  holds zero (one that ends at zero holds it): NOT SHOWN AT H0'S SIZE, or UNDETECTABLE AT THIS SIZE by the
  note below. No goal problem of 4 lines or more, or no stored three-round model (a smoke run): NOT READ.
- **The note beside an interval that holds zero.** From the ceiling's report of the seed, when it is on the
  box, can be read and is not inconclusive: each of its two models' primary over the rows it had seen
  (2,000; 8,000) is the gain the ceiling shows for each proof; times H0's rows it is the gain expected here.
  This run resolves half the width of its primary's interval, or the noise floor's size when that is larger.
  If the expected gain reaches that at either dose, a gain of the ceiling's size could have been seen: NOT
  SHOWN AT H0'S SIZE. If at neither: UNDETECTABLE AT THIS SIZE. With no ceiling report to read, the verdict
  is NOT SHOWN AT H0'S SIZE and the report says the note was not made.
- **The secondary reads.** By length group and on all of G for five pairs: `with` minus `without`, the
  noise floor, and `with`, `without` and the stored three-round model each minus the base. The three rungs
  for the first two. Goal problems solved at least once and reliably BY ATTEMPTS ALONE, for the four
  models, from each one's attempts cut into episodes of 8 as the replay tool cuts them (no Lean); the counts
  WITH ASSEMBLY are not in the stage's report, the replay tool makes them afterwards. Goal problems solved
  at 93 attempts, `with` against `without`, gained and lost. The lines of the proofs each model verifies, as
  the ceiling counts them. The share of a model's attempts whose first step is a `have`: of every attempt,
  whatever became of it, the first line that is neither empty nor a comment begins with the word `have`.
- **The smoke stage** reads `ladder_l1_smoke` alone. `without`'s rows are that run's own stored training
  examples (`training_examples_challenger.jsonl`, 12 rows in the pulled copy); no L2 run is read, so G gets
  its first sampling only and there is no noise floor. H0 is an 8-row fixture (`data/ladder_l3d_fixture`:
  8 of the 36 proofs the round replay of item 2 assembled, the model's own, not minimised), with the least
  number of rows lowered to 8 by the stage's environment. The fixture's goal problem has no shipped proof
  length, so no branch is named.
- **Measured on the stored rows before the run** (no GPU, no Lean; the prepare step and the report against
  the pulled copies of seed 0). `without` is 1,685 rows, 211 optimizer steps. The report's count by attempts
  alone gives the table above at seed 0 (the base 57 and 3, t = 1/4 61 and 7, t = 1/10 59 and 15). And the
  size this read resolves: the primary's quantity between two STORED three-round models of seed 0 (t = 1/4
  minus t = 1/10, 50 successes against 52 in 21,390 attempts) is −0.00009 [−0.00140, +0.00112], a half-width
  of about 0.0013 per attempt where the base's rate is 0.0026. An interval of `with` minus `without` of that
  width is clear of zero only for a difference of about half the base's rate.

**H0 as harvested (2026-10-08, by the tool, Lean only): 92 proofs. Step 1 is not run on it.** Of the 3,496
pool problems no stored attempt verified (3,643 sides, 65,789 failed attempts, at most 48 a side), a pool
stood on 1,598 and an assembled proof resolved 92 (76 Lean Workbook, 16 STP; all on the statement's side).
The yield flattens with attempts: 45 after 8 attempts a side, 67 after 12, 79 after 16, 88 after 24, 92 after
48. Minimised, the proofs have a median of 4 lines and the longest 12 (593 pool blocks before, 248 after; 80
of the 92 were made shorter; none had to be used as assembled). 843 closer checks timed out and count as
failures. 92 is under the 100 fixed above as Step 1's first check, so Step 1 as designed is not run: what the
loop can supply of its own from what is stored is too thin for one training to show. That is a finding of
itself. H0 is kept (`data/ladder_l3d/harvest_h0.jsonl`): its proofs are the loop's own and go into Step 2.

**The stored models with assembly (2026-10-08, Lean only, `tools/ladder_goal_assembly.py`; seed 0, the goal
set, 11 episodes of 8 attempts a problem).** Attempts alone, then with assembly. The base: 117 then 164
episodes resolved (48 by an assembled proof), 57 then 77 goal problems solved at least once, 3 then 3
reliably. The loop's three-round model at t = 1/10: 193 then 232 (45), 59 then 77, 15 then 15. Training gives
the reliability and assembly the reach, and today neither moves the other: that is the gap Step 2 tests. (The
ceiling's 8,000-proof model, its stored attempts only: 1,280 then 1,412, 171 by an assembled proof, 242 then
262, 106 then 119: the search finds more with a stronger model.)

**Step 2, made exact before it is built (2026-10-08): the loop with assembly in the round, six rounds, and a
twin trained without the assembled proofs.** It goes ahead on the ceiling's read (a proof of a problem
beyond the model is worth many of its own: `ladder-ceiling-RESULT.md`) and is the enlargement by generation
that Step 1's first check asks for. One seed first, by the seed rule.

- **What it is.** The L2 stage's arm at t = 1/10 again (the same candidates, batches, seeds and sampling
  seeds), with two changes: SIX rounds in place of three, and after each batch's attempts the Lean-only
  assembly (`domain/repair/replay.py`, L3c's rules and sizes) over the 8 attempts, in the order drawn, of
  every side of every problem that none of its attempts resolved. No generation is added.
- **What counts.** A problem an assembled proof resolves is resolved. For the challenger's reward and its
  refit, k stays the number of verified attempts, and a problem with k = 0 that assembly resolved counts as
  k = 1. A proof verified on the side a published answer rules out is the soundness alarm, as everywhere.
- **The training set of a round** is the L2 arm's (one verified proof for each problem resolved in the
  rounds so far, the model trained from the base on all of them), and with it: the minimised assembled proof
  of each problem only assembly resolved (minimised as H0's are: blocks out one at a time from the last, the
  text that is left checked once more, the proof as assembled if that fails), and, from round 1, H0's rows
  for problems the rounds have not resolved themselves (a problem has one proof: the round's own when there
  is one).
- **The twin.** After round six one more model is trained from the base, by the same code and in the same
  seeded order with rows only left out: `without`, on the six rounds' one-shot proofs alone (no assembled
  proof of a round, none of H0). The round-six model is `with`. What differs between them is the assembled
  proofs and nothing else.
- **Measured** after round six, `with` and `without` alike, as the ceiling's and L2's models are: 8 episodes
  on the three rungs, 93 one-shot attempts on each goal problem with L2's sampling seeds. Afterwards, Lean
  only: each one's goal attempts in episodes of 8 with assembly (`tools/ladder_goal_assembly.py`).
- **Primary: the goal problems whose shortest published proof is 4 lines or more (230), successes per attempt
  over 93 one-shot attempts, `with` minus `without`,** paired by problem, 95% bootstrap over problems.
- **Secondary:** the same on all of G and by length group; the three rungs; goal problems solved at least
  once and reliably (at least half of 11 episodes of 8), by attempts alone and with assembly, for `with`,
  `without`, the stored three-round model at t = 1/10 and the base; by round: problems resolved by an
  attempt, problems only assembly resolved, the share of the unresolved it resolved (does the yield grow as
  the model is trained), the mean pass rate of the picks; the lines of the proofs each model verifies and the
  share of its attempts that open with a `have`.
- **Branches.** Interval clear of zero and above: ASSEMBLED PROOFS TEACH, at the rate the loop makes them;
  two more seeds, and the loop with assembly is the loop from here. Interval holds zero: with the ceiling's
  figures beside it (a proof of the first 2,000 is worth about +0.000013 per attempt on this quantity), the
  note says whether a gain of that size for each assembled proof trained on could have been seen; if it
  could, NOT SHOWN (assembled proofs teach less than published ones, each); if not, UNDETECTABLE AT THIS
  SIZE. Interval below zero: THE ASSEMBLED PROOFS COST ONE-SHOT ATTEMPTS.
- **Can this run see a win (else INCONCLUSIVE).** `with` was trained on at least 150 assembled proofs (H0's
  and the rounds') that `without` was not. Both final trainings took (the mean loss over the last tenth of
  the rows below the mean over the first tenth). Both models still write proofs (under 5% of rung attempts
  without an answer). Every assembled row stands once in the record of what `with` was trained on and none in
  `without`'s.
- **Fixed before the run, on the build's two cautions.** (1) The count may fall short on yield alone: H0
  gives at most 92 and the stored rounds assembled 10 to 27 each. If the first check fails by the count and
  nothing else, the verdict is INCONCLUSIVE (execution), nothing is said of the proofs, and the arm is taken
  on two rounds at a time (the twin trained and both measured again) until `with` holds 150; no other
  setting changes. (2) The interval is a bootstrap over problems and does not hold what two trainings differ
  by. So a positive read at one seed is a licence to run the two other seeds and nothing more: ASSEMBLED
  PROOFS TEACH is concluded only on three seeds, each of its own sign and the pooled interval clear of zero.
- **Budget, for the job's estimate:** six rounds of 8,000 generations and their Lean checks, about half as
  many checks again for assembly, six trainings of a set that grows by about 600 rows a round, one more for
  the twin, two measurements of 42,817 generations: about four and a half GPU-hours.

**Made exact by the build (Step 2) (2026-10-08, before any run).**

*The arm (`gpu/ladder_l2.py` with `gpu/ladder_assembly.py`; rules in `domain/ladder_round/assembly.py`).*

- **Where it is configured.** `ladder_loop.l2_assembly_arms.t010_assembly`: the target rate (0.10) and the
  number of rounds (6). `l2_arms` is left as it is, and so are the three-round arms: the same step lists, the
  same files (asserted). The run directory is `ladder_l2_t010_assembly_seed<N>`. Rounds 4, 5 and 6 sample
  with the places after round 3's (1014, 1015, 1016 at seed 0). A smoke run takes its number of rounds from
  `RLVR_LEAN_LADDER_L2_ROUNDS`.
- **No model is measured between the rounds,** so round r needs the training of M(r − 1) and not its
  measurement, the stop rule (which is read on a round's measurement) is not read, and there is no control.
  The stage's measurements, its control and its report refuse to run for this arm.
- **The error positions cost no check.** The round's checker has every rejected attempt's answer in hand: a
  check session now keeps Lean's raw answers, and the episode step of this arm stores, beside a block's
  attempts, the positioned errors of each attempt Lean rejected (`episodes_<set>_errors_<block>.jsonl`). No
  attempt is checked again. A rejected attempt whose errors were not kept is refused, not passed over.
- **What is replayed.** For every problem of the batch with k = 0, each side that was sent to Lean, as an
  episode of its own: its attempts with status `lean_error`, a completion and no forbidden token, each at its
  place in the order drawn (1 to 8). An attempt that reached the token cap, ran into the Lean limit or got
  no verdict has no place in it: there is nothing to cut it by. The loop over an episode's attempts is
  `replay.assemble`, the three calls the harvest and the goal replay make. Sizes are L3c's (12 pool blocks,
  8 kept closers).
- **Assembly's own checks are the solver's:** the round's client settings (its requests in flight, the 30 s
  Lean limit of an episode, no priority of its own). A file whose Lean header timed out is asked once more; a
  check the pool did not take is asked again and, if it is still refused, the step fails with nothing of the
  batch's assembly recorded. Every episode, and then every proof being minimised, advances as its own checks
  come back, so a closing step at the Lean limit holds up its own episode only; the files and their number
  are what one episode after another would send. The batch is marked done only after its assembly.
- **What the checks are expected to number.** The replay of seed 0's stored rounds at t = 1/10 (item 2 above)
  made, over three rounds and 1,315 unresolved problems, 10,693 re-checks, 1,390 pool checks and 3,499 closer
  checks. The re-checks are the ones not made here. With about 7 checks to minimise each assembled proof
  that is about 1,700 checks a round, a fifth of a round's 8,140 attempt checks.
- **What is stored.** `assembly_r<r>_b<b>.jsonl`: one row for every assembled resolution (problem, side,
  round, batch, the attempt it came after, the proof as assembled and as minimised, blocks and lines before
  and after, the checks). The step `ladder_l2_assembly_r<r>_b<b>` records the batch's counts: unresolved
  problems, sides and attempts replayed, pools that stood, resolved by assembly, checks by kind, timeouts.
  `episodes_<set>_problems.jsonl` gets `resolved_by_assembly` beside its counts: true only for a problem
  with k = 0 that an assembled proof resolved.
- **The alarm** is raised before anything of the batch's assembly is recorded, with the proof in its message.
  One problem cannot have an assembled proof on both sides without it.
- **Where k = 1 is read.** `resolved` stays the number of verified attempts in every stored row. The
  challenger reads it through one function (`assembly.counted`): the labels of every refit, the batch's and
  the round's reward, the report's mean reward by round. The picks' mean pass rate in the report is the
  solver's own (verified attempts over attempts) and counts no assembled problem.
- **The training examples of a round** (`training_examples_r<r>.jsonl`) hold, each row with an `id` and its
  `origin`: the round's one-shot rows (`attempt`), the minimised proof of each problem only assembly resolved
  (`assembled`), and H0's rows for the problems rounds 1 to r have not resolved themselves (`h0`). M(r) is
  trained on the `attempt` and `assembled` rows of rounds 1 to r and on round r's `h0` rows: one proof a
  problem. An H0 problem a round proposes and does not resolve keeps its H0 row. H0 is read from the
  snapshot, refused with a held-out or base-map problem, a problem twice or a row that is not an assembled
  proof, and held to the file the run was prepared on.
- **The order of training** for all of this arm's models is a content hash (SHA-256 of
  `<seed>:l3d2_order:<row id>`, the task's seed), not the round's seeded shuffle: in a shuffle, leaving rows
  out moves the rest, and here a row's place among the others does not depend on which others there are.
  The trainings are the ceiling's one pass (the round's recipe, the task's seed for the adapter, every row's
  loss, the training loop's own record of the rows). Stored: `training_set_m<r>.jsonl` (the rows in the order
  trained) and `training_loss_m<r>.json`; the adapter is where every arm's is (`adapters/m<r>`).

*The twin, the measurements and the report (stage `ladder_l3d2`; `gpu/ladder_l3d2.py`; rules in
`domain/ladder_round/l3d2.py`; report in `reporting/ladder_l3d2.py`).*

- **The stage** runs, in the arm's run directory: the arm's prepare step; `ladder_l3d2_prepare`; the
  embedding; `ladder_l2_round_<r>` and `ladder_l2_train_<r>` for r = 1 to 6; `ladder_l3d2_train_without`;
  `ladder_l3d2_measure_with`; `ladder_l3d2_measure_without`; `ladder_l3d2_report`. Every task of the stage
  names the arm, and a step run for another arm is refused. The steps of rounds 4 to 6 are registered by this
  stage: the L2 stage's own table of steps is the three rounds'.
- **`ladder_l3d2_prepare` comes before anything is sampled.** It reads what the ceiling reads (L1's run of
  the seed; `ladder_l2_seed<N>` for the base's control attempts; `ladder_l2_t010_seed<N>` for the stored
  three-round model) and refuses, with the task to run, when one is missing or would not pair. It puts the
  two models' sets (`l3d2_rungs_<model>`, `l3d2_reach_<model>`, `l3d2_more_<model>`) beside the arm's own.
- **`with` is M(6)** (the last round's model, where the arm saved it). **`without`** is trained from the base
  by the same function on the rows of origin `attempt` of M(6)'s training set, in that set's order with the
  other rows left out, with the same seed (`l3d2_training_without.jsonl`, `l3d2_loss_without.json`,
  `adapters/without`). Every adapter is kept.
- **The four checks.** (1) The assembled proofs `with` was trained on and `without` was not are counted from
  the two training records: the rows of origin `assembled` or `h0` that stand in `with`'s record and not in
  `without`'s; at least 150 (`minimum_assembled`). (2) and (3) are Step 1's, on these two trainings and
  models. (4) is Step 1's fourth with "assembled row" for "row of H0": each once in `with`'s record, none in
  `without`'s, and each record the order that was prepared.
- **The branch** is read on the primary alone, as the Branches bullet says: this step names no noise floor.
  An interval that ends at zero holds zero. The note takes `gain_of_a_published_proof` (+0.000013) times
  the count of check (1) as the gain expected, and half the width of the primary's interval as what the run
  resolves: NOT SHOWN when the first reaches the second, UNDETECTABLE AT THIS SIZE when it does not.
- **The report prints the four checks first,** then the primary and the branch, then the secondary reads.
  The table by round gives: the picks; the problems an attempt resolved; the problems only assembly resolved
  and their share of what the attempts left unresolved; the picks' mean pass rate; the mean reward as the
  challenger reads k; the round's training rows by origin; the share of its own rows that are refutations.
  The reliable counts by attempts alone are Step 1's function; the counts with assembly are the replay
  tool's, afterwards.
- **The smoke stage** (`ladder_l3d2_smoke`) is the L2 smoke run's world (the fixtures, `ladder_l1_smoke`,
  four problems a round in two batches) with two rounds, assembly on, the 8-row H0 fixture, and 8 assembled
  proofs asked for in place of 150. No L2 smoke run holds what the real run reads beside its models, so
  none is read: G gets its first sampling only and no stored model stands beside. `with` is M(2).
- **Measured before the run** (no GPU, no Lean): both prepare steps against the pulled copies of seed 0
  accept the stored runs, the whole pool as candidates (51,631; six rounds of 1,000 leave none short) and
  H0's 92 rows.

**Step 2's outcome (seed 0, 2026-10-08; `ladder-l3d2-RESULT.md`): NOT SHOWN, and the run could have seen
it.** The four checks pass (156 assembled proofs between the two models: 98 of the rounds, 58 of H0). Primary,
`with` minus `without` on the 230 goal problems of 4 lines or more: −0.00019 [−0.00122, +0.00061], 2.66
against 2.85 per 1,000; at a published proof's worth 156 proofs would have given +0.0020 and the run resolves
about ±0.0009, so an assembled proof is worth under a third of a published one. Why: minimised, the assembled
proofs are as short as the model's own (median 3 lines; the ceiling's published proofs 8). The yield did not
grow with training (8 to 24 a round, 98 in six). Six rounds stand above the stored three on the goal set
(13.0 and 13.6 successes per 1,000 against 7.2; reliably 19 and 23 against 15), all of it at 1 to 3 lines.
With assembly at inference the six-round model solves 92 goal problems at least once and 26 reliably. No
further seed is run for this read: it is not positive and it is bounded.

### L4: the loop from a model pretrained on published proofs — APPROVED by the owner 2026-10-08 ("3": both as arms)

**The decision.** After the ceiling (`ladder-ceiling-RESULT.md`) and L3d's Step 2 (`ladder-l3d2-RESULT.md`)
the owner was put three options and chose the third: the loop from the base AND the loop from a model
pretrained on published proofs, so that what the loop adds on top of pretraining is itself measured. This is
the fallback the owner named on 2026-10-04. The base arm exists: `ladder_l3d2_seed0` (six rounds at t = 1/10
with assembly). This section is the other arm. Everything it produces is labelled **pretrained on published
proofs**: distillation of other provers, followed by the loop.

**What changes in the rules, and what does not.** A model pretrained on published proofs may now be kept and
used in the rounds of this arm. No held-out problem is in any training text, as ever. The training text
(other people's published proofs) is not exported. The base arm stays what it is.

**The pool is cut in two, by problem, before anything is trained.** A model pretrained on a problem's
published proof and then asked to prove that problem is being asked to remember. So the loop of this arm never
draws a problem whose proof the pretraining saw: each pool candidate goes to one half by a hash of its id
(SHA-256 of `<seed>:l4_half:<problem id>`, even or odd, seed 0), `pretrain` or `loop`. The pretraining file is
made from the `pretrain` half alone; the rounds' candidates are the `loop` half alone. Near variants of a
problem can still fall on both sides, as they do between the pool and the held-out set (the ceiling's note
measures what that is worth).

**The pretraining file** (`data/ladder_l4/pretraining.jsonl`; `tools/ladder_ceiling_set.py` with this
stage's settings; Lean checks only):

- **Eligible and chosen:** every candidate of the `pretrain` half on the side `true` (about 25,000 of the
  50,188; no cut by predicted pass rate: the whole of what is published there, easy and hard).
- **The proof** (changed 2026-10-08 before any run; the first version re-checked up to three published proofs
  a problem, as the ceiling's file does, and at the pool's daytime pace its 56,000 checks would have taken
  most of a day): **the certificate the pool build verified for the problem.** Every pool problem is in the
  pool because one of its published proofs, as published or as renamed, verified under v4.27 when the pool
  was built, and `pool.jsonl` names that proof by the hash of its Lean file. That file is the one a solver's
  attempt is checked in (the same header, theorem, proof and axiom report). So the proof is taken from the
  problem's published proofs by that hash and is not checked again. A problem whose certificate and statement
  together pass 2,400 characters is left out.
- **A control, by Lean:** a seeded sample of 300 rows of the file is checked as a solver's attempt is. At
  least 98% must verify (the ceiling's file found a verified proof for 98.2% of its problems), or the file is
  not used.
- **Written:** one row a problem, in a seeded shuffled order. All of them are trained on.

**The run, seed 0, in two tasks.**

1. **Pretraining** (`ladder_l4_pretrain`). From the base, the round's recipe, one pass over the file in its
   order: the adapter `pre`, KEPT. Measured as every model here is (8 episodes on the rungs, 93 one-shot
   attempts on each goal problem with L2's sampling seeds). **And `pre`'s own map** (added 2026-10-08 before
   any run, on the build's caution): the challenger starts from a map of pass rates, 8 attempts on each of
   the base map's problems, and the map the pool build stored is the BASE's. An arm whose solver is `pre`
   from its first attempt would be aimed by a map of a much weaker model: its early picks would be far too
   easy and the rounds' proofs with them, and a null or a loss could then be the aim's and not the loop's. The
   rule is one rule for both arms, the map of the model the arm starts from. So this task attempts the base
   map's problems again with `pre` (the same problems, sides, 8 attempts and sampling seed as the stored
   map) and the arm's challenger reads that map wherever the base arm's reads the base's.
2. **The loop from it** (`ladder_l4`). The arm of L3d's Step 2 again (t = 1/10, six rounds of 1,000, assembly
   after each batch, a problem only assembly resolves counted as k = 1), with two differences and no other:
   the candidates are the `loop` half, and every model of the arm is trained FROM `pre` in place of from the
   base (round r's model: `pre` trained one more pass on the proofs of rounds 1 to r). No H0 (its proofs came
   from the base arm's history and half its problems are in the `pretrain` half). After round six the twin
   (`without`: from `pre` on the rounds' one-shot proofs alone) and the two measurements, as in Step 2.

**The read, fixed before any run.** Paired by problem, 95% bootstrap over problems, fresh samples on both
sides of every difference.

- **The goal set is drawn again for the stronger model:** G′ is the goal problems (of the 392) that `pre`
  does not solve in its 32-attempt sampling. Its other sampling (61 attempts) is fresh for those problems.
- **Primary: on G′, successes per attempt over the 61 fresh attempts, the round-six model (`with`) minus
  `pre`.** It is the loop's own question asked of the pretrained model: does the loop take it to problems it
  could not solve.
- **Beside it, the base arm's own:** the same quantity for the base arm from stored rows (the goal problems
  the base does not solve in its 32-attempt sampling; the base arm's round-six model minus the base, over the
  61), so the two arms' gains stand side by side.
- **Secondary:** all of G by proof length and the three rungs, for `pre`, `with` and `without`, each against
  the base and `with` against `pre`; goal problems solved at least once and reliably, by attempts alone and
  with assembly; G′ by proof length; `with` minus `without` (do assembled proofs teach at this strength: the
  rounds' count of them, their lines once minimised); by round: resolved by an attempt, only by assembly, the
  picks' mean pass rate; the share of distinct attempts (does pretraining plus rounds narrow the model).
- **Branches.** Primary's interval clear of zero and above: THE LOOP ADDS ON TOP OF PRETRAINING; two more
  seeds of the arm (the pretraining is not repeated: the same `pre`), and this is the loop from here.
  Interval holds zero: NOT SHOWN ON TOP OF PRETRAINING at this size; the note says what the base arm's own
  gain would have looked like here (its size against this interval's half-width). Interval below zero: THE
  ROUNDS COST THE PRETRAINED MODEL on what it could not solve (the rounds' own proofs, easier than what it was
  pretrained on, pull it back): the loop is then not run on top of pretraining as it stands.
- **Can this run see a win (else INCONCLUSIVE).** The pretraining took: `pre` solves at least 150 goal
  problems in its 93 attempts (the ceiling's models solved 209 and 245). G′ holds at least 80 problems. The
  two measured trainings of the arm ran (changed 2026-10-08 before any run, on the build's caution: a model
  that starts from `pre` is trained on proofs it or its like wrote, so its loss need not fall, and "the loss
  fell" in each of seven trainings would fail by noise: in the base arm the first training's fall was 0.028
  against a standard error of 0.027. What the check is for is a training that did not happen or went wrong.
  So, for `with` and for `without`: the adapter's weights differ from `pre`'s, and the mean loss over the last
  tenth of its rows is not above the mean over the first tenth by more than two standard errors of their
  difference. The other five trainings are reported the same way and decide nothing) and each measured
  model still writes proofs (under 5% of rung attempts without an answer). No row of any training of this arm
  is a problem of the `pretrain` half or a held-out problem.
- **One seed first.** A positive read licenses the two other seeds and nothing more, as in Step 2.

**Budget, for the jobs' estimates** (brought up to date 2026-10-08 with the three changes above). The file:
no Lean check, and 300 for its control. The pretraining task: 24,866 rows (two hours at the ceiling's pace),
one measurement and the map (32,464 generations): about three and a half hours. The loop: six rounds with
assembly, six trainings and the twin (each a pass over at most a few thousand rows), two measurements: four
to seven hours, by how much longer the pretrained model's proofs are.

**The file as built (2026-10-08).** All 24,867 true-side candidates of the `pretrain` half matched the
certificate the pool build verified (3,627 of them a renamed one); one was left out for its length. 24,866
rows: 8,155 Lean Workbook, 16,711 STP; proof lines median 7, mean 9.2, 12,420 of 8 lines or more. The control:
300 of 300 sampled rows verified as a solver's attempt.

**Made exact by the build (2026-10-08, before any run).**

*The pretraining file (`tools/ladder_ceiling_set.py --file l4_pretrain`; settings
`ladder_loop.l4.pretraining_file`).*

- **How the certificate is found.** The pool build's own functions form the check, as its plan forms it: a
  problem's stored row of `candidates.jsonl` is read into a candidate (`candidate_from_row`), the renamed
  certificates of `renamed.jsonl` follow its published ones (`certificate_from_entry`), and
  `selection.steps` gives one check a certificate: for a problem on the side `true`, the file
  `certificate_source(certificate, with_fingerprint=True)`, whose SHA-256 is the check's hash. The
  certificate whose check has the hash the problem's row of `pool.jsonl` names is the row's proof. No Lean
  file is written by hand and none is sent.
- **Left out, each counted:** a problem with no such certificate; one whose statement, as L2 reads it, is not
  the theorem that certificate proves (the file the pool build verified held that theorem; a pair that was
  not verified as it stands is not trained on); one whose proof holds a forbidden token; one whose statement
  and proof together pass 2,400 characters. When more than 1% of the chosen problems have no such
  certificate the tool stops with nothing written: the hash is then not rebuilt as the pool build made it.
- **A row** is the ceiling file's fields (no predicted rate), `half`, and the certificate's `certificate_sha`
  and `certificate_renamed`; its `certificate_source` is the matched certificate's. The summary holds what
  matched by source, how many by a renamed certificate, and what each rule left out; no check status.
- **On the real inputs** (a dry run, nothing sent or written): 24,867 eligible, all 24,867 matched (STP
  18,228, Goedel 3,337, InternLM's proofs 3,246, InternLM's rows 56; 3,627 by a renamed certificate); no
  statement differs from the pool row's or from its certificate's theorem; no forbidden token; 1 problem too
  long. 24,866 rows (8,155 Lean Workbook, 16,711 STP); proof lines median 7, mean 9.2, half of them 8 or
  more: the certificate is the first of a problem's proofs that verified when the pool was built, not its
  shortest.
- **The control** (`--control 300`): the sample is seeded (`control.seed`) over the rows of the WRITTEN file,
  in the file's order; each is the file a solver's attempt is checked as
  (`imports_first(build_proof_source(statement, proof))`), sent as background work with a 60 s Lean limit
  and judged as an attempt is. Stored beside the file (`pretraining.control.json`): the file's SHA-256, the
  rows sampled, how many verified, the statuses, and the id and status of every one that did not. It passes
  at 98% or more of the sample and exits non-zero under it. The stage does not read that file: whether the
  pretraining file is used is the operator's to hold to it.

*The pretraining (stage `ladder_l4_pretrain`; `gpu/ladder_l4.py`; rules in `domain/ladder_round/l4.py`; report
in `reporting/ladder_l4.py`).*

- **The stage** is the ceiling's shape with one model, in a run directory of its own
  (`ladder_l4_pretrain_seed<N>`): `ladder_l4_pretrain_prepare`, `_train`, `_measure`, `_map`, `_report`. It reads on
  the box what the ceiling reads (L1's run of the seed; `ladder_l2_seed<N>` for the base's control attempts;
  `ladder_l2_t010_seed<N>` for the stored three-round model) and refuses, with the task to run, when one is
  missing or would not pair.
- **The file is refused,** with nothing written, when it is not in the snapshot, holds no row, or holds a row
  that is a held-out problem (the snapshot's, or the run's), a base-map problem, not of the `pretrain` half
  by `half_of`, not on the side `true`, a problem twice, or longer as a training example than the 2,048
  tokens the recipe cuts at (counted with the model's tokenizer). Recorded: the rows by kind and by the
  length group of their own proofs, their tokens, the file's SHA-256. The run's own copy of the rows holds
  no statement and no proof. The training step refuses a file that changed since.
- **The training** is the ceiling's one pass (`one_pass`): from the base, the file's order, nothing shuffled,
  the task's seed for the adapter, every row's loss stored, ONE adapter saved at the end: `adapters/pre`.
  No step of the stage deletes it.
- **A re-queue during the training starts the pass again from the base.** The pass is one optimizer and one
  schedule, and only the adapter is ever written. A resume from an adapter alone would restart the
  optimizer's moments and the warm-up in the middle of the file, which is not the one pass asked for; an
  exact one needs the optimizer's state written and read back on the GPU, which nothing here can test. Cost:
  up to the whole pass (about two hours). Everything else resumes: the prepare step, a finished training,
  and the measurement block by block.
- **The measurement** is the ceiling's (`measure_model`): 8 episodes on the rungs, G's two samplings with
  L2's seeds (32, then 61), so `pre` pairs by problem with every stored model.
- **`pre`'s own map** (`ladder_l4_pretrain_map`, after the measurement; `l4_map_pre.jsonl`). The base map's
  problems are the snapshot's (`data/ladder_l0/base_map.jsonl`, 4,000), and the stored map is the BASE's rows
  the challenger reads (`data/ladder_l2/base_results.jsonl`, the set `base_map`). The stored map was made by
  `ladder_l0b` with `base_map.episodes` (8) and `base_map.sampling_seed` (101): the step takes both from
  there. Each problem gets its exact negation by the rule that gave the stored map's (L0's, as every round
  uses it: 4,000 exactness checks, the solver's priority). THE SIDES: before anything is sampled, the sides
  the episode step would attempt for each problem (one, or two when the audit rule names it and its negation
  is exact) are held to the `sides` of its stored row, and a problem that differs stops the step with nothing
  sampled. A fixture's stored rows are hand-made and are not held to (the step records that). The episodes
  are a set of the run, resumable by block; the rows written are the stored map's fields, in the base map's
  order. A map on which Lean gave no verdict for more than 2% of its attempts is not kept: the step fails
  and a rerun samples it again. The report prints it beside the base's: problems by k of 8, mean pass rate.
  Checked without Lean: with today's settings and the negations the stored rows imply, the plan is the stored
  map's for all 4,000 problems (3,942 on one side, 58 on two: 32,464 attempts, the stored map's own number).
- **G′ is drawn from the FIRST sampling** (the 32 attempts with L1's seed for G): the goal problems with no
  verified attempt there, in the goal set's order. Its ids are stored (`l4_goal_set_again.jsonl`, and in the
  report) and the arm reads them from there: it does not draw them again.
- **The two checks are read here,** on `pre`: it solves at least 150 goal problems in all its 93 attempts;
  G′ holds at least 80. A FAIL does not fail the task: the report is written and says so, and the arm's own
  prepare step then refuses to run. The training's loss (first tenth of the rows against the last) and the
  share of `pre`'s rung attempts without an answer are printed beside them for information.
- **Beside `pre`** stand the base, the stored three-round model, and the ceiling's two models when the
  ceiling's run of the seed is on the box (`ladder_ceiling_seed<N>`: its report readable, its six sets
  sampled as this run samples); they are read as stored models are, from their rows and attempts. That run
  is not needed: when it is absent or would not pair, it is not read, the reason is recorded, and nothing is
  refused.
- **Every printed line** of both stages begins `l4 (pretrained on published proofs):`, and every stored
  summary and report carries the label.
- **The smoke stage** (`ladder_l4_pretrain_smoke`) reads `ladder_l1_smoke` alone, with a 12-row fixture in
  the file's place (`data/ladder_l4_fixture`: rows of the ceiling's fixture that fall in the `pretrain`
  half; published proofs, never exported) and both minimums at zero, so that the arm's smoke run can stand
  on it. Its data are the L2 smoke run's fixtures, so its map is of the fixture's four base-map problems:
  the ones the arm's smoke run reads.

*The arm (stage `ladder_l4`; `gpu/ladder_l2.py`, `gpu/ladder_assembly.py` and `gpu/ladder_l3d2.py` as for
Step 2, with `gpu/ladder_l4.py`).*

- **Where it is configured.** `ladder_loop.l2_assembly_arms.t010_assembly_pre`: the base arm's target rate
  and rounds, and four settings the base arm does not have: `start: pre`, `candidates: loop_half`,
  `h0: false`, `map: start_model`. The base arm's entry, step lists, files and summaries are what they were
  (asserted). The run directory is `ladder_l2_t010_assembly_pre_seed<N>`.
- **The stage is Step 2's steps again,** in the arm's run directory, with two of its own:
  `ladder_l4_prepare` right after the arm's prepare step and before anything is sampled, and
  `ladder_l4_report` in the place of Step 2's report. The sets and files Step 2's steps write keep their
  names there (`l3d2_rungs_with`, `l3d2_training_without.jsonl`, and so on); this arm's own are `l4_...`.
- **`ladder_l4_prepare` refuses,** with nothing written: a pretraining run of the seed whose report, file of
  G′, measurement or map is not there; one whose report is not to be read or whose two checks failed; and, on
  the GPU, a missing `adapters/pre`. It copies into the arm's run what the report reads of that run (G′,
  `pre`'s per-problem rows and what it wrote).
- **The candidates are the `loop` half and nothing else knows another:** the embeddings (the base map's
  statements and that half's, under a key of their own), the scores, the proposals. The arm's prepare step
  refuses a candidate of the other half and a half too small for the arm's rounds; every batch's proposal
  refuses one again before any fit. Of the pool's 51,631 candidates the arm keeps 26,029 (25,321 true, 708
  false); six rounds use 6,000.
- **Round 1 is attempted by `pre`:** the engine serves the stored adapter beside the base (a request id of
  its own, 100; a round's own models have 1 to 6). Rounds 2 to 6 are attempted by M(1) to M(5) as in every
  arm.
- **Every model is trained from `pre`:** a fresh adapter is attached as for every training, `pre`'s weights
  are loaded into it, and the pass goes on from there with the round's recipe (a new optimizer, the same
  warm-up). M(r) is `pre` and one pass over the rows of rounds 1 to r; it does not start from M(r − 1). The
  twin is `pre` and one pass over the one-shot rows of M(6)'s set. Before its first step a training
  compares the B matrices the model holds with the file's (their absolute sum; a fresh adapter's is zero)
  and refuses when they differ; both sums are recorded. After its pass the adapter it SAVED is compared with
  `pre`'s, file against file: how many numbers there are and how many are not the same
  (`against_the_start_adapter`, in the training's summary).
- **No H0:** the arm reads no harvest file, and its training rows are of origin `attempt` or `assembled`.
- **The challenger starts from `pre`'s own map** (`map: start_model`; the spec's "And `pre`'s own map"). The
  arm's data are read through one function, and it puts the rows of `l4_map_pre.jsonl` in the place of the
  base map's stored rows: the first fit and every refit read them from there, with the weights the base map
  has in every arm (as old as round 1). The arm's OWN prepare step reads the map before anything else and
  refuses, naming the task to run, when it is not there, when the step that made it recorded another
  sampling seed or number of attempts than `base_map`'s, or when it is not exactly the base map's problems
  with that number of attempts each. An arm that asks for the map and names no start is a configuration
  error. The base arm reads the base's stored map as it did (asserted on every one of its fits).

*The read (`reporting/ladder_l4.py`).*

- **The checks, in this order.** (1) and (2) are the pretraining report's, carried over as it read them.
  (3) The two measured trainings ran (below). (4) `pre`, `with` and `without`: under 5% of each one's rung
  attempts without an answer. (5) No row of any of the seven trainings is a problem of the `pretrain` half
  or a held-out problem, read from the training loop's own record of the rows.
- **The two measured trainings ran** (`with`'s, M(6), and `without`'s; both must pass). *The adapter differs
  from `pre`'s:* at the end of a training the file of the adapter it saved and the file of `pre` are compared
  BIT FOR BIT, over the tensors both hold under one name with one type and one shape
  (`infrastructure/adapter_files.py`, which reads the two files and needs no model; tested on hand-written
  files); the check passes when at least one number is not the same. It is a count of unequal numbers between
  two stored files, so it cannot pass or fail by rounding: the sums of the B matrices, which a training
  records rounded, are not what it reads; a tensor held with another type or shape, whose numbers could
  differ by their rounding alone, is not compared and counts for nothing; and an adapter that IS `pre`'s
  gives zero. A training that recorded no comparison (the stand-in trains nothing) does not pass.
  *The loss did not rise:* the first and the last tenth of the training's rows (a tenth rounded down, at
  least one row; a row's loss is its mean loss per target token, read before the update of its step); the
  STANDARD ERROR of the difference of the two means is the square root of the first tenth's sample variance
  over its number of rows plus the last tenth's over its own (the two tenths hold different rows, taken as
  independent); the check passes when the last mean is not above the first by more than 2 of them
  (`standard_errors_allowed`). A tenth of one row has no variance, and nothing is then allowed. No fall is
  asked for. The other five trainings are read by the same function, printed on a line of their own and kept
  in the report beside the checks (`the_other_trainings`); they decide nothing.
- **The primary** is read on G′ over the SECOND sampling alone (the 61 attempts with the seed of L2's
  control): the first chose G′. `with` minus `pre`, paired by problem, bootstrap over the problems of G′.
- **Beside it,** read only when the base arm's run of the seed is on the box
  (`ladder_l2_t010_assembly_seed<N>`): the base's own G′ (the goal problems the base does not solve in its
  32 attempts), and over the same 61 attempts the base arm's last model minus the base. From the stored
  rows at seed 0 this is +0.0066 per attempt [+0.0031, +0.0108] on 355 problems.
- **The note** beside an interval that holds zero: the size of the base arm's gain against half the width
  of the primary's interval ("would have been seen here" when it reaches it). An interval that ends at zero
  holds zero.
- **Secondary.** All of G over the 93 attempts by proof length, and the three rungs, for `pre`, `with` and
  `without` against the base, `with` against `pre`, `with` against `without`, and the stored three-round
  model; solved at least once and reliably by attempts alone for each; G′ by proof length; `with` minus
  `without` on G′ with the count of assembled proofs trained on and the lines of the rounds' assembled
  proofs once minimised; the table by round; the share of distinct attempts. The counts with assembly are
  the replay tool's, after the run, as in Step 2.
- **The smoke stage** (`ladder_l4_smoke`) is Step 2's smoke world from the pretraining smoke run's adapter:
  two rounds of two problems in two batches of one (five of the fixture's twelve candidates are of the
  `loop` half). It has one sampling of G, so its primary is not read.
- **Measured before the run** (no GPU, no Lean): both stages' prepare steps against the pulled copies of
  seed 0 accept the stored runs, the ceiling's two models and the base arm's rows (the arm's with a stand-in
  map of the 4,000 base-map problems in the pretraining run's place: the arm's data then hold those 4,000
  rows and no row of the base's map). The pretraining's prepare step accepts the COMMITTED file: 24,866
  rows, 3,109 optimizer steps, the SHA-256 its control checked (the tokens were counted by the stand-in
  there; the model's tokenizer counts them on the box). With the ceiling's
  8,000-proof model standing in for `pre`, the two checks pass (245 goal problems solved; G′ of 190, of
  which 125 with a published proof of 4 lines or more), and an interval on G′ is about 0.002 per attempt
  wide each way, so a gain of the base arm's size would be seen. The loss check on the base arm's seven
  stored trainings: the standard error is 0.028 at M(1) (68 rows a tenth) and 0.010 at M(6) and the twin
  (336 and 320), so what is allowed at the two measured trainings is a rise of about 0.02 on a loss near 0.2;
  all seven fell (by 0.028 to 0.063).

## Fixtures (these become the tests)

1. An episode that proves the statement resolves the problem; so does one that proves the exact negation; one
   that proves neither does not.
2. A statement with no certificate checked under the current pin is never sent to the solver and never enters
   a training set.
3. A problem published both ways is not in the pool. A known-false problem whose exact negation cannot be built
   or has no certificate is not in the pool.
4. The challenger's reward for k of n at target rate t is `p (1 − p)^a / [t (1 − t)^a]` with p = k/n and
   a = 1/t − 1: 0 at k = 0 and at k = n; 1 at p = t; at n = 8 and t = 1/4 it is 0, 0.79, 1.00, 0.87, 0.59, 0.31,
   0.11, 0.02, 0 for k = 0 to 8 (to two places); at t = 1/2 it is 4p(1 − p), equal for k and n − k. The band
   computed from t = 1/4 is 0.13 to 0.41.
5. Every proposed problem with a valid certificate gets its n episodes; none is dropped or kept by a pass-rate
   estimate first.
6. No problem of H, and none of the 693, appears in the pool, a training set or the challenger's data.
7. A published proof never appears as a training target.
8. The training target is drawn from the problem's verified proofs with the round's seed; the same seed gives
   the same target; the shortest proof has no special standing.
9. An episode stops at the loop cap, the token cap or the Lean time cap, and a capped episode is a failure.
10. A verified proof on the side the certificate contradicts stops the round.
11. The stop rule fires on an interval entirely below zero and not on one that contains zero.
12. Gain by k: the trained model and the model the round started from attempt the same problems, each with
    fresh samples and the same number of episodes; the change is paired by problem and grouped by the k of the
    round; no episode that placed a problem is used on either side.
13. A rung holds only problems with at least one success in the base's placing episodes, and its edges are
    computed from t.

## Known limits

- STP's conjectures were written from Lean Workbook statements, ours included, and the release does not say
  which statement each came from. A conjecture close to a problem of G can be in the pool. The duplicate check
  removes exact and same-shape copies only; reach is also reported on the problems of G with no near neighbour
  in the training sets.
- STP's "statement" rows include a few hundred miniF2F-style names. Any miniF2F evaluation must exclude them;
  the pool does not use them.
- Which Lean version each published proof was made with is not checked here. L0's certificate check under our
  pin is the test.

## Decided by the owner (2026-10-04)

1. **The challenger chooses from the pool and does not write new statements, until the novel stage.** Yes. The
   pool includes STP's 703,996 conjectures, written by another lab's model, each with a published proof.
2. **Published proofs are certificates only, never training targets.** Yes, "though we might want to pretrain
   on some of these later if we can't make progress, see if it helps against a holdout." That is the later
   option under Behavior; it is not part of L0 to L2.
3. **The negation is the exact one.** Yes. The 958 published disproofs with hypotheses stay out of the pool
   until their exact negation has a certificate.
4. **Rewrite `∑ x in s` to `∑ x ∈ s` for v4.27.** Yes, with rewritten statements marked.
5. **The target rate t is 1/4.** "1/4 is good." It is the owner's shape (k = 1 to 3 of 8 above k = 4) and what
   the 13b lead and the flat pass@32 point to; it costs 7% of proposals that yield no proof. The alternatives
   were 1/3, the peak of the per-step model (k = 2 and 3 on top, k = 1 below k = 4; 3% yield none), and 1/2,
   the earlier draft. L1's gain-by-k table is what would change it for L2, by a spec edit.

## Open questions (the owner's)

None.
