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
stated, because the pool's 718 candidates predicted in the band are used up in round 1. **No further rounds
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
2. **The state.** The kept prefix, then `sorry` at the cut line's indentation, checked by Lean. The state is
   the goal Lean reports at that `sorry`, provided no error lies before it. If none can be had (a cut inside
   a structure the prefix does not close, a timeout), that loop is a blind attempt and is counted as
   "no state": never dropped.
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
