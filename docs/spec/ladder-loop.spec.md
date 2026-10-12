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

**Further seeds of the arm, made exact before they run (2026-10-09).** The branch for a positive read is
"two more seeds of the arm (the pretraining is not repeated: the same `pre`)". What follows fixes, from the
code as it is, what a seed of the arm changes and what it does not, for seeds 1 and 2. It adds no branch and
no threshold, and the two tasks are queued only on that branch.

- **One pretraining for every seed of the arm** (`ladder_loop.l4.pretraining_seed`: 0, read in one function,
  `ladder_l2.pretraining_seed`). The arm of seed N reads the pretraining run of the SETTING's seed,
  `ladder_l4_pretrain_seed0`, wherever seed 0's arm reads it, and writes nothing there: the adapter `pre`
  (round 1's attempts, and the start of all seven trainings); `pre`'s own map (the challenger's first fit and
  every refit; it was made with `base_map.sampling_seed`, which no task's seed moves); G′
  (`l4_goal_set_again.jsonl`, 210 problems: ONE G′, so every seed's primary is read on the same problems);
  the pretraining report's two checks, carried into every seed's report as read; and `pre`'s stored
  per-problem rows on the rungs and on G's two samplings, which are the `pre` side of every difference at
  every seed (8, 32 and 61 attempts, sampled with seed 0's sampling seeds 1002, 1001 and 1020). A task that
  names another run (the smoke run) still reads that one. The pretraining stage is not changed: its run
  directory is its own task's seed's, and there is no job for a seed other than 0.
- **The arm's prepare steps refuse as at seed 0,** before anything is sampled and naming the pretraining
  task of seed 0, never one of seed N: a pretraining run whose map, report, file of G′, measurement or
  adapter is missing, or whose checks failed. **One refusal more:** a pretraining run whose prepare step
  recorded another seed than `pretraining_seed`.
- **What the seed moves:** everything the arm's steps draw with the task's seed N.
  - The rounds' attempts: sampling seeds 1000 + 100 N + 11 to 16 (seed 0: 1011 to 1016; seed 1: 1111 to
    1116; seed 2: 1211 to 1216).
  - The challenger's proposals: the folds of its fit and the random tenth of each batch's places are seeded
    by N, and from the second batch on its refits read what that seed's attempts gave. Round 1's first fit is
    on the same map at every seed.
  - The rounds' training proofs: which of a problem's verified attempts is its training proof.
  - The seven trainings (M(1) to M(6) and the twin): the order of the rows, a content hash of N and each
    row's id. The adapter seed is N too; by the code it moves nothing here, since it initialises the fresh
    adapter that `pre`'s weights are then loaded over and dropout is 0 (read from the code, not measured).
  - The two measurements, of `with` and of `without`: the rungs with sampling seed 1000 + 100 N + 2, G's
    first sampling (32) with + 1, its second (61) with + 20: 1102, 1101 and 1120 at seed 1; 1202, 1201 and
    1220 at seed 2. They are the seeds of that seed's L1 run (`ladder_l1_seed<N>`) and of its L2 control
    (`ladder_l2_seed<N>`), which the arm reads with the stored three-round model (`ladder_l2_t010_seed<N>`):
    the base and that model stand beside `with` and `without` from seed N's own stored rows. All three ran at
    seeds 0, 1 and 2. The 2% of problems whose ruled-out side is still attempted also follow the sampling
    seed; that side resolves nothing, so no count moves with it.
- **What the seed does not move.** Checked on the pulled copies of the three seeds: the held-out groups, G's
  392 problems and the rungs' 712, the same and in the same order in the L1 runs of seeds 0, 1 and 2 (L0
  placed them); and the second sampling's size, 61 attempts a goal problem at each seed. By the code: the
  candidates (the `loop` half) and their embeddings, which seed 0's arm made (the key holds no seed).
- **The primary of seed N is still paired by problem, on fresh attempts.** `with` of seed N over the 61
  attempts of its second sampling, minus `pre` over its own 61, problem by problem of G′. `pre`'s FIRST
  sampling alone chose G′: nothing `with` sampled, at any seed, chose it, and `pre`'s second sampling is as
  fresh as it was. What seed 0 has and seeds 1 and 2 do not is one sampling seed on both sides: there the
  two sides share the problems and not the random numbers. And the `pre` side is the same 61 attempts in
  every seed's difference: three arms against ONE measurement of `pre` (open question 1).
- **Beside the primary at seeds 1 and 2: nothing.** The base arm ran at seed 0 only
  (`ladder_l2_t010_assembly_seed0`). The arm of seed N looks for the base arm's run of seed N and does not
  find it: the report says so where the base arm's gain would stand, nothing is refused, and no check reads
  it. The base arm's own gain is what seed 0's report reads.
- **A further seed's report says what it stands on:** one line after the checks' lines and one entry
  (`a_further_seed`): the pretraining's seed and run, what was read from it, and this seed's sampling seeds
  beside `pre`'s. Seed 0's report has neither, and the arm at seed 0 writes what it wrote (asserted).
- **The read over the three seeds, fixed now** (`reporting/ladder_l4.py`, `over_seeds`; the command
  `python -m rlvr_lean.tools.ladder_l4_seeds` on the three pulled task directories; no GPU, no Lean). Per
  seed: the primary as that seed's report reads it, computed again from the stored rows and held to the
  report's figure. Pooled: for each problem of G′ the mean over the seeds of its difference, 95% bootstrap
  over the problems of G′ (the form of the three-seed reads of L1b, L2 and L2t), by the function the
  primary is read with, on each problem's attempts added over the seeds (three times 61 of `with`, and
  `pre`'s 61 once for each seed: with the same number of attempts at every seed that is the mean of the
  three differences). Beside it: each seed's sign, and how many seeds have their own interval clear of
  zero, above and below. Refused: seeds that do not hold one G′ or one `pre`. A seed whose report is
  INCONCLUSIVE, is not to be read or has no primary is named, its primary is not given, and nothing is
  pooled. No verdict is named from these figures: the branches above stand (open question 2).
- **Budget:** each further seed is one task of the arm's size (four to seven hours); nothing is pretrained,
  mapped or embedded again.
- **Made exact by the build, and measured before any run** (2026-10-09; no GPU, no Lean, nothing queued).
  The setting is read by `ladder_l2.pretraining_seed` alone; `ladder_l2.pretraining_run` names the run an arm
  reads and `ladder_l4.pretrain_directory` stays the pretraining stage's own. A further seed's own prepare
  step records `a_further_seed` (the two seeds, `pre`'s sampling seeds, the base arm's run it looked for);
  the report's line and entry are made from it, and a seed that is not a further one records and prints what
  it did. The command takes each seed's pulled task directory (or its `steps` directory) and reads four
  files of it: the report, G′, `with`'s rows and `pre`'s rows on the second sampling. Holding a seed's
  primary to its report's figure, interval included, refuses no honest run: four intervals the box wrote
  (`pre` minus the base on all of G and on the three rungs) come out the same to the last place when
  computed again here from the pulled rows.
  *Seed 0 is what it was:* in the scripted world, the arm's stage at seed 0 (two rounds, the base arm's run
  beside) writes the same 119 files with the code before the build and after it, and the pretraining the
  same 41; on the pulled copies of seed 0, the arm's three prepare steps write the same 22 files (paths and
  wall-clock figures aside). *Seed 2, on the pulled copies:* the three prepare steps run to their end from
  seed 0's pretraining run: 26,029 candidates, the map's 4,000 problems (mean pass rate 0.237), G′ of 210
  problems all within that seed's G, `pre`'s 61 attempts on the same 392 problems as that seed's base, the
  sampling seeds 1202, 1201 and 1220 (rounds 1211 to 1216); seed 0's base arm's run is in the store and is
  not read; no pretraining directory of seed 2 is looked for or made; G′, `pre`'s rows, the held-out groups
  and the measured problems are byte for byte seed 0's. *Seed 1:* the arm looks for the pretraining at
  `ladder_l4_pretrain_seed0` and reads and checks its map; its own prepare step could not be run to its end
  here, because the pulled copy of L1's seed-1 run lacks its two measurement markers and the base's attempts
  (they were on the box when the L2 tasks of seed 1 read them; not confirmed since). Not shown without the
  box: the adapter `pre` served and loaded at a further seed (seed 0's run does both), and the embeddings
  found again.

### L4r: is the pretrained model capped by the size of its adapter? (a labelled check of the pretraining; fixed 2026-10-09 before any run)

**Under the owner's word of 2026-10-08 (night): "keep iterating as much as you can on the RLVR towards getting
it to solve never before solved math problems and at more reliable rates, without overfitting."** Everything
this stage makes is labelled **pretrained on published proofs**, as L4's is; it trains on L4's pretraining file
and on nothing else (the `pretrain` half of the pool: no held-out problem, no base-map problem, no problem of
the `loop` half), and that file is never exported.

**Why.** `pre` (24,866 published proofs, one pass) reaches what the ceiling's 8,000-proof model reached and no
more (66.7 against 64.7 successes per 1,000 attempts on G; 238 goal problems against 245;
`ladder-l4-RESULT.md`). Read from `pre`'s stored training record: the loss on rows it had not yet trained on
(one pass, each row read before any update that saw it) fell from 0.266 over the first twentieth of the pass
to 0.225 over the second, was 0.195 at the middle and 0.189 at the end: flat over the last two fifths. Every
model of this project is one adapter of rank 16 on the 7B model (`lora.rank: 16`, alpha 32, 37.5 million
trained numbers). One cause that fits all of it is that the adapter is full: then "more published proofs buy
no more" is a fact about the adapter and not about the model or the data, and the arm's rounds, which go on
training that same adapter, compete for room that is not there. That is a cause of the kind this repo rules
out before a mechanism is judged (the run could not have shown the effect), and it costs one training to
test. It also sizes the larger run the owner has named as the step after this one: an adapter, or the whole
model.

**What runs. ONE change.** From the base, one pass over the same file in the same order, the same seed,
learning rate (1e-4), batch (8), sequence limit and target modules as `pre`; the adapter's rank is 64 and its
alpha 128 (the ratio alpha / rank stays 2, so the adapter's scale is `pre`'s). The model is `pre_r64`, in a
run directory of its own (`ladder_l4_pretrain_r64_seed<N>`); `pre`'s run is only read. It is measured as `pre`
was, with `pre`'s sampling seeds: 8 episodes on the three rungs, and G twice (32 and 61 one-shot attempts),
so that it pairs by problem with `pre`'s stored rows. No map is made (a map is made only for a model an arm
will start from). The model server's largest adapter rank is raised to 64 for this stage alone. If the box
cannot hold rank 64 at the sequence limit (an out-of-memory failure in the first steps), the run is VOID and
rank 32 with alpha 64 is run in its place, under the same read; the note says which ran.

**The read, fixed before the run.**
- **Checks first (any FAIL: INCONCLUSIVE, nothing else is said).** The training took: the mean loss over the
  last tenth of its rows is below the mean over the first tenth. It still writes proofs: under 5% of its rung
  attempts without an answer. Lean answered: at most 2% of each measured set without an answer (a set over
  that is sampled again by queueing the task again, as in the ceiling).
- **Primary: all of G (392 problems), successes per attempt over the 93 attempts, `pre_r64` minus `pre`,
  paired by problem, 95% bootstrap over problems.**
- **Secondary:** the same by proof length (1, 2-3, 4-7, 8 or more lines; 4 or more); goal problems solved in 93
  attempts (gained, lost, sign test); solved in at least one episode of 8 and reliably; the three rungs; the
  loss on rows not yet trained on by twentieth of the pass, beside `pre`'s; the share of attempts that time
  out in Lean and the share of distinct attempts; how many of the goal problems that nothing stored has ever
  solved (81 on 2026-10-09) it solves.
- **Branches.** Interval clear of zero and above: THE ADAPTER'S SIZE CAPPED THE PRETRAINED MODEL. The arm is
  then run again from `pre_r64` (its own map, the same halves, every model of the arm at that rank) before
  more seeds are bought at rank 16, and every result of this project that reads "not shown" at rank 16 is
  marked as read under that cap. Interval holds zero: NOT THE CAP AT THIS SIZE; rank 16 stays, and the note
  gives the interval's half-width as the size this run could have seen. Interval below zero: the larger
  adapter is worse after one pass at this learning rate; rank 16 stays and nothing is concluded about a
  larger adapter trained longer.
- **What it cannot say.** Rank 64 is not the whole model: a null does not show that full training would not
  help. One learning rate and one pass: an adapter four times the size may want either changed, and this run
  changes neither. One seed, a first run by the seed rule; a positive read is confirmed by what is built on
  it (the arm from `pre_r64`), not by a second pretraining.
- **Budget.** The training is `pre`'s 3,109 steps (112 minutes at rank 16; the larger adapter adds little to
  a step); the measurement 46 to 55 minutes as `pre`'s was: about three hours on the box, 42,817 generations
  and as many Lean checks. Memory, corrected by the build's arithmetic before any run (the figures first
  written here, 10.5 GiB and 13 GiB, were both wrong): `pre` peaked at 9.78 GiB allocated and a training is
  capped at 11.94 GiB (3 GiB left for the desktop and half a GiB of overhead); rank 64 is expected to peak
  near 11.45 GiB, 96% of the cap, so it may not fit. That is the case the fallback above is for: an
  out-of-memory failure comes within the first 21 optimizer steps (the longest row is row 164), and rank 32
  (about 10.3 GiB, 87%) is then queued.
- **The adapter** `pre_r64` is kept on the box until the read: it is the arm's starting model if the first
  branch is taken, and it is deleted otherwise.
- **Made exact by the build, and measured before any run** (2026-10-09; no GPU, no Lean, nothing queued;
  `gpu/ladder_l4_rank.py`, `domain/ladder_round/l4_rank.py`, `reporting/ladder_l4_rank.py`). The read and
  the branches above are not changed.
  *The stages.* `ladder_l4_rank` runs the check at the setting's rank (`ladder_loop.l4.rank_check`: 64 with
  alpha 128; the model `pre_r64`, the run `ladder_l4_pretrain_r64_seed<N>`). `ladder_l4_rank_fallback` runs
  the smaller one (32 with 64; `pre_r32`, `ladder_l4_pretrain_r32_seed<N>`) and is queued only after the
  first failed out of memory. `ladder_l4_rank_smoke` runs the first, at the real rank, on the pretraining
  smoke run's world. A stage names its check by a variable; the setting has one reader; the pretraining's
  three steps became functions both stages call, and the pretraining stage reads neither the variable nor
  the setting. *`pre`'s stage is what it was:* in the scripted world the pretraining writes the same 41
  files (40 with the stand-in, which saves no adapter) and the arm the same 117 with the code before the
  build and after it, key order included (paths and wall-clock figures aside), and the pretraining the same
  again when its task carries the check's variable. No step deletes the check's adapter.
  *The same seed.* The check is trained with its task's seed, and a task whose seed is not
  `ladder_loop.l4.pretraining_seed` is refused: "the same seed" is `pre`'s.
  *What of `pre`'s run is read,* each refused when missing, naming the task to run, before anything is
  written: its report (one that is not to be read is refused); the markers of its prepare step (the seed it
  recorded; what the one change is held against) and of its measure step (what `pre` wrote); its training
  record (`l4_pretrain_loss.json`, one loss for each row of this file); and for each set it was measured on
  its per-problem rows and that set's own marker. The rows, what `pre` wrote and its losses are copied into
  the check's run by the prepare step, and the report reads those copies.
  *The one change is held, not assumed.* The prepare step compares what it is about to record with what
  `pre`'s prepare step recorded and refuses, naming each, anything but the adapter's rank and alpha that is
  not the same: the file's SHA-256, its rows and their order, the seed, the batch, the steps, the sequence
  limit, the learning rate, the warm-up, the dropout, the target modules. It refuses a rank that is not
  above `pre`'s and an alpha / rank that is not `pre`'s. `pre`'s stored sets must also pair with what the
  run will measure: the sampling seeds and attempts `pre`'s prepare step recorded, and each set held, as the
  ceiling holds a stored run's, to its seed, its attempts, exactly this run's problems, and Lean having
  answered.
  *The adapter saved is read back* before the training step is marked done: the rank and alpha PEFT wrote
  beside the weights (`adapter_config.json`) and the rank of every matrix in the weights file must be the
  check's, or the step fails and nothing of the run is measured (a run the change did not reach would read
  NOT THE CAP for no reason). The count of its numbers and the most memory the allocator held are recorded.
  *Out of memory.* An out-of-memory error of the training (PyTorch's own, or a CUDA error that says so)
  fails the step and names the fallback's job (`ladder_l4_rank32_seed0.json`: a job config, not shipped in this
  copy; this copy's code names the fallback's stage and its command in its place); nothing smaller is
  tried inside the task and the VOID run's directory is left as it is. The fallback's report says in its
  first line that it is the fallback, and for which rank. A fallback out of memory says that no smaller rank
  is configured.
  *The checks.* The tenth is `l4.loss_share_of_rows` (rounded down, at least one row); "under 5%" is
  `l4.maximum_share_without_an_answer`; "Lean answered" is read on the six sets the report reads (the
  check's three and `pre`'s stored three), and a set at exactly 2% passes. A failing third check also makes
  the report NOT TO BE READ: its step fails, and the task queued again samples that set of the check's model
  again from the kept adapter.
  *The read.* The primary is read by the function the ceiling's and L4's reports read successes per attempt
  with (`per_attempt`), on all of G over every attempt of both samplings. An interval that ends at zero
  holds zero. A world with no goal problem reads NOT READ (a case the read did not name). A twentieth of the
  pass is rows `n i // 20` up to `n (i + 1) // 20`: on `pre`'s stored record this gives the figures above
  (0.266, 0.225, 0.195 at the tenth twentieth, 0.189 at the last). Time-outs are a model's attempts that
  timed out in Lean over its attempts, on the rungs and on G. The goal problems nothing stored has ever
  solved are GIVEN and never computed by the stage: a file `l4_rank_never_solved.jsonl` in the run
  directory, one `problem_id` a row. The report says how many of them are goal problems of the run, how many
  the check's model solves in its 93 attempts and, as a check of the list, how many `pre` does (0 for a list
  that is what it says); with no file it says "not given". The report step can be run again on a pulled
  copy with that file beside (no GPU, no Lean). Not in the read above and not reported: the base, the stored
  three-round model and the ceiling's models (the prepare step, being the pretraining's, still stores their
  rows in the run).
  *Memory, by arithmetic; the Budget bullet's figures are round, and these replace them.* The adapter holds
  2,342,400 numbers a unit of rank (30 layers; four matrices of 4,096 + 4,096 and three of 4,096 + 11,008;
  the box counted 37,478,400 in a rank-16 file): 149,913,600 at rank 64 and 74,956,800 at rank 32. In
  training each is held four times as a 32-bit float (the weight, its gradient, AdamW's two moments): 1.68
  GiB more than `pre` at rank 64, 0.56 GiB at rank 32. `pre`'s 10.5 is GB of 10^9 bytes ALLOCATED, 9.78 GiB.
  The training's allocator is capped at 0.773 of the card's 15.45 GiB: 11.94 GiB, not 13 (the desktop's 3
  GiB and 0.5 GiB of process overhead are taken off, `pipeline._cap_torch_memory`; the log of `pre`'s task
  prints that fraction). Rank 64 is so expected to peak at 11.45 GiB allocated: 0.49 GiB under the cap, 96%
  of it, BEFORE the allocator's own fragmentation, which the cap counts and `pre`'s run did not record. The
  card's own figure for `pre`'s training (12,329 MiB at its peak, 970 MiB held by the desktop when the task
  began) puts that process 1.31 GiB above what it had allocated, overhead and fragmentation together; with
  the same excess rank 64 would stand 0.33 GiB over what the cap and its overhead allow, unless the
  allocator gives back what it caches, which it tries before it fails. RANK 64 MAY NOT FIT, and that is not
  known before it is tried. The file's longest row (1,536 tokens) is row 164, in optimizer step 21: a run
  the box cannot hold fails in its first minutes, as the fallback rule supposes. Rank 32: 10.34 GiB, 87% of
  the cap, and 0.79 GiB to spare by the card's figure. The model server at rank 64 holds adapter buffers
  four times as large, taken from a sampling cache of about 2.4 GiB (5,248 tokens at rank 16) that must
  keep 1.88 GiB for one 4,096-token sequence: not measured; the smoke run starts that engine once.
  *Measured before the run, on the pulled copies of seed 0* (four markers of `pre`'s first task, which the
  pulled copy of its second does not hold, were made again from that copy's own step results; whether the
  box still holds the originals is not verified from here: the prepare step refuses, naming them, if it
  does not). The prepare step runs to its end for both ranks: 24,866 rows, 3,109 steps, the committed
  file's SHA-256 the one `pre` recorded. What it records is what the box recorded for `pre` in every entry
  but the paths, the tokens (counted by the stand-in here), the recipe's rank and alpha and each stage's
  own entry; the 712 + 392 + 392 problems it will measure are `pre`'s own with their negations, in order.
  The stage's report step, with `pre`'s own rows standing in for `pre_r64`'s: +0.00000 [+0.00000,
  +0.00000], NOT THE CAP AT THIS SIZE, and on the `pre` side the result note's figures (66.68 successes per
  1,000 attempts; 238 goal problems solved; 234 in an episode of 8 and 94 reliably; 3.6% of its goal
  attempts timed out in Lean). The smoke stage's prepare step, with the stage's own variables, runs to its
  end on the pulled copies of the two smoke runs it reads (12 rows, 2 steps, one sampling of G). Not shown
  without the box: a rank-64 adapter of this model trained from this config, saved by PEFT in the two files
  the read-back reads, and served by a model server started for rank 64; and the memory of the real
  training. The smoke run is the first three of these.

### L4t: what should a round train on? Three one-change checks on the rounds already made (fixed 2026-10-09 before any run)

**Under the owner's word of 2026-10-08 (night)** to keep iterating toward solving never-solved problems, more
reliably, "without overfitting". Labelled **pretrained on published proofs**, as everything built on `pre` is.

**Why.** The arm's read at seed 0 (`ladder-l4-RESULT.md`): six rounds from `pre` make the model more reliable
and narrower. Per attempt it succeeds 17% more often on the goal set; it solves fewer distinct goal problems
(227 against 238 in 93 attempts; the twin 216), its attempts are less varied (the share of distinct attempts on
G falls from 0.82 to 0.71) and it verifies fewer long proofs; on the problems `pre` could not solve nothing is
shown. Read from the stored rounds: the challenger is calibrated (its picks' predicted pass rate is 0.24 on
average and so is the realised one), and it does not aim at 1/10, because with 8 attempts the EXPECTED reward
is nearly flat from 0.10 to 0.25 (a problem nobody solves pays nothing); a quarter of each round's picks are
solved by half the attempts or more, and their short proofs are four in ten of the training rows. Three causes
of the narrowing are each one change away, and none needs a new round: the rounds' attempts and the adapters
are stored on the box. A fourth, that the adapter is full, is L4r's.

**What runs.** One stage, three checks, each a model or a measurement made from the seed-0 arm's stored run
(`ladder_l2_t010_assembly_pre_seed0`: the rounds' training examples with each problem's k of 8, the adapter
`with`) and from `pre`. Every model is trained FROM the start adapter, one pass, the arm's recipe, in the
order of the content hash the arm uses, and measured as `with` was (8 episodes on the three rungs; G twice,
32 and 61 attempts, `pre`'s sampling seeds), so that each pairs by problem with `pre`, `with` and `without`.
The start adapter is ONE setting (`pre`; `pre_r64` if L4r's first branch is taken, every model then at that
rank). In the order they run:

1. **`rehearse`: the published proofs stay in the mix.** The twin's 3,422 one-shot rows, and with them as
   many rows of the pretraining file (a seeded draw of rows `pre` was trained on, no row twice), one pass over
   both in the hash's order. If the narrowing is the model leaving what it was pretrained on for its own
   style, this keeps the breadth and the gain in reliability.
2. **`reward_rows`: train in proportion to the reward.** Each one-shot row of the rounds enters the training
   set with probability r(k), the challenger's reward at the arm's target rate for its problem's k of 8 (0.97
   at k = 1, 0.48 at 2, 0.14 at 3, 0.03 at 4, about nothing above), by a hash of the task's seed and the
   row's id; an assembled row counts as k = 1. No quota and no screen: the solver is trained on what the
   challenger was paid for. If the narrowing comes from the easy rows, this removes them; it is also the first
   training set made mostly of proofs at the model's limit.
3. **`hot`: is the breadth lost only entropy?** No training. `with` and `pre` are each sampled again on G at
   temperature 1.2 (the config's is 1.0), 93 attempts with seeds of their own, no rungs. If a trained model's
   narrowness is a sharper distribution and nothing else, a higher temperature gives the distinct problems
   back while it keeps more successes per attempt than `pre` at either temperature.

**The read, fixed before the run.** For `rehearse` and `reward_rows`, each against the stored `without` (the
same rounds' rows by the old rule) and against `pre`:
- **Checks first (any FAIL: that model is INCONCLUSIVE, the others are read).** Its training ran (L4's third
  check: the adapter is not the start adapter's, the loss did not rise by more than two standard errors);
  under 5% of its rung attempts without an answer; at most 2% of each measured set without an answer; no row
  it was trained on is a held-out problem or, among its rounds' rows, a problem of the `pretrain` half.
- **Primary, breadth: the goal problems solved at least once in the 93 attempts, the model against
  `without`, paired by problem: gained, lost, the two-sided sign test.**
- **Beside it, reliability kept:** all of G, successes per attempt over 93 attempts, the model minus `pre`
  (95% bootstrap over problems); `without`'s is +0.0114.
- **Secondary:** G′ over the 61 fresh attempts against `pre` (the arm's primary); by proof length; solved
  reliably; the three rungs; the share of distinct attempts; the lines of verified proofs; the goal problems
  nothing stored had solved (81 on 2026-10-09) that it solves; the rows it was trained on (count, lines).
- **Branches, each model by itself.** Gained more than lost against `without` with the sign test under
  0.05, AND the interval for all of G against `pre` above zero: THE RULE GIVES THE BREADTH BACK AND KEEPS
  THE GAIN; it replaces the old rule in the next arm. The first and not the second: it gives the breadth back
  by undoing the training, and is not taken. The second and not the first (named on 2026-10-09 after the
  build pointed out it was missing, before any run): IT KEEPS THE GAIN AND DOES NOT GIVE THE BREADTH BACK,
  the old rule's result again by another road; it does not replace the old rule. Neither: not this lever at
  this size. For the last two the note gives the count
  that would have been seen (the sign test's smallest decisive split at the observed number of changed
  problems).
- **`hot` is read, not branched:** distinct goal problems solved in 93 attempts and successes per attempt
  for `with` and `pre` at 1.0 (stored) and 1.2, the share of distinct attempts, and the same on G′. It says
  how much of the breadth is sampling, and which temperature later measurements of trained models should
  also report.
- **What it cannot say.** The rounds were attempted by models trained the old way, so this is the training
  rule alone and not the loop under it: a rule that wins here is confirmed by the next arm, not by this
  stage. One seed. `rehearse` trains again on published proofs `pre` has seen: a model that only gains from
  seeing them twice would show it against `pre`, and the note says so if `rehearse` beats `pre` on G′.
- **Budget.** A training is seven to twenty-five minutes (about 190 to 856 optimizer steps), each full measurement 35
  to 50 minutes, the two measurements of `hot` about 35 each without the rungs: about three and a half
  hours, about 160,000 generations and as many Lean checks, no new round.

**L4r's first branch was taken: what L4t runs (fixed 2026-10-09, after L4r's read and before any run of L4t).**
L4r read THE ADAPTER'S SIZE CAPPED THE PRETRAINED MODEL (`pre_r64` minus `pre` on all of G: +0.0133 [+0.0077,
+0.0192]). By the setting above the start adapter is then `pre_r64` and every model is at rank 64. The stored
`with` and `without` are rank-16 models built on `pre`, so nothing stored is the old rule at this rank, and the
cap is itself a suspect for the narrowing. So:
- **A third trained model, `old_rule`, runs FIRST:** from `pre_r64`, one pass over the twin's 3,422 one-shot rows
  in the twin's order (what `without` is, at rank 64). **It takes `without`'s place in every read above:** the
  primary of `rehearse` and of `reward_rows` is the goal problems solved in 93 attempts against `old_rule`, and
  `pre_r64` takes `pre`'s place (reliability kept: all of G per attempt against `pre_r64`; G′ is `pre_r64`'s:
  the goal problems it does not solve in its 32-attempt sampling, read over its 61 fresh attempts).
- **`old_rule` is read by itself, and that is the cap's question:** `old_rule` against `pre_r64`, the goal
  problems solved in 93 attempts (gained, lost, the sign test), the share of distinct attempts on G, the share
  of its verified proofs with 8 lines or more, and all of G per attempt; beside each, rank 16's stored figures
  (`without` against `pre`: 216 against 238 solved; 0.70 against 0.82 distinct; 12% against 23%; +0.0114). Named
  outcomes: lost more than gained with the sign test under 0.05: THE OLD RULE NARROWS AT RANK 64 TOO (the cap
  was not the cause; the training rule is). Otherwise, with all of G per attempt against `pre_r64` above zero:
  NO NARROWING SHOWN AT RANK 64 (the old rule may stand at this rank; the note gives the smallest decisive
  split). Otherwise: NOT READ AS EITHER.
- **The order** is `old_rule`, `reward_rows`, `rehearse`, then `hot` (now `old_rule` and `pre_r64` at 1.2), so
  that the cap's question is answered first and the last steps can give way to the arm.
- **The rows are the rank-16 rounds'.** The proofs were written by models built on `pre`, and each row's k is
  that model's. `pre_r64` is stronger, so some of what was its limit for them is not for it: this is the
  training rule on given rows at rank 64, not the rounds `pre_r64` would have made. What it settles is
  confirmed by the arm from `pre_r64`, which is still what follows.
- **Budget.** Three trainings (about 15, 6 and 30 minutes at rank 64) and three full measurements (L4r's took
  74 minutes with another job on the pool; 50 to 75), then two of `hot` (about 45 each): five to six hours,
  about 202,000 generations.

**Made exact by the build (2026-10-09; nothing queued, nothing run on a GPU; the read and the branches above are
not changed).**
- **The stage.** `ladder_l4_rows` (`gpu/ladder_l4_rows.py`; rules `domain/ladder_round/l4_rows.py`; report
  `reporting/ladder_l4_rows.py`), run directory `ladder_l4_rows_seed<N>`, settings `ladder_loop.l4.rows` (one
  reader). Eight steps, each resumable: prepare; train and measure `rehearse`; train and measure `reward_rows`;
  measure `hot` for `with`, then for `pre`; report. Smoke: `ladder_l4_rows_smoke`, in the L4 smoke runs' world.
- **A fact corrected, by the pre-flight on the pulled copy of the seed-0 arm.** `reward_rows` is 1,349 rows
  (1,262 one-shot and all 87 assembled; by k: 1,009 at k = 1, 273 at 2, 62 at 3, 4 at 4, 1 at 5), which is
  **169** optimizer steps, not "about 190": the Budget's range is 169 to 856. `rehearse` is 6,844 rows, 856 steps.
  Of `reward_rows` 54% of the proofs have 4 lines or more and 15% have 8 or more (all the rounds' rows: 45% and
  11%); of `rehearse` 60% and 31%.
- **The rows the rule reads** are the last model's training set (`assembly.training_set`: the one-shot and the
  assembled rows of every round, one proof a problem). k is the pick's verified attempts of 8 as the challenger
  read it (`assembly.counted`: an assembled row is k = 1); r(k) is the challenger's own function
  (`episodes.reward`) at the target rate the arm's run recorded. A row of another origin (H0) is refused: the
  rule names no k for it.
- **The two draws.** A row's number in [0, 1) is the first 13 hexadecimal digits (52 bits) of SHA-256 of
  `<seed>:l4_rows_keep:<row id>` over 16^13; it is kept when that is under r(k). The rehearsal is the rows of the
  smallest SHA-256 of `<seed>:l4_rows_rehearsal:<problem>#pretraining` among the rows `pre` was trained on. Every
  row's k, reward and draw, kept or not, is stored (`l4_rows_reward_draws.jsonl`).
- **The rehearsal rows' text is not stored in the run.** They are other people's published proofs: the stored
  training set names each by its id, its problem and its place in the file, and the training step reads the text
  from the snapshot's file, held to the SHA-256 `pre`'s prepare step recorded. Every other row is stored with
  its text, as the twin's are.
- **A case the branches did not name, named before any run.** The build found that the branches named both
  conditions, the first alone, and neither. THE SECOND AND NOT THE FIRST (the breadth is not given back, the
  gain is kept: what `without` itself is) was named on 2026-10-09 before the stage was queued: IT KEEPS THE
  GAIN AND DOES NOT GIVE THE BREADTH BACK, reported with the same count as "neither" (the smallest decisive
  split). With too few changed problems for any split to be decisive (fewer than six), the note says that no
  split is.
- **The checks' reach.** The third check reads the model's own three sets and the stored sets of `pre` and of
  `without` it is read against (those were held to it at the prepare step too). The fourth reads the rows the
  training loop recorded: the rehearsal rows are of the `pretrain` half by construction and are named as such;
  a rehearsal row that is NOT of that half, like a round's row that is, fails it.
- **`hot`.** Sampling seeds 1040 and 1041 (no other measurement's; the prepare step refuses one the two runs
  recorded). Its sets carry `hot` in their names (`l4_reach_hot_with`). `ladder_ceiling.Model` has one new field,
  `rungs` (true for every other caller), so that G alone is sampled. On G′ the table gives the 93 attempts and,
  beside them, the second sampling alone: at 1.0 `pre`'s first sampling CHOSE G′ and has no success there. For
  information, each model at 1.2 against itself at 1.0 (gained, lost; the paired difference per attempt).
- **Pairing.** Every model here is measured with the sampling seeds `pre`'s run recorded, and the prepare step
  refuses unless `with` and `without` were measured with the same: true at seed 0, not at a further seed of the
  arm, where this stage is refused as it stands.
- **From `pre_r64`: the amendment above ("L4r's first branch was taken"), made exact by its build** (2026-10-09;
  nothing queued, nothing run on a GPU, nothing sent to Lean; the read and the outcomes are not changed; no fact
  of the amendment was found wrong: its rank-16 figures, its 3,422 rows and its 202,000 generations are what the
  stored rows and the arithmetic give).
  *The stage and the setting.* `ladder_loop.l4.rows.start` STAYS `pre`, and the stage `ladder_l4_rows` and its
  two jobs are what they were. The stage `ladder_l4_rows_r64` (smoke: `ladder_l4_rows_r64_smoke`) names the
  start by a variable (`RLVR_LEAN_LADDER_L4_ROWS_START=pre_r64`), as the rank check's stages name their check;
  the setting itself set to `pre_r64` is the same run. The three models of another start are a setting of their
  own (`models_from_another_start: [old_rule, reward_rows, rehearse]`), which the stage's steps follow. Run
  directory `ladder_l4_rows_pre_r64_seed<N>`. Ten steps: prepare; train and measure `old_rule`, `reward_rows`,
  `rehearse`; `hot` for `old_rule`, then for the start model (the step is `ladder_l4_rows_measure_hot_start`,
  its sets `l4_<sampling>_hot_pre_r64`: a step's name is registered without a config); report.
  *Where the start lives, and its rank.* A start that is not `pre` is a model of `ladder_loop.l4.rank_check`,
  found by the rank check's own functions: `ladder_l4_pretrain_r64_seed<pretraining_seed>/adapters/pre_r64`.
  The rank and alpha every training attaches are the ones that run's prepare step RECORDED in its recipe (64 and
  128), held to the setting's check and to the adapter that run read back; never `lora.rank`. One function
  hands a training the config with that rank and alpha, and a measure step the same with the model server's
  largest adapter rank raised to it (the rank stage's own `config_of`); from `pre` the config is handed on as
  it is. The adapter each training saves is read back (the rank and alpha in its config file, the rank of every
  matrix), and one that is not the start adapter's fails its step unmarked, so nothing of it is measured.
  *What is read, and refused before anything is written.* Of the rank check's run: its report (refused,
  naming `ladder_l4_rank_seed0.json` (a job config, not shipped in this copy), when it is missing, not to be read,
  or its checks failed); the markers of
  its prepare, train and measure steps; its seed, its model and its recorded rank; `pre_r64`'s per-problem rows
  on its three sets with their markers, held to this run's problems, seeds and attempts and to Lean having
  answered; and its file's SHA-256 and rows, which must be the ones `pre`'s prepare step recorded (the rehearsal
  rows are drawn among rows both were trained on). Of the arm's run: what it always read, less `with`'s adapter
  and the two measured models' markers; ITS RECORDED START MUST BE `pre` whatever this run starts from, and the
  summary says so (`rows_made_by`). Of `pre`'s run: its prepare step's marker and `l4_pretraining_rows.jsonl`.
  *G′.* Made by the pretraining's rule (`l4.goal_set_again`) from `pre_r64`'s stored first sampling and stored
  in this run (`l4_goal_set_again.jsonl`); `pre`'s own is not read. Its count by length group is given by the
  REPORT and not by the prepare step: the length of a held-out problem's published proof is read by reports
  and by nothing before them. From the pulled rows: 192 (17 of one line, 45 of 2-3, 74 of 4-7, 56 of 8 or more).
  *`old_rule`.* The twin's file's rows in the file's order, each with its k, stored with its text as the
  twin's are; 428 optimizer steps. Its four checks are the other models' (the third reads its own sets and the
  start model's). Read by itself (`by_itself` in the report): the four figures against `pre_r64`, and beside
  each the same figure of `without` against `pre`, COMPUTED from their stored rows when those are on the box
  and pair with this run (G's two samplings, their markers, the two measure steps' markers); when they are not,
  the report says NOT THERE with the reason, and nothing is refused. From the pulled rows that gives the
  amendment's figures: 216 against 238 solved (gained 25, lost 47, p = 0.013), 0.702 against 0.817 distinct,
  11.8% against 22.9%, +0.0114 [+0.0026, +0.0203]. The outcome's name carries the START ADAPTER'S rank (64
  here). A narrowing is decided first: an old rule that loses breadth by the sign test narrows, whatever it
  gains per attempt. An interval that ends at zero is not above zero. `NO NARROWING SHOWN` and `NOT READ AS
  EITHER` give the smallest split at which a narrowing would have been seen (the mirror of the other
  branches'), or say that none is decisive with fewer than six changed problems.
  *The two others.* The primary is against `old_rule`'s rows as this run measures them; reliability, G′ and
  the rungs against `pre_r64`; the third check reads the model's own sets, `pre_r64`'s and `old_rule`'s. When
  `old_rule` is INCONCLUSIVE each of the two reads NOT READ (a name the report already had), with that reason,
  and what was measured is kept under `measured_and_not_read`.
  *A case the amendment did not name: the smoke run.* The smoke `pre_r64`'s own report reads INCONCLUSIVE
  (twelve rows have no loss to compare), which the refusal above stops. The smoke stage alone sets
  `RLVR_LEAN_LADDER_L4_ROWS_START_CHECKS=smoke`: it goes on from a rank check whose checks failed, never from
  a report that is not to be read, and the prepare step records it.
  *Measured before any run, on the pulled copies* (`pre`'s prepare marker and two set markers made again from
  that copy's own step results, as for L4r). The prepare step runs to its end from `pre_r64`: `old_rule`
  3,422 rows, `reward_rows` 1,349, `rehearse` 6,844 (428, 169 and 856 optimizer steps); G′ 192; every model
  at rank 64. The smoke stage's prepare step runs to its end on the pulled copies of the three smoke runs. The
  report, with `pre_r64`'s own rows standing in for the three models, reads every difference as zero
  (`old_rule` NOT READ AS EITHER, the two others NOT THIS LEVER AT THIS SIZE); with `pre`'s stored rows
  standing in for one model, that model's all of G against `pre_r64` reads -0.0133 [-0.0192, -0.0077], L4r's
  primary with its sign turned, which is the check that the references switched. Generations: 3 x 42,817 +
  2 x 36,985 = 202,421. The task writes about 140 MB to its out directory (five measurements' attempts).
  *From `pre`, nothing moves.* In the scripted world a run from `pre` writes the same 79 files and returns the
  same 8 summaries with the code before the build and after it, key order included (paths and wall-clock
  figures aside), and its report on eleven hand-made cases (every branch, a failed check, a set Lean did not
  answer, a smoke run's one sampling, no goal problem) is the same report.
  *Not shown without the box:* a stored rank-64 adapter loaded into a rank-64 training by the real loop, and
  the memory of these trainings (the rank check's own peaked at 12.82 GB reserved, which is the cap; they attach
  the same adapter and their longest rows are the same file's). The smoke run is the first run of both.
- **The smoke run's minimum.** The fixture's problems are solved by every attempt or by none, so the rule may
  keep no row; `RLVR_LEAN_LADDER_L4_ROWS_MINIMUM=2` (the smoke stage alone) fills `reward_rows` to two rows by the
  smallest draws, each marked as kept by the run's minimum and not by the rule. Every other run refuses an empty set.
- **The never-solved problems** are read from `l4_rows_never_solved.jsonl` in the run directory when it is there
  (one `problem_id` a row); the stage does not compute them.

### L4b: the arm again, from the larger pretrained model (fixed 2026-10-09, after L4r's read and BEFORE L4t's)

**Under the owner's word of 2026-10-08 (night)** and by L4r's own branch: "the arm is then run again from
`pre_r64` (its own map, the same halves, every model of the arm at that rank) before more seeds are bought at
rank 16". Labelled **pretrained on published proofs**.

**What runs.** L4's two tasks' shape again, with what the two results name and nothing else changed:
1. **`pre_r64`'s own map**, a step of its own made first: the base map's 4,000 problems attempted again by
   `pre_r64`, 8 attempts each, the stored map's sampling seed and sides, as `pre`'s map was made. The
   challenger starts from it (one rule for every arm: the map of the model the arm starts from).
2. **The arm `t010_assembly_pre_r64`:** target rate 1/10, six rounds of 1,000 problems from the `loop` half,
   the Lean-only assembly after every batch, round 1 attempted by `pre_r64`, every model of the arm and the
   twin trained FROM `pre_r64` at its rank (64, alpha 128; read from the adapter, not set twice), no H0, the
   two measurements with `pre_r64`'s sampling seeds.
3. **The training rule is ONE setting of the arm, and L4t names it.** Decided here, before L4t is read, so
   that the rule is not chosen by the result it is then judged on:
   - a rule that L4t reads as THE RULE GIVES THE BREADTH BACK AND KEEPS THE GAIN is the arm's rule; if both
     `reward_rows` and `rehearse` do, the one with the larger (gained minus lost) against `old_rule`, and
     `reward_rows` on a tie (it needs no published proof in a round);
   - else, if `old_rule` reads NO NARROWING SHOWN AT RANK 64: the old rule (one proof for every solved
     problem, as every arm so far);
   - else (the old rule narrows at rank 64 too and no rule gives the breadth back): THE ARM IS NOT RUN as
     specified; what L4t's `hot` and its secondary reads say goes to a new section first.
   In a round the rule is applied to the rows of rounds 1 to r as L4t applies it to all six: `reward_rows`
   keeps a row by the hash of the task's seed and the row's id against r(k), so a row kept for M(r) is kept
   for every later model; `rehearse` adds as many rows of the pretraining file as the model has one-shot rows,
   by the same nested hash draw. The twin is the same rule with the assembled rows left out.

**The read, fixed before the run.** L4's, with `pre_r64` in `pre`'s place:
- **Checks first** (any FAIL: INCONCLUSIVE): L4's five, read on `pre_r64` (its report's checks pass; G′, the
  goal problems `pre_r64` does not solve in its 32-attempt sampling, holds at least 80: it holds 192; the two
  measured trainings ran; each measured model still writes proofs; no barred row).
- **Primary: on G′ (192 problems) over the 61 fresh attempts, `with` minus `pre_r64`, paired by problem, 95%
  bootstrap over problems.** Branches as L4's: interval clear of zero and above, THE LOOP ADDS ON TOP OF
  PRETRAINING (two more seeds of this arm from the same `pre_r64`); interval holds zero, NOT SHOWN at this
  size; below zero, THE ROUNDS COST THE PRETRAINED MODEL.
- **Beside it, the owner's "without overfitting", read on every branch:** the goal problems solved at least
  once in 93 attempts, `with` against `pre_r64` (gained, lost, the two-sided sign test), and the share of
  distinct attempts on G. Lost more than gained with the sign test under 0.05 is written NARROWER whatever
  the primary says, and a primary that "adds" with it is reported as "adds per attempt, on fewer problems".
- **Secondary:** L4's (all of G by proof length; the rungs; solved reliably; `with` against `without`; the
  rounds' table: picks by k, training rows by origin, by k and by lines, the rule's kept share), the rank-16
  arm's figures beside each, and the goal problems nothing stored has solved that each model solves.
- **Budget.** The map about 45 to 60 minutes (32,464 attempts; `pre_r64`'s attempts take longer to check).
  The arm: rank 16's took 279 minutes; at rank 64 the trainings and the two measurements are longer (L4r's
  measurement took 74 minutes): about six to seven hours. One seed.

**Made exact by the build (2026-10-09; nothing queued, nothing run on a GPU, nothing sent to Lean; the read, the rule
for choosing the training rule and the branches above are not changed).**
- **The stages, and how the rule is chosen.** Three stages, one a rule: `ladder_l4b_old`, `ladder_l4b_reward_rows`,
  `ladder_l4b_rehearse` (`gpu/ladder_l4b.py` on the L2 stage's arm; rules `domain/ladder_round/l4b.py`; report
  `reporting/ladder_l4b.py`), each with a smoke stage, and six jobs (`ladder_l4b_<rule>_seed0.json`,
  `..._smoke.json`: job configs, not shipped in this copy). THE RULE IS ONE SETTING OF THE ARM (`ladder_loop.l2_assembly_arms.t010_assembly_pre_r64.rule`: `old`),
  read in `ladder_l2.assembly_arm` and nowhere else; a stage names another by `RLVR_LEAN_LADDER_L2_RULE`. So the rule is
  chosen by which job is queued, with no code or config changed at queue time. A run with another rule than `old` has the
  rule in its run directory's name (`ladder_l2_t010_assembly_pre_r64_<rule>_seed<N>`), and a run keeps the rule it was
  prepared with. An arm that states no rule (the base arm; the arm from `pre`) is refused the variable, and is trained and
  recorded as it always was. An arm with a rule reads no H0 (refused: a rule gives a k to a round's own rows only).
- **Where a start model lives is ONE function** (`gpu/ladder_l4_start.py`: `the_start`, with `config_from`,
  `read_the_start_run`, `the_adapter_saved`, `pretrained_on`, `examples_of`, moved out of L4t's module), asked by L4t and
  by the arm (`ladder_l2.the_start_of`). `pre`: the ONE pretraining's run. `pre_r64`: the rank check's run and adapter, by
  that check's own functions. The rank and alpha are the ones that run's prepare step RECORDED (64 and 128), held to the
  setting's check and to what that run read back from the adapter's two files; the arm's own prepare step stores them
  (`start_recipe`) and refuses, before anything is written and naming `ladder_l4_rank_seed0.json` (a job config, not
  shipped in this copy), a check's run without
  its report or markers, one not to be read or whose checks failed, and another seed, model or rank. Every training of
  the arm and of the twin attaches that rank (`ladder_l2.config_for`); the adapter each saves is read back (its config
  file's rank and alpha, the rank of every matrix) and one at another rank fails its step unmarked; every step that
  samples (the map, the six rounds, the two measurements) starts the model server with that rank as its largest. From
  `pre` and from the base the config handed on is the config itself, the same object.
- **`pre_r64`'s own map lives in a run directory of its own:** `ladder_l4_map_pre_r64_seed<pretraining seed>`
  (`l4_map_pre_r64.jsonl`; step `ladder_l4b_map`, the stage's FIRST step). The rank check's run is only read: no later
  stage writes there (`pre`'s map stays beside `pre`, where its own stage wrote it). It is `pre`'s map step under
  `pre_r64`'s names (`ladder_l4.the_map_of`: one function, the pretraining's body unchanged): the same refusals, the sides
  held to the stored map's, the 2% rule, block-resumable. One map for every rule and seed of the arm (it is sampled with
  `base_map.sampling_seed`, which no task's seed moves): a later task finds it done, and a failed arm keeps it. Before it
  samples, the map step reads and refuses the check's run as the arm's prepare step does. `start_map` then refuses, in
  the arm's own prepare step, a map that is missing or was made with another seed, attempts or set of problems, naming
  the step to run.
- **The rule inside a round** (`l4b.rule_set`, called where M(r)'s set is built, `ladder_assembly.train_round`; L4t's own
  functions: `with_k`, `reward_draws`, `kept_rows`, `rehearsal_rows`). k is the problem's k of 8 as the challenger read
  it in the row's own round (an assembled row: 1). `old`: the rows and the order every arm has. `reward_rows`: the rows
  whose draw is under r(k); each kept row carries its k, reward and draw; M(r)'s set is M(6)'s rows of rounds 1 to r, in
  M(6)'s order (asserted). `rehearse`: EVERY row of the rounds so far (the assembled ones too) and as many rows of the
  pretraining file as the model has ONE-SHOT rows, the first of one hash order of the rows the START model was
  pretrained on (its run's `l4_pretraining_rows.jsonl`; M(r)'s are the first of M(r + 1)'s, asserted); stored by id and
  place in the file, never with text; the text is read from the snapshot's file, held to the SHA-256 the start model's run
  recorded, by the training and again by the twin's. THE TWIN is the last model's set, in its order, with the assembled
  rows left out: under `rehearse` that is L4t's `rehearse` set exactly. A training's summary says the rule, the rows by
  origin and the kept share of the rounds' rows in all and by k (`rule`).
- **The checks on `pre_r64`** are read by the arm's own prepare step (`ladder_l4b_prepare`), which refuses a model that
  fails one. The first is BOTH L4's own (it solves at least 150 goal problems in its 93 attempts: 245) and the section's
  ("its report's checks pass": the three checks of the rank check's report, and that report not INCONCLUSIVE). G′ is made
  there by the pretraining's rule from `pre_r64`'s stored first sampling and stored in the arm's run (the rank check's
  run stored none). The FIFTH check reads the rounds' rows as ever; under `rehearse` the rehearsal rows stand apart, are
  counted and named as rows of the `pretrain` half, and one is barred only when it is a held-out problem or is NOT of
  that half; under any other rule a row of that half is barred whatever it is called.
- **The breadth beside the primary** (`l4b.breadth_beside`, `with_breadth`; `build_l4_report(breadth=True)`): the goal
  problems solved at least once over both samplings (93 attempts), by the function L4's secondary read uses, and the two
  shares of distinct attempts on G. The branch's NAME stays one of L4's three; `reported_as` adds `, NARROWER` when lost
  exceeds gained with the sign test under 0.05, and for THE LOOP ADDS it reads "... NARROWER: adds per attempt, on fewer
  problems". INCONCLUSIVE says nothing else (what was measured is kept, the breadth with it). A run with one sampling (a
  smoke run: the primary NOT READ) still has its breadth read.
- **Beside, never a reason to refuse.** The base arm's own gain stands beside the primary as in L4, when its run is on
  the box. The rank-16 arm's figures are ITS STORED REPORT's (`report_ladder_l4.json` of `ladder_l2_t010_assembly_pre_seed<N>`,
  copied by the prepare step into `l4b_rank_16.json`), not computed again from its rows. The never-solved ids are read
  from `l4b_never_solved.jsonl` in the arm's run directory; the report step can be run again.
- **No fact of the section was found wrong.** One is made more exact: the rank is "read from the adapter" through what
  the run that made it recorded and read back from its files, not by opening the adapter again in every step. And the
  Budget by arithmetic, on the rank-16 rounds' sizes: the seven trainings are 1,998 optimizer steps under `old`, 777 under
  `reward_rows` and 3,958 under `rehearse` (at `pre`'s 2.2 seconds a step: about 72, 28 and 143 minutes), so the arm with
  `rehearse` is about an hour longer than with `old`. Not measured.
- **Measured before any run, on the pulled copies of seed 0** (read through links; a stand-in map of `pre`'s 4,000 map
  rows under `pre_r64`'s names). The arm's three prepare steps run to their end with each rule: 26,029 candidates of
  51,631; the start's rank 64 and alpha 128; the map's 4,000 rows in the base map's place; check 1 passes (245 of 392
  solved, 150 asked; the rank report's three checks pass); G′ 192 (80 asked); `pre_r64`'s rows on 712 + 392 + 392
  problems; the rank-16 arm's report read (NOT SHOWN ON TOP OF PRETRAINING); the run directories
  `ladder_l2_t010_assembly_pre_r64_seed0`, `..._reward_rows_seed0`, `..._rehearse_seed0`. THE RULES ON THE RANK-16 ARM'S
  STORED ROUNDS, as if they were this arm's (rows of M(1) to M(6); the twin): `old` 625, 1,232, 1,830, 2,389, 2,960,
  3,509; 3,422. `reward_rows` 258, 502, 728, 947, 1,152, 1,349 (33 to 169 optimizer steps); 1,262; M(6) is L4t's set
  (1,009 rows at k = 1, 273 at 2, 62 at 3, 4 at 4, 1 at 5; all 87 assembled). `rehearse` 1,238, 2,434, 3,622, 4,727, 5,848,
  6,931 (613 to 3,422 rehearsal rows); 6,844, which is L4t's `rehearse`. Nested under both, and no text in a rehearsal row.
  The smoke stages' three prepare steps run to their end, with each rule, on the pulled copies of the three smoke runs
  they read (a stand-in map of the fixture's four rows): 5 candidates, rank 64, the smoke rank check's failed check
  waived and recorded, check 1 FAIL as said below, the fixture's SHA-256 the one that run recorded. Whether the box
  still holds those runs is not verified from here: the prepare steps refuse, naming the task, if it does not.
- **The stages before it write what they wrote.** In the scripted world, with the code before the build and after it:
  the arm from `pre` (its pretraining with it) the same 158 files, the base arm the same 227, L4t from `pre` the same 237
  and L4t from `pre_r64` the same 300, and every step's summary the same key for key (paths and wall-clock figures aside).
  Four assertions that state where things live were brought up to date (an arm more; `config_from` in the shared module;
  three smoke stages more that read the rank check's smoke run).
- **The smoke stages** (`ladder_l4b_<rule>_smoke`) read `ladder_l1_smoke`, `ladder_l4_pretrain_smoke` and
  `ladder_l4_rank_smoke`, and write `ladder_l4b_map_smoke` and `ladder_l4b_<rule>_smoke`. The smoke `pre_r64`'s own report
  reads INCONCLUSIVE, so these stages alone set `RLVR_LEAN_LADDER_L2_START_CHECKS=smoke`, go on, and their own report
  reads INCONCLUSIVE on its first check; `RLVR_LEAN_LADDER_L2_RULE_MINIMUM=2` fills a set the rule would leave empty, each
  such row marked as not kept by the rule.
- **Not shown without the box:** a rank-64 adapter served for a round's attempts and loaded into a round's training; the
  memory of those trainings (the rank check's own reached the cap of reserved memory; these attach the same adapter and
  cut rows at the same length); the real map. The smoke run is the first run of the first.
- **Mutated.** Fifty one-change mutations of what this build added (the three rules and their nesting, the twin, the kept
  share, NARROWER and how a branch is said, the fifth check with rehearsal rows, the rank handed to every training and
  every engine, the adapters read back, the run's rule and directory, the map's place and seed, G′'s sampling, the
  prepare steps' refusals, the report's start model). The stage's tests noticed 46 as first written; four assertions were
  added for the four they missed (a held-out rehearsal row of the `pretrain` half; G′ drawn from the second sampling;
  check 1 not reading the rank report's own checks; the primary read against the twin), and all fifty are noticed.

**Outcome of L4t (seed 0, 2026-10-09; `ladder-l4t-RESULT.md`), and of L4b by its own rule: THE ARM IS NOT RUN AS
SPECIFIED.** From `pre_r64`, every model at rank 64, all checks PASS, 93 attempts a goal problem.
- `old_rule`: **THE OLD RULE NARROWS AT RANK 64 TOO.** 217 goal problems solved against `pre_r64`'s 245 (gained 25,
  lost 53, p = 0.002); all of G per attempt −0.00107 [−0.01141, +0.00960] (78.9 against 80.0 per 1,000). The loss
  is long proofs: on the problems of 4 lines or more, 99 solved against 125 (gained 12, lost 38), and the share of
  its verified proofs with 8 lines or more is 0.125 against 0.254. The cap was not the cause; the training rule is.
- `reward_rows`: **IT GIVES THE BREADTH BACK BY UNDOING THE TRAINING.** 241 against `old_rule`'s 217 (gained 48,
  lost 24, p = 0.006); all of G against `pre_r64` +0.00395 [−0.00461, +0.01267].
- `rehearse`: **the same branch.** 234 against 217 (gained 41, lost 24, p = 0.046); +0.00502 [−0.00453, +0.01478].
- `hot` (1.2), read and not branched: `old_rule` 215 solved against 217 at 1.0 and 67.5 against 78.9 per 1,000;
  `pre_r64` 240 against 245 and 62.4 against 80.0. Temperature costs successes and gives no breadth.
- So no rule reads THE RULE GIVES THE BREADTH BACK AND KEEPS THE GAIN and `old_rule` narrows: L4b's third case.
- **A fault in the bar, said plainly.** "Keeps the gain" asked for more successes per attempt on ALL of G than
  `pre_r64`, an interval above zero. That bar was set from rank 16, where the old rule had +0.011. At rank 64 the
  old rule itself has none (−0.001), so the branch's words "by undoing the training" say more than was measured:
  there was no gain on all of G to undo. The verdicts stand as written; what follows reads the same rows by the
  ARM's own read, which was fixed (L4b) before L4t was read.
- **By the arm's read, from the same stored rows** (secondary here, not a verdict): on G′ of `pre_r64` (192
  problems, the 61 fresh attempts) `old_rule` +0.00538 [+0.00000, +0.01272], `reward_rows` +0.00615 [+0.00171,
  +0.01101], `rehearse` +0.00734 [+0.00205, +0.01358] (13.8, 14.5 and 15.7 against 8.4 per 1,000). G′ is chosen by
  `pre_r64`'s own first sampling, which favours any model that differs from it; on a set no rank-64 model chose
  (the 154 goal problems the rank-16 `pre` does not solve in 93 attempts) the four models have 4.9, 8.5, 7.5 and
  7.6 successes per 1,000 and solve 38, 28, 37 and 31 problems: MORE SUCCESSES, ON NO MORE PROBLEMS. Breadth
  against `pre_r64`: `reward_rows` gained 32, lost 36 (p = 0.72); `rehearse` gained 27, lost 38 (p = 0.21). Long
  proofs (4 lines or more, per attempt, against `pre_r64`): `old_rule` −0.00636 [−0.01786, +0.00622], `reward_rows`
  −0.01014 [−0.01903, −0.00117], `rehearse` +0.00809 [−0.00295, +0.01996]. Solved reliably: 107 (`pre_r64`), 97,
  108, 114.
- **Never solved by anything stored:** 81 before the arm, 58 now; this run's five samplings add 11 (each model 4
  to 8, mostly different ones, once or twice each). At 186 attempts `pre_r64` with its own sampling at 1.2 solves
  263 goal problems, and with a trained model's 93 attempts 270 to 277.

### L4t2: two more one-change checks, on what L4t left open (fixed 2026-10-09, after L4t's read and before any run)

**Under the owner's word of 2026-10-08 (night).** Labelled **pretrained on published proofs**. No new round: the
stored rounds, `pre_r64`, L4t's own sets and measurements, which are only read.

**What L4t left open, and why it is asked before the arm (about an hour a model against ten for the arm):**
1. `rehearse` is the one rule that adds on G′, is not narrower and holds the long proofs. But half its rows are
   published proofs `pre_r64` has already seen once: A SECOND PASS OVER THEM ALONE WOULD SHOW THE SAME. An arm
   with rehearsal rows cannot be read until that is separated.
2. `reward_rows` (a third of the rows, the hard problems' rows) keeps the breadth and adds on G′, and loses the
   long proofs. Rehearsal holds them. The two together are not measured.

**What runs.** Two models, each trained FROM `pre_r64` at its rank, one pass, the arm's recipe, in the content-hash
order, and measured as L4t's models were (the rungs, G twice, `pre_r64`'s recorded sampling seeds):
- `rehearse_only`: the 3,422 rehearsal rows of L4t's `rehearse` set and nothing else, in that set's order.
- `reward_rehearse`: L4t's `reward_rows` set (1,349 rows) and as many rehearsal rows as it has one-shot rows
  (1,262: the first of the same hash order, so they are among `rehearse`'s), 2,611 rows.
The rehearsal rows' text is never stored in the run (L4t's rule: read from the snapshot's file, held to the hash).

**Three reads of a model M against `pre_r64`, the ARM's own (L4b), fixed here before the run:**
- ADDS: G′ (192), the 61 fresh attempts, M minus `pre_r64`, paired by problem, 95% bootstrap: interval above zero.
- NOT NARROWER: goal problems solved in the 93 attempts: not (lost more than gained with the sign test under 0.05).
- HOLDS THE LONG PROOFS: the goal problems of 4 lines or more (230), successes per attempt over the 93 attempts, M
  minus `pre_r64`: the interval is not wholly below zero.
From the stored rows: `old_rule` fails the first two; `reward_rows` fails the third; `rehearse` passes all three.

**Branches, fixed before the run.**
- **Question 1, `rehearse_only`.** On G′ over the 61 fresh attempts, `rehearse` minus `rehearse_only`, paired:
  interval above zero, THE LOOP'S ROWS ADD BESIDE THE REHEARSAL. Else, if `rehearse_only` itself ADDS: THE GAIN IS
  A SECOND PASS OVER PUBLISHED PROOFS (not shown to be the loop's). Else: NOT SEPARATED AT THIS SIZE (differences
  on G′ resolve about 0.005).
- **Question 2, `reward_rehearse`.** The three reads. All three: IT ADDS, KEEPS THE BREADTH AND HOLDS THE LONG
  PROOFS. Otherwise the read it fails is named.
- **The arm's rule after L4t2** (this replaces item 3 of L4b; L4b's read does not change):
  - Question 1 reads A SECOND PASS: no arm with rehearsal rows is run, because it could not show that the loop
    adds. What stands goes to the owner: a second pass over published proofs helps (a fact about the pretraining
    recipe), and the loop's own rows trade long proofs for successes on short ones.
  - Otherwise: the arm is run from `pre_r64` with a rule that passes all three reads: `reward_rehearse` if it
    passes and its (gained minus lost) against `pre_r64` is at least `rehearse`'s; else `rehearse`. THE ARM THEN
    CARRIES ITS OWN SEPARATION: after the last round one more model is trained from `pre_r64` on the last model's
    rehearsal rows alone and measured as the twin is, and the report reads `with` against it beside the primary.
- **What cannot be read here.** The rows are the rank-16 rounds': the arm makes its own, at `pre_r64`'s frontier.
  One seed. A choice between two rules by reads on the same goal set is exploratory, and is said so in the note.
- **Budget.** Trainings of about 15 and 10 minutes; a measurement took 39 to 54 minutes with the pool's 37
  workers and takes about twice that with 15: two to four hours.

**Made exact by the build (2026-10-09; nothing queued, nothing run on a GPU, nothing sent to Lean; the two sets, the
three reads and the branches above are not changed, and no fact of the section was found wrong).**
- **The stage.** `ladder_l4_rows2_r64` (`gpu/ladder_l4_rows2.py`; rules `domain/ladder_round/l4_rows2.py`; report
  `reporting/ladder_l4_rows2.py`), settings `ladder_loop.l4.rows2` (`start: pre_r64`, `models: [rehearse_only,
  reward_rehearse]`; one reader; `pre` is refused: L4t from it has no `old_rule`). The stage names NO variable. Run
  directory `ladder_l4_rows2_pre_r64_seed<N>`. Six steps, each resumable: prepare; train and measure
  `rehearse_only`; train and measure `reward_rehearse`; report. `rehearse_only` is first: Question 1 is what an arm
  with rehearsal rows waits on. Jobs `ladder_l4_rows2_r64_seed0.json` (est_minutes 300) and `..._smoke.json`
  (job configs, not shipped in this copy).
- **What is read, and refused before anything is written.** TWO runs, only read. L4t's from `pre_r64`
  (`ladder_l4_rows_pre_r64_seed<N>`), FINISHED: the markers of its prepare and report steps, its report (one not to
  be read is refused), its problems, its held-out groups, its G′, its two stored sets, what its three models wrote
  and their per-problem rows with each set's marker. The rank check's, as L4t reads it (its report, its three
  markers, `pre_r64`'s rows), and the rows `pre_r64` was pretrained on (`l4_pretraining_rows.jsonl`). **The arm's
  run and `pre`'s run are NOT read:** the sets are made from L4t's stored sets, not from the rounds again, and the
  pretraining rows are the start model's own record (the same file: L4t held the two records to each other). Held
  to each other here, each a refusal: L4t's seed, start and three models; the rank and alpha L4t trained at and the
  check's; how each was measured (sampling seeds, attempts); G′, made again from `pre_r64`'s first sampling by
  `l4.goal_set_again`, which must be the one L4t stored; the pretraining file's SHA-256 (the snapshot's, the check's
  record and L4t's record are one); every stored set's problems, seed, attempts, and Lean having answered.
- **The two sets** (`l4_rows2.rehearse_only_set`, `reward_rehearse_set`). `rehearse_only` is `l4b.rehearsal_of` on
  L4t's stored `rehearse` rows: its rehearsal rows in its order with the others left out, which is the content-hash
  order of them. `reward_rehearse` is every row of L4t's stored `reward_rows` set as stored (its k, reward and draw
  with it) and `l4_rows.rehearsal_rows` (L4t's own draw) asked for as many rows as that set has ONE-SHOT rows; an
  assembled row brings none (1,262 of its 1,349). The prefix is HELD, not assumed: L4t's stored rehearsal rows must
  be that function's draw for their own count, by id and by place in the file (another file or seed is refused),
  and the rows drawn must be its first; then `assembly.training_order`. Each row is numbered by its place in its own
  set. A rehearsal row that carries a text field is refused (`refuse_text`); the training step reads the text from
  the snapshot's file, held to the SHA-256 the START model's run recorded.
- **Trained and measured by L4t's own two step bodies, not by copies.** `ladder_l4_rows.train_a_model` and
  `measure_a_model` under the names a `Stage` gives (L4t's are the defaults; this stage gives its own), and
  `read_the_trained` for a report. L4t's two steps call them. The adapter saved is read back and held to rank 64
  and alpha 128, the model server is started with that rank, the temperature is the config's: as L4t from `pre_r64`.
- **The report.** L4t's report's pieces are functions two reports call (`reporting/ladder_l4_rows.py`: `world_of`,
  `the_four_checks`, `check_lines`, `secondary_reads`, `secondary_lines`). In order: each trained model's four
  checks; the three reads of the two models, then of `old_rule`, `reward_rows` and `rehearse` from L4t's stored
  rows; Question 1; Question 2; ONE line for the arm's rule with the numbers it used; then L4t's secondary reads of
  each trained model BY ITSELF against `pre_r64`, L4t's three models beside in each table.
- **The three reads at the edge of zero** (`l4_rows2.three_reads`). Intervals are the reports' own, rounded to five
  places. ADDS: the low end above zero; one that ends at zero is not above it (`old_rule`'s +0.00000). NOT
  NARROWER: L4b's own `NARROWER`, by its function (`l4b.breadth_beside`). HOLDS THE LONG PROOFS: the high end not
  below zero; one that ends at zero is not wholly below it.
- **Cases the section did not name, named before any run.** (1) A check of a trained model fails: that model is
  INCONCLUSIVE, none of its three reads is made, and a question that stands on it is INCONCLUSIVE (L4t's rule).
  (2) A stored model that L4t's OWN report read as INCONCLUSIVE is not read here either (at seed 0: none). (3) A
  read with nothing measured (one sampling of G: the smoke run) is NOT READ, and a model passes "all three" only
  when all three were made. (4) Question 2, otherwise: `IT FAILS: <read>`, the reads joined by AND. (5) Question 1,
  NOT SEPARATED: its line gives what THIS difference resolves, half the width of its interval, in place of "about
  0.005". (6) The arm's rule: "at least" is >=, so a tie goes to `reward_rehearse`. `NO RULE IS NAMED` when
  Question 1 is INCONCLUSIVE or NOT READ, when a rule's three reads could not all be made (an INCONCLUSIVE
  `reward_rehearse` does NOT fall through to `rehearse`: nothing is said about its training rule, and the task is
  queued again), and when neither rule passes all three (not possible with `rehearse`'s stored rows, which pass).
- **The smoke stage** (`ladder_l4_rows2_r64_smoke`) reads `ladder_l4_rows_r64_smoke` and `ladder_l4_rank_smoke` and
  nothing else, and writes `ladder_l4_rows2_r64_smoke`. The smoke `pre_r64`'s own report reads INCONCLUSIVE, which a
  real run is refused for: this stage sets no variable for it and goes on ONLY because L4t's smoke run, which it
  reads, recorded that IT went on (`checks_waived_for_a_smoke_run`), and records that.
- **Measured before any run, on the pulled copies** (seed 0; copies in a scratch store; no GPU, no Lean). The
  prepare step runs to its end: `rehearse_only` 3,422 rows, 428 optimizer steps; `reward_rehearse` 2,611 rows
  (1,349 and 1,262), 327 optimizer steps; the 1,262 are the first of `rehearse`'s 3,422 in the draw's order, at the
  same places of the file; `rehearse`'s 3,422 are the draw among the 24,866 rows; no rehearsal row carries text; G′
  192; rank 64; the two runs it reads not written to. **The three reads from the stored rows are the section's:**
  `old_rule` +0.00538 [+0.00000, +0.01272], gained 25, lost 53 (p = 0.002), −0.00636 [−0.01786, +0.00622]: fails
  the first two. `reward_rows` +0.00615 [+0.00171, +0.01101], gained 32, lost 36, −0.01014 [−0.01903, −0.00117]:
  fails the third. `rehearse` +0.00734 [+0.00205, +0.01358], gained 27, lost 38, +0.00809 [−0.00295, +0.01996]: all
  three; its gained minus lost is −11. The smoke stage's prepare step runs to its end on the pulled copies of the
  two smoke runs (4 and 4 rows; the waiver read from L4t's smoke run; `reward_rows` INCONCLUSIVE there and not read).
  By L4t's own timings (760 s, 319 s and 1,735 s for 428, 169 and 856 steps) the two trainings are about 16 and 11
  minutes; 85,634 generations.
- **L4t writes what it wrote.** In the scripted world, with the code before the build and after it, L4t from `pre`
  writes the same 237 files (79 its own) and L4t from `pre_r64` the same 300 (100 its own), every step's summary the
  same (paths and wall-clock figures aside). On the REAL rows: L4t's report built again from the pulled steps of
  seed 0 is the same byte for byte before and after, and is the report the box stored (its `adapters` entry aside).
  One assertion that states where things live was brought up to date (one more smoke stage reads the rank check's
  smoke run).
- **Not shown without the box:** a training whose EVERY row is a rehearsal row, its text read from the snapshot's
  file (L4t's `rehearse` mixed them with the rounds' rows); the two trainings' memory (a training takes one row at
  a time, and every row here is a row of L4t's `rehearse` or `reward_rows` trainings, which fitted: 12.81 GB
  reserved at the most). The smoke run is the first run of the first.
- **Mutated.** Ninety-three one-change mutations of what this build added (the two sets and their order, the
  prefix and what holds it, the refusal of text, each of the three reads and its edge at zero, the reads' reference
  and their sets in the report, Question 1's three outcomes and its difference, Question 2, every case of the arm's
  rule and the line that says it, what the prepare step holds L4t's run and the start model's to, the stage's names
  handed to L4t's two bodies, the smoke stage's runs). The stage's tests noticed 91 as written; two assertions were
  added for the two they missed (G′ made from the START model's first sampling and no other model's; what each
  stored model wrote, and its rows, are its own run's), and all 93 are noticed.

### L4c: the arm from `pre_r64` with a rule L4t2 names (fixed 2026-10-09, while L4t2 runs and BEFORE its read)

**Under the owner's word of 2026-10-08 (night).** Labelled **pretrained on published proofs**. This is L4b's arm
(`pre_r64`'s own map; target 1/10; six rounds of 1,000 from the `loop` half; assembly after each batch; every model
at rank 64; the twin; the two measurements) with what "The arm's rule after L4t2" asks and nothing else. It is
built before L4t2 is read so that the card does not wait; WHICH job is queued, or none, is L4t2's line "the arm's
rule" and nothing else: `reward_rehearse`, `rehearse`, or no arm (NO ARM WITH REHEARSAL ROWS; NO RULE IS NAMED).

**What is added to L4b's build.**
1. **A fourth training rule of the arm, `reward_rehearse`.** In a round, on the rows of rounds 1 to r: the rows
   `reward_rows` keeps (the same hash draw against r(k), so a row kept for M(r) is kept for every later model),
   and as many rows of the pretraining file as that kept set has ONE-SHOT rows, the first of the hash order
   `rehearse` draws from (so M(r)'s rehearsal rows are the first of M(r + 1)'s, and are among `rehearse`'s). Stored
   by id and place, never with text. The twin is the same rule with the assembled rows left out.
2. **The arm's own separation, for the two rules with rehearsal rows** (`rehearse`, `reward_rehearse`). After the
   last round ONE more model, `rehearsal_alone`: trained FROM `pre_r64` at its rank, one pass, on the rehearsal
   rows of the last model's set alone, in that set's order with the other rows left out; measured as the twin is
   (the rungs, G twice, the same sampling seeds). Under `old` and `reward_rows` the arm has no such model and
   writes what L4b's build wrote.

**The read, fixed before any run.** L4b's, unchanged (five checks; the primary on G′ over the 61 fresh attempts,
`with` minus `pre_r64`; the breadth beside it and NARROWER), and beside the primary, for a rule with rehearsal rows:
- **The loop's rows, apart from the rehearsal:** on G′ over the 61 fresh attempts, `with` minus `rehearsal_alone`,
  paired by problem, 95% bootstrap. Interval above zero: THE ROUNDS ADD BESIDE THE REHEARSAL. Else, if
  `rehearsal_alone` itself has an interval above zero against `pre_r64`: THE GAIN IS NOT SHOWN TO BE THE ROUNDS'
  (a second pass over published proofs gives it). Else: NOT SEPARATED AT THIS SIZE. A primary that reads THE LOOP
  ADDS ON TOP OF PRETRAINING is REPORTED with this line after it, and is not called the loop's gain unless the
  first of the three is read.
- **The long proofs:** the goal problems of 4 lines or more, successes per attempt over the 93 attempts, `with`
  minus `pre_r64`; an interval wholly below zero is written SHORTER beside the branch, as NARROWER is.
- `rehearsal_alone` has the checks a measured model has (its training ran; it still writes proofs; Lean answered);
  one that fails makes the separation NOT READ and changes nothing else.

**Budget.** L4b's (the map 45 to 60 minutes, the arm six to seven hours, with the pool's 37 workers) and about an
hour more for `rehearsal_alone`; with 15 workers the map and the measurements take about twice as long. One seed.

**Made exact by the build (2026-10-09; nothing queued, nothing run on a GPU, nothing sent to Lean; the two additions, the
read and its outcomes above are not changed, and no fact of the section was found wrong).**
- **The fourth rule.** `reward_rehearse` is one more value of the arm's ONE setting (`l4b.RULES`; the arm's own stays
  `old`), applied where the three others are (`l4b.rule_set`, from `ladder_assembly.train_round`): ONE call of L4t's
  draw and keep rule for the two rules that keep by the reward (`kept_rows(reward_draws(...))`), ONE call of its
  rehearsal draw for the two rules with rehearsal rows (`rehearsal_rows`, asked for as many rows as the KEPT set has
  one-shot rows; an assembled row brings none), then `assembly.training_order`. So a model's rounds' rows ARE
  `reward_rows`' set, row for row with its k, reward and draw, and its rehearsal rows are the first of the order
  `rehearse`'s are drawn in. A rule with rehearsal rows is one of `l4b.WITH_REHEARSAL` (`rehearse`,
  `reward_rehearse`), asked wherever `rehearse` alone was asked before: the pretraining rows handed to a round's
  training without text, the text read from the snapshot's file held to the SHA-256 the start model's run recorded
  (`ladder_l2.rule_inputs`, `examples_of`), that file held by the arm's prepare step before anything is sampled, the
  fifth check naming the rehearsal rows. The twin is `l4b.twin_of`, unchanged. Stages `ladder_l4b_reward_rehearse`
  and `..._smoke`; run directory `ladder_l2_t010_assembly_pre_r64_reward_rehearse_seed<N>`; jobs
  `ladder_l4b_reward_rehearse_seed0.json` and `..._smoke.json` (job configs, not shipped in this copy).
- **`rehearsal_alone`** (`domain/ladder_round/l4c.py`; two steps in `gpu/ladder_l4b.py`, written as the twin's two
  and on the same functions). `ladder_l4b_train_rehearsal_alone`: the rehearsal rows of the last model's STORED set
  (`training_set_m<last>.jsonl`), in its order, each with its place there (`l4c.rehearsal_alone_set`, on
  `l4b.rehearsal_of`); stored as `l4b_training_rehearsal_alone.jsonl` by id and place (a row with a text field is
  refused); trained by `ladder_ceiling.one_pass` from the start adapter at the rank and alpha the arm's prepare step
  stored, with the twin's recipe and adapter seed, the text from `ladder_l2.examples_of`; the adapter
  (`adapters/rehearsal_alone`) read back and held to rank 64 before the step is marked done; its rows' losses and the
  training loop's own record in `l4b_loss_rehearsal_alone.json`. It needs the arm's L4b prepare step and the last
  round's training, and refuses a last set with no rehearsal row (nothing is written).
  `ladder_l4b_measure_rehearsal_alone`: `ladder_ceiling.measure_model` with what `ladder_l3d2_prepare` recorded for the
  twin (the rungs, every sampling of G, the same sampling seeds), under a model server started for rank 64. Its three
  sets (`l3d2_<part>_rehearsal_alone`) get `with`'s problems, put into `problems.jsonl` BY THIS STEP once the model is
  trained, so a run directory prepared before this build needs nothing redone. In the stage's order the two stand
  after `ladder_l3d2_measure_without` and before `ladder_l4b_report`, a guard before each, in the stages of the two
  rules and in their smoke stages, and in no other; under `old` and `reward_rows` each of the two steps refuses.
- **The report** (`build_l4b_report(rehearsal_alone=...)` on `build_l4_report(long_proofs=..., rehearsal_alone=...)`;
  the rules in `l4c.py`). With the two arguments off, which is every default, the report is L4b's byte for byte.
  `rehearsal_alone` IS NO MODEL OF ANY OTHER READ: it is in none of L4's tables, its sets are not among those that
  make a report NOT TO BE READ, and the five checks do not know it (its rows are the last model's rehearsal rows,
  which the fifth check reads there). ITS THREE CHECKS are the functions the run's own checks use: its training ran
  (`l4.the_training_ran`: the adapter saved is not the start adapter's, the loss did not rise), it still writes
  proofs (`still_writes_proofs` on its rung attempts), Lean answered (at most 2% of each of its three sets without an
  answer). A failed one: the separation is NOT READ; `ok`, the branch's name, the primary, the breadth, the long
  proofs and every secondary read are what they were (asserted key for key); and the task queued again samples a set
  Lean did not answer again (the measure step's own rule). THE SEPARATION (`l4c.separation`): the difference is
  `per_attempt` on G′ over the second sampling, `with` minus `rehearsal_alone`; "`rehearsal_alone` itself has an
  interval above zero against `pre_r64`" is L4t2's read ADDS, by its own function (`l4_rows2.three_reads`). THE LONG
  PROOFS (`l4c.long_proofs_beside`): L4t2's read HOLDS THE LONG PROOFS, by the same function, on `with` against the
  start model; its interval is L4's own figure for that pair on the problems of 4 lines or more; SHORTER is that read
  failing.
- **Cases the section did not name, named before any run.** (1) At the edge of zero, as L4t2's reads: an interval that
  ends at zero is not above it and not wholly below it. (2) NOT READ, besides a failed check: nothing measured on both
  sides (one sampling of G: a smoke run); a failed check is said first. (3) NOT SEPARATED: its line gives what THIS
  difference resolves, half the width of its interval. (4) THE SEPARATION IS READ ON EVERY BRANCH that is not
  INCONCLUSIVE (its line stands beside the primary); only a primary that reads THE LOOP ADDS is REPORTED with it.
  `reported_as` is L4b's own words, then `, SHORTER` when it is, then `; apart from the rehearsal: <outcome>`.
  `the_loops_gain` is true only for THE ROUNDS ADD BESIDE THE REHEARSAL, false for the three others (NOT READ among
  them), and not said for a primary that does not add. (5) INCONCLUSIVE says nothing else: neither read is printed
  or branched on, and what was measured is kept under `measured_and_not_read`. (6) The report's `check` stays L4b
  (the stage, the run and the report file are L4b's); `spec_of_the_separation` names this section. (7) A smoke
  run's minimum under `reward_rehearse` (`RLVR_LEAN_LADDER_L2_RULE_MINIMUM`): the rows it adds are rows of the kept
  set, marked as not the rule's, and a one-shot row among them brings a rehearsal row; the nesting is the rule's, not
  the minimum's (as for `reward_rows`).
- **A reading the section left open, and the one taken: SHORTER is read where the separation is.** The list "beside
  the primary" is headed "for a rule with rehearsal rows", and item 2 says that under `old` and `reward_rows` the arm
  writes what L4b's build wrote. So the long proofs are read, and SHORTER written, under `rehearse` and
  `reward_rehearse` and under no other rule; under the two others the report is L4b's byte for byte. Reading the long
  proofs under every rule is one argument (`long_proofs=rehearsing` in `reporting/ladder_l4b.py`), and then those
  two rules' reports are no longer what L4b's build wrote.
- **Under `old` and `reward_rows` the arm writes what it wrote; so do the stages before it.** In the scripted world,
  with the code before the build and after it: the arm from `pre_r64` with `old` the same 168 files and with
  `reward_rows` the same 168 (121 its own each), every step's summary the same key for key; the arm from `pre` the
  same 158, the base arm the same 227, L4t from `pre` the same 237 and from `pre_r64` the same 300, L4t2 the same 199.
  With `rehearse`: of the 168 files L4b's build wrote 165 are the same; `problems.jsonl` holds the model's three sets
  after its own rows, the report and its marker are new; 21 files are added. The four job files of the two other
  rules are L4b's byte for byte (their SHA-256 is asserted). The assertions of L4b's three test files that state the
  rules, the steps and the jobs were brought up to date (a fourth rule; for a rule with rehearsal rows two more
  steps, one more training and one more measured model, an hour more), and one of the rank check's (one more smoke
  stage reads its smoke run); nothing they assert of `old` and `reward_rows` is changed.
- **Measured before any run, on the pulled copies** (seed 0; links and copies in a scratch store; no GPU, no Lean).
  The two rules on the rank-16 arm's stored rounds, as if they were this arm's (24,866 rows `pre_r64` was pretrained
  on; the snapshot's file is the one its run recorded). `reward_rehearse`: M(1) to M(6) 504, 974, 1,418, 1,843, 2,232,
  2,611 rows (one-shot 246, 472, 690, 896, 1,080, 1,262; assembled 12, 30, 38, 51, 72, 87; rehearsal as many as
  one-shot); the twin 2,524 (1,262 and 1,262); `rehearsal_alone` 1,262. M(6)'s kept rows are L4t's stored
  `reward_rows` set, id for id in its order (1,349: 1,262 one-shot, 87 assembled), and M(6) is L4t2's
  `reward_rehearse` set, id for id in its order (2,611), by L4t2's own function on the pulled L4t run and by the set
  its build's pre-flight stored; its rehearsal rows are the first 1,262 of `rehearse`'s 3,422 in the draw's order, at
  the same places of the file. `rehearse`: 1,238 to 6,931 and the twin 6,844 (L4b's figures); `rehearsal_alone`
  3,422, which is L4t2's `rehearse_only` set, id for id in its order. Under both: every model's rounds' rows are the
  last model's rows of its rounds, in its order; M(r)'s rehearsal rows are the first of M(r + 1)'s; under
  `reward_rehearse` they are among `rehearse`'s in every round; no rehearsal row carries text; the text of every row
  of `rehearsal_alone` is read from the snapshot's file, held to the recorded SHA-256. OPTIMIZER STEPS:
  `reward_rehearse` 63, 122, 178, 231, 279, 327, the twin 316 (1,516 for the seven), `rehearsal_alone` 158;
  `rehearse` 155, 305, 453, 591, 731, 867, the twin 856 (3,958), `rehearsal_alone` 428. By L4t's timings at rank 64
  (12.8 minutes for 428 steps, 5.5 for 169; a measurement 39 to 54 minutes with 37 workers) `rehearsal_alone` is 45
  to 70 minutes: the section's hour. The arm's three prepare steps run to their end with each of the two rules
  (26,029 candidates; rank 64; check 1 245 of 392; G′ 192; `..._rehearse_seed0`, `..._reward_rehearse_seed0`).
- **The smoke run of `rehearse` queued again resumes.** A copy of the run directory the smoke run of L4b's build left
  (`ladder_l4b_rehearse_smoke`, as pulled: 122 files), with the stage's steps run in it again: the eleven steps that
  run finished return what is stored and no file is written again (the embedding step was not called: the stand-in
  has another key; on the box it returns what is stored while the shared embeddings are there);
  `ladder_l4b_train_rehearsal_alone` RUNS (4 rows of M(2)'s 8, one optimizer step, the fixture's text read from the
  snapshot's file); `ladder_l4b_measure_rehearsal_alone` reaches its sampling (it stops there for want of the pool's
  key), its two sets put beside `with`'s; the report is built again. So the same job queued again runs the two steps
  and the report in that directory, WHILE THE BOX'S STORE STILL HOLDS IT and the rank check's smoke adapter: a worker
  container made again holds neither, and the three smoke runs the stage reads are then queued first.
- **The jobs.** `est_minutes` 600 for the two real jobs (540 and the hour); 35 for the two smoke jobs (the smoke run
  of L4b's build took 3.4 minutes; the two steps add about one). Each real job says that WHICH is queued, or none, is
  L4t2's line and nothing else.
- **Not shown without the box:** `rehearsal_alone` trained at rank 64 on rows of the real file taken from an ARM's
  last set (its rows are of the kind L4t's `rehearse` trained on, which fitted, 12.81 GB reserved at the most; L4t2's
  smoke run showed a training whose every row is a rehearsal row); its adapter served and measured; `problems.jsonl`
  gaining its sets in a run directory of the box. The smoke run of each of the two rules is the first run of all
  three. With 15 workers the map and the three measurements take about twice as long, and 600 minutes may not hold.
- **Left as L4b's build has it.** L4's own second line of the report counts a rehearsal row among "rows of the
  rounds" and the twin's among "one-shot rows" under a rule with rehearsal rows (it did under `rehearse`); the rows by
  origin are in the rule's own line and in each training's summary. L4b's read is not changed, its words neither.
- **Mutated.** Eighty-nine one-change mutations of what this build added (the fourth rule and its nesting; the count
  by ONE-SHOT rows of the kept set; the twin; `rehearsal_alone`'s set and order, how it is trained and measured, the
  two steps' refusals and their place in the stages; the separation's three outcomes, its edge at zero and its two
  references; its three checks and what a failed one changes; SHORTER and what it is read on; how a branch is said;
  the two rules that are not changed, and their report). The tests noticed 87 as first written; two assertions were
  added for the two they missed (a report that read the twin's rows on the rungs as `rehearsal_alone`'s: the stand-in
  gives every model the same attempts; the arm's prepare step not holding the snapshot's file to the start model's
  record under `reward_rehearse`), and all 89 are noticed.

**Outcome of L4t2 (seed 0, 2026-10-09; `ladder-l4t2-RESULT.md`), and of L4c by its own rule: NO ARM WITH REHEARSAL
ROWS IS RUN.** Both models' checks pass; 93 attempts a goal problem; every read against `pre_r64`.
- **Question 1: THE GAIN IS A SECOND PASS OVER PUBLISHED PROOFS.** On G′ over the 61 fresh attempts `rehearse`
  minus `rehearse_only` is −0.00051 [−0.00606, +0.00512] (15.7 against 16.2 per 1,000), and `rehearse_only` itself
  adds: +0.00786 [+0.00367, +0.01264] against `pre_r64` (16.2 against 8.4). Its three reads all pass (gained 35,
  lost 28; 4 lines or more +0.00930 [−0.00108, +0.01978]). The published rows alone do what the mix did, and more
  beside: 252 goal problems solved (the mix 234, `pre_r64` 245), 124 solved reliably (114, 107), 11 of the 81
  never-solved (4, 7).
- **Question 2: `reward_rehearse` ADDS, KEEPS THE BREADTH AND HOLDS THE LONG PROOFS.** G′ +0.01076 [+0.00470,
  +0.01819] (19.1 against 8.4 per 1,000); gained 32, lost 30; 4 lines or more +0.00753 [−0.00290, +0.01819]. It is
  also the first model above `pre_r64` on ALL of G per attempt: +0.01040 [+0.00137, +0.01937] (90.4 against 80.0).
- **The arm's rule: NO ARM WITH REHEARSAL ROWS** (Question 1 reads a second pass). L4c is built and not run.
- **What is not known.** Whether `reward_rehearse`'s gain is its 1,349 rows of the rounds or its 1,262 published
  rows: against `rehearse_only` (3,422 published rows and none of the rounds) it is +0.00290 [−0.00290, +0.00905]
  on G′ and +0.00359 [−0.00499, +0.01262] on all of G, not separated, and its own control (1,262 published rows
  alone) was not measured. And how much a second pass can give: `rehearse_only` is 14% of one.
- **A fault of the first run:** its second training ran out of memory at the cap on a fragmented cache (task
  15f70151: VOID for that model; the first model's stored steps were kept). Fixed in f0faa6624 (the allocator's
  segments may grow; its cache is released between rows; no number a training computes changes), and the run
  queued again resumed at that training, which then fitted (12.25 GB reserved at its peak).

### L4p: how much more pretraining? The dose of a second pass (fixed 2026-10-09, after L4t2's read and before any run)

**Under the owner's word of 2026-10-08 (night).** Labelled **pretrained on published proofs**. A labelled check of
the PRETRAINING, as L4r was: no round, no row of a round. It is run while L4t2's outcome is with the owner because
every way on from here needs it: a pretraining recipe needs to know where more passes stop helping, and a loop
arm read against a start model that more of its own pretraining still improves cannot be read.

**Why.** `rehearse_only` trained `pre_r64` again on 3,422 of the 24,866 published proofs it had seen once (16
minutes) and is the broadest and most reliable model measured. Two explanations, which the dose tells apart:
MORE TRAINING HELPS (the gain should grow with the rows), or ANY SHORT FURTHER TRAINING HELPS (a model that ended
its one pass at a decayed learning rate is moved a little, and the gain is flat in the rows).

**What runs.** Two models, each trained FROM `pre_r64` at its rank, one pass over its rows, the arm's recipe (as
`rehearse_only` was), in the content-hash order, and measured as L4t's models were:
- `second_half`: the first 12,433 rows of the hash order L4t's rehearsal draw uses (so `rehearse_only`'s 3,422 are
  its first rows);
- `second_full`: all 24,866 rows of the pretraining file, in that order.
With `rehearse_only` (stored, read and never trained again) that is the dose at 14%, 50% and 100% of a second
pass along ONE order. The rows' text is never stored in the run (L4t's rule).

**The read, fixed before the run.** On all of G (392 problems, the 93 attempts), successes per attempt, paired by
problem, 95% bootstrap over problems:
- **Primary: `second_full` minus `rehearse_only`.** Interval above zero: MORE OF THE SECOND PASS HELPS (and a
  third pass is specified next). Interval holds zero: THE FIRST SEVENTH GAVE WHAT A SECOND PASS GIVES (the gain is
  not the amount of training; the recipe is one pass and a short second one). Interval below zero: MORE COSTS.
- **Overfitting, read on every branch and for both models against `pre_r64`:** NARROWER (lost more goal problems
  than gained in the 93 attempts, sign test under 0.05); the three rungs (an interval wholly below zero on a rung
  is named); the share of distinct attempts on G. (No held-out problem is in the training text, so a model
  cannot copy a goal problem's proof: what more passes can cost here is breadth, and that is what is read.)
- **Checks first** (any FAIL: INCONCLUSIVE for that model): L4t's four (the training ran; it still writes proofs;
  Lean answered; no barred row: every row is of the `pretrain` half and none is held out).
- **Secondary:** the three reads of L4t2 for each model; `second_half` against `rehearse_only` and `second_full`
  against `second_half` (the shape of the dose); by the length of the shortest published proof; solved reliably;
  the lines of verified proofs; the goal problems nothing stored had solved.
- **What it cannot say.** Whether MORE DATA would do what more passes do (the `loop` half's published proofs are
  untouched, and using them ends the separation every loop arm stands on: that is the owner's decision). One seed.
- **Budget.** Trainings of about 58 and 117 minutes (2.25 seconds an optimizer step, `rehearse_only`'s own), two
  measurements of about 50 minutes with 28 pool workers: about four and a half hours.

**Added the same day, after the owner's question and before any run: a third model, trained and measured FIRST,
`matched_1262`.** The owner (2026-10-09): "I just want to make sure that this is actually creating a feedback
cycle that is able to push the model beyond capabilities that we're able to get through pre-training alone."
L4t2 left `reward_rehearse` (the rounds' hard rows and 1,262 published rows) without its own control.
- `matched_1262`: the first 1,262 rows of the same hash order and nothing else, from `pre_r64`, as the two others
  are. They are EXACTLY the published rows of L4t2's `reward_rehearse` (held, by id and place). It is also the
  dose at 5%.
- **The read (the owner's question at the size of one pass: do the model's own proofs of hard problems add
  anything that the same published rows do not?).** On G′ over the 61 fresh attempts, `reward_rehearse` (stored)
  minus `matched_1262`, paired by problem, 95% bootstrap. Interval above zero: THE ROUNDS' HARD ROWS ADD BESIDE
  THE SAME PUBLISHED ROWS. Interval holds zero: NOT SEPARATED AT THIS SIZE (it resolves about 0.006). Interval
  below zero: THEY COST. Beside it, deciding nothing: the same difference on all of G over the 93 attempts, the
  goal problems solved (gained, lost, the sign test), solved reliably, and the never-solved.
- **Two jobs, one run directory.** `matched_1262` and this read are a job of their own (about 70 minutes), so the
  read arrives before the two long trainings; the dose job then finds that model done and goes on. The dose's
  primary is NOT READ in the first job's report.
- It decides no arm by itself: one pass on stale rows resolves about 0.006, and the six-round arm with its own
  separation (L4c) is the test of the cycle.

**Made exact by the build (2026-10-09; nothing queued, nothing run on a GPU, nothing sent to Lean; the three models,
the two reads and their branches above are not changed; ONE fact of the section's "Why" was found wrong, and what
this size can see is said, both below).**
- **The stages.** `ladder_l4_dose_r64` (`gpu/ladder_l4_dose.py`; rules `domain/ladder_round/l4_dose.py`; report
  `reporting/ladder_l4_dose.py`), settings `ladder_loop.l4.dose` (`start: pre_r64`, `models: [matched_1262,
  second_half, second_full]`; one reader; a start that is not L4t2's own is refused: the dose is of one start model).
  It names NO variable. Run directory `ladder_l4_dose_pre_r64_seed<N>`. Eight steps, each resumable: prepare; train
  and measure `matched_1262`, then `second_half`, then `second_full`; report. **Two jobs, one run directory:**
  `ladder_l4_dose_r64_matched` is the prepare step, `matched_1262` trained and measured, and
  `ladder_l4_dose_report_matched`; the whole stage, queued after it, finds those three steps done, goes on, and writes
  the whole report (queued alone it runs all three). Jobs `ladder_l4_dose_r64_matched_seed0.json` (est_minutes
  90), `ladder_l4_dose_r64_seed0.json` (420) and `ladder_l4_dose_r64_smoke.json` (25): job configs, not shipped in
  this copy.
- **What is read, and refused before anything is written.** TWO runs, only read. L4t2's
  (`ladder_l4_rows2_pre_r64_seed<N>`), FINISHED: the markers of its prepare and report steps, its report (one not to
  be read is refused), its problems, its held-out groups, its G′, and for EACH of its two models (`rehearse_only`,
  `reward_rehearse`) its stored training set, what it wrote, and its per-problem rows with each set's marker. The rank
  check's, as L4t2 reads it (its report, its three markers, `pre_r64`'s rows, `l4_pretraining_rows.jsonl`). Held to
  each other, each a refusal: L4t2's seed and start, and both models among its models; the rank and alpha L4t2
  trained at and the check's; how each was measured; G′, made again from `pre_r64`'s first sampling, which must be
  the one L4t2 stored; the pretraining file's SHA-256 (the snapshot's, the check's record and L4t2's record are one);
  each stored set's problems, seed, attempts, and Lean having answered. No adapter but `pre_r64`'s is read.
- **The three sets** (`l4_dose.dose_sets`). All are L4t's own draw (`l4_rows.rehearsal_rows`, the same call with
  another count): `second_full` is the draw of EVERY row, `second_half` the draw of half of them (`rows_of`),
  `matched_1262` the draw of AS MANY ROWS AS L4t2's STORED `reward_rehearse` SET HAS PUBLISHED ROWS (1,262 at seed
  0: the count is that set's and is written nowhere as a number; the model keeps its name whatever the count). HELD,
  not assumed, each a refusal: `second_full` holds every row of the file exactly once (by place and by id);
  `second_half`'s rows are the first of `second_full`'s; the rows L4t2's run stored for `rehearse_only` are
  rehearsal rows and no other, are not more than `second_half`'s, are the first of the order BY ID AND BY PLACE IN
  THE FILE (another file or seed is refused), and stand in the content-hash order; `reward_rehearse` has at least
  one published row and not more than `rehearse_only` has rows; `matched_1262`'s rows are the first of the order,
  are EXACTLY the published rows of `reward_rehearse` by id and by place, and those stand in the content-hash order
  (so `matched_1262` is that set with its rounds' rows left out and none moved). Then `assembly.training_order` for
  each. Each row is numbered by its place in its own set. A row that carries a text field is refused
  (`l4_rows2.refuse_text`), and a set with a held-out problem or a row that is not of the `pretrain` half
  (`l4_rows.refuse_barred`); the training step reads the text from the snapshot's file, held to the SHA-256 the START
  model's run recorded.
- **Trained and measured by L4t's own two step bodies, not by copies** (`ladder_l4_rows.train_a_model`,
  `measure_a_model`, `read_the_trained`, under this stage's `Stage`), as L4t2's two models were: the adapter saved is
  read back and held to rank 64 and alpha 128, the model server is started with that rank, the temperature is the
  config's.
- **The report**, in order: each measured model's four checks (L4t's functions); THE ROUNDS' HARD ROWS
  (`l4_dose.matched_branch`: on G′ over the fresh attempts, the stored `reward_rehearse` minus `matched_1262`; then,
  beside it and deciding nothing, the same difference on all of G over all the attempts, the goal problems solved
  with the sign test, those each solves reliably, and the never-solved when their file is given); the dose's primary
  and its branch (`l4_dose.dose_branch`); the overfitting reads of each model against `pre_r64` (`l4_dose.overfitting`:
  NARROWER by `l4b.breadth_beside`, which L4t2's read NOT NARROWER is the reverse of on the same count; the rungs
  from the model's own secondary reads); the dose (all of G against `pre_r64` for the FOUR doses, `rehearse_only`'s
  from the rows L4t2's run stored; then its three steps, each dose minus the one before it); L4t2's three reads of
  each model (`l4_rows2.three_reads`); L4t's secondary reads of each model BY ITSELF against `pre_r64`
  (`secondary_reads`: G′ over the fresh attempts, by length, solved by attempts alone and reliably, the rungs,
  distinct attempts, the lines of verified proofs, the never-solved ids from `l4_dose_never_solved.jsonl` when given,
  the rows it was trained on), `pre_r64` and `rehearse_only` beside, and `reward_rehearse` beside `matched_1262`.
- **The report of the first job** (`ladder_l4_dose_report_matched`) is the same report, in the same file, on every
  model of the stage that is trained and measured by then: after the first job, `matched_1262` alone. A model that
  is not measured yet is NOT MEASURED YET, never INCONCLUSIVE; the dose's primary and every read that stands on such
  a model is NOT READ and says so (its branch reads NOT READ with the model named); nothing of it is computed. What
  stands on `matched_1262` and the stored models alone is read there: its checks, the rounds' hard rows, its
  overfitting reads, its three reads, its secondary reads, its line of the dose and the first step (`rehearse_only`
  minus `matched_1262`). The whole stage's own report step REFUSES while a model's steps are not done, and its
  report replaces the first job's; the first job queued again after the whole stage ran writes the whole report.
- **Readings the section left open, and the ones taken.** (1) **"The first 12,433 rows" is HALF THE FILE, by rule**
  (the first ⌊n/2⌋ of the order; 12,433 of 24,866), not a number in the config: the smoke run then exercises the rule
  itself. (2) **"In the content-hash order" / "in that order".** The hash order of the DRAW says WHICH rows a dose
  holds (each dose's rows are the first of the next one's). Every model is TRAINED in the arm's content-hash order
  (`assembly.training_order`, another label of the same hash), as `rehearse_only` was: so a smaller dose's rows stand in
  the larger one's training order with the others left out, and a larger training does NOT begin with the smaller
  one's rows. (3) **L4t's run is NOT read.** What this stage needs of it (the problems, the groups, G′) L4t2's run
  stored after holding it to L4t's, and G′ is made again here from `pre_r64`'s own rows. (4) **A rung is named** when
  its interval's high end is below zero; one that ends at zero is not wholly below it. (5) **Beside a model in its
  tables** stand `pre_r64` and the stored models its reads are against; L4t's three models do not (they are not
  doses). (6) **The dose's table** (all of G against `pre_r64` for the four doses) stands with the steps the section
  lists; it names no outcome. (7) **`matched_1262` is a dose, so the shape has a third step,** `rehearse_only` minus
  `matched_1262`, which the first job's report already reads. (8) **"The goal problems solved (gained, lost, the sign
  test)" beside the rounds' hard rows** is `reward_rehearse` against `matched_1262`, gained being by the first and not
  by the second.
- **Cases the section did not name, named before any run.** (1) THE PRIMARY STANDS ON `second_full` AND
  `rehearse_only` ALONE: a failed check of `second_full` makes it INCONCLUSIVE; a failed check of `second_half` or of
  `matched_1262` makes THAT model INCONCLUSIVE by itself (none of its reads is made, and the steps of the shape that
  stand on it are NOT READ) and the primary is still read. (2) THE ROUNDS' HARD ROWS STAND ON `matched_1262` AND
  `reward_rehearse` ALONE: a failed check of `matched_1262` makes them INCONCLUSIVE; no other model's does. (3) A
  stored model that L4t2's OWN report read as INCONCLUSIVE: what is read against it is NOT READ (the primary for
  `rehearse_only`, the rounds' hard rows for `reward_rehearse`; at seed 0 both pass their checks). (4) Nothing
  measured on both sides: NOT READ. (5) An interval that holds zero: the line gives what THIS difference resolves,
  half the width of its interval (the section's "about 0.006" is said by the run's own figure). (6) THE FIRST SEVENTH
  is the branch's name whatever share `rehearse_only` is (a smoke run's is a third). (7) OVERFITTING NAMES NO BRANCH
  OF ITS OWN and does not move the primary's: NARROWER and each rung named are said in a line a model and in the
  headline, on every branch; an INCONCLUSIVE model's are not read. (8) A file of an odd number of rows: half, rounded
  down. (9) A run with one sampling of G (the smoke run): the primary is read on that sampling's attempts; the
  rounds' hard rows and ADDS have no fresh attempts and are NOT READ. (10) An INCONCLUSIVE or NOT READ outcome keeps
  what was measured apart (`measured_and_not_read`) and prints none of it.
- **A fact of the section's "Why" that the code does not bear out.** The second explanation says a model "ended its
  one pass at a decayed learning rate". THE RECIPE HAS NO DECAY: every training here and `pre_r64`'s own pass warm up
  over 5 optimizer steps and then hold 1e-4 to the end (`ladder_dose._train_with_readings`: `min(1, (step + 1) /
  warmup_steps)`). What a further training from the stored adapter has that the pass going on would not: a fresh
  optimizer (its running averages start at zero), five warm-up steps, another order of the rows, and rows seen a
  second time. The two explanations and what tells them apart (the gain grows with the rows, or is flat in them) stand
  as written; only that mechanism for the second is not the recipe's.
- **What this size can see, said before the run.** THE DOSE: on the stored rows `rehearse_only` minus `pre_r64` on all
  of G is +0.00680 [−0.00233, +0.01569] (86.82 against 80.01 per 1,000; 252 goal problems solved against 245, gained
  35, lost 28): half the width of that interval is 0.0090, and L4t2's `reward_rehearse` minus `rehearse_only` on all
  of G had 0.0088. THE PRIMARY RESOLVES ABOUT 0.009 PER ATTEMPT, which is more than the whole gain being dosed
  (0.0068, itself not clear of zero on all of G; it is on G′). So `second_full` would have to add about 0.009 per
  attempt ON TOP of `rehearse_only` (more than twice its gain over `pre_r64`) to read MORE OF THE SECOND PASS HELPS; a
  second pass that merely doubled the gain would most likely read THE FIRST SEVENTH. The read is not changed (it is
  fixed above); a holds-zero outcome is to be read with the size its own line gives, and the shape beside it (the
  steps, G′ in the three reads). THE ROUNDS' HARD ROWS: on the stored rows `reward_rehearse` minus `pre_r64` on G′
  over the 61 fresh attempts is +0.01076 [+0.00470, +0.01819] (19.13 against 8.37 per 1,000), half a width of
  0.0068: the section's "about 0.006". `rehearse_only` (2.7 times the published rows) reads +0.00786 there, so a
  `matched_1262` near it leaves about +0.003, well inside what the read resolves: NOT SEPARATED AT THIS SIZE is the
  likely outcome unless 1,262 published rows give much less than 3,422 do.
- **The smoke stage** (`ladder_l4_dose_r64_smoke`) runs the WHOLE stage's eight steps, reads `ladder_l4_rows2_r64_smoke`
  and `ladder_l4_rank_smoke` and nothing else, and writes `ladder_l4_dose_r64_smoke`. ITS SIZES ARE THE RULE'S OWN, no
  variable names them: on the fixture's 12 rows `matched_1262` is the first 2 (the 2 published rows of L4t2's smoke
  `reward_rehearse` set), `second_half` the first 6 and `second_full` all 12, with `rehearse_only`'s 4 between: four
  doses that differ, so every nesting and every row once are held on rows that differ. The smoke `pre_r64`'s own
  report reads INCONCLUSIVE, which a real run is refused for: this stage sets no variable for it and goes on ONLY
  because L4t2's smoke run, which it reads, recorded that IT went on, and records that. The first job's stage has no
  smoke job: its steps are the whole stage's first three and a report with two models not given, which the stage's
  tests run.
- **Measured before any run, on the pulled copies** (seed 0; copies in a scratch store; no GPU, no Lean). The prepare
  step runs to its end: `matched_1262` 1,262 rows, 158 optimizer steps; `second_half` 12,433 rows, 1,555;
  `second_full` 24,866 rows, 3,109 (`rehearse_only` 3,422 and 428; `reward_rehearse` 2,611 rows, 1,349 of the rounds
  and 1,262 published). `matched_1262`'s rows ARE the 1,262 published rows of the stored `reward_rehearse` set, by id
  and place, in that set's order with its rounds' rows left out; they are the first 1,262 of `rehearse_only`'s 3,422
  in the draw's order, by id and place, which are the first 3,422 of `second_half`'s, which are the first 12,433 of
  `second_full`'s; `second_full` holds every row of the file once (24,866 places, ids and problems); no row carries
  text, and the text of every row of the three sets is read from the snapshot's file at the recorded SHA-256; no
  barred row; G′ 192; rank 64; the two runs it reads not written to. The report built with `pre_r64`'s own rows
  standing in for the three models gives the two stored lines above, and with `matched_1262` alone given it reads the
  dose's primary NOT READ and the rounds' hard rows. The smoke stage's prepare step runs to its end on the pulled
  copies of the two smoke runs (2, 6 and 12 rows, `rehearse_only`'s 4 between; the waiver read from L4t2's smoke
  run; L4t2's smoke `reward_rehearse` is INCONCLUSIVE in its own report, so the rounds' hard rows are NOT READ
  there). THE PULLED COPY OF L4t2's REAL RUN is the copy of the task that resumed it and lacks the markers the first
  task wrote (its prepare step's, and `rehearse_only`'s training, measure step and three sets): for the pre-flight
  they were made again from that copy's own step summaries, which hold them key for key; the box has the originals,
  which the resumed task read.
- **L4t and L4t2 write what they wrote.** No function of theirs is changed (the stage is three new modules, three
  stages in `runner/entry.py`, its steps in `gpu/__main__.py`'s table, one block of the config). In the scripted world,
  with the code before the build and after it: L4t from `pre` the same 237 files (79 its own), from `pre_r64` the same
  300 (100), L4t2 the same 199 (61), every step's summary the same (paths and wall-clock figures aside); and after
  this stage ran beside them their own tasks queued again return what is stored and move no file (asserted). Two
  assertions that state where things live were brought up to date (one more smoke stage reads the rank check's smoke
  run; three more job files begin `ladder_l4_`).
- **Not shown without the box:** a training of more than 856 optimizer steps from a stored adapter (L4t's `rehearse`,
  the longest so far; `pre_r64`'s own pass, from the base, was these 3,109 steps in 7,062 s at 12.32 GB allocated and
  12.82 GB reserved, at the cap; since f0faa6624 the cache is released between rows); that the minutes hold with fewer
  than 28 pool workers (a measurement takes about twice as long with 15: 90 for the first job leaves about 30 over
  one measurement of 50); the whole stage finding the first job's three steps done in the box's store (asserted in
  the scripted world, where a step that is done returns what is stored). The smoke run is the first run of the stage.
- **Mutated.** Two hundred and six one-change mutations of what this build added (the three sets and their order;
  every nesting and what holds it, by id and by place; every row of the file once; `matched_1262`'s count, its rows
  and its order; the rounds' hard rows' three outcomes, their edge at zero, their two sides, their set and attempts,
  and what stands beside; the primary's three branches, its edge at zero, its two sides, its set and attempts; whose
  checks decide which read; NARROWER, the naming of a rung and the distinct shares; the dose's table and its steps; a
  model that is not measured yet, and the two report steps; what the prepare step holds L4t2's run and the start
  model's to; the stage's names handed to L4t's two bodies; the three stages; and sixteen changes to L4t's and L4t2's
  OWN code, which this stage's tests must notice because it stands on it). The first build's tests (two models)
  noticed 129 of its 132 as written, and the amended build's 204 of its 206; five assertions were added for the five
  missed (`second_full` held to every row once by PLACE as well as by id; NARROWER counted against the start model and
  not against `rehearse_only`; the share of distinct attempts beside a model's being the start model's; no figure of
  the rounds' hard rows given when `reward_rehearse` is INCONCLUSIVE where it was measured; a model counted as
  measured only when BOTH its steps are done), and all 206 are noticed.

**Outcome of the first job (seed 0, 2026-10-10 UTC; `ladder-l4p-RESULT.md`, part 1): THE ROUNDS' HARD ROWS ARE NOT
SEPARATED FROM THE SAME PUBLISHED ROWS, AND THE SMALLEST DOSE IS THE BEST MODEL MEASURED.** `matched_1262`'s four
checks pass. The dose's primary is not read yet (`second_half` and `second_full` are not measured).
- **The owner's question at one pass: NOT SEPARATED AT THIS SIZE.** On G′ over the 61 fresh attempts
  `reward_rehearse` minus `matched_1262` is −0.00111 [−0.00709, +0.00487] (19.1 against 20.2 per 1,000). Beside it,
  deciding nothing: on all of G −0.00875 [−0.01778, +0.00027] (90.4 against 99.2); goal problems solved 247 against
  264, gained 22, lost 39 (p = 0.04); solved reliably 118 against 134. On the stored rounds, one pass, the model's
  own proofs of hard problems add nothing that the same published rows do not, and cost breadth.
- **`matched_1262` against `pre_r64`:** all of G +0.01915 [+0.01037, +0.02790] (99.2 against 80.0 per 1,000); goal
  problems solved 264 against 245 (gained 44, lost 25, p = 0.03); 134 solved reliably against 107; 4 lines or more
  +0.02272 [+0.01164, +0.03427]; its three reads all pass; no rung below zero. 158 optimizer steps.
- **The dose is not monotone:** `rehearse_only` (3,422 rows) minus `matched_1262` (1,262) on all of G is −0.01234
  [−0.02057, −0.00398]. "More training helps" does not describe it. NOT UNDERSTOOD, and one thing is NOT MEASURED:
  how far two trainings of one recipe differ. Every interval of L4t, L4t2 and L4p is over problems only; a
  difference between two single trained models is not yet shown to be more than that spread.
- **A measurement check made on this result, and cleared.** `pre_r64`'s stored attempts hold more Lean timeouts
  (4.3% of its goal attempts) than any model measured since (2.3% to 3.5%), and one pool server answers many times
  slower when other jobs load its machine: a proof that times out is a failed attempt, so the stored baseline
  could have been low. Every timed-out goal attempt of both models was checked again, Lean only: 6.9% of
  `pre_r64`'s 1,583 and 5.6% of `matched_1262`'s 1,278 verify on the second check (26% and 19% get an answer). With
  those counted on both sides: 83.0 and 101.1 per 1,000, and the difference +0.01810 where the stored rows give
  +0.01915. THE STORED BASELINE IS LOW BY ABOUT 0.001 AGAINST A LATER MODEL, and no more.
- **What follows.** By the owner's word of 2026-10-09 the arm of L4c (`reward_rehearse`, six fresh rounds from
  `pre_r64`, with its own `rehearsal_alone` control) is run next, and the dose's two long trainings after it.

**Outcome of the whole stage (seed 0, 2026-10-10 UTC; `ladder-l4p-RESULT.md`, part 2): INCONCLUSIVE BY ITS OWN
CHECK, and the figures say a whole second pass is the best model measured.** `second_half`'s checks pass.
`second_full` fails "its training ran": the mean loss over the last tenth of its rows (0.1385) is above the first
tenth's (0.1271) by more than two standard errors. So the primary is not read as a branch.
- **The check was the wrong one for a second pass, said plainly.** It was written for trainings on rows a model
  has not seen, where a loss that rises means the training did not take. On a second pass the first rows are
  ones `pre_r64` saw last and fits best, and the loss on already-seen rows RISES across the pass while proof
  success rises with it (the third time here that the loss on proof text and proof success part ways). The
  verdict stands as written; the figures below are read outside it and name no branch.
- **All of G, per 1,000 attempts, goal problems solved in 93, solved reliably, of the 81 never solved:**
  `pre_r64` 80.0, 245, 107, 7; `matched_1262` (5% of a pass) 99.2, 264, 134, 11; `rehearse_only` (14%) 86.8, 252,
  124, 11; `second_half` (50%) 99.3, 255, 124, 10; `second_full` (100%) 108.9, 267, 139, 13. (Of the 81, 37 are
  now solved by something and 44 by nothing.) On G′ over the 61
  fresh attempts: 8.4, 20.2, 16.2, 17.6, 26.9. `second_full` minus `rehearse_only` on all of G, outside the
  rule: +0.0221.
- **The spread between two trainings is now measured, roughly, and it is large.** Five models trained from
  `pre_r64` on published rows alone (the four doses and L4c's `rehearsal_alone`: 95.4, 258, 135) are not in the
  order of their doses: 99.2, 95.4, 86.8, 99.3, 108.9. Leaving the whole pass out, four trainings that differ
  only in which rows and how many have a standard deviation of about 6 per 1,000 on all of G, 2 on G′ and 5
  problems solved. NO DIFFERENCE BETWEEN TWO SINGLE TRAINED MODELS OF L4t, L4t2 OR L4p SMALLER THAN ABOUT TWICE
  THAT IS AN EFFECT. What stays outside it: the published-only models against those that hold rows of the
  rounds (problems solved 252 to 267 against 217 to 247), and the whole second pass on G′.
- **What it says for the loop.** The pretraining-only bar is not 80.0 and 245: with one more pass over the same
  published proofs it is 108.9 and 267, and it has not been shown to stop there.

**Outcome of L4c (seed 0, 2026-10-10 UTC; `ladder-l4c-RESULT.md`): THE LOOP ADDS ON TOP OF PRETRAINING; APART FROM
THE REHEARSAL, THE GAIN IS NOT SHOWN TO BE THE ROUNDS'.** It was run although L4t2's rule read "no arm with
rehearsal rows": that rule stood on such an arm being unreadable, this arm carries its own control, and the owner
asked for the test. All five checks pass, and `rehearsal_alone`'s three; the three models were measured one after
another on one pool.
- **Primary:** `with` minus `pre_r64` on G′ over the 61 fresh attempts +0.00478 [+0.00060, +0.00973] (13.2 against
  8.4 per 1,000).
- **The separation:** `with` minus `rehearsal_alone` on the same attempts −0.00615 [−0.01101, −0.00162] (13.2
  against 19.3), and `rehearsal_alone` itself adds (+0.01093 [+0.00555, +0.01699]). The primary is reported with
  this line and is not called the loop's gain.
- **Breadth and the long proofs:** against `pre_r64` not NARROWER by the test (229 to 245, gained 29, lost 45, p =
  0.08) and not SHORTER (+0.0111 [−0.0002, +0.0228] on 4 lines or more). Against its own control: 229 goal problems
  solved to 258, gained 19, lost 48 (p = 0.001); solved reliably 120 to 135; 8 lines or more solved 20 to 32.
- **The assembled proofs do not teach** (`with` minus `without` on G′ −0.0013 [−0.0032, +0.0008]; 106 proofs).
- **What the read does not hold fixed:** the control took 189 optimizer steps and `with` 391, and among the
  published-only models more steps have meant less. Against the published-only model nearest in steps
  (`rehearse_only`, 428) `with` still solves fewer goal problems (229 to 252) and is lower on G′ (13.2 to 16.2),
  and is higher per attempt on all of G (92.2 to 86.8).
- **Not a KILL of the idea.** A positive control cleared the same reads through the same harness; the budget
  (about 48,000 attempts, 8 a problem) is a small fraction of the loops known to move a model of this size, and
  8 attempts a problem decide what a round can find: a proof the model could already write. AT THIS SIZE AND WITH
  THESE ROWS the rounds' own proofs sharpen the model and narrow it, twice (the stored rounds in L4p, six fresh
  rounds here).

### L5: do DEEP solves teach what ordinary solves do not? One step-matched check (fixed 2026-10-10, before any run)

**Under the owner's word of 2026-10-09:** "Perhaps as results come out of that model and it is able to solve a
problem that we didn't previously see a solution to, we could pre-train on that? I just want to make sure that
this is actually creating a feedback cycle that is able to push the model beyond capabilities that we're able to
get through pre-training alone." Labelled **pretrained on published proofs** (it starts from `pre_r64`). A scout:
one seed, no round.

**Why.** L4p and L4c read the same thing twice: the model's own proofs, found within 8 attempts, sharpen it and
narrow it, and the published rows beside them do the adding. A proof found within 8 attempts is one the model
could already write. The owner's idea is about the other kind: a problem the model had NOT solved. This asks
whether such solves teach, with the two things L4c's read did not hold fixed held fixed: the optimizer steps and
the published rows.

**What runs.**
1. **The deep search.** The problems of `pre_r64`'s own map that it failed in all 8 attempts (1,985 of the 4,000;
   base-map problems, so no model here has seen their published proofs and no round draws them). A seeded draw
   of 1,000 of them; `pre_r64` makes 56 more attempts at each (64 with the map's 8), the round's episode in every
   other respect (the same prompt, limits, one side by the audit rule, Lean's verdict). A problem solved here is
   a DEEP solve: its training row is one of its verified attempts, drawn by the seed, as a round draws one. No
   assembly, no round, no challenger. Budget: 56,000 attempts.
2. **Three models, each FROM `pre_r64` at its rank, one pass, the same number of rows (2D) and so of optimizer
   steps,** where D is the number of deep solves (capped at 400; a run with fewer than 150 is refused: below):
   - `deep`: the D deep rows and the first D rows of the rehearsal's hash order (published proofs);
   - `shallow`: D rows of ORDINARY solves and the same D published rows. An ordinary solve is a map problem
     `pre_r64` solved in 1 to 3 of its 8 map attempts (1,114 problems): a seeded draw of D of them, one verified
     attempt each, drawn by the seed;
   - `published`: the first 2D rows of the hash order and nothing else.
   Each set in the content-hash order; no published row stored with text. Each model measured as L4t's models
   are (the rungs; G twice with `pre_r64`'s recorded sampling seeds), one after another in one task.

**The read, fixed before the run.** Every difference paired by problem, 95% bootstrap over problems.
- **Checks first** (any FAIL: INCONCLUSIVE, nothing else said). The deep search found at least 150 deep solves
  (fewer: the three sets would be too small to read; the search is then the finding, and it is said). Each
  training ran; each model still writes proofs; Lean answered (at most 2% of a set without an answer); no barred
  row (no held-out problem; the published rows of the `pretrain` half).
- **Primary: `deep` minus `published` on G′ (the 192 goal problems `pre_r64` does not solve in its first 32
  attempts) over the 61 fresh attempts.** Interval above zero: DEEP SOLVES ADD WHAT THE SAME NUMBER OF PUBLISHED
  PROOFS DO NOT. Interval holds zero: NOT SEPARATED AT THIS SIZE (it resolves about 0.006). Interval below zero:
  THEY COST.
- **Breadth, on every branch:** the goal problems solved in the 93 attempts, `deep` against `published` (gained,
  lost, the two-sided sign test); lost more than gained under 0.05 is written NARROWER beside the branch.
- **The mechanism: `deep` minus `shallow`,** the same two reads. It is the claim under test: that a solve the
  model could not sample extends it where one it could does not. Above zero on G′, or more gained than lost
  under 0.05: DEPTH MATTERS. Otherwise NOT SHOWN.
- **The replication L4c owes: `shallow` minus `published`,** the same two reads, at equal steps and equal
  published rows.
- **Secondary:** all of G per attempt and by the length of the shortest published proof; solved reliably; the
  three rungs; the never-solved when given; each against `pre_r64` (stored); the deep search itself (problems
  solved by the attempt of the first success: 9 to 16, 17 to 32, 33 to 64; the lines of the deep rows beside the
  ordinary ones').
- **What it cannot say.** One pass and no round: whether deep solves COMPOUND (a model trained on them reaching
  deeper ones) is the loop this would license, not this check. One seed: between two trainings of one recipe
  the goal set has moved by about 0.004 per attempt and 6 problems (L4p's and L4c's published-only models).
  A depth of 64 is shallow against a search of thousands.
- **Branches for what follows.** DEEP SOLVES ADD, or DEPTH MATTERS with the primary not below zero: two more
  seeds, then a loop whose rounds ARE deep searches. THEY COST and NARROWER: the model's own proofs, deep or not,
  are not rows to train on at this size, and that goes to the owner with the three results together. Otherwise:
  not separated, and the depth is the thing to raise before the seeds.
- **Budget.** The search about two and a half hours with 15 pool workers (about 380 attempts a minute, the
  arm's own rate); three trainings of a few minutes; three measurements of about 90 minutes: about seven and a
  half hours.

**Amended the same day, after L4p's whole stage was read and before any run of L5: THE PRIMARY MOVES TO A LARGER
SET, the held-aside hard problems.** Why: L4p measured the spread between two trainings of one kind (about 0.002
per attempt on G′ and 5 goal problems, one standard deviation), and G′ is 192 problems, where a difference
resolves 0.006. Three models of D rows or 2D would most likely all read NOT SEPARATED there whatever is true.
- **The held-aside set, HA:** the candidates of the deep search that the draw did NOT take (985 of the 1,985;
  a seeded draw of `deep.held_aside` 600 of them). They are problems `pre_r64` failed 8 of 8, of the same pool
  and kind as the ones searched, and no model here trains on them or has seen their published proofs. They are
  the cycle's own test: after training on deep solves of some hard problems, does a model solve OTHER hard
  problems it could not?
- **One more measurement for each of FOUR models** (`pre_r64` itself, `published`, `shallow`, `deep`): 16 fresh
  attempts at each problem of HA, one sampling seed of its own shared by the four, the round's episode. `pre_r64`
  is measured again there because its 8 stored map attempts CHOSE the set (all failures by construction).
- **PRIMARY (this replaces the primary above; G′ is read beside it as the held-out goal set's word):** on HA,
  successes per attempt, `deep` minus `published`, paired by problem, 95% bootstrap over problems, the same three
  names (DEEP SOLVES ADD WHAT THE SAME NUMBER OF PUBLISHED PROOFS DO NOT / NOT SEPARATED AT THIS SIZE / THEY
  COST); and beside it the problems of HA solved at least once, gained and lost, the sign test (NARROWER as
  above).
- **The mechanism and the replication are read on HA too** (`deep` minus `shallow`; `shallow` minus `published`),
  by the same two reads, and on G′ and the goal problems as written above, beside.
- **Each trained model against `pre_r64` on HA** (successes per attempt and problems solved) is secondary: it is
  what "more hard problems solved after the training" is, for each kind of row.
- The selection is the same for the three trained models (each is `pre_r64` moved by a training of equal steps),
  so the three differences among them are fair; a trained model against `pre_r64` on HA is favoured by the
  selection, as on G′, and is said so.
- **Budget:** 38,400 more attempts (about an hour and three quarters with 15 pool workers): about nine and a
  quarter hours in all.

**Made exact by the build (2026-10-10; nothing queued, nothing run on a GPU, nothing sent to Lean; the read above and
its amendment are not changed; the readings taken where the section left something open, two figures of the section
that the stored runs put otherwise, and what is fragile, are below).**
- **The stages.** `ladder_l5_deep` (`gpu/ladder_l5.py`; rules `domain/ladder_round/l5.py`; report
  `reporting/ladder_l5.py`), settings `ladder_loop.l5` (`start: pre_r64`, which must be `l4.rows2`'s own start;
  `models: [published, shallow, deep]`; `deep`: `problems` 1,000, `attempts` 56, `sampling_place` 50, `held_aside` 600,
  `held_aside_attempts` 16, `held_aside_sampling_place` 51, `cap` 400, `minimum` 150; `shallow`: `solved_at_least` 1,
  `solved_at_most` 3; one reader). It names NO variable. Run directory `ladder_l5_deep_pre_r64_seed<N>`. FOURTEEN steps,
  each resumable: `ladder_l5_search_prepare`; `ladder_l5_search`; `ladder_l5_prepare` (the three sets);
  `ladder_l5_held_aside_start`; then for `published`, `shallow`, `deep` in turn `ladder_l5_train_<m>`,
  `ladder_l5_measure_<m>`, `ladder_l5_held_aside_<m>`; `ladder_l5_report`. Jobs `ladder_l5_deep_seed0.json`
  (est_minutes 660) and `ladder_l5_deep_smoke.json` (30): job configs, not shipped in this copy.
- **What is read, and refused before anything is written.** THREE runs, only read. `pre_r64`'s own map
  (`ladder_l4_map_pre_r64_seed<pretraining seed>`, by `ladder_l4_start.the_start`): its step's marker (the model it
  recorded; its attempts and sampling seed held to the stored map's by `l4.check_start_map`, as an arm holds them), its
  file, its per-problem rows (held to the file), its own `problems.jsonl` (the base map's problems with their exact
  negations AS THE MAP SAMPLED THEM: held to the snapshot's statements; no exactness check is made again) and its
  stored attempts. L4t2's FINISHED run and the rank check's, as L4p reads them (seed and start; the rank and alpha; how a
  model is measured; G′ made again from `pre_r64`'s first sampling and held to L4t2's; the pretraining file's SHA-256),
  without L4t2's two models: no read here stands on them. L4p's run and L4c's are not read.
- **The deep search.** The candidates are the map's rows with k = 0; their number is THE MAP'S (1,985 at seed 0) and is
  recorded, not held to a figure. The draw: the `deep.problems` of the smallest content hash of the task's seed and the
  problem's id (label `l5_deep_search`); fewer candidates than the draw is refused. `pre_r64`, served at rank 64, makes
  56 attempts at each by `ladder_loop.run_episodes` under the config's own `episode` settings (the round's), with the
  sampling seed `round.sampling_seed` + 100 × the task's seed + place 50 (1050 at seed 0): the places of every stored
  sampling are 0 to 3, 11 to 16, 20 and 30 to 33, the base's placing seed is 100, the map's 101, L4t's hot ones 1040
  and 1041, and the prepare step refuses a seed one of the samplings it can see used. ONE THING IS THE SEARCH'S OWN: a
  block holds 18 problems and not 128 (`block_of`: as many problems as make a round's block in ATTEMPTS, 128 × 8 / 56),
  because a block is what a kill loses and what waits on Lean together: 56 blocks. A search on which Lean gave no
  verdict for more than 2% of its attempts is not kept and is sampled again by the task queued again (as a map is).
- **A deep solve's row** is drawn by `training_set.training_examples`, the function a round draws its row by (the
  verified attempt first in a seeded order of attempt ids, on whichever side was proved), and carries its problem's k of
  56, the lines of its proof and `first_success`: the place of the problem's FIRST verified episode, counting the map's
  8 (9 to 64). Its origin is `deep`; an ordinary solve's row is one of the map's own stored verified attempts, by the
  same function and seed, of origin `shallow`.
- **The three sets** (`l5.the_sets`, held from the outside by `hold_the_sets`). D is the deep solves, or `deep.cap` of
  them by a seeded draw of its own (label `l5_deep_rows`, by problem). The ordinary solves: a seeded draw (label
  `l5_shallow`) of D of the map's problems with k of 1 to 3; fewer than D is refused. The published rows are
  `l4_rows.rehearsal_rows`' (L4t's, L4t2's and L4p's draw, with another count): `published` the draw of 2D, `deep` and
  `shallow` the draw of D, HELD to be the first D of the 2D by id and place (so they are the rows L4p's `matched` logic
  would take for a set with D published rows: asserted against `l4_dose.dose_sets`). Each set in
  `assembly.training_order`. Refused: sizes that are not 2D; a row of another origin; an id twice; published rows that are
  not the first of the order by id and place; a problem in both `deep` and `shallow`; an own row whose problem the
  pretraining file holds; a published row with text; a held-out problem; a published row not of the `pretrain` half;
  an own row that is no base-map problem; a row of a held-aside problem.
- **The held-aside set** (the amendment). A draw OF ITS OWN (label `l5_held_aside`) of `deep.held_aside` of the
  candidates the search's draw did not take (985 at seed 0), made by the search's prepare step; fewer left than the draw
  is refused. Four measurements, each a step of its own: 16 attempts at each problem, ONE sampling seed (place 51: 1051),
  blocks of 64 problems, by `run_episodes` as the search runs; a set Lean did not answer is measured again, and no other,
  when the task is queued again. THE START MODEL'S STEP STANDS AFTER THE THREE SETS ARE BUILT, not right after the
  search, so that a run with too few deep solves does not spend it.
- **The report**, in order: the checks; the primary on HA and its branch; G′ beside it; the mechanism on HA, G′ beside;
  the replication on HA, G′ beside; one line for what follows; each trained model against `pre_r64` on HA, with the line
  that the selection favours it, and against the stored `pre_r64` on all of G (per attempt, and the goal problems
  solved); each pair on all of G and by length; L4t's secondary reads of each model against the stored `pre_r64`; the
  deep search's table; the lines of the rows. The never-solved ids are read from `l5_never_solved.jsonl` when given.
- **Readings the section left open, and the ones taken.** (1) **Fewer than 150 deep solves: the stage does not fail.**
  Its prepare step builds no set and says why; the ten steps after it train and measure nothing (each returns that, and
  marks nothing); the report reads INCONCLUSIVE with the check named and the search's table as THE FINDING. (2) **"Any
  FAIL: INCONCLUSIVE, nothing else said" is ONE verdict for the run** (in L4p a model was INCONCLUSIVE by itself): a
  failed check of the search, of ANY of the three models, or of Lean's answers on a held-aside set; what was measured
  is kept apart (`measured_and_not_read`) and none of it is printed. (3) **"No barred row" is this stage's own fourth
  check** (`l5.no_barred_row`): L4t's bars a round's row whose id hashes to the `pretrain` half, and a base-map problem
  is in neither half's files while about half of them hash there. Here: no held-out problem; the published rows of the
  `pretrain` half; the own rows base-map problems; and, held when the sets are built, no own row's problem in the
  pretraining file. (4) **"Lean answered" for the search** is not a check of the report: such a search is not kept (above).
  For a HELD-ASIDE set it is a check, and a set that fails it also makes the report not to be read (the step fails, and
  the task queued again measures that set again). (5) **The first success counts EPISODES** (an audited problem's
  episode is an attempt at each side). (6) **The table's bands double from the map's attempts** (`l5.bands`: 9 to 16, 17
  to 32, 33 to 64 at 8 and 64). (7) **DEPTH MATTERS**: the interval on HA above zero (one that ends at zero is not), or
  gained more than lost (gained = by `deep` and not by `shallow`) with the sign test under 0.05. (8) **The
  replication's names** are ABOVE ZERO / HOLDS ZERO / BELOW ZERO, with NARROWER beside. (9) **What follows, in the
  section's order:** DEEP SOLVES ADD first, whatever the breadth says; THEY COST without NARROWER is "Otherwise", and its
  line says so (the words "not separated" there name the branch, not the primary). Nothing is named when the primary is
  INCONCLUSIVE or was not read. (10) **G′ beside** is the same pair on G′ over the fresh attempts and on the goal
  problems solved, printed after each read and stored under `beside_on_the_goal_set`; it names no branch; a run with one
  sampling of G has none. (11) **A holds-zero outcome gives what ITS difference resolves** (half the width of its
  interval), on HA.
- **Two figures of the section, as the stored map puts them.** "Budget: 56,000 attempts" is 56,000 EPISODES: the audit
  rule samples and checks the ruled-out side too for a seeded 2% of problems, by the search's own seed 11 of the 1,000
  drawn (seed 0): 56,616 attempts. The held-aside set's "38,400" is likewise episodes. The figures 1,985, 1,114 and 985
  are the stored map's.
- **Fragile, said before the run.** (a) THE CHECK "ITS TRAINING RAN" ON `published`: its rows are 2D rows `pre_r64` has
  seen once, and L4p's whole stage found that check wrong for a second pass (the loss on seen rows rises). `published`'s
  rows are the first 2D of `matched_1262`'s 1,262, whose check passed; but by reading (2) ONE failed check makes the
  whole run INCONCLUSIVE. (b) A FURTHER SEED reads L4t2's run OF ITS SEED (for the measured sets' problems, the groups
  and G′), and that run exists at seed 0 only: "two more seeds" needs a build step first (read the one L4t2 run, as an
  arm reads the one pretraining). (c) The task writes about 180 MB to its out directory (about 223,000 attempts).
- **The smoke stage** (`ladder_l5_deep_smoke`) runs all fourteen steps, reads `ladder_l4b_map_smoke`,
  `ladder_l4_rows2_r64_smoke` and `ladder_l4_rank_smoke` and nothing else, and writes `ladder_l5_deep_smoke`. THE
  FIXTURE'S MODEL SOLVED EVERY BASE-MAP PROBLEM IN ITS MAP (k = 3, 8, 8, 8): the rule gives the search no problem.
  `RLVR_LEAN_LADDER_L5_SMOKE_PROBLEMS=1`, set by this stage ALONE: the search attempts one problem the rule did not
  give, the held-aside set holds one other (one and one: four problems, and the held-aside set shares none with the
  search or a set), `shallow` takes its ordinary solve among the two left (the rule's own when it is there), one deep
  solve is enough, and every problem and row so taken is marked `taken_by: the run's minimum` and counted in the
  report's line A SMOKE RUN. NO REAL RUN CAN TAKE THAT PATH: the prepare step refuses the variable unless the run
  directory is one the task names AND both the snapshot's data and the map's own record say fixture; without the
  variable a map with no candidate, and fewer ordinary solves than deep rows, are refusals. Its models will most
  likely read INCONCLUSIVE (a training of two rows has no loss to compare), as L4p's smoke models did.
- **Measured before any run, on the pulled copies** (seed 0; copies in a scratch store; no GPU, no Lean; the map's own
  `problems.jsonl` is not in the pulled copy and was made again from the snapshot's base map and the stored rows'
  one-side reasons). 1,985 candidates; 1,000 drawn; 600 of the 985 left held aside, disjoint from the searched; 1,114
  ordinary solves. With a STAND-IN deep result (330 of the drawn problems marked solved by a made-up text): three sets of
  660 rows and 83 optimizer steps each; `deep` and `shallow` hold the same 330 published rows by id and place, the first
  330 of the order and of `published`'s 660; no published row with text; no barred row; the held-aside set disjoint from
  the 1,320 problems of the three sets; the 330 ordinary solves' proofs have a median of 7 lines (75% of 4 or more, 46%
  of 8 or more) against 7 (73%, 48%) for the published rows beside them; the three runs not written to. The smoke
  stage's prepare steps run to their end on the pulled smoke copies (1 searched, 1 held aside, D = 1, sets of 2 rows).
- **L4t, L4t2 and L4p write what they wrote.** No function of theirs is changed (three new modules, two stages in
  `runner/entry.py`, the steps in `gpu/__main__.py`'s table, one block of the config). In the scripted world, with the
  code before the build and after it: L4t from `pre` the same 237 files (79 its own), from `pre_r64` 300 (100), L4t2 199
  (61), L4p 178 (79), every step's summary the same; and after this stage ran beside them their own tasks queued again
  return what is stored and move no file (asserted). Three assertions that state where things live were brought up to
  date (one more smoke stage reads the rank check's smoke run and the map's).
- **Not shown without the box:** an episode step that serves a stored rank-64 adapter for 56 samples a prompt outside a
  measurement (a measurement samples 61); the search's rate (the budget's 380 attempts a minute is the arm's own, at 8
  samples a prompt); a training on rows of two kinds; four engines started in turn for one set; the map's own
  `problems.jsonl` on the box (read for the first time by a stage other than the map's). The smoke run is the first run
  of the stage.
- **Mutated.** One hundred and seventy one-change mutations of what this build added, each applied to a copy of the
  tree and never to the working tree (the candidates' rule; the search's draw and its refusals; the held-aside set's draw,
  that it is a draw of its own, and its two refusals; the one-verified-attempt draw and the place of the first success;
  D, the cap and its draw; the ordinary solves; the minimum's refusal in the rule and in the stage; the three sets, each
  hold of the same published rows by id and by place, the sizes and the order; what is barred; the table's bands; every
  outcome of the primary, of the mechanism and of the replication with their edges at zero; the branch for what
  follows; in the report the primary's SET (the held-aside set, not G′), each pair's two sides, DEPTH MATTERS decided on
  the held-aside set, whose checks make the run INCONCLUSIVE, and what is kept apart; in the stage the smoke-only path's
  guard in each of its forms, the two sampling seeds, the four models measured with one seed each from its own adapter,
  whose rows the report reads, the block rule; the registration; and seven changes to code of OTHER stages that this one
  stands on: the round's draw of a row, L4t's draw, its two step bodies and its checks, L4b's NARROWER). Before the
  run two things were given to the stage's tests because the first commit's could not see them: the scripted model
  answers the held-aside set differently for each of the four models (it answered alike, so a report that read every
  model from the start model's rows would have passed), and a real run with fewer ordinary solves than deep rows is
  refused. The first run (164 mutations): 154 noticed, 10 not. Two of the ten were the same program as the code (a
  looser bound that the slice beside it already held) and were replaced by mutations that change what a real run
  takes. Seven asked for assertions, which were added: a problem never solved and without its result is refused; a real
  run with NO ordinary solve is refused though the map holds problems solved more often (two mutations: the stage
  handing the rule `loosely`, and the stage reading every run as a smoke run); the deep row is drawn by the task's seed
  (the scripted search now verifies four attempts of its problem); DEPTH MATTERS by the breadth of the HELD-ASIDE set
  where the goal set's says nothing; a held-aside draw that names a searched problem is refused where it is made; and
  `deep`'s and `shallow`'s published rows are held to one order even were each set's own order not held. THE TENTH WAS
  IN L4t's OWN MEASURING BODY (every model measured from the adapter under the start model's name) and was missed a
  second time: the scripted world serves no adapter (the stand-in engine takes none), so no test of this stage saw whose
  adapter a step asks the model server for; the summaries' paths alone were held. A test AS ON THE GPU was added (the
  search and the start model's held-aside step ask for `pre_r64`'s adapter in its own run; each trained model's rungs, G
  twice and the held-aside set for the adapter THIS run trained; each under a name of its own; an adapter that is no
  longer there is refused), with three mutations of this stage's own two serving lines. Then the section's secondary
  read that the first commit's report had left out was added (each trained model against the stored `pre_r64` on all of
  G, per attempt and goal problems solved), with three mutations of its own. The last run, all 170 against the tree
  with these additions: 168 noticed, 2 not, both in that new read (read against `published` in place of the start
  model; gained and lost exchanged): the hand-made `published` stood on the start model's own rows and no model lost a
  goal problem. The read is now held on a `published` that is below the start model and loses three problems, and the
  two are noticed (they and the five beside them were run again; nothing but those assertions changed after the run, so
  the 168 were not run a third time). All 170 are noticed.

**Outcome of L5 at seed 0 (2026-10-11 UTC; `ladder-l5-RESULT.md`): NOT SEPARATED AT THIS SIZE on the primary, and a
lead on the goal set.** All checks pass. The deep search solved 520 of the 1,000 problems (D = 400; three sets of
800 rows, 100 optimizer steps).
- **Primary (HA, 600 problems, 16 fresh attempts):** `deep` minus `published` +0.00063 [−0.00604, +0.00729] (51.5
  against 50.8 per 1,000; `shallow` 48.9; `pre_r64` 36.2); problems solved 217 to 214. The mechanism (`deep` minus
  `shallow`, +0.00260 [−0.00365, +0.00885]; 217 to 204): NOT SHOWN. The replication (`shallow` minus `published`,
  −0.00198 [−0.00792, +0.00407]): HOLDS ZERO.
- **The goal set, beside, naming no branch:** all of G per 1,000 attempts `pre_r64` 80.0, `published` 103.1,
  `shallow` 104.5, `deep` 112.5; goal problems solved 245, 262, 257, 275; solved reliably 107, 146, 136, 151; of 8
  lines or more solved 25, 25, 22, 36. `deep` minus `published` on all of G +0.00941 [+0.00063, +0.01816]; `deep`
  minus `shallow` on G′ +0.00683 [+0.00307, +0.01093] and goal problems gained 39, lost 21 (p = 0.03). `deep` has
  the most goal problems solved and solved reliably of any model measured, and it is the first model trained on
  its own proofs that is not behind its published-only control in breadth.
- **Why it is a lead and not a result.** The primary was moved to HA before the run and shows nothing. Each model
  is one training, and two trainings of one kind differ by about 0.006 per attempt on all of G and 5 goal
  problems (L4p): the goal set's gaps are one to two and a half of those. The three models were measured one
  after another and the pool's main server was installed anew between them; `deep`'s goal attempts timed out in
  Lean more often (4.0% and 4.7% against `published`'s 2.5%), which counts against it if the pool was slower.
- **What follows, by the standing seed rule** (a scout whose primary is not negative goes to three seeds): two
  further seeds, below.

### L5, further seeds (fixed 2026-10-11, after seed 0's read and before they run)

**What a further seed N (1, 2) is.** The SAME search, the same held-aside set and the same stored measurement of
`pre_r64` on it (seed 0's run, only read: nothing of them is sampled again), and the stored runs seed 0 read. By
the task's seed: which 400 of the 520 deep solves are taken and which verified attempt of each; which 400
ordinary solves and which attempt; the published rows (the first D and the first 2D of the hash order of THAT
seed: another draw of the pretraining file); the order each set is trained in. The three models again, each from
`pre_r64`, measured on the rungs and on G with the sampling seeds seed 0 used, and on HA (16 attempts, seed 0's
sampling seed), so that a problem's attempts differ between seeds by the model alone.

**The reads, fixed before they run.**
- **The primary, over the three seeds:** on HA, successes per attempt, `deep` minus `published`: each problem's
  mean over the three seeds of its difference, paired, 95% bootstrap over problems; each seed's own difference
  and sign beside. Interval above zero: DEEP SOLVES ADD WHAT THE SAME NUMBER OF PUBLISHED PROOFS DO NOT. Holds
  zero: NOT SHOWN AT THREE SEEDS. Below zero: THEY COST.
- **The goal set's lead, CONFIRMED OR NOT ON THE TWO NEW SEEDS ALONE** (seed 0 found it, so seed 0 cannot confirm
  it): on all of G over the 93 attempts, `deep` minus `published`, each problem's mean over seeds 1 and 2, 95%
  bootstrap over problems. THE LEAD IS CONFIRMED when the interval is above zero AND the difference is positive
  at both seeds; otherwise NOT CONFIRMED. Beside it: the goal problems solved (gained, lost) at each seed; the
  same for `deep` minus `shallow` (the depth) and `shallow` minus `published`; and the three seeds pooled, with
  seed 0 marked as the seed that found the lead.
- **The spread, measured at last:** for each kind of model, its three seeds' figures (all of G per attempt, goal
  problems solved, HA per attempt) and their standard deviation.
- **Timeouts:** each set's share of attempts that timed out in Lean is printed; where two models' shares differ by
  more than a point, their timed-out attempts are checked again (Lean only, no measurement running) and the
  reads are given both ways.
- **Branches.** CONFIRMED on the goal set, or ADD on HA: the next stage is a loop whose rounds are deep searches,
  specified then. THEY COST on HA: stop, and the three seeds go to the owner. Otherwise: the lead is not
  confirmed, deep solves are no worse than published rows at equal steps (unlike ordinary ones in L4c), and the
  depth or the dose is the thing to raise.
- **Budget.** A seed: three trainings of a few minutes, three measurements of about 75 minutes and three of about
  25 on HA with 15 pool workers: about five hours and a quarter. Two seeds, one task each, one after the other.

**Amended 2026-10-11, after a closer look at seed 0 and BEFORE seeds 1 and 2 run: where the lead sits, and a
read of reach.** Seed 0's stored attempts, looked at again (no run; exploratory, and said as such):
- **Per attempt the lead is reliability.** 84% of `deep`'s lead over `published` on all of G sits on the 200
  goal problems `pre_r64` solves in its first 32 attempts (200.9 against 185.4 per 1,000, +0.0155 [−0.0007,
  +0.0320]). On G′, the 192 it fails, over all 93 attempts: +0.0031 [−0.0020, +0.0083]. So HA and the goal set do
  not disagree: on problems the start model fails, `deep` and `published` are level on both.
- **The sign of reach is a count, on the long problems.** Goal problems of 8 lines or more solved at least once
  in 93 attempts: `deep` 36, `published` 25 (gained 14, lost 3, p = 0.013), `shallow` 22 (gained 17, lost 3).
  Per attempt on those 74 `deep` is not above `published` (−0.0042 [−0.0169, +0.0070]): it reaches more long
  problems and each of them rarely. One slice of several looked at, one training each.

So confirming the lead on all of G would confirm mostly reliability. Two reads are added, on seeds 1 and 2 alone
like the lead's, and the lead's own rule stands as written:
- **The split.** `deep` minus `published` per attempt over the 93 attempts, each problem's mean over the two new
  seeds, 95% bootstrap over problems, (a) on the 200 and (b) on G′. A confirmed lead is named RELIABILITY when
  (b) holds zero and REACH TOO when (b) is above zero.
- **Reach on the long problems.** On the 74 goal problems of 8 lines or more: for each problem and model, the
  share of the two new seeds in which the model solves it at least once in its 93 attempts; `deep` minus
  `published`, paired, 95% bootstrap over the 74. REACH IS CONFIRMED ON THE LONG PROBLEMS when the interval is
  above zero AND `deep` solves more of them than `published` at both new seeds; otherwise NOT CONFIRMED. Beside
  it: each model's count at each seed with gained and lost, and the same read for `deep` minus `shallow`.
- **Branches, replacing "CONFIRMED on the goal set" above.** REACH CONFIRMED ON THE LONG PROBLEMS, or ADD on HA:
  the next stage is a loop whose rounds are deep searches, with ordinary solves matched in length as a control
  (depth is not yet apart from length: 64% of the deep rows have 8 lines or more, 44% of the shallow ones). The
  lead confirmed as RELIABILITY and reach not: deep solves are the rows of its own a model can train on without
  narrowing, which ordinary solves are not; reach is not shown and the depth of the search is the thing to
  raise. THEY COST on HA: stop, and the three seeds go to the owner. Otherwise as written above.

**Made exact by the build (2026-10-11; nothing queued, nothing run on a GPU, nothing sent to Lean; the reads above and
their amendment are not changed; the readings taken where the section left something open, what was found otherwise,
what was NOT built, and what is fragile, are below).**
- **A further seed is the SAME stage and the same fourteen steps** (`ladder_l5_deep --seeds N`; no new stage, no new
  variable). `ladder_loop.l5.search_seed` (0; `the_settings` is its one reader) names the seed whose task makes the deep
  search, the held-aside set and the start model's measurement on it. A task of another seed is a further seed
  (`ladder_l5.a_further_seed`) and writes `ladder_l5_deep_pre_r64_seed<N>`; a task of THAT seed is the search's own task,
  and every step, summary and file of it is what it was. The two sampling seeds (the search's 1050, the held-aside set's
  1051) are derived from the search's seed at every seed of the task. Jobs `ladder_l5_deep_seed1.json`,
  `_seed2.json` (est_minutes 360): job configs, not shipped in this copy.
- **What it reads, and never writes.** Seed 0's FINISHED run (`read_the_search`) and the three stored runs seed 0 read:
  `pre_r64`'s map, the run of the check of the adapter's rank, and L4t2's run OF SEED 0 (L5's own "Fragile (b)": there is
  no L4t2 run of another seed, and none is looked for).
- **Refused by its first step, before anything is written, naming the task to run.** MISSING OR UNFINISHED: one of the
  markers of seed 0's search-prepare, search, prepare, start-model held-aside and report steps, its report, its
  `problems.jsonl`, one of the two draws' files, or the rows, marker or attempts of the search or of `pre_r64`'s
  held-aside measurement is not there. A report NOT TO BE READ or INCONCLUSIVE. A run that recorded another seed, start
  or models. MADE OTHERWISE (`held_to_the_search`, what that run recorded against what this task reads now and would
  do): the start adapter's rank and alpha; how a model is measured; the pretraining file's SHA-256; the size of G′; the
  map's problems by k; the problems searched and the problems held aside (BOTH DRAWS ARE MADE AGAIN from the map with
  the search's seed and held to the stored files: the task's own seed draws neither); the attempts a problem and the
  sampling seed of each; the cap and the minimum of deep solves (so D is seed 0's D). The two stored samplings
  themselves are held to their seed, their attempts and their problems, and refused when Lean did not answer one. The
  second step holds the search it finds again in the stored rows (its table, its deep solves, D) to what seed 0's own
  step counted.
- **What is not made again.** `ladder_l5_search` and `ladder_l5_held_aside_start` return `run: false` with the reason,
  mark nothing and sample nothing. `pre_r64`'s rows on the held-aside set are COPIED into the further seed's run
  (`l5_stored_pre_r64_held_aside.jsonl`), so that its report can be built again from its pulled directory and the read
  over the seeds can hold the seeds to one `pre_r64`. Its `problems.jsonl` holds the sets it samples and no other.
- **By the task's seed, and by nothing else:** which deep solves the cap keeps (`l5.capped`, among all the search
  solved); which verified attempt of each (`l5.solve_rows`); which ordinary solves and which verified map attempt; the
  published rows (`l4_rows.rehearsal_rows` with that seed) and each set's order (`assembly.training_order`); the adapter
  seed of the three trainings. The rungs and G are measured with the sampling seeds `pre_r64`'s run recorded, as seed
  0's were.
- **The further seed's own report** is `build_l5_report` on its three models and seed 0's rows of `pre_r64`: the same
  reads and names. Its first line says A FURTHER SEED in the place of "a scout: one seed"; one line says whose search
  and held-aside set it read; its line WHAT FOLLOWS is the scout's rule on that one seed and is not the branch of the
  three (the line above says so). TIMEOUTS ARE PRINTED BY A FURTHER SEED'S REPORT ALONE (keys `a_further_seed`,
  `timed_out_in_lean`): a line added to seed 0's report would change its `lines`. Seed 0's shares are in its report's
  `attempts` already, and the read over the seeds gives them for every seed.
- **"More than a point"** is more than 0.01 of a set's attempts, between two of the FOUR models (the stored `pre_r64`
  among them) on one kind of set (the rungs, a sampling of G, the held-aside set): `l5.timed_out`. At seed 0 eight
  pairs, all on G's two samplings (`deep` 0.0469 and 0.0399 against `published` 0.0249 and 0.0254; `pre_r64` 0.0443
  and 0.0420; `shallow` 0.0313 and 0.0272).
- **NOT BUILT: "checked again (Lean only)" and "the reads given both ways".** The pairs are named by a further seed's
  report and by the read over the seeds; no step and no tool sends a timed-out attempt to Lean again. By the
  section's rule seed 0 already asks for that check.
- **The read over the seeds** is `reporting.ladder_l5.over_seeds`, run by `python -m rlvr_lean.tools.ladder_l5_seeds
  <the pulled task directories that are there>` (no GPU, no Lean). The seed whose report says nothing of a further seed
  made the search and FOUND the lead; the others are the new seeds. Each seed's primary is computed again from its rows
  and HELD to its report's. "Each problem's mean over the seeds of its difference" is `per_attempt` on each problem's
  attempts ADDED over the seeds, as L4's read is (every seed makes as many attempts at a problem: held, refused
  otherwise). The bootstrap is over the held-aside problems in the order a seed's own report reads them (sorted) and
  over G in the groups' order. Refused: seeds without one search (its problems and what it solved), one held-aside set,
  one measurement of `pre_r64` or one G and G′; two seeds that each made a search; rows that do not give a report's
  primary. A seed that is INCONCLUSIVE or not to be read is named, its other files are not read, and NOTHING is pooled
  (the spread neither).
- **Fewer seeds than three.** The primary is NAMED only over exactly three seeds, the one that made the search among
  them; the lead, its split and the reach only on exactly two new seeds; a third further seed would be read by none of
  them. Otherwise each is NOT READ, the figure over the seeds that are there is still given, and the read lists what it
  could not make. The two new seeds without seed 0 give the lead, its split and the reach, and no primary.
- **Readings of the amendment.** (1) A confirmed lead whose (b) is BELOW zero is named RELIABILITY too, and its line
  says BELOW: the section names the two other cases. (2) THE BRANCH IS READ IN THE SECTION'S ORDER
  (`l5.after_the_seeds`): reach confirmed or ADD first, even under THEY COST; then the lead confirmed as RELIABILITY
  with reach not, also before THEY COST; then THEY COST; then "otherwise". (3) A lead confirmed as REACH TOO whose reach
  on the long problems is not confirmed, with no ADD and no COST, has no branch of its own in the section: it is read
  as "otherwise", and its line says so and does not say "the lead is not confirmed". (4) "The same read" of `deep`
  minus `shallow` (beside the reach) and "the same two reads" of the two other pairs (beside the lead) give their
  figures and conditions and NAME NOTHING. (5) The long problems are the length group "8+" of the stage's by-length
  read (74). (6) The spread's standard deviation is the sample's (n − 1), and it is given for G′ per 1,000 fresh
  attempts too, the figure L4p's spread was read on.
- **A run the task names at a further seed** (the smoke stage queued with `--seeds 1`) writes `<the named
  run>_seed<N>` and reads the named run: the search's own run is never written over. IT DOES NOT EXERCISE THE PATH ON
  THE BOX: the smoke run of seed 0 reads INCONCLUSIVE (`shallow`: its training ran; two rows have no loss to compare),
  which a further seed is refused for, at its first step and with nothing written. No smoke stage was added.
- **Found otherwise than the amendment's figures.** On G′ over the 93 attempts the report's own bootstrap (10,000
  resamples, seed 0, G′ in its stored order) gives +0.00308 [−0.00190, +0.00812], the amendment +0.0031 [−0.0020,
  +0.0083]: the point agrees, the upper end differs in the fourth place (another draw). Every other figure of the
  section and of the amendment that the read gives for seed 0 agrees.
- **Measured before any run, on the pulled copies** (links in a scratch store; no GPU, no Lean; the pulled directories
  only read; the map's own `problems.jsonl` and L4t2's prepare marker are not in the pulled copies and were made again
  in scratch). The two prepare steps ran to their end for seeds 1 and 2: 400 of the 520 deep solves each, 307 and 309
  of those problems among seed 0's 400 (164 and 157 with the same attempt) and 300 in both new seeds; 400 ordinary
  solves, 139 and 146 among seed 0's (147 in both new seeds); the published rows the first 400 and the first 800 of
  each seed's own order, 25 and 26 of the 800 also among seed 0's; no published row with text; 100 optimizer steps a
  model. THE THREE SEEDS ARE THEREFORE NOT THREE DRAWS OF DEEP SOLVES: about three quarters of a seed's deep problems
  are another seed's, and the spread the read gives is over the cap's draw, the attempt, the published rows, the order
  and the adapter seed. On seed 0's pulled run alone the read gives the stored report's figures and the amendment's:
  per 1,000 on the held-aside set 36.15, 50.83, 48.85, 51.46; the primary +0.00063 [−0.00604, +0.00729]; all of G
  103.06, 104.51, 112.46; goal problems solved 262, 257, 275; `deep` minus `published` on all of G +0.00941 [+0.00063,
  +0.01816]; on the 200 goal problems not in G′ 200.91 against 185.43; long problems solved 36, 25, 22 (`deep` against
  `published` gained 14, lost 3).
- **The search's own seed is what it was.** In the scripted world, with the code before the build and after it, seed
  0's run is the same 108 files and 14 step summaries (timings and the copy's own path apart); two runs of the code
  before the build differ in nothing there. Asserted in the suite: no summary and no report of seed 0 holds a
  further seed's key or line. Three assertions that state how things are were brought up to date (the setting's block;
  the sampling seeds of a task of another seed, 1050 and 1051 where they were 1150 and 1151; the stage's four jobs).
- **Fragile, and not shown without the box.** (a) A further seed's report step and the read over three seeds ran at
  the real size only on STAND-IN rows (seed 0's models' rows linked in the place of a further seed's own): the
  plumbing, and nothing else. (b) Seed 0's run on the box must still hold the search's 56 attempt files and its
  `problems.jsonl`; only the pulled copy was read. (c) ONE failed check of one model makes a further seed
  INCONCLUSIVE, and the read then pools nothing: with one of the two new seeds INCONCLUSIVE the lead cannot be read.
  `published` is again a short second pass over rows `pre_r64` has seen (the loss check L4p found wrong for one), on
  another draw of them. (d) The three measurements of a seed are hours apart and the pool changes between them: the
  timeout shares are printed and their check is not built.
- **Mutated.** One hundred and eleven one-change mutations of what this build added, each applied to a copy of the
  tree and never to the working tree: what is a further seed and whose run it reads; the seed that decides each draw (the
  search's and the held-aside set's by seed 0, the cap's, the two attempts', the ordinary solves', the published rows'
  and the order by the task's); that no search is run and the start model is not measured again; each refusal, each
  thing the search's run is held to, and the two stored samplings; the report of the search's own seed and of a
  further one; the primary over the seeds, its three names and its edges at zero, its three seeds; the lead on the new
  seeds alone and each of its two conditions; the split's two sets and its names; the reach's set, its seeds and each
  of its two conditions; the spread; the branch and its order; what the read refuses; the command; the setting; the
  jobs. The first run: 108 noticed, 3 not. The three asked for assertions, which were added: the held-aside set is
  seed 0's draw in a world where another seed would draw another (two candidates left the draw no choice); which
  verified MAP attempt is an ordinary solve's row is by the task's seed (both ordinary solves of the scripted map now
  have two); and the held-aside problems are read in the order a seed's report reads them whatever order the draw
  stored them in (the hand-made ones were sorted already; on seed 0's real rows that mutation gives another interval
  and the read refuses). All 111 are noticed.

**Amended 2026-10-11, BEFORE seeds 1 and 2 run: a rise of the training loss is printed and decides nothing at a
further seed** ("Fragile (c)" above, settled before it can happen and not after).
- **The fault in the check, known before these seeds.** The loss half of "its training ran" reads the mean loss
  over the last tenth of a training's rows against the first tenth's (80 rows each here) and fails above two
  standard errors. With no trend at all that is one training in forty by chance: six trainings in the two seeds,
  about one chance in eight that one of them fails and its seed reads INCONCLUSIVE. L4p found the same half wrong
  where the start model has already been trained on rows of the kind (`second_full`: the loss rose over the pass
  and the model was the best measured). At seed 0 `shallow`'s loss rose by 1.3 standard errors and passed.
- **The rule for seeds 1 and 2.** The other checks stand and each still makes a seed INCONCLUSIVE: the adapter
  saved is not the start adapter's, the model still writes proofs, Lean answered, no barred row, each held-aside
  set answered. A training whose loss rose by more than two standard errors, with its adapter half passing, is
  NAMED on a line of its own and does not stop the seed from being read or pooled.
- **Built only if a training trips it.** The stage as queued would write INCONCLUSIVE for such a seed. The report
  step and the read over the seeds are then made again on the pulled directory by this rule (no GPU, no Lean),
  and the note says that the box's own report read otherwise.

**What these reads can resolve, written 2026-10-11 while seed 1 runs and before any of its rows is pulled** (from
seed 0's intervals; the pooling over seeds shares the problems, so each standard error lies between one seed's and
one seed's over the square root of the number of seeds; approximations, said so that a NOT CONFIRMED is read for what
it is):
- **The primary on HA over three seeds:** one seed's standard error is 0.0034 per attempt, three seeds' 0.0020 to
  0.0034. A difference is found four times in five only when it is 0.006 to 0.010 per attempt, an eighth to a fifth
  of `published`'s own rate there (0.051). Smaller: UNDETECTABLE AT THREE SEEDS, and it is to be written so.
- **The goal set's lead on the two new seeds:** one seed's standard error is 0.0045, two seeds' 0.0032 to 0.0045.
  If the true lead is seed 0's (+0.0094) it is confirmed about three times in five to four in five; if it is half
  of that, as a lead picked after the fact usually is, one time in five to one in three. NOT CONFIRMED is therefore
  the likely outcome even of a real lead of that size, and by itself it is not a refutation.
- **Reach on the long problems:** seed 0's difference is 11 problems of 74 (0.15 of them), one seed's standard
  error 0.053. If the truth is that, it is confirmed four times in five or more; if it is half, one time in three
  to one in two.
- **The spread of each kind of model over its three seeds is the figure that outlasts the verdicts:** it says what
  one trained model's difference from another is worth in every stage after this.
- **The pool during seed 1.** The Lean pool went from 24 workers to about 12 between its first and second model
  (two servers on boxes that other jobs were loading: one slow before it was paused, one drained), so its three models are
  not measured on one pool. Its timed-out attempts are checked again, HA's among them, and its reads are given
  both ways, as the Timeouts rule above already asks.

**Outcome of L5 over three seeds (2026-10-11; `ladder-l5-RESULT.md`, Part 2).** Seeds 1 and 2 ran from cd8372bfc; all
checks pass at every seed.
- **The primary: NOT SHOWN AT THREE SEEDS.** On the held-aside set `deep` minus `published` is +0.00146 [−0.00292,
  +0.00573]; at the seeds +0.0006, +0.0007, +0.0030.
- **The goal set's lead, on seeds 1 and 2 alone: THE LEAD IS CONFIRMED,** +0.02034 [+0.01436, +0.02658] (+0.0177 and
  +0.0230). **Its split: REACH TOO** (on G′ over the 93 attempts +0.00339 [+0.00036, +0.00661]). **Reach on the long
  problems: NOT CONFIRMED** (+0.054 [−0.007, +0.122]; 24 against 19 and 24 against 21). **The branch: "otherwise".**
- **Over trainings** (each seed's model one draw; not a read fixed in advance, added because the rule's intervals are
  over problems): `deep` minus `published` on all of G +9.4, +17.7, +23.0 per 1,000 (mean +16.7, t on two degrees
  4.2, p = 0.052); goal problems solved +13, +33, +33; on G′ +3.1, +3.9, +2.9. `shallow` minus `published` +1.5,
  +30.5, +16.5. `deep` minus `shallow` +8.0, −12.8, +6.5: DEPTH IS NOT SHOWN TO MATTER, which was the stage's question.
- **The spread, the figure that outlasts the verdicts** (standard deviation over three seeds, all of G per 1,000 and
  goal problems solved): `published` 13.9 and 23.3; `shallow` 7.6 and 6.1; `deep` 7.9 and 12.5. On the held-aside
  set 4.1, 4.9 and 4.3 per 1,000.
- **Said of my own note:** after seed 1 I wrote that seed 0's lead "did not repeat"; as the lead was defined it did.
  What did not repeat is `deep` over `shallow`. The timed-out attempts of seeds 1 and 2 are not yet checked again.

### L6: a second round. Does the cycle compound? (fixed 2026-10-11, while L5's further seeds run and BEFORE their read)

**Why.** The owner's bet (2026-10-09) is a feedback cycle: "this loop could be used to push it beyond the reach of
things we even have pre-training data for". L5 asked whether ONE harvest of deep solves teaches what the same number
of published proofs does not. The cycle is the next question whatever L5's three seeds say, short of THEY COST: the
model trained on round 1's deep solves searches again, and what it finds is trained on. Published rows here are a
second pass over proofs the start model has already seen, and there are no more of them in the `pretrain` half; the
loop's rows are new at every round. If the cycle compounds, round 2 shows it.

**When it runs.** After L5's read over three seeds, on every branch but THEY COST. Under THEY COST it does not run and
the three seeds go to the owner.

**Round 1 is L5's, read and not made again** (seed 0's finished run): the search (520 problems solved of 1,000), the
held-aside set HA (600), `pre_r64`'s measurement on HA, and M(1), seed 0's `deep` model (its adapter is kept).

**Search 2.** M(1), served at rank 64, makes 64 attempts at each of the 865 hard problems that are neither round 1's
solves nor held aside:
- **(a) the 480** that search 1 left unsolved (`pre_r64` failed 64 of 64);
- **(b) the 385** candidates that neither draw took (`pre_r64` failed 8 of 8).
55,360 attempts, a sampling seed of its own. A problem it solves is a ROUND-2 solve; its row is one verified attempt,
drawn by the task's seed as a round draws. Fewer than 150 round-2 solves: nothing is trained and the search is the
finding. HA is not searched and no row of an HA problem is trained on.

**Four models, each trained from `pre_r64`** (the ladder's rule: M(r) is trained from the start on every round's
rows), one pass, 2N rows each, N = 520 + the round-2 solves (every deep solve of both rounds, no cap), the published
rows being the first of ONE hash order of the pretraining file (stored by id and place, never with text):
- **`published2`**: the first 2N published rows. The bar at equal steps.
- **`one_round`**: round 1's 520 solves and the first 2N − 520 published rows.
- **`two_rounds`**: the N solves of both rounds and the first N published rows. This is M(2).
- **`long_ordinary`**: N ORDINARY solves (map problems `pre_r64` solved in at least one of its 8 map attempts, one
  verified map attempt each) and the first N published rows. The ordinary solves are drawn to have the line counts
  of `two_rounds`' own N rows, bin by bin (1, 2 to 3, 4 to 7, 8 to 15, 16 or more lines); a bin the map cannot fill
  is filled from the next shorter one and the shortfall is printed.

**Measured,** each as L5's models are: the three rungs, G twice (32 and 61 attempts, the sampling seeds `pre_r64`'s
run recorded) and HA (16 attempts, L5's sampling seed), so each pairs by problem with every stored model. One more
measurement of a stored model: `second_full` (L4p: a whole second pass, 3,109 steps, the best pretraining-only model)
on HA, when its adapter is on the box; NOT THERE otherwise, and nothing is refused for it.

**The reads, fixed before the run.** Per attempt, paired by problem, 95% bootstrap over problems; an interval that
ends at zero holds zero.
- **PRIMARY, the bet at equal steps:** on HA, `two_rounds` minus `published2`. Above zero: THE CYCLE ADDS WHAT THE
  SAME NUMBER OF PUBLISHED PROOFS DOES NOT. Holds zero: NOT SEPARATED AT THIS SIZE. Below: IT COSTS.
- **COMPOUNDING:** on HA, `two_rounds` minus `one_round` (round 2's solves against the same number of published
  rows, round 1's solves in both). THE SECOND ROUND ADDS / NOT SHOWN / IT COSTS.
- **HARD TO FIND, OR LONG:** on HA, `two_rounds` minus `long_ordinary`. Above zero: HARD TO FIND MATTERS BEYOND
  LENGTH. Holds zero: NOT SEPARATED FROM LENGTH. Below: LONG ORDINARY SOLVES DO BETTER.
- **REACH, on the goal set:** the 74 goal problems of 8 lines or more solved at least once in 93 attempts,
  `two_rounds` against `published2`: gained, lost, the sign test. REACH SHOWN AT THIS SEED when more are gained than
  lost with p under 0.05; otherwise NOT SHOWN.
- **Beside, naming nothing:** every pair on all of G and its split (the 200 `pre_r64` solves in its first 32
  attempts; G′); goal problems solved and solved reliably; the never-solved when their ids are given; `two_rounds`
  against M(1)'s and `second_full`'s stored rows on G, and against `second_full` on HA; for each model the rungs,
  the share of distinct attempts and NARROWER (the overfitting reads); each set's share of timeouts.
- **The search's table:** round-2 solves in (a) and in (b), by the attempt of the first success; the lines of their
  proofs beside round 1's.

**Checks, read first; any FAIL makes the run INCONCLUSIVE.** The search found at least 150 solves; for each trained
model its adapter is not the start adapter's, it still writes proofs, Lean answered, no barred row (no held-out
problem, no HA problem, published rows of the `pretrain` half, its own rows base-map problems); Lean answered on
each HA set. A rise of a training's loss is printed and decides nothing (the rule of "L5, further seeds").

**What follows, by the scout's table, fixed now.** The primary's point at or above zero, or REACH SHOWN, or THE
SECOND ROUND ADDS: two further seeds of the four models on the same two searches. The primary below zero by more
than twice the spread L5's three seeds measured between trainings of one kind on HA: the cycle at this size does
not clear the bar, it is written up, and the question of a larger model goes to the owner. In between: stated as
it is, and the owner chooses.

**What it cannot say.** One seed. One M(1) made search 2. Parts (a) and (b) are trained on together. `published2`
is a second pass over seen rows; the published proofs of the `loop` half stay unused by the owner's rule, so
"pretraining on new published data" is not the bar here. HA resolves about 0.007 per attempt at one seed.

**Budget.** Search 2 about two hours; four trainings of about seven minutes; four measurements of about 70 minutes
and five of about 20 on HA with 24 pool workers: about nine hours. A smoke stage first.

**Made exact by the build (2026-10-11; nothing queued, nothing run on a GPU, nothing sent to Lean; the reads above are
not changed; the readings taken where the section left something open, the figures of the section that the stored runs
put otherwise, what was NOT built, and what is fragile, are below).**
- **The stages.** `ladder_l6_cycle` (`gpu/ladder_l6.py`; rules `domain/ladder_round/l6.py`; report
  `reporting/ladder_l6.py`), settings `ladder_loop.l6` (`start: pre_r64`, which must be L5's; `models: [published2,
  one_round, two_rounds, long_ordinary]`; `search_seed` 0; `search`: `attempts` 64, `sampling_place` 52, `minimum` 150;
  `long_ordinary`: `solved_at_least` 1, `line_bins` [1, 2, 4, 8, 16]; one reader). It names NO variable. Run directory
  `ladder_l6_cycle_pre_r64_seed<N>`. SEVENTEEN steps, each resumable: `ladder_l6_search_prepare`; `ladder_l6_search`;
  `ladder_l6_prepare` (the four sets); then for `published2`, `one_round`, `two_rounds`, `long_ordinary` in turn
  `ladder_l6_train_<m>`, `ladder_l6_measure_<m>`, `ladder_l6_held_aside_<m>`; `ladder_l6_held_aside_second_full`;
  `ladder_l6_report`. Jobs `ladder_l6_cycle_seed0.json` (est_minutes 720) and `ladder_l6_cycle_smoke.json` (30):
  job configs, not shipped in this copy.
- **L5's functions are called, not copied.** Four things of `gpu/ladder_l5.py` became functions two stages call, every
  default L5's: what its first step reads, holds and draws before it writes (`read_for_the_search`, with
  `read_the_search` and `held_to_the_search` under the words an `OfTheSearch` gives); the deep search's body
  (`search_deep`, under the names a `Search` gives); one model's measurement on the held-aside set
  (`on_the_held_aside_set`); and `_not_run`. The trainings and the measurements on the rungs and G are L4t's own two
  bodies, as L5's are. So search 2 is the deep search's own code (the round's episode, blocks of `block_of`, a search
  Lean did not answer not kept), and a model here is measured on HA by the code that measured round 1's.
- **What is read, never written, and refused by the first step before anything is written.** ROUND 1 is the run of the
  seed that made L5's deep search (`ladder_l5.search_directory`: `ladder_l5_deep_pre_r64_seed0`). It is read as a
  further seed of L5 reads it, by L5's own code and with its refusals in this stage's words: missing or unfinished;
  a report not to be read or INCONCLUSIVE; another seed, start or models; MADE OTHERWISE (the rank and alpha, how a
  model is measured, the pretraining file, G', the map's problems by k, BOTH DRAWS made again from the map and held to
  the stored files, the attempts and the sampling seed of the search and of the held-aside set, the cap and the
  minimum), and the two stored samplings held to their seed, attempts and problems. With it every stored run that task
  read: `pre_r64`'s map, L4t2's run, the run of the check of the adapter's rank. BESIDE THAT, this stage's own: a
  round 1 that trained no model; its `deep` model without its training or measuring marker, without its stored rows on
  G, or measured there with other seeds or attempts than `pre_r64` was; its adapter not there; a stored search that
  does not count the solves its own step counted, or holds no verified attempt for a problem it counts as solved (held
  here, before the two hours of search 2, and not when the rows are drawn); a search 2 with no problem; a problem of
  it that is held out or no base-map problem; a sampling seed a stored sampling used (L5's two among them).
- **Search 2.** (a) is the problems search 1 attempted and did not solve, in its draw's order; (b) the map's rows with
  k = 0 that neither search 1's draw nor the held-aside set took, in the map's order (`l6.second_search`). HELD, each a
  refusal: a problem search 1 solved is one it attempted; none is both searched and held aside or stands twice; a
  problem a draw took by the rule is a candidate; and round 1's solves, (a), (b) and HA partition the candidates. The
  counts are the stored files' and are recorded (`the_search.partition`), never numbers of the code. Sampling seed
  `round.sampling_seed` + 100 × `l6.search_seed` + place 52 (1052), of the seed the SETTING names, as L5's two are.
  Blocks of 16 problems (55 of them at seed 0). A round-2 row carries its part and `first_success`, the place of the
  problem's first verified episode among M(1)'s OWN attempts, from 1: M(1) made none at these problems before. The
  table's bands are 1 to 8 (a map's attempts), then doubling: 9 to 16, 17 to 32, 33 to 64, for each part by itself.
- **The rows, by the task's seed.** One verified attempt of every problem a search solved, by `l5.solve_rows` (the
  function a round draws by): round 1's from round 1's stored attempts, round 2's from this run's. Origins `round_1`,
  `round_2`, `ordinary`. At seed 0 the 520 round-1 rows are the very attempts L5's own step drew (the same function and
  seed: measured on the pulled run), of which its sets held 400.
- **The four sets** (`l6.the_sets`, held from the outside by `hold_the_sets`), each in `assembly.training_order`. Held,
  each a refusal: 2N rows and no id twice; `two_rounds` N own rows with at least one of each round and N published;
  `one_round` round 1's solves, THE SAME ROWS as `two_rounds`', and 2N less as many published; `long_ordinary` N
  ordinary solves and THE SAME N published rows as `two_rounds`, by id and place, in the same order; `published2` 2N
  published rows; the published rows of each the FIRST of the one hash order, each smaller draw the first rows of each
  larger; no problem twice among a set's own rows; no problem both a deep solve and an ordinary one; no own row of a
  problem the pretraining file holds; no published row with text; no held-out problem; published rows of the
  `pretrain` half; own rows base-map problems; no row of a held-aside problem, and no held-aside problem searched.
- **The ordinary solves matched in length** (`l6.matched_in_length`). An ordinary solve is a map problem `pre_r64`
  solved in at least `long_ordinary.solved_at_least` (1) of its 8 map attempts (2,015 at seed 0), and its row ONE stored
  verified map attempt drawn by the task's seed with L5's own function; ITS lines are the row's (as the report's "lines
  of the rows" counts them). The bins are the setting's lower edges. Within a bin a seeded draw of its own (label
  `l6_long_ordinary`: the smallest content hash of the task's seed and the problem's id). From the longest bin down,
  what a bin cannot give is asked of the next shorter one beside that bin's own; the table (`matched_in_length`) gives
  for each bin what is wanted, what the map offers, what longer bins asked of it, what was taken and its shortfall, and
  the report prints it.
- **Measured.** Each model by L4t's own two bodies (the rungs; G twice with the sampling seeds `pre_r64`'s run
  recorded), then by L5's own body on HA: 16 attempts, round 1's sampling seed for that set (1051), blocks of 64.
  `pre_r64`'s rows there are round 1's, copied (`l6_stored_pre_r64_held_aside.jsonl`). The set's attempts, sampling
  seed and block size are the ones round 1's run RECORDED, and L5's body refuses a config that now gives others.
  `second_full` is the LAST step before the report, so that the reads the section names are in hand first.
- **Readings the section left open, and the ones taken.** (1) **"No barred row ... no HA problem"**: the fourth check is
  L5's own with one thing more, no row of a held-aside problem (`l6.no_barred_row`). (2) **"Lean answered on each HA
  set"** reads the four models' sets, round 1's stored one of `pre_r64`, and `second_full`'s when it was measured; such
  a set of this run's own is measured again when the task is queued again. (3) **The loss.** The adapter half of L4t's
  "its training ran" is the check (`its_adapter_is_not_the_start_adapters`); each training's loss at both ends is
  PRINTED on a line of its own, a rise beyond the allowance NAMED there, and it is in no list of failed checks. (4)
  **ONE verdict for the run**, as L5: any failed check, of any model, and the four reads are INCONCLUSIVE, what was
  measured kept apart and none of it printed. (5) **Fewer than 150 round-2 solves: the stage does not fail.** No set is
  built, the thirteen steps after it return that and mark nothing, the report reads INCONCLUSIVE and gives the search's
  table as THE FINDING. (6) **REACH** is on the length group "8+" of the stage's by-length read (74), over all the
  attempts: SHOWN when more are gained than lost AND the sign test is under 0.05. (7) **What follows, in the section's
  order:** the point at or above zero, or REACH SHOWN, or THE SECOND ROUND ADDS, first and whatever the spread; then,
  below zero, the kill by the spread, where below by EXACTLY twice the spread is "in between". (8) **"The spread L5's
  three seeds measured between trainings of one kind on HA"** is three figures, one a kind of model (the standard
  deviation, n − 1, of its seeds' successes per attempt there). THE LARGEST is taken: a kill then asks the most. It is
  read from `l6_l5_spread.json` when that file is GIVEN in the run directory (what `python -m
  rlvr_lean.tools.ladder_l5_seeds --out` writes). NOT GIVEN: no file, a read that pooled nothing or is not over exactly
  three seeds, a kind without a standard deviation; the line then gives the primary's point, says so and names no kill
  (`THE SPREAD IS NOT GIVEN: NO KILL IS NAMED`). (9) **"Every pair"** is the six pairs of the four models, each on all
  of G, on the goal problems not in G' and on G', over ALL the attempts (as L5's amendment split its lead), with the
  goal problems solved. (10) **M(1) is read on G alone** (the section's words); its rows on HA and L5's `published` and
  `shallow` are not read. (11) **`second_full` "when it is on the box"**: its rows on G are read when L4p's run holds
  its prepare, training and measuring markers, is from this start, and its two stored samplings of G pair (the
  problems, the seeds, the attempts); it is measured on HA when, beside that, its adapter's directory is there when
  ITS step runs. Otherwise NOT THERE with the reason, in the first step's record and in the report; a task queued again
  after the adapter is back measures it, and no other step runs again. (12) **A shortfall in the SHORTEST bin is
  refused** in a real run: the section fills from the next shorter bin and names no fill from a longer one. (13) **Each
  read's line gives the held-aside problems solved** (gained, lost) beside its difference, and names nothing by them.
  (14) **A task of another seed than `l6.search_seed` is refused** by the first step: further seeds of L6 and a read
  over seeds are not built. Search 2's sampling seed is the setting's seed's and every draw of rows is by the task's,
  so that a further seed can later read this search.
- **Figures of the section, as the stored runs put them.** "55,360 attempts" are EPISODES: the audit rule samples the
  other side too for a seeded 2% of problems (search 1: 56,616 attempts for 56,000 episodes). **THE BUDGET IS NOT NINE
  HOURS BY L5's MEASURED TIMES.** Seed 0 of L5 (15 pool workers) took 134 minutes for its search (422 attempts a
  minute), 73 to 87 minutes for a model's rungs and G (81 on average), 22 to 25 for a held-aside set, 4 for a training
  of 800 rows; its generator waited on Lean 87 seconds of the search and not at all on a held-aside set, so the steps
  are bound by generation and 24 pool workers shorten them little. By those times: search 2 about 134 minutes, four
  trainings of 8 to 14 (1,640 rows at 300 round-2 solves, 2,770 at most), four measurements, five held-aside sets: 611
  minutes at the averages, 664 at the worst of each (ten and a quarter to eleven hours). est_minutes is 720.
- **The smoke stage** (`ladder_l6_cycle_smoke`) runs all seventeen steps and reads `ladder_l5_deep_smoke`, the three
  smoke runs that task read, and `ladder_l4_dose_r64_smoke` for `second_full`; it writes `ladder_l6_cycle_smoke`. HOW
  IT GOES ON FROM ROUND 1's SMOKE RUN: that run's report is INCONCLUSIVE (`shallow`: its training ran; two rows have no
  loss to compare), and the rule gives search 2 no problem (the fixture's model solved every base-map problem in its
  map, and search 1 solved its one). `RLVR_LEAN_LADDER_L6_SMOKE_PROBLEMS=2`, set by this stage ALONE: the INCONCLUSIVE
  round 1 is read all the same and the first step RECORDS it (`smoke.round_one_read_though_inconclusive`, with the
  failed checks; a report not to be read is refused here too); round 1's two looser draws are made again as that run
  recorded them, WITHOUT L5's own smoke-only variable, which stays its smoke stage's alone; search 2 attempts the two
  map problems no search took and that are not held aside; one round-2 solve is enough; `long_ordinary` takes its
  ordinary solves among the map problems the searches attempted, and a shortfall in the shortest bin is filled from
  what is left. Every problem and row so taken is marked `taken_by: the run's minimum` and counted in the report's line
  A SMOKE RUN. NO REAL RUN CAN TAKE THAT PATH: the first step refuses the variable unless the run directory is one the
  task names AND the map is the fixtures'.
- **Measured before any run, on the pulled copies** (links in a scratch store; no GPU, no Lean; the pulled directories
  only read; the map's own `problems.jsonl`, L4t2's and L4p's prepare markers are not in the pulled copies and were
  made again in scratch). The first step ran to its end for seed 0: 1,985 candidates = 520 round-1 solves + 480 left
  unsolved (a) + 385 in neither draw (b) + 600 held aside; search 2 is the 865, none held aside, each with k = 0 in the
  map; sampling seed 1052; 55 blocks of 16; `second_full`'s rows on G pair. With a STAND-IN search 2 (a seeded 300 of
  the 865 marked solved, each by one stand-in verified attempt whose text is one of round 1's own deep proofs): N =
  820; four sets of 1,640 rows, 205 optimizer steps each; `one_round` 520 and 1,120, `two_rounds` 520, 300 and 820,
  `long_ordinary` 820 and 820, `published2` 1,640; the published rows nested and the same 820 in `two_rounds` and
  `long_ordinary`; no published row with text, no barred row, no row of a held-aside problem. THE LENGTH MATCHING AT
  THAT SIZE FILLS EVERY BIN, BARELY: by bin (1, 2-3, 4-7, 8-15, 16+) 20, 76, 192, 341 and 191 rows are wanted and the
  map offers 411, 616, 445, 347 and 196. With round 1's lines (64% of its rows have 8 lines or more) the two long bins
  run short from about N = 835 and are then filled from the shorter ones (8-15 first, then 4-7, which has room). The
  cascade cannot reach the shortest bin at any N this run can have (at most 1,385 rows are wanted, and the bins above
  the shortest offer 1,604 ordinary solves); the refusal of reading (12) would take more than 411 one-line deep proofs
  (round 1 has 12). The smoke stage's first step and its set-building step ran to their end on the pulled smoke copies
  (round 1's smoke run read though INCONCLUSIVE, and recorded; search 2 takes the two map problems left, both by the
  run's minimum; `second_full`'s rows pair; N = 3 with both marked solved, four sets of 6 rows).
- **L5 writes what it wrote.** In the scripted world, with the code before the build and after it: L5's run at the
  search's own seed is the same 108 files and 14 step summaries, and a further seed's the same 97 and 14 (timings and
  the copy's own path apart). Asserted in the suite: with this stage run beside it, L5's own task queued again returns
  what is stored and moves no file, a further seed of L5 run AFTER this stage leaves the files a further seed leaves,
  and round 1's run, L4p's and every run round 1 read are file for file what they were after this stage ran. Three
  assertions that state where things live were brought up to date (one more smoke stage names the rank check's smoke
  run, the map's, L5's smoke run and L4t2's). THE SCRIPTED WORLD of this stage's tests is the fixtures' with a base map
  of twelve problems (the fixtures' four and eight more, in a data directory of the test's own): four problems cannot
  hold a round-1 solve, an (a), a (b), a held-aside problem and the ordinary solves at once.
- **Fragile, and not shown without the box.** (a) M(1)'s adapter is served for a search from a run it was not trained
  for a search in, and `second_full` from L4p's run for an episode step: the smoke stage is the first run of both. (b)
  M(1)'s rate in a search is not known (its proofs are longer than `pre_r64`'s, and its goal attempts timed out in Lean
  more often). (c) Round 1's run on the box must still hold all 56 attempt files of its search: only the pulled copy
  was read. (d) The report step ran at the real size on nothing: the four models do not exist; its plumbing is the
  scripted world's. (e) One failed check of one model makes the run INCONCLUSIVE. (f) The five held-aside sets are
  measured hours apart; the timeout shares are printed and their check is not built, as in L5.
- **Mutated.** One hundred and fifty-seven one-change mutations of what this build added, each applied to a copy of the
  tree and never to the working tree: search 2's two parts, each refusal of the partition and a smoke run's joiners; the
  minimum; the table; the bins, the direction of the fill, the shortfall, the seeded draw within a bin and a smoke
  run's looser fill; each set's composition and size, N, and each thing the sets are held to (the sizes, the origins,
  the ids, the order, round 1's same rows, the nesting, the first of the order, the same published rows, deep against
  ordinary, the pretraining's problems); the held-aside bar of the fourth check; each read's three names and its two
  edges at zero, and a failed check; reach's two conditions; the spread (the largest, three seeds, per 1,000, the
  held-aside set's figure); what follows (the point at zero, reach, compounding, once and exactly twice the spread, the
  spread not given, an INCONCLUSIVE run); in the report the reads' set (the held-aside set, not G), their two sides,
  each read's own figure, reach's other side, its set and its samplings, the loss rise deciding and the adapter half
  not deciding, each kind of failed check, the point what follows reads, the split of G, the stored models' other
  side; in the stage whose adapter makes search 2 and under whose names, which model M(1) is, the seed of the sampling
  seed, a further seed, an INCONCLUSIVE round 1 in a real run, each guard of the smoke-only variable, each refusal of
  round 1 and of M(1), the stored samplings' seeds, a search 2 without a problem, a smoke run's draws, the minimum, the
  seed of each of the three draws of a row, the ordinary rule, the held-aside set held out of the sets, a barred row,
  the held-aside set's sampling seed, where the stored model is served from and when it is NOT THERE; sixteen changes
  to L5's own module (the reader's allowance and its default, its smoke guard, the caller's least number, the search's
  marker, set and served name, the held-aside set's files and directory, `_not_run`, the further seed's hook); the
  registration, the smoke stage's environment, the setting. The first run: 143 noticed, 14 not. One of the fourteen
  was the same program in every real run (`long_ordinary`'s looser candidates are empty there: no problem a search
  attempted was solved in the map) and was replaced by one that changes what a smoke run takes. Thirteen asked for
  assertions, which were added: a smoke run's search 2 passes over a held-aside problem that is first in the hash
  order; a whole family of sets without a solve of round 2 is refused; as many long problems gained as lost is not
  more; what follows reads the PRIMARY's point (below `published2` and level with `one_round` is a kill, not further
  seeds); overfitting is each model against the start model; the minimum of round-2 solves is the setting's in a real
  run; a real run refuses a shortfall in the shortest bin; a held-out problem among a set's rows is refused where the
  sets are built; a stored model whose rows would not pair is not measured though its adapter is there; round 1's row,
  round 2's and an ordinary solve's are each drawn by the task's seed in a world where seed 1 draws another (three
  mutations: in the first world seed 1 drew the same attempts); and the smoke world keeps L5's real minimum of deep
  solves, so that the minimum round 1 recorded is held. One thing of the code moved with them: the file of the deep
  solves is written with the sets, and no longer before a refusal that says nothing was written. The last run, all 157
  against the tree with these additions: 157 noticed.

### L5n: what is one training worth? The spread between trainings, and a recipe that lowers it (fixed 2026-10-11, after L5's seed 1 and while its seed 2 runs)

**Why, and what it overturns.** L5's seed 1 turned seed 0's order over. All of G, successes per 1,000 attempts, seed 0
and seed 1: `published` 103.1 and 79.5 (the start model's own 80.0, and fewer goal problems solved than it, 217
against 245); `shallow` 104.5 and 110.0; `deep` 112.5 and 97.2. Goal problems solved: 262 and 217, 257 and 261, 275
and 250. Every check passes at both seeds and `published` was measured on a healthy pool. Two trainings of one kind
differ by up to 24 per 1,000 and 45 goal problems, where L4p had put the spread at 6 and 5 from models of four
different doses. **So no difference between two single trained models since L4t stands as an effect, in either
direction:** not "the model's own proofs narrow it" (L4t, L4c, L4p part 1), not "the published rows do the adding",
not seed 0's lead of `deep`. Every model trained from `pre_r64` so far, on published rows or on its own, for 100
steps or 3,109, lies in one band of about 95 ± 10 per 1,000. Each stage's interval was over problems and never over
trainings; that is my fault in the reads, not a fault of the runs. Until one training is worth more, no stage that
compares single trained models can be read, L6 among them. This is a fault of the measurement and it is dealt with
before another arm is bought.

**What the recipe is** (`ladder_dose._train_with_readings`): AdamW at a constant learning rate of 1e-4 after five
warm-up steps, no decay, no weight decay, no dropout; a training from `pre_r64` loads that adapter, so a task's seed
changes WHICH ROWS are drawn and THEIR ORDER and nothing else. A step of Adam is about the size of the learning rate
whatever the gradient, so with no decay the model saved is the last point of a walk. That is the hypothesis; nothing
here assumes it.

**The check.** ONE fixed set of rows: the 800 published rows of L5 seed 0's `published` (by id and place; the model
that scored 103.1). From `pre_r64` at its rank, FOUR ORDERS of those rows (the content-hash order of order seeds 1 to
4; none is the order seed 0 trained in), and each order trained twice:
- **`constant_1` to `constant_4`:** the recipe as it is.
- **`decay_1` to `decay_4`:** ONE change: after the warm-up the learning rate falls linearly to zero at the end of the
  pass.

Then two models that are not trained: **`mean_constant`** and **`mean_decay`**, the adapter whose every number is the
mean of its four adapters' (the four start from one adapter and move little, so the mean is to first order the
adapter of the mean model; it is read back and served like any other).

Ten models. Each is measured on the goal set's FIRST sampling alone: 32 attempts a problem with the sampling seed
`pre_r64`'s run recorded, so it pairs by problem with every stored model's first sampling. No rungs, no second
sampling, no held-aside set: this stage reads a spread, not a model.

**The reads, fixed before the run.** The figure of a model is its successes per attempt on all of G over the 32
attempts. For a fixed set of problems its sampling noise is binomial (from each problem's own rate) and is printed
beside every figure: about 0.003.
- **THE SPREAD OF THE RECIPE AS IT IS:** the standard deviation (n − 1) of the four `constant` figures, with the
  sampling noise beside it and, from L5, the standard deviation of `published` over its three seeds (rows and order
  both drawn) on the same 32 attempts. ORDER ALONE MAKES THE SPREAD when the first is at least half of the second;
  otherwise THE ROWS DRAWN MAKE MOST OF IT.
- **DOES DECAY LOWER IT:** the standard deviation of the four `decay` figures against the `constant` one. DECAY LOWERS
  THE SPREAD when it is at most half; NOT SHOWN otherwise. Said with it: with four against four only a ratio of about
  three is beyond chance (F at 3 and 3 degrees, 9.28).
- **AND ITS LEVEL:** `decay_k` minus `constant_k` for the four orders, their mean with its t interval at 3 degrees.
  DECAY COSTS when the interval is below zero; otherwise its level is given and named nothing.
- **AVERAGING:** each averaged model's figure against the mean of its four models' figures and against the best of
  them. AVERAGING GIVES AT LEAST THE MEAN OF ITS MODELS when it is not below that mean by more than twice its sampling
  noise.
- **Beside, naming nothing:** each model's goal problems solved in the 32 attempts and their spread; its share of
  distinct attempts; its share of timeouts; L5 seed 0's own `published` (the same rows in seed 0's order, the recipe
  as it is) from its stored first sampling, as a fifth draw of `constant` that is not in the standard deviation.

**Checks, read first; any FAIL makes the run INCONCLUSIVE.** Each trained adapter is not the start adapter's and is
read back at its rank; each averaged adapter's numbers ARE the mean of its four (read back) and it is served; Lean
answered (at most 0.02 of a set without an answer); no row is anything but one of the 800.

**What follows, fixed now.**
- **DECAY LOWERS THE SPREAD and does not cost:** every later stage trains with it. L5's three kinds are then trained
  again with it before L6 runs (specified then, with this stage's cheap measurement for the spread).
- **Not that, and AVERAGING GIVES AT LEAST THE MEAN:** a later stage trains each model four times and measures the
  averaged adapter.
- **Neither:** the spread is the recipe's own at this size (its learning rate, its length), and the choice (a lower
  learning rate, or more seeds an arm) goes to the owner with these figures.

**L6 waits for this read** (amending its "When it runs"): one training a model, as L6 is specified, cannot resolve
a difference under about 25 per 1,000 beside this spread. L6's smoke run stands; its real run is queued with the
recipe this stage names.

**What it cannot say.** One set of rows, published rows only; four orders; G alone and 32 attempts; nothing of the
held-aside set. It measures the recipe's noise from `pre_r64`, not from another start.

**Budget.** Eight trainings of about four minutes and ten measurements of about 28: about five hours and a quarter.
A smoke stage first.

**Made exact by the build (2026-10-11; nothing queued, nothing run on a GPU, nothing sent to Lean; the reads above are
not changed; the readings taken where the section left something open, the figures of the section that the stored runs
put otherwise, what was NOT built, and what is fragile, are below).**
- **The stages.** `ladder_l5_noise` (`gpu/ladder_l5_noise.py`; rules `domain/ladder_round/l5_noise.py`; report
  `reporting/ladder_l5_noise.py`), settings `ladder_loop.l5_noise` (`orders: [1, 2, 3, 4]`, `recipes: [constant, decay]`,
  `sampling: reach`, `attempts: 32`, `published_seeds: [0, 1, 2]`; one reader; the start is L5's own setting, and no
  setting of this block names it). It names NO variable. Run directory `ladder_l5_noise_pre_r64_seed<N>`. TWENTY-TWO
  steps, each resumable: `ladder_l5_noise_prepare`; then for each order seed k in turn `ladder_l5_noise_train_constant_k`,
  `ladder_l5_noise_measure_constant_k`, `ladder_l5_noise_train_decay_k`, `ladder_l5_noise_measure_decay_k` (ORDER BY
  ORDER, so that a run cut short leaves whole pairs); `ladder_l5_noise_average_constant`,
  `ladder_l5_noise_measure_mean_constant`, `ladder_l5_noise_average_decay`, `ladder_l5_noise_measure_mean_decay`;
  `ladder_l5_noise_report`. Jobs `ladder_l5_noise_seed0.json` (est_minutes 420) and `ladder_l5_noise_smoke.json` (30):
  job configs, not shipped in this copy.
- **Existing functions are called, not copied.** L5's finished run is read by L5's own first-step reading
  (`ladder_l5.read_the_search`, `read_for_the_search`, under the words an `OfTheSearch` gives: a further seed's refusals,
  as L6 reads round 1). A training is L4t's own body (`ladder_l4_rows.train_a_model`); a measurement the ceiling's own
  (`ladder_ceiling.measure_model` with `rungs=False` and ONE sampling handed to it, as L4t's `hot` samples G alone).
  THREE FUNCTIONS TAKE A NEW ARGUMENT THAT IS OFF BY DEFAULT: `ladder_dose._train_with_readings(decay=False,
  learning_rates=False)`, `ladder_ceiling.one_pass(decay=False, learning_rates=False)` and `train_a_model(schedule=None)`
  (a `Schedule`: whether the decay is asked, and what the summary says of the order). `ladder_l5.own_run` takes a seed
  (default: the task's), for the runs of L5's other seeds.
- **The rows, and what is refused by the first step before anything is written.** L5's own refusals (missing or
  unfinished; a report not to be read or INCONCLUSIVE; another seed, start or models; MADE OTHERWISE, with the three
  stored runs that task read). BESIDE THEM, this stage's own: a run that trained no model; `published` without its
  training or measuring marker, its stored set (`l5_training_published.jsonl`), its training's own record
  (`l5_loss_published.json`) or its stored first sampling; a stored set that is not as many rows as its training
  recorded and its own step counted (2D); THE FIXED ROWS (`l5_noise.the_rows`): a row with text, a row that is no
  published row, an id twice, a row that is not the row of the pretraining file at its place, a set that is not the
  first as many rows of L5's own draw for seed 0 (`l4_rows.rehearsal_rows`), a set that does not stand in the order
  that task trained in, a training whose own record of the rows is not these rows in this order; a barred row
  (`l5.refuse_barred`); A RECIPE THAT IS NOT L5's (the learning rate, the warm-up steps, the batch, the adapter's rank,
  alpha and modules that run recorded: `constant` is the recipe that trained the model that is read beside); a first
  sampling of `pre_r64` that is not the setting's (its name and its 32 attempts; the sampling SEED is the recorded one
  and no setting); a start adapter whose numbers are of a type the averaged adapters are not written in; an order that
  is the order L5 trained in, or two orders alike (`l5_noise.the_orders`); A TASK OF ANOTHER SEED than L5's search seed
  (nothing here is drawn by the task's seed: such a task would train the same eight models again; further seeds of this
  stage are NOT BUILT).
- **The decay, exactly** (`domain/ladder_round/dose.py`: `warmed`, `decayed`, `scheduler_factor`). For optimizer step s,
  counted from 1, of a training of T steps with W warm-up steps: the recipe as it is has the factor min(1, s / W); with
  the decay the factor is s / W for s ≤ W (the recipe's own, 1 at s = W) and (T + 1 − s) / (T + 1 − W) for s > W: a
  straight line from 1 at the warm-up's last step to ZERO AT THE STEP AFTER THE LAST ONE. "Zero at the end of the pass"
  is read so: the last step is still a step (its factor is 1 / (T + 1 − W)). At T = 100, W = 5: steps 1, 5, 6, 50, 99,
  100 have 0.2, 1, 1, 1, 1, 1 as it is and 0.2, 1, 95/96, 51/96, 2/96, 1/96 with the decay (0.98958, 0.53125, 0.02083,
  0.01042). The factors of a pass add up to 98.0 as it is and 50.5 with the decay: a decayed training moves about half
  as far in all, which is said beside any read of its level. A training no longer than its warm-up has no step after
  it, and every factor is the recipe's own. The scheduler counts the updates made so far, so it is handed the factor of
  step s at s − 1; with the decay off it is handed the very expression it always was.
- **Every training records its learning rate, and its step HOLDS it** (a hold the section does not ask for in those
  words). The rate is read from the optimizer before each step's update, stored beside that step's row in the model's
  loss file and summed up in its summary (`learning_rate`). `train_a_model` refuses, with the step not marked done, a
  training that recorded none, fewer than it made steps, or one that is not its schedule's within a billionth of the
  rate (`dose.held_to_the_schedule`): a `decay` whose rate did not fall is a `constant` under another name.
- **The averaged adapters** (`infrastructure/adapter_files.py`: `mean_of`, `is_the_mean`, beside `compared`; numpy and
  the file format alone, no torch). The file is the FIRST model's, byte for byte (its header: the names, types, shapes
  and places as the library laid them out), with every tensor's numbers written over. A NUMBER'S MEAN IS SUMMED IN
  FLOAT64 AND ROUNDED ONCE TO THE TENSOR'S OWN TYPE: for float32 it is the nearest float32 to the mean, within half a
  unit in the last place (a relative 6e-8). Written for float16, float32 and float64; bfloat16 and whole numbers are
  refused, and the first step holds the start adapter's types to that before anything is trained. The models' adapter
  configs must agree (a list in them compared sorted: a set of module names is saved in no fixed order) and the first
  is copied. READ BACK, the step holds: every number is that mean BIT FOR BIT (made again from the four files); it is
  NONE OF THEM (it differs from each in at least one number); and its config and every matrix have the start adapter's
  rank and alpha. Refused otherwise, the step not marked. Its summary also compares it with the start adapter.
- **A measurement** is 32 attempts at each goal problem with the sampling seed `pre_r64`'s run recorded for its first
  sampling (1001), under the set `l4_reach_<model>`, the model server started for rank 64. No set of a rung, of the
  second sampling or of the held-aside set exists in this run.
- **Readings the section left open, and the ones taken.** (1) **The figure** is successes over attempts on all of G (an
  audited problem's episodes counted once); its noise is the section's formula on that model's own rows. (2) **"The
  sampling noise beside it"** (the first read) is the root of the mean square of the four `constant` models' noises.
  (3) **L5's `published` over its three seeds** is read BY THE REPORT STEP, from the runs of the setting's seeds, so that
  a seed that finishes while this stage runs is read; the fixed rows' own seed is the copy the first step held. A seed
  is THERE when its report step is done and its report is to be read, `published` is trained and measured, and its
  stored first sampling pairs with this run's (the problems, the seed, the attempts, Lean having answered). A seed
  whose own report read INCONCLUSIVE is read, and said (a rise of a training's loss decides nothing of its figure). With
  any seed NOT THERE the spread is given and the name is NOT READ; the rows of each seed that is there are copied into
  this run (`l5_noise_stored_published_seed<N>.jsonl`). (4) **The edges.** "At least half" and "at most half" hold at
  exactly half; a recipe with no spread at all has none to lower (NOT SHOWN). (5) **"Only a ratio of about three is
  beyond chance"** is said from the upper 5% point of F at as many degrees each (9.28 at 3 and 3: a ratio of 3.0 between
  two standard deviations). (6) **The level's interval** is the mean ± t × the standard deviation of the paired
  differences over the root of the orders (3.182 at four orders; 4.303 and 12.706 at three and two, a smoke run's);
  an interval that ends at zero is not below it; its other name is NAMED NOTHING. (7) **AVERAGING is named for each
  averaged model by itself**; exactly twice its noise below the mean still gives; the outcome the section does not name
  is AVERAGING GIVES LESS THAN THE MEAN OF ITS MODELS. (8) **WHAT FOLLOWS, the second case, reads `mean_constant`**, the
  averaged adapter of the recipe a later stage would then train with; `mean_decay`'s name stands beside. The three
  cases are named EVERY LATER STAGE TRAINS WITH THE DECAY / A LATER STAGE TRAINS EACH MODEL FOUR TIMES AND MEASURES THE
  AVERAGED ADAPTER / THE SPREAD IS THE RECIPE'S OWN AT THIS SIZE: THE CHOICE GOES TO THE OWNER. (9) **The checks.** "No
  row is anything but one of the 800" is read on the training loop's own record, by id and place, and asks too that
  each fixed row stands once. "It is served": its set was sampled by the step that asked the model server for THAT
  adapter. ONE verdict for the run: any failed check of any model, and the reads are INCONCLUSIVE, what was measured
  kept apart and none of it printed. A set Lean did not answer also fails the report step, and the task queued again
  samples that set again. (10) **Beside**: the share of distinct attempts is the first sampling's own (L5's `published`
  of seed 0: 0.914 on its 32 attempts, where its own run's 0.813 is over all 93).
- **Figures of the section, as the stored runs put them.** "The model that scored 103.1" scored it over all 93
  attempts. ON THE FIRST SAMPLING ALONE, which is all this stage reads, L5's `published` has 103.79 per 1,000 at seed 0
  (1,302 of 12,544; 220 goal problems solved) and 78.60 at seed 1 (986; 177). The sampling noise "about 0.003" is
  0.0020 to 0.0024 on the six stored first samplings of L5's models. THE BUDGET IS 337 MINUTES BY THE MEASURED STEP
  TIMES, not 312: a training step took 238 to 254 seconds (it loads the model) and a measurement starts the model
  server before its 28 minutes. est_minutes is 420, which leaves a quarter for a slower pool.
- **The smoke stage** (`ladder_l5_noise_smoke`) runs fourteen steps (two of the four orders) and reads
  `ladder_l5_deep_smoke` and the three smoke runs that task read; it writes `ladder_l5_noise_smoke`. HOW IT GOES ON:
  `RLVR_LEAN_LADDER_L5_NOISE_SMOKE_ORDERS=2`, set by this stage ALONE and refused by the first step unless the run
  directory is one the task names AND the map is the fixtures'. Under it, each recorded: L5's smoke run, whose report
  is INCONCLUSIVE, is read all the same (one NOT TO BE READ never is); only the first two orders are trained; the fixed
  rows are TWO rows, which have two orders, and an order that is the one L5 trained in (order seed 2, on the pulled
  smoke run) or another's is trained all the same; and an averaged adapter that IS one of its models is kept and
  served (two rows are one batch and one optimizer step whatever their order: the four trainings are most likely one
  training, number for number). L5's own smoke-only variable is not set: that run's two looser draws are made again as
  it recorded them. WHAT IT DOES NOT EXERCISE: a learning rate that FALLS (a training of two rows is one step, inside
  the warm-up: the decay is asked for, its schedule built, its one recorded rate held, and that rate is the recipe's
  own); four orders; the reads on rows that can be read.
- **Measured before any run, on the pulled copies** (links in a scratch store; no GPU, no Lean; the pulled directories
  only read). The first step ran to its end for seed 0: the 800 rows are L5's stored set, id for id and place for place
  in its order, the first 800 of its draw, with no text and no barred row; eight sets of 800 rows, 100 optimizer steps
  each; no order seed of 1 to 4 gives seed 0's order and no two give one. Two orders share 6 to 13 of their first 80
  rows and 4 to 11 of their last 80 (by chance: 8), and at most 3 rows stand at the same place in two orders; against
  seed 0's order, 4 to 10 of the first 80 and 4 to 12 of the last. The smoke stage's first step ran to its end on the
  pulled smoke copies. THE SCHEDULER ITSELF, on a CPU: the optimizer's rate before each of 100 steps is the schedule's
  bit for bit, as it is and with the decay (torch 2.8). THE TRAINING FUNCTION ITSELF, on a CPU with a made-up model of
  a few hundred numbers and the real optimizer and scheduler: with both options off it returns what the code before
  the build returns, number for number; with the rates recorded it is that result and the 100 rates; with the decay the
  first six steps' losses are the recipe's own and the later ones are not. The mean of four made-up adapters, read
  back: 0 of its numbers are not the mean, and the largest gap to the mean in float64 is 6e-8. The report's reads on
  STAND-IN rows (the stored first samplings of L5's six models in the ten models' places) reach every name.
- **Existing stages write what they wrote.** In the scripted world, with the code before the build and after it: L5 at
  the search's own seed is the same 108 files and 14 step summaries, a further seed's the same 97 and 14, L6's the same
  149 and 17 (timings and the copy's own path apart; two runs of the code before the build differ in nothing).
  Asserted in the suite: a training of L5 and of L6 hands the pass the start adapter and nothing more, and no summary
  or loss file of theirs holds a learning rate; with this stage run beside it, L5's own task queued again returns what
  is stored and moves no file, and a further seed of L5 run AFTER this stage leaves the files a further seed leaves.
  Five assertions that state where things live were brought up to date (one more smoke stage names the rank check's
  smoke run, the map's, L5's smoke run and L4t2's; L5's own stages and jobs are named by its stage).
- **Fragile, and not shown without the box.** (a) THE FIRST LEARNING RATE THAT FALLS ON THE BOX is step 6 of
  `ladder_l5_noise_train_decay_1`, about 35 minutes into the real run: the smoke run cannot show one. A training whose
  rates are not its schedule's fails its step, unmarked, after its four minutes. (b) An averaged adapter is a file no
  library wrote: the smoke run is the first time the model server takes one. (c) The adapters' numbers are taken to be
  float32 (149,913,600 of them, about 600 MB a file): the first step refuses another type, and the smoke run meets the
  real files first. (d) L5's seed 2 may not be finished when the report step runs: the first read is then NOT READ, and
  the task queued again builds the report again with the seeds that are there. (e) ONE failed check of one of ten
  models makes the run INCONCLUSIVE. (f) The ten measurements are hours apart on a pool that changes: the timeout
  shares are printed, and no check is built on them.
- **Mutated.** One hundred and fifty-nine one-change mutations of what this build added, each applied to a copy of the
  tree and never to the working tree: the decay's factor (inside the warm-up, after it, in the middle, at the last step, in
  a training no longer than its warm-up), the recipe's own factor, and what the scheduler is handed with the decay off;
  the steps of a training; what a record of the rates is held to; the three functions' options (off for every other
  stage, on as asked here, the stand-in's); L5's run of a given seed; the fixed rows (text, place, the draw and its
  seed, the order, the training's own record) and each order (its seed, the one trained before, two alike); the check
  of the rows; the figure and each term of its noise; each read's inputs, threshold and names (half, the seeds of L5,
  the t and F values, the level's sign, its interval and its edge at zero, twice the noise, the mean and not the best);
  what follows in each of its cases; in the report whose figures each read takes, each check, whose checks decide, an
  INCONCLUSIVE run; in the stage the sampling (its attempts, its seed, the rungs, the second sampling), each refusal,
  each guard of the smoke-only variable and each thing it allows, which rows and which recipe a model is trained with,
  which adapters an average takes and each thing it is held to, whose adapter is served, the seeds of L5 and when one
  is NOT THERE; the mean of adapters (a sum, three of the four, summed in float32, written at another place, a file
  that is not the mean or is one of its adapters, each refusal); the registration and the setting. The first run: 155
  noticed, 4 not. The four asked for assertions, which were added: a row that is none of the fixed rows fails the
  check though every fixed row stands beside it; an audited problem's episodes are counted once in the figure; a real
  run refuses an order that is the one L5 trained in (the stage handing the rule `loosely` for every run went
  unseen); an averaged model is not measured before its own step has made its adapter. The last run, all 159 against
  the tree with these additions: 159 noticed. SIX MORE, OF THE TRAINING FUNCTION ITSELF, which the suite cannot see
  (its environment has no torch), each put to that function's run on a CPU: the decay not handed to the scheduler, or
  counted over the rows and not the steps; the recipe as it is handed the decay; the rate read after the step's
  update, or taken from the config; the rates put in every result. Six noticed.

**Outcome of L5n (2026-10-11; `docs/spec/ladder-l5n-RESULT.md`).** Run in two tasks: the first ended at 18:10 UTC
for infrastructure with three models measured (VOID; the engine did not start, fixed at ea0a74813), the second
(`rlvr_lean/ladder_l5_noise_seed0_r2`, 20:25 to 23:11 UTC) went on from its steps. All checks pass. Successes per 1,000
attempts on all of G: `constant_1` to `_4` 112.3, 107.9, 102.2, 101.7 (standard deviation 5.0; sampling noise of one
figure 2.3); `decay_1` to `_4` 103.9, 109.9, 106.5, 96.7 (5.6); `mean_constant` 110.0 against its models' mean 106.0;
`mean_decay` 107.4 against 104.2. The reads: THE ROWS DRAWN MAKE MOST OF IT (0.00504 against L5's `published` over its
seeds, 0.01520); DOES DECAY LOWER IT: NOT SHOWN; its level NAMED NOTHING (−0.00179 [−0.01126, +0.00768]); AVERAGING
GIVES AT LEAST THE MEAN OF ITS MODELS, for both. **What follows, by the rule above: a later stage trains each model four
times and measures the averaged adapter.** L5d's setting was changed to `mean_constant` at 28f015f6a. Said beside, and
deciding nothing: the same 800 rows in five orders (these four and L5's own) give 101.7 to 112.3, where the two other
draws of 800 rows gave 78.6 and 76.5 in the one order each was trained in, so seed 0's level was not its order's;
whether the two weak draws are weak by their rows or by that order is not told apart (the note's "The reading that is
left open"), and L5d's fresh draws, each in four orders, tell it.

### L5d: the three kinds again, over fresh draws, measured on the goal set's first sampling (fixed 2026-10-11, before L5n's read and before any fresh seed runs)

**Why.** L5's three seeds left one sign for the owner's bet that repeats: on the goal set, the two models trained on
the model's own proofs beside published ones are above the model trained on published ones alone at 3 seeds of 3. It
is not a result. It was found on those seeds (the read fixed in advance named `deep` minus `published` on the
held-aside set, which shows nothing); `published` is as much low at seeds 1 and 2 as the others are high; and three
trainings resolve little. On the goal set's FIRST sampling alone (32 attempts a problem, what this stage measures),
successes per 1,000 attempts at seeds 0, 1, 2: `published` 103.8, 78.6, 76.5; `shallow` 105.0, 107.6, 95.1; `deep`
110.2, 98.1, 100.8. The mean of `shallow` and `deep`, minus `published`: +3.8, +24.2, +21.5 (mean +16.5, standard
deviation 11.1, t 2.58 on 2 degrees, two-sided p 0.12). `deep` minus `shallow`: +5.2, −9.6, +5.7 (mean +0.4): depth
is not shown to matter, which is why the two are read together here. The first sampling's figures are within about 2
per 1,000 of the 93 attempts', so the second sampling, the rungs and the held-aside set buy little for this question
and cost two thirds of a seed's five hours and a half.

**What runs: a further seed of L5, with the measurement cut to the one sampling.** For each fresh seed N: L5's
finished search is READ (seed 0's run; nothing of it is sampled again), and the three sets are built BY L5's OWN RULE
for the task's seed: which 400 of the 520 deep solves and which verified attempt of each; which 400 ordinary solves
and which attempt; the published rows (the first 400 and the first 800 of the hash order of that seed); the order.
Three models, each from `pre_r64` at its rank, one pass over its 800 rows. Each is measured on G's first sampling
alone: 32 attempts a problem with the sampling seed `pre_r64`'s run recorded (1001), as L5n measures. No rungs, no
second sampling, no held-aside set.

**The recipe is the one L5n names, and it is taken as named** (L5n's "What follows", fixed before its run):
- EVERY LATER STAGE TRAINS WITH THE DECAY: each of the three models is trained once, with the decay.
- A LATER STAGE TRAINS EACH MODEL FOUR TIMES AND MEASURES THE AVERAGED ADAPTER: each of the three sets is trained in
  four orders (order seeds 1 to 4 of its own rows, as L5n orders rows) with the recipe as it is; the model measured is
  the adapter whose every number is the mean of the four, by L5n's own function and held to what L5n holds it to.
- THE SPREAD IS THE RECIPE'S OWN AT THIS SIZE: the recipe as it is, one training a model. More seeds an arm is one of
  the two choices that case puts to the owner, and it is what this stage is.
- L5n INCONCLUSIVE, or not read: the recipe as it is, one training a model, and that is said beside every read.

**The seeds, and the owner's word.** Fresh seeds only: 3, 4, 5; and 6, 7, 8 WITH THE OWNER'S APPROVAL, ASKED FOR ON
2026-10-11 BEFORE ANY FRESH SEED RUNS (the repo's rule: more than three seeds needs it in advance). The number is
fixed before the read: SIX when the approval has come before seed 5's figures are read, THREE otherwise. Seeds added
after a read at three are a new replication, read on its own seeds alone. Why six: by the exact paired-t power, with
the effect and the standard deviation found (16.5 and 11.1, the latter known from three figures only: 5.8 to 70 at
95%), three seeds confirm the effect 31 times in 100 and six 83 times; the interval's half-width is 27.5 per 1,000 at
three and 11.6 at six. For half the effect: 12 and 32 in 100.

**The reads, fixed before any fresh seed runs.** A model's figure is its successes per attempt on all of G over the
32 attempts (L5n's figure). Every interval is over SEEDS (trainings), never over problems: mean ± t × the standard
deviation of the per-seed differences over the root of the seeds (t 4.303 at three, 2.571 at six).
- **Primary: the mean of `shallow` and `deep`, minus `published`, at each fresh seed; the fresh seeds alone.**
  Interval above zero: THE MODEL'S OWN PROOFS BESIDE PUBLISHED ONES BEAT PUBLISHED ONES ALONE ON THE GOAL SET.
  Interval below zero: THEY COST. Interval holds zero: when its half-width is above 16.5 per 1,000 (the effect found),
  UNDETECTABLE AT THIS NUMBER OF SEEDS, with the interval and no word about the effect; otherwise NOT SHOWN.
- **Beside, each with its interval, naming nothing:** `deep` minus `published`; `shallow` minus `published`; `deep`
  minus `shallow` (the depth); the number of fresh seeds at which the primary's difference is positive.
- **Where it sits:** the primary's difference (a) on the goal problems `pre_r64` solves in its own first sampling
  (200) and (b) on G′ (the 192 it does not). Named RELIABILITY when (a)'s interval is above zero and (b)'s holds zero;
  REACH TOO when (b)'s is above zero. Read only when the primary names the first outcome.
- **Counts, at each seed:** goal problems solved in the 32 attempts by each model (gained and lost against
  `published`), and of the 74 long problems (8 lines or more).
- **The spread:** each kind's figures over the fresh seeds and their standard deviation, with the sampling noise
  beside. `published`'s figure at each seed is the fresh look at the lead L5n's first figures gave: whether the bar's
  level goes with which published rows were drawn.
- **Exploratory, marked as such:** when the recipe is the one L5 trained with, L5's three seeds beside the fresh ones
  on the same first sampling, and the nine pooled, with seeds 0 to 2 marked as the ones that found the sign.
- **Timeouts:** each set's share of attempts that timed out in Lean is printed; where two models of one seed differ
  by more than a point, the seed's timed-out attempts are checked again (Lean only, no measurement running) and the
  reads are given both ways.

**Checks, read first; any FAIL at a seed leaves that seed out and the run INCONCLUSIVE until the seed is run again.**
Each set is L5's rule for its seed (made again and held, as a further seed of L5 holds it): no row with text, no
barred row (no held-out problem; the published rows of the `pretrain` half; no problem of G or of the held-aside
set). Each trained adapter is not the start adapter's and is read back at its rank; an averaged adapter is the mean of
its four, bit for bit, and is the one served. Lean answered (at most 0.02 of a set without an answer). The stored
first sampling of `pre_r64` is the setting's (its problems, its seed, its 32 attempts).

**What follows, fixed now.**
- THE MODEL'S OWN PROOFS BEAT: the bet's first condition holds on the goal set at this size. L6 (a second round) runs
  with this recipe and with as many seeds an arm as this stage's measured standard deviation asks for its own
  difference; the cycle at full size has its evidence to put to the owner.
- UNDETECTABLE or NOT SHOWN: the sign of L5's three seeds is not confirmed. The interval, the standard deviation and
  the seeds a decision would cost go to the owner; L6 does not run on it.
- THEY COST: stop. The nine seeds go to the owner.

**What it cannot say.**
- G alone and 32 attempts. Nothing of the held-aside set: reach on OTHER hard problems was not shown at three seeds
  in L5 and is not measured here.
- **The bar is a second pass.** `published` trains on rows `pre_r64` was already trained on; the model's own proofs
  are of problems it never trained on. So the first outcome says: its own proofs of new problems are worth more than
  reading published proofs again. It does not say they are worth more than PUBLISHED proofs of new problems. That
  comparison needs a fourth kind: the same 400 problems as `deep`, with each problem's published proof (its pool
  certificate) in place of the model's own. Those problems are outside the `pretrain` half, so it is not built without
  the owner's word (asked 2026-10-11 with the seeds). With the word it is specified then, and trained on these same
  seeds beside the three.
- **One search.** The 520 deep solves and the 1,114 ordinary ones come from seed 0's search and map. Two seeds' deep
  rows share about 308 of 400 problems; two seeds' published rows share about 25 of 800. The spread over seeds here
  is over draws and trainings, not over searches.
- One start model, one size of set, one pass.

**Budget.** A seed, by the measured step times of L5n: three trainings of about four minutes and three measurements
of about half an hour: an hour and three quarters. With the averaged recipe, twelve trainings and three averages:
about two hours and a half. Three seeds: five to eight hours. Six: ten to fifteen. One task a seed, one after
another; a smoke stage first.

**Made exact by the build (2026-10-11; nothing queued, nothing run on a GPU, nothing sent to Lean; the reads, the
checks, the seeds and "What follows" above are not changed; the readings taken where the section left something open,
what was NOT built, what the smoke stage does not exercise, what was measured before any run and what is fragile are
below).**
- **The stages.** `ladder_l5_draws` and `ladder_l5_draws_smoke` (`gpu/ladder_l5_draws.py`; rules
  `domain/ladder_round/l5_draws.py`; report `reporting/ladder_l5_draws.py`; the read over the seeds again from pulled
  task directories, `tools/ladder_l5_draws.py`), settings `ladder_loop.l5_draws` (`seeds: [3, 4, 5]`,
  `more_seeds: [6, 7, 8]`, `more_seeds_approved: false`, `sampling: reach`, `attempts: 32`, `recipe: constant`; one
  reader; the start and the search's seed are L5's own settings, L5's seeds and the order seeds L5n's, and no setting
  of this block names them). Run directory `ladder_l5_draws_pre_r64_seed<N>`. Jobs `ladder_l5_draws_seed3.json`
  to `_seed8.json` (est_minutes 210 each: twice the section's hour and three quarters, and above the averaged
  recipe's two hours and a half) and `ladder_l5_draws_smoke.json` (30), with the `resources` of L5n's jobs key for
  key: job configs, not shipped in this copy. L5n's jobs hold no `setup` key, and these hold none.
- **TWENTY-THREE steps, those of the LARGEST recipe.** The stage's list in `runner/entry.py` is fixed before the recipe
  is known, so it holds every recipe's steps: `ladder_l5_draws_prepare`; then model by model (`published`, `shallow`,
  `deep`) `ladder_l5_draws_train_<model>`, `ladder_l5_draws_train_<model>_1` to `_4`, `ladder_l5_draws_average_<model>`,
  `ladder_l5_draws_measure_<model>`; `ladder_l5_draws_report`. A STEP THE RUN'S RECIPE DOES NOT USE MAKES NOTHING: it
  returns `run: false` with the reason and marks nothing. `constant` and `decay` use eight (the first step, three
  trainings, three measurements, the report); `mean_constant` twenty (the first step, twelve trainings, three
  averages, three measurements, the report). Each step is resumable; a finished one returns what it stored. The
  report step ALONE is built again each time its task is queued, so that the read over the seeds takes the seeds
  that are there by then.
- **Existing functions are called, not copied.** L5's finished run is read by L5's own first-step reading
  (`ladder_l5.read_the_search`, `read_for_the_search`, under the words an `OfTheSearch` gives: a further seed's
  refusals, as L5n and L6 read it). THE THREE SETS ARE DRAWN BY THE VERY CODE A FURTHER SEED OF L5 DRAWS THEM BY: two
  bodies were cut out of L5's three sets step, which now calls them (`ladder_l5.deep_solves_by_the_seed`: the deep
  solves, which of them the cap keeps and which verified attempt of each; `sets_by_the_seed`: the ordinary solves, the
  published rows, the three sets and their barred rows; with `the_stored_search` and `counted_by_the_search`). A
  training is L4t's own body (`ladder_l4_rows.train_a_model`); a measurement the ceiling's own
  (`ladder_ceiling.measure_model` with `rungs=False` and ONE sampling handed to it, as L5n measures). The averaged
  adapter is L5n's own body and check, each cut out of L5n's step and report, which now call them
  (`ladder_l5_noise.the_mean_adapter`, under the names a `Stage` gives; `reporting/ladder_l5_noise.the_mean_check`).
- **The recipe is a SETTING, held to L5n's report by the first step** (`l5_draws.the_recipe`). The first step reads
  L5n's REAL run (`ladder_l5_noise_pre_r64_seed0`: its report's marker and its report), only reads it, and never
  refuses for its absence. Its report there, to be read, not INCONCLUSIVE and naming a case of its "what follows": the
  setting must be that case's recipe (`decay`, `mean_constant`, `constant`, in the section's order), refused otherwise
  with both named and L5n's task. Its run not there, not finished, not to be read, INCONCLUSIVE or naming no case: the
  setting must be `constant`, and a sentence saying which of these it was is recorded (`the_recipe`) and printed by
  the report and by the tool beside the primary, where it sits and what follows. SO A SEED QUEUED BEFORE L5n's REPORT
  IS WRITTEN TRAINS `constant` AND SAYS SO: the jobs say to queue after it. A RUN KEEPS ITS RECIPE: every later step
  refuses when the setting names another than the run was prepared with (the run is moved away to run that seed with
  another). SEEDS OF TWO RECIPES ARE NEVER READ TOGETHER: the rule refuses, and the report and the tool then give NOT
  READ with each seed's recipe.
- **Each recipe, exactly.** `constant`: the call L5 trains a seed's model by, with no schedule handed to it; no
  learning rate is recorded and none held, as in L5's own runs. `decay`: L5n's `decay` (the factor of its "The decay,
  exactly"), each step's rate recorded and held to that schedule by the training's own step, as L5n holds it; the set
  in the seed's own order. `mean_constant`: each set in `assembly.training_order` of the order seeds L5n's setting
  names (1 to 4), stored as `l5_draws_training_<model>_<k>.jsonl` (a published row by id and place, never with text),
  trained with the recipe as it is and its rates recorded and held; the four are `<model>_1` to `_4`, the averaged
  model keeps the kind's name and its adapter is `adapters/<model>`; no model is then trained in the seed's own
  order. AN ORDER SEED THAT IS THE TASK'S SEED GIVES THE SEED'S OWN ORDER (order seed 3 at seed 3, 4 at seed 4:
  measured below). L5n refuses such an order because the model trained in it stands beside; here none does, so it is
  trained all the same and named (`the_orders`). Two order seeds that give one order are refused. The first step
  holds the start adapter's types to those the mean is written in, under `mean_constant` alone. The learning-rate
  factor the first step records is the run's own recipe's, and none under `constant`.
- **The seeds.** A task's seed is held by the first step (`l5_draws.the_seed`): a seed of L5's own (L5n's
  `published_seeds`), a seed of `more_seeds` while `more_seeds_approved` is false, and a seed of neither list are
  refused with nothing written. THE READS ARE OVER `seeds` WHILE THE APPROVAL IS NOT RECORDED AND OVER BOTH LISTS ONCE
  IT IS (`fixed_seeds`), read from the setting when the report is built. The code cannot hold WHEN the approval came:
  `more_seeds_approved` is set to true only if it came before seed 5's figures were read. Once it is true the read
  over three is no longer made, and the read is NOT READ until 6, 7 and 8 are there. A NEW REPLICATION after a read
  at three is the setting `seeds: [6, 7, 8]`, `more_seeds: []`: the runs of 3, 4 and 5 are then named as not of the
  fixed number and left out. The settings reader refuses seeds that are not distinct whole numbers, an approval that
  is not true or false, and a number of seeds the interval is not given for (it is given for two to nine: t 12.706,
  4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306).
- **What else the first step refuses before anything is written.** L5's own refusals (its run missing or unfinished;
  a report not to be read or INCONCLUSIVE; made otherwise, with the three stored runs that task read). A sampling that
  is not L5n's (the settings reader) or not the FIRST sampling `pre_r64`'s run recorded, by its name and its attempts
  (the sampling SEED is the recorded one, 1001, and no setting). A recipe (the learning rate, the warm-up steps, the
  batch, the adapter's rank, alpha and modules) that is not the one L5's run recorded. Fewer deep solves than L5's
  minimum. A set with a problem of G or of the held-aside set (`l5_draws.refuse_problems_of`; L5's own rule already
  refuses the held-aside ones and the barred rows). The smoke-only variable outside a smoke run. A refused first step
  leaves the run's directory EMPTY (the store makes it), and no file in it.
- **What the first step writes.** The three sets (`l5_draws_training_<model>.jsonl`), and under `mean_constant` the
  twelve ordered ones; the goal problems of each model's set (`problems.jsonl`, the sets `l4_reach_<model>`: L5's and
  L5n's name for a model's first sampling, so that every stored first sampling pairs by problem); and COPIES of the
  groups, G′, the held-aside problems and `pre_r64`'s stored first sampling. L5's own files of the deep solves and of
  the ordinary solves are not written again: the sets hold their rows.
- **Readings the section left open, and the ones taken.** (1) **The figure** is L5n's (`l5_noise.figure`: successes
  over attempts, an audited problem's episodes counted once, its binomial noise), on all of G, on (a) and on (b).
  (2) **The parts.** (a) is the goal problems `pre_r64` solved at least once in the stored first sampling this run
  copied; (b) the ones it did not, HELD to be the stored G′ (L5's reading made G′ from that sampling); were they
  not, no figure would be given on a part and where it sits would be NOT READ. (3) **The primary's outcome** is
  named from the exact mean and half-width (what is printed is rounded to the sixth decimal, and where it sits is
  named from the rounded); an interval that ends at zero holds zero; a half-width that is exactly 16.5 per 1,000 is
  not above it (NOT SHOWN). (4) **Where it sits** names its two cases and, for any other pair of intervals, NAMED
  NOTHING with both given; REACH TOO is read first; an interval on G′ that begins at zero holds zero; NOT READ when
  a part holds no problem at a seed. (5) **The spread** is each kind's figure at each fresh seed, the standard
  deviation over the seeds (n − 1) and, beside, the root of the mean square of its figures' sampling noises.
  (6) **The counts** are of the goal problems solved at least once in the 32 attempts, gained and
  lost against `published`, and the same among the long problems (the last length group, 8 lines or more, as L5's
  reports count them). (7) **A missing seed before a failed one.** With a seed of the fixed number not there the read
  is NOT READ, naming it, and a seed whose check failed is said beside; with all there and one failed, INCONCLUSIVE,
  naming the seed and its checks. A seed is THERE when its report step is done and its report is of that seed. A seed
  that is there and is not of the fixed number is named and left out. Seeds that were not measured alike (the
  sampling, its attempts, its seed, the number of problems) are not pooled either. (8) **A seed's own verdict.** Any
  failed check of any of its models, and the seed is INCONCLUSIVE: its figures are kept apart
  (`measured_and_not_read`) and not printed, and it hands none to the read. A set Lean did not answer also fails the
  report step, and the task queued again samples that set again from its adapter. Any other failed check: the run is
  moved away and the seed run again. (9) **"The seeds a decision would cost"** are given as the interval's half-width
  at each number of seeds from two to nine with the standard deviation measured, and the least number at which it is
  at most the effect found; no power is computed by the code. (10) **Exploratory.** L5's seeds are read by the report
  step from their own runs when those are on the box and finished (NOT THERE otherwise, never refused), on the same
  sampling and the same parts, under `constant` alone; their figures are stored in the report (`the_seeds_of_l5`), so
  that the tool needs no run of L5; the pool is read only when all three are there. The tool takes them from the
  first report that holds them all. (11) **The check of the start's sampling** holds this run's copy: its problems are
  the goal problems, each once, each with the setting's attempts; its name and attempts are the setting's AS IT IS
  WHEN THE REPORT IS BUILT; a sampling seed is recorded. (12) **The rows a model was trained on** are the training
  loop's own record, by id, held to the stored set (each row once, none else); the barred rows are L5's own check
  (`l5.no_barred_row`). (13) **The read over the seeds is made by a seed's report step**, from the runs on the box
  then: the last seed to finish makes it, any seed's task queued again makes it again, and the tool makes it from the
  pulled task directories (`python -m rlvr_lean.tools.ladder_l5_draws <one directory a seed> [--out read.json]`; no
  GPU, no Lean; it refuses, with exit code 2, a directory without its report, a report that is not L5d's and two
  directories of one seed). A seed's OWN report is not built by the tool: its report step reads `pre_r64`'s run and
  the snapshot's pretraining file beside the seed's run.
- **OPEN: the timed-out attempts are NOT checked again.** The section's "Timeouts" asks that, where two models of one
  seed differ by more than a point, the seed's timed-out attempts be checked again and the reads given both ways.
  BUILT: each set's share is printed and stored, and the pairs more than a point apart are named (`l5.timed_out`, as
  L5's further seeds name them), with `checked_again: false`. NOT BUILT: the check again (a Lean-only step or tool
  that asks those attempts again and makes each read a second time on the answers), as it is not built for L5. Where
  a pair is named, the reads stand as given once and the report says so.
- **The smoke stage** (`ladder_l5_draws_smoke`, task seed 3) runs the same twenty-three steps and reads
  `ladder_l5_deep_smoke` and the three smoke runs that task read; it writes `ladder_l5_draws_smoke` (with a recipe
  that is not `constant`, `ladder_l5_draws_smoke_<recipe>`, so that one recipe's smoke run does not stand in
  another's way). HOW IT GOES ON: `RLVR_LEAN_LADDER_L5_DRAWS_SMOKE=1`, set by this stage ALONE and refused by the
  first step unless the run directory is one the task names AND the map is the fixtures'. Under it, each recorded:
  L5's smoke run, whose report is INCONCLUSIVE, is read all the same (one NOT TO BE READ never is); its two looser
  draws are made again as it recorded them (L5's own smoke-only variable is not set); one deep solve is enough (D =
  1, three sets of two rows); orders of two rows that are alike are trained all the same; and an averaged adapter
  that IS one of its models is kept and served. THE RECIPE IS THE SETTING'S, HELD TO L5n's REAL REPORT as a real
  run's is. WHAT IT DOES NOT EXERCISE: the two recipes the setting does not name (as the config stands, a training
  with the decay asked for, the four orders, an averaged adapter made and served: when L5n names one of them the
  setting is changed and THE SMOKE IS QUEUED AGAIN under it, before the first real seed); a learning rate that FALLS
  under any recipe (a training of two rows is one step, inside the warm-up); the read over the seeds (a run the task
  names reads no other seed's run: NOT READ); where it sits (the fixtures' start model solves the one goal problem,
  so G′ is empty); L5's seeds 1 and 2 beside; the refusals of a real run (an INCONCLUSIVE run of L5, a further seed
  without the approval, two orders alike); the real sizes.
- **Measured before any run, on the pulled copies** (links in a scratch store; no GPU, no Lean; the pulled directories
  only read: the sizes and times of their 489, 82, 80 and 700 files are as before). The first step ran to its end
  for each of the seeds 3 to 8 (6 to 8 with the approval set in a scratch copy of the settings); it REFUSED, with no
  file written, the seeds 0, 1 and 2, the seeds 6, 7 and 8 without the approval, and seed 9. AT EVERY SEED: the
  search of seed 0 (1,000 problems, 520 deep solves), D = 400; `published` 800 published rows, `shallow` 400 ordinary
  solves and 400 published rows, `deep` 400 deep rows and 400 published rows; NO ID TWICE IN A SET, no problem twice,
  no problem of G (392) or of the held-aside set (600), no published row with text; `shallow` and `deep` hold the
  SAME 400 published rows, which are among `published`'s 800; 100 optimizer steps a training; the adapter seed is
  the task's; the sampling `reach`, 32 attempts, sampling seed 1001; the recipe `constant`, said beside every read
  (L5n's report is not among the pulled copies). The parts of G are the section's: `pre_r64` solves 200 goal
  problems in its stored first sampling and G′ holds the 192 others; 74 goal problems are long. THE SETS ARE WHAT A
  FURTHER SEED OF L5 BUILDS: L5's own two prepare steps, run as a further seed of L5 for each of the six seeds in the
  same scratch store, build D = 400 and three sets equal to this stage's row for row and field for field. WHAT TWO
  SEEDS SHARE, over the 15 pairs of the six seeds: DEEP PROBLEMS 300 to 311 of 400 (mean 306.3; seeds 3 and 4: 303,
  3 and 5: 309, 4 and 5: 304), of which 152 to 172 with the same verified attempt (mean 164.7); ordinary problems
  134 to 163 of 400 (mean 145.7); PUBLISHED ROWS of `published` 18 to 31 of 800 (mean 26.1; 31, 25, 29 for those
  three pairs); published rows of `deep` and `shallow` 2 to 11 of 400 (mean 6.7). The section's "about 308" and
  "about 25" are these. All 520 deep solves stand in some seed's set and 106 in every one of the six; of the seeds
  3, 4 and 5, 515 in some and 231 in each. The six seeds' `published` sets hold 4,420 distinct rows of their 4,800.
  THE FOUR ORDERS, by the rule alone (no such run is prepared while the setting is `constant`): four distinct orders
  of each set at every seed and no two alike; at seed 3 the order seed 3, and at seed 4 the order seed 4, give the
  seed's own order.
- **Existing stages write what they wrote.** In the scripted world, with the code before the build and after it: L5
  at the search's own seed is the same 108 files and 14 step summaries, a further seed's the same 97 and 14, L6's the
  same 149 and 17, L5n's the same 112 and 22 (timings and the copy's own path apart). Asserted in the suite: with
  this stage run beside them, L5's own task queued again returns what is stored and moves no file, a further seed of
  L5 run AFTER this stage leaves the files a further seed leaves, L5n run after it leaves L5n's files and hands its
  trainings what it handed them, its averaged adapters made by the body the two stages share; L6 writes its files
  with this stage in the code. Three assertions that state where things live were brought up to date (one more smoke
  stage names the rank check's smoke run, the map's, L5's smoke run and L4t2's).
- **Fragile, and not shown without the box.** (a) THE RECIPE DEPENDS ON WHEN A SEED IS QUEUED: before L5n's report is
  written a seed trains `constant`. If L5n then names another recipe, that seed's run is of another recipe than the
  later ones, is not read with them, and is moved away and run again. (b) ONE failed check of one model makes its
  seed INCONCLUSIVE and the read over the seeds INCONCLUSIVE until that seed is run again. (c) The read is NOT READ
  until the last seed of the fixed number has written its report; the report that makes it is the last seed's, and
  an earlier seed's report on the box still says NOT READ until its task is queued again. (d) Fifteen of the
  twenty-three steps make nothing under `constant` and `decay`; each still starts its process behind the guard
  between steps. A guard after a step that held no model passes only while the card's other users hold at most 300
  MiB more than at the guard before it (as after L5's own steps that make nothing); one that fails stops the task,
  and the task queued again goes on from the first step not done. What the fifteen cost in time is not measured.
  (e) The sampling engine starts with 0.3 GiB to spare beside the desktop's reserve, three times a seed; a start
  that fails is run again by the entry shim, twice at most. (f) Under `mean_constant` an averaged adapter is made and
  served here for the first time outside L5n: its smoke run and its real run come first. (g) A seed's three
  measurements are half an hour apart and the seeds hours or days apart on a pool that changes: the timeout shares
  are printed, no check is built on them, and the check again is not built (OPEN, above). (h) The report step reads
  `pre_r64`'s run and, for the exploratory lines, L5's three runs on the box: those not there are NOT THERE, and the
  exploratory lines NOT READ.
- **Mutated.** Two hundred and forty-four one-change mutations of what this build added, each applied to a copy of the
  tree and never to the working tree: the recipe held to L5n's report (a run that is not there, a report INCONCLUSIVE or
  not to be read, which case asks for which recipe, what is said beside the reads); each recipe's trainings and
  orders; the seeds and the approval; each check (a published row with text, a set out of its order, the rule and its
  seed, a goal or held-aside problem, a row twice, missing or a stranger, the stored sampling); the parts of G, the
  counts and each difference; the interval over seeds (the deviation, the root of the seeds, each critical value), the
  primary's four outcomes and their edges, the effect found; where it sits and its edges; what a number of seeds
  would resolve; what follows in each case; the read over the seeds (two recipes, seeds measured otherwise, a missing
  seed, a failed one, one that is not of the fixed number, the part and the difference each read takes, the
  exploratory pool); in the stage the settings' refusals, L5n's report, each refusal of the first step, each guard
  of the smoke-only variable and each thing it allows, by whose seed the sets are drawn, what is copied, a run that
  keeps its recipe, the steps of another recipe, each recipe's schedule, the average, the measurement (the rungs, a
  second sampling, whose adapter, the rank), the other seeds' runs and L5's, the report step; in the report each
  check, whose checks decide, an INCONCLUSIVE seed and what it hands on, what is printed; the tool; the registration
  and the setting; and, put to the tests of L5, its further seeds, L6 and L5n as well, the two functions cut out of
  L5's three sets step and L5n's averaging body and check. The first run: 215 noticed, 29 not. The twenty-nine asked
  for assertions, which were added. ON THE RULES: a training of nothing on a set of nothing does not pass; the
  outcome is named from the interval as computed and not as it is printed (one that begins three ten-millionths
  above zero; a half-width four hundred-millionths above the effect found); a seed at which the difference is zero
  is not counted positive; an interval on G′ that begins at zero is RELIABILITY; a half-width that is exactly the
  effect found resolves; the exploratory pool holds L5's own differences and the fresh seeds' primary ones on all of
  G; the tool reads over six seeds once the approval is recorded. ON THE STAGE: the start adapter's type is held
  under the averaged recipe alone; a real run asks the setting's minimum of deep solves (the scripted world's is two
  now); the search's count is held to what its own task recorded; G′ is copied as L5's reading hands it on, and a
  stored sampling it was not made from gives no figure on a part; a real run refuses an averaged adapter that is one
  of its models, and its report fails one; the other seeds are read from their runs (the further ones too; an
  unfinished run, a report of another seed and a run the task names are not there); the stored first sampling is
  held to the setting as it is when the report is built; an adapter read back at another rank fails; a row of a goal
  problem is barred and of G, a row of a held-aside problem of the held-aside set, both on the training loop's own
  record and not on the stored set; the long problems are the last length group; and, in L5's own two functions,
  through this stage, the map is held to the one the search was prepared on and a set with a held-out row is refused
  (the tests of L5, its further seeds, L6 and L5n that the pass ran had seen neither). The last run, all 244 against
  the tree with these additions: 244 noticed. The pass's own control, a copy with nothing changed, passes its tests
  (48 of this stage's; 133 with those of L5, its further seeds, L6 and L5n).

**Outcome of L5d at three seeds (2026-10-12 UTC; `docs/spec/ladder-l5d-RESULT.md`): THE MODEL'S OWN PROOFS
BESIDE PUBLISHED ONES BEAT PUBLISHED ONES ALONE ON THE GOAL SET.** The number of seeds is THREE: seed 5 finished unread
at 10:59 UTC, and the owner answered the ask before it was read (about 11:06 UTC: "let's do 3 seeds. I
don't need super high confidence to keep pushing in this direction"). Seeds 6 to 8 were not run. All checks pass at
each seed. The primary, per 1,000 attempts: seed 3 +13.91, seed 4 +13.07, seed 5 +11.08; mean +12.69, interval over
the three seeds [+9.08, +16.30]. Per 1,000 on all of G, `published` / `shallow` / `deep`: 90.88 / 99.41 / 110.17,
95.34 / 105.31 / 111.53, 97.58 / 102.84 / 114.48. Beside: `deep` minus `published` +17.46 [+13.41, +21.50];
`shallow` minus `published` +7.92 [+1.93, +13.91]; `deep` minus `shallow` +9.54 [+2.31, +16.77]. Where it sits: REACH
TOO (on the 200 [+9.19, +31.70]; on G' [+0.26, +8.97]). The standard deviation of the primary's difference over the
seeds is 1.45. Seed 3's difference with the second check of its timed-out attempts is +13.08; seed 4 owes none; seed
5's is in the note. **What follows by the rule above:** the bet's first condition holds on the goal set at this size
and a second round runs. It is run from the best model and not from `pre_r64` (L6b, below, with its reasons): from
the twice-pretrained model the same difference read -14.87 at the one seed of L5f, with own proofs that the weaker
model found. L6 as fixed does not run before L6b is read.

### L5s: the dose. Every own solve in stock that the model finds less than half the time, against the same number of published proofs (fixed 2026-10-12, after L5d's seed 3 and before any run of this stage)

**Under the owner's words** (2026-10-12): "make as much progress as you can tonight on improving this solver's solve
rate against the heldout tasks." Labelled **pretrained on published proofs** (it starts from `pre_r64`).

**Why.** The one lever with a repeated sign for the solve rate on the held-out goal set is the model's own
Lean-checked proofs of problems it never trained on: beside published proofs they are above published proofs alone
at 3 seeds of 3 in L5 (+16.5 per 1,000 attempts on average) and at L5d's first fresh seed (+13.9; `published` 90.9,
`shallow` 99.4, `deep` 110.2, each the average of four orders). Every one of those models was trained on 400 of its
own proofs. Nothing here has measured what MORE of them does. And nothing trained from `pre_r64` has read above
113 per 1,000: 800 rows of any kind, 12,433 and 24,866 published rows all land between 76 and 113. In stock, found
and checked already, are 1,634 own solves of problems `pre_r64` solves less than half the time: the 520 deep solves
of L5's search (failed 8 of 8, solved within 64) and the 1,114 base-map problems it solved in 1 to 3 of its 8 map
attempts. This stage trains on ALL of them, four times the dose, and reads whether the solve rate goes with the dose.
It is the question the full-size cycle (`ladder-cycle.spec.md`, seventeen times the dose) rests on, asked tonight at
the size that needs no new search.

**What runs: one task, no search, no sampling but the measurements.** The seed is 3, L5d's first fresh seed: this
stage's sets then HOLD that seed's rows (its 400 deep solves, its 400 ordinary solves, its first 800 published rows),
and its two models stand beside that seed's three with the same published order, the same recipe and the same
measurement. L5's finished search is READ as L5d reads it. Two sets, built with L5's own functions for the task's
seed:
- `stock`: EVERY deep solve of the search, one verified attempt of each drawn by the seed as L5 draws it; EVERY
  ordinary solve as L5 defines one (its own setting: a base-map problem solved in 1 to 3 of its 8 map attempts), one
  stored verified map attempt of each drawn by the seed as L5 draws it; and the first N rows of the seed's hash order
  of the pretraining file, N the number of own rows. Expected 520 + 1,114 = 1,634 own rows and 3,268 rows in all; the
  counts are the stored runs', recorded, never numbers of the code.
- `published_stock`: the first 2N rows of that order and nothing else. The bar at equal steps.
Each set is trained from `pre_r64` at its rank, one pass, in four orders (order seeds 1 to 4 of its own rows) with
the recipe as it is, and the model measured is the adapter that is the mean of the four (L5n's function, held as L5n
holds it): the recipe L5n named, as L5d runs it. Each of the two is measured on G's first sampling alone, 32 attempts
a problem with the sampling seed `pre_r64`'s run recorded (1001). `stock` is trained, averaged and measured first,
then `published_stock`. A published row is stored by id and place, never with text.

**The reads, fixed before the run.** One seed: a scout, by the repo's seed rule. A model's figure is its successes
per attempt on all of G over the 32 attempts. Two averaged models of the same rows differ by a standard deviation of
about 0.0048 (the order's part of an averaged model about 0.0025, the sampling noise 0.0023, each twice).
- **Checks, read first; any FAIL: INCONCLUSIVE, and nothing else is said.** L5d's own for a seed: each set is its rule
  made again and held (`stock` holds every deep solve and every ordinary solve the rule gives and no other own row;
  the published rows are the first N and the first 2N of the seed's order), no row with text, no barred row (no
  held-out problem; published rows of the `pretrain` half alone; no problem of G or of the held-aside set); each
  trained adapter is not the start adapter's and is read back at its rank; each averaged adapter is the mean of its
  four, bit for bit, and is the one served; Lean answered (at most 0.02 of a set without an answer); the stored first
  sampling of `pre_r64` is the setting's.
- **The dose (the scout's read).** `stock` minus `deep` of L5d's seed 3 (read from that seed's finished run; when
  the run is not on the box the read is NOT READ and that is said). At or above zero: THE SOLVE RATE DOES NOT FALL
  WITH THE DOSE: ESCALATE, seeds 4 and 5 of this stage run (three seeds conclude). Below zero by more than 0.010
  (twice the standard deviation above): MORE OF ITS OWN PROOFS COSTS AT THIS SEED; no further seed. Between: NOT
  SHOWN AT ONE SEED; no further seed is queued by this read.
- **Its own proofs at four times the dose.** `stock` minus `published_stock`: L5d's primary at this dose, printed
  beside that seed's own (+0.01391 at 400). One seed names nothing; three do, with L5d's rule and names.
- **The steps without its own proofs.** `published_stock` minus `published` of L5d's seed 3 (3,268 rows against
  800 of the same order), printed. `stock` differs from that seed's models by its steps too; this is the same change
  of steps with no own proof.
- **The solve rate against the start model.** Each of the two against `pre_r64`'s stored first sampling, paired by
  problem: per attempt on all of G, on the 200 goal problems `pre_r64` solves in that sampling and on G′ (the 192 it
  does not); the goal problems solved in the 32 attempts, gained and lost.
- **Overfitting, for each model against `pre_r64`** (the owner: "without overfitting"): the goal problems lost
  beside the ones gained; the share of distinct attempts on G; the 74 long goal problems (8 lines or more) solved,
  gained and lost.
- **Timeouts:** each set's share; where the two differ by more than a point, the timed-out attempts are checked
  again (Lean only, no measurement running) and the reads are given both ways.

**What follows, fixed now.** ESCALATE: seeds 4 and 5 of this stage, each after L5d's own run of that seed; over the
three, `stock` minus `published_stock` is read with L5d's primary rule, and the dose with an interval over seeds.
The averaged adapter that reads highest on G's first sampling is then the one measured in full (the rungs, the
second sampling, the held-aside set), a step that is not built here. COSTS, or NOT SHOWN: no further seed; the dose
is not the lever at this size, and that goes into the owner's decision on the full-size cycle with its interval.

**What it cannot say.** One seed. The bar is a second pass over proofs `pre_r64` has trained on, where the own
proofs are of problems it has not (as in L5d). The own proofs are of base-map problems, found by `pre_r64` alone.
Nothing here is on the held-aside set or the rungs. A model's figure on 32 attempts.

**Budget.** Eight trainings of 3,268 rows (about 16 minutes each at 800 rows in 3.9 minutes) and two measurements of
about 28 minutes: about three hours and a quarter on the one GPU. A smoke stage on the fixtures first.

**Made exact by the build (2026-10-12; nothing queued, nothing run on a GPU, nothing sent to Lean; the sets, the
reads, the checks and "What follows" above are not changed; the readings taken where the section left something
open, what was NOT built, what the smoke stage does not exercise and what is fragile are below).**
- **The stages.** `ladder_l5_stock` and `ladder_l5_stock_smoke` (`gpu/ladder_l5_stock.py`; rules
  `domain/ladder_round/l5_stock.py`; report `reporting/ladder_l5_stock.py`), settings `ladder_loop.l5_stock`
  (`seeds: [3]`, `more_seeds: [4, 5]`, `escalated: false`; one reader, `the_settings`; the start and the search's
  seed are L5's own settings, the sampling, its attempts and the recipe L5d's, the order seeds L5n's, and no setting
  of this block names them). Run directory `ladder_l5_stock_pre_r64_seed<N>`. Jobs `ladder_l5_stock_seed3.json`
  (est_minutes 230: the budget's three hours and a quarter, and room for a slower Lean pool) and
  `ladder_l5_stock_smoke.json` (30), with the `resources` of L5d's jobs key for key. No job is written for seeds 4
  and 5: they are written when the escalation is recorded.
- **The seeds.** A task's seed is the scout's (`seeds`) or, ONCE `escalated` IS TRUE, one of `more_seeds`
  (`l5_stock.the_seed`): a task of a further seed is refused by the first step until the setting records the
  escalation, and a task of any other seed always. The settings reader refuses seeds that are not distinct whole
  numbers, a seed that is none of L5d's (`l5_draws.seeds` and `more_seeds`: this stage's sets hold the rows of L5d's
  own run of the seed), an `escalated` that is not true or false, and a recipe of L5d's that is not `mean_constant`
  (this stage trains each set in four orders and measures the averaged adapter, and has no other path). Nothing in
  the code queues a seed or sets `escalated`: after a report that reads ESCALATE the setting is changed by hand, in a
  commit, and seeds 4 and 5 are queued, each after L5d's own run of that seed.
- **FOURTEEN steps,** in this order: `ladder_l5_stock_prepare`; `ladder_l5_stock_train_stock_1` to `_4`,
  `ladder_l5_stock_average_stock`, `ladder_l5_stock_measure_stock`; `ladder_l5_stock_train_published_stock_1` to
  `_4`, `ladder_l5_stock_average_published_stock`, `ladder_l5_stock_measure_published_stock`;
  `ladder_l5_stock_report`. Each is resumable and a finished one returns what it stored; the report step ALONE is
  built again each time its task is queued, so that the dose is read once L5d's run of the seed is there. No step
  deletes an adapter, and no step writes into a run it reads.
- **Existing functions are called, not copied; three bodies were cut out of the stages that held them, which now call
  them.** L5's finished run is read by L5d's own first-step reading, `ladder_l5_draws.drawn_by_the_seed` (cut out of
  L5d's first step: L5's run and every stored run its task read by `ladder_l5.read_the_search` and
  `read_for_the_search`, a further seed's refusals, how a model is measured and trained, the deep solves by the task's
  seed, L5d's three sets at that seed, and the averaged recipe's hold of the start adapter's type), under the words
  an `OfASeed` gives (who reads, and which two variables name a smoke run). The ordinary solves are drawn by
  `ladder_l5.ordinary_by_the_seed` (cut out of `sets_by_the_seed`: the map's rows, the ordinary solves the rule and,
  in a smoke run alone, the run's minimum give, and one stored verified map attempt of each drawn by the seed). The
  checks of each model and their lines are `reporting/ladder_l5_draws.checks_of_the_models` and `lines_of_the_checks`
  (cut out of L5d's report). `l5_draws.the_counts` takes the model a count is against (`against`, `published` by
  default). A training is L4t's own body (`ladder_l4_rows.train_a_model` with the schedule L5d's `mean_constant`
  hands it), an average L5n's (`ladder_l5_noise.the_mean_adapter`), a measurement the ceiling's
  (`ladder_ceiling.measure_model` with `rungs=False` and the one sampling). L5, a further seed of it, L5n, L5d and L6
  write what they wrote: their own stage tests pass unchanged (the four edits to earlier tests are lists of the
  stages that name a variable, each with this stage's smoke added).
- **The two sets, made exact.** `l5_stock.the_sets`, held by `hold_the_sets`.
  - EVERY deep solve: the problems L5's search solved, with no cap; one verified attempt of each, drawn by the task's
    seed by `ladder_l5.deep_solves_by_the_seed` (the function a further seed of L5 draws by, the cap lifted).
  - EVERY ordinary solve: the problems of the start model's map solved in `l5.shallow.solved_at_least` to
    `solved_at_most` of its attempts (1 to 3 of 8), in the map's order, less the problems the search attempted and the
    held-aside ones (`l5_stock.ordinary_of_the_rule`). Both of those were drawn from the problems the map failed in
    all 8 attempts, so at the real sizes the leave-out removes no ordinary solve: it is the rule L5 draws under,
    stated. There is no draw of WHICH ordinary solves; which stored verified map attempt of each is drawn by the seed.
  - The published rows: `l4_rows.rehearsal_rows(pretraining, count, seed)`, the call L5's and L5d's published rows are
    the first of, with N and with 2N. A file with fewer than 2N rows is refused.
  - Each set is stored in the content-hash order of the task's seed (`assembly.training_order`) and, as
    `<model>_<k>`, in that of each order seed; an order seed that is the task's seed (3 at seed 3) gives the seed's own
    order and is trained all the same, and two order seeds that give one order are refused (`l5_draws.four_orders`).
  - HELD, each a refusal by the first step with nothing written: the sizes (N own rows and N published in `stock`, 2N
    published and no other in `published_stock`), at least one deep and one ordinary solve, no id twice, no problem
    with two own rows, no own row of a problem of the pretraining file, the published rows being the first N and 2N of
    the seed's order by id and place, no row with text, no held-aside or searched problem out of place
    (`l5.refuse_overlap`), no barred row (`l5.refuse_barred`: no held-out problem, published rows of the `pretrain`
    half alone), no problem of G or of the held-aside set (`l5_draws.refuse_problems_of`), and THAT SEED's ROWS:
    every row of L5d's `deep` and `shallow` at the seed is a row of `stock` and every row of its `published` a row of
    `published_stock`, by id (`l5_stock.holds_the_seeds_rows`; L5d's three sets are made again by L5d's own function,
    not read from L5d's run).
  - The ordinary solves as drawn are stored in `l5_stock_ordinary_solves.jsonl` (each with the map's k and
    `taken_by`); the counts are in the first step's summary (`the_stock`) and in the report.
- **The first check makes the rule again** (`the_rule_again`, `l5_stock.the_sets_are_the_rule`): the report step
  reads the problems the search solved from the search run's own per-problem rows and the ordinary solves from the
  map run's, and holds the stored sets to them (every one, no other own row), to the seed's published order, to
  carrying no text, and each ordered set to being its model's rows in its order seed's order. Then L5d's own checks
  for a seed, by L5d's functions, and the start model's stored first sampling being the setting's.
- **L5d's run of the seed** is found by L5d's own function (`ladder_l5_draws.own_run`:
  `ladder_l5_draws_pre_r64_seed<N>`) and read by the report step alone, from `ladder_l5_draws_report.done.json` and
  `report_ladder_l5_draws.json` (what that report says of its own seed, `of_this_seed`). It is NOT READ, with why,
  and nothing is refused, when the run is not on the box or has no report; when its report is of another seed or of
  another recipe than this run's; when it reads INCONCLUSIVE or is not to be read; and when it was measured otherwise
  than this run (the sampling, its attempts, its sampling seed, the number of goal problems): `l5_stock.of_l5d`. Then
  the dose and "the steps without its own proofs" read NOT READ, "its own proofs at this dose" is given without L5d's
  primary beside, and "what follows" names nothing.
- **The dose is decided on whole numbers** (`l5_stock.the_dose`: the two models' successes and attempts as fractions,
  never a rounded figure). At or above zero is a difference of zero or more; "below zero by more than 0.010" is a
  difference under −0.010, so a difference of exactly −0.010 reads NOT SHOWN AT ONE SEED. The standard deviation of
  two averaged models (0.0048) is printed in the dose's line and decides nothing by itself. The names are the
  section's, in capitals as written there.
- **The reads beside it, made exact.** The parts of G are the start model's own stored first sampling's (the goal
  problems it solves there; G′, held to the stored G′). "Solved" is at least one success in the 32 attempts; gained
  and lost are against the start model's stored rows, paired by problem. The share of distinct attempts on G is each
  measurement's own record of it, and the start model's as its own run recorded it. The long goal problems are those
  whose shortest published proof has 8 lines or more (`data/ladder_l0/heldout_proof_lines.jsonl`, read by the report
  and by nothing before it). Timeouts are `l5.timed_out` on the two sets.
- **A report that is not to be read** (Lean gave no verdict on more than 0.02 of a set) has `ok` false and fails its
  step, as L5d's does; the task queued again samples that set again from its adapter.
- **NOT BUILT.** The read over three seeds (each seed's report keeps what it needs under `of_this_seed`: its
  figures, its dose and `stock` minus `published_stock`). The check again in Lean of the timed-out attempts where
  the two sets' shares differ by more than a point: the report names the case and says no step does it, so the reads
  "both ways" are NOT given by this stage and are owed by hand, as for L5 and L5d. The measurement in full of the
  best averaged adapter (the section says so).
- **The smoke stage,** `ladder_l5_stock_smoke`: the fourteen steps at seed 3 on what the smoke runs left on the box
  (`ladder_l5_deep_smoke` and the three smoke runs that task read), in `ladder_l5_stock_smoke`. It ALONE sets
  `RLVR_LEAN_LADDER_L5_STOCK_SMOKE=1`, which the first step refuses unless the run directory is one the task names
  (`RLVR_LEAN_LADDER_L5_STOCK_RUN`) and the base map is the fixtures'; it sets neither L5's smoke-only variable nor
  L5d's. With it: L5's INCONCLUSIVE smoke run is read all the same and that is recorded; one deep solve is enough;
  where the fixtures' map gives the rule fewer ordinary solves than there are deep solves, as many as there are deep
  solves are taken, the others by the run's minimum and marked so (`taken_by`), and the first check holds the sets to
  the rule's and those; orders that are alike are trained; an averaged adapter that is one of its models is kept.
  The stage names L5d's run variable (`RLVR_LEAN_LADDER_L5_DRAWS_RUN=ladder_l5_draws_smoke`), so the dose is read
  against `ladder_l5_draws_smoke_mean_constant` when `ladder_l5_draws_smoke` has run under the recipe in force, and is
  NOT READ otherwise. IT DOES NOT EXERCISE: the real sizes and so the memory and the time of a training of 3,268 rows;
  a learning rate after the warm-up; the rule's own ordinary solves at their number; the refusals of an INCONCLUSIVE
  run of L5, of a further seed and of two orders alike; the reads on G′ and on the long goal problems. Nothing it
  measures is a result.
- **Fragile.** One failed check of one model makes the run INCONCLUSIVE. The dose is one seed's difference of two
  models whose sets differ in size as well as in their own proofs; "the steps without its own proofs" stands beside
  it for that and names nothing. A training here is four times L5d's rows, the longest pass a rank-64 adapter has
  made on this card since the rank check's own, and its memory is not measured before the run. A seed's report built
  before L5d's run of that seed is finished says NOT READ until its task is queued again.

**Outcome of L5s (seed 3, 2026-10-12 UTC; `docs/spec/ladder-l5s-RESULT.md`): THE DOSE IS NOT SHOWN AT ONE SEED,
and by the stage's own rule no further seed runs.** One task (`ladder_l5_stock_seed3_r1`, 04:33 to 07:11 UTC, from
`6d5c1c4fd`), all checks pass. Per 1,000 attempts on G's first sampling: `stock` (1,634 own solves and 1,634
published rows) 105.55, `published_stock` (3,268 published rows) 111.85; `stock` minus that seed's `deep` -4.62.
Beside it, naming nothing: `stock` minus `published_stock` -6.30 (at 800 rows the own-proof kinds led by +13.91);
`published_stock` minus that seed's `published` +20.97; goal problems solved in 32 attempts 225 and 224 (the start
model 200); the timeout shares under a point apart. Every route from `pre_r64` measured so far lands between 105 and
112 per 1,000; the own deep solves reached that band with a quarter of the rows and nothing went past it. L5f asks
whether they go past it from the twice-pretrained model.

### L5f: the three kinds again, from the twice-pretrained model. Do the model's own proofs still add after a whole second pass over published ones? (fixed 2026-10-12, after L5d's seeds 3 and 4 and before any run of this stage)

**Under the owner's words.** 2026-10-12: "make as much progress as you can tonight on improving this solver's solve
rate against the heldout tasks." And the bet this project tests (2026-10-09): "I just want to make sure that this is
actually creating a feedback cycle that is able to push the model beyond capabilities that we're able to get through
pre-training alone". Labelled **pretrained on published proofs** (it starts from `second_full`).

**Why.** Two routes from `pre_r64` now read the same on G's first sampling (32 attempts a problem, per 1,000
attempts; `pre_r64` itself 77.7):
- MORE PRETRAINING: `second_full`, a whole second pass over the 24,866 published rows (L4p; one training): 110.25,
  224 goal problems solved.
- THE MODEL'S OWN PROOFS: `deep`, 800 rows of which 400 are deep solves and 400 published, the average of four
  orders (L5d): 110.17 at seed 3 and 111.53 at seed 4 (226 and 219 solved); and at both seeds the two kinds with own
  proofs are above the same number of published rows alone (+13.91 and +13.07).

Nothing has trained the model's own proofs FROM the twice-pretrained model. If the two gains are one gain (any more
training of this model gives it, and it saturates), the three kinds from `second_full` read alike and near
`second_full`. If the own proofs carry something a second pass over published proofs does not, the kinds with own
proofs read above `published` from `second_full` too, and above `second_full`: the best model stored, and the bet's
own question at the strongest pretraining we have.

**What runs: one task, L5d's stage with ONE change, the start adapter.** The seed is 3. The three sets are L5d's for
that seed (L5's rule, made by L5's own functions) and are HELD to be, id for id and order for order, the sets L5d's
finished run of seed 3 trained from `pre_r64`: the two runs then differ by the start alone. Each set is trained from
`second_full`'s stored adapter (L4p's run of the pretraining's seed; only read; rank 64 and alpha 128, as that run
recorded them) in four orders (order seeds 1 to 4) with the recipe as it is, the model measured is the adapter that
is the mean of the four, and each of the three is measured on G's first sampling alone, 32 attempts a problem with
the sampling seed `pre_r64`'s run recorded (1001), which is the one `second_full` was measured with. It writes a run
directory of its own, with the start's name in it. L5d from `pre_r64` writes what it wrote.
The deep and the ordinary solves are `pre_r64`'s own (the once-pretrained model found them), not `second_full`'s; the
published rows of every set are rows `second_full` has trained on twice. No row carries text; a published row by id
and place.

**The reads, fixed before the run.** One seed: a scout. A model's figure is its successes per attempt on all of G
over the 32 attempts. An averaged model's figure has a standard deviation of about 0.0034 (the order's part 0.0025,
the sampling noise 0.0023); the primary's difference of one seed (a mean of two models minus a third) about 0.0042.
- **Checks, read first; any FAIL: INCONCLUSIVE, and nothing else is said.** L5d's own for a seed, with the start
  adapter `second_full`'s; the sets are the ones L5d's run of the seed trained; the stored first sampling of
  `second_full` is the setting's (392 problems, 32 attempts, sampling seed 1001); `second_full`'s adapter is there, is
  at its rank and is not `pre_r64`'s. L4p's own check of `second_full`'s loss (which rose over its pass, and which
  L4p's note found to be the wrong check for a second pass) is printed and decides nothing, as in L6.
- **The stack (the scout's read).** The mean of `shallow` and `deep`, minus `published`, all three from
  `second_full`: L5d's primary. At or above zero: THE MODEL'S OWN PROOFS DO NOT FALL BEHIND PUBLISHED ONES AFTER A
  SECOND PASS: ESCALATE, seeds 4 and 5 of this stage run, each after L5d's own run of that seed (three seeds
  conclude, with L5d's primary rule and names). Below zero by more than 0.0085 (twice the standard deviation above):
  THEY COST AFTER A SECOND PASS AT THIS SEED; no further seed. Between: NOT SHOWN AT ONE SEED; no further seed is
  queued by this read.
- **The level, beside it and naming nothing at one seed.** Each of the three minus `second_full`'s stored first
  sampling, paired by problem (does ANY further training add to the twice-pretrained model: 800 rows it has seen
  twice, for `published`); and each of the three minus the same kind of L5d's seed 3 (the same rows from `pre_r64`:
  what the second pass adds under them). The highest figure of the three is printed beside every stored model's
  first-sampling figure it is read against (`pre_r64` 77.7, `second_full` 110.25, L5d's three at this seed).
- **The solve rate against the two start models.** Each of the three against `second_full`'s stored first sampling
  and against `pre_r64`'s: per attempt on all of G, on the 200 goal problems `pre_r64` solves in its first sampling
  and on G′ (the 192 it does not); the goal problems solved in the 32 attempts, gained and lost.
- **Overfitting, for each model against `second_full`** (the owner: "without overfitting"): the goal problems lost
  beside the ones gained; the share of distinct attempts on G; the 74 long goal problems (8 lines or more) solved,
  gained and lost.
- **Timeouts:** each set's share; where two differ by more than a point, the timed-out attempts are checked again
  (Lean only, no measurement running) and the reads are given both ways.

**What follows, fixed now.** ESCALATE: seeds 4 and 5 of this stage, each after L5d's own run of that seed; over the
three, the primary is read with L5d's rule. The averaged adapter that reads highest on G's first sampling, of this
stage and of L5s, is then the one measured in full (the rungs, the second sampling, the held-aside set), a step that
is not built here. COSTS, or NOT SHOWN: no further seed; the own proofs' gain at this dose is not shown beyond a
second pass over published proofs, and that goes to the owner beside L5s's read and the full-size cycle's decision.

**What it cannot say.** One seed. `second_full` is ONE training in one order: every model here inherits that
order's luck, so the differences between the three kinds do not carry it, and "what the second pass adds under them"
does (a single training's order is worth about 0.005). The own proofs are of base-map problems and were found by
`pre_r64`, not by the model they are trained into here: a round of search BY the twice-pretrained model is another
experiment (L6's shape). Nothing here is on the held-aside set or the rungs. A model's figure on 32 attempts.

**Budget.** Twelve trainings of 800 rows (about four minutes each) and three measurements of about 25 minutes: about
an hour and three quarters on the one GPU, as a seed of L5d. A smoke stage on the fixtures first (it reads the smoke
run of L4p for its `second_full`).

**Made exact by the build (2026-10-12; nothing queued, nothing run on a GPU, nothing sent to Lean; the reads, the
checks and "What follows" above are not changed; the readings taken where the section left something open, what was
NOT built, what the smoke stage does not exercise and what is fragile are below).**
- **The stages and their names.** `ladder_l5_full` and `ladder_l5_full_smoke` (`gpu/ladder_l5_full.py`; rules
  `domain/ladder_round/l5_full.py`; report `reporting/ladder_l5_full.py`), settings `ladder_loop.l5_full` (`start:
  second_full`, `seeds: [3]`, `more_seeds: [4, 5]`, `escalated: false`; one reader, `the_settings`). THE RUN DIRECTORY
  is `ladder_l5_draws_second_full_seed<N>`: L5d's own name with the start's name in the start's place, as L4t names a
  run from another start. The stage, its steps (`ladder_l5_full_*`), its files (`l5_full_*`,
  `report_ladder_l5_full.json`) and its jobs (`ladder_l5_full_seed3.json`, est_minutes 130;
  `ladder_l5_full_smoke.json`, 30; the `resources` of L5d's jobs key for key) are NOT named `ladder_l5_draws_...`:
  L5d's own tests hold the jobs `ladder_l5_draws*.json` and the steps of that name to be L5d's alone, and L5d
  from `pre_r64` is to write what it wrote. No job is written for seeds 4 and 5.
- **The seeds.** As L5s's, by L5s's own two functions (`l5_stock.the_seed`, `fixed_seeds`): a task's seed is the
  scout's or, ONCE `escalated` IS TRUE, one of `more_seeds`; a task of a further seed is refused by the first step
  until then, and of any other seed always. The settings reader refuses a start that is not `second_full` (the one
  start this section specifies), seeds that are not distinct whole numbers or are not seeds of L5d's, an `escalated`
  that is not true or false, and a recipe of L5d's that is not `mean_constant`. Nothing in the code queues a seed or
  sets `escalated`.
- **TWENTY steps,** in L5d's order: `ladder_l5_full_prepare`; then for `published`, for `shallow` and for `deep`:
  `ladder_l5_full_train_<model>_1` to `_4`, `ladder_l5_full_average_<model>`, `ladder_l5_full_measure_<model>`;
  `ladder_l5_full_report`. Each is resumable and a finished one returns what it stored; the report step ALONE is
  built again each time its task is queued. No step deletes an adapter, and no step writes into a run it reads.
- **THE ONE CHANGE, and how it is made.** L4t's training body takes the start model it is HANDED
  (`ladder_l4_rows.train_a_model(start_from=...)`, off by default: every other caller hands it none and it finds the
  start itself, as before; asserted). This stage hands it `second_full`: the start `pre_r64` resolves to
  (`ladder_l4_start.the_start`, with its rank, its alpha and its check) under the name `second_full`, with the
  directory and adapter of L4p's run. So every training loads `second_full`'s adapter into a fresh adapter at rank 64
  and alpha 128, every adapter saved is read back and held to that rank, the count of numbers that are not the start
  adapter's is against `second_full`'s file, the averaged adapters are L5n's own function handed the same start, and
  the three measurements are served at rank 64 (`config_from(serving=True)`). The recipe, the learning rates recorded
  and held, the adapter seed (the task's) and the order seeds are L5d's `mean_constant`'s.
- **Where `second_full` lives is stated once.** `ladder_l6.l4p_directory` (L4p's run of the pretraining's seed; the
  variable L6 names for it, which a smoke run sets) and L4p's own names for its markers and files; this module holds
  neither a run's name nor an adapter's path. It is read by L6's own reading of the stored model
  (`ladder_l6.read_the_stored_model`: L4p's markers of its prepare step and of `second_full`'s training and
  measurement, and `second_full`'s rows on G held to the problems, the attempts and the sampling seeds the start model
  was measured with). L6 says NOT THERE and goes on; HERE THAT IS REFUSED, naming L4p's task, because every model is
  trained from that adapter.
- **What the first step holds of `second_full`, each a refusal with nothing written.** L4p's run recorded rank 64 and
  alpha 128 for its models, and saved `second_full` read back at them. L4p's own bit-for-bit comparison at the end of
  that training recorded numbers that are not `pre_r64`'s. That rank and alpha are the ones L5's run trained its
  models at. And ON A GPU BOX (the test the trainings themselves make of where they run), the files themselves, as
  every later step reads them: the adapter is there; read back, it is at rank 64 and alpha 128 with every matrix of
  rank 64; compared with `pre_r64`'s file number for number, some are not `pre_r64`'s; and its tensors are of a type
  the averaged adapters are written in. Its stored first sampling (392 problems, 32 attempts, sampling seed 1001) is
  copied into the run (`l4_stored_second_full_<sampling>.jsonl`) beside `pre_r64`'s. The report's two checks of it
  read what the first step stored.
- **L4p's own checks of `second_full` are printed and decide nothing.** The first step reads what L4p's report says
  of the model (`can_this_model_be_read`), stores the names of the checks that failed and the two mean losses of its
  pass, and prints them; the report prints them in a line of its own after the checks. Nothing is refused for them and
  no check of this stage reads them. With L4p's report not on the box the line says so.
- **L5d's run of the seed** is found by L5d's own function (`ladder_l5_draws.own_run`, by L5s's `read_l5d`) and read
  by L5s's own reading of what its report says of its seed (`l5_stock.of_l5d`). THE FIRST STEP REFUSES, naming the
  task (`ladder_l5_draws_seed<N>.json`), a run that is not there or has no report, of another seed or another recipe
  than `mean_constant`, INCONCLUSIVE, not to be read, or measured otherwise than this run (the sampling, its
  attempts, its sampling seed, the number of goal problems); and one that does not hold its fifteen stored sets
  (`l5_draws_training_<name>.jsonl`: the three, and each in its four orders) and its twelve trainings' own records
  (`l5_draws_loss_<name>.json`).
- **The sets are held to L5d's, twice.** The first step makes the three sets by L5d's own first-step reading
  (`ladder_l5_draws.drawn_by_the_seed`, under the words an `OfASeed` gives: L5's finished run, a further seed's
  refusals, L5's rule by the task's seed) and each in its four orders (`l5_draws.four_orders`), and holds them
  (`l5_full.hold_to_theirs`): each of the fifteen is, id for id and place for place, the set L5d's run stored under
  that name, and each of the twelve ordered sets is what that run's training loop RECORDED it trained on. The refusal
  names the first set and the first place that differ. The report makes the same comparison again from this run's
  stored sets and L5d's run as it is then (`the_sets_are_theirs`), and that is the check "the sets are the ones L5d's
  run of the seed trained". A set that is L5d's rule but was not what L5d's run trained fails it.
- **The checks, in the report's order.** Each set is L5's rule for its seed, made again and held (L5d's own); the
  sets are the ones L5d's run trained; `second_full`'s adapter is there, at its rank, and is not `pre_r64`'s;
  `second_full`'s stored first sampling is the setting's; then L5d's own checks of the models
  (`reporting/ladder_l5_draws.checks_of_the_models`, `lines_of_the_checks`), each adapter counted against
  `second_full`'s. Any FAIL: the headline is INCONCLUSIVE, the failed checks are named, and no read is made.
- **The stack is decided on whole numbers** (`l5_full.the_stack`: each model's successes and attempts as fractions,
  never a rounded figure). At or above zero is a difference of zero or more; "below zero by more than 0.0085" is a
  difference under −0.0085, so a difference of exactly −0.0085 reads NOT SHOWN AT ONE SEED. The two standard
  deviations the section gives (0.0034, 0.0042) are printed in its line and decide nothing by themselves. The names
  are the section's, in capitals as written there. A model with no attempt on G makes it NOT READ.
- **The level, made exact** (`l5_full.the_level`). Each of the three minus `second_full`'s stored first sampling is
  per attempt on all of G (the goal problems solved, gained and lost against it, paired by problem, are in the read
  against the two start models). Each minus the same kind of L5d's run of the seed is read from what L5d's REPORT says of its seed when the report
  step runs: with that run gone, of another recipe, INCONCLUSIVE or not to be read by then, this difference is NOT
  READ, with why, and nothing else changes. The highest figure of the three is printed with its model's name (the
  first of `published`, `shallow`, `deep` on a tie) beside `pre_r64`'s, `second_full`'s and L5d's three.
- **Against the two start models, made exact.** L5s's own function (`l5_stock.against_the_start`), once for each
  start. The parts of G are `pre_r64`'s in BOTH: the goal problems `pre_r64` solves in its own stored first sampling
  and G′ (held to the stored G′), as the section names them. "Solved" is at least one success in the 32 attempts;
  gained and lost are against that start model's stored rows.
- **Overfitting and timeouts, made exact.** Against `second_full`: lost beside gained, the share of distinct attempts
  on G (each measurement's own record; `second_full`'s as L4p's run recorded it, over all its attempts on G, and said
  so), and the long goal problems (shortest published proof of 8 lines or more,
  `data/ladder_l0/heldout_proof_lines.jsonl`, read by the report and by nothing before it). Timeouts are
  `l5.timed_out` on the three sets, with `second_full`'s own share beside; a pair more than a point apart is NAMED,
  and the report says no step checks it again.
- **A report that is not to be read** (Lean gave no verdict on more than 0.02 of a set) has `ok` false and fails its
  step, as L5d's does; the task queued again samples that set again from its adapter.
- **Existing functions are called, not copied.** L5's run and the three sets: L5d's first-step reading. `second_full`:
  L6's. L5d's run: L5s's. A training: L4t's body. An average and its check: L5n's. A measurement: the ceiling's, with
  no rung and the one sampling. The checks of the models and their lines: L5d's report's. L5, a further seed of it,
  L5n, L5d, L5s and L6 write what they wrote: their own stage tests pass unchanged (the five edits to earlier tests
  are lists of the stages that name a variable, each with this stage's smoke added).
- **NOT BUILT.** The read over three seeds (each seed's report keeps what it needs under `of_this_seed`). The check
  again in Lean of the timed-out attempts where two sets' shares differ by more than a point: the report names the
  case, so the reads "both ways" are NOT given by this stage and are owed by hand, as for L5, L5d and L5s. The
  measurement in full of the best averaged adapter (the section says so). Job files for seeds 4 and 5.
- **The smoke stage,** `ladder_l5_full_smoke`: the twenty steps at seed 3 on what the smoke runs left on the box, in
  `ladder_l5_full_smoke`. It ALONE sets `RLVR_LEAN_LADDER_L5_FULL_SMOKE=1`, which the first step refuses unless the
  run directory is one the task names (`RLVR_LEAN_LADDER_L5_FULL_RUN`) and the base map is the fixtures'; it sets
  neither L5's smoke-only variable nor L5d's. It names L6's variable for L4p's run
  (`RLVR_LEAN_LADDER_L6_L4P_RUN=ladder_l4_dose_r64_smoke`) and L5d's run variable
  (`RLVR_LEAN_LADDER_L5_DRAWS_RUN=ladder_l5_draws_smoke`). NEEDS ON THE BOX, all only read: `ladder_l4_dose_r64_smoke`
  run to its end (its `adapters/second_full`, its markers and rows); `ladder_l5_draws_smoke_mean_constant` run to its
  end with a report that is not INCONCLUSIVE (its stored sets are what this run's are held to; the first step refuses
  without it); `ladder_l5_deep_smoke` and the three smoke runs that task read (`ladder_l4b_map_smoke`,
  `ladder_l4_rows2_r64_smoke`, `ladder_l4_rank_smoke`). With the variable: L5's INCONCLUSIVE smoke run is read all the
  same and that is recorded; one deep solve is enough; orders that are alike are trained; an averaged adapter that is
  one of its models is kept. IT EXERCISES FOR THE FIRST TIME ON THE BOX a training whose start adapter is L4p's
  `second_full`, an averaged adapter of such trainings served, and the hold of the sets to L5d's stored ones. IT DOES
  NOT EXERCISE: the real sizes and so the memory and the time of a real training; a learning rate after the warm-up;
  the refusals of an INCONCLUSIVE run of L5 or of L5d, of a further seed and of two orders alike; the reads on G′ and
  on the long goal problems (G is one problem). Nothing it measures is a result, and its line THE STACK is no read.
- **Fragile.** One failed check of one model makes the run INCONCLUSIVE. The first step refuses a real run for one
  missing file of three stored runs: L4p's markers and `second_full`'s stored rows and attempts on G (what L6's
  reading asks for), and L5d's fifteen stored sets and twelve loss records of the seed. `second_full` is ONE training:
  what "the level" reads against L5d's kinds carries that training's order (about 0.005). The memory of a training
  from `second_full` is that of a training from `pre_r64` at the same rank and rows, and is not measured before the
  run. A seed's report built after L5d's run of the seed is moved or run again reads the level's second difference
  NOT READ, or fails the check of the sets.

**Outcome of L5f (seed 3, 2026-10-12 UTC): THEY COST AFTER A SECOND PASS AT THIS SEED, and by the stage's own rule
no further seed runs: the outcome goes to the owner.** One task (`ladder_l5_full_seed3_r1`, 07:20 to 09:04 UTC, from
`2220ba9a7`), all checks pass. Per 1,000 attempts on G's first sampling, each trained from `second_full` (110.25):
`published` 144.77, `shallow` 126.35, `deep` 133.45; the stack (the mean of `shallow` and `deep`, minus `published`)
-14.87, past the -8.5 line. Each is above `second_full` (+34.52, +16.10, +23.20) and above the same kind trained
from `pre_r64` at this seed (+53.89, +26.95, +23.28); goal problems solved in 32 attempts 235, 245, 236 (`second_full`
224). The timeout shares are more than a point apart (0.0256, 0.0335, 0.0473): the second check of the timed-out
attempts is owed and the reads are given both ways in the result note. `published` here is the highest figure any
model of this project has read on G's first sampling; L5c asks what made it.

### L5c: the continuation. What lifted the published kind by 34 per 1,000 from the twice-pretrained model, and does more of it go higher? (fixed 2026-10-12, after L5f's seed 3 and before any run of this stage)

Under the owner's words of 2026-10-12: "make as much progress as you can tonight on improving this solver's solve
rate against the heldout tasks". Labelled **pretrained on published proofs**. No row of this stage is one of the
model's own proofs: it is a stage about the published rows and the recipe.

**Why.** L5f's `published` (from `second_full`: the first 800 published rows of seed 3's order, trained in four
orders, the averaged adapter measured) reads 144.77 per 1,000 on G's first sampling against `second_full`'s 110.25.
Every route from `pre_r64` had landed between 105 and 112. Two things changed at once between `second_full` and that
model: 800 more rows of training, and the AVERAGE of four trainings. `second_full` is one training at a constant
learning rate whose loss rose over its pass (0.127 over its first rows, 0.138 over its last); from `pre_r64`, L5n
read the average of four at +4 per 1,000 over the mean of its four. Which of the two made the +34.5 is not known,
and neither is whether more of either goes higher. Each answer is a recipe: a wider average, a second averaged step,
or more rows.

**What runs.** One task at seed 3. Everything is trained from an adapter that is only read, at rank 64, with the
recipe as it is (constant learning rate, one pass), on rows of seed 3's order of the pretraining file (the order
L5d's `published` and L5s's `published_stock` are the first rows of). Four models, each measured on G's first
sampling alone (32 attempts at each of the 392 goal problems, sampling seed 1001), in this order:

- **`single`**: L5f's stored `published_1` (the first of the four trainings its `published` is the average of), ONLY
  READ and measured. No training.
- **`wide`**: the average of EIGHT adapters: L5f's stored `published_1` to `published_4` and four new trainings of the
  same 800 rows from `second_full` in the orders of order seeds 5 to 8.
- **`again`**: from L5f's stored averaged `published` (not from `second_full`), the NEXT 800 rows of the order (rows
  801 to 1,600) trained in four orders (order seeds 1 to 4), averaged. A second averaged step.
- **`more`**: from `second_full`, the first 3,268 rows of the order (L5s's `published_stock` set, held to it id for
  id) trained in four orders, averaged. Four times the rows in one step.

**The reads, fixed before the run.** On whole numbers of successes over the 12,544 episodes, per 1,000 attempts.
The bar is L5f's stored `published` (144.77), paired by problem. One seed: each read is a scout's.

- **Checks, read first; any FAIL makes the run INCONCLUSIVE:** L5f's run of the seed is there, finished, not
  INCONCLUSIVE, with its four `published` adapters and its averaged one at rank 64, each not the start adapter's;
  each set is its rule made again (rows of the seed's order, no row with text, no held-out problem, no problem of G
  or of the held-aside set); `wide`'s first 800 rows and `more`'s 3,268 are, id for id, L5f's `published` set and
  L5s's `published_stock` set; each averaged adapter is the mean of its members and is the one served; Lean answered.
- **WHAT MADE IT, `single` minus `published`:** below zero by more than 13 per 1,000: THE AVERAGE IS PART OF THE GAIN.
  Otherwise ONE CONTINUATION GIVES IT. Beside it, `single` minus `second_full`'s stored first sampling: what 800 rows
  of one training add.
- **A WIDER AVERAGE, `wide` minus `published`;** **A SECOND STEP, `again` minus `published`;** **MORE ROWS, `more`
  minus `published`.** Each: at or above +8.5 per 1,000 it ADDS: ESCALATE (two further seeds of that model and of
  the bar); below zero by more than 8.5 it COSTS; between NOT SHOWN AT ONE SEED.
- **Beside, naming nothing:** each model against `second_full` and against `pre_r64` (all of G, the 200 goal
  problems `pre_r64` solves in its first sampling, G', goal problems solved, the 74 long ones); goal problems lost
  beside gained against `second_full`; the share of distinct attempts.
- **Timeouts:** each set's share; two more than a point apart are checked again and read both ways (no step does it).

13 is twice the standard deviation of a difference between one training and an average of four (order spread 5.0
for one, about 2.5 for four, binomial 2.5 each); 8.5 is L5f's own line for two averaged models.

**What follows.** The highest model of this stage and of L5f is the candidate for a measurement in full (both
samplings of G, the held-aside set, the rungs), which is not built. A model that reads ADDS gets its two further
seeds before anything is said of it. If nothing adds, L5f's `published` recipe stands as the best known and the
question of what made it is answered by `single` alone.

**What it cannot say.** One seed and one draw of rows; `second_full` is one training. Nothing here is about the
model's own proofs. The published rows are ones `second_full` has trained on twice: a gain on G from a third
reading of them is a gain of the recipe, and whether it holds on the held-aside set is not measured here.

**Budget.** One measurement with no training; four trainings of 800 rows and a measurement, twice; four trainings
of 3,268 rows and a measurement: about 2 h 40 at 16 minutes a measurement. Its smoke run first.

**Made exact by the build (2026-10-12; nothing queued, nothing run on a GPU, nothing sent to Lean; the reads, the
checks and "What follows" above are not changed; the readings taken where the section left something open, what was
NOT built, what the smoke stage does not exercise and what is fragile are below).**
- **The stages and their names.** `ladder_l5_cont` and `ladder_l5_cont_smoke` (`gpu/ladder_l5_cont.py`; rules
  `domain/ladder_round/l5_cont.py`; report `reporting/ladder_l5_cont.py`), settings `ladder_loop.l5_cont` (`start:
  second_full`, `seeds: [3]`, `more_seeds: [4, 5]`, `escalated: false`; one reader, `the_settings`). THE RUN DIRECTORY
  is its own, `ladder_l5_cont_second_full_seed<N>`; its files are `l5_cont_*` and `report_ladder_l5_cont.json`; its
  jobs `ladder_l5_cont_seed3.json` (est_minutes 190) and `ladder_l5_cont_smoke.json` (30), with the `resources`
  of L5f's jobs key for key. No job is written for seeds 4 and 5. Nothing of L5f's, L5d's, L5s's, L4p's or L5's run
  is written: each is read, and left file for file as it was (asserted in the scripted world).
- **The seeds.** As L5f's, by L5s's own two functions (`l5_stock.the_seed`, `fixed_seeds`): a task's seed is the
  scout's or, ONCE `escalated` IS TRUE, one of `more_seeds`. The settings reader refuses a start that is not
  `second_full` (and not L5f's own), seeds that are not distinct whole numbers or are not seeds of L5f's, and an
  `escalated` that is not true or false. Nothing in the code queues a seed or sets `escalated`.
- **TWENTY-ONE steps,** in the section's order: `ladder_l5_cont_prepare`; `ladder_l5_cont_measure_single`;
  `ladder_l5_cont_train_wide_5` to `_8`, `ladder_l5_cont_average_wide`, `ladder_l5_cont_measure_wide`;
  `ladder_l5_cont_train_again_1` to `_4`, `ladder_l5_cont_average_again`, `ladder_l5_cont_measure_again`;
  `ladder_l5_cont_train_more_1` to `_4`, `ladder_l5_cont_average_more`, `ladder_l5_cont_measure_more`;
  `ladder_l5_cont_report`. Each is resumable and a finished one returns what it stored; the report step ALONE is
  built again each time its task is queued. No step deletes an adapter, and no step writes into a run it reads.
- **What is called, not copied.** L5's finished search and every stored run its task read: L5d's own first-step
  reading (`ladder_l5_draws.drawn_by_the_seed`, under the words an `OfASeed` gives). `second_full`: L5f's own reading
  and start (`ladder_l5_full.read_second_full`, `the_start`), so everything L5f's first step refuses about it is
  refused here in the same words. L5f's run of the seed: found by `ladder_l5_full.own_run` (this module holds no run
  name of another stage and no adapter path). The trainings: L4t's body with `start_from`, as L5f hands it. The
  averaged adapters: L5n's `the_mean_adapter`. The measurements: `measure_model` with `rungs=False`. The checks of
  the models and their lines: L5d's (`reporting/ladder_l5_draws.checks_of_the_models`, `lines_of_the_checks`). The
  reads against a stored model: `l5_stock.against_the_start`. THREE FUNCTIONS WERE CUT OR WIDENED FOR IT, each off by
  default so that its first owner writes what it wrote (their tests pass unchanged, but for the lists of stages that
  name a variable): `ladder_l5_stock.stock_by_the_seed` (L5s's two sets and what its rule counted, cut out of L5s's
  first step, which now calls it); `ladder_l5_noise.the_mean_adapter(kept=...)` (adapters ANOTHER run keeps, averaged
  before the run's own; handed none, it averages its own run's alone); and `lines_of_the_checks(mean_of=...)` (the
  words for what an averaged adapter is the mean of: "its four's" unless told otherwise).
- **The four models, made exact.** `single` has no adapter and no training set of its own: its measurement is served
  from L5f's stored `published_1` where that run keeps it, and its step records the adapter it was served from.
  `wide` is ONE averaged adapter of EIGHT: L5f's stored `published_1` to `published_4`, only read and in that order,
  then this run's `wide_5` to `wide_8`, each from `second_full`. `again`'s four trainings load L5f's AVERAGED
  `published` (`the_bar`: `pre_r64`'s start, with its rank, alpha and check, under the bar's name and with L5f's run's
  directory and adapter), so the count of numbers that are not the start adapter's is against THAT file for `again`,
  and against `second_full`'s for `wide` and `more`. Every adapter saved is read back at rank 64 and alpha 128; every
  measurement is served at rank 64. The recipe is L5d's `mean_constant` (constant learning rate, one pass, each step's
  rate recorded and held); the adapter seed is the task's.
- **How "the first rows", "the next rows" and "held id for id" are made** (`l5_cont.the_sets`, `hold_the_sets`,
  `hold_the_members`). The seed's order is `l4_rows.rehearsal_rows` of the pretraining file for the task's seed, the
  draw L5d's `published` and L5s's `published_stock` are the first rows of. `wide` is its first F rows, F the number
  of rows L5f's run STORED as `published` (800 at the real sizes: read from that run, never a number of the code);
  `again` rows F + 1 to 2F; `more` its first S rows, S the size of L5s's `published_stock` BY L5s's RULE FOR THE SEED,
  made here by calling the rule on the stored search and map (twice the own solves in stock; 3,268 expected,
  RECORDED). L5s's stored run is NOT read, so a run of L5s that is gone or of another seed changes nothing. Each set
  is in `assembly.training_order` of the task's seed, and each trained set in that of its order seed. HELD, each a
  refusal: `wide` is, id for id and place for place, L5f's stored `published` set; `more` is, id for id and place for
  place, the rule's `published_stock` set, and holds every row of `wide`; `again` has as many rows as `wide` and none
  of them; every row is a published row by id and place, without text; L5f's four stored ordered sets are `wide`'s
  rows in the order of their order seeds, and its four training loops recorded those rows, each once (so `wide` is
  the mean of eight trainings of ONE set); no row of a held-out problem, none that is not of the `pretrain` half,
  none of a problem of G or of the held-aside set; and no two of `wide`'s EIGHT order seeds (1 to 8), nor of another
  model's four, give one order.
- **L5f's run of the seed, and what the first step holds of it** (`read_l5fs_run`; each a refusal with nothing
  written, naming the task that writes it). What its report says of its seed (`l5_cont.of_l5f`): there,
  of this seed, from `second_full`, of the recipe `mean_constant`, not INCONCLUSIVE, to be read, and measured as this
  run measures (the sampling, its attempts, its sampling seed, the number of goal problems). Its files: the markers
  of its first step, of its four trainings of `published`, of that model's averaging and of its measurement; its
  stored `published` set and that set in its four orders; the four trainings' loss records; its `published` rows on
  the first sampling. Its adapters, AS ITS OWN STEPS RECORDED THEM (`l5_cont.theirs_are_held`): each of the four saved
  and read back at rank 64 and alpha 128 with numbers that are not `second_full`'s, and the averaged one read back as
  the mean of the four, none of them (a smoke run's recorded exception apart), at that rank. ON A GPU BOX, also the
  five FILES as every later step reads them: there; read back at rank 64 and alpha 128; not `second_full`'s number
  for number; of a type the mean is written in. The averaged adapter is NOT averaged again from the four files by
  the first step: that it is their mean stands on L5f's own read-back.
- **The checks, in the report's order.** Of the run: each set is its rule made again (the three sets and the twelve
  ordered ones from the pretraining rows and the seed; no row with text; `wide` is L5f's stored set AS THAT RUN HOLDS
  IT WHEN THE REPORT IS BUILT; L5f's four trainings are orders of it and recorded its rows; `more`'s size is L5s's
  rule COUNTED AGAIN from the stored search and map: `the_sets_are_the_rule`); L5f's run of the seed carries the
  stage (the first step's hold of its adapters, and its report READ AGAIN as it is then: a run that has since gone,
  or reads INCONCLUSIVE, fails it); `second_full`'s adapter is there, at its rank, and is not `pre_r64`'s (L5f's
  check); `single`'s set was sampled from L5f's stored `published_1`; `second_full`'s stored first sampling is the
  setting's; the bar's stored first sampling is the setting's (392 problems, 32 attempts, the recorded sampling
  seed). Then L5d's own checks: of each of the twelve trainings (its adapter is not its start adapter's and is at its
  rank; it was trained on its set and nothing else, on the loop's own record; no barred row; no problem of G or of
  the held-aside set), of each averaged model (its numbers are the mean of its members', it is none of them, at its
  rank, and the one served; Lean answered), and of `single` (Lean answered). Any FAIL: the headline is INCONCLUSIVE,
  the failed checks are named, and no read is made.
- **The reads are decided on whole numbers** (`l5_cont.what_made_it`, `the_read`: successes and attempts as
  fractions, never a rounded figure). "Below by more than 13 per 1,000" is a difference under −0.013: exactly −0.013
  reads ONE CONTINUATION GIVES IT. "At or above +8.5" is a difference of 0.0085 or more; "below zero by more than
  8.5" is under −0.0085, so exactly −0.0085 reads NOT SHOWN AT ONE SEED. The bar is the copy this run's first step
  made of L5f's stored `published` rows (`l4_stored_l5f_published_<sampling>.jsonl`), paired by problem. A model
  with no attempt on G makes its read NOT READ.
- **Beside, made exact.** Each of the four against the bar, against `second_full` and against `pre_r64`, by L5s's
  own function, once for each: per attempt on all of G, on the goal problems `pre_r64` solves in its own stored
  first sampling and on G′ (the parts are `pre_r64`'s in all three), the goal problems solved, gained and lost, and
  the long ones (8 lines or more). Overfitting is the lost beside the gained against `second_full` and the share of
  distinct attempts. TIMEOUTS: each of the four sets' share and the two stored sets' beside; a pair more than a point
  apart is NAMED and no step checks it again.
- **What follows, made exact** (`l5_cont.what_follows`). With a model that reads ADDS: `ESCALATE: TWO FURTHER SEEDS
  OF EACH MODEL THAT ADDS AND OF THE BAR`, naming the models and the seeds. With none: `NOTHING ADDS: L5f's
  `published` RECIPE STANDS AS THE BEST KNOWN`. Either way the CANDIDATE for a measurement in full is named: the
  highest per attempt on all of G of this stage's four models and the bar (the bar on a tie with it). The
  measurement in full is not built.
- **After an escalation.** Set `ladder_loop.l5_cont.escalated: true`. A task of seed 4 or 5 is then refused by its
  first step until L5F's OWN RUN of that seed is on the box and finished, which needs `ladder_loop.l5_full.escalated`
  too and, before it, L5d's own run of that seed: no run holds the bar at seeds 4 and 5 when this is written. The
  read over three seeds is not built.
- **NOT BUILT.** The further seeds' bar and the read over three seeds; the check again in Lean of the timed-out
  attempts where two shares are more than a point apart; the measurement in full of the candidate (both samplings of
  G, the held-aside set, the rungs); job files for seeds 4 and 5.
- **The smoke stage** (`ladder_l5_cont_smoke`; `--seeds 3`). It ALONE sets
  `RLVR_LEAN_LADDER_L5_CONT_SMOKE=1`, refused unless the run directory is one the task names
  (`RLVR_LEAN_LADDER_L5_CONT_RUN=ladder_l5_cont_smoke`) and the base map is the fixtures'. It names L5's run and the
  three that task read, L6's variable for L4p's run (`ladder_l4_dose_r64_smoke`) and L5f's run variable
  (`ladder_l5_full_smoke`); it sets no other stage's smoke-only variable and not L5d's run variable. ON THE BOX IT
  NEEDS: `ladder_l5_full_smoke` run to its end with a report that is not INCONCLUSIVE; `ladder_l4_dose_r64_smoke`
  run to its end; `ladder_l5_deep_smoke` with `ladder_l4b_map_smoke`, `ladder_l4_rows2_r64_smoke` and
  `ladder_l4_rank_smoke`. L5d's smoke run is not read. It runs all twenty-one steps: L5f's smoke `published` is two
  rows, so `wide` is those two, `again` the next two and `more` four (one deep solve and one ordinary solve in
  stock). WHAT IT DOES NOT EXERCISE: the real sizes; a learning rate past the warm-up (each of its trainings is one
  optimizer step); eight orders that differ (two rows have two orders: pairs alike are trained all the same and
  RECORDED, and an averaged adapter that is one of its members is kept and SAID); a read that means anything (the
  fixture has one goal problem). WHAT IT IS THE FIRST RUN ON THE BOX OF: a mean of eight adapters of two runs; a
  training FROM an averaged adapter; a measurement served from an adapter another run keeps.
- **What could not be tested off the box.** The reads of the REAL adapters by the first step on a GPU box (L5f's
  five read back at rank 64, of a type the mean is written in, and not `second_full`'s); that the real `more` by
  L5s's rule comes to 3,268 rows on the stored search and map as they are, and that the real orders differ (they do
  by the hash, and the first step refuses if two are alike); that L5f's real report says of its seed what `of_l5f`
  asks (the measurement's four figures equal this run's); the memory and time of four passes of 3,268 rows from
  `second_full` at rank 64 on this card (L5s's own were of the same size, from `pre_r64`) and of a training from an
  averaged adapter.
- **Fragile.** One failed check of one model makes the run INCONCLUSIVE. The first step refuses for one missing file
  of L5f's run of the seed or of L4p's run. The bar is ONE stored sampling of ONE averaged model: every read here
  carries its draw, and `single` is one training, whose order spread (5.0 per 1,000 from `pre_r64`) is the reason for
  the line at 13. The tests: `tests/rlvr_lean/test_ladder_l5_cont.py` (the rules on hand-made rows, 32) and
  `test_ladder_l5_cont_stage.py` (the stage on the scripted world, 8).

### L5m: the best model, measured in full. Does the gain hold on the held-aside set and on fresh attempts? (fixed 2026-10-12, before L5c's read and before any run of this stage)

Under the owner's words: "towards getting it to solve never before solved math problems and at more reliable rates,
without overfitting", and of 2026-10-12: "make as much progress as you can tonight on improving this solver's solve
rate against the heldout tasks". Labelled **pretrained on published proofs**.

**Why.** L5f's `published` (144.77 per 1,000) and L5c's four models are measured on G's first sampling alone: 32
attempts at 392 problems, the one set of 12,544 episodes every model of L5d, L5s, L5f and L5c is read on. More than
a dozen models have been compared on that one sampling, and the highest of them carries the selection. Two
measurements no stage of the night has selected on say whether the gain is the model's: G's SECOND sampling (61
fresh attempts a problem) and the HELD-ASIDE set (600 problems, 16 attempts).

**What runs.** No training. One task. The CANDIDATE is one stored averaged adapter, named in the settings before
the task is queued and after L5c's report: the model that reads highest on G's first sampling among L5f's three
and L5c's four (a tie: the earlier made). It is measured in full, as L5's models are: the rungs, G's two samplings
(32 and 61 attempts, the sampling seeds `pre_r64`'s run recorded), the held-aside set (16 attempts). Its first
sampling is measured AGAIN here, not copied from the stage that made it. The BAR is `second_full`, the model the
candidate was trained from: G's two samplings and the rungs are read from L4p's stored attempts; its held-aside set
is measured here from its stored adapter (only read, served at its rank) unless a stored measurement of it with the
same setting exists, which is then read.

**The reads, fixed before the run.** Per attempt, paired by problem, 95% bootstrap over problems: these are reads
about two stored models on problems, not about a recipe over trainings.

- **Checks, read first; any FAIL makes the run INCONCLUSIVE:** the candidate's adapter and `second_full`'s are
  there, at rank 64, and are not each other's; the candidate is the adapter the stage that made it measured (its
  stored record of the mean); every set is the setting's (problems, attempts, sampling seed); no held-out problem,
  problem of G or held-aside problem is in a row the candidate's stage trained on (that stage's own record); Lean
  answered.
- **THE HELD-ASIDE SET (the primary):** the candidate minus `second_full`. The interval above zero: THE GAIN HOLDS
  ON THE HELD-ASIDE SET. Spanning zero: NOT SHOWN ON THE HELD-ASIDE SET. Below zero: IT DOES NOT HOLD THERE.
- **FRESH ATTEMPTS:** the candidate minus `second_full` on all of G over the 61 attempts of the second sampling:
  THE GAIN HOLDS ON FRESH ATTEMPTS, NOT SHOWN, or IT DOES NOT HOLD, by the same rule.
- **THE MEASUREMENT AGAIN, naming nothing:** the candidate's first sampling here minus its stored one, per 1,000.
- **Beside, naming nothing:** all of G over the 93 attempts; G' over the 61; goal problems solved in 93, solved
  reliably (half the attempts or more), and of the 147 `pre_r64` never solved in 93; the 74 long ones; each rung
  (the easy rung is where L4r read a cost); the share of distinct attempts; timeouts by set.
- **What follows.** Holds on both: the candidate is the best model measured in full, said without the caution of
  selection, and is the start the next cycle is proposed from. Not shown or not holding on the held-aside set: the
  night's first-sampling comparisons are said on the page to be selection on G until a further seed shows otherwise.

**What it cannot say.** One candidate: one seed, one draw of rows. G and the held-aside set are of one source.
Nothing here is about the model's own proofs.

**Budget.** About 51,700 episodes for the candidate and 9,600 for the bar's held-aside set: about 1 h 20 with 30
pool workers. Its smoke run first.

**The order, amended 2026-10-12 at 10:58 UTC, before any run of this stage and before L5c's real run.** The stage
was built before L5c's real run could start, and the highest model known is L5f's `published`. The first task
measures THAT model now (the committed setting), so that the check free of the selection is in hand before more
models are compared on the first sampling; L5c's real run follows it. If L5c's report then names a model that reads
higher on G's first sampling, a second task measures that one against the same bar, and both are reported. Nothing
else of the section changes: the reads, the bar and the rules are as fixed above.

**Made exact by the build (2026-10-12; nothing queued, nothing run on a GPU, nothing sent to Lean; the reads, the
checks and "What follows" above are not changed; the readings taken where the section left something open, what was
NOT built, what the smoke stage does not exercise and what is fragile are below).**
- **The stages and their names.** `ladder_l5_measure` and `ladder_l5_measure_smoke` (`gpu/ladder_l5_measure.py`;
  rules `domain/ladder_round/l5_measure.py`; report `reporting/ladder_l5_measure.py`), settings
  `ladder_loop.l5_measure` (one reader, `the_settings`). Its steps are named `ladder_l5m_...` and its files
  `l5m_...`, because `ladder_l5_measure_<model>` is already a step of L5's own. THE RUN DIRECTORY is its own and is
  named by its candidate, `ladder_l5_measure_<of>_<model>_seed<N>` (as committed
  `ladder_l5_measure_l5_full_published_seed3`); its report is `report_ladder_l5_measure.json`; its jobs
  `ladder_l5_measure_seed3.json` (est_minutes 110) and `ladder_l5_measure_smoke.json` (30), with the
  `resources` of L5f's jobs key for key. No adapter directory exists in a run of this stage. Nothing of L5f's,
  L5c's, L4p's, L5's or L6's run is written: each is read, and left file for file as it was (asserted in the
  scripted world).
- **The candidate, in the settings.** `ladder_loop.l5_measure.candidate` is three values: `of` (the stage that made
  it: `l5_full` or `l5_cont`), `seed` (its seed in that stage, which must be one that stage's own settings list)
  and `model` (its name there: `published`, `shallow`, `deep` of `l5_full`; `wide`, `again`, `more` of `l5_cont`).
  `bar` is `second_full` and nothing else. Anything else is refused by the reader. As committed: `of: l5_full`,
  `seed: 3`, `model: published`. It is SET BY HAND after L5c's report and before the task is queued; for a model
  of L5c the change is two lines, `of: l5_cont` and `model: wide` (or `again`, `more`). Nothing in the code chooses
  the candidate or reads L5c's report to choose it. THE TASK'S SEED IS THE CANDIDATE'S: a task of another seed is
  refused by the first step. A RUN KEEPS ITS CANDIDATE: a later step in a run directory prepared for another
  candidate is refused.
- **FIVE steps.** `ladder_l5m_prepare`; `ladder_l5m_measure_candidate` (the three rungs, then G's two samplings);
  `ladder_l5m_held_aside_candidate`; `ladder_l5m_held_aside_second_full`; `ladder_l5m_report`. Each is resumable
  and a finished one returns what it stored; the report step ALONE is built again each time its task is queued. No
  step trains, averages or deletes an adapter, and no step writes into a run it reads.
- **What is called, not copied.** L5's finished search and every stored run its task read: L5's own first-step
  reading (`ladder_l5.read_the_search`, `read_for_the_search`). `second_full`: L5f's own reading and start
  (`ladder_l5_full.read_second_full`, `the_start`), so everything L5f's first step refuses about it is refused here
  in the same words. The candidate's run: found by `ladder_l5_full.own_run` or `ladder_l5_cont.own_run` (this module
  holds no run name of another stage and no adapter path). A stored set held to the setting:
  `ladder_ceiling._stored`. The rungs and G: `ladder_ceiling.measure_model`, with the rungs, as L5's models are
  measured. The held-aside set: L5's own body, `ladder_l5.on_the_held_aside_set`, handed the adapter's directory.
  The refusals of a barred row: `l5.refuse_barred` and `l5_draws.refuse_problems_of`; the same two checks in the
  report by their own functions. NO FUNCTION OF AN EARLIER STAGE WAS CHANGED: L5 to L6 write what they wrote (their
  tests pass unchanged, but for six lists of stages that name a variable, which now hold `ladder_l5_measure_smoke`).
- **The candidate, held by the first step.** By the record of the stage that made it, before anything is written:
  its run's report of its seed (finished, not INCONCLUSIVE, to be read, from `second_full`, recipe `mean_constant`,
  measured on the same first sampling); its record of the mean (`ladder_l5_full_average_<model>` or
  `ladder_l5_cont_average_<model>`: the mean of its members, none of them alone, saved at rank 64 and alpha 128)
  and of its measurement (sampled from the adapter this stage would serve); its stored rows on G's first sampling,
  held to the setting; and, for every training the adapter is the mean of, that training's marker, stored ordered
  set and loss record. For `wide` the members are eight: L5f's four in L5F's run and L5c's own four. ON A GPU BOX
  the adapter's files are read too: missing, read back at another rank or alpha, or number for number
  `second_full`'s, the step refuses. The rows the candidate was trained on are written to `l5m_trained_rows.jsonl`
  by id and place (`training`, `id`, `problem_id`, `origin`, `row_of_the_file`), NEVER with text.
- **The bar.** On the rungs and on G's two samplings its rows are L4p's stored attempts (`l4_rungs_second_full`,
  `l4_reach_second_full`, `l4_more_second_full`), each held to the setting's problems, attempts and sampling seed,
  copied into this run and held again by the report. ON THE HELD-ASIDE SET the section says "unless a stored
  measurement of it with the same setting exists, which is then read": the one stage that would have made such a
  measurement is L6 (its step `ladder_l6_held_aside_second_full`), so the first step looks in L6's run
  (`ladder_l6_cycle_pre_r64_seed0`, or the run L6's own variable names). It is READ when that run holds the step's
  marker and its set, sampled from the same adapter, on the same problems, with 16 attempts and the same sampling
  seed; its rows are then copied to `l5m_stored_second_full_held_aside.jsonl`, the bar's step makes nothing and
  marks nothing, and the report says where the rows are from. Otherwise NOTHING IS REFUSED: the first step records
  why, and the bar is MEASURED HERE from the adapter L4p's run keeps, served at rank 64. L6 has not run for real,
  so the real run is expected to MEASURE it (the budget above counts it).
- **The sizes, which are read and not written in the code.** The rungs at 8 episodes a problem with sampling seed
  1002; G's first sampling at 32 attempts with 1001 and its second at 61 with 1020 (the samplings `pre_r64`'s run
  recorded: a run that recorded another number of samplings than two is refused); the held-aside set at 16 attempts
  with 1051 (L5's own setting and seed). The candidate's first sampling is sampled again with the seed its stored
  one was sampled with, from the same adapter.
- **The checks, five.** (1) The two adapters are there at the rank and are not each other's: the candidate's by
  its own stage's record (and by the files, when the first step read them), the bar's as L5f holds it, and the
  bar's held-aside set was sampled from the bar's adapter. (2) The candidate is the adapter its stage measured: the
  record of the mean and of its measurement, read AGAIN as they are when the report is built, with that run's
  report of its seed, and every measurement of this run was sampled from that one adapter. (3) Every set is the
  setting's: nine sets (the candidate's four; the bar's three stored ones and its held-aside set, stored or
  measured; the candidate's stored first sampling), each by its problems, its attempts a problem and the sampling
  seed its own step recorded. `pre_r64`'s stored sets are not among them: L5's first-step reading holds them. (4)
  No barred row in what the candidate was trained on: no held-out problem, no published row outside the `pretrain`
  half, no own row outside the base map, no problem of G, no held-aside problem. (5) Lean answered, on the five sets this run samples (four when the
  bar's held-aside set is read); a set it did not answer makes the report NOT TO BE READ and fails its step.
- **The reads, made exact.** Both by one rule (`l5_measure.by_the_interval`), the candidate minus `second_full`,
  per attempt, paired by problem, 95% bootstrap over problems: the interval's lower end above zero HOLDS; its upper
  end below zero DOES NOT HOLD; otherwise NOT SHOWN, so an interval that ENDS at zero spans it. FRESH ATTEMPTS is
  all of G over the second sampling alone (61 attempts a problem at the real sizes). THE MEASUREMENT AGAIN is the
  candidate's first sampling here minus its stored one. BESIDE: all of G over both samplings (93 attempts); the
  first sampling as measured here; G' over the fresh attempts; goal problems solved in 93; SOLVED RELIABLY, read as
  twice a problem's successes at or above its attempts over all 93; the goal problems `pre_r64` never solved in its
  93 stored attempts (counted from its stored rows: 147 is expected and is not a number of the code); the long ones
  (8 lines or more of the published proof, whose length the report alone reads); each rung; the share of distinct
  attempts; timeouts by set, where a pair more than a point apart is NAMED.
- **What follows, and the outcome the section does not name.** Holds on both: THE CANDIDATE IS THE BEST MODEL
  MEASURED IN FULL. Not shown or not holding on the held-aside set, whatever the fresh attempts read: SELECTION ON G
  until a further seed shows otherwise. The section names nothing for a gain that HOLDS ON THE HELD-ASIDE SET AND
  NOT ON FRESH ATTEMPTS: the report says exactly that ("NOT NAMED BY THE SECTION"), claims nothing of it, and it
  goes to the owner.
- **Other readings taken where the section left something open.** L5c's `single` is refused as a candidate: it is
  one training (L5f's stored `published_1`), not an averaged adapter with a record of the mean, and the section
  says "one stored averaged adapter". "No ... row the candidate's stage trained on" is read on the trainings the
  candidate's adapter is the mean of; for `again`, which was trained FROM L5f's averaged `published`, the rows of
  that start are not read again here (L5f's own run and L5c's first step held them). A tie between two candidates
  is not something the code sees: the settings name one.
- **NOT built.** The check again in Lean of timed-out attempts (a pair of sets more than a point apart is named,
  and no step checks it). A candidate that is one training. A second candidate, or a further seed, in one run. The
  counts with assembly.
- **The smoke stage.** `ladder_l5_measure_smoke` runs every step from the smoke runs on the box: it names L5's,
  L4p's, L5f's and L5c's runs by the variables L5c's smoke stage sets (without L5c's smoke-only variable), its own
  run directory `ladder_l5_measure_smoke`, and ALONE sets `RLVR_LEAN_LADDER_L5_MEASURE_SMOKE=1`: L5's smoke run
  reads INCONCLUSIVE and is read all the same, and a candidate whose averaged adapter is one of its members (which
  L5f's smoke stage records of its own run) is measured all the same; both are recorded and the report says A
  SMOKE RUN. The first step refuses that variable in a run directory of the candidate's own, or on the pool's base
  map. It does NOT name L6's run variable, so the smoke run MEASURES the bar on the held-aside set, as the real run
  does. ON THE BOX it needs, each run to its end: `ladder_l5_full_smoke` (its report not INCONCLUSIVE),
  `ladder_l4_dose_r64_smoke`, and `ladder_l5_deep_smoke` with the three smoke runs that task read
  (`ladder_l4b_map_smoke`, `ladder_l4_rows2_r64_smoke`, `ladder_l4_rank_smoke`); a candidate of L5c needs
  `ladder_l5_cont_smoke` too. WHAT IT DOES NOT EXERCISE: the real sizes; the path where the bar's held-aside set is
  read from a stored run; a candidate of L5c; an averaged adapter that differs from its members; G's second
  sampling and the read on fresh attempts (next bullet).
- **The smoke world holds ONE sampling of G** (added 2026-10-12, after the first smoke run on the box, task
  333ceb41, was refused at its first step). The smoke `pre_r64`'s run reads L1's smoke run alone, with no stored
  run of L2 beside it, so it recorded G's first sampling and no second, and every smoke run after it (L5's own
  among them) measured its models on that one. A SMOKE run of this stage goes on with it: the candidate is
  measured on the rungs, on that one sampling and on the held-aside set; the bar's stored sets are the rungs' and
  that sampling's; the first step records it (`smoke.one_sampling_of_g`); the report makes the primary on the
  held-aside set, reads FRESH ATTEMPTS and what follows as NOT READ, and says so in its smoke line. A REAL RUN IS
  REFUSED for a start run that recorded one sampling, as it was (the real start run recorded two), and the report
  refuses a run of one sampling whose first step did not record that it went on from one. The scripted tests alone
  exercise the second sampling and the read on fresh attempts; the test of the box's own smoke world builds every
  smoke run by its own smoke stage, in the box's order.
- **Fragile.** One failed check makes the run INCONCLUSIVE. The first step refuses for one missing file of the
  candidate's run. The bar on the rungs and on G is one stored sampling of L4p's run. NOT TESTED HERE AGAINST THE
  REAL RUNS, and so where the first step could refuse on the box: the candidate's real adapter files, read back
  (rank, alpha, numbers that are not `second_full`'s); whether L4p's real run holds the three stored sets with the
  problems, attempts and sampling seeds this config gives; whether L5f's real markers carry the fields read here
  (its averaging record's `of`, `read_back`, `adapter_saved`, `against_the_start_adapter`; its measuring record's
  `adapter`); and L5c's real run, which had not finished when this was built.
- **Tests.** `tests/rlvr_lean/test_ladder_l5_measure.py` (the rules on hand-made rows) and
  `test_ladder_l5_measure_stage.py` (the stage on the scripted world: the whole stage, the branches of the two
  reads, each failed check, the GPU path of the first step and of the served adapters, the bar read from a stored
  run and measured otherwise, a candidate of L5c, the refusals, the entry point, the smoke stage, the registration,
  the two jobs).

**Outcome of L5m (seed 3, 2026-10-12 UTC; `docs/spec/ladder-l5m-RESULT.md`): THE GAIN HOLDS ON THE HELD-ASIDE
SET AND ON FRESH ATTEMPTS.** One task (`ladder_l5_measure_seed3_r1`, task c8e2546c, 12:18 to 14:17 UTC, from
`a2bab6705`, 16 pool workers); all checks pass. The candidate is L5f's `published`, as the amended order names it; the
bar `second_full`. THE HELD-ASIDE SET (the primary): +0.01615 [+0.00771, +0.02469], 77.5 against 61.35 per 1,000 over
600 problems. FRESH ATTEMPTS: +0.03739 [+0.02643, +0.04839], 145.62 against 108.23 over the 392 goal problems. The
measurement again: 146.44 against the stored 144.77 (+0.00167 [-0.00303, +0.00654]). Beside: all of G over 93
attempts 145.9 against 108.93; G' on fresh attempts 42.09 against 26.9; goal problems solved in 93 attempts 264
against 267 (gained 23, lost 26), solved reliably 32 against 16; the long ones +0.01642 [-0.00044, +0.03342]; the
rungs +0.0512, +0.0266 (holds zero), +0.0425; distinct attempts on G 0.70 against 0.80. **What follows by the rule
above:** THE CANDIDATE IS THE BEST MODEL MEASURED IN FULL: THE NEXT CYCLE IS PROPOSED FROM IT. Said beside it: the
gain is in how often a problem is solved, not in how many problems are. L5c's real run follows; if its report names
a model that reads higher, a second task measures that one.

### L6b: a second round, from the best model. Do proofs the stronger model finds add where the weaker model's did not? (fixed 2026-10-12, after L5d's read at three seeds and L5f's seed 3, before L5c's and L5m's reads and before any run of this stage)

Under the owner's words of 2026-10-12, on the ask "three fresh seeds or six": "let's do 3 seeds. I don't need super
high confidence to keep pushing in this direction". Labelled **pretrained on published proofs**.

**Why.** L5d reads, over three fresh seeds from the once-pretrained model: THE MODEL'S OWN PROOFS BESIDE PUBLISHED
ONES BEAT PUBLISHED ONES ALONE ON THE GOAL SET (+12.69 per 1,000, interval over seeds +9.08 to +16.30). Its rule sends
the bet to a second round. L6 as fixed on 2026-10-11 is a second round from `pre_r64`, searched by seed 0's `deep`
model (about 110 per 1,000), one training a model, about nine hours. Since then the best model moved: L5f's averaged
`published`, trained from the twice-pretrained model, reads 144.77, and from that stronger start the own-proof kinds
read BELOW the published kind at the one seed run (-14.87). Every own proof in L5f was found by `pre_r64`, the weaker
model, before the second pass. The bet is a cycle: the improved model searches, and what IT finds is trained on. That
has not been run from the stronger model. L6b runs it, at the size of a scout. L6 as fixed does not run before L6b is
read; its section stands unchanged.

**The start, B.** L5f's stored averaged `published` of seed 3 (the adapter L5m measures in full and L5c's `again` is
trained from). Only read.

**The search.** B, served at rank 64, makes 64 attempts at each of the 865 hard problems of L6's "Search 2": the 480
that search 1 left unsolved (`pre_r64` failed 64 of 64) and the 385 candidates that neither draw took (`pre_r64`
failed 8 of 8). 55,360 attempts, L6's sampling place. The held-aside set is not searched. A problem B solves is a NEW
SOLVE; its row is one verified attempt, drawn by the task's seed as L6 draws it. N is the number of new solves.
Fewer than 150: nothing is trained and the search is the finding.

**Two models are trained, each from B,** with the recipe as it is (four orders, order seeds 1 to 4, one pass,
constant learning rate, the averaged adapter measured), 800 rows each, K = the smaller of 400 and N:
- **`new_own`**: K new solves (drawn by the seed when N is above 400) and the published rows 801 to 1,600 - K of
  seed 3's order.
- **`old_own`**: K of `pre_r64`'s deep solves (the first K, in their stored order, of the 400 that L5d's `deep` set
  of seed 3 holds) and the same published rows.

**The bar is L5c's stored `again`,** only read: from B, the published rows 801 to 1,600 of the same order, the same
recipe, measured the same way. The three models differ in their own rows alone: none, `pre_r64`'s, B's.

**Measured:** each of the two on G's first sampling alone (32 attempts at each of the 392 goal problems, sampling
seed 1001), as L5f's and L5c's models are.

**The reads, fixed before the run.** On whole numbers of successes over the 12,544 episodes, per 1,000 attempts,
paired by problem. One seed: each read is a scout's.

- **Checks, read first; any FAIL makes the run INCONCLUSIVE:** B is L5f's averaged `published` of the seed (the mean
  of its four, at rank 64, not `second_full`'s); L5c's run of the seed is there, finished and not INCONCLUSIVE, with
  `again`'s set (rows 801 to 1,600 of the order) and its stored first sampling of the setting; the searched set is
  L6's Search 2 set, made by L6's own function and held to L6's own checks (no problem held out, of G, of the
  held-aside set or of round 1's solves); each set is its rule made again (no row with text among the published rows,
  no barred row); the two own sets hold the same published rows; each trained adapter is not B's, each averaged adapter is the
  mean of its four and is the one served; Lean answered (at most 0.02 of the search and of each set without an
  answer).
- **THE SEARCH, a finding of its own:** N, and of it how many of the 480 (problems `pre_r64` never solved in 64
  attempts) and of the 385; how many are solved in under half of B's 64 attempts; their line counts beside round 1's.
- **PRIMARY, its own new proofs from the best model: `new_own` minus `again`.** At or above +8.5 per 1,000: THE
  MODEL'S OWN NEW PROOFS ADD FROM THE BEST MODEL. Below zero by more than 8.5: THEY COST THERE AT THIS SEED. Between:
  NOT SHOWN AT ONE SEED.
- **WHOSE PROOFS: `new_own` minus `old_own`.** At or above +8.5: PROOFS THE STRONGER MODEL FOUND TEACH MORE THAN THE
  WEAKER MODEL'S. Below zero by more than 8.5: THEY TEACH LESS. Between: NOT SHOWN AT ONE SEED.
- **Beside, naming nothing:** `old_own` minus `again` (L5f's read, one step up); each of the three against B's stored
  first sampling (144.77), `second_full` and `pre_r64`: all of G, the 200 goal problems `pre_r64` solves in its first
  sampling, G', goal problems solved (gained and lost against B), solved reliably (16 of 32 or more), the 74 long
  ones, the share of distinct attempts.
- **Timeouts:** each set's share; two more than a point apart are checked again and read both ways (no step does it).

8.5 is L5f's own line for two averaged models.

**What follows, fixed now.**
- THE MODEL'S OWN NEW PROOFS ADD: the cycle turns from the best model. Two further seeds of `new_own` and of the bar
  (not built: no run holds `again` at another seed) before anything is said over seeds; the highest model is a
  candidate for a measurement in full; the cycle at full size has this beside L5d's read to put to the owner.
- NOT SHOWN AT ONE SEED: said as it is, with the search's yield. A third round is not run on it.
- THEY COST THERE: from the best model, at this size and seed, published rows read again are worth more than the
  model's own proofs, old or new. It goes to the owner with L5f's read; the full-size cycle is not proposed on it.
- Whatever the primary reads, WHOSE PROOFS says whether the search by the stronger model is the part worth scaling.

**What it cannot say.** One seed, one search, one draw. G's first sampling alone, the set more than a dozen models
have been compared on; nothing of the held-aside set. The bar is published rows B's lineage has read twice already;
the own proofs are of problems it never trained on: as in L5d, a win is "its own proofs of new problems beat reading
published proofs again", not "beat published proofs of new problems" (the fourth kind, not built without the owner's
word). K own rows in 800: a dose of at most half. The new solves are of harder problems than the old ones, so WHOSE
PROOFS does not tell "found by a stronger model" from "of harder problems".

**Budget.** The search about an hour and a half to two hours with 30 pool workers; eight trainings of 800 rows
(about four minutes each) and two measurements of about 16 minutes: about three hours. Its smoke run first. It runs
after L5c's real run (the bar is read from it).

**Made exact by the build (2026-10-12; nothing queued, nothing run on a GPU, nothing sent to Lean; the reads, the
checks and "What follows" above are not changed; the readings taken where the section left something open, what was
NOT built, what the smoke stage does not exercise and what is fragile are below).**
- **The stages and their names.** `ladder_l6b` and `ladder_l6b_smoke` (`gpu/ladder_l6b.py`; rules
  `domain/ladder_round/l6b.py`; report `reporting/ladder_l6b.py`), settings `ladder_loop.l6b` (`start: published`,
  `bar: again`, `seeds: [3]`; one reader, `the_settings`, which refuses another start, another bar, a seed that L5c's
  setting does not fix, and any other key). THE RUN DIRECTORY is its own, `ladder_l6b_published_seed<N>`; its files
  are `l6b_*` and `report_ladder_l6b.json`; its jobs `ladder_l6b_seed3.json` (est_minutes 200) and
  `ladder_l6b_smoke.json` (30), with the `resources` of L5c's jobs key for key. The stage names no variable. Nothing
  of L5f's, L5c's, L5d's, L4p's, L5's or L6's run is written: each one read is left file for file as it was
  (asserted in the scripted world, for the real-shaped runs and for the box's smoke runs).
- **SIXTEEN steps,** in the section's order: `ladder_l6b_search_prepare`; `ladder_l6b_search`; `ladder_l6b_prepare`
  (the two sets, and each set in its four orders); `ladder_l6b_train_new_own_1` to `_4`,
  `ladder_l6b_average_new_own`, `ladder_l6b_measure_new_own`; `ladder_l6b_train_old_own_1` to `_4`,
  `ladder_l6b_average_old_own`, `ladder_l6b_measure_old_own`; `ladder_l6b_report`. Each is resumable and a finished
  one returns what it stored; the report step ALONE is built again each time its task is queued. No step deletes an
  adapter, and every adapter is kept (eight trainings and two averaged ones).
- **What is called, not copied.** L5's finished search and every stored run its task read: L5d's own first-step
  reading (`ladder_l5_draws.drawn_by_the_seed`). `second_full`: L5f's reading (`ladder_l5_full.read_second_full`).
  B, in L5f's run of the seed: L5c's own reading and start (`ladder_l5_cont.read_l5fs_run`, `the_bar`), so what L5c's
  first step refuses about L5f's run is refused here in the same words under this stage's; this module holds no run
  name of another stage and no adapter path. L5d's run of the seed: L5f's reading of it
  (`ladder_l5_full.read_l5ds_run`). THE SEARCH'S PROBLEMS: L6's own rule, by two functions CUT OUT of L6's first step
  for this stage and called by that step too (`ladder_l6.solved_by_search_one`, `the_second_search`; L6 writes what
  it wrote and its tests pass unchanged), with L6's sampling seed (`ladder_l6.search_seed`). The search's body: L5's
  `search_deep`, handed B's adapter. A new solve's row: `l5.solve_rows`. The trainings: L4t's body with `start_from`,
  as L5c's `again` uses it. The averaged adapters: L5n's `the_mean_adapter`. The measurements: `measure_model` with
  `rungs=False`. The checks of the trainings and models and their lines: L5d's. The reads against a stored model:
  `l5_stock.against_the_start`. Earlier stages' tests pass unchanged, but for the lists of stages that name a
  variable.
- **K, where the section says "the smaller of 400 and N".** `l6b.the_dose`: the smaller of HALF THE ROWS OF A SET,
  THE OLD OWN ROWS L5d's `deep` set of the seed holds, and N. At the real sizes this is the section's number (half
  of 800 is 400, and L5d's set holds 400 own rows); written so, both models always hold the same K. A set's size is
  the number of rows L5f's `published` set holds (800), read from that run.
- **The published rows.** "Rows 801 to 1,600 - K of seed 3's order" is taken as: the NEXT rows of the seed's order
  of the pretraining file after the ones L5f's `published` set holds, which are L5c's `again` set; the first step
  holds them to the set L5c's run stores as `again`, ID FOR ID AND PLACE FOR PLACE, and refuses otherwise; each model
  takes the first 800 - K of them, the SAME ones. A published row is stored by id and place, never with text.
- **The old own rows.** The 400 own rows of the `deep` set that L5d's run of the seed stores, in its stored order
  (`l6b_old_solves.jsonl`); the first step makes L5d's rule again for the seed and refuses a stored set that is not
  it. `old_own` takes the first K.
- **The search.** Model `published`, served from B's adapter in L5F'S run; the 865 problems are whatever L6's rule
  gives on the stored runs (the first step records the partition: attempted by search 1, solved by it, part a, part
  b, held aside); the sampling seed is L6's (the seed of the search 0, place 52: 1052), so if L6 itself ever runs,
  its search 2 samples the same problems at the same seed with another model. A NEW SOLVE is a problem with at least
  one verified attempt; its row is ONE verified attempt, drawn by the TASK's seed; when N is above K the K of the set
  are drawn by the task's seed too (`l5.capped`). "Line counts beside round 1's" is taken as: the line counts of the
  new solves beside those of the old own rows.
- **"Fewer than 150 new solves" is not a check.** The minimum is L6's setting (`ladder_loop.l6.search.minimum`).
  Below it the sets step builds no set and records why, the twelve training, averaging and measuring steps make and
  mark nothing, and the report (not INCONCLUSIVE for that) reads the primary and whose proofs as NOT READ, says what
  the search found, and what follows is THE SEARCH IS THE FINDING.
- **The bar.** L5c's stored `again`: its report, the markers of its four trainings and of its averaged adapter, its
  stored set and its rows on G's first sampling, copied to `l4_stored_l5c_again_reach.jsonl`. ITS ADAPTERS ARE NOT
  READ. The first step refuses L5c's run when it is not there, not finished, of another seed, start or recipe,
  INCONCLUSIVE, not to be read or measured otherwise; when a marker, its stored set or its rows are missing; when its
  first step recorded that `again` was trained from another adapter than B's; when its rows on the sampling would
  not pair (other problems, attempts or sampling seed). The stored first samplings of `pre_r64`, `second_full` and B
  are copied beside it and shown, naming nothing.
- **The checks of the report,** each read on what is stored THEN: B is L5f's averaged `published` (L5c's own check
  of L5f's run, and: the search was served from B's adapter, every training started from it); L5c's run of the seed
  holds the bar (its report read again; the next rows are still its `again` set); THE SEARCHED SET IS L6's SEARCH 2
  (the rule made again gives the stored problems in their order and parts; none is held out, of G, held aside or
  solved by round 1; the search sampled exactly those, each once, with its attempts; the new solves made again from
  the stored search are the stored ones); the stored first sampling of the start models and of the bar are the
  setting's; Lean answered the search; each set is its rule made again (K, the same published rows in both, none
  with text, the next rows L5c's, each of the four orders its order seed's); then L5d's checks of each training and
  each model. Two order seeds that give one order of a set are refused outside a smoke run.
- **The reads.** On whole numbers of successes and attempts over all of G's first sampling, with L5f's line
  (`l5_cont.A_LINE`, 8.5 per 1,000): at or above plus the line, below minus the line, or between. Beside, naming
  nothing: `old_own` minus the bar, and timeouts by model. WHAT FOLLOWS reads the primary alone, with WHOSE PROOFS
  said beside it.
- **NOT built.** A further seed and the read over seeds (What follows names "two further seeds of `new_own` and of
  the bar"; no run holds B or the bar at another seed, and the settings reader refuses one). The check again in
  Lean of timed-out attempts. A measurement in full of either model (the rungs, G's second sampling, the held-aside
  set). A second seed of the search. Any deletion of adapters.
- **The smoke stage.** `ladder_l6b_smoke` alone sets `RLVR_LEAN_LADDER_L6B_SMOKE_PROBLEMS=2`, refused unless the run
  directory is one the task names and the map is the fixtures': L5's smoke run reads INCONCLUSIVE and is read all the
  same (recorded); the fixture's map leaves search 2 no problem, so the search attempts the two map problems no
  search took, each marked as NOT the rule's; one new solve is enough; K is 1 (half of L5f's smoke set of two rows),
  so a set is one row of the model's own and one published row; orders of those two rows that are alike are trained
  all the same; an averaged adapter that is one of its members is kept and said. ON THE BOX IT NEEDS EIGHT SMOKE
  RUNS AND NO OTHER: `ladder_l5_full_smoke`, `ladder_l5_cont_smoke`, `ladder_l5_draws_smoke_mean_constant`,
  `ladder_l4_dose_r64_smoke`, `ladder_l5_deep_smoke`, `ladder_l4b_map_smoke`, `ladder_l4_rows2_r64_smoke`,
  `ladder_l4_rank_smoke`. L6's smoke run (`ladder_l6_cycle_smoke`) is not read. THE TEST OF THE BOX'S SMOKE WORLD
  builds that world as it is, every run by its own smoke stage in its own environment and in the box's order (L1's;
  the pretraining's, the arm's, the rank check's, L4t's, L4t2's; L4p's; L5's; L5d's; L5f's; L5c's), moves every run
  but those eight away, and runs this stage's smoke task: all sixteen steps run, the eight runs are left file for
  file, and a real run is refused for such a round 1. The stage reads G's FIRST sampling alone, so the smoke world's
  one sampling is all it asks for.
- **What the smoke run does NOT exercise.** K above 1; a set of 800 rows and a training of 100 optimizer steps from
  an averaged adapter; a search of 865 problems; the too-few-solves path, unless B solves neither of its two
  problems on the box, in which case NOTHING IS TRAINED there and the trainings are not exercised (the scripted
  tests alone then); the reads at their lines (the stand-in sizes say nothing).
- **What is fragile.** L5c's real run did not exist when this was built: the first step's reading of it is
  exercised on its smoke run and on scripted runs alone, and it refuses until that run has finished with a report
  that is not INCONCLUSIVE. The first step refuses for one missing file of L5f's, L5c's or L5d's run. One failed
  check makes the run INCONCLUSIVE. The bar is ONE stored sampling of one averaged model, measured by another task
  at another hour of the pool; nothing here measures it again. The search is 55,360 attempts through the pool, the
  longest single step of the run. In the scripted world the real-shaped run has ONE new solve (with two, the four
  rows of `old_own` give one order at two order seeds, which the stage refuses): the seeded draw of K among more new
  solves is exercised by the rules' tests and by the smoke-world test alone. One seed, one search, one draw: every
  read is a scout's.

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

Both are L4's, from "Further seeds of the arm, made exact before they run" (2026-10-09). Neither stops seeds
1 and 2 from running; each says which reading was built.

1. **`pre` is measured once.** At seeds 1 and 2, `with` is sampled with that seed's sampling seeds and the
   `pre` side is seed 0's stored attempts. So those two primaries are paired by problem without the common
   random numbers seed 0's has, and all three differences share the same `pre` attempts: a low draw of `pre`
   on G′ would lift all three, and three seeds clear of zero are not three independent confirmations. The
   pooled interval does carry `pre`'s sampling noise, problem by problem. Built: the reading that changes
   least (the spec's "the same `pre`", no new measurement). The alternative is to measure `pre` again with
   each further seed's sampling seeds (one measurement a seed, 42,817 generations, 46 to 55 minutes), which
   would give each seed a `pre` side of its own. Is one measurement of `pre` enough for the three-seed read?
2. **What the three seeds must show is not said in L4's own words.** Its "One seed first" says a positive
   read licenses the two other seeds and nothing more, "as in Step 2", and Step 2's rule is: concluded only
   on three seeds, each of its own sign and the pooled interval clear of zero. Built: the read prints what
   that rule reads (each seed's sign, the pooled interval) and the count of seeds whose own interval is clear
   of zero, and names no verdict. Is Step 2's rule the one L4's three seeds are read by?
