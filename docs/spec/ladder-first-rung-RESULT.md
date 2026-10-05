# Ladder, first rung (13b): the run did not show building; the untrained band below got worse (one seed)

Spec: the first experiment's spec section 13b (the arm, the primary, the secondaries, the census and
the branches, pre-registered 2026-10-04; its two exact rules, the VOID range and "inside the base's luck", were
written before the run was queued). Read on 2026-10-04 at one seed.

**Verdict.** One pass of training on the 50% band did **not** lift the band below it. On the 46 conjectures
the base proves 1 or 2 times in 12 and the adapter never saw, fresh samples read **adapter minus base −0.0344
[−0.0634, −0.0036]** (base 0.1123, adapter 0.0779): the interval lies below zero, so the pre-registered branch
is **STOP and diagnose**. On statements the base never proved, the adapter is inside the base's own luck in
both directions. This is one seed, 46 conjectures, one pass and one selection: **the run did not show
building**; it does not show that a ladder cannot be built.

The diagnosis (section 6) changes what the branch's sentence should be taken to mean. The loss is not special
to training on the band: on the below-band conjectures every adapter has sampled, all four native-format
adapters fall below the base, whatever they trained on, and the adapters trained in the legacy format on the
same band do not fall as far. What the adapter writes got shorter, and the proofs it lost are the long ones.

The census is the other finding: **the base proves the negation of 420 of the 974 never-proved conjectures
(43.1%)**, so at least 28% of the 1,500 conjectures the challenger kept are false as stated.

## What ran

One task, `ladder_13b` (74bbbae5, 8 h 10 min on the GPU box, all three parts finished; commit f2b3c76af).

- **Arm `ladder_half_native`:** the `half_pass_rate` selection (137 pool conjectures, 3 to 10 proved of 12),
  native format, seed 0, Phase A's hyperparameters, one pass = 18 optimizer steps.
- **Fresh samples**, sampling seed 1 (the stored samples used seed 0), base then adapter set by set in one
  engine session: 12 each on the 46 below-band conjectures (31 proved once, 15 twice; 37 in the pool, 9 in the
  holdout; **0 of the 46 are in the training set**, checked by the run), 8 each on the 899 workbook-holdout
  problems the base never proved, 12 each on the 974 never-proved conjectures.
- **Standard evaluation** of the adapter (19,464 samples, seed 0) and its held-out loss by part.
- **Census**, base model only: for each never-proved conjecture the theorem `¬ (∀ binders, statement)`,
  compile-checked, checked exact by Lean, 12 samples (seed 1).
- **Reach**, base model only: 64 samples (seed 2) on each of the 899.

Reports: `experiments/rlvr_lean/ladder_13b/steps/ladder_report.jsonl`, `negation_census.json`, `base_reach.json`.

## Before the branch: how the fresh samples were checked

| Set | Model | Checks | Timeouts | Server errors | Median check (s) |
|---|---|---|---|---|---|
| Below band (the primary) | base | 552 | 0 | 0 | 0.766 |
| | adapter | 552 | 0 | 0 | 0.685 |
| Workbook, never proved | base | 7,192 | 0.626% | 0.250% | 1.557 |
| | adapter | 7,192 | 0.528% | 0.236% | 1.093 |
| Conjectures, never proved | base | 11,688 | 0.205% | 0.180% | 0.542 |
| | adapter | 11,688 | 0.180% | 0.094% | 0.446 |

Every check of the primary was judged, for both models. On the other two sets the adapter has slightly fewer
unjudged checks than the base. The conditions do not explain any difference below.

## 1. The primary: pass rate on the 46 untrained below-band conjectures (fresh, paired)

| Base | Adapter | Difference | 95% interval | Branch |
|---|---|---|---|---|
| 0.1123 (62 of 552) | 0.0779 (43 of 552) | −0.0344 | [−0.0634, −0.0036] | **STOP and diagnose** |

- The VOID check passes: the base's fresh rate, 0.1123, lies inside [0.083, 0.188] (its stored rate on the
  same conjectures was 0.1105; the predicted fresh rate was 0.132).
- Per conjecture: the adapter is lower on 24, higher on 8, equal on 14. Proved at least once: base 36, adapter
  24 (15 only by the base, 3 only by the adapter).

## 2. The secondaries: statements the base never proved, proved at least once in fresh samples

| Set | Base | Adapter | Both | Interval of the difference |
|---|---|---|---|---|
| 974 never-proved conjectures, 12 samples | 13 | 11 | 4 | [−0.0103, +0.0062] |
| 899 unsolved workbook problems, 8 samples | 4 | 5 | 1 | [−0.0045, +0.0067] |

Both inside the base's luck. The base's 13 sits at the low end of what its stored counts predicted (25, 99%
range 13 to 39); 420 of those 974 conjectures are false (section 4) and cannot be proved by anyone.

## 3. The standard evaluation: adapter against the base's stored samples

| Measure (base) | Adapter | Difference | 95% interval | Label |
|---|---|---|---|---|
| Workbook-holdout pass@1 (0.0744) | 0.0758 | +0.0014 | [−0.0016, +0.0044] | no measurable change |
| Conjecture-holdout pass@1 (0.2956) | 0.3069 | +0.0113 | [+0.0031, +0.0200] | improved |
| miniF2F-test pass@1 (0.3636) | 0.3646 | +0.0010 | [−0.0059, +0.0082] | no measurable change |
| miniF2F-test pass@32 (0.4631) | 0.4631 | 0.0000 | [−0.0164, +0.0164] | no measurable change |

- Solved at least once: conjecture holdout 112 (base 113; 2 new, 3 lost), workbook holdout 118 (114; 6 new, 2
  lost), miniF2F-test 113 (113; 2 new, 2 lost).
- Distinct attempts on the conjecture holdout 0.567 (base 0.730; three passes on Phase A's picks: 0.36 to
  0.39). The adapter starts 99.96% of its attempts with the sequence-start token.
- Held-out loss, evaluation read: NF4 0.2474 → 0.2137 (reduction 0.0336 [+0.0200, +0.0488], improved); FP8
  0.2382 → 0.2079 (0.0303 [+0.0185, +0.0434], improved). By part the fall is the ending (newline 0.167 → 0.095
  nats, fence 0.040 → 0.011); the body goes 0.1921 → 0.1870 nats per token. One pass does not overfit: during
  training the loss falls at every read, 0.2454 → 0.2111 at step 18.
- Lean checks: timeouts 0.375%, server errors 0.298%.

Beside three passes on Phase A's picks in the same format (`native-format-RESULT.md`; a different selection,
so not a controlled comparison): one pass on the band gives most of the conjecture-holdout gain (+0.0113
against +0.0129) and little of the workbook gain (+0.0014 against +0.0099). Where the conjecture-holdout gain
sits (bands from the base's first six stored
samples, its rate from the other six, so nothing is selected on the number compared):

| The base proved it, in six samples | Statements | Base | Adapter | Difference |
|---|---|---|---|---|
| never | 187 | 0.0045 | 0.0022 | −0.0022 |
| once or twice | 9 | 0.2037 | 0.2407 | +0.0370 |
| three to five times | 40 | 0.6875 | 0.7312 | +0.0437 |
| every time | 60 | 0.9500 | 0.9833 | +0.0333 |

It gains where the base already succeeds and loses, in proportion, where it almost never does.

## 4. The census: never-proved conjectures whose negation the base proves

| | Conjectures |
|---|---|
| Never proved by the base in 12 stored samples | 974 |
| Negation built / compiles | 974 / 974 |
| Negation exact (Lean confirms it equals `¬ (the conjecture's type)`) | 931 |
| Negation inexact (the statement uses a variable it never binds) | 43 |
| **Disproved: the negation has a verified proof** | **420 (43.1%)** |
| of which with an exact negation | 417 of 931 (44.8%) |
| of which with an inexact negation | 3 of 43 (7.0%) |
| Undetermined | 554 (56.9%) |

- An inexact negation is the stronger claim ("for every x, not ..." where the true negation is "not for every
  x ..."): a proof of it still shows the conjecture false, but a conjecture false only for some x cannot be
  disproved that way. So 420 is a lower bound, and the 43 inexact ones are mostly undetermined for that reason.
- The disproofs are firm: 262 of the 420 are proved in at least 6 of 12 samples, 62 in all 12; 45 in exactly
  one. They are counterexamples: 248 open with `push_neg` and 142 with `intro`.
- By kind of statement: inequalities 48.2% disproved (of 650), equalities and equivalences 54.1% (of 159),
  existentials 25.0% (of 68), statements that are already negations 4.1% (of 97).
- Of the conjecture holdout's 296 statements, 81 are known false (27.4%): its pass@1 has a ceiling below 0.73.
- No contradiction: none of the conjectures either model proved afresh (13 and 11) is among the disproved.
- Lean verdicts of the 11,688 negation attempts: 2,945 verified, 3 timeouts, no server errors.

## 5. The base's reach on the 899 (last and optional)

64 samples each (seed 2): **21 of the 899 proved at least once**. The 4 the base proved in its 8 fresh samples
are all among them, so **878 remain unproved by the base in all 80 samples** (8 stored, 8 fresh, 64): that set
is "unsolved by the base model" for later rounds. 102 of 57,536 attempts verified; timeouts 0.58%, server
errors 0.30%. Seven of the 21 were proved once in 64. The adapter's 5 fresh-proved problems (section 2) are
all within the base's 21: it proved nothing the base cannot.

## 6. Diagnosis: go and see, then compete the causes (reading only, no new GPU work)

**What the adapter wrote on the 46 conjectures, against the base (552 fresh attempts each):**

| | Base | Adapter |
|---|---|---|
| Proof tokens per attempt, mean (median) | 46.1 (32) | 34.2 (24) |
| Finish reason | all stopped | all stopped (none hit the length cap) |
| One-line attempts | 15.4% | 27.4% |
| Attempts shorter than the conjecture's own shortest verified proof | 56.7% | 72.1% |
| Verified among attempts at least that long | 52 of 239 | 22 of 154 |
| Verified among shorter attempts | 10 of 313 | 21 of 398 |
| Attempts with a `have` step (verified) | 110 (16) | 44 (6) |
| Distinct attempts of 12, per conjecture | 9.26 | 7.70 |
| Verified proofs shorter than the base's usual attempt for that conjecture | 16 | 23 |
| Verified proofs at least as long | 46 | 20 |

The adapter's attempts are shorter on 44 of the 46 conjectures. It finds MORE short proofs (+7) and far fewer
long ones (−26). Does the loss track the length of the known proof? Not its absolute length (Spearman +0.01),
but its length against what the base usually writes: where the known proof is shorter than the base's median
attempt (16 conjectures) the adapter reads 17 of 192 against 19; where it is at least as long (30
conjectures), 26 of 360 against 43. Seventeen of the nineteen lost proofs are there.

**The rivals, with what the data says about each:**

| Cause | Layer | Prediction | What was read | Status |
|---|---|---|---|---|
| (c) Nothing changed; 46 conjectures at one seed is noise | L0 | per-conjecture changes look like binomial sampling | Size: sign-flip test p = 0.047, binomial null z = −2.0 (p = 0.050). Direction: 24 conjectures down, 8 up (sign test p = 0.007); proved at least once 36 → 24 (15 lost, 3 gained, p = 0.0075). An independent sample of the same adapter (its standard evaluation, other sampling seed) on the 9 below-band conjectures in the holdout: 10 of 108, against the base's 14 (stored) and 18 (fresh) | **Not excluded by size alone (p ≈ 0.05); unlikely given the direction and the replication.** |
| (d) A measurement difference between the two models' samples | L0/L1 | unjudged checks, cut-off proofs, a format slip, an inconsistent verifier | 0 timeouts and 0 server errors on both; every attempt ended by itself; the adapter starts all 552 with the sequence-start token; no proof text was ever judged two ways (798 texts); the base's fresh rate equals its stored one (62 against 61) with different texts | **Nothing found.** |
| (a) Brevity: the target is each conjecture's shortest proof, so the round teaches short proofs | L3 | shorter attempts; long proofs lost, short ones kept | the table above: −26 long proofs, +7 short; `have` attempts 110 → 44 at an unchanged success rate (14.5%, 13.6%), which alone is 10 of the 19 lost. Splitting the loss into "shorter attempts" and "less success at a given length" gives −13 and −6, or −6 and −13, depending on the order | **Supported as what changed in the attempts. It does not separate the native adapters from the legacy ones (next table).** |
| (b) Narrowing: fewer distinct attempts costs most where a proof is found once in twelve | L4 | fewer distinct attempts; more mass on the base's commonest (failing) attempt | distinct attempts 9.26 → 7.70, entropy 2.05 → 1.79 nats, and conjectures that narrowed more lost more (Spearman +0.30, p = 0.047). But the adapter does NOT pile onto the base's commonest attempt (19.9% of its samples repeat it, the base's own fresh samples 19.0%): it writes different, shorter attempts (53% of its texts are ones the base had already written, against 68% of the base's own fresh texts) | **Partly supported (less variety), refuted in the "repeats the base's mode" form.** |
| (e) Not the band: any round in the native format does this | L2 | other native adapters lose on below-band conjectures too; legacy adapters trained on the same band do not | next table | **Supported, on 9 conjectures.** |

**Every adapter on the 9 below-band conjectures of the holdout** (each adapter's own 108 standard-evaluation
samples; none trained on them; the base read 14 stored and 18 fresh):

| Adapter | Format, passes | Verified of 108 | Without one conjecture, of 96 |
|---|---|---|---|
| base | | 14 stored, 18 fresh | 13, 17 |
| learning-progress picks, seeds 0, 1, 2 | legacy, 3 | 16, 14, 14 | 9, 10, 7 |
| random, seed 0 | legacy, 3 | 15 | 6 |
| 50% band (`half_pass_rate`), seeds 0, 1, 2 | legacy, 3 | 21, 19, 14 | 12, 12, 6 |
| below-band heuristic, seeds 0, 2 | legacy, 3 | 29, 35 | 17, 23 |
| learning-progress picks, seeds 0, 1, 2 | native, 3 | 6, 6, 4 | 6, 5, 3 |
| 50% band, this run | native, 1 | 10 | 6 |

- All four native adapters are below every legacy adapter and below the base. The three that trained on Phase
  A's picks, not on the band, lose at least as much as the ladder adapter. So "training on the band made harder
  conjectures worse" is not what the data supports; "a round in the native format made rarely-proved
  conjectures worse, whatever it trained on" is.
- The one conjecture set apart (`a219a34a9eca58b5`) is proved by `rw [h₃]` alone; the base adds a `ring` that
  fails, 10 or 11 times in 12. The legacy adapters learned to stop (4 to 12 of 12), the three-pass native ones
  did not (0, 1, 1), the ladder adapter half-way (4). It carries 75 of the legacy adapters' 177 proofs. Without
  it the legacy adapters trained on easy or mid conjectures are below the base too (6 to 12 of 96 against 13
  and 17), though less than the native ones (3 to 6).
- The only adapters ABOVE the base on these conjectures are the two trained on below-band conjectures
  themselves (the difficulty heuristic: 29 and 35 of 108).
- Nine conjectures. The 46-conjecture fresh-sample reading exists for one adapter only.

**What the reading does not settle (assumptions, labelled):** why the native format is harder on rare
conjectures than the legacy one. The native round fits its examples harder at the same settings (last training
loss 0.0057 against 0.0183 on the same batches, `native-format-RESULT.md`), which is the candidate; it is not
tested. Attempt length does not separate the two formats on these 9 conjectures (mean 24 to 28 tokens for the
legacy adapters trained on easy or mid conjectures, 24 to 27 for the native ones; the base 33).

**Levers, each a change to make and measure** (none built or queued; the measure for all is the fresh
below-band pass rate, adapter minus base, now −0.034, with more than 46 conjectures):

1. **Train on a longer proof.** The target is each conjecture's SHORTEST verified proof; use a random or the
   longest one. Addresses (a). Also read: verified proofs at least as long as the base's usual attempt (46 →
   20 now). Drop it if the below-band difference stays below zero.
2. **Leave the closing tokens out of the loss.** What a round learns first is to end the proof (the newline and
   fence fall, the body does not). Addresses (a) at its source. Also read: mean attempt length (46.1 → 34.2 now).
3. **A smaller dose in the native format.** Fewer steps or a lower learning rate; the held-out loss by part
   shows the body at its best by step 8 to 18 and worse than the base's by the second pass. Addresses (e).
4. **Train on the rung itself.** Put below-band conjectures in the training set: the only adapters above the
   base on untrained below-band conjectures are the ones that did. This is the ladder's own logic turned
   around: a rung is lifted by training on it, not on the one above.
5. **Keep the variety.** Train on every distinct verified proof of a conjecture, not one. Addresses (b).

All five need more below-band conjectures than 46 to read; the 4,609 generated conjectures never sampled are
where they are.

## What this does and does not show

- **Shown:** at one seed, one pass on the 50% band lowered the fresh pass rate on the untrained band below it
  (−0.034, interval just below zero) and did not change how many never-proved statements get proved.
- **Shown:** at least 43% of the never-proved conjectures, 28% of the 1,500 kept, are false. A challenger
  rewarded for conjectures the solver fails would be rewarded for these.
- **Shown (9 conjectures, 13 adapters):** the loss on rarely-proved conjectures is common to the native-format
  adapters and absent or smaller in the legacy ones; training on below-band conjectures is what raised them.
- **Not shown:** that learning cannot build from one rung to the next. One seed, 46 conjectures, one pass, one
  selection, and a training target (the shortest proof) that pushes against long proofs.
- **Not shown:** which of brevity, the dose and the loss of variety is the cause. They are confounded in every
  adapter trained so far; levers 1 to 5 are how to separate them.

## Consequences

- No further seed of `ladder_half_native` is queued: the branch is STOP, and the diagnosis says the next run
  should change the round (levers 1 to 4), not repeat it.
- Every pass-rate figure on self-generated conjectures has a ceiling set by the false ones: the conjecture
  holdout holds at least 81 false statements of 296. The census belongs in the pipeline before selection.
- The primary needs more conjectures to be read at this effect size: the interval's upper end is −0.0036.
