# Phase B result: the learning-progress score selects no better than a random draw (three seeds)

> **CORRECTION, 2026-10-04: the verdict on the score is INCONCLUSIVE (execution), not a refutation.**
> The numbers below stand. Their reading does not. A diagnosis
> (the diagnosis ledger of the held-out loss and the score, three GPU probes) found a defect in the
> pipeline: the training targets leave out the sequence-start token the base model writes between the prompt
> and the proof (its first choice there in 113 of 114 held-out proofs).
> - **The held-out loss figures are that token.** In all twelve adapters 97% to 101% of the fall from 0.941
>   comes from the first target position (11.96 nats per proof at the base, under 0.04 after training). With
>   the token put back the base's held-out loss is 0.245, and one round lowers it by 0.02 to 0.03.
> - **The score measured that token.** It carries 99.8% of the reference gradient and a median 93% of each
>   example's. The ranking is repeatable (0.987 across two random projections) and is about how typical an
>   example's first-token gradient is, not about its proof. So this experiment compared a token-typicality
>   ranking with a random draw; it did not test selection by learning progress. The sentence below beginning
>   "This refutes this score" is withdrawn, and so is the "scoring check" as evidence about the idea.
> - **Still standing:** the pass@1 and pass@32 measurements (Lean checked those proofs) and the comparison of
>   the two difficulty bands with random. **Not known:** whether the pass@1 gains survive the fix, since every
>   adapter here also learned to stop writing the token.

Spec: the first experiment's spec section 13 (arms, seeds, the rule pre-registered 2026-10-03 before
any Phase B result). Read on 2026-10-03 at three seeds, the ceiling.

**Verdict.** On the deciding measure, training on the 137 examples the learning-progress score picks is
**not measurably better than training on 137 random ones**. It is better, by small margins, than both
difficulty-band selections. The score also fails its own check: it lowers held-out loss no more than random
selection does. Every arm, random included, improves pass@1 over the base model, and no arm moves pass@32.

This refutes **this score** (cosine of B-only LoRA gradients) as a way to choose training examples. It does not
refute choosing by learning progress in general: an effect smaller than about ±0.0023 on this measure would
not have been seen.

## What ran

Four arms, each N = 137 self-generated conjecture proofs, Phase A's hyperparameters, seeds 0, 1, 2. The
learning-progress arm is Phase A's own adapters and evaluations, reused. Tasks: `phase_b_seed0_r2` (982eefa7),
then seeds 1–2 across `phase_b_seeds012` (efbe391e, cancelled part-way) and `phase_b_seeds012_r4` (0bd4e6bd).
Report: `experiments/rlvr_lean/phase_b_seeds012_r4/steps/report.json`.

| Arm | Mean base pass rate of its examples | Mean proof length (tokens) |
|---|---|---|
| `learning_progress_cosine` | 0.777 | 18.3 |
| `random` | 0.779 | 19.6 |
| `difficulty_heuristic` (band ≤ 0.25, filled upward) | 0.358 | 33.3 |
| `half_pass_rate` (closest to 0.5) | 0.526 | 23.5 |

## 1. The deciding measure: workbook-holdout pass@1, learning progress minus each arm

1,013 problems, paired over problems, 3 seeds. Learning progress scores 0.0845.

| Other arm | Its pass@1 | Difference | 95% interval | Label |
|---|---|---|---|---|
| `random` | 0.0826 | +0.0019 | [−0.0004, +0.0042] | no measurable difference |
| `difficulty_heuristic` | 0.0803 | +0.0042 | [+0.0014, +0.0070] | LP better |
| `half_pass_rate` | 0.0822 | +0.0023 | [+0.0004, +0.0043] | LP better |

The `half_pass_rate` label is marginal: see "Conditions" for a bias of up to 0.0002 against that arm, against
a lower bound of +0.0004.

## 2. The prompt's rule: miniF2F-test pass@32, learning progress minus the heuristic

−0.0014 [−0.0109, +0.0082], **inconclusive**, as section 13 predicted: pass@32 did not move for any arm.

## 3. The scoring check: held-out-loss reduction, learning progress against random

| Stack | LP reduction | Random reduction | Difference | Label |
|---|---|---|---|---|
| NF4 (training) | 0.7186 | 0.7139 | +0.0047 [−0.0034, +0.0130] | no measurable difference |
| FP8 (serving) | 0.7505 | 0.7481 | +0.0024 [−0.0058, +0.0112] | no measurable difference |

Section 13 says what this means: *"If LP does not lower it more, the scoring is not doing what it claims."* It
does not. Two more facts say the same thing from the selection's side: its examples have the same difficulty
profile as random's (table above), and it shares examples with the random selection at exactly the rate two
random draws would (Jaccard 0.197 measured; 0.199 expected for two draws of 137 from the pool of 413).

## 4. Every arm against the base model

| Measure (base) | LP | Random | Heuristic | 50% band |
|---|---|---|---|---|
| Workbook holdout pass@1 (0.0744) | +0.0101 improved | +0.0082 improved | +0.0059 improved | +0.0078 improved |
| Conjecture holdout pass@1 (0.2956) | +0.0175 improved | +0.0115 improved | +0.0171 improved | +0.0185 improved |
| miniF2F-test pass@1 (0.3636) | +0.0104 improved | +0.0058 no change | +0.0105 improved | +0.0092 improved |
| miniF2F-test pass@32 (0.4631) | −0.0000 no change | +0.0014 no change | +0.0014 no change | +0.0027 no change |

No arm read degraded at any seed, so none was stopped. Phase A's finding holds for every selection: one round
sharpens (pass@1 up) and does not expand (pass@32 flat).

The other comparisons of section 13 item 4 (learning progress minus each arm on conjecture-holdout pass@1 and
miniF2F-test pass@1) all read no measurable difference. The heuristic arm lowers held-out loss less than
learning progress does (NF4 +0.0195 [+0.0091, +0.0308], FP8 +0.0150 [+0.0052, +0.0249]).

## Proof variety (not part of the pre-registered report)

Distinct proof texts as a share of attempts on the conjecture holdout, 12 samples per statement, computed from
the six evaluations whose files are on the dev machine: random 54.8%; heuristic 51.4% and 54.4%; 50% band 48.8%,
48.9%, 49.5%. The base model's figure is 73%, and the learning-progress arm's was the lowest at 45% (both
measured at seed 0). Every selection narrows what the model writes.

## Conditions that differed between evaluations

The comparison is paired over problems, not over time. Three of the twelve arm-seed evaluations
(`difficulty_heuristic` seed 2, `half_pass_rate` seeds 1 and 2) ran after an incident and differ in these ways:

- **A smaller sampling cache.** The first run of seeds 1–2 filled the GPU box's VRAM and was cancelled.
  The re-run leaves 3 GB for the desktop, so the sampling engine's cache was 2.4 GiB instead of 4.1. The model,
  sampling settings and per-request seeds are the same; fewer sequences sample at once.
- **The FP8 held-out loss is scored in groups** and before the samples instead of after (a 114-prompt call
  overflowed the smaller cache and crashed the engine once). The loss is a deterministic forward pass.
- **Lean checks ran slightly slower.** Timeouts were 0.45% of attempts in all three, against 0.40–0.43% in the
  three seed-0 evaluations; median check time 0.56–0.71 s against 0.47–0.59 s. Other work shared the machine.
  If every extra timeout had been a proof that would otherwise verify, workbook pass@1 would be about 0.0003
  lower in such an evaluation: at most 0.0001 on the heuristic arm's three-seed mean and 0.0002 on the 50%
  band's. That cannot change the `random` or the heuristic label; it is half the margin of the 50% band's.

The evaluation files of `random` seeds 1–2 and `difficulty_heuristic` seed 1 are on the GPU box's store only
(their task was cancelled before its files were copied), so their timeout rates and variety are not in the
figures above. Their results are in the report.

## What this does and does not show

- **Shown:** at one round and N = 137, this learning-progress score is indistinguishable from random in what it
  picks and in what training on its picks does. The bed can tell selections apart when they differ: the
  heuristic arm is separated from learning progress by +0.0042 on the same measure.
- **Shown:** choosing harder examples (pass rate ≤ 0.25, or near 0.5) does not help pass@1 on any of the three
  sets after one round, and transfers slightly worse to the workbook holdout than easy or random examples.
- **Not shown:** that learning-progress selection cannot work. The estimator may be the weak part (Phase A
  already found the dot-product version to be a proof-length detector, Spearman −0.98), and a single round on a
  pass@1 measure rewards reinforcing what the model already does.
- **Not shown:** anything about expanding what the model can prove. No arm moved pass@32.

## Consequences

- Phase B's code (the extra arms, the 50% band selection, the per-arm pipeline) did not produce a winner and
  stays on its branch; this write-up is what lands.
- Phase A's pipeline on the main line trains on the learning-progress selection. This result says a random selection
  would do as well without the gradient pass over the pool; changing that default belongs to the loop's spec.
- For the multi-round loop: random selection is the baseline to beat, and the measure to move is pass@32 or a
  solved-problem count, which one round of any selection did not move.
