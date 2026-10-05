# Native-format result (13a): the pass@1 gain survives the format fix (three seeds)

Spec: the first experiment's spec section 6 item 2a (the fix) and section 13a (the scout, its primary
and its branches, fixed 2026-10-04 before the run). Read on 2026-10-04: seed 0 as the pre-registered scout,
then seeds 1 and 2 as the escalation its branch named.

**Verdict.** With the training pairs in the model's native format, one round on Phase A's own 137 examples
raises workbook-holdout pass@1 by **+0.0099 [+0.0057, +0.0143] at three seeds**, against **+0.0101
[+0.0056, +0.0148]** for Phase A's stored adapters at the same seeds. The difference between the two formats is
−0.0002 [−0.0033, +0.0030]: no measurable change. Phase A's pass@1 gain was not the habit of dropping the
sequence-start token. The native format does not add to the gain either, and it does not move pass@32.

## What ran

Arm `native_same_picks`: Phase A's 137 examples (its learning-progress selection), Phase A's hyperparameters
(54 steps = 3 passes, batch 8, lr 1e-4), seeds 0, 1, 2, every (prompt, proof) pair encoded as prompt,
sequence-start token (context), proof. Nothing else differs from Phase A's adapters, whose stored evaluations
are the control and were not re-run; the base model's stored samples are shared. Tasks: `native_scout_seed0`
(19d42458, seed 0), then part 2 of `ladder_13b` (74bbbae5, seeds 1 and 2). Code: commits 93c3f6f2c and
f2b3c76af. Reports: `experiments/rlvr_lean/native_scout_seed0/steps/native_scout_report.jsonl` and
`experiments/rlvr_lean/ladder_13b/steps/native_scout_seeds_report.jsonl`.

## 1. The primary: workbook-holdout pass@1, adapter minus base (1,013 problems, paired)

| Read | Base | Native | Difference | 95% interval | Legacy control | Native minus legacy |
|---|---|---|---|---|---|---|
| Seed 0 (the scout) | 0.0744 | 0.0829 | +0.0085 | [+0.0042, +0.0130] | 0.0843 (+0.0099), reproduced | −0.0014 [−0.0052, +0.0025] |
| Seeds 0, 1, 2 | 0.0744 | 0.0843 | +0.0099 | [+0.0057, +0.0143] | 0.0845 (+0.0101 [+0.0056, +0.0148]) | −0.0002 [−0.0033, +0.0030] |

Scout branch: **ESCALATE** (difference ≥ 0; the stored control read the pre-registered 0.0744 and 0.0843).
Per seed, native minus base: +0.0085, +0.0101, +0.0110; legacy minus base: +0.0099, +0.0095, +0.0109.

## 2. Also read, three seeds, each adapter against base

| Measure | Native | Legacy | Native minus legacy |
|---|---|---|---|
| Conjecture-holdout pass@1 | +0.0129 [+0.0033, +0.0223] improved | +0.0175 [+0.0078, +0.0277] improved | −0.0046 [−0.0134, +0.0041] |
| miniF2F-test pass@1 | +0.0102 [−0.0011, +0.0226] no measurable change | +0.0104 [+0.0015, +0.0202] improved | −0.0002 [−0.0072, +0.0067] |
| miniF2F-test pass@32 | −0.0041 [−0.0178, +0.0082] no measurable change | 0.0000 [−0.0164, +0.0164] no measurable change | −0.0041 [−0.0164, +0.0068] |

What the adapters write, per seed (seeds 0, 1, 2; the base beside them):

| | Base | Legacy | Native |
|---|---|---|---|
| Conjecture holdout: solved at least once (of 296) | 113 | 111, 110, 111 | 106, 108, 105 |
| Workbook holdout: solved at least once (of 1,013) | 114 | 120, 117, 115 | 114, 118, 115 |
| miniF2F-test: solved at least once (of 244) | 113 | 113, 111, 115 | 111, 113, 112 |
| Proof length, conjecture holdout (mean tokens) | 43.7 | 31.6, 29.3, 32.1 | 27.4, 28.8, 27.9 |
| Proof length, workbook holdout | 96.8 | 85.7, 84.5, 84.9 | 72.7, 70.9, 67.8 |
| Distinct attempts, conjecture holdout | 0.730 | 0.453, 0.455, 0.452 | 0.387, 0.392, 0.362 |
| Attempts that start with the sequence-start token | 99.98% | 0.01% | 98.6%, 98.9%, 98.5% |

The native adapters write shorter and less varied proofs than the legacy ones and solve fewer conjecture-holdout
statements at least once (106, 108, 105 against 111, 110, 111; the base 113).

## 3. What the round teaches, read by part of the target

The held-out loss (114 proofs never trained on) by part, during training (seed 0; seeds 1 and 2 have the same
shape: 0.2454 → 0.2107 and 0.2089 at step 18 → 0.2285 and 0.2330 at step 54):

| Step | Mean per-token loss | Body, nats per token | Newline that ends the proof, nats | Fence, nats | First token, nats |
|---|---|---|---|---|---|
| 0 (base) | 0.2454 | 0.1905 | 0.168 | 0.041 | 0.0003 |
| 6 | 0.2260 | 0.1878 | 0.122 | 0.025 | 0.0003 |
| 12 | 0.2078 | 0.1881 | 0.080 | 0.013 | 0.0003 |
| 18 (one pass) | 0.2092 | 0.1876 | 0.092 | 0.015 | 0.0003 |
| 36 (two passes) | 0.2096 | 0.1937 | 0.075 | 0.012 | 0.0003 |
| 54 (three passes) | 0.2272 | 0.2187 | 0.044 | 0.007 | 0.0002 |

- The loss is lowest at step 12 and the fall is the ENDING: the newline that closes the proof and the fence.
  The proof body does not improve (0.1905 → 0.1872 at best) and is worse than the base's from the second
  pass on. The round teaches the model to stop sooner, then memorises its 137 examples (training loss
  0.2495 → 0.0057 at seed 0).
- Evaluation read, loss reduction against the base in the same format, per seed: NF4 0.0232, 0.0213, 0.0162
  (each "no measurable change"; base 0.2474); FP8 0.0297, 0.0256, 0.0239 (each "improved"; base 0.2382).
  Legacy, for scale: NF4 0.9410 → 0.2205, of which 97% to 101% was the missing token.
- On the same batches the native round fits its examples harder than the legacy one did (last training loss
  0.0057 against 0.0183 at seed 0; third-pass mean 0.027 against 0.049), and its held-out body loss rises more
  (+0.023 against +0.005, evaluation read, seed 0). Cause not tested.

## 4. Conditions

Lean checks of the three native evaluations: timeouts 0.385%, 0.442%, 0.411%; server errors 0.221%, 0.170%,
0.257% (legacy seed 0: 0.416% and 0.288%). During seed 0's evaluation an installer ran on the Lean server's
machine from about 04:00 UTC: the workbook checks before and after that time read 0.342% and 0.561% timeouts
against 0.407% and 0.612% for the legacy evaluation on the same statements, and the median check was faster
than the legacy one throughout (times reconstructed from chunk order; the verifier stores none). An unjudged
check counts as a failure, so the conditions could only understate the native figure.

## What this does and does not show

- **Shown:** the pass@1 gain of one round on these 137 examples is the same with and without the format
  defect, on all three sets. Phase A's central claim stands on its pass@1 measures.
- **Shown:** the gain is sharpening. It comes with shorter proofs and less variety, the held-out proof body
  does not improve, and pass@32 does not move (native −0.0041, inside ±0.016).
- **Not shown:** that a round teaches anything about harder problems. `ladder-first-rung-RESULT.md` reads the
  opposite on rarely-proved conjectures, for every native adapter.
- **Not shown:** anything about selection by learning progress. The repaired score has not been compared
  with a random draw (section 13a: "the next comparison").
- **A defect left in:** the adapter is never trained to write the sequence-start token (it is context in
  training) and skips it in 1.1% to 1.5% of attempts, almost all on the workbook holdout; the base writes it in
  99.98%.

## Consequences

- The native format is the pipeline's format from here on (`encode_pair`, `ArmNames.target_format`); legacy
  arms and files keep their names and their figures.
- Three passes at lr 1e-4 overfit in this format: the held-out loss is lowest at step 12 and the body is worse
  than the base's by the second pass. Phase A's settings were chosen for a round whose first steps went to one
  token; they are not known to suit the native format.
