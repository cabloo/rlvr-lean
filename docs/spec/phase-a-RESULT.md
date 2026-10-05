# rlvr_lean Phase A — one round of RLVR on self-generated Lean conjectures

**One round of self-generated conjectures → verified proofs → QLoRA fine-tuning makes DeepSeek-Prover-V1.5-SFT
more reliable, and that carries to held-out problems: pass@1 rises on the conjecture holdout (+1.8 points),
on held-out Lean Workbook problems (+1.0) and on miniF2F-test (+1.0), all three seeds agreeing. It does not
reach new problems: miniF2F-test pass@32, the pre-registered transfer label, is unchanged (46.3% → 46.3%).**

> **CORRECTION, 2026-10-04. The two held-out loss rows below do not measure learning about proofs.**
> The pipeline's training targets leave out the sequence-start token that the base model writes between the
> prompt and the proof (its first choice at that position in 113 of 114 held-out proofs). That one position
> costs the base model 11.96 nats per proof and under 0.04 after training, and it is 97% to 101% of the fall
> in every adapter. With the token put back, the base model's held-out loss is 0.245, not 0.941, and one round
> lowers it by 0.02 to 0.03. The same token is what the learning-progress score ranked (99.8% of its
> reference gradient). The pass@1 and pass@32 rows are Lean-checked measurements and stand as numbers;
> whether the pass@1 gains survive the fix has not been measured, since every adapter here also learned to
> stop writing the token. Diagnosis and probes: the diagnosis ledger of the held-out loss and
> the score (not part of this copy).

- **Question (owner, 2026-10-02):** does this style of RLVR process improve the model, and does that translate
  to a holdout? Baselines (random and difficulty-heuristic selection) are Phase B, deferred.
- **Spec:** the first experiment's spec. Milestones 1 to 3 have result notes of their own
  (not part of this copy).
- **Runs:** tasks `04831307` (`rlvr_lean_m4/full_seed0`) and `22bd4130` (`rlvr_lean_m4/full_seeds012`) on
  the GPU box (RTX 5080), 2026-10-03; Lean verification on the Lean server host's Kimina server.

## Results (pre-registered Phase A rules, spec §8)

Paired over problems, base model vs trained adapter; a problem's adapter score is the mean of its three
seeds' estimates; pass@k by the unbiased estimator of Chen et al. (2021); 95% percentile bootstrap over
problems, 10,000 resamples. **Improved** = interval entirely above zero; **no measurable change** = it
contains zero.

| Measure | Problems | Base | Adapter | Difference | 95% interval | Label | Per seed (0 / 1 / 2) |
|---|---|---|---|---|---|---|---|
| **Improvement** — conjecture holdout pass@1 | 296 | 0.296 | 0.313 | +0.018 | [+0.008, +0.028] | **improved** | +0.016 / +0.020 / +0.016 |
| **Transfer** — miniF2F-test pass@32 | 244 | 0.463 | 0.463 | 0.000 | [−0.016, +0.016] | **no measurable change** | 0.000 / −0.008 / +0.008 |
| Workbook holdout pass@1 | 1,013 | 0.074 | 0.085 | +0.010 | [+0.006, +0.015] | improved | +0.0099 / +0.0095 / +0.0109 |
| miniF2F-test pass@1 | 244 | 0.364 | 0.374 | +0.010 | [+0.002, +0.020] | improved | +0.008 / +0.010 / +0.013 |
| Held-out loss, training base (reduction) | 114 | 0.941 | 0.222 | +0.719 | [+0.593, +0.856] | improved | all three |
| Held-out loss, serving base (reduction) | 114 | 0.964 | 0.214 | +0.751 | [+0.616, +0.896] | improved | all three |

**Pre-registered labels: Improvement = improved. Transfer = no measurable change** (miniF2F-test pass@32).
The supporting endpoints say which way the transfer points: pass@1 improves on both held-out sets.

**Reading.** Higher pass@1 with unchanged pass@32 means the adapter samples correct proofs more often for
problems the base model could already solve, without solving new ones: sharpening, not expansion. The
relative gains are +6% (conjecture holdout), +14% (workbook holdout) and +3% (miniF2F-test pass@1).

**Superseded by the correction above:** the pattern this paragraph describes (the shorter the proof, the
larger the fall) is what a constant cost per proof, divided by the proof's length, looks like. The fall is
11.9 nats per proof whatever its length.

**The held-out loss overstates the gain.** It fell fourfold, but that is mostly style: 12 of the 114 held-out
proofs are one-word tactics that also appear among the training proofs (`rfl`, `nlinarith`, `ring`), and
their loss fell by 2.1; proofs of 30 tokens or fewer fell by 1.06, longer ones by 0.24. It is not leakage:
only 1 of 114 held-out statements is close to a training conjecture (similarity 0.92; the next is 0.85). The
gain does survive the move from the 4-bit training base to the FP8 serving base (0.72 vs 0.75), so the
quantization mismatch does not eat it.

**Sanity anchor.** The base model's miniF2F-test pass@32 here is 46.3%. DeepSeek report 48.2% at 32
samples, with chain-of-thought prompting, the informal problem statement and a 300 s limit; this run uses
no CoT, no informal statement, FP8 weights and a 60 s limit.

## What was trained on

| Stage | Count |
|---|---|
| Lean Workbook problems (one formalization each) | 89,219 → 2,000 seed statements, 2,000 reward statements (6,157 of 6,420 candidates compile under the pin) |
| Raw generated conjectures (2,000 prompts × 4) | 8,000 |
| Unparseable / forbidden token / duplicate text | 136 / 3 / 336 |
| Do not compile / duplicate elaborated type / vacuous | 1,175 / 197 / 44 |
| Survived (76%) → kept | 6,109 → 1,500 (296 in the conjecture holdout) |
| Solvable (≥ 1 of 12 proofs verify) | 526 of 1,500 (35%); 413 outside the holdout form the pool |
| Pool pass-rate histogram (verified of 12: count) | 1: 27, 2: 10, 3: 20, 4: 14, 5: 18, 6: 14, 7: 21, 8: 24, 9: 25, 10: 31, 11: 50, **12: 159** |
| Training set | N = ⌊413 / 3⌋ = **137** conjectures, one canonical (shortest) proof each, selected by the cosine learning-progress score; 54 optimizer steps (3 epochs, effective batch 8) per seed, ~47 s each |

38% of the pool is trivial for the base model (all 12 proofs verify). How each selection rule (computed for all
four, trained only for one) chose from it:

| Method | Mean pass rate | Trivial (12/12) | In the band (0, 0.25] | Median proof tokens |
|---|---|---|---|---|
| Random | 0.78 | 60 | 17 | 16 |
| Difficulty heuristic | 0.36 | 0 | 57 | 25 |
| Learning progress, dot product | 0.89 | 78 | 1 | 7 |
| **Learning progress, cosine (trained)** | 0.78 | 50 | 11 | 17 |

- **The dot-product score is a proof-length detector.** Spearman with proof length was −0.98 over the 413
  pool conjectures (−0.93 in the smoke test), which is why Phase A trained on the cosine variant (decision
  recorded in the spec, assumption A1).
- **The cosine score's set looks like a random draw** by difficulty and length. So this result measures
  fine-tuning on self-generated, verified conjecture proofs, not the value of the selection score. Whether
  learning progress beats random or the heuristic is exactly Phase B.
- **Score diagnostics:** stability under a second random projection, Spearman 0.999. Dot product against the
  "frontier" reward gradient (band statements only), 0.999. Dot product against cosine, 0.21. Selected-set
  overlaps are low (Jaccard 0.06–0.21), so the four rules really do pick different data.

## Throughput and run times

| | Value |
|---|---|
| Sampling (vLLM 0.30.0, FP8, RTX 5080) | 1,400–2,700 generated tokens/s depending on the batch (about 2,100 over the 34,000-proof sampling stage) |
| Generated tokens per proof | 46 (conjectures), 99 (workbook) on average |
| Verification (Kimina, 16 workers on the shared Lean server host) | 7.6–7.9 proofs/s while sampling; 11.8 on miniF2F |
| Verification statuses, proof sampling (34,000) | 5,931 verified, 27,884 Lean error, 99 timeout, 65 server error (0.19%), 21 rejected by the token filter |
| Run 1 (seed 0, everything) | 2 h 21 min: data 1 min, conjectures 9 min, proof sampling 72 min, scoring 2 min, training 1 min, evaluation 55 min |
| Run 2 (seeds 1 and 2) | 1 h 27 min: training 2 min, evaluation 86 min |
| Peak card memory | 15,685 MiB, against the 15.9 GB the job queue measures for the card (vLLM reserves 0.85 of it; the desktop's display holds about 1,400 MiB) |

## What went wrong, and what it cost

| Problem | Effect | Fix |
|---|---|---|
| transformers 5.x loads this model's tokenizer as a sentencepiece-style tokenizer that drops every space and newline | Milestone 2 run 3 passed every step on garbled prompts (3.9% pass@1) | Generic fast tokenizer plus a start-token template, checked against the model's own tokenizer file before any use |
| The FP8 exporter quantized the token embedding | vLLM refused the checkpoint | Embedding excluded; the export checks its own output |
| vLLM's FlashInfer sampler compiles on first use; the worker image has no CUDA compiler | Sampling crashed | PyTorch sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`) |
| Theorem names were read from comments in the pin gate | 14 false failures out of 489 | Comment-free, line-anchored parsing |
| Four snippets per verification request | One slow proof idled three workers; the gate took twice as long | One snippet per request |
| Summed resident memory | Read 33 GB on a 30 GB box | Proportional set size |
| The dot-product score tracks proof length | Its selection is the 137 shortest, mostly trivial, proofs | Trained on the cosine variant |
| 65 server errors and 99 timeouts in 34,000 proof checks | Counted as failures for both base and adapter alike | Within the 1% limit |
| miniF2F copy | Six statements are reported defective (DeepSeek-Prover-V1.5 issue #26: five big-operator binder escapes and `mathd_algebra_314`) | Kept for base and adapter alike, so they cannot bias the paired difference |
| Lean server hardening (outbound block, 16-CPU cap) | Written, not yet applied | Needs the owner's installer re-run |

## What this says about the larger project

- **The loop works mechanically and it does improve the solver**, on held-out problems too, in one round, from
  137 examples. That is the precondition for a challenger/solver loop.
- **It sharpens rather than expands** (pass@1 up, pass@32 flat), the usual pattern for a round of
  verifier-filtered self-training. Expanding coverage is where a challenger has to earn its keep: by
  generating problems just past what the solver can reach.
- **The conjecturer wastes much of its output on trivial statements**: 38% of solvable conjectures are 12/12.
  A difficulty filter removes them; the dot-product learning-progress score prefers them. The cosine score
  is indifferent to them. Phase B (learning progress against random and the heuristic) is the direct test of
  whether learning progress picks better data, and its pieces are already computed for this pool.

## Reproduce

Two tasks on the GPU box, one after the other, each given the Lean server's key:

    python -m rlvr_lean.runner.entry --stage phase_a --profile full --seeds 0 --kimina-api-key <key> --out <dir>
    python -m rlvr_lean.runner.entry --stage phase_a --profile full --seeds 0,1,2 --kimina-api-key <key> --out <dir>

Both run the stage `phase_a` at the profile `full`; every stage resumes from the GPU
box's store, and every artifact is mirrored to `experiments/rlvr_lean_m4/<task>/steps/`.
