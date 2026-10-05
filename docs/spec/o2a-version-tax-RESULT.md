# O2a result: the pipeline runs at Lean v4.27 through the pool, and the version tax on identical proofs is small (2026-10-04)

Spec: the OEIS Open spec (not part of this copy), O2a (items 9a to 9g). The default pin is
still `v4.9`; nothing here changes a v4.9 run.

**The read (pre-registered in O2: "small" if the v4.27 pass rate is within 20%, relative, of the v4.9 rate).**
The 34,000 stored base-model attempts of Phase A were checked again at Lean v4.27 through the `oeis` pool. On
the statements that compile at both pins, v4.27 verifies **0.988 [0.974, 1.002]** of what v4.9 verifies, per
attempt: a relative loss of 1.2%, and the whole interval is above 0.8. **The tax is small.** By O2's rule a
prover with this tax needs no version-adaptation stage before O4; DeepSeek-Prover-V1.5-SFT is the only prover
measured.

Two things sit beside that number and are not in it:

- **8% of the statements do not compile at v4.27 at all**, and 98% of those for one reason: the removed
  notation `∑ x in s, f x` (now `∑ x ∈ s, f x`). The same notation takes out 63 of miniF2F's 488 statements.
- **The 1.2% is a net figure**: 219 attempts that verified at v4.9 fail at v4.27, and 151 that failed at v4.9
  verify at v4.27.

## Accepted when (item 9g)

| Criterion | Result |
|---|---|
| The existing v4.9 suite is green with the default pin | ✅ 338 passed before, 404 after (66 new), 5 skipped both times. `test_lean_pool_*` not run (another piece of work) |
| The client's by-address, by-name TLS path is tested against a real TLS server | ✅ `tests/rlvr_lean/test_kimina_client_tls.py`: a server on loopback whose certificate names ONLY a host is answered at its address when the name is given (and receives that name as SNI), and refused without the name, with another name, or under another authority. Live: every request of this run (35,310) went to the pool's LAN address over HTTPS, an address the pool's certificate does not carry, with the certificate checked against the pool's name |
| The tax report exists | ✅ a JSON report and its summary (written by `rlvr_lean.tools.version_tax`; not part of this copy); complete, 34,000 of 34,000 attempts |
| One smoke stage that only verifies has run end to end at v4.27 through the pool | ✅ from the dev container, not from the GPU box: `python -m rlvr_lean.gpu prepare_data` at profile `smoke`, config `lean.pin: v4.27`, a private store (below). ⏳ The same through the stage runner on the GPU box waits for the GPU |

## Statements that still compile (item 9d)

Every statement of the stored run was compiled at v4.27 with the pipeline's own compile check (statement +
`sorry` + the fingerprint command). All of them compiled at v4.9. Intervals: 95% Wilson.

| Set | At v4.9 | Compile at v4.27 | Survival | Fail on `∑ x in s` | Fail otherwise |
|---|---|---|---|---|---|
| Seed statements | 2,000 | **1,825** | 91.25% [89.93, 92.41] | 171 | 4 (missing names) |
| Reward statements | 2,000 | **1,833** | 91.65% [90.36, 92.78] | 166 | 1 (missing name) |
| Conjectures (not part of 9d; needed for 9e) | 1,500 | 1,397 | 93.13% [91.74, 94.31] | 99 | 4 |

`unexpected token 'in'; expected ','` is the message in 436 of the 445 failures, and exactly the statements
that contain the old binder (171, 166 and 99 by text) fail with it. Example:

    theorem lean_workbook_49459 (n : ℕ) : ∑ k in Finset.range (n+1), (n.choose k) = 2^n := by
    -- v4.27: unexpected token 'in'; expected ','

The dropped reward and conjecture statements carried 2,572 of the 34,000 attempts, 196 of them verified at
v4.9 (30 statements proved). None of their attempts verifies at v4.27 (0 of 2,572: a check of the plumbing,
since a proof cannot verify under a statement that does not parse).

## The version tax on identical texts (item 9e)

Compared: 31,375 attempts on 3,230 statements (31,428 on statements that compile at both pins, minus 51 that
v4.9 gave no answer for and 6 that v4.27 gave none for). Each attempt is the stored text, with its own
statement, the pipeline's header and the pipeline's lexical filter. Intervals: 95%, bootstrap over
statements, 10,000 resamples, paired.

| | Statements | Attempts | Per attempt, v4.9 | Per attempt, v4.27 | v4.27 / v4.9 | Proved at least once, v4.9 | v4.27 | v4.27 / v4.9 |
|---|---|---|---|---|---|---|---|---|
| **All** | 3,230 | 31,375 | 18.28% [17.02, 19.59] | 18.06% [16.80, 19.35] | **0.988 [0.974, 1.002]** | 22.45% (725) | 22.57% (729) | 1.006 [0.989, 1.023] |
| Conjectures (12 samples each) | 1,397 | 16,751 | 27.47% | 27.33% | 0.995 [0.979, 1.011] | 36.72% (513) | 37.08% (518) | 1.010 [0.991, 1.029] |
| Workbook reward statements (8 each) | 1,833 | 14,624 | 7.75% | 7.45% | 0.961 [0.929, 0.989] | 11.57% (212) | 11.51% (211) | 0.995 [0.961, 1.032] |

Verified attempts: 5,735 at v4.9, 5,667 at v4.27. Statements proved at both pins: 707; only at v4.9: 18; only
at v4.27: 22.

By the statement's base pass rate in the stored run (verified over samples drawn, at v4.9):

| Base pass rate | Statements | Attempts | Verified at v4.9 | Verified at v4.27 | v4.27 / v4.9 | Statements still proved |
|---|---|---|---|---|---|---|
| 0 (never proved) | 2,505 | 23,523 | 0 | 65 | not defined | 22 newly proved |
| (0, 0.25] | 102 | 1,076 | 176 | 201 | 1.142 [0.988, 1.322] | 97 of 102 |
| (0.25, 0.75] | 231 | 2,436 | 1,395 | 1,324 | 0.949 [0.914, 0.982] | 221 of 231 |
| (0.75, 1] | 392 | 4,340 | 4,164 | 4,077 | 0.979 [0.967, 0.989] | 389 of 392 |

The loss is in the middle and upper bands (2% to 5%); the low band and the never-proved statements gain.
No band is near the 20% line.

**If the dropped statements are counted as losses too** (not the spec's measure): 5,931 verified at v4.9 over
all 34,000 attempts, 5,667 at v4.27, a ratio of 0.956. Still small, and three quarters of that loss (196 of
264) is proofs of statements that no longer parse, not proofs that stopped working.

## What fails at v4.27

| | Unknown identifier | Other error | Timeout | Rejected before Lean |
|---|---|---|---|---|
| All attempts not verified at v4.27 (25,708) | 2,136 | 23,530 | 25 | 17 |
| **Verified at v4.9, not at v4.27 (219, on 72 statements)** | **101** | **118** | **0** | 0 |

"Unknown identifier" is any error naming something the environment does not have: `Unknown identifier`,
`Unknown constant`, `Invalid field ...: The environment does not contain ...`. The message that
`#print axioms` adds when the attempt's own theorem failed to be declared (`Unknown constant
`<its own name>``) is an echo and is not counted. `unknown tactic` is the parser's message for text it cannot
read and is an other error.

**Unknown identifier (101).** Lemmas renamed or removed between the 2024 and the 2026 Mathlib. The names:
`div_le_iff` 30, `add_left_neg` 18, `div_le_div_iff` 14, `Real.sqrt_eq_iff_sq_eq` 7, `add_eq_zero_iff` 7,
`Function.funext_iff` 5, `div_lt_div_iff` 5, `le_div_iff` 3, `le_or_lt` 3, and eight more once or twice.

    theorem conjecture_1b43f2ceffd0cc00 {X ι} (f : X → ι) : (fun x ↦ f x) = f ↔ True := by
      simp [Function.funext_iff]            -- Unknown identifier `Function.funext_iff`

    theorem conjecture_2a3a9636337f1f21 : ∀ x : ℝ, |x + 3| - |x - 1| ≤ |4 * x + 6| := by
      intro x
      cases' le_total 0 (x + 3) with h₀ h₀ <;> cases' le_total 0 (x - 1) with h₁ h₁ <;>
        cases' le_total 0 (4 * x + 6) with h₂ h₂ <;>
          simp_all only [abs_of_nonneg, abs_of_nonpos, add_left_neg, add_right_neg,
            sub_eq_add_neg, neg_add_rev] <;>
            nlinarith                       -- Unknown identifier `add_left_neg`, and `add_right_neg`

    theorem conjecture_7492eb1ccea3aa1f ... : (fun (x : ℝ) => Real.sqrt (y * z + x)) x = Real.sqrt (y * z + x) := by
      simp [Real.sqrt_eq_iff_sq_eq]         -- Unknown constant `Real.sqrt_eq_iff_sq_eq`

**Other error (118).** Tactics that became stricter or changed what they leave behind: `unsolved goals` 48,
`No goals to be solved` 24, an ambiguous name 15, `made no progress` 7, and smaller kinds.

    theorem conjecture_0633ba6fcce1ddaa : ¬(∀ x y z : ℕ, x^2 + y^2 + z^2 = 2 * (x*y + y*z + z*x) ↔ ...) := by
      intro h; have h₁ := h 1 1 1; have h₂ := h 1 1 2
      norm_num at h₁ h₂                     -- No goals to be solved (the first hypothesis already closed it)

    theorem conjecture_13a1949e9d8d5f6f (p q r : Prop) : ((p ∧ q ∧ r) ∧ (q ∧ p ∧ r)) ↔ (p ∧ q ∧ r) := by
      constructor <;> simp_all only [and_imp, and_assoc, and_left_comm] <;> tauto
                                            -- Ambiguous term and_left_comm: _root_.and_left_comm, Nat.and_left_comm

    theorem conjecture_19455868f2698a0e (x : ℝ) (hx : x ∈ Set.Icc 0 1) : ∃ f : ℝ → ℝ, ContinuousOn f Set.univ ∧ ... := by
      use fun y => x; simp [continuousOn_const, hx]     -- unsolved goals ⊢ Continuous fun y => x

The ambiguity is the prompt header's doing: it opens `Nat` and `Real`, and at v4.27 `Nat.div_pos` and
`Nat.and_left_comm` exist beside the root names. The header is part of the model's prompt and was not changed.

**Timeout (0 lost).** 25 attempts time out at v4.27 (limit 120 s); v4.9 had timed out on 20 of them, failed
5 with an error, and verified none.

**Gained (151, on 22 newly proved statements and others).** At v4.9 they failed with `linarith failed` 94,
`unsolved goals` 24, `no goals to be solved` 14, `simp made no progress` 10, and others. Typical: `field_simp`
then `nlinarith [sq_nonneg (a - b), ...]` on a three-variable inequality now closes; `exact funext_iff` now
resolves (the reverse of the loss above).

Name drift is much wider than the tax shows. Among the compared attempts, the FIRST error names something
unknown in 666 at v4.9 and in 1,837 at v4.27; but 1,099 of the 1,200 new ones are proofs that did not verify
at v4.9 either. The share of failed attempts whose first error is an unknown name, which O4 would watch, is
7.1% at v4.27 against 2.6% at v4.9.

## Checks that could have moved the read

| Check | Result |
|---|---|
| v4.9's own repeatability: proof texts sampled more than once for a statement | 2,279 texts, 7,433 attempts, **0** with differing v4.9 verdicts. The stored verdicts are not noisy |
| Attempts verified at v4.27 on a statement that does not compile at v4.27 | 0 of 2,572 |
| Verified at v4.27 only because a "failed" warning is no longer an error (the rule that differs) | 0 |
| Verified at v4.27 only after more than v4.9's 60 s (the limit that differs: 120 s) | 2; without them the ratio is 0.988 |
| Attempts with no answer at v4.27 that v4.9 had verified (a loss the report could not see) | 0 of 6 (v4.9: 2 timeouts, 4 with no answer either) |
| The pool's cache: the first 967 checks sent a second time | 967 of 967 answered from the cache, every status, message and time identical to the first answer |
| Why the 219 lost attempts fail | they fall into 18 kinds of Lean message, every one about the proof's own tactics or names; none is about the header, the statement or `#print axioms`. About 30 were read in full |
| Lexically rejected attempts | 21, the stored run's own count |

## The smoke stage at v4.27 (items 9c, 9d, 9g)

`prepare_data`, profile `smoke`, the shipped config with `lean.pin: v4.27` and nothing else changed, a private
store whose v4.9 run directory held the stored v4.9 smoke run's statements (`rlvr_lean_m3/smoke_1`); the
certificate went through the entry's own `store_ca_certificate`. 32 s.

| | |
|---|---|
| v4.9's statements compiled again | 160: **53 of 60 seed** and **85 of 100 reward** survive |
| Written | `<store>/runs-v4.27/smoke/`: statements, miniF2F files, `prepare_data.done.json` |
| v4.9's run directory | byte-identical before and after |
| miniF2F statements that compile at v4.27 | **425 of 488** (488 of 488 at v4.9). All 63 failures use `∑ x in s` (31 of the 244 test statements) |

## The pool under its first sustained load

34,325 distinct Lean files, 8 requests in flight, one file per request, Lean limit 120 s, 2026-10-04 04:37 to
05:48 UTC. The host was busy throughout (load average 18 to 33 on 32
threads).

| | |
|---|---|
| Answers | 34,319 of 34,325 |
| Rate | **9.0 checks/s** over the main pass (33,352 in 61.5 min): 20/s on statement checks, 8.5/s on proofs. Median Lean time 0.40 s, 99th percentile 4.2 s |
| Whole run | 68 minutes for the 34,000 attempts (v4.9, 16 workers, during Phase A: 7.9/s) |
| Transport failures, HTTP 503, health waits | **0** |
| HTTP 500 | 24 requests, all `{"detail":"JSON decode error"}`, all on **6 files**, each of which kills its Lean worker every time (`norm_num` on `|x 2012| = 2 ^ (Nat.fib 2012 - 1) / 2`; `ring_nf` on `(x + 2)^2023 - x^2023`). v4.9 had timed out or crashed on the same six. Left as no answer, in neither pin's counts |
| The Lean server | `UP` in all 19 status reads (before, during, at each failed request, after) and never marked DOWN in the 16 proxy logs fetched at a failed request |
| The cache backend | marked DOWN once, for under a minute, between 05:46:39 and 05:47:44 UTC, when the run was down to its last 4 requests (below). It cost no check |
| Cache | before: 0 hits, 0 misses, 40 entries. After: **967 hits, 34,343 misses, 34,338 entries** (32 MB). Hits plus misses equal the 35,310 requests sent, so none bypassed the cache. Every definitive answer was stored over TLS; the 21 timeouts and the crashes were not, as designed |

Three things for the pool's owner, not fixed here:

1. **The cache's health check timed out once**, from the proxy log:

       Server cache/cache is DOWN, reason: Layer7 timeout, check duration: 1000ms. 0 active and 1 backup
       servers left. Running on backup. 4 sessions active, 0 requeued, 0 remaining in queue.
       Server cache/cache is UP, reason: Layer7 check passed, code: 200, check duration: 0ms. 1 active and 1
       backup servers online. 0 sessions requeued, 0 total in queue.

   By their timing the four sessions were worker-killing files, each held for minutes. It has the shape of
   the Lean server's flap seen right after the install (a 2 s health timeout during a slow check). Under the
   full load of 8 checks at a time it is in none of the logs fetched.
2. **A file that kills its worker costs six worker deaths per check.** The proxy log shows each such request
   with `retries 2` and a total of 122 to 166 s: HAProxy sends it to the only server three times (40 to 58 s
   each, until the worker dies), and the client asks twice. With one server, `retry-on 500` cannot reach a
   different one. The tool now asks such a file once per run and gives up after two runs.
3. **The proxy log holds about 40 seconds at this load** (two access lines per check, `logs` capped at 500
   lines), so a flap can only be read if the log is fetched within seconds of it. A flap of the Lean server
   would also have shown in the client as 503s or connection failures, of which there were none.

## What was built

| | |
|---|---|
| The setting | `lean.pin: v4.9 \| v4.27` in `config/experiment.yaml`, with `lean.pool` (name, port, Lean timeout 120 s, 8 in flight) for v4.27. v4.9's endpoint is the `kimina` section, untouched |
| What a pin changes | `domain/verification/pin.py`: a "failed" warning is an error (v4.9 only), imports above a leading comment (v4.27 only), the pool endpoint, the run directory, whose statements it filters. A test fails if any other pipeline code names a pin |
| 9b | `KiminaClientSettings.tls_server_name` (httpx's `sni_hostname`: the name is sent for SNI and is what the certificate is checked against). `lean_settings` resolves the pool's name and connects to the address |
| The certificate to a task | as the key: `-- --kimina-api-key <pool key> --kimina-ca-certificate "$(base64 -w0 oeis_ca.crt)"` at queue time. The entry writes it to `<store>/lean-ca/<hash>.crt` and sets `RLVR_LEAN_KIMINA_CA_FILE` for every step. Never printed |
| 9c | `<store>/runs-v4.27/<profile>/`; v4.9 keeps `<store>/runs/<profile>/`. A step that selected no pin is refused |
| 9d | `prepare_data` at v4.27 compiles v4.9's seed and reward statements again and keeps the survivors, each in its set |
| 9e | `python -m rlvr_lean.tools.version_tax` (resumable, 8 in flight, no answer is never a Lean failure) and `domain/evaluation/version_tax.py` |

## Not measured

- Sampling or conjecturing at v4.27, the other provers of O2 item 8, any training (item 9f). The tax here is
  the reference prover's; its verified proofs are short (median 22 tokens) and lean on automation. A prover
  that writes long proofs full of lemma names meets the renames far more often.
- ~~A queued task at v4.27 on the GPU box.~~ **Measured 2026-10-04:** `rlvr_lean/ladder_l0b_smoke_r1` (the
  ladder loop's L0 smoke stage, on the GPU box) ran to `done` in 2 minutes: the box reached the `oeis` pool
  over TLS with the key and the authority's certificate handed at queue time, the data files travelled in
  the snapshot, the real model sampled 448 attempts and Lean verified 188 of them, with no check left
  unanswered. Not stressed by it: an attempt near the 1,024-token cap (the longest was 151 tokens) and a
  check near the 30 s limit (the slowest took 0.96 s); the fixture's statements are easy.
- The adapters' attempts and the miniF2F attempts of the stored run: only the 34,000 of `proof_attempts.jsonl`.
- Repeatability at v4.27: each distinct file was checked once.
- Comparator: this is the screen, not a claim.

## Open questions

1. **The dropped statements.** Rewriting `∑ x in s` to `∑ x ∈ s` would bring back about 8% of the workbook
   statements and 63 miniF2F statements. They would then no longer be "unchanged under both pins" (O2 item 7).
   Wanted? Until answered they are dropped, and miniF2F's 63 stay in the evaluation set at v4.27, where no
   proof of them can verify (9d names only seed and reward statements).
2. **9d on a box with no v4.9 run.** `prepare_data` at v4.27 filters v4.9's statements of the same profile and
   refuses if they are not in the store. The other reading (choose fresh candidates at v4.27) was not built.
3. **The Lean timeout at v4.27** is 120 s (the spec says "longer"; no number). It made no difference here.
4. **How a single task picks the pin.** Only the config: a v4.27 run is queued from a commit whose
   `lean.pin` is `v4.27`. The entry has no pin argument, as the spec asks for ONE setting.
5. **The header opens `Nat` and `Real`**, which costs 15 proofs to new name clashes. It is the model's own
   prompt; left alone.
