# Ladder loop, L0 (the data): a pool of 55,631 problems with a published answer, a goal set of 392, three fillable rungs

Spec: `docs/spec/ladder-loop.spec.md`, milestone L0. Nothing was trained.
Run 2026-10-04 at Lean v4.27 through the `oeis` pool. Tasks: `rlvr_lean/ladder_l0b_r1` (both attempt
steps; failed at a GPU guard afterwards) and `rlvr_lean/ladder_l0b_r2` (the report). Store:
`experiments/rlvr_lean/ladder_l0`; exported sets: `src/rlvr_lean/data/ladder_l0`, `src/rlvr_lean/data/ladder_l1`
(derived data, not shipped in this copy: `ladder_pool export` and `ladder_round export` write them).

**Verdict.** Every gate L1 depends on is met: each held-out rung holds at least 137 problems (166 / 155 / 391),
the goal set is 392 problems, and the pool holds problems in the band for the base model (6.0% of a random
draw). L1 seed 0 was queued the same day.

## What L0 had to decide

| Question | Answer |
|---|---|
| How many published certificates survive v4.27 | 5,004 of 6,500 sampled candidates (77%) after both passes; 6,883 of 10,290 slice candidates after the first pass alone, 7,875 after the renaming pass |
| The size of G | **392** of 1,000 held-out Lean Workbook problems unresolved by the base in 32 episodes (387 known true, 5 known false) |
| How much of the pool sits in the band for the base | **239 of 4,000** random pool problems (6.0%); Lean Workbook 11.2%, STP conjectures 1.7% |
| Whether each rung has at least 137 problems | Yes: below the band 166, in it 155, above it 391 |

The band is the pass rates the reward scores at least 0.8 at t = 1/4: 0.127 to 0.409. In 32 placing episodes
that is 1 to 4 successes below, 5 to 13 in, 14 or more above.

## The pool

Sources pinned by revision and SHA-256: `internlm/Lean-Workbook` (the `proof` field and `wkbk_1009.parquet`),
`Goedel-LM/Lean-workbook-proofs`, `kfdong/STP_Lean_0320`. 88,961 candidates: 37,126 Lean Workbook known true,
1,835 known false (published disproofs with no hypotheses, their exact negation checked), 50,000 STP
conjectures (the first 50,000 of 696,211 in seeded order).

| | |
|---|---|
| Candidates verified, both passes over all 88,961 | **67,857** (76.3%): 59,478 by a certificate as published, 8,379 more by a renamed one |
| Not verified | 21,025 with no certificate that verifies, 79 whose negation Lean did not confirm as exact |
| Verified with a rewritten statement (`∑ x in s`) | 5,963 |
| Dropped as the same problem as a held-out or set-aside one | 1,360 |
| Dropped as copies within the pool | 8,866 |
| **Pool** | **55,631**: 18,207 Lean Workbook known true, 1,556 known false, 35,868 STP conjectures |

L1 seed 0 drew its 20,000 candidates when the pool held 43,069 (the bulk check was still running); the pool
reached 55,631 at 21:17 UTC, after 159,101 Lean checks in all.

**The slice** (spec: "The held-out sets do not wait for the whole pool"), through both passes:

| Part | Verified |
|---|---|
| The present holdout's candidates | 243 of 309 |
| Lean Workbook sample | 1,041 of 1,300 |
| STP sample | 1,834 of 2,500 |
| All candidates | 5,004 of 6,500 |

**The renaming pass.** The first pass left 3,400 slice problems unsettled; 2,822 had a certificate that names
a library name Mathlib has since renamed; 992 verified by a renamed certificate. The three most frequent
names were `le_div_iff` (962 problems), `div_le_div_iff` (944) and `div_le_iff` (659); a problem can carry
more than one. On a 489-proof sample before the pass, STP conjectures verified at 44 of 80; after both passes
the STP sample verified at 73%.

**H** (drawn once, seed 20261004): 1,000 Lean Workbook problems (243 from the present holdout; 943 known true,
57 known false) and 1,000 STP conjectures. **Set aside:** the 693 holdout statements with no published answer
(692 compile at v4.27). **Base-map sample:** 4,000 (2,197 STP, 1,690 Lean Workbook known true, 113 known false).

## The base model on H (32 episodes each) and on the base map (8 each)

An episode is one native-format attempt at the side the certificate does not rule out (owner, 2026-10-04); a
seeded 2% of problems kept both sides (40 of H, 58 of the base map) and raised no alarm.

| | H | Base map |
|---|---|---|
| Problems | 2,000 | 4,000 |
| Resolved at least once | 712 | 1,141 |
| Never resolved | 1,288 (392 Lean Workbook, 896 STP) | 2,859 |
| Attempts | 65,280 | 32,464 |
| Verified attempts | 11,414 | 5,328 (4,914 at the statement, 414 at the negation: 7.8%) |
| Lean timeouts (30 s) | 189 | 71 |
| Capped at 1,024 tokens | 66 | 52 |
| No answer | 2 | 2 |

- **STP conjectures are far harder for the base than Lean Workbook problems:** 104 of 1,000 held-out STP
  conjectures were resolved at least once, against 608 of 1,000 Lean Workbook problems. In the band: 1.7%
  against 11.2%.
- **Base map by k of 8:** 2,859 / 196 / 140 / 99 / 94 / 95 / 136 / 160 / 221 for k = 0 to 8. A random draw's
  mean reward is 0.121: the control arm of L1 can expect a proof from about 285 of its 1,000 problems.

## Where the time went

| Step | Wall time | Of which generation |
|---|---|---|
| Held-out attempts | 69 min | 23 min (4.38 M tokens) |
| Base-map attempts | 49 min | 16 min (2.18 M tokens) |

The attempt step waited on Lean about two thirds of the time, with the pool between 62% and 88% busy: a block
was two chunks, and each block waited for its own checks before the next began. L1 runs with the rolling
pipeline (`e6fd5ce96`) and a pool of 37 workers (it was 28 for this run, 11 at its start of day).

## What went wrong, and what was changed

- **The first run failed after both attempt steps were done.** The stage's last GPU guard asked the whole card
  to be back at its starting level; the desktop held 485 MiB more by then, against a 300 MiB margin. The guard
  now also passes when memory has fallen 6,000 MiB below the step's own peak (`07fced232`).
- **A failed task delivers no step files, and its rerun skipped the finished steps without delivering them.**
  A finished attempt set now mirrors its per-problem results again (`07fced232`).

## Departures from the spec

- **Lean checks:** about 104,000 for the GPU half (6,000 exactness, about 98,000 attempts), against the 198,000
  the both-sides episode would have cost; the spec's "281,000 for L0" was written before the slice and before
  the ruled-out side was dropped.
- **G is 392, not the "about 560" estimated:** the base resolves more held-out Lean Workbook problems in 32
  one-sided episodes (608 of 1,000) than the estimate assumed.
- **H and the base-map sample were drawn from the slice, before the whole pool was checked** (the spec's slice
  rule). The whole pool's two passes finished afterwards and only grew the pool; H and the sample are as drawn.

## Not measured

- Whether the base's 32-episode zero is stable: G was placed once, with one sampling seed. L1 measures the
  base afresh on G (its luck).
- The 958 published disproofs with hypotheses, and the 15,370 candidates no certificate settles: they are out
  of the pool, not judged.
- Near-copies of G's problems among STP's conjectures (spec, Known limits): the duplicate check removes exact
  and same-shape copies only.
