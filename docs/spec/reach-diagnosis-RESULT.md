# Why the loop has not reached new problems, read from stored results: the problems it cannot solve need longer proofs, each attempt throws its verified steps away, and pooling those steps solves problems that 558 sampled attempts did not

A diagnosis, 2026-10-06, after the owner's question: "how do we push this further? ideally to reliably
accomplishing tasks the model couldn't accomplish before". No GPU and no generation: stored results of
`ladder-l2-RESULT.md`, `ladder-l2t-RESULT.md` and `ladder-l3a2-RESULT.md`, the published proofs of the held-out
problems, and about 14,000 Lean checks through the pool at background priority. It closes with a proposal;
nothing here is a verdict fixed before a run.

## 1. What separates the problems the model never solves: the length of the proof

The shortest published proof of each held-out problem (the fewest lines of any of its published proofs), by
group (a group is the base model's own pass rate):

| Group | Problems | Median lines | One line | At most 3 lines | Nine tenths within |
|---|---|---|---|---|---|
| Above the band (base solves 77%) | 391 | 1 | 72% | 96% | 2 lines |
| In the band (27%) | 155 | 2 | 33% | 77% | 5 lines |
| Below the band (7%) | 166 | 3 | 18% | 59% | 8 lines |
| Goal set (never in 32 attempts) | 392 | 4 | 9% | 41% | 10 lines |

The goal set is not another kind of mathematics. Its proofs are a few intermediate facts and a closing step:
median 4 lines, nine tenths within 10.

(Corrected 2026-10-08: the first version of this table and the next took the published proof with the fewest
characters and counted its lines. They now use the fewest lines of any published proof, which is what the
shipped `data/ladder_l0/heldout_proof_lines.jsonl` and L3c's report use. The shape is the same and the numbers
moved a little; the goal problems with a proof of 4 lines or more are 230, not 248.)

## 2. Training helps where a short proof exists and not where a long one is needed

The goal set by its shortest published proof; successes per 1,000 attempts over 279 attempts a problem (three
seeds of 93), the base against the model after three rounds at t = 1/10:

| Shortest published proof | Problems | Base ever solved | Base per 1,000 | Trained per 1,000 | Ratio |
|---|---|---|---|---|---|
| 1 line | 37 | 15 | 9.5 | 22.9 | 2.4 |
| 2 to 3 lines | 125 | 34 | 4.7 | 10.6 | 2.3 |
| 4 to 7 lines | 156 | 29 | 2.9 | 3.9 | 1.4 |
| 8 lines or more | 74 | 4 | 0.4 | 0.3 | 0.7 |

This is "reliability, not reach" with its cause attached. A proof written in one shot is verified all or
nothing, so its chance falls with every step it needs; training on the model's own proofs raises the chance
of a step it already takes and does not supply the steps it never takes.

## 3. The pieces are there and are thrown away

On the hard problems (the goal set and the below-band rung), 6,436 failed first attempts of the second repair
check have a known first error. **In 3,740 of them (58%) the error is on the last line: every step before it
verified.** The next attempt starts from nothing.

## 4. Why resuming from one failed attempt was not the answer

The first repair step against a fresh attempt on the same episodes, by the shortest published proof (second
repair check, base model):

| Shortest published proof | Problems | Repair step verifies | Fresh attempt verifies | Within 5 attempts: alternate against blind |
|---|---|---|---|---|
| 1 line | 67 | 4.1% | 2.8% | 13.4% against 12.9% |
| 2 to 3 lines | 193 | 3.8% | 2.4% | 11.7% against 10.9% |
| 4 to 7 lines | 203 | 1.9% | 1.7% | 7.1% against 7.0% |
| 8 or more | 95 | 1.2% | 1.1% | 4.4% against 3.9% |

Repair's edge is on the short-proof problems. Where more steps are needed, resuming keeps one attempt's
intermediate facts, and those were that attempt's plan, which had not worked.

## 5. The probe: pool the verified lemmas of all of a problem's failed attempts, then close

From the stored failed fresh attempts of each hard problem (attempts 1 and 3 of each of its 12 episodes, up to
24), mechanically:

- **a run:** an attempt's leading top-level `have` steps that end before its first error, with the names they
  bind made unique to the attempt. 2,823 runs on 411 of the 558 problems; checked alone with `sorry`, 2,821
  stand (so the extraction is sound);
- **a pool:** whole runs one after another, at most 12 `have` blocks (410 of 411 stand together);
- **closers:** six plain closing tactics, and the closing steps the model wrote in its failed attempts (up to 8
  a problem);
- **checks:** the pool with each closer, and each closer with no lemma at all.

| | Problems |
|---|---|
| Hard problems | 558 |
| Solved by some whole attempt in that run (12 episodes of up to 5 attempts, two arms) | 203 |
| Unsolved by every whole attempt in the run | 355 |
| Closed with pooled lemmas | 32 |
| Closed only with the lemmas (no closer does it alone) | 30 |
| **Of those, unsolved by every whole attempt in the run** | **15** |
| Of those, in the goal set | 14 |
| Of those, never solved by the base in 279 attempts | 10 |
| Of those, never solved by the base nor by the models after three rounds (558 attempts) | 9 |

- By the shortest published proof, of the problems unsolved in the run: 1 of 31 (1 line), 8 of 109 (2 to 3),
  4 of 139 (4 to 7), 2 of 76 (8 or more: published proofs of 8 and 10 lines).
- The assembled proofs are 11 to 22 lines.
- Of the 15, 10 close with a closing step the model wrote, 5 with a plain tactic.
- **Ablation, the 30 closed only with lemmas:** in 21 the lemmas of ONE attempt with a closer from the list
  suffice (one attempt had the facts, another the closing step); in 9 lemmas of SEVERAL attempts are needed
  together. Of the 15 new ones, 8 and 7.

No model wrote anything for this. It is the model's own verified work, kept instead of discarded, and Lean.

## What this says and does not say

- It says the premise holds in its crudest form: partial results that verified, kept across attempts, make
  proofs that sampled attempts did not, including of problems no model here ever solved.
- It does not say how large the effect is for an episode that generates. The probe used up to 24 attempts'
  pieces a problem and recombined them blindly; an episode has fewer attempts and can show the model the pool.
- It is one run's stored attempts, held-out problems, base model. It is not a comparison at equal cost and
  not a verdict: the run of 12 episodes had about a hundred whole attempts a problem, the probe none.

## Proposal

1. **An episode that accumulates** (to be specified with its read fixed first, `ladder-loop.spec.md`): the
   attempts on a problem share a pool of verified lemmas; a generation either starts fresh or continues from
   the pool, shown with Lean's state; failed closing steps are tried again when the pool grows. Measured
   against blind attempts at equal generations on the hard problems, by running total and by the length of
   the published proof, base model, no training.
2. **Then train on what it assembles:** the long verified proofs are the training examples the loop has never
   had. Reach first from the search, reliability after from training on it.
3. **A ceiling, with the owner's leave** (published proofs are certificates, not training text): one
   fine-tune on published proofs of pool problems the base cannot solve, labelled a diagnostic, measured on the
   goal set by proof length. It says whether this model can write the longer proofs in one shot when shown
   them, which bounds what any version of step 2 can reach.

From here on the goal set is reported by the length of its shortest published proof; the 4 to 7 and 8 to 15
line groups are the measure of reach.

## Files

Scripts of this reading were run from the session's scratch directory and are not kept as tools: the tables
are reproducible from the run directories named in the three result notes and from
`experiments/rlvr_lean/ladder_l0/steps/candidates.jsonl` (the published proofs).
