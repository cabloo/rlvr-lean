# Ladder loop, the ceiling (a labelled diagnostic, seed 0): the model can learn longer proofs from examples, by a wide margin; trained once on 8,000 published proofs of problems the base cannot solve, it solves 245 of the 392 goal problems to the base's 59 and solves 106 of them reliably, against 3 for the base and 15 for the loop's own best model

Spec: `docs/spec/ladder-loop.spec.md`, "The ceiling: a labelled diagnostic", approved by the owner
2026-10-08. Run 2026-10-08 (UTC) at Lean v4.27 through the `oeis` pool, on the
GPU box: tasks `rlvr_lean/ladder_ceiling_smoke_r1` (3 minutes) and
`rlvr_lean/ladder_ceiling_seed0_r1` (140 minutes: 38 of training, 55 and 46 of measurement), from 898e646f1.

**This is an exception and not a change of rule.** Published proofs are certificates and not training text.
This trained once on them, as a diagnostic. Both adapters were deleted by the report step; no model trained
this way was used in a round or kept; the training file is not exported. Every number here is a ceiling.

**Verdict by the read fixed before the run: THE MODEL CAN LEARN LONGER PROOFS FROM EXAMPLES.**

- **The three checks pass.** The training took (mean loss 0.265 over the first 500 rows, 0.176 over the last
  500). The model still writes proofs (none of its 5,832 rung attempts without an answer). It is not broken
  on easy problems (0.768 on the above-band rung against the base's 0.769).
- **Primary (the 230 goal problems whose shortest published proof is 4 lines or more; successes per attempt
  over 93 one-shot attempts; the 8,000-proof model minus the base): +0.0478 [+0.0380, +0.0584].** 50.4
  successes per 1,000 attempts against 2.62 (1,078 against 56 in 21,390 attempts): 19 times the base.
- **The reference, fixed before the run:** the loop's own gain there is +0.00064 over three seeds; twice it
  is +0.00128. The primary is 75 times the loop's own gain. This seed's three-round model beside it: −0.0002
  [−0.0014, +0.0011].
- **One seed, by the rule:** the interval is nowhere near zero, so no further seed is called for.

## The goal set, 93 one-shot attempts a problem (seed 0)

Successes per 1,000 attempts, by the shortest published proof:

| Shortest published proof | Problems | Base | Three rounds of the loop (t = 1/10) | 2,000 published proofs | 8,000 published proofs |
|---|---|---|---|---|---|
| 1 line | 37 | 9.9 | 23.0 | 82.5 | 97.1 |
| 2 to 3 lines | 125 | 4.7 | 11.4 | 62.3 | 81.4 |
| 4 to 7 lines | 156 | 3.5 | 3.4 | 37.8 | 64.6 |
| 8 lines or more | 74 | 0.9 | 0.4 | 7.6 | 20.5 |
| All of G | 392 | 4.0 | 7.2 | 44.1 | 64.7 |

Goal problems solved at least once in 93 attempts:

| | Solved | Against the base: gained / lost | 4 lines or more (230) | 8 lines or more (74) |
|---|---|---|---|---|
| The base | 59 | | 22 | 2 |
| Three rounds of the loop (t = 1/10) | 59 | 20 / 20 | 21 | 2 |
| 2,000 published proofs | 209 | 154 / 4 | 103 | 15 |
| 8,000 published proofs | 245 | 187 / 1 | 126 | 25 |

The loop's own training doubles the rate where a short proof exists and does nothing at 4 lines or more.
Training on published proofs of problems the base cannot solve raises the rate at every length, most where
the loop's training gave nothing: 19 times at 4 to 7 lines, 24 times at 8 or more.

## The held-out rungs (8 episodes a problem), pass rate

| Rung | Base | Three rounds of the loop | 2,000 published proofs | 8,000 published proofs |
|---|---|---|---|---|
| Below the band (166) | 0.064 | 0.095 | 0.179 | 0.209 |
| In the band (155) | 0.268 | 0.363 | 0.358 | 0.406 |
| Above the band (391) | 0.769 | 0.836 | 0.746 | 0.768 |

The two kinds of training are complementary. The loop's own proofs raise the easy rung (+0.068) and the
published proofs of hard problems do not (−0.001 [−0.022, +0.020]); on the hard rung it is the reverse
(+0.031 against +0.145).

## Is it a leak? (read after the run; not part of the fixed read)

A gain this large is first a suspect. The training file holds no held-out problem (the prepare step refuses
one), but both datasets hold variants of one another's problems, so each goal problem was set against the
8,000 training statements (the statement without its name; shared character 5-grams over all of either).

- **No goal problem's statement is in the training file.** The nearest statement's overlap: median 0.47,
  nine tenths under 0.73; 21 goal problems have a near variant (0.8 or more: a term reordered, an inequality
  turned, a factor added).
- **The gain does not rest on the near ones.** The primary again on the goal problems of 4 lines or more whose
  nearest training statement is under a given overlap:

| Nearest training statement | Problems | Base per 1,000 | 8,000 proofs per 1,000 | Difference |
|---|---|---|---|---|
| Any | 230 | 2.6 | 50.4 | +0.0478 [+0.0379, +0.0583] |
| Under 0.8 | 212 | 2.6 | 51.1 | +0.0484 [+0.0382, +0.0595] |
| Under 0.6 | 157 | 3.0 | 49.5 | +0.0465 [+0.0345, +0.0591] |
| Under 0.5 | 123 | 3.4 | 47.2 | +0.0438 [+0.0296, +0.0596] |
| Under 0.4 | 83 | 4.4 | 41.8 | +0.0374 [+0.0228, +0.0543] |
| Under 0.3 | 42 | 6.4 | 32.3 | +0.0259 [+0.0077, +0.0489] |

  It falls as the problems get less like anything trained on, and is 5 times the base at the far end. So it
  is learning that carries to problems of the same kind, strongest near what was seen, and not a copy.
- **What a timeout does here.** 2.0% of the 8,000-proof model's goal attempts and 2.9% of the 2,000-proof
  model's timed out at 30 s and count as failures. Other Lean work of this session shared the pool during
  both measurements, the first most. Any error from that is against the ceiling's models.

## In the owner's terms: problems solved reliably (read after the run)

A goal problem is solved reliably when it is resolved in at least half of its episodes of 8 one-shot
attempts (11 episodes a problem here, from the same 93 attempts in the order drawn). Seed 0:

| | Solved in at least one episode | A quarter | Half (reliably) | Nine tenths |
|---|---|---|---|---|
| The base | 57 | 20 | 3 | 0 |
| Three rounds of the loop (t = 1/10) | 59 | 25 | 15 | 1 |
| 2,000 published proofs | 208 | 120 | 70 | 26 |
| 8,000 published proofs | 242 | 174 | 106 | 39 |

- Of the 106, the base won no episode of 11 on 69, and at most one on 88: these are problems it could not do.
- By the shortest published proof, the 8,000-proof model reliably solves 14 of 37, 42 of 125, 44 of 156 and
  6 of 74 (1, 2 to 3, 4 to 7, 8 or more lines); the loop's model 5, 8, 2 and 0.
- On the 220 goal problems with no near statement in the training file (overlap under 0.5): 53 reliably,
  against 3 for the base and 9 for the loop's model.

## With assembly on top (Lean only, after the run; `tools/ladder_goal_assembly.py`)

The same stored attempts, each model's, in the same 11 episodes of 8, with the episode's Lean-only assembly
added (`ladder-l3c-RESULT.md`, addendum). Attempts alone, then with assembly:

| | Episodes resolved (of 4,312) | By an assembled proof | Solved at least once | Reliably |
|---|---|---|---|---|
| The base | 117, then 164 | 48 | 57, then 77 | 3, then 3 |
| Three rounds of the loop (t = 1/10) | 193, then 232 | 45 | 59, then 77 | 15, then 15 |
| 8,000 published proofs | 1,280, then 1,412 | 171 | 242, then 262 | 106, then 119 |

- **For the base and the loop's model, training and assembly add and do not compound:** training gives the
  reliability (3 to 15), assembly gives the reach (to 77 with either model), and neither moves the other.
- **At the ceiling model's strength the search finds more:** 171 assembled proofs across the same episodes
  against 45 to 48, and they add reliable problems too (106 to 119). What the loop can supply of its own grows
  with the model it searches with.

## What the model writes

- **More varied, not narrower.** Distinct attempts on G: 92% for both ceiling models, against the base's 81%
  and the loop's model's 75%. On the rungs 97% against 95% and 89%.
- **Longer proofs that verify, in number.** On G the base's 145 verified proofs have a median of 5 lines;
  the 8,000-proof model verifies 2,358, median 4, 63% of them 4 lines or more and 20% 8 or more (longest 25).
  The loop's model: 263, median 2, 35% of 4 lines or more.
- **The dose.** 2,000 proofs give most of it (209 goal problems solved, 245 with 8,000). Per proof trained on,
  the first 2,000 are worth more than three times the next 6,000 on the primary (+0.0127 against +0.0037 per
  1,000 proofs). The training loss was still falling at the end (0.194 at rows 1,500 to 2,000, 0.176 at the last
  500).

## What this says and does not say

- **It says what limits the loop is what it trains on, not the model.** The same model, recipe and one pass,
  with about as many examples (2,000 against the loop's 1,685): published proofs of problems it cannot solve
  take the goal set from 4.0 successes per 1,000 to 44.1; its own proofs of problems it can sometimes solve
  take it to 7.2. The loop feeds the model proofs it could already find.
- **It says the aim is reachable by training:** 106 never-solved problems reliably, and 245 at all.
- **It does not say the loop can get there on its own.** These proofs were written by other, stronger
  provers. What the loop can supply of the same kind is what its search finds beyond one-shot attempts: the
  assembled proofs of L3c, about 15 a round today (`ladder-loop.spec.md`, L3d).
- **It is one seed, one model, one family of problems** (the goal set is Lean Workbook problems; the
  training file is half Lean Workbook, half STP). The size leaves no doubt of the sign; the exact figures are
  one training's.
- **It sizes L3d's first step.** At the 2,000-proof dose a proof is worth about +0.000013 per attempt on the
  primary. If an assembled proof taught as much, a harvest of 150 would be worth about +0.002, which is at
  the edge of what one seed resolves there (about ±0.0013).

## For the owner: the decision this puts

The owner's rule is that published proofs are certificates only, with this named in advance (2026-10-04):
"we might want to pretrain on some of these later if we can't make progress, see if it helps against a
holdout." It helps against the holdout by more than everything else measured put together.

1. **Keep the rule.** The loop goes on with its own proofs, and L3d asks whether the few assembled ones can
   do some of what published ones do. Slow, and far below this ceiling on any reading so far.
2. **Use the fallback: start the loop from a model pretrained on published proofs** (all that verify on pool
   problems, never a held-out one), and ask the loop's question again from there: does it take THAT model to
   problems it cannot solve, with assembly in the round. The goal set would be re-drawn for the stronger
   model. This is distillation of other provers followed by the loop, and would be reported as such.
3. **Both, as arms.** The loop from the base and from the pretrained model, the same rounds, so that what
   the loop adds on top of pretraining is itself measured.

Recommended: 3 if the GPU time is there, else 2. L3d's first step is cheap, approved, and runs in any case.

## Files

`experiments/rlvr_lean/ladder_ceiling_seed0_r1/steps/` (the report `report_ladder_ceiling.json`, the
measurements, the loss of every row); the training file `src/rlvr_lean/data/ladder_ceiling/training.jsonl`
(other people's published proofs: not shipped in this copy) and its tool `tools/ladder_ceiling_set.py`; the stage `gpu/ladder_ceiling.py`, its report
`reporting/ladder_ceiling.py`. The two reads made after the run (the overlap with the training file, the
reliable count) were made with scripts that are not kept as tools; the reliable count is
`domain/ladder_round/reliable.py`'s.
