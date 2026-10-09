# Ladder loop, L4 (the loop from a model pretrained on published proofs, seed 0): NOT SHOWN ON TOP OF PRETRAINING. Six rounds from the pretrained model make it more reliable on what it could already do (+17% successes per attempt on the goal set, +11 points on the middle rung) and narrower (fewer distinct goal problems, 227 against 238; less varied attempts); on the 210 goal problems it could not solve the gain is +0.0012 [−0.0022, +0.0058]. The pretraining itself: one pass over 24,866 published proofs gives a model that solves 238 of the 392 goal problems and 94 reliably, and three times the ceiling's proofs buy no more reach

Everything here is **pretrained on published proofs**: distillation of other provers, then the loop. It is the
arm the owner chose on 2026-10-08 beside the base arm (`ladder-l3d2-RESULT.md`), so that what the loop adds on
top of pretraining is itself measured.

Spec: `docs/spec/ladder-loop.spec.md`, "L4: the loop from a model pretrained on published proofs" and its
"Made exact by the build". Why: `ladder-ceiling-RESULT.md`. Run 2026-10-08 and
10-09 (UTC) at Lean v4.27 through the `oeis` pool, on the GPU box, from 428c600e6: tasks
`rlvr_lean/ladder_l4_pretrain_seed0_r1` (the training, the rungs and the first goal sampling; it then FAILED on
its own safeguard, see "The fault") and `rlvr_lean/ladder_l4_pretrain_seed0_r2` (76 minutes: the second goal
sampling checked again, the map sampled again, the report). One seed, a first run by the seed rule.

## Part 1: the pretrained model `pre`

**What was run.** From the base, ONE pass over `data/ladder_l4/pretraining.jsonl` (other people's published
proofs: not shipped in this copy) in the file's order: 24,866
rows (8,155 Lean Workbook, 16,711 STP), each a problem of the pool's `pretrain` half with the certificate the
pool build verified for it (median 7 lines); 3,109 optimizer steps, 112 minutes; mean loss 0.246 over the first
tenth of the rows, 0.190 over the last. No held-out problem, no base-map problem and no problem of the `loop`
half is in the file. The adapter is kept: it is the arm's starting model. It was measured as the ceiling's
models were (8 episodes on the three rungs; G twice, 32 and 61 one-shot attempts with the stored sampling
seeds), so it pairs by problem with the stored base, the three-round loop model and both ceiling models.

**Both checks pass.** `pre` solves 238 of the 392 goal problems in its 93 attempts (150 asked). G′, the goal
problems it does not solve in its first 32 attempts, holds 210 (80 asked): 17 of 1 line, 47 of 2-3, 84 of 4-7,
62 of 8 or more.

### The goal set (392 problems the base failed 32 times in a row at placing), 93 one-shot attempts each

Successes per 1,000 attempts, by the length of the shortest published proof:

| | Base | Loop, three rounds | Ceiling, 2,000 proofs | Ceiling, 8,000 proofs | **`pre`, 24,866 proofs** |
|---|---|---|---|---|---|
| 1 line (37) | 9.9 | 23.0 | 82.5 | 97.1 | 109.3 |
| 2-3 lines (125) | 4.7 | 11.4 | 62.3 | 81.4 | 91.6 |
| 4-7 lines (156) | 3.5 | 3.4 | 37.8 | 64.6 | 61.2 |
| 8 lines or more (74) | 0.9 | 0.4 | 7.6 | 20.5 | 14.8 |
| 4 lines or more (230) | 2.62 | 2.43 | 28.1 | 50.4 | 46.3 |
| **All of G (392)** | 3.98 | 7.21 | 44.1 | 64.7 | **66.7** |
| Problems solved in 93 attempts | 59 | 59 | 209 | 245 | **238** |
| Solved in at least one episode of 8 / reliably (half of 11 episodes) | 57 / 3 | 59 / 15 | 208 / 70 | 242 / 106 | **234 / 94** |
| The same, 4 lines or more | 22 / 1 | 21 / 2 | 102 / 25 | 125 / 50 | 111 / 39 |

`pre` minus the base on all of G: +0.0627 [+0.0521, +0.0739] successes per attempt; solved 238 to 59 (gained
183, lost 4). The rungs, `pre` minus the base: below the band +0.200 [+0.159, +0.240], in it +0.146 [+0.098,
+0.192], above it +0.050 [+0.027, +0.073] (the 8,000-proof ceiling: +0.145, +0.138, −0.001).

- **Three times the proofs, no more reach.** 66.7 per 1,000 against the 8,000-proof ceiling's 64.7; 238
  problems against 245; 94 reliable against 106; on proofs of 8 lines or more it is behind (14.8 against 20.5).
  The two files are not one file at two sizes: the ceiling's is 8,000 problems the base was predicted to fail,
  with up to their three shortest proofs; this one is every true problem of half the pool with one certificate
  each, easy problems included. Read as: more published proofs of this kind do not move the goal set further
  with this model and recipe; they do lift the easy rung (above the band +0.050, where the ceiling did nothing).
  It is not a clean dose curve.
- **More of its attempts time out in Lean:** 3.5% and 3.9% of its goal attempts against 0.15% to 0.25% for the
  base and the loop model. A timeout is a failure here, so its reach is if anything under-measured; it is also
  what feeds the pool's process leak (below).

### What the arm starts from: a model with a middle, whose own proofs are long at its limit

`pre`'s own map (the base map's 4,000 pool problems, 8 attempts each, the stored sampling seed), the map the
arm's challenger starts from:

| k of 8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | mean pass rate |
|---|---|---|---|---|---|---|---|---|---|---|
| The base's stored map | 2,859 | 196 | 140 | 99 | 94 | 95 | 136 | 160 | 221 | 0.167 |
| `pre` | 2,088 | 491 | 307 | 192 | 166 | 115 | 136 | 200 | 305 | 0.237 |

At the reward's target (k = 1 to 3) the base had 435 of 4,000 problems; `pre` has 990. And the proofs it
verifies there are long (read after the run from the map's stored attempts; lines of a verified proof):

| The problem's k of 8 | Problems | Verified proofs | Median lines | 4 lines or more | 8 lines or more |
|---|---|---|---|---|---|
| 1 | 491 | 491 | 8 | 80% | 52% |
| 2-3 | 499 | 1,190 | 4 | 59% | 28% |
| 4-5 | 281 | 1,239 | 3 | 34% | 11% |
| 6-8 | 641 | 4,656 | 2 | 13% | 2% |

This is what the base arm never had: its own proofs were short at every k, and once minimised so were the
assembled ones (`ladder-l3d2-RESULT.md`). A round from `pre` aimed at k = 1 to 3 trains on proofs of 4 to 8
lines that the model wrote itself.

### Is it a leak, or learning that only carries to near copies? (read after the run, no Lean)

For each goal problem, the nearest statement in the pretraining file (the statement without its name,
character 5-gram overlap). No goal statement is in the file. Near twins are: 34 goal problems have a
pretraining statement at 0.8 or over (the same inequality with two terms swapped, or an STP variant such as
`P ↔ True` of the same `P`), 8 at 0.9 or over.

| Goal problems whose nearest pretraining statement overlaps | Problems | Base, per 1,000 | Ceiling 8,000 | `pre` | `pre` minus base | In G′ |
|---|---|---|---|---|---|---|
| under 0.3 | 57 | 6.8 | 39.4 | 46.4 | +0.0396 [+0.0177, +0.0651] | 41 |
| 0.3 to 0.5 | 127 | 3.7 | 68.2 | 67.5 | +0.0638 [+0.0445, +0.0862] | 67 |
| 0.5 to 0.7 | 134 | 3.9 | 78.5 | 79.5 | +0.0756 [+0.0590, +0.0940] | 58 |
| 0.7 and over | 74 | 2.3 | 53.2 | 57.7 | +0.0554 [+0.0344, +0.0799] | 44 |
| 4 lines or more, under 0.5 | 108 | 2.1 | 39.1 | 39.9 | +0.0378 [+0.0233, +0.0540] | 76 |

The gain holds where nothing near was trained on, and the problems with a near twin are not where it is
largest: not a copy. It is smaller on the least similar problems, as the ceiling's was. The arm's read will be
given by the same distance, to the pretraining file and to the rounds' own training statements together.

### The fault of the first task (infrastructure, not the model)

`ladder_l4_pretrain_seed0_r1` trained `pre` and measured the rungs and the first goal sampling, then failed on
the stage's own rule: Lean gave no verdict on 6.3% of the second goal sampling and of the map (2% allowed).
Two of the pool's four Lean servers were dead at their process limit: each container leaks processes (the
server kills a Lean process group and waits only for its direct child, so the orphan stays a zombie in the
container's process count), and at 4,096 it answers every check with HTTP 500 while still reporting healthy.
The servers were restarted; the second task kept the adapter and the clean measurements, checked the unanswered
set again (0 without an answer) and sampled the map again (9 of 32,464). The first goal sampling kept 134
attempts without an answer (1.05%, under the rule's 2%); they are failures in every count, and they only
choose G′: the arm's primary is read on the second sampling. A durable fix for the leak is being built
separately.

## Part 2: the arm (six rounds from `pre` on the `loop` half, assembly in the round, a twin)

Task `rlvr_lean/ladder_l4_seed0_r1`, from 3817cfc75, 2026-10-09 04:26 to 09:05 UTC (279 minutes: six
rounds and their trainings 170, the twin 12, the two measurements 49 and 35). It was recorded as done at 09:26:
the job runner's result volume was full from about 08:20 and it took no write until room was made; the
run itself was not touched, and every file it wrote was pulled afterwards.

**What was run.** The arm `t010_assembly_pre`: target rate 1/10, six rounds of 1,000 problems drawn from the
`loop` half alone (26,029 candidates; no problem whose published proof the pretraining saw), the challenger
started from `pre`'s own map, round 1 attempted by `pre`, the Lean-only assembly after every batch, every model
trained FROM `pre` (one more pass over one proof a problem of the rounds so far). `with` is M(6): `pre` and
3,509 rows of the rounds (3,422 one-shot proofs, 87 assembled). `without` is its twin on the 3,422 alone.
Both measured as `pre` was, with `pre`'s sampling seeds.

**The five checks pass.** The pretraining took (238 goal problems); G′ holds 210; the two measured trainings
ran (all but 5 and 3 of each adapter's 37,478,400 numbers differ from `pre`'s; loss 0.180 over the first tenth
of the rows and 0.165 over the last, for both); every model still writes proofs (no rung attempt of `with`
without an answer); no row of the seven trainings (15,967 rows) is a problem of the `pretrain` half or a
held-out problem. The start adapter was loaded: the B matrices' sum read back is the file's, M(1)'s first-step
loss is 0.132, and `with` stands where a model built on `pre` stands and not where the base arm's did.

### The primary: not shown on top of pretraining

**On G′ (the 210 goal problems `pre` does not solve in its first 32 attempts), successes per attempt over the
61 attempts of the second sampling, `with` minus `pre`: +0.00125 [−0.00219, +0.00578]** (8.74 per 1,000
against 7.49; 112 successes against 96 in 12,810 attempts each). The interval holds zero: **NOT SHOWN ON TOP OF
PRETRAINING at this size.** The run resolves about ±0.004; the base arm's own gain on its own goal set drawn
again, read beside it from stored rows, is +0.0066 [+0.0031, +0.0108] (7.76 against 1.20 per 1,000) and would
have been seen here.

| G′ by the shortest published proof | Problems | `pre`, per 1,000 | `with` | `with` minus `pre` | Solved in the 61: `pre`, `with` (gained, lost) |
|---|---|---|---|---|---|
| 1 line | 17 | 12.5 | 13.5 | +0.0010 [−0.0058, +0.0068] | 6, 5 (1, 2) |
| 2-3 lines | 47 | 12.9 | 19.9 | +0.0070 [−0.0056, +0.0262] | 19, 13 (6, 12) |
| 4-7 lines | 84 | 6.4 | 6.6 | +0.0002 [−0.0031, +0.0037] | 22, 20 (11, 13) |
| 8 lines or more | 62 | 3.4 | 1.9 | −0.0016 [−0.0042, +0.0011] | 9, 5 (4, 8) |
| **All of G′** | 210 | 7.49 | 8.74 | **+0.0012 [−0.0022, +0.0058]** | 56, 43 (22, 35) |

More successes on fewer problems: `with` solves 43 of the 210 in its 61 attempts where `pre` solves 56.

### What the rounds did do: more reliable, and narrower

| | Base | `pre` | `without` | `with` |
|---|---|---|---|---|
| All of G (392), successes per 1,000 over 93 attempts | 3.98 | 66.7 | 78.1 | 77.8 |
| 1 line (37) | 9.9 | 109.3 | 138.9 | 136.9 |
| 2-3 lines (125) | 4.7 | 91.6 | 109.8 | 106.2 |
| 4-7 lines (156) | 3.5 | 61.2 | 66.5 | 68.6 |
| 8 lines or more (74) | 0.9 | 14.8 | 18.5 | 19.9 |
| Goal problems solved in 93 attempts | 59 | 238 | 216 | 227 |
| Solved in at least one episode of 8 / reliably | 57 / 3 | 234 / 94 | 210 / 98 | 220 / 98 |
| The same, 4 lines or more (230) | 22 / 1 | 111 / 39 | 99 / 38 | 104 / 37 |
| Rung below the band, pass rate minus the base's | | +0.200 | +0.213 | +0.216 |
| Rung in the band | | +0.146 | +0.269 | +0.255 |
| Rung above the band | | +0.050 | +0.096 | +0.098 |
| Share of attempts that are distinct, on the rungs / on G | 0.946 / 0.813 | 0.870 / 0.817 | 0.781 / 0.702 | 0.784 / 0.713 |
| Proofs verified on G: median lines, share of 8 or more | 5, 27% | 4, 23% | 4, 12% | 4, 14% |

- **More reliable.** `with` minus `pre` on all of G: +0.0111 successes per attempt [+0.0027, +0.0197], 17% more;
  on the rung in the band +0.109 [+0.069, +0.149], above it +0.048 [+0.030, +0.065], below it +0.017 [−0.013,
  +0.047]. By length the gain is clear at 1 to 3 lines and not at 4 or more (+0.0066 [−0.0024, +0.0172]).
- **Narrower.** Fewer distinct goal problems: 227 against `pre`'s 238 in 93 attempts (gained 28, lost 39, sign
  test p = 0.22); 220 against 234 solved in at least one episode of 8. The share of distinct attempts falls from
  0.82 to 0.71 on G and from 0.87 to 0.78 on the rungs, and the share of its verified proofs with 8 lines or
  more from 23% to 14%. The reliable count barely moves: 98 against 94 (37 against 39 at 4 lines or more).
  This is the dose curve's finding again (L1b: a second pass helped the middle and lost goal problems), now on
  a model that could write long proofs: one more pass over its own proofs concentrates it.
- **Never solved by anything stored here** (81 of the 392 on 2026-10-09: 5, 10, 27 and 39 by length): `with`
  solves 4 in its 93 attempts (two of 2-3 lines, one of 4-7, one of 8 or more) and `without` 3 (two of 2-3
  lines, one of 8 or more), five problems between them, each ONCE in 93 attempts. `pre`, counted the same way before it was added to the ledger, had
  solved 15 that nothing before it had. Reached once is not solved reliably.

### Why: what the rounds trained on

| Round | Picks: k = 0, k = 1-3, k of 4 or more (of 8) | Picks' mean pass rate | Training rows (one-shot, assembled) | Their lines: median, 4 or more, 8 or more |
|---|---|---|---|---|
| 1 | 387, 406, 207 | 0.230 | 625 (613, 12) | 4, 55%, 17% |
| 2 | 411, 350, 239 | 0.245 | 607 (589, 18) | 3, 38%, 7% |
| 3 | 410, 340, 250 | 0.256 | 598 (590, 8) | 3, 37%, 7% |
| 4 | 454, 336, 210 | 0.232 | 559 (546, 13) | 4, 52%, 13% |
| 5 | 450, 286, 264 | 0.265 | 571 (550, 21) | 3, 47%, 11% |
| 6 | 466, 297, 237 | 0.242 | 549 (534, 15) | 3, 42%, 10% |

- **The picks were easier than the target.** The reward's target is a pass rate of 1/10; the picks' mean pass
  rate was 0.23 to 0.27 in every round, as it was in the base arm (0.19 to 0.35): four in ten picks were not
  solved at all, a quarter were solved by half the attempts or more, and a third fell where the reward peaks.
- **So the training rows were not the long proofs `pre` writes at its limit.** Of the 3,422 one-shot rows 45%
  have 4 lines or more and 11% have 8 or more (the base arm: 38% and 4%); `pre`'s own proofs on problems it
  solves once in 8 have a median of 8 lines (Part 1), and those are a minority of what a round keeps: one
  proof for every solved problem, most of them problems it solves often.
- **A quarter of round 6's rows are refutations** (25.7%, from 2 to 3% in rounds 1 to 4 and 8% in round 5): the
  challenger found that known-false problems sit near its target, as it did in L2.
- **Assembled proofs are longer here and still do not teach.** The 87 the rounds assembled have a median of 5
  lines once minimised (69% of 4 or more; the base arm's: 3 lines). `with` minus `without`: on G′ −0.0012
  [−0.0026, +0.0001]; on the goal problems of 4 lines or more +0.0019 [−0.0005, +0.0044]; distinct goal
  problems solved 227 against 216 (gained 22, lost 11, p = 0.08). Nothing here is clear of zero.

### The overfit read (after the run, no Lean): the gain is not carried by near copies

For each problem of G′, the nearest statement the arm was trained on (the rounds' 3,509 training statements,
and the pretraining file's 24,866; character 5-gram overlap). No problem of G′ has a rounds' statement at 0.8
or over (3 at 0.7 or over). `with` minus `pre` on the primary, by the nearest statement in either:

| Nearest trained-on statement overlaps | Problems | `pre`, per 1,000 | `with` | `with` minus `pre` |
|---|---|---|---|---|
| under 0.3 | 36 | 2.7 | 0.9 | −0.0018 [−0.0046, +0.0009] |
| 0.3 to 0.5 | 67 | 7.6 | 11.5 | +0.0039 [−0.0042, +0.0171] |
| 0.5 to 0.7 | 60 | 10.7 | 11.8 | +0.0011 [−0.0038, +0.0063] |
| 0.7 and over | 47 | 7.0 | 7.0 | +0.0000 [−0.0049, +0.0042] |

No band is clear of zero and the nearest band is not where the (unshown) gain sits. What does show is the
other end: on the problems least like anything trained on, the round-six model does no better and solves fewer
(2 against 5).

### What it says, and what it does not

- **The loop, as built, turns a pretrained model's occasional successes into frequent ones and does not take
  it to problems it could not solve.** That is the base arm's result one level up: there the rounds moved
  short-proof problems only; here they move every length the model already reaches, and G′ not at all.
- **It costs breadth.** The owner's condition was "without overfitting": by the count of distinct problems
  and of distinct attempts, six passes over its own proofs overfit the model to its own style. Reliability
  bought that way is not the reliability asked for.
- **One seed.** By the spec's branch this is "not shown at this size", and no further seed of this arm is
  bought as it stands: the point estimate is a sixth of `pre`'s rate and the breadth lost is the larger fact.
- **It does not say the loop cannot add.** Two causes that this run cannot tell from "the mechanism is
  insufficient" are named and tested next: (1) every model here is one rank-16 adapter, and `pre`'s own
  training loss was flat for the last two fifths of its pass: an adapter that is full can only learn the
  rounds' proofs by giving something up, which is what narrowing is (the check L4r, queued after this run:
  the same pretraining at rank 64); (2) the picks' pass rate was 2.4 times the target in both arms, so the
  rounds have never trained mostly on the proofs at the model's limit.
- **The counts with the Lean-only assembly at inference** are in the addendum below.

## Addendum, 2026-10-09: with the Lean-only assembly at inference (replays of the stored attempts, no generation)

Each model's stored goal attempts cut into episodes of 8 in the order drawn, with the episode's Lean-only part
added (`tools/ladder_goal_assembly`); goal problems solved in at least one episode / in a quarter / in half
(reliably) / in nine tenths of their episodes, by attempts alone and then with assembly.

| | Episodes a problem | Solved at least once | In a quarter | Reliably (half) | Nine tenths |
|---|---|---|---|---|---|
| `with` (M(6)), both samplings | 11 | 220 → 229 | 153 → 165 | 98 → 106 | 48 → 48 |
| `without` (the twin), both samplings | 11 | 210 → 222 | 147 → 156 | 98 → 108 | 51 → 53 |
| `pre`, the second sampling alone | 7 | 212 → 223 | 156 → 170 | 99 → 104 | 33 → 36 |
| `with`, the second sampling alone | 7 | 193 → 205 | 147 → 160 | 97 → 108 | 43 → 45 |
| `without`, the second sampling alone | 7 | 198 → 209 | 145 → 161 | 92 → 97 | 42 → 44 |

`pre`'s first-sampling attempts were not pulled from the box (the task that made them failed on the pool
fault, and the re-run did not rewrite them), so `pre` is read on its 61 attempts alone, and the arm's two
models are read both ways; only the last three rows are like for like. (For scale, with both samplings: the
8,000-proof ceiling model 242 → 262 and 106 → 119; the base arm's six-round model 72 → 92 and 23 → 26.)

- **The same picture with assembly as without it.** Like for like, `pre` solves more distinct goal problems
  (223 against 205 and 209) and the round-six model about as many reliably (108 against 104; the twin 97), and
  wins nine tenths of its episodes on more (45 against 36): what the rounds add is at the reliable end.
- **Assembly adds about the same to each:** 9 to 12 problems solved at least once and 5 to 11 solved
  reliably, on a pretrained model as on the rounds' models. It is still worth keeping in an episode, and it
  does not change which model is broader.
- Where the proof needs 4 lines or more (230 problems, the second sampling alone, with assembly): solved at
  least once 108 (`pre`), 97 (`with`), 100 (`without`); reliably 45, 46 and 36.
