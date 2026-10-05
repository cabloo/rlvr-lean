# Ladder loop, L1b (the dose curve, three seeds): a second pass over the same proofs makes the model more reliable on problems it can already solve and narrower on the ones it cannot; the hardest rung does not separate from zero and the goal set prefers one pass

Spec: `docs/spec/ladder-loop.spec.md`, "L1b: the dose curve" and "L1b at three seeds". Run 2026-10-04 at
Lean v4.27 through the `oeis` pool (37 workers), on the
GPU box, about an hour a seed. Tasks: `rlvr_lean/ladder_l1b_seed0_r1` (and `_r2`, which added the
goal set at two passes), `ladder_l1b_seed1_r1`, `ladder_l1b_seed2_r1`.

**Verdict by the read fixed before seeds 1 and 2 ran.**

- **Primary (below-band rung, two passes minus one pass): +0.0108 [−0.0013, +0.0233]. UNDETECTABLE at three
  seeds.** By seed +0.0226, +0.0098, +0.0000. Seed 0 alone read ESCALATE; seeds 1 and 2 alone read ONE PASS
  STANDS.
- **Secondary (in-band rung, named after seeing seed 0): +0.0403 [+0.0207, +0.0613].** By the letter of the
  spec this sends L2 to two passes on the in-band evidence.
- **Guard as written (goal set, two passes against the base afresh): 41 gained, 44 lost, p = 0.83. Not
  triggered.**
- **The guard compared against the wrong thing.** The choice is between one pass and two, and against ONE
  PASS two passes gain 32 goal problems and lose 63 (p = 0.002), the same direction at every seed. This
  comparison was not pre-registered and is reported as a go-and-see.
- No seed is VOID: every one-pass checkpoint reproduces its seed's L1 result. No checkpoint is overfitting by
  the spec's rule (loss AND a rung's pass rate below its one-pass value).

**What I recommend, which departs from the letter of the branch:** keep one pass per round. The second pass
buys reliability on problems the model can already solve sometimes and pays for it on the goal set, which is
what the loop exists to move. The guard's stated purpose ("two passes cost reach: both numbers go to the owner
as a trade, and L2 does not switch by default") applies, so this is the owner's decision and the default
stays at one pass until it is made.

## What ran

Each seed's challenger arm again, from the base, on the training proofs its L1 round stored (741, 742, 740),
for three passes (279 steps at a constant learning rate after warm-up, so the two-pass checkpoint is what a
two-pass run would end at). Loss was read at 61 points on 150 of the training proofs and on 150 held-out
proofs (verified proofs the base wrote for held-out rung problems, 50 per rung; none trained on). Adapters
were saved after 0.5, 1, 1.5, 2 and 3 passes; each got 8 episodes on the 712 held-out rung problems with L1's
sampling seed, and at 1, 2 and 3 passes 32 episodes on the 392 goal problems.

## The owner's three questions

| Question | Answer at three seeds |
|---|---|
| Are we overfitting? | By loss, yes from the second pass at every seed. On the held-out rungs no pass rate falls below its one-pass value. On the goal set two and three passes solve fewer problems than one pass. |
| Did L1 measure at the peak? | For problems already solved sometimes, no: two passes add +4.0 points in the band and +2.6 above it. For the below-band rung one pass is not shown to be short. For the goal set one pass is the best dose measured. |
| Would training longer help the held-out set? | Repeating the same proofs helps what the model can already do and narrows it on what it cannot. |

## Loss (nats per proof-body token; seeds 0, 1, 2)

| After | Start | 1 pass | 2 passes | 3 passes |
|---|---|---|---|---|
| 150 of the training proofs | 0.200, 0.189, 0.207 | 0.140, 0.130, 0.149 | 0.050, 0.052, 0.062 | 0.019, 0.020, 0.030 |
| 150 held-out proofs | 0.160, 0.181, 0.173 | 0.163, 0.180, 0.176 | 0.197, 0.207, 0.195 | 0.243, 0.280, 0.226 |

The held-out loss is flat through the first pass at every seed and steps up within about 15 steps of each new
pass starting, which is the signature of the same proofs coming round again.

## Held-out pass rate (checkpoint minus base; per problem the mean over seeds; 95% bootstrap over problems)

| After | Below the band (166; base 0.067) | In the band (155; base 0.270) | Above the band (391; base 0.767) | Distinct attempts |
|---|---|---|---|---|
| 0.5 pass | −0.002 [−0.008, +0.005] | +0.029 [+0.014, +0.043] | +0.030 [+0.022, +0.038] | 92% |
| 1 pass | +0.004 [−0.005, +0.013] | +0.034 [+0.018, +0.051] | +0.055 [+0.046, +0.063] | 91% |
| 1.5 passes | +0.006 [−0.008, +0.021] | +0.062 [+0.036, +0.087] | +0.083 [+0.071, +0.096] | 83% |
| 2 passes | +0.015 [+0.001, +0.030] | +0.075 [+0.050, +0.099] | +0.081 [+0.069, +0.092] | 85% |
| 3 passes | +0.004 [−0.011, +0.020] | +0.080 [+0.050, +0.111] | +0.095 [+0.080, +0.109] | 77% |

| Against one pass | Below the band | In the band | Above the band | Below, by seed |
|---|---|---|---|---|
| 1.5 passes | +0.002 [−0.010, +0.014] | +0.027 [+0.007, +0.048] | +0.028 [+0.018, +0.038] | +0.011, +0.005, −0.010 |
| **2 passes** | **+0.011 [−0.001, +0.023]** | +0.040 [+0.021, +0.061] | +0.026 [+0.017, +0.036] | +0.023, +0.010, +0.000 |
| 3 passes | +0.000 [−0.013, +0.014] | +0.046 [+0.021, +0.071] | +0.040 [+0.028, +0.052] | −0.006, +0.021, −0.015 |

The base's distinct-attempt share is 95%. VOID check at one pass, measured against L1: seed 0 in-band +0.028
(L1 +0.038 [+0.010, +0.065]), above +0.058 (+0.057 [+0.044, +0.069]); seed 1 +0.020 (+0.037 [+0.010,
+0.063]), +0.044 (+0.043 [+0.031, +0.057]); seed 2 +0.054 (+0.056 [+0.030, +0.081]), +0.062 (+0.054 [+0.041,
+0.067]).

## The goal set (392 problems the base never resolved while being placed; 32 episodes each)

| Problems solved | Base afresh | 1 pass | 2 passes | 3 passes |
|---|---|---|---|---|
| Seed 0 | 37 | 34 (+12 −15) | 24 (+10 −23) | 25 (+10 −22) |
| Seed 1 | 22 | 40 (+20 −2) | 28 (+13 −7) | 27 (+15 −10) |
| Seed 2 | 26 | 39 (+22 −9) | 30 (+18 −14) | 25 (+16 −17) |
| All seeds | 85 | 113 | 82 | 77 |
| Against the base afresh: gained, lost | | 54, 26 (p = 0.002) | 41, 44 (p = 0.83) | 41, 49 (p = 0.46) |
| Against one pass: gained, lost | | | 32, 63 (p = 0.002) | 31, 67 (p < 0.001) |
| Successful episodes, all seeds | 133 | 185 | 160 | 172 |

- Against one pass the loss of goal problems is the same at every seed: two passes 11 gained and 21 lost, 10
  and 22, 11 and 20; three passes 12 and 21, 11 and 24, 8 and 22.
- Successful episodes fall less than problems solved: the successes land on fewer problems.
- **One pass's own gain over the base is likely and not settled.** L1's own measurement of the same recipe
  (another run of the same training) solved 27, 31 and 35: 41 gained and 33 lost (p = 0.42). Two measurements
  of one recipe differ by 20 problems over three seeds, which is the size of this instrument's noise.

## Why loss and pass rate disagree

The held-out loss is the trained model's surprise at proofs **the base** wrote. Training pulls the model
toward the proof shapes it was trained on; the base's other ways of writing a proof become less likely (loss
up) while the model's own attempts verify more often (pass rate up on the rungs). What the loss tracks is the
narrowing, which the distinct-attempt share shows directly (95%, 91%, 85%, 77% at 0, 1, 2 and 3 passes) and
which the goal set prices.

## Is the model learning to refute instead of prove? (go and see; not pre-registered)

| Held-out rung problems (three seeds) | n | Base | 1 pass | 2 passes | 3 passes |
|---|---|---|---|---|---|
| Known true (proved as stated) | 660 | 0.491 | +0.036 [+0.029, +0.042] | +0.063 [+0.054, +0.072] | +0.070 [+0.059, +0.082] |
| Known false (proved by the negation) | 52 | 0.550 | +0.074 [+0.044, +0.103] | +0.079 [+0.039, +0.120] | +0.072 [+0.022, +0.123] |
| Lean Workbook | 608 | 0.555 | +0.045 [+0.038, +0.052] | +0.073 [+0.063, +0.083] | +0.082 [+0.069, +0.094] |
| STP conjectures | 104 | 0.151 | +0.000 [−0.010, +0.011] | +0.011 [−0.007, +0.031] | +0.004 [−0.018, +0.026] |

- The training set is lopsided twice, at every seed: 41% to 42% of its proofs are refutations (7% of the rung
  problems and 5 of the 392 goal problems are known false), and 2% to 3% are of STP conjectures (15% of the
  rung problems).
- The refutations did not bend the proofs of true statements. Known-true problems gain at every dose, and at
  seed 0 the share of the model's attempts at true statements that open with `intro`, `push_neg` or
  `by_contra` (the openers of 62% of the base's refutations) is 12.0% at the base and 12.9%, 12.3%, 12.8%
  after 1, 2 and 3 passes. An episode is handed its side by the harness; the model does not choose to refute.
- STP problems got nothing at any dose. They are 67 of the 166 below-band problems, which is part of why
  that rung moves least.
- So the cost of the refutation share is what it displaces, not damage.

## Is the challenger picking problems solved a quarter of the time? (L1's three seeds; go and see)

| 1,000 picks, 8 episodes each (mean of three seeds) | k = 0 | k = 1 to 3 (the band) | k = 4 to 7 | k = 8 | Mean pass rate | Mean reward |
|---|---|---|---|---|---|---|
| Challenger | 26% | 34% | 36% | 5% | 0.37 | 0.40 |
| Random draw | 70% | 12% | 13% | 5% | 0.17 | 0.13 |

- Three times a random draw's share in the band, and not centred on 1/4: more picks land above the band (41%)
  than in it.
- **The candidates are thin in the middle.** Of the 20,000, 14,222 are predicted below 0.10 and 3,402 at 0.50
  or above; 693 are predicted between 0.25 and 0.50 and the challenger took all of them in every seed. The
  other 207 scored picks come from just outside; 100 picks are placed at random by design (74% of those
  never solved at seed 0).
- **The estimate is calibrated on average and loose per problem.** Predicted 0.37, realised 0.37. Picks
  predicted at 0.25 to 0.35 land in the band 41% of the time, 28% are never solved, 31% land above. Realised
  reward is flat (0.43 to 0.45) from a predicted 0.25 to 0.60, so about 40% in the band and a reward near
  0.45 is the ceiling of this estimate on these candidates.
- The training set follows: 53% to 56% of the training proofs come from problems solved at least half the
  time. 378 picks are known false (4% of the candidates), since they are rarely at zero (18% against 31% for
  the known-true picks).

## The reward and the two sides (the owner's question, 2026-10-04; computed from stored data, no new run)

The owner, on holding known-false picks to a share: "the model should be rewarded the right way, not us put in
hacks ... Can we fix it through the reward function? Perhaps negations are worth less reward?"

**Why the challenger picks refutations.** L1's 20,000 candidates by the predicted pass rate:

| | Candidates | Below 0.10 | 0.10 to 0.25 | 0.25 to 0.50 | 0.50 and above | Mean expected reward |
|---|---|---|---|---|---|---|
| Known true | 19,265 | 14,211 (74%) | 1,615 (8%) | 376 (2%) | 3,063 (16%) | 0.120 |
| Known false | 735 | 11 (1%) | 68 (9%) | 317 (43%) | 339 (46%) | 0.399 |

Refutations are where the middle of the pool is: 317 of the 693 candidates predicted between 0.25 and 0.50.

**A lower reward for a refutation is a switch.** The reward of a known-false problem multiplied by v; the 900
scored picks of round 1 chosen again; outcomes expected from the 6,000 picks whose k was measured:

| v | Known false of 900 | Expected never solved | Expected in the band | Expected proofs of true statements |
|---|---|---|---|---|
| 1.0 (as run) | 375 (42%) | 20.5% | 36.2% | 407 |
| 0.9 | 258 (29%) | 21.2% | 36.2% | 501 |
| 0.8 | 51 (6%) | 22.5% | 35.8% | 656 |
| 0.7 and below | 0 | 23.0% | 35.6% | 693 |
| Whole pool scored, 1.0 | 48% | 22.7% | 36.8% | 356 |
| Whole pool scored, 0.9 and below | 0 | 22.3% | 37.9% | 699 |

The expected reward of the 900th pick is 0.414 and of the best about 0.47: by the challenger's own rule
(n = 8, t = 1/4, dispersion 0.3) the expected reward is 0.44 at a true pass rate of 0.25, 0.47 at 0.35, 0.43
at 0.50 and 0.36 at 0.60. Nothing between "all the refutations the reward likes" and "none" is reachable with
one multiplier. The known-true candidates that would replace them are expected in the band as often.

**The reward as written moves off them as the solver learns.** Change in pass rate after one pass, on held-out
problems never trained on (three seeds):

| | Below the band | In the band | Above the band |
|---|---|---|---|
| Known true | +0.003 (163 problems) | +0.035 (139) | +0.051 (358) |
| Known false | +0.069 (3) | +0.031 (16) | +0.095 (33) |

With every candidate's predicted rate moved by its cell and the best 900 chosen among the candidates not picked
in round 1: 99 known false (11%), mean pass rate of the picks 0.334. With the predictor not moved (supply
alone): 198 (22%), mean rate 0.414. Round 1: 375 (42%), 0.394. With 20,000 candidates drawn afresh each round
and the predictor not moved, the share goes 42%, 37%, 30% over three rounds (whole pool: 48%, 40%, 31%): supply
alone does not resolve it in three rounds; the solver's own improvement has to reach the challenger.

**It reaches the challenger one model late as the round is specified:** round r's proposals are made before
any result of M(r) exists. L2 therefore makes a round's proposals in batches and refits the challenger after
each (spec, "L2: three rounds").

These are estimates from stored scores. L2 measures the share by batch and by round.

## What is not settled

- The below-band rung: +0.011 [−0.001, +0.023] for the second pass. Three seeds do not separate it from zero.
- A second pass over the same proofs against the same number of steps on fresh ones (several verified proofs
  per problem: a round verifies about 2,900 attempts for its 741 solved problems and trains on one each; or a
  second round). Not compared here.
- The held-out loss is on the base's proofs. A loss on proofs written by someone else (the published
  certificates, which are never trained on) would say whether the model is getting worse at proofs in
  general or only moving away from the base. Not measured.
