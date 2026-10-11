# Ladder loop, L4c (six fresh rounds from the larger pretrained model, reward-weighted rows with rehearsal, and its own control; seed 0): THE GAIN IS NOT SHOWN TO BE THE ROUNDS'. Against the pretrained model the six-round model adds on its failures (13.2 successes per 1,000 against 8.4, +0.0048 [+0.0006, +0.0097]). Against a model trained on the same published rows and none of the rounds' it is behind: 13.2 against 19.3 on those failures (−0.0062 [−0.0110, −0.0017]), 229 goal problems solved against 258 (gained 19, lost 48, p = 0.001), 120 solved reliably against 135. The model's own proofs, six rounds of them, made it sharper on what it could do and narrower; the published rows beside them did the adding.

Labelled **pretrained on published proofs**, as everything built on that file is.

Spec: `docs/spec/ladder-loop.spec.md`, "L4c: the arm from `pre_r64` with a rule L4t2 names" on "L4b: the arm
again, from the larger pretrained model": the read, the separation and its three outcomes fixed before the run.
Why it ran although L4t2's rule said no arm with rehearsal rows: that rule stood on such an arm being unreadable,
this arm carries its own control, and the owner asked (2026-10-09) for the test of whether the loop "is actually
creating a feedback cycle that is able to push the model beyond capabilities that we're able to get through
pre-training alone". Run 2026-10-10 (UTC) at Lean v4.27 through the `oeis` pool
(15 workers on two servers from the first half hour on: the other two were paused), on the GPU box:
tasks `rlvr_lean/ladder_l4b_reward_rehearse_smoke_r1` and `rlvr_lean/ladder_l4b_reward_rehearse_seed0_r1` (539
minutes), from 4c1892f95. One seed.

**What was run.** `pre_r64`'s own map (4,000 problems, 8 attempts). Then six rounds: each round the challenger
picks 1,000 problems of the `loop` half (whose published proofs no model here has seen), the current model makes
8 attempts at each, Lean verifies, assembly runs on what the attempts left unsolved, and M(r) is trained FROM
`pre_r64` on the rule's rows of rounds 1 to r. The rule `reward_rehearse`: a row of the rounds is kept with the
probability of its problem's reward at the target rate 1/10 (so mostly problems one solver in eight solved), and
as many published rows as the kept set has one-shot rows are added. Then three models measured the same way,
one after another, on the same pool: `with` (M(6): 1,617 rows of the rounds and 1,511 published rows),
`without` (the twin: the same without the 106 assembled proofs) and `rehearsal_alone` (the 1,511 published rows
and nothing else). All five checks pass, and `rehearsal_alone`'s three.

## The reads fixed before the run

| 392 goal problems | `pre_r64` (start) | `with` (six rounds) | `without` (twin) | `rehearsal_alone` (control) |
|---|---|---|---|---|
| Rows of the rounds / published rows | | 1,617 / 1,511 | 1,511 / 1,511 | 0 / 1,511 |
| G′ (192), the 61 fresh attempts, per 1,000 | 8.4 | 13.2 | 14.4 | 19.3 |
| All of G, 93 attempts, per 1,000 | 80.0 | 92.2 | 93.4 | 95.4 |
| Goal problems solved in 93 | 245 | 229 | 234 | 258 |
| of 8 lines or more (74) | 25 | 20 | 21 | 32 |
| Solved reliably | 107 | 120 | 121 | 135 |
| Of the 81 never solved before | 7 | 8 | 8 | 10 |

- **Primary: `with` minus `pre_r64` on G′ over the 61 fresh attempts: +0.00478 [+0.00060, +0.00973].** The
  interval is clear of zero and above: the branch's name is THE LOOP ADDS ON TOP OF PRETRAINING.
- **Apart from the rehearsal: THE GAIN IS NOT SHOWN TO BE THE ROUNDS'.** `with` minus `rehearsal_alone` on the
  same attempts is −0.00615 [−0.01101, −0.00162], and `rehearsal_alone` itself adds (+0.01093 [+0.00555,
  +0.01699]). By the spec the primary is REPORTED with this line and is not called the loop's gain.
- **Breadth:** `with` against `pre_r64` 229 to 245, gained 29, lost 45 (p = 0.08): not NARROWER by the test.
  Against its own control it is: gained 19, lost 48 (p = 0.001).
- **The long proofs:** `with` minus `pre_r64` on the problems of 4 lines or more +0.0111 [−0.0002, +0.0228]: not
  SHORTER. Against the control −0.0104 [−0.0219, +0.0007].
- **The assembled proofs do not teach:** `with` minus `without` on G′ −0.0013 [−0.0032, +0.0008] (106 proofs,
  median 6 lines).

## What the rounds did

Picks 1,000 a round; solved by an attempt 604, 664, 638, 615, 562, 540; the picks' mean pass rate 0.21, 0.27,
0.25, 0.25, 0.22, 0.19 (the challenger moves toward the target of 1/10); assembly solved 22, 10, 13, 25, 20, 18
that the attempts left. Of the rounds' 3,731 rows the rule kept 1,617: 1,165 of the 1,205 from problems one
solver in eight solved, 360 of 747 at two, 75 of 518 at three, 17 beyond.

## What it says

- **At this size the loop's own proofs do not push the model past what published proofs give it; they pull it
  back.** Twice now, with the stored rounds in one pass (`ladder-l4p-RESULT.md`) and with six fresh rounds at
  the larger model's own frontier, the same published rows WITHOUT the model's own proofs solve more problems
  (264 against 247; 258 against 229) and do better where the start model fails.
- **What the model's own proofs do is sharpen.** Trained on them it succeeds more often on problems it could
  already reach (120 solved reliably against the start model's 107; the rungs up) and reaches fewer (229 against
  245). A verified proof the model found in 8 attempts is, by construction, one it could already write.
- **The control is a replicate of the small second pass.** 1,511 published rows give 95.4 per 1,000, 258
  problems, 135 reliably; L4p's 1,262 gave 99.2, 264, 134. Two trainings of nearly one recipe differ by 4 per
  1,000 and 6 problems: the differences read above are several times that.

## What it does not say

- **The control is not matched in steps, and one control is one training.** `with` took 391 optimizer steps,
  `rehearsal_alone` 189. CORRECTED after the dose's two long trainings (the same day): I first wrote that among
  published-only models more steps had meant less. That was three points and it was noise: the five
  published-only models from `pre_r64` have 99.2, 95.4, 86.8, 99.3 and 108.9 successes per 1,000 at 158, 189,
  428, 1,555 and 3,109 steps, and solve 264, 258, 252, 255 and 267 goal problems. Two trainings of one kind
  differ by about 6 per 1,000 and 5 problems (one standard deviation). Against THAT spread: every
  published-only model solves more goal problems than `with` (229) or `without` (234), by 18 to 38, and does
  better on G′ (16.2 to 26.9 against 13.2 and 14.4); per attempt on all of G `with` (92.2) is inside the
  published-only range. So the narrowing and the loss on the start model's failures stand against five controls
  of every step count; a loss per attempt on all of G is not shown.
- **That a loop cannot do this.** This one trains on proofs found within 8 attempts, in six rounds of 1,000
  problems: about 48,000 attempts. The loops known to move a model of this size ran orders of magnitude more.
  What is shown is narrower: rows the model can already sample do not extend it. Rows it cannot (a solve that
  takes hundreds of attempts, or an assembly of pieces) were 106 of 3,128 here, and too few to read.
- **Anything at more than one seed**, though the direction is the same in two separate comparisons.

## The gates of a verdict on the idea

Not a KILL. The run could have seen a win (a positive control cleared the same reads on this goal set through
this harness: the published rows alone), and its models were measured together on one pool. But its budget is a
small fraction of the reference, and the bed's sizes (8 attempts a problem) decide what kind of row a round can
produce. The honest verdict: AT THIS SIZE AND WITH THESE ROWS, NOT THE ROUNDS' GAIN; the idea of a loop that
reaches past its data is not judged by it.

## What follows

The dose of a second pass (L4p part 2) is running: it is the pretraining-only ceiling any loop result must
clear, and it says what training steps alone do. After it, by the owner's direction, the next test of the loop
is on rows the model CANNOT already sample: solves that take deep search, each one a problem it had failed many
times, read against a step-matched published-only control.
