# Ladder loop, L3a (the repair check, no training, seed 0): not shown at five attempts; the first resumed attempt verifies about twice as often as a blind one on hard problems, and the loop then repeats itself

Spec: `docs/spec/ladder-loop.spec.md`, "L3a: the repair check, no training". Run 2026-10-05 (UTC) at Lean
v4.27 through the `oeis` pool (37 workers), on the
GPU box, the base model, nothing trained: 21 minutes. Tasks `rlvr_lean/ladder_l3a_smoke_r1` and
`rlvr_lean/ladder_l3a_seed0_r1`. One seed: this is a check of a search step, not a trained arm.

**Verdict by the read fixed before the run: NOT SHOWN AT THIS SIZE.** Not a kill and not a win.

- **Both "can this run see a win" checks pass.** A state came back for 97.5% of the failed first attempts
  (2,223 of 2,281; half was needed). On the above-band rung the resumed next attempt verifies 55.5% of the
  time against the blind one's 65.9% (half of blind's was needed).
- **Primary (G and the below-band rung, 558 problems, 1,900 episodes; resolved within 5 attempts; resume with
  the state minus blind): +0.0063 [−0.0099, +0.0220]** (0.0784 against 0.0721). Without the trimmed loops
  (kept lines that were already a whole proof): 0.0000 [−0.0157, +0.0161].
- **The single step (given a failed first attempt, the next attempt verifies), hard problems:** blind 20 of
  1,880 (1.1%), with the state 36 (1.9%), without the state 16 (0.9%). Resolved within 2 attempts, with the
  state minus blind: **+0.0125 [+0.0027, +0.0228]**; on the below-band rung +0.036 [+0.006, +0.066] (26
  against 14 of 315); on G +0.003 [−0.003, +0.008] (10 against 6 of 1,565).
- **G by problem (resolved in any of 4 episodes):** 18 with the state against 19 blind; gained 11, lost 12.
- **Budget.** The resuming arm generated 54% of the blind arm's tokens on the primary's problems (363,473
  against 674,541), so the primary is read as it stands. Its prompts are 2.5 times as long (2.56 M tokens
  against 1.04 M) and it sends twice the Lean checks (14,664 against 7,383: one more for each state).
- **What the model did with it.** A state in 98.5% of repair loops. The first step written repeats the step
  that had just failed in 35.3% of them (2,810 of 7,968), and none of those verified. From one loop to the
  next the cut stays on the same line 76% of the time.

So the branch is the middle one: an interval that contains zero, and the two diagnostics go to the owner. They
say the model does not ignore the state, and it does repeat itself.

## What ran

For each problem, first attempts sampled blind from the base model: 4 on each of the 392 goal problems, 2 on
each of the 712 rung problems (2,992 episodes; 711 verified at once). Each of the 2,281 failed first attempts
was continued three ways, up to 4 more attempts each: blind (whole proofs from the plain prompt), resume with
the state (the kept lines before the first error, then Lean's goals there in the prover's own comment format),
resume without the state (the same kept lines, no comment). 28,496 attempts, 39,259 Lean checks, 1.68 M
generated tokens. No soundness alarm (60 audit attempts on the side a certificate rules out, none verified).

## Resolved within 5 attempts (each problem's mean over its episodes; 95% bootstrap over problems)

| Set | Blind | With the state | Without the state | With the state minus blind | Without minus blind | With minus without |
|---|---|---|---|---|---|---|
| G (392) | 0.013 | 0.017 | 0.008 | +0.004 [−0.005, +0.013] | −0.006 [−0.013, +0.001] | +0.010 [+0.003, +0.017] |
| Below the band (166) | 0.211 | 0.223 | 0.160 | +0.012 [−0.036, +0.063] | −0.051 [−0.096, −0.006] | +0.063 [+0.024, +0.105] |
| **Hard: both (558)** | 0.072 | 0.078 | 0.053 | **+0.006 [−0.010, +0.022]** | −0.019 [−0.034, −0.005] | +0.026 [+0.013, +0.039] |
| In the band (155) | 0.697 | 0.561 | 0.529 | −0.136 [−0.194, −0.077] | −0.168 [−0.232, −0.103] | +0.032 [−0.016, +0.081] |
| Above the band (391) | 0.990 | 0.935 | 0.925 | −0.055 [−0.076, −0.036] | −0.065 [−0.084, −0.046] | +0.010 [−0.003, +0.023] |

Three things are in this table.

1. **The state is information the model uses.** With it, resuming beats resuming without it on the hard
   problems (+0.026, interval clear of zero) and on each hard set.
2. **Keeping the prefix alone costs.** Resuming without the state is worse than starting over on every set.
3. **On problems the model can often solve, starting over beats resuming,** with or without the state: 14
   points in the band and 5.5 above it. A fresh attempt may take another approach; a resumed one is held to
   the approach that just failed.

## Attempt by attempt, hard problems (verified / attempts made at that loop)

| | Attempt 2 | Attempt 3 | Attempt 4 | Attempt 5 |
|---|---|---|---|---|
| Blind | 20 / 1,880 (1.06%) | 21 / 1,860 (1.13%) | 21 / 1,839 (1.14%) | 9 / 1,818 (0.50%) |
| Resume with the state | 36 / 1,880 (1.91%) | 26 / 1,844 (1.41%) | 14 / 1,818 (0.77%) | 5 / 1,804 (0.28%) |
| Resume without the state | 16 / 1,880 (0.85%) | 12 / 1,864 (0.64%) | 12 / 1,852 (0.65%) | 5 / 1,840 (0.27%) |

The first resumed attempt is worth nearly two blind ones; the fourth is worth a quarter of one. The blind
arm's low last column was checked for a mechanical cause and none was found (the same token counts, Lean
times and finish reasons as the other loops; every sampling seed distinct; by set 2 of 1,549 on G and 7 of
269 below the band, against 6, 2, 8 and 14, 19, 13 before).

## Why the resumed loop fades: it writes what it wrote

Share of attempts whose whole proof is an exact copy of an earlier attempt in the same episode:

| | Attempt 2 | Attempt 3 | Attempt 4 | Attempt 5 |
|---|---|---|---|---|
| Blind | 0.5% | 1.0% | 1.5% | 1.8% |
| Resume with the state | 17.1% | 30.4% | 39.0% | 45.9% |
| Resume without the state | 23.3% | 37.7% | 48.9% | 54.0% |

By the fifth attempt nearly half of the resuming arm's generations and Lean checks go to a proof Lean has
already rejected. The step that fails is mostly the closing one: 69% of the repair loops follow a failed
`nlinarith` or `linarith`.

| Failed step | Repair loops | Writes the same line again | Starts with the same tactic | Verified |
|---|---|---|---|---|
| `nlinarith` | 4,543 | 35% | 78% | 1.6% |
| `linarith` | 1,144 | 52% | 67% | 2.3% |
| `have` | 449 | 11% | 61% | 6.2% |
| `rw` | 380 | 19% | 56% | 7.1% |
| `simp` | 214 | 10% | 45% | 11.2% |
| `omega` | 202 | 77% | 78% | 0.0% |
| `positivity` | 125 | 70% | 70% | 0.8% |

Where the model was shown the state and wrote something else, it does find proofs. Two from the goal set:
after `nlinarith [pow_two_nonneg (a - b), ...]` failed on `a ^ 3 + 2 * b ^ 3 + 2 * c ^ 3 ≤ 4` with each
variable in [0, 1], it wrote three `have : a ^ 2 - a ≤ 0` steps and closed the goal; after `linarith` failed
on `(b * (2 * a) / (a + b)) ^ 2 ≤ a * b`, it wrote `field_simp`, `ring_nf`, `nlinarith`. Resumed attempts
that do not repeat the failed step verify at 1.51% on the hard problems (72 of 4,769), against the blind
arm's 0.96% (71 of 7,397).

Known corner, seen in the smoke run: when the error is inside an `all_goals` block, the state shown is the
branch that is still open, not the branch where the step failed.

## What this does and does not say

- It does **not** say repair fails. The primary's interval contains zero at this size, and the arm that
  resumed spent half the generated tokens.
- It says **where a repair step pays for this model, untrained:** once, straight after a failure, on a hard
  problem, with the state shown. It says where it does not: repeated from the same cut, and on problems where
  a fresh attempt often works.
- **A reading composed from the stored rows, not a run arm, and made after seeing the above:** one repair
  step and then blind attempts (attempt 2 from the resuming arm, attempts 3 to 5 from the blind arm's first
  three) resolves 0.0855 of hard episodes within 5 attempts against blind's 0.0721, +0.0134 [+0.0049,
  +0.0224], and loses nothing in or above the band (+0.026 [−0.010, +0.061], +0.003 [−0.004, +0.009]). It
  swaps the blind arm's fifth attempt, which was unusually poor here, for the repair step, so it restates the
  single step more than it adds to it. It is a candidate for a check with its own read, not a result.

## For the owner: what to do with repair

1. **A second check, no training, about 25 minutes:** an episode that repairs once after each fresh failure,
   never sends Lean a proof it has already rejected, and otherwise starts over; against blind at equal
   attempts and at equal generated tokens, on fresh sampling seeds, read fixed first. It tests the composed
   reading above honestly.
2. **Train on repair (L3b):** rounds whose episodes include a repair step, and training on the
   failed-then-repaired proofs, so that not repeating the rejected step is learned. This run has 251 such
   proofs; a round of 1,000 problems would give more.
3. **Leave repair** and put the GPU on the lower target and more rounds.

Recommended: 1, then 2 if it holds. The check is cheap, and it settles how a repair step should sit inside a
round before anything is trained on it.

## Files

`experiments/rlvr_lean/ladder_l3a_seed0_r1/steps/`: `report_ladder_l3a.json` (every number above except the
attempt-by-attempt, copies and failed-step tables, which are read from `repair_attempts_<loop>.jsonl` and
`repair_states_<loop>.jsonl`), `repair_episodes.jsonl`, `repair_problems.jsonl`.
