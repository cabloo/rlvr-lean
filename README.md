# rlvr-lean

**Reinforcement learning with verifiable rewards for a Lean 4 theorem prover, on one consumer GPU.**

A 7B open prover (DeepSeek-Prover-V1.5) is trained on its own proofs, with the Lean compiler as the only judge
of what counts. A *challenger* picks problems the prover solves about a quarter of the time, a *solver*
attempts them, Lean checks every attempt, and the solver is fine-tuned on what verified. The question the
project asks is narrow and hard: **does this loop take the model to problems it could not solve before, or
does it only make it more reliable on the ones it already could?**

The answer so far, at three seeds and with every read fixed before its run, has two parts. **Training on its
own proofs makes the model more reliable on held-out problems and does not take it further.** What does take
it further is not training at all: **an episode that keeps the steps Lean verified, instead of starting each
attempt from nothing, solves 92 of the never-solved problems where the same number of blind attempts solves
65** (28 gained, 1 lost), 20 of them problems no model here had solved in 558 attempts. Those problems are
reached, not yet reliable; training on the proofs the episode assembles is the step now being built.

The repository contains the loop and the full record of results, including the ones that came out negative.
The distributed Lean checking pool built to feed it is a project of its own, [lean-pool](https://github.com/cabloo/lean-pool).

<p align="center"><img src="docs/figures/rungs_by_round.svg" width="620" alt="Held-out pass rate over the base model after one, two and three rounds, on three difficulty rungs"></p>

## Results at a glance

| Stage | Question | Result (three seeds, 95% intervals) |
|---|---|---|
| One round on self-written conjectures | Does a round of RLVR help on held-out problems? | pass@1 up on three held-out sets (+1.0 to +1.8 points, every interval above zero); miniF2F-test pass@32 unchanged (46.3% to 46.3%) |
| Selection ablation | Does a gradient-based "learning progress" score pick better training proofs than a random draw? | No measurable difference, then found **inconclusive**: a pipeline defect meant the score was ranking one token ([below](#a-defect-worth-describing)) |
| Problem pool | How many published answers survive a re-check under a current Lean? | 67,857 of 88,961; 206 problems were published as both proved *and* disproved |
| One round, challenger against random picks | Does aiming at a target pass rate beat a random draw? | Yes on every held-out rung (+1.3, +2.8 and +3.0 points over the random arm) |
| Dose curve | Train longer on the same proofs? | A second pass adds 4.0 points on mid-difficulty problems and **loses** ground on never-solved ones (32 gained, 63 lost) |
| Three rounds | Does the ladder climb, and does it reach new problems? | Every rung rises (hardest +2.0 points [+0.7, +3.4]). On 392 never-solved problems the trained model succeeds on 1.5 times as many attempts but solves about the same problems (64 gained, 52 lost, p = 0.31) |
| Equal compute | Is training a better use of 24,000 attempts than plain sampling? | No: the base model given those attempts solves 158 of the never-solved problems against the trained model's 96 |
| Lower reward target | Does aiming the challenger at a pass rate of 1/10 instead of 1/4 keep its picks hard, and does the model reach further? | The picks stay on target. Hardest rung: no difference shown (+0.6 points [−0.4, +1.6]). Never-solved problems: a third more successes per attempt at every seed; more problems solved in one seed of three |
| Repair, no training (one seed, two checks) | Is an attempt that resumes from Lean's proof state better than a fresh one? | One repair step is (2.7% against 1.9% verify on hard problems); a second adds nothing. Over five attempts the lead is +0.5 points [−0.2, +1.2], and the same problems get solved (49 against 47) |
| Accumulating episode, no training | Does an episode that keeps its verified lemmas across 8 generations solve more than 8 blind attempts? | **Yes, at every seed.** Hard episodes resolved: +1.0 points [+0.5, +1.6]. Never-solved problems solved: 92 against 65 (28 gained, 1 lost); where the proof needs 4 lines or more, 36 against 28 (8 gained, 0 lost). The same generations, 91% of the tokens, 1.5 times the Lean checks |

Every stage was run from a written spec whose pass, fail and void conditions were committed before the run.
The specs and result notes are in [`docs/spec/`](docs/spec/).

## The loop

```mermaid
flowchart LR
    P[("Pool: 55,631 problems<br/>with a published answer,<br/>re-checked by Lean")] --> C
    C["Challenger<br/>predicts each problem's pass rate,<br/>picks by expected reward"] -->|1,000 problems| S
    S["Solver<br/>8 attempts each<br/>(prove it, or refute it)"] -->|proofs| L
    L{{"Lean 4 + Mathlib<br/>37 workers, 4 machines"}} -->|k of 8 verified| R
    R["Reward for the challenger<br/>peaks at a pass rate of 1/4"] --> C
    L -->|one verified proof per solved problem| T
    T["Fine-tune the solver<br/>(QLoRA, one pass, from the base)"] --> S
    T --> M["Measure on held-out problems:<br/>three difficulty rungs + a goal set<br/>the base never solved"]
```

- **Problems come with a published answer.** A problem is a statement from Lean Workbook or STP together with
  a published proof of it, or of its exact negation, that our own Lean accepts. Nothing the model wrote is in
  the pool, and published proofs are used as certificates only, never as training targets: the solver trains
  only on proofs it found itself.
- **Either side counts.** A problem is resolved by proving the statement or by proving its negation, so
  refuting a false statement is rewarded like proving a true one.
- **The challenger's reward** for a problem that `k` of `n` solver attempts resolved, with `p = k/n` and a
  target rate `t`:

  $$r(k) = \frac{p\,(1-p)^{a}}{t\,(1-t)^{a}}, \qquad a = \frac{1}{t} - 1$$

  It is 0 for a problem never solved or always solved, and 1 at the target. With `t = 1/4` and `n = 8` it pays
  0.79, 1.00, 0.87, 0.59 for 1 to 4 successes. There is no difficulty filter anywhere else: difficulty falls
  out of this reward.
- **Held-out measurement.** 2,000 problems are held out before anything is trained. Those the base model
  sometimes solves are cut into three rungs by its pass rate; the 392 it never solved in 32 attempts are the
  *goal set*. Every comparison is paired by problem with fresh samples on both sides, because a problem placed
  as "hard" by a noisy count reads easier the next time with no training at all.

## What the experiments say

### Training makes the model more reliable, on held-out problems, at every difficulty it can already touch

Three rounds, each model trained from the base on all rounds' proofs so far (about 720, 1,470 and 2,190),
lift the pass rate on every held-out rung: by 7.6 points where the base already solves 77% of attempts, by
8.7 where it solves 27%, and by 2.0 [+0.7, +3.4] where it solves 7%. The third round adds nothing that
separates from zero.

### Training alone has not reached problems the base could not solve

<p align="center"><img src="docs/figures/goal_set.svg" width="760" alt="Goal problems solved and successes per 1,000 attempts, base model against the three-round model, at 32 and at 93 attempts"></p>

On the goal set, with both models given 93 attempts per problem, the trained model succeeds on 5.4 attempts
per 1,000 against the base's 3.6 (difference +1.8 [+0.6, +3.2]). But those extra successes fall on problems
both models solve: problem by problem it gains 64 and loses 52 (p = 0.31). And charged for its own training
attempts, the loop loses to brute force: the base model, simply sampled three times as much, solves 158 goal
problems to the trained model's 96. This is the same shape the first experiment showed on miniF2F (pass@1 up,
pass@32 flat), one level harder.

### Training longer on the same proofs narrows the model

<p align="center"><img src="docs/figures/dose_curve.svg" width="760" alt="Loss on training and held-out proofs over three passes, and held-out pass rate by rung at five checkpoints"></p>

Held-out loss says stop after one pass; held-out pass rate on solvable problems says two; the goal set says
one. The three disagree because repeated passes concentrate the model on fewer proof shapes (distinct
attempts fall from 95% to 77%), which helps where it can already win and hurts where a problem needs a rare
attempt. The held-out loss was measuring the narrowing, not the proving.

### The pool has no middle

<p align="center"><img src="docs/figures/pool_shape.svg" width="620" alt="Candidate problems by predicted pass rate: most near zero, many near one, few in between"></p>

For this model, published problems are mostly either out of reach or easy. Of 51,631 candidates, 2,555 are
predicted inside the band the reward aims at and only 718 in its upper half, where the challenger's expected
reward peaks; one round uses those up. After that its picks drift *easier* (mean pass rate 0.37, 0.47, 0.47
over three rounds), the opposite of the intended ladder. Its predictor is calibrated on average but loose
per problem, so an expected-reward maximiser aims above the target. Both effects are properties of the data
and the estimate, not of the reward's shape.

<p align="center"><img src="docs/figures/challenger_picks.svg" width="760" alt="Share of refutations among the challenger's picks and mean pass rate of its picks, by batch, three seeds"></p>

Refutations of false statements are where the pool's middle is (they are 4% of candidates and 45% of first-
round picks). A quota would fix the share by decree, and a discount on their reward is a switch, not a dial:
20% off leaves 6% of the picks, 30% off leaves none. Left alone, with the challenger refit four times a round so it sees the current solver, the
share falls by itself as the solver learns to refute.

### Aiming the reward lower trains on fewer easy proofs, and that helps where it is hard

<p align="center"><img src="docs/figures/lower_target.svg" width="760" alt="Goal problems solved and successes per 1,000 attempts at 93 attempts each, by seed: base model, three rounds at target 1/4, three rounds at target 1/10"></p>

The same three rounds were run again with one setting changed: the reward's target, 1/10 in place of 1/4, so
that its expected-reward peak sits at a quarter instead of above a third. The challenger then stays where it
was aimed (its picks average a pass rate of 0.27, 0.34 and 0.24 over the rounds, against 0.37, 0.47 and 0.47)
and refutations fall to a twentieth of its picks, with no quota.

The measure fixed before the run, the hardest held-out rung, shows no difference (+0.6 points [−0.4, +1.6],
the same sign at each seed). On the never-solved problems the lower target's model succeeds on about a third
more attempts than the other at every seed (7.2 per 1,000 against 5.4; the base 3.6). It solves more of those
problems in one seed of three and exactly as many as the base in the other two, so that is not counted as
reach. It gives up about a point on the easy rung.

What changed is the training set: about as many proofs from problems few solvers cracked (2,486 against
2,601) and about 1,450 fewer from problems most did. Leaving the easy proofs out keeps the model more varied
(90% of its attempts are distinct, against 87%), which is the dose curve's finding from the other side.

### Feeding Lean's answer back: one repair step puts the solver ahead, a second adds nothing

<p align="center"><img src="docs/figures/repair.svg" width="760" alt="Share of hard episodes resolved within one to five attempts: starting over against resuming from Lean's proof state after every failure, and against repairing once and then starting over; the lead in points is marked at two and at five attempts"></p>

Everything above is blind resampling: a failed attempt tells the model nothing. Two checks with the untrained
base model asked what happens when it does. A failed proof is cut before its first error, Lean reports the
proof state at the cut, and the model resumes from the kept lines and that state, in the comment format the
prover's authors trained it on. The figure shows running totals, because a per-attempt rate at later attempts
would compare different survivors: the arm that resolved more early is left with the harder episodes.

- **The state is information the model uses.** Resuming with it resolves 2.6 points more hard episodes than
  resuming from the same kept lines without it [+1.3, +3.9].
- **One repair step straight after a failure beats a fresh attempt,** measured twice on separate samples:
  1.9% against 1.1% of 1,880 failures verify, then 2.7% against 1.9% of 6,562.
- **The lead it makes is kept, and it is small.** Within two attempts the repairing episode is ahead by 0.7
  points [+0.1, +1.4]; within five, by 0.5 [−0.2, +1.2] (9.0% of 6,696 hard episodes against 8.5%), which is
  no longer separable from zero. The first check, which resumed after every failure, ends at +0.6 [−1.0, +2.2].
- **A second repair step adds nothing.** On the 6,179 episodes still open in both arms after three attempts,
  it verifies 94 times and a fresh attempt 98. The episodes a repair can fix are taken by the first one; and
  resumed every time, the model rewrites what Lean rejected (46% exact copies by the fifth attempt).
- **Reach does not move.** The same goal problems get solved either way (49 against 47).

The second check was sized to resolve the gain that the first one's data suggested (+1.3 points). It did not
repeat, which is what fixing the read before the run is for.

### What the never-solved problems have in common: they need longer proofs

Reading the stored results by the length of each problem's shortest published proof gave the cause. The
problems the base never solves are not another kind of mathematics: their proofs are a few intermediate facts
and a closing step (median 4 lines, against 1 on the easiest rung). And training helps exactly where a short
proof exists:

| Shortest published proof | Never-solved problems | Base, successes per 1,000 attempts | After three rounds | Ratio |
|---|---|---|---|---|
| 1 line | 37 | 9.5 | 22.9 | 2.4 |
| 2 to 3 lines | 125 | 4.7 | 10.6 | 2.3 |
| 4 to 7 lines | 156 | 2.9 | 3.9 | 1.4 |
| 8 lines or more | 74 | 0.4 | 0.3 | 0.7 |

A proof written in one shot is verified all or nothing, so its chance falls with every step it needs.
Training on the model's own proofs raises the chance of a step it already takes and does not supply the steps
it never takes. Meanwhile the pieces are there and are thrown away: in 58% of failed first attempts on hard
problems, the first error is on the last line, so every step before it had verified, and the next attempt
starts from nothing.

### An episode that keeps what verified reaches problems that sampling does not

<p align="center"><img src="docs/figures/accumulating_episode.svg" width="760" alt="Share of hard episodes resolved within one to eight generations, eight blind attempts against an episode that keeps its verified lemmas; and never-solved problems solved by each, by the length of the shortest published proof"></p>

So the episode was changed and the model was not. An episode gets up to 8 generations on a problem and holds a
pool of lemmas that only grows. After each failed proof, its leading `have` steps that Lean accepted are added
to the pool (and checked together); the closing step of a failed proof is kept and tried again whenever the
pool has grown; and a generation either starts fresh or continues from the pool, shown Lean's proof state.
The comparison is 8 blind attempts from the same first attempt, with the same untrained base model, read by
running totals.

- **It resolves more hard episodes at every seed,** +1.1, +1.1 and +0.8 points; over the three, +1.0
  [+0.5, +1.6] (12.9% of 10,044 episodes against 11.8%). The lead grows with every generation, where a repair
  step's lead was made once and held.
- **It solves problems sampling does not.** Of the 392 never-solved problems, 92 against 65 over 18 episodes
  each: 28 gained, 1 lost. 20 of the 92 were never solved by the base model or by any trained model here in
  558 attempts.
- **The longer proofs move too.** Where the shortest published proof is 4 lines or more: 36 against 28, 8
  gained and none lost (sign test p = 0.008). Most of the gain is still on short-proof problems, the ones
  blind attempts kept nearly solving.
- **The cost is Lean time, not GPU time.** The same number of generations, 91% of the generated tokens, and
  1.5 times the Lean checks, because an assembled proof (the pooled lemmas with a kept closing step) is a
  candidate that cost no generation. Of the 28 gained problems, 23 were won by such a proof.
- **Reached is not reliable.** Of the 28 gained problems, 19 were won in one episode of 18. And the untrained
  model does little with a pool it is shown: three quarters of its continuations go straight to a closing
  tactic, and 27% repeat a proof Lean already rejected.

For scale: the base model with 279 blind attempts a problem solved 82 of these problems, and the models after
three rounds of training, with 279, solved 94. The accumulating episode solves 92 with 144 generations of the
untrained model.

### A defect worth describing

The first two experiments reported a fourfold drop in held-out loss after one round, and a selection score
that picked no better than chance. Both were one bug. The prover writes a sequence-start token between its
prompt and its proof; the training pairs omitted it. That single position accounted for 97% to 101% of every
measured loss drop and 99.8% of the gradient the selection score was built on. The pass-rate results, which
Lean checked, survived the fix unchanged (+0.99 points against +1.01); the loss-based conclusions were
withdrawn in the result notes, and the loss is now reported by part.

### A published label is not a certificate

Of 140,214 Lean Workbook statements, 206 are published as both proved and disproved, because one source's
"disproved" negates the conclusion while keeping the hypotheses, which is vacuous when the hypotheses
contradict each other. Every certificate is therefore re-checked under the pinned Lean, against the *exact*
negation where it is a disproof. Mathlib's lemma renames since the proofs were published cost another 8,379
certificates, recovered with a 29-entry rename table taken from Mathlib's own deprecation records.

## Engineering

- **Verification is the bottleneck, so it got its own project.** [lean-pool](https://github.com/cabloo/lean-pool) puts HAProxy, a
  shared result cache and per-machine load agents in front of several
  [Kimina Lean Servers](https://github.com/project-numina/kimina-lean-server), with an admission test a
  server must pass before it joins, optional mutual TLS, request priorities and a capacity signal clients
  follow. The experiments ran on 37 Lean workers across four machines at 30 to 50 checks per second.
- **Sampling and checking roll.** The GPU samples in chunks while a finaliser settles blocks as their checks
  return, so neither side waits for the other. Every step is idempotent against a store and resumes at the
  first block not done.
- **A soundness alarm.** A verified proof on the side a published certificate rules out stops the run: either
  the certificate or the checker is wrong.
- **One 16 GB card.** QLoRA on a 4-bit base for training (peak 8.7 GB with gradient checkpointing), vLLM with
  an FP8 export and LoRA adapters for sampling at about 2,500 tokens per second.
- **Statistics.** Paired by problem, bootstrap over problems, sign tests for solved/unsolved flips, one seed
  as a scout and three to conclude, and a distinction kept between a run that failed and an idea that failed.
- **Tests.** About 790 tests for the loop (the pool's 1,100 are in its own repository). The model and Lean are
  replaced by stand-ins, so the suite runs with no GPU and no network.

## Repository layout

```
src/rlvr_lean/
  domain/           pure rules: verification status, the reward and the band, the challenger's
                    predictor, training-set selection, the repair cut, estimators (pass@k, bootstrap)
  infrastructure/   the Lean client, the artifact store, the verification service
  gpu/              the steps that run on the GPU (sampling, training, measuring), one process each
  reporting/        the read of each stage, exactly as fixed before its run
  runner/           the stage runner: which steps make a stage, in what order
  tools/            building the problem pool, re-checking certificates, the Lean-version check
  config/           one experiment config
tests/              the specs' fixtures as tests; stand-ins for the model and for Lean
docs/spec/          the specs and the result notes, as written during the project
docs/results/       the result summaries the figures are drawn from
docs/figures/       the figures and the script that draws them
```

## Running it

```bash
uv run --group dev pytest          # the test suite: no GPU, no network
python docs/figures/make_figures.py   # redraw the figures from docs/results/ (needs matplotlib)
```

The GPU stages need a 16 GB card, a Kimina Lean Server (or [lean-pool](https://github.com/cabloo/lean-pool) in front of several) and the
model weights. A stage is one command, and the stages are listed in
[`src/rlvr_lean/runner/entry.py`](src/rlvr_lean/runner/entry.py); each has a `_smoke` variant that runs on the
shipped fixtures, for example
`PYTHONPATH=src python -m rlvr_lean.runner.entry --stage ladder_l1_smoke --profile full --out out/`. The settings are in
[`src/rlvr_lean/config/experiment.yaml`](src/rlvr_lean/config/experiment.yaml). The problem pool is rebuilt
from the published datasets with `python -m rlvr_lean.tools.ladder_pool` (it is not shipped: 30 MB derived
from Lean Workbook and STP). The one data file that is shipped,
[`heldout_proof_lines.jsonl`](src/rlvr_lean/data/ladder_l0/heldout_proof_lines.jsonl), holds the held-out
problems' ids and the line count of each one's shortest published proof, which the reports group by; it holds
no statement and no proof.

## Status

Training on one-shot proofs makes the prover more reliable on what it can already sometimes do. Reach came
from the search: an episode that keeps what verified solves problems that sampling does not, with an untrained
model. Neither is yet the aim, which is to do reliably what could not be done before. Three things are open.

- **Training on what the episode assembles.** The proofs it verifies on the never-solved problems are longer
  than the blind attempts' (median 6 lines against 4, the longest 21 against 11), and they are the training
  examples the loop has never had. The next stage puts the accumulating episode inside the round and asks
  whether the trained model then solves the reached problems reliably, and in one shot.
- **A ceiling for that.** One fine-tune on published proofs of problems the base cannot solve, as a labelled
  diagnostic and never kept, says whether this model can learn to write longer proofs when shown them at all.
  It bounds what the stage above can reach.
- **A deeper pool.** About 650,000 published proofs have not been re-checked yet; the pool's thin middle is
  the scarcest thing the loop has.

## Credits

Research direction and decisions: Zane Hooper. The code, the experiment runs and the result notes were
produced with [Claude Code](https://claude.com/claude-code), working from written specs with reads fixed
before each run; the notes in `docs/spec/` are kept as they were written, with machine names removed.

Built on [DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5),
[Lean 4](https://lean-lang.org/) and [Mathlib](https://github.com/leanprover-community/mathlib4),
[Kimina Lean Server](https://github.com/project-numina/kimina-lean-server),
[Lean Workbook](https://huggingface.co/datasets/internlm/Lean-Workbook),
[Goedel-Prover's proofs](https://huggingface.co/datasets/Goedel-LM/Lean-workbook-proofs),
[STP](https://huggingface.co/datasets/kfdong/STP_Lean_0320), [vLLM](https://github.com/vllm-project/vllm) and
[PEFT](https://github.com/huggingface/peft).
