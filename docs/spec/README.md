# Specs and result notes

These are the working documents of the project, kept as they were written, with the names of the machines
removed. Each experiment was specified before it ran, with what would count as a pass, a failure or a void run
fixed in advance; each result note reads the run against that.

They were written during the work, addressed to the project's owner, so they say "the owner" where a decision
or a question came from the owner, and they keep the corrections that later runs forced.

| File | What it is |
|---|---|
| [`ladder-loop.spec.md`](ladder-loop.spec.md) | The challenger and solver loop: the rules, the reward, the held-out sets, every stage's read |
| [`ladder-l0-RESULT.md`](ladder-l0-RESULT.md) | The problem pool: published answers re-checked by Lean; the held-out sets |
| [`ladder-l1-RESULT.md`](ladder-l1-RESULT.md) | One round, challenger against a random draw, three seeds |
| [`ladder-l1b-RESULT.md`](ladder-l1b-RESULT.md) | The dose curve: one, two and three passes over the same proofs |
| [`ladder-l2-RESULT.md`](ladder-l2-RESULT.md) | Three rounds; the goal set at equal attempts and at equal compute |
| [`ladder-l2t-RESULT.md`](ladder-l2t-RESULT.md) | Three rounds again with the reward aimed at a pass rate of 1/10 in place of 1/4, three seeds |
| [`ladder-l3a-RESULT.md`](ladder-l3a-RESULT.md) | The repair check, no training: resuming a failed proof from Lean's proof state against starting over |
| [`ladder-l3a2-RESULT.md`](ladder-l3a2-RESULT.md) | The second repair check: one repair step after each fresh failure, then a fresh attempt |
| [`reach-diagnosis-RESULT.md`](reach-diagnosis-RESULT.md) | Why the loop had not reached new problems: proof length, discarded verified steps, and a probe that pools them |
| [`ladder-l3c-RESULT.md`](ladder-l3c-RESULT.md) | An episode that keeps what verified: more hard episodes resolved and more never-solved problems reached, three seeds |
| [`ladder-ceiling-RESULT.md`](ladder-ceiling-RESULT.md) | The ceiling, a labelled diagnostic: one training on other provers' published proofs, to see whether the model can learn longer proofs; not a result of the loop, one seed |
| [`ladder-l3d2-RESULT.md`](ladder-l3d2-RESULT.md) | Six rounds with the assembly in the round, and a twin trained without the assembled proofs: not shown, one seed |
| [`phase-a-RESULT.md`](phase-a-RESULT.md) | The first experiment: one round on self-written conjectures |
| [`phase-b-selection-RESULT.md`](phase-b-selection-RESULT.md) | Selection by a learning-progress score against a random draw, and its correction |
| [`native-format-RESULT.md`](native-format-RESULT.md) | The missing-token defect: does the gain survive the fix |
| [`ladder-first-rung-RESULT.md`](ladder-first-rung-RESULT.md) | A first, one-seed look at whether learning builds upward |
| [`o2a-version-tax-RESULT.md`](o2a-version-tax-RESULT.md) | Moving from Lean 4.9 to 4.27: what it costs the prover |

Not included: the first experiment's spec and the notes on installing the Lean servers, which mostly describe
the machines the project ran on. Where a note cites "the first experiment's spec", that is the document meant.
Nor is any training text made of other people's published proofs (the ceiling's file and the pretraining
file): the notes name those files, and the tool that builds them from the pool is in `src/rlvr_lean/tools/`.
Run names such as `ladder_l2_seed0_r1`, short task ids and commit hashes refer to the private working
repository and are kept only as labels.
