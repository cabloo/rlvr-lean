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
| [`ladder-l4-RESULT.md`](ladder-l4-RESULT.md) | The same six rounds from a model pretrained on published proofs: not shown on top of pretraining, one seed; corrected 2026-10-11: "more reliable and narrower" is one training against one, not established |
| [`ladder-l4r-RESULT.md`](ladder-l4r-RESULT.md) | Is the pretrained model capped by the size of its adapter? The same pretraining at rank 64 in the place of 16, one seed; corrected 2026-10-11: one pretraining against one, "the adapter's size capped it" is not established |
| [`ladder-l4t-RESULT.md`](ladder-l4t-RESULT.md) | What should a round train on? Three training rules on the stored rounds, from the larger pretrained model, one seed; corrected 2026-10-11: one training against one, "the old rule narrows at rank 64 too" is not established |
| [`ladder-l4t2-RESULT.md`](ladder-l4t2-RESULT.md) | Two more one-change checks on the stored rounds, from the larger pretrained model, one seed; corrected 2026-10-11: one training against one, "the gain is a second pass over the published proofs" is not established |
| [`ladder-l4p-RESULT.md`](ladder-l4p-RESULT.md) | How much more pretraining? Part 1, the matched control: the model's own proofs of hard problems are not separated from the same published proofs alone. Part 2, the dose of a second pass: inconclusive by its own check; one seed; corrected 2026-10-11: one training against one, "a whole second pass is the best model" is not established, and the note's estimate of the spread between two trainings (about 6 per 1,000 and 5 goal problems) was several times too small |
| [`ladder-l4c-RESULT.md`](ladder-l4c-RESULT.md) | Six fresh rounds from the larger pretrained model with a control trained on the same published rows alone: the gain is not shown to be the rounds'; not a verdict on the idea, one seed; corrected 2026-10-11: one training against one, that the model's own proofs narrow it is not established, and neither is the contrary |
| [`ladder-l5-RESULT.md`](ladder-l5-RESULT.md) | Do deep solves teach what ordinary solves do not? Three seeds: not shown on the held-aside problems, and depth not shown to matter; on the goal set the models trained on the model's own proofs beside published ones were above the model trained on published proofs alone at three seeds of three, found on those seeds and not yet confirmed; corrected 2026-10-11: what did not repeat from seed 0 is the deep-solves model ahead of the ordinary-solves one |
| [`ladder-l5n-RESULT.md`](ladder-l5n-RESULT.md) | What is one training worth? One fixed set of 800 published proofs in four orders: a standard deviation of 5.0 successes per 1,000 attempts from the order alone, 15.2 between three draws of 800 proofs; a falling learning rate not shown to lower it; the average of four adapters gives at least their mean |
| [`ladder-cycle.spec.md`](ladder-cycle.spec.md) | The deep-search cycle at full size (L7 in the loop's numbering): a spec that is PROPOSED, not approved and not built |
| [`phase-a-RESULT.md`](phase-a-RESULT.md) | The first experiment: one round on self-written conjectures |
| [`phase-b-selection-RESULT.md`](phase-b-selection-RESULT.md) | Selection by a learning-progress score against a random draw, and its correction |
| [`native-format-RESULT.md`](native-format-RESULT.md) | The missing-token defect: does the gain survive the fix |
| [`ladder-first-rung-RESULT.md`](ladder-first-rung-RESULT.md) | A first, one-seed look at whether learning builds upward |
| [`o2a-version-tax-RESULT.md`](o2a-version-tax-RESULT.md) | Moving from Lean 4.9 to 4.27: what it costs the prover |

Not included: the first experiment's spec and the notes on installing the Lean servers, which mostly describe
the machines the project ran on. Where a note cites "the first experiment's spec", that is the document meant.
Nor is the note of the challenger test that `ladder-cycle.spec.md` cites.
Nor is any training text made of other people's published proofs (the ceiling's file and the pretraining
file): the notes name those files, and the tool that builds them from the pool is in `src/rlvr_lean/tools/`.
Run names such as `ladder_l2_seed0_r1`, short task ids and commit hashes refer to the private working
repository and are kept only as labels.
