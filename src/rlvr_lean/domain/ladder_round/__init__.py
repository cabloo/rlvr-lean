"""The ladder loop's ROUND: the challenger that chooses problems, the training set a round gives, and L1's read.
Spec: docs/spec/ladder-loop.spec.md, "A round", "The challenger in this stage", "L1's read, fixed now". Pure
(numpy only): rows in, rows out. The GPU steps that drive it are `rlvr_lean.gpu.ladder_round`.
"""

CHALLENGER_ARM = "challenger"       # the challenger chooses the round's problems
RANDOM_ARM = "random"               # the equal-compute control: a random draw of the pool, the same number of episodes
ARMS = (CHALLENGER_ARM, RANDOM_ARM)
