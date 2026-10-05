"""The repair check (L3a): a failed proof is cut at its first error, Lean is asked for the proof state there, and the
prover resumes from it. Spec: docs/spec/ladder-loop.spec.md, "L3a: the repair check, no training". Pure (numpy
only): text and rows in, text and rows out. The GPU steps that drive it are `rlvr_lean.gpu.ladder_l3a`.

`cut.py` holds what is done to ONE failed proof (the cut, the file that asks for the state, the state read from Lean's
answer, the prompt, the proof that is checked); `read.py` holds the read fixed before the run. `alternate.py` is the
second check (L3a2: one repair step after each fresh failure, then start over): its arms, its known copies, its read.
"""

FIRST = "first"                     # the blind first attempt of an episode: the three arms share it
BLIND = "blind"                     # the arm that goes on with whole proofs from the plain prompt
RESUME_WITH_STATE = "resume_with_state"         # the repair loop: the kept proof lines, then Lean's state as a comment
RESUME_WITHOUT_STATE = "resume_without_state"   # the same cut and kept lines, no state comment
ARMS = (BLIND, RESUME_WITH_STATE, RESUME_WITHOUT_STATE)
RESUME_ARMS = (RESUME_WITH_STATE, RESUME_WITHOUT_STATE)
