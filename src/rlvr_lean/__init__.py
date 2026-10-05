"""RLVR for a Lean prover: self-generated conjectures, verified proofs, learning-progress selection.

Specs and result notes: docs/spec/ (the spec is the source of truth for the behaviour here).

Layout (domain-driven; the domain imports no infrastructure):
    domain/          pure rules - what a verified proof is, how selection scores work, the estimators
    infrastructure/  adapters behind domain-side interfaces - the Kimina HTTP client, later vLLM and PEFT
"""
