"""A published proof, written back as a Lean file our own Lean can check. Spec: docs/spec/ladder-loop.spec.md,
"A problem, and the pool".

A CERTIFICATE is a published proof of a statement (the problem is known true) or of its exact negation (known
false). It shows the problem can be resolved; it is never a training target. Four published forms are read:

  a tactic script      the Lean Workbook file's own `proof` field (InternLM): tactic lines, unindented
  tactic rows          InternLM's search results: one row per tactic with the goal before and after; a
                       published proof is ONE chain of rows ending in "no goals"
  a whole file         Goedel-Prover: header, a comment, the theorem and its proof
  a prompt and target  STP: the prover's prompt (ending at `:= by`) and what the model wrote after it

Comments are removed from every published proof before it is sent: they are prose (Goedel's hold whole
paragraphs), they can trip the lexical filter for nothing, and Lean does not read them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rlvr_lean.domain.conjecturing.dedup import strip_lean_comments
from rlvr_lean.domain.conjecturing.negation import negate_statement, split_signature
from rlvr_lean.domain.conjecturing.parsing import _first_index_outside_brackets, _is_lean_identifier
from rlvr_lean.domain.conjecturing.statement_checks import fingerprint_command
from rlvr_lean.domain.verification.lean_source import LEAN_HEADER, theorem_name_of

TRUE_SIDE, FALSE_SIDE = "true", "false"

# The published files a certificate can come from.
INTERNLM_PROOFS = "internlm_proofs"     # internlm/Lean-Workbook lean_workbook.json, field `proof`
INTERNLM_ROWS = "internlm_rows"         # internlm/Lean-Workbook wkbk_1009.parquet, status proved or disproved
GOEDEL = "goedel"                       # Goedel-LM/Lean-workbook-proofs
STP = "stp"                             # kfdong/STP_Lean_0320
SOURCES = (INTERNLM_PROOFS, INTERNLM_ROWS, GOEDEL, STP)

NO_GOALS = "no goals"
_PROOF_STUB = re.compile(r":=\s*by(\s+sorry)?\s*\Z")
_DECLARATION = re.compile(r"^(?:theorem|lemma)\s", re.MULTILINE)

# What STP's prover was shown before the theorem (its released prompts): an instruction, then a Lean file
# that starts with this header. A Mathlib row starts with that file's own copyright comment instead.
STP_PROMPT_HEAD = "Complete the following Lean 4 code:\n\n```lean4\n"
STP_HEADER = "import Mathlib\nimport Aesop\nset_option maxHeartbeats 0\nopen BigOperators Real Nat Topology Rat\n\n"


@dataclass(frozen=True)
class Certificate:
    problem_id: str
    side: str                # TRUE_SIDE: a proof of the statement; FALSE_SIDE: a proof of its exact negation
    source: str              # one of SOURCES
    theorem: str             # the theorem this proves, up to and including `:= by` and a newline
    proof: str               # the indented tactic block under it, as it is sent
    rewritten: int = 0       # places where the pin's notation rewrite changed the published text
    # The library names the renaming pass replaced in this proof (`renames.py`); empty for a proof as
    # published. It marks the certificate and is NOT part of the Lean file's shape: a renamed certificate is a
    # different text, hence a different check, by its proof alone.
    renamed: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return theorem_name_of(self.theorem)


def certificate_source(certificate: Certificate, with_fingerprint: bool) -> str:
    """The Lean file for one certificate: header, theorem, proof, then the axiom report every proof attempt
    carries. `with_fingerprint` appends the command that prints the hash of the theorem's elaborated type
    (the duplicate check's second key); it is asked of a proof of the STATEMENT, not of a negation."""
    text = f"{LEAN_HEADER}{certificate.theorem}{certificate.proof.rstrip()}\n\n#print axioms {certificate.name}\n"
    return text + fingerprint_command(certificate.name) if with_fingerprint else text


def sha_of(lean_file: str) -> str:
    """The key a check is stored and resumed by: identical files are one check."""
    return hashlib.sha256(lean_file.encode()).hexdigest()


def workbook_statement(formal_statement: str) -> str | None:
    """`theorem ... := by\\n` for a Lean Workbook row (`... := by sorry`), or None if it is not a theorem."""
    body = _PROOF_STUB.sub("", formal_statement.strip()).rstrip()
    return body + " := by\n" if body.startswith("theorem ") else None


def tactic_block(script: str, indent: str = "  ") -> str:
    """An unindented tactic script as the body of a `by` block: every line moved right by `indent`."""
    return "\n".join(indent + line.rstrip() if line.strip() else "" for line in script.strip("\n").split("\n"))


def block_after_by(text: str) -> str:
    """What a published proof wrote after `by`, as lines of their own under the theorem. A proof that starts
    on a new line keeps its indentation; one that continues the `by` line has that first tactic moved onto a
    line at the indentation of the lines after it (two spaces when it is the only line)."""
    lines = [line.rstrip() for line in text.split("\n")]
    first, rest = lines[0].strip(), [line for line in lines[1:] if line]      # blank lines (removed comments) go
    if not first:
        return "\n".join(rest)
    indent = min((len(line) - len(line.lstrip()) for line in rest), default=2)
    if indent == 0:                     # the whole script is unindented
        return tactic_block("\n".join([first] + rest))
    return "\n".join([" " * indent + first] + rest)


def chain_order(rows: list[dict]) -> list[dict] | None:
    """Tactic rows in the order they were applied, when they form ONE chain: a single row whose goal no other
    row produced, each row's `state_after` the next row's `state_before`, the last ending in "no goals", and
    every row used. None otherwise (a search tree with a branch, a published proof that is not finished)."""
    produced = {row["state_after"] for row in rows}
    roots = [row for row in rows if row["state_before"] not in produced]
    if len(roots) != 1:
        return None
    following: dict[str, list[dict]] = {}
    for row in rows:
        following.setdefault(row["state_before"], []).append(row)
    ordered = [roots[0]]
    while ordered[-1]["state_after"] != NO_GOALS:
        nexts = following.get(ordered[-1]["state_after"], [])
        if len(nexts) != 1 or len(ordered) >= len(rows):
            return None
        ordered.append(nexts[0])
    return ordered if len(ordered) == len(rows) else None


def proof_from_rows(rows: list[dict]) -> str | None:
    """The proof text of a published chain of tactic rows, or None when the rows are not one chain."""
    ordered = chain_order(rows)
    return None if ordered is None else "\n".join(tactic_block(row["tactic"]) for row in ordered)


def published_file_proof(full_proof: str, name: str) -> tuple[str, str] | None:
    """(the theorem `name` as the file states it, up to `:= by\\n`; its proof with comments removed) from a
    published Lean file, or None when the file does not declare `name` with a `by` proof."""
    code = strip_lean_comments(full_proof)
    declaration = re.search(rf"^theorem\s+{re.escape(name)}(?![\w'.!?])", code, re.MULTILINE)
    if declaration is None:
        return None
    rest = code[declaration.start():]
    assignment = _first_index_outside_brackets(rest, ":=")
    if assignment is None:
        return None
    after = rest[assignment + 2:]
    opening = re.match(r"\s*by(?![\w'])", after)
    if opening is None:
        return None
    proof = block_after_by(after[opening.end():])
    return (rest[:assignment].rstrip() + " := by\n", proof) if proof.strip() else None


def stp_statement(prompt: str) -> str | None:
    """The statement `theorem ... := by\\n` an STP prompt ends with, or None when the prompt is not ONE theorem
    under STP's own header that stops at `:= by` (a Mathlib file, a `def`, a prompt cut at `:=`)."""
    if not prompt.startswith(STP_PROMPT_HEAD + STP_HEADER):
        return None
    body = strip_lean_comments(prompt[len(STP_PROMPT_HEAD) + len(STP_HEADER):]).strip()
    if not body.startswith(("theorem ", "lemma ")) or len(_DECLARATION.findall(body)) != 1:
        return None
    assignment = _first_index_outside_brackets(body, ":=")
    if assignment is None or body[assignment + 2:].strip() != "by":
        return None
    return "\n".join(line.rstrip() for line in body[:assignment].rstrip().split("\n")) + " := by\n"


def stp_proof(target: str) -> str | None:
    """The proof an STP row's target holds (what the model wrote after the prompt's `by`), comments removed;
    None when nothing is left."""
    proof = block_after_by(strip_lean_comments(target))
    return proof if proof.strip() else None


# ------------------------------------------------------------------------------------ known false: the negation
# InternLM's "disproved" keeps the statement's hypotheses and negates only its conclusion: its rows prove
# `binders ⊢ ¬ TYPE`. That is the exact negation `¬ (∀ binders, TYPE)` only when no binder is a hypothesis
# (a statement whose hypotheses contradict each other is provable BOTH ways: 206 published cases). Whether a
# binder is a hypothesis is read twice, from the statement and from the first row's context, and both must
# say "no hypotheses"; Lean then decides whether the built certificate is a proof.
_RELATION = re.compile(r"[=<>≤≥≠∣∧∨¬∀∃∈∉↔⊆≡]|\bFact\b|Prime|Coprime|Continuous|Even\b|Odd\b|Irrational|Monotone|StrictMono|"
                       r"Injective|Surjective|Bijective|IsLeast|IsGreatest|Squarefree|Differentiable|Summable|Tendsto")
_TYPE_NOISE = re.compile(r"ℝ≥0∞|ℝ≥0|ℕ\+")
_OPENING, _CLOSING = "([{⦃", ")]}⦄"

NO_HYPOTHESES, HAS_HYPOTHESES, UNPARSED = "no_hypotheses", "has_hypotheses", "unparsed"


def is_proposition(type_text: str) -> bool:
    """A HEURISTIC: a binder's type is a proposition when it holds a relation, a connective or a known predicate."""
    return _RELATION.search(_TYPE_NOISE.sub("T", type_text)) is not None


def binder_groups(binders: str) -> list[str] | None:
    """The bracketed groups of a binder list, or None when something stands outside every bracket."""
    groups, depth, start, outside = [], 0, 0, []
    for index, character in enumerate(binders):
        if character in _OPENING:
            if depth == 0:
                start = index
            depth += 1
        elif character in _CLOSING:
            depth -= 1
            if depth == 0:
                groups.append(binders[start:index + 1])
        elif depth == 0:
            outside.append(character)
    return groups if depth == 0 and not "".join(outside).strip() else None


def _context_propositions(state_before: str) -> int:
    """How many entries of a goal's context are propositions (a wrapped entry continues on indented lines)."""
    context = state_before.split("\n⊢")[0] if "\n⊢" in state_before else ""
    entries: list[str] = []
    for line in context.split("\n"):
        if line[:1].isspace() and entries:
            entries[-1] += " " + line.strip()
        elif line.strip():
            entries.append(line)
    return sum(1 for entry in entries if " : " in entry and is_proposition(entry.split(" : ", 1)[1]))


def hypotheses_reading(statement: str, first_state_before: str) -> str:
    """NO_HYPOTHESES only when the statement's binders and the first row's context both hold no proposition;
    UNPARSED when the statement cannot be split into name, binders and type."""
    try:
        _, binders, _ = split_signature(statement)
    except ValueError:
        return UNPARSED
    groups = binder_groups(binders)
    if groups is None:
        return UNPARSED
    from_statement = sum(1 for group in groups if ":" in group and is_proposition(group[1:-1].split(":", 1)[1]))
    return NO_HYPOTHESES if from_statement == 0 and _context_propositions(first_state_before) == 0 else HAS_HYPOTHESES


def _explicit_variables(binders: str) -> list[str] | None:
    """The variable names of a binder list made only of explicit groups `(x y : T)`, in order; None otherwise."""
    groups = binder_groups(binders)
    if groups is None:
        return None
    names: list[str] = []
    for group in groups:
        colon = _first_index_outside_brackets(group[1:-1], ":")
        if not group.startswith("(") or colon is None:
            return None
        group_names = group[1:-1][:colon].split()
        if not group_names or not all(_is_lean_identifier(name) for name in group_names):
            return None
        names.extend(group_names)
    return names


def negation_certificate(statement: str, published_proof: str, negation_name: str) -> tuple[str, str] | None:
    """(the exact negation `theorem <negation_name> : ¬ (∀ BINDERS, TYPE) := by\\n`, its proof) from a
    published proof of `BINDERS ⊢ ¬ TYPE`, for a statement WITHOUT hypotheses.

    No binders: the published proof is the proof. Variables only: the published proof shows `¬ TYPE` for every
    value, the negation follows at any one value, and `default` names one (the variable's type must be
    inhabited, which Lean checks). None when a binder is not an explicit `(x : T)` group: the wrapper is not
    built, and the problem stays out of the pool."""
    _, binders, statement_type = split_signature(statement)
    theorem = negate_statement(statement, negation_name)
    if not binders:
        return theorem, published_proof
    names = _explicit_variables(binders)
    if names is None:
        return None
    values = " ".join("default" for _ in names)
    proof = (f"  have published_for_every_value {binders} : ¬ ({statement_type}) := by\n"
             f"{tactic_block(published_proof)}\n"
             f"  intro claimed_for_every_value\n"
             f"  exact published_for_every_value {values} (claimed_for_every_value {values})")
    return theorem, proof


def negation_of(problem_id: str) -> str:
    """The name of a problem's exact negation, as a theorem."""
    return f"negation_of_{problem_id}"
