"""The accumulating episode (L3c): an episode that keeps what verified. Spec: docs/spec/ladder-loop.spec.md,
"L3c: an episode that keeps what verified (no training)" and its "Made exact before the build (2026-10-07)". Pure:
text in, pool out; rows in, numbers out. The cut at the first error, the file that asks Lean for a state and the
prover's comment format are L3a's (`cut.py`); the known copies are L3a2's (`alternate.py`).

THE EPISODE UNDER TEST ("accumulate"). Up to `generations` generations, stopping at the first verified proof. It holds a
POOL: an ordered list of verified lemmas (top-level `have` steps), empty at the start, which only grows.

  fresh       a whole proof from the plain prompt. Generation 1 is fresh, and so is any generation whose pool is empty
              or has not changed since the last continuing one
  continue    the plain prompt, the pool's lemmas as the proof so far, and Lean's state after them in the prover's own
              comment format; the model continues, and the proof checked is the pool and the continuation
  harvest     from a failed proof, its leading top-level `have` steps that end before its first error (of a
              continuing proof: the leading `have` steps of the CONTINUATION). They join the pool under names made
              unique to their generation (`h₁` of generation 3 is `h₁_g3`; an anonymous one stays anonymous). A step that
              binds a pattern, a `have` with no proof of its own, or anything that is not a `have`, ends the harvest. A
              statement the pool already holds is not added, and the name it bound is read as the pooled lemma's in
              what follows. A harvest that would take the pool past its size is cut at its tail
  pool check  whenever the pool changes: the pool, then `all_goals sorry`. It stands when Lean reports no error at all
              and exactly one goal at the `sorry` (the theorem's own, with the pooled facts as hypotheses): that goal is
              the state the next continuing generation is shown. Otherwise the newest harvest is taken back out
  closers     a failed proof that was lemmas and then ONE closing step leaves that step as a kept closer, its names read
              as the pooled ones. Whenever the pool has grown (or a closer was just kept), each kept closer whose pooled
              names all exist is checked after the pool. A proof that verifies this way resolves the episode: ASSEMBLED

The other arm is BLIND: whole proofs from the plain prompt throughout. A sampling seed does not depend on the arm, so a
fresh generation of the accumulate arm is the blind arm's own sample at that position while both are open.

KNOWN COPIES (L3a2's rule): a proof whose text Lean has already rejected in the episode is not sent again. In the
accumulate arm that holds for an assembled proof too, and the texts Lean has rejected include the assembled proofs it
rejected. ONE pair is known without its text being equal: the pool right after a proof's own harvest, then that proof's
own closer, is the proof Lean has just rejected under its pooled names (`OWN_PROOF`); it is not sent.

  top_steps, bound_name, harvest        a failed proof's leading lemmas and its closing step
  pool_source, read_pool                the file that checks a pool, and what Lean's answer says of it
  closers_to_check, assembled_proof     which kept closers are tried after a pool, and the proof each makes
  episode_state, kind_of                what one episode holds before its next generation, from its stored rows
  episode_rows, problem_rows            one row an episode, one a problem
  accumulate_read                       the primary, reach, the running totals, the budget, the checks, the branch
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.ladder_round.rounds import sign_test
from rlvr_lean.domain.problem_pool.episodes import NO_ANSWER, VERIFIED
from rlvr_lean.domain.repair import BLIND, FIRST
from rlvr_lean.domain.repair import cut as rules
from rlvr_lean.domain.repair.alternate import KNOWN_COPY, checked_text, rejected
from rlvr_lean.domain.repair.cut import _CLOSING, _OPENING
from rlvr_lean.domain.repair.read import (
    BLIND_ATTEMPT,
    HALF,
    HARD,
    INCONCLUSIVE,
    MORE_TOKENS_ALLOWED,
    NOT_SHOWN,
    RESUMED,
    SETS,
    Branch,
    _of,
    _share,
    gained_against_lost,
    paired,
    resolved,
    resolved_by_attempt,
)
from rlvr_lean.domain.verification.lean_source import _strip_comments

ACCUMULATE = "accumulate"           # the episode under test: it keeps a pool of verified lemmas
ARMS = (BLIND, ACCUMULATE)
FRESH, CONTINUE = "fresh", "continue"       # the kind of a generation: a whole proof, or one that goes on from the pool
ASSEMBLED = "assembled"             # how an episode was resolved: a kept closer verified after the pool (nothing generated)
POOL_INDENTATION = rules.DEFAULT_INDENTATION        # every pooled lemma, the `sorry` after them and the state comment stand here

# A line at the steps' own indentation that begins with one of these goes on with the step before it.
GOES_ON = ("<;>", "·", "|", ".", "all_goals", "any_goals")

# A pool check's outcome: `STANDS`, or why the newest harvest is taken back out (also one of `cut.NO_STATE_REASONS`).
STANDS = "state"                                    # no error, one goal at the `sorry`: Lean's state after the pool
ERROR_IN_THE_POOL = "error_in_the_pool"             # an "unsolved goals" around the `sorry`: a pooled `have` was left open
SEVERAL_GOALS = "several_goals"                     # more than one goal after the pool: a pooled step opened a goal of its own
# A closer check that was not sent: `alternate.KNOWN_COPY` (Lean already rejected that text in the episode), or
OWN_PROOF = "own_proof"                             # the pool right after a proof's own harvest, then that proof's own closer
# Why a failed proof left the pool nothing (also `KNOWN_COPY`, and `cut.NO_ERROR_POSITION`, `cut.ERROR_ON_THE_STATEMENT`).
LAYOUT = "layout_not_understood"                    # a line stands left of the proof's first step
OFF_THE_POOL = "not_at_the_pools_indentation"       # a continuation whose first step does not stand where the pool's lemmas do
ERROR_IN_THE_POOLS_LINES = "error_in_the_pools_lines"       # a continuing proof whose first error lies above the continuation
# Where a harvest ended.
THE_END, THE_ERROR, NOT_A_HAVE, NO_PROOF, HAVE_GOES_ON = "the_end", "the_error", "not_a_have", "have_without_a_proof", "have_that_goes_on"

MINIMUM_POOL_SHARE = 0.30           # "a non-empty pool in at least 30% of hard episodes by their last generation"
MINIMUM_STANDING_SHARE = 0.95       # "the pool as harvested stands with `sorry` in at least 95% of the checks made of it"
LONG_PROOF = 8                      # lines: the read gives the share of verified proofs with at least this many

THE_SOLVERS_EPISODE = "THE ACCUMULATING EPISODE IS THE SOLVER'S EPISODE FROM HERE"      # primary above zero, interval clear of zero
REACHES = "THE EPISODE REACHES PROBLEMS WITHOUT RESOLVING MORE EPISODES"                # reach ahead, the primary's interval through zero
# Not one of the spec's three branches (its text names none for an interval that lies below zero): reported as that.
RESOLVES_FEWER = "THE ACCUMULATING EPISODE RESOLVES FEWER EPISODES THAN BLIND ATTEMPTS"
BRANCHES = (THE_SOLVERS_EPISODE, NOT_SHOWN, REACHES, RESOLVES_FEWER, INCONCLUSIVE)      # NOT_SHOWN: the interval contains zero, reach not ahead
REACH_AHEAD_BELOW = 0.05            # "reach ahead (gained more than lost, sign test under 0.05)"

_HEAD = re.compile(r"(\s*have)(?![\w'])(\s*)([^\s:(\[{⟨]*)")       # a step's first line: `have`, then the name it binds (or none)
_NAME = re.compile(r"[\w'₀-₉.!?]+")


# ----------------------------------------------------------------------------------------------- the steps
def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def top_steps(lines: Sequence[str]) -> tuple[list[str], int, list[tuple[int, int, bool]]] | None:
    """(the lines with their comments taken out, the indentation of the proof's own steps, its top-level steps), or
    None when the layout is not understood (a line of code left of the first one). A step is `(start, end, goes on)`:
    the lines `start` to `end` - 1, and whether one of them after the first stands at the steps' own indentation (it
    begins with one of `GOES_ON`: `have h : P := by simp` and then `<;> linarith` is not a lemma alone). A blank line
    and a line that is only a comment are no step and end none."""
    code = [line.rstrip() for line in _strip_comments("\n".join(lines)).split("\n")]
    base = next((_indent(line) for line in code if line.strip()), None)
    if base is None:
        return None
    steps: list[list] = []
    for index, line in enumerate(code):
        if not line.strip():
            continue
        indent = _indent(line)
        if indent < base:
            return None
        if indent == base and not line.lstrip().startswith(GOES_ON):
            steps.append([index, index + 1, False])
        elif steps:
            steps[-1][1], steps[-1][2] = index + 1, steps[-1][2] or indent == base
        else:
            return None
    return code, base, [(start, end, goes_on) for start, end, goes_on in steps]


def token(name: str) -> re.Pattern:
    """A name where it stands as a whole name: not inside a longer one, not as a field (`x.h`)."""
    return re.compile(r"(?<![\w'.])" + re.escape(name) + r"(?![\w'])")


def renamed(text: str, names: Mapping[str, str]) -> str:
    """`text` with every name of `names` read as its new one, in ONE pass: a new name is never read again as an old."""
    if not names:
        return text
    pattern = re.compile(r"(?<![\w'.])(?:" + "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True)) + r")(?![\w'])")
    return pattern.sub(lambda found: names[found.group(0)], text)


def bound_name(first: str) -> str | None:
    """The name a `have` step binds, from its first line: a name, "" for an anonymous one (`have : P := ...` binds
    `this`), None when the step is not a `have` that binds one name (a pattern such as `have ⟨x, hx⟩ := h`, `haveI`,
    another tactic)."""
    head = _HEAD.match(first)
    if head is None:
        return None
    if not head.group(3):
        return "" if first[head.end():].lstrip().startswith(":") else None
    return head.group(3) if _NAME.fullmatch(head.group(3)) else None


def _assigned_at(text: str) -> int | None:
    """Where the first `:=` outside every bracket begins: what a `have` states ends there and its proof begins."""
    depth = 0
    for index, character in enumerate(text):
        if character in _OPENING:
            depth += 1
        elif character in _CLOSING:
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(":=", index):
            return index
    return None


def _squeezed(text: str) -> str:
    return " ".join(text.split())


def proof_line_count(proof: str) -> int:
    """A proof's lines as the published proofs' are counted (`data/heldout_proof_lines.py`): the lines that are
    neither empty nor a comment."""
    return len([line for line in proof.strip("\n").split("\n") if line.strip() and not line.strip().startswith("--")])


# ----------------------------------------------------------------------------------------------- the harvest
@dataclass(frozen=True)
class Harvest:
    """What one failed proof leaves its episode (spec items 2 and 3)."""
    blocks: tuple[dict, ...] = ()       # the lemmas that join the pool, in the proof's order: each `{text, name, statement, generation}`
    duplicates: int = 0                 # leading lemmas whose statement the pool already held: not added, their names read as the pooled ones
    over_the_size: int = 0              # leading lemmas the pool had no room for: the tail of the harvest
    closer: dict | None = None          # the ONE closing step after the lemmas, `{text, needs, generation}`; None: the proof was not lemmas and then one step
    own_proof: bool = False             # the pool before, these lemmas and this closer ARE the failed proof, under its pooled names
    ended: str = THE_END                # where the harvest ended (`THE_ERROR`, `NOT_A_HAVE`, ...), or why there was none (`LAYOUT`, `OFF_THE_POOL`)


def harvest(written: str, kept: int, pool: Sequence[Mapping], generation: int, size: int, continues: bool = False) -> Harvest:
    """The lemmas and the closing step one failed proof leaves (spec items 2 and 3; this module's docstring).

    `written`: what the steps are read from: the whole proof of a fresh generation; of a continuing one (`continues`)
    what the model wrote after the pool. `kept`: how many of its lines lie before the first error (L3a's cut:
    `Cut.kept_lines`, less the pool's lines in a continuing proof). `pool`: the lemmas held so far. `size`: the blocks a
    pool may hold (`accumulate.pool_blocks`).

    A lemma is a top-level `have` that binds one name or none, has a proof of its own (a `:=` outside every bracket)
    and ends before the first error. Its text is its lines of code (comments and blank lines left out) at the pool's
    indentation, with the name it binds made unique to the generation (`h` is `h_g<generation>`; bound a second time
    in the same proof, `h_g<generation>'`) and every lemma name of the proof above it read as the pooled one. Two
    lemmas are the same when both are named, or both anonymous, and state the same thing (the text between the name
    and `:=`, white space squeezed; of a `have h := t`, the term): the second is not added and its name is read as the
    first's. An anonymous one is added all the same when the proof goes on to say `this`.

    The CLOSER: when every step but the last was such a lemma and the last is not a `have`, that last step, with the
    lemma names read as the pooled ones. `needs` names the pooled lemmas of this proof it mentions: where one of them
    did not reach the pool the closer is not tried."""
    parsed = top_steps(rules.proof_lines(written))
    if parsed is None:
        return Harvest(ended=LAYOUT)
    code, base, steps = parsed
    if continues and base != len(POOL_INDENTATION):
        return Harvest(ended=OFF_THE_POOL)
    shift = len(POOL_INDENTATION) - base

    def placed(text: str) -> str:
        return "\n".join(" " * shift + line if shift >= 0 else line[-shift:] for line in text.split("\n"))

    held: dict[tuple[bool, str], str] = {}
    for block in pool:
        held.setdefault((bool(block["name"]), block["statement"]), block["name"])
    names: dict[str, str] = {}
    blocks, bound, duplicates, over, taken, ended = [], set(), 0, 0, 0, THE_END
    for start, end, goes_on in steps:
        lines = [code[index] for index in range(start, end) if code[index].strip()]
        name = bound_name(lines[0])
        if name is None or end > kept or goes_on:
            ended = NOT_A_HAVE if name is None else THE_ERROR if end > kept else HAVE_GOES_ON
            break
        head = _HEAD.match(lines[0])
        rest = renamed("\n".join([lines[0][head.end():], *lines[1:]]), names)
        at = _assigned_at(rest)
        if at is None:
            ended = NO_PROOF
            break
        statement = _squeezed(rest[:at]) or ":= " + _squeezed(rest[at + 2:])
        taken += 1
        if not name:
            names.pop("this", None)             # from here `this` is this lemma, whatever an earlier `have this` bound
        known = held.get((bool(name), statement))
        if known is not None and (name or not any(token("this").search(line) for line in code[end:])):
            duplicates += 1
            if name:
                names[name] = known
            continue
        pooled = ""
        if name:
            pooled = f"{name}_g{generation}"
            while pooled in bound:
                pooled += "'"
            names[name] = pooled
        if len(pool) + len(blocks) >= size:
            over += 1
            continue
        bound.add(pooled)
        held.setdefault((bool(name), statement), pooled)
        blocks.append({"text": placed(lines[0][:head.start(3)] + pooled + rest), "name": pooled, "statement": statement, "generation": generation})
    closer = None
    if steps and taken == len(steps) - 1 and not code[steps[-1][0]].lstrip().startswith("have"):
        start, end, _ = steps[-1]
        text = placed(renamed("\n".join(code[index] for index in range(start, end) if code[index].strip()), names))
        closer = {"text": text, "needs": sorted({new for new in names.values() if token(new).search(text)}), "generation": generation}
    return Harvest(tuple(blocks), duplicates, over, closer, closer is not None and not duplicates and not over and (continues or not pool), ended)


# -------------------------------------------------------------------------------------------------- the pool
def pool_lines(pool: Sequence[Mapping]) -> list[str]:
    return [line for block in pool for line in block["text"].split("\n")]


def pool_text(pool: Sequence[Mapping]) -> str:
    return rules.kept_text(pool_lines(pool))


def pooled_names(pool: Sequence[Mapping]) -> set[str]:
    return {block["name"] for block in pool if block["name"]}


def pool_source(statement: str, pool: Sequence[Mapping]) -> tuple[str, int, int]:
    """(the Lean file that checks a pool, the 1-based line of its `sorry`, its 0-based column): the pool's lemmas, then
    `all_goals sorry` where they stand. It is L3a's state file with the pool as the kept lines."""
    return rules.state_source(statement, rules.Cut(rules.WHOLE_PROOF, tuple(pool_lines(pool)), POOL_INDENTATION, None, {}))


def read_pool(raw: Mapping[str, Any], line: int, column: int) -> tuple[str | None, str]:
    """(Lean's state after the pool, `STANDS`), or (None, why the pool does not stand). L3a's `read_state`, and
    stricter: a pool is lemmas above the theorem's own goal, so its file has NO error (L3a allows the "unsolved goals"
    of a block the `sorry` is inside, which here is a pooled `have` left open) and exactly ONE goal at the `sorry` (a
    step that opened a goal of its own is not a lemma). A file with no goal left there closed the theorem's goal: not
    lemmas either (`cut.NO_GOALS_AT_THE_CUT`)."""
    state, why = rules.read_state(raw, line, column)
    if why is not None:
        return None, why
    if rules.errors_of(raw):
        return None, ERROR_IN_THE_POOL
    goals = 0
    for entry in raw["response"].get("sorries") or []:
        position = entry.get("pos") if isinstance(entry.get("pos"), Mapping) else {}
        goals += (position.get("line"), position.get("column")) == (line, column) and bool(str(entry.get("goal") or "").strip())
    return (state, STANDS) if goals == 1 else (None, SEVERAL_GOALS)


def closers_to_check(pool: Sequence[Mapping], closers: Sequence[Mapping], grew: bool, kept_now: Mapping | None) -> list[Mapping]:
    """The kept closers that are checked after `pool` now: every one when the pool has just grown, the one just kept
    when it has not (the others were checked after this pool already); of those, the ones whose pooled names all
    exist. None with an empty pool: there is nothing to check a closer after."""
    if not pool:
        return []
    names = pooled_names(pool)
    return [closer for closer in (closers if grew else [kept_now] if kept_now is not None else []) if set(closer["needs"]) <= names]


def assembled_proof(pool: Sequence[Mapping], closer: Mapping) -> str:
    """The proof a kept closer makes after a pool: the pool's lemmas, then the closing step."""
    return rules.resumed_proof(pool_lines(pool), closer["text"] + "\n")


# ---------------------------------------------------------------------------------------------- the episode
def empty_state() -> dict:
    """What the accumulate arm holds before its first generation."""
    return {"pool": [], "state": None, "made_at": None, "changed": False, "closers": [], "rejected": {}, "assembled": {}}


def is_refused(row: Mapping) -> bool:
    """Whether a closer check's proof counts as one Lean has rejected: it was sent and failed, or it is the proof Lean
    had just rejected under its pooled names (`OWN_PROOF`)."""
    return row["status"] == OWN_PROOF or (bool(row["sent_to_lean"]) and row["status"] not in (VERIFIED, NO_ANSWER))


def episode_state(chain: Sequence[Mapping], pools: Sequence[Mapping], closers: Sequence[Mapping]) -> dict:
    """What the accumulate arm of one episode holds before its next generation, from its stored rows: `chain`, the
    first attempt and the arm's attempts after it; `pools`, the episode's pool rows; `closers`, its closer-check rows.

      pool       the blocks of the latest pool state that stands (a harvest taken back out left the one before it)
      state      Lean's state after that pool
      made_at    the loop whose harvest made that pool state (None: the pool is empty)
      changed    whether the pool has changed since the last continuing generation (or none has continued yet)
      closers    the kept closers, in the order they were kept
      rejected   the proofs Lean has rejected in the episode: text -> the loop it was rejected at
      assembled  those of them that were never an attempt's own text: proofs assembled from the pool and a kept closer"""
    standing = [row for row in pools if row["stands"]]
    continued = [row["loop"] for row in chain if row.get("kind") == CONTINUE]
    refused = rejected(chain)
    assembled = {}
    for row in closers:
        if is_refused(row) and checked_text(row["proof"]) not in refused:
            assembled.setdefault(checked_text(row["proof"]), row["loop"])
    return {"pool": standing[-1]["blocks"] if standing else [], "state": standing[-1]["state"] if standing else None,
            "made_at": standing[-1]["loop"] if standing else None,
            "changed": bool(standing) and (not continued or standing[-1]["loop"] >= continued[-1]),
            "closers": [row["kept_closer"] for row in chain if row.get("kept_closer")], "rejected": {**assembled, **refused}, "assembled": assembled}


def kind_of(state: Mapping) -> str:
    """Spec item 5: a generation continues when the pool is not empty and has changed since the last continuing
    generation (an unchanged pool would be shown the same prompt again), and is fresh otherwise."""
    return CONTINUE if state["pool"] and state["changed"] else FRESH


def _generation(row: Mapping) -> dict:
    """One generation of one arm as the read uses it."""
    return {"loop": row["loop"], "kind": row["kind"], "how": row["how"], "had_state": bool(row["had_state"]), "no_state": row["no_state"],
            "status": row["status"], "token_count": row["token_count"], "prompt_tokens": row["prompt_tokens"], "sent_to_lean": bool(row["sent_to_lean"]),
            "pool_blocks": row["pool_blocks"], "shared": bool(row["shared"])}


def episode_rows(problems: Sequence[Mapping], attempts: Sequence[Mapping], pools: Sequence[Mapping], closers: Sequence[Mapping],
                 generations: int) -> list[dict]:
    """One row for every episode on the side its problem's certificate allows: how and when each arm resolved it.
    `attempts`: every stored attempt row (loop 0 is generation 1, arm `first`, which the arms share; loops 1 to
    `generations` - 1 are the arms'); `pools`, `closers`: every stored pool state and closer check. Refused unless
    each arm went on from every failed first generation to the generation that resolved it or to the last one.

    An arm's `resolved_at` counts generations (1 is the shared first one) and `resolved_by` says how: a `fresh` proof
    (a whole proof from the plain prompt), a `continue` generation (the pool in its prompt), or, in the accumulate arm,
    an `assembled` proof: a kept closer that verified after the pool, counted within the generation after whose
    harvest it was found. The accumulate arm also carries its pool checks and its closer checks, generation by
    generation (those made after generation 1 are at loop 0), and the assembled proof's text."""
    firsts: dict[tuple[str, int], Mapping] = {}
    later: dict[tuple[str, str, int, int], Mapping] = {}
    for row in attempts:
        if row.get("audit"):
            continue
        if row["arm"] == FIRST:
            firsts[row["problem_id"], row["episode"]] = row
        else:
            later[row["arm"], row["problem_id"], row["episode"], row["loop"]] = row
    of_pools: dict[tuple[str, int], list[Mapping]] = {}
    for row in pools:
        of_pools.setdefault((row["problem_id"], row["episode"]), []).append(row)
    of_closers: dict[tuple[str, int], list[Mapping]] = {}
    for row in closers:
        of_closers.setdefault((row["problem_id"], row["episode"]), []).append(row)
    found = []
    for problem in problems:
        for episode in sorted(episode for problem_id, episode in firsts if problem_id == problem["problem_id"]):
            key = (problem["problem_id"], episode)
            first = firsts[key]
            assembled = {row["loop"]: row for row in reversed(of_closers.get(key, [])) if row["status"] == VERIFIED}
            entry = {"problem_id": problem["problem_id"], "group": problem["group"], "side": problem["side"], "episode": episode,
                     "failed_first": first["status"] != VERIFIED,
                     "first": {**{name: first[name] for name in ("status", "token_count", "prompt_tokens", "sent_to_lean")},
                               "proof_lines": proof_line_count(first["proof"]) if first["status"] == VERIFIED else None}, "arms": {}}
            for arm in ARMS:
                own: dict = {"resolved_at": None, "resolved_by": None, "proof_lines": None, "loops": []}
                if not entry["failed_first"]:
                    own.update({"resolved_at": 1, "resolved_by": FRESH, "proof_lines": entry["first"]["proof_lines"]})
                for loop in range(generations):
                    if own["resolved_at"] is not None:
                        break
                    if loop:
                        row = later.get((arm, *key, loop))
                        if row is None:
                            raise ValueError(f"{key[0]} episode {episode}: the {arm} arm has no generation at loop {loop} and had not resolved "
                                             "before it: the read needs whole episodes")
                        own["loops"].append(_generation(row))
                        if row["status"] == VERIFIED:
                            own.update({"resolved_at": loop + 1, "resolved_by": CONTINUE if row["how"] == RESUMED else FRESH,
                                        "proof_lines": proof_line_count(row["proof"])})
                    if arm == ACCUMULATE and own["resolved_at"] is None and loop in assembled:
                        own.update({"resolved_at": loop + 1, "resolved_by": ASSEMBLED, "proof_lines": proof_line_count(assembled[loop]["proof"])})
                if arm == ACCUMULATE:
                    standing = [row for row in of_pools.get(key, []) if row["stands"]]
                    chain = [first, *(later[arm, *key, step["loop"]] for step in own["loops"])]
                    own.update({
                        "assembled_proof": assembled[own["resolved_at"] - 1]["proof"] if own["resolved_by"] == ASSEMBLED else None,
                        "pool_blocks": len(standing[-1]["blocks"]) if standing else 0,
                        # The lemmas of a generation's harvest, the pool states it made and what Lean said of each.
                        "pool_checks": [{"loop": row["loop"], "outcome": row["outcome"], "stands": bool(row["stands"]), "blocks": len(row["blocks"]),
                                         "harvested": row["harvested"]} for row in of_pools.get(key, [])],
                        "kept_closers": sum(bool(row.get("kept_closer")) for row in chain),
                        "closer_checks": [{"loop": row["loop"], "status": row["status"], "sent_to_lean": bool(row["sent_to_lean"]),
                                           "from_generation": row["from_generation"], "pool_blocks": row["pool_blocks"]} for row in of_closers.get(key, [])]})
                entry["arms"][arm] = own
            found.append(entry)
    return found


def problem_rows(rows: Sequence[Mapping], generations: int) -> list[dict]:
    """One row a problem: its episodes, how many failed at the first generation, how many each arm resolved, and how
    many of the accumulate arm's were resolved by a continuing generation and by an assembled proof."""
    by_problem: dict[str, list[Mapping]] = {}
    for row in rows:
        by_problem.setdefault(row["problem_id"], []).append(row)
    return [{"problem_id": problem_id, "group": own[0]["group"], "side": own[0]["side"], "episodes": len(own),
             "failed_first": sum(row["failed_first"] for row in own),
             **{f"resolved_{arm}": sum(resolved(row, arm, generations) for row in own) for arm in ARMS},
             **{f"{how}_{ACCUMULATE}": sum(row["arms"][ACCUMULATE]["resolved_by"] == how for row in own) for how in (CONTINUE, ASSEMBLED)},
             "with_a_pool": sum(row["arms"][ACCUMULATE]["pool_blocks"] > 0 for row in own)}
            for problem_id, own in by_problem.items()]


# ------------------------------------------------------------------------------------------ running totals
def running_totals(rows: Sequence[Mapping], generations: int) -> dict:
    """Episodes resolved WITHIN k generations, k = 1 to `generations`, as counts by arm: the only way the arms are
    compared (the correction of 2026-10-06: after the first generation the two arms hold different open
    episodes, so a rate among the episodes each has left compares different survivors). With each: the episodes only
    one arm resolved within k, and the two-sided sign test on those two counts."""
    result = {}
    for within in range(1, generations + 1):
        pairs = [(resolved(row, ACCUMULATE, within), resolved(row, BLIND, within)) for row in rows]
        only_first, only_second = sum(first and not second for first, second in pairs), sum(second and not first for first, second in pairs)
        result[str(within)] = {ACCUMULATE: sum(first for first, _ in pairs), BLIND: sum(second for _, second in pairs),
                               "lead": sum(first for first, _ in pairs) - sum(second for _, second in pairs),
                               f"only_{ACCUMULATE}": only_first, f"only_{BLIND}": only_second, "sign_test_p": sign_test(only_first, only_second)}
    return result


def resolutions(rows: Sequence[Mapping], generations: int) -> dict:
    """How the accumulate arm's resolved episodes came: a fresh proof (and how many of those at the first generation,
    which the arms share), a continuing generation, an assembled proof. Beside it, the blind arm's count."""
    own = [row["arms"][ACCUMULATE] for row in rows if resolved(row, ACCUMULATE, generations)]
    by = Counter(arm["resolved_by"] for arm in own)
    return {"episodes": len(rows), "resolved": len(own), FRESH: by[FRESH], "of_which_at_the_first_generation": sum(arm["resolved_at"] == 1 for arm in own),
            CONTINUE: by[CONTINUE], ASSEMBLED: by[ASSEMBLED], f"resolved_{BLIND}": sum(resolved(row, BLIND, generations) for row in rows)}


def _lines(counts: Sequence[int]) -> dict:
    if not counts:
        return {"proofs": 0, "median": None, "mean": None, "share_with_8_lines_or_more": None, "longest": None}
    return {"proofs": len(counts), "median": statistics.median(counts), "mean": round(sum(counts) / len(counts), 3),
            "share_with_8_lines_or_more": _share(sum(count >= LONG_PROOF for count in counts), len(counts)), "longest": max(counts)}


def verified_lines(rows: Sequence[Mapping], generations: int) -> dict:
    """The line counts of the proofs each arm verified (the proof that resolved each episode; lines counted as the
    published proofs' are): every one, those found after the first generation (the first is the same proof in both
    arms), and the accumulate arm's by how it resolved."""
    result = {}
    for arm in ARMS:
        own = [row["arms"][arm] for row in rows if resolved(row, arm, generations)]
        result[arm] = {**_lines([entry["proof_lines"] for entry in own]),
                       "after_the_first_generation": _lines([entry["proof_lines"] for entry in own if entry["resolved_at"] > 1])}
    result[ACCUMULATE]["by_how"] = {how: _lines([row["arms"][ACCUMULATE]["proof_lines"] for row in rows if resolved(row, ACCUMULATE, generations)
                                                 and row["arms"][ACCUMULATE]["resolved_by"] == how and row["arms"][ACCUMULATE]["resolved_at"] > 1])
                                    for how in (FRESH, CONTINUE, ASSEMBLED)}
    return result


# ------------------------------------------------------------------------------------------ what the arm did
def by_generation(rows: Sequence[Mapping], generations: int) -> dict:
    """By arm and generation: the generations made (the episodes that arm still had open), how many verified, how many
    were known copies; for the accumulate arm also their kinds, and the episodes an assembled proof resolved after
    that generation's harvest. NOT a comparison of the arms: after generation 1 the two arms hold different open
    episodes. It says what a generation was and did, in its own arm."""
    result: dict = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for loop in range(1, generations):
            steps = [step for row in rows for step in row["arms"][arm]["loops"] if step["loop"] == loop]
            entry = {"generations": len(steps), "verified": sum(step["status"] == VERIFIED for step in steps),
                     "known_copies": sum(step["status"] == KNOWN_COPY for step in steps)}
            if arm == ACCUMULATE:
                held = [step for step in steps if step["how"] == RESUMED]
                entry.update({FRESH: sum(step["kind"] == FRESH for step in steps), CONTINUE: sum(step["kind"] == CONTINUE for step in steps),
                              "continuing_with_the_pool_in_the_prompt": len(held), "of_which_verified": sum(step["status"] == VERIFIED for step in held),
                              "of_which_known_copies": sum(step["status"] == KNOWN_COPY for step in held),
                              "shared_with_the_blind_arm": sum(step["shared"] for step in steps),
                              "resolved_by_an_assembled_proof_after_it": sum(own["resolved_by"] == ASSEMBLED and own["resolved_at"] == loop + 1
                                                                             for own in (row["arms"][ACCUMULATE] for row in rows))})
            result[arm][str(loop + 1)] = entry
    return result


def second_generation(rows: Sequence[Mapping]) -> dict:
    """The first continuing generation, like for like: on the episodes whose first generation failed and whose second
    generation in the accumulate arm had the pool in its prompt, that generation against the blind arm's second on the
    SAME episodes (every one of them is open in both arms: nothing but the second generation differs yet)."""
    held = [row for row in rows if row["failed_first"] and row["arms"][ACCUMULATE]["loops"] and row["arms"][ACCUMULATE]["loops"][0]["how"] == RESUMED]
    pairs = [(row["arms"][ACCUMULATE]["loops"][0]["status"] == VERIFIED, row["arms"][BLIND]["loops"][0]["status"] == VERIFIED) for row in held]
    only_first, only_second = sum(first and not second for first, second in pairs), sum(second and not first for first, second in pairs)
    return {"failed_first_generations": sum(row["failed_first"] for row in rows), "with_the_pool_in_the_second_prompt": len(held),
            f"verified_{ACCUMULATE}": sum(first for first, _ in pairs), f"verified_{BLIND}": sum(second for _, second in pairs),
            f"only_{ACCUMULATE}": only_first, f"only_{BLIND}": only_second, "sign_test_p": sign_test(only_first, only_second),
            "known_copies": sum(row["arms"][ACCUMULATE]["loops"][0]["status"] == KNOWN_COPY for row in held)}


def pool_use(rows: Sequence[Mapping]) -> dict:
    """What the pools were: the episodes that held one at their end and how large, the pool checks by outcome, the
    harvests taken back out, the closers kept, and the closer checks by status."""
    arms = [row["arms"][ACCUMULATE] for row in rows]
    checks = [check for arm in arms for check in arm["pool_checks"]]
    closers = [check for arm in arms for check in arm["closer_checks"]]
    sizes = [arm["pool_blocks"] for arm in arms if arm["pool_blocks"]]
    return {"episodes": len(rows), "failed_first_generations": sum(row["failed_first"] for row in rows),
            "with_a_pool_at_the_end": len(sizes), "share_with_a_pool_at_the_end": _share(len(sizes), len(rows)),
            "mean_blocks_where_there_is_a_pool": round(sum(sizes) / len(sizes), 3) if sizes else None, "largest_pool": max(sizes, default=0),
            "pool_checks": len(checks), "pool_checks_by_outcome": dict(Counter(check["outcome"] for check in checks).most_common()),
            "harvests_taken_back_out": sum(not check["stands"] for check in checks), "lemmas_pooled": sum(check["harvested"] for check in checks if check["stands"]),
            "kept_closers": sum(arm["kept_closers"] for arm in arms), "closer_checks": len(closers),
            "closer_checks_sent_to_lean": sum(check["sent_to_lean"] for check in closers),
            "closer_checks_by_status": dict(Counter(check["status"] for check in closers).most_common()),
            "episodes_resolved_by_an_assembled_proof": sum(arm["resolved_by"] == ASSEMBLED for arm in arms)}


# -------------------------------------------------------------------------------------------------- budget
def arm_spend(rows: Sequence[Mapping], arm: str, through: int | None = None) -> dict:
    """What `arm` spent after the first generations (which the arms share): its generations, the tokens it generated,
    the tokens of its prompts, its Lean checks and its known copies. The accumulate arm's Lean checks are its
    attempts that were sent, its pool checks (one for every pool state a harvest made, those after generation 1
    too) and its closer checks that were sent. `through`: only up to that loop."""
    steps = [step for row in rows for step in row["arms"][arm]["loops"] if through is None or step["loop"] <= through]
    sent = sum(step["sent_to_lean"] for step in steps)
    own = [row["arms"][arm] for row in rows] if arm == ACCUMULATE else []
    pools = sum(1 for entry in own for check in entry["pool_checks"] if through is None or check["loop"] <= through)
    closers = sum(check["sent_to_lean"] for entry in own for check in entry["closer_checks"] if through is None or check["loop"] <= through)
    return {"generations": len(steps), "generated_tokens": sum(step["token_count"] for step in steps), "prompt_tokens": sum(step["prompt_tokens"] for step in steps),
            "attempts_sent_to_lean": sent, "pool_checks": pools, "closer_checks": closers, "lean_checks": sent + pools + closers,
            "known_copies": sum(step["status"] == KNOWN_COPY for step in steps)}


def budget(rows: Sequence[Mapping], primary_rows: Sequence[Mapping], generations: int, resamples: int, seed: int) -> dict:
    """Generations, generated tokens, prompt tokens, Lean checks and known copies by arm, and the comparison at equal
    tokens, as L3a2 makes it. `rows`: every episode (the cost of the run); `primary_rows`: the primary's episodes, on
    which the equal-token read is made."""
    by_arm = {arm: arm_spend(rows, arm) for arm in ARMS}
    firsts = {"generations": len(rows), "generated_tokens": sum(row["first"]["token_count"] for row in rows),
              "prompt_tokens": sum(row["first"]["prompt_tokens"] for row in rows), "lean_checks": sum(bool(row["first"]["sent_to_lean"]) for row in rows)}
    hard = {arm: arm_spend(primary_rows, arm) for arm in ARMS}
    blind_tokens, own_tokens = hard[BLIND]["generated_tokens"], hard[ACCUMULATE]["generated_tokens"]
    ratio = round(own_tokens / blind_tokens, 4) if blind_tokens else None
    checks = round(hard[ACCUMULATE]["lean_checks"] / hard[BLIND]["lean_checks"], 4) if hard[BLIND]["lean_checks"] else None
    result = {"what": "after the first generations, which the two arms share: an arm's generations, the tokens it generated, the tokens of its prompts, its "
                      "Lean checks (each attempt sent; for the accumulate arm also one for every pool state a harvest made, the one after generation 1 "
                      "too, and one for every closer check sent) and its known copies (attempts whose proof Lean had already rejected in the episode: "
                      "not sent, tokens counted). Each arm is counted in full: where the two arms' generations are one sample and one check (the same "
                      "prompt and seed), both count it",
              "first_generations": firsts, "by_arm": by_arm, "by_arm_on_the_primarys_problems": hard,
              "generated_tokens_accumulate_over_blind_on_the_primarys_problems": ratio,
              "lean_checks_accumulate_over_blind_on_the_primarys_problems": checks,
              "accumulate_generates_more_than_10_percent_more": bool(ratio is not None and ratio > 1 + MORE_TOKENS_ALLOWED)}
    if not result["accumulate_generates_more_than_10_percent_more"]:
        result["equal_tokens"] = "the accumulate arm generated no more than 10% more tokens than the blind arm: the primary is read as it stands"
        return result
    # The spec: "the blind arm is read at the number of generations that matches". As in L3a and L3a2, the blind arm was
    # sampled to `generations` - 1 more and the match needs more than that, so that read cannot be made from this run.
    # What can be read at equal tokens from what was sampled is its mirror: the accumulate arm at the loops whose tokens match.
    matching = round((generations - 1) * ratio, 2)
    kept_loops = 0
    for through in range(1, generations):
        if arm_spend(primary_rows, ACCUMULATE, through)["generated_tokens"] <= blind_tokens * (1 + MORE_TOKENS_ALLOWED):
            kept_loops = through
    result["equal_tokens"] = {
        "blind_generations_after_the_first_that_match": matching, "blind_generations_after_the_first_that_were_sampled": generations - 1,
        "the_blind_arm_at_the_matching_number_can_be_read": False,
        "why": f"the blind arm was sampled to {generations - 1} more generations an episode and the accumulate arm's tokens match {matching}",
        "mirror": {"what": f"the accumulate arm read at its first {kept_loops} generations after the first, whose generated tokens are within 10% of the "
                           f"blind arm's at {generations - 1}; the blind arm at all of its generations",
                   "accumulate_loops_read": kept_loops,
                   "accumulate_generated_tokens": arm_spend(primary_rows, ACCUMULATE, kept_loops)["generated_tokens"], "blind_generated_tokens": blind_tokens,
                   "primary": paired(primary_rows, ACCUMULATE, BLIND, 1 + kept_loops, resamples, seed, within_against=generations)}}
    return result


# ------------------------------------------------------------------------------- can this run see a win
def can_see_a_win(rows: Sequence[Mapping]) -> dict:
    """The three checks of the spec. A non-empty pool in at least 30% of the hard episodes by their last generation
    (every hard episode counts, the ones the first generation resolved too: they have no pool). A state for at least
    half of the continuing generations (a generation counts as continuing by its kind; it had a state when the pool
    stood in its prompt). And the pool as harvested stands in at least 95% of the checks made of it. A check that
    cannot be read (no hard episode, no continuing generation, no pool check) has not passed. All three are read on
    the counts: a share in the report is rounded."""
    hard = _of(rows, SETS[HARD])
    with_a_pool = sum(row["arms"][ACCUMULATE]["pool_blocks"] > 0 for row in hard)
    failed = sum(row["failed_first"] for row in hard)
    pools_pass = bool(hard) and with_a_pool >= MINIMUM_POOL_SHARE * len(hard)
    continuing = [step for row in rows for step in row["arms"][ACCUMULATE]["loops"] if step["kind"] == CONTINUE]
    held = sum(step["how"] == RESUMED for step in continuing)
    states_pass = bool(continuing) and held >= HALF * len(continuing)
    checks = [check for row in rows for check in row["arms"][ACCUMULATE]["pool_checks"]]
    standing = sum(check["stands"] for check in checks)
    stands_pass = bool(checks) and standing >= MINIMUM_STANDING_SHARE * len(checks)
    return {
        "a_pool_in_the_hard_episodes": {
            "hard_episodes": len(hard), "with_a_pool_by_their_last_generation": with_a_pool, "share": _share(with_a_pool, len(hard)),
            # Beside the spec's check: of the hard episodes whose first generation failed (the others never had a pool to make).
            "hard_episodes_whose_first_generation_failed": failed, "share_of_those": _share(with_a_pool, failed),
            "needed": "at least 30%", "passes": pools_pass},
        "a_state_for_the_continuing_generations": {
            "continuing_generations": len(continuing), "with_a_state": held, "share": _share(held, len(continuing)),
            "without_a_state_by_reason": dict(Counter(step["no_state"] for step in continuing if step["how"] == BLIND_ATTEMPT).most_common()),
            "needed": "at least half", "passes": states_pass},
        "the_pool_stands": {
            "pool_checks": len(checks), "stand": standing, "share": _share(standing, len(checks)),
            "by_outcome": dict(Counter(check["outcome"] for check in checks).most_common()), "needed": "at least 95%", "passes": stands_pass},
        "passes": pools_pass and states_pass and stands_pass}


# -------------------------------------------------------------------------------------------------- branch
def reach_is_ahead(reach: Mapping) -> bool:
    """The spec's "reach ahead": more problems of G gained than lost, with a sign test under 0.05."""
    return reach["gained"] > reach["lost"] and reach["sign_test_p"] is not None and reach["sign_test_p"] < REACH_AHEAD_BELOW


def accumulate_branch(primary: Mapping, reach: Mapping, checks: Mapping) -> Branch:
    """The branch the numbers select, in the spec's words. The three "can this run see a win" checks come first: when
    any fails the run is INCONCLUSIVE, not a verdict. `reach`: the problems of G resolved in any of their episodes,
    gained against lost."""
    if not checks["passes"]:
        pools, states, stands = checks["a_pool_in_the_hard_episodes"], checks["a_state_for_the_continuing_generations"], checks["the_pool_stands"]
        failed = []
        if not pools["passes"]:
            failed.append(f"a pool held a lemma in {pools['share']} of the hard episodes by their last generation (at least 30% is needed)")
        if not states["passes"]:
            failed.append(f"a state came back for {states['share']} of the continuing generations (at least half is needed)")
        if not stands["passes"]:
            failed.append(f"the pool stood in {stands['share']} of the checks made of it (at least 95% is needed)")
        return Branch(INCONCLUSIVE, "this run could not have seen a win: " + "; ".join(failed) + ": the harvest, the pool check or the prompt is broken, "
                                    "or the episode is too short for a pool. Fix it and run again; this is not a verdict")
    if primary.get("mean") is None:
        return Branch(INCONCLUSIVE, "the primary could not be computed: no episode on G or on the below-band rung")
    interval = f"{primary['mean']} [{primary['low']}, {primary['high']}]"
    split = f"gained {reach['gained']}, lost {reach['lost']} (p = {reach['sign_test_p']})"
    if primary["low"] > 0:
        return Branch(THE_SOLVERS_EPISODE, f"the primary is {interval}, above zero with an interval clear of zero: the accumulating episode is the solver's "
                                           "episode from here, and L3d trains on the proofs it assembles (its own read; the question there is whether "
                                           f"the model then writes the longer proofs itself). Reach on G: {split}")
    if primary["high"] < 0:
        return Branch(RESOLVES_FEWER, f"the primary is {interval}, below zero with an interval clear of zero: the accumulating episode resolves fewer hard "
                                      "episodes than blind attempts at equal generations. The spec names no branch for this; it is reported as that and "
                                      f"needs a decision. Reach on G: {split}")
    if reach_is_ahead(reach):
        return Branch(REACHES, f"the primary is {interval}, an interval through zero, and reach is ahead on G ({split}): the episode reaches problems "
                               "without resolving more episodes; reported as that, and L3d is still the next step")
    return Branch(NOT_SHOWN, f"the primary is {interval}, an interval that contains zero, and reach is not ahead on G ({split}): not shown; the probe's "
                             "gain came from more attempts' pieces than an episode has, and the size of the pool is the next thing to vary, which "
                             "needs a decision")


# ------------------------------------------------------------------------------------------------ the read
def accumulate_read(rows: Sequence[Mapping], generations: int, resamples: int, seed: int) -> dict:
    """Everything the spec's read names that needs no published proof (the reads by proof length are the report's:
    `reporting/ladder_l3c.py`), from the episode rows. Every comparison is accumulate minus blind by RUNNING TOTAL
    (resolved within k generations), paired by episode (the same first generation), each problem's mean over its
    episodes, a 95% bootstrap over problems."""
    hard, goal = _of(rows, SETS[HARD]), _of(rows, SETS[GOAL])
    primary = paired(hard, ACCUMULATE, BLIND, generations, resamples, seed)
    reach = gained_against_lost(goal, ACCUMULATE, BLIND, generations)
    checks = can_see_a_win(rows)
    branch = accumulate_branch(primary, reach, checks)
    return {
        "primary": primary,
        "reach": reach,
        # The running total within 2, 3, ... generations, by set: the paired difference and its interval, and the counts.
        "within_generations": {set_name: {str(within): paired(_of(rows, groups), ACCUMULATE, BLIND, within, resamples, seed) for within in range(2, generations + 1)}
                               for set_name, groups in SETS.items()},
        "running_totals": {set_name: running_totals(_of(rows, groups), generations) for set_name, groups in SETS.items()},
        "resolved_within_generations": {set_name: resolved_by_attempt(_of(rows, groups), generations, ARMS) for set_name, groups in SETS.items()},
        "resolutions": {set_name: resolutions(_of(rows, groups), generations) for set_name, groups in SETS.items()},
        "verified_proof_lines": {set_name: verified_lines(_of(rows, groups), generations) for set_name, groups in SETS.items()},
        "pools": {set_name: pool_use(_of(rows, groups)) for set_name, groups in SETS.items()},
        "second_generation": {set_name: second_generation(_of(rows, groups)) for set_name, groups in SETS.items()},
        "by_generation": {set_name: by_generation(_of(rows, groups), generations) for set_name, groups in SETS.items()},
        "budget": budget(rows, hard, generations, resamples, seed),
        "can_this_run_see_a_win": checks,
        "branch": {"name": branch.name, "reason": branch.reason},
    }
