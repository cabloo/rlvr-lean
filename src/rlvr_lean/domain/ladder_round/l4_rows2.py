"""L4t2: two more one-change checks, on what L4t left open. Spec: docs/spec/ladder-loop.spec.md, "L4t2: two more
one-change checks, on what L4t left open" and its "Made exact by the build". Pure.

L4t read `rehearse` as the one rule that adds on G', is not narrower and holds the long proofs, and `reward_rows` as the
rule that keeps the breadth, adds on G' and loses the long proofs. Two things were left open, and each is ONE model
away, trained from the start model on rows L4t already stored. NO NEW ROUND, and no new draw of the rounds' rows:

  rehearse_only     the rehearsal rows of L4t's `rehearse` set and nothing else, in that set's order
                    (`rehearse_only_set`). Half of `rehearse`'s rows are published proofs the start model has seen
                    once: a second pass over them alone would show the same, and this is that second pass
  reward_rehearse   L4t's `reward_rows` set, and with it as many rehearsal rows as it has ONE-SHOT rows: the first of
                    the hash order L4t's rehearsal draw uses (`l4_rows.rehearsal_rows`), so they are the first of
                    `rehearse`'s own. One order, the arm's content hash (`reward_rehearse_set`)

A rehearsal row is a published proof: a set names it by its id and its place in the file, never with its text.

  the three reads   of a model against the start model, the ARM's own (L4b), fixed before the run (`three_reads`):
                    ADDS (G', the fresh attempts: the interval above zero), NOT NARROWER (the goal problems solved in
                    all the attempts: not lost more than gained with the sign test under the level, L4b's own
                    `NARROWER`), HOLDS THE LONG PROOFS (the goal problems of 4 lines or more, per attempt over all the
                    attempts: the interval not wholly below zero)
  Question 1        `rehearse` minus `rehearse_only` on G' over the fresh attempts: do the loop's rows add beside the
                    rehearsal, or is the gain a second pass over published proofs? (`question_1`)
  Question 2        the three reads of `reward_rehearse` (`question_2`)
  the arm's rule    which rule the arm from the start model is run with, by the spec's rule (`the_arms_rule`)

A model whose own checks failed is INCONCLUSIVE and none of its reads is made; what stands on it is not read either.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import ATTEMPT, training_order
from rlvr_lean.domain.ladder_round.ceiling import interval_text
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4_rows import OLD_RULE, PRETRAINING, REHEARSE, REWARD_ROWS, SIGN_TEST_LEVEL, rehearsal_rows
from rlvr_lean.domain.ladder_round.l4b import breadth_beside, rehearsal_of

REHEARSE_ONLY, REWARD_REHEARSE = "rehearse_only", "reward_rehearse"
MODELS = (REHEARSE_ONLY, REWARD_REHEARSE)           # the two trained models, in the order they are trained and measured
STORED = (OLD_RULE, REWARD_ROWS, REHEARSE)          # L4t's three, read here from the rows its run stored
TEXT_FIELDS = ("theorem", "completion", "statement", "proof")       # where a row holds text: a round's own example, and a row of the pretraining file

ADDS, NOT_NARROWER, HOLDS = "ADDS", "NOT NARROWER", "HOLDS THE LONG PROOFS"        # the three reads, in the spec's words
READS = (ADDS, NOT_NARROWER, HOLDS)
KEY = {ADDS: "adds", NOT_NARROWER: "not_narrower", HOLDS: "holds_the_long_proofs"}        # ... and as a report's keys

LOOPS_ROWS_ADD = "THE LOOP'S ROWS ADD BESIDE THE REHEARSAL"             # Question 1: `rehearse` minus `rehearse_only` above zero
SECOND_PASS = "THE GAIN IS A SECOND PASS OVER PUBLISHED PROOFS"        # ... else, `rehearse_only` itself ADDS: not shown to be the loop's
NOT_SEPARATED = "NOT SEPARATED AT THIS SIZE"                           # ... else
QUESTION_1 = (LOOPS_ROWS_ADD, SECOND_PASS, NOT_SEPARATED, INCONCLUSIVE, NOT_READ)
ALL_THREE = "IT ADDS, KEEPS THE BREADTH AND HOLDS THE LONG PROOFS"     # Question 2: `reward_rehearse` passes the three reads
NO_ARM = "NO ARM WITH REHEARSAL ROWS"                                   # the arm's rule: Question 1 reads a second pass
NO_RULE = "NO RULE IS NAMED"                                            # ... a case the section does not name: what its rule stands on could not be read
EXPLORATORY = "a choice between two rules by reads on the same goal set is exploratory"


def fails(reads: Sequence[str]) -> str:
    """Question 2, otherwise: the read (or the reads) `reward_rehearse` fails, named."""
    return "IT FAILS: " + " AND ".join(reads)


# ------------------------------------------------------------------------------------------------- the sets
def rehearse_only_set(rehearse: Sequence[Mapping]) -> list[Mapping]:
    """`rehearse_only`: the rehearsal rows of L4t's `rehearse` set and nothing else, in that set's order with the other
    rows left out. Refused (ValueError) when the set holds none."""
    rows = rehearsal_of(rehearse)
    if not rows:
        raise ValueError(f"L4t's `{REHEARSE}` set holds no rehearsal row among its {len(rehearse)}: there is nothing to train `{REHEARSE_ONLY}` on. Refused")
    return rows


def _placed(row: Mapping) -> tuple:
    """A rehearsal row as a set names it: its id, its problem and its place in the file (never its text)."""
    return row["id"], row["problem_id"], row["row_of_the_file"]


def reward_rehearse_set(reward_rows: Sequence[Mapping], rehearse: Sequence[Mapping], pretraining: Sequence[Mapping], seed: int) -> list[Mapping]:
    """`reward_rehearse`: L4t's `reward_rows` set and as many rehearsal rows as it has ONE-SHOT rows, in ONE order, the
    arm's content hash. The rehearsal rows are the FIRST of the hash order L4t's rehearsal draw uses
    (`l4_rows.rehearsal_rows`, the same call with a smaller count), so they are among `rehearse`'s own. `reward_rows`,
    `rehearse`: the two sets as L4t's run stored them. `pretraining`: the rows the start model was pretrained on, in
    the file's order, WITHOUT their text. `seed`: the task's, which is L4t's.

    Refused (ValueError): `rehearse`'s stored rehearsal rows are not this draw's (another file, or another seed: a
    prefix of another draw would be other rows); the rows drawn are not the first of it; an id stands twice."""
    of_rehearse = rehearsal_of(rehearse)
    drawn = rehearsal_rows(pretraining, len(of_rehearse), seed)         # L4t's whole draw again, in the DRAW's order: the smallest hash first
    if sorted(map(_placed, of_rehearse)) != sorted(map(_placed, drawn)):
        other = sorted(set(map(_placed, of_rehearse)) ^ set(map(_placed, drawn)))
        raise ValueError(f"L4t's `{REHEARSE}` set holds {len(of_rehearse)} rehearsal rows, and they are not the {len(drawn)} rows of the smallest hash among the "
                         f"{len(pretraining)} rows the start model was pretrained on (first that differs: {other[0][0]}): that set was not drawn from this file with "
                         "this seed. Refused")
    one_shot = [row for row in reward_rows if row["origin"] == ATTEMPT]
    first = rehearsal_rows(pretraining, len(one_shot), seed)
    if list(map(_placed, first)) != list(map(_placed, drawn[:len(one_shot)])):
        raise ValueError(f"the {len(first)} rehearsal rows drawn for `{REWARD_REHEARSE}` are not the first {len(one_shot)} of the hash order `{REHEARSE}`'s were drawn "
                         "in. Refused")
    rows = [*reward_rows, *first]
    twice = sorted(key for key, times in Counter(row["id"] for row in rows).items() if times > 1)
    if twice:
        raise ValueError(f"{len(twice)} ids stand twice in `{REWARD_REHEARSE}` (first: {twice[0]}). Refused")
    return training_order(rows, seed)


def refuse_text(rows: Sequence[Mapping], what: str) -> None:
    """A rehearsal row is a published proof: a set names it by its id and its place in the file and NEVER holds its
    text (the training step reads that from the pretraining file). Refused (ValueError) when one carries a text field."""
    carrying = [row["id"] for row in rows if row["origin"] == PRETRAINING and any(field in row for field in TEXT_FIELDS)]
    if carrying:
        raise ValueError(f"{len(carrying)} rehearsal rows of {what} carry a text field (first: {carrying[0]}): a rehearsal row is a published proof, named by its id and "
                         "its place in the file and never stored with its text. Refused")


# ------------------------------------------------------------------------------------------ the three reads
def three_reads(on_again: Mapping, solved: Mapping, long: Mapping, start: str, failed_checks: Sequence[str] = (), level: float = SIGN_TEST_LEVEL) -> dict:
    """The three reads of ONE model against the start model, the arm's own (L4b), fixed before the run. `on_again`:
    G', successes per attempt over the fresh attempts, the model minus `start`, paired by problem, with its interval.
    `solved`: the goal problems solved at least once in all the attempts, the model against `start` (`gained`, `lost`,
    `sign_test_p`). `long`: the goal problems of 4 lines or more, successes per attempt over all the attempts, the
    model minus `start`, with its interval.

      ADDS                    the interval on G' is above zero (one that ends at zero is not above it)
      NOT NARROWER            NOT (lost more than gained with the sign test under `level`): L4b's `NARROWER`, by its own
                              function (`l4b.breadth_beside`)
      HOLDS THE LONG PROOFS   the interval on the problems of 4 lines or more is not wholly below zero

    Each read `passes` (True or False), or is not read (None) when nothing was measured for it (a run with one
    sampling of G has no fresh attempts). `all_three`: True when the three pass; False when one fails; None when none
    fails and one is not read. `failed_checks`: the checks the model ITSELF failed: it is then INCONCLUSIVE, none of
    its reads is made, and what was measured is kept apart and not read."""
    breadth = breadth_beside(solved, None, None, start, level)
    measured = {
        KEY[ADDS]: {"what": f"G', successes per attempt over the fresh attempts, the model minus `{start}`, paired by problem, 95% bootstrap over problems: the "
                            "interval is above zero",
                    **{key: on_again.get(key) for key in ("problems", "mean", "low", "high")},
                    "passes": None if on_again.get("mean") is None else bool(on_again["low"] > 0)},
        KEY[NOT_NARROWER]: {"what": f"the goal problems solved at least once in all the attempts, the model against `{start}`: NOT (lost more than gained with the "
                                    f"two-sided sign test under {level:g})",
                            **{key: solved[key] for key in ("problems", "gained", "lost", "sign_test_p")}, "gained_minus_lost": solved["gained"] - solved["lost"],
                            "counts": breadth["counts"], "narrower": breadth["narrower"], "smallest_decisive_split": breadth["smallest_decisive_split"],
                            "passes": None if not solved.get("problems") else not breadth["narrower"]},
        KEY[HOLDS]: {"what": f"the goal problems of 4 lines or more, successes per attempt over all the attempts, the model minus `{start}`, 95% bootstrap over "
                             "problems: the interval is not wholly below zero",
                     **{key: long.get(key) for key in ("problems", "mean", "low", "high")},
                     "passes": None if long.get("mean") is None else not long["high"] < 0}}
    if failed_checks:
        return {"against": start, "failed_checks": list(failed_checks), **{KEY[read]: {"what": measured[KEY[read]]["what"], "passes": None} for read in READS},
                "passed": [], "fails": [], "not_read": list(READS), "all_three": None, "gained_minus_lost": None,
                "said": f"NOT READ: this model is {INCONCLUSIVE} ({', '.join(failed_checks)}), and none of its reads is made",
                "measured_and_not_read": {KEY[read]: {key: value for key, value in measured[KEY[read]].items() if key not in ("what", "passes")} for read in READS}}
    passed, failed, unread = ([read for read in READS if measured[KEY[read]]["passes"] is verdict] for verdict in (True, False, None))
    figures = {ADDS: f"G', the fresh attempts: {interval_text(on_again)}", NOT_NARROWER: breadth["counts"] if solved.get("problems") else "no goal problem was measured",
               HOLDS: f"4 lines or more: {interval_text(long)}"}
    verdict = {True: "PASS", False: "FAIL", None: "not read"}
    return {"against": start, "failed_checks": [], **measured, "passed": passed, "fails": failed, "not_read": unread,
            "all_three": False if failed else None if unread else True, "gained_minus_lost": measured[KEY[NOT_NARROWER]]["gained_minus_lost"],
            "said": "; ".join(f"{read}: {verdict[measured[KEY[read]]['passes']]} ({figures[read]})" for read in READS)
                    + (f". FAILS {' AND '.join(failed)}" if failed else f". NOT READ: {', '.join(unread)}" if unread else ". ALL THREE")}


def _margin(reads: Mapping) -> str:
    """A model's gained minus lost against the start model, with the two counts, as a reason says it."""
    breadth = reads[KEY[NOT_NARROWER]]
    return f"{reads['gained_minus_lost']:+d} (gained {breadth['gained']}, lost {breadth['lost']})"


# ---------------------------------------------------------------------------------------------- the branches
def question_1(difference: Mapping, of_rehearse_only: Mapping, of_rehearse: Mapping) -> dict:
    """Question 1, fixed before the run. `difference`: on G' over the fresh attempts, `rehearse` minus `rehearse_only`,
    paired by problem, with its interval. `of_rehearse_only`, `of_rehearse`: each one's three reads (`three_reads`).

    Its interval above zero: THE LOOP'S ROWS ADD BESIDE THE REHEARSAL. Else, if `rehearse_only` itself ADDS against the
    start model: THE GAIN IS A SECOND PASS OVER PUBLISHED PROOFS (not shown to be the loop's). Else: NOT SEPARATED AT
    THIS SIZE, with what this difference resolves (half the width of its interval). A model of the two that failed a
    check of its own: INCONCLUSIVE. Nothing measured on both sides (a run with one sampling of G): NOT READ."""
    failed = {name: list(reads["failed_checks"]) for name, reads in ((REHEARSE, of_rehearse), (REHEARSE_ONLY, of_rehearse_only)) if reads["failed_checks"]}
    if failed:
        return {"name": INCONCLUSIVE, "failed_checks": failed,
                "reason": "; ".join(f"`{name}` is {INCONCLUSIVE} ({', '.join(checks)})" for name, checks in failed.items())
                          + ": the difference stands on both models and is not read; nothing is said about what the loop's rows add beside the rehearsal"}
    if difference.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": {}, "reason": "G' holds no problem with fresh attempts on both sides: the difference has nothing to be read on"}
    read, start = interval_text(difference), of_rehearse_only["against"]
    alone = of_rehearse_only[KEY[ADDS]]
    said_alone = f"`{REHEARSE_ONLY}` minus `{start}` on the same attempts: {interval_text(alone)}"
    if difference["low"] > 0:
        return {"name": LOOPS_ROWS_ADD, "failed_checks": {}, "rehearse_only_adds": alone["passes"],
                "reason": f"on G' over the fresh attempts `{REHEARSE}` minus `{REHEARSE_ONLY}` is above zero ({read}): the loop's rows add beside the rehearsal "
                          f"({said_alone})"}
    if alone["passes"]:
        return {"name": SECOND_PASS, "failed_checks": {}, "rehearse_only_adds": True,
                "reason": f"on G' over the fresh attempts `{REHEARSE}` minus `{REHEARSE_ONLY}` is not above zero ({read}), and `{REHEARSE_ONLY}` itself ADDS ({said_alone}, "
                          "above zero): the gain is a second pass over published proofs, not shown to be the loop's"}
    resolves = round((difference["high"] - difference["low"]) / 2, 5)
    return {"name": NOT_SEPARATED, "failed_checks": {}, "rehearse_only_adds": False, "resolves": resolves,
            "reason": f"on G' over the fresh attempts `{REHEARSE}` minus `{REHEARSE_ONLY}` is not above zero ({read}), and `{REHEARSE_ONLY}` does not itself ADD "
                      f"({said_alone}, not above zero): not separated at this size. This difference resolves {resolves:.5f} per attempt (half the width of its interval)"}


def question_2(of_reward_rehearse: Mapping) -> dict:
    """Question 2, fixed before the run: the three reads of `reward_rehearse` (`three_reads`). All three: IT ADDS,
    KEEPS THE BREADTH AND HOLDS THE LONG PROOFS. Otherwise the read it fails is named (`fails`). INCONCLUSIVE when it
    failed a check of its own; NOT READ when it fails no read and one could not be made."""
    reads = of_reward_rehearse
    if reads["failed_checks"]:
        return {"name": INCONCLUSIVE, "failed_checks": list(reads["failed_checks"]),
                "reason": f"`{REWARD_REHEARSE}` could not have been read ({', '.join(reads['failed_checks'])}): none of its three reads is made"}
    if reads["fails"]:
        return {"name": fails(reads["fails"]), "failed_checks": [], "fails": list(reads["fails"]), "reason": f"against `{reads['against']}`: {reads['said']}"}
    if reads["not_read"]:
        return {"name": NOT_READ, "failed_checks": [], "fails": [],
                "reason": f"against `{reads['against']}`: {reads['said']}: it fails no read that was made, and all three are asked"}
    return {"name": ALL_THREE, "failed_checks": [], "fails": [], "reason": f"against `{reads['against']}`: {reads['said']}: it adds, keeps the breadth and holds the long proofs"}


def the_arms_rule(first: Mapping, of_reward_rehearse: Mapping, of_rehearse: Mapping) -> dict:
    """Which rule the arm from the start model is run with after L4t2, by the spec's rule ("The arm's rule after
    L4t2"). `first`: Question 1's branch. `of_reward_rehearse`, `of_rehearse`: each one's three reads (`three_reads`).

    Question 1 reads A SECOND PASS: NO ARM WITH REHEARSAL ROWS (it could not show that the loop adds). Otherwise the arm
    is run with a rule that passes all three reads: `reward_rehearse` if it passes and its (gained minus lost) against
    the start model is AT LEAST `rehearse`'s; else `rehearse`. `rule` is then the rule's name, and the choice is
    exploratory. NO RULE IS NAMED (a case the section does not name) when what the rule stands on could not be read
    (Question 1 INCONCLUSIVE or NOT READ; a rule of the two INCONCLUSIVE, or with a read that could not be made and
    none failed), and when neither rule passes all three."""
    start = of_rehearse["against"]
    numbers = {"question_1": first["name"],
               **{name: {"all_three": reads["all_three"], "fails": list(reads["fails"]), "not_read": list(reads["not_read"]), "gained_minus_lost": reads["gained_minus_lost"]}
                  for name, reads in ((REWARD_REHEARSE, of_reward_rehearse), (REHEARSE, of_rehearse))}}
    if first["name"] == SECOND_PASS:
        return {"name": NO_ARM, "rule": None, "numbers": numbers,
                "reason": f"Question 1 reads {SECOND_PASS}: no arm with rehearsal rows is run, because it could not show that the loop adds. What stands needs a "
                          "decision: a second pass over published proofs helps (a fact about the pretraining recipe), and the loop's own rows trade long proofs for "
                          "successes on short ones"}
    if first["name"] in (INCONCLUSIVE, NOT_READ):
        return {"name": NO_RULE, "rule": None, "numbers": numbers,
                "reason": f"Question 1 is {first['name']}: whether an arm with rehearsal rows could show that the loop adds is not read, and no rule is named for it"}
    unread = [name for name, reads in ((REWARD_REHEARSE, of_reward_rehearse), (REHEARSE, of_rehearse)) if reads["all_three"] is None]
    if unread:
        return {"name": NO_RULE, "rule": None, "numbers": numbers,
                "reason": f"Question 1 reads {first['name']} (not a second pass), and the three reads of " + " and of ".join(f"`{name}`" for name in unread)
                          + " could not all be made: the rule is chosen by them, and no rule is named"}
    said = f"Question 1 reads {first['name']} (not a second pass); gained minus lost against `{start}`: `{REWARD_REHEARSE}` {_margin(of_reward_rehearse)}, `{REHEARSE}` {_margin(of_rehearse)}"
    carries = f"the arm is run from `{start}` with it, and carries its own separation. Exploratory: {EXPLORATORY}"
    if of_reward_rehearse["all_three"] and of_reward_rehearse["gained_minus_lost"] >= of_rehearse["gained_minus_lost"]:
        return {"name": REWARD_REHEARSE, "rule": REWARD_REHEARSE, "numbers": numbers, "exploratory": EXPLORATORY,
                "reason": f"{said}. `{REWARD_REHEARSE}` passes all three reads and its gained minus lost is at least `{REHEARSE}`'s: the arm's rule is `{REWARD_REHEARSE}`; {carries}"}
    if of_rehearse["all_three"]:
        why = (f"`{REWARD_REHEARSE}` passes all three reads and its gained minus lost is under `{REHEARSE}`'s" if of_reward_rehearse["all_three"] else
               f"`{REWARD_REHEARSE}` does not pass all three reads (it fails {' and '.join(of_reward_rehearse['fails'])})")
        return {"name": REHEARSE, "rule": REHEARSE, "numbers": numbers, "exploratory": EXPLORATORY,
                "reason": f"{said}. {why}, and `{REHEARSE}` passes all three: the arm's rule is `{REHEARSE}`; {carries}"}
    why = (f"`{REWARD_REHEARSE}` passes all three reads with a gained minus lost under `{REHEARSE}`'s, so the section's rule names `{REHEARSE}`, which fails "
           f"{' and '.join(of_rehearse['fails'])}" if of_reward_rehearse["all_three"] else
           f"`{REWARD_REHEARSE}` fails {' and '.join(of_reward_rehearse['fails'])} and `{REHEARSE}` fails {' and '.join(of_rehearse['fails'])}")
    return {"name": NO_RULE, "rule": None, "numbers": numbers,
            "reason": f"{said}. {why}: the arm is run with a rule that passes all three reads, and the section's rule names none here"}
