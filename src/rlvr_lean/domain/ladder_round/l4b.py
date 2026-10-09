"""L4b: the arm again, from the larger pretrained model. Spec: docs/spec/ladder-loop.spec.md, "L4b: the arm again,
from the larger pretrained model" and its "Made exact by the build". Pure.

THE TRAINING RULE IS ONE SETTING OF AN ARM (`rule`), and L4t names it. Inside a round it is applied to the rows of
rounds 1 to r as L4t applies it to all six (`rule_set`), with L4t's own functions (`l4_rows`):

  old           what every arm does: one proof for every solved problem (the rounds' one-shot and assembled rows)
  reward_rows   a row is kept when its draw (a content hash of the task's seed and the row's id) is under r(k), the
                challenger's reward at the arm's target rate for its problem's k of n (an assembled row: k = 1). The
                draw does not depend on r: A ROW KEPT FOR M(r) IS KEPT FOR EVERY LATER MODEL
  rehearse      all the rounds' rows, and with them as many rows of the pretraining file as the model has one-shot
                rows, the rows of the smallest content hash: M(r)'S REHEARSAL ROWS ARE THE FIRST OF M(r + 1)'S

The twin is the same rule with the assembled rows left out (`twin_of`): the last model's set, in its order, without them.

THE READ is L4's with the start model in `pre`'s place, and BESIDE THE PRIMARY, on every branch, the condition "without
overfitting" (`breadth_beside`, `with_breadth`): the goal problems solved at least once in all the attempts, `with`
against the start model, and the share of distinct attempts on G. Lost more than gained with the sign test under 0.05
is written NARROWER whatever the primary says.
"""

from __future__ import annotations

from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import ASSEMBLED, ATTEMPT, FROM_ASSEMBLY, training_order, training_set
from rlvr_lean.domain.ladder_round.l4 import ADDS, INCONCLUSIVE
from rlvr_lean.domain.ladder_round.l4_rows import (
    PRETRAINING,
    SIGN_TEST_LEVEL,
    THE_MINIMUM,
    kept_rows,
    rehearsal_rows,
    rehearse_set,
    reward_draws,
    smallest_decisive_split,
    with_k,
)
from rlvr_lean.domain.repair.accumulate import proof_line_count

OLD, REWARD_ROWS, REHEARSE = "old", "reward_rows", "rehearse"
RULES = (OLD, REWARD_ROWS, REHEARSE)            # what an arm's `rule` may say; an arm that says none is trained as every arm was
NARROWER = "NARROWER"                           # lost more goal problems than gained against the start model, the sign test under the level
ON_FEWER = "adds per attempt, on fewer problems"       # how a primary that "adds" is reported when the model is also NARROWER
ORIGINS = (ATTEMPT, ASSEMBLED, PRETRAINING)     # where a row of a set made by a rule is from (an arm with a rule reads no H0)


# ------------------------------------------------------------------------------------------- the rule in a round
def rule_set(by_round: Mapping[int, Sequence[Mapping]], picks: Sequence[Mapping], number: int, rule: str, target_rate: float, seed: int,
             pretraining: Sequence[Mapping] | None = None, at_least: int = 0) -> list[Mapping]:
    """What M(`number`) is trained on under `rule`, IN THE ORDER IT IS TRAINED (the arm's content hash). `by_round`: the
    rounds' stored training examples. `picks`: every batch's per-problem results of rounds 1 to `number` (each problem's
    k of n). `pretraining` (`rehearse` alone): the rows the start model was pretrained on, in the file's order, WITHOUT
    their text. `at_least` (`reward_rows` in a smoke run alone): `l4_rows.kept_rows`' minimum.

    `old`: the rows themselves, as every arm orders them. The two others: every row of the rounds gets its `lines`, its
    `k` and `n` (`l4_rows.with_k`: refused for a row no round picked, an H0 row, or stored rounds that do not agree).
    `reward_rows`: the rows whose draw is under their reward, each with its `reward`, `draw` and `kept_by`.
    `rehearse`: every row of the rounds, and as many rehearsal rows as there are one-shot rows among them."""
    rows = training_set(by_round, number)
    if rule == OLD:
        return training_order(rows, seed)
    if rule not in RULES:
        raise ValueError(f"{rule!r} is not a training rule: an arm's rule is one of {list(RULES)}")
    of_the_rounds = with_k([{**row, "lines": proof_line_count(row["completion"])} for row in rows], picks)
    if rule == REWARD_ROWS:
        return kept_rows(reward_draws(of_the_rounds, target_rate, seed), seed, at_least)
    if pretraining is None:
        raise ValueError(f"the rule `{REHEARSE}` draws its rehearsal rows among the rows the start model was pretrained on, and none were given")
    one_shot = [row for row in of_the_rounds if row["origin"] == ATTEMPT]
    rehearsal = rehearsal_rows(pretraining, len(one_shot), seed)
    rehearse_set(one_shot, rehearsal, seed)         # as many of the one as of the other, and no id twice: refused otherwise
    return training_order([*of_the_rounds, *rehearsal], seed)


def twin_of(ordered: Sequence[Mapping]) -> list[Mapping]:
    """The twin's rows under a rule: `ordered` (what the last model was trained on, in its order) with the assembled
    rows left out and none moved. The rehearsal rows stay: the twin is the same rule without the assembled proofs."""
    return [row for row in ordered if row["origin"] not in FROM_ASSEMBLY]


def rehearsal_of(rows: Sequence[Mapping]) -> list[Mapping]:
    """A set's rehearsal rows, in the set's order."""
    return [row for row in rows if row["origin"] == PRETRAINING]


def by_origin(rows: Sequence[Mapping]) -> dict[str, int]:
    """A set made by a rule, by where its rows are from: the rounds' one-shot rows, their assembled rows, the rehearsal rows."""
    return {origin: sum(row["origin"] == origin for row in rows) for origin in ORIGINS}


def kept_share(of_the_rounds: Sequence[Mapping], kept: Collection[str]) -> dict:
    """The rule's kept share: of the rounds' rows (`id`, `k`), how many are in the set (`kept`: its rows' ids), in all and
    by k. Under `old` and under `rehearse` every row of the rounds is kept; under `reward_rows` the share follows r(k)."""
    def share(rows: Sequence[Mapping]) -> dict:
        inside = sum(row["id"] in kept for row in rows)
        return {"rows": len(rows), "kept": inside, "share": round(inside / len(rows), 5) if rows else None}

    return {**share(of_the_rounds), "by_k": {str(k): share([row for row in of_the_rounds if row["k"] == k]) for k in sorted({row["k"] for row in of_the_rounds})}}


def record(rule: str, rows: Sequence[Mapping], of_the_rounds: Sequence[Mapping]) -> dict:
    """What a training's summary says of its rule: the rule, the set's rows by origin, and the kept share of the rounds'
    rows in all and by k. `rows`: the set. `of_the_rounds`: every row of rounds 1 to r with its k (`with_k`)."""
    return {"rule": rule, "rows": len(rows), "rows_by_origin": by_origin(rows), "rows_of_the_rounds": len(of_the_rounds),
            "kept_of_the_rounds_rows": kept_share(of_the_rounds, {row["id"] for row in rows}),
            "kept_by_the_runs_minimum": sum(row.get("kept_by") == THE_MINIMUM for row in rows)}


# ------------------------------------------------------------------------------------- the breadth, beside the primary
def breadth_beside(solved: Mapping, distinct_of_with: float | None, distinct_of_the_start: float | None, start: str, level: float = SIGN_TEST_LEVEL) -> dict:
    """The condition "without overfitting", read beside the primary on every branch. `solved`: the goal problems solved at
    least once in all the attempts, `with` against the start model, paired by problem (`gained_lost`: `gained`, `lost`,
    `sign_test_p`). The two shares of distinct attempts on G. NARROWER: lost more than gained with the sign test under
    `level`."""
    gained, lost, p = solved["gained"], solved["lost"], solved["sign_test_p"]
    narrower = bool(lost > gained and p is not None and p < level)
    counts = f"gained {gained}, lost {lost}" + (f" (p = {p})" if p is not None else " (no problem changed)")
    return {"what": f"the goal problems solved at least once in all the attempts, `with` against `{start}`, paired by problem, with the two-sided sign test; and the share "
                    f"of distinct attempts on G. Lost more than gained with the sign test under {level:g} is written {NARROWER}, whatever the primary says",
            "problems": solved["problems"], "solved_by_with": solved["resolved_after"], "solved_by_the_start": solved["resolved_before"], "gained": gained, "lost": lost,
            "sign_test_p": p, "level": level, "counts": counts, "share_of_distinct_attempts_on_g": {"with": distinct_of_with, start: distinct_of_the_start},
            "narrower": narrower, "smallest_decisive_split": smallest_decisive_split(gained + lost, level)}


def with_breadth(branch: Mapping, breadth: Mapping) -> dict:
    """L4's branch with the breadth read beside it. The branch's NAME is L4's and does not move. `reported_as` is how it
    is said: with NARROWER after it when the model lost more goal problems than it gained, and a primary that "adds" is
    then reported as "adds per attempt, on fewer problems". An INCONCLUSIVE run says nothing else: the breadth is kept
    with what was measured and not read."""
    if branch["name"] == INCONCLUSIVE:
        return {**branch, "narrower": None, "reported_as": branch["name"]}
    if not breadth["narrower"]:
        return {**branch, "narrower": False, "reported_as": branch["name"]}
    said = f"{branch['name']}, {NARROWER}" + (f": {ON_FEWER}" if branch["name"] == ADDS else "")
    return {**branch, "narrower": True, "reported_as": said,
            "reason": f"{branch['reason']}. {NARROWER}: against the start model it lost more goal problems than it gained with the sign test under {breadth['level']:g} "
                      f"({breadth['counts']})" + (f"; the primary is reported as: {ON_FEWER}" if branch["name"] == ADDS else "")}
