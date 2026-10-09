"""L4t: what should a round train on? Spec: docs/spec/ladder-loop.spec.md, "L4t: what should a round train on?
Three one-change checks on the rounds already made" and its "Made exact by the build". Pure.

The arm from `pre` made the pretrained model more reliable and NARROWER. Three causes are each one change away, and none
needs a new round: the rounds' attempts and the adapters are stored. Two of the three are a TRAINING SET made from the
stored rounds, each trained from the start adapter by the arm's recipe, and their rules are here:

  rehearse      the twin's one-shot rows, and with them as many rows of the pretraining file: a draw, by a content hash of
                the task's seed and each row's id, among the rows `pre` was trained on, no row twice (`rehearsal_rows`)
  reward_rows   each row of the rounds is kept with probability r(k), the challenger's reward at the arm's target rate for
                its problem's k of n (`episodes.reward`: the ONE place the reward is written); an assembled row counts as
                k = 1 (`assembly.counted`: k as the challenger reads it). The draw is a number in [0, 1) made from a
                content hash of the task's seed and the row's id (`uniform`): a row is kept when it is under r(k). No
                quota and no screen (`with_k`, `reward_draws`, `kept_rows`)

Both are trained in ONE order, the arm's content hash (`assembly.training_order`). The third check (`hot`) trains
nothing: it is a measurement, and has no rule here.

  the checks    the fourth of a trained model's checks: no row it was trained on is a held-out problem or, among its
                rounds' rows, a problem of the `pretrain` half (`no_barred_row`; before any training, `refuse_barred`)
  the branch    in the spec's words (`rows_branch`), with the sign test's smallest decisive split at the observed number
                of changed problems (`smallest_decisive_split`)

ANOTHER START THAN `pre` (spec, "L4r's first branch was taken: what L4t runs"). Nothing stored is the old rule at the
start adapter's rank, so a third model is trained and measured FIRST: `old_rule`, the twin's rows exactly, in the twin's
order (what `without` is, from that start). It takes `without`'s place in the two other models' reads and the start
model takes `pre`'s (`models_of`, `rows_branch`'s `against` and `start`), and it is read BY ITSELF against the start
model, which is the cap's question (`old_rule_branch`: the three named outcomes, with the start adapter's rank in them).
"""

from __future__ import annotations

from collections import Counter
from typing import Collection, Mapping, Sequence

from rlvr_lean.domain.ladder_round.assembly import ASSEMBLED, ATTEMPT, counted, training_order
from rlvr_lean.domain.ladder_round.l3d import WITHOUT
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, LOOP, NOT_READ, PRE, PRETRAIN, half_of
from rlvr_lean.domain.ladder_round.rounds import sign_test
from rlvr_lean.domain.problem_pool.episodes import STATEMENT_SIDE, reward
from rlvr_lean.domain.problem_pool.selection import rank

REHEARSE, REWARD_ROWS = "rehearse", "reward_rows"
MODELS = (REHEARSE, REWARD_ROWS)                    # the two trained models, in the order they are trained and measured
OLD_RULE = "old_rule"                               # from another start than `pre`: the twin's rows, the old rule at that start's rank
MODELS_FROM_ANOTHER_START = (OLD_RULE, REWARD_ROWS, REHEARSE)      # ... and the three models then, in the order they are trained and measured
HOT = "hot"                                         # the third check: `with` and `pre` sampled again at another temperature
PRETRAINING = "pretraining"                         # where a rehearsal row is from: the pretraining file (a published proof)
REHEARSAL_LABEL, KEEP_LABEL = "l4_rows_rehearsal", "l4_rows_keep"      # the two draws' own labels in the content hash
THE_RULE, THE_MINIMUM = "the rule", "the run's minimum"                # why a row is in `reward_rows` (the second: a smoke run alone)
HASH_DIGITS = 13                                    # hexadecimal digits of the hash a draw is made from: 52 bits, exactly a float
SIGN_TEST_LEVEL = 0.05                              # the spec's: "with the sign test under 0.05"

GIVES_BACK = "THE RULE GIVES THE BREADTH BACK AND KEEPS THE GAIN"       # gained more than lost (sign test under 0.05) AND all of G against `pre` above zero
UNDOES = "IT GIVES THE BREADTH BACK BY UNDOING THE TRAINING"            # the first and not the second: not taken
NOT_THIS_LEVER = "NOT THIS LEVER AT THIS SIZE"                          # neither
NOT_NAMED = "IT KEEPS THE GAIN AND DOES NOT GIVE THE BREADTH BACK"     # the second and not the first: the old rule's result again; not taken
BRANCHES = (GIVES_BACK, UNDOES, NOT_THIS_LEVER, NOT_NAMED, INCONCLUSIVE, NOT_READ)
NOT_READ_AS_EITHER = "NOT READ AS EITHER"                               # `old_rule` by itself: neither of its two named outcomes


def narrows(rank: int) -> str:
    """`old_rule` by itself, lost more than gained against the start model with the sign test under the level: the cap
    was not the cause of the narrowing, the training rule is. The spec's words are for rank 64; the rank is the start
    adapter's."""
    return f"THE OLD RULE NARROWS AT RANK {rank} TOO"


def no_narrowing(rank: int) -> str:
    """`old_rule` by itself, otherwise, with all of G per attempt against the start model above zero: the old rule may
    stand at this rank."""
    return f"NO NARROWING SHOWN AT RANK {rank}"


def old_rule_outcomes(rank: int) -> tuple[str, ...]:
    """Every outcome `old_rule`'s own read can have at a start adapter of rank `rank`."""
    return (narrows(rank), no_narrowing(rank), NOT_READ_AS_EITHER, INCONCLUSIVE, NOT_READ)


def models_of(start: str) -> tuple[str, ...]:
    """The trained models of a run from `start`, in the order they are trained and measured: the two from `pre`; from
    any other start `old_rule` first, then the two (the cap's question is answered first)."""
    return MODELS if start == PRE else MODELS_FROM_ANOTHER_START


def reference_of(start: str) -> str:
    """The model that stands in `without`'s place in every read: the stored `without` from `pre`; from any other start
    `old_rule`, which this run trains and measures (nothing stored is the old rule at that start's rank)."""
    return WITHOUT if start == PRE else OLD_RULE


# ------------------------------------------------------------------------------------------------ the draws
def uniform(seed: int, label: str, key: str) -> float:
    """A number in [0, 1) from the content hash every draw here uses (`selection.rank`): its first `HASH_DIGITS`
    hexadecimal digits over 16 ** `HASH_DIGITS`. The same wherever and whenever it is asked."""
    return int(rank(seed, label, key)[:HASH_DIGITS], 16) / 16 ** HASH_DIGITS


def rehearsal_id(problem_id: str) -> str:
    """A rehearsal row's id (the pretraining file holds one row a problem): `<problem>#pretraining`."""
    return f"{problem_id}#{PRETRAINING}"


# ---------------------------------------------------------------------------------------------- reward_rows
def with_k(rows: Sequence[Mapping], picks: Sequence[Mapping]) -> list[dict]:
    """Each row of the rounds with its problem's k of n AS THE CHALLENGER READS IT (`assembly.counted`): the verified
    attempts of the n the round gave the problem, and 1 for a problem only assembly resolved. `rows`: the rounds'
    training rows (origins `attempt` and `assembled`). `picks`: every batch's per-problem results.

    Refused (ValueError): a problem picked twice; a row whose problem no round picked; a row that is neither one-shot
    nor assembled (H0's: the rule names no k for it); a one-shot row of a problem no attempt resolved; an assembled
    row of a problem an attempt resolved, or that assembly did not."""
    by_problem: dict[str, tuple[Mapping, Mapping]] = {}
    for pick, read in zip(picks, counted(picks)):
        if pick["problem_id"] in by_problem:
            raise ValueError(f"{pick['problem_id']} was picked twice by the rounds: a problem has one k. Refused")
        by_problem[pick["problem_id"]] = (pick, read)
    result = []
    for row in rows:
        if row["origin"] not in (ATTEMPT, ASSEMBLED):
            raise ValueError(f"the row {row['id']} is of origin {row['origin']!r}: the rule gives a k to a round's one-shot and assembled rows, and to no other. Refused")
        if row["problem_id"] not in by_problem:
            raise ValueError(f"the row {row['id']} is of {row['problem_id']}, which no round picked: it has no k. Refused")
        pick, read = by_problem[row["problem_id"]]
        by_assembly = pick["resolved"] == 0 and bool(pick.get("resolved_by_assembly"))
        if (row["origin"] == ATTEMPT and pick["resolved"] < 1) or (row["origin"] == ASSEMBLED and not by_assembly):
            raise ValueError(f"the row {row['id']} is of origin {row['origin']!r} and its problem's pick has {pick['resolved']} verified attempts"
                             f"{' and an assembled proof' if by_assembly else ''}: the stored rounds do not agree. Refused")
        result.append({**row, "k": read["resolved"], "n": read["episodes"]})
    return result


def reward_draws(rows: Sequence[Mapping], target_rate: float, seed: int) -> list[dict]:
    """Every row (each with its `k` and `n`: `with_k`) with the reward of its problem (`reward`: r(k) at the arm's target
    rate, the challenger's own function), its draw (`draw`: `uniform` of the task's seed and the row's id) and whether
    the rule keeps it (`kept`: the draw is under the reward). A row's draw does not depend on which other rows there are."""
    drawn = []
    for row in rows:
        paid, draw = reward(row["k"], row["n"], target_rate), uniform(seed, KEEP_LABEL, row["id"])
        drawn.append({**row, "reward": round(paid, 6), "draw": draw, "kept": draw < paid})
    return drawn


def kept_rows(draws: Sequence[Mapping], seed: int, at_least: int = 0) -> list[dict]:
    """`reward_rows`: the rows the rule keeps, in the arm's order (`assembly.training_order`), each saying why it is
    there (`kept_by`). `at_least` (a smoke run alone; 0: the rule and nothing else): when the rule keeps fewer rows,
    the rows it did not keep join in the order of their draws, smallest first, until there are that many (or none is
    left), each marked as kept by the run's minimum and NOT by the rule."""
    kept = [{**row, "kept_by": THE_RULE} for row in draws if row["kept"]]
    if len(kept) < at_least:
        others = sorted((row for row in draws if not row["kept"]), key=lambda row: (row["draw"], row["id"]))
        kept += [{**row, "kept_by": THE_MINIMUM} for row in others[:at_least - len(kept)]]
    return training_order(kept, seed)


# ------------------------------------------------------------------------------------------------- rehearse
def rehearsal_rows(pretraining: Sequence[Mapping], count: int, seed: int) -> list[dict]:
    """`count` rows of the pretraining file, drawn among `pretraining` (the rows `pre` was trained on, in the file's
    order: `problem_id`, and whatever else is to be kept of a row) by a content hash of the task's seed and each row's
    id: the `count` rows of the smallest hash. NO ROW TWICE. Each is given its id, its origin (`pretraining`) and its
    place in the file (`row_of_the_file`); it has no k. Refused (ValueError): a problem twice among the rows, and fewer
    rows than `count`."""
    twice = sorted(problem for problem, times in Counter(row["problem_id"] for row in pretraining).items() if times > 1)
    if twice:
        raise ValueError(f"{len(twice)} problems stand twice among the rows `pre` was trained on (first: {twice[0]}): a rehearsal row is drawn once. Refused")
    if count > len(pretraining):
        raise ValueError(f"`pre` was trained on {len(pretraining)} rows and {count} are asked for the rehearsal (as many as the twin's): no row is drawn twice. Refused")
    placed = [{**row, "id": rehearsal_id(row["problem_id"]), "side": STATEMENT_SIDE, "origin": PRETRAINING, "k": None, "row_of_the_file": position}
              for position, row in enumerate(pretraining)]
    return sorted(placed, key=lambda row: rank(seed, REHEARSAL_LABEL, row["id"]))[:count]


def rehearse_set(twin: Sequence[Mapping], rehearsal: Sequence[Mapping], seed: int) -> list[Mapping]:
    """`rehearse`: the twin's rows and the rehearsal rows in ONE order, the arm's content hash. Refused (ValueError)
    unless there are as many rehearsal rows as twin's rows, and no id stands twice."""
    if len(rehearsal) != len(twin):
        raise ValueError(f"the rehearsal holds {len(rehearsal)} rows and the twin {len(twin)}: `rehearse` is the twin's rows and AS MANY of the pretraining file. Refused")
    rows = [*twin, *rehearsal]
    twice = sorted(key for key, times in Counter(row["id"] for row in rows).items() if times > 1)
    if twice:
        raise ValueError(f"{len(twice)} ids stand twice in `rehearse` (first: {twice[0]}). Refused")
    return training_order(rows, seed)


# --------------------------------------------------------------------------------------------- what is barred
def barred(rows: Sequence[Mapping], heldout: Collection[str], half_seed: int) -> dict:
    """Of a training set's rows (`problem_id`, `origin`): the held-out problems among them; among the ROUNDS' rows (every
    origin but `pretraining`) the problems of the `pretrain` half; and among the rehearsal rows the problems that are
    NOT of the `pretrain` half (a rehearsal row is a row of the pretraining file, which is that half and no other)."""
    rounds = [row for row in rows if row["origin"] != PRETRAINING]
    rehearsal = [row for row in rows if row["origin"] == PRETRAINING]
    return {"rows": len(rows), "rows_of_the_rounds": len(rounds), "rehearsal_rows": len(rehearsal),
            "held_out": sorted({row["problem_id"] for row in rows if row["problem_id"] in heldout}),
            "of_the_pretrain_half_among_the_rounds_rows": sorted({row["problem_id"] for row in rounds if half_of(row["problem_id"], half_seed) != LOOP}),
            "rehearsal_rows_not_of_the_pretrain_half": sorted({row["problem_id"] for row in rehearsal if half_of(row["problem_id"], half_seed) != PRETRAIN})}


def refuse_barred(rows: Sequence[Mapping], heldout: Collection[str], half_seed: int, what: str) -> dict:
    """A training set is refused (ValueError) BEFORE anything is trained when it holds a held-out problem, a round's row
    of the `pretrain` half, or a rehearsal row that is not of the `pretrain` half. Returns what was counted."""
    found = barred(rows, heldout, half_seed)
    for key, said in (("held_out", "are held-out problems"), ("of_the_pretrain_half_among_the_rounds_rows", "are rounds' rows of the `pretrain` half"),
                      ("rehearsal_rows_not_of_the_pretrain_half", "are rehearsal rows that are not of the `pretrain` half")):
        if found[key]:
            raise ValueError(f"{len(found[key])} problems of {what} {said} (first: {found[key][0]}): refused, nothing was written")
    return found


def no_barred_row(trained: Sequence[Mapping], heldout: Collection[str], half_seed: int) -> dict:
    """The fourth check of a trained model, read on the rows the TRAINING LOOP recorded (not on the prepared file): no
    row it was trained on is a held-out problem or, among its rounds' rows, a problem of the `pretrain` half. The
    rehearsal rows ARE of the `pretrain` half (published proofs `pre` was trained on) and are named as such."""
    found = barred(trained, heldout, half_seed)
    first = [*found["held_out"], *found["of_the_pretrain_half_among_the_rounds_rows"], *found["rehearsal_rows_not_of_the_pretrain_half"]]
    return {"what": "no row it was trained on is a held-out problem or, among its rounds' rows, a problem of the `pretrain` half; its rehearsal rows are rows of "
                    "the pretraining file, of the `pretrain` half, and are named as such",
            "rows": found["rows"], "rows_of_the_rounds": found["rows_of_the_rounds"], "rehearsal_rows_of_the_pretrain_half": found["rehearsal_rows"],
            "held_out_problems": len(found["held_out"]), "problems_of_the_pretrain_half_among_the_rounds_rows": len(found["of_the_pretrain_half_among_the_rounds_rows"]),
            "rehearsal_rows_not_of_the_pretrain_half": len(found["rehearsal_rows_not_of_the_pretrain_half"]), "first": first[:5], "passes": bool(trained) and not first}


# ----------------------------------------------------------------------------------------------- the branch
def smallest_decisive_split(changed: int, level: float = SIGN_TEST_LEVEL) -> dict:
    """The count that would have been seen: of `changed` problems that differ between two models, the least number
    GAINED (more than lost) at which the two-sided sign test is under `level` (`rounds.sign_test`, the one the read
    uses). `gained` is None when no split of that many problems is decisive (fewer than six, at 0.05)."""
    for gained in range(changed // 2 + 1, changed + 1):
        p = sign_test(gained, changed - gained)
        if p is not None and p < level:
            return {"changed": changed, "gained": gained, "lost": changed - gained, "sign_test_p": p, "level": level}
    return {"changed": changed, "gained": None, "lost": None, "sign_test_p": None, "level": level}


def _seen(split: Mapping) -> str:
    if split["gained"] is None:
        return (f"with the {split['changed']} problems that changed, no split is decisive at {split['level']:g}: a difference in breadth could not have been "
                "seen at this size")
    return (f"the count that would have been seen: with the {split['changed']} problems that changed, the sign test's smallest decisive split is "
            f"{split['gained']} gained to {split['lost']} lost (p = {split['sign_test_p']})")


def rows_branch(checks: Mapping, breadth: Mapping, kept: Mapping, level: float = SIGN_TEST_LEVEL, against: str = WITHOUT, start: str = PRE,
                against_failed: Sequence[str] | None = None) -> dict:
    """One trained model's branch, fixed before the run. `breadth` (the primary): the goal problems solved at least
    once in all the attempts, the model against `without`, paired by problem (`gained`, `lost`, `sign_test_p`). `kept`
    (beside it): all of G, successes per attempt, the model minus `pre`, with its interval.

    A check failed: INCONCLUSIVE, by itself. THE FIRST: gained more than lost with the sign test under `level`. THE
    SECOND: the interval for all of G against `pre` is above zero. Both: the rule gives the breadth back and keeps the
    gain. The first and not the second: it gives the breadth back by undoing the training. Neither: not this lever at
    this size, with the count that would have been seen. THE SECOND AND NOT THE FIRST (named in the spec on 2026-10-09,
    before any run): it keeps the gain and does not give the breadth back, the old rule's result again; not taken, with
    the same count.

    FROM ANOTHER START THAN `pre`: `against` is the model in `without`'s place (`old_rule`, measured by the same run)
    and `start` the one in `pre`'s; the four names are the same. `against_failed`: the checks `against` itself failed,
    when it is a model of this run and is INCONCLUSIVE: the primary is then NOT READ against it, and says so."""
    failed = [name for name, check in checks.items() if not check["passes"]]
    if failed:
        return {"name": INCONCLUSIVE, "failed_checks": failed,
                "reason": f"this model could not have been read ({', '.join(failed)}): nothing is said about its training rule. The other models are read by themselves"}
    if against_failed:
        return {"name": NOT_READ, "failed_checks": [], "not_read_against": against,
                "reason": f"`{against}`, which stands in `without`'s place, is INCONCLUSIVE ({', '.join(against_failed)}): this model's primary is against it and is not "
                          "read; nothing is said about its training rule"}
    if not breadth.get("problems") or kept.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": "no goal problem was measured on both sides: the primary has nothing to be read on"}
    gained, lost, p = breadth["gained"], breadth["lost"], breadth["sign_test_p"]
    first, second = bool(gained > lost and p is not None and p < level), bool(kept["low"] > 0)
    split = smallest_decisive_split(gained + lost, level)
    counts = f"gained {gained}, lost {lost}" + (f", p = {p}" if p is not None else ", no problem changed")
    interval = f"{kept['mean']:+.5f} [{kept['low']:+.5f}, {kept['high']:+.5f}]"
    read = {"failed_checks": [], "the_first": first, "the_second": second, "smallest_decisive_split": split}
    if first and second:
        return {"name": GIVES_BACK, **read,
                "reason": f"gained more than lost against `{against}` with the sign test under {level:g} ({counts}), and the interval for all of G against `{start}` is "
                          f"above zero ({interval}): the rule gives the breadth back and keeps the gain; it replaces the old rule in the next arm"}
    if first:
        return {"name": UNDOES, **read,
                "reason": f"gained more than lost against `{against}` with the sign test under {level:g} ({counts}), and the interval for all of G against `{start}` is "
                          f"not above zero ({interval}): it gives the breadth back by undoing the training, and is not taken"}
    if second:
        return {"name": NOT_NAMED, **read,
                "reason": f"the breadth is not given back (against `{against}`: {counts}; not gained more than lost with the sign test under {level:g}) and the gain "
                          f"is kept (all of G against `{start}`: {interval}, above zero): it keeps the gain and does not give the breadth back, which is the old "
                          f"rule's result again; it does not replace the old rule. For the breadth, {_seen(split)}"}
    return {"name": NOT_THIS_LEVER, **read,
            "reason": f"neither (against `{against}`: {counts}; all of G against `{start}`: {interval}, not above zero): not this lever at this size. For the breadth, "
                      f"{_seen(split)}"}


def old_rule_branch(checks: Mapping, breadth: Mapping, kept: Mapping, rank: int, start: str, level: float = SIGN_TEST_LEVEL) -> dict:
    """`old_rule` READ BY ITSELF, which is the cap's question (spec, "L4r's first branch was taken: what L4t runs").
    `breadth`: the goal problems solved at least once in all the attempts, `old_rule` against the start model, paired by
    problem (`gained`, `lost`, `sign_test_p`). `kept`: all of G, successes per attempt, `old_rule` minus the start model,
    with its interval. `rank`: the start adapter's, which every model of the run has.

    A check failed: INCONCLUSIVE (and the two other models are then not read against it). LOST MORE THAN GAINED with
    the sign test under `level`: the old rule narrows at this rank too (the cap was not the cause; the training rule
    is). Otherwise, with the interval for all of G above zero: no narrowing shown at this rank (the old rule may stand
    at this rank), with the smallest split at which a narrowing would have been seen. Otherwise: not read as either."""
    failed = [name for name, check in checks.items() if not check["passes"]]
    if failed:
        return {"name": INCONCLUSIVE, "failed_checks": failed,
                "reason": f"`{OLD_RULE}` could not have been read ({', '.join(failed)}): nothing is said about the old rule at rank {rank}, and the models read against "
                          "it are not read"}
    if not breadth.get("problems") or kept.get("mean") is None:
        return {"name": NOT_READ, "failed_checks": [], "reason": "no goal problem was measured on both sides: there is nothing to be read on"}
    gained, lost, p = breadth["gained"], breadth["lost"], breadth["sign_test_p"]
    narrowed, above = bool(lost > gained and p is not None and p < level), bool(kept["low"] > 0)
    split = smallest_decisive_split(gained + lost, level)
    counts = f"gained {gained}, lost {lost}" + (f", p = {p}" if p is not None else ", no problem changed")
    interval = f"{kept['mean']:+.5f} [{kept['low']:+.5f}, {kept['high']:+.5f}]"
    seen = (f"with the {split['changed']} problems that changed, no split is decisive at {level:g}: a narrowing could not have been seen at this size"
            if split["gained"] is None else
            f"a narrowing would have been seen at {split['gained']} lost to {split['lost']} gained (the sign test's smallest decisive split with the "
            f"{split['changed']} problems that changed, p = {split['sign_test_p']})")
    read = {"failed_checks": [], "lost_more_than_gained": narrowed, "the_gain_is_above_zero": above, "smallest_decisive_split": split, "rank": rank}
    if narrowed:
        return {"name": narrows(rank), **read,
                "reason": f"lost more than gained against `{start}` with the sign test under {level:g} ({counts}; all of G per attempt against `{start}`: {interval}): "
                          f"the old rule narrows at rank {rank} too. The cap was not the cause; the training rule is"}
    if above:
        return {"name": no_narrowing(rank), **read,
                "reason": f"not lost more than gained against `{start}` with the sign test under {level:g} ({counts}), and the interval for all of G per attempt "
                          f"against `{start}` is above zero ({interval}): no narrowing shown at rank {rank}; the old rule may stand at this rank. For the breadth, {seen}"}
    return {"name": NOT_READ_AS_EITHER, **read,
            "reason": f"not lost more than gained against `{start}` with the sign test under {level:g} ({counts}), and the interval for all of G per attempt against "
                      f"`{start}` is not above zero ({interval}): not read as either. For the breadth, {seen}"}
