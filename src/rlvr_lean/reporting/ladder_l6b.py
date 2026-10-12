"""L6b's report: a second round, from the best model. Does the improved model's own search yield proofs that add? Spec:
docs/spec/ladder-loop.spec.md, "L6b: a second round, from the best model" (its checks, its reads, what follows).
Pure: rows in, a report out, on the rules of `domain/ladder_round/l6b.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. A model is ONE sampling of the goal set: no rung, no second sampling, no held-aside set.

  the checks        FIRST, each PASS or FAIL. This stage's: B is L5f's averaged `published` of the seed, as the first
                    step held it and as its report reads now, and the search and every training were served and
                    started from it; L5c's run of the seed holds the bar (its report read AGAIN, the next rows of the
                    seed's order held to its stored `again` set); the searched set is L6's search 2 made again
                    (`l6b.the_searched_is_l6s`), and the new solves made again from the stored search are as many as
                    were counted; each set is its rule made again (`l6b.the_sets_are_the_rule`); the stored first
                    samplings of B and of the bar are the setting's; Lean answered the search. Then L5d's own checks of
                    the models BY L5d's OWN FUNCTIONS (`reporting/ladder_l5_draws.py`: `checks_of_the_models`,
                    `lines_of_the_checks`). ANY FAIL: THE RUN IS INCONCLUSIVE and nothing else is said (what was
                    measured is kept under `measured_and_not_read`)
  the reads         THE SEARCH (`l6b.the_yield`); THE PRIMARY, `new_own` minus L5c's stored `again` (`the_primary`);
                    WHOSE PROOFS, `new_own` minus `old_own` (`whose_proofs`); beside, naming nothing, `old_own` minus
                    the bar and each of the three against B's, `second_full`'s and `pre_r64`'s stored first samplings
                    (`against_the_start`, L5s's own); the timeouts; what follows (`what_follows`)

TOO FEW NEW SOLVES: no model was trained; the checks of the run are read, the search is the finding, and both reads are
NOT READ. A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task
queued again samples such a set again from its adapter.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l5 import timed_out
from rlvr_lean.domain.ladder_round.l5_cont import against_the_start
from rlvr_lean.domain.ladder_round.l5_draws import AGAIN as G_AGAIN
from rlvr_lean.domain.ladder_round.l5_draws import ALL, SOLVED, on_the_parts, the_counts, the_parts, the_start_is_the_settings
from rlvr_lean.domain.ladder_round.l5_measure import reliably
from rlvr_lean.domain.ladder_round.l5_noise import figure
from rlvr_lean.domain.ladder_round.l6 import PART_A, PART_B
from rlvr_lean.domain.ladder_round.l6b import (
    BAR,
    MODELS,
    NEW_OWN,
    OLD_OWN,
    START,
    hold_the_next,
    of_l5c,
    published_next,
    the_primary,
    the_searched_is_l6s,
    the_sets_are_the_rule,
    what_follows,
    whose_proofs,
)
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids
from rlvr_lean.reporting.ladder_ceiling import goal_sets, lean_did_not_answer
from rlvr_lean.reporting.ladder_l2 import MAXIMUM_SHARE_WITHOUT_AN_ANSWER, _health
from rlvr_lean.reporting.ladder_l3c import LENGTH_GROUPS
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY, _verdict
from rlvr_lean.reporting.ladder_l5_cont import l5fs_run
from rlvr_lean.reporting.ladder_l5_draws import PART_SAID, THE_RUN, THE_SAMPLING, _per_1000, checks_of_the_models, lines_of_the_checks
from rlvr_lean.reporting.ladder_l5_draws import THE_START as THE_STARTS_SAMPLING
from rlvr_lean.reporting.ladder_l5_noise import LEAN_ANSWERED

L4 = "l4"
QUESTION = "a second round, from the best model"
SPEC = "docs/spec/ladder-loop.spec.md, L6b: a second round, from the best model"
LONG = LENGTH_GROUPS[-1]                    # the goal problems whose shortest published proof has 8 lines or more
THE_BEST = f"l5f_{START}"                   # B, L5f's stored averaged `published` of the seed, as the counts and the timeouts name it
THE_BAR = f"l5c_{BAR}"                      # L5c's stored `again` of the seed, likewise
B_IS_L5FS = "b_is_l5fs_averaged_published_and_everything_here_is_from_it"
L5CS_RUN = "l5cs_run_of_the_seed_holds_the_bar"
THE_SEARCHED = "the_searched_set_is_l6s_search_2"
THE_SETS = "each_set_is_its_rule_made_again"
THE_BARS_SAMPLING = "the_stored_first_sampling_of_the_bar_is_the_settings"
SEARCH_ANSWERED = "lean_answered_the_search"
CANNOT_SAY = [
    "ONE SEED, one search and one draw of its rows: a scout, by the repo's seed rule. Each read names what follows, and no effect",
    "G alone and one sampling of it, 32 attempts a problem: nothing of the rungs, of the second sampling or of the held-aside set",
    "THE BAR is published rows that B's lineage has read twice before: a gain over it says the model's own new proofs are worth more than a further reading of published "
    "ones, and not that they are worth more than PUBLISHED proofs of new problems",
    "K rows of the model's own in a set (at most half of it): nothing of a larger dose",
    "The new solves are of HARDER problems than the old ones (problems the first search left unsolved, or that no search took): WHOSE PROOFS does not separate \"the "
    "stronger model found them\" from \"they are of harder problems\"",
    "B is one averaged model of one seed, and every model here, and the bar, inherits it"]


def _signed(value: float | None) -> str:
    return "not read" if value is None else _per_1000(value, True)


def the_best_model(of_the_first_step: Mapping, l5f: Mapping, seed: int, recipe: str, measurement: Mapping, searched_from: str | None, trains: Mapping[str, Mapping],
                   adapter: str | None) -> dict:
    """The check of B: L5f's run of the seed, by L5c's own check of it (`reporting/ladder_l5_cont.l5fs_run`: its four
    trainings and its averaged adapter as the first step held them, its report read AGAIN now); and everything here is
    FROM it: the search was served from its adapter, and every training names it as what it was trained from."""
    own = l5fs_run(of_the_first_step, l5f, seed, recipe, measurement)
    not_from = sorted(name for name, entry in trains.items() if entry.get("trained_from") != f"the stored adapter {adapter}")
    served = bool(adapter) and searched_from == adapter
    return {**own, "what": f"B is L5f's stored averaged `{START}` of this seed: that run is there, finished and not {INCONCLUSIVE}, of this seed and recipe and measured as "
                           "this run; the averaged adapter is the mean of its four trainings, at the rank, and not the twice-pretrained start's (as the first step held "
                           "them); the search was served from it, and every training here was started from it",
            "adapter": adapter, "the_search_was_served_from": searched_from, "trainings_not_from_it": not_from, "passes": bool(own["passes"]) and served and not not_from}


def l5cs_run(of_the_first_step: Mapping, l5c: Mapping, seed: int, recipe: str, measurement: Mapping, smoke: bool, pretraining: Sequence[Mapping], size: int,
             stored_again: Sequence[str] | None) -> dict:
    """The check of L5c's run of the seed: its report, read AGAIN now, still carries this stage (`l6b.of_l5c`; a SMOKE
    run reads one that is INCONCLUSIVE all the same), its `again` was trained from B (as the first step held it), and
    the next rows of the seed's order of the pretraining file are, id for id, the set that run stores as `again` NOW."""
    why = of_l5c(l5c, seed, recipe, measurement, though_inconclusive=smoke)
    try:
        held, refused = hold_the_next(published_next(pretraining, seed, size), seed, stored_again), None
    except ValueError as error:
        held, refused = None, str(error)
    return {"what": f"L5c's run of this seed is there, finished and not {INCONCLUSIVE}, of this seed and recipe and measured as this run; its `{BAR}` was trained from B; "
                    f"and the next rows of the seed's order are the set it stores as `{BAR}`, id for id",
            "run": of_the_first_step.get("run"), "trained_from": of_the_first_step.get("trained_from"), "its_report_now": why is None, "why_not": why,
            "read_though_inconclusive": bool(smoke and l5c.get("inconclusive")), "the_next_rows_are_its_again": held is not None, "rows": None if held is None else held.get("rows"),
            "refused": refused, "passes": why is None and held is not None}


def the_searched(of_the_search: Mapping, prepare: Mapping, new_again: Sequence[Mapping], new_refused: str | None, goal_ids: Sequence[str], aside: Sequence[str],
                 heldout: Sequence[str]) -> dict:
    """The check of the searched set (`l6b.the_searched_is_l6s`), and that the new solves made again from the stored
    search are as many as the sets step counted."""
    again = of_the_search["again"]
    own = the_searched_is_l6s(of_the_search["problems"], again["problems"], again["refused"], of_the_search["results"], prepare["the_search"]["episodes_each"], heldout,
                              goal_ids, aside, again["solved_by_search_one"])
    counted, made = prepare["new_solves"], None if new_refused else len(new_again)
    return {**own, "new_solves": counted, "new_solves_made_again": made, "new_solves_refused": new_refused, "passes": own["passes"] and made == counted}


def build_l6b_report(prepare: Mapping, of_the_search: Mapping, trains: Mapping[str, Mapping], trained: Mapping[str, Sequence[Mapping]], averaged: Mapping[str, Mapping],
                     measured: Mapping[str, Mapping], rows: Mapping[str, Sequence[Mapping]], sets: Mapping[str, Sequence[Mapping]], pretraining: Sequence[Mapping],
                     now: Mapping, stored: Mapping[str, Sequence[Mapping]], groups: Sequence[Mapping], lengths: Mapping[str, Mapping], again: Sequence[str],
                     aside: Sequence[str], settings: Mapping, base_map: Sequence[str], of_the_setting: Mapping) -> dict:
    """`prepare`: the sets step's summary. `of_the_search`: the search step's `summary`, the search's stored `problems`
    and per-problem `results`, L6's rule made `again` on the stored runs as they are now (`problems`, or None with
    `refused`; `solved_by_search_one`), and the `heldout` problems. `trains`, `trained`, `averaged`, `measured`, `rows`,
    `sets`, `pretraining`, `groups`, `lengths`, `again` (G'), `aside`, `settings`, `base_map`, `of_the_setting`: as
    L5d's report takes them (`reporting/ladder_l5_draws.build_l5_draws_report`); all empty of models when nothing was
    trained. `now`: what the report step read again: `l5f` and `l5c` (what each run's report says of its own seed, or
    `{"there": False, "why": ...}`), `l5cs_again` (the ids L5c's run stores as `again`), `old` (the own rows of L5d's
    stored `deep` set), `new` (B's new solves made again from the stored search; `new_refused` when that refused).
    `stored`: the stored first samplings as this run copied them: `best` (B), `bar` (L5c's `again`), `second_start`
    (`second_full`), `first_start` (`pre_r64`)."""
    start, first, second, seed, sampling = prepare["start"], prepare["first_start"], prepare["second_start"], prepare["seed"], prepare["the_sampling"]
    ran = bool(prepare["trainings"]["run"])
    models = list(MODELS) if ran else []
    if start != START or sorted(rows) != sorted(models) or sorted(trains) != sorted(prepare["trained"]) or sorted(averaged) != sorted(prepare["averaged"]):
        raise ValueError(f"the report is given a run from `{start}` with the rows of {sorted(rows)}, the trainings of {sorted(trains)} and the averages of {sorted(averaged)}: it "
                         f"reads a run from L5f's `{START}` and, when its trainings ran, its models ({list(MODELS)}), each measured")
    goal_ids = group_ids(groups, GOAL)
    if any(sorted(row["problem_id"] for row in own) != sorted(goal_ids) for own in rows.values()):
        raise ValueError("the models were not measured on the goal problems, each once: every model here is one sampling of the one goal set")
    rank, attempts, recipe, smoke = (prepare.get("start_recipe") or {}).get("rank"), sampling["episodes"], prepare["recipe"]["name"], prepare.get("smoke")
    long_ids = goal_sets(goal_ids, lengths).get(LONG, [])
    of_the_best, of_the_bars_run, search, enough = prepare["the_best"], prepare["l5cs_run"], prepare["the_search"], prepare["enough_new_solves"]
    the_measurement = {"sampling": sampling["name"], "episodes": attempts, "sampling_seed": sampling["sampling_seed"], "problems": len(goal_ids)}
    size, results = of_the_best["rows_of_its_set"], of_the_search["results"]

    # ---- the checks, FIRST: this stage's of the run, then (when its trainings ran) of the sets, and L5d's own of the models
    health = _health(results)
    checks: dict[str, dict] = {THE_RUN: {
        B_IS_L5FS: the_best_model(of_the_best, now["l5f"], seed, recipe, the_measurement, of_the_search["summary"].get("adapter"), trains, prepare.get("start_adapter")),
        L5CS_RUN: l5cs_run(of_the_bars_run, now["l5c"], seed, recipe, the_measurement, bool(smoke), pretraining, size, now["l5cs_again"]),
        THE_SEARCHED: the_searched(of_the_search, prepare, now["new"], now["new_refused"], goal_ids, aside, [*of_the_search["heldout"], *(row["problem_id"] for row in groups)]),
        THE_STARTS_SAMPLING: the_start_is_the_settings(stored["best"], goal_ids, sampling, of_the_setting["sampling"], of_the_setting["attempts"]),
        THE_BARS_SAMPLING: the_start_is_the_settings(stored["bar"], goal_ids, sampling, of_the_setting["sampling"], of_the_setting["attempts"]),
        SEARCH_ANSWERED: {"what": f"Lean gave a verdict on all but at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of the search's attempts", "maximum": MAXIMUM_SHARE_WITHOUT_AN_ANSWER,
                          "share_without_an_answer": health["share_without_an_answer"], "passes": not lean_did_not_answer(results)}}}
    if ran:
        in_orders = {name: sets[name] for name in prepare["trained"]}
        checks[THE_RUN][THE_SETS] = the_sets_are_the_rule({model: sets[model] for model in MODELS}, in_orders, now["new"], now["old"] or [], pretraining, seed, now["l5cs_again"])
        checks.update(checks_of_the_models(prepare, trains, trained, averaged, measured, rows, sets, goal_ids, aside, [row["problem_id"] for row in groups],
                                           settings["half_seed"], base_map))
    failed = [f"{'the run' if name == THE_RUN else f'`{name}`'}: {check}" for name, of in checks.items() for check, entry in of.items() if not entry["passes"]]
    inconclusive = bool(failed)
    not_to_be_read = [name for name in models if not checks[name][LEAN_ANSWERED]["passes"]]

    # ---- the figures; the parts of G are THE FIRST START's own first sampling's (the goal problems it solves there, and G'), held to the G' the first step copied
    try:
        parts = the_parts(stored["first_start"], again)
    except ValueError:      # the stored sampling is not the one G' was made from: no figure is given on a part
        parts = {ALL: list(goal_ids), SOLVED: [], G_AGAIN: []}
    share_distinct = lambda own: ((own.get("distinct_attempts") or {}).get(GOAL) or {}).get("mean_share_distinct")      # noqa: E731
    figures = {name: {part: {**own, **({"share_of_distinct_attempts": share_distinct(measured[name])} if part == ALL else {})}
                      for part, own in on_the_parts(rows[name], parts).items()} for name in models}
    stored_figures = {THE_BEST: on_the_parts(stored["best"], parts), THE_BAR: on_the_parts(stored["bar"], parts), second: on_the_parts(stored["second_start"], parts),
                      first: on_the_parts(stored["first_start"], parts)}
    on_g, of_the_bar = {name: figures[name][ALL] for name in models}, stored_figures[THE_BAR][ALL]

    # ---- THE READS: the search; the primary; whose proofs; beside; the timeouts; what follows
    found = prepare["the_search_found"]
    primary, whose = the_primary(failed, on_g.get(NEW_OWN), of_the_bar), whose_proofs(failed, on_g.get(NEW_OWN), on_g.get(OLD_OWN))
    if not ran and not inconclusive:
        primary = whose = {"name": NOT_READ, "reason": f"no model was trained ({prepare['trainings'].get('why')})"}
    follows = what_follows(primary, whose, bool(enough["passes"]) or inconclusive)
    read_rows = {THE_BAR: stored["bar"], **{name: rows[name] for name in models}}
    stored_rows = {THE_BEST: stored["best"], second: stored["second_start"], first: stored["first_start"]}
    against = {name: {other: against_the_start(own, theirs, parts, long_ids) for other, theirs in stored_rows.items()} for name, own in read_rows.items()}
    counts = the_counts({THE_BEST: stored["best"], **read_rows}, long_ids, against=THE_BEST)
    reliable = {name: len(reliably(own, goal_ids)) for name, own in {THE_BEST: stored["best"], **read_rows}.items()}
    old_minus_bar = (None if not ran or None in (on_g[OLD_OWN]["per_attempt"], of_the_bar["per_attempt"]) else round(on_g[OLD_OWN]["per_attempt"] - of_the_bar["per_attempt"], 6))
    shares = {THE_BEST: share_distinct(of_the_best.get("wrote") or {}), THE_BAR: share_distinct(of_the_bars_run.get("wrote") or {}),
              **{name: figures[name][ALL]["share_of_distinct_attempts"] for name in models}}
    of_timeouts = timed_out({**{name: _health(own) for name, own in read_rows.items()}, THE_BEST: _health(stored["best"])},
                            {THE_SAMPLING: {name: name for name in (*models, THE_BAR, THE_BEST)}})
    of_timeouts.update({"checked_again": False, "why": "no step checks the timed-out attempts again: where a pair is named here, it is to be done by hand (Lean only, no "
                                                       "measurement running) and the reads given both ways"})
    difference = lambda read: read.get("difference")      # noqa: E731
    of_this_seed = {"seed": seed, "start": start, "recipe": recipe, "ok": not not_to_be_read, "inconclusive": inconclusive, "failed_checks": failed,
                    "the_measurement": the_measurement, "trainings_run": ran, "new_solves": None if inconclusive else prepare["new_solves"],
                    "figures": None if inconclusive or not ran else figures, "the_bar": None if inconclusive else stored_figures[THE_BAR],
                    "the_reads": None if inconclusive or not ran else {"primary": difference(primary), "whose_proofs": difference(whose)}}

    of_b, of_their_run, of_the_searched, of_the_bests, of_the_bars, of_the_answer = (checks[THE_RUN][key] for key in (
        B_IS_L5FS, L5CS_RUN, THE_SEARCHED, THE_STARTS_SAMPLING, THE_BARS_SAMPLING, SEARCH_ANSWERED))
    by_part, dose = search["by_part"], prepare.get("K")
    lines = [
        f"{SAY}: L6b, {QUESTION.upper()}. Seed {seed}. B is L5f's stored averaged `{START}` of this seed ({of_the_best['run']}), ONLY READ. B, served at rank {rank}, made "
        f"{search['episodes_each']} attempts at each of {search['problems']:,} problems of L6's search 2 ({by_part[PART_A]:,} that round 1's search attempted and left "
        f"unsolved, {by_part[PART_B]:,} candidates neither draw took), sampling seed {search['sampling_seed']}; the held-aside set was not searched. "
        + (f"TWO MODELS FROM B, the averaged recipe (four orders, one pass, the recipe as it is), {size:,} rows each, K = {dose}: `{NEW_OWN}` (K of B's new solves and the "
           f"first {size - dose:,} of the next published rows of the seed's order) and `{OLD_OWN}` (the first K own rows of L5d's `deep` set of this seed, `{first}`'s deep "
           f"solves, and THE SAME published rows). THE BAR is L5c's stored `{BAR}` ({of_the_bars_run['run']}): from the same B, the next {size:,} published rows, the same "
           f"recipe. EACH MODEL IS MEASURED on the goal set's first sampling alone: {attempts} attempts at each of {len(goal_ids)} goal problems, sampling seed "
           f"{sampling['sampling_seed']}; every read is paired by problem" if ran else f"NO MODEL WAS TRAINED: {prepare['trainings'].get('why')}")]
    if smoke:
        lines.append(f"{SAY}: A SMOKE RUN: L5's smoke run " + (f"reads INCONCLUSIVE ({', '.join(smoke['l5s_failed_checks'])}) and was read all the same; "
                                                              if smoke.get("l5_read_though_inconclusive") else "was read; ")
                     + ("L5c's smoke run reads INCONCLUSIVE and was read all the same; " if smoke.get("l5c_read_though_inconclusive") else "")
                     + f"the search attempts at least {smoke['problems_at_least']} problems ({search['taken_by_the_runs_minimum']} NOT by the rule), one new solve is enough, "
                       "and orders of a set's handful of rows that are alike are trained all the same. Nothing here is a read of the second round")
    lines += [
        f"{SAY}: CHECK, B is L5f's averaged `{START}` of this seed, and everything here is from it ({of_b['adapter']}): its four trainings and its averaged adapter held by "
        f"the first step: {of_b['adapters_held_by_the_first_step']}" + ("; the files themselves were read" if of_b["read_now_of_the_adapters"] else
                                                                      "; its files were not read by the first step (no GPU box)")
        + "; its report now: " + ("read, of this seed and recipe, measured alike" if of_b["its_report_now"] else f"NOT ({of_b['why_not']})")
        + f"; the search was served from {of_b['the_search_was_served_from']}; trainings not from it: {of_b['trainings_not_from_it']}: {_verdict(of_b)}",
        f"{SAY}: CHECK, L5c's run of this seed holds the bar ({of_their_run['run']}): its report now: "
        + ("read, of this seed and recipe, measured alike" + (" (INCONCLUSIVE, and read all the same: a smoke run)" if of_their_run["read_though_inconclusive"] else "")
           if of_their_run["its_report_now"] else f"NOT ({of_their_run['why_not']})")
        + f"; its `{BAR}` was trained from {of_their_run['trained_from']}; the next rows of the seed's order are its stored `{BAR}` set: "
        + (f"{of_their_run['rows']:,} rows, id for id" if of_their_run["the_next_rows_are_its_again"] else f"NOT ({of_their_run['refused']})") + f": {_verdict(of_their_run)}",
        f"{SAY}: CHECK, the searched set is L6's search 2 ({of_the_searched['problems']:,} problems; the rule made again gives {of_the_searched['problems_of_the_rule']}"
        + ("" if of_the_searched["refused"] is None else f", REFUSED: {of_the_searched['refused']}")
        + f"): the rule made again: {of_the_searched['is_the_rule_made_again']}; held out {of_the_searched['held_out']}, of G {of_the_searched['of_the_goal_set']}, held aside "
        f"{of_the_searched['of_the_held_aside_set']}, solved by round 1's search {of_the_searched['solved_by_search_one']}; sampled as prepared: "
        f"{of_the_searched['sampled_as_prepared']} (attempts a problem that are not {search['episodes_each']}: {of_the_searched['attempts_a_problem_that_are_not_the_settings']}); "
        f"new solves {of_the_searched['new_solves']}, made again from the stored search {of_the_searched['new_solves_made_again']}: {_verdict(of_the_searched)}",
        f"{SAY}: CHECK, Lean answered the search (at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of its attempts without an answer): {of_the_answer['share_without_an_answer']}: "
        f"{_verdict(of_the_answer)}",
        f"{SAY}: CHECK, the stored first sampling of the bar (L5c's `{BAR}`) is the setting's ({of_the_bars['problems']} problems of {of_the_bars['goal_problems']}, sampling "
        f"seed {of_the_bars['sampling_seed']}, attempts a problem that are not {attempts}: {of_the_bars['attempts_a_problem_that_are_not_the_settings']}): "
        f"{_verdict(of_the_bars)}"]
    if ran:
        of_the_sets = checks[THE_RUN][THE_SETS]
        lines += [
            f"{SAY}: CHECK, each set is its rule made again (K = {of_the_sets['K']} of {of_the_sets['new_solves']} new solves and of {of_the_sets['old_own_rows']} old own rows; "
            f"`{NEW_OWN}` {of_the_sets['rows'][NEW_OWN]:,} rows, `{OLD_OWN}` {of_the_sets['rows'][OLD_OWN]:,}): "
            + ("made again" if of_the_sets["refused"] is None else f"REFUSED ({of_the_sets['refused']})")
            + f"; sets that are not the rule: {of_the_sets['sets_that_are_not_the_rule']}; published rows with text: {of_the_sets['published_rows_with_text']}; the two hold the "
            f"same published rows: {of_the_sets['the_two_hold_the_same_published_rows']}; the next rows are L5c's `{BAR}` set: {of_the_sets['the_next_rows_are_l5cs_again']}; sets "
            f"not in their order seed's order: {of_the_sets['sets_not_in_their_order_seeds_order']}: {_verdict(of_the_sets)}",
            *lines_of_the_checks(prepare, checks, rank, attempts, not_to_be_read)]
    else:
        lines.append(f"{SAY}: CHECK, the stored first sampling of B (L5f's `{START}`) is the setting's ({of_the_bests['problems']} problems of {of_the_bests['goal_problems']}, "
                     f"sampling seed {of_the_bests['sampling_seed']}, attempts a problem that are not {attempts}: "
                     f"{of_the_bests['attempts_a_problem_that_are_not_the_settings']}): {_verdict(of_the_bests)}")
    if inconclusive:
        lines.append(f"{SAY}: THIS RUN IS {INCONCLUSIVE}: {'; '.join(failed)}. Nothing else is said of it")
    else:
        of_lines = found["lines"]
        lines.append(
            f"{SAY}: THE SEARCH, a finding of its own: B solved {found['new_solves']:,} of the {found['problems']:,} problems (NEW SOLVES; {enough['minimum']} are asked for a "
            f"training): {found[PART_A]['solved']:,} of the {found[PART_A]['problems']:,} that round 1's search left unsolved in its attempts, {found[PART_B]['solved']:,} of the "
            f"{found[PART_B]['problems']:,} that no search took; {found['solved_in_under_half_of_its_attempts']:,} of them in under half of its {found['episodes_each']} "
            f"attempts; successes per 1,000 attempts {found['per_1000']}. Lines of a new solve's row: median {of_lines['new_solves']['median']}, mean "
            f"{of_lines['new_solves']['mean']}, 8 or more {of_lines['new_solves']['share_with_8_lines_or_more']}; beside, of round 1's rows as L5d's set holds them: median "
            f"{of_lines['old_own_rows']['median']}, mean {of_lines['old_own_rows']['mean']}, 8 or more {of_lines['old_own_rows']['share_with_8_lines_or_more']}")
        if ran:
            named = {THE_BEST: f"B (L5f's `{START}`)", THE_BAR: f"the bar (L5c's `{BAR}`)", second: f"`{second}`", first: f"`{first}`", **{name: f"`{name}`" for name in models}}

            def figure_said(part: str) -> str:
                return ", ".join(f"`{name}` {_per_1000(figures[name][part]['per_attempt'])} ({_per_1000(figures[name][part]['sampling_noise'])})" for name in models)

            def against_said(name: str, other: str) -> str:
                own = against[name][other]
                return (f"{named[name]} " + ", ".join(f"{_per_1000(own['per_attempt'][part]['model'])} against {_per_1000(own['per_attempt'][part]['start'])} "
                                                       f"({_signed(own['per_attempt'][part]['difference'])}) on {PART_SAID[part].replace('the start model', f'`{first}`')} "
                                                       f"({own['per_attempt'][part]['problems']})" for part in (ALL, SOLVED, G_AGAIN))
                        + f"; goal problems solved {own['solved']} against {own['solved_by_the_start']} (+{own['gained']}, -{own['lost']}); long ones {own['long']['solved']} "
                          f"against {own['long_solved_by_the_start']}")

            lines += [
                f"{SAY}: THE FIGURES, successes per 1,000 attempts over the {attempts} attempts, each with its binomial sampling noise: on all of G ({len(parts[ALL])} "
                f"problems) {figure_said(ALL)}; on the goal problems `{first}` solves in its own first sampling ({len(parts[SOLVED])}) {figure_said(SOLVED)}; on G' "
                f"({len(parts[G_AGAIN])}) {figure_said(G_AGAIN)}. THE BAR, L5c's stored `{BAR}`, on all of G: {_per_1000(of_the_bar['per_attempt'])}; B: "
                f"{_per_1000(stored_figures[THE_BEST][ALL]['per_attempt'])}; `{second}`: {_per_1000(stored_figures[second][ALL]['per_attempt'])}; `{first}`: "
                f"{_per_1000(stored_figures[first][ALL]['per_attempt'])}",
                f"{SAY}: THE PRIMARY (a scout's read): {primary['name']}. {primary['reason']}",
                f"{SAY}: WHOSE PROOFS (a scout's read): {whose['name']}. {whose['reason']}",
                f"{SAY}: BESIDE, NAMING NOTHING, paired by problem (per 1,000 attempts, the model against the stored model). `{OLD_OWN}` minus the bar on all of G: "
                f"{_signed(old_minus_bar)}. Against B's stored first sampling: " + "; ".join(against_said(name, THE_BEST) for name in read_rows)
                + f". Against `{second}`'s: " + "; ".join(against_said(name, second) for name in read_rows)
                + f". Against `{first}`'s: " + "; ".join(against_said(name, first) for name in read_rows)
                + f". Goal problems solved against B ({counts[THE_BEST]['solved']}): "
                + "; ".join(f"{named[name]} {counts[name]['solved']} (+{counts[name]['gained']}, -{counts[name]['lost']})" for name in read_rows)
                + ". Solved reliably (in half its attempts or more): " + "; ".join(f"{named[name]} {reliable[name]}" for name in (THE_BEST, *read_rows))
                + f". The {len(long_ids)} long goal problems (8 lines or more) solved: " + "; ".join(f"{named[name]} {counts[name]['long']['solved']}" for name in (THE_BEST, *read_rows))
                + ". The share of distinct attempts on G: " + ", ".join(f"{named[name]} {shares[name]}" for name in (*models, THE_BAR, THE_BEST))
                + " (the stored models' as their own runs recorded them)",
                f"{SAY}: TIMEOUTS, each set's share of attempts that timed out in Lean: "
                + "; ".join(f"{named[name]} {of_timeouts['share'][name]}" for name in (*models, THE_BAR, THE_BEST))
                + (f". MORE THAN A POINT APART: {'; '.join(' and '.join(pair['models']) for pair in of_timeouts['apart'])}: the timed-out attempts are to be checked again (Lean "
                   "only, no measurement running) and the reads given both ways; no step does it" if of_timeouts["apart"] else ". No two are more than a point apart")]
        lines.append(f"{SAY}: WHAT FOLLOWS: {follows['name']}. {follows['reason'][:1].upper()}{follows['reason'][1:]}")
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(f'`{name}`' for name in not_to_be_read)}. The step fails; "
                     "the task queued again measures such a set again from its adapter")

    headline = (f"L6b (a second round, from the best model; {LABEL}) seed {seed}, from L5f's averaged `{START}`, against L5c's stored `{BAR}`: "
                + (f"THIS RUN IS {INCONCLUSIVE} ({'; '.join(failed)})" if inconclusive else
                   f"THE SEARCH: {found['new_solves']:,} new solves of {found['problems']:,} problems ({found[PART_A]['solved']:,} of {found[PART_A]['problems']:,} left "
                   f"unsolved by round 1's search, {found[PART_B]['solved']:,} of {found[PART_B]['problems']:,} no search took)"
                   + ("; per 1,000 attempts on all of G " + ", ".join(f"`{name}` {_per_1000(figures[name][ALL]['per_attempt'])}" for name in models)
                      + f" (the bar {_per_1000(of_the_bar['per_attempt'])}, B {_per_1000(stored_figures[THE_BEST][ALL]['per_attempt'])}); THE PRIMARY: {primary['name']} "
                      f"({_signed(difference(primary))}); WHOSE PROOFS: {whose['name']} ({_signed(difference(whose))})" if ran else "; NO MODEL WAS TRAINED"))
                + f". What follows: {follows['name']}"
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in the sets of {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    kept = lambda what: {"measured_and_not_read": what} if inconclusive else what      # noqa: E731
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L6b", "headline": headline, "lines": lines, "ok": not not_to_be_read, "seed": seed,
        "stand_in_engine": bool(prepare.get("stand_in_engine") or of_the_search["summary"].get("stand_in_engine")
                                or any(entry.get("stand_in_engine") for entry in (*trains.values(), *averaged.values(), *measured.values()))),
        "start": start, "first_start": first, "second_start": second, "rank": rank, "models": models, "recipe": recipe, "trainings": prepare["trainings"],
        "branches": {"primary": primary["name"], "whose_proofs": whose["name"], "what_follows": follows["name"]},
        "inconclusive": inconclusive, "failed_checks": failed,
        "can_this_run_be_read": {"what": "the checks the spec reads first; when any fails the run is INCONCLUSIVE and nothing else is said of it", **checks},
        "of_this_seed": of_this_seed,
        "the_search": {"prepared": search, "sampled": {key: value for key, value in of_the_search["summary"].items() if key != "pipeline"},
                       "enough_new_solves": enough, **kept({"found": found})},
        "figures": {"what": f"each model's successes per attempt over the {attempts} attempts of the first sampling, with its binomial sampling noise, on all of G, on the goal "
                            f"problems `{first}` solves in its own first sampling and on G'", "problems": {part: len(own) for part, own in parts.items()},
                    "the_stored_models": {name: own[ALL] for name, own in stored_figures.items()}, **kept(figures)},
        "the_bar": {"what": f"L5c's stored `{BAR}` of this seed's run, on the goal set's first sampling: what the primary is against, paired by problem",
                    "run": of_the_bars_run["run"], "file": sampling["stored_bar"], **kept({"figures": stored_figures[THE_BAR]})},
        "the_primary": primary if inconclusive else {"what": f"`{NEW_OWN}` minus L5c's stored `{BAR}`, per attempt on all of G: a scout's read at one seed", **primary},
        "whose_proofs": whose if inconclusive else {"what": f"`{NEW_OWN}` minus `{OLD_OWN}`, per attempt on all of G: a scout's read at one seed", **whose},
        "beside": kept({"what": f"naming nothing: `{OLD_OWN}` minus the bar; each model and the bar against B's, `{second}`'s and `{first}`'s stored first samplings, paired by "
                                "problem; the goal problems each solves, gained and lost against B; solved reliably (in half its attempts or more); the long goal problems; "
                                "the share of distinct attempts on G", "old_own_minus_the_bar": old_minus_bar, "against_the_stored_models": against, "counts": counts,
                        "long_problems": len(long_ids), "solved_reliably": reliable, "share_of_distinct_attempts": shares}),
        "timed_out_in_lean": of_timeouts, "what_follows": follows,
        "the_seeds": prepare["the_seeds"], "the_start_model": prepare["the_start_model"], "the_best": of_the_best, "l5cs_run": of_the_bars_run, "l5ds_run": prepare["l5ds_run"],
        "round_one": prepare["round_one"], "the_dose": prepare.get("the_dose"), "the_sets": prepare.get("the_sets"), "the_orders": prepare.get("the_orders"),
        "the_sampling": sampling, "training_sets": prepare["training_sets"], "training_recipe": prepare["recipe"],
        "models_made": {name: {"model": f"`{name}`: trained by nothing: the mean of {', '.join(f'`{of}`' for of in averaged[name].get('of') or [])}",
                               "trainings": {of: {key: value for key, value in trains[of].items() if key != "checkpoints"} for of in prepare["the_averaged"][name]["of"]},
                               "averaging": averaged[name]} for name in models},
        "smoke": smoke, "what_it_cannot_say": list(CANNOT_SAY),
        "attempts": {**{name: _health(rows[name]) for name in models}, "the_search": health}, "not_to_be_read": not_to_be_read,
        "not_built": [f"the further seeds: two more of `{NEW_OWN}` and of the bar need L5f's `{START}` and L5c's `{BAR}` at those seeds, which no run holds; and the read over "
                      "three seeds", "a further seed of the search (one search, by the seed L6's setting names)",
                      "the check again in Lean of the timed-out attempts where two shares differ by more than a point: the pairs are named",
                      "the measurement in full (the rungs, the second sampling, the held-aside set) of the highest model"],
        "not_measured": ["the three rungs, the goal set's second sampling and the held-aside set: this stage reads one sampling of the goal set",
                         f"B, L5c's `{BAR}`, `{second}` and `{first}`: their rows are the stored ones of their own runs", "anything at another temperature than the config's"]}
