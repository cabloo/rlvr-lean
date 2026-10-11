"""L6's report: a second round. Does the cycle compound? Spec: docs/spec/ladder-loop.spec.md, "L6: a second round.
Does the cycle compound?" ("The reads, fixed before the run", "Checks, read first", "What follows"). Pure: rows in, a
report out. Its pieces are L4t's report's own (`reporting/ladder_l4_rows.py`: the world a model is read on, the checks of
a trained model), L5's (`reporting/ladder_l5.py`) and the rules of `domain/ladder_round/l6.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. One seed.

  the checks        FIRST, each PASS or FAIL: search 2 found enough round-2 solves; for each trained model (`published2`,
                    `one_round`, `two_rounds`, `long_ordinary`) the adapter it saved is not the start adapter's, it still
                    writes proofs, Lean answered, and no row it was trained on is barred (no held-out problem, no
                    held-aside problem, its published rows of the `pretrain` half, its own rows base-map problems:
                    `l6.no_barred_row`); and Lean answered on each held-aside set. ANY FAIL: INCONCLUSIVE, and nothing else is said (what was measured is kept under
                    `measured_and_not_read`). A RISE OF A TRAINING'S LOSS IS PRINTED AND DECIDES NOTHING (the rule of
                    "L5, further seeds": that half of L4t's check fails one training in forty by chance)
  the primary       ON THE HELD-ASIDE SET, successes per attempt, `two_rounds` minus `published2`, paired by problem, 95%
                    bootstrap over problems (`l6.primary_branch`)
  compounding       `two_rounds` minus `one_round` there (`l6.compounding`)
  hard to find, or long   `two_rounds` minus `long_ordinary` there (`l6.hard_or_long`)
  reach             the goal problems of 8 lines or more solved at least once in all the attempts, `two_rounds` against
                    `published2`: gained, lost, the sign test (`l6.reach`)
  what follows      the scout's table, in one line (`l6.what_follows`), with the spread L5's three seeds measured when the
                    read over them is GIVEN in the run directory
  beside            naming nothing: every pair on all of G and its split (the goal problems not in G', and G'); each
                    model's goal problems solved and solved reliably, and the never-solved when their ids are given;
                    `two_rounds` against M(1)'s and `second_full`'s stored rows on G, and against `second_full` on the
                    held-aside set when it was measured there; each model's overfitting reads against the start model
                    (`l4_dose.overfitting`); each set's share of attempts that timed out in Lean (`l5.timed_out`)
  the search        search 2's table: the round-2 solves in (a) and in (b), by the attempt of the first success; the lines
                    of their proofs beside round 1's

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples such a set of this run's own again from the adapter. THE LENGTH OF A HELD-OUT PROBLEM'S PUBLISHED PROOF is read
here and nowhere before.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE
from rlvr_lean.domain.ladder_round.l4_dose import SECOND_FULL, overfitting
from rlvr_lean.domain.ladder_round.l4_rank import never_solved
from rlvr_lean.domain.ladder_round.l5 import DEEP, timed_out
from rlvr_lean.domain.ladder_round.l6 import (
    LONG_ORDINARY,
    MODELS,
    ONE_ROUND,
    PART_A,
    PART_B,
    PUBLISHED2,
    TWO_ROUNDS,
    compounding,
    hard_or_long,
    no_barred_row,
    primary_branch,
    reach,
    spread_of_l5,
    what_follows,
)
from rlvr_lean.domain.ladder_round.read import GOAL, paired_change
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import ALL, RUNG_PART, _rate, attempts_on_g, goal_sets, per_attempt, solved
from rlvr_lean.reporting.ladder_l2 import MAXIMUM_SHARE_WITHOUT_AN_ANSWER, _split
from rlvr_lean.reporting.ladder_l3c import LENGTH_GROUPS
from rlvr_lean.reporting.ladder_l4 import LABEL, NOT_MEASURED, SAY, _verdict
from rlvr_lean.reporting.ladder_l4_rows import check_lines, on_a_set, the_four_checks, world_of
from rlvr_lean.reporting.ladder_l5 import _stats, held_aside_key, pair_name, sets_by_kind

L4 = "l4"
QUESTION = "a second round: does the cycle compound?"
SPEC = "docs/spec/ladder-loop.spec.md, L6: a second round. Does the cycle compound?"
M1, STORED = DEEP, SECOND_FULL          # the two stored models `two_rounds` is read against, beside: round 1's `deep` (M(1)), and L4p's whole second pass
LONG = LENGTH_GROUPS[-1]                # the goal problems whose shortest published proof has 8 lines or more: the reach's
IS = {PUBLISHED2: "the first 2N rows of the hash order L4t's rehearsal draw uses among the rows the start model was pretrained on, and nothing else: the bar at equal steps",
      ONE_ROUND: "round 1's deep solves (one verified attempt each) and as many of the first rows of that order as make 2N",
      TWO_ROUNDS: "the N deep solves of both rounds (one verified attempt each) and the first N rows of that order: M(2)",
      LONG_ORDINARY: "N ordinary solves (map problems the start model solved in a map attempt, one verified map attempt each) drawn to have the line counts of `two_rounds`' "
                     "own rows bin by bin, and the SAME first N rows of that order"}
# The three reads on the held-aside set, each `after` minus `before`, by the key the report stores it under; then the three other pairs, which are read beside on G alone.
READS = (("primary", TWO_ROUNDS, PUBLISHED2), ("compounding", TWO_ROUNDS, ONE_ROUND), ("hard_to_find_or_long", TWO_ROUNDS, LONG_ORDINARY))
PAIRS = (*((after, before) for _, after, before in READS), (ONE_ROUND, PUBLISHED2), (LONG_ORDINARY, PUBLISHED2), (LONG_ORDINARY, ONE_ROUND))
THE_SEARCH = "search_2_found_enough"                                    # the first check, as a failed check names it
THE_HELD_ASIDE = "lean_answered_on_the_held_aside_sets"                 # the last check, as a failed check names it
THE_ADAPTER = "its_adapter_is_not_the_start_adapters"                   # a model's first check: the adapter half of L4t's "its training ran", and that half alone
CANNOT_SAY = [
    "One seed: each model is one training",
    "One M(1) made search 2: another training of round 1's rows would have found other round-2 solves",
    "Parts (a) and (b) of search 2 are trained on together",
    "`published2` is a second pass over rows the start model has seen; the published proofs of the `loop` half stay unused by the loop's rule, so \"pretraining on new "
    "published data\" is not the bar here",
    "The held-aside set resolves about 0.007 per attempt at one seed",
    "A trained model against the start model on the held-aside set is favoured by the selection (the start model's 8 stored map attempts chose the set): only the "
    "differences among the trained models are fair there"]


def _table(found: Mapping) -> str:
    """Search 2's table in one line."""
    def of_a_part(part: str, said: str) -> str:
        own = found[part]
        by_band = "; ".join(f"{key.replace('_to_', ' to ')}: {count}" for key, count in own["solved_by_first_success"].items())
        return f"({part}) {said}: {own['solved']:,} of {own['problems']:,} solved (a share of {own['share_solved']}); by the attempt of the first success: {by_band}"

    return (f"{found['problems']:,} problems attempted ({found['episodes_each']} attempts each), {found['solved']:,} solved (a share of {found['share_solved']}). "
            + of_a_part(PART_A, "the problems search 1 left unsolved") + ". " + of_a_part(PART_B, "the candidates neither draw took")
            + f". {found['attempts']:,} attempts, {found['successes']:,} of {found['episodes']:,} episodes resolved their problem ({found['per_1000']} per 1,000); "
            f"{found['without_an_answer']} attempts without an answer from Lean")


def _loss(name: str, loss: Mapping) -> str:
    """One training's loss at both ends, as the line that PRINTS it says it (it decides nothing)."""
    if loss.get("rise") is None:
        return f"`{name}`: too few rows to compare"
    return (f"`{name}`: {loss['last']} over the last {loss['rows_compared']} rows against {loss['first']} over the first, a rise of {loss['rise']} where {loss['allowed']} would "
            "be allowed" + ("" if loss["passes"] else ": IT ROSE BY MORE"))


def build_l6_report(prepare: Mapping, trains: Mapping[str, Mapping], losses: Mapping[str, Sequence[float]], trained: Mapping[str, Sequence[Mapping]],
                    groups: Sequence[Mapping], lengths: Mapping[str, Mapping], stored: Mapping[str, Mapping], models: Mapping[str, Mapping],
                    again: Sequence[str], settings: Mapping, evaluation: Mapping, base_map: Sequence[str], held_aside: Mapping[str, Sequence[Mapping]],
                    beside: Mapping[str, Sequence[Sequence[Mapping]]], never: Sequence[str] | None = None, spread: Mapping | None = None) -> dict:
    """`prepare`: the summary of the step that built the four sets (`start`; `enough_round_two_solves`;
    `the_search_found`; `round_one`; `the_held_aside`; `training_sets`, empty when search 2 found too few round-2 solves
    and nothing was trained). `trains`, `losses`, `trained`: by trained model, its training step's summary, its rows'
    losses in the order trained, and the rows its training loop recorded. `groups`: the held-out groups. `lengths`: by
    problem, its shortest published proof's length group. `stored`: the start model, from the rows its own run stored.
    `models`: the four trained models, each `rungs`, `goal` (one list of per-problem rows for each sampling of G) and
    what it wrote; NONE when nothing was trained. `again`: the start model's G'. `settings`: `ladder_loop.l4` (the
    checks' own). `base_map`: the ids of the base map's problems. `held_aside`: by model, its per-problem rows on the
    held-aside set: the START MODEL (round 1's stored rows), the four trained ones and, when it was measured there, the
    stored model `second_full` (none when nothing was trained). `beside`: by STORED model that `two_rounds` is read
    against (round 1's `deep`, M(1); `second_full` when its run was on the box), its rows on G, a list a sampling.
    `never`: the ids of the goal problems nothing stored had solved, as GIVEN in the run directory, or None. `spread`:
    the read over L5's seeds as GIVEN in the run directory (`l6.spread_of_l5` reads the spread from it), or None."""
    start, enough, found, of_round_one = prepare["start"], prepare["enough_round_two_solves"], prepare["the_search_found"], prepare["round_one"]
    rank = (prepare.get("start_recipe") or {}).get("rank")
    measured = tuple(name for name in MODELS if name in models)
    if measured and (measured != MODELS or not enough["passes"]):
        raise ValueError(f"the report is given the trained models {list(measured)}: it reads all four ({list(MODELS)}), or none when search 2 found too few round-2 solves "
                         "and nothing was trained")
    on_the_set = (start, *measured, *((STORED,) if STORED in held_aside else ())) if measured else ()        # the models measured on the held-aside set
    ids = {name: sorted(row["problem_id"] for row in held_aside.get(name, ())) for name in on_the_set}
    if sorted(held_aside) != sorted(on_the_set) or any(own != ids[start] or len(set(own)) != len(own) or not own for own in ids.values()):
        raise ValueError(f"the report is given held-aside rows of {sorted(held_aside)}: the start model `{start}`, the four trained models and at most `{STORED}` are measured "
                         "there, each on the same problems, or no model when nothing was trained")
    aside_ids = ids[start] if measured else []
    who = {start: stored[start], **{name: models[name] for name in measured}}
    world = world_of(prepare, who, groups, lengths, again, settings, evaluation, never, also={held_aside_key(name): held_aside[name] for name in on_the_set})
    goal_ids, on_g, attempts, resamples, seed = world.goal_ids, world.on_g, world.attempts, world.resamples, world.seed
    of_the_set, of_the_search = prepare["the_held_aside"], prepare["the_search"]
    in_again = set(again)
    not_again, long_ids = [key for key in goal_ids if key not in in_again], world.sets.get(LONG, [])

    # ---- the checks, FIRST: search 2's; each trained model's four (the adapter half of L4t's first, L4t's second and third, L5's own fourth); Lean's answers on the held-aside sets
    checks_of, loss_of, check_lines_of = {}, {}, {}
    for name in measured:
        own_sets = [*world.sets_of[name], *world.sets_of[start]]
        of_l4t = the_four_checks(world, name, trains[name], losses[name], trained[name], own_sets, f"the stored ones of `{start}`")
        ran, barred = of_l4t["its_training_ran"], no_barred_row(trained[name], world.heldout, settings["half_seed"], base_map, aside_ids)
        differs, loss_of[name] = ran["the_adapter_differs_from_the_start"], ran["the_loss_did_not_rise"]        # THE LOSS IS PRINTED AND DECIDES NOTHING: it is no check here
        checks_of[name] = {THE_ADAPTER: differs, "it_still_writes_proofs": of_l4t["it_still_writes_proofs"], "lean_answered": of_l4t["lean_answered"], "no_barred_row": barred}
        of = f"{SAY}: `{name}`"
        check_lines_of[name] = [
            f"{of} ({IS[name]}; {trains[name]['rows']:,} rows, {trains[name]['steps']:,} optimizer steps, from `{start}`), CHECK 1, the adapter it saved is not the start "
            "adapter's: " + (f"{differs['elements_changed']:,} of the {differs['elements']:,} numbers of its adapter are not the start adapter's" if differs["compared"]
                             else "its adapter was not compared with the start adapter's") + f": {_verdict(differs)}",
            *check_lines(world, of, name, IS[name], trains[name], of_l4t, own_sets)[1:3],
            f"{of}, CHECK 4, no barred row (no held-out problem; no held-aside problem; its published rows of the `pretrain` half; the rows of the model's own are base-map "
            f"problems): {barred['held_out_problems']} held-out problems, {barred['held_aside_problems']} held-aside problems, "
            f"{barred['published_rows_not_of_the_pretrain_half']} of its {barred['published_rows_of_the_pretrain_half']:,} "
            f"published rows not of the `pretrain` half, {barred['own_rows_not_of_the_base_map']} of its {barred['rows_of_the_models_own']:,} own rows not base-map problems: "
            f"{_verdict(barred)}"]
    silent = [held_aside_key(name) for name in on_the_set if held_aside_key(name) in world.not_to_be_read]
    answered = {"what": f"Lean gave a verdict on all but at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of the attempts of each held-aside set: the start model's (round 1's), the "
                        "four trained models' and the stored model's when it was measured there; a set of this run's own is sampled again by the task queued again",
                "maximum": MAXIMUM_SHARE_WITHOUT_AN_ANSWER, "share_without_an_answer": {held_aside_key(name): world.health[held_aside_key(name)]["share_without_an_answer"] for name in on_the_set},
                "sets_not_answered": silent, "passes": bool(on_the_set) and not silent}
    failed = [*([] if enough["passes"] else [THE_SEARCH]), *(f"`{name}`: {check}" for name in measured for check, entry in checks_of[name].items() if not entry["passes"]),
              *([THE_HELD_ASIDE] if measured and not answered["passes"] else [])]
    inconclusive = bool(failed)
    rose = [name for name in measured if loss_of[name].get("rise") is not None and not loss_of[name]["passes"]]

    # ---- the reads ON THE HELD-ASIDE SET, each `after` minus `before`; reach on the long goal problems; what follows
    on_ha = lambda after, before: per_attempt(held_aside[after], held_aside[before], aside_ids, resamples, seed)      # noqa: E731
    figures = {key: ({"difference": on_ha(after, before), "solved": solved(held_aside[after], held_aside[before], aside_ids)} if measured else {"difference": dict(NOT_MEASURED)})
               for key, after, before in READS}
    long_solved = solved(on_g[TWO_ROUNDS], on_g[PUBLISHED2], long_ids) if measured else {}
    primary = primary_branch(failed, figures["primary"]["difference"])
    of_compounding = compounding(failed, figures["compounding"]["difference"])
    of_length = hard_or_long(failed, figures["hard_to_find_or_long"]["difference"])
    of_reach = reach(failed, long_solved)
    of_the_spread = spread_of_l5(spread)
    follows = what_follows(primary, of_compounding, of_reach, figures["primary"]["difference"], of_the_spread)
    branch_of = {"primary": primary, "compounding": of_compounding, "hard_to_find_or_long": of_length}

    # ---- beside, naming nothing: every pair on G and its split; each model's own counts; `two_rounds` against the stored models; overfitting; timeouts
    def on_goal(after: Sequence[Mapping], before: Sequence[Mapping]) -> dict:
        return {"on_all_of_g": per_attempt(after, before, goal_ids, resamples, seed), "on_the_goal_problems_not_in_the_goal_set_again": per_attempt(after, before, not_again, resamples, seed),
                "on_the_goal_set_again": per_attempt(after, before, list(again), resamples, seed), "goal_problems_solved": solved(after, before, goal_ids)}

    pairs = {pair_name(after, before): on_goal(on_g[after], on_g[before]) for after, before in PAIRS} if measured else {}
    everyone = (start, *measured) if measured else ()
    each = {name: {"held_aside": on_a_set(held_aside[name], aside_ids), "all_of_g": on_a_set(on_g[name], goal_ids), "solved_reliably": world.alone[name][ALL]["reliably"],
                   "long_goal_problems": on_a_set(on_g[name], long_ids), "never_solved_before": never_solved(never, on_g[name], on_g[start], goal_ids)} for name in everyone}
    against_stored = {}
    for name in (beside if measured else ()):
        of_the_stored = attempts_on_g(beside[name])
        against_stored[name] = {"on_g": on_goal(on_g[TWO_ROUNDS], of_the_stored), "all_of_g": on_a_set(of_the_stored, goal_ids), "long_goal_problems": on_a_set(of_the_stored, long_ids)}
    if measured and STORED in held_aside:
        against_stored.setdefault(STORED, {})["on_the_held_aside_set"] = {"difference": on_ha(TWO_ROUNDS, STORED), "solved": solved(held_aside[TWO_ROUNDS], held_aside[STORED], aside_ids),
                                                                          "held_aside": on_a_set(held_aside[STORED], aside_ids)}
    overfit = {name: overfitting(solved(on_g[name], on_g[start], goal_ids),
                                 {rung: paired_change(who[name][RUNG_PART], who[start][RUNG_PART], world.rung_ids[rung], resamples, seed) for rung in RUNGS},
                                 who[name]["distinct_attempts"][GOAL]["mean_share_distinct"], who[start]["distinct_attempts"][GOAL]["mean_share_distinct"], start) for name in measured}
    kinds = sets_by_kind(start, measured, len(prepare["goal_samplings"]))
    if STORED in on_the_set:
        kinds["held_aside"][STORED] = held_aside_key(STORED)
    timeouts = timed_out(world.health, kinds) if measured else None

    count, lines_of, bins = prepare["deep_solves"], prepare.get("lines_of_the_rows") or {}, (prepare.get("matched_in_length") or {})
    not_the_rules = {key: value for key, value in (prepare.get("rows_not_taken_by_the_rule") or {}).items() if key != "what" and value}
    smoke, each_on_the_set = prepare.get("smoke"), of_the_set["episodes_each"]
    fresh = world.fresh

    def read_said(key: str, after: str, before: str) -> str:
        """One read ON THE HELD-ASIDE SET, as a line gives it."""
        own = figures[key]
        return (f"THE HELD-ASIDE SET ({len(aside_ids):,} problems `{start}` failed in every map attempt, that no search took and no model trains on), successes per attempt "
                f"over {each_on_the_set} fresh attempts a problem, `{after}` minus `{before}`, paired by problem, 95% bootstrap over problems: {_rate(own['difference'])}; "
                f"held-aside problems solved at least once, `{after}` against `{before}`: {own['solved']['resolved_after']} to {own['solved']['resolved_before']}, "
                f"{_split(own['solved'])}")

    def own_said(entry: Mapping) -> str:
        return f"{entry['solved_at_least_once']} solved, {entry['per_1000']} per 1,000"

    lines = [
        f"{SAY}: L6, {QUESTION.upper()} Seed {prepare['seed']}, one seed. ROUND 1 IS L5's, only read ({of_round_one['run']}): its deep search solved {of_round_one['solved']:,} "
        f"of {of_round_one['attempted']:,} problems, and M(1) is its model `{of_round_one['model']}`. SEARCH 2: M(1), served at rank {rank}, made {of_the_search['episodes_each']} "
        f"attempts at {of_the_search['problems']:,} problems, sampling seed {of_the_search['sampling_seed']}: (a) the {of_the_search['by_part'][PART_A]:,} search 1 left unsolved, "
        f"(b) the {of_the_search['by_part'][PART_B]:,} of the {of_the_search['partition']['candidates']:,} candidates neither draw took; a problem solved there is a ROUND-2 solve"
        + (f". FOUR MODELS, each trained FROM `{start}` AT RANK {rank}, one pass, the arm's recipe, on {2 * count:,} rows (N = {count:,}: every deep solve of both rounds; the "
           f"same number of rows and of optimizer steps), in the content-hash order, and measured as L5's models were: "
           + ", ".join(f"`{name}` ({IS[name]})" for name in MODELS) + f". THE HELD-ASIDE SET is round 1's ({of_the_set['problems']:,} problems, {each_on_the_set} fresh attempts "
           f"each, ONE sampling seed, {of_the_set['sampling_seed']}). THE PRIMARY is ON THE HELD-ASIDE SET, `{TWO_ROUNDS}` minus `{PUBLISHED2}`; COMPOUNDING is `{TWO_ROUNDS}` minus "
           f"`{ONE_ROUND}` and HARD TO FIND, OR LONG `{TWO_ROUNDS}` minus `{LONG_ORDINARY}`, there too; REACH is on the goal problems of 8 lines or more"
           if measured else ". NO MODEL WAS TRAINED: search 2 found too few round-2 solves")]
    if smoke or not_the_rules:
        lines.append(f"{SAY}: A SMOKE RUN: round 1's run " + (f"reads INCONCLUSIVE ({', '.join(smoke['round_ones_failed_checks'])}) and was read all the same; "
                                                              if smoke and smoke.get("round_one_read_though_inconclusive") else "was read; ")
                     + f"{of_the_search['taken_by_the_runs_minimum']} of search 2's problems and "
                     + (", ".join(f"{value} of `{key}`'s own rows" for key, value in not_the_rules.items()) or "no row")
                     + " are of problems the rule did not give, so that every step has rows to run on. Nothing here is a read of the rule")
    if measured and not fresh:
        lines.append(f"{SAY}: the models have ONE sampling of G here (a smoke run): G was attempted {attempts} times a problem")
    lines.append(f"{SAY}: CHECK, search 2 found at least {enough['minimum']} round-2 solves: {enough['round_two_solves']:,}: {_verdict(enough)}")
    for name in measured:
        lines += check_lines_of[name]
    if measured:
        lines.append(f"{SAY}: CHECK, Lean answered on each held-aside set (at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} without an answer): "
                     + "; ".join(f"{key} {share}" for key, share in answered["share_without_an_answer"].items()) + f": {_verdict(answered)}")
        lines.append(f"{SAY}: PRINTED, DECIDING NOTHING: each training's loss (the mean over the last tenth of its rows against the first; "
                     f"{settings['standard_errors_allowed']:g} standard errors of their difference would be allowed): " + "; ".join(_loss(name, loss_of[name]) for name in measured)
                     + (f". THE LOSS ROSE BY MORE THAN THAT IN {', '.join(f'`{name}`' for name in rose)}: named, and the run is read all the same" if rose else ""))
    if not measured or STORED not in on_the_set:
        lines.append(f"{SAY}: `{STORED}` (L4p's whole second pass) on the held-aside set: NOT THERE (" + ((prepare["the_stored_model"].get("why") or "its adapter is not on the box, "
                     "or its step has not run") if measured else "nothing was trained, and nothing was measured") + "); nothing is refused for it")
    if inconclusive:
        lines.append(f"{SAY}: {INCONCLUSIVE}. {primary['reason']}" + ("" if enough["passes"] else f". {prepare['trainings']['why']}"))
    else:
        lines += [f"{SAY}: PRIMARY, the bet at equal steps. {read_said('primary', TWO_ROUNDS, PUBLISHED2)}",
                  f"{SAY}: PRIMARY: {primary['name']}. {primary['reason']}",
                  f"{SAY}: COMPOUNDING (round 2's solves against the same number of published rows, round 1's solves in both). {read_said('compounding', TWO_ROUNDS, ONE_ROUND)}",
                  f"{SAY}: COMPOUNDING: {of_compounding['name']}. {of_compounding['reason']}",
                  f"{SAY}: HARD TO FIND, OR LONG (ordinary solves matched in length in the place of the deep ones). {read_said('hard_to_find_or_long', TWO_ROUNDS, LONG_ORDINARY)}",
                  f"{SAY}: HARD TO FIND, OR LONG: {of_length['name']}. {of_length['reason']}",
                  f"{SAY}: REACH, on the goal set ({len(long_ids)} goal problems whose shortest published proof has 8 lines or more, solved at least once in the {attempts} "
                  f"attempts): {of_reach['name']}. {of_reach['reason']}",
                  f"{SAY}: WHAT FOLLOWS: {follows['name']}. {follows['reason']}",
                  f"{SAY}: BESIDE, naming nothing: on the held-aside set ({len(aside_ids):,} problems), problems solved and successes per 1,000 attempts: "
                  + "; ".join(f"`{name}` {own_said(each[name]['held_aside'])}" for name in everyone)
                  + f" (`{start}`'s rows are round 1's; its 8 stored map attempts chose the set, which favours every trained model against it)",
                  f"{SAY}: BESIDE, naming nothing: on all of G ({len(goal_ids)} problems, {attempts} attempts each), goal problems solved / solved reliably (in at least half of "
                  "their episodes of 8) and successes per 1,000 attempts: "
                  + "; ".join(f"`{name}` {each[name]['all_of_g']['solved_at_least_once']} / {each[name]['solved_reliably']}, {each[name]['all_of_g']['per_1000']}" for name in everyone)
                  + f"; long goal problems ({len(long_ids)}) solved: " + ", ".join(f"`{name}` {each[name]['long_goal_problems']['solved_at_least_once']}" for name in everyone)
                  + "; the goal problems nothing stored had solved: "
                  + (f"of the {each[start]['never_solved_before']['goal_problems']} listed, " + ", ".join(f"`{name}` {each[name]['never_solved_before']['solved']}" for name in measured)
                     if each[start]["never_solved_before"]["given"] else "not given (no file of their ids is in the run directory)")]
        for after, before in PAIRS:
            pair = pairs[pair_name(after, before)]
            lines.append(f"{SAY}: BESIDE, naming nothing: `{after}` minus `{before}`, successes per attempt over the {attempts} attempts: all of G ({len(goal_ids)}): "
                         f"{_rate(pair['on_all_of_g'])}; the {len(not_again)} goal problems NOT in G' (`{start}` solves them in its first sampling): "
                         f"{_rate(pair['on_the_goal_problems_not_in_the_goal_set_again'])}; G' ({len(again)}): {_rate(pair['on_the_goal_set_again'])}; goal problems solved "
                         f"{pair['goal_problems_solved']['resolved_after']} to {pair['goal_problems_solved']['resolved_before']}, {_split(pair['goal_problems_solved'])}")
        for name in (M1, STORED):
            own = against_stored.get(name)
            what = "M(1), the model round 1 trained, which made search 2" if name == M1 else "L4p's whole second pass, the best pretraining-only model stored"
            if not own or "on_g" not in own:
                lines.append(f"{SAY}: BESIDE, naming nothing: `{TWO_ROUNDS}` against the stored `{name}` ({what}) on G: NOT THERE (its stored rows are not on the box)")
                continue
            lines.append(f"{SAY}: BESIDE, naming nothing: `{TWO_ROUNDS}` against the stored `{name}` ({what}) on all of G: {_rate(own['on_g']['on_all_of_g'])}; goal problems "
                         f"solved {own['on_g']['goal_problems_solved']['resolved_after']} to {own['on_g']['goal_problems_solved']['resolved_before']}, "
                         f"{_split(own['on_g']['goal_problems_solved'])}; long goal problems solved by `{name}`: {own['long_goal_problems']['solved_at_least_once']}")
        on_the_set_too = (against_stored.get(STORED) or {}).get("on_the_held_aside_set")
        lines.append(f"{SAY}: BESIDE, naming nothing: `{TWO_ROUNDS}` against `{STORED}` ON THE HELD-ASIDE SET: "
                     + (f"{_rate(on_the_set_too['difference'])}; problems solved {on_the_set_too['solved']['resolved_after']} to {on_the_set_too['solved']['resolved_before']}, "
                        f"{_split(on_the_set_too['solved'])}" if on_the_set_too else "NOT THERE (it was not measured there)"))
        for name in measured:
            lines.append(f"{SAY}: BESIDE, naming nothing: OVERFITTING, `{name}` against `{start}`: {overfit[name]['said']}")
        lines.append(f"{SAY}: BESIDE, naming nothing: TIMEOUTS, the share of each measured set's attempts that timed out in Lean: "
                     + "; ".join(f"{key} {share}" for key, share in timeouts["share"].items()) + ". WHERE TWO MODELS' SHARES DIFFER BY MORE THAN A POINT: "
                     + ("; ".join(f"{entry['kind']}: `{entry['models'][0]}` {entry['shares'][0]} against `{entry['models'][1]}` {entry['shares'][1]}" for entry in timeouts["apart"])
                        if timeouts["apart"] else "nowhere"))
    lines.append(f"{SAY}: SEARCH 2" + ("" if enough["passes"] else ", WHICH IS THE FINDING") + f": {_table(found)}")
    of_the_rows = {key: value for key, value in lines_of.items() if key != "what"} or {f"{key}_rows": value for key, value in found["lines_of_the_deep_rows"].items()}
    lines.append(f"{SAY}: THE LINES OF THE ROWS (" + ("the proofs trained on" if measured else "the deep solves of both rounds; nothing was trained") + "): "
                 + "; ".join(f"{key.replace('_', ' ')}: {_stats(value)}" for key, value in of_the_rows.items()))
    if measured:
        lines.append(f"{SAY}: THE ORDINARY SOLVES MATCHED IN LENGTH (`{LONG_ORDINARY}`'s own rows against `{TWO_ROUNDS}`', by the lines of a row's proof): "
                     + "; ".join(f"{entry['lines']} lines: {entry['wanted']} wanted, the map offers {entry['offered']}, {entry['taken']} taken"
                                 + (f", {entry['shortfall']} SHORT and filled from the next shorter bin" if entry["shortfall"] else "") for entry in bins["bins"])
                     + (f". SHORTFALL: {bins['rows_short']} rows, in {', '.join(bins['bins_short'])}" if bins["bins_short"] else ". Every bin was filled"))
    if world.not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(world.not_to_be_read)}. The step fails; the task queued again "
                     "measures such a set of this run's own again from the adapter")

    of_the_primary = figures["primary"]["difference"]
    headline = (f"L6 ({QUESTION.rstrip('?')}; {LABEL}) seed {prepare['seed']}, from `{start}`: " + (INCONCLUSIVE if inconclusive else primary["name"])
                + f". Search 2 solved {found['solved']:,} of {found['problems']:,} problems" + (f"; N = {count:,}" if measured else "")
                + ("" if inconclusive or of_the_primary.get("mean") is None
                   else f". Primary (the held-aside set, {len(aside_ids):,} problems, `{TWO_ROUNDS}` minus `{PUBLISHED2}`): {_rate(of_the_primary)}")
                + ("" if inconclusive else f". Compounding (`{TWO_ROUNDS}` minus `{ONE_ROUND}`): {of_compounding['name']}. Hard to find, or long (`{TWO_ROUNDS}` minus "
                   f"`{LONG_ORDINARY}`): {of_length['name']}. Reach: {of_reach['name']}. What follows: {follows['name']}")
                + f". {primary['reason']}"
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(world.not_to_be_read)}" if world.not_to_be_read else ""))

    def read_or_kept(key: str) -> dict:
        """A read as the report stores it: its figures on the held-aside set; or, INCONCLUSIVE, what was measured kept apart."""
        branch = branch_of[key]
        if not measured:
            return {**branch, "difference": {"mean": None}}
        return {**branch, "difference": {"mean": None}, "measured_and_not_read": figures[key]} if inconclusive else {**branch, **figures[key]}

    kept = lambda what: {"measured_and_not_read": what} if inconclusive else what      # noqa: E731
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L6", "headline": headline, "lines": lines, "ok": not world.not_to_be_read, "seed": prepare["seed"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in (*trains.values(), *models.values()))),
        "start": start, "rank": rank, "measured": list(measured), "deep_solves": count, "rows_a_set": prepare["rows_a_set"],
        "branches": {"the_primary": primary["name"], "compounding": of_compounding["name"], "hard_to_find_or_long": of_length["name"], "reach": of_reach["name"],
                     "what_follows": follows["name"]},
        "inconclusive": inconclusive, "failed_checks": failed,
        "can_this_run_be_read": {"what": "the checks the spec reads first; when any fails the run is INCONCLUSIVE and nothing else is said. A rise of a training's loss is no "
                                         "check here: it is printed, and decides nothing",
                                 "search_2": enough, **{name: checks_of[name] for name in measured}, **({"the_held_aside_sets": answered} if measured else {})},
        "the_loss_of_each_training": {"what": "PRINTED, DECIDING NOTHING (spec, L5, further seeds: the loss half of L4t's check fails one training in forty by chance, and is wrong "
                                              "where the start model has already been trained on rows of the kind): the mean loss over the last tenth of a training's rows "
                                              "against the first tenth's, and whether it rose by more than the standard errors allowed", **loss_of, "rose_by_more": rose},
        "primary": {"what": f"ON THE HELD-ASIDE SET (problems `{start}` failed in every map attempt, that no search took and no model trains on): successes per attempt, "
                            f"`{TWO_ROUNDS}` minus `{PUBLISHED2}`, paired by problem, a 95% bootstrap interval over problems. In its figures `of_the_base` is the side compared "
                            f"with: `{PUBLISHED2}`", **read_or_kept("primary")},
        "compounding": {"what": f"on the held-aside set, `{TWO_ROUNDS}` minus `{ONE_ROUND}`: round 2's solves against the same number of published rows, round 1's solves in both",
                        **read_or_kept("compounding")},
        "hard_to_find_or_long": {"what": f"on the held-aside set, `{TWO_ROUNDS}` minus `{LONG_ORDINARY}`: the deep solves against ordinary solves matched in length",
                                 **read_or_kept("hard_to_find_or_long")},
        "reach": {"what": f"on the goal set: the goal problems whose shortest published proof has 8 lines or more that are solved at least once in all the attempts, "
                          f"`{TWO_ROUNDS}` against `{PUBLISHED2}`: gained, lost, the two-sided sign test. REACH SHOWN AT THIS SEED when more are gained than lost with p under 0.05",
                  "long_goal_problems": len(long_ids), **of_reach, **({} if not measured else {"measured_and_not_read": long_solved} if inconclusive else {"solved": long_solved})},
        "what_follows": follows, "the_spread_of_l5": of_the_spread,
        "the_held_aside_set": {**of_the_set, "problems_measured": len(aside_ids)},
        "beside": {"what": "naming nothing: each model's own counts; every pair on all of G and its split (the goal problems not in G', and G'); `two_rounds` against the "
                           "stored models; each model's overfitting reads against the start model; each measured set's share of attempts that timed out in Lean",
                   **kept({"each_model": each, "pairs": pairs, "two_rounds_against_the_stored_models": against_stored, "overfitting": overfit, "timed_out_in_lean": timeouts})},
        "models": {name: {"model": f"`{name}`: from `{start}`, one pass over {trains[name]['rows']:,} rows: {IS[name]}",
                          "training": {key: value for key, value in trains[name].items() if key != "checkpoints"}} for name in measured},
        "search_2": {**found, "prepared": of_the_search, "the_finding": not enough["passes"]}, "round_one": of_round_one,
        "matched_in_length": bins or None, "lines_of_the_rows": lines_of or None, "rows_not_taken_by_the_rule": prepare.get("rows_not_taken_by_the_rule"), "smoke": smoke,
        "what_it_cannot_say": list(CANNOT_SAY),
        "the_stored_models": {start: f"`{start}`: {LABEL}; the model of the check of the adapter's rank; its rows on the rungs and on G are that run's, its attempts at the "
                                     "held-aside set round 1's run's: only read here", M1: f"`{M1}` of round 1: M(1), which made search 2; its rows on G are round 1's run's",
                              STORED: prepare["the_stored_model"], "runs": prepare["stored_models"]},
        "training_sets": prepare["training_sets"], "trainings": prepare["trainings"],
        "heldout": {"goal_set": len(goal_ids), "goal_set_again": len(again), "goal_set_by_length_group": {group: len(ids_) for group, ids_ in world.sets.items()},
                    "goal_set_again_by_length_group": {group: len(ids_) for group, ids_ in goal_sets([problem_id for problem_id in goal_ids if problem_id in in_again], lengths).items()},
                    "rungs": {rung: len(world.rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "contradicted_side_setting", "start",
                                                "start_adapter", "l4t2_run", "map_run")},
        "start_recipe": prepare.get("start_recipe"), "attempts": world.health, "not_to_be_read": list(world.not_to_be_read),
        "not_measured": [f"`{start}` on the rungs, on G and on the held-aside set (its rows are the stored ones of the runs named under the_stored_models)",
                         "a first round: round 1 is L5's run, only read", "further seeds of the four models, and a read over seeds: neither is built",
                         "anything at another temperature than the config's"],
    }
