"""L5m's report: the best model, measured in full. Does the gain hold on the held-aside set and on fresh attempts?
Spec: docs/spec/ladder-loop.spec.md, "L5m: the best model, measured in full" ("The reads, fixed before the run").
Pure: rows in, a report out, on the rules of `domain/ladder_round/l5_measure.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. Two stored models, nothing trained: THE CANDIDATE (one stored averaged adapter) and THE BAR
(`second_full`, the model it was trained from). Every interval is a bootstrap OVER PROBLEMS, paired by problem.

  the checks        FIRST, each PASS or FAIL: the two adapters are there, at the rank, and not each other's
                    (`adapters_are_held`); the candidate is the adapter the stage that made it measured, by that stage's
                    own record of the mean, read AGAIN as that run is when the report is built, and the one every
                    measurement here was sampled from (`is_the_one_measured`, `of_its_run`); every set is the setting's:
                    its problems, its attempts, its sampling seed (`sets_are_the_settings`); no row the candidate was
                    trained on is held out, of G or of the held-aside set (`l5.no_barred_row`, `l5_draws.no_problem_of`,
                    on that stage's own record); Lean answered. ANY FAIL: THE RUN IS INCONCLUSIVE and nothing else is
                    said (what was measured is kept under `measured_and_not_read`). L4p's OWN CHECKS OF `second_full`
                    ARE PRINTED AND DECIDE NOTHING
  the reads         THE HELD-ASIDE SET (the primary) and FRESH ATTEMPTS (all of G over the second sampling), each the
                    candidate minus the bar by `by_the_interval`; THE MEASUREMENT AGAIN (the candidate's first sampling
                    here minus its stored one), naming nothing; beside, naming nothing: all of G over all the attempts,
                    G' over the fresh ones, the goal problems solved, solved reliably, of the ones `pre_r64` never
                    solved, the long ones, each rung, the share of distinct attempts, the timeouts; what follows

A report in which Lean did not answer a set this run sampled is NOT TO BE READ (`ok` is false, and the step fails): the
task queued again samples such a set again from its adapter.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE
from rlvr_lean.domain.ladder_round.l5 import no_barred_row, timed_out
from rlvr_lean.domain.ladder_round.l5_draws import MEAN_CONSTANT, no_problem_of
from rlvr_lean.domain.ladder_round.l5_measure import (
    BAR,
    CANDIDATE,
    ON_FRESH,
    ON_THE_SET,
    adapters_are_held,
    by_the_interval,
    is_the_one_measured,
    never_solved,
    of_its_run,
    reliably,
    sets_are_the_settings,
    what_follows,
)
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import _rate, attempts_on_g, goal_sets, lean_did_not_answer, per_attempt, solved
from rlvr_lean.reporting.ladder_l2 import _health
from rlvr_lean.reporting.ladder_l3c import LENGTH_GROUPS
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY, _verdict
from rlvr_lean.reporting.ladder_l5_full import the_start_adapter
from rlvr_lean.reporting.ladder_l5_noise import LEAN_ANSWERED

L4 = "l4"
QUESTION = "the best model, measured in full"
SPEC = "docs/spec/ladder-loop.spec.md, L5m: the best model, measured in full"
RUNG_PART, HELD_ASIDE = "rungs", "held_aside"       # the two kinds of set that are not a sampling of G, as the step names a model's rows
LONG = LENGTH_GROUPS[-1]                            # the goal problems whose shortest published proof has 8 lines or more: "the long ones"
THE_ADAPTERS = "the_two_adapters_are_there_at_the_rank_and_are_not_each_others"
THE_ONE_MEASURED = "the_candidate_is_the_adapter_its_stage_measured"
THE_SETS = "every_set_is_the_settings"
NO_BARRED_ROW = "no_barred_row_in_what_the_candidate_was_trained_on"
CHECKS = (THE_ADAPTERS, THE_ONE_MEASURED, THE_SETS, NO_BARRED_ROW, LEAN_ANSWERED)
SOLVED_KEYS = ("problems", "resolved_after", "resolved_before", "gained", "lost", "sign_test_p")


def _difference(model: Sequence[Mapping], base: Sequence[Mapping], problem_ids: Sequence[str], evaluation: Mapping) -> dict:
    """The model minus the base per attempt over `problem_ids`, paired by problem with its bootstrap interval over
    problems (`per_attempt`, the ceiling's own); NOT MEASURED, with why, where the two sides do not pair (a problem on
    one side alone, or with another number of attempts: the sets check then fails, and nothing is read)."""
    try:
        return per_attempt(model, base, problem_ids, evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"])
    except ValueError as error:
        return {"problems": 0, "mean": None, "low": None, "high": None, "per_1000": None, "per_1000_of_the_base": None, "why_not": str(error)}


def _solved(model: Sequence[Mapping], base: Sequence[Mapping], problem_ids: Sequence[str]) -> dict:
    """The problems of `problem_ids` each side solved at least once, gained against lost (`solved`, the ceiling's own)."""
    try:
        own = solved(model, base, problem_ids)
    except ValueError as error:
        return {**{key: None for key in SOLVED_KEYS}, "why_not": str(error)}
    return {key: own[key] for key in SOLVED_KEYS}


def _summed(samplings: Sequence[Sequence[Mapping]]) -> list[dict]:
    """One model's rows over its samplings of G, added problem by problem; nothing where they do not hold the same problems."""
    try:
        return attempts_on_g(samplings)
    except (ValueError, KeyError):
        return []


def _counted(own: Mapping) -> str:
    return (f"{own['resolved_after']} against {own['resolved_before']} (+{own['gained']}, -{own['lost']}; sign test p = {own['sign_test_p']})"
            if own.get("resolved_after") is not None else "not counted")


def _share(own: Mapping | None, part: str) -> float | None:
    return ((own or {}).get(part) or {}).get("mean_share_distinct")


def build_l5_measure_report(prepare: Mapping, measured: Mapping, on_the_set: Mapping[str, Mapping | None], candidate: Mapping[str, Sequence[Mapping]],
                            bar: Mapping[str, Sequence[Mapping]], first_start: Mapping[str, Sequence[Mapping]], stored_first: Sequence[Mapping],
                            recorded: Mapping[str, Mapping], problems: Mapping[str, Sequence[str]], groups: Sequence[Mapping], lengths: Mapping[str, Mapping],
                            again: Sequence[str], trained: Sequence[Mapping], heldout: Sequence[str], half_seed: int, base_map: Sequence[str], evaluation: Mapping,
                            now: Mapping, of_the_setting: Mapping) -> dict:
    """`prepare`: the prepare step's summary. `measured`: the candidate's measuring step's (the rungs and G's two
    samplings). `on_the_set`: by `candidate` and `second_full`, the summary of the step that measured it on the
    held-aside set (None for the bar when its rows are a stored run's). `candidate`, `bar`: each model's per-problem
    rows by part (`rungs`, each sampling of G by its name, `held_aside`); `first_start`: `pre_r64`'s stored ones on the
    rungs and on G. `stored_first`: the candidate's stored first sampling, as its own run measured it. `recorded`: what
    each set THIS run sampled recorded of itself (by part; the held-aside sets as `held_aside_<model>`). `problems`: by
    part, the problems the first step wrote for the candidate's set. `groups`, `lengths`, `again`: the held-out groups,
    the published proofs' lengths, G'. `trained`: the rows the candidate's trainings recorded (no text). `heldout`,
    `half_seed`, `base_map`: what a barred row is read against. `evaluation`: the bootstrap's resamples and seed.
    `now`: the candidate's run AS IT IS NOW (`entry`: what its report says of its seed; `averaging`, `measuring`: its
    two steps' records). `of_the_setting`: the held-aside set's attempts and sampling seed as the config gives them now."""
    of_it, of_the_bar, samplings, smoke = prepare["the_candidate"], prepare["the_bar"], list(prepare["goal_samplings"]), prepare.get("smoke")
    # A SMOKE RUN in a world whose smoke runs recorded ONE sampling of G (the first step recorded that it went on from it): no second sampling, no read on fresh attempts.
    one_sampling = bool((smoke or {}).get("one_sampling_of_g")) and len(samplings) == 1
    if (len(samplings) != 2 and not one_sampling) or of_it["model"] is None or prepare["bar"] != BAR:
        raise ValueError(f"the report is given a run against `{prepare['bar']}` with {len(samplings)} samplings of the goal set: it reads a candidate against `{BAR}` on the "
                         "first sampling (measured again) and the second (fresh attempts)")
    sampling, fresh = samplings[0], None if one_sampling else samplings[1]
    first_name, fresh_name = sampling["name"], fresh["name"] if fresh else None
    on_g = [first_name, *([fresh_name] if fresh else [])]              # the samplings of G this run holds: two, or a smoke run's one
    parts = [RUNG_PART, *on_g]
    goal_ids, aside_ids = group_ids(groups, GOAL), list(problems[HELD_ASIDE])
    if any(sorted(problems[name]) != sorted(goal_ids) for name in on_g):
        raise ValueError("the candidate's two samplings of G were not prepared on the goal problems, each once: every read on G here is over the one goal set")
    seed, model, made_by, first = prepare["seed"], of_it["model"], of_it["made_by"], prepare["first_start"]
    no_fresh = {"problems": 0, "mean": None, "low": None, "high": None, "per_1000": None, "per_1000_of_the_base": None,
                "why_not": "the models have ONE sampling of G here (a smoke run: the smoke runs recorded its first sampling alone): there are no fresh attempts"}
    rank, alpha = (prepare.get("start_recipe") or {}).get("rank"), (prepare.get("start_recipe") or {}).get("alpha")
    of_the_bars, of_stored, of_the_aside = prepare["the_bars_held_aside"], prepare["the_stored_sets"], prepare["the_held_aside"]
    bar_stored = bool(of_the_bars["read_from_a_stored_run"])
    said = f"`{model}` of {made_by}"

    # ---- the checks, FIRST
    served_bar = None if bar_stored else (on_the_set.get(BAR) or {}).get("adapter")
    held = adapters_are_held(of_it["held"], the_start_adapter(of_the_bar["summary"]), rank, alpha)
    the_adapters = {**held, "the_bars_held_aside_set_was_sampled_from": served_bar, "the_bar_was_served": bar_stored or served_bar == prepare["bar_adapter"],
                    "passes": held["passes"] and (bar_stored or served_bar == prepare["bar_adapter"])}
    why = of_its_run(now.get("entry") or {"there": False, "why": "its report was not read"}, of_it["seed"], BAR, MEAN_CONSTANT, prepare["the_measurement"]["first_sampling"])
    record = is_the_one_measured(now.get("averaging") or {}, now.get("measuring") or {}, of_it["adapter"], rank, alpha,
                                 [measured.get("adapter"), (on_the_set.get(CANDIDATE) or {}).get("adapter")], smoke=bool(smoke))
    the_one = {**record, "run": of_it["run"], "its_report_now": why is None, "why_not": why, "passes": record["passes"] and why is None}
    setting = {RUNG_PART: (prepare["rung_episodes"], prepare["sampling_seeds"][RUNG_PART]), first_name: (sampling["episodes"], sampling["sampling_seed"]),
               **({fresh_name: (fresh["episodes"], fresh["sampling_seed"])} if fresh else {}), HELD_ASIDE: (of_the_setting["attempts"], of_the_setting["sampling_seed"])}

    def one(rows: Sequence[Mapping], part: str, own: Mapping | None) -> dict:
        return {"rows": rows, "problem_ids": problems[part], "attempts": setting[part][0], "sampling_seed": setting[part][1],
                "recorded_attempts": (own or {}).get("episodes_each"), "recorded_seed": (own or {}).get("sampling_seed")}

    sets = {**{f"the candidate ({part})": one(candidate[part], part, recorded.get(part)) for part in parts},
            "the candidate (held-aside)": one(candidate[HELD_ASIDE], HELD_ASIDE, recorded.get(f"{HELD_ASIDE}_{CANDIDATE}")),
            **{f"`{BAR}` ({part}, stored)": one(bar[part], part, of_stored["the_bar"].get(part)) for part in parts},
            f"`{BAR}` (held-aside{', stored' if bar_stored else ''})": one(bar[HELD_ASIDE], HELD_ASIDE,
                                                                         of_stored["the_bars_held_aside"] if bar_stored else recorded.get(f"{HELD_ASIDE}_{BAR}")),
            "the candidate's stored first sampling": one(stored_first, first_name, of_stored["the_candidates_first_sampling"])}
    the_sets = sets_are_the_settings(sets)
    barred, of_g = no_barred_row(trained, set(heldout), half_seed, set(base_map)), no_problem_of(trained, set(goal_ids), set(aside_ids))
    no_barred = {"what": "no row the candidate was trained on (the record of the stage that made it: each training loop's own row ids) is a held-out problem, a published "
                         "row outside the `pretrain` half, an own row outside the base map, a problem of G or a held-aside problem",
                 "rows": len(trained), "trainings": sorted({row["training"] for row in trained}), "barred": barred, "of_g_or_of_the_held_aside_set": of_g,
                 "passes": barred["passes"] and of_g["passes"]}
    sampled_here = {**{f"the candidate ({part})": candidate[part] for part in parts}, "the candidate (held-aside)": candidate[HELD_ASIDE],
                    **({} if bar_stored else {f"`{BAR}` (held-aside)": bar[HELD_ASIDE]})}
    not_to_be_read = [name for name, rows in sampled_here.items() if lean_did_not_answer(rows)]
    answered = {"what": "Lean gave a verdict on all but a small share of the attempts of every set this run sampled (the stored sets were held so by the first step)",
                "share_without_an_answer": {name: _health(rows)["share_without_an_answer"] for name, rows in sampled_here.items()}, "not_to_be_read": not_to_be_read,
                "passes": not not_to_be_read}
    checks = {THE_ADAPTERS: the_adapters, THE_ONE_MEASURED: the_one, THE_SETS: the_sets, NO_BARRED_ROW: no_barred, LEAN_ANSWERED: answered}
    failed = [name for name in CHECKS if not checks[name]["passes"]]
    inconclusive = bool(failed)

    # ---- THE READS, in the section's order
    on_the_aside = _difference(candidate[HELD_ASIDE], bar[HELD_ASIDE], aside_ids, evaluation)
    on_fresh = _difference(candidate[fresh_name], bar[fresh_name], goal_ids, evaluation) if fresh else no_fresh
    primary = by_the_interval(failed, on_the_aside, ON_THE_SET, "the held-aside set")
    second = by_the_interval(failed, on_fresh, ON_FRESH, f"all of G over the {fresh['episodes']} fresh attempts of the second sampling" if fresh else
                             "G's second sampling (a smoke run: the smoke runs recorded ONE sampling of G, and there are no fresh attempts)")
    follows = what_follows(primary, second)
    measured_again = _difference(candidate[first_name], stored_first, goal_ids, evaluation)
    both, of_the_bar_both, of_pre = (_summed([own[name] for name in on_g]) for own in (candidate, bar, first_start))
    attempts = sampling["episodes"] + (fresh["episodes"] if fresh else 0)
    long_ids, never = goal_sets(goal_ids, lengths).get(LONG, []), never_solved(of_pre, goal_ids)
    reliable, reliable_of_the_bar = set(reliably(both, goal_ids)), set(reliably(of_the_bar_both, goal_ids))
    rung_ids = {name: group_ids(groups, name) for name in RUNGS}
    beside = {
        "what": "naming nothing: the candidate minus the bar, paired by problem. All of G over all its attempts and over the first sampling as measured here; G' over the "
                f"fresh attempts; the goal problems solved, solved reliably (half the attempts or more), and of the ones `{first}` never solved; the long ones; each rung; "
                "the held-aside problems solved; the share of distinct attempts",
        "all_of_g": {"attempts_a_problem": attempts, **_difference(both, of_the_bar_both, goal_ids, evaluation)},
        "first_sampling_measured_here": {"attempts_a_problem": sampling["episodes"], **_difference(candidate[first_name], bar[first_name], goal_ids, evaluation)},
        "g_again": {"what": f"G': the goal problems `{first}` does not solve in its own first sampling, over the fresh attempts alone",
                    "attempts_a_problem": fresh["episodes"] if fresh else 0,
                    **(_difference(candidate[fresh_name], bar[fresh_name], list(again), evaluation) if fresh else no_fresh)},
        "solved": {"attempts_a_problem": attempts, **_solved(both, of_the_bar_both, goal_ids)},
        "solved_reliably": {"what": "solved in half its attempts or more", "attempts_a_problem": attempts, "candidate": len(reliable), "bar": len(reliable_of_the_bar),
                            "gained": len(reliable - reliable_of_the_bar), "lost": len(reliable_of_the_bar - reliable)},
        "never_solved_by_the_first_start": {"what": f"the goal problems `{first}` solved in none of its {attempts} attempts", "problems": len(never),
                                            **{key: value for key, value in _solved(both, of_the_bar_both, never).items() if key != "problems"}},
        "long": {"what": "the goal problems whose shortest published proof has 8 lines or more", "problems": len(long_ids),
                 "per_attempt": _difference(both, of_the_bar_both, long_ids, evaluation), "solved": _solved(both, of_the_bar_both, long_ids)},
        "rungs": {name: _difference(candidate[RUNG_PART], bar[RUNG_PART], rung_ids[name], evaluation) for name in RUNGS},
        "held_aside_solved": _solved(candidate[HELD_ASIDE], bar[HELD_ASIDE], aside_ids),
        "share_of_distinct_attempts": {"what": "the mean share of a prompt's attempts that are distinct, as each model's own measuring step counted it: on the rungs and on G",
                                       "candidate": {part: _share(measured.get("distinct_attempts"), part) for part in (RUNG_PART, GOAL)},
                                       BAR: {part: _share((of_the_bar.get("wrote") or {}).get("distinct_attempts"), part) for part in (RUNG_PART, GOAL)},
                                       "the_candidate_as_its_own_run_measured_it": _share((of_it.get("wrote") or {}).get("distinct_attempts"), GOAL)}}
    health = {**{f"candidate_{part}": _health(candidate[part]) for part in (*parts, HELD_ASIDE)}, **{f"{BAR}_{part}": _health(bar[part]) for part in (*parts, HELD_ASIDE)},
              "candidate_stored_first_sampling": _health(stored_first)}
    of_timeouts = timed_out(health, {
        "the rungs": {CANDIDATE: f"candidate_{RUNG_PART}", BAR: f"{BAR}_{RUNG_PART}"},
        f"G's first sampling (`{first_name}`)": {CANDIDATE: f"candidate_{first_name}", BAR: f"{BAR}_{first_name}", "the candidate as stored": "candidate_stored_first_sampling"},
        **({f"G's second sampling (`{fresh_name}`)": {CANDIDATE: f"candidate_{fresh_name}", BAR: f"{BAR}_{fresh_name}"}} if fresh else {}),
        "the held-aside set": {CANDIDATE: f"candidate_{HELD_ASIDE}", BAR: f"{BAR}_{HELD_ASIDE}"}})
    of_timeouts.update({"checked_again": False, "why": "no step checks the timed-out attempts again: where a pair is named here, it is to be done by hand (Lean only, no "
                                                       "measurement running) and the reads given both ways"})
    own_rows = barred["rows_of_the_models_own"]
    cannot_say = [
        "One candidate: one seed, one draw of rows. It says whether THIS adapter's gain holds, not whether the recipe that made it does",
        "The candidate was CHOSEN for its figure on G's first sampling: its first sampling here, and every figure over all its attempts, carry that selection; the two reads "
        "(the held-aside set, the fresh attempts) do not",
        "G and the held-aside set are of one source: neither says anything of problems of another kind",
        f"`{BAR}` is ONE training in one order: the bar carries that order's luck",
        ("Nothing here is about the model's own proofs: no row the candidate was trained on is one" if not own_rows else
         f"{own_rows} of the rows the candidate was trained on are the model's own proofs: the reads are of this adapter, and say nothing of own proofs as a kind")]

    own_checks, the_saved, read_now = of_the_bar["summary"]["l4ps_own_checks"], (of_it["held"]["recorded"] or {}).get("adapter_saved"), of_it["held"]["read_now"]
    lines = [
        f"{SAY}: L5m, {QUESTION.upper()}. Seed {seed}. NO TRAINING. THE CANDIDATE is the stored averaged adapter {said} ({of_it['adapter']}; the mean of "
        f"{len(of_it['the_mean_of'])} trainings), chosen for its figure on G's first sampling, the one set more than a dozen models have been compared on. THE BAR is `{BAR}` "
        f"({prepare['bar_adapter']}), the model it was trained from. The candidate is measured here as L5's models are: {prepare['rung_episodes']} episodes on each of "
        f"{len(problems[RUNG_PART])} rung problems, "
        + (f"{sampling['episodes']} and {fresh['episodes']} attempts at each of {len(goal_ids)} goal problems (its first sampling MEASURED AGAIN, sampling seed "
           f"{sampling['sampling_seed']}; its second, seed {fresh['sampling_seed']})" if fresh else
           f"{sampling['episodes']} attempts at each of {len(goal_ids)} goal problems (its first sampling MEASURED AGAIN, sampling seed {sampling['sampling_seed']}; NO second "
           "sampling)") + f", and {of_the_aside['episodes_each']} at each of "
        f"{len(aside_ids)} held-aside problems (seed {of_the_aside['sampling_seed']}). The bar's rungs and G are L4p's stored attempts; on the held-aside set it is "
        + (f"READ FROM A STORED RUN ({of_the_bars['looked_in']})" if bar_stored else "MEASURED HERE from its stored adapter") + ". Every interval is a bootstrap over problems"]
    if smoke:
        lines.append(f"{SAY}: A SMOKE RUN: L5's smoke run " + (f"reads INCONCLUSIVE ({', '.join(smoke['l5s_failed_checks'])}) and was read all the same; "
                                                              if smoke.get("l5_read_though_inconclusive") else "was read; ")
                     + "the candidate is a smoke run's adapter. "
                     + ("" if fresh else "The models have ONE sampling of G here (the smoke runs recorded its first sampling alone): the candidate is measured on that one, "
                                         "and the read on fresh attempts, G' beside it and what follows are NOT MADE. ")
                     + "Nothing here is a read of a model")
    lines += [
        f"{SAY}: CHECK, the candidate's adapter and `{BAR}`'s are there, at rank {rank} and alpha {alpha}, and are not each other's: the candidate's "
        + (f"was saved at rank {the_saved.get('rank')} and alpha {the_saved.get('alpha')}" if the_saved else "has no recorded read back")
        + f", {(of_it['held']['recorded'] or {}).get('numbers_not_its_start_adapters')} of its {(of_it['held']['recorded'] or {}).get('numbers')} numbers not its start "
        "adapter's (its own stage's record)"
        + (f"; read now at rank {read_now['read_back']['rank']} and alpha {read_now['read_back']['alpha']}, {read_now['against_the_bar']['elements_changed']:,} of its "
           f"{read_now['against_the_bar']['elements']:,} numbers not the bar's" if read_now else "; its files were not read by the first step (no GPU box)")
        + f"; the bar's: {'held' if held['the_bar']['passes'] else 'NOT held'} as L5f holds it"
        + ("" if bar_stored else f", and its held-aside set was sampled from {served_bar}") + f": {_verdict(the_adapters)}",
        f"{SAY}: CHECK, the candidate is the adapter {made_by} measured ({of_it['run']}): by that run's own record it is the mean of {record['of']} "
        f"({record['elements_not_the_mean']} of {record['elements']} numbers not that mean; the mean: {record['is_the_mean']}; at its rank: {record['at_its_rank']}), the one "
        f"its own measurement was sampled from ({record['one_adapter']}) and the one every measurement here was sampled from ({record['served_here']}); its report now: "
        + ("read, of this seed, start and recipe, measured alike" if why is None else f"NOT ({why})") + f": {_verdict(the_one)}",
        f"{SAY}: CHECK, every set is the setting's (its problems, its attempts a problem, the sampling seed its own step recorded): {len(the_sets['sets'])} sets, "
        + "; ".join(f"{name} {own['problems']} problems of {own['problems_of_the_setting']}, {own['recorded_attempts']} attempts, seed {own['recorded_sampling_seed']}"
                    for name, own in the_sets["sets"].items())
        + f"; not the setting's: {the_sets['not_the_settings']}: {_verdict(the_sets)}",
        f"{SAY}: CHECK, no row the candidate was trained on is barred ({made_by}'s own record: {no_barred['rows']:,} rows of {len(no_barred['trainings'])} trainings): "
        f"held-out problems {barred['held_out_problems']}, published rows not of the `pretrain` half {barred['published_rows_not_of_the_pretrain_half']}, own rows not of "
        f"the base map {barred['own_rows_not_of_the_base_map']}, problems of G {of_g['problems_of_the_goal_set']}, held-aside problems "
        f"{of_g['problems_of_the_held_aside_set']}: {_verdict(no_barred)}",
        f"{SAY}: CHECK, Lean answered (the share of each sampled set's attempts without a verdict): "
        + "; ".join(f"{name} {share}" for name, share in answered["share_without_an_answer"].items()) + f": {_verdict(answered)}",
        f"{SAY}: L4p's OWN CHECKS OF `{BAR}`, PRINTED AND DECIDING NOTHING (a second pass over rows the model has trained on has no falling loss to show; L6 treats it so): "
        + (f"its report reads the model {INCONCLUSIVE} for {own_checks['failed']}" if own_checks["failed"] else
           "its report failed none of them" if own_checks["report_there"] else "its report is not on the box")]
    if inconclusive:
        lines.append(f"{SAY}: THIS RUN IS {INCONCLUSIVE}: {'; '.join(failed)}. Nothing else is said of it")
    else:
        rate = lambda own: f"{_rate(own)} ({own['problems']} problems)"      # noqa: E731
        solves, reliable_, of_never, long = beside["solved"], beside["solved_reliably"], beside["never_solved_by_the_first_start"], beside["long"]
        shares = beside["share_of_distinct_attempts"]
        lines += [
            f"{SAY}: THE HELD-ASIDE SET (the primary): {primary['name']}. {primary['reason']}. Held-aside problems solved at least once: {_counted(beside['held_aside_solved'])}",
            f"{SAY}: FRESH ATTEMPTS: {second['name']}. {second['reason']}",
            f"{SAY}: THE MEASUREMENT AGAIN, naming nothing: the candidate's first sampling here minus its stored one ({made_by}'s own measurement, the same sampling seed), "
            f"per attempt: {rate(measured_again)}",
            f"{SAY}: BESIDE, NAMING NOTHING, the candidate minus `{BAR}`, paired by problem. All of G over the {attempts} attempts: {rate(beside['all_of_g'])}; over the first "
            f"sampling as measured here ({sampling['episodes']} attempts, the selected-on sampling): {rate(beside['first_sampling_measured_here'])}; G' over the "
            + (f"{fresh['episodes']} fresh attempts: {rate(beside['g_again'])}" if fresh else "fresh attempts: not read (ONE sampling of G here, a smoke run)"),
            f"{SAY}: BESIDE, the goal problems. Solved in {attempts} attempts: {_counted(solves)}; solved reliably (half the attempts or more): {reliable_['candidate']} against "
            f"{reliable_['bar']} (+{reliable_['gained']}, -{reliable_['lost']}); of the {of_never['problems']} `{first}` never solved in its {attempts}: {_counted(of_never)}; "
            f"the {long['problems']} long ones (8 lines or more): {rate(long['per_attempt'])}, solved {_counted(long['solved'])}",
            f"{SAY}: BESIDE, each rung (the easy rung is where L4r read a cost): "
            + "; ".join(f"{name} {rate(beside['rungs'][name])}" for name in RUNGS)
            + f". The share of distinct attempts: on the rungs {shares['candidate'][RUNG_PART]} against {shares[BAR][RUNG_PART]}; on G {shares['candidate'][GOAL]} against "
            f"{shares[BAR][GOAL]}",
            f"{SAY}: TIMEOUTS, each set's share of attempts that timed out in Lean: " + "; ".join(f"{name} {share}" for name, share in of_timeouts["share"].items())
            + (f". MORE THAN A POINT APART: {'; '.join(pair['kind'] + ' (' + ' and '.join(pair['models']) + ')' for pair in of_timeouts['apart'])}: the timed-out attempts are "
               "to be checked again (Lean only, no measurement running) and the reads given both ways; no step does it" if of_timeouts["apart"] else
               ". No two of one kind of set are more than a point apart"),
            f"{SAY}: WHAT FOLLOWS: {follows['name']}. {follows['reason']}"]
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}. The step fails; the task queued again samples "
                     "such a set again from its adapter")

    headline = (f"L5m ({QUESTION}; {LABEL}) seed {seed}, the candidate {said} against `{BAR}`: "
                + (f"THIS RUN IS {INCONCLUSIVE} ({'; '.join(failed)})" if inconclusive else
                   f"THE HELD-ASIDE SET: {primary['name']} ({_rate(on_the_aside)}); FRESH ATTEMPTS: {second['name']}"
                   + (f" ({_rate(on_fresh)})" if fresh else " (ONE sampling of G here, a smoke run)") + "; the measurement again "
                   f"{_rate(measured_again)}")
                + f". What follows: {follows['name']}"
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    kept = lambda what: {"measured_and_not_read": what} if inconclusive else what      # noqa: E731
    candidate_of = {key: of_it[key] for key in ("of", "seed", "model")}
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L5m", "headline": headline, "lines": lines, "ok": not not_to_be_read, "seed": seed,
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any((entry or {}).get("stand_in_engine") for entry in (measured, *on_the_set.values()))),
        "candidate": candidate_of, "bar": BAR, "first_start": first, "rank": rank,
        "branches": {"the_held_aside_set": primary["name"], "fresh_attempts": second["name"], "what_follows": follows["name"]},
        "inconclusive": inconclusive, "failed_checks": failed,
        "can_this_run_be_read": {"what": "the checks the spec reads first; when any fails the run is INCONCLUSIVE and nothing else is said of it", **checks},
        "l4ps_own_checks_of_the_bar": own_checks,
        "of_this_seed": {"seed": seed, "candidate": candidate_of, "bar": BAR, "ok": not not_to_be_read, "inconclusive": inconclusive, "failed_checks": failed,
                         "the_held_aside_set": None if inconclusive else on_the_aside, "fresh_attempts": None if inconclusive or not fresh else on_fresh},
        "the_held_aside_set": primary if inconclusive else {
            "what": f"THE PRIMARY: the candidate minus `{BAR}` on the held-aside set, successes per attempt, paired by problem, bootstrap over problems. A set no stage of "
                    "the night selected on", **primary, "difference": on_the_aside, "problems": len(aside_ids), "attempts_a_problem": of_the_aside["episodes_each"]},
        "fresh_attempts": second if inconclusive or not fresh else {
            "what": f"the candidate minus `{BAR}` on all of G over the second sampling alone: attempts no stage of the night selected on", **second, "difference": on_fresh,
            "problems": len(goal_ids), "attempts_a_problem": fresh["episodes"]},
        "the_measurement_again": kept({"what": "the candidate's first sampling as measured here minus its stored one (the same adapter, attempts and sampling seed, another "
                                               "run of the model server): what a measurement is worth, naming nothing", **measured_again}),
        "beside": kept(beside), "timed_out_in_lean": of_timeouts, "what_follows": follows,
        "the_candidate": of_it, "the_bar": of_the_bar, "the_search": prepare["the_search"], "the_measurement": prepare["the_measurement"], "the_held_aside": of_the_aside,
        "the_bars_held_aside": of_the_bars, "smoke": smoke, "what_it_cannot_say": cannot_say,
        "attempts": health, "not_to_be_read": not_to_be_read,
        "not_built": ["the check again in Lean of the timed-out attempts where two shares differ by more than a point: the pairs are named",
                      "a candidate that is ONE training (L5c's `single`): this stage measures a stored AVERAGED adapter, held by its record of the mean",
                      "a further seed or a second candidate in one run: one task measures one stored adapter",
                      "the counts with assembly: `python -m rlvr_lean.tools.ladder_goal_assembly` on the stored goal attempts"],
        "not_measured": [f"`{BAR}` on the rungs and on G: its rows are L4p's stored attempts",
                         f"`{first}`: its stored rows name G' and the goal problems it never solved; it is not sampled",
                         "anything at another temperature than the config's"]}
