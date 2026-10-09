"""L4's two reports: the model PRETRAINED ON PUBLISHED PROOFS (`pre`), and the loop from it. Spec:
docs/spec/ladder-loop.spec.md, "L4: the loop from a model pretrained on published proofs", "The read, fixed before
any run". Pure: rows in, a report out. Their pieces are the ceiling report's and L3d's, read on this stage's models.

Everything here is labelled pretrained on published proofs: distillation of other provers, followed by the loop. One-shot
attempts only. Every comparison is over the same problems and the same number of attempts on each, paired by problem,
with a 95% bootstrap interval over problems.

`build_pretrain_report`   `pre` against the base, beside the stored three-round model and the ceiling's two models when
                          they were read; the goal set drawn again (G': the goal problems `pre` does not solve in its
                          first sampling, with their ids: the arm reads them from this report's run); the first two
                          "can this run see a win" checks; and `pre`'s own map beside the base's stored one (the
                          problems by k, the mean pass rate): what the arm's challenger starts from
`build_l4_report`         the arm. The checks FIRST (any FAIL: INCONCLUSIVE, and nothing else is said: what was measured
                          is kept under `measured_and_not_read`); the PRIMARY on G' over the attempts of the SECOND
                          sampling (fresh for those problems: nothing of it chose them), `with` minus `pre`; beside it
                          the base arm's own gain, the same quantity from its stored rows; the branch; the secondary reads

The counts WITH ASSEMBLY are not here: `tools/ladder_goal_assembly.py` makes them after the run, Lean only. THE LENGTH
OF A HELD-OUT PROBLEM'S PUBLISHED PROOF is read here and nowhere before.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.ceiling import interval_text, still_writes_proofs
from rlvr_lean.domain.ladder_round.l3d import WITH, WITHOUT
from rlvr_lean.domain.ladder_round.l3d2 import round_table
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, PRE, arm_checks, goal_set_again, l4_branch, map_summary, pretraining_checks, the_training_ran
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids, paired_change
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import ALL, FOUR_PLUS, LOOP, RUNG_PART, _named, _rate, attempts_on_g, goal_sets, lean_did_not_answer, per_attempt, solved, verified_lines
from rlvr_lean.reporting.ladder_l2 import _health, _split
from rlvr_lean.reporting.ladder_l3d1 import BASE, by_attempts_alone

L4 = "l4"
LABEL = "pretrained on published proofs"
SAY = f"{L4} ({LABEL})"                 # what every printed line of both reports begins with
CEILING_PREFIX = "ceiling_"             # the ceiling's two models, as the pretraining's report names them
PRIMARY = "with_minus_pre"
# Every pair that is read on all of G and on the rungs: (the model, what it is compared with).
PAIRS = {PRIMARY: (WITH, PRE), "with_minus_without": (WITH, WITHOUT), "pre_minus_base": (PRE, BASE), "with_minus_base": (WITH, BASE),
         "without_minus_base": (WITHOUT, BASE), "loop_minus_base": (LOOP, BASE)}
NOT_MEASURED = {"problems": 0, "mean": None, "low": None, "high": None}


def names_of(prepare: Mapping, models: Mapping[str, Mapping]) -> dict[str, str]:
    """How each model is called in a line: the base; `pre`, `with`, `without`; the stored three-round model with its
    arm's target rate; the ceiling's two models by the proofs each had been trained on."""
    rate = prepare.get("loop_target_rate")
    at = f" at t = {Fraction(str(rate))}" if rate is not None else ""
    names = {BASE: "the base"}
    for name, entry in models.items():
        names[name] = (f"the stored three-round model{at}" if name == LOOP else
                       f"the ceiling's {entry['rows']:,}-proof model" if name.startswith(CEILING_PREFIX) else f"`{name}`")
    return names


def _sets_health(who: Mapping[str, Mapping]) -> tuple[dict, list[str]]:
    """(by measured set, whether Lean answered; the sets that are NOT TO BE READ: Lean gave no verdict on too much)."""
    measured = {}
    for name, entry in who.items():
        measured[f"rungs_{name}"] = entry[RUNG_PART]
        measured.update({f"goal_{name}_{index}": rows for index, rows in enumerate(entry[GOAL], start=1)})
    return {name: _health(rows) for name, rows in measured.items()}, sorted(name for name, rows in measured.items() if lean_did_not_answer(rows))


def _what_they_wrote(who: Mapping[str, Mapping]) -> tuple[dict, dict, dict]:
    lines = {name: {part: verified_lines(entry["verified_proof_lines"][part]) for part in (GOAL, RUNG_PART)} for name, entry in who.items()}
    return lines, {name: entry["distinct_attempts"] for name, entry in who.items()}, {name: entry["opens_with_a_have"] for name, entry in who.items()}


def _lines_line(names: Mapping[str, str], lines_verified: Mapping[str, Mapping]) -> str:
    return "; ".join(f"{names[name]} {entry[GOAL]['proofs']}, {entry[GOAL]['median']}, {entry[GOAL]['mean']}, {entry[GOAL]['share_with_4_lines_or_more']}, "
                     f"{entry[GOAL]['share_with_8_lines_or_more']}, {entry[GOAL]['longest']}" for name, entry in lines_verified.items())


def _verdict(check: Mapping) -> str:
    return "PASS" if check["passes"] else "FAIL"


def _by_k(summary: Mapping) -> str:
    return ", ".join(f"{k}: {count:,}" for k, count in summary["problems_by_k"].items())


def adapter_against_the_start(train: Mapping) -> Mapping | None:
    """What one training of the arm recorded when it compared the adapter it saved with the start adapter's file (it
    saves one adapter, so the record holds one entry); None for a training that recorded none (the stand-in)."""
    recorded = train.get("against_the_start_adapter") or {}
    return next(iter(recorded.values()), None)


def _ran(name: str, entry: Mapping) -> str:
    """One training as the check's line says it: what of its adapter is not the start's, and its loss at both ends."""
    differs, loss = entry["the_adapter_differs_from_the_start"], entry["the_loss_did_not_rise"]
    adapter = (f"{differs['elements_changed']:,} of the {differs['elements']:,} numbers of its adapter are not the start adapter's" if differs["compared"]
               else "its adapter was not compared with the start adapter's")
    return (f"{name}: {adapter}; loss {loss['last']} over the last {loss['rows_compared']} rows against {loss['first']} over the first, a rise of {loss['rise']} "
            f"where {loss['allowed']} is allowed (standard error {loss['standard_error']})")


# -------------------------------------------------------------------------------------------- the pretraining
def build_pretrain_report(prepare: Mapping, train: Mapping, groups: Sequence[Mapping], lengths: Mapping[str, Mapping], base: Mapping, models: Mapping[str, Mapping],
                          minimums: Sequence[int], settings: Mapping, evaluation: Mapping, maps: Mapping) -> dict:
    """`prepare` and `train`: those steps' summaries. `groups`: the held-out groups. `lengths`: by problem, its shortest
    published proof's length group. `base` and each of `models` (`pre`; beside it `loop` when a stored three-round model
    was read, and `ceiling_small` and `ceiling_full` when the ceiling's run was, each with the `rows` it was trained
    on): `rungs` (per-problem rows on the three rungs), `goal` (one list of per-problem rows for each sampling of G),
    and what the model wrote (`verified_proof_lines`, `distinct_attempts`, `opens_with_a_have`, `episodes_of_8`).
    `minimums`: (the goal problems `pre` must solve, the problems G' must hold). `settings`: `ladder_loop.l4`.
    `maps`: the rows of `pre`'s own map and of the base's stored one (`pre`, `base`), with the `sampling_seed` and
    the `file` the arm reads."""
    resamples, seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    goal_ids, rung_ids = group_ids(groups, GOAL), {name: group_ids(groups, name) for name in RUNGS}
    sets = goal_sets(goal_ids, lengths)
    who, names = {BASE: base, **models}, names_of(prepare, models)
    on_g = {name: attempts_on_g(entry[GOAL]) for name, entry in who.items()}
    by_length = {name: {group: per_attempt(on_g[name], on_g[BASE], ids, resamples, seed) for group, ids in sets.items()} for name in models}
    solved_on_g = {name: {group: solved(on_g[name], on_g[BASE], ids) for group, ids in sets.items()} for name in models}
    rungs = {name: {rung: paired_change(models[name][RUNG_PART], base[RUNG_PART], rung_ids[rung], resamples, seed) for rung in RUNGS} for name in models}
    alone = by_attempts_alone(who, sets)
    lines_verified, distinct, have = _what_they_wrote(who)

    # ---- the goal set drawn again for the stronger model, and the two checks
    again = goal_set_again(models[PRE][GOAL][0], goal_ids)
    in_again = set(again)
    again_by_length = {group: sum(problem_id in in_again for problem_id in ids) for group, ids in sets.items()}
    checks = pretraining_checks(on_g[PRE], again, goal_ids, *minimums)
    took, enough = checks["the_pretraining_took"], checks["the_goal_set_again_is_large_enough"]
    writes = still_writes_proofs(models[PRE][RUNG_PART], settings["maximum_share_without_an_answer"])
    health, not_to_be_read = _sets_health(who)
    passes = all(check["passes"] for check in checks.values())
    the_maps = {name: map_summary(maps[name]) for name in (PRE, BASE)}

    attempts, first = prepare["attempts_a_goal_problem"], prepare["goal_samplings"][0]["episodes"]
    kinds = ", ".join(f"{count:,} {kind}" for kind, count in prepare["rows_by_kind"].items())
    lines = [
        f"{SAY}: L4, THE PRETRAINING, seed {prepare['seed']}. `pre`: ONE pass from the base over {prepare['rows']:,} published proofs of the `pretrain` half of the pool "
        f"({kinds}), in the file's order, {train['steps']:,} optimizer steps. Distillation of other provers. The adapter is KEPT: it is the starting model of the arm. "
        "No held-out problem, no base-map problem and no problem of the `loop` half is in the file",
        f"{SAY}: CHECK 1, the pretraining took: `pre` solves {took['goal_problems_solved']} of the {took['goal_problems']} goal problems in its {attempts} one-shot "
        f"attempts a problem; at least {took['minimum']} is asked: {_verdict(took)}",
        f"{SAY}: CHECK 2, the goal set drawn again is large enough: G' (the goal problems `pre` does not solve in its {first}-attempt sampling) holds "
        f"{enough['problems']}; at least {enough['minimum']} is asked: {_verdict(enough)}",
        f"{SAY}: G' by the length of the shortest published proof: " + "; ".join(f"{_named(group)}: {again_by_length[group]} of {len(ids)}" for group, ids in sets.items()),
        f"{SAY}: for information, the training loss of `pre`: mean over the first {train['rows_compared']} rows {train['mean_loss_over_the_first_rows']}, over the last "
        f"{train['mean_loss_over_the_last_rows']}; attempts on the three rungs without an answer: {writes['share']} of {writes['attempts']:,}"]
    if prepare.get("stored_runs") is None:
        lines.append(f"{SAY}: L2's stored runs were not read (a smoke run): G was attempted {attempts} times a problem, and no stored model stands beside `pre`")
    for name in models:
        lines.append(f"{SAY}: successes per attempt on G by the length of the shortest published proof, {names[name]} minus the base: "
                     + "; ".join(f"{_named(group)} ({len(ids)}): {_rate(by_length[name][group])}" for group, ids in sets.items()))
    for name in models:
        lines.append(f"{SAY}: goal problems solved at {attempts} attempts, {names[name]} against the base: "
                     + "; ".join(f"{_named(group)}: {solved_on_g[name][group]['resolved_after']} to {solved_on_g[name][group]['resolved_before']}, "
                                 f"{_split(solved_on_g[name][group])}" for group in (ALL, FOUR_PLUS)))
    for group in (ALL, FOUR_PLUS):
        lines.append(f"{SAY}: goal problems solved at least once / reliably (in at least half of their episodes of 8 one-shot attempts), BY ATTEMPTS ALONE, "
                     f"{_named(group)} ({len(sets[group])}): "
                     + "; ".join(f"{names[name]} {alone[name][group]['solved_at_least_once']} / {alone[name][group]['reliably']}" for name in who))
    for name in models:
        lines.append(f"{SAY}: the three rungs ({prepare['rung_episodes']} episodes a problem), {names[name]} minus the base: "
                     + "; ".join(f"{rung} {interval_text(rungs[name][rung])}" for rung in RUNGS))
    lines.append(f"{SAY}: the lines of the proofs each model verifies on G (proofs, median, mean, share with 4 lines or more, with 8 or more, longest): "
                 + _lines_line(names, lines_verified))
    lines.append(f"{SAY}: the share of distinct attempts (on the rungs, on G): "
                 + "; ".join(f"{names[name]} {entry[RUNG_PART]['mean_share_distinct']}, {entry[GOAL]['mean_share_distinct']}" for name, entry in distinct.items()))
    lines.append(f"{SAY}: THE MAP THE ARM'S CHALLENGER STARTS FROM is `pre`'s own ({maps['file']}): the base map's {the_maps[PRE]['problems']:,} problems attempted again by "
                 f"`pre`, {the_maps[PRE]['episodes_each']} attempts each, sampling seed {maps['sampling_seed']}. Problems by k of {the_maps[PRE]['episodes_each']}: `pre` "
                 f"{_by_k(the_maps[PRE])}, mean pass rate {the_maps[PRE]['mean_pass_rate']}; the base's stored map {_by_k(the_maps[BASE])}, mean pass rate "
                 f"{the_maps[BASE]['mean_pass_rate']}")
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}. The step fails; the task queued again "
                     "measures such a set of this run's own again from the kept adapter")
    on_all = by_length[PRE][ALL]
    headline = (f"L4's pretraining ({LABEL}) seed {prepare['seed']}: `pre`, one pass over {prepare['rows']:,} published proofs. The pretraining took {_verdict(took)} "
                f"({took['goal_problems_solved']} goal problems solved, at least {took['minimum']} asked); G' {_verdict(enough)} ({enough['problems']} problems, at least "
                f"{enough['minimum']} asked). On all of G, `pre` minus the base: {_rate(on_all)}"
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, L4: the loop from a model pretrained on published proofs (the pretraining)", "stage": L4, "label": LABEL,
        "headline": headline, "lines": lines, "ok": not not_to_be_read, "checks_pass": passes, "seed": prepare["seed"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or train.get("stand_in_engine") or models[PRE].get("stand_in_engine")),
        "can_this_run_see_a_win": {"what": "the first two of the arm's checks, read on `pre`: the arm is not run when one fails (its prepare step refuses)", **checks},
        "goal_set_again": {"what": f"G': the goal problems `pre` does not solve in its {first}-attempt sampling (the FIRST sampling of G). Its other sampling is fresh for "
                                   "these problems: nothing of it chose them. The arm's primary is read on them",
                           "problems": len(again), "by_length_group": again_by_length, "problem_ids": again},
        "for_information": {"the_training_took": {"first": train["mean_loss_over_the_first_rows"], "last": train["mean_loss_over_the_last_rows"],
                                                  "rows_compared": train["rows_compared"], "passes": train.get("the_training_took")},
                            "pre_still_writes_proofs": writes},
        "against_the_base": {
            "by_length_group": {"what": "successes per attempt, the model minus the base, paired by problem, by the length group of each goal problem's shortest "
                                        "published proof, on the problems of 4 lines or more and on all of G", "problems": {group: len(ids) for group, ids in sets.items()},
                                "models": {name: {"model": names[name], **by_length[name]} for name in models}},
            "goal_problems_solved": {"what": f"goal problems solved at least once in their {attempts} attempts: gained = by the model and not by the base, lost = the "
                                             "reverse; two-sided sign test; by length group", "models": {name: {"model": names[name], **solved_on_g[name]} for name in models}},
            "the_three_rungs": {"what": f"fresh pass rate on each held-out rung ({prepare['rung_episodes']} episodes a problem), the model minus the base, paired by problem",
                                "problems": {rung: len(rung_ids[rung]) for rung in RUNGS}, "models": {name: {"model": names[name], **rungs[name]} for name in models}}},
        "goal_problems_solved_by_attempts_alone": {
            "what": "each model's one-shot attempts at a goal problem cut into episodes of 8 in the order drawn; goal problems resolved in at least one, a quarter, half "
                    "(RELIABLY) and nine tenths of their episodes. No Lean here: the counts with assembly come from tools/ladder_goal_assembly.py",
            **{name: {"model": names[name], **alone[name]} for name in who}},
        "verified_proof_lines": {"what": "the line counts of the proofs each model verifies, on G over all its attempts and on the three rungs", **lines_verified},
        "distinct_attempts": {"what": "the share of a model's attempts at one problem and side that are distinct", **distinct},
        "opens_with_a_have": {"what": "the share of a model's attempts whose first step is a `have`, on the three rungs and on G", **have},
        "the_map_the_arm_starts_from": {
            "what": "`pre`'s own map: the base map's problems attempted again by `pre` with the stored map's sides, number of attempts and sampling seed. The arm's "
                    "challenger reads it wherever the base arm's reads the base's stored map, which stands beside it here",
            "file": maps["file"], "sampling_seed": maps["sampling_seed"], PRE: the_maps[PRE], BASE: the_maps[BASE]},
        "models": {**names, PRE: f"`pre`: {LABEL}, one pass from the base over {prepare['rows']:,} proofs of the pool's `pretrain` half; kept"},
        "training": {key: value for key, value in train.items() if key != "checkpoints"},
        "pretraining_file": {key: prepare[key] for key in ("pretraining_file", "pretraining_file_sha256", "rows", "half", "half_seed", "rows_by_kind", "rows_by_proof_lines",
                                                           "tokens", "longest_example_tokens", "max_sequence_tokens", "order", "steps", "recipe")},
        "the_ceiling": {key: value for key, value in (prepare.get("ceiling") or {}).items()},
        "heldout": {"goal_set": len(goal_ids), "goal_set_by_length_group": {group: len(ids) for group, ids in sets.items()},
                    "rungs": {rung: len(rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "stored_runs", "loop_arm",
                                                "stored_measurements", "contradicted_side_setting")},
        "attempts": health, "not_to_be_read": not_to_be_read,
        "not_measured": ["the base, the stored three-round model and the ceiling's two models (their rows are the stored ones of the runs named under sizes.stored_runs "
                         "and the_ceiling)",
                         "anything with assembly on the goal set: the counts by episode with assembly are made after the run, Lean only (tools/ladder_goal_assembly.py)",
                         "the loop from `pre`: that is the arm's task (stage ladder_l4)"],
    }


# ------------------------------------------------------------------------------------------------ the arm
def build_l4_report(prepare: Mapping, own: Mapping, arm_prepare: Mapping, again: Sequence[str], trains: Mapping[str, Mapping], row_losses: Mapping[str, Sequence[float]],
                    trained_on: Mapping[str, Sequence[str]], assembled_lines: Sequence[int], rounds: Mapping[int, Mapping], groups: Sequence[Mapping],
                    lengths: Mapping[str, Mapping], base: Mapping, models: Mapping[str, Mapping], base_arm: Sequence[Sequence[Mapping]] | None, settings: Mapping,
                    target_rate: float, evaluation: Mapping) -> dict:
    """`prepare`: the summary of the step that read the stored runs into the arm's run (`ladder_l3d2_prepare`); `own`:
    `ladder_l4_prepare`'s (the pretraining run it stands on and that run's two checks); `arm_prepare`: the arm's own.
    `again`: G', the ids the pretraining's report stored. `trains`, `row_losses`, `trained_on`: by training of the arm
    (M(1) to the last, and the twin), its step's summary (with what it recorded `against_the_start_adapter`), its
    rows' losses and the problems of the rows it was trained on. `assembled_lines`: the lines, once minimised, of
    every proof the rounds assembled. `rounds`: by round,
    its `summary`, `results` and `examples`. `base` and `models` (`pre`, `with`, `without`, and `loop` when a stored
    three-round model was read): as the pretraining's report takes them. `base_arm`: the base arm's last model on G, one
    list of per-problem rows a sampling, from its stored run; None when that run was not read. `settings`:
    `ladder_loop.l4`. `target_rate`: the arm's."""
    resamples, seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    goal_ids, rung_ids = group_ids(groups, GOAL), {name: group_ids(groups, name) for name in RUNGS}
    sets = goal_sets(goal_ids, lengths)
    who = {BASE: base, **{name: models[name] for name in (PRE, WITH, WITHOUT, LOOP) if name in models}}
    names = names_of(prepare, {name: entry for name, entry in who.items() if name != BASE})
    pairs = {name: pair for name, pair in PAIRS.items() if pair[0] in who and pair[1] in who}
    on_g = {name: attempts_on_g(entry[GOAL]) for name, entry in who.items()}
    by_length = {name: {group: per_attempt(on_g[model], on_g[other], ids, resamples, seed) for group, ids in sets.items()} for name, (model, other) in pairs.items()}
    rungs = {name: {rung: paired_change(who[model][RUNG_PART], who[other][RUNG_PART], rung_ids[rung], resamples, seed) for rung in RUNGS}
             for name, (model, other) in pairs.items()}
    solved_on_g = {name: {group: solved(on_g[model], on_g[other], ids) for group, ids in sets.items()} for name, (model, other) in pairs.items()
                   if name in (PRIMARY, "with_minus_without")}
    alone = by_attempts_alone(who, sets)
    lines_verified, distinct, have = _what_they_wrote(who)
    table = round_table({number: {"results": entry["results"], "examples": entry["examples"], "assembly": entry["summary"].get("assembly")}
                         for number, entry in rounds.items()}, target_rate)

    # ---- the primary: on G', over the attempts of the SECOND sampling of G alone (fresh for G': the first chose it)
    samplings = prepare["goal_samplings"]
    fresh = samplings[1] if len(samplings) > 1 else None
    in_again = set(again)
    stray = sorted(in_again - set(goal_ids))
    if stray:
        raise ValueError(f"{len(stray)} problems of G' are not goal problems of this run (first: {stray[0]}): the pretraining's report is of another held-out set")
    second = lambda entry: entry[GOAL][1]      # noqa: E731

    def on_the_fresh_attempts(model: str, other: str, ids: Sequence[str]) -> dict:
        return per_attempt(second(who[model]), second(who[other]), ids, resamples, seed) if fresh else dict(NOT_MEASURED)

    primary = on_the_fresh_attempts(WITH, PRE, list(again))
    again_by_length = {group: on_the_fresh_attempts(WITH, PRE, [problem_id for problem_id in ids if problem_id in in_again]) for group, ids in sets.items()}
    twin_on_again = on_the_fresh_attempts(WITH, WITHOUT, list(again))
    # ---- beside it: the base arm's own gain, the same quantity from its stored rows
    base_again = goal_set_again(base[GOAL][0], goal_ids)
    beside = None
    if base_arm is not None and fresh:
        beside = {"what": f"the base arm ({own['base_arm']}), from its stored rows: on the goal problems the base does not solve in its {samplings[0]['episodes']}-attempt "
                          f"sampling, successes per attempt over the {fresh['episodes']} attempts of the second sampling, its last model minus the base",
                  "run": own.get("base_arm_run"), "goal_set_again_of_the_base": len(base_again), **per_attempt(base_arm[1], second(base), base_again, resamples, seed)}

    # ---- the checks; then the branch
    measured_here = {name: who[name][RUNG_PART] for name in (PRE, WITH, WITHOUT)}
    last = prepare["rounds"][-1]
    of_the_measured = (f"M({last})", f"`{WITHOUT}`")           # the two trainings whose models are measured: `with`'s and `without`'s
    as_recorded = lambda name: {"row_losses": row_losses[name], "against_the_start_adapter": adapter_against_the_start(trains[name])}      # noqa: E731
    checks = arm_checks(own["the_two_checks_of_the_pretraining"], {name: as_recorded(name) for name in of_the_measured if name in row_losses}, measured_here, trained_on,
                        {row["problem_id"] for row in groups}, settings["half_seed"], settings)
    # The other trainings (the models of the rounds before the last) are read the same way and decide nothing.
    others = {name: the_training_ran(row_losses[name], adapter_against_the_start(trains[name]), settings) for name in row_losses if name not in of_the_measured}
    took_pre, enough, ran, writes, barred = (checks[key] for key in (
        "the_pretraining_took", "the_goal_set_again_is_large_enough", "the_two_measured_trainings_ran", "each_measured_model_still_writes_proofs",
        "no_training_row_is_of_the_pretrain_half_or_held_out"))
    branch = l4_branch(checks, primary, beside)
    inconclusive = branch["name"] == INCONCLUSIVE
    health, not_to_be_read = _sets_health(who)

    attempts = prepare["attempts_a_goal_problem"]
    of_the_fresh = f"the {fresh['episodes']} attempts of the second sampling" if fresh else "a second sampling, which this run does not have"
    said = lambda name: f"{names[pairs[name][0]]} minus {names[pairs[name][1]]}"      # noqa: E731
    against = lambda name: f"{names[pairs[name][0]]} against {names[pairs[name][1]]}"      # noqa: E731
    assembled = {"what": "the proofs the rounds assembled: how many `with` was trained on and `without` was not, and the lines of every assembled proof once minimised",
                 "assembled_proofs_trained_on": (trains[f"M({last})"].get("rows_by_origin") or {}).get("assembled", 0),
                 "assembled_in_the_rounds": len(assembled_lines),
                 "lines_once_minimised": verified_lines({str(lines): count for lines, count in sorted(Counter(assembled_lines).items())})}
    measured = {
        "primary": {"what": f"G' (the goal problems `pre` does not solve in its {samplings[0]['episodes']}-attempt sampling): successes per attempt over {of_the_fresh}, "
                            "`with` minus `pre`, paired by problem, a 95% bootstrap interval over problems. In its figures `of_the_base` is the side compared with: `pre`",
                    "goal_set_again": len(again), **primary},
        "beside_the_primary": beside,
        "secondary": {
            "by_length_group": {"what": f"successes per attempt over all {attempts} attempts a problem by the length group of each goal problem's shortest published "
                                        "proof, on the problems of 4 lines or more and on all of G; for every pair of models read",
                                "problems": {group: len(ids) for group, ids in sets.items()}, "pairs": {name: {"pair": said(name), **by_length[name]} for name in pairs}},
            "the_three_rungs": {"what": f"fresh pass rate on each held-out rung ({prepare['rung_episodes']} episodes a problem), paired by problem; for every pair read",
                                "problems": {rung: len(rung_ids[rung]) for rung in RUNGS}, "pairs": {name: {"pair": said(name), **rungs[name]} for name in pairs}},
            "goal_problems_solved_by_attempts_alone": {
                "what": "each model's one-shot attempts at a goal problem cut into episodes of 8 in the order drawn; goal problems resolved in at least one, a "
                        "quarter, half (RELIABLY) and nine tenths of their episodes. No Lean here: the counts with assembly come from tools/ladder_goal_assembly.py",
                **{name: {"model": names[name], **alone[name]} for name in who}},
            "goal_problems_solved": {"what": f"goal problems solved at least once in their {attempts} attempts, gained against lost with a two-sided sign test, by "
                                             "length group: `with` against `pre`, and `with` against `without`",
                                     **{name: {"pair": against(name), **solved_on_g[name]} for name in solved_on_g}},
            "the_goal_set_again_by_length_group": {"what": "the primary's quantity by the length group of each problem of G'", **again_by_length},
            "with_minus_without": {"what": f"do assembled proofs teach at this strength: `with` minus its twin, on G' over {of_the_fresh}; on all of G it is under "
                                           "by_length_group", "on_the_goal_set_again": twin_on_again, **assembled},
            "by_round": {"what": "by round of the arm: its picks, the problems an attempt resolved, the problems only assembly resolved and their share of what the "
                                 "attempts left unresolved, the picks' mean pass rate (verified attempts over attempts), the mean reward with k as the challenger "
                                 "reads it (an assembled problem counts as k = 1), the round's training rows by origin, the share of its own rows that are refutations",
                         "rows": table},
            "verified_proof_lines": {"what": "the line counts of the proofs each model verifies, on G over all its attempts and on the three rungs", **lines_verified},
            "opens_with_a_have": {"what": "the share of a model's attempts whose first step is a `have`, on the three rungs and on G", **have},
            "distinct_attempts": {"what": "the share of a model's attempts at one problem and side that are distinct (does pretraining plus rounds narrow the model)",
                                  **distinct}}}

    # ---- the lines: the checks first, then the read in the spec's order
    lines = [
        f"{SAY}: L4, THE LOOP FROM A MODEL {LABEL.upper()}, seed {prepare['seed']}. The arm {prepare['arm']}: {len(prepare['rounds'])} rounds with assembly after each "
        f"batch, its candidates the `loop` half of the pool, round 1 attempted by `pre` ({own['pretraining_rows']:,} published proofs of the `pretrain` half) and every "
        f"model trained FROM `pre`. `with` is M({last}), `pre` trained one more pass on {trains[f'M({last})']['rows']:,} rows of the rounds; `without` is its twin, from "
        f"`pre` on the {trains[f'`{WITHOUT}`']['rows']:,} one-shot rows among them. All measured in one-shot attempts",
        f"{SAY}: CHECK 1, the pretraining took: `pre` solves {took_pre['goal_problems_solved']} of the {took_pre['goal_problems']} goal problems; at least "
        f"{took_pre['minimum']} is asked: {_verdict(took_pre)}",
        f"{SAY}: CHECK 2, G' holds {enough['problems']} problems; at least {enough['minimum']} is asked: {_verdict(enough)}",
        f"{SAY}: CHECK 3, the two measured trainings of the arm ran (the adapter saved is not the start adapter's; the mean loss over the last tenth of its rows is "
        f"not above the first tenth's by more than {settings['standard_errors_allowed']:g} standard errors of their difference): "
        + "; ".join(_ran(name, ran[name]) for name in of_the_measured if name in ran) + f": {_verdict(ran)}",
        f"{SAY}: CHECK 4, each measured model still writes proofs (attempts on the three rungs without an answer; under {settings['maximum_share_without_an_answer']} "
        "is asked): " + "; ".join(f"`{name}` {writes[name]['share']} of {writes[name]['attempts']:,}" for name in measured_here) + f": {_verdict(writes)}",
        f"{SAY}: CHECK 5, no row of any training of the arm is a problem of the `pretrain` half or a held-out problem: {barred['barred_problems']} such problems in "
        f"{sum(barred['rows'].values()):,} rows over {len(barred['rows'])} trainings: {_verdict(barred)}",
        f"{SAY}: for information, deciding nothing, the other {len(others)} trainings read the same way: " + "; ".join(_ran(name, entry) for name, entry in others.items())]
    if prepare.get("stored_runs") is None:
        lines.append(f"{SAY}: L2's stored runs were not read (a smoke run): G was attempted {attempts} times a problem in ONE sampling, so the primary has no fresh "
                     "attempts to be read on, and no stored model stands beside")
    if not inconclusive:
        lines.append(f"{SAY}: PRIMARY. G' ({len(again)} goal problems `pre` does not solve in its {samplings[0]['episodes']}-attempt sampling), successes per attempt over "
                     f"{of_the_fresh}, `with` minus `pre`, paired by problem, 95% bootstrap over problems: {_rate(primary)}"
                     + (f" ({primary['successes']} successes against {primary['successes_of_the_base']} in {primary['attempts_each']:,} attempts each)"
                        if primary.get("mean") is not None else ""))
        lines.append(f"{SAY}: BESIDE IT, the base arm's own gain from its stored rows (the {beside['goal_set_again_of_the_base']} goal problems the base does not solve in "
                     f"its {samplings[0]['episodes']}-attempt sampling, its last model minus the base, over the same {fresh['episodes']} attempts): {_rate(beside)}"
                     if beside is not None else f"{SAY}: BESIDE IT: the base arm's stored rows were not read (its run is not on this box, or this run has one sampling)")
    lines.append(f"{SAY}: {'INCONCLUSIVE' if inconclusive else 'BRANCH: ' + branch['name']}. {branch['reason']}")
    if not inconclusive:
        lines.append(f"{SAY}: SECONDARY, G' by the length of the shortest published proof, `with` minus `pre` over {of_the_fresh}: "
                     + "; ".join(f"{_named(group)} ({again_by_length[group]['problems']}): {_rate(again_by_length[group])}" for group in sets))
        for name in pairs:
            lines.append(f"{SAY}: SECONDARY, all of G over {attempts} attempts a problem by the length of the shortest published proof, {said(name)}: "
                         + "; ".join(f"{_named(group)} ({len(ids)}): {_rate(by_length[name][group])}" for group, ids in sets.items()))
        for name in pairs:
            lines.append(f"{SAY}: SECONDARY, the three rungs ({prepare['rung_episodes']} episodes a problem), {said(name)}: "
                         + "; ".join(f"{rung} {interval_text(rungs[name][rung])}" for rung in RUNGS))
        for group in (ALL, FOUR_PLUS):
            lines.append(f"{SAY}: SECONDARY, goal problems solved at least once / reliably (in at least half of their episodes of 8 one-shot attempts), BY ATTEMPTS "
                         f"ALONE, {_named(group)} ({len(sets[group])}): "
                         + "; ".join(f"{names[name]} {alone[name][group]['solved_at_least_once']} / {alone[name][group]['reliably']}" for name in who)
                         + ". With assembly: not in this report (tools/ladder_goal_assembly.py, after the run)")
        for name in solved_on_g:
            lines.append(f"{SAY}: SECONDARY, goal problems solved at {attempts} attempts, {against(name)}: "
                         + "; ".join(f"{_named(group)}: {solved_on_g[name][group]['resolved_after']} to {solved_on_g[name][group]['resolved_before']}, "
                                     f"{_split(solved_on_g[name][group])}" for group in (ALL, FOUR_PLUS)))
        stats = assembled["lines_once_minimised"]
        lines.append(f"{SAY}: SECONDARY, do assembled proofs teach at this strength: `with` minus `without` on G' over {of_the_fresh}: {_rate(twin_on_again)}; `with` was "
                     f"trained on {assembled['assembled_proofs_trained_on']:,} assembled proofs that `without` was not; the {stats['proofs']:,} proofs the rounds assembled, "
                     f"once minimised: median {stats['median']} lines, mean {stats['mean']}, share with 4 lines or more {stats['share_with_4_lines_or_more']}, with 8 or "
                     f"more {stats['share_with_8_lines_or_more']}, longest {stats['longest']}")
        lines.append(f"{SAY}: SECONDARY, by round (picks; resolved by an attempt; only assembly resolved, and its share of what the attempts left unresolved; the "
                     "picks' mean pass rate; the share of the round's own training rows that are refutations): "
                     + "; ".join(f"round {row['round']}: {row['picks']}; {row['resolved_by_an_attempt']}; {row['only_assembly_resolved']}, "
                                 f"{row['share_of_the_unresolved']}; {row['mean_pass_rate']}; {row['share_of_refutations']}" for row in table))
        lines.append(f"{SAY}: SECONDARY, the lines of the proofs each model verifies on G (proofs, median, mean, share with 4 lines or more, with 8 or more, longest): "
                     + _lines_line(names, lines_verified))
        lines.append(f"{SAY}: SECONDARY, the share of distinct attempts (on the rungs, on G): "
                     + "; ".join(f"{names[name]} {entry[RUNG_PART]['mean_share_distinct']}, {entry[GOAL]['mean_share_distinct']}" for name, entry in distinct.items()))
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}. The step fails; the task queued again "
                     "measures such a set of this run's own again from the kept adapter")
    headline = (f"L4 (the loop from a model {LABEL}) seed {prepare['seed']}: {branch['name']}. "
                + ("" if inconclusive else f"Primary (G', {len(again)} goal problems `pre` does not solve in its first sampling, over {of_the_fresh}, `with` minus "
                                           f"`pre`): {_rate(primary)}; the base arm's own gain beside it: {_rate(beside) if beside is not None else 'not read'}. ")
                + f"Checks: the pretraining took {_verdict(took_pre)}, G' {_verdict(enough)} ({enough['problems']}), the two measured trainings ran {_verdict(ran)}, still write "
                  f"proofs {_verdict(writes)}, no barred row {_verdict(barred)}. " + branch["reason"]
                + (f" NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, L4: the loop from a model pretrained on published proofs", "stage": L4, "label": LABEL, "headline": headline,
        "lines": lines, "ok": not not_to_be_read, "seed": prepare["seed"], "arm": prepare["arm"], "rounds": prepare["rounds"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in trains.values())
                                or any(models[name].get("stand_in_engine") for name in (WITH, WITHOUT))),
        "branch": branch, "inconclusive": inconclusive,
        "can_this_run_see_a_win": {"what": "the checks the spec reads before the branch; when any fails the run is INCONCLUSIVE and nothing else is said", **checks},
        "the_other_trainings": {"what": "the trainings whose models are not measured (the rounds before the last), read as the third check reads the two that "
                                        "are: for information, they decide nothing", **others},
        **({"primary": None, "beside_the_primary": None, "secondary": None,
            "measured_and_not_read": {"why": "a check failed: the run is INCONCLUSIVE, and these numbers say nothing about what the loop adds on top of pretraining. "
                                             "They are kept for whoever repairs the run", **measured}} if inconclusive else measured),
        "models": {BASE: "the base model, from stored attempts: nothing here measured it again",
                   PRE: f"`pre`: {LABEL} ({own['pretraining_rows']:,} proofs of the pool's `pretrain` half), measured by its own stage: its rows are that run's",
                   WITH: f"`with`: M({last}), the arm's last model: `pre` trained one more pass on the rounds' proofs",
                   WITHOUT: f"`without`: the twin of M({last}), from `pre` on its one-shot rows alone", **({LOOP: names[LOOP]} if LOOP in who else {})},
        "trainings": {name: {key: value for key, value in entry.items() if key != "checkpoints"} for name, entry in trains.items()},
        "the_pretraining": {key: own.get(key) for key in ("pretraining_run", "start_adapter", "pretraining_rows", "pretraining_file_sha256", "goal_set_again")},
        "the_arm": {key: arm_prepare.get(key) for key in ("arm", "target_rate", "rounds", "problems_a_round", "batches", "solvers", "candidates",
                                                          "candidates_of_the_whole_pool", "start", "start_adapter", "h0", "h0_rows", "sampling_seeds", "data")},
        "heldout": {"goal_set": len(goal_ids), "goal_set_again": len(again), "goal_set_by_length_group": {group: len(ids) for group, ids in sets.items()},
                    "rungs": {rung: len(rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "stored_runs", "loop_arm",
                                                "stored_measurements", "contradicted_side_setting")},
        "attempts": health, "not_to_be_read": not_to_be_read,
        "not_measured": ["the base, `pre` and the stored three-round model (their rows are the stored ones of the runs named under sizes.stored_runs and the_pretraining)",
                         "the base arm (its last model's rows are its own run's, read when that run is on the box)",
                         "the models of the rounds before the last: only the last round's model and its twin are measured",
                         "anything with assembly on the goal set: the counts by episode with assembly are made after the run, Lean only (tools/ladder_goal_assembly.py)",
                         "the stop rule and the equal-compute control of the L2 stage: this arm has neither"],
    }
