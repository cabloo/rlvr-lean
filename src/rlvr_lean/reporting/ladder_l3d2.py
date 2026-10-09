"""L3d Step 2's report: the loop with assembly in the round, `with` against its twin `without`. Spec:
docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches", "Step 2, made exact before it is built".
Pure: rows in, a report out. Its pieces are the ceiling report's and Step 1's, read on this stage's two models.

One-shot attempts only. Every comparison is over the same problems and the same number of attempts on each, paired by
problem, with a 95% bootstrap interval over problems.

  the checks   the four "can this run see a win" checks FIRST, each with its number and PASS or FAIL. Any FAIL makes
               the report INCONCLUSIVE, and it then says nothing else: what was measured is kept under
               `measured_and_not_read`
  primary      the goal problems whose shortest published proof is 4 lines or more: successes per attempt, `with`
               minus `without`
  the branch   in the spec's words; beside an interval that holds zero, the note from the ceiling's fixed figure
  secondary    the same on all of G and by length group; the three rungs; goal problems solved at least once and
               reliably BY ATTEMPTS ALONE for `with`, `without`, the stored three-round model and the base; the table
               by round; the lines of the proofs each model verifies; the share of attempts that open with a `have`

The counts WITH ASSEMBLY are not here: `tools/ladder_goal_assembly.py` makes them after the run, Lean only. THE LENGTH
OF A HELD-OUT PROBLEM'S PUBLISHED PROOF is read here and nowhere before.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.ceiling import interval_text
from rlvr_lean.domain.ladder_round.l3d import ARMS, WITH, WITHOUT
from rlvr_lean.domain.ladder_round.l3d2 import INCONCLUSIVE, L3D2, can_this_run_see_a_win, l3d2_branch, round_table
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids, paired_change
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import ALL, FOUR_PLUS, LOOP, RUNG_PART, _named, _rate, attempts_on_g, goal_sets, lean_did_not_answer, per_attempt, solved, verified_lines
from rlvr_lean.reporting.ladder_l2 import _health, _split
from rlvr_lean.reporting.ladder_l3d1 import BASE, by_attempts_alone, model_names

PRIMARY = "with_minus_without"
PAIRS = {PRIMARY: (WITH, WITHOUT), "with_minus_base": (WITH, BASE), "without_minus_base": (WITHOUT, BASE), "loop_minus_base": (LOOP, BASE)}


def build_l3d2_report(prepare: Mapping, arm_prepare: Mapping, trains: Mapping[str, Mapping], losses: Mapping[str, Mapping], orders: Mapping[str, Sequence[Mapping]],
                      rounds: Mapping[int, Mapping], groups: Sequence[Mapping], lengths: Mapping[str, Mapping], base: Mapping, models: Mapping[str, Mapping],
                      settings: Mapping, target_rate: float, evaluation: Mapping) -> dict:
    """`prepare`: this stage's prepare summary; `arm_prepare`: the arm's. `trains` and `losses`: by model, its
    training step's summary and its stored loss file (`row_losses`, `rows_trained`). `orders`: by model, the rows it
    was to be trained on (`id`, `origin`), in order. `rounds`: by round, its `summary`, `results` and `examples`.
    `groups`, `lengths`, `base` and `models` (`without`, `with`, and `loop` when a stored three-round model was
    read): as Step 1's report takes them. `settings`: `ladder_loop.l3d.step_2`. `target_rate`: the arm's."""
    resamples, seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    goal_ids, rung_ids = group_ids(groups, GOAL), {name: group_ids(groups, name) for name in RUNGS}
    sets = goal_sets(goal_ids, lengths)
    who = {BASE: base, **models}
    names = model_names(prepare, models)
    pairs = {name: pair for name, pair in PAIRS.items() if pair[0] in who and pair[1] in who}
    on_g = {name: attempts_on_g(entry[GOAL]) for name, entry in who.items()}
    by_length = {name: {group: per_attempt(on_g[model], on_g[other], ids, resamples, seed) for group, ids in sets.items()} for name, (model, other) in pairs.items()}
    rungs = {name: {rung: paired_change(who[model][RUNG_PART], who[other][RUNG_PART], rung_ids[rung], resamples, seed) for rung in RUNGS}
             for name, (model, other) in pairs.items()}
    solved_on_g = {group: solved(on_g[WITH], on_g[WITHOUT], ids) for group, ids in sets.items()}
    alone = by_attempts_alone(who, sets)
    lines_verified = {name: {part: verified_lines(entry["verified_proof_lines"][part]) for part in (GOAL, RUNG_PART)} for name, entry in who.items()}
    have = {name: entry["opens_with_a_have"] for name, entry in who.items()}
    table = round_table({number: {"results": entry["results"], "examples": entry["examples"], "assembly": entry["summary"].get("assembly")}
                         for number, entry in rounds.items()}, target_rate)

    # ---- the checks; then the branch
    checks = can_this_run_see_a_win(orders, {arm: losses[arm]["rows_trained"] for arm in ARMS}, {arm: losses[arm]["row_losses"] for arm in ARMS},
                                    {arm: models[arm][RUNG_PART] for arm in ARMS}, settings, prepare["minimum_assembled"])
    enough, took, writes, record = (checks[key] for key in ("with_was_trained_on_enough_assembled_proofs", "both_trainings_took",
                                                             "both_models_still_write_proofs", "every_assembled_row_is_in_withs_record_once_and_none_in_withouts"))
    primary = by_length[PRIMARY][FOUR_PLUS]
    branch = l3d2_branch(checks, primary, enough["assembled_proofs"], settings["gain_of_a_published_proof"])
    inconclusive = branch["name"] == INCONCLUSIVE

    # ---- whether Lean answered (the health of each set: not the third check, which is about the models)
    measured_sets = {}
    for name, entry in who.items():
        measured_sets[f"rungs_{name}"] = entry[RUNG_PART]
        measured_sets.update({f"goal_{name}_{index}": rows for index, rows in enumerate(entry[GOAL], start=1)})
    health = {name: _health(rows) for name, rows in measured_sets.items()}
    not_to_be_read = sorted(name for name, rows in measured_sets.items() if lean_did_not_answer(rows))

    attempts, last = prepare["attempts_a_goal_problem"], prepare["rounds"][-1]
    said = lambda name: f"{names[pairs[name][0]]} minus {names[pairs[name][1]]}"      # noqa: E731
    measured = {
        "primary": {"what": f"the goal problems whose shortest published proof is 4 lines or more: successes per attempt over {attempts} one-shot attempts a "
                            "problem, `with` minus `without`, paired by problem, a 95% bootstrap interval over problems", **primary},
        "secondary": {
            "by_length_group": {"what": "the primary's quantity by the length group of each goal problem's shortest published proof, on the problems of 4 lines or "
                                        "more and on all of G; for every pair of models read", "problems": {group: len(ids) for group, ids in sets.items()},
                                "pairs": {name: {"pair": said(name), **by_length[name]} for name in pairs}},
            "the_three_rungs": {"what": f"fresh pass rate on each held-out rung ({prepare['rung_episodes']} episodes a problem), paired by problem; for every pair read",
                                "problems": {rung: len(rung_ids[rung]) for rung in RUNGS}, "pairs": {name: {"pair": said(name), **rungs[name]} for name in pairs}},
            "goal_problems_solved_by_attempts_alone": {
                "what": "each model's one-shot attempts at a goal problem cut into episodes of 8 in the order drawn; goal problems resolved in at least one, a "
                        "quarter, half (RELIABLY) and nine tenths of their episodes. No Lean here: the counts with assembly come from tools/ladder_goal_assembly.py",
                **{name: {"model": names[name], **alone[name]} for name in who}},
            "goal_problems_solved_with_against_without": {"what": f"goal problems solved at least once in their {attempts} attempts: gained = by `with` and not by "
                                                                  "`without`, lost = the reverse; two-sided sign test; by length group", **solved_on_g},
            "by_round": {"what": "by round of the arm: its picks, the problems an attempt resolved, the problems only assembly resolved and their share of what the "
                                 "attempts left unresolved, the picks' mean pass rate (verified attempts over attempts), the mean reward with k as the challenger "
                                 "reads it (an assembled problem counts as k = 1), the round's training rows by origin, the share of its own rows that are refutations",
                         "rows": table},
            "verified_proof_lines": {"what": "the line counts of the proofs each model verifies, on G over all its attempts and on the three rungs", **lines_verified},
            "opens_with_a_have": {"what": "the share of a model's attempts whose first step is a `have`, on the three rungs and on G", **have},
            "distinct_attempts": {"what": "the share of a model's attempts at one problem and side that are distinct", **{name: entry["distinct_attempts"] for name, entry in who.items()}}}}

    # ---- the lines: the checks first, then the read in the spec's order
    verdict = lambda check: "PASS" if check["passes"] else "FAIL"      # noqa: E731
    lines = [
        f"{L3D2}: L3D STEP 2, THE LOOP WITH ASSEMBLY IN THE ROUND, seed {prepare['seed']}. The arm {prepare['arm']}: {len(prepare['rounds'])} rounds with assembly after "
        f"each batch. `with` is M({last}), trained on {trains[WITH]['rows']:,} rows; `without` is its twin, trained from the base on the {trains[WITHOUT]['rows']:,} "
        "one-shot rows among them, in the same order with the others left out. Both measured in one-shot attempts",
        f"{L3D2}: CHECK 1, `with` was trained on enough assembled proofs that `without` was not: {enough['assembled_proofs']:,} ({enough['from_the_rounds']:,} of the "
        f"rounds, {enough['from_h0']:,} of H0); at least {enough['minimum']} is asked: {verdict(enough)}",
        f"{L3D2}: CHECK 2, both trainings took (the mean training loss over the last tenth of the rows against the first tenth): "
        + "; ".join(f"`{arm}` {took[arm]['last']} against {took[arm]['first']} over {took[arm]['rows_compared']} rows" for arm in ARMS) + f": {verdict(took)}",
        f"{L3D2}: CHECK 3, both models still write proofs (attempts on the three rungs without an answer; under {settings['maximum_share_without_an_answer']} is asked): "
        + "; ".join(f"`{arm}` {writes[arm]['share']} of {writes[arm]['attempts']:,}" for arm in ARMS) + f": {verdict(writes)}",
        f"{L3D2}: CHECK 4, every assembled row stands once in the record of what `with` was trained on and none in `without`'s: {record['counted_once_in_with']:,} of "
        f"{record['assembled_rows']:,} once in `with`'s {record['rows_trained'][WITH]:,} rows, {record['counted_in_without']} in `without`'s "
        f"{record['rows_trained'][WITHOUT]:,}; each record is the prepared order: {'yes' if all(record['as_prepared'].values()) else 'NO'}: {verdict(record)}"]
    if prepare.get("stored_runs") is None:
        lines.append(f"{L3D2}: L2's stored runs were not read (a smoke run): G was attempted {attempts} times a problem and no three-round model stands beside")
    if not inconclusive:
        lines.append(f"{L3D2}: PRIMARY. The goal problems whose shortest published proof is 4 lines or more ({primary['problems']} problems), successes per attempt "
                     f"over {attempts} one-shot attempts a problem, `with` minus `without`, paired by problem, 95% bootstrap over problems: {_rate(primary)}"
                     + (f" ({primary['successes']} successes against {primary['successes_of_the_base']} in {primary['attempts_each']:,} attempts each)"
                        if primary.get("mean") is not None else ""))
    lines.append(f"{L3D2}: {'INCONCLUSIVE' if inconclusive else 'BRANCH: ' + branch['name']}. {branch['reason']}")
    if not inconclusive:
        for name in pairs:
            lines.append(f"{L3D2}: SECONDARY, the same by the length of the shortest published proof, {said(name)}: "
                         + "; ".join(f"{_named(group)} ({len(ids)}): {_rate(by_length[name][group])}" for group, ids in sets.items()))
        lines.append(f"{L3D2}: SECONDARY, the three rungs ({prepare['rung_episodes']} episodes a problem), {said(PRIMARY)}: "
                     + "; ".join(f"{rung} {interval_text(rungs[PRIMARY][rung])}" for rung in RUNGS))
        for group in (ALL, FOUR_PLUS):
            lines.append(f"{L3D2}: SECONDARY, goal problems solved at least once / reliably (in at least half of their episodes of 8 one-shot attempts), BY "
                         f"ATTEMPTS ALONE, {_named(group)} ({len(sets[group])}): "
                         + "; ".join(f"{names[name]} {alone[name][group]['solved_at_least_once']} / {alone[name][group]['reliably']}" for name in who)
                         + ". With assembly: not in this report (tools/ladder_goal_assembly.py, after the run)")
        lines.append(f"{L3D2}: SECONDARY, goal problems solved at {attempts} attempts, `with` against `without`: "
                     + "; ".join(f"{_named(group)}: {solved_on_g[group]['resolved_after']} to {solved_on_g[group]['resolved_before']}, {_split(solved_on_g[group])}"
                                 for group in (ALL, FOUR_PLUS)))
        lines.append(f"{L3D2}: SECONDARY, by round (picks; resolved by an attempt; only assembly resolved, and its share of what the attempts left unresolved; the "
                     "picks' mean pass rate; the share of the round's own training rows that are refutations): "
                     + "; ".join(f"round {row['round']}: {row['picks']}; {row['resolved_by_an_attempt']}; {row['only_assembly_resolved']}, "
                                 f"{row['share_of_the_unresolved']}; {row['mean_pass_rate']}; {row['share_of_refutations']}" for row in table))
        lines.append(f"{L3D2}: SECONDARY, the lines of the proofs each model verifies on G (proofs, median, mean, share with 4 lines or more, with 8 or more, longest): "
                     + "; ".join(f"{names[name]} {entry[GOAL]['proofs']}, {entry[GOAL]['median']}, {entry[GOAL]['mean']}, {entry[GOAL]['share_with_4_lines_or_more']}, "
                                 f"{entry[GOAL]['share_with_8_lines_or_more']}, {entry[GOAL]['longest']}" for name, entry in lines_verified.items()))
        lines.append(f"{L3D2}: SECONDARY, the share of attempts whose first step is a `have` (on the rungs, on G): "
                     + "; ".join(f"{names[name]} {entry[RUNG_PART]['share']}, {entry[GOAL]['share']}" for name, entry in have.items()))
    if not_to_be_read:
        lines.append(f"{L3D2}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}. The step fails; the task queued again "
                     "measures such a set of this run's own again from the kept adapter")
    headline = (f"L3d Step 2 (the loop with assembly in the round, `with` against its twin) seed {prepare['seed']}: {branch['name']}. "
                + ("" if inconclusive else f"Primary (goal problems with a published proof of 4 lines or more, {primary['problems']} problems, {attempts} attempts each, "
                                           f"`with` minus `without`, {enough['assembled_proofs']:,} assembled proofs between them): {_rate(primary)}. ")
                + f"Checks: assembled proofs {verdict(enough)} ({enough['assembled_proofs']} of at least {enough['minimum']}), both trainings took {verdict(took)}, both "
                  f"still write proofs {verdict(writes)}, the records {verdict(record)}. " + branch["reason"]
                + (f" NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, L3d: train on what the episode reaches (Step 2)", "stage": L3D2, "headline": headline, "lines": lines,
        "ok": not not_to_be_read, "seed": prepare["seed"], "arm": prepare["arm"], "rounds": prepare["rounds"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(trains[arm].get("stand_in_engine") or models[arm].get("stand_in_engine") for arm in ARMS)),
        "branch": branch, "inconclusive": inconclusive,
        "can_this_run_see_a_win": {"what": "the four checks the spec reads before the branch; when any fails the run is INCONCLUSIVE and nothing else is said", **checks},
        **({"primary": None, "secondary": None,
            "measured_and_not_read": {"why": "a check failed: the run is INCONCLUSIVE, and these numbers say nothing about what assembled proofs teach. They are "
                                             "kept for whoever repairs the run", **measured}} if inconclusive else measured),
        "models": {BASE: "the base model, from stored attempts: nothing here measured it again", WITH: f"`with`: M({last}), the arm's last model",
                   WITHOUT: f"`without`: the twin of M({last}), trained from the base on its one-shot rows alone", **({LOOP: names[LOOP]} if LOOP in models else {})},
        "trainings": {arm: {key: value for key, value in trains[arm].items() if key != "checkpoints"} for arm in ARMS},
        "the_arm": {key: arm_prepare.get(key) for key in ("arm", "target_rate", "rounds", "problems_a_round", "batches", "solvers", "h0_file", "h0_file_sha256", "h0_rows",
                                                          "sampling_seeds")},
        "heldout": {"goal_set": len(goal_ids), "goal_set_by_length_group": {group: len(ids) for group, ids in sets.items()},
                    "rungs": {rung: len(rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "stored_runs", "loop_arm",
                                                "stored_measurements", "contradicted_side_setting")},
        "attempts": health, "not_to_be_read": not_to_be_read,
        "not_measured": ["the base and the stored three-round model (their rows are the stored ones of the runs named under sizes.stored_runs)",
                         "the models of the rounds before the last: only the last round's model and its twin are measured",
                         "anything with assembly on the goal set: the counts by episode with assembly are made after the run, Lean only (tools/ladder_goal_assembly.py)",
                         "the stop rule and the equal-compute control of the L2 stage: this arm has neither"],
    }
