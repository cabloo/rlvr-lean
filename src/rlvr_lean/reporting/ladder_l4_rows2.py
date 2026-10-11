"""L4t2's report: two more one-change checks, on what L4t left open. Spec: docs/spec/ladder-loop.spec.md, "L4t2: two
more one-change checks, on what L4t left open" ("Three reads of a model M against `pre_r64`", "Branches, fixed before the
run"). Pure: rows in, a report out. Its pieces are L4t's report's own (`reporting/ladder_l4_rows.py`: the world a model
is read on, the four checks of a trained model, its secondary reads) and the rules of `domain/ladder_round/l4_rows2.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. Every read is against the START MODEL (`pre_r64`), whose rows are its own run's.

  each trained model (`rehearse_only`, then `reward_rehearse`)
    the checks       L4t's four, FIRST, each PASS or FAIL: its training ran; it still writes proofs; Lean answered; no
                     row it was trained on is barred. Any FAIL: THAT MODEL is INCONCLUSIVE, none of its reads is made
                     (what was measured is kept under `measured_and_not_read`), and what stands on it is not read
  the three reads    ADDS, NOT NARROWER, HOLDS THE LONG PROOFS (`l4_rows2.three_reads`): of each of the two trained
                     models and, FROM THE ROWS L4t's RUN STORED, of `old_rule`, `reward_rows` and `rehearse`. A stored
                     model L4t's own report read as INCONCLUSIVE is not read here either
  Question 1         on G' over the fresh attempts, `rehearse` minus `rehearse_only` (`l4_rows2.question_1`)
  Question 2         the three reads of `reward_rehearse` (`l4_rows2.question_2`)
  the arm's rule     ONE line: the rule the spec's "The arm's rule after L4t2" names, with the numbers it used
                     (`l4_rows2.the_arms_rule`)
  secondary          L4t's, of each trained model BY ITSELF against the start model: G' over the fresh attempts; by
                     proof length; solved by attempts alone and reliably; the three rungs; the share of distinct
                     attempts; the lines of verified proofs; the goal problems nothing stored had solved, when their ids
                     were GIVEN; the rows it was trained on. L4t's three stored models stand beside in each table

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples such a set of this run's own again from the kept adapter. THE LENGTH OF A HELD-OUT PROBLEM'S PUBLISHED PROOF is
read here and nowhere before.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE
from rlvr_lean.domain.ladder_round.l4_rows import REHEARSE
from rlvr_lean.domain.ladder_round.l4_rows2 import MODELS, REHEARSE_ONLY, REWARD_REHEARSE, STORED, question_1, question_2, the_arms_rule, three_reads
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import FOUR_PLUS, goal_sets, per_attempt, solved
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY
from rlvr_lean.reporting.ladder_l4_rows import check_lines, secondary_lines, secondary_reads, the_four_checks, world_of

L4 = "l4"
QUESTION = "two more one-change checks, on what L4t left open"
SPEC = "docs/spec/ladder-loop.spec.md, L4t2: two more one-change checks, on what L4t left open"
IS = {REHEARSE_ONLY: "the rehearsal rows of L4t's `rehearse` set and nothing else, in that set's order: a second pass over published proofs the start model was "
                     "pretrained on",
      REWARD_REHEARSE: "L4t's `reward_rows` set and as many rehearsal rows as it has one-shot rows, the first of the hash order L4t's rehearsal draw uses"}
CANNOT_SAY = [
    "The rows are the rank-16 rounds': the arm makes its own, at the start model's frontier",
    "One seed",
    "A choice between two rules by reads on the same goal set is exploratory"]


def build_rows2_report(prepare: Mapping, trains: Mapping[str, Mapping], losses: Mapping[str, Sequence[float]], trained: Mapping[str, Sequence[Mapping]],
                       groups: Sequence[Mapping], lengths: Mapping[str, Mapping], stored: Mapping[str, Mapping], models: Mapping[str, Mapping],
                       again: Sequence[str], settings: Mapping, evaluation: Mapping, never: Sequence[str] | None = None) -> dict:
    """`prepare`: the prepare step's summary (`start`; `l4t_read`: what L4t's own report said of its three models).
    `trains`, `losses`, `trained`: by trained model, its training step's summary, its rows' losses in the order
    trained, and the rows its training loop recorded. `groups`: the held-out groups. `lengths`: by problem, its
    shortest published proof's length group. `stored`: the start model and L4t's three (`old_rule`, `reward_rows`,
    `rehearse`), each from the rows its own run stored; `models`: the two trained here; each `rungs`, `goal` (one list
    of per-problem rows for each sampling of G) and what it wrote. `again`: the start model's G'. `settings`:
    `ladder_loop.l4`. `never`: the ids of the goal problems nothing stored had solved, as GIVEN in the run directory,
    or None when no file was given."""
    start = prepare["start"]
    rank = (prepare.get("start_recipe") or {}).get("rank")
    who = {start: stored[start], **{name: stored[name] for name in STORED}, **{name: models[name] for name in MODELS}}
    world = world_of(prepare, who, groups, lengths, again, settings, evaluation, never)
    goal_ids, sets, on_g, attempts, fresh = world.goal_ids, world.sets, world.on_g, world.attempts, world.fresh
    beside = (start, *STORED)                                   # the models named beside one trained model in its secondary reads
    said_of_l4t = (prepare.get("l4t_read") or {}).get("failed_checks") or {}

    # ---- each trained model: L4t's four checks first; a stored model is held to what L4t's own report read of it
    checks_of, sets_read = {}, {}
    for name in MODELS:
        sets_read[name] = [*world.sets_of[name], *world.sets_of[start]]       # what this model's reads stand on: its own sets, and the start model's
        checks_of[name] = the_four_checks(world, name, trains[name], losses[name], trained[name], sets_read[name], f"the stored ones of `{start}`")
    failed_of = {**{name: list(said_of_l4t.get(name) or []) for name in STORED},
                 **{name: [check for check, entry in checks_of[name].items() if not entry["passes"]] for name in MODELS}}

    # ---- the three reads of every model against the start model, and the branches
    reads = {name: three_reads(world.on_again(name, start), solved(on_g[name], on_g[start], goal_ids),
                               per_attempt(on_g[name], on_g[start], sets[FOUR_PLUS], world.resamples, world.seed), start, failed_of[name])
             for name in (*MODELS, *STORED)}
    difference = world.on_again(REHEARSE, REHEARSE_ONLY)
    first = question_1(difference, reads[REHEARSE_ONLY], reads[REHEARSE])
    second = question_2(reads[REWARD_REHEARSE])
    rule = the_arms_rule(first, reads[REWARD_REHEARSE], reads[REHEARSE])

    # ---- each trained model, by itself: its checks, its three reads, and L4t's secondary reads against the start model
    read, check_lines_of, secondary_lines_of = {}, {}, {}
    for name in MODELS:
        inconclusive = bool(failed_of[name])
        measured = {"secondary": secondary_reads(world, name, start, beside, STORED, trained[name])}
        read[name] = {"model": f"`{name}`: from `{start}`, one pass over {trains[name]['rows']:,} rows: {IS[name]}", "inconclusive": inconclusive,
                      "can_this_model_be_read": {"what": "the four checks the spec reads first; when any fails THIS MODEL is INCONCLUSIVE, by itself", **checks_of[name]},
                      "failed_checks": failed_of[name], "the_three_reads": reads[name],
                      **({"secondary": None, "measured_and_not_read": {"why": "a check failed: this model is INCONCLUSIVE, and these numbers say nothing about its training "
                                                                              "rule. They are kept for whoever repairs the run", **measured}} if inconclusive else measured),
                      "training": {key: value for key, value in trains[name].items() if key != "checkpoints"}}
        of = f"{SAY}: `{name}`"
        check_lines_of[name] = check_lines(world, of, name, IS[name], trains[name], checks_of[name], sets_read[name])
        secondary_lines_of[name] = [] if inconclusive else secondary_lines(world, of, name, start, beside, STORED, measured["secondary"])

    def of_whose_rows(name: str) -> str:
        return "measured by this run" if name in MODELS else "from the rows L4t's run stored"

    sizes = prepare["training_sets"]
    of_the_fresh = f"the {fresh['episodes']} fresh attempts" if fresh else "fresh attempts, which this run does not have"
    lines = [
        f"{SAY}: L4t2, {QUESTION.upper()}, seed {prepare['seed']}. NO NEW ROUND, and no new draw of the rounds' rows: from the sets and the measured rows L4t's run "
        f"stored ({prepare['l4t_run']}) and from `{start}`. EVERY MODEL IS AT RANK {rank}, the start adapter's, trained FROM `{start}`, one pass, the arm's recipe, in the "
        f"content-hash order, and measured as L4t's models were with the sampling seeds `{start}`'s run recorded. `{REHEARSE_ONLY}`: the {sizes[REHEARSE_ONLY]['rows']:,} "
        f"rehearsal rows of L4t's `{REHEARSE}` set and nothing else (published proofs `{start}` has seen once: a second pass). `{REWARD_REHEARSE}`: "
        f"{sizes[REWARD_REHEARSE]['rows']:,} rows, L4t's `reward_rows` set ({sizes[REWARD_REHEARSE]['rows_of_reward_rows']:,}) and as many rehearsal rows as it has "
        f"one-shot rows ({sizes[REWARD_REHEARSE]['rehearsal_rows']:,}: the first of the hash order `{REHEARSE}`'s were drawn in). Every read is against `{start}`: THE THREE "
        f"READS are ADDS (G', {len(again)} goal problems `{start}` does not solve in its first sampling, successes per attempt over {of_the_fresh}: the interval above "
        f"zero), NOT NARROWER (the goal problems solved in the {attempts} attempts: not lost more than gained with the sign test under 0.05) and HOLDS THE LONG PROOFS "
        f"(the {len(sets[FOUR_PLUS])} goal problems of 4 lines or more, successes per attempt over the {attempts} attempts: the interval not wholly below zero)"]
    if not fresh:
        lines.append(f"{SAY}: the stored models have ONE sampling of G here (a smoke run): G was attempted {attempts} times a problem, and G' has no fresh attempts to be "
                     "read on: ADDS and Question 1 are not read")
    for name in MODELS:
        lines += check_lines_of[name]
    lines += [f"{SAY}: `{name}`, THE THREE READS against `{start}` ({of_whose_rows(name)}): {reads[name]['said']}" for name in (*MODELS, *STORED)]
    lines += [
        f"{SAY}: QUESTION 1, `{REHEARSE}` minus `{REHEARSE_ONLY}` on G' over {of_the_fresh}, paired by problem, 95% bootstrap over problems: {first['name']}. {first['reason']}",
        f"{SAY}: QUESTION 2, `{REWARD_REHEARSE}`, the three reads: {second['name']}. {second['reason']}",
        f"{SAY}: THE ARM'S RULE AFTER L4t2: {'`' + rule['rule'] + '`' if rule['rule'] else rule['name']}. {rule['reason']}"]
    for name in MODELS:
        lines += secondary_lines_of[name]
    if world.not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(world.not_to_be_read)}. The step fails; the task queued again "
                     "measures such a set of this run's own again from the kept adapter")

    headline = (f"L4t2 ({QUESTION}; {LABEL}) seed {prepare['seed']}, from `{start}`. Question 1 (`{REHEARSE}` minus `{REHEARSE_ONLY}` on G'): {first['name']}. Question 2 "
                f"(`{REWARD_REHEARSE}`): {second['name']}. The arm's rule after L4t2: {'`' + rule['rule'] + '`' if rule['rule'] else rule['name']}. The three reads against "
                f"`{start}`: " + "; ".join(f"`{name}` " + ("ALL THREE" if reads[name]["all_three"] else INCONCLUSIVE if reads[name]["failed_checks"] else
                                                       f"fails {' and '.join(reads[name]['fails'])}" if reads[name]["fails"] else "not read") for name in (*MODELS, *STORED))
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(world.not_to_be_read)}" if world.not_to_be_read else ""))
    in_again = set(again)
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L4t2", "headline": headline, "lines": lines, "ok": not world.not_to_be_read, "seed": prepare["seed"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in (*trains.values(), *models.values()))),
        "start": start, "rank": rank, "branches": {"question_1": first["name"], "question_2": second["name"], "the_arms_rule": rule["name"]},
        "models": read,
        "the_three_reads": {"what": f"of each model against `{start}`, the arm's own (L4b), fixed before the run: ADDS (G', the fresh attempts of the second sampling: the "
                                    "interval above zero), NOT NARROWER (the goal problems solved in all the attempts: not lost more than gained with the sign test under "
                                    "0.05), HOLDS THE LONG PROOFS (the goal problems of 4 lines or more, per attempt over all the attempts: the interval not wholly below "
                                    f"zero). `{'`, `'.join(MODELS)}`: measured by this run. `{'`, `'.join(STORED)}`: from the rows L4t's run stored",
                            "goal_set_again": len(again), "goal_problems": len(goal_ids), "goal_problems_of_4_lines_or_more": len(sets[FOUR_PLUS]),
                            "attempts_a_goal_problem": attempts, "fresh_attempts": fresh["episodes"] if fresh else None, **reads},
        "question_1": {"what": f"on G' over the fresh attempts, `{REHEARSE}` minus `{REHEARSE_ONLY}`, paired by problem, 95% bootstrap over problems: do the loop's rows "
                               "add beside the rehearsal, or is the gain a second pass over published proofs?", "difference": difference, **first},
        "question_2": {"what": f"the three reads of `{REWARD_REHEARSE}`: the hard problems' rows with rehearsal", **second},
        "the_arms_rule": {"what": "which rule the arm from the start model is run with, by the spec's rule (\"The arm's rule after L4t2\"): no arm with rehearsal rows "
                                  f"when Question 1 reads a second pass; otherwise `{REWARD_REHEARSE}` if it passes the three reads and its gained minus lost against "
                                  f"`{start}` is at least `{REHEARSE}`'s, else `{REHEARSE}`", **rule},
        "what_it_cannot_say": list(CANNOT_SAY),
        "the_stored_models": {start: f"`{start}`: {LABEL}; the model of the check of the adapter's rank, measured by that stage; its rows are that run's",
                              **{name: f"`{name}`: L4t's model, trained from `{start}` and measured by L4t's run; its rows are that run's" for name in STORED},
                              "runs": prepare["stored_models"], "what_l4t_read": prepare.get("l4t_read")},
        "training_sets": prepare["training_sets"],
        "heldout": {"goal_set": len(goal_ids), "goal_set_again": len(again), "goal_set_by_length_group": {group: len(ids) for group, ids in sets.items()},
                    "goal_set_again_by_length_group": {group: len(ids) for group, ids in goal_sets([problem_id for problem_id in goal_ids if problem_id in in_again], lengths).items()},
                    "rungs": {rung: len(world.rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "contradicted_side_setting", "start",
                                                "start_adapter", "l4t_run")},
        "start_recipe": prepare.get("start_recipe"), "attempts": world.health, "not_to_be_read": list(world.not_to_be_read),
        "not_measured": [f"`{start}` and L4t's three models (their rows are the stored ones of the runs named under the_stored_models)",
                         "a new round: the rows are the rank-16 rounds', and the arm makes its own", "the arm's own separation, which the arm carries (one more model trained "
                         "from the start model on its last model's rehearsal rows alone)", "anything at another temperature than the config's", "anything with assembly on the goal set"],
    }
