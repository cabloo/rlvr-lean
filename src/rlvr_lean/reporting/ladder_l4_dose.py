"""L4p's report: how much more pretraining? The dose of a second pass. Spec: docs/spec/ladder-loop.spec.md, "L4p:
how much more pretraining? The dose of a second pass" ("The read, fixed before the run"; "Added the same day ... a
third model, trained and measured FIRST, `matched_1262`"). Pure: rows in, a report out. Its pieces are L4t's report's own
(`reporting/ladder_l4_rows.py`: the world a model is read on, the four checks of a trained model, its secondary reads),
L4t2's three reads (`l4_rows2.three_reads`) and the rules of `domain/ladder_round/l4_dose.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. A LABELLED CHECK OF THE PRETRAINING: no round, and no row of a round.

  each trained model (`matched_1262`, then `second_half`, then `second_full`)
    the checks       L4t's four, FIRST, each PASS or FAIL: its training ran; it still writes proofs; Lean answered; no
                     row it was trained on is barred (every row is of the `pretrain` half and none is held out). Any
                     FAIL: THAT MODEL is INCONCLUSIVE, none of its reads is made, and what stands on it is not read
  the rounds' rows   on G' over the fresh attempts, the stored `reward_rehearse` minus `matched_1262`, paired by problem,
                     95% bootstrap over problems, and its outcome (`l4_dose.matched_branch`); beside it, deciding
                     nothing: the same difference on all of G, the goal problems solved, solved reliably, the never-solved
  the primary        all of G, successes per attempt over all the attempts, `second_full` minus `rehearse_only`, paired
                     by problem, 95% bootstrap over problems, and its branch (`l4_dose.dose_branch`)
  overfitting        on every branch, of each trained model against the start model (`l4_dose.overfitting`): NARROWER,
                     the rungs wholly below zero, the share of distinct attempts on G
  the dose           all of G against the start model for the four doses (`rehearse_only` from the rows L4t2's run
                     stored), and its steps, each dose minus the one before it
  the three reads    L4t2's, of each trained model against the start model (`l4_rows2.three_reads`)
  secondary          L4t's, of each trained model BY ITSELF against the start model: G' over the fresh attempts; by
                     proof length; solved by attempts alone and reliably; the three rungs; the share of distinct
                     attempts; the lines of verified proofs; the goal problems nothing stored had solved, when their ids
                     were GIVEN; the rows it was trained on. The start model and L4t2's stored models stand beside

THE REPORT OF THE JOB THAT TRAINS AND MEASURES `matched_1262` ALONE is this report with the two other models not given:
each is NOT MEASURED YET, the dose's primary and every read that stands on one of them is NOT READ and said as such
(never INCONCLUSIVE), and what stands on `matched_1262` and the stored models alone is read.

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples such a set of this run's own again from the kept adapter. THE LENGTH OF A HELD-OUT PROBLEM'S PUBLISHED PROOF is
read here and nowhere before.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE
from rlvr_lean.domain.ladder_round.l4_dose import DOSES, MATCHED, MODELS, SECOND_FULL, SECOND_HALF, STORED, dose_branch, matched_branch, overfitting
from rlvr_lean.domain.ladder_round.l4_rank import never_solved
from rlvr_lean.domain.ladder_round.l4_rows2 import REHEARSE_ONLY, REWARD_REHEARSE, three_reads
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import ALL, FOUR_PLUS, _rate, goal_sets, per_attempt, solved
from rlvr_lean.reporting.ladder_l2 import _split
from rlvr_lean.reporting.ladder_l4 import LABEL, NOT_MEASURED, SAY
from rlvr_lean.reporting.ladder_l4_rows import check_lines, secondary_lines, secondary_reads, the_four_checks, world_of

L4 = "l4"
QUESTION = "how much more pretraining? The dose of a second pass"
SPEC = "docs/spec/ladder-loop.spec.md, L4p: how much more pretraining? The dose of a second pass"
IS = {MATCHED: "the first rows of the hash order L4t's rehearsal draw uses among the rows the start model was pretrained on, as many as L4t2's `reward_rehearse` set has "
               "published rows: they ARE those rows, and nothing else",
      SECOND_HALF: "the first half of the hash order L4t's rehearsal draw uses among the rows the start model was pretrained on (`rehearse_only`'s rows are its first): half "
                   "of a second pass over published proofs",
      SECOND_FULL: "every row of the pretraining file, the draw of every row (`second_half`'s rows are its first): a whole second pass over published proofs"}
STEPS = tuple(zip(DOSES[1:], DOSES[:-1]))               # the shape of the dose: each dose minus the one before it
# What a trained model's reads stand on beside the start model: the stored models whose rows are on the other side of one of them.
READ_AGAINST = {MATCHED: (REWARD_REHEARSE, REHEARSE_ONLY), SECOND_HALF: (REHEARSE_ONLY,), SECOND_FULL: (REHEARSE_ONLY,)}
CANNOT_SAY = [
    "Whether MORE DATA would do what more passes do: the `loop` half's published proofs are untouched, and using them ends the separation every loop arm stands on (which "
    "needs a decision)",
    "One seed",
    "`reward_rehearse` minus `matched_1262` decides no arm by itself: one pass on the rank-16 rounds' rows, and the arm with its own separation is the test of the cycle"]


def step_name(after: str, before: str) -> str:
    """A step of the dose, as the report's keys name it."""
    return f"{after}_minus_{before}"


def build_dose_report(prepare: Mapping, trains: Mapping[str, Mapping], losses: Mapping[str, Sequence[float]], trained: Mapping[str, Sequence[Mapping]],
                      groups: Sequence[Mapping], lengths: Mapping[str, Mapping], stored: Mapping[str, Mapping], models: Mapping[str, Mapping],
                      again: Sequence[str], settings: Mapping, evaluation: Mapping, never: Sequence[str] | None = None) -> dict:
    """`prepare`: the prepare step's summary (`start`; `l4t2_read`: what L4t2's own report said of its two models).
    `trains`, `losses`, `trained`: by trained model, its training step's summary, its rows' losses in the order
    trained, and the rows its training loop recorded. `groups`: the held-out groups. `lengths`: by problem, its
    shortest published proof's length group. `stored`: the start model and L4t2's two (`rehearse_only`,
    `reward_rehearse`), each from the rows its own run stored; `models`: the models trained here THAT ARE MEASURED
    (`matched_1262` always; the two others are left out by the job that trains and measures `matched_1262` alone);
    each `rungs`, `goal` (one list of per-problem rows for each sampling of G) and what it wrote. `again`: the start
    model's G'. `settings`: `ladder_loop.l4`. `never`: the ids of the goal problems nothing stored had solved, as GIVEN
    in the run directory, or None when no file was given."""
    start = prepare["start"]
    rank = (prepare.get("start_recipe") or {}).get("rank")
    measured_here = tuple(name for name in MODELS if name in models)
    not_yet = tuple(name for name in MODELS if name not in models)
    who = {start: stored[start], **{name: stored[name] for name in STORED}, **{name: models[name] for name in measured_here}}
    world = world_of(prepare, who, groups, lengths, again, settings, evaluation, never)
    goal_ids, sets, on_g, attempts, fresh = world.goal_ids, world.sets, world.on_g, world.attempts, world.fresh

    # ---- each trained model: L4t's four checks first; L4t2's two stored models are held to what its own report read of them
    checks_of, sets_read = {}, {}
    for name in measured_here:
        sets_read[name] = [*world.sets_of[name], *world.sets_of[start], *(key for other in READ_AGAINST[name] for key in world.sets_of[other])]
        checks_of[name] = the_four_checks(world, name, trains[name], losses[name], trained[name], sets_read[name],
                                          f"the stored ones of `{start}` and of " + " and ".join(f"`{other}`" for other in READ_AGAINST[name]))
    said_of_l4t2 = (prepare.get("l4t2_read") or {}).get("failed_checks") or {}
    failed_of = {**{name: list(said_of_l4t2.get(name) or []) for name in STORED},
                 **{name: [check for check, entry in checks_of[name].items() if not entry["passes"]] for name in measured_here}}

    def on_all(name: str, minus: str) -> dict:
        """All of G, successes per attempt over all the attempts, `name` minus `minus`, paired by problem."""
        return per_attempt(on_g[name], on_g[minus], goal_ids, world.resamples, world.seed)

    def unless_unread(names: Sequence[str], figures: Callable[[], Mapping]) -> dict:
        """What stands on `names`, read; or NOT READ: one of them is not measured yet (nothing is computed), or is
        INCONCLUSIVE (what was measured is kept apart)."""
        waiting = [name for name in names if name in not_yet]
        if waiting:
            return {"read": False, "not_measured": waiting, "why": " and ".join(f"`{name}`" for name in waiting) + (" is" if len(waiting) == 1 else " are") + " not measured yet"}
        failed = {name: failed_of[name] for name in names if failed_of[name]}
        if failed:
            return {"read": False, "failed_checks": failed, "measured_and_not_read": dict(figures()),
                    "why": "; ".join(f"`{name}` is {INCONCLUSIVE} ({', '.join(checks)})" for name, checks in failed.items()) + ": not read"}
        return {"read": True, **figures()}

    # ---- the rounds' hard rows beside the same published rows: the stored `reward_rehearse` minus `matched_1262`, on G' over the fresh attempts
    difference = world.on_again(REWARD_REHEARSE, MATCHED)
    hard_rows = matched_branch(failed_of[MATCHED], difference, failed_of[REWARD_REHEARSE])
    hard_rows_unread = hard_rows["name"] == INCONCLUSIVE or "not_read_against" in hard_rows
    unsolved = {name: never_solved(never, on_g[name], on_g[start], goal_ids) for name in (REWARD_REHEARSE, MATCHED)}
    beside_it = {"what": f"beside it, deciding nothing: `{REWARD_REHEARSE}` minus `{MATCHED}` on all of G over the {attempts} attempts; the goal problems each solves at least "
                         f"once (gained = by `{REWARD_REHEARSE}` and not by `{MATCHED}`, lost = the reverse; the two-sided sign test); those each solves reliably (in at least "
                         "half of its episodes of 8 one-shot attempts); and of the goal problems nothing stored had solved, when their ids were given",
                 "on_all_of_g": on_all(REWARD_REHEARSE, MATCHED), "solved": solved(on_g[REWARD_REHEARSE], on_g[MATCHED], goal_ids),
                 "solved_reliably": {name: world.alone[name][ALL]["reliably"] for name in (REWARD_REHEARSE, MATCHED)},
                 "never_solved_before": {"given": unsolved[MATCHED]["given"], "goal_problems": unsolved[MATCHED].get("goal_problems"),
                                         **{name: unsolved[name].get("solved") for name in (REWARD_REHEARSE, MATCHED)}}}

    # ---- the dose's primary and its branch
    primary = on_all(SECOND_FULL, REHEARSE_ONLY) if SECOND_FULL in models else dict(NOT_MEASURED)
    branch = dose_branch(failed_of.get(SECOND_FULL, []), primary, start, failed_of[REHEARSE_ONLY], measured=SECOND_FULL in models)
    inconclusive = branch["name"] == INCONCLUSIVE
    set_aside = inconclusive or "not_read_against" in branch or "not_measured" in branch       # the primary is not read: not measured yet, a failed check, or an INCONCLUSIVE `rehearse_only`

    # ---- each trained model, by itself against the start model: its overfitting reads, L4t2's three reads, L4t's secondary reads
    against_the_start = {name: solved(on_g[name], on_g[start], goal_ids) for name in DOSES if name not in not_yet}
    read, check_lines_of, secondary_lines_of, overfit, reads = {}, {}, {}, {}, {}
    for name in not_yet:
        waiting = {"measured": False, "said": f"NOT READ: `{name}` is not measured yet (the job that trains and measures `{MATCHED}` alone writes this report)"}
        read[name], overfit[name], reads[name] = {"model": f"`{name}`: {IS[name]}", **waiting}, dict(waiting), dict(waiting)
    for name in measured_here:
        unread = bool(failed_of[name])
        beside = (start, *READ_AGAINST[name])                   # the models named beside this one in its secondary reads
        measured = {"secondary": secondary_reads(world, name, start, beside, READ_AGAINST[name], trained[name])}
        overfit[name] = overfitting(against_the_start[name], measured["secondary"]["the_three_rungs"][f"minus_{start}"], who[name]["distinct_attempts"][GOAL]["mean_share_distinct"],
                                    who[start]["distinct_attempts"][GOAL]["mean_share_distinct"], start, failed_of[name])
        reads[name] = three_reads(world.on_again(name, start), against_the_start[name], per_attempt(on_g[name], on_g[start], sets[FOUR_PLUS], world.resamples, world.seed),
                                  start, failed_of[name])
        read[name] = {"model": f"`{name}`: from `{start}`, one pass over {trains[name]['rows']:,} rows: {IS[name]}", "measured": True, "inconclusive": unread,
                      "can_this_model_be_read": {"what": "the four checks the spec reads first; when any fails THIS MODEL is INCONCLUSIVE, by itself", **checks_of[name]},
                      "failed_checks": failed_of[name], "overfitting": overfit[name], "the_three_reads": reads[name],
                      **({"secondary": None, "measured_and_not_read": {"why": "a check failed: this model is INCONCLUSIVE, and these numbers say nothing about a second pass. "
                                                                              "They are kept for whoever repairs the run", **measured}} if unread else measured),
                      "training": {key: value for key, value in trains[name].items() if key != "checkpoints"}}
        of = f"{SAY}: `{name}`"
        check_lines_of[name] = check_lines(world, of, name, IS[name], trains[name], checks_of[name], sets_read[name])
        secondary_lines_of[name] = [] if unread else secondary_lines(world, of, name, start, beside, READ_AGAINST[name], measured["secondary"])

    read, overfit, reads = ({name: of[name] for name in MODELS} for of in (read, overfit, reads))       # in the stage's order of models, whichever are measured

    # ---- the dose: all of G against the start model for the four doses, and its steps (a step that stands on a model not measured yet or INCONCLUSIVE is not read)
    sizes = prepare["training_sets"]
    rows = {REHEARSE_ONLY: prepare["rehearse_only"]["rows"], **{name: sizes[name]["rows"] for name in MODELS}}
    of_the_file = prepare["rows_the_start_was_pretrained_on"]
    share = {name: round(rows[name] / of_the_file, 5) for name in DOSES}
    of_the_doses = {name: {"rows": rows[name], "share_of_a_second_pass": share[name],
                           **unless_unread([name], lambda name=name: {"per_attempt": on_all(name, start), "solved": against_the_start[name]})} for name in DOSES}
    steps = {step_name(after, before): {"rows_more": rows[after] - rows[before], **unless_unread([after, before], lambda after=after, before=before: on_all(after, before))}
             for after, before in STEPS}

    def of_a_dose(name: str) -> str:
        entry = of_the_doses[name]
        if not entry["read"]:
            return f"`{name}` ({rows[name]:,} rows, {share[name]:.0%} of a second pass): NOT READ ({entry['why']})"
        found = entry["solved"]
        return (f"`{name}` ({rows[name]:,} rows, {share[name]:.0%} of a second pass): {_rate(entry['per_attempt'])}; goal problems solved {found['resolved_after']} to "
                f"{found['resolved_before']}, {_split(found)}")

    def of_a_step(after: str, before: str) -> str:
        entry = steps[step_name(after, before)]
        return (f"`{after}` minus `{before}` ({entry['rows_more']:,} rows more): {_rate(entry)}" if entry["read"] else
                f"`{after}` minus `{before}`: NOT READ ({entry['why']})")

    of_the_fresh = f"the {fresh['episodes']} fresh attempts" if fresh else "fresh attempts, which this run does not have"
    published = prepare["reward_rehearse"]
    lines = [
        f"{SAY}: L4p, {QUESTION.upper()}, seed {prepare['seed']}. A LABELLED CHECK OF THE PRETRAINING: no round, and no row of a round. EVERY MODEL IS AT RANK {rank}, the "
        f"start adapter's, trained FROM `{start}`, one pass over its rows, the arm's recipe, in the content-hash order, and measured as L4t's models were with the sampling "
        f"seeds `{start}`'s run recorded. The rows are rows of the {of_the_file:,} published proofs `{start}` was pretrained on, along ONE order, the hash order L4t's "
        f"rehearsal draw uses: `{MATCHED}` its first {rows[MATCHED]:,} (EXACTLY the published rows of L4t2's `{REWARD_REHEARSE}` set, which holds {published['rows']:,} rows: "
        f"its control), `{REHEARSE_ONLY}` its first {rows[REHEARSE_ONLY]:,} (L4t2's model: stored in {prepare['l4t2_run']}, read and never trained again), `{SECOND_HALF}` its "
        f"first {rows[SECOND_HALF]:,}, `{SECOND_FULL}` all {rows[SECOND_FULL]:,}: the dose at " + ", ".join(f"{share[name]:.0%}" for name in DOSES[:-1])
        + f" and {share[SECOND_FULL]:.0%} of a second pass. THE ROUNDS' HARD ROWS are read on G' over {of_the_fresh}, the stored `{REWARD_REHEARSE}` minus `{MATCHED}`. THE "
        f"DOSE'S PRIMARY is on all of G ({len(goal_ids)} problems, the {attempts} attempts), successes per attempt, `{SECOND_FULL}` minus `{REHEARSE_ONLY}`, paired by problem, "
        f"95% bootstrap over problems. OVERFITTING is read on every branch, of each model against `{start}`"]
    if not_yet:
        lines.append(f"{SAY}: THIS IS THE REPORT OF THE JOB THAT TRAINS AND MEASURES `{MATCHED}` ALONE: " + " and ".join(f"`{name}`" for name in not_yet)
                     + " are NOT MEASURED YET. The dose's primary and every read that stands on one of them is NOT READ here (not INCONCLUSIVE: nothing of them failed), and "
                     "the report of the whole stage, which finds this model done and goes on, reads them")
    if not fresh:
        lines.append(f"{SAY}: the stored models have ONE sampling of G here (a smoke run): G was attempted {attempts} times a problem, and G' has no fresh attempts to be "
                     "read on: the rounds' hard rows and ADDS are not read")
    for name in measured_here:
        lines += check_lines_of[name]
    if not hard_rows_unread:
        lines.append(f"{SAY}: THE ROUNDS' HARD ROWS BESIDE THE SAME PUBLISHED ROWS. G' ({len(again)} goal problems `{start}` does not solve in its first sampling), successes per "
                     f"attempt over {of_the_fresh}, `{REWARD_REHEARSE}` (stored: the rounds' hard rows and these {rows[MATCHED]:,} published rows) minus `{MATCHED}` (these "
                     f"published rows alone), paired by problem, 95% bootstrap over problems: {_rate(difference)}")
    lines.append(f"{SAY}: THE ROUNDS' HARD ROWS: {hard_rows['name']}. {hard_rows['reason']}")
    if not hard_rows_unread:
        found, reliably, of_the_never = beside_it["solved"], beside_it["solved_reliably"], beside_it["never_solved_before"]
        lines.append(f"{SAY}: BESIDE IT, deciding nothing. All of G over the {attempts} attempts, `{REWARD_REHEARSE}` minus `{MATCHED}`: {_rate(beside_it['on_all_of_g'])}; goal "
                     f"problems solved {found['resolved_after']} to {found['resolved_before']}, {_split(found)}; solved reliably `{REWARD_REHEARSE}` {reliably[REWARD_REHEARSE]}, "
                     f"`{MATCHED}` {reliably[MATCHED]}; the goal problems nothing stored had solved: "
                     + (f"of the {of_the_never['goal_problems']} listed, `{REWARD_REHEARSE}` {of_the_never[REWARD_REHEARSE]}, `{MATCHED}` {of_the_never[MATCHED]}"
                        if of_the_never["given"] else "not given (no file of their ids is in the run directory)"))
    if not set_aside:
        lines.append(f"{SAY}: PRIMARY. All of G ({primary['problems']} problems), successes per attempt over the {attempts} attempts a problem, `{SECOND_FULL}` minus "
                     f"`{REHEARSE_ONLY}`, paired by problem, 95% bootstrap over problems: {_rate(primary)}"
                     + (f" ({primary['successes']} successes against {primary['successes_of_the_base']} in {primary['attempts_each']:,} attempts each)"
                        if primary.get("mean") is not None else ""))
    lines.append(f"{SAY}: {'INCONCLUSIVE' if inconclusive else 'BRANCH: ' + branch['name']}. {branch['reason']}")
    lines += [f"{SAY}: OVERFITTING, `{name}` against `{start}` (read on every branch): {overfit[name]['said']}" for name in MODELS]
    lines.append(f"{SAY}: THE DOSE, all of G ({len(goal_ids)} problems, {attempts} attempts a problem), successes per attempt, each dose minus `{start}`: "
                 + "; ".join(of_a_dose(name) for name in DOSES) + f". `{REHEARSE_ONLY}` is from the rows L4t2's run stored")
    lines.append(f"{SAY}: THE SHAPE OF THE DOSE, all of G, successes per attempt, each dose minus the one before it: " + "; ".join(of_a_step(after, before) for after, before in STEPS))
    lines += [f"{SAY}: `{name}`, THE THREE READS of L4t2 against `{start}` (ADDS on G' over {of_the_fresh}; NOT NARROWER; HOLDS THE LONG PROOFS): {reads[name]['said']}"
              for name in MODELS]
    for name in measured_here:
        lines += secondary_lines_of[name]
    if world.not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(world.not_to_be_read)}. The step fails; the task queued again "
                     "measures such a set of this run's own again from the kept adapter")

    def overfit_said(name: str) -> str:
        own = overfit[name]
        if name in not_yet:
            return f"`{name}` not measured yet"
        if own["failed_checks"]:
            return f"`{name}` {INCONCLUSIVE}"
        below = own["rungs_wholly_below_zero"]
        return (f"`{name}` " + ("NARROWER" if own["narrower"] else "not narrower") + f" ({own['counts']})"
                + (f", wholly below zero on the rung{'s' if len(below) > 1 else ''} {', '.join(below)}" if below else ", no rung wholly below zero"))

    headline = (f"L4p ({QUESTION.split('?')[0]}; {LABEL}) seed {prepare['seed']}, from `{start}`: {branch['name']}"
                + (f" (THE DOSE'S PRIMARY: {' and '.join(f'`{name}`' for name in not_yet)} not measured yet)" if not_yet else "") + ". "
                + ("" if set_aside else f"Primary (all of G, {primary['problems']} problems, {attempts} attempts each, `{SECOND_FULL}` minus `{REHEARSE_ONLY}`): "
                                        f"{_rate(primary)}. ")
                + f"The rounds' hard rows beside the same published rows (`{REWARD_REHEARSE}` minus `{MATCHED}` on G'): {hard_rows['name']}"
                + ("" if hard_rows_unread or difference.get("mean") is None else f" ({_rate(difference)})")
                + f". Overfitting against `{start}`: " + "; ".join(overfit_said(name) for name in MODELS) + f". {branch['reason']}"
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(world.not_to_be_read)}" if world.not_to_be_read else ""))
    in_again = set(again)
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L4p", "headline": headline, "lines": lines, "ok": not world.not_to_be_read, "seed": prepare["seed"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in (*trains.values(), *models.values()))),
        "start": start, "rank": rank, "measured": list(measured_here), "not_measured_yet": list(not_yet),
        "branches": {"the_dose": branch["name"], "the_rounds_hard_rows": hard_rows["name"]}, "branch": branch, "inconclusive": inconclusive,
        "primary": {"what": f"all of G: successes per attempt over the {attempts} attempts a problem, `{SECOND_FULL}` minus `{REHEARSE_ONLY}`, paired by problem, a 95% "
                            f"bootstrap interval over problems. In its figures `of_the_base` is the side compared with: `{REHEARSE_ONLY}`, from the rows L4t2's run stored",
                    **({"mean": None, **({} if "not_measured" in branch else {"measured_and_not_read": primary})} if set_aside else primary)},
        "the_rounds_hard_rows": {"what": f"the question at the size of one pass: do the model's own proofs of hard problems add anything that the same published rows "
                                         f"do not? On G' over the fresh attempts, `{REWARD_REHEARSE}` (L4t2's model, from the rows its run stored: the rounds' hard rows and "
                                         f"as many published rows) minus `{MATCHED}` (those published rows and nothing else), paired by problem, 95% bootstrap over problems. "
                                         "In its figures `of_the_base` is the side compared with: `matched_1262`. It decides no arm by itself",
                                 "published_rows": rows[MATCHED], "rows_of_reward_rehearse": published["rows"],
                                 "difference": {"mean": None, "measured_and_not_read": difference} if hard_rows_unread else difference, **hard_rows,
                                 "beside": {"what": beside_it["what"], "measured_and_not_read": {key: value for key, value in beside_it.items() if key != "what"}}
                                 if hard_rows_unread else beside_it},
        "models": read,
        "overfitting": {"what": f"read on every branch, of each trained model against `{start}`: NARROWER (lost more goal problems than gained in the {attempts} attempts, "
                                "the sign test under 0.05), each rung whose interval is wholly below zero, the share of distinct attempts on G. No held-out problem is in "
                                "the training text, so a model cannot copy a goal problem's proof: what more passes can cost here is breadth, and that is what is read",
                        **overfit},
        "the_dose": {"what": f"the dose along ONE order, the hash order L4t's rehearsal draw uses among the rows `{start}` was pretrained on: each dose's rows are the first "
                             f"of the next one's. `against_the_start`: all of G, successes per attempt and the goal problems solved, each dose against `{start}`. `steps`: "
                             f"all of G, successes per attempt, each dose minus the one before it. `{REHEARSE_ONLY}` is L4t2's model, from the rows its run stored",
                     "rows_the_start_was_pretrained_on": of_the_file, "rows": rows, "share_of_a_second_pass": share, "against_the_start": of_the_doses, "steps": steps},
        "the_three_reads": {"what": f"L4t2's three reads of each trained model against `{start}`: ADDS (G', the fresh attempts of the second sampling: the interval above "
                                    "zero), NOT NARROWER (the goal problems solved in all the attempts: not lost more than gained with the sign test under 0.05), HOLDS THE "
                                    "LONG PROOFS (the goal problems of 4 lines or more, per attempt over all the attempts: the interval not wholly below zero)",
                            "goal_set_again": len(again), "goal_problems": len(goal_ids), "goal_problems_of_4_lines_or_more": len(sets[FOUR_PLUS]),
                            "attempts_a_goal_problem": attempts, "fresh_attempts": fresh["episodes"] if fresh else None, **reads},
        "what_it_cannot_say": list(CANNOT_SAY),
        "the_stored_models": {start: f"`{start}`: {LABEL}; the model of the check of the adapter's rank, measured by that stage; its rows are that run's",
                              REHEARSE_ONLY: f"`{REHEARSE_ONLY}`: L4t2's model, trained from `{start}` on the first rows of the same hash order and measured by L4t2's run; "
                                             "its rows are that run's, and it is never trained again",
                              REWARD_REHEARSE: f"`{REWARD_REHEARSE}`: L4t2's model, trained from `{start}` on the rounds' hard rows and the published rows `{MATCHED}` is "
                                               "trained on alone, and measured by L4t2's run; its rows are that run's",
                              "runs": prepare["stored_models"], "what_l4t2_read": prepare.get("l4t2_read")},
        "training_sets": prepare["training_sets"],
        "heldout": {"goal_set": len(goal_ids), "goal_set_again": len(again), "goal_set_by_length_group": {group: len(ids) for group, ids in sets.items()},
                    "goal_set_again_by_length_group": {group: len(ids) for group, ids in goal_sets([problem_id for problem_id in goal_ids if problem_id in in_again], lengths).items()},
                    "rungs": {rung: len(world.rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "contradicted_side_setting", "start",
                                                "start_adapter", "l4t2_run")},
        "start_recipe": prepare.get("start_recipe"), "attempts": world.health, "not_to_be_read": list(world.not_to_be_read),
        "not_measured": [f"`{start}`, `{REHEARSE_ONLY}` and `{REWARD_REHEARSE}` (their rows are the stored ones of the runs named under the_stored_models)",
                         *(f"`{name}`: not measured yet (the job that trains and measures `{MATCHED}` alone)" for name in not_yet),
                         "a round, or a row of a round: this is a check of the pretraining", "a third pass, which is specified next only when the primary reads that more helps",
                         "the `loop` half's published proofs, which no model here has seen", "anything at another temperature than the config's",
                         "anything with assembly on the goal set"],
    }
