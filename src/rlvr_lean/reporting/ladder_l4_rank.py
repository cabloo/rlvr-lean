"""L4r's report: is the pretrained model capped by the size of its adapter? Spec: docs/spec/ladder-loop.spec.md,
"L4r: is the pretrained model capped by the size of its adapter?", "The read, fixed before the run". Pure: rows in, a
report out. Its pieces are the ceiling report's and L4's, read on the check's model against `pre`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. Every comparison is the check's model (`pre_r<rank>`) minus `pre` over the same problems and
the same number of attempts on each, paired by problem, with a 95% bootstrap interval over problems.

  the checks   FIRST, each with its number and PASS or FAIL: the training took (the mean loss over the last tenth of its
               rows is below the mean over the first tenth); it still writes proofs (under 5% of its rung attempts
               without an answer); Lean answered (at most 2% of each set read without an answer). Any FAIL makes the
               report INCONCLUSIVE, and it then says nothing else: what was measured is kept under
               `measured_and_not_read`, for whoever repairs the run
  primary      all of G: successes per attempt over all the attempts a problem (93), `pre_r<rank>` minus `pre`
  the branch   in the spec's words
  secondary    the same by proof length; goal problems solved (gained, lost, sign test); solved in at least one episode
               of 8 and reliably; the three rungs; the loss on rows not yet trained on by twentieth of the pass, beside
               `pre`'s; the share of attempts that time out in Lean and the share of distinct attempts; how many of the
               goal problems nothing stored has ever solved it solves, when their ids were GIVEN

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples that set of the check's model again from the kept adapter. THE LENGTH OF A HELD-OUT PROBLEM'S PUBLISHED PROOF is
read here and nowhere before.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.ceiling import interval_text
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, PRE
from rlvr_lean.domain.ladder_round.l4_rank import FALLBACK, TWENTIETHS, by_twentieth, never_solved, rank_branch, training_checks
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids, paired_change
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import ALL, FOUR_PLUS, RUNG_PART, _named, _rate, attempts_on_g, goal_sets, per_attempt, solved
from rlvr_lean.reporting.ladder_l2 import MAXIMUM_SHARE_WITHOUT_AN_ANSWER, _split
from rlvr_lean.reporting.ladder_l3d1 import by_attempts_alone
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY, _sets_health, _verdict

L4 = "l4"
QUESTION = "is the pretrained model capped by the size of its adapter?"
NAMED_IN_A_LINE = 20                # the solved problems a printed line names; the report's file holds every one


def cannot_say(name: str, rank: int, rank_of_pre: int) -> list[str]:
    """What the check cannot say, in the spec's words."""
    return [f"Rank {rank} is not the whole model: a null does not show that full training would not help",
            f"One learning rate and one pass: an adapter {rank / rank_of_pre:g} times the size may want either changed, and this run changes neither",
            f"One seed, a first run by the seed rule; a positive read is confirmed by what is built on it (the arm from `{name}`), not by a second pretraining"]


def _share(count: int, of: int) -> float | None:
    return round(count / of, 5) if of else None


def timed_out(model: Mapping) -> dict:
    """The share of one model's attempts that time out in Lean, on the three rungs and on G (all its samplings)."""
    def of(rows: Sequence[Mapping]) -> dict:
        attempts, timed = sum(row["episodes"] * row["sides"] for row in rows), sum(row["attempts_timed_out"] for row in rows)
        return {"attempts": attempts, "timed_out_in_lean": timed, "share": _share(timed, attempts)}

    return {RUNG_PART: of(model[RUNG_PART]), GOAL: of([row for rows in model[GOAL] for row in rows])}


def _series(values: Sequence[float | None]) -> str:
    return ", ".join("-" if value is None else f"{value:.3f}" for value in values)


def _named_problems(problem_ids: Sequence[str]) -> str:
    """Solved problems as a line names them: every one up to `NAMED_IN_A_LINE`, then how many more (the report's file holds them all)."""
    if not problem_ids:
        return ""
    more = len(problem_ids) - NAMED_IN_A_LINE
    return f" ({', '.join(problem_ids[:NAMED_IN_A_LINE])}" + (f", and {more} more: all are in the report's file)" if more > 0 else ")")


def build_rank_report(prepare: Mapping, train: Mapping, losses: Mapping[str, Sequence[float]], groups: Sequence[Mapping], lengths: Mapping[str, Mapping],
                      pre: Mapping, model: Mapping, settings: Mapping, evaluation: Mapping, never: Sequence[str] | None = None) -> dict:
    """`prepare` and `train`: the summaries of the check's two steps (`prepare["rank_check"]`: the check, its model, the
    one change and what was read of `pre`). `losses`: by model (`pre`, and the check's own name), its rows' losses in
    the order trained, each read before the update of its step. `groups`: the held-out groups. `lengths`: by problem,
    its shortest published proof's length group. `pre` and `model` (the check's): `rungs` (per-problem rows on the
    three rungs), `goal` (one list of per-problem rows for each sampling of G), `distinct_attempts` and
    `episodes_of_8`. `settings`: `ladder_loop.l4`. `never`: the ids of the goal problems nothing stored has ever
    solved, as GIVEN in the run directory, or None when no file was given."""
    resamples, seed = evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"]
    check = prepare["rank_check"]
    name, of_pre = check["model"], check[PRE]
    goal_ids, rung_ids = group_ids(groups, GOAL), {rung: group_ids(groups, rung) for rung in RUNGS}
    sets = goal_sets(goal_ids, lengths)
    who = {PRE: pre, name: model}
    on_g = {key: attempts_on_g(entry[GOAL]) for key, entry in who.items()}

    # ---- the primary: all of G, successes per attempt over every attempt a problem, the check's model minus `pre`, paired by problem
    primary = per_attempt(on_g[name], on_g[PRE], goal_ids, resamples, seed)
    by_length = {group: per_attempt(on_g[name], on_g[PRE], ids, resamples, seed) for group, ids in sets.items() if group != ALL}
    solved_on_g = {group: solved(on_g[name], on_g[PRE], ids) for group, ids in sets.items()}
    alone = by_attempts_alone(who, sets)
    rungs = {rung: paired_change(model[RUNG_PART], pre[RUNG_PART], rung_ids[rung], resamples, seed) for rung in RUNGS}
    loss = {key: by_twentieth(losses[key]) for key in who}
    out_of_time = {key: timed_out(entry) for key, entry in who.items()}
    distinct = {key: entry["distinct_attempts"] for key, entry in who.items()}
    unsolved = never_solved(never, on_g[name], on_g[PRE], goal_ids)

    # ---- the checks, then the branch
    health, not_to_be_read = _sets_health(who)
    checks = training_checks(losses[name], model[RUNG_PART], settings)
    took, writes = checks["the_training_took"], checks["it_still_writes_proofs"]
    answered = checks["lean_answered"] = {
        "what": f"Lean gave a verdict on all but at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of the attempts of each set read: the check's model's own three, which "
                "the task queued again samples again, and `pre`'s stored ones",
        "maximum": MAXIMUM_SHARE_WITHOUT_AN_ANSWER, "share_without_an_answer": {set_name: entry["share_without_an_answer"] for set_name, entry in health.items()},
        "sets_not_answered": not_to_be_read, "passes": not not_to_be_read}
    branch = rank_branch(checks, primary, name, of_pre["rank"])
    inconclusive = branch["name"] == INCONCLUSIVE

    attempts = prepare["attempts_a_goal_problem"]
    saved = train.get("adapter_saved") or {}
    numbers = saved.get("numbers") or check.get("trained_parameters")
    measured = {
        "primary": {"what": f"all of G: successes per attempt over the {attempts} attempts a problem, `{name}` minus `pre`, paired by problem, a 95% bootstrap interval "
                            "over problems. In its figures `of_the_base` is the side compared with: `pre`", **primary},
        "secondary": {
            "by_length_group": {"what": "the primary's quantity by the length group of each goal problem's shortest published proof (1, 2-3, 4-7, 8 or more lines), and "
                                        "on the problems of 4 lines or more", "problems": {group: len(sets[group]) for group in by_length}, **by_length},
            "goal_problems_solved": {"what": f"goal problems solved at least once in their {attempts} attempts: gained = by `{name}` and not by `pre`, lost = the reverse; "
                                             "two-sided sign test; on all of G and by length group", **solved_on_g},
            "goal_problems_solved_by_attempts_alone": {
                "what": "each model's one-shot attempts at a goal problem cut into episodes of 8 in the order drawn; goal problems resolved in at least one, a quarter, "
                        "half (RELIABLY) and nine tenths of their episodes", **alone},
            "the_three_rungs": {"what": f"fresh pass rate on each held-out rung ({prepare['rung_episodes']} episodes a problem), `{name}` minus `pre`, paired by problem",
                                "problems": {rung: len(rung_ids[rung]) for rung in RUNGS}, **rungs},
            "loss_by_twentieth": {"what": f"the mean loss over each of {TWENTIETHS} consecutive parts of the pass, of rows not yet trained on (a row's mean loss per target "
                                          "token, read before the update of the step it was in): the check's model's beside `pre`'s", **loss},
            "timed_out_in_lean": {"what": "the share of a model's attempts that time out in Lean, on the three rungs and on G", **out_of_time},
            "distinct_attempts": {"what": "the share of a model's attempts at one problem and side that are distinct, on the three rungs and on G", **distinct},
            "never_solved_before": unsolved}}

    # ---- the lines: the checks first, then the primary, the branch and the secondary reads
    change = check["the_one_change"]
    fallback = (f" THIS IS THE FALLBACK: rank {check['rank']} is run in the place of rank {check['in_the_place_of_rank']}, which the box could not hold (that run is "
                "VOID)." if check["check"] == FALLBACK else "")
    lines = [
        f"{SAY}: L4r, {QUESTION.upper()} A labelled check of the pretraining, seed {prepare['seed']}. `{name}`: `pre`'s pass again (from the base, ONE pass over the same "
        f"{prepare['rows']:,} published proofs in the file's order, {train['steps']:,} optimizer steps; the same seed, learning rate, batch, sequence limit and target "
        f"modules) with ONE change: the adapter's rank is {change['rank']['check']} and its alpha {change['alpha']['check']}, where `pre` has {change['rank'][PRE]} and "
        f"{change['alpha'][PRE]}" + (f" ({numbers:,} trained numbers)" if numbers else "") + f".{fallback} It is measured as `pre` was, with `pre`'s sampling seeds, and "
        f"read against `pre`'s stored rows ({of_pre['run']}), which are only read. No map is made",
        f"{SAY}: CHECK 1, the training took: the mean loss over the last tenth of its rows ({took['rows_compared']:,}) is {took['last']}, over the first tenth it was "
        f"{took['first']}: {_verdict(took)}",
        f"{SAY}: CHECK 2, it still writes proofs: {writes['share']} of `{name}`'s {writes['attempts']:,} attempts on the three rungs got no answer "
        f"({writes['capped_at_the_token_limit']} reached the token cap, {writes['without_a_verdict_from_lean']} had no verdict from Lean); under {writes['maximum']} is "
        f"asked: {_verdict(writes)}",
        f"{SAY}: CHECK 3, Lean answered: the share of attempts without an answer in each set read (at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} is asked): "
        + "; ".join(f"{set_name} {entry['share_without_an_answer']} of {entry['attempts']:,}" for set_name, entry in health.items()) + f": {_verdict(answered)}"]
    if not inconclusive:
        lines.append(f"{SAY}: PRIMARY. All of G ({primary['problems']} problems), successes per attempt over the {attempts} attempts a problem, `{name}` minus `pre`, "
                     f"paired by problem, 95% bootstrap over problems: {_rate(primary)}"
                     + (f" ({primary['successes']} successes against {primary['successes_of_the_base']} in {primary['attempts_each']:,} attempts each)"
                        if primary.get("mean") is not None else ""))
    lines.append(f"{SAY}: {'INCONCLUSIVE' if inconclusive else 'BRANCH: ' + branch['name']}. {branch['reason']}")
    if not inconclusive:
        both = lambda value: "; ".join(f"`{key}` {value(key)}" for key in who)      # noqa: E731
        lines += [
            f"{SAY}: SECONDARY, the same by the length of the shortest published proof, `{name}` minus `pre`: "
            + "; ".join(f"{_named(group)} ({len(sets[group])}): {_rate(entry)}" for group, entry in by_length.items()),
            f"{SAY}: SECONDARY, goal problems solved at {attempts} attempts, `{name}` against `pre`: "
            + "; ".join(f"{_named(group)}: {solved_on_g[group]['resolved_after']} to {solved_on_g[group]['resolved_before']}, {_split(solved_on_g[group])}"
                        for group in (ALL, FOUR_PLUS)),
            f"{SAY}: SECONDARY, goal problems solved in at least one episode of 8 one-shot attempts / reliably (in at least half of their episodes): "
            + "; ".join(f"{_named(group)} ({len(sets[group])}): " + ", ".join(f"`{key}` {alone[key][group]['solved_at_least_once']} / {alone[key][group]['reliably']}"
                                                                              for key in who) for group in (ALL, FOUR_PLUS)),
            f"{SAY}: SECONDARY, the three rungs ({prepare['rung_episodes']} episodes a problem), `{name}` minus `pre`: "
            + "; ".join(f"{rung} {interval_text(rungs[rung])}" for rung in RUNGS),
            f"{SAY}: SECONDARY, the loss on rows not yet trained on, by twentieth of the pass: " + both(lambda key: _series(loss[key])),
            f"{SAY}: SECONDARY, the share of attempts that time out in Lean (on the rungs, on G): "
            + both(lambda key: f"{out_of_time[key][RUNG_PART]['share']}, {out_of_time[key][GOAL]['share']}")
            + ". The share of distinct attempts (on the rungs, on G): "
            + both(lambda key: f"{distinct[key][RUNG_PART]['mean_share_distinct']}, {distinct[key][GOAL]['mean_share_distinct']}"),
            f"{SAY}: SECONDARY, the goal problems nothing stored has ever solved: "
            + (f"of the {unsolved['goal_problems']} listed, `{name}` solves {unsolved['solved']}" + _named_problems(unsolved["problem_ids"])
               + f"; `pre` solves {unsolved['solved_by_pre']} of them"
               + (f"; {unsolved['not_goal_problems_of_this_run']} more ids of the file are not goal problems of this run and are not counted"
                  if unsolved["not_goal_problems_of_this_run"] else "")
               if unsolved["given"] else "not given (no file of their ids is in the run directory; the stage does not compute it)")]
    if not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}. The step fails; the task queued again "
                     "measures such a set of this run's own again from the kept adapter")
    headline = (f"L4r ({QUESTION[:-1]}; {LABEL}) seed {prepare['seed']}, `{name}` at rank {check['rank']}: {branch['name']}. "
                + ("" if inconclusive else f"Primary (all of G, {primary['problems']} problems, {attempts} attempts each, `{name}` minus `pre`): {_rate(primary)}. ")
                + f"Checks: the training took {_verdict(took)} ({took['last']} against {took['first']}), still writes proofs {_verdict(writes)} ({writes['share']} without "
                  f"an answer), Lean answered {_verdict(answered)}. " + branch["reason"]
                + (f" NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, L4r: is the pretrained model capped by the size of its adapter?", "stage": L4, "label": LABEL,
        "headline": headline, "lines": lines, "ok": not not_to_be_read, "seed": prepare["seed"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or train.get("stand_in_engine") or model.get("stand_in_engine")),
        "branch": branch, "inconclusive": inconclusive,
        "the_check": {**{key: check.get(key) for key in ("check", "stage", "model", "rank", "alpha", "in_the_place_of_rank", "max_lora_rank", "trained_parameters",
                                                         "trained_parameters_of_pre", "counted_from", "the_one_change", PRE)},
                      "adapter_saved": train.get("adapter_saved"), "peak_allocated_gb": train.get("peak_allocated_gb"), "peak_reserved_gb": train.get("peak_reserved_gb")},
        "can_this_run_see_a_win": {"what": "the three checks the spec reads before the branch; when any fails the run is INCONCLUSIVE and nothing else is said", **checks},
        **({"primary": None, "secondary": None,
            "measured_and_not_read": {"why": "a check failed: the run is INCONCLUSIVE, and these numbers say nothing about the size of the adapter. They are kept for "
                                             "whoever repairs the run", **measured}} if inconclusive else measured),
        "what_it_cannot_say": cannot_say(name, check["rank"], of_pre["rank"]),
        "models": {PRE: f"`pre`: {LABEL}, one pass from the base over {of_pre['rows']:,} proofs of the pool's `pretrain` half at rank {of_pre['rank']}; its rows are its "
                        "own run's, stored: nothing here measured it again",
                   name: f"`{name}`: {LABEL}, the same pass at rank {check['rank']} and alpha {check['alpha']}; kept until the read"},
        "training": {key: value for key, value in train.items() if key != "checkpoints"},
        "pretraining_file": {key: prepare[key] for key in ("pretraining_file", "pretraining_file_sha256", "rows", "half", "half_seed", "rows_by_kind", "rows_by_proof_lines",
                                                           "tokens", "longest_example_tokens", "max_sequence_tokens", "order", "steps", "recipe")},
        "heldout": {"goal_set": len(goal_ids), "goal_set_by_length_group": {group: len(ids) for group, ids in sets.items()},
                    "rungs": {rung: len(rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "contradicted_side_setting")},
        "attempts": health, "not_to_be_read": not_to_be_read,
        "not_measured": ["`pre` (its rows are the stored ones of the run named under the_check.pre): the check measures its own model alone",
                         "the base, the stored three-round model and the ceiling's models: the read is against `pre`",
                         "a map of the check's model: one is made only for a model an arm will start from",
                         "anything with assembly on the goal set"],
    }
