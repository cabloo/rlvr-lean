"""L5's report: do DEEP solves teach what ordinary solves do not? Spec: docs/spec/ladder-loop.spec.md, "L5: do DEEP
solves teach what ordinary solves do not? One step-matched check" ("The read, fixed before the run", and its amendment:
"THE PRIMARY MOVES TO A LARGER SET, the held-aside hard problems"). Pure: rows in, a report out. Its pieces are L4t's
report's own (`reporting/ladder_l4_rows.py`: the world a model is read on, the checks of a trained model, its secondary
reads) and the rules of `domain/ladder_round/l5.py`.

Everything here is labelled pretrained on published proofs, as L4's reports are: every line begins with L4's label.
One-shot attempts only. A scout: one seed, no round.

  the checks        FIRST, each PASS or FAIL: the deep search found enough deep solves; for each trained model
                    (`published`, `shallow`, `deep`) its training ran, it still writes proofs, Lean answered, and no row
                    it was trained on is barred (no held-out problem; its published rows of the `pretrain` half; its own
                    rows base-map problems); and Lean answered on each of the four held-aside sets. ANY FAIL:
                    INCONCLUSIVE, and nothing else is said (what was measured is kept under `measured_and_not_read`).
                    Too few deep solves: nothing was trained, and the search is the finding
  the primary       ON THE HELD-ASIDE SET (HA: problems the start model failed in every map attempt, that the search
                    did not take and no model trains on), successes per attempt, `deep` minus `published`, paired by
                    problem, 95% bootstrap over problems, and its branch (`l5.primary_branch`); the breadth beside it on
                    every branch (the problems of HA solved at least once: NARROWER)
  the mechanism     `deep` minus `shallow`, the same two reads on HA (`l5.mechanism`)
  the replication   `shallow` minus `published`, the same two reads on HA, named plainly (`l5.replication`)
  beside each       the same pair on G' over the fresh attempts and on the goal problems solved in all the attempts: the
                    held-out goal set's word, which names NO branch
  what follows      the spec's branch for what follows, in one line (`l5.what_follows`)
  secondary         each trained model against the start model ON HA (the selection favours the trained model there, as
                    on G', and it is said), and on all of G per attempt against its stored rows; for each of the three
                    pairs, all of G per attempt and by the length of the shortest published proof; L4t's secondary reads
                    of each model BY ITSELF against the stored start model; the deep search itself (the problems drawn and solved, by the attempt of the first success;
                    its attempts and successes per 1,000; the lines of the deep rows beside the ordinary solves' and the
                    published rows')

A report in which Lean did not answer a set is NOT TO BE READ (`ok` is false, and the step fails): the task queued again
samples such a set of this run's own again from the adapter. THE LENGTH OF A HELD-OUT PROBLEM'S PUBLISHED PROOF is read
here and nowhere before.

A FURTHER SEED's report (spec "L5, further seeds") is this report on its own three models, the same reads and the same
names, against the start model's rows the search's own run stored, with two lines of its own (whose search and
held-aside set it read; each measured set's share of attempts that timed out in Lean, and where two models' shares
differ by more than a point) and two keys of its own (`a_further_seed`, `timed_out_in_lean`). The report of the search's
own seed has neither: it is what it was. THE READ OVER THE SEEDS is `over_seeds`, below; it is no step of the stage.
"""

from __future__ import annotations

import statistics
from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ
from rlvr_lean.domain.ladder_round.l4b import NARROWER
from rlvr_lean.domain.ladder_round.l5 import (
    DEEP,
    MODELS,
    NEW_SEEDS,
    PUBLISHED,
    SHALLOW,
    after_the_seeds,
    mechanism,
    no_barred_row,
    primary_branch,
    primary_over_the_seeds,
    reach_on_the_long,
    replication,
    the_lead,
    the_split,
    timed_out,
    what_follows,
)
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids, paired_change
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import ALL, _named, _rate, attempts_on_g, goal_sets, per_attempt, solved
from rlvr_lean.reporting.ladder_l2 import MAXIMUM_SHARE_WITHOUT_AN_ANSWER, _split
from rlvr_lean.reporting.ladder_l3c import LENGTH_GROUPS
from rlvr_lean.reporting.ladder_l4 import HELD_TO, LABEL, NOT_MEASURED, NOT_TO_BE_READ, SAY, _verdict
from rlvr_lean.reporting.ladder_l4_rows import check_lines, secondary_lines, secondary_reads, the_four_checks, world_of

L4 = "l4"
QUESTION = "do deep solves teach what ordinary solves do not?"
SPEC = "docs/spec/ladder-loop.spec.md, L5: do DEEP solves teach what ordinary solves do not? One step-matched check"
IS = {PUBLISHED: "the first 2D rows of the hash order L4t's rehearsal draw uses among the rows the start model was pretrained on, and nothing else",
      SHALLOW: "D rows of ordinary solves (map problems the start model solved in a few of its map attempts, one verified map attempt each) and the first D rows of that order",
      DEEP: "the D deep rows (one verified attempt of each problem the deep search solved) and the SAME first D rows of that order"}
PAIRS = ((DEEP, PUBLISHED), (DEEP, SHALLOW), (SHALLOW, PUBLISHED))      # the primary's, the mechanism's and the replication's, each `after` minus `before`
THE_SEARCH = "the_deep_search_found_enough"                             # the first check, as a failed check names it
THE_HELD_ASIDE = "lean_answered_on_the_held_aside_sets"                 # the last check, as a failed check names it
FAVOURED = ("the held-aside problems are ones the start model FAILED in every map attempt: that selection favours a trained model against it, as on G'. The three "
            "differences among the trained models are fair: the selection is the same for each")
CANNOT_SAY = [
    "One pass and no round: whether deep solves COMPOUND (a model trained on them reaching deeper ones) is the loop this would license, not this check",
    "One seed: between two trainings of one recipe the goal set has moved by about 0.004 per attempt and 6 problems (L4p's and L4c's published-only models)",
    "A depth of 64 is shallow against a search of thousands",
    "A trained model against the start model on the held-aside set is favoured by the selection (the start model's 8 map attempts chose the set): only the differences "
    "among the three trained models are fair there"]


def pair_name(after: str, before: str) -> str:
    """A pair of models, as the report's keys name it."""
    return f"{after}_minus_{before}"


def held_aside_key(name: str) -> str:
    """A model's held-aside set, as the report names it among the measured sets."""
    return f"held_aside_{name}"


def sets_by_kind(start: str, models: Sequence[str], samplings: int) -> dict[str, dict[str, str]]:
    """A report's measured sets by KIND of set (the rungs, each sampling of G, the held-aside set): for each model
    measured on it, its set by the name the report gives it under `attempts`."""
    everyone = (start, *models)
    return {"rungs": {name: f"rungs_{name}" for name in everyone},
            **{f"goal_{index}": {name: f"goal_{name}_{index}" for name in everyone} for index in range(1, samplings + 1)},
            "held_aside": {name: held_aside_key(name) for name in everyone}}


def _pieces(table: Mapping) -> str:
    """The deep search's table in one line."""
    by_band = "; ".join(f"{key.replace('_to_', ' to ')}: {count}" for key, count in table["solved_by_first_success"].items())
    return (f"{table['problems']:,} problems attempted ({table['episodes_each']} more attempts each, after the map's {table['attempts_before_each']}), {table['solved']:,} "
            f"solved (a share of {table['share_solved']}); by the attempt of the first success, counting the map's: {by_band}; {table['attempts']:,} attempts, "
            f"{table['successes']:,} of {table['episodes']:,} episodes resolved their problem ({table['per_1000']} per 1,000); {table['without_an_answer']} attempts without "
            "an answer from Lean")


def _stats(stats: Mapping) -> str:
    """The lines of a set of proofs (`verified_lines`), as a line gives them."""
    return (f"{stats['proofs']:,} proofs, median {stats['median']}, mean {stats['mean']}, share with 4 lines or more {stats['share_with_4_lines_or_more']}, with 8 or more "
            f"{stats['share_with_8_lines_or_more']}, longest {stats['longest']}")


def build_l5_report(prepare: Mapping, trains: Mapping[str, Mapping], losses: Mapping[str, Sequence[float]], trained: Mapping[str, Sequence[Mapping]],
                    groups: Sequence[Mapping], lengths: Mapping[str, Mapping], stored: Mapping[str, Mapping], models: Mapping[str, Mapping],
                    again: Sequence[str], settings: Mapping, evaluation: Mapping, base_map: Sequence[str], held_aside: Mapping[str, Sequence[Mapping]],
                    never: Sequence[str] | None = None) -> dict:
    """`prepare`: the summary of the step that built the three sets (`start`; `enough_deep_solves`; `the_search_found`;
    `the_held_aside`; `training_sets`, empty when the search found too few deep solves and nothing was trained).
    `trains`, `losses`, `trained`: by trained model, its training step's summary, its rows' losses in the order
    trained, and the rows its training loop recorded. `groups`: the held-out groups. `lengths`: by problem, its
    shortest published proof's length group. `stored`: the start model, from the rows its own run stored. `models`: the
    three trained models, each `rungs`, `goal` (one list of per-problem rows for each sampling of G) and what it wrote;
    NONE when nothing was trained. `again`: the start model's G'. `settings`: `ladder_loop.l4` (the checks' own).
    `base_map`: the ids of the base map's problems. `held_aside`: by model, the START MODEL and the three trained
    ones, its per-problem rows on the held-aside set (none when nothing was trained). `never`: the ids of the goal
    problems nothing stored had solved, as GIVEN in the run directory, or None when no file was given."""
    start, enough, found = prepare["start"], prepare["enough_deep_solves"], prepare["the_search_found"]
    rank = (prepare.get("start_recipe") or {}).get("rank")
    further = prepare.get("a_further_seed")         # A FURTHER SEED: what its first step recorded of the search's own run; the search's own seed has no such key
    measured = tuple(name for name in MODELS if name in models)
    if measured and (measured != MODELS or not enough["passes"]):
        raise ValueError(f"the report is given the trained models {list(measured)}: it reads all three ({list(MODELS)}), or none when the deep search found too few deep "
                         "solves and nothing was trained")
    on_the_set = (start, *measured) if measured else ()                 # the FOUR models measured on the held-aside set
    ids = {name: sorted(row["problem_id"] for row in held_aside.get(name, ())) for name in on_the_set}
    if sorted(held_aside) != sorted(on_the_set) or any(own != ids[start] or len(set(own)) != len(own) or not own for own in ids.values()):
        raise ValueError(f"the report is given held-aside rows of {sorted(held_aside)}: the start model `{start}` and the three trained models are measured there, each on "
                         "the same problems, or no model when nothing was trained")
    aside_ids = ids[start] if measured else []
    who = {start: stored[start], **{name: models[name] for name in measured}}
    world = world_of(prepare, who, groups, lengths, again, settings, evaluation, never, also={held_aside_key(name): held_aside[name] for name in on_the_set})
    goal_ids, sets, on_g, attempts, fresh = world.goal_ids, world.sets, world.on_g, world.attempts, world.fresh
    of_the_set, of_the_search = prepare["the_held_aside"], prepare["the_search"]

    # ---- the checks, FIRST: the deep search's, each trained model's four (L4t's, with this stage's own fourth), then Lean's answers on the held-aside sets
    checks_of, sets_read, check_lines_of = {}, {}, {}
    for name in measured:
        sets_read[name] = [*world.sets_of[name], *world.sets_of[start]]
        of_l4t = the_four_checks(world, name, trains[name], losses[name], trained[name], sets_read[name], f"the stored ones of `{start}`")
        barred = no_barred_row(trained[name], world.heldout, settings["half_seed"], base_map)
        checks_of[name] = {**of_l4t, "no_barred_row": barred}
        of = f"{SAY}: `{name}`"
        check_lines_of[name] = [
            *check_lines(world, of, name, IS[name], trains[name], of_l4t, sets_read[name])[:3],
            f"{of}, CHECK 4, no barred row (no held-out problem; its published rows of the `pretrain` half; the rows of the model's own are base-map problems): "
            f"{barred['held_out_problems']} held-out problems, {barred['published_rows_not_of_the_pretrain_half']} of its {barred['published_rows_of_the_pretrain_half']:,} "
            f"published rows not of the `pretrain` half, {barred['own_rows_not_of_the_base_map']} of its {barred['rows_of_the_models_own']:,} own rows not base-map problems: "
            f"{_verdict(barred)}"]
    silent = [held_aside_key(name) for name in on_the_set if held_aside_key(name) in world.not_to_be_read]
    answered = {"what": f"Lean gave a verdict on all but at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} of the attempts of each held-aside set: the start model's and the three "
                        "trained models', which the task queued again samples again", "maximum": MAXIMUM_SHARE_WITHOUT_AN_ANSWER,
                "share_without_an_answer": {held_aside_key(name): world.health[held_aside_key(name)]["share_without_an_answer"] for name in on_the_set},
                "sets_not_answered": silent, "passes": bool(on_the_set) and not silent}
    failed = [*([] if enough["passes"] else [THE_SEARCH]), *(f"`{name}`: {check}" for name in measured for check, entry in checks_of[name].items() if not entry["passes"]),
              *([THE_HELD_ASIDE] if measured and not answered["passes"] else [])]
    inconclusive = bool(failed)

    # ---- the three pairs, each `after` minus `before`: ON THE HELD-ASIDE SET (the reads), and beside them on G' and the goal problems (which name no branch)
    def of_a_pair(after: str, before: str) -> dict:
        return {"on_the_held_aside_set": per_attempt(held_aside[after], held_aside[before], aside_ids, world.resamples, world.seed),
                "held_aside_solved": solved(held_aside[after], held_aside[before], aside_ids),
                "on_the_goal_set_again": world.on_again(after, before), "goal_problems_solved": solved(on_g[after], on_g[before], goal_ids),
                "on_all_of_g": per_attempt(on_g[after], on_g[before], goal_ids, world.resamples, world.seed),
                "by_length_group": {group: per_attempt(on_g[after], on_g[before], ids_, world.resamples, world.seed) for group, ids_ in sets.items() if group != ALL}}

    pairs = {pair_name(after, before): of_a_pair(after, before) for after, before in PAIRS} if measured else {}
    figures = lambda after, before, key: pairs[pair_name(after, before)][key] if measured else dict(NOT_MEASURED) if key == "on_the_held_aside_set" else {}      # noqa: E731
    primary = primary_branch(failed, figures(DEEP, PUBLISHED, "on_the_held_aside_set"), figures(DEEP, PUBLISHED, "held_aside_solved"))
    depth = mechanism(failed, figures(DEEP, SHALLOW, "on_the_held_aside_set"), figures(DEEP, SHALLOW, "held_aside_solved"))
    again_ = replication(failed, figures(SHALLOW, PUBLISHED, "on_the_held_aside_set"), figures(SHALLOW, PUBLISHED, "held_aside_solved"))
    follows = what_follows(primary, depth)

    # ---- each trained model against the start model ON THE HELD-ASIDE SET; and BY ITSELF against the stored start model: L4t's secondary reads
    against_the_start = {name: {"per_attempt": per_attempt(held_aside[name], held_aside[start], aside_ids, world.resamples, world.seed),
                                "solved": solved(held_aside[name], held_aside[start], aside_ids)} for name in measured}
    on_all_of_g = {name: {"per_attempt": per_attempt(on_g[name], on_g[start], goal_ids, world.resamples, world.seed), "solved": solved(on_g[name], on_g[start], goal_ids)}
                   for name in measured}
    secondary, secondary_lines_of = {}, {}
    for name in measured:
        others = tuple(other for other in MODELS if other != name)
        secondary[name] = secondary_reads(world, name, start, (start, *others), others, trained[name])
        secondary_lines_of[name] = secondary_lines(world, f"{SAY}: `{name}`", name, start, (start, *others), others, secondary[name])

    count, lines_of = prepare["rows_of_each_kind"], prepare.get("lines_of_the_rows") or {}
    of_the_fresh = f"the {fresh['episodes']} fresh attempts" if fresh else "fresh attempts, which this run does not have"
    not_the_rules = {key: value for key, value in (prepare.get("rows_not_taken_by_the_rule") or {}).items() if key != "what" and value}
    each = of_the_set["episodes_each"]

    def on_the_set_said(after: str, before: str, read: Mapping) -> str:
        """A pair's two reads ON THE HELD-ASIDE SET, as a line gives them."""
        pair = pairs[pair_name(after, before)]
        return (f"THE HELD-ASIDE SET ({len(aside_ids):,} problems `{start}` failed in every map attempt, that the search did not take and no model trains on), successes "
                f"per attempt over {each} fresh attempts a problem, `{after}` minus `{before}`, paired by problem, 95% bootstrap over problems: "
                f"{_rate(pair['on_the_held_aside_set'])}; held-aside problems solved at least once, `{after}` against `{before}`: {pair['held_aside_solved']['resolved_after']} to "
                f"{pair['held_aside_solved']['resolved_before']}, {_split(pair['held_aside_solved'])}" + (f": {NARROWER}" if read.get("narrower") else ""))

    def beside_said(after: str, before: str) -> str:
        """The same pair on the held-out goal set, which names no branch."""
        pair = pairs[pair_name(after, before)]
        return (f"BESIDE IT, the held-out goal set's word, naming no branch: G' ({len(again)} goal problems `{start}` does not solve in its first sampling), successes per "
                f"attempt over {of_the_fresh}, `{after}` minus `{before}`: {_rate(pair['on_the_goal_set_again'])}; goal problems solved at least once in the {attempts} attempts, "
                f"`{after}` against `{before}`: {pair['goal_problems_solved']['resolved_after']} to {pair['goal_problems_solved']['resolved_before']}, "
                f"{_split(pair['goal_problems_solved'])}")

    lines = [
        f"{SAY}: L5, {QUESTION.upper()} Seed {prepare['seed']}, " + (f"A FURTHER SEED of the scout of seed {further['search_seed']}, which made the search: no round" if further
                                                                     else "a scout: one seed, no round")
        + f". THE DEEP SEARCH: `{start}` made {of_the_search['episodes_each']} more attempts "
        f"({of_the_search['attempts_with_the_maps']} with its map's) at {of_the_search['problems']:,} of the {of_the_search['candidates']:,} problems of its own map that it "
        f"failed in every map attempt (base-map problems: no model here has seen their published proofs), sampling seed {of_the_search['sampling_seed']}; a problem solved "
        "there is a DEEP solve, its row one of its verified attempts"
        + (f". THREE MODELS, each trained FROM `{start}` AT RANK {rank}, one pass, the arm's recipe, on {2 * count:,} rows (D = {count:,}: the same number of rows and of "
           f"optimizer steps), in the content-hash order, and measured as L4t's models were with the sampling seeds `{start}`'s run recorded: `{PUBLISHED}` ({IS[PUBLISHED]}), "
           f"`{SHALLOW}` ({IS[SHALLOW]}), `{DEEP}` ({IS[DEEP]}). THE HELD-ASIDE SET: {of_the_set['problems']:,} of the {of_the_set['candidates_not_searched']:,} candidates the "
           f"search's draw did NOT take, a draw of its own; `{start}` and the three trained models each made {each} fresh attempts at every one, with ONE sampling seed "
           f"({of_the_set['sampling_seed']}). THE PRIMARY is ON THE HELD-ASIDE SET, `{DEEP}` minus `{PUBLISHED}`; THE MECHANISM is `{DEEP}` minus `{SHALLOW}` and THE REPLICATION "
           f"`{SHALLOW}` minus `{PUBLISHED}`, there too; G' and the goal problems stand beside each and name no branch"
           if measured else ". NO MODEL WAS TRAINED: the search found too few deep solves for the three sets")]
    if further:
        lines.append(f"{SAY}: A FURTHER SEED: seed {prepare['seed']} reads the deep search, the held-aside set and the measurement of `{start}` on it from the run of seed "
                     f"{further['search_seed']} ({further['search_run']}), which made them ONCE; nothing of them was sampled again, and this run's models were measured with the "
                     f"sampling seeds that run's were (the held-aside set: {of_the_set['sampling_seed']}), so a problem's attempts differ between the seeds by the model alone. "
                     f"THIS SEED'S OWN, by its seed: which {count:,} of the search's {enough['deep_solves']:,} deep solves and which verified attempt of each, which ordinary "
                     "solves and which attempt, the published rows (the first D and the first 2D of the hash order of this seed), each set's training order, the three "
                     f"trainings. The lines below read THIS seed alone, as seed {further['search_seed']}'s report reads it; what the seeds say together, and the branch for what "
                     "follows them, is the read over the seeds (python -m rlvr_lean.tools.ladder_l5_seeds), not this report's line WHAT FOLLOWS")
    if not_the_rules or (measured and of_the_set["taken_by_the_runs_minimum"]):
        lines.append(f"{SAY}: A SMOKE RUN: rows of problems the rule did not give were taken, so that every step has rows to run on: "
                     + ", ".join(f"{value} of `{key}`'s own rows" for key, value in not_the_rules.items()) + f"; {of_the_search['taken_by_the_runs_minimum']} of the "
                     f"search's problems; {of_the_set['taken_by_the_runs_minimum']} of the held-aside problems. Nothing here is a read of the rule")
    if measured and not fresh:
        lines.append(f"{SAY}: the models have ONE sampling of G here (a smoke run): G was attempted {attempts} times a problem, and G' has no fresh attempts: what stands "
                     "beside the reads on the held-aside set is not measured there")
    lines.append(f"{SAY}: CHECK, the deep search found at least {enough['minimum']} deep solves: {enough['deep_solves']:,}: {_verdict(enough)}")
    for name in measured:
        lines += check_lines_of[name]
    if measured:
        lines.append(f"{SAY}: CHECK, Lean answered on each held-aside set (at most {MAXIMUM_SHARE_WITHOUT_AN_ANSWER} without an answer): "
                     + "; ".join(f"{key} {share}" for key, share in answered["share_without_an_answer"].items()) + f": {_verdict(answered)}")
    if inconclusive:
        lines.append(f"{SAY}: {INCONCLUSIVE}. {primary['reason']}" + ("" if enough["passes"] else f". {prepare['trainings']['why']}"))
    else:
        lines += [f"{SAY}: PRIMARY. {on_the_set_said(DEEP, PUBLISHED, primary)}",
                  f"{SAY}: BRANCH: {primary['name']}" + (f", {NARROWER}" if primary.get("narrower") else "") + f". {primary['reason']}",
                  f"{SAY}: PRIMARY, {beside_said(DEEP, PUBLISHED)}",
                  f"{SAY}: THE MECHANISM (the claim under test: a solve the model could not sample extends it where one it could does not). {on_the_set_said(DEEP, SHALLOW, depth)}",
                  f"{SAY}: THE MECHANISM: {depth['name']}. {depth['reason']}",
                  f"{SAY}: THE MECHANISM, {beside_said(DEEP, SHALLOW)}",
                  f"{SAY}: THE REPLICATION L4c owes, at equal steps and equal published rows. {on_the_set_said(SHALLOW, PUBLISHED, again_)}",
                  f"{SAY}: THE REPLICATION: {again_['name']}" + (f", {NARROWER}" if again_.get("narrower") else "") + f". {again_['reason']}",
                  f"{SAY}: THE REPLICATION, {beside_said(SHALLOW, PUBLISHED)}",
                  f"{SAY}: WHAT FOLLOWS: {follows['name']}. {follows['reason']}",
                  f"{SAY}: SECONDARY, each trained model against `{start}` ON THE HELD-ASIDE SET ({len(aside_ids):,} problems, {each} attempts each; successes per attempt, and "
                  "the problems solved at least once): "
                  + "; ".join(f"`{name}`: {_rate(against_the_start[name]['per_attempt'])}, solved {against_the_start[name]['solved']['resolved_after']} to "
                              f"{against_the_start[name]['solved']['resolved_before']}, {_split(against_the_start[name]['solved'])}" for name in measured) + f". NOTE: {FAVOURED}",
                  f"{SAY}: SECONDARY, each trained model against the stored `{start}` on ALL OF G ({len(goal_ids)} problems, {attempts} attempts each; successes per attempt, and "
                  "the goal problems solved at least once): "
                  + "; ".join(f"`{name}`: {_rate(on_all_of_g[name]['per_attempt'])}, solved {on_all_of_g[name]['solved']['resolved_after']} to "
                              f"{on_all_of_g[name]['solved']['resolved_before']}, {_split(on_all_of_g[name]['solved'])}" for name in measured)]
        for after, before in PAIRS:
            pair = pairs[pair_name(after, before)]
            lines.append(f"{SAY}: SECONDARY, `{after}` minus `{before}`, successes per attempt over the {attempts} attempts: all of G ({len(goal_ids)}): "
                         f"{_rate(pair['on_all_of_g'])}; by the length of the shortest published proof: "
                         + "; ".join(f"{_named(group)} ({len(sets[group])}): {_rate(entry)}" for group, entry in pair["by_length_group"].items()))
        for name in measured:
            lines += secondary_lines_of[name]
    lines.append(f"{SAY}: THE DEEP SEARCH" + ("" if enough["passes"] else ", WHICH IS THE FINDING") + f": {_pieces(found)}")
    of_the_rows = {key: value for key, value in lines_of.items() if key != "what"} or {"deep_rows": found["lines_of_the_deep_rows"]}      # nothing trained: the deep solves' own
    lines.append(f"{SAY}: THE LINES OF THE ROWS (" + ("the proofs trained on" if measured else "the deep solves found; nothing was trained") + "): "
                 + "; ".join(f"{key.replace('_', ' ')}: {_stats(value)}" for key, value in of_the_rows.items()))
    timeouts = timed_out(world.health, sets_by_kind(start, measured, len(prepare["goal_samplings"]))) if further and measured else None
    if timeouts:        # the section's "Timeouts": printed by a further seed's report alone, so that the report of the search's own seed is what it was
        lines.append(f"{SAY}: TIMEOUTS, the share of each measured set's attempts that timed out in Lean: " + "; ".join(f"{key} {share}" for key, share in timeouts["share"].items())
                     + ". WHERE TWO MODELS' SHARES DIFFER BY MORE THAN A POINT: "
                     + ("; ".join(f"{entry['kind']}: `{entry['models'][0]}` {entry['shares'][0]} against `{entry['models'][1]}` {entry['shares'][1]}" for entry in timeouts["apart"])
                        + ": their timed-out attempts are to be checked again (Lean only, no measurement running) and the reads given both ways; no step of this stage does it"
                        if timeouts["apart"] else "nowhere"))
    if world.not_to_be_read:
        lines.append(f"{SAY}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(world.not_to_be_read)}. The step fails; the task queued again "
                     "measures such a set of this run's own again from the adapter")

    of_the_primary = pairs[pair_name(DEEP, PUBLISHED)]["on_the_held_aside_set"] if measured else dict(NOT_MEASURED)
    headline = (f"L5 ({QUESTION.rstrip('?')}; {LABEL}) seed {prepare['seed']}, from `{start}`: " + (INCONCLUSIVE if inconclusive else primary["name"])
                + ("" if inconclusive or not primary.get("narrower") else f", {NARROWER}")
                + (f". The deep search solved {found['solved']:,} of {found['problems']:,} problems" + (f"; D = {count:,}" if measured else ""))
                + ("" if inconclusive or of_the_primary.get("mean") is None
                   else f". Primary (the held-aside set, {len(aside_ids):,} problems, `{DEEP}` minus `{PUBLISHED}`): {_rate(of_the_primary)}")
                + ("" if inconclusive else f". The mechanism (`{DEEP}` minus `{SHALLOW}`): {depth['name']}. The replication (`{SHALLOW}` minus `{PUBLISHED}`): {again_['name']}"
                   + (f", {NARROWER}" if again_.get("narrower") else "") + f". What follows: {follows['name']}")
                + f". {primary['reason']}"
                + (f". NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(world.not_to_be_read)}" if world.not_to_be_read else ""))
    in_again = set(again)
    beside_keys = ("on_the_goal_set_again", "goal_problems_solved")

    def read_or_kept(branch: Mapping, after: str, before: str) -> dict:
        """A pair's read as the report stores it: its figures on the held-aside set and, beside, on the held-out goal
        set; or, INCONCLUSIVE, what was measured kept apart."""
        if not measured:
            return {**branch, "difference": {"mean": None}}
        pair = pairs[pair_name(after, before)]
        own = {"difference": pair["on_the_held_aside_set"], "solved": pair["held_aside_solved"],
               "beside_on_the_goal_set": {"what": "the held-out goal set's word, naming no branch: the same pair on G' over the fresh attempts, and the goal problems solved "
                                                  "at least once in all the attempts", **{key: pair[key] for key in beside_keys}}}
        return {**branch, "difference": {"mean": None}, "measured_and_not_read": own} if inconclusive else {**branch, **own}

    on_g_alone = {key: {part: pair[part] for part in ("on_all_of_g", "by_length_group")} for key, pair in pairs.items()}
    return {
        "spec": SPEC, "stage": L4, "label": LABEL, "check": "L5", "headline": headline, "lines": lines, "ok": not world.not_to_be_read, "seed": prepare["seed"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(entry.get("stand_in_engine") for entry in (*trains.values(), *models.values()))),
        "start": start, "rank": rank, "measured": list(measured), "rows_of_each_kind": count,
        "branches": {"the_primary": primary["name"], "the_mechanism": depth["name"], "the_replication": again_["name"], "what_follows": follows["name"]},
        "inconclusive": inconclusive, "failed_checks": failed,
        "can_this_run_be_read": {"what": "the checks the spec reads first; when any fails the run is INCONCLUSIVE and nothing else is said",
                                 "the_deep_search": enough, **{name: checks_of[name] for name in measured}, **({"the_held_aside_sets": answered} if measured else {})},
        "primary": {"what": f"ON THE HELD-ASIDE SET (problems `{start}` failed in every map attempt, that the deep search did not take and no model trains on): successes per "
                            f"attempt, `{DEEP}` minus `{PUBLISHED}`, paired by problem, a 95% bootstrap interval over problems. In its figures `of_the_base` is the side "
                            f"compared with: `{PUBLISHED}`. Beside it on every branch, the held-aside problems solved at least once (`breadth`)",
                    **read_or_kept(primary, DEEP, PUBLISHED)},
        "the_mechanism": {"what": f"the claim under test: `{DEEP}` minus `{SHALLOW}`, the same two reads on the held-aside set. DEPTH MATTERS when the interval is above "
                                  "zero, or more held-aside problems were gained than lost with the sign test under 0.05", **read_or_kept(depth, DEEP, SHALLOW)},
        "the_replication": {"what": f"the replication L4c owes, at equal steps and equal published rows: `{SHALLOW}` minus `{PUBLISHED}`, the same two reads on the held-aside "
                                    "set, named plainly", **read_or_kept(again_, SHALLOW, PUBLISHED)},
        "what_follows": follows,
        "the_held_aside_set": {**of_the_set, "problems_measured": len(aside_ids), "note": FAVOURED,
                               **({"against_the_start": {"measured_and_not_read": against_the_start} if inconclusive else against_the_start} if measured else {})},
        "pairs": {"what": "secondary: for each pair, all of G per attempt over all the attempts and by the length group of each goal problem's shortest published proof",
                  **({"measured_and_not_read": on_g_alone} if inconclusive else on_g_alone)},
        "against_the_start_on_all_of_g": {"what": f"secondary: each trained model against the stored `{start}` on all of G: successes per attempt over all the attempts, "
                                                  "paired by problem, 95% bootstrap over problems, and the goal problems solved at least once (gained, lost, the sign test)",
                                          **({"measured_and_not_read": on_all_of_g} if inconclusive else on_all_of_g)},
        "models": {name: {"model": f"`{name}`: from `{start}`, one pass over {trains[name]['rows']:,} rows: {IS[name]}",
                          **({"secondary": None, "measured_and_not_read": {"why": "a check failed: the run is INCONCLUSIVE, and these numbers say nothing about what deep solves "
                                                                                  "teach. They are kept for whoever repairs the run", "secondary": secondary[name]}}
                             if inconclusive else {"secondary": secondary[name]}),
                          "training": {key: value for key, value in trains[name].items() if key != "checkpoints"}} for name in measured},
        "the_deep_search": {**found, "prepared": of_the_search, "the_finding": not enough["passes"]},
        "lines_of_the_rows": lines_of or None, "rows_not_taken_by_the_rule": prepare.get("rows_not_taken_by_the_rule"),
        "what_it_cannot_say": list(CANNOT_SAY),
        "the_stored_model": {start: f"`{start}`: {LABEL}; the model of the check of the adapter's rank, measured by that stage; its rows on the rungs and on G are that "
                                    "run's. Its attempts at the held-aside set are " + (f"those of the run of seed {further['search_seed']}, which made the search: only read "
                                                                                           "here" if further else "THIS run's") + " (its 8 stored map attempts chose the set)",
                             "runs": prepare["stored_models"]},
        "training_sets": prepare["training_sets"], "trainings": prepare["trainings"],
        "heldout": {"goal_set": len(goal_ids), "goal_set_again": len(again), "goal_set_by_length_group": {group: len(ids_) for group, ids_ in sets.items()},
                    "goal_set_again_by_length_group": {group: len(ids_) for group, ids_ in goal_sets([problem_id for problem_id in goal_ids if problem_id in in_again], lengths).items()},
                    "rungs": {rung: len(world.rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "contradicted_side_setting", "start",
                                                "start_adapter", "l4t2_run", "map_run")},
        "start_recipe": prepare.get("start_recipe"), "attempts": world.health, "not_to_be_read": list(world.not_to_be_read),
        "not_measured": [f"`{start}` on the rungs and on G (its rows are the stored ones of the run named under the_stored_model)", "a round, an assembled proof or a "
                         "challenger: the deep search is attempts and Lean's verdicts alone", "whether deep solves compound: one pass and no round",
                         "anything at another temperature than the config's"],
        **({"a_further_seed": further, "timed_out_in_lean": timeouts} if further else {}),       # absent at the search's own seed: its report is what it was
    }


# ---------------------------------------------------------------------------------- the read over the seeds
SEEDS_SPEC = ("docs/spec/ladder-loop.spec.md, L5, further seeds (the reads, fixed before they run; amended 2026-10-11 before seeds 1 and 2 ran: where the lead sits, "
              "and a read of reach)")
LONG = LENGTH_GROUPS[-1]                    # the goal problems whose shortest published proof has 8 lines or more: the amendment's "long problems"
# The figures of a kind of model whose spread over its seeds is given, as a line names each.
SPREAD_OF = {"all_of_g_per_1000": "all of G per 1,000 attempts", "goal_problems_solved": "goal problems solved", "held_aside_per_1000": "the held-aside set per 1,000 attempts",
             "goal_set_again_per_1000": "G' per 1,000 fresh attempts"}
# What a seed is given as, beside its report: one search (`searched`, `solves`), one held-aside set (`aside`), one G and G' (`groups`, `again`), and each model's rows.
ONE_SEARCH, ONE_SET, ONE_GOAL_SET = ("searched", "solves"), ("aside",), ("groups", "again")


def _not_read(report: Mapping) -> str | None:
    """Why a seed is not read, or None when it is: its report is not to be read, a check failed (INCONCLUSIVE: nothing
    else is said of such a run), or its primary had nothing to be read on."""
    if not report.get("ok", True):
        return NOT_TO_BE_READ
    if report.get("inconclusive"):
        return INCONCLUSIVE
    return NOT_READ if ((report.get("primary") or {}).get("difference") or {}).get("mean") is None else None


def _counted(rows: Sequence[Mapping]) -> dict:
    return {row["problem_id"]: (row["resolved"], row["episodes"]) for row in rows}


def _per_1000(rows: Sequence[Mapping], ids: Sequence[str]) -> float | None:
    """One model's successes per 1,000 attempts over `ids`."""
    wanted = set(ids)
    own = [row for row in rows if row["problem_id"] in wanted]
    attempts = sum(row["episodes"] for row in own)
    return round(1000 * sum(row["resolved"] for row in own) / attempts, 2) if attempts else None


def _solves(rows: Sequence[Mapping], ids: Sequence[str]) -> set[str]:
    """The problems of `ids` one model solves at least once in its attempts."""
    wanted = set(ids)
    return {row["problem_id"] for row in rows if row["problem_id"] in wanted and row["resolved"] > 0}


def _named_seeds(seeds: Sequence[int]) -> str:
    return ", ".join(str(seed) for seed in seeds) if seeds else "none"


def over_seeds(by_seed: Mapping[int, Mapping], lengths: Mapping[str, Mapping], resamples: int, bootstrap_seed: int) -> dict:
    """THE READ OVER THE SEEDS of L5 (spec, "L5, further seeds" and its amendment). `by_seed`: for each seed that is
    here, from its pulled run: `report`; `searched` (the problems the deep search attempted) and `solves` (by problem
    it solved: its k, its n and the place of its first success); `aside` (the held-aside problems); `groups` (the
    held-out groups) and `again` (G'); `held_aside` and `goal`: by model (the start model and the three trained ones)
    its per-problem rows on the held-aside set, and on G one list a sampling. `lengths`: by goal problem, the length
    group of its shortest published proof.

    The seed whose report says nothing of a further seed MADE the search and found the goal set's lead; the others are
    the NEW seeds. Per seed: its own figures, the primary computed again from its rows and HELD to its report's.
      the primary     on the held-aside set, `deep` minus `published`, each problem's mean over the THREE seeds of its
                      difference (`per_attempt` on each problem's attempts added over the seeds: every seed makes as
                      many), each seed's own difference and sign beside (`l5.primary_over_the_seeds`)
      the lead        ON THE NEW SEEDS ALONE: all of G over all the attempts, `deep` minus `published`, each problem's
                      mean over the two new seeds (`l5.the_lead`); beside it the goal problems solved at each seed, the
                      same two reads of `deep` minus `shallow` and `shallow` minus `published`, and every seed pooled
                      with the seed that found the lead marked
      the split       the same difference (a) on the goal problems NOT in G' and (b) on G' (`l5.the_split`)
      the reach       on the goal problems of 8 lines or more, the share of the new seeds in which a model solves the
                      problem at least once, `deep` minus `published`, paired (`l5.reach_on_the_long`); beside it each
                      model's count at each seed with gained and lost, and the same read of `deep` minus `shallow`
      the spread      for each kind of model its seeds' figures and their standard deviation (n - 1)
      what follows    the section's branch, in one line (`l5.after_the_seeds`)
    A read that needs seeds that are not here is NOT READ and is listed (`could_not_read`): it works on one seed.

    Refused (ValueError): no seed; seeds that do not share ONE search, ONE held-aside set, ONE measurement of the start
    model or ONE G and G'; two seeds that each made a search; rows that do not give a report's primary. A seed whose
    report is INCONCLUSIVE or not to be read is named, nothing is said of it, and NOTHING IS POOLED."""
    if not by_seed:
        raise ValueError("the read over the seeds of L5 was given no seed")
    seeds = sorted(by_seed)
    why_not = {seed: _not_read(by_seed[seed]["report"]) for seed in seeds}
    read = [seed for seed in seeds if why_not[seed] is None]
    searched_by = {seed: (by_seed[seed]["report"].get("a_further_seed") or {}).get("search_seed", seed) for seed in seeds}
    if len(set(searched_by.values())) != 1:
        raise ValueError(f"the seeds do not share ONE search: by seed, the seed whose task made the search it read is {searched_by}. Every further seed reads the one search, "
                         "and one task alone makes it")
    found_by = searched_by[seeds[0]]                                    # the seed that made the search, and so found the lead
    new = [seed for seed in read if seed != found_by]                   # the NEW seeds, of those that are read
    starts = {by_seed[seed]["report"].get("start") for seed in seeds}
    if len(starts) != 1:
        raise ValueError(f"the seeds are of other start models ({sorted(map(str, starts))}): every seed of L5 is from the one start")
    (start,) = starts
    everyone = (start, *MODELS)
    first = by_seed[read[0]] if read else None
    for seed in read[1:]:
        entry = by_seed[seed]
        for what, keys in (("search", ONE_SEARCH), ("held-aside set", ONE_SET), ("G and G'", ONE_GOAL_SET)):
            other = [key for key in keys if entry[key] != first[key]]
            if other:
                raise ValueError(f"seed {seed} does not hold the ONE {what} seed {read[0]} holds ({', '.join(other)} differ): every seed of L5 reads one search, one "
                                 "held-aside set and one held-out goal set")
        if _counted(entry["held_aside"][start]) != _counted(first["held_aside"][start]) or [_counted(rows) for rows in entry["goal"][start]] != [_counted(rows) for rows in first["goal"][start]]:
            raise ValueError(f"seed {seed} holds other rows of `{start}` than seed {read[0]}: every seed of L5 is read against the ONE measurement of `{start}` (on the "
                             "held-aside set, the search's own run's; on G, the stored ones)")
        for name in MODELS:
            if {key: value[1] for key, value in _counted(entry["held_aside"][name]).items()} != {key: value[1] for key, value in _counted(first["held_aside"][name]).items()} \
                    or [{key: value[1] for key, value in _counted(rows).items()} for rows in entry["goal"][name]] != [{key: value[1] for key, value in _counted(rows).items()} for rows in first["goal"][name]]:
                raise ValueError(f"seed {seed} made other numbers of attempts for `{name}` than seed {read[0]}: a problem's mean over the seeds is read on its attempts added "
                                 "over them, which takes as many at every seed")
    ids, on_g, on_ha = {}, {}, {}
    if read:
        goal_ids, again = group_ids(first["groups"], GOAL), list(first["again"])
        in_again = set(again)
        # The held-aside problems in the order a seed's own report reads them (sorted), G in the groups' order: a bootstrap over problems is a draw of places in that order.
        ids = {"aside": sorted(row["problem_id"] for row in first["aside"]), "goal": goal_ids, "again": again, "not_again": [key for key in goal_ids if key not in in_again],
               "long": goal_sets(goal_ids, lengths).get(LONG, [])}
        on_g = {seed: {name: attempts_on_g(by_seed[seed]["goal"][name]) for name in everyone} for seed in read}
        on_ha = {seed: {name: by_seed[seed]["held_aside"][name] for name in everyone} for seed in read}
    difference = lambda after, before, over: per_attempt(after, before, over, resamples, bootstrap_seed)      # noqa: E731

    # ---- each seed by itself: what its report read, computed again from its rows; and the figures its report does not give
    each = {}
    for seed in seeds:
        report = by_seed[seed]["report"]
        of_the_seed = {"read": why_not[seed] is None, "why_not": why_not[seed], "made_the_search": seed == found_by, "branches": report.get("branches")}
        if why_not[seed] is not None:
            each[seed] = of_the_seed
            continue
        goal, fresh = by_seed[seed]["goal"], len(by_seed[seed]["goal"][start]) > 1
        models = {name: {"held_aside_per_1000": _per_1000(on_ha[seed][name], ids["aside"]), "all_of_g_per_1000": _per_1000(on_g[seed][name], ids["goal"]),
                         "goal_problems_solved": len(_solves(on_g[seed][name], ids["goal"])),
                         "goal_set_again_per_1000": _per_1000(goal[name][1], ids["again"]) if fresh else None,
                         "long_problems_solved": len(_solves(on_g[seed][name], ids["long"]))} for name in everyone}
        pairs = {pair_name(after, before): {
            "on_the_held_aside_set": difference(on_ha[seed][after], on_ha[seed][before], ids["aside"]),
            "held_aside_solved": solved(on_ha[seed][after], on_ha[seed][before], ids["aside"]),
            "on_all_of_g": difference(on_g[seed][after], on_g[seed][before], ids["goal"]),
            "goal_problems_solved": solved(on_g[seed][after], on_g[seed][before], ids["goal"]),
            "on_the_goal_problems_not_in_the_goal_set_again": difference(on_g[seed][after], on_g[seed][before], ids["not_again"]),
            "on_the_goal_set_again": difference(on_g[seed][after], on_g[seed][before], ids["again"]),
            "long_problems_solved": solved(on_g[seed][after], on_g[seed][before], ids["long"])} for after, before in PAIRS}
        primary, of_the_report = pairs[pair_name(DEEP, PUBLISHED)]["on_the_held_aside_set"], report["primary"]["difference"]
        differs = [key for key in HELD_TO if primary.get(key) != of_the_report.get(key)]
        if differs:
            raise ValueError(f"seed {seed}: the primary computed from its stored rows is not its report's ({', '.join(differs)}: {[primary.get(key) for key in differs]} "
                             f"against {[of_the_report.get(key) for key in differs]}): these are not the rows that report read, or the bootstrap's settings are not the run's")
        samplings = len(goal[start])
        each[seed] = {**of_the_seed, "models": models, "pairs": pairs, "timed_out_in_lean": timed_out(report["attempts"], sets_by_kind(start, MODELS, samplings))}

    # ---- over the seeds: nothing is pooled over a seed that is not read
    all_read = len(read) == len(seeds)
    could_not_read, pooled = [], None
    if all_read:
        over = lambda among, rows_, after, before, problems: difference(attempts_on_g([rows_[seed][after] for seed in among]),      # noqa: E731
                                                                       attempts_on_g([rows_[seed][before] for seed in among]), problems) if among else dict(NOT_MEASURED)
        at = lambda among, pair, key: {seed: each[seed]["pairs"][pair][key]["mean"] for seed in among}      # noqa: E731
        the_pair = pair_name(DEEP, PUBLISHED)
        on_the_set = over(read, on_ha, DEEP, PUBLISHED, ids["aside"])
        at_each = at(read, the_pair, "on_the_held_aside_set")
        # The primary takes the seed that made the search AND the two further: without that seed nothing is named, however many seeds are here.
        primary = {"difference": on_the_set, "at_each_seed": at_each, **primary_over_the_seeds(on_the_set, at_each if found_by in read else {})}
        on_goal = {pair_name(after, before): {"difference": over(new, on_g, after, before, ids["goal"]), "at_each_new_seed": at(new, pair_name(after, before), "on_all_of_g"),
                                              "goal_problems_solved_at_each_seed": {seed: each[seed]["pairs"][pair_name(after, before)]["goal_problems_solved"] for seed in read},
                                              "every_seed_pooled": over(read, on_g, after, before, ids["goal"])} for after, before in PAIRS}
        lead = {**on_goal[the_pair], **the_lead(on_goal[the_pair]["difference"], on_goal[the_pair]["at_each_new_seed"])}
        not_again, on_again = over(new, on_g, DEEP, PUBLISHED, ids["not_again"]), over(new, on_g, DEEP, PUBLISHED, ids["again"])
        split = {"on_the_goal_problems_not_in_the_goal_set_again": not_again, "on_the_goal_set_again": on_again, **the_split(lead, on_again)}
        long_solved = {seed: {name: _solves(on_g[seed][name], ids["long"]) for name in MODELS} for seed in read}

        def share_of_the_new_seeds(name: str) -> list[dict]:
            """For each long problem, in how many of the NEW seeds the model solves it at least once in all its attempts."""
            return [{"problem_id": key, "resolved": sum(key in long_solved[seed][name] for seed in new), "episodes": len(new)} for key in ids["long"]]

        def reach_of(after: str, before: str) -> dict:
            shares = paired_change(share_of_the_new_seeds(after), share_of_the_new_seeds(before), ids["long"], resamples, bootstrap_seed) if new else dict(NOT_MEASURED)
            counts = {seed: (len(long_solved[seed][after]), len(long_solved[seed][before])) for seed in new}
            return {"difference": shares, "counts_at_each_new_seed": counts, **reach_on_the_long(shares, counts, after=after, before=before)}

        reach = reach_of(DEEP, PUBLISHED)
        beside_reach = {key: value for key, value in reach_of(DEEP, SHALLOW).items() if key != "name"}          # the same read, which names nothing
        spread = {}
        for name in MODELS:
            spread[name] = {}
            for figure in SPREAD_OF:
                values = {seed: each[seed]["models"][name][figure] for seed in read}
                known = [value for value in values.values() if value is not None]
                spread[name][figure] = {"by_seed": values, "mean": round(statistics.mean(known), 3) if known else None,
                                        "standard_deviation": round(statistics.stdev(known), 3) if len(known) > 1 and len(known) == len(values) else None}
        follows = after_the_seeds(primary, lead, split, reach)
        pooled = {"the_primary": primary, "the_lead": lead, "beside_the_lead": {key: value for key, value in on_goal.items() if key != the_pair}, "the_split": split,
                  "reach_on_the_long_problems": reach, "beside_the_reach": {pair_name(DEEP, SHALLOW): beside_reach}, "the_spread": spread, "what_follows": follows}
        missing = NEW_SEEDS - len(new)
        if found_by not in read or missing:
            could_not_read.append(f"the primary over the three seeds (it takes the seed that made the search, {found_by}, and {NEW_SEEDS} further; here: "
                                  f"{_named_seeds(read)})")
        if missing:
            could_not_read += [f"{what} (read on the {NEW_SEEDS} new seeds alone; here: {_named_seeds(new)})"
                               for what in ("the goal set's lead, confirmed or not", "the split of the lead", "reach on the long problems")]
        if len(read) < 2:
            could_not_read.append("the spread (a standard deviation takes two seeds)")
        if follows["name"] == NOT_READ:
            could_not_read.append("the branch for what follows")

    # ---- the lines
    lines = [f"{SAY}: L5, THE READ OVER {len(seeds)} SEED{'S' if len(seeds) > 1 else ''} ({_named_seeds(seeds)}), each from `{start}`. Seed {found_by} made the ONE deep "
             f"search and found the goal set's lead; the new seeds, of those here and read: {_named_seeds(new)}. Every difference is paired by problem, 95% bootstrap over problems"]
    for seed in seeds:
        entry = each[seed]
        if not entry["read"]:
            lines.append(f"{SAY}: seed {seed}: {entry['why_not']}: nothing is said of what its models did, and nothing is pooled over it")
            continue
        models, pairs = entry["models"], entry["pairs"]
        of_the_pair = pairs[pair_name(DEEP, PUBLISHED)]
        lines.append(f"{SAY}: seed {seed}" + (" (IT MADE THE SEARCH AND FOUND THE LEAD)" if entry["made_the_search"] else " (a new seed)") + ": per 1,000 attempts on the "
                     f"held-aside set ({len(ids['aside']):,} problems): " + ", ".join(f"`{name}` {models[name]['held_aside_per_1000']}" for name in everyone)
                     + f"; on all of G ({len(ids['goal'])}): " + ", ".join(f"`{name}` {models[name]['all_of_g_per_1000']}" for name in everyone)
                     + "; goal problems solved: " + ", ".join(f"`{name}` {models[name]['goal_problems_solved']}" for name in everyone)
                     + f"; on G' ({len(ids['again'])}) over the fresh attempts: " + ", ".join(f"`{name}` {models[name]['goal_set_again_per_1000']}" for name in everyone)
                     + f"; long problems (8 lines or more, {len(ids['long'])}) solved: " + ", ".join(f"`{name}` {models[name]['long_problems_solved']}" for name in everyone))
        lines.append(f"{SAY}: seed {seed}, `{DEEP}` minus `{PUBLISHED}`: on the held-aside set {_rate(of_the_pair['on_the_held_aside_set'])} (its report's primary: "
                     f"{entry['branches']['the_primary']}), problems solved {_split(of_the_pair['held_aside_solved'])}; on all of G {_rate(of_the_pair['on_all_of_g'])}, goal "
                     f"problems solved {_split(of_the_pair['goal_problems_solved'])}; on the {len(ids['not_again'])} goal problems NOT in G' "
                     f"{_rate(of_the_pair['on_the_goal_problems_not_in_the_goal_set_again'])}; on G' over all the attempts {_rate(of_the_pair['on_the_goal_set_again'])}; long "
                     f"problems solved {_split(of_the_pair['long_problems_solved'])}")
        for after, before in PAIRS[1:]:
            pair = pairs[pair_name(after, before)]
            lines.append(f"{SAY}: seed {seed}, `{after}` minus `{before}`: on the held-aside set {_rate(pair['on_the_held_aside_set'])}; on all of G {_rate(pair['on_all_of_g'])}, "
                         f"goal problems solved {_split(pair['goal_problems_solved'])}; long problems solved {_split(pair['long_problems_solved'])}")
        apart = entry["timed_out_in_lean"]["apart"]
        lines.append(f"{SAY}: seed {seed}, TIMEOUTS: two models' shares of attempts that timed out in Lean differ by more than a point "
                     + ("on " + "; ".join(f"{found['kind']} (`{found['models'][0]}` {found['shares'][0]}, `{found['models'][1]}` {found['shares'][1]})" for found in apart)
                        + ": their timed-out attempts are to be checked again (Lean only) and the reads given both ways, which this command does not do" if apart else "nowhere"))
    if pooled is None:
        unread = [seed for seed in seeds if why_not[seed] is not None]
        lines.append(f"{SAY}: NOT POOLED: seed{'s' if len(unread) > 1 else ''} {_named_seeds(unread)} ({', '.join(why_not[seed] for seed in unread)}): nothing is said of such "
                     "a run, and nothing is pooled over it")
    else:
        primary, lead, split, reach, follows = (pooled[key] for key in ("the_primary", "the_lead", "the_split", "reach_on_the_long_problems", "what_follows"))
        lines.append(f"{SAY}: THE PRIMARY, on the held-aside set, `{DEEP}` minus `{PUBLISHED}`, each problem's mean over the seeds {_named_seeds(read)} of its difference: "
                     f"{_rate(primary['difference'])}; at each seed: " + ", ".join(f"seed {seed} {value:+.5f}" for seed, value in sorted(primary["at_each_seed"].items()))
                     + f": {primary['name']}. {primary['reason']}")
        lines.append(f"{SAY}: THE GOAL SET'S LEAD, on the new seeds ALONE ({_named_seeds(new)}), all of G over all the attempts, `{DEEP}` minus `{PUBLISHED}`: "
                     f"{_rate(lead['difference'])}: {lead['name']}" + (f", {split['name']}" if split["name"] else "") + f". {lead['reason']}")
        lines.append(f"{SAY}: THE LEAD, BESIDE IT: goal problems solved at each seed, `{DEEP}` against `{PUBLISHED}`: "
                     + "; ".join(f"seed {seed} {_split(counted)}" for seed, counted in lead["goal_problems_solved_at_each_seed"].items())
                     + f"; every seed here pooled ({_named_seeds(read)}; seed {found_by} FOUND the lead and confirms nothing): {_rate(lead['every_seed_pooled'])}")
        for key, own in pooled["beside_the_lead"].items():
            lines.append(f"{SAY}: THE LEAD, BESIDE IT, the same two reads of `{key.replace('_minus_', '` minus `')}`, naming nothing: on the new seeds {_rate(own['difference'])} ("
                         + (", ".join(f"seed {seed} {value:+.5f}" for seed, value in sorted(own["at_each_new_seed"].items())) or "no new seed")
                         + "); goal problems solved at each seed: " + "; ".join(f"seed {seed} {_split(counted)}" for seed, counted in own["goal_problems_solved_at_each_seed"].items())
                         + f"; every seed here pooled: {_rate(own['every_seed_pooled'])}")
        lines.append(f"{SAY}: THE SPLIT of the lead, on the new seeds alone, `{DEEP}` minus `{PUBLISHED}` over all the attempts: (a) on the {len(ids['not_again'])} goal problems "
                     f"NOT in G' (`{start}` solves them in its first sampling): {_rate(split['on_the_goal_problems_not_in_the_goal_set_again'])}; (b) on G' ({len(ids['again'])}): "
                     f"{_rate(split['on_the_goal_set_again'])}: " + (split["name"] or "NO NAME") + f". {split['reason']}")
        lines.append(f"{SAY}: REACH ON THE LONG PROBLEMS ({len(ids['long'])} goal problems whose shortest published proof has 8 lines or more), on the new seeds alone: "
                     f"{reach['name']}. {reach['reason']}")
        lines.append(f"{SAY}: REACH, BESIDE IT: long problems solved at each seed: "
                     + "; ".join(f"seed {seed}: " + ", ".join(f"`{name}` {each[seed]['models'][name]['long_problems_solved']}" for name in MODELS)
                                 + f" (`{DEEP}` against `{PUBLISHED}` {_split(each[seed]['pairs'][pair_name(DEEP, PUBLISHED)]['long_problems_solved'])}; against `{SHALLOW}` "
                                 f"{_split(each[seed]['pairs'][pair_name(DEEP, SHALLOW)]['long_problems_solved'])})" for seed in read)
                     + f"; the same read of `{DEEP}` minus `{SHALLOW}`, naming nothing: {pooled['beside_the_reach'][pair_name(DEEP, SHALLOW)]['reason']}")
        for name in MODELS:
            lines.append(f"{SAY}: THE SPREAD of `{name}` over its seeds ({_named_seeds(read)}; the standard deviation is the sample's, n - 1): "
                         + "; ".join(f"{SPREAD_OF[figure]}: " + ", ".join(str(value) for value in own["by_seed"].values()) + f" (sd {own['standard_deviation']})"
                                     for figure, own in pooled["the_spread"][name].items()))
        lines.append(f"{SAY}: WHAT FOLLOWS: {follows['name']}. {follows['reason']}")
        if could_not_read:
            lines.append(f"{SAY}: NOT READ, for the seeds that are not here: " + "; ".join(could_not_read))
    return {
        "spec": SEEDS_SPEC, "stage": L4, "label": LABEL, "check": "L5",
        "what": "per seed: its own figures, computed again from its stored rows. Over the seeds: the primary on the held-aside set over the three seeds; on the NEW seeds "
                "alone the goal set's lead, its split and the reach on the long problems; the spread of each kind of model; the section's branch for what follows",
        "seeds": seeds, "seeds_read": read, "made_the_search": found_by, "new_seeds": new, "start": start,
        "sizes": {key: len(value) for key, value in ids.items()}, "by_seed": each, "pooled": pooled,
        "not_pooled": None if pooled is not None else {str(seed): why_not[seed] for seed in seeds if why_not[seed] is not None},
        "could_not_read": could_not_read, "lines": lines,
    }
