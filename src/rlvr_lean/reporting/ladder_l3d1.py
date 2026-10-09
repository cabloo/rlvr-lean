"""L3d Step 1's report: do assembled proofs teach? Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the
episode reaches", "The read, fixed before the run". Pure: rows in, a report out. Its pieces are the ceiling report's
(`reporting/ladder_ceiling.py`: the goal set by proof length, successes per attempt paired by problem, the lines of the
proofs a model verifies), read on other pairs of models.

One-shot attempts only. Every comparison is over the same problems and the same number of attempts on each, paired by
problem, with a 95% bootstrap interval over problems.

  primary      the goal problems whose shortest published proof is 4 lines or more: successes per attempt, `with`
               minus `without` (the ceiling's primary, so the two can be set side by side per proof trained on)
  noise floor  the same quantity, `without` minus the STORED three-round model: the same training examples, another
               run and another order. It is what two trainings on the same data differ by; which of the two is
               subtracted from the other is arbitrary, so it is read as a size (`l3d.noise_band`), and a primary is
               read as an effect only when its interval is clear of zero AND its point is outside that
  secondary    the same by length group and on all of G; the three rungs; goal problems solved at least once and
               reliably (in at least half of their episodes of 8 one-shot attempts, in the order drawn) BY ATTEMPTS
               ALONE, for `with`, `without`, the stored three-round model and the base; the lines of the proofs each
               model verifies; the share of its attempts whose first step is a `have`
  the checks   the four "can this run see a win" checks, each with its number and PASS or FAIL. Any FAIL makes the
               report INCONCLUSIVE, and it then says nothing else: what was measured is kept under
               `measured_and_not_read`
  the branch   in the spec's words

The counts WITH ASSEMBLY are not here: they need Lean, and `tools/ladder_goal_assembly.py` makes them after the run from
each model's stored goal attempts. THE LENGTH OF A HELD-OUT PROBLEM'S PUBLISHED PROOF is read here and nowhere before.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

from rlvr_lean.domain.ladder_round import reliable
from rlvr_lean.domain.ladder_round.ceiling import interval_text
from rlvr_lean.domain.ladder_round.l3d import ARMS, H0, INCONCLUSIVE, L3D1, WITH, WITHOUT, can_this_run_see_a_win, l3d1_branch, noise_band
from rlvr_lean.domain.ladder_round.read import GOAL, group_ids, paired_change
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.reporting.ladder_ceiling import ALL, FOUR_PLUS, LOOP, RUNG_PART, _named, _rate, attempts_on_g, goal_sets, lean_did_not_answer, per_attempt, solved, verified_lines
from rlvr_lean.reporting.ladder_l2 import _health, _split

BASE = "base"
PRIMARY, NOISE_FLOOR = "with_minus_without", "noise_floor"
# Every pair that is read: (the model, what it is compared with). The first is the primary's, the second the noise floor's.
PAIRS = {PRIMARY: (WITH, WITHOUT), NOISE_FLOOR: (WITHOUT, LOOP), "with_minus_base": (WITH, BASE), "without_minus_base": (WITHOUT, BASE), "loop_minus_base": (LOOP, BASE)}


def model_names(prepare: Mapping, models: Mapping[str, Mapping]) -> dict[str, str]:
    rate = prepare.get("loop_target_rate")
    at = f" at t = {Fraction(str(rate))}" if rate is not None else ""
    return {BASE: "the base", **{name: f"the stored three-round model{at}" if name == LOOP else f"`{name}`" for name in models}}


def by_attempts_alone(models: Mapping[str, Mapping], sets: Mapping[str, Sequence[str]]) -> dict:
    """For each model, the goal problems solved at least once, in a quarter, in half (reliably) and in nine tenths of
    their episodes of 8 one-shot attempts, by attempts alone (`reliable.reliability`), on each set of G."""
    return {name: {group: reliable.reliability(entry["episodes_of_8"], ids) for group, ids in sets.items()} for name, entry in models.items()}


def build_l3d1_report(prepare: Mapping, trains: Mapping[str, Mapping], losses: Mapping[str, Mapping], orders: Mapping[str, Sequence[Mapping]],
                      groups: Sequence[Mapping], lengths: Mapping[str, Mapping], base: Mapping, models: Mapping[str, Mapping], settings: Mapping,
                      evaluation: Mapping, ceiling: Mapping[str, Mapping] | None = None) -> dict:
    """`prepare`: that step's summary. `trains` and `losses`: by model, its training step's summary and its stored
    loss file (`row_losses`, `rows_trained`). `orders`: by model, the rows the prepare step fixed (`id`, `source`), in
    order. `groups`: the held-out groups. `lengths`: by problem, its shortest published proof's length group. `base`
    and each of `models` (`without`, `with`, and `loop` when a stored three-round model was read): `rungs` (per-problem
    rows on the three rungs), `goal` (one list of per-problem rows for each sampling of G), and what the model wrote
    (`verified_proof_lines`, `distinct_attempts`, `opens_with_a_have`, `episodes_of_8`). `settings`:
    `ladder_loop.l3d.step_1`. `ceiling`: the ceiling's two doses (`{rows, mean}` each), when its report was read."""
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
    distinct = {name: entry["distinct_attempts"] for name, entry in who.items()}

    # ---- the checks; then the branch
    h0_count = prepare["h0_rows"]
    checks = can_this_run_see_a_win(h0_count, prepare["minimum_h0"], {arm: losses[arm]["row_losses"] for arm in ARMS}, {arm: models[arm][RUNG_PART] for arm in ARMS},
                                    orders, {arm: losses[arm]["rows_trained"] for arm in ARMS}, settings)
    primary, floor = by_length[PRIMARY][FOUR_PLUS], by_length[NOISE_FLOOR][FOUR_PLUS] if NOISE_FLOOR in by_length else None
    branch = l3d1_branch(checks, primary, floor, h0_count, ceiling)
    inconclusive, band = branch["name"] == INCONCLUSIVE, noise_band(floor)

    # ---- whether Lean answered (the health of each set: not the third check, which is about the models)
    measured_sets = {}
    for name, entry in who.items():
        measured_sets[f"rungs_{name}"] = entry[RUNG_PART]
        measured_sets.update({f"goal_{name}_{index}": rows for index, rows in enumerate(entry[GOAL], start=1)})
    health = {name: _health(rows) for name, rows in measured_sets.items()}
    not_to_be_read = sorted(name for name, rows in measured_sets.items() if lean_did_not_answer(rows))

    attempts, rows_of = prepare["attempts_a_goal_problem"], {arm: prepare["trainings"][arm]["rows"] for arm in ARMS}
    said = lambda name: f"{names[pairs[name][0]]} minus {names[pairs[name][1]]}"      # noqa: E731
    measured = {
        "primary": {"what": f"the goal problems whose shortest published proof is 4 lines or more: successes per attempt over {attempts} one-shot attempts a "
                            "problem, `with` minus `without`, paired by problem, a 95% bootstrap interval over problems", **primary},
        "noise_floor": None if floor is None else {
            "what": f"the same quantity, `without` minus {names[LOOP]} (the same training examples; another run and another order): what two trainings on the same "
                    "data differ by. Which is subtracted from which is arbitrary, so it is read as a size, `band`: the end of its interval farthest from zero. "
                    "No primary whose point is within it is read as an effect", "band": None if band is None else round(band, 5), **floor},
        "secondary": {
            "by_length_group": {"what": "the primary's quantity (successes per attempt, paired by problem) by the length group of each goal problem's shortest "
                                        "published proof (1, 2-3, 4-7, 8+ lines), on the problems of 4 lines or more and on all of G; for every pair of models read",
                                "problems": {group: len(ids) for group, ids in sets.items()}, "pairs": {name: {"pair": said(name), **by_length[name]} for name in pairs}},
            "the_three_rungs": {"what": f"fresh pass rate on each held-out rung ({prepare['rung_episodes']} episodes a problem), paired by problem; for every pair read",
                                "problems": {rung: len(rung_ids[rung]) for rung in RUNGS}, "pairs": {name: {"pair": said(name), **rungs[name]} for name in pairs}},
            "goal_problems_solved_by_attempts_alone": {
                "what": "each model's stored one-shot attempts at a goal problem cut into episodes of 8 in the order drawn (each sampling on its own; the last "
                        "few attempts are left over); an episode is resolved when one of its 8 verified. Goal problems resolved in at least one, a quarter, "
                        "half (RELIABLY) and nine tenths of their episodes. No Lean here: the counts with assembly come from tools/ladder_goal_assembly.py",
                **{name: {"model": names[name], **alone[name]} for name in who}},
            "goal_problems_solved_with_against_without": {"what": f"goal problems solved at least once in their {attempts} attempts: gained = by `with` and not by "
                                                                  "`without`, lost = the reverse; two-sided sign test; by length group", **solved_on_g},
            "verified_proof_lines": {"what": "the line counts of the proofs each model verifies (lines that are neither empty nor a comment), on G over all its "
                                             "attempts and on the three rungs: how many, the median, the mean, the shares with 4 lines or more and with 8 or "
                                             "more, the longest", **lines_verified},
            "opens_with_a_have": {"what": "the share of a model's attempts (every one, whatever became of it) whose first step is a `have`, on the three rungs and "
                                          "on G: whether it writes differently", **have},
            "distinct_attempts": {"what": "the share of a model's attempts at one problem and side that are distinct, averaged over problems and sides", **distinct}}}

    # ---- the lines, in the spec's order
    heading = (f"{L3D1}: L3D STEP 1, DO ASSEMBLED PROOFS TEACH? seed {prepare['seed']}. Two models from the base, one pass each, the round's recipe: `without` on "
               f"the {rows_of[WITHOUT]:,} training examples the three rounds stored, in a seeded order; `with` on the same rows in the same relative order and "
               f"H0's {h0_count:,} assembled proofs at seeded places among them ({rows_of[WITH]:,} rows). Both measured in one-shot attempts")
    lines = [heading]
    if prepare.get("stored_runs") is None:
        lines.append(f"{L3D1}: L2's runs were not read (a smoke run): `without` is the L1 run's own training examples, G was attempted {attempts} times a problem "
                     "and no three-round model stands beside (no noise floor)")
    if not inconclusive:
        counted = lambda entry: (f" ({entry['successes']} successes against {entry['successes_of_the_base']} in {entry['attempts_each']:,} attempts each)"      # noqa: E731
                                 if entry.get("mean") is not None else "")
        lines.append(f"{L3D1}: PRIMARY. The goal problems whose shortest published proof is 4 lines or more ({primary['problems']} problems), successes per attempt "
                     f"over {attempts} one-shot attempts a problem, `with` minus `without`, paired by problem, 95% bootstrap over problems: {_rate(primary)}"
                     + counted(primary))
        lines.append(f"{L3D1}: THE NOISE FLOOR, the same quantity, `without` minus {names[LOOP]} (the same training examples, another run and order): {_rate(floor)}"
                     f"{counted(floor)}. Two trainings on the same data differ here by up to {band:.5f} either way, and no primary whose point is within that is "
                     "read as an effect" if floor is not None and floor.get("mean") is not None else f"{L3D1}: THE NOISE FLOOR: no stored three-round model was read")
        for name in pairs:
            lines.append(f"{L3D1}: SECONDARY, the same by the length of the shortest published proof, {said(name)}: "
                         + "; ".join(f"{_named(group)} ({len(ids)}): {_rate(by_length[name][group])}" for group, ids in sets.items()))
        for name in (PRIMARY, NOISE_FLOOR):
            if name in pairs:
                lines.append(f"{L3D1}: SECONDARY, the three rungs ({prepare['rung_episodes']} episodes a problem), {said(name)}: "
                             + "; ".join(f"{rung} {interval_text(rungs[name][rung])}" for rung in RUNGS))
        for group in (ALL, FOUR_PLUS):
            lines.append(f"{L3D1}: SECONDARY, goal problems solved at least once / reliably (in at least half of their episodes of 8 one-shot attempts), BY "
                         f"ATTEMPTS ALONE, {_named(group)} ({len(sets[group])}): "
                         + "; ".join(f"{names[name]} {alone[name][group]['solved_at_least_once']} / {alone[name][group]['reliably']}" for name in who)
                         + ". With assembly: not in this report (tools/ladder_goal_assembly.py, after the run)")
        lines.append(f"{L3D1}: SECONDARY, goal problems solved at {attempts} attempts, `with` against `without`: "
                     + "; ".join(f"{_named(group)}: {solved_on_g[group]['resolved_after']} to {solved_on_g[group]['resolved_before']}, {_split(solved_on_g[group])}"
                                 for group in (ALL, FOUR_PLUS)))
        for part, where in ((GOAL, "on G"), (RUNG_PART, "on the rungs")):
            lines.append(f"{L3D1}: SECONDARY, the lines of the proofs each model verifies {where} (proofs, median, mean, share with 4 lines or more, with 8 or more, "
                         "longest): " + "; ".join(
                             f"{names[name]} {entry[part]['proofs']}, {entry[part]['median']}, {entry[part]['mean']}, {entry[part]['share_with_4_lines_or_more']}, "
                             f"{entry[part]['share_with_8_lines_or_more']}, {entry[part]['longest']}" for name, entry in lines_verified.items()))
        lines.append(f"{L3D1}: SECONDARY, the share of attempts whose first step is a `have` (on the rungs, on G): "
                     + "; ".join(f"{names[name]} {entry[RUNG_PART]['share']}, {entry[GOAL]['share']}" for name, entry in have.items()))
    enough, took, writes, trained = (checks[key] for key in ("h0_holds_enough_proofs", "both_trainings_took", "both_models_still_write_proofs", "with_was_trained_on_h0"))
    verdict = lambda check: "PASS" if check["passes"] else "FAIL"      # noqa: E731
    lines += [
        f"{L3D1}: CHECK 1, H0 holds enough proofs: {enough['h0_rows']:,}; at least {enough['minimum']} is asked: {verdict(enough)}",
        f"{L3D1}: CHECK 2, both trainings took (the mean training loss over the last tenth of the rows against the first tenth): "
        + "; ".join(f"`{arm}` {took[arm]['last']} against {took[arm]['first']} over {took[arm]['rows_compared']} rows" for arm in ARMS) + f": {verdict(took)}",
        f"{L3D1}: CHECK 3, both models still write proofs (attempts on the three rungs without an answer; under {settings['maximum_share_without_an_answer']} is asked): "
        + "; ".join(f"`{arm}` {writes[arm]['share']} of {writes[arm]['attempts']:,} ({writes[arm]['capped_at_the_token_limit']} at the token cap, "
                    f"{writes[arm]['without_a_verdict_from_lean']} without a verdict from Lean)" for arm in ARMS) + f": {verdict(writes)}",
        f"{L3D1}: CHECK 4, `with` was trained on H0: {trained['counted_once_in_with']:,} of H0's {trained['h0_rows']:,} rows are counted once in the record of the "
        f"{trained['rows_trained'][WITH]:,} rows its training steps were made on, {trained['counted_in_without']} in `without`'s {trained['rows_trained'][WITHOUT]:,}; "
        f"each record is the prepared order: {'yes' if all(trained['as_prepared'].values()) else 'NO'}: {verdict(trained)}",
        f"{L3D1}: {'INCONCLUSIVE' if inconclusive else 'BRANCH: ' + branch['name']}. {branch['reason']}"]
    note = branch.get("could_have_been_seen")
    if note and note["doses"]:
        lines.append(f"{L3D1}: beside it, the ceiling's two doses (its primary over the published proofs it was trained on, times H0's {h0_count:,}): "
                     + "; ".join(f"after {dose['rows']:,} proofs {dose['gain']:+.5f}, so {dose['expected_here']:+.6f} expected here "
                                 f"({'could' if dose['could_have_been_seen'] else 'could not'} have been seen)" for dose in note["doses"].values())
                     + f"; this run resolves {note['resolves']:.5f}")
    if not_to_be_read:
        lines.append(f"{L3D1}: NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}. The step fails; the task queued again "
                     "measures such a set of this run's own again from the kept adapter")
    headline = (f"L3d Step 1 (do assembled proofs teach?) seed {prepare['seed']}: {branch['name']}. "
                + ("" if inconclusive else f"Primary (goal problems with a published proof of 4 lines or more, {primary['problems']} problems, {attempts} attempts each, "
                                           f"`with` minus `without`, H0 = {h0_count:,} proofs): {_rate(primary)}; the noise floor (`without` minus the stored "
                                           f"three-round model): {_rate(floor) if floor is not None else 'not read'}. ")
                + f"Checks: H0 {verdict(enough)} ({enough['h0_rows']}), both trainings took {verdict(took)}, both still write proofs {verdict(writes)}, `with` trained "
                  f"on H0 {verdict(trained)} ({trained['counted_once_in_with']} of {trained['h0_rows']}). " + branch["reason"]
                + (f" NOT TO BE READ: too many attempts without a verdict from Lean in {', '.join(not_to_be_read)}" if not_to_be_read else ""))
    return {
        "spec": "docs/spec/ladder-loop.spec.md, L3d: train on what the episode reaches (Step 1)", "stage": L3D1, "headline": headline, "lines": lines,
        "ok": not not_to_be_read, "seed": prepare["seed"],
        "stand_in_engine": bool(prepare.get("stand_in_engine") or any(trains[arm].get("stand_in_engine") or models[arm].get("stand_in_engine") for arm in ARMS)),
        "branch": branch, "inconclusive": inconclusive,
        "can_this_run_see_a_win": {"what": "the four checks the spec reads before the branch; when any fails the run is INCONCLUSIVE and nothing else is said", **checks},
        **({"primary": None, "noise_floor": None, "secondary": None,
            "measured_and_not_read": {"why": "a check failed: the run is INCONCLUSIVE, and these numbers say nothing about what assembled proofs teach. They are "
                                             "kept for whoever repairs the run", **measured}} if inconclusive else measured),
        "the_ceilings_doses": ceiling,
        "models": {BASE: "the base model, from stored attempts: nothing here measured it again", **{name: names[name] for name in models}},
        "trainings": {arm: {key: value for key, value in trains[arm].items() if key != "checkpoints"} for arm in ARMS},
        "training": {"recipe": prepare["recipe"], "order": prepare["order"], "rounds_files": prepare["rounds_files"], "rounds_rows": prepare["rounds_rows"],
                     "h0_file": prepare["h0_file"], "h0_file_sha256": prepare["h0_file_sha256"], "h0_rows": h0_count, "h0_sides": prepare["h0_sides"],
                     "trainings": prepare["trainings"], "longest_example_tokens": prepare["longest_example_tokens"],
                     "max_sequence_tokens": prepare["max_sequence_tokens"], "h0_rows_in_with": sum(row["source"] == H0 for row in orders[WITH])},
        "heldout": {"goal_set": len(goal_ids), "goal_set_by_length_group": {group: len(ids) for group, ids in sets.items()},
                    "rungs": {rung: len(rung_ids[rung]) for rung in RUNGS}},
        "sizes": {key: prepare[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "stored_runs", "loop_arm",
                                                "stored_measurements", "contradicted_side_setting")},
        "attempts": health, "not_to_be_read": not_to_be_read,
        "not_measured": ["the base and the stored three-round model (their rows are the stored ones of the runs named under sizes.stored_runs)",
                         "anything with assembly: the counts by episode with assembly are made after the run, Lean only (tools/ladder_goal_assembly.py)",
                         "the matched control (`without` and as many more one-shot proofs as H0 holds): trained only if this run reads ASSEMBLED PROOFS TEACH",
                         "the standard evaluation of Phase A (its statements are not built at v4.27)"],
    }
