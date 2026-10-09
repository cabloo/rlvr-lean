"""L4: the loop from a model PRETRAINED ON PUBLISHED PROOFS. Spec: docs/spec/ladder-loop.spec.md, "L4: the loop from a
model pretrained on published proofs" and its "Made exact by the build". Two stages, each step its own process
(`rlvr_lean.runner.entry`). Everything they produce is labelled pretrained on published proofs: distillation of other
provers, followed by the loop. The training text is other people's published proofs and is not distributed.

STAGE `ladder_l4_pretrain`, run directory `ladder_l4_pretrain_seed<seed>`: the ceiling stage's shape with one model.

  ladder_l4_pretrain_prepare   reads on the box what the ceiling reads (L1's run; L2's two), and the pretraining file
                               from the package's data (`data/ladder_l4/pretraining.jsonl`): REFUSED when it is missing, when a
                               row is a held-out or base-map problem, is not of the `pretrain` half, is not on the side
                               `true`, repeats a problem, or would be cut as a training example. No GPU, no Lean
  ladder_l4_pretrain_train     from the base, the round's recipe, ONE pass over every row in the file's order
                               (`ladder_ceiling.one_pass`); every row's loss; the adapter `pre`, KEPT (`adapters/pre`)
  ladder_l4_pretrain_measure   `pre` as the ceiling's models are (`measure_model`): 8 episodes on the rungs, 93 attempts
                               on every goal problem with L2's sampling seeds
  ladder_l4_pretrain_map       `pre`'s OWN MAP: the base map's problems attempted again by `pre`, with the sides, the
                               number of attempts and the sampling seed of the stored map (the base's); its rows are
                               stored in the shape the challenger reads the base's (`l4_map_pre.jsonl`), and the arm's
                               challenger starts from them. Block-resumable, as a measurement is
  ladder_l4_pretrain_report    `pre` against the base (`reporting/ladder_l4.py`), the goal set drawn again (G': the
                               goal problems `pre` does not solve in its 32-attempt sampling; its ids are stored for
                               the arm), the first two "can this run see a win" checks, and the map beside the base's

STAGE `ladder_l4`, the arm's run directory (`ladder_l2_t010_assembly_pre_seed<seed>`): L3d Step 2's steps again for the
arm `l4.arm` (`gpu/ladder_l2.py`, `gpu/ladder_assembly.py`, `gpu/ladder_l3d2.py`: six rounds with assembly, the twin, the
two measurements), every model of it trained FROM `pre` and round 1 attempted by `pre`, its candidates the `loop` half of
the pool, no H0; with two steps of its own:

  ladder_l4_prepare            (right after the arm's prepare step, before anything is sampled) REFUSES a pretraining
                               run whose report is not written or whose two checks failed, and a missing `pre`
                               adapter; copies into the arm's run what its report reads of that run (G', `pre`'s
                               stored rows) and, when it is on the box, of the base arm's run
  ladder_l4_report             the read fixed before the run (`reporting/ladder_l4.py`)

Every adapter of both stages is kept: `pre` is the arm's starting model, and the arm's models are this arm's own.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from collections import Counter

from rlvr_lean.data.heldout_proof_lines import length_group
from rlvr_lean.data.ladder_round_export import BASE_MAP_SET
from rlvr_lean.domain.ladder_round.ceiling import CHECKPOINTS, check_examples_fit, the_training_took, training_example, training_rows
from rlvr_lean.domain.ladder_round.l3d import ARMS, WITHOUT, tenth
from rlvr_lean.domain.ladder_round.l4 import MAP_FILE, MAP_STEP, PRE, PRETRAIN, map_rows, map_summary, refuse_the_other_half
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.problem_pool.episodes import side_plan
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, ladder_l3d2, ladder_loop, ladder_round, pipeline
from rlvr_lean.gpu.ladder_ceiling import (
    LOOP,
    MORE,
    REACH,
    RUNG_PART,
    Model,
    Reading,
    StoredRuns,
    _file_rows,
    _need,
    example_tokens,
    forget_measurement,
    goal_set,
    measure_model,
    measured_sizes,
    one_pass,
    read_stored_runs,
    rung_set,
    stored_file,
    what_a_model_wrote,
    write_stored_runs,
)
from rlvr_lean.gpu.ladder_dose import _episodes, _rows, _runs
from rlvr_lean.gpu.ladder_l3d1 import WROTE, _wrote, what_else_a_model_wrote
from rlvr_lean.gpu.ladder_round import BASE, Engines, _as_sets, _results, _stand_in, _write_json, add_problems, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.reporting.ladder_ceiling import lean_did_not_answer

L4 = "l4"
LABEL = "pretrained on published proofs"
PRETRAIN_STAGE, ARM_STAGE = "ladder_l4_pretrain", "ladder_l4"
RUN_VARIABLE = ladder_l2.L4_PRETRAIN_RUN_VARIABLE       # another run directory of the pretraining stage than `ladder_l4_pretrain_seed<seed>` (a smoke run)
SOURCE_VARIABLE = "RLVR_LEAN_LADDER_L4_SOURCE"          # another L1 run directory to read than `ladder_l1_seed<seed>`
STORED_VARIABLE = "RLVR_LEAN_LADDER_L4_STORED"          # `none`: L2's stored runs are not read (a smoke run)
FILE_VARIABLE = "RLVR_LEAN_LADDER_L4_FILE"              # another pretraining file than the package's (the fixture)
MINIMUMS_VARIABLE = "RLVR_LEAN_LADDER_L4_MINIMUMS"      # other minimums for the two checks than the settings', as `solved,again` (a smoke run: `0,0`)
PACKAGE_FILE = Path(__file__).resolve().parents[1] / "data" / "ladder_l4" / "pretraining.jsonl"
SETTING = "ladder_loop.l4.loop_arm"
WHAT = "L4's pretraining file"
HOW = ("it is built on the dev machine by `tools/ladder_ceiling_set.py --file l4_pretrain` (from the pool's own certificates; its control is run with "
       "`--control`) and committed with the code")
PREPARE, TRAIN, MEASURE, REPORT = (f"{PRETRAIN_STAGE}_{name}" for name in ("prepare", "train", "measure", "report"))
MAP = MAP_STEP                                          # `pre`'s own map: the step, and its marker (the arm reads it by this name)
MAP_EPISODES = f"{L4}_map_{PRE}"                        # the set its episodes are stored under
MAP_PROBLEMS_FILE = f"{L4}_map_problems.json"           # what was recorded when the map's problems were built: their sides against the stored map's
ARM_PREPARE, ARM_REPORT = f"{ARM_STAGE}_prepare", f"{ARM_STAGE}_report"
REPORT_FILE, ARM_REPORT_FILE = "report_ladder_l4_pretrain.json", "report_ladder_l4.json"
STORED_MODELS_FILE, GROUPS_FILE, LOSS_FILE, ROWS_FILE = f"{L4}_stored_models.json", f"{L4}_heldout_groups.jsonl", f"{L4}_pretrain_loss.json", f"{L4}_pretraining_rows.jsonl"
AGAIN_FILE, PRE_FILE = f"{L4}_goal_set_again.jsonl", f"{L4}_stored_pre.json"
BASE_ARM = "base_arm"                                   # the base arm's last model, from that arm's stored run
READING = Reading(reader="L4", models="L4's models", not_read="that set is not to be read, and no report of this run could be read against it", setting=SETTING)
MAX_LEFTOVER_GB = pipeline.MAX_LEFTOVER_GB


def _say(message: str) -> None:
    print(f"{L4} ({LABEL}): {message}", flush=True)


def pretrain_directory(config: dict) -> Path:
    return _runs(config) / (os.environ.get(RUN_VARIABLE) or f"{PRETRAIN_STAGE}_seed{training_seed(config)}")


def _store(config: dict) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    return ArtifactStore(pretrain_directory(config), Path(mirror) if mirror else None)


def source_directory(config: dict) -> Path:
    return _runs(config) / (os.environ.get(SOURCE_VARIABLE) or f"ladder_l1_seed{training_seed(config)}")


def pretraining_file() -> Path:
    override = os.environ.get(FILE_VARIABLE)
    return Path(override) if override else PACKAGE_FILE


def minimums(config: dict) -> tuple[int, int]:
    """(the goal problems `pre` must solve, the problems G' must hold): the settings', or a smoke run's own."""
    override, own = os.environ.get(MINIMUMS_VARIABLE), config["ladder_loop"]["l4"]
    return tuple(int(part) for part in override.split(",")) if override else (own["minimum_solved_by_pre"], own["minimum_goal_set_again"])


def read_the_ceiling(config: dict, read: StoredRuns) -> dict:
    """The ceiling's two models, from what its run of this seed stored ON THIS BOX (`ladder_ceiling_seed<seed>`: read,
    never written), when its report is there and can be read: for each, its per-problem rows on the rungs and on G's
    samplings (`kept`) and what it wrote (`wrote`), read as the stored three-round model is. They stand beside `pre`
    in the report and decide nothing. A run that is not there, a report that is not to be read, or a set sampled
    otherwise than this run samples: not read, with the reason."""
    directory = _runs(config) / f"ladder_ceiling_seed{training_seed(config)}"
    report, prepared = directory / ladder_ceiling.REPORT_FILE, directory / f"{ladder_ceiling.PREPARE}.done.json"
    if not report.exists() or not prepared.exists():
        return {"read": False, "why": f"{report} is not on this box"}
    if not json.loads(report.read_text()).get("ok"):
        return {"read": False, "why": f"{report} is not to be read"}
    rows_seen, measure = json.loads(prepared.read_text())["checkpoints"], config["ladder_loop"]["measure"]
    ids = {RUNG_PART: [row["problem_id"] for row in read.rung_problems], GOAL: [row["problem_id"] for row in read.goal_problems]}
    parts = [(RUNG_PART, RUNG_PART, measure["rung_episodes"]), *((sampling["name"], GOAL, sampling["episodes"]) for sampling in read.samplings)]
    kept, wrote, models = {}, {}, {}
    try:
        for name in CHECKPOINTS:
            who = ceiling_name(name)
            sets = {part: rung_set(name) if part == RUNG_PART else goal_set(part, name) for part, _, _ in parts}
            lacking = ladder_ceiling._missing(directory, sets.values())
            if lacking:
                raise RuntimeError(f"{directory} does not hold {lacking}")
            for part, of, episodes in parts:
                kept[who, part] = ladder_ceiling._stored(directory, sets[part], ids[of], read.seeds[part], episodes, f"the ceiling's model {name} ({part})", READING)[0]
            on_rungs = ladder_ceiling._stored_attempts(directory, sets[RUNG_PART])
            by_sampling = [ladder_ceiling._stored_attempts(directory, sets[sampling["name"]]) for sampling in read.samplings]
            wrote[who] = {**what_a_model_wrote(on_rungs, [attempt for attempts in by_sampling for attempt in attempts]), **what_else_a_model_wrote(on_rungs, by_sampling)}
            models[who] = {"rows": rows_seen[name]["rows_seen"]}
    except (RuntimeError, ValueError, KeyError, OSError) as error:       # a run that is not needed never stops the pretraining
        return {"read": False, "why": str(error)}
    return {"read": True, "run": str(directory), "models": models, "kept": kept, "wrote": wrote}


def ceiling_name(checkpoint: str) -> str:
    return f"ceiling_{checkpoint}"


# --------------------------------------------------------------------------------------- the pretraining
def ladder_l4_pretrain_prepare(config: dict) -> dict:
    store = _store(config)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    seed, source, stored = training_seed(config), source_directory(config), ladder_ceiling.stored_directories(config, STORED_VARIABLE, SETTING)
    read = read_stored_runs(config, source, stored, READING)
    path = pretraining_file()
    rows, sha256 = _file_rows(path, WHAT, HOW)
    heldout, base_map, _ = ladder_loop.load_problems(ladder_loop.data_directory())
    chosen = training_rows(rows, {row["problem_id"] for row in heldout} | {row["problem_id"] for row in read.groups}, {row["problem_id"] for row in base_map},
                           len(rows), WHAT)             # every row is trained on; a held-out, base-map, other-side or repeated row is refused
    if not chosen:
        raise ValueError(f"{WHAT} holds no row: refused")
    half_seed = config["ladder_loop"]["l4"]["half_seed"]
    refuse_the_other_half(chosen, PRETRAIN, half_seed, WHAT)
    count, counted_by = example_tokens(config)
    tokens, training = [count(training_example(row)) for row in chosen], config["training"]
    check_examples_fit([row["problem_id"] for row in chosen], tokens, training["max_sequence_tokens"], WHAT)
    batch = training["effective_batch"]
    ceiling = read_the_ceiling(config, read) if stored else {"read": False, "why": "L2's stored runs are not read (a smoke run), and the ceiling's is not either"}
    write_stored_runs(store, read, (PRE,), L4)
    for (who, part), own_rows in ceiling.pop("kept", {}).items():
        store.write_rows(stored_file(who, part, L4), own_rows)
    _write_json(store, STORED_MODELS_FILE, {"stage": L4, "label": LABEL, **{who: _wrote(read, who) for who in read.wrote}, **ceiling.pop("wrote", {})})
    # The rows the run trains on, WITHOUT their statements and proofs: the published proofs stay in the training file.
    store.write_rows(ROWS_FILE, [{"row": position, "problem_id": row["problem_id"], "kind": row["kind"], "proof_lines": row["proof_lines"],
                                  "length_group": length_group(row["proof_lines"]), "tokens": tokens[position]} for position, row in enumerate(chosen)])
    arm = ladder_ceiling.loop_arm(config, SETTING) if stored else None
    count_by = lambda key: {name: sum(key(row) == name for row in chosen) for name in sorted({key(row) for row in chosen})}      # noqa: E731
    summary = {"stage": L4, "label": LABEL, "seed": seed, "source_run": str(source), "stored_runs": {who: str(directory) for who, directory in stored.items()} if stored else None,
               "loop_arm": arm, "loop_target_rate": config["ladder_loop"]["l2_arms"][arm].get("target_rate") if arm else None, "stand_in_engine": _stand_in(),
               "ceiling": ceiling, "pretraining_file": str(path), "pretraining_file_sha256": sha256, "rows": len(chosen), "half": PRETRAIN, "half_seed": half_seed,
               "rows_by_kind": count_by(lambda row: row["kind"]), "rows_by_proof_lines": count_by(lambda row: length_group(row["proof_lines"])),
               "tokens": sum(tokens), "longest_example_tokens": max(tokens), "tokens_counted_by": counted_by, "max_sequence_tokens": training["max_sequence_tokens"],
               "order": "the file's: no row is moved", "effective_batch": batch, "steps": -(-len(chosen) // batch),
               "recipe": {"what": "the round's (`ladder_round._train`), from the base: one pass, the native format, the round's own example builder",
                          "learning_rate": training["learning_rate"], "warmup_steps": training["warmup_steps"], "effective_batch": batch,
                          "lora": config["lora"], "adapter_seed": seed},
               "minimums": dict(zip(("goal_problems_solved_by_pre", "goal_set_again"), minimums(config))), **measured_sizes(config, read)}
    store.mark_done(PREPARE, summary)
    _say(f"prepared, seed {seed}: {len(chosen)} rows of {path.name} in the file's order, {summary['steps']} optimizer steps; {summary['attempts_a_goal_problem']} "
         f"attempts on each of {len(read.goal_problems)} goal problems")
    return summary


def ladder_l4_pretrain_train(config: dict) -> dict:
    """The pretraining: from the base, one pass over the file's rows in its order; the adapter `pre` is kept."""
    store = _store(config)
    if store.is_done(TRAIN):
        return store.done_summary(TRAIN)
    _need(store, PREPARE, "the pretraining", PRETRAIN_STAGE)
    prepared, seed = store.done_summary(PREPARE), training_seed(config)
    rows, sha256 = _file_rows(pretraining_file(), WHAT, HOW)
    if sha256 != prepared["pretraining_file_sha256"]:
        raise RuntimeError(f"{WHAT} has SHA-256 {sha256} and this run was prepared on {prepared['pretraining_file_sha256']}: a run trains on the file its prepare step checked")
    examples, batch = [training_example(row) for row in rows], prepared["effective_batch"]
    _say(f"pretraining from the base on {len(examples)} published proofs in the file's order, {prepared['steps']} optimizer steps")
    result, step_rows, row_losses, _ = one_pass(config, examples, {PRE: prepared["steps"]}, seed, batch, store.root / ladder_assembly.ADAPTERS, f"{PRETRAIN_STAGE}_seed{seed}")
    result.pop("checkpoints")
    _write_json(store, LOSS_FILE, {
        "stage": L4, "label": LABEL, "seed": seed, "stand_in_engine": _stand_in(), "rows": len(row_losses), "steps": result["steps"], "effective_batch": batch,
        "what": "row_losses: each training row's mean loss per target token, in the file's order, read BEFORE the update of the optimizer step it was in",
        "row_losses": row_losses, "training_steps": [{**row, "rows_seen": min(row["step"] * batch, len(row_losses))} for row in step_rows]})
    took = the_training_took(row_losses, tenth(len(row_losses), config["ladder_loop"]["l4"]["loss_share_of_rows"]))
    summary = {"stage": L4, "label": LABEL, "model": PRE, "seed": seed, "rows": len(examples), "passes": 1, "order": prepared["order"], "stand_in_engine": _stand_in(),
               "adapter": str(store.root / ladder_assembly.ADAPTERS / PRE), "first_step_loss": step_rows[0]["mean_loss"], "last_step_loss": step_rows[-1]["mean_loss"],
               "mean_loss_over_the_first_rows": took["first"], "mean_loss_over_the_last_rows": took["last"], "rows_compared": took["rows_compared"],
               "the_training_took": took["passes"], **result}
    store.mark_done(TRAIN, summary)         # the adapter is saved, and kept
    if summary.get("allocated_after_cleanup_gb", 0) > MAX_LEFTOVER_GB:
        raise RuntimeError(f"{summary['allocated_after_cleanup_gb']} GB is still allocated on the GPU after the pretraining was released")
    return summary


def ladder_l4_pretrain_measure(config: dict) -> dict:
    store = _store(config)
    _need(store, PREPARE, "the measurement of L4's pretrained model `pre`", PRETRAIN_STAGE)
    prepared = store.done_summary(PREPARE)
    return measure_model(config, store, prepared, Model(
        name=PRE, called="the pretrained model `pre`", whose="L4's pretrained model `pre`", marker=MEASURE, trained_by=TRAIN,
        adapter=store.root / ladder_assembly.ADAPTERS / PRE,
        no_adapter="the adapter `pre` is kept by this stage, and one trained again would not be the model its other measurements came from. The measurement cannot be made",
        detail=f"{prepared['rows']} published proofs", summary={"stage": L4, "label": LABEL, "model": PRE, "rows": prepared["rows"]},
        stage=PRETRAIN_STAGE, label=L4, also=what_else_a_model_wrote))


def stored_map(config: dict) -> tuple[list[dict], list[dict], bool]:
    """(the base map's problems; the stored map's rows, the BASE's, as L2's challenger reads them; whether this is a
    fixture, whose rows are hand-made). From the package's data: `data/ladder_l0` and the round data L2 reads."""
    data = ladder_round._data(config, ladder_l2.data_directory())
    return data["base_map"], [row for row in data["base_results"] if row["set"] == BASE_MAP_SET], bool(data["summary"].get("fixture"))


def ladder_l4_pretrain_map(config: dict) -> dict:
    """`pre`'s OWN MAP (spec, "And `pre`'s own map"): the base map's problems attempted again by `pre`, with what the
    stored map (the base's) was made with: the same problems, the same sides, `base_map.episodes` attempts, and
    `base_map.sampling_seed`. The problems get their exact negations by the rule that gave the stored map's
    (`ladder_round.with_negations`, L0's), and BEFORE anything is sampled each problem's sides, as the episode step
    would sample them, are held to the stored map's row: a map on other sides is not the base map of another model.
    The episodes are a set of this run (`run_episodes`: a rerun resumes at the first block not done); the map is
    then written in the shape the challenger reads the base's. A map Lean did not answer is not kept."""
    store = _store(config)
    if store.is_done(MAP):
        return store.done_summary(MAP)
    for needed in (PREPARE, TRAIN):
        _need(store, needed, "`pre`'s own map", PRETRAIN_STAGE)
    settings, episode = config["ladder_loop"]["base_map"], config["ladder_loop"]["episode"]
    episodes, seed = settings["episodes"], settings["sampling_seed"]
    base_map, of_the_base, fixture = stored_map(config)
    stored = {row["problem_id"]: row for row in of_the_base}
    other = sorted(key for key, row in stored.items() if row["episodes"] != episodes)
    if other or sorted(stored) != sorted(row["problem_id"] for row in base_map):
        raise RuntimeError(f"the stored map is not {episodes} attempts on each of the base map's {len(base_map)} problems ({len(stored)} rows; {len(other)} with another "
                           "number of attempts): `pre`'s map would not be the same map of another model. Nothing was sampled")
    if not any(row["set"] == MAP_EPISODES for row in store.read_rows("problems.jsonl")):
        problems, checks = ladder_round.with_negations(config, base_map)
        planned = {problem["problem_id"]: len(side_plan(problem, episode, seed)) for problem in problems}
        differ = sorted(key for key, sides in planned.items() if sides != stored[key]["sides"])
        if differ and not fixture:
            raise RuntimeError(f"{len(differ)} of the base map's {len(problems)} problems would be attempted on other sides than the stored map's were (first: {differ[0]}, "
                               f"{planned[differ[0]]} against {stored[differ[0]]['sides']}): the episode settings or an exactness check are not what they were when the "
                               "stored map was made. Nothing was sampled")
        _write_json(store, MAP_PROBLEMS_FILE, {
            "stage": L4, "label": LABEL, "problems": len(problems), "episodes_each": episodes, "sampling_seed": seed, **checks,
            "sides": {str(sides): count for sides, count in sorted(Counter(planned.values()).items())},
            "sides_of_the_stored_map": {str(sides): count for sides, count in sorted(Counter(row["sides"] for row in of_the_base).items())},
            "problems_on_other_sides_than_the_stored_map": len(differ),
            "sides_held_to_the_stored_map": not fixture, **({"why_not": "a fixture's stored rows are hand-made: their sides are not what an episode step sampled"} if fixture else {})})
        add_problems(store, _as_sets(problems, [MAP_EPISODES]))
    if _stand_in():
        adapter = None
    else:
        directory = store.root / ladder_assembly.ADAPTERS / PRE
        if not directory.is_dir():
            raise RuntimeError(f"{directory} is not there: the adapter `pre` is kept by this stage, and the map is `pre`'s. The map cannot be made")
        from vllm.lora.request import LoRARequest

        adapter = LoRARequest(f"{PRETRAIN_STAGE}_{PRE}", 1, str(directory))
    _say(f"`pre`'s own map: {episodes} attempts on each of the base map's {len(base_map)} problems, sampling seed {seed}")
    sampled = _episodes(config, store, MAP_EPISODES, episodes, seed, Engines(enable_lora=True).kit(adapter))
    results = _results(store, MAP_EPISODES)
    if lean_did_not_answer(results):        # the challenger is not aimed by a map with holes: the set is sampled again by a rerun
        forget_measurement(store, MAP_EPISODES)
        raise RuntimeError(f"Lean gave no verdict on too many attempts of {MAP_EPISODES}: the map is not kept. Queue the task again: this step samples it again")
    rows = map_rows(results, [row["problem_id"] for row in base_map])
    store.write_rows(MAP_FILE, rows)
    summary = {"stage": L4, "label": LABEL, "model": PRE, "what": "the base map's problems attempted again by `pre`, as the stored map (the base's) was made",
               "set": MAP_EPISODES, "file": str(store.path(MAP_FILE)), "problems": len(rows), "episodes_each": episodes, "sampling_seed": seed,
               "fixture": fixture, "problems_built": json.loads(store.path(MAP_PROBLEMS_FILE).read_text()),
               "map": map_summary(rows), "map_of_the_base": map_summary(of_the_base),
               "attempts": sampled.get("attempts"), "statuses": sampled.get("statuses"), "generated_tokens": sampled.get("generated_tokens"),
               "generation_seconds": sampled.get("generation_seconds"), "pipeline": sampled.get("pipeline"), "stand_in_engine": _stand_in()}
    store.mark_done(MAP, summary)
    _say(f"`pre`'s own map is stored ({MAP_FILE}): mean pass rate {summary['map']['mean_pass_rate']} against the base's {summary['map_of_the_base']['mean_pass_rate']}")
    return summary


def _read(store: ArtifactStore, prepared: dict, label: str, name: str, wrote: dict) -> dict:
    """One model as a report reads it: its rows on the rungs, a list of rows for each sampling of G, what it wrote."""
    return {RUNG_PART: _results(store, rung_set(name, label)), GOAL: [_results(store, goal_set(sampling["name"], name, label)) for sampling in prepared["goal_samplings"]],
            **{key: wrote[key] for key in WROTE}}


def ladder_l4_pretrain_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l4 import build_pretrain_report

    store = _store(config)
    for marker in (PREPARE, TRAIN, MEASURE, MAP):
        _need(store, marker, "the pretraining's report", PRETRAIN_STAGE)
    prepared, samplings = store.done_summary(PREPARE), store.done_summary(PREPARE)["goal_samplings"]
    stored_models = json.loads(store.path(STORED_MODELS_FILE).read_text())
    stored = lambda who: {RUNG_PART: store.read_rows(stored_file(who, RUNG_PART, L4)),      # noqa: E731
                          GOAL: [store.read_rows(stored_file(who, sampling["name"], L4)) for sampling in samplings], **stored_models[who]}
    measured = store.done_summary(MEASURE)
    models = {PRE: {**_read(store, prepared, L4, PRE, measured), "stand_in_engine": measured["stand_in_engine"]}}
    if prepared["stored_runs"]:
        models[LOOP] = stored(LOOP)
    for who, entry in (prepared["ceiling"].get("models") or {}).items():       # the ceiling's two models, when its run was read: beside, deciding nothing
        models[who] = {**entry, **stored(who)}
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    made = store.done_summary(MAP)
    maps = {"sampling_seed": made["sampling_seed"], "file": MAP_FILE, BASE: stored_map(config)[1], PRE: store.read_rows(MAP_FILE)}
    report = build_pretrain_report(prepared, store.done_summary(TRAIN), store.read_rows(GROUPS_FILE), proof_lengths(), stored(BASE), models, minimums(config),
                                   config["ladder_loop"]["l4"], config["evaluation"], maps)
    report["adapter"] = {"kept": True, "what": f"`pre` ({LABEL}) is KEPT: it is the starting model of the arm", "directory": str(store.root / ladder_assembly.ADAPTERS / PRE)}
    store.write_rows(AGAIN_FILE, [{"problem_id": problem_id} for problem_id in report["goal_set_again"]["problem_ids"]])
    _write_json(store, REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in (STORED_MODELS_FILE, LOSS_FILE, MAP_PROBLEMS_FILE):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(REPORT, {"stage": L4, "label": LABEL, "headline": report["headline"], "ok": report["ok"], "checks_pass": report["checks_pass"]})
    return report


# ------------------------------------------------------------------------------------------------ the arm
def _arm_store(config: dict) -> ArtifactStore:
    wanted, named = config["ladder_loop"]["l4"]["arm"], ladder_l2.arm_name()
    if named != wanted:
        raise RuntimeError(f"the stage `{ARM_STAGE}` runs the arm {wanted} of ladder_loop.{ladder_l2.ASSEMBLY_ARMS} and {ladder_l2.L2_ARM_VARIABLE} names {named!r}: "
                           "its steps are run by the stage, which sets it")
    return ladder_l3d2._store(config)


def ladder_l4_prepare(config: dict) -> dict:
    """What the arm stands on, checked BEFORE anything is sampled: the pretraining run of this seed (its report
    written, its two checks passed, the adapter `pre` there). What the arm's report reads of that run, and of the
    base arm's when it is on the box, is copied into the arm's own run directory."""
    store = _arm_store(config)
    if store.is_done(ARM_PREPARE):
        return store.done_summary(ARM_PREPARE)
    _need(store, ladder_l2.PREPARE, "L4's own prepare step", ARM_STAGE)
    seed, directory, start = training_seed(config), pretrain_directory(config), ladder_l2.start_directory(config)
    task = f"the task of stage `{PRETRAIN_STAGE}` for seed {seed} (`python -m rlvr_lean.runner.entry --stage {PRETRAIN_STAGE} --seeds {seed}`; for the smoke run, stage `{PRETRAIN_STAGE}_smoke`)"
    lacking = [name for name in (REPORT_FILE, AGAIN_FILE, f"{MEASURE}.done.json", f"{PREPARE}.done.json", MAP_FILE, f"{MAP}.done.json") if not (directory / name).exists()]
    if lacking:
        raise RuntimeError(f"{directory} does not hold {lacking}. L4's arm starts from the model {task} pretrained, and reads its report. Run that task to its end first; "
                           "nothing was written.")
    report, of_pre = json.loads((directory / REPORT_FILE).read_text()), json.loads((directory / f"{PREPARE}.done.json").read_text())
    failed = [name for name, check in report["can_this_run_see_a_win"].items() if isinstance(check, dict) and not check["passes"]]
    if failed or not report.get("ok", True):
        raise RuntimeError(f"the pretraining run {directory.name} cannot carry the arm: " + (f"its checks failed ({', '.join(failed)})" if failed else "its report is not to be read")
                           + ". The arm is not run on it; nothing was written.")
    if not _stand_in() and not start.is_dir():
        raise RuntimeError(f"{start} is not there: the adapter `pre` is what every model of the arm is trained from and what attempts round 1. Run {task} again; nothing was written.")
    again = [row["problem_id"] for row in _rows(directory / AGAIN_FILE)]
    store.write_rows(AGAIN_FILE, [{"problem_id": problem_id} for problem_id in again])
    parts = [RUNG_PART, *(sampling["name"] for sampling in of_pre["goal_samplings"])]
    for part in parts:
        store.write_rows(stored_file(PRE, part, L4), _rows(directory / f"episodes_{L4}_{part}_{PRE}_problems.jsonl"))
    _write_json(store, PRE_FILE, {"label": LABEL, **{key: json.loads((directory / f"{MEASURE}.done.json").read_text())[key] for key in WROTE}})
    # The base arm's own gain stands beside the primary: its last model's stored rows on G, when its run is on this box (a smoke run reads none).
    base_arm, copied = config["ladder_loop"]["l4"]["base_arm"], None
    of_the_base_arm = _runs(config) / f"ladder_l2_{base_arm}_seed{seed}"
    files = {part: of_the_base_arm / f"episodes_{ladder_l3d2.L3D2}_{part}_with_problems.jsonl" for part in (REACH, MORE)}
    if os.environ.get(ladder_l3d2.STORED_VARIABLE) != "none" and all(path.exists() for path in files.values()):
        for part, path in files.items():
            store.write_rows(stored_file(BASE_ARM, part, L4), _rows(path))
        copied = str(of_the_base_arm)
    summary = {"stage": L4, "label": LABEL, "seed": seed, "arm": ladder_l2.arm_name(), "pretraining_run": str(directory), "start_adapter": str(start),
               "pretraining_rows": of_pre["rows"], "pretraining_file_sha256": of_pre["pretraining_file_sha256"], "goal_set_again": len(again),
               "the_two_checks_of_the_pretraining": {name: check for name, check in report["can_this_run_see_a_win"].items() if isinstance(check, dict)},
               "map": store.done_summary(ladder_l2.PREPARE).get("map"), "map_file": store.done_summary(ladder_l2.PREPARE).get("map_file"),
               "pre_parts": parts, "base_arm": base_arm, "base_arm_run": copied, "stand_in_engine": _stand_in()}
    store.mark_done(ARM_PREPARE, summary)
    _say(f"the arm {summary['arm']} is prepared, seed {seed}: from `pre` ({of_pre['rows']} published proofs); G' holds {len(again)} goal problems; the base arm's rows "
         f"{'are read from ' + copied if copied else 'are not on this box: its own gain will not stand beside the primary'}")
    return summary


def ladder_l4_report(config: dict) -> dict:
    from rlvr_lean.gpu.ladder_l3c import proof_lengths
    from rlvr_lean.reporting.ladder_l4 import build_l4_report

    store, rounds = _arm_store(config), ladder_l2.rounds_of(config)
    last = rounds[-1]
    for marker in (ARM_PREPARE, ladder_l3d2.PREPARE, f"ladder_l2_train_{last}", ladder_l3d2.TRAIN_WITHOUT, *(ladder_l3d2.measure_marker(arm) for arm in ARMS)):
        _need(store, marker, "L4's report", ARM_STAGE)
    own, prepared, sizes = store.done_summary(ARM_PREPARE), store.done_summary(ladder_l3d2.PREPARE), ladder_l2._sizes(ladder_l2.arm_config(config), store)
    base, models = ladder_l3d2.measured_models(store, prepared)
    samplings = prepared["goal_samplings"]
    models[PRE] = {RUNG_PART: store.read_rows(stored_file(PRE, RUNG_PART, L4)), GOAL: [store.read_rows(stored_file(PRE, sampling["name"], L4)) for sampling in samplings],
                   **json.loads(store.path(PRE_FILE).read_text())}
    base_arm = [store.read_rows(stored_file(BASE_ARM, part, L4)) for part in (REACH, MORE)] if own["base_arm_run"] else None
    # Every training of the arm (each round's model, then the twin): its step's summary, its rows' losses, the problems of the rows it was to be trained on.
    trainings = {f"M({number})": (f"ladder_l2_train_{number}", ladder_assembly.loss_file(number), ladder_assembly.training_set_file(number)) for number in rounds}
    trainings[f"`{WITHOUT}`"] = (ladder_l3d2.TRAIN_WITHOUT, ladder_l3d2.LOSS_WITHOUT_FILE, ladder_l3d2.TRAINING_WITHOUT_FILE)
    trains = {name: store.done_summary(marker) for name, (marker, _, _) in trainings.items()}
    stored_losses = {name: json.loads(store.path(loss).read_text()) for name, (_, loss, _) in trainings.items()}
    losses = {name: entry["row_losses"] for name, entry in stored_losses.items()}
    # What each training WAS trained on is the training loop's own record of row ids, read back to their problems.
    problem_of = {name: {row["id"]: row["problem_id"] for row in store.read_rows(rows)} for name, (_, _, rows) in trainings.items()}
    trained_on = {name: [problem_of[name][row_id] for row_id in entry["rows_trained"]] for name, entry in stored_losses.items()}
    assembled = [row for number in rounds for batch in range(1, sizes["batches"] + 1) for row in store.read_rows(ladder_assembly.assembly_file(number, batch))]
    # The length of a held-out problem's published proof is read HERE, by the report, and by nothing before it.
    report = build_l4_report(
        prepared, own, store.done_summary(ladder_l2.PREPARE), [row["problem_id"] for row in store.read_rows(AGAIN_FILE)], trains, losses, trained_on,
        [row["lines_after"] for row in assembled], ladder_l3d2.rounds_read(store, rounds, sizes["batches"]), store.read_rows(ladder_l3d2.GROUPS_FILE),
        proof_lengths(), base, models, base_arm, config["ladder_loop"]["l4"], ladder_l2.arm_config(config)["ladder_loop"]["challenger"]["target_rate"],
        config["evaluation"])
    report["adapters"] = {"kept": True, "what": "every model of the arm is kept, and `pre` with them (in the pretraining run's directory)",
                          "directory": str(store.root / ladder_assembly.ADAPTERS), "start_adapter": own["start_adapter"],
                          "there": sorted(path.name for path in (store.root / ladder_assembly.ADAPTERS).glob("*") if path.is_dir())}
    _write_json(store, ARM_REPORT_FILE, report)
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in sorted(path.name for path in store.root.glob("*.done.json")):
        if "_block_" not in name:
            store.mirror(name)
    for name in (ladder_l3d2.STORED_MODELS_FILE, PRE_FILE, ladder_l3d2.LOSS_WITHOUT_FILE, *(ladder_assembly.loss_file(number) for number in rounds)):
        store.mirror(name)
    for line in report["lines"]:
        print(line, flush=True)
    store.mark_done(ARM_REPORT, {"stage": L4, "label": LABEL, "headline": report["headline"], "branch": report["branch"], "ok": report["ok"]})
    return report


STEPS = {PREPARE: ladder_l4_pretrain_prepare, TRAIN: ladder_l4_pretrain_train, MEASURE: ladder_l4_pretrain_measure, MAP: ladder_l4_pretrain_map,
         REPORT: ladder_l4_pretrain_report, ARM_PREPARE: ladder_l4_prepare, ARM_REPORT: ladder_l4_report}
