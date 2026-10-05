"""The ladder loop's L2: three rounds of the challenger arm, then the equal-compute control. Spec:
docs/spec/ladder-loop.spec.md, "L2: three rounds". Each step is its own
process (`rlvr_lean.runner.entry`, stage `ladder_l2`).

NAMING, as in `domain/ladder_round/rounds.py` and in the report: rounds are 1, 2, 3. Round r starts from M(r - 1) and
trains M(r); M(0) is the base model.

  ladder_l2_prepare       reads L1's run directory for this seed ON THE BOX (never writes there): G and the rungs,
                          those problems with their exact negations, the base's fresh results on them. Checks the
                          shipped data (the WHOLE pool as candidates; none of H, none of the base map) and fixes
                          the sampling seeds. No GPU, no Lean.
  ladder_l2_embed         the base model's embedding of every base-map and candidate statement, and its leading
                          directions, ONCE PER BOX: stored outside the run directory under a key (a hash of the model
                          and of every prompt), so another seed's task reuses them and embeds nothing
  ladder_l2_round_<r>     round r in ONE process with one engine: `round.batches` batches. A batch: the challenger
                          refitted on everything seen so far (the base map and every finished batch), its proposals
                          from the candidates not yet proposed, their exact negations, n episodes of M(r - 1) on
                          each. Then the round's training set: one verified proof of every problem with k >= 1
  ladder_l2_train_<r>     M(r): from the BASE, one pass, native format, over every round's training set so far
  ladder_l2_measure_<r>   M(r) on the three rungs (8 episodes) and on G (32) with L1's sampling seeds, so that it pairs
                          by problem with the base's stored results and with the other rounds; distinct attempts. The
                          stop rule is read here: an interval entirely below zero on the below-band rung ends the loop
  ladder_l2_control       after the last round, the base: the loop's attempt episodes spread over G's problems (61 on
                          each of 392), with a sampling seed no other measurement uses
  ladder_l2_control_trained   (added 2026-10-05) the LAST model: the control's number of episodes on every problem
                          of G, with the control's sampling seed, in a set of its own (`control_m3`), so the two
                          models are compared at the same number of attempts. A run that finished before the step
                          existed gets it alone when its task is queued again
  ladder_l2_report        the read fixed before the run, the challenger's table by batch and by round, the branch

The run directory is its own (`ladder_l2_seed<seed>`): nothing here can mark a step of L1's run done. A rerun
resumes at the first step, the first batch and, inside a batch's episodes, the first block that is not done. A
verified proof on the side a certificate contradicts stops the step with the soundness alarm's exit code, as in L1.
The gain by k is NOT measured in L2 (L1 measured it at three seeds and t stays 1/4).
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rlvr_lean.data.ladder_round_export import BASE_MAP_SET
from rlvr_lean.domain.ladder_round.challenger import SCORED, expected_rewards, fact_features, feature_names, fit_pass_rate_model, fit_projection
from rlvr_lean.domain.ladder_round.read import arm_reward, group_ids, paired_change, stop_rule
from rlvr_lean.domain.ladder_round.rounds import (
    FIRST_ROUND,
    batch_sizes,
    control_episodes,
    model_name,
    propose_batch,
    refit_observations,
    weights_summary,
)
from rlvr_lean.domain.ladder_round.training_set import training_examples, training_summary
from rlvr_lean.domain.problem_pool.certificates import FALSE_SIDE
from rlvr_lean.domain.problem_pool.episodes import BELOW_BAND, RUNGS
from rlvr_lean.domain.proving import build_prover_prompt
from rlvr_lean.domain.verification.pin import lean_pin_from_config
from rlvr_lean.gpu import ladder_loop, ladder_round, pipeline
from rlvr_lean.gpu.ladder_round import BASE, Engines, _attempts, _results, _stand_in, _write_json, distinct_attempts, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

L2_DATA_VARIABLE = "RLVR_LEAN_LADDER_L2_DATA"            # another data directory than the package's (the fixture, a pre-flight)
L2_RUN_VARIABLE = "RLVR_LEAN_LADDER_L2_RUN"              # another run directory than `ladder_l2_seed<seed>` (a smoke run)
L2_SOURCE_VARIABLE = "RLVR_LEAN_LADDER_L2_SOURCE"        # another L1 run directory to read than `ladder_l1_seed<seed>`
L2_PROBLEMS_VARIABLE = "RLVR_LEAN_LADDER_L2_PROBLEMS"    # another `round.problems` (a smoke run: the fixture has twelve candidates)
L2_BATCHES_VARIABLE = "RLVR_LEAN_LADDER_L2_BATCHES"      # another `round.batches` (the same)
PACKAGE_L2_DATA = Path(__file__).resolve().parents[1] / "data" / "ladder_l2"
ROUNDS = (1, 2, 3)                                       # the stage's steps; `round.rounds` must give these
PREPARE, EMBED, CONTROL_STEP, REPORT = "ladder_l2_prepare", "ladder_l2_embed", "ladder_l2_control", "ladder_l2_report"
CONTROL_TRAINED_STEP = "ladder_l2_control_trained"       # added 2026-10-05: the last model's own extra attempts on G
CONTROL = "control"                                      # the set the control's episodes are stored under
CONTROL_TRAINED = f"control_m{ROUNDS[-1]}"               # the set the last model's extra episodes are stored under (`control_m3`)
EMBEDDINGS = "ladder_embeddings"                         # under the store's root, beside the runs: one directory per key
MAX_LEFTOVER_GB = pipeline.MAX_LEFTOVER_GB
PROJECTION_ROWS = 8192                                   # statements projected at a time (a block of 4,096-wide vectors, not all of them)
SOURCE_FILES = ("problems.jsonl", "heldout_groups.jsonl", f"episodes_rungs_{BASE}_problems.jsonl", f"episodes_reach_{BASE}_problems.jsonl",
                f"episodes_rungs_{BASE}.done.json", f"episodes_reach_{BASE}.done.json")
# Sampling seeds: `round.sampling_seed` + 100 x the training seed + a place. The rungs and G are sampled with L1's own
# seeds for them (`ladder_round.SAMPLING_KINDS`: places 1 and 2), so every model pairs by problem with the base's fresh
# results L1 stored. The rounds' attempt episodes and the control have places of their own, which no measurement of
# L1, L1b or L2 shares.
ROUND_SEED_PLACE = 10                                    # + the round's number: 11, 12, 13
CONTROL_SEED_PLACE = 20


# ------------------------------------------------------------------------------------------------- the run
def _runs(config: dict) -> Path:
    return pipeline.STORE / lean_pin_from_config(ladder_loop.ladder_config(config)).runs_directory


def _store(config: dict) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    run = os.environ.get(L2_RUN_VARIABLE) or f"ladder_l2_seed{training_seed(config)}"
    return ArtifactStore(_runs(config) / run, Path(mirror) if mirror else None)


def source_directory(config: dict) -> Path:
    """L1's run directory for this seed. It is only ever READ, with plain file reads: nothing is created in it."""
    return _runs(config) / (os.environ.get(L2_SOURCE_VARIABLE) or f"ladder_l1_seed{training_seed(config)}")


def data_directory() -> Path:
    override = os.environ.get(L2_DATA_VARIABLE)
    return Path(override) if override else PACKAGE_L2_DATA


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def loop_sizes(config: dict) -> dict:
    """The loop's sizes: the config's, or a smoke run's own number of problems and batches."""
    settings = config["ladder_loop"]["round"]
    if tuple(range(FIRST_ROUND, FIRST_ROUND + settings["rounds"])) != ROUNDS:
        raise ValueError(f"ladder_loop.round.rounds is {settings['rounds']} and the stage runs rounds {ROUNDS}: change both together")
    problems = int(os.environ.get(L2_PROBLEMS_VARIABLE) or settings["problems"])
    batches = int(os.environ.get(L2_BATCHES_VARIABLE) or settings["batches"])
    return {"problems": problems, "batches": batches, "batch_sizes": batch_sizes(problems, batches), "solvers": settings["solvers"]}


def sampling_seeds(config: dict) -> dict:
    first = config["ladder_loop"]["round"]["sampling_seed"] + 100 * training_seed(config)
    return {"rungs": ladder_round.sampling_seed(config, f"rungs_{BASE}"), "reach": ladder_round.sampling_seed(config, f"reach_{BASE}"),
            **{f"round_{number}": first + ROUND_SEED_PLACE + number for number in ROUNDS}, CONTROL: first + CONTROL_SEED_PLACE}


def batch_set(number: int, batch: int) -> str:
    return f"round_r{number}_b{batch}"


def adapter_directory(store: ArtifactStore, number: int) -> Path:
    return store.root / "adapters" / f"m{number}"


def stopped_after(store: ArtifactStore) -> int | None:
    """The round whose measurement fired the stop rule (the loop ran no further), or None."""
    for number in ROUNDS:
        marker = f"ladder_l2_measure_{number}"
        if not store.is_done(marker):
            return None
        if store.done_summary(marker).get("stop_rule_fires"):
            return number
    return None


def _skipped(store: ArtifactStore, what: str) -> dict | None:
    """Spec, "A round" step 6: a round whose interval for the below-band rung lies entirely below zero stops the
    loop for diagnosis. The steps after it run nothing and say so; the report is still written."""
    stopped = stopped_after(store)
    if stopped is None:
        return None
    below = store.done_summary(f"ladder_l2_measure_{stopped}")["below_band_minus_base"]
    return {"skipped": f"the stop rule fired after round {stopped} ({model_name(stopped)} minus the base on the below-band rung: {below['mean']} "
                       f"[{below['low']}, {below['high']}], entirely below zero): the loop stopped for diagnosis and {what} was not run",
            "stopped_after_round": stopped}


def _need(store: ArtifactStore, marker: str, what: str) -> None:
    if not store.is_done(marker):
        raise RuntimeError(f"{what} needs the step {marker} of this run, which is not done: the stage `ladder_l2` runs the steps in order")


def _sizes(config: dict, store: ArtifactStore) -> dict:
    """The loop's sizes, refused when they are not the ones this run was prepared with: its batches are stored by
    their number, and a run resumed with other sizes would mix two loops."""
    sizes, prepared = loop_sizes(config), store.done_summary(PREPARE)
    recorded = {"problems": prepared["problems_a_round"], "batches": prepared["batches"], "batch_sizes": prepared["batch_sizes"], "solvers": prepared["solvers"]}
    if sizes != recorded:
        raise RuntimeError(f"{store.root} was prepared with {recorded} and the config now gives {sizes}: a run keeps the sizes it began with")
    return sizes


# ------------------------------------------------------------------------------------------------- prepare
def ladder_l2_prepare(config: dict) -> dict:
    store = _store(config)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    sizes, seed, source = loop_sizes(config), training_seed(config), source_directory(config)
    attempt_files = sorted(source.glob(f"episodes_rungs_{BASE}_attempts_*.jsonl")) if source.is_dir() else []
    missing = [name for name in SOURCE_FILES if not (source / name).exists()]
    if missing or not attempt_files:
        if not attempt_files:
            missing.append(f"episodes_rungs_{BASE}_attempts_*.jsonl (the base model's rung attempts)")
        raise RuntimeError(
            f"{source} does not hold {missing}. "
            f"L2 reads the run directory the L1 task for seed {seed} wrote on this box (stage `ladder_l1` with --seeds {seed}: "
            f"`python -m rlvr_lean.runner.entry --stage ladder_l1 --seeds {seed}`; for the smoke run, stage `ladder_l1_smoke`). "
            "Run that task to its end first; nothing was written.")
    groups = _rows(source / "heldout_groups.jsonl")
    problems = _rows(source / "problems.jsonl")
    rung_problems = [row for row in problems if row["set"] == f"rungs_{BASE}"]
    goal_problems = [row for row in problems if row["set"] == f"reach_{BASE}"]
    measure, seeds = config["ladder_loop"]["measure"], sampling_seeds(config)
    stored = {kind: json.loads((source / f"episodes_{kind}_{BASE}.done.json").read_text()) for kind in ("rungs", "reach")}
    for kind, episodes in (("rungs", measure["rung_episodes"]), ("reach", measure["reach_episodes"])):
        if stored[kind].get("problems") and (stored[kind]["sampling_seed"] != seeds[kind] or stored[kind]["episodes_each"] != episodes):
            raise RuntimeError(f"L1 measured the base on its {kind} with sampling seed {stored[kind]['sampling_seed']} and {stored[kind]['episodes_each']} "
                               f"episodes; this config gives {seeds[kind]} and {episodes}: the rounds would not pair with L1's stored results")
    data = ladder_round._data(config, data_directory())         # refuses a candidate that is in H or in the base map (fixture 6)
    if {row["problem_id"] for row in groups} != {row["problem_id"] for row in data["heldout"]}:
        raise RuntimeError(f"{source} placed another held-out set than this snapshot's H ({ladder_loop.data_directory()}): the candidates are held "
                           "out against the snapshot's, and the rungs and G would be measured on the run's. Nothing was written.")

    def as_set(rows: list[dict], set_name: str) -> list[dict]:
        return [{**row, "set": set_name} for row in rows]

    own = [row for number in ROUNDS for row in as_set(rung_problems, f"rungs_m{number}")]
    own += [row for number in ROUNDS for row in as_set(goal_problems, f"reach_m{number}")]
    own += as_set(goal_problems, CONTROL)
    store.write_rows("problems.jsonl", own)
    store.write_rows("heldout_groups.jsonl", groups)
    store.write_rows("base_rungs.jsonl", _rows(source / f"episodes_rungs_{BASE}_problems.jsonl"))
    store.write_rows("base_reach.jsonl", _rows(source / f"episodes_reach_{BASE}_problems.jsonl"))
    wanted = len(ROUNDS) * sizes["problems"]
    summary = {"seed": seed, "source_run": str(source), "data_directory": str(data_directory()), "fixture": bool(data["summary"].get("fixture")),
               "stand_in_engine": _stand_in(), "lean_pin": config["ladder_loop"]["lean_pin"], "data": data["counts"],
               "candidates_exported": data["summary"].get("candidates_wanted"),
               "rounds": list(ROUNDS), "problems_a_round": sizes["problems"], "batches": sizes["batches"], "batch_sizes": sizes["batch_sizes"],
               "solvers": sizes["solvers"], "candidates_short_by": max(0, wanted - data["counts"]["candidates"]),
               "goal_set": len(goal_problems), "rungs": {name: len(group_ids(groups, name)) for name in RUNGS}, "rung_problems": len(rung_problems),
               "heldout_in_neither": sum(row["group"] is None for row in groups),
               "sampling_seeds": seeds, "rung_episodes": measure["rung_episodes"], "reach_episodes": measure["reach_episodes"],
               "base_distinct_attempts_on_the_rungs": distinct_attempts([row for path in attempt_files for row in _rows(path)]),
               "l1_contradicted_side_setting": stored["rungs"].get("contradicted_side_setting"),
               "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all")}
    store.mark_done(PREPARE, summary)
    return summary


# --------------------------------------------------------------------------------------------------- embed
def embedding_key(config: dict, problem_ids: list[str], prompts: list[str], stand_in: bool) -> str:
    """What the stored embeddings are OF: the model, how a statement is read, and every prompt in its order.
    Another seed's task has the same key and embeds nothing; other data, another model or the stand-in has another."""
    digest = hashlib.sha256(json.dumps({"model": config["model"], "weights": "NF4 base (the training stack's)",
                                        "what": "the mean of the last layer's states over the prover prompt's tokens",
                                        "max_tokens": config["ladder_loop"]["challenger"]["embedding_max_tokens"],
                                        "stand_in": stand_in}, sort_keys=True).encode())
    for problem_id, prompt in zip(problem_ids, prompts):
        digest.update(b"\0" + problem_id.encode() + b"\0" + prompt.encode())
    return digest.hexdigest()


def _save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def ladder_l2_embed(config: dict) -> dict:
    store = _store(config)
    data = ladder_round._data(config, data_directory())
    rows = [*data["base_map"], *data["candidates"]]
    problem_ids, statements = [row["problem_id"] for row in rows], [row["statement"] for row in rows]
    components = config["ladder_loop"]["challenger"]["embedding_components"]
    key = embedding_key(config, problem_ids, [build_prover_prompt(statement) for statement in statements], _stand_in())
    shared = ArtifactStore(pipeline.STORE / EMBEDDINGS / key[:16])      # no mirror: half a gigabyte, and nothing a reader needs
    projected_name = f"projected_{components}.npy"
    if store.is_done(EMBED) and store.done_summary(EMBED)["key"] == key and shared.path(projected_name).exists():
        return store.done_summary(EMBED)
    started = time.monotonic()
    embedded_before = shared.is_done("embeddings")
    if not embedded_before:
        if _stand_in():
            vectors = ladder_round.stand_in_embeddings(statements)
        else:
            import torch

            vectors = ladder_round._embed(config, statements)
            gc.collect()
            torch.cuda.empty_cache()
        _save_array(shared.path("embeddings.npy"), vectors.astype(np.float16))
        shared.path("embedding_ids.json").write_text(json.dumps(problem_ids))
        shared.mark_done("embeddings", {"key": key, "statements": len(rows), "dimension": int(vectors.shape[1]),
                                        "seconds": round(time.monotonic() - started, 1), "stand_in_engine": _stand_in()})
        del vectors
    embedded = shared.done_summary("embeddings")
    if embedded["key"] != key:
        raise RuntimeError(f"{shared.root} holds embeddings under another key than this data's: remove that directory and run the step again")
    projected_before = shared.path(projected_name).exists()
    if not projected_before:
        # The leading directions of every embedding the stage holds, as L1 fitted them (on what is STORED, at its
        # precision), and each statement's place along them: 32 numbers a statement in place of 4,096.
        stored = np.load(shared.path("embeddings.npy")).astype(np.float64)
        projection = fit_projection(stored, components)
        _save_array(shared.path(projected_name), np.vstack([projection.apply(stored[start:start + PROJECTION_ROWS])
                                                            for start in range(0, len(stored), PROJECTION_ROWS)]))
        del stored
    summary = {"key": key, "directory": str(shared.root), "projected": projected_name, "statements": len(rows), "base_map": len(data["base_map"]),
               "candidates": len(data["candidates"]), "dimension": embedded["dimension"], "components_asked": components,
               "embeddings_reused": embedded_before, "projection_reused": projected_before, "seconds_embedding_when_made": embedded["seconds"],
               "seconds": round(time.monotonic() - started, 1), "stand_in_engine": _stand_in()}
    store.mark_done(EMBED, summary)
    return summary


def _statements(store: ArtifactStore, data: dict) -> dict:
    """Every statement the challenger reads (the base map's, then the candidates') with its features: its place
    along the leading directions of the embeddings, and the published facts."""
    embedded = store.done_summary(EMBED)
    directory = Path(embedded["directory"])
    facts = {row["problem_id"]: row for row in data["base_map_facts"]}
    rows = [*({**row, **facts[row["problem_id"]]} for row in data["base_map"]), *data["candidates"]]
    problem_ids = [row["problem_id"] for row in rows]
    if json.loads((directory / "embedding_ids.json").read_text()) != problem_ids:
        raise RuntimeError(f"{directory} does not hold the embeddings of this data's statements in their order: run {EMBED} again")
    projected = np.load(directory / embedded["projected"])
    return {"position": {problem_id: index for index, problem_id in enumerate(problem_ids)},
            "features": np.hstack([projected, fact_features(rows)]), "names": feature_names(int(projected.shape[1]))}


# ------------------------------------------------------------------------------------------------ sampling
def _episodes(config: dict, store: ArtifactStore, set_name: str, episodes: int, seed: int, kit) -> dict:
    if not any(row["set"] == set_name for row in store.read_rows("problems.jsonl")):
        store.write_rows(f"episodes_{set_name}_problems.jsonl", [])      # a set with no problem is measured as such
        return {"set": set_name, "problems": 0, "episodes_each": episodes, "attempts": 0, "statuses": {}}
    with ladder_loop.lean_sessions(config) as sessions:      # ONE client for the set: its requests in flight are `lean_in_flight`
        return ladder_loop.run_episodes(config, set_name, episodes, seed, store, kit, sessions)


def _adapter(store: ArtifactStore, number: int):
    """The adapter of M(`number`) as vLLM serves it beside the base; None for M(0), the base, and for the stand-in."""
    if number < FIRST_ROUND or _stand_in():
        return None
    from vllm.lora.request import LoRARequest

    return LoRARequest(f"ladder_l2_m{number}", number, str(adapter_directory(store, number)))


# --------------------------------------------------------------------------------------------------- round
def _off_the_main_thread(function, *arguments):
    """Run `function` in a worker thread and wait for it. A batch's exactness checks go to Lean through
    `asyncio.run` (`check_lean_sources`), and from the second batch on the sampling engine of this process is
    already loaded. `asyncio.run` unsets its thread's event loop when it returns; whether a loaded engine minds
    that on the main thread has never been tried here, and a GPU run is not where to find out. In a thread of
    its own the call is what `VerificationService` has always done between two calls to the engine."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ladder-negations") as executor:
        return executor.submit(function, *arguments).result()


def _batches_before(sizes: dict, number: int, batch: int) -> list[tuple[int, int]]:
    return [(earlier, part) for earlier in ROUNDS for part in range(1, sizes["batches"] + 1) if (earlier, part) < (number, batch)]


def _propose(config: dict, store: ArtifactStore, data: dict, statements: dict, sizes: dict, number: int, batch: int) -> list[dict]:
    """One batch's proposals, made once: the challenger refitted on the base map and on every finished batch,
    every candidate scored, the batch chosen among those not yet proposed, and its problems stored with their
    exact negations. A rerun returns what is stored."""
    marker, name = f"ladder_l2_propose_r{number}_b{batch}", f"proposals_r{number}_b{batch}.jsonl"
    if store.is_done(marker):
        return store.read_rows(name)
    challenger, seed = config["ladder_loop"]["challenger"], training_seed(config)
    position, features = statements["position"], statements["features"]
    results = {row["problem_id"]: row for row in data["base_results"] if row["set"] == BASE_MAP_SET}
    before = _batches_before(sizes, number, batch)
    finished = [{**row, "round": earlier, "batch": part} for earlier, part in before for row in _results(store, batch_set(earlier, part))]
    seen = refit_observations([results[row["problem_id"]] for row in data["base_map"]], finished, number, challenger["recency_decay"])
    model, chosen = fit_pass_rate_model(seen["problem_ids"], features[[position[problem_id] for problem_id in seen["problem_ids"]]], seen["resolved"],
                                        seen["episodes"], names=statements["names"], ridge_grid=challenger["ridge_grid"], folds=challenger["folds"],
                                        dispersion_grid=challenger["dispersion_grid"], seed=seed, weights=seen["weights"])
    candidate_ids = [row["problem_id"] for row in data["candidates"]]
    rates = model.predict(features[[position[problem_id] for problem_id in candidate_ids]])
    scores = expected_rewards(rates, sizes["solvers"], challenger["target_rate"], model.dispersion)
    proposed = {row["problem_id"] for earlier, part in before for row in store.read_rows(f"proposals_r{earlier}_b{part}.jsonl")}
    barred = {row["problem_id"] for row in data["heldout"]} | {row["problem_id"] for row in data["base_map"]}
    picks = [{**row, "round": number, "batch": batch}
             for row in propose_batch(candidate_ids, rates, scores, proposed, barred, sizes["batch_sizes"][batch - 1], challenger["random_share"], seed)]
    by_id = {row["problem_id"]: row for row in data["candidates"]}
    # Every proposed problem gets its n episodes: none is dropped or kept by a pass-rate estimate first (fixture 5).
    problems, checks = _off_the_main_thread(ladder_round.with_negations, config, [by_id[row["problem_id"]] for row in picks])
    ladder_round.add_problems(store, ladder_round._as_sets(problems, [batch_set(number, batch)]))
    still_open = [index for index, problem_id in enumerate(candidate_ids) if problem_id not in proposed]
    if batch == 1:
        # What the challenger thought of the pool when the round began: every candidate not yet proposed.
        store.write_rows(f"candidate_scores_r{number}.jsonl", [{"problem_id": candidate_ids[index], "predicted_rate": round(float(rates[index]), 5),
                                                                "expected_reward": round(float(scores[index]), 5)} for index in still_open])
    false_open = [index for index in still_open if by_id[candidate_ids[index]]["side"] == FALSE_SIDE]
    target = challenger["target_rate"]
    store.write_rows(name, picks)
    store.mark_done(marker, {
        "round": number, "batch": batch, "attempted_by": model_name(number - 1), "observations": len(seen["problem_ids"]),
        "observations_by_the_round_that_gave_them": weights_summary(seen),
        "fit": {key: value for key, value in chosen.items() if key != "out_of_fold_rates"},
        "candidates_not_yet_proposed": {
            "problems": len(still_open), "known_false": len(false_open),
            "mean_predicted_rate": round(float(rates[still_open].mean()), 4) if still_open else None,
            "best_expected_reward": round(float(scores[still_open].max()), 4) if still_open else None,
            "predicted_at_or_above_the_target_rate": int((rates[still_open] >= target).sum()),
            "known_false_predicted_at_or_above_the_target_rate": int((rates[false_open] >= target).sum())},
        "proposals": {"problems": len(picks), "scored": sum(row["how"] == SCORED for row in picks),
                      "known_false": sum(by_id[row["problem_id"]]["side"] == FALSE_SIDE for row in picks),
                      "mean_predicted_rate": round(float(np.mean([row["predicted_rate"] for row in picks])), 4) if picks else None,
                      "mean_expected_reward": round(float(np.mean([row["score"] for row in picks])), 4) if picks else None},
        **checks})
    return picks


def ladder_l2_round(config: dict, number: int) -> dict:
    """Round `number`: its batches, one after the other in this ONE process (the engine is loaded once), each
    attempted by M(number - 1); then the round's training set. A batch that is done is not touched again."""
    store, marker = _store(config), f"ladder_l2_round_{number}"
    if store.is_done(marker):
        return store.done_summary(marker)
    skipped = _skipped(store, f"round {number}")
    if skipped:
        return skipped
    for needed in (PREPARE, EMBED, *([f"ladder_l2_measure_{number - 1}"] if number > FIRST_ROUND else [])):
        _need(store, needed, f"round {number}")
    sizes, settings, seed = _sizes(config, store), config["ladder_loop"], training_seed(config)
    target, floor = settings["challenger"]["target_rate"], settings["challenger"]["band_reward"]
    sampling_seed = store.done_summary(PREPARE)["sampling_seeds"][f"round_{number}"]
    data = ladder_round._data(config, data_directory())
    statements = _statements(store, data)
    kit = Engines(enable_lora=number > FIRST_ROUND).kit(_adapter(store, number - 1))
    sets = [batch_set(number, batch) for batch in range(1, sizes["batches"] + 1)]
    batches = []
    for batch, set_name in enumerate(sets, start=1):
        batch_marker = f"ladder_l2_batch_r{number}_b{batch}"
        if not store.is_done(batch_marker):
            picks = _propose(config, store, data, statements, sizes, number, batch)
            episodes = _episodes(config, store, set_name, sizes["solvers"], sampling_seed, kit)
            store.mark_done(batch_marker, {"round": number, "batch": batch, "set": set_name, "problems": len(picks), "episodes": episodes,
                                           "reward": arm_reward(_results(store, set_name), target, floor)})
        batches.append(store.done_summary(batch_marker))
    # The round's training set: one verified proof of every problem with k >= 1, drawn with the round's seed, on
    # whichever side was proved. The proofs are the solver's own attempts (fixtures 7 and 8).
    batch_of = {row["problem_id"]: batch for batch, set_name in enumerate(sets, start=1) for row in _results(store, set_name)}
    problems = [row for row in store.read_rows("problems.jsonl") if row["set"] in sets]
    examples = training_examples(problems, [attempt for set_name in sets for attempt in _attempts(store, set_name)], seed)
    store.write_rows(f"training_examples_r{number}.jsonl", [{**example, "round": number, "batch": batch_of[example["problem_id"]]} for example in examples])
    results = [row for set_name in sets for row in _results(store, set_name)]
    summary = {"round": number, "attempted_by": model_name(number - 1), "seed": seed, "sampling_seed": sampling_seed,
               "problems": len(results), "attempt_episodes": sum(row["episodes"] for row in results),
               "reward": arm_reward(results, target, floor), "training_set": training_summary(examples),
               "batches": [{key: entry[key] for key in ("batch", "set", "problems", "reward")} for entry in batches],
               "attempts": sum(entry["episodes"]["attempts"] for entry in batches),
               "generated_tokens": sum(entry["episodes"].get("generated_tokens", 0) for entry in batches),
               "generation_seconds": round(sum(entry["episodes"].get("generation_seconds", 0) for entry in batches), 1),
               "stand_in_engine": _stand_in()}
    store.mark_done(marker, summary)
    return summary


# ------------------------------------------------------------------------------------------------ training
def ladder_l2_train(config: dict, number: int) -> dict:
    """M(number): trained FROM THE BASE, one pass, on every round's training set so far (`ladder_round._train`:
    the same model, adapter, optimizer and order rule as L1's one round)."""
    store, marker = _store(config), f"ladder_l2_train_{number}"
    if store.is_done(marker):
        return store.done_summary(marker)
    skipped = _skipped(store, f"the training of {model_name(number)}")
    if skipped:
        return skipped
    _need(store, f"ladder_l2_round_{number}", f"the training of {model_name(number)}")
    seed = training_seed(config)
    by_round = {earlier: store.read_rows(f"training_examples_r{earlier}.jsonl") for earlier in ROUNDS if earlier <= number}
    examples = [example for earlier in sorted(by_round) for example in by_round[earlier]]
    if not examples:
        raise RuntimeError(f"rounds 1 to {number} resolved no problem: there is nothing to train {model_name(number)} on")
    summary = {"round": number, "model": model_name(number), "seed": seed, "trained_from": "the base", "rounds_trained_on": sorted(by_round),
               "examples_by_round": {str(earlier): len(rows) for earlier, rows in sorted(by_round.items())}, **training_summary(examples),
               "stand_in_engine": _stand_in()}
    if _stand_in():
        summary["note"] = "no adapter was trained: a pre-flight without a GPU"
    else:
        import torch

        from rlvr_lean.gpu.model_utils import training_text

        pipeline._cap_torch_memory(config)
        pairs = [training_text(example["theorem"], example["completion"]) for example in examples]
        summary.update(ladder_round._train(config, pairs, seed, adapter_directory(store, number), f"ladder_l2_m{number}_seed{seed}"))
        gc.collect()
        torch.cuda.empty_cache()
        summary["allocated_after_cleanup_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
    store.mark_done(marker, summary)       # the adapter is saved: keep it whatever follows
    if summary.get("allocated_after_cleanup_gb", 0) > MAX_LEFTOVER_GB:
        raise RuntimeError(f"{summary['allocated_after_cleanup_gb']} GB is still allocated on the GPU after {model_name(number)}'s adapter was trained and released")
    return {key: value for key, value in summary.items() if key != "losses"}


# ------------------------------------------------------------------------------------------------- measure
def ladder_l2_measure(config: dict, number: int) -> dict:
    """M(number) on the three held-out rungs and on G, with L1's sampling seeds: the same problems, the same random
    numbers and the same sides as L1's base measurement and as every other round, so each pairs by problem with
    them. The stop rule is read here, on the below-band rung against the base."""
    store, marker = _store(config), f"ladder_l2_measure_{number}"
    if store.is_done(marker):
        return store.done_summary(marker)
    skipped = _skipped(store, f"the measurement of {model_name(number)}")
    if skipped:
        return skipped
    _need(store, f"ladder_l2_train_{number}", f"the measurement of {model_name(number)}")
    prepare, measure, evaluation = store.done_summary(PREPARE), config["ladder_loop"]["measure"], config["evaluation"]
    kit = Engines(enable_lora=True).kit(_adapter(store, number))
    summary = {"round": number, "model": model_name(number),
               "rungs": _episodes(config, store, f"rungs_m{number}", measure["rung_episodes"], prepare["sampling_seeds"]["rungs"], kit),
               "reach": _episodes(config, store, f"reach_m{number}", measure["reach_episodes"], prepare["sampling_seeds"]["reach"], kit),
               "distinct_attempts_on_the_rungs": distinct_attempts(_attempts(store, f"rungs_m{number}"))}
    below = paired_change(_results(store, f"rungs_m{number}"), store.read_rows("base_rungs.jsonl"),
                          group_ids(store.read_rows("heldout_groups.jsonl"), BELOW_BAND), evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"])
    summary.update({"below_band_minus_base": below, "stop_rule_fires": stop_rule(below), "stand_in_engine": _stand_in()})
    store.mark_done(marker, summary)
    return summary


# ------------------------------------------------------------------------------------------------- control
def ladder_l2_control(config: dict) -> dict:
    """The equal-compute control, after the last round: the BASE gets every attempt episode the rounds ran, the
    same number on each problem of G, with a sampling seed of its own. They are read on top of the episodes of
    its fresh reach measurement (L1's)."""
    store = _store(config)
    if store.is_done(CONTROL_STEP):
        return store.done_summary(CONTROL_STEP)
    skipped = _skipped(store, "the equal-compute control")
    if skipped:
        return skipped
    _need(store, f"ladder_l2_measure_{ROUNDS[-1]}", "the equal-compute control")
    prepare = store.done_summary(PREPARE)
    loop_episodes = sum(store.done_summary(f"ladder_l2_round_{number}")["attempt_episodes"] for number in ROUNDS)
    each = control_episodes(loop_episodes, prepare["goal_set"])
    summary = {"model": model_name(0), "loop_attempt_episodes": loop_episodes, "goal_set": prepare["goal_set"], "episodes_each": each,
               "episodes_not_spent": loop_episodes - each * prepare["goal_set"], "on_top_of_the_fresh_reach_episodes": prepare["reach_episodes"],
               "sampling_seed": prepare["sampling_seeds"][CONTROL], "stand_in_engine": _stand_in()}
    if each:
        summary["episodes"] = _episodes(config, store, CONTROL, each, prepare["sampling_seeds"][CONTROL], Engines(enable_lora=False).kit())
    else:
        summary["note"] = "no control episodes: G is empty, or the loop ran fewer attempt episodes than G has problems"
    store.mark_done(CONTROL_STEP, summary)
    return summary


def _ensure_problems(store: ArtifactStore, wanted: str, held: str) -> None:
    """The prepare step writes no problems under `wanted` (a run prepared before the set existed has none, and a
    new run is prepared the same way): they are the goal problems the set `held` holds, with this set's name."""
    rows = store.read_rows("problems.jsonl")
    if any(row["set"] == wanted for row in rows):
        return
    own = [{**row, "set": wanted} for row in rows if row["set"] == held]
    if not own:
        raise RuntimeError(f"{store.root} holds no problem of the set {held}: the set {wanted} cannot be made from it")
    store.write_rows("problems.jsonl", rows + own)


def ladder_l2_control_trained(config: dict) -> dict:
    """The same extra attempts for the trained model (spec, "Added 2026-10-05, after seeds 0 and 1"): the LAST
    model gets on every problem of G exactly the control's number of episodes, with the control's sampling seed
    (the same problems, the same sides), in a set of its own. With its reach episodes that is as many attempts
    as the base has with the control's: the two models at equal attempts. It does not replace the control. A run
    that finished before this step existed gets it alone when its task is queued again: every other step
    returns what is stored."""
    store, last = _store(config), ROUNDS[-1]
    if store.is_done(CONTROL_TRAINED_STEP):
        return store.done_summary(CONTROL_TRAINED_STEP)
    skipped = _skipped(store, f"{model_name(last)}'s extra attempts on G")
    if skipped:
        return skipped
    for needed in (CONTROL_STEP, f"ladder_l2_train_{last}"):
        _need(store, needed, f"{model_name(last)}'s extra attempts on G")
    prepare, each = store.done_summary(PREPARE), store.done_summary(CONTROL_STEP)["episodes_each"]
    seed = prepare["sampling_seeds"][CONTROL]
    summary = {"model": model_name(last), "set": CONTROL_TRAINED, "goal_set": prepare["goal_set"], "episodes_each": each,
               "the_same_number_and_sampling_seed_as": CONTROL, "on_top_of_the_reach_episodes": prepare["reach_episodes"],
               "sampling_seed": seed, "stand_in_engine": _stand_in()}
    if each:
        adapter = adapter_directory(store, last)
        if not _stand_in() and not adapter.is_dir():
            raise RuntimeError(f"{adapter} is not there: {model_name(last)}'s adapter is what this step samples from, and one trained again "
                               "would not be the model this run's other measurements came from. The measurement cannot be added to this run")
        _ensure_problems(store, CONTROL_TRAINED, CONTROL)
        summary["episodes"] = _episodes(config, store, CONTROL_TRAINED, each, seed, Engines(enable_lora=True).kit(_adapter(store, last)))
    else:
        summary["note"] = "no extra episodes: the control ran none"
    store.mark_done(CONTROL_TRAINED_STEP, summary)
    return summary


# -------------------------------------------------------------------------------------------------- report
def ladder_l2_report(config: dict) -> dict:
    from rlvr_lean.reporting.ladder_l2 import build_l2_report

    store = _store(config)
    sizes, prepare = _sizes(config, store), store.done_summary(PREPARE)
    rounds = {}
    for number in ROUNDS:
        if not store.is_done(f"ladder_l2_round_{number}"):
            break
        parts = range(1, sizes["batches"] + 1)
        trained, measured = store.is_done(f"ladder_l2_train_{number}"), store.is_done(f"ladder_l2_measure_{number}")
        rounds[number] = {
            "summary": store.done_summary(f"ladder_l2_round_{number}"),
            "proposals": {batch: store.read_rows(f"proposals_r{number}_b{batch}.jsonl") for batch in parts},
            "results": {batch: _results(store, batch_set(number, batch)) for batch in parts},
            "fits": {batch: store.done_summary(f"ladder_l2_propose_r{number}_b{batch}") for batch in parts},
            "examples": store.read_rows(f"training_examples_r{number}.jsonl"),
            "training": {key: value for key, value in store.done_summary(f"ladder_l2_train_{number}").items() if key != "losses"} if trained else None,
            "measure": store.done_summary(f"ladder_l2_measure_{number}") if measured else None,
            "rungs": _results(store, f"rungs_m{number}") if measured else None,
            "reach": _results(store, f"reach_m{number}") if measured else None}
    control = None
    if store.is_done(CONTROL_STEP):
        control = {"summary": store.done_summary(CONTROL_STEP)}
        control["results"] = _results(store, CONTROL) if control["summary"]["episodes_each"] else None
    control_trained = None          # a run that has not had the step (one finished before it existed) is reported as it always was
    if store.is_done(CONTROL_TRAINED_STEP):
        control_trained = {"summary": store.done_summary(CONTROL_TRAINED_STEP)}
        control_trained["results"] = _results(store, CONTROL_TRAINED) if control_trained["summary"]["episodes_each"] else None
    report = build_l2_report(prepare, store.read_rows("heldout_groups.jsonl"), store.read_rows("base_rungs.jsonl"), store.read_rows("base_reach.jsonl"),
                             rounds, control, list(ROUNDS), config["ladder_loop"], config["evaluation"],
                             {"embed": store.done_summary(EMBED) if store.is_done(EMBED) else None, "stopped_after_round": stopped_after(store)},
                             control_trained)
    _write_json(store, "report_ladder_l2.json", report)
    # Everything a reader needs beside the report goes out again from here (a rerun is another task with an output
    # directory of its own, and a task that failed hands over no step files): the per-problem results, the
    # proposals, the training examples, the held-out groups, and what each step recorded when it finished. Not the
    # attempts and not the per-block files, which are large and which the step that wrote them delivered.
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in sorted(path.name for path in store.root.glob("*.done.json")):
        if "_block_" not in name:
            store.mirror(name)
    store.mark_done(REPORT, {"headline": report["headline"], "branch": report["branch"]})
    return report


def _for_round(step, number: int):
    def run(config: dict) -> dict:
        return step(config, number)
    return run


STEPS = {
    PREPARE: ladder_l2_prepare,
    EMBED: ladder_l2_embed,
    **{f"ladder_l2_{name}_{number}": _for_round(step, number) for number in ROUNDS
       for name, step in (("round", ladder_l2_round), ("train", ladder_l2_train), ("measure", ladder_l2_measure))},
    CONTROL_STEP: ladder_l2_control,
    CONTROL_TRAINED_STEP: ladder_l2_control_trained,
    REPORT: ladder_l2_report,
}
