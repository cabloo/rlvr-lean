"""The L2 stage's arm WITH ASSEMBLY (L3d Step 2): what a batch does after its attempts, and how the arm's models are
trained. Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches", "Step 2, made exact before
it is built" and "Made exact by the build (Step 2)". Called by `gpu/ladder_l2.py` for an arm of
`ladder_loop.l2_assembly_arms`, and by no other arm; the rules are `domain/ladder_round/assembly.py`.

  assemble_batch   after a batch's attempts: every side of every problem that none of its attempts resolved is put
                   through the episode's Lean-only part over its 8 attempts in the order drawn
                   (`domain/repair/replay.py`, L3c's sizes), and each assembled proof is minimised as the harvest
                   minimises. THE ERROR POSITIONS are the round's own: the episode step kept, for every attempt Lean
                   rejected, the errors of the answer it already had (`ladder_loop.run_episodes(keep_errors=True)`),
                   so no attempt is checked again. Assembly's own checks (a pool's, a closer's, a minimisation's)
                   are the SOLVER's: the round's Lean service at the round's priority, and the batch is not settled
                   until they are back. A check the pool did not take is asked again and never read as a failure.
                   Every episode, and then every proof being minimised, advances as ITS OWN checks come back (a
                   thread each over the one client): a closing step that runs into the Lean limit holds up its
                   own episode and no other. The Lean files and their number are what one episode after another
                   would send.
  round_examples   the round's training examples: its one-shot rows and its assembled rows, each with its origin, and
                   H0's rows for the problems the rounds so far have not resolved themselves
  train_round      M(r): from the base, one pass over the training set in the order of a content hash
                   (`ladder_ceiling.one_pass`: the round's recipe, every row's loss, and the training loop's own
                   record of the rows), so that the twin is this order with rows left out
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping

from rlvr_lean.domain.ladder_round.assembly import (
    ATTEMPT,
    EXAMPLE_FIELDS,
    H0,
    assembled_rows,
    batch_counts,
    by_origin,
    check_harvest,
    episodes_to_assemble,
    h0_rows,
    raise_on_the_ruled_out_side,
    round_rows,
    to_minimise,
    training_order,
    training_set,
    with_assembly,
)
from rlvr_lean.domain.ladder_round.ceiling import the_training_took
from rlvr_lean.domain.ladder_round.l3d import tenth
from rlvr_lean.domain.problem_pool.selection import SoundnessAlarm
from rlvr_lean.domain.repair import replay
from rlvr_lean.gpu import ladder_loop
from rlvr_lean.gpu.ladder_l3a import lean_pool  # noqa: F401 - ONE client at the round's settings (an episode's Lean limit, no priority of its own)
from rlvr_lean.gpu.ladder_round import _attempts, _results, _stand_in, _write_json, training_seed
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

H0_VARIABLE = "RLVR_LEAN_LADDER_L3D2_H0"                 # another harvest than the package's (the fixture)
PACKAGE_H0 = Path(__file__).resolve().parents[1] / "data" / "ladder_l3d" / "harvest_h0.jsonl"
ADAPTERS = "adapters"
IN_PARALLEL = 64                                         # episodes (then proofs) advancing at once; the client's own limit is what reaches Lean


def assembly_marker(number: int, batch: int) -> str:
    return f"ladder_l2_assembly_r{number}_b{batch}"


def assembly_file(number: int, batch: int) -> str:
    return f"assembly_r{number}_b{batch}.jsonl"


def training_set_file(number: int) -> str:
    return f"training_set_m{number}.jsonl"


def loss_file(number: int) -> str:
    return f"training_loss_m{number}.json"


# ------------------------------------------------------------------------------------------------------ H0
def harvest_file() -> Path:
    override = os.environ.get(H0_VARIABLE)
    return Path(override) if override else PACKAGE_H0


def read_harvest(heldout: list[dict], base_map: list[dict]) -> tuple[list[dict], dict]:
    """(H0's rows, what is recorded of the file). It travels with the code; a copy without it, and an H0 with
    a held-out or base-map problem, are refused."""
    path = harvest_file()
    if not path.exists():
        raise RuntimeError(f"the harvest H0 {path} is not in this snapshot: it is built on the dev machine by `tools/ladder_harvest.py` (Lean only) and "
                           "committed with the code. Nothing was written")
    content = path.read_bytes()
    rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    check_harvest(rows, {row["problem_id"] for row in heldout}, {row["problem_id"] for row in base_map})
    return rows, {"h0_file": str(path), "h0_file_sha256": hashlib.sha256(content).hexdigest(), "h0_rows": len(rows)}


# ------------------------------------------------------------------------------------------- after a batch
def solver_check(pool, config: dict) -> Callable[[dict[str, str]], dict[str, dict]]:
    """Lean's raw answer for every file of a batch of assembly's checks, through the round's client. A file whose
    Lean header timed out (a cold worker) is asked once more; one the POOL did not take is asked again, up to
    `episode.pool_refusal_rounds` times, and then the step fails with nothing recorded (`LeanPoolRefused`), as an
    attempt's check does: a refusal is never read as a proof that failed."""
    rounds = config["ladder_loop"]["episode"].get("pool_refusal_rounds", 2)

    def ask(sources: dict[str, str]) -> dict[str, dict]:
        return {raw["id"]: raw for raw in pool.submit_sources({name: sources[name] for name in sorted(sources)}).result()}

    def check(sources: dict[str, str]) -> dict[str, dict]:
        if not sources:
            return {}
        answers = ask(sources)
        cold = [name for name, raw in answers.items() if ladder_loop._header_timed_out(replay.judged(name, raw))]
        if cold:
            answers.update(ask({name: sources[name] for name in cold}))
        for _ in range(rounds):
            refused = [name for name, raw in answers.items() if ladder_loop._pool_refused(replay.judged(name, raw))]
            if not refused:
                break
            answers.update(ask({name: sources[name] for name in refused}))
        refused = [name for name, raw in answers.items() if ladder_loop._pool_refused(replay.judged(name, raw))]
        if refused:
            raise ladder_loop.LeanPoolRefused(f"the Lean pool did not take {len(refused)} of {len(sources)} checks of a batch's assembly, asked again {rounds} "
                                              "times: nothing was recorded for the batch's assembly, run the step again")
        return answers
    return check


def each_on_its_own(items: list, work: Callable) -> tuple[Counter, Counter]:
    """`work(item, stats)` for every item, each in a thread of its own, so that one waits only for its own checks.
    Returns (what the calls returned, summed; their stats, summed). A soundness alarm is raised before any other
    failure, the first by the items' order."""
    def one(item) -> tuple[Counter, Counter]:
        stats: Counter = Counter()
        return Counter(work(item, stats) or {}), stats

    counts, stats, failures = Counter(), Counter(), []
    if items:
        with ThreadPoolExecutor(max_workers=min(IN_PARALLEL, len(items)), thread_name_prefix="ladder-assembly") as executor:
            futures = [executor.submit(one, item) for item in items]
        for future in futures:
            if future.exception() is not None:
                failures.append(future.exception())
            else:
                counts.update(future.result()[0])
                stats.update(future.result()[1])
    if failures:
        raise next((error for error in failures if isinstance(error, SoundnessAlarm)), failures[0])
    return counts, stats


def _errors(store: ArtifactStore, set_name: str) -> dict[str, list]:
    """Lean's positioned errors for the batch's rejected attempts, as the episode step kept them."""
    return {row["attempt_id"]: row["errors"] for path in sorted(store.root.glob(f"episodes_{set_name}_errors_*.jsonl"))
            for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def assemble_batch(config: dict, store: ArtifactStore, number: int, batch: int, set_name: str) -> dict:
    """The assembly of one batch, once its attempts are settled; a rerun returns what is stored. Writes the row of
    every assembled resolution (`assembly_r<r>_b<b>.jsonl`) and puts `resolved_by_assembly` beside the counts of
    the batch's per-problem results."""
    marker, results_file = assembly_marker(number, batch), f"episodes_{set_name}_problems.jsonl"
    if not store.is_done(marker):
        sizes = {"pool_blocks": config["ladder_loop"]["accumulate"]["pool_blocks"], "kept_closers": config["ladder_loop"]["accumulate"]["kept_closers"]}
        results = [{key: value for key, value in row.items() if key != "resolved_by_assembly"} for row in _results(store, set_name)]
        problems = [row for row in store.read_rows("problems.jsonl") if row["set"] == set_name]
        episodes, not_kept = episodes_to_assemble(problems, results, _attempts(store, set_name), _errors(store, set_name))
        if not_kept:
            raise RuntimeError(f"{len(not_kept)} rejected attempts of {set_name} have no stored errors (first: {not_kept[0]}): the episode step of this arm keeps "
                               "Lean's errors for every attempt it rejects, and assembly cuts a failed proof by them. Nothing was recorded for the batch's assembly")
        with lean_pool(config) as pool:
            check = solver_check(pool, config)
            checks, stats = each_on_its_own([episode for episode in episodes if episode["chain"]], lambda episode, own: replay.assemble(
                [episode], check, sizes, episode["attempts"], own, raise_on_the_ruled_out_side))
            proofs = to_minimise(episodes)
            stats.update(each_on_its_own(proofs, lambda proof, own: replay.minimise([proof], check, own))[1])
        rows = assembled_rows(proofs, number, batch)
        store.write_rows(assembly_file(number, batch), rows)
        store.mark_done(marker, {"round": number, "batch": batch, "set": set_name, **batch_counts(results, episodes, rows, checks, stats), "sizes": sizes,
                                 "error_positions": "the round's own checks (kept by the episode step): no attempt was checked again"})
    # The batch's per-problem results say what assembly resolved; written again from what is stored, so a rerun gives the same file.
    store.write_rows(results_file, with_assembly(_results(store, set_name), store.read_rows(assembly_file(number, batch))))
    store.mirror(results_file)
    return store.done_summary(marker)


# -------------------------------------------------------------------------------------- the training set
def round_examples(store: ArtifactStore, number: int, sets: list[str], examples: list[dict], batch_of: Mapping[str, int], rounds: tuple[int, ...],
                   harvest: list[dict]) -> list[dict]:
    """The rows of `training_examples_r<number>.jsonl` for this arm: the round's own (one-shot, then assembled), and
    H0's rows for the problems rounds 1 to `number` have not resolved themselves."""
    assembled = [row for batch in range(1, len(sets) + 1) for row in store.read_rows(assembly_file(number, batch))]
    own = round_rows(examples, assembled, number, batch_of)
    resolved = {row["problem_id"] for row in own}
    for earlier in rounds:
        if earlier < number:
            resolved |= {row["problem_id"] for row in store.read_rows(f"training_examples_r{earlier}.jsonl") if row["origin"] != H0}
    return own + h0_rows(harvest, resolved, number)


def train_round(config: dict, store: ArtifactStore, number: int, rounds: tuple[int, ...], arm: str, start: Path | None = None) -> dict:
    """M(number) of this arm: from the base, one pass over its training set in the order of a content hash of the
    task's seed and each row's id, the round's recipe. Stores the set in that order (`training_set_m<r>.jsonl`),
    every row's loss and the training loop's own record of the rows (`training_loss_m<r>.json`). `start`: the stored
    adapter an arm's models are trained FROM in the place of the base (L4: `pre`)."""
    from rlvr_lean.gpu.ladder_ceiling import one_pass      # imported here: that module reads this stage's (`ladder_l2`)

    seed, batch = training_seed(config), config["training"]["effective_batch"]
    by_round = {earlier: store.read_rows(f"training_examples_r{earlier}.jsonl") for earlier in rounds if earlier <= number}
    rows = training_order(training_set(by_round, number), seed)
    if not rows:
        raise RuntimeError(f"rounds 1 to {number} resolved no problem and H0 is empty: there is nothing to train M({number}) on")
    store.write_rows(training_set_file(number), [{"row": position, **row} for position, row in enumerate(rows)])
    examples = [{field: row[field] for field in EXAMPLE_FIELDS} for row in rows]
    steps = -(-len(rows) // batch)
    result, step_rows, row_losses, positions = one_pass(config, examples, {f"m{number}": steps}, seed, batch, store.root / ADAPTERS,
                                                        f"ladder_l2_{arm}_m{number}_seed{seed}", positions=True, **({"start": start} if start is not None else {}))
    result.pop("checkpoints")
    _write_json(store, loss_file(number), {
        "round": number, "model": f"M({number})", "seed": seed, "stand_in_engine": _stand_in(), "rows": len(row_losses), "steps": result["steps"],
        "effective_batch": batch,
        "what": "row_losses: each training row's mean loss per target token, in the order trained, read BEFORE the update of the optimizer step it was in. "
                "rows_trained: the id of the row each of those losses is of, from the positions the training loop recorded step by step",
        "row_losses": row_losses, "rows_trained": [rows[position]["id"] for position in positions],
        "training_steps": [{**row, "rows_seen": min(row["step"] * batch, len(row_losses))} for row in step_rows]})
    took = the_training_took(row_losses, tenth(len(row_losses), 0.1))
    return {"round": number, "model": f"M({number})", "seed": seed, "trained_from": "the base" if start is None else f"the stored adapter {start}",
            "rounds_trained_on": sorted(by_round),
            "rows": len(rows), "rows_by_origin": by_origin(rows),
            "rows_by_round": {str(earlier): sum(row["round"] == earlier and row["origin"] != H0 for row in rows) for earlier in sorted(by_round)},
            "order": "a content hash of the task's seed and each row's id (`assembly.training_order`): leaving rows out moves no other",
            "one_shot_rows": sum(row["origin"] == ATTEMPT for row in rows), "first_step_loss": step_rows[0]["mean_loss"],
            "last_step_loss": step_rows[-1]["mean_loss"], "mean_loss_over_the_first_rows": took["first"], "mean_loss_over_the_last_rows": took["last"],
            "rows_compared": took["rows_compared"], "stand_in_engine": _stand_in(), **result}
