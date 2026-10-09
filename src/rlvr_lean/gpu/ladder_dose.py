"""The ladder loop's L1b: the dose curve. Spec: docs/spec/ladder-loop.spec.md, "L1b: the dose curve".

One seed's challenger arm again, from the base, on the SAME training proofs L1 stored, for `dose.passes` passes in
place of one, with the loss read over time on the training set and on held-out proofs, and the held-out pass rate
measured at checkpoints. Each step is its own process (`rlvr_lean.runner.entry`, stage `ladder_l1b`).

  ladder_l1b_prepare          reads L1's run directory for this seed ON THE BOX (never writes there): the stored
                              training examples, the rung and goal problems with their negations, the base's rung
                              attempts and results. Draws the held-out proofs and the training sample, fixes the
                              schedule, and refuses when anything held out is a training example. No GPU, no Lean.
  ladder_l1b_train            the training with readings: the per-step loss by part (read before the update), the
                              loss on the fixed pairs at the reading points, an adapter saved at every checkpoint
  ladder_l1b_measure_<name>   one checkpoint: 8 episodes on every problem of the three rungs with L1's sampling seed
                              for the rungs (so it pairs by problem with L1's stored base results and with the other
                              checkpoints), reach on G at the checkpoints `dose.reach_at` names, distinct attempts
  ladder_l1b_report           the three loss series, the read of the spec's table, the VOID check and the branch

The run directory is its own (`ladder_l1b_seed<seed>`): nothing here can mark a step of L1's run done.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

from rlvr_lean.domain.ladder_round import CHALLENGER_ARM
from rlvr_lean.domain.ladder_round.dose import (
    check_nothing_held_out_is_trained_on,
    checkpoint_steps,
    heldout_pairs,
    parts_row,
    reading_steps,
    run_dose,
    steps_per_pass,
    training_sample,
)
from rlvr_lean.domain.problem_pool.episodes import RUNGS
from rlvr_lean.domain.training.target_format import NATIVE
from rlvr_lean.domain.verification.pin import lean_pin_from_config
from rlvr_lean.gpu import ladder_loop, ladder_round, pipeline
from rlvr_lean.gpu.ladder_round import BASE, Engines, _attempts, _stand_in, _write_json, distinct_attempts, training_seed
from rlvr_lean.infrastructure.adapter_files import ADAPTER_FILE, compared
from rlvr_lean.infrastructure.artifact_store import ArtifactStore

DOSE_RUN_VARIABLE = "RLVR_LEAN_LADDER_DOSE_RUN"          # another run directory than `ladder_l1b_seed<seed>` (a smoke run)
DOSE_SOURCE_VARIABLE = "RLVR_LEAN_LADDER_DOSE_SOURCE"    # another L1 run directory to read than `ladder_l1_seed<seed>`
ARM = CHALLENGER_ARM
PREPARE, TRAIN, REPORT = "ladder_l1b_prepare", "ladder_l1b_train", "ladder_l1b_report"
CHECKPOINTS = ("p050", "p100", "p150", "p200", "p300")   # the stage's measure steps; `dose.checkpoints` must give these names
MAX_LEFTOVER_GB = pipeline.MAX_LEFTOVER_GB
SOURCE_FILES = (f"training_examples_{ARM}.jsonl", "problems.jsonl", "heldout_groups.jsonl", f"episodes_rungs_{BASE}_problems.jsonl",
                f"episodes_reach_{BASE}_problems.jsonl", f"episodes_rungs_{BASE}.done.json", f"episodes_reach_{BASE}.done.json",
                "report_ladder_l1.json")


# ------------------------------------------------------------------------------------------------- the run
def _runs(config: dict) -> Path:
    return pipeline.STORE / lean_pin_from_config(ladder_loop.ladder_config(config)).runs_directory


def _store(config: dict) -> ArtifactStore:
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    run = os.environ.get(DOSE_RUN_VARIABLE) or f"ladder_l1b_seed{training_seed(config)}"
    return ArtifactStore(_runs(config) / run, Path(mirror) if mirror else None)


def source_directory(config: dict) -> Path:
    """L1's run directory for this seed. It is only ever READ, with plain file reads: nothing is created in it."""
    return _runs(config) / (os.environ.get(DOSE_SOURCE_VARIABLE) or f"ladder_l1_seed{training_seed(config)}")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def adapter_directory(store: ArtifactStore, name: str) -> Path:
    return store.root / "adapters" / name


def _dose(config: dict) -> dict:
    dose = config["ladder_loop"]["dose"]
    names = tuple(checkpoint_steps(8, 8, dose["checkpoints"]))
    if names != CHECKPOINTS:
        raise ValueError(f"ladder_loop.dose.checkpoints gives {names} and the stage measures {CHECKPOINTS}: change both together")
    return dose


def _passes(dose: dict) -> dict[str, float]:
    return dict(zip(CHECKPOINTS, dose["checkpoints"]))


def reach_checkpoints(config: dict) -> list[str]:
    """The checkpoints whose reach on G is read: what the config says NOW, whatever the prepare step recorded. A
    finished run can so be asked for one more reach measurement (the three-seed escalation reads G at 2 passes
    too): its measure step then runs that measurement and nothing else."""
    dose = _dose(config)
    passes = _passes(dose)
    return [name for name in CHECKPOINTS if passes[name] in dose["reach_at"]]


def _ensure_reach_problems(store: ArtifactStore, name: str) -> None:
    """A run prepared before this checkpoint was asked for reach has no problems under its set name: they are the
    goal problems another checkpoint's set holds, with this set's name."""
    rows, wanted = store.read_rows("problems.jsonl"), f"reach_{name}"
    if any(row["set"] == wanted for row in rows):
        return
    held = next((f"reach_{other}" for other in CHECKPOINTS if any(row["set"] == f"reach_{other}" for row in rows)), None)
    if held is None:
        raise RuntimeError(f"{store.root} holds the goal problems under no checkpoint's set: reach at {name} cannot be added to this run")
    store.write_rows("problems.jsonl", rows + [{**row, "set": wanted} for row in rows if row["set"] == held])


# ------------------------------------------------------------------------------------------------- prepare
def ladder_l1b_prepare(config: dict) -> dict:
    store = _store(config)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    dose, seed, source = _dose(config), training_seed(config), source_directory(config)
    attempt_files = sorted(source.glob(f"episodes_rungs_{BASE}_attempts_*.jsonl")) if source.is_dir() else []
    missing = [name for name in SOURCE_FILES if not (source / name).exists()]
    if missing or not attempt_files:
        if not attempt_files:
            missing.append(f"episodes_rungs_{BASE}_attempts_*.jsonl (the base model's rung attempts)")
        raise RuntimeError(
            f"{source} does not hold {missing}. "
            f"L1b reads the run directory the L1 task for seed {seed} wrote on this box (stage `ladder_l1` with --seeds {seed}: "
            f"`python -m rlvr_lean.runner.entry --stage ladder_l1 --seeds {seed}`; for the smoke run, stage `ladder_l1_smoke`). "
            "Run that task to its end first; nothing was written.")
    examples = _rows(source / f"training_examples_{ARM}.jsonl")
    groups = _rows(source / "heldout_groups.jsonl")
    problems = _rows(source / "problems.jsonl")
    rung_problems = [row for row in problems if row["set"] == f"rungs_{BASE}"]
    goal_problems = [row for row in problems if row["set"] == f"reach_{BASE}"]
    base_attempts = [row for path in attempt_files for row in _rows(path)]
    measure = config["ladder_loop"]["measure"]
    seeds = {"rungs": ladder_round.sampling_seed(config, f"rungs_{BASE}"), "reach": ladder_round.sampling_seed(config, f"reach_{BASE}")}
    stored = {kind: json.loads((source / f"episodes_{kind}_{BASE}.done.json").read_text()) for kind in ("rungs", "reach")}
    for kind, episodes in (("rungs", measure["rung_episodes"]), ("reach", measure["reach_episodes"])):
        if stored[kind].get("problems") and (stored[kind]["sampling_seed"] != seeds[kind] or stored[kind]["episodes_each"] != episodes):
            raise RuntimeError(f"L1 measured the base on its {kind} with sampling seed {stored[kind]['sampling_seed']} and {stored[kind]['episodes_each']} "
                               f"episodes; this config gives {seeds[kind]} and {episodes}: the checkpoints would not pair with L1's stored results")
    pairs = heldout_pairs(groups, rung_problems, base_attempts, seed, dose["heldout_per_rung"])
    check_nothing_held_out_is_trained_on(examples, pairs, groups)
    sample = training_sample(examples, seed, dose["training_sample"])
    batch = config["training"]["effective_batch"]
    per_pass = steps_per_pass(len(examples), batch)
    checkpoints = checkpoint_steps(len(examples), batch, dose["checkpoints"])
    total = dose["passes"] * per_pass
    readings = reading_steps(total, dose["reading_every_step_until"], dose["reading_every"], list(checkpoints.values()))

    def as_set(rows: list[dict], set_name: str) -> list[dict]:
        return [{**row, "set": set_name} for row in rows]

    passes = _passes(dose)
    own = [row for name in CHECKPOINTS for row in as_set(rung_problems, f"rungs_{name}")]
    own += [row for name in CHECKPOINTS if passes[name] in dose["reach_at"] for row in as_set(goal_problems, f"reach_{name}")]
    store.write_rows("problems.jsonl", own)
    store.write_rows("training_examples.jsonl", examples)
    store.write_rows("training_sample.jsonl", [{"position": position, "attempt_id": examples[position]["attempt_id"]} for position in sample])
    store.write_rows("heldout_pairs.jsonl", pairs)
    store.write_rows("heldout_groups.jsonl", groups)
    store.write_rows("base_rungs.jsonl", _rows(source / f"episodes_rungs_{BASE}_problems.jsonl"))
    store.write_rows("base_reach.jsonl", _rows(source / f"episodes_reach_{BASE}_problems.jsonl"))
    l1_report = json.loads((source / "report_ladder_l1.json").read_text())
    reference = {"source_run": source.name, "rungs_trained_minus_base": l1_report["arms"][ARM]["rungs_trained_minus_base"],
                 "reach_on_g": l1_report["arms"][ARM]["reach_on_g"], "headline": l1_report.get("headline"),
                 "base_distinct_attempts_on_the_rungs": distinct_attempts(base_attempts)}
    _write_json(store, "l1_reference.json", reference)
    by_rung = {rung: sum(pair["rung"] == rung for pair in pairs) for rung in RUNGS}
    summary = {"seed": seed, "arm": ARM, "source_run": str(source), "stand_in_engine": _stand_in(),
               "training_examples": len(examples),
               "training_examples_sha256": hashlib.sha256((source / f"training_examples_{ARM}.jsonl").read_bytes()).hexdigest(),
               "training_sample": len(sample), "heldout_pairs": len(pairs), "heldout_pairs_by_rung": by_rung,
               "heldout_pairs_short_by": {rung: dose["heldout_per_rung"] - by_rung[rung] for rung in RUNGS},
               "heldout_pairs_on_the_negation": sum(pair["side"] == "negation" for pair in pairs),
               "rung_problems": len(rung_problems), "goal_problems": len(goal_problems),
               "passes": dose["passes"], "effective_batch": batch, "steps_per_pass": per_pass, "steps": total,
               "checkpoint_steps": checkpoints, "checkpoint_passes": passes, "reading_steps": readings,
               "sampling_seeds": seeds, "rung_episodes": measure["rung_episodes"], "reach_episodes": measure["reach_episodes"],
               "reach_at": [name for name in CHECKPOINTS if passes[name] in dose["reach_at"]],
               "l1_contradicted_side_setting": stored["rungs"].get("contradicted_side_setting"),
               "contradicted_side_setting": config["ladder_loop"]["episode"].get("contradicted_side", "all")}
    store.mark_done(PREPARE, summary)
    return summary


# --------------------------------------------------------------------------------------------------- train
def _stand_in_calls(examples: int, sample: list[int], pairs: list[dict]):
    """Stand-ins for the three calls of `run_dose`: made-up losses that fall with the updates on the training
    side and turn back up on the held-out side. NOT a model: a pre-flight of the schedule and the files."""
    state = {"updates": 0}

    def losses(value: float, length: int) -> list[float]:
        return [round(2 * value, 4), *[round(value, 4)] * length, round(value / 4, 4), round(value / 8, 4)]

    def train_step(batch: list[int]) -> list[list[float]]:
        rows = [losses(1.0 / (1 + state["updates"]) + 0.001 * (index % 7), 3 + index % 4) for index in batch]
        state["updates"] += 1
        return rows

    def read(step: int) -> dict:
        held = 0.3 + 0.002 * (step - 4) ** 2
        row = {"training_sample": parts_row([losses(1.0 / (1 + step), 4) for _ in sample])} if sample else {"training_sample": None}
        row["heldout"] = parts_row([losses(held, 4) for _ in pairs]) if pairs else None
        row["heldout_by_rung"] = {rung: parts_row([losses(held, 4) for pair in pairs if pair["rung"] == rung])
                                  for rung in RUNGS if any(pair["rung"] == rung for pair in pairs)}
        return row

    return train_step, read, lambda name, step: None


def _train_with_readings(config: dict, examples: list[dict], sample: list[int], pairs: list[dict], schedule: dict, seed: int,
                         directory: Path, tensorboard_run: str, orders: list[list[int]] | None = None, per_example: bool = False,
                         positions: bool = False, start: Path | None = None) -> dict:
    """The training of `ladder_round._train` (same model, adapter, optimizer, warm-up and order of examples), for
    `dose.passes` passes, with the per-step loss by part, the readings on the fixed pairs and the checkpoints.
    EVERY GPU object is a local of this function, so nothing outlives it. `orders`, `per_example` and `positions`
    are `run_dose`'s: another order of the examples than the round's, each example's own loss beside its step's,
    and the positions of the examples each step was made on. `start` (L4's arm): a stored adapter the training goes
    on FROM, in the place of a fresh one on the base; its weights are loaded into the adapter just attached, and the
    training is refused, before its first step, unless the B matrices the model then holds are the file's (a fresh
    adapter's are zero, a pretrained one's are not: their absolute sum is compared with the file's). After the pass,
    each adapter it SAVED is compared with the start adapter's, file against file and bit for bit
    (`against_the_start_adapter`, by `infrastructure.adapter_files.compared`: how many of its numbers are not the
    same): a trained adapter that IS the one it started from is a training that did not happen."""
    import torch

    from rlvr_lean.runner.heartbeat import ScalarEventWriter
    from rlvr_lean.gpu.model_utils import attach_lora, encode_pair, load_nf4_base, position_losses, training_text

    training = config["training"]
    model, tokenizer = load_nf4_base(pipeline.NF4_DIR)
    peft_model = attach_lora(model, config["lora"], seed=seed)
    if start is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        stored = load_file(str(Path(start) / ADAPTER_FILE))
        set_peft_model_state_dict(peft_model, stored)
        of_the_file = sum(float(tensor.float().abs().sum()) for key, tensor in stored.items() if "lora_B" in key)
        loaded = sum(float(parameter.detach().float().abs().sum()) for name, parameter in peft_model.named_parameters() if "lora_B" in name)
        if of_the_file <= 0 or abs(loaded - of_the_file) > 1e-3 * of_the_file:
            raise RuntimeError(f"the adapter {start} did not load: its B matrices sum to {of_the_file} in absolute value and the model to train holds {loaded} "
                               "(a fresh adapter on the base holds 0). Nothing was trained")
        start_loaded = {"adapter": str(start), "b_matrices_absolute_sum_of_the_file": round(of_the_file, 4), "b_matrices_absolute_sum_loaded": round(loaded, 4)}
    peft_model.train()          # gradient checkpointing is applied in training mode only (see `ladder_round._train`)
    optimizer = torch.optim.AdamW([parameter for parameter in peft_model.parameters() if parameter.requires_grad],
                                  lr=training["learning_rate"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / training["warmup_steps"]))

    def encode(theorem: str, completion: str):
        return encode_pair(tokenizer, *training_text(theorem, completion), training["max_sequence_tokens"], NATIVE)

    encoded = [encode(example["theorem"], example["completion"]) for example in examples]
    sample_encoded = [encoded[position] for position in sample]
    heldout_encoded = [(pair["rung"], encode(pair["theorem"], pair["completion"])) for pair in pairs]
    root = os.environ.get("RLVR_LEAN_TB_DIR")
    tensorboard = ScalarEventWriter(Path(root) / tensorboard_run) if root else None
    reading_seconds, counters = [], {"step": 0}

    def train_step(batch: list[int]) -> list[list[float]]:
        per_pair, step_loss = [], 0.0
        for index in batch:
            ids, mask = encoded[index]
            input_ids = torch.tensor([ids], device="cuda")
            labels = torch.tensor([[token if keep else -100 for token, keep in zip(ids, mask)]], device="cuda")
            output = peft_model(input_ids=input_ids, labels=labels)     # the call of `mean_proof_token_loss`: the same loss, the same gradient
            loss = output.loss / len(batch)
            loss.backward()
            step_loss += float(loss.detach())
            with torch.no_grad():       # the same forward pass, by position: what the loss was BEFORE this step's update
                positions = torch.tensor([position for position in range(1, len(ids)) if mask[position]], device=output.logits.device)
                log_probabilities = torch.log_softmax(output.logits[0][positions - 1].float(), dim=-1)
                per_pair.append((-log_probabilities.gather(1, input_ids[0, positions].unsqueeze(1)).squeeze(1)).tolist())
            del loss, output
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        counters["step"] += 1
        if tensorboard is not None:
            parts = parts_row(per_pair)
            tensorboard.scalars(counters["step"], {"train/loss": step_loss, "train/body_nats_per_token": parts["body_nats_per_token"],
                                                   "train/first_token_nats": parts["first_token_nats"]})
        return per_pair

    def read(step: int) -> dict:
        began = time.monotonic()
        peft_model.eval()               # forward passes only, without gradients; then back to training mode
        with torch.no_grad():
            on_sample = [position_losses(peft_model, ids, mask)[0] for ids, mask in sample_encoded]
            on_heldout = [(rung, position_losses(peft_model, ids, mask)[0]) for rung, (ids, mask) in heldout_encoded]
        peft_model.train()
        row = {"training_sample": parts_row(on_sample) if on_sample else None,
               "heldout": parts_row([losses for _, losses in on_heldout]) if on_heldout else None,
               "heldout_by_rung": {rung: parts_row([losses for name, losses in on_heldout if name == rung])
                                   for rung in RUNGS if any(name == rung for name, _ in on_heldout)}}
        reading_seconds.append(time.monotonic() - began)
        if tensorboard is not None:
            values = {f"{name}/{key}": row[name][key] for name in ("training_sample", "heldout") if row[name]
                      for key in ("mean_loss", "body_nats_per_token", "first_token_nats")}
            values.update({f"heldout_{rung}/body_nats_per_token": parts["body_nats_per_token"] for rung, parts in row["heldout_by_rung"].items()})
            if values:
                tensorboard.scalars(step, values)
        return row

    def save(name: str, step: int) -> None:
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(target))

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = run_dose(len(encoded), training["effective_batch"], schedule["passes"], seed, schedule["reading_steps"],
                      schedule["checkpoint_steps"], train_step, read, save, orders=orders, per_example=per_example, positions=positions)
    seconds = time.monotonic() - started
    # What was SAVED against what the training started from, file against file and bit for bit: a count of the numbers
    # that are not the same, so nothing here passes or fails by rounding.
    against = {name: compared(directory / name / ADAPTER_FILE, Path(start) / ADAPTER_FILE) for name in schedule["checkpoint_steps"]} if start is not None else {}
    return {**result, "target_format": NATIVE, "sequence_start_token_id": int(tokenizer.bos_token_id),
            **({"start_adapter_loaded": start_loaded, "against_the_start_adapter": against} if start is not None else {}),
            "seconds": round(seconds, 1), "seconds_in_readings": round(sum(reading_seconds), 1),
            "seconds_per_reading": round(sum(reading_seconds) / len(reading_seconds), 2) if reading_seconds else None,
            "seconds_training": round(seconds - sum(reading_seconds), 1),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "gradient_checkpointing_active": bool(peft_model.training and getattr(peft_model.base_model.model, "is_gradient_checkpointing", False))}


def ladder_l1b_train(config: dict) -> dict:
    store = _store(config)
    if store.is_done(TRAIN):
        return store.done_summary(TRAIN)
    schedule, seed = store.done_summary(PREPARE), training_seed(config)
    examples = store.read_rows("training_examples.jsonl")
    sample = [row["position"] for row in store.read_rows("training_sample.jsonl")]
    pairs = store.read_rows("heldout_pairs.jsonl")
    if _stand_in():
        batch = config["training"]["effective_batch"]
        result = run_dose(len(examples), batch, schedule["passes"], seed, schedule["reading_steps"], schedule["checkpoint_steps"],
                          *_stand_in_calls(len(examples), sample, pairs))
        result["note"] = "no adapter was trained and every loss is made up: a pre-flight without a GPU"
    else:
        import torch

        pipeline._cap_torch_memory(config)
        result = _train_with_readings(config, examples, sample, pairs, schedule, seed, store.root / "adapters", f"ladder_l1b_seed{seed}")
        gc.collect()
        torch.cuda.empty_cache()
        result["allocated_after_cleanup_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
    curves = {"seed": seed, "steps": result["steps"], "steps_per_pass": result["steps_per_pass"], "stand_in_engine": _stand_in(),
              "training_steps": result.pop("training_steps"), "readings": result.pop("readings"), "checkpoints": result["checkpoints"]}
    _write_json(store, "loss_curves.json", curves)
    summary = {"seed": seed, "arm": ARM, "examples": len(examples), "passes": schedule["passes"], "stand_in_engine": _stand_in(),
               "readings": len(curves["readings"]), "first_step_loss": curves["training_steps"][0]["mean_loss"],
               "last_step_loss": curves["training_steps"][-1]["mean_loss"], **result}
    store.mark_done(TRAIN, summary)         # the checkpoints are saved: keep them whatever follows
    if summary.get("allocated_after_cleanup_gb", 0) > MAX_LEFTOVER_GB:
        raise RuntimeError(f"{summary['allocated_after_cleanup_gb']} GB is still allocated on the GPU after the dose curve's training was released")
    return summary


# ------------------------------------------------------------------------------------------------- measure
def _episodes(config: dict, store: ArtifactStore, set_name: str, episodes: int, seed: int, kit) -> dict:
    if not any(row["set"] == set_name for row in store.read_rows("problems.jsonl")):
        store.write_rows(f"episodes_{set_name}_problems.jsonl", [])      # a set with no problem is measured as such
        return {"set": set_name, "problems": 0, "episodes_each": episodes, "attempts": 0, "statuses": {}}
    with ladder_loop.lean_sessions(config) as sessions:
        return ladder_loop.run_episodes(config, set_name, episodes, seed, store, kit, sessions)


def ladder_l1b_measure(config: dict, name: str) -> dict:
    """One checkpoint on the three held-out rungs (and on G where `dose.reach_at` says), with L1's sampling seeds:
    the same problems, the same random numbers and the same sides as L1's base measurement and as every other
    checkpoint, so each pairs by problem with them. A checkpoint that is done and is now asked for reach as well
    gets the reach measurement alone: its rung results are kept as they are."""
    store, marker = _store(config), f"ladder_l1b_measure_{name}"
    reach_wanted = name in reach_checkpoints(config)
    summary = store.done_summary(marker) if store.is_done(marker) else None
    if summary is not None and (not reach_wanted or "reach" in summary):
        return summary
    prepare, measure = store.done_summary(PREPARE), config["ladder_loop"]["measure"]
    engines = Engines(enable_lora=True)
    if _stand_in():
        adapter = None
    else:
        from vllm.lora.request import LoRARequest

        adapter = LoRARequest(f"ladder_l1b_{name}", 1, str(adapter_directory(store, name)))
    kit = engines.kit(adapter)
    if summary is None:
        summary = {"checkpoint": name, "step": prepare["checkpoint_steps"][name], "pass": prepare["checkpoint_passes"][name],
                   "rungs": _episodes(config, store, f"rungs_{name}", measure["rung_episodes"], prepare["sampling_seeds"]["rungs"], kit)}
    if reach_wanted:
        _ensure_reach_problems(store, name)
        summary["reach"] = _episodes(config, store, f"reach_{name}", measure["reach_episodes"], prepare["sampling_seeds"]["reach"], kit)
    summary["distinct_attempts_on_the_rungs"] = distinct_attempts(_attempts(store, f"rungs_{name}"))
    summary["stand_in_engine"] = _stand_in()
    store.mark_done(marker, summary)
    return summary


# -------------------------------------------------------------------------------------------------- report
def ladder_l1b_report(config: dict) -> dict:
    from rlvr_lean.reporting.ladder_l1b import build_l1b_report

    store = _store(config)
    prepare, train = store.done_summary(PREPARE), store.done_summary(TRAIN)
    measured = {name: store.done_summary(f"ladder_l1b_measure_{name}") for name in CHECKPOINTS}
    report = build_l1b_report(
        prepare, {key: value for key, value in train.items() if key != "checkpoints"}, json.loads(store.path("loss_curves.json").read_text()),
        store.read_rows("heldout_groups.jsonl"), store.read_rows("base_rungs.jsonl"), store.read_rows("base_reach.jsonl"),
        {name: store.read_rows(f"episodes_rungs_{name}_problems.jsonl") for name in CHECKPOINTS},
        {name: store.read_rows(f"episodes_reach_{name}_problems.jsonl") for name in CHECKPOINTS if "reach" in measured[name]},
        measured, json.loads(store.path("l1_reference.json").read_text()), config["evaluation"])
    _write_json(store, "report_ladder_l1b.json", report)
    # Everything a reader needs beside the report goes out again from here (a rerun is another task with an output
    # directory of its own): the per-problem results, the pairs, the curves. Not the attempts, not the per-block files.
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    for name in ("loss_curves.json", "l1_reference.json"):
        store.mirror(name)
    store.mark_done(REPORT, {"headline": report["headline"], "branch": report["branch"]})
    return report


def _for_checkpoint(name: str):
    def run(config: dict) -> dict:
        return ladder_l1b_measure(config, name)
    return run


STEPS = {
    PREPARE: ladder_l1b_prepare,
    TRAIN: ladder_l1b_train,
    **{f"ladder_l1b_measure_{name}": _for_checkpoint(name) for name in CHECKPOINTS},
    REPORT: ladder_l1b_report,
}
