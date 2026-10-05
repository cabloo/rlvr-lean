"""The ladder loop's L1: ONE round, two arms that differ only in who chooses the round's problems.
Spec: docs/spec/ladder-loop.spec.md, "A round", "The challenger in this stage", "The equal-compute control",
"L1's read, fixed now". Each step is run as its own process by `rlvr_lean.runner.entry`.

  ladder_l1_prepare          the data that travels with the code (L0's H and base map, the round's candidates,
                             the base's placing results), checked; G and the rungs cut from the placing results;
                             the exact negation of every held-out problem that is measured. No GPU.
  ladder_l1_embed            the base model's embedding of every base-map and candidate statement, once
  ladder_l1_propose          the challenger fitted on the base map, its calibration on held-back problems, and
                             BOTH arms' problems: the challenger's choice and the uniform draw. No GPU.
  ladder_l1_base             the base's fresh episodes on G (its luck) and on the rungs, once for both arms
  ladder_l1_round_<arm>      the round: n episodes of the base on the arm's problems (k of n for each)
  ladder_l1_train_<arm>      one pass, native format, on one verified proof of every problem with k >= 1
  ladder_l1_measure_<arm>    the trained model on G, on the rungs and again on the round's problems, and the
                             base again on the round's problems (the gain by k compares fresh with fresh)
  ladder_l1_report           L1's read and the branch it selects

Both arms use the same base, the same seed and one box. M(0) is the base model, so the round's episodes and the
"model the round started from" are the base's. Each kind of measurement has a sampling seed of its own, shared by
both arms and by both sides of a comparison (`SAMPLING_KINDS`). The steps run at
the ladder loop's Lean pin in a run directory of their own (`ladder_l1_seed<seed>`), and a rerun resumes at the
first step, and the first block, that is not done. A verified proof on the side a certificate contradicts stops
the step with the soundness alarm's exit code, as in L0.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np

from rlvr_lean.data.ladder_round_export import BASE_MAP_SET, HELDOUT_SET, check_round_data, load_round_data
from rlvr_lean.domain.ladder_round import ARMS, CHALLENGER_ARM, RANDOM_ARM
from rlvr_lean.domain.ladder_round.challenger import (
    calibration,
    expected_rewards,
    feature_names,
    features_of,
    fit_pass_rate_model,
    fit_projection,
    propose,
    random_draw,
    recency_weights,
)
from rlvr_lean.domain.ladder_round.read import GOAL, arm_reward, group_ids, heldout_groups, rung_ids
from rlvr_lean.domain.ladder_round.training_set import training_examples, training_summary
from rlvr_lean.domain.problem_pool.episodes import NEGATION_NOT_EXACT, RUNGS
from rlvr_lean.domain.problem_pool.selection import rank
from rlvr_lean.domain.proving import build_prover_prompt
from rlvr_lean.domain.training.target_format import NATIVE
from rlvr_lean.domain.verification.pin import lean_pin_from_config
from rlvr_lean.gpu import ladder_loop, pipeline
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.infrastructure.verification_service import check_lean_sources

ROUND_DATA_VARIABLE = "RLVR_LEAN_LADDER_ROUND_DATA"      # another data directory than the package's (the fixture, a pre-flight)
ROUND_RUN_VARIABLE = "RLVR_LEAN_LADDER_ROUND_RUN"        # another run directory than `ladder_l1_seed<seed>` (a smoke run)
PACKAGE_ROUND_DATA = Path(__file__).resolve().parents[1] / "data" / "ladder_l1"
BASE = "base"
PREPARE, EMBED, PROPOSE, BASE_STEP, REPORT = "ladder_l1_prepare", "ladder_l1_embed", "ladder_l1_propose", "ladder_l1_base", "ladder_l1_report"
# Each KIND of measurement samples with a seed of its own: `round.sampling_seed` + 100 x the training seed + its place
# here. A kind's seed is shared by both arms and by the base and the trained model, so both sides of every comparison
# are sampled with the same random numbers (and a problem both arms proposed gets ONE k from the base, not two). The
# round's episodes and the fresh re-attempt of the round's problems (`gain`) never share one, nor does anything here
# share L0's placing seeds.
SAMPLING_KINDS = ("round", "reach", "rungs", "gain")
SAMPLED_SETS = ("reach_base", "rungs_base", *(f"{name}_{arm}" for name in ("round", "reach", "rungs", "gain_trained", "gain_base") for arm in ARMS))
MAX_LEFTOVER_GB = pipeline.MAX_LEFTOVER_GB
STAND_IN_EMBEDDING = 16


# ------------------------------------------------------------------------------------------------- the run
def training_seed(config: dict) -> int:
    """The round's seed: the task's (`--seeds`, ONE seed per task) or the config's."""
    override = os.environ.get("RLVR_LEAN_TRAINING_SEEDS")
    if not override:
        return config["ladder_loop"]["round"]["seed"]
    seeds = [int(seed) for seed in override.split(",")]
    if len(seeds) != 1:
        raise ValueError(f"a ladder round runs ONE seed per task (its run directory is the seed's); got --seeds {override}")
    return seeds[0]


def round_data_directory() -> Path:
    override = os.environ.get(ROUND_DATA_VARIABLE)
    return Path(override) if override else PACKAGE_ROUND_DATA


def _store(config: dict) -> ArtifactStore:
    pin = lean_pin_from_config(ladder_loop.ladder_config(config))
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    run = os.environ.get(ROUND_RUN_VARIABLE) or f"ladder_l1_seed{training_seed(config)}"
    return ArtifactStore(pipeline.STORE / pin.runs_directory / run, Path(mirror) if mirror else None)


def _stand_in() -> bool:
    return bool(os.environ.get(ladder_loop.STAND_IN_VARIABLE))


def sampling_seed(config: dict, set_name: str) -> int:
    if set_name not in SAMPLED_SETS:
        raise ValueError(f"{set_name!r} is not a set a round samples")
    kind = SAMPLING_KINDS.index(set_name.split("_", 1)[0])
    return config["ladder_loop"]["round"]["sampling_seed"] + 100 * training_seed(config) + kind


def _write_json(store: ArtifactStore, name: str, value: dict) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    store.path(name).write_text(text)
    mirror = os.environ.get("RLVR_LEAN_STEP_DIR")
    if mirror:
        Path(mirror).mkdir(parents=True, exist_ok=True)
        Path(mirror, name).write_text(text)


def _data(config: dict, directory: Path | None = None) -> dict:
    """Everything that travels with the code, checked against its recorded hashes and against the held-out rule.
    `directory`: another round data directory than L1's (L2's holds the whole pool as candidates)."""
    heldout, base_map, l0_summary = ladder_loop.load_problems(ladder_loop.data_directory())
    candidates, base_map_facts, base_results, summary = load_round_data(directory or round_data_directory(), l0_summary)
    counts = check_round_data(heldout, base_map, candidates, base_map_facts, base_results, config["ladder_loop"]["goal"]["base_episodes"])
    return {"heldout": heldout, "base_map": base_map, "candidates": candidates, "base_map_facts": base_map_facts,
            "base_results": base_results, "summary": summary, "counts": counts}


def add_problems(store: ArtifactStore, rows: list[dict]) -> None:
    """Put `rows` into the run's `problems.jsonl` (what the episode step reads), replacing the sets they name."""
    replaced = {row["set"] for row in rows}
    kept = [row for row in store.read_rows("problems.jsonl") if row["set"] not in replaced] if store.path("problems.jsonl").exists() else []
    store.write_rows("problems.jsonl", kept + rows)


def with_negations(config: dict, rows: list[dict]) -> tuple[list[dict], dict]:
    """The problems as the episode step reads them, each with its exact negation where Lean shows the built one
    exact, and its one-side reason otherwise. The rule is L0's (`ladder_loop.ladder_prepare`)."""
    problems, exactness = ladder_loop.build_problems(rows, [])
    raw = check_lean_sources(ladder_loop._lean_settings(config), exactness) if exactness else {}
    without_an_answer = sorted(key for key, answer in raw.items() if answer.get("error"))
    if len(without_an_answer) > ladder_loop.MAXIMUM_EXACTNESS_WITHOUT_AN_ANSWER * max(1, len(exactness)):
        raise RuntimeError(f"{len(without_an_answer)} of {len(exactness)} exactness checks got no answer from the Lean pool "
                           f"(first: {raw[without_an_answer[0]].get('error')}): nothing was recorded, run the step again")
    for problem in problems:
        if problem["negation"] is not None and not pipeline._compiles(raw[problem["problem_id"]]):
            problem["negation"], problem["one_side_reason"] = None, NEGATION_NOT_EXACT
    return problems, {"exactness_checks": len(exactness), "exactness_without_an_answer": len(without_an_answer),
                      "two_sided": sum(problem["negation"] is not None for problem in problems),
                      "known_false_with_one_side": sum(problem["side"] == "false" and problem["negation"] is None for problem in problems)}


def _as_sets(problems: list[dict], set_names: list[str]) -> list[dict]:
    return [{**problem, "set": name} for name in set_names for problem in problems]


# ------------------------------------------------------------------------------------------------- prepare
def ladder_l1_prepare(config: dict) -> dict:
    store = _store(config)
    if store.is_done(PREPARE):
        return store.done_summary(PREPARE)
    data, settings = _data(config), config["ladder_loop"]
    placing = {row["problem_id"]: row for row in data["base_results"] if row["set"] == HELDOUT_SET}
    groups = heldout_groups(data["heldout"], placing, settings["challenger"]["target_rate"], settings["challenger"]["band_reward"])
    goal, on_a_rung = set(group_ids(groups, GOAL)), set(rung_ids(groups))
    measured, checks = with_negations(config, [row for row in data["heldout"] if row["problem_id"] in goal | on_a_rung])
    rows = []
    for who in (BASE, *ARMS):
        rows += _as_sets([problem for problem in measured if problem["problem_id"] in goal], [f"reach_{who}"])
        rows += _as_sets([problem for problem in measured if problem["problem_id"] in on_a_rung], [f"rungs_{who}"])
    add_problems(store, rows)
    store.write_rows("heldout_groups.jsonl", groups)
    sizes = {name: len(group_ids(groups, name)) for name in RUNGS}
    minimum = settings["goal"]["rung_minimum"]
    summary = {"seed": training_seed(config), "fixture": bool(data["summary"].get("fixture")), "lean_pin": settings["lean_pin"],
               "data": data["counts"], "goal_set": len(goal), "rungs": sizes, "rung_minimum": minimum,
               "rungs_below_the_minimum": [name for name in RUNGS if sizes[name] < minimum],
               "heldout_in_neither": sum(row["group"] is None for row in groups), **checks}
    store.mark_done(PREPARE, summary)
    return summary


# --------------------------------------------------------------------------------------------------- embed
def stand_in_embeddings(statements: list[str]) -> np.ndarray:
    """Fixed pseudo-random vectors, one per statement, for a pre-flight without a GPU. NOT the model's."""
    rows = [[byte / 255.0 for byte in hashlib.sha256(statement.encode()).digest()[:STAND_IN_EMBEDDING]] for statement in statements]
    return np.asarray(rows, dtype=np.float32)


def _embed(config: dict, statements: list[str]) -> np.ndarray:
    """The base model's embedding of each statement: the mean of its last layer's states over the prover
    prompt's tokens. The training-stack model (NF4), under the desktop's memory cap; no gradient, no adapter."""
    import torch

    from rlvr_lean.gpu.model_utils import load_nf4_base

    pipeline._cap_torch_memory(config)
    settings = config["ladder_loop"]["challenger"]
    model, tokenizer = load_nf4_base(pipeline.NF4_DIR)
    model.eval()
    core = model.base_model                 # the transformer without its output head: no logits over the vocabulary are needed
    encoded = [tokenizer(build_prover_prompt(statement))["input_ids"][:settings["embedding_max_tokens"]] for statement in statements]
    padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]))       # like lengths together: less padding
    vectors = np.zeros((len(statements), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(order), settings["embedding_batch"]):
            batch = order[start:start + settings["embedding_batch"]]
            width = max(len(encoded[index]) for index in batch)
            ids = torch.full((len(batch), width), padding, dtype=torch.long, device="cuda")
            mask = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
            for row, index in enumerate(batch):
                ids[row, :len(encoded[index])] = torch.tensor(encoded[index], dtype=torch.long)
                mask[row, :len(encoded[index])] = 1
            # Padding sits on the right of a causal model, so no real token's state sees it; the mask keeps it out of the mean.
            states = getattr(core(input_ids=ids, attention_mask=mask), "last_hidden_state", None)
            if states is None:              # a model class whose core is itself: ask the whole model for its states
                states = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            states = states.float()
            weights = mask.unsqueeze(-1).float()
            vectors[batch] = ((states * weights).sum(dim=1) / weights.sum(dim=1)).cpu().numpy()
            del states
    return vectors


def ladder_l1_embed(config: dict) -> dict:
    store = _store(config)
    if store.is_done(EMBED):
        return store.done_summary(EMBED)
    data = _data(config)
    rows = [*data["base_map"], *data["candidates"]]
    statements = [row["statement"] for row in rows]
    started = time.monotonic()
    vectors = stand_in_embeddings(statements) if _stand_in() else _embed(config, statements)
    np.save(store.path("embeddings.npy"), vectors.astype(np.float16))
    store.path("embedding_ids.json").write_text(json.dumps([row["problem_id"] for row in rows]))
    summary = {"statements": len(rows), "base_map": len(data["base_map"]), "candidates": len(data["candidates"]),
               "dimension": int(vectors.shape[1]), "seconds": round(time.monotonic() - started, 1), "stand_in_engine": _stand_in()}
    store.mark_done(EMBED, summary)
    return summary


# ------------------------------------------------------------------------------------------------- propose
def _embeddings(store: ArtifactStore, problem_ids: list[str]) -> np.ndarray:
    order = {problem_id: index for index, problem_id in enumerate(json.loads(store.path("embedding_ids.json").read_text()))}
    vectors = np.load(store.path("embeddings.npy")).astype(np.float64)
    return vectors[[order[problem_id] for problem_id in problem_ids]]


def fit_challenger(config: dict, store: ArtifactStore, data: dict) -> tuple[np.ndarray, dict]:
    """(each candidate's expected reward, what the fit was). The observations are the base map: the base's k of n
    on a random sample of the pool. Calibration is read on a seeded share of it held back from a first fit;
    the model that scores the candidates is then fitted on all of it."""
    settings = config["ladder_loop"]
    challenger, seed = settings["challenger"], training_seed(config)
    results = {row["problem_id"]: row for row in data["base_results"] if row["set"] == BASE_MAP_SET}
    facts = {row["problem_id"]: row for row in data["base_map_facts"]}
    observed = [{**row, **facts[row["problem_id"]]} for row in data["base_map"]]
    observed_ids = [row["problem_id"] for row in observed]
    candidate_ids = [row["problem_id"] for row in data["candidates"]]
    observed_vectors, candidate_vectors = _embeddings(store, observed_ids), _embeddings(store, candidate_ids)
    projection = fit_projection(np.vstack([observed_vectors, candidate_vectors]), challenger["embedding_components"])
    names = feature_names(projection)
    features = features_of(observed, observed_vectors, projection)
    resolved = np.array([results[problem_id]["resolved"] for problem_id in observed_ids])
    episodes = np.array([results[problem_id]["episodes"] for problem_id in observed_ids])
    weights = recency_weights([0] * len(observed), 0, challenger["recency_decay"])      # L1: every observation is the base map's
    arguments = dict(names=names, ridge_grid=challenger["ridge_grid"], folds=challenger["folds"], dispersion_grid=challenger["dispersion_grid"], seed=seed)

    held_back = sorted(range(len(observed)), key=lambda index: rank(seed, "challenger_held_back", observed_ids[index]))
    held_back = set(held_back[:int(len(observed) * challenger["held_back_share"])])
    kept = [index for index in range(len(observed)) if index not in held_back]
    report = {"observations": len(observed), "held_back_for_calibration": len(held_back)}
    target, floor, solvers = challenger["target_rate"], challenger["band_reward"], settings["round"]["solvers"]
    if held_back and len(kept) >= 2:
        first, _ = fit_pass_rate_model([observed_ids[index] for index in kept], features[kept], resolved[kept], episodes[kept], weights=weights[kept], **arguments)
        back = sorted(held_back)
        report["calibration_on_held_back_problems"] = calibration(first.predict(features[back]), resolved[back], episodes[back], target, floor, first.dispersion)
    model, chosen = fit_pass_rate_model(observed_ids, features, resolved, episodes, weights=weights, **arguments)
    report["fit_on_all_observations"] = {key: value for key, value in chosen.items() if key != "out_of_fold_rates"}
    rates = model.predict(features_of(data["candidates"], candidate_vectors, projection))
    scores = expected_rewards(rates, solvers, target, model.dispersion)
    report["candidates"] = {"problems": len(candidate_ids), "mean_predicted_rate": round(float(rates.mean()), 4),
                            "mean_expected_reward": round(float(scores.mean()), 4), "best_expected_reward": round(float(scores.max()), 4)}
    store.write_rows("candidate_scores.jsonl", [{"problem_id": problem_id, "predicted_rate": round(float(rate), 5), "expected_reward": round(float(score), 5)}
                                                for problem_id, rate, score in zip(candidate_ids, rates, scores)])
    return scores, report


def ladder_l1_propose(config: dict) -> dict:
    store = _store(config)
    if store.is_done(PROPOSE):
        return store.done_summary(PROPOSE)
    data, settings, seed = _data(config), config["ladder_loop"], training_seed(config)
    scores, fit = fit_challenger(config, store, data)
    candidate_ids = [row["problem_id"] for row in data["candidates"]]
    wanted = settings["round"]["problems"]
    proposals = {CHALLENGER_ARM: propose(candidate_ids, scores, wanted, settings["challenger"]["random_share"], seed),
                 RANDOM_ARM: random_draw(candidate_ids, wanted, seed, scores)}
    by_id = {row["problem_id"]: row for row in data["candidates"]}
    chosen = sorted({row["problem_id"] for rows in proposals.values() for row in rows})
    problems, checks = with_negations(config, [by_id[problem_id] for problem_id in chosen])
    built = {problem["problem_id"]: problem for problem in problems}
    rows = []
    for arm in ARMS:
        store.write_rows(f"proposals_{arm}.jsonl", proposals[arm])
        # Every proposed problem gets its n episodes: none is dropped or kept by a pass-rate estimate first.
        rows += _as_sets([built[row["problem_id"]] for row in proposals[arm]], [f"round_{arm}", f"gain_trained_{arm}", f"gain_base_{arm}"])
    add_problems(store, rows)
    fit["mean_expected_reward_of_the_proposals"] = {arm: round(float(np.mean([row["score"] for row in proposals[arm]])), 4) for arm in ARMS if proposals[arm]}
    _write_json(store, "challenger_fit.json", fit)
    summary = {"seed": seed, "problems": {arm: len(proposals[arm]) for arm in ARMS}, "wanted": wanted,
               "in_both_arms": len({row["problem_id"] for row in proposals[CHALLENGER_ARM]} & {row["problem_id"] for row in proposals[RANDOM_ARM]}),
               "challenger_fit": fit, **checks}
    store.mark_done(PROPOSE, summary)
    return summary


# ------------------------------------------------------------------------------------------------ sampling
class _AdapterEngine:
    """The sampling engine with ONE adapter applied to everything it generates (the episode step passes none)."""

    def __init__(self, engine, adapter) -> None:
        self.engine, self.adapter = engine, adapter

    def generate(self, prompts, parameters, lora_request=None):
        return self.engine.generate(prompts, parameters, lora_request=self.adapter)


class Engines:
    """One sampling engine per process, built at first use: vLLM is loaded once, however many sets a step samples."""

    def __init__(self, enable_lora: bool) -> None:
        self.enable_lora, self.engine, self.parameters_of = enable_lora, None, None

    def kit(self, adapter=None) -> Callable:
        def build(config: dict):
            if self.engine is None:
                if _stand_in():
                    from rlvr_lean.gpu.stand_in_engine import StandInEngine, stand_in_parameters

                    print("STAND-IN ENGINE: no model is loaded; attempts are drawn from fixed proofs (a pre-flight, not a measurement)", flush=True)
                    self.engine, self.parameters_of = StandInEngine(), stand_in_parameters
                else:
                    self.engine = pipeline._engine(config, enable_lora=self.enable_lora)
                    self.parameters_of = lambda samples, seed, max_tokens: pipeline._sampling_parameters(config, samples, seed=seed, max_tokens=max_tokens)
            return (self.engine if adapter is None else _AdapterEngine(self.engine, adapter)), self.parameters_of
        return build


def _episodes(config: dict, store: ArtifactStore, set_name: str, episodes: int, kit: Callable) -> dict:
    if not any(row["set"] == set_name for row in store.read_rows("problems.jsonl")):
        # A set with no problem (no problem of H on a rung, say) is measured as such, not treated as a failure.
        store.write_rows(f"episodes_{set_name}_problems.jsonl", [])
        return {"set": set_name, "problems": 0, "episodes_each": episodes, "attempts": 0, "statuses": {}}
    with ladder_loop.lean_sessions(config) as sessions:      # ONE client for the set: its requests in flight are `lean_in_flight`
        return ladder_loop.run_episodes(config, set_name, episodes, sampling_seed(config, set_name), store, kit, sessions)


def _results(store: ArtifactStore, set_name: str) -> list[dict]:
    return store.read_rows(f"episodes_{set_name}_problems.jsonl")


def _attempts(store: ArtifactStore, set_name: str) -> list[dict]:
    rows = []
    for path in sorted(store.root.glob(f"episodes_{set_name}_attempts_*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def distinct_attempts(attempts: list[dict]) -> dict:
    """The share of an attempt set's completions that are distinct, per problem and side, averaged."""
    by_prompt: dict[tuple[str, str], list[str]] = {}
    for attempt in attempts:
        by_prompt.setdefault((attempt["problem_id"], attempt["side"]), []).append(attempt["completion"])
    shares = [len(set(completions)) / len(completions) for completions in by_prompt.values()]
    return {"prompts": len(shares), "mean_share_distinct": round(sum(shares) / len(shares), 4) if shares else None}


def ladder_l1_base(config: dict) -> dict:
    """The base's FRESH episodes on G and on the rungs, made once and shared by both arms."""
    store = _store(config)
    if store.is_done(BASE_STEP):
        return store.done_summary(BASE_STEP)
    measure, kit = config["ladder_loop"]["measure"], Engines(enable_lora=False).kit()
    summary = {"reach": _episodes(config, store, f"reach_{BASE}", measure["reach_episodes"], kit),
               "rungs": _episodes(config, store, f"rungs_{BASE}", measure["rung_episodes"], kit)}
    summary["stand_in_engine"] = _stand_in()
    store.mark_done(BASE_STEP, summary)
    return summary


def ladder_l1_round(config: dict, arm: str) -> dict:
    """The round itself: n episodes of the model the round starts from (M(0), the base) on the arm's problems."""
    store, marker = _store(config), f"ladder_l1_round_{arm}"
    if store.is_done(marker):
        return store.done_summary(marker)
    settings = config["ladder_loop"]
    episodes = _episodes(config, store, f"round_{arm}", settings["round"]["solvers"], Engines(enable_lora=False).kit())
    reward = arm_reward(_results(store, f"round_{arm}"), settings["challenger"]["target_rate"], settings["challenger"]["band_reward"])
    summary = {"arm": arm, "episodes": episodes, "reward": reward, "stand_in_engine": _stand_in()}
    store.mark_done(marker, summary)
    return summary


# ------------------------------------------------------------------------------------------------ training
def adapter_directory(store: ArtifactStore, arm: str) -> Path:
    return store.root / "adapters" / arm


def _train(config: dict, pairs: list[tuple[str, str]], seed: int, directory: Path, tensorboard_run: str) -> dict:
    """Train and save one adapter: native format, `round.epochs` passes (one), Phase A's optimizer settings.
    EVERY GPU object is a local of this function, so nothing outlives it (see `pipeline._train_one_adapter`)."""
    import torch

    from rlvr_lean.runner.heartbeat import ScalarEventWriter
    from rlvr_lean.gpu.model_utils import attach_lora, encode_pair, load_nf4_base, mean_proof_token_loss

    training, epochs = config["training"], config["ladder_loop"]["round"]["epochs"]
    model, tokenizer = load_nf4_base(pipeline.NF4_DIR)
    peft_model = attach_lora(model, config["lora"], seed=seed)
    # `from_pretrained` returns the model in evaluation mode, and gradient checkpointing (asked for in
    # `attach_lora`) is applied only in training mode. Without this line every layer's activations are kept
    # for the backward pass, memory grows with the example's length (8.5 GB on the smoke run's short proofs,
    # 12.35 GB on seed 0's longest), and seed 1 ran out of memory under the desktop's reserve
    # (ladder_l1_seed1_r1, 2026-10-04). The mathematics is unchanged: `lora.dropout` is 0 and the model has no
    # other layer that differs between the two modes.
    peft_model.train()
    optimizer = torch.optim.AdamW([parameter for parameter in peft_model.parameters() if parameter.requires_grad],
                                  lr=training["learning_rate"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / training["warmup_steps"]))
    encoded = [encode_pair(tokenizer, prompt, target, training["max_sequence_tokens"], NATIVE) for prompt, target in pairs]
    root = os.environ.get("RLVR_LEAN_TB_DIR")
    tensorboard = ScalarEventWriter(Path(root) / tensorboard_run) if root else None
    order_rng, losses = random.Random(seed), []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for _ in range(epochs):
        order = list(range(len(encoded)))
        order_rng.shuffle(order)
        for batch_start in range(0, len(order), training["effective_batch"]):
            batch = order[batch_start:batch_start + training["effective_batch"]]
            step_loss = 0.0
            for index in batch:
                loss = mean_proof_token_loss(peft_model, *encoded[index]) / len(batch)
                loss.backward()
                step_loss += float(loss.detach())
                del loss
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(round(step_loss, 4))
            if tensorboard is not None:
                tensorboard.scalars(len(losses), {"train/loss": step_loss})
    directory.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(directory))
    return {"examples": len(pairs), "epochs": epochs, "steps": len(losses), "target_format": NATIVE,
            "sequence_start_token_id": int(tokenizer.bos_token_id), "first_loss": losses[0], "last_loss": losses[-1], "losses": losses,
            "seconds": round(time.monotonic() - started, 1), "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "gradient_checkpointing_active": bool(peft_model.training and getattr(peft_model.base_model.model, "is_gradient_checkpointing", False))}


def ladder_l1_train(config: dict, arm: str) -> dict:
    """M(1) for this arm: trained from the base on one verified proof of every problem of the round with k >= 1,
    drawn with the round's seed, on whichever side was proved. The proofs are the solver's own attempts."""
    store, marker = _store(config), f"ladder_l1_train_{arm}"
    if store.is_done(marker):
        return store.done_summary(marker)
    from rlvr_lean.gpu.model_utils import training_text

    seed = training_seed(config)
    problems = [row for row in store.read_rows("problems.jsonl") if row["set"] == f"round_{arm}"]
    examples = training_examples(problems, _attempts(store, f"round_{arm}"), seed)
    if not examples:
        raise RuntimeError(f"the {arm} arm's round resolved no problem: there is nothing to train on")
    store.write_rows(f"training_examples_{arm}.jsonl", examples)
    summary = {"arm": arm, "seed": seed, **training_summary(examples), "stand_in_engine": _stand_in()}
    if _stand_in():
        summary["note"] = "no adapter was trained: a pre-flight without a GPU"
    else:
        import torch

        pipeline._cap_torch_memory(config)
        pairs = [training_text(example["theorem"], example["completion"]) for example in examples]
        summary.update(_train(config, pairs, seed, adapter_directory(store, arm), f"ladder_l1_{arm}_seed{seed}"))
        gc.collect()
        torch.cuda.empty_cache()
        summary["allocated_after_cleanup_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
    store.mark_done(marker, summary)       # the adapter is saved: keep it whatever follows
    if summary.get("allocated_after_cleanup_gb", 0) > MAX_LEFTOVER_GB:
        raise RuntimeError(f"{summary['allocated_after_cleanup_gb']} GB is still allocated on the GPU after the {arm} adapter was trained and released")
    return {key: value for key, value in summary.items() if key != "losses"}


# ------------------------------------------------------------------------------------------------- measure
def ladder_l1_measure(config: dict, arm: str) -> dict:
    """The trained model on G (reach), on the rungs and on the round's own problems, and the base AGAIN on the
    round's problems: the gain by k compares fresh episodes with fresh episodes."""
    store, marker = _store(config), f"ladder_l1_measure_{arm}"
    if store.is_done(marker):
        return store.done_summary(marker)
    settings = config["ladder_loop"]
    measure, solvers = settings["measure"], settings["round"]["solvers"]
    engines = Engines(enable_lora=True)
    if _stand_in():
        adapter = None
    else:
        from vllm.lora.request import LoRARequest

        adapter = LoRARequest(f"ladder_l1_{arm}", 1 + ARMS.index(arm), str(adapter_directory(store, arm)))
    trained, base = engines.kit(adapter), engines.kit()
    summary = {"arm": arm,
               "reach": _episodes(config, store, f"reach_{arm}", measure["reach_episodes"], trained),
               "rungs": _episodes(config, store, f"rungs_{arm}", measure["rung_episodes"], trained),
               "gain_trained": _episodes(config, store, f"gain_trained_{arm}", solvers, trained),
               "gain_base": _episodes(config, store, f"gain_base_{arm}", solvers, base),
               "distinct_attempts": {"the_trained_model_on_the_rounds_problems": distinct_attempts(_attempts(store, f"gain_trained_{arm}")),
                                     "the_base_on_the_rounds_problems": distinct_attempts(_attempts(store, f"gain_base_{arm}")),
                                     "the_trained_model_on_the_rungs": distinct_attempts(_attempts(store, f"rungs_{arm}")),
                                     "the_base_on_the_rungs": distinct_attempts(_attempts(store, f"rungs_{BASE}"))},
               "stand_in_engine": _stand_in()}
    store.mark_done(marker, summary)
    return summary


# -------------------------------------------------------------------------------------------------- report
def ladder_l1_report(config: dict) -> dict:
    from rlvr_lean.reporting.ladder_l1 import build_l1_report

    store = _store(config)
    arms = {}
    for arm in ARMS:
        training = store.done_summary(f"ladder_l1_train_{arm}")
        arms[arm] = {"round": _results(store, f"round_{arm}"), "reach": _results(store, f"reach_{arm}"), "rungs": _results(store, f"rungs_{arm}"),
                     "gain_trained": _results(store, f"gain_trained_{arm}"), "gain_base": _results(store, f"gain_base_{arm}"),
                     "training": {key: value for key, value in training.items() if key != "losses"},
                     "distinct_attempts": store.done_summary(f"ladder_l1_measure_{arm}")["distinct_attempts"],
                     "proposals": store.read_rows(f"proposals_{arm}.jsonl")}
    prepare = store.done_summary(PREPARE)
    context = {"fixture": prepare["fixture"], "stand_in_engine": any(store.done_summary(f"ladder_l1_measure_{arm}")["stand_in_engine"] for arm in ARMS),
               "challenger_fit": store.done_summary(PROPOSE)["challenger_fit"],
               "steps": {PREPARE: prepare, EMBED: store.done_summary(EMBED),
                         **{f"ladder_l1_round_{arm}": store.done_summary(f"ladder_l1_round_{arm}")["episodes"] for arm in ARMS}}}
    report = build_l1_report(store.read_rows("heldout_groups.jsonl"),
                             {"reach": _results(store, f"reach_{BASE}"), "rungs": _results(store, f"rungs_{BASE}")},
                             arms, config["ladder_loop"], config["evaluation"], training_seed(config), context)
    _write_json(store, "report_ladder_l1.json", report)
    # A rerun is another task with an output directory of its own, and a task that failed hands over no step
    # files (seed 1's first task ran out of memory after five steps; its rerun delivered the report and not
    # what those steps had stored). Everything a reader needs beside the report goes out again from here: the
    # per-problem results, the proposals, the training examples, the held-out groups. Not the attempts and not
    # the per-block files, which are large and which the step that wrote them delivered.
    for name in sorted(path.name for path in store.root.glob("*.jsonl")):
        if "_attempts_" not in name and not name.rsplit("_", 1)[-1][:4].isdigit():
            store.mirror(name)
    store.mark_done(REPORT, {"headline": report["headline"], "branch": report["branch"]})
    return report


def _for_arm(step: Callable, arm: str) -> Callable:
    def run(config: dict) -> dict:
        return step(config, arm)
    return run


STEPS = {
    PREPARE: ladder_l1_prepare,
    EMBED: ladder_l1_embed,
    PROPOSE: ladder_l1_propose,
    BASE_STEP: ladder_l1_base,
    **{f"ladder_l1_{name}_{arm}": _for_arm(step, arm) for arm in ARMS
       for name, step in (("round", ladder_l1_round), ("train", ladder_l1_train), ("measure", ladder_l1_measure))},
    REPORT: ladder_l1_report,
}
