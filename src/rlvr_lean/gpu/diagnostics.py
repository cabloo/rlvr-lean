"""Diagnostic steps against a finished run's store. Ledger: the held-out saturation and learning-progress score diagnosis.

Each is one process run by `rlvr_lean.runner.entry --stage diagnose_tokens|diagnose_gradients --profile full`.
They read the store (the base models, the data and every trained adapter are already there), train nothing
that is kept, and write only `diagnose_*.jsonl` and the step's JSON.

  diagnose_tokens     forward only: the loss of every target POSITION under the base model and each trained
                      adapter, what the base would rather write at each position, and how it ends a proof
                      when left to continue on its own.
  diagnose_gradients  the scoring gradients of spec section 5, taken apart by target position.
"""

from __future__ import annotations

import math

from rlvr_lean.domain.diagnosis import PARTS, longest_other_proof, loss_by_part
from rlvr_lean.domain.training.arms import arm_names

DIAGNOSED_ARMS = ("learning_progress_cosine", "random", "difficulty_heuristic", "half_pass_rate")
DIAGNOSED_SEEDS = (0, 1, 2)
ALL_SETS_SEED = 0            # this seed's adapters are read on every target set; the others on the held-out set only
ENDINGS_READ = 24            # held-out pairs whose ending the model is left to write itself
DEVICE = "cuda"              # a preflight on a machine without a GPU sets this to "cpu"


def target_sets(store) -> dict[str, list[dict]]:
    """The (prompt, target) pairs each diagnostic reads, built exactly as the pipeline builds them.

    heldout          reward-VALIDATION statements, canonical (shortest verified) proof: the held-out loss's pairs
    heldout_longest  the same statements, their LONGEST verified proof with a different text (where one exists)
    reward_gradient  reward-GRADIENT statements, canonical proof: the pairs the score's reference is the mean of
    pool             the selection pool, canonical proof: what every arm draws its training examples from
    """
    from rlvr_lean.domain.selection.canonical_proof import choose_canonical_proof
    from rlvr_lean.gpu.model_utils import training_text
    from rlvr_lean.gpu.pipeline import _verified_by_statement

    verified, _ = _verified_by_statement(store)
    completions = {row["attempt_id"]: row["completion"] for row in store.read_rows("proof_attempts.jsonl")}

    def pair(identifier: str, statement: str, attempt_id: str) -> dict:
        prompt, target = training_text(statement, completions[attempt_id])
        return {"id": identifier, "attempt_id": attempt_id, "prompt": prompt, "target": target}

    sets: dict[str, list[dict]] = {"heldout": [], "heldout_longest": [], "reward_gradient": [], "pool": []}
    for row in sorted(store.read_rows("statements_reward.jsonl"), key=lambda r: r["statement_id"]):
        statement_id = row["statement_id"]
        if statement_id not in verified:
            continue
        canonical = choose_canonical_proof(verified[statement_id])
        if row["half"] == "validation":
            sets["heldout"].append(pair(statement_id, row["statement"], canonical))
            other = longest_other_proof(verified[statement_id], completions, canonical)
            if other is not None:
                sets["heldout_longest"].append(pair(statement_id, row["statement"], other))
        elif row["half"] == "gradient":
            sets["reward_gradient"].append(pair(statement_id, row["statement"], canonical))
    for conjecture in sorted(store.read_rows("conjectures.jsonl"), key=lambda c: c["conjecture_id"]):
        conjecture_id = conjecture["conjecture_id"]
        if conjecture["holdout"] or conjecture_id not in verified:
            continue
        sets["pool"].append(pair(conjecture_id, conjecture["statement"], choose_canonical_proof(verified[conjecture_id])))
    return sets


def trained_adapters(config: dict) -> list:
    """ArmNames of every arm and seed this diagnosis reads, in a fixed order."""
    reused = config["selection"]["phase_a_method"]
    return [arm_names(arm, seed, reused) for arm in DIAGNOSED_ARMS for seed in DIAGNOSED_SEEDS]


def position_losses(model, ids: list[int], mask: list[bool]) -> tuple[list[float], list[int], list[float]]:
    """The pipeline's per-position read (`model_utils.position_losses`), on this module's device."""
    from rlvr_lean.gpu import model_utils

    return model_utils.position_losses(model, ids, mask, device=DEVICE)


def greedy_continuation(model, ids: list[int], steps: int) -> list[tuple[int, float]]:
    """The model's own next `steps` tokens after `ids`, each its first choice: (token id, log-probability)."""
    import torch

    current, written = list(ids), []
    for _ in range(steps):
        logits = model(input_ids=torch.tensor([current], device=DEVICE)).logits[0, -1].float()
        value, token = torch.log_softmax(logits, dim=-1).max(dim=0)
        written.append((int(token), float(value)))
        current.append(int(token))
    return written


def adapter_size(peft_model, lora: dict) -> dict:
    """How far a trained adapter moved the weights: norms of B, and of the weight change (alpha/r)·B·A."""
    named = dict(peft_model.named_parameters())
    scaling = lora["alpha"] / lora["rank"]
    b_squared, change_squared, b_absolute, b_entries, modules = 0.0, 0.0, 0.0, 0, 0
    for name, b_weight in named.items():
        if "lora_B" not in name:
            continue
        a_weight = named[name.replace("lora_B", "lora_A")].detach().float()
        b_matrix = b_weight.detach().float()
        b_squared += float((b_matrix ** 2).sum())
        b_absolute += float(b_matrix.abs().sum())
        b_entries += b_matrix.numel()
        change_squared += float(((b_matrix @ a_weight) ** 2).sum()) * scaling ** 2
        modules += 1
    return {"modules": modules, "b_norm": round(math.sqrt(b_squared), 4), "b_mean_absolute": b_absolute / max(1, b_entries),
            "weight_change_norm": round(math.sqrt(change_squared), 4)}


def _rounded(values: list[float]) -> list[float]:
    return [round(float(value), 4) for value in values]


def diagnose_tokens(config: dict) -> dict:
    """Per-position loss of every target under the base and each trained adapter (forward only).

    Sensitivity check, in the result: the held-out mean of these per-position losses must reproduce the
    pipeline's stored held-out NF4 loss for the base and for every adapter."""
    import torch
    from peft import PeftModel

    from rlvr_lean.gpu.model_utils import encode_with_proof_mask, load_nf4_base
    from rlvr_lean.gpu.pipeline import NF4_DIR, _adapter_dir, _cap_torch_memory, _store

    _cap_torch_memory(config)
    store = _store()
    sets = target_sets(store)
    max_tokens = config["training"]["max_sequence_tokens"]
    model, tokenizer = load_nf4_base(NF4_DIR)
    encoded = {name: [encode_with_proof_mask(tokenizer, p["prompt"], p["target"], max_tokens) for p in pairs] for name, pairs in sets.items()}
    target_rows = []
    for name, pairs in sets.items():
        for pair, (ids, mask) in zip(pairs, encoded[name]):
            target_ids = [token for token, keep in zip(ids, mask) if keep]
            target_rows.append({"set": name, "id": pair["id"], "attempt_id": pair["attempt_id"], "prompt_tokens": len(ids) - len(target_ids),
                                "target_ids": target_ids, "target_tokens": [tokenizer.decode([token]) for token in target_ids]})
    store.write_rows("diagnose_targets.jsonl", target_rows)

    rows, ending_rows, summary, stored_agreement, sizes, missing = [], [], {}, {}, {}, []

    def read(variant: str, current_model, set_names: list[str]) -> None:
        summary[variant] = {}
        for set_name in set_names:
            per_pair = []
            for pair, (ids, mask) in zip(sets[set_name], encoded[set_name]):
                losses, top_ids, top_values = position_losses(current_model, ids, mask)
                per_pair.append(losses)
                rows.append({"variant": variant, "set": set_name, "id": pair["id"], "nll": _rounded(losses), "top1": top_ids,
                             "top1_logprob": _rounded(top_values), "top1_tokens": [tokenizer.decode([token]) for token in top_ids]})
            if per_pair:
                summary[variant][set_name] = loss_by_part(per_pair)
        print(f"read {variant}: " + ", ".join(f"{name} {summary[variant][name]['mean_loss']:.4f}" for name in summary[variant]), flush=True)

    def compare_with_stored(variant: str, file_name: str) -> None:
        """This read against the pipeline's own stored held-out loss, pair by pair."""
        if not store.path(file_name).exists():
            return
        stored = {row["statement_id"]: row["loss"] for row in store.read_rows(file_name)}
        mine = {row["id"]: sum(row["nll"]) / len(row["nll"]) for row in rows if row["variant"] == variant and row["set"] == "heldout"}
        shared = sorted(set(stored) & set(mine))
        if shared:
            stored_agreement[variant] = {"pairs": len(shared), "stored_mean": sum(stored[i] for i in shared) / len(shared),
                                         "read_mean": sum(mine[i] for i in shared) / len(shared),
                                         "max_absolute_difference": max(abs(stored[i] - mine[i]) for i in shared)}

    def endings(variant: str, current_model) -> None:
        """Leave the model to finish the proof itself: after the last body token, and after the newline."""
        for pair, (ids, _) in list(zip(sets["heldout"], encoded["heldout"]))[:ENDINGS_READ]:
            after_body = greedy_continuation(current_model, ids[:-2], 4)
            after_newline = greedy_continuation(current_model, ids[:-1], 3)
            ending_rows.append({"variant": variant, "id": pair["id"], "last_body_token": tokenizer.decode([ids[-3]]),
                                "after_body": [[token, tokenizer.decode([token]), round(value, 4)] for token, value in after_body],
                                "after_newline": [[token, tokenizer.decode([token]), round(value, 4)] for token, value in after_newline]})

    all_sets = list(sets)
    with torch.no_grad():
        model.eval()
        read("base", model, all_sets)
        compare_with_stored("base", "heldout_loss_nf4_base.jsonl")
        endings("base", model)
        for names in trained_adapters(config):
            variant, directory = names.tensorboard_run, _adapter_dir(names)
            if not (directory / "adapter_config.json").exists():
                missing.append(variant)
                continue
            peft_model = PeftModel.from_pretrained(model, str(directory), adapter_name=names.key)
            peft_model.eval()
            read(variant, peft_model, all_sets if names.seed == ALL_SETS_SEED else ["heldout"])
            compare_with_stored(variant, names.nf4_loss_file)
            sizes[variant] = adapter_size(peft_model, config["lora"])
            if names.seed == ALL_SETS_SEED:
                endings(variant, peft_model)
            model = peft_model.unload()               # back to the bare base for the next adapter
    store.write_rows("diagnose_tokens.jsonl", rows)
    store.write_rows("diagnose_endings.jsonl", ending_rows)
    return {"sets": {name: len(pairs) for name, pairs in sets.items()}, "parts": list(PARTS), "variants_read": list(summary),
            "adapters_missing": missing, "loss_by_part": summary, "agreement_with_stored_heldout_loss": stored_agreement,
            "adapter_size": sizes, "rows": len(rows)}


# ------------------------------------------------------------------------------------- diagnose_gradients
# Probe 1 found that the base model writes a beginning-of-sequence token between the prompt and the proof
# (its first choice there, p = 0.994), and the pipeline's targets do not hold one. So the scoring gradient is
# taken three ways: of the target's FIRST token alone, of the REST of the target, and of the whole target in
# the model's NATIVE format (that token put back; its own position is not a target).
VIEWS = ("first", "rest", "native")
REFERENCES = ("reward_first", "reward_rest",                      # the score's reference (their sum), by part of the target
              "reward_half_a", "reward_half_b",                   # the score's reference from each half of the reward pairs
              "reward_native_half_a", "reward_native_half_b",     # the native-format reference from each half
              "heldout", "heldout_native")                        # both references from the held-out pairs instead


class PartGradients:
    """The scoring gradient of spec section 5 (B-only, rescaled per module) of one (prompt, target) pair, by
    view. `first` and `rest` are gradients of SUMMED token losses in the pipeline's format, so the pair's
    scoring gradient is (first + rest) / target tokens; `native` is the summed loss of the whole target with
    the beginning-of-sequence token restored before it."""

    def __init__(self, peft_model, bos_token_id: int) -> None:
        self.peft_model = peft_model
        self.bos_token_id = bos_token_id
        named = dict(peft_model.named_parameters())
        self.b_parameters = []
        for name, parameter in named.items():
            if "lora_B" in name:
                rank, d_in = named[name.replace("lora_B", "lora_A")].shape
                self.b_parameters.append((parameter, math.sqrt(3.0 * d_in / rank)))

    def gradients(self, ids: list[int], mask: list[bool]) -> tuple[dict, dict, int]:
        """(view -> flat gradient, loss name -> summed loss, target tokens)."""
        import torch

        from rlvr_lean.domain.diagnosis import native_format

        positions = [position for position in range(1, len(ids)) if mask[position]]
        native_ids, native_mask = native_format(ids, mask, self.bos_token_id)
        native_positions = [position for position in range(1, len(native_ids)) if native_mask[position]]
        views = {"first": (ids, positions[:1]), "rest": (ids, positions[1:]), "native": (native_ids, native_positions)}
        gradients, losses = {}, {}
        for view in VIEWS:
            view_ids, chosen_positions = views[view]
            self.peft_model.zero_grad(set_to_none=True)
            input_ids = torch.tensor([view_ids], device=DEVICE)
            logits = self.peft_model(input_ids=input_ids).logits[0]
            chosen = torch.tensor(chosen_positions, device=logits.device)
            log_probabilities = torch.log_softmax(logits[chosen - 1].float(), dim=-1)
            token_losses = -log_probabilities.gather(1, input_ids[0, chosen].unsqueeze(1)).squeeze(1)
            loss = token_losses.sum()
            loss.backward()
            losses[view] = float(loss.detach())
            if view == "native":
                losses["native_first"] = float(token_losses[0].detach())
            gradients[view] = torch.cat([parameter.grad.detach().float().flatten() * scale for parameter, scale in self.b_parameters])
            del loss, logits, log_probabilities, token_losses
        self.peft_model.zero_grad(set_to_none=True)
        return gradients, losses, len(positions)


def _diagnose_gradients(config: dict, a_seed: int, tag: str) -> dict:
    """Every GPU object is a local of this function (see `_train_one_adapter` on why that matters)."""
    import torch

    from rlvr_lean.gpu.model_utils import attach_lora, encode_with_proof_mask, load_nf4_base
    from rlvr_lean.gpu.pipeline import NF4_DIR, _cap_torch_memory, _store

    _cap_torch_memory(config)
    store = _store()
    sets = target_sets(store)
    max_tokens = config["training"]["max_sequence_tokens"]
    model, tokenizer = load_nf4_base(NF4_DIR)
    scorer = PartGradients(attach_lora(model, config["lora"], seed=a_seed, trainable_a=False), int(tokenizer.bos_token_id))
    encoded = {name: [encode_with_proof_mask(tokenizer, p["prompt"], p["target"], max_tokens) for p in sets[name]]
               for name in ("reward_gradient", "heldout", "pool")}

    sums: dict = {name: None for name in REFERENCES}
    counts = {name: 0 for name in REFERENCES}
    loss_rows = []

    def add(name: str, vector) -> None:
        sums[name] = vector.clone() if sums[name] is None else sums[name].add_(vector)
        counts[name] += 1

    def record(set_name: str, identifier: str, tokens: int, losses: dict) -> None:
        loss_rows.append({"set": set_name, "id": identifier, "target_tokens": tokens, "loss": {name: round(value, 4) for name, value in losses.items()}})

    for index, (pair, (ids, mask)) in enumerate(zip(sets["reward_gradient"], encoded["reward_gradient"])):
        gradients, losses, tokens = scorer.gradients(ids, mask)
        record("reward_gradient", pair["id"], tokens, losses)
        half = "a" if index % 2 == 0 else "b"
        add("reward_first", gradients["first"] / tokens)
        add("reward_rest", gradients["rest"] / tokens)
        add(f"reward_half_{half}", (gradients["first"] + gradients["rest"]) / tokens)
        add(f"reward_native_half_{half}", gradients["native"] / tokens)
    for pair, (ids, mask) in zip(sets["heldout"], encoded["heldout"]):
        gradients, losses, tokens = scorer.gradients(ids, mask)
        record("heldout", pair["id"], tokens, losses)
        add("heldout", (gradients["first"] + gradients["rest"]) / tokens)
        add("heldout_native", gradients["native"] / tokens)
    del gradients
    missing = [name for name in REFERENCES if sums[name] is None]
    if missing:
        raise RuntimeError(f"no gradient was accumulated for {missing}: the reward or held-out pairs are empty")
    references = torch.stack([sums[name] / counts[name] for name in REFERENCES])       # each row is a MEAN gradient
    sums.clear()
    reference_gram = (references @ references.T).tolist()
    print(f"references built from {len(encoded['reward_gradient'])} reward pairs and {len(encoded['heldout'])} held-out pairs", flush=True)

    # The pool: every view's dot with every reference, and the views' own Gram matrix. Also the mean of the
    # pool's NORMALISED gradients per view: N·|mean|² gives the pool's mean pairwise cosine.
    unit_names = ("first", "rest", "whole", "native")
    unit_sums: dict = {name: None for name in unit_names}
    rows = []
    for pair, (ids, mask) in zip(sets["pool"], encoded["pool"]):
        gradients, losses, tokens = scorer.gradients(ids, mask)
        record("pool", pair["id"], tokens, losses)
        stacked = torch.stack([gradients[view] for view in VIEWS])
        rows.append({"id": pair["id"], "target_tokens": tokens, "dots": (stacked @ references.T).tolist(), "gram": (stacked @ stacked.T).tolist()})
        whole = gradients["first"] + gradients["rest"]
        for name, vector in (("first", gradients["first"]), ("rest", gradients["rest"]), ("whole", whole), ("native", gradients["native"])):
            unit = vector / vector.norm().clamp_min(1e-30)
            unit_sums[name] = unit if unit_sums[name] is None else unit_sums[name].add_(unit)
        del gradients, stacked, whole
    store.write_rows(f"diagnose_gradients_{tag}.jsonl", rows)
    store.write_rows(f"diagnose_format_losses_{tag}.jsonl", loss_rows)
    pool_size = len(rows)
    unit_means = {name: vector / pool_size for name, vector in unit_sums.items() if vector is not None}
    return {"a_seed": a_seed, "views": list(VIEWS), "references": list(REFERENCES), "reference_counts": counts,
            "reference_gram": reference_gram, "pool": pool_size, "b_parameters": len(scorer.b_parameters),
            "gradient_dimensions": int(references.shape[1]), "bos_token_id": scorer.bos_token_id,
            "pool_mean_of_unit_gradients": {name: {"squared_norm": float(vector @ vector), "dots_with_references": (references @ vector).tolist()}
                                            for name, vector in unit_means.items()},
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE == "cuda" else None}


def diagnose_gradients(config: dict) -> dict:
    """The scoring gradients by view, with the scoring pass's own random A (`selection.scoring_seed`)."""
    return _diagnose_gradients(config, config["selection"]["scoring_seed"], "scoring_seed")


def diagnose_gradients_second(config: dict) -> dict:
    """The same with an independent random A (`selection.stability_seed`): is the ranking the projection's?"""
    return _diagnose_gradients(config, config["selection"]["stability_seed"], "stability_seed")


# ----------------------------------------------------------------------------------- diagnose_native_round
ROUND_ARM = "random"                 # the selection both rounds train on (the arm with no selection rule to confound)
ROUND_SEED = 0
EVERY_STEP_UNTIL = 12                # held-out read at every optimizer step up to here, then every `heldout_loss_every`
MAX_LEFTOVER_GB = 1.0                # as in `train_adapter`: the card must be this empty before the next model loads


def _one_round(config: dict, examples: list, heldout_pairs: list, native: bool) -> dict:
    """One training round exactly as `pipeline._train_one_adapter` runs it (same optimizer, schedule, batch
    order and seed), with the pairs in the pipeline's format or in the model's native format, and the held-out
    loss read BY PART. Nothing is saved. Every GPU object is a local of this function."""
    import random

    import torch

    from rlvr_lean.domain.diagnosis import native_format
    from rlvr_lean.gpu.model_utils import attach_lora, encode_with_proof_mask, load_nf4_base, mean_proof_token_loss
    from rlvr_lean.gpu.pipeline import NF4_DIR

    training = config["training"]
    model, tokenizer = load_nf4_base(NF4_DIR)
    peft_model = attach_lora(model, config["lora"], seed=ROUND_SEED)
    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=training["learning_rate"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / training["warmup_steps"]))
    bos_token_id = int(tokenizer.bos_token_id)

    def encode(prompt: str, target: str) -> tuple:
        ids, mask = encode_with_proof_mask(tokenizer, prompt, target, training["max_sequence_tokens"])
        return native_format(ids, mask, bos_token_id) if native else (ids, mask)

    encoded = [encode(prompt, target) for prompt, target in examples]
    heldout_encoded = [encode(pair["prompt"], pair["target"]) for pair in heldout_pairs]
    total_steps = training["epochs"] * math.ceil(len(encoded) / training["effective_batch"])
    order_rng, losses, curve = random.Random(ROUND_SEED), [], []

    def read(step: int) -> None:
        was_training = peft_model.training
        peft_model.eval()
        with torch.no_grad():
            per_pair = [position_losses(peft_model, ids, mask)[0] for ids, mask in heldout_encoded]
            size = adapter_size(peft_model, config["lora"])
        peft_model.train(was_training)
        parts = loss_by_part(per_pair)
        curve.append({"step": step, "mean_loss": round(parts["mean_loss"], 4), "body_per_token": round(parts["body_per_token"], 4),
                      "nats_per_proof": {part: round(value, 4) for part, value in parts["nats_per_proof"].items()},
                      "contribution": {part: round(value, 4) for part, value in parts["contribution"].items()},
                      "b_mean_absolute": size["b_mean_absolute"], "weight_change_norm": size["weight_change_norm"]})

    read(0)
    for _ in range(training["epochs"]):
        order = list(range(len(encoded)))
        order_rng.shuffle(order)
        for batch_start in range(0, len(order), training["effective_batch"]):
            batch = order[batch_start:batch_start + training["effective_batch"]]
            step_loss = 0.0
            for index in batch:
                ids, mask = encoded[index]
                loss = mean_proof_token_loss(peft_model, ids, mask) / len(batch)
                loss.backward()
                step_loss += float(loss.detach())
                del loss
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(round(step_loss, 4))
            step = len(losses)
            if step <= EVERY_STEP_UNTIL or step % training["heldout_loss_every"] == 0 or step == total_steps:
                read(step)
    print(f"round ({'native' if native else 'pipeline'} format): held-out {curve[0]['mean_loss']} -> {curve[-1]['mean_loss']}, "
          f"body per token {curve[0]['body_per_token']} -> {curve[-1]['body_per_token']}", flush=True)
    return {"native": native, "examples": len(examples), "steps": len(losses), "train_losses": losses, "heldout_curve": curve,
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


def diagnose_native_round(config: dict) -> dict:
    """The lever, tested: one round on the random selection (seed 0) with every pair in the model's native
    format, then the same round in the pipeline's format as the control. The control must reproduce the
    stored held-out curve of `random_seed0`; the native round shows what the round teaches once the first
    token costs nothing."""
    import gc

    import torch

    from rlvr_lean.gpu.model_utils import training_text
    from rlvr_lean.gpu.pipeline import _cap_torch_memory, _heldout_pairs, _store, _training_examples

    _cap_torch_memory(config)
    store = _store()
    completions = {row["attempt_id"]: row["completion"] for row in store.read_rows("proof_attempts.jsonl")}
    statements = {c["conjecture_id"]: c["statement"] for c in store.read_rows("conjectures.jsonl")}
    examples = [training_text(statements[e["conjecture_id"]], completions[e["proof_attempt_id"]])
                for e in _training_examples(store, config, ROUND_ARM)]
    heldout_pairs = _heldout_pairs(store)
    rounds = {}
    for name, native in (("native", True), ("pipeline", False)):
        rounds[name] = _one_round(config, examples, heldout_pairs, native)
        gc.collect()
        torch.cuda.empty_cache()
        leftover = round(torch.cuda.memory_allocated() / 1e9, 3)
        rounds[name]["allocated_after_cleanup_gb"] = leftover
        if leftover > MAX_LEFTOVER_GB:
            raise RuntimeError(f"{leftover} GB is still allocated after the {name} round; the next model would load on top of it")
    marker = arm_names(ROUND_ARM, ROUND_SEED, config["selection"]["phase_a_method"]).train_marker
    stored = store.done_summary(marker) if store.is_done(marker) else {}
    return {"arm": ROUND_ARM, "seed": ROUND_SEED, "rounds": rounds,
            "stored_pipeline_round": {"train_losses": stored.get("losses"), "heldout_loss_curve": stored.get("heldout_loss_curve")}}


STEPS = {
    "diagnose_tokens": diagnose_tokens,
    "diagnose_gradients": diagnose_gradients,
    "diagnose_gradients_second": diagnose_gradients_second,
    "diagnose_native_round": diagnose_native_round,
}
