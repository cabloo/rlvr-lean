"""L6, the stage end to end on what round 1 (L5's FINISHED run), the runs that task read and L4p's run stored: the refusals
of its first step, search 2's two parts and the partition, whose adapter makes the search, the four sets of 2N rows, each
model trained from `pre_r64` at rank 64 and measured by L4t's own two step bodies and on the held-aside set by L5's own
body, the stored model `second_full` measured there when its adapter is on the box (NOT THERE otherwise), the report; too
few round-2 solves (nothing trained, the search is the finding); the runs it reads left file for file as they were; the
smoke stage and its smoke-only path, which no real run can take; the registration. Spec:
docs/spec/ladder-loop.spec.md, "L6: a second round. Does the cycle compound?". The engine and Lean are scripted;
nothing touches a GPU or the network. The rules and the report on hand-made rows are `test_ladder_l6.py`."""

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l4_rows_stage import _all  # noqa: E402
from test_ladder_l4_stage import FIXTURE, _on_the_gpu  # noqa: E402
from test_ladder_l5_seeds import FURTHER_FILE, NOT_MADE_AGAIN  # noqa: E402 - what a further seed of L5 leaves, frozen there
from test_ladder_l5_stage import OWN_FILES as L5_FILES  # noqa: E402 - what L5 leaves at the search's own seed, frozen there
from test_ladder_l5_stage import REPORT_KEYS as L5_REPORT_KEYS  # noqa: E402
from test_ladder_l5_stage import STEPS as L5_STEPS  # noqa: E402
from test_ladder_l5_stage import (  # noqa: E402, F401
    CONFIG,
    MAP_SET,
    PACKAGE,
    SEEDS,
    _draw_order,
    _file_rows,
    _l4p,
    _placed,
    _stage5,
    _write_the_map,
    arm,
    deep,
    dose,
    l4,
    loop,
    r64,
    rows,
    rows2,
    stage,
)

from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import THE_MINIMUM, THE_RULE  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5 import HELD_ASIDE_LABEL, SEARCH_LABEL  # noqa: E402
from rlvr_lean.domain.ladder_round.l6 import (  # noqa: E402
    COMPOUNDING,
    FOLLOWS,
    LENGTH,
    LONG_ORDINARY,
    MODELS,
    ONE_ROUND,
    ORDINARY,
    PRIMARY,
    PUBLISHED2,
    REACH,
    ROUND_1,
    ROUND_2,
    TWO_ROUNDS,
    bin_of,
)
from rlvr_lean.domain.problem_pool.selection import rank  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_dose, ladder_l4, ladder_l4_dose, ladder_l4_rank, ladder_l4_rows, ladder_l4_start, ladder_l5, ladder_l6, ladder_loop, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import StandInOutput, StandInSample, stand_in_parameters  # noqa: E402
from rlvr_lean.reporting.ladder_ceiling import per_attempt  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import SAY  # noqa: E402

RUN, L4P_RUN, SMOKE = ladder_l6.RUN_VARIABLE, ladder_l6.L4P_RUN_VARIABLE, ladder_l6.SMOKE_VARIABLE
STEPS = ["ladder_l6_search_prepare", "ladder_l6_search", "ladder_l6_prepare",
         *(f"ladder_l6_{step}_{model}" for model in MODELS for step in ("train", "measure", "held_aside")), "ladder_l6_held_aside_second_full", "ladder_l6_report"]
TRAININGS = tuple(f"ladder_l6_train_{model}" for model in MODELS)
ATTEMPTS = CONFIG["ladder_loop"]["l6"]["search"]["attempts"]                # 64: no measurement of this world asks an engine for as many samples a prompt
EDGES = CONFIG["ladder_loop"]["l6"]["long_ordinary"]["line_bins"]
# The hand-made map of `pre_r64` on TWELVE base-map problems (the fixtures' four and eight more): by problem, the map attempts (from 0, of 8) that verified. Seven failed every
# time: the candidates. Five are ordinary solves (one of them solved FOUR times: no ordinary solve of L5's, and one here), their verified map attempts of other lengths
# (an attempt's proof has two lines more than its place): three of 2 or 3 lines, two of 4 to 7 (each with two verified attempts, of which seed 0 draws one and seed 1 the
# other), and none of 8 or more.
CANDIDATES = tuple(f"fixture_p{index}" for index in range(1, 8))
MAP_VERIFIED = {**{key: () for key in CANDIDATES}, "fixture_p8": (3, 5), "fixture_p9": (2, 5), "fixture_p10": (0,), "fixture_p11": (0, 1, 2, 3), "fixture_p12": (1,)}
SOLVED_IN_THE_MAP = tuple(key for key, verified in MAP_VERIFIED.items() if verified)
DRAWN = sorted(CANDIDATES, key=lambda key: rank(0, SEARCH_LABEL, key))[:4]                 # search 1's draw of four of the seven candidates
HELD = min((key for key in CANDIDATES if key not in DRAWN), key=lambda key: rank(0, HELD_ASIDE_LABEL, key))      # the held-aside problem: one of the three it left
NEITHER = [key for key in CANDIDATES if key not in DRAWN and key != HELD]                  # (b): the two candidates neither draw took, in the map's order
SOLVED_1, UNSOLVED_1 = (DRAWN[0], DRAWN[2]), (DRAWN[1], DRAWN[3])                           # search 1 solves two of its four, and leaves two: (a)
SEARCHED_2 = [*UNSOLVED_1, *NEITHER]                                                        # search 2's four problems, (a) then (b)
# What M(1) writes in search 2: by problem, the attempts (from 0, of 64) whose proof closes the goal, each with the lines of that proof. One of (a) and one of (b) are solved.
SCRIPT_2 = {UNSOLVED_1[0]: {20: 9, 40: 9}, NEITHER[1]: {5: 20, 9: 20, 33: 20}}
VERIFIED_1, VERIFIED_2 = (10, 30, 41, 52), sum(map(len, SCRIPT_2.values()))                 # the attempts of search 1 (from 0, of 56) that close a problem it solves; search 2's verified attempts
SOLVED_2 = (UNSOLVED_1[0], NEITHER[1])
ON_THE_SET = dict(zip(("pre_r64", "published", "shallow", "deep", *MODELS, "second_full"), range(2, 11)))      # the held-aside problem's successes of 16: the k-th model measured there closes the goal in its first 1 + k attempts
OWN_FILES = sorted([
    *(f"adapters/{model}/{name}" for model in MODELS for name in ("adapter_config.json", "adapter_model.safetensors")),
    *(f"l6_loss_{model}.json" for model in MODELS), "l6_stored_models.json", *(f"l6_training_{model}.jsonl" for model in MODELS), "l6_deep_solves.jsonl", "l6_ordinary_solves.jsonl",
    "l6_search_problems.jsonl", "l6_held_aside_problems.jsonl", "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl", *(f"l4_stored_pre_r64_{part}.jsonl" for part in ("rungs", "reach", "more")),
    "l6_stored_pre_r64_held_aside.jsonl", *(f"l6_stored_{who}_{part}.jsonl" for who in ("deep", "second_full") for part in ("reach", "more")),
    *(f"ladder_l6_{step}_{model}.done.json" for model in MODELS for step in ("train", "measure", "held_aside")), "ladder_l6_held_aside_second_full.done.json",
    "ladder_l6_search_prepare.done.json", "ladder_l6_search.done.json", "ladder_l6_prepare.done.json", "ladder_l6_report.done.json", "problems.jsonl", "report_ladder_l6.json"])


class CycleSolver:
    """`other` (L5's scripted solver) for every measurement and for search 1; for SEARCH 2 (64 samples a prompt) what
    `script` says: by problem, the attempts whose proof closes the goal, with the lines of that proof. `searched`: every
    prompt of search 2, with what the engine was asked for."""

    def __init__(self, other, script):
        self.other, self.script, self.searched = other, script, []

    @property
    def calls(self):
        return self.other.calls

    @property
    def asked(self):
        return self.other.asked

    def generate(self, prompts, parameters, lora_request=None):
        if parameters.n != ATTEMPTS:
            return self.other.generate(prompts, parameters, lora_request)
        outputs = []
        for prompt in prompts:
            name = re.findall(r"theorem (\S+)", prompt)[-1]
            self.searched.append((name, parameters.n, parameters.seed, parameters.max_tokens))
            self.other.asked.append((name, parameters.n, parameters.seed))
            closes = self.script.get(name.removeprefix("negation_of_"), {})
            outputs.append(StandInOutput([StandInSample(f"  have r{index} : fact_a := by good\n" * (closes[index] - 1) + "  done\n" if index in closes else "  bad_step\n  done\n",
                                                        [0] * 6, "stop") for index in range(parameters.n)]))
        return outputs


def _more_base_map_problems(tmp_path, monkeypatch, more=8):
    """A data directory of the fixtures' held-out set and a base map of the fixtures' four problems and `more` beside
    (known true), in the place of the fixtures': search 2 needs more problems than four."""
    source, target = Path(os.environ[ladder_loop.DATA_VARIABLE]), tmp_path / "ladder_l0_with_more"
    target.mkdir()
    shutil.copy(source / "heldout.jsonl", target / "heldout.jsonl")
    base_map = _file_rows(source / "base_map.jsonl")
    base_map += [{"problem_id": f"fixture_p{index}", "kind": "lean_workbook", "side": "true", "statement": f"theorem fixture_p{index} (a : ℝ) : a ^ 2 + {index} ≥ 0 := by\n", "rewritten": 0}
                 for index in range(len(base_map) + 1, len(base_map) + more + 1)]
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in base_map)
    (target / "base_map.jsonl").write_text(text)
    summary = json.loads((source / "summary.json").read_text())
    summary["files"]["base_map.jsonl"] = {"rows": len(base_map), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    (target / "summary.json").write_text(json.dumps(summary))
    monkeypatch.setenv(ladder_loop.DATA_VARIABLE, str(target))
    return target


def _stage6(config, monkeypatch, until=None, stage_name="ladder_l6_cycle"):
    """The stage's steps in its order, the trainings by their GPU path."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l6.STEPS:
            continue
        run = ladder_l6.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if step in TRAININGS else run(config)
        if step == until:
            break
    return summaries


def _round_one(deep, monkeypatch, tmp_path, verified=MAP_VERIFIED, solved=SOLVED_1, settings=None, l4p=True):  # noqa: F811
    """`deep`'s world with what L6 reads on the box: L4p's run (when `l4p`), a base map of twelve problems with a
    hand-made map of `pre_r64` on it, and ROUND 1: L5's own stage RUN TO ITS END there (its search solves `solved`).
    The stand-in training's losses rise with a row's place, which L5's report reads as INCONCLUSIVE, so for round 1's
    trainings every row's loss is level; this stage's own trainings keep the rising ones. Returns round 1's summaries."""
    config = deep.config
    if l4p:
        with monkeypatch.context() as patched:
            patched.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (deep.dose.solver, stand_in_parameters)))
            _l4p(config, patched)
    _more_base_map_problems(tmp_path, monkeypatch)
    _write_the_map(config, verified)
    config["ladder_loop"]["l5"]["deep"].update(settings or {"problems": 4, "held_aside": 1, "minimum": 1})
    deep.solver.script.clear()
    deep.solver.script.update({key: VERIFIED_1 for key in solved})
    made_up = ladder_dose._stand_in_calls
    with monkeypatch.context() as patched:
        patched.setattr(ladder_dose, "_stand_in_calls", lambda examples, sample, pairs: ((lambda batch: [[2.0, 1.0, 1.0, 1.0, 0.25, 0.125] for _ in batch]), *made_up(examples, sample, pairs)[1:]))
        first = _stage5(config, patched)
    return first


@pytest.fixture
def cycle(deep, monkeypatch, tmp_path):  # noqa: F811
    """Round 1 run to its end (`_round_one`), L4p's run beside it, and this stage's task, which carries NO variable.
    `solver.searched`: every prompt of search 2. One round-2 solve is enough (the setting's, made small)."""
    for name in (RUN, L4P_RUN, SMOKE):
        monkeypatch.delenv(name, raising=False)
    config = deep.config
    first = _round_one(deep, monkeypatch, tmp_path)
    root = deep.store().root
    assert root.name == "ladder_l5_deep_pre_r64_seed0" and first["ladder_l5_report"]["inconclusive"] is False and first["ladder_l5_report"]["ok"] is True
    assert [row["problem_id"] for row in _file_rows(root / "l5_search_problems.jsonl")] == DRAWN and [row["problem_id"] for row in _file_rows(root / "l5_held_aside_problems.jsonl")] == [HELD]
    config["ladder_loop"]["l6"]["search"]["minimum"] = 1
    solver = CycleSolver(deep.solver, {key: dict(value) for key, value in SCRIPT_2.items()})
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        deep.kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"],
                          "block_problems": given["ladder_loop"]["episode"]["block_problems"]}), (solver, stand_in_parameters))[1]))
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l6_cycle"])        # THE STAGE names no variable
    l4p = deep.dose.store().root
    del deep.calls[:], deep.kits[:]
    return SimpleNamespace(config=config, deep=deep, first=first, round_one=root, l4p=l4p, calls=deep.calls, kits=deep.kits, solver=solver, asked=len(deep.solver.asked),
                           runs=ladder_l5._runs(config), store=lambda: ladder_l6._store(config, ladder_l6.the_settings(config)))


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_reads_round_1_searches_again_with_m1_builds_the_four_sets_trains_and_measures_them_and_reports(cycle, monkeypatch):
    config, calls, kits, solver, deep_ = cycle.config, cycle.calls, cycle.kits, cycle.solver, cycle.deep
    read_only = {path: _all(path) for path in (cycle.round_one, cycle.l4p, deep_.map, deep_.l4t2, deep_.rank, deep_.l4t, deep_.arm, deep_.pre)}
    summaries = _stage6(config, monkeypatch)
    store = cycle.store()
    assert store.root.name == "ladder_l6_cycle_pre_r64_seed0" and store.root.parent == cycle.round_one.parent and list(summaries) == STEPS == list(ladder_l6.STEPS)
    assert {path: _all(path) for path in read_only} == read_only               # ROUND 1's RUN, L4p's AND EVERY RUN ROUND 1 READ WERE ONLY READ: file for file
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    assert sorted(path.name for path in cycle.runs.iterdir() if path.name.startswith(("ladder_l5", "ladder_l6"))) == ["ladder_l5_deep_pre_r64_seed0", "ladder_l6_cycle_pre_r64_seed0"]

    # ---- the first step: round 1 as it was read; search 2's two parts; the partition; a sampling seed of its own
    read, first = summaries["ladder_l6_search_prepare"], cycle.first
    m1 = cycle.round_one / "adapters" / "deep"
    assert (read["stage"], read["label"], read["check"], read["seed"], read["start"], read["models"]) == ("l4", "pretrained on published proofs", "L6", 0, "pre_r64", list(MODELS))
    of_round_one, of_the_search, of_the_set = read["round_one"], read["the_search"], read["the_held_aside"]
    assert (of_round_one["run"], of_round_one["seed"], of_round_one["model"], of_round_one["adapter"]) == (str(cycle.round_one), 0, "deep", str(m1))
    assert (of_round_one["attempted"], of_round_one["solved"], of_round_one["inconclusive"], of_round_one["headline"]) == (4, 2, False, first["ladder_l5_report"]["headline"])
    searched = store.read_rows("l6_search_problems.jsonl")
    assert [(row["problem_id"], row["part"]) for row in searched] == [*((key, "a") for key in UNSOLVED_1), *((key, "b") for key in NEITHER)]       # (a) in the draw's order, (b) in the map's
    assert all(row == {"problem_id": row["problem_id"], "part": row["part"], "map_k": 0, "map_n": 8, "taken_by": THE_RULE} for row in searched)
    assert not {HELD, *SOLVED_1} & {row["problem_id"] for row in searched}                                                           # NEITHER a round-1 solve NOR the held-aside problem
    assert of_the_search["partition"] == {"candidates": 7, "attempted_by_search_1": 4, "solved_by_search_1": 2, "a_left_unsolved_by_search_1": 2, "b_in_neither_draw": 2, "held_aside": 1,
                                          "taken_by_the_runs_minimum": 0}
    assert (of_the_search["problems"], of_the_search["by_part"], of_the_search["episodes_each"], of_the_search["sampling_seed"], of_the_search["sampling_place"]) == (4, {"a": 2, "b": 2}, 64, 1052, 52)
    assert (of_the_search["model"], of_the_search["block_problems"], of_the_search["minimum_round_two_solves"], of_the_search["taken_by_the_runs_minimum"]) == ("deep", 1, 1, 0)
    used = of_the_search["seeds_of_the_stored_samplings_it_is_none_of"]
    assert 1052 not in used.values() and (used["L5's deep search"], used["L5's held-aside set"]) == (1050, 1051) and len(used) == 11 and read["smoke"] is None
    by_id = {row["problem_id"]: row for row in _file_rows(deep_.map / "problems.jsonl")}
    of_a_set = lambda name: [row for row in store.read_rows("problems.jsonl") if row["set"] == name]      # noqa: E731
    assert of_a_set("l6_search") == [{**by_id[key], "set": "l6_search"} for key in SEARCHED_2]                                      # each problem as the map sampled it
    # THE HELD-ASIDE SET is round 1's: its problem, its attempts and ITS sampling seed; this run's four models and the stored one are measured there, `pre_r64`'s rows copied.
    assert store.read_rows("l6_held_aside_problems.jsonl") == _file_rows(cycle.round_one / "l5_held_aside_problems.jsonl")
    assert (of_the_set["problems"], of_the_set["episodes_each"], of_the_set["sampling_seed"], of_the_set["block_problems"]) == (1, 16, 1051, 2)
    assert of_the_set["measured"] == [*MODELS, "second_full"] and of_the_set["sets"] == {name: f"l6_held_aside_{name}" for name in (*MODELS, "second_full")}
    assert all(of_a_set(f"l6_held_aside_{name}") == [{**by_id[HELD], "set": f"l6_held_aside_{name}"}] for name in (*MODELS, "second_full"))
    assert store.read_rows("l6_stored_pre_r64_held_aside.jsonl") == _file_rows(cycle.round_one / "episodes_l5_held_aside_pre_r64_problems.jsonl")
    assert of_the_set["of_the_start"] == {"file": "l6_stored_pre_r64_held_aside.jsonl", "set": "l5_held_aside_pre_r64", "run": str(cycle.round_one), "successes": ON_THE_SET["pre_r64"]}
    # What `two_rounds` is read against, beside: M(1)'s stored rows on G (round 1's run) and `second_full`'s (L4p's run), copied as stored.
    for who, source in (("deep", cycle.round_one), ("second_full", cycle.l4p)):
        assert all(store.read_rows(f"l6_stored_{who}_{part}.jsonl") == _file_rows(source / f"episodes_l4_{part}_{who}_problems.jsonl") for part in ("reach", "more"))
    of_the_stored = read["the_stored_model"]
    assert (of_the_stored["model"], of_the_stored["there"], of_the_stored["run"], of_the_stored["adapter"]) == ("second_full", True, str(cycle.l4p), str(cycle.l4p / "adapters" / "second_full"))
    assert read["stored_models"] == {"pre_r64": {"run": str(deep_.rank)}, "deep": {"run": str(cycle.round_one)}, "second_full": {"run": str(cycle.l4p)}}
    theirs = first["ladder_l5_search_prepare"]
    assert all(read[key] == theirs[key] for key in ("the_map", "start_recipe", "start_run_read", "sampling_seeds", "rung_episodes", "goal_samplings", "pretraining_file_sha256", "goal_set_again"))
    assert read["recipe"] == theirs["recipe"] and store.read_rows("l4_goal_set_again.jsonl") == _file_rows(cycle.round_one / "l4_goal_set_again.jsonl")

    # ---- search 2: M(1) served at rank 64, 64 attempts at each of the four problems, its own sampling seed, the round's limits
    search = summaries["ladder_l6_search"]
    side = lambda key: f"negation_of_{key}" if key == "fixture_p3" else key      # noqa: E731
    assert solver.searched == [(side(key), 64, 1052, 1024) for key in SEARCHED_2]
    assert kits[0] == {"temperature": 1.0, "adapter": None, "max_lora_rank": 64, "rank": 64, "block_problems": 1}                   # the first engine of the task: search 2's
    assert (search["model"], search["adapter"], search["set"], search["problems"], search["episodes_each"], search["sampling_seed"]) == ("deep", str(m1), "l6_search", 4, 64, 1052)
    assert (search["check"], search["resolved_at_least_once"], search["attempts"], search["statuses"]) == ("L6", 2, 4 * 64, {"lean_error": 4 * 64 - VERIFIED_2, "verified": VERIFIED_2})
    results = {row["problem_id"]: row["resolved"] for row in store.read_rows("episodes_l6_search_problems.jsonl")}
    assert results == {key: len(SCRIPT_2.get(key, {})) for key in SEARCHED_2} and store.is_done("episodes_l6_search")

    # ---- the four sets: 2N rows each, N every deep solve of both rounds; the published rows the first of the ONE order, nested
    prepared = summaries["ladder_l6_prepare"]
    assert prepared["enough_round_two_solves"] == {"what": prepared["enough_round_two_solves"]["what"], "round_two_solves": 2, "minimum": 1, "passes": True} and prepared["trainings"] == {"run": True}
    assert (prepared["round_one_solves"], prepared["round_two_solves"], prepared["deep_solves"], prepared["rows_a_set"]) == (2, 2, 4, 8) and all(prepared[key] == read[key] for key in ladder_l6.CARRIED)
    sets = {model: store.read_rows(f"l6_training_{model}.jsonl") for model in MODELS}
    assert [len(sets[model]) for model in MODELS] == [8, 8, 8, 8] and all([row["row"] for row in own] == list(range(8)) for own in sets.values())
    assert {model: prepared["training_sets"][model]["rows_by_origin"] for model in MODELS} == {
        PUBLISHED2: {"pretraining": 8}, ONE_ROUND: {"pretraining": 6, "round_1": 2}, TWO_ROUNDS: {"pretraining": 4, "round_1": 2, "round_2": 2}, LONG_ORDINARY: {"ordinary": 4, "pretraining": 4}}
    assert {prepared["training_sets"][model]["optimizer_steps"] for model in MODELS} == {1} and {prepared["training_sets"][model]["rows"] for model in MODELS} == {8}
    own = lambda model, origin: [row for row in sets[model] if row["origin"] == origin]      # noqa: E731
    # Round 1's rows: one verified attempt of each problem search 1 solved, from ROUND 1's stored attempts, drawn by the task's seed; the SAME rows in `one_round` and `two_rounds`.
    attempts_1 = {row["attempt_id"]: row for path in sorted(cycle.round_one.glob("episodes_l5_deep_search_attempts_*.jsonl")) for row in _file_rows(path) if row["status"] == "verified"}
    assert sorted(row["problem_id"] for row in own(TWO_ROUNDS, ROUND_1)) == sorted(SOLVED_1) and sorted(row["id"] for row in own(ONE_ROUND, ROUND_1)) == sorted(row["id"] for row in own(TWO_ROUNDS, ROUND_1))
    for row in own(TWO_ROUNDS, ROUND_1):
        of_the_problem = [key for key, attempt in attempts_1.items() if attempt["problem_id"] == row["problem_id"]]
        assert row["id"] == min(of_the_problem, key=lambda key: rank(0, "training_target", key)) and row["completion"] == attempts_1[row["id"]]["completion"] and len(of_the_problem) == 4
        assert (row["round"], row["first_success"], row["k"], row["n"], row["map_k"], row["taken_by"], row["lines"]) == (1, 19, 4, 56, 0, THE_RULE, 2)
    by_seed_1 = lambda ids: min(ids, key=lambda key: rank(1, "training_target", key))      # noqa: E731 - BY THE TASK'S SEED: another seed would draw another attempt of a problem
    assert {row["id"] for row in own(TWO_ROUNDS, ROUND_1)} != {by_seed_1([key for key, attempt in attempts_1.items() if attempt["problem_id"] == problem]) for problem in SOLVED_1}
    # Round 2's rows: one verified attempt of each problem search 2 solved, from THIS run's search, with its part and the place of its first success among M(1)'s own attempts.
    attempts_2 = {row["attempt_id"]: row for path in sorted(store.root.glob("episodes_l6_search_attempts_*.jsonl")) for row in _file_rows(path) if row["status"] == "verified"}
    assert sorted(row["problem_id"] for row in own(TWO_ROUNDS, ROUND_2)) == sorted(SOLVED_2) and len(attempts_2) == VERIFIED_2 == 5
    by_problem = {row["problem_id"]: row for row in own(TWO_ROUNDS, ROUND_2)}
    for key, part, first_success, lines in ((UNSOLVED_1[0], "a", 21, 9), (NEITHER[1], "b", 6, 20)):
        row = by_problem[key]
        assert (row["round"], row["part"], row["first_success"], row["k"], row["n"], row["lines"], row["taken_by"]) == (2, part, first_success, len(SCRIPT_2[key]), 64, lines, THE_RULE)
        assert row["id"] in attempts_2 and row["completion"] == attempts_2[row["id"]]["completion"]
    for problem in SOLVED_2:
        assert by_problem[problem]["id"] == min((key for key, attempt in attempts_2.items() if attempt["problem_id"] == problem), key=lambda key: rank(0, "training_target", key))
    assert {row["id"] for row in by_problem.values()} != {by_seed_1([key for key, attempt in attempts_2.items() if attempt["problem_id"] == problem]) for problem in SOLVED_2}
    solves = store.read_rows("l6_deep_solves.jsonl")
    assert [(row["round"], row["in_the_sets"]) for row in solves] == [(1, True)] * 2 + [(2, True)] * 2 and {row["id"] for row in solves} == {row["id"] for row in sets[TWO_ROUNDS] if row["origin"] != "pretraining"}
    # The ORDINARY solves: four of the five map problems `pre_r64` solved, one stored verified MAP attempt each, matched to the deep rows' lines bin by bin (2, 2, 9 and 20 lines).
    ordinary = own(LONG_ORDINARY, ORDINARY)
    map_attempts = {row["attempt_id"]: row for row in _file_rows(deep_.map / f"episodes_{MAP_SET}_attempts_0000.jsonl")}
    assert len(ordinary) == 4 and {row["problem_id"] for row in ordinary} <= set(SOLVED_IN_THE_MAP) and len({row["problem_id"] for row in ordinary}) == 4
    for row in ordinary:
        verified = [key for key, attempt in map_attempts.items() if attempt["problem_id"] == row["problem_id"] and attempt["status"] == "verified"]
        assert row["id"] == min(verified, key=lambda key: rank(0, "training_target", key)) and row["completion"] == map_attempts[row["id"]]["completion"]
        assert (row["map_k"], row["taken_by"], row["bin"]) == (len(MAP_VERIFIED[row["problem_id"]]), THE_RULE, bin_of(row["lines"], EDGES))
    of_the_map = lambda problem: [key for key, attempt in map_attempts.items() if attempt["problem_id"] == problem and attempt["status"] == "verified"]      # noqa: E731
    assert {row["id"] for row in ordinary} != {by_seed_1(of_the_map(row["problem_id"])) for row in ordinary}                                # ... and WHICH map attempt, by the task's seed
    matched = prepared["matched_in_length"]
    assert [entry_["lines"] for entry_ in matched["bins"]] == ["1", "2-3", "4-7", "8-15", "16+"] and [entry_["wanted"] for entry_ in matched["bins"]] == [0, 2, 0, 1, 1]
    assert sum(entry_["taken"] for entry_ in matched["bins"]) == 4 and sum(entry_["offered"] for entry_ in matched["bins"]) == 5 == matched["ordinary_solves_the_map_offers"]
    assert matched["bins"][-1] == {"lines": "16+", "wanted": 1, "offered": 0, "asked_by_longer_bins": 0, "taken": 0, "shortfall": 1} and "16+" in matched["bins_short"]      # NO map proof is that long
    assert matched["bins"][-2]["asked_by_longer_bins"] == 1 and matched["rows_short"] == sum(entry_["shortfall"] for entry_ in matched["bins"]) and matched["bins"][0]["shortfall"] == 0
    assert all(entry_["taken"] == min(entry_["offered"], entry_["wanted"] + entry_["asked_by_longer_bins"]) for entry_ in matched["bins"])
    assert [{key: row[key] for key in ("problem_id", "id", "map_k", "lines", "bin", "taken_by")} for row in sorted(ordinary, key=lambda row: row["id"])] == sorted(
        store.read_rows("l6_ordinary_solves.jsonl"), key=lambda row: row["id"])
    # The published rows: the first 8, 6, 4 and 4 of the ONE hash order, by id and place; `two_rounds` and `long_ordinary` hold the SAME four; none with text.
    pretraining = ladder_l4_start.pretrained_on(deep_.rank, prepared["pretraining_file_sha256"], "pre_r64")[0]
    order = _draw_order([{**row, "id": f"{row['problem_id']}#pretraining", "origin": "pretraining", "row_of_the_file": place} for place, row in enumerate(pretraining)])
    for model, count in ((PUBLISHED2, 8), (ONE_ROUND, 6), (TWO_ROUNDS, 4), (LONG_ORDINARY, 4)):
        assert list(map(_placed, _draw_order(sets[model]))) == list(map(_placed, order[:count])) and len(order) == 12
        assert [row["id"] for row in sets[model]] == [row["id"] for row in training_order(sets[model], 0)]
        assert not any(field in row for row in sets[model] if row["origin"] == "pretraining" for field in TEXT_FIELDS)              # no published row with its text
        assert prepared["training_sets"][model]["barred"]["held_out"] == [] and prepared["training_sets"][model]["barred"]["own_rows_not_of_the_base_map"] == []
        assert HELD not in {row["problem_id"] for row in sets[model]}                                                                 # NO SET HOLDS A ROW OF THE HELD-ASIDE PROBLEM
    found = prepared["the_search_found"]
    assert (found["problems"], found["solved"], found["episodes"], found["successes"]) == (4, 2, 4 * 64, VERIFIED_2)
    assert (found["a"]["problems"], found["a"]["solved"], found["a"]["solved_by_first_success"]) == (2, 1, {"1_to_8": 0, "9_to_16": 0, "17_to_32": 1, "33_to_64": 0})
    assert (found["b"]["problems"], found["b"]["solved"], found["b"]["solved_by_first_success"]) == (2, 1, {"1_to_8": 1, "9_to_16": 0, "17_to_32": 0, "33_to_64": 0})
    assert prepared["rows_not_taken_by_the_rule"][TWO_ROUNDS] == prepared["rows_not_taken_by_the_rule"][LONG_ORDINARY] == 0
    assert (prepared["lines_of_the_rows"]["round_1_rows"]["proofs"], prepared["lines_of_the_rows"]["round_2_rows"]["proofs"], prepared["lines_of_the_rows"]["round_2_rows"]["longest"]) == (2, 2, 20)

    # ---- each model, in the setting's order: trained FROM `pre_r64` at rank 64 on its set in its order, measured, and measured on the held-aside set with ROUND 1's seed
    start = deep_.rank / "adapters" / "pre_r64"
    assert [(call["lora"]["rank"], call["lora"]["alpha"], call["more"]) for call in calls] == [(64, 128, {"start": start})] * 4
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l6_cycle_pre_r64_seed0_{model}" for model in MODELS]
    for call, model in zip(calls, MODELS):
        own_examples = {row["id"]: row for row in sets[model] if row["origin"] != "pretraining"}
        assert [example["problem_id"] for example in call["examples"]] == [row["problem_id"] for row in sets[model]] and all(example["theorem"] and example["completion"] for example in call["examples"])
        assert sorted(example["completion"] for example in call["examples"] if example["completion"] in {row["completion"] for row in own_examples.values()}) == sorted(
            row["completion"] for row in own_examples.values())
    # EIGHTEEN ENGINES, each started for rank 64: search 2's; for each model the rungs, G twice and the held-aside set; then the stored model's on the held-aside set.
    assert [kit["max_lora_rank"] for kit in kits] == [64] * 18 and [kit["block_problems"] for kit in kits] == [1, *[5, 5, 5, 2] * 4, 2]
    held_side = side(HELD)
    assert [asked for asked in deep_.solver.asked[cycle.asked:] if asked[2] == 1051] == [(held_side, 16, 1051)] * 5                    # FIVE models there, each with round 1's sampling seed
    on_the_set = {"pre_r64": store.read_rows("l6_stored_pre_r64_held_aside.jsonl")}
    for name in (*MODELS, "second_full"):
        measured = summaries[f"ladder_l6_held_aside_{name}"]
        assert (measured["model"], measured["set"], measured["problems"], measured["episodes_each"], measured["sampling_seed"], measured["check"]) == (name, f"l6_held_aside_{name}", 1, 16, 1051, "L6")
        assert measured["adapter"] == str(cycle.l4p / "adapters" / "second_full" if name == "second_full" else store.root / "adapters" / name) and store.is_done(f"ladder_l6_held_aside_{name}")
        on_the_set[name] = store.read_rows(f"episodes_l6_held_aside_{name}_problems.jsonl")
        assert [(row["problem_id"], row["episodes"], row["resolved"]) for row in on_the_set[name]] == [(HELD, 16, ON_THE_SET[name])] and measured["successes"] == ON_THE_SET[name]
    for model in MODELS:
        train, measured = summaries[f"ladder_l6_train_{model}"], summaries[f"ladder_l6_measure_{model}"]
        assert (train["model"], train["rows"], train["steps"], train["adapter"]) == (model, 8, 1, str(store.root / "adapters" / model)) and train["adapter_saved"]["rank"] == 64
        assert measured["rungs"]["set"] == f"l4_rungs_{model}" and store.is_done(f"ladder_l6_measure_{model}")
        assert list(summaries).index(f"ladder_l6_held_aside_{model}") == list(summaries).index(f"ladder_l6_measure_{model}") + 1

    # ---- the report
    report = summaries["ladder_l6_report"]
    assert json.loads(store.path("report_ladder_l6.json").read_text()) == json.loads(json.dumps(report, default=str)) and report["ok"] is True
    assert (report["label"], report["check"], report["start"], report["rank"], report["measured"], report["deep_solves"], report["rows_a_set"]) == (
        "pretrained on published proofs", "L6", "pre_r64", 64, list(MODELS), 4, 8)
    assert all(line.startswith(f"{SAY}: ") for line in report["lines"]) and list(report["branches"]) == ["the_primary", "compounding", "hard_to_find_or_long", "reach", "what_follows"]
    branches = report["branches"]
    assert branches["the_primary"] in PRIMARY and branches["compounding"] in COMPOUNDING and branches["hard_to_find_or_long"] in LENGTH and branches["reach"] in REACH and branches["what_follows"] in FOLLOWS
    assert list(report["can_this_run_be_read"]) == ["what", "search_2", *MODELS, "the_held_aside_sets"]
    assert list(report["can_this_run_be_read"]["the_held_aside_sets"]["share_without_an_answer"]) == [f"held_aside_{name}" for name in ("pre_r64", *MODELS, "second_full")]
    # THE STAND-IN'S LOSSES RISE with a row's place: every training's rise is PRINTED and decides nothing; the run is read.
    assert report["inconclusive"] is False and report["failed_checks"] == [] and report["the_loss_of_each_training"]["rose_by_more"] == list(MODELS)
    assert any("PRINTED, DECIDING NOTHING" in line and "THE LOSS ROSE BY MORE THAN THAT IN `published2`, `one_round`, `two_rounds`, `long_ordinary`: named, and the run is read all the same" in line
               for line in report["lines"])
    # EACH READ IS ON THE HELD-ASIDE ROWS THE STEPS STORED, `two_rounds` against its own other side.
    resamples = config["evaluation"]["bootstrap_resamples"]
    for key, before in (("primary", PUBLISHED2), ("compounding", ONE_ROUND), ("hard_to_find_or_long", LONG_ORDINARY)):
        assert report[key]["difference"] == per_attempt(on_the_set[TWO_ROUNDS], on_the_set[before], [HELD], resamples, config["evaluation"]["bootstrap_seed"])
        assert (report[key]["difference"]["successes"], report[key]["difference"]["successes_of_the_base"]) == (ON_THE_SET[TWO_ROUNDS], ON_THE_SET[before])
    beside = report["beside"]
    assert {name: own_["held_aside"]["successes"] for name, own_ in beside["each_model"].items()} == {name: ON_THE_SET[name] for name in ("pre_r64", *MODELS)}
    assert list(beside["two_rounds_against_the_stored_models"]) == ["deep", "second_full"]
    of_the_stored = beside["two_rounds_against_the_stored_models"]["second_full"]["on_the_held_aside_set"]["difference"]
    assert (of_the_stored["successes"], of_the_stored["successes_of_the_base"]) == (ON_THE_SET[TWO_ROUNDS], ON_THE_SET["second_full"])
    assert report["adapters"]["there"] == sorted(MODELS) and report["adapters"]["kept"] is True and report["search_2"]["the_finding"] is False
    assert any("SEARCH 2: 4 problems attempted (64 attempts each), 2 solved" in line for line in report["lines"]) and report["the_spread_of_l5"]["given"] is False
    assert store.done_summary("ladder_l6_report") == {"stage": "l4", "label": "pretrained on published proofs", "check": "L6", "headline": report["headline"], "branches": branches,
                                                      "measured": list(MODELS), "ok": True}
    # ---- queued again: every step returns what is stored; nothing is sampled or trained; nothing that was read moved
    trained, asked = len(calls), len(deep_.solver.asked)
    again = _stage6(config, monkeypatch)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in summaries.items() if "report" not in name}
    assert (len(calls), len(deep_.solver.asked)) == (trained, asked) and again["ladder_l6_report"]["branches"] == branches and {path: _all(path) for path in read_only} == read_only


def test_as_on_the_gpu_search_2_serves_m1_from_round_1s_run_and_every_measurement_the_adapter_of_its_own_model(cycle, monkeypatch):
    config, kits = cycle.config, cycle.kits
    # The stand-in engine takes no adapter, so in the scripted world none is asked for. As on the GPU every sampling step asks the model server for ITS model's adapter.
    with monkeypatch.context() as patched:
        for module in (ladder_l5, ladder_ceiling):
            patched.setattr(module, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        summaries = _stage6(config, patched)
        store = cycle.store()
        m1, kept, stored = cycle.round_one / "adapters" / "deep", store.root / "adapters", cycle.l4p / "adapters" / "second_full"
        served = lambda name, directory: (f"ladder_l6_cycle_{name}", 1, str(directory))      # noqa: E731
        # SEARCH 2 IS M(1)'s: round 1's `deep`, in ROUND 1's run (not `pre_r64`'s adapter, and no adapter of this run). Then for each model its rungs, G twice and the
        # held-aside set: the adapter THIS run trained. Then the stored model on the held-aside set: `second_full`, in L4p's run.
        assert [kit["adapter"] for kit in kits] == [served("deep", m1)] + [served(model, kept / model) for model in MODELS for _ in range(4)] + [served("second_full", stored)]
        assert [kit["max_lora_rank"] for kit in kits] == [64] * 18 and summaries["ladder_l6_search"]["stand_in_engine"] is False and len(summaries) == len(STEPS)
        assert m1 != cycle.deep.rank / "adapters" / "pre_r64" and m1.is_dir() and stored.is_dir()
        # An adapter that is no longer there: the measurement is refused, not made from another model's.
        shutil.rmtree(kept / TWO_ROUNDS)
        for step in ("ladder_l6_held_aside_two_rounds", "ladder_l6_measure_two_rounds"):
            store.path(f"{step}.done.json").unlink()
            with pytest.raises(RuntimeError, match="adapters/two_rounds is not there: the adapter `two_rounds` is kept by this stage"):
                ladder_l6.STEPS[step](config)
            assert not store.is_done(step)
        # M(1)'s adapter gone from round 1's run: search 2 is not made from another model's; and the first step refuses it before anything is written.
        store.path("ladder_l6_search.done.json").unlink()
        m1.rename(m1.with_name("away"))
        with pytest.raises(RuntimeError, match="adapters/deep is not there: the adapter `deep` is its own run's, only read here, and the deep search is its. Nothing was sampled"):
            ladder_l6.ladder_l6_search(config)
        for path in list(store.root.iterdir()):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        with pytest.raises(RuntimeError, match="adapters/deep is not there: the adapter `deep` of round 1 is M.1., which makes search 2. Queue the task of stage `ladder_l5_deep` for seed 0"):
            ladder_l6.ladder_l6_search_prepare(config)
        assert not any(store.root.iterdir())


def test_the_first_step_refuses_before_anything_is_written(cycle, monkeypatch):
    config, root = cycle.config, cycle.round_one
    own_root = cycle.runs / "ladder_l6_cycle_pre_r64_seed0"
    task = "the task of stage `ladder_l5_deep` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l5_deep --seeds 0`; for the smoke run, stage `ladder_l5_deep_smoke`."
    why = "L6 makes no first round of its own: round 1 is L5's finished run of seed 0, whose search, held-aside set and `deep` model .M.1., which makes search 2. are read from that run"

    def refused(match, error=RuntimeError):
        with pytest.raises(error, match=match):
            ladder_l6.ladder_l6_search_prepare(config)
        assert not own_root.exists() or not any(own_root.iterdir())             # NOTHING WAS WRITTEN

    def with_changed(path, change, match):
        kept = path.read_text()
        path.write_text(change(kept))
        refused(match)
        path.write_text(kept)

    def without(path, match):
        path.rename(path.with_name(path.name + ".away"))
        refused(match)
        path.with_name(path.name + ".away").rename(path)

    as_json = lambda change: (lambda text: json.dumps(change(json.loads(text))))      # noqa: E731
    as_rows = lambda change: (lambda text: "".join(json.dumps(row) + "\n" for row in change([json.loads(line) for line in text.splitlines()])))      # noqa: E731
    # ---- ROUND 1 MISSING: no run of L5 at all; UNFINISHED: a step's marker, its report, a draw, its problems or a stored sampling is not there (L5's own reading, in L6's words)
    root.rename(root.with_name("elsewhere"))
    refused(f"ladder_l5_deep_pre_r64_seed0 does not hold .*the FINISHED run of the deep search. {why}. Run {task} to its end first; nothing was written")
    root.with_name("elsewhere").rename(root)
    for name in ("ladder_l5_search.done.json", "ladder_l5_prepare.done.json", "ladder_l5_report.done.json", "report_ladder_l5.json", "l5_search_problems.jsonl", "l5_held_aside_problems.jsonl",
                 "episodes_l5_deep_search_problems.jsonl", "episodes_l5_held_aside_pre_r64_problems.jsonl"):
        without(root / name, f"ladder_l5_deep_pre_r64_seed0 does not hold .*{name}.*the FINISHED run of the deep search. L6 makes no first round of its own")
    # ---- A REPORT NOT TO BE READ, or INCONCLUSIVE: no second round stands on such a run (a real run; the smoke stage alone reads an INCONCLUSIVE one)
    report = root / "report_ladder_l5.json"
    with_changed(report, as_json(lambda own: {**own, "ok": False}), "the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is not to be read .Lean gave no verdict on too much of a set.: "
                 "no second round stands on such a run. L6 makes no first round of its own")
    with_changed(report, as_json(lambda own: {**own, "inconclusive": True, "failed_checks": ["`shallow`: its_training_ran"]}),
                 "the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is INCONCLUSIVE .`shallow`: its_training_ran.: no second round stands on such a run")
    # ---- MADE OTHERWISE than this task would make it: another start, rank, search or held-aside set (L5's own holding, in L6's words)
    first = root / "ladder_l5_search_prepare.done.json"
    other = "the deep search's run ladder_l5_deep_pre_r64_seed0 was made with another start model, rank, search or held-aside set than this task would make: "
    inside = lambda key, **changed: as_json(lambda own: {**own, key: {**own[key], **changed}})      # noqa: E731
    with_changed(first, as_json(lambda own: {**own, "start": "pre"}), "the run ladder_l5_deep_pre_r64_seed0 recorded seed 0, the start 'pre'")
    with_changed(first, inside("start_recipe", rank=32), other + "the start adapter's rank and alpha .that run: .32, 128.; this task: .64, 128...*L6's models are read beside that run's, on its search "
                 "and its held-aside set. Nothing was written")
    with_changed(first, inside("the_held_aside", sampling_seed=1151), other + "the held-aside set's attempts a problem and its sampling seed .that run: .16, 1151.")
    with_changed(root / "l5_search_problems.jsonl", as_rows(lambda own: [*own[:-1], {**own[-1], "problem_id": HELD}]), other + "the problems searched")
    with_changed(root / "l5_held_aside_problems.jsonl", as_rows(lambda own: [{**own[0], "problem_id": NEITHER[0]}]), other + "the held-aside problems")
    with_changed(root / "episodes_l5_held_aside_pre_r64.done.json", as_json(lambda own: {**own, "sampling_seed": 7}), "measured `pre_r64` on the held-aside set with sampling seed 7 and 16 episodes")
    # ---- NO M(1): round 1 trained no model; its `deep` model was not trained or measured; its stored rows on G are not there, or would not pair; its search does not agree with itself
    prepared = root / "ladder_l5_prepare.done.json"
    with_changed(prepared, as_json(lambda own: {**own, "trainings": {"run": False, "why": "too few deep solves"}}), "round 1's run ladder_l5_deep_pre_r64_seed0 trained no model .too few deep solves.: there is "
                 "no M.1. to make search 2. Nothing was written")
    for name in ("ladder_l5_train_deep.done.json", "ladder_l5_measure_deep.done.json", "episodes_l4_reach_deep_problems.jsonl", "episodes_l4_more_deep.done.json"):
        without(root / name, f"ladder_l5_deep_pre_r64_seed0 does not hold .*{name}.*what round 1 recorded of `deep`, M.1., or its stored rows on G. Run {task} to its end first; nothing was written")
    with_changed(root / "episodes_l4_more_deep.done.json", as_json(lambda own: {**own, "sampling_seed": 5}), "ladder_l5_deep_pre_r64_seed0 measured `deep` .more. with sampling seed 5")
    with_changed(prepared, as_json(lambda own: {**own, "deep_solves": 3}), "round 1's stored search .ladder_l5_deep_pre_r64_seed0. counts 2 problems as solved, its stored attempts hold a verified one for "
                 "2, and its own step counted 3 deep solves: the run does not agree with itself. Nothing was written")
    # ... or one of its stored attempt files is gone (the first block's: a problem it solved): held HERE, before search 2 is made, and not when the rows are drawn
    without(root / "episodes_l5_deep_search_attempts_0000.jsonl", "counts 2 problems as solved, its stored attempts hold a verified one for 1, and its own step counted 2 deep solves")
    # ---- the partition, held where it is made: a round 1 that solved a problem it did not attempt
    with monkeypatch.context() as patched:
        read_round_one = ladder_l6.read_round_one
        patched.setattr(ladder_l6, "read_round_one", lambda *arguments: {**read_round_one(*arguments), "solved": [*SOLVED_1, HELD]})
        refused("round 1 does not agree with the map: 0 problems a draw took by the rule are no candidates .*1 problems search 1 solved are none it attempted.*Refused .round 1's run "
                "ladder_l5_deep_pre_r64_seed0; the map ladder_l4_map_pre_r64_seed0.. Nothing was written")
    # ---- SEARCH 1 LEFT NOTHING: every problem it attempted solved, and no candidate in neither draw
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l6, "second_search", lambda *arguments: {"problems": [], "partition": {}})
        refused("search 2 has no problem to attempt: search 1 solved every problem it attempted, and its draw and the held-aside set took every candidate")
    # ---- search 2's sampling seed: its own, and no stored sampling's (L5's two among them); the settings
    settings = config["ladder_loop"]["l6"]
    for place, said in ((2, "`pre_r64`'s rungs"), (1, "`pre_r64`'s reach"), (40, "L4t's hot sampling 1")):
        monkeypatch.setitem(settings["search"], "sampling_place", place)
        refused(f"search 2's sampling seed is {1000 + place} .ladder_loop.l6.search.sampling_place {place}. and .*{said}.* already sampled with it: it has a sampling seed of its own")
    for place in (50, 51):
        monkeypatch.setitem(settings["search"], "sampling_place", place)
        refused(f"ladder_loop.l6.search.sampling_place is {place} and L5's two are 50 and 51.*search 2's sampling seed is neither the first search's nor the held-aside set's", ValueError)
    monkeypatch.setitem(settings["search"], "sampling_place", 52)
    with monkeypatch.context() as patched:                  # ... and were the settings' own refusal not there, the stored samplings L5's run recorded are held too
        patched.setattr(ladder_l6, "search_seed", lambda config_, cycle_: 1051)
        refused("search 2's sampling seed is 1051 .*L5's held-aside set.* already sampled with it")
    for key, value, said in (("attempts", 0, "ladder_loop.l6.search.attempts is 0"), ("minimum", True, "ladder_loop.l6.search.minimum is True"), ("attempts", 2.5, "ladder_loop.l6.search.attempts is 2.5")):
        kept = settings["search"][key]
        monkeypatch.setitem(settings["search"], key, value)
        refused(said, ValueError)
        monkeypatch.setitem(settings["search"], key, kept)
    for key, value, said in (("models", ["two_rounds", "one_round", "published2", "long_ordinary"], "ladder_loop.l6.models is .* and the stage trains and measures .'published2', 'one_round', 'two_rounds', "
                              "'long_ordinary'.: change both together"), ("start", "pre", "ladder_loop.l5.start is `pre` and ladder_loop.l4.rows2.start is `pre_r64`"),
                             ("search_seed", -1, "ladder_loop.l6.search_seed is -1: it is the seed of the ONE task that makes search 2")):
        kept = settings[key]
        monkeypatch.setitem(settings, key, value)
        if key == "start":                                  # another start than L5's: L5's own start is what it was, and L6's is held to it
            monkeypatch.setitem(settings, key, kept)
            monkeypatch.setitem(config["ladder_loop"]["l5"], "start", "pre")
            refused(said, ValueError)
            monkeypatch.setitem(config["ladder_loop"]["l5"], "start", "pre_r64")
            monkeypatch.setitem(settings, key, "pre")
            refused("ladder_loop.l6.start is `pre` and ladder_loop.l5.start is `pre_r64`: round 1 is L5's finished run from that start", ValueError)
        else:
            refused(said, ValueError)
        monkeypatch.setitem(settings, key, kept)
    for edges in ([2, 4], [1, 4, 4], [1, 2.5], []):
        monkeypatch.setitem(settings["long_ordinary"], "line_bins", edges)
        refused("the line bins' lower edges are .*: whole numbers that begin at 1 and rise", ValueError)
    monkeypatch.setitem(settings["long_ordinary"], "line_bins", [1, 2, 4, 8, 16])
    # ---- A TASK OF ANOTHER SEED: a further seed of L6 is not built, and nothing of another seed is written or looked for
    monkeypatch.setenv(SEEDS, "1")
    with pytest.raises(RuntimeError, match="this task's seed is 1 and ladder_loop.l6.search_seed is 0: the task of that seed makes search 2, and a further seed of L6 .* is not built. Nothing was written"):
        ladder_l6.ladder_l6_search_prepare(config)
    assert not any((cycle.runs / "ladder_l6_cycle_pre_r64_seed1").iterdir()) and ladder_l6.search_seed(config, ladder_l6.the_settings(config)) == 1052        # the SETTING's seed's, at every seed
    monkeypatch.delenv(SEEDS)
    # ---- and after every refusal the step runs to its end; a later step before its own is refused
    with pytest.raises(RuntimeError, match="search 2 needs the step ladder_l6_search_prepare of this run, which is not done: the stage `ladder_l6_cycle` runs the steps in order"):
        ladder_l6.ladder_l6_search(config)
    assert ladder_l6.ladder_l6_search_prepare(config)["the_search"]["problems"] == 4 and (own_root / "ladder_l6_search_prepare.done.json").exists()
    with pytest.raises(RuntimeError, match="L6's four training sets needs the step ladder_l6_search of this run"):
        ladder_l6.ladder_l6_prepare(config)
    with pytest.raises(RuntimeError, match="the training of `two_rounds` needs the step ladder_l6_prepare of this run"):
        ladder_l6.STEPS["ladder_l6_train_two_rounds"](config)
    with pytest.raises(RuntimeError, match="the measurement of `second_full` on the held-aside set needs the step ladder_l6_prepare of this run"):
        ladder_l6.STEPS["ladder_l6_held_aside_second_full"](config)
    with pytest.raises(RuntimeError, match="L6's report needs the step ladder_l6_prepare of this run"):
        ladder_l6.ladder_l6_report(config)


def test_with_too_few_round_two_solves_nothing_is_trained_and_the_report_says_the_search_is_the_finding(cycle, monkeypatch):
    config, calls, solver = cycle.config, cycle.calls, cycle.solver
    solver.script.clear()                                                       # search 2 solves NO problem, and one round-2 solve is asked
    summaries = _stage6(config, monkeypatch)
    store = cycle.store()
    prepared = summaries["ladder_l6_prepare"]
    assert list(summaries) == STEPS and prepared["enough_round_two_solves"]["passes"] is False and (prepared["round_one_solves"], prepared["round_two_solves"], prepared["deep_solves"], prepared["rows_a_set"]) == (2, 0, 0, 0)
    assert prepared["training_sets"] == {} and prepared["trainings"]["run"] is False and "search 2 found 0 round-2 solves and 1 are asked: nothing is trained" in prepared["trainings"]["why"]
    # NO SET, NO TRAINING, NO MEASUREMENT (the held-aside set's among them, the stored model's too): the thirteen steps say so, mark nothing and sample nothing.
    not_run = {"stage": "l4", "label": "pretrained on published proofs", "check": "L6", "run": False, "why": prepared["trainings"]["why"]}
    for step in STEPS[3:-1]:
        assert summaries[step] == not_run and not store.is_done(step)
    assert len(STEPS[3:-1]) == 13 and not [path.name for path in store.root.glob("l6_training_*")] and not [path.name for path in store.root.glob("episodes_l6_held_aside_*")]
    assert calls == [] and not (store.root / "adapters").exists() and len(cycle.kits) == 1 and not store.path("l6_ordinary_solves.jsonl").exists()
    assert [(row["round"], row["in_the_sets"]) for row in store.read_rows("l6_deep_solves.jsonl")] == [(1, False)] * 2 and store.is_done("ladder_l6_prepare")
    report = summaries["ladder_l6_report"]
    assert report["inconclusive"] is True and report["failed_checks"] == ["search_2_found_enough"] and report["measured"] == [] and report["ok"] is True
    assert report["branches"] == {"the_primary": INCONCLUSIVE, "compounding": INCONCLUSIVE, "hard_to_find_or_long": INCONCLUSIVE, "reach": INCONCLUSIVE, "what_follows": NOT_READ}
    assert report["search_2"]["the_finding"] is True and (report["search_2"]["solved"], report["search_2"]["problems"]) == (0, 4)
    assert any(line.startswith(f"{SAY}: SEARCH 2, WHICH IS THE FINDING: 4 problems attempted") for line in report["lines"]) and report["adapters"]["there"] == []
    assert any(line == f"{SAY}: CHECK, search 2 found at least 1 round-2 solves: 0: FAIL" for line in report["lines"]) and "no model was trained" in report["adapters"]["what"]
    assert any("`second_full` (L4p's whole second pass) on the held-aside set: NOT THERE (nothing was trained, and nothing was measured)" in line for line in report["lines"])
    # THE REAL MINIMUM IS THE SETTING'S: two round-2 solves are fewer than its 150.
    assert CONFIG["ladder_loop"]["l6"]["search"]["minimum"] == 150


def test_the_minimum_of_round_two_solves_is_the_settings_and_a_real_run_refuses_a_shortfall_in_the_shortest_bin(cycle, monkeypatch):
    config = cycle.config
    # THE MINIMUM IS THE SETTING'S in a real run: search 2 solves two problems, and three are asked: nothing is trained.
    config["ladder_loop"]["l6"]["search"]["minimum"] = 3
    summaries = _stage6(config, monkeypatch, until="ladder_l6_prepare")
    store, prepared = cycle.store(), summaries["ladder_l6_prepare"]
    assert summaries["ladder_l6_search_prepare"]["the_search"]["minimum_round_two_solves"] == 3 and summaries["ladder_l6_search_prepare"]["smoke"] is None
    assert (prepared["round_two_solves"], prepared["enough_round_two_solves"]["passes"], prepared["trainings"]["run"]) == (2, False, False)
    assert "search 2 found 2 round-2 solves and 3 are asked" in prepared["trainings"]["why"] and not [path.name for path in store.root.glob("l6_training_*")]
    # A SHORTFALL IN THE SHORTEST BIN IS REFUSED in a real run, and never filled from a longer bin: round 2's proofs have ONE line here, and no proof of the map is that short.
    for path in list(store.root.iterdir()):
        path.unlink()
    config["ladder_loop"]["l6"]["search"]["minimum"] = 1
    cycle.solver.script.update({UNSOLVED_1[0]: {20: 1}, NEITHER[1]: {5: 1}})
    _stage6(config, monkeypatch, until="ladder_l6_search")
    with pytest.raises(RuntimeError, match="the map offers 5 ordinary solves, 0 of them in the shortest bin .`1`., and 2 of the 4 rows asked for are still missing there.*and no bin is shorter. Refused "
                                           ".the map ladder_l4_map_pre_r64_seed0"):
        ladder_l6.ladder_l6_prepare(config)
    assert not store.is_done("ladder_l6_prepare") and not [path.name for path in store.root.glob("l6_training_*")] and not store.path("l6_ordinary_solves.jsonl").exists()
    assert not store.path("l6_deep_solves.jsonl").exists()                      # NOTHING WAS WRITTEN: the file of the deep solves is written with the sets, not before a refusal


def test_the_stored_model_is_not_there_without_l4ps_run_and_nothing_is_refused_for_it(cycle, monkeypatch):
    config = cycle.config
    # L4p's run is NOT ON THE BOX: the first step says so and refuses nothing; no rows of `second_full` are copied; its step measures nothing and marks nothing; the report says NOT THERE.
    cycle.l4p.rename(cycle.l4p.with_name("l4p_elsewhere"))
    summaries = _stage6(config, monkeypatch)
    store = cycle.store()
    of_the_stored = summaries["ladder_l6_search_prepare"]["the_stored_model"]
    assert of_the_stored["there"] is False and "ladder_l4_dose_pre_r64_seed0 does not hold" in of_the_stored["why"] and of_the_stored["sets_on_g"] == []
    assert list(summaries["ladder_l6_search_prepare"]["stored_models"]) == ["pre_r64", "deep"] and not list(store.root.glob("l6_stored_second_full_*"))
    step = summaries["ladder_l6_held_aside_second_full"]
    assert (step["run"], step["not_there"], step["model"]) == (False, True, "second_full") and step["why"] == of_the_stored["why"] and not store.is_done("ladder_l6_held_aside_second_full")
    assert not list(store.root.glob("episodes_l6_held_aside_second_full*")) and len(cycle.kits) == 17                               # no engine was started for it
    report = summaries["ladder_l6_report"]
    assert report["inconclusive"] is False and report["ok"] is True and list(report["beside"]["two_rounds_against_the_stored_models"]) == ["deep"]
    assert list(report["can_this_run_be_read"]["the_held_aside_sets"]["share_without_an_answer"]) == [f"held_aside_{name}" for name in ("pre_r64", *MODELS)]
    lines = "\n".join(report["lines"])
    assert "`second_full` (L4p's whole second pass) on the held-aside set: NOT THERE (" in lines and "nothing is refused for it" in lines
    assert "`two_rounds` against the stored `second_full` (L4p's whole second pass, the best pretraining-only model stored) on G: NOT THERE" in lines
    assert "`two_rounds` against `second_full` ON THE HELD-ASIDE SET: NOT THERE (it was not measured there)" in lines and "`two_rounds` against the stored `deep` (M(1)" in lines
    cycle.l4p.with_name("l4p_elsewhere").rename(cycle.l4p)


def test_the_stored_models_rows_are_read_when_its_adapter_is_gone_and_rows_that_would_not_pair_are_not_there(cycle, monkeypatch):
    config = cycle.config
    # ITS ADAPTER IS GONE, its rows on G are there: they are read beside; on the held-aside set it is NOT THERE, and nothing is refused.
    adapter = cycle.l4p / "adapters" / "second_full"
    adapter.rename(adapter.with_name("away"))
    summaries = _stage6(config, monkeypatch)
    store = cycle.store()
    assert summaries["ladder_l6_search_prepare"]["the_stored_model"]["there"] is True and store.path("l6_stored_second_full_reach.jsonl").exists()
    step = summaries["ladder_l6_held_aside_second_full"]
    assert (step["run"], step["not_there"]) == (False, True) and "adapters/second_full is not there" in step["why"] and not store.is_done("ladder_l6_held_aside_second_full")
    against = summaries["ladder_l6_report"]["beside"]["two_rounds_against_the_stored_models"]
    assert list(against) == ["deep", "second_full"] and "on_g" in against["second_full"] and "on_the_held_aside_set" not in against["second_full"]
    # THE ADAPTER IS BACK and the task is queued again: that step alone measures, and the report reads it.
    adapter.with_name("away").rename(adapter)
    asked = len(cycle.deep.solver.asked)
    again = _stage6(config, monkeypatch)
    assert again["ladder_l6_held_aside_second_full"]["model"] == "second_full" and store.is_done("ladder_l6_held_aside_second_full") and len(cycle.deep.solver.asked) == asked + 1
    assert "on_the_held_aside_set" in again["ladder_l6_report"]["beside"]["two_rounds_against_the_stored_models"]["second_full"]
    # ROWS THAT WOULD NOT PAIR (another sampling seed), a run of another start, a model that was not measured: NOT THERE, each with why, and never a refusal.
    cycle_ = ladder_l6.the_settings(config)
    read = {"plan": {"goal_samplings": summaries["ladder_l6_search_prepare"]["goal_samplings"]}, "goal_problems": [row for row in store.read_rows("problems.jsonl") if row["set"] == "l4_reach_two_rounds"]}
    assert ladder_l6.read_the_stored_model(config, cycle_, read)["there"] is True
    marker = cycle.l4p / "episodes_l4_more_second_full.done.json"
    kept = marker.read_text()
    marker.write_text(json.dumps({**json.loads(kept), "sampling_seed": 5}))
    found = ladder_l6.read_the_stored_model(config, cycle_, read)
    assert found["there"] is False and "measured `second_full` (more) with sampling seed 5" in found["why"] and "on_g" not in found
    marker.write_text(kept)
    prepared = cycle.l4p / "ladder_l4_dose_prepare.done.json"
    kept = prepared.read_text()
    prepared.write_text(json.dumps({**json.loads(kept), "start": "pre"}))
    assert "recorded the start 'pre', and this stage's is `pre_r64`" in ladder_l6.read_the_stored_model(config, cycle_, read)["why"]
    prepared.write_text(kept)
    (cycle.l4p / "ladder_l4_dose_measure_second_full.done.json").unlink()
    assert "does not hold ['ladder_l4_dose_measure_second_full.done.json']" in ladder_l6.read_the_stored_model(config, cycle_, read)["why"]


def test_a_stored_model_whose_rows_would_not_pair_is_not_measured_though_its_adapter_is_there(cycle, monkeypatch):
    config = cycle.config
    # L4p's run is on the box with its adapter, and one of `second_full`'s samplings of G was made with another seed: it would not pair. NOT THERE, by what the FIRST step
    # recorded, on G and on the held-aside set alike; nothing is refused, nothing is measured from the adapter that is there.
    marker = cycle.l4p / "episodes_l4_more_second_full.done.json"
    marker.write_text(json.dumps({**json.loads(marker.read_text()), "sampling_seed": 5}))
    summaries = _stage6(config, monkeypatch)
    store = cycle.store()
    of_the_stored = summaries["ladder_l6_search_prepare"]["the_stored_model"]
    assert of_the_stored["there"] is False and "measured `second_full` (more) with sampling seed 5" in of_the_stored["why"] and (cycle.l4p / "adapters" / "second_full").is_dir()
    step = summaries["ladder_l6_held_aside_second_full"]
    assert (step["run"], step["not_there"], step["why"]) == (False, True, of_the_stored["why"]) and not store.is_done("ladder_l6_held_aside_second_full") and len(cycle.kits) == 17
    assert not list(store.root.glob("episodes_l6_held_aside_second_full*")) and not list(store.root.glob("l6_stored_second_full_*"))
    assert list(summaries["ladder_l6_report"]["beside"]["two_rounds_against_the_stored_models"]) == ["deep"] and summaries["ladder_l6_report"]["inconclusive"] is False


def test_no_set_holds_a_held_aside_problem_and_a_search_lean_did_not_answer_is_not_kept(cycle, monkeypatch):
    config, solver = cycle.config, cycle.solver
    ladder_l6.ladder_l6_search_prepare(config)
    store = cycle.store()
    # ---- Lean gave no verdict on too many attempts of search 2: the step fails, unmarked, and the set is forgotten; the task queued again samples it again
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l5, "lean_did_not_answer", lambda rows_: True)
        with pytest.raises(RuntimeError, match="Lean gave no verdict on too many attempts of l6_search: the search is not kept. Queue the task again: this step samples it again"):
            ladder_l6.ladder_l6_search(config)
    assert not store.is_done("ladder_l6_search") and not store.is_done("episodes_l6_search") and not list(store.root.glob("episodes_l6_search_block_*.done.json"))
    sampled = len(solver.searched)
    assert ladder_l6.ladder_l6_search(config)["resolved_at_least_once"] == 2 and len(solver.searched) == 2 * sampled and store.is_done("ladder_l6_search")
    # ---- a held-aside file that names a problem search 2 attempted (a store gone wrong): the four sets are refused, and nothing of them is written
    kept = store.path("l6_held_aside_problems.jsonl").read_text()
    store.path("l6_held_aside_problems.jsonl").write_text(json.dumps({"problem_id": SOLVED_2[0], "map_k": 0, "map_n": 8, "taken_by": THE_RULE}) + "\n")
    with pytest.raises(RuntimeError, match="1 problems of the held-aside set are problems the deep search attempted.*no model here trains on a held-aside problem, and none of them was searched. Refused"):
        ladder_l6.ladder_l6_prepare(config)
    assert not store.is_done("ladder_l6_prepare") and not [path.name for path in store.root.glob("l6_training_*")]
    # ... one that names an ordinary solve: the ordinary draw passes it over (five are offered no longer, and four are still taken)
    store.path("l6_held_aside_problems.jsonl").write_text(json.dumps({"problem_id": "fixture_p10", "map_k": 1, "map_n": 8, "taken_by": THE_RULE}) + "\n")
    prepared = ladder_l6.ladder_l6_prepare(config)
    assert prepared["matched_in_length"]["ordinary_solves_the_map_offers"] == 4 and "fixture_p10" not in {row["problem_id"] for row in store.read_rows("l6_training_long_ordinary.jsonl")}
    store.path("ladder_l6_prepare.done.json").unlink()
    store.path("l6_held_aside_problems.jsonl").write_text(kept)
    # ---- A HELD-OUT PROBLEM among a set's rows (a store gone wrong: the held-out groups name an ordinary solve the set takes): refused, and nothing of the sets is written
    taken = store.read_rows("l6_ordinary_solves.jsonl")[0]["problem_id"]
    for path in (*store.root.glob("l6_training_*"), store.path("l6_ordinary_solves.jsonl")):
        path.unlink()
    groups = store.path("l4_heldout_groups.jsonl").read_text()
    store.path("l4_heldout_groups.jsonl").write_text(groups + json.dumps({"problem_id": taken, "group": None}) + "\n")
    with pytest.raises(RuntimeError, match=f"1 problems of the training set of `long_ordinary` are held-out problems .first: {taken}.: refused, nothing was written"):
        ladder_l6.ladder_l6_prepare(config)
    assert not store.is_done("ladder_l6_prepare") and not [path.name for path in store.root.glob("l6_training_*")] and not store.path("l6_ordinary_solves.jsonl").exists()
    store.path("l4_heldout_groups.jsonl").write_text(groups)
    # ---- a search prepared with other sizes than the settings now give is not made
    store.path("ladder_l6_search.done.json").unlink()
    monkeypatch.setitem(config["ladder_loop"]["l6"]["search"], "attempts", 40)
    with pytest.raises(RuntimeError, match="this run was prepared for a search of 64 attempts a problem with sampling seed 1052 in blocks of 1 problems, and the config now gives 40"):
        ladder_l6.ladder_l6_search(config)


def test_the_steps_run_through_the_gpu_entry_point(cycle, monkeypatch, tmp_path):
    monkeypatch.setattr(milestone2, "load_config", lambda path: cycle.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert cycle.store().is_done(ladder_l6.REPORT)


# ---------------------------------------------------------------------------------------- the smoke-only path
SMOKE_MAP = {"fixture_p1": tuple(range(8)), "fixture_p2": (0, 1, 2), "fixture_p3": tuple(range(8)), "fixture_p4": tuple(range(8))}      # as the box's smoke map has it: no problem failed every time


def _smoke_world(deep, monkeypatch):  # noqa: F811
    """`deep`'s world as the smoke runs leave the box, on the fixtures' own four base-map problems: a map in which the
    model solved EVERY problem (k = 8, 3, 8, 8), L5's stage run to its end BY ITS OWN SMOKE-ONLY PATH in a run the task
    names (one problem searched and one held aside, each by the run's minimum; its report INCONCLUSIVE: the stand-in's
    losses rise), and L4p's run. Returns round 1's summaries; L5's smoke-only variable is the smoke stage's and is gone
    again, L5's run variable stays (this stage's smoke task names round 1's run by it)."""
    config = deep.config
    with monkeypatch.context() as patched:
        patched.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (deep.dose.solver, stand_in_parameters)))
        _l4p(config, patched)
    _write_the_map(config, SMOKE_MAP)
    config["ladder_loop"]["l5"]["deep"]["minimum"] = CONFIG["ladder_loop"]["l5"]["deep"]["minimum"]      # the REAL minimum, 150: L5's smoke-only path alone makes one deep solve enough
    monkeypatch.setenv(ladder_l5.RUN_VARIABLE, "ladder_l5_deep_smoke")
    deep.solver.script.clear()
    deep.solver.script.update({f"fixture_p{index}": (4,) for index in range(1, 5)})          # in search 1 every problem is solved at its fifth attempt
    with monkeypatch.context() as patched:
        patched.setenv(ladder_l5.SMOKE_VARIABLE, "1")
        first = _stage5(config, patched)
    assert first["ladder_l5_report"]["inconclusive"] is True and first["ladder_l5_report"]["ok"] is True and first["ladder_l5_prepare"]["rows_of_each_kind"] == 1
    return first


def test_the_smoke_stage_goes_on_from_round_1s_inconclusive_smoke_run_takes_its_problems_loosely_and_says_so(deep, monkeypatch):  # noqa: F811
    for name in (RUN, L4P_RUN, SMOKE):
        monkeypatch.delenv(name, raising=False)
    config = deep.config
    first = _smoke_world(deep, monkeypatch)
    runs = ladder_l5._runs(config)
    round_one, l4p = runs / "ladder_l5_deep_smoke", deep.dose.store().root
    ids = [f"fixture_p{index}" for index in range(1, 5)]
    searched_1 = min(ids, key=lambda key: rank(0, SEARCH_LABEL, key))
    aside = min((key for key in ids if key != searched_1), key=lambda key: rank(0, HELD_ASIDE_LABEL, key))
    left = [key for key in ids if key not in (searched_1, aside)]
    assert [row["problem_id"] for row in _file_rows(round_one / "l5_search_problems.jsonl")] == [searched_1] and [row["problem_id"] for row in _file_rows(round_one / "l5_held_aside_problems.jsonl")] == [aside]
    failed = first["ladder_l5_report"]["failed_checks"]
    solver = CycleSolver(deep.solver, {key: {7: 3} for key in ids})                          # in search 2 every problem is solved at its eighth attempt, in three lines
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (solver, stand_in_parameters)))
    # ---- A REAL RUN IS REFUSED FOR SUCH A ROUND 1 (its report is INCONCLUSIVE; its draws are no draws of the rule), in the seed's own directory and in one the task names.
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE .*: no second round stands on such a run"):
        ladder_l6.ladder_l6_search_prepare(config)
    # THE SMOKE-ONLY VARIABLE without a run directory of the task's naming; and with one, on a map that is no fixture's: refused, and nothing is written.
    monkeypatch.setenv(SMOKE, "2")
    said = "RLVR_LEAN_LADDER_L6_SMOKE_PROBLEMS is set .2.: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L6_RUN. and on the fixtures' base map"
    with pytest.raises(RuntimeError, match=said + ".*reads a round 1 whose checks passed and takes its problems by the rule and by nothing else. Nothing was written"):
        ladder_l6.ladder_l6_search_prepare(config)
    assert not (runs / "ladder_l6_cycle_pre_r64_seed0").exists() or not any((runs / "ladder_l6_cycle_pre_r64_seed0").iterdir())
    monkeypatch.setenv(RUN, "ladder_l6_cycle_smoke")
    marker = deep.map / "ladder_l4b_map.done.json"
    recorded = json.loads(marker.read_text())
    marker.write_text(json.dumps({**recorded, "fixture": False}))
    with pytest.raises(RuntimeError, match=said):
        ladder_l6.ladder_l6_search_prepare(config)
    assert not (runs / "ladder_l6_cycle_smoke").exists() or not any((runs / "ladder_l6_cycle_smoke").iterdir())
    marker.write_text(json.dumps(recorded))
    # A report NOT TO BE READ is refused for a smoke run too.
    report_file = round_one / "report_ladder_l5.json"
    kept = report_file.read_text()
    report_file.write_text(json.dumps({**json.loads(kept), "ok": False}))
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is not to be read"):
        ladder_l6.ladder_l6_search_prepare(config)
    report_file.write_text(kept)
    # ---- THE SMOKE RUN: its own directory, the fixtures' map, the variable. Round 1 is read all the same, and that is RECORDED; its looser draws are made again as it recorded them.
    read_only = {path: _all(path) for path in (round_one, l4p, deep.map, deep.l4t2, deep.rank)}
    summaries = _stage6(config, monkeypatch, stage_name="ladder_l6_cycle_smoke")
    store = ladder_l6._store(config, ladder_l6.the_settings(config))
    assert store.root.name == "ladder_l6_cycle_smoke" and list(summaries) == STEPS and {path: _all(path) for path in read_only} == read_only and os.environ.get(ladder_l5.SMOKE_VARIABLE) is None
    read = summaries["ladder_l6_search_prepare"]
    assert read["smoke"]["problems_at_least"] == 2 and read["smoke"]["round_one_read_though_inconclusive"] is True and read["smoke"]["round_ones_failed_checks"] == failed != []
    assert (read["round_one"]["run"], read["round_one"]["inconclusive"], read["round_one"]["failed_checks"], read["round_one"]["solved"]) == (str(round_one), True, failed, 1)
    # The rule gives search 2 NO problem (search 1 solved its one; no candidate): the two map problems no search took and that are not held aside join, each marked.
    of_the_search = read["the_search"]
    assert of_the_search["partition"] == {"candidates": 0, "attempted_by_search_1": 1, "solved_by_search_1": 1, "a_left_unsolved_by_search_1": 0, "b_in_neither_draw": 0, "held_aside": 1,
                                          "taken_by_the_runs_minimum": 2}
    searched = store.read_rows("l6_search_problems.jsonl")
    assert sorted(row["problem_id"] for row in searched) == sorted(left) and [row["problem_id"] for row in searched] == sorted(left, key=lambda key: rank(0, "l6_search", key))
    assert {(row["part"], row["taken_by"]) for row in searched} == {("b", THE_MINIMUM)} and (of_the_search["taken_by_the_rule"], of_the_search["taken_by_the_runs_minimum"], of_the_search["minimum_round_two_solves"]) == (0, 2, 1)
    assert (of_the_search["episodes_each"], of_the_search["sampling_seed"]) == (64, 1052)                                              # the real numbers are untouched
    # N = 3: round 1's one solve and round 2's two; the ordinary solves are taken among the map problems the searches attempted (all were solved in the map), each marked.
    prepared = summaries["ladder_l6_prepare"]
    assert (prepared["round_one_solves"], prepared["round_two_solves"], prepared["deep_solves"], prepared["rows_a_set"]) == (1, 2, 3, 6) and prepared["rows_not_taken_by_the_rule"] == {
        "what": prepared["rows_not_taken_by_the_rule"]["what"], TWO_ROUNDS: 3, LONG_ORDINARY: 3}
    sets = {model: store.read_rows(f"l6_training_{model}.jsonl") for model in MODELS}
    assert [len(sets[model]) for model in MODELS] == [6, 6, 6, 6] and all(aside not in {row["problem_id"] for row in own} for own in sets.values())
    ordinary = [row for row in sets[LONG_ORDINARY] if row["origin"] == ORDINARY]
    assert sorted(row["problem_id"] for row in ordinary) == sorted([searched_1, *left]) and {row["taken_by"] for row in ordinary} == {THE_MINIMUM} and prepared["matched_in_length"]["ordinary_solves_the_map_offers"] == 0
    assert all("#l4_map_pre_r64#" in row["id"] for row in ordinary) and all("#l6_search#" in row["id"] for row in sets[TWO_ROUNDS] if row["origin"] == ROUND_2)      # a MAP attempt's proof, not the search's
    assert {prepared["training_sets"][model]["rows_by_origin"].get("pretraining") for model in MODELS} == {6, 5, 3} and len(_file_rows(FIXTURE)) == 12
    # The four models trained and measured, the stored model measured on the held-aside set, and the report says A SMOKE RUN.
    report = summaries["ladder_l6_report"]
    assert report["measured"] == list(MODELS) and report["ok"] is True and summaries["ladder_l6_held_aside_second_full"]["model"] == "second_full" and report["smoke"] == read["smoke"]
    assert any(f"A SMOKE RUN: round 1's run reads INCONCLUSIVE ({', '.join(failed)}) and was read all the same; 2 of search 2's problems and 3 of `two_rounds`'s own rows, 3 of `long_ordinary`'s own rows "
               "are of problems the rule did not give" in line for line in report["lines"])
    assert sorted(path.name for path in (store.root / "adapters").iterdir()) == sorted(MODELS)
    # WITHOUT THE VARIABLE the minimum is the setting's, and nothing is taken loosely.
    monkeypatch.delenv(SMOKE)
    assert ladder_l6.smoke_problems() == 0 and CONFIG["ladder_loop"]["l6"]["search"]["minimum"] == 150


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_the_smoke_variable_is_the_smoke_stages_alone():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l6_cycle"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l6.STEPS) == STEPS and ladder_l6.ENTRY_STAGE == "ladder_l6_cycle" and len(STEPS) == 17
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l6_cycle_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    # THE SMOKE STAGE's environment: the runs L5's smoke stage names (by L5's own variables, WITHOUT its smoke-only one), L4p's smoke run, a run of its own, and ITS smoke-only variable.
    of_l5 = entry.step_fields(entry.STAGES["ladder_l5_deep_smoke"][2])[2]["environment"]
    assert smoke[0][2]["environment"] == {**{name: value for name, value in of_l5.items() if name != ladder_l5.SMOKE_VARIABLE}, RUN: "ladder_l6_cycle_smoke", L4P_RUN: "ladder_l4_dose_r64_smoke", SMOKE: "2"}
    assert smoke[0][2]["environment"][ladder_l5.RUN_VARIABLE] == "ladder_l5_deep_smoke" == of_l5[ladder_l5.RUN_VARIABLE] and ladder_l5.SMOKE_VARIABLE in of_l5
    of_l4p = entry.step_fields(entry.STAGES["ladder_l4_dose_r64_smoke"][2])[2]["environment"]
    assert of_l4p[ladder_l4_dose.RUN_VARIABLE] == smoke[0][2]["environment"][L4P_RUN] and of_l4p[ladder_l4_rank.RUN_VARIABLE] == smoke[0][2]["environment"][ladder_l4_rank.RUN_VARIABLE]
    # THE ORDER IS THE SETTING'S: search 2 first; then each model trained, measured and measured on the held-aside set before the next; then the stored model there; then the report.
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    settings = whole["ladder_loop"]["l6"]
    assert settings["models"] == list(MODELS) and [step for _, step, _ in steps if step and "_train_" in step] == [f"ladder_l6_train_{model}" for model in settings["models"]] == list(TRAININGS)
    assert [step for _, step, _ in steps if step and "_held_aside_" in step] == [f"ladder_l6_held_aside_{who}" for who in (*settings["models"], "second_full")]
    own = ladder_l6.the_settings(whole)
    assert (own.start, own.models, own.attempts, own.sampling_place, own.minimum, own.least, own.edges, own.seed_of_the_search) == ("pre_r64", MODELS, 64, 52, 150, 1, (1, 2, 4, 8, 16), 0)
    assert own.deep == ladder_l5.the_settings(whole) and ladder_l6.run_name(own, 0) == "ladder_l6_cycle_pre_r64_seed0" and ladder_l6.search_seed(whole, own) == 1052
    assert ladder_l6.round_one_directory(whole, own).name == "ladder_l5_deep_pre_r64_seed0" and ladder_l6.l4p_directory(whole, own).name == "ladder_l4_dose_pre_r64_seed0"
    assert (ladder_l6.M1, ladder_l6.STORED, ladder_l6.SEARCH_SET, ladder_l5.held_aside_set("two_rounds", "l6"), ladder_l5.held_aside_set("pre_r64")) == ("deep", "second_full", "l6_search", "l6_held_aside_two_rounds",
                                                                                                                                                 "l5_held_aside_pre_r64")
    # A BLOCK IS ABOUT AS MANY ATTEMPTS AS A ROUND'S (128 problems x 8): 16 problems of search 2's 64 attempts (55 blocks of its 865 at seed 0).
    assert ladder_l5.block_of(whole, 64) == 16 and -(-865 // 16) == 55
    # ONE reader of its settings; its steps are its own, and only its two stages run them.
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["ladder_loop"]["l6"]' in path.read_text()] == ["gpu/ladder_l6.py"]
    others = set()
    for module in (ladder_round, ladder_loop, ladder_ceiling, ladder_l4, ladder_l4_rank, ladder_l4_rows, ladder_l4_dose, ladder_l5):
        others |= set(module.STEPS)
    assert not set(ladder_l6.STEPS) & others and all(name.startswith("ladder_l6_") for name in ladder_l6.STEPS)
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l6.STEPS for step in entry.STAGES[name])) == ["ladder_l6_cycle", "ladder_l6_cycle_smoke"]
    # ITS VARIABLES are named by its smoke stage alone, and no other module reads them.
    names = lambda variable: sorted(name for name, own_ in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own_))      # noqa: E731
    # (L5f's smoke stage names L4p's run by this stage's variable for it too: its start adapter is that run's `second_full`, found by `l4p_directory`.)
    assert names(RUN) == names(SMOKE) == ["ladder_l6_cycle_smoke"] and names(L4P_RUN) == ["ladder_l5_cont_smoke", "ladder_l5_full_smoke", "ladder_l5_measure_smoke", "ladder_l6_cycle_smoke", "ladder_l6b_smoke"] and SMOKE == "RLVR_LEAN_LADDER_L6_SMOKE_PROBLEMS"
    assert sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if SMOKE in path.read_text()) == ["gpu/ladder_l6.py", "runner/entry.py"]
    # ONE BODY EACH: its train and measure steps are L4t's own two bodies, its search and its held-aside measurements L5's own, its report reads its models by L4t's own reader.
    assert ladder_l6.train_a_model is ladder_l4_rows.train_a_model and ladder_l6.measure_a_model is ladder_l4_rows.measure_a_model and ladder_l6.read_the_trained is ladder_l4_rows.read_the_trained
    assert ladder_l6.THE_SEARCH == ladder_l5.Search(marker="ladder_l6_search", set_name="l6_search", stage=ladder_l6.THE_STAGE, said="second search") and ladder_l5.Search().stage is ladder_l5.THE_STAGE
    own_stage = ladder_l6.THE_STAGE
    assert (own_stage.name, own_stage.steps, own_stage.files, own_stage.reader, own_stage.prepare, own_stage.start_said_by) == ("ladder_l6_cycle", "ladder_l6", "l6", "L6", "ladder_l6_prepare", "ladder_loop.l6.start")
    source = (PACKAGE / "gpu" / "ladder_l6.py").read_text()
    assert "ladder_l5.search_deep(" in source and "ladder_l5.on_the_held_aside_set(" in source and "ladder_l5.read_the_search(" in source and "ladder_l5.read_for_the_search(" in source
    assert "_episodes(" not in source and "LoRARequest" not in source           # it serves no adapter and runs no episode but through those bodies
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l6.STEPS" in main and "ladder_l6," in main


# ------------------------------------------------------------------- the stage that existed before this one
def test_l5_at_the_searchs_own_seed_and_at_a_further_seed_writes_what_it_wrote_with_this_stage_run_beside_it(cycle, monkeypatch):
    config, deep_, first = cycle.config, cycle.deep, cycle.first
    as_it_wrote = _all(cycle.round_one)
    # ROUND 1 IS L5's OWN PATH at the search's seed: its files, steps and report keys are what `test_ladder_l5_stage.py` holds them to; nothing of it names this stage.
    assert sorted(name for name in as_it_wrote if not name.startswith("episodes_")) == L5_FILES and list(first) == L5_STEPS and list(first["ladder_l5_report"]) == L5_REPORT_KEYS
    assert not [name for name in as_it_wrote if "l6_" in name] and "ladder_l6" not in json.dumps(first, default=str) and not [line for line in first["ladder_l5_report"]["lines"] if "L6" in line]
    _stage6(config, monkeypatch)
    root = cycle.store().root
    own = _all(root)
    # L5's own task, queued again AFTER this stage ran in the same store: every step returns what is stored, nothing is trained or sampled, no file of either run moves.
    trained, asked = len(cycle.calls), len(deep_.solver.asked)
    again = _stage5(config, monkeypatch)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in first.items() if "report" not in name}
    assert again["ladder_l5_report"]["lines"] == first["ladder_l5_report"]["lines"] and again["ladder_l5_report"]["branches"] == first["ladder_l5_report"]["branches"]
    assert (len(cycle.calls), len(deep_.solver.asked)) == (trained, asked) and _all(cycle.round_one) == as_it_wrote and _all(root) == own
    # A FURTHER SEED OF L5, run AFTER this stage: it reads round 1's run as it always did and writes its own files, and no other: this stage's run is nothing to it.
    monkeypatch.setenv(SEEDS, "1")
    further = _stage5(config, monkeypatch)
    of_the_seed = ladder_l5._store(config, ladder_l5.the_settings(config)).root
    assert of_the_seed.name == "ladder_l5_deep_pre_r64_seed1" and list(further) == L5_STEPS and all(further[step]["run"] is False for step in NOT_MADE_AGAIN)
    assert sorted(name for name in _all(of_the_seed) if not name.startswith("episodes_")) == sorted({*L5_FILES, FURTHER_FILE} - {f"{step}.done.json" for step in NOT_MADE_AGAIN})
    assert list(further["ladder_l5_report"]) == [*L5_REPORT_KEYS[:-1], "a_further_seed", "timed_out_in_lean", "adapters"] and further["ladder_l5_search_prepare"]["a_further_seed"]["search_run"] == str(cycle.round_one)
    assert _all(cycle.round_one) == as_it_wrote and _all(root) == own and "ladder_l6" not in json.dumps(further, default=str)
    assert sorted(path.name for path in cycle.runs.iterdir() if path.name.startswith(("ladder_l5", "ladder_l6"))) == ["ladder_l5_deep_pre_r64_seed0", "ladder_l5_deep_pre_r64_seed1", "ladder_l6_cycle_pre_r64_seed0"]
    monkeypatch.delenv(SEEDS)
    # L5's names are its own as they were; this stage's are beside them and never the same.
    of_l5 = ladder_l5.THE_STAGE
    assert (of_l5.name, of_l5.steps, of_l5.files, of_l5.reader, of_l5.prepare, of_l5.start_said_by) == ("ladder_l5_deep", "ladder_l5", "l5", "L5", "ladder_l5_prepare", "ladder_loop.l5.start")
    assert not set(ladder_l5.STEPS) & set(ladder_l6.STEPS) and list(ladder_l5.STEPS) == L5_STEPS and ladder_l5.Search() == ladder_l5.Search(marker="ladder_l5_search", set_name="l5_deep_search", stage=of_l5, said="deep search")
    assert ladder_l5.OfTheSearch() == ladder_l5.OfTheSearch(who="a further seed", why=None, stands="no further seed stands beside such a run",
                                                           models="A further seed's models are read beside that run's, on its search and its held-aside set", inconclusive=False)
