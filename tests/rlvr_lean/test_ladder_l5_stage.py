"""L5, the stage end to end on what the start model's own map, L4t2's FINISHED run and the start model's run stored: the
deep search's problems and the held-aside set drawn from the map (the refusals of its prepare step), the search itself as
a round's episodes run, the three sets of 2D rows, each model trained from `pre_r64` at rank 64 and measured by L4t's own
two step bodies, the held-aside set measured by the start model and the three trained ones with one sampling seed, the
report; too few deep solves (nothing trained, the search is the finding); the cap; the runs it reads left file for file as
they were, and L4t (from `pre` and from `pre_r64`), L4t2 and L4p writing what they wrote; the smoke stage and its
smoke-only path, which no real run can take; the registration. Spec: docs/spec/ladder-loop.spec.md,
"L5: do DEEP solves teach what ordinary solves do not? One step-matched check" and its amendment. The engine and Lean are
scripted; nothing touches a GPU or the network. The rules and the report on hand-made rows are `test_ladder_l5.py`."""

import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_assembly import arm  # noqa: E402, F401
from test_ladder_l2_stage import ARM, loop  # noqa: E402, F401
from test_ladder_l4_dose_stage import OWN_FILES as L4P_FILES  # noqa: E402 - what L4p leaves, frozen there
from test_ladder_l4_dose_stage import STEPS as L4P_STEPS  # noqa: E402
from test_ladder_l4_dose_stage import _stage_p as _l4p  # noqa: E402 - L4p's own stage, as its tests run it
from test_ladder_l4_dose_stage import dose  # noqa: E402, F401
from test_ladder_l4_rows2_stage import OWN_FILES as L4T2_FILES  # noqa: E402
from test_ladder_l4_rows2_stage import STEPS as L4T2_STEPS  # noqa: E402
from test_ladder_l4_rows2_stage import _stage2 as _l4t2  # noqa: E402
from test_ladder_l4_rows2_stage import rows2  # noqa: E402, F401
from test_ladder_l4_rows_r64 import OWN_FILES as L4T_FILES_FROM_PRE_R64  # noqa: E402
from test_ladder_l4_rows_r64 import STEPS as L4T_STEPS_FROM_PRE_R64  # noqa: E402
from test_ladder_l4_rows_r64 import _stage as _l4t  # noqa: E402
from test_ladder_l4_rows_r64 import _trainings, r64  # noqa: E402, F401
from test_ladder_l4_rows_stage import OWN_FILES as L4T_FILES_FROM_PRE  # noqa: E402
from test_ladder_l4_rows_stage import STEPS as L4T_STEPS_FROM_PRE  # noqa: E402
from test_ladder_l4_rows_stage import _all, rows  # noqa: E402, F401
from test_ladder_l4_rows_stage import _stage as _l4t_from_pre  # noqa: E402
from test_ladder_l4_stage import FIXTURE, _file_rows, _on_the_gpu, l4  # noqa: E402, F401
from test_ladder_round import _run_stage, stage  # noqa: E402, F401

from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ, map_rows  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import REHEARSAL_LABEL, THE_MINIMUM, THE_RULE  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5 import CAP_LABEL, DEEP, FOLLOWS, HELD_ASIDE_LABEL, MECHANISM, MODELS, PRIMARY, PUBLISHED, REPLICATION, SEARCH_LABEL, SHALLOW, SHALLOW_LABEL  # noqa: E402
from rlvr_lean.domain.problem_pool.episodes import problem_result  # noqa: E402
from rlvr_lean.domain.problem_pool.selection import SoundnessAlarm, rank  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_l2, ladder_l3d2, ladder_l4, ladder_l4_dose, ladder_l4_rank, ladder_l4_rows, ladder_l4_rows2, ladder_l4_start, ladder_l4b, ladder_l5, ladder_loop, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import StandInOutput, StandInSample, stand_in_parameters  # noqa: E402
from rlvr_lean.reporting.ladder_ceiling import per_attempt  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import SAY  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
RUN, L4T2_RUN, SMOKE, MAP_RUN, RANK_RUN = ladder_l5.RUN_VARIABLE, ladder_l5.L4T2_RUN_VARIABLE, ladder_l5.SMOKE_VARIABLE, ladder_l4_start.MAP_RUN_VARIABLE, ladder_l4_rank.RUN_VARIABLE
L4T_START, L4T_MINIMUM = ladder_l4_rows.START_VARIABLE, ladder_l4_rows.MINIMUM_VARIABLE
SEEDS = "RLVR_LEAN_TRAINING_SEEDS"
STEPS = ["ladder_l5_search_prepare", "ladder_l5_search", "ladder_l5_prepare", "ladder_l5_held_aside_start",
         "ladder_l5_train_published", "ladder_l5_measure_published", "ladder_l5_held_aside_published", "ladder_l5_train_shallow", "ladder_l5_measure_shallow", "ladder_l5_held_aside_shallow",
         "ladder_l5_train_deep", "ladder_l5_measure_deep", "ladder_l5_held_aside_deep", "ladder_l5_report"]
TRAININGS = ("ladder_l5_train_published", "ladder_l5_train_shallow", "ladder_l5_train_deep")
FOUR = ("pre_r64", *MODELS)                                                 # the four models that are measured on the held-aside set, in the order they are
RANK_64 = {"rank": 64, "alpha": 128}
ATTEMPTS = CONFIG["ladder_loop"]["l5"]["deep"]["attempts"]                  # 56: no measurement of this world asks an engine for as many samples a prompt
HELD_ATTEMPTS = CONFIG["ladder_loop"]["l5"]["deep"]["held_aside_attempts"]  # 16: nor for as many: the held-aside set's alone
ON_THE_SET = dict(zip(("pre_r64", "published", "shallow", "deep"), (2, 3, 4, 5)))      # the held-aside problem's successes of 16, by model: the k-th to be measured closes the goal in its first 2 + k attempts
MAP_SET = "l4_map_pre_r64"
# The hand-made map of `pre_r64` on the fixtures' four base-map problems: by problem, the map attempts (from 0, of 8) that verified. Two failed every time (one of them known
# false: its side is the negation): one is searched, the other held aside. Two are ordinary solves (twice, once).
MAP_VERIFIED = {"fixture_p1": (), "fixture_p2": (3, 6), "fixture_p3": (), "fixture_p4": (5,)}
CANDIDATES, ORDINARY = ("fixture_p1", "fixture_p3"), ("fixture_p2", "fixture_p4")
SEARCHED = min(CANDIDATES, key=lambda key: rank(0, SEARCH_LABEL, key))      # the candidate the deep search's draw of one takes ...
HELD = next(key for key in CANDIDATES if key != SEARCHED)                   # ... and the one it leaves: the held-aside set
# What the scripted solver writes in the deep search: by problem, the attempts (from 0, of 56) whose proof closes the goal.
DEEP_SCRIPT = {"fixture_p1": (10, 30), "fixture_p2": (20,), "fixture_p3": (47, 50, 52, 55)}
FIRST_SUCCESS = {"fixture_p1": 19, "fixture_p2": 29, "fixture_p3": 56}      # ... so each one's first success, counting the map's 8
OWN_FILES = sorted([
    *(f"adapters/{model}/{name}" for model in MODELS for name in ("adapter_config.json", "adapter_model.safetensors")),
    *(f"l5_loss_{model}.json" for model in MODELS), "l5_stored_models.json", *(f"l5_training_{model}.jsonl" for model in MODELS), "l5_deep_solves.jsonl", "l5_ordinary_solves.jsonl",
    "l5_search_problems.jsonl", "l5_held_aside_problems.jsonl", "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl", *(f"l4_stored_pre_r64_{part}.jsonl" for part in ("rungs", "reach", "more")),
    *(f"ladder_l5_{step}_{model}.done.json" for model in MODELS for step in ("train", "measure", "held_aside")), "ladder_l5_held_aside_start.done.json",
    "ladder_l5_search_prepare.done.json", "ladder_l5_search.done.json", "ladder_l5_prepare.done.json", "ladder_l5_report.done.json", "problems.jsonl", "report_ladder_l5.json"])
REPORT_KEYS = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "start", "rank", "measured", "rows_of_each_kind", "branches", "inconclusive",
               "failed_checks", "can_this_run_be_read", "primary", "the_mechanism", "the_replication", "what_follows", "the_held_aside_set", "pairs", "against_the_start_on_all_of_g",
               "models", "the_deep_search",
               "lines_of_the_rows", "rows_not_taken_by_the_rule", "what_it_cannot_say", "the_stored_model", "training_sets", "trainings", "heldout", "sizes", "start_recipe", "attempts",
               "not_to_be_read", "not_measured", "adapters"]


class DeepSolver:
    """The world's scripted solver for every measurement of the rungs and of G; for the deep search (56 samples a prompt)
    what `script` says: by problem, the attempts whose proof closes the goal; and for the held-aside set (16 samples a
    prompt) a proof that closes the goal in the first 2 + k attempts of the k-th call, so that the four models that are
    measured there differ. `searched`: every prompt of the search, with what the engine was asked for; `asked`: every
    prompt of every call, with its samples and seed."""

    def __init__(self, other, script):
        self.other, self.script, self.searched, self.asked, self.on_the_set = other, script, [], [], 0

    @property
    def calls(self):
        return self.other.calls

    def generate(self, prompts, parameters, lora_request=None):
        self.asked += [(re.findall(r"theorem (\S+)", prompt)[-1], parameters.n, parameters.seed) for prompt in prompts]
        if parameters.n == HELD_ATTEMPTS:
            self.on_the_set += 1
            return [StandInOutput([StandInSample("  done\n" if index <= self.on_the_set else "  bad_step\n  done\n", [0] * 4, "stop") for index in range(parameters.n)]) for _ in prompts]
        if parameters.n != ATTEMPTS:
            return self.other.generate(prompts, parameters, lora_request)
        outputs = []
        for prompt in prompts:
            name = re.findall(r"theorem (\S+)", prompt)[-1]
            self.searched.append((name, parameters.n, parameters.seed, parameters.max_tokens))
            closes = self.script.get(name.removeprefix("negation_of_"), ())
            outputs.append(StandInOutput([StandInSample(f"  have h{index} : fact_a := by good\n  done\n" if index in closes else "  bad_step\n  done\n", [0] * 6, "stop")
                                          for index in range(parameters.n)]))
        return outputs


def _write_the_map(config, verified=MAP_VERIFIED, directory=None, model="pre_r64", episodes=8, sampling_seed=101, fixture=True):
    """A FINISHED map run of `pre_r64`, hand-made (it is only read): its marker, its file, its problems as the map's step
    builds them, its per-problem rows and its stored attempts with their texts. Returns its directory."""
    directory = directory or ladder_l4_start.the_start(config, "pre_r64").map_directory
    directory.mkdir(parents=True, exist_ok=True)
    base_map = ladder_loop.load_problems(ladder_loop.data_directory())[1]
    problems = [{**problem, "set": MAP_SET} for problem in ladder_loop.build_problems(base_map, [])[0]]
    attempts, results = [], []
    for problem in problems:
        key, side = problem["problem_id"], "negation" if problem["side"] == "false" else "statement"
        own = [{"attempt_id": f"{key}#{MAP_SET}#{side}#{episode}", "problem_id": key, "side": side, "episode": episode, "token_count": 9, "finish_reason": "stop",
                "completion": f"  have m{episode} : fact_a := by good\n" * (1 + episode) + "  done\n" if episode in verified[key] else "  bad_step\n  done\n",
                "status": "verified" if episode in verified[key] else "lean_error", "seconds": 0.1, "first_error": ""} for episode in range(episodes)]
        attempts += own
        results.append(problem_result(problem, own, episodes, {side: "checked"}))
    write = lambda name, rows_: (directory / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_))      # noqa: E731
    write("problems.jsonl", problems)
    write(f"episodes_{MAP_SET}_problems.jsonl", results)
    write(f"episodes_{MAP_SET}_attempts_0000.jsonl", attempts)
    write("l4_map_pre_r64.jsonl", map_rows(results, [row["problem_id"] for row in base_map]))
    (directory / f"episodes_{MAP_SET}.done.json").write_text(json.dumps({"set": MAP_SET, "problems": len(results), "episodes_each": episodes, "sampling_seed": sampling_seed}))
    (directory / "ladder_l4b_map.done.json").write_text(json.dumps({"stage": "l4", "model": model, "set": MAP_SET, "problems": len(results), "episodes_each": episodes,
                                                                     "sampling_seed": sampling_seed, "fixture": fixture}))
    return directory


def _stage5(config, monkeypatch, until=None, stage_name="ladder_l5_deep"):
    """The stage's steps in its order, the trainings by their GPU path."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l5.STEPS:
            continue
        run = ladder_l5.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if step in TRAININGS else run(config)
        if step == until:
            break
    return summaries


def _draw_order(rows_, seed=0):
    """Published rows in the order L4t's rehearsal draw takes them: the smallest content hash of the seed and the row's id first."""
    return sorted((row for row in rows_ if row["origin"] == "pretraining"), key=lambda row: rank(seed, REHEARSAL_LABEL, row["id"]))


def _placed(row):
    return row["id"], row["problem_id"], row["row_of_the_file"]


@pytest.fixture
def deep(dose, monkeypatch):  # noqa: F811
    """`dose`'s world (`pre`'s run, the arm's, the run of the check of the adapter's rank, L4t's run and L4t2's FINISHED
    run from `pre_r64`) with a hand-made map of `pre_r64` in its own run directory: what this stage reads on the box.
    The fixtures' base map holds four problems, so the search draws 1 of the two candidates, the other is held aside,
    and one deep solve is enough (the settings', made small). The task carries NO variable. `solver.searched`: every
    prompt of the deep search."""
    for name in (RUN, L4T2_RUN, SMOKE, MAP_RUN):
        monkeypatch.delenv(name, raising=False)
    config = dose.config
    config["ladder_loop"]["l5"]["deep"].update({"problems": 1, "held_aside": 1, "minimum": 1})
    the_map = _write_the_map(config)
    solver = DeepSolver(dose.solver, dict(DEEP_SCRIPT))
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        dose.kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"],
                          "block_problems": given["ladder_loop"]["episode"]["block_problems"]}), (solver, stand_in_parameters))[1]))
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l5_deep"])        # THE STAGE names no variable
    del dose.calls[:], dose.kits[:]
    return SimpleNamespace(config=config, dose=dose, calls=dose.calls, kits=dose.kits, l4t2=dose.l4t2, l4t=dose.l4t, rank=dose.rank, arm=dose.arm, pre=dose.pre, map=the_map,
                           solver=solver, lean=dose.lean, store=lambda: ladder_l5._store(config, ladder_l5.the_settings(config)))


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_searches_deep_builds_the_three_sets_trains_each_from_pre_r64_at_rank_64_measures_them_and_reports(deep, monkeypatch):
    config, calls, kits, solver = deep.config, deep.calls, deep.kits, deep.solver
    read_only = {path: _all(path) for path in (deep.map, deep.l4t2, deep.rank, deep.l4t, deep.arm, deep.pre)}
    summaries = _stage5(config, monkeypatch)
    store = deep.store()
    assert store.root.name == "ladder_l5_deep_pre_r64_seed0" and store.root.parent == deep.l4t2.parent and list(summaries) == STEPS == list(ladder_l5.STEPS)
    assert {path: _all(path) for path in read_only} == read_only               # THE MAP, L4t2's RUN AND THE START MODEL's WERE ONLY READ: file for file; so were the others
    assert (config["lora"]["rank"], config["vllm"]["max_lora_rank"], config["sampling"]["temperature"], config["ladder_loop"]["episode"]["block_problems"]) == (16, 16, 1.0, 5)
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES

    # ---- the search's prepare step: the map, the candidates, the seeded draw, the held-aside set, a sampling seed of its own for each
    read = summaries["ladder_l5_search_prepare"]
    start = deep.rank / "adapters" / "pre_r64"
    assert (read["stage"], read["label"], read["check"], read["seed"], read["start"]) == ("l4", "pretrained on published proofs", "L5", 0, "pre_r64")
    assert (read["start_adapter"], read["start_run"], read["l4t2_run"], read["map_run"], read["models"]) == (str(start), str(deep.rank), str(deep.l4t2), str(deep.map), list(MODELS))
    of_the_map, of_the_search, of_the_set = read["the_map"], read["the_search"], read["the_held_aside"]
    assert (of_the_map["problems"], of_the_map["episodes_each"], of_the_map["sampling_seed"], of_the_map["set"]) == (4, 8, 101, MAP_SET)
    assert of_the_map["problems_by_k"] == {"0": 2, "1": 1, "2": 1} and of_the_map["failed_in_every_attempt"] == 2 and of_the_map["ordinary_solves"]["problems"] == 2
    assert (of_the_search["candidates"], of_the_search["problems"], of_the_search["episodes_each"], of_the_search["attempts_with_the_maps"]) == (2, 1, 56, 64)
    assert (of_the_search["sampling_seed"], of_the_search["sampling_place"], of_the_search["block_problems"], of_the_search["taken_by_the_rule"], of_the_search["taken_by_the_runs_minimum"]) == (1050, 50, 1, 1, 0)
    assert not {1050, 1051} & set(of_the_search["seeds_of_the_stored_samplings_it_is_none_of"].values()) and len(of_the_search["seeds_of_the_stored_samplings_it_is_none_of"]) == 9
    assert read["smoke"] is None and of_the_search["minimum_deep_solves"] == 1 and of_the_search["cap"] == 400
    drawn = store.read_rows("l5_search_problems.jsonl")
    assert drawn == [{"problem_id": SEARCHED, "map_k": 0, "map_n": 8, "taken_by": THE_RULE}]                                        # ONE of the two failed 8 of 8, by the hash
    by_id = {row["problem_id"]: row for row in _file_rows(deep.map / "problems.jsonl")}
    of_a_set = lambda name: [row for row in store.read_rows("problems.jsonl") if row["set"] == name]      # noqa: E731
    assert of_a_set("l5_deep_search") == [{**by_id[SEARCHED], "set": "l5_deep_search"}]                                              # the problem as the map sampled it, its negation and all
    # THE HELD-ASIDE SET: the candidate the search's draw did NOT take; the same problems for the four models that are measured on it; ONE sampling seed, not the search's.
    assert store.read_rows("l5_held_aside_problems.jsonl") == [{"problem_id": HELD, "map_k": 0, "map_n": 8, "taken_by": THE_RULE}] and HELD != SEARCHED
    assert (of_the_set["candidates_not_searched"], of_the_set["problems"], of_the_set["episodes_each"], of_the_set["sampling_seed"], of_the_set["sampling_place"]) == (1, 1, 16, 1051, 51)
    assert (of_the_set["measured"], of_the_set["sets"], of_the_set["block_problems"]) == (list(FOUR), {name: f"l5_held_aside_{name}" for name in FOUR}, 2)
    assert (of_the_set["taken_by_the_rule"], of_the_set["taken_by_the_runs_minimum"]) == (1, 0)
    assert all(of_a_set(f"l5_held_aside_{name}") == [{**by_id[HELD], "set": f"l5_held_aside_{name}"}] for name in FOUR)
    recipe = read["start_recipe"]
    assert (recipe["rank"], recipe["alpha"], recipe["rank_of_the_config"]) == (64, 128, 16) and read["recipe"]["lora"] == {**config["lora"], **RANK_64}
    of_l4t2 = json.loads((deep.l4t2 / "ladder_l4_rows2_prepare.done.json").read_text())
    assert all(read[key] == of_l4t2[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings", "pretraining_file_sha256", "goal_set_again"))
    assert store.read_rows("l4_goal_set_again.jsonl") == _file_rows(deep.l4t2 / "l4_goal_set_again.jsonl") and store.read_rows("l4_heldout_groups.jsonl") == _file_rows(deep.l4t2 / "l4_heldout_groups.jsonl")

    # ---- the deep search: `pre_r64` served at rank 64, 56 attempts at the problem, its own sampling seed, the round's limits, ONE side by the audit rule
    search = summaries["ladder_l5_search"]
    side = f"negation_of_{SEARCHED}" if SEARCHED == "fixture_p3" else SEARCHED
    assert solver.searched == [(side, 56, 1050, config["ladder_loop"]["episode"]["max_new_tokens"])] and config["ladder_loop"]["episode"]["max_new_tokens"] == 1024
    assert kits[0] == {"temperature": 1.0, "adapter": None, "max_lora_rank": 64, "rank": 64, "block_problems": 1}                   # the first engine of the task: the search's
    assert (search["model"], search["set"], search["problems"], search["episodes_each"], search["sampling_seed"], search["resolved_at_least_once"]) == ("pre_r64", "l5_deep_search", 1, 56, 1050, 1)
    solved_in = len(DEEP_SCRIPT[SEARCHED])
    assert search["attempts"] == 56 and search["statuses"] == {"lean_error": 56 - solved_in, "verified": solved_in} and search["adapter"] == str(start) and search["block_problems"] == 1
    result = store.read_rows("episodes_l5_deep_search_problems.jsonl")
    assert [(row["problem_id"], row["resolved"], row["episodes"], row["sides"]) for row in result] == [(SEARCHED, solved_in, 56, 1)]
    attempts = [row for path in sorted(store.root.glob("episodes_l5_deep_search_attempts_*.jsonl")) for row in _file_rows(path)]
    assert len(attempts) == 56 and {row["attempt_id"].split("#")[1] for row in attempts} == {"l5_deep_search"} and store.is_done("episodes_l5_deep_search")

    # ---- the three sets: 2D rows each; a deep solve's row is one of its verified attempts with the place of its first success; the same published rows
    prepared = summaries["ladder_l5_prepare"]
    assert prepared["enough_deep_solves"] == {"what": prepared["enough_deep_solves"]["what"], "deep_solves": 1, "minimum": 1, "passes": True} and prepared["trainings"] == {"run": True}
    assert (prepared["deep_solves"], prepared["rows_of_each_kind"]) == (1, 1) and all(prepared[key] == read[key] for key in ladder_l5.CARRIED)
    sets = {model: store.read_rows(f"l5_training_{model}.jsonl") for model in MODELS}
    assert [len(sets[model]) for model in MODELS] == [2, 2, 2] and all([row["row"] for row in own] == [0, 1] for own in sets.values())
    assert {model: prepared["training_sets"][model]["rows_by_origin"] for model in MODELS} == {PUBLISHED: {"pretraining": 2}, SHALLOW: {"pretraining": 1, "shallow": 1}, DEEP: {"deep": 1, "pretraining": 1}}
    assert {prepared["training_sets"][model]["optimizer_steps"] for model in MODELS} == {1} and {prepared["training_sets"][model]["rows"] for model in MODELS} == {2}
    (deep_row,) = [row for row in sets[DEEP] if row["origin"] == DEEP]
    assert (deep_row["problem_id"], deep_row["first_success"], deep_row["k"], deep_row["n"], deep_row["map_k"], deep_row["taken_by"]) == (SEARCHED, FIRST_SUCCESS[SEARCHED], solved_in, 56, 0, THE_RULE)
    verified = {row["attempt_id"]: row for row in attempts if row["status"] == "verified"}
    assert deep_row["id"] in verified and deep_row["completion"] == verified[deep_row["id"]]["completion"]                           # the row IS a verified attempt of the search
    assert deep_row["id"] == min(verified, key=lambda key: rank(0, "training_target", key)) and 9 <= deep_row["first_success"] <= 64  # ... drawn as a round draws one
    assert len(verified) == 4 and deep_row["id"] != min(verified, key=lambda key: rank(1, "training_target", key)) and deep_row["id"] != min(verified)      # by THE TASK'S seed; not the first verified
    assert (deep_row["side"], deep_row["theorem"]) == (("negation", by_id[SEARCHED]["negation"]) if SEARCHED == "fixture_p3" else ("statement", by_id[SEARCHED]["statement"]))
    (shallow_row,) = [row for row in sets[SHALLOW] if row["origin"] == SHALLOW]
    ordinary = min(ORDINARY, key=lambda key: rank(0, SHALLOW_LABEL, key))                                                            # a draw of one of the two ordinary solves, by ITS hash
    map_attempts = {row["attempt_id"]: row for row in _file_rows(deep.map / f"episodes_{MAP_SET}_attempts_0000.jsonl")}
    assert (shallow_row["problem_id"], shallow_row["first_success"], shallow_row["k"], shallow_row["n"], shallow_row["map_k"], shallow_row["taken_by"]) == (
        ordinary, MAP_VERIFIED[ordinary][0] + 1, len(MAP_VERIFIED[ordinary]), 8, len(MAP_VERIFIED[ordinary]), THE_RULE)
    assert map_attempts[shallow_row["id"]]["status"] == "verified" and shallow_row["completion"] == map_attempts[shallow_row["id"]]["completion"]      # a stored MAP attempt's text
    assert store.read_rows("l5_ordinary_solves.jsonl") == [{"problem_id": ordinary, "map_k": len(MAP_VERIFIED[ordinary]), "map_n": 8, "taken_by": THE_RULE}]
    pretraining = ladder_l4_start.pretrained_on(deep.rank, prepared["pretraining_file_sha256"], "pre_r64")[0]
    order = _draw_order([{**row, "id": f"{row['problem_id']}#pretraining", "origin": "pretraining", "row_of_the_file": place} for place, row in enumerate(pretraining)])
    assert list(map(_placed, _draw_order(sets[DEEP]))) == list(map(_placed, _draw_order(sets[SHALLOW]))) == list(map(_placed, order[:1])) == list(map(_placed, _draw_order(sets[PUBLISHED])[:1]))
    assert list(map(_placed, _draw_order(sets[PUBLISHED]))) == list(map(_placed, order[:2])) and len(order) == 12
    for model in MODELS:
        assert [row["id"] for row in sets[model]] == [row["id"] for row in training_order(sets[model], 0)]
        assert not any(field in row for row in sets[model] if row["origin"] == "pretraining" for field in TEXT_FIELDS)              # no published row with its text
        assert prepared["training_sets"][model]["barred"]["held_out"] == [] and prepared["training_sets"][model]["barred"]["own_rows_not_of_the_base_map"] == []
        assert HELD not in {row["problem_id"] for row in sets[model]}                                                                 # NO SET HOLDS A ROW OF A HELD-ASIDE PROBLEM
    assert store.read_rows("l5_deep_solves.jsonl") == [{**{key: value for key, value in deep_row.items() if key != "row"}, "in_the_set": True}]
    found = prepared["the_search_found"]
    assert (found["problems"], found["solved"], found["episodes"], found["attempts"], found["successes"]) == (1, 1, 56, 56, solved_in) and sum(found["solved_by_first_success"].values()) == 1
    assert prepared["rows_not_taken_by_the_rule"][DEEP] == prepared["rows_not_taken_by_the_rule"][SHALLOW] == 0 and prepared["lines_of_the_rows"]["deep_rows"]["proofs"] == 1

    # ---- each model, in the order `published`, `shallow`, `deep`: trained FROM `pre_r64` at rank 64 on its set in its order, then measured under an engine started for rank 64
    assert [(call["lora"]["rank"], call["lora"]["alpha"], call["more"]) for call in calls] == [(64, 128, {"start": start})] * 3
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l5_deep_pre_r64_seed0_{model}" for model in MODELS]
    assert [list(call["schedule"]["checkpoint_steps"].items()) for call in calls] == [[(model, 1)] for model in MODELS] and all(call["orders"] == [[0, 1]] for call in calls)
    for call, model in zip(calls, MODELS):
        own_examples = {row["problem_id"]: row for row in sets[model] if row["origin"] != "pretraining"}
        assert [example["problem_id"] for example in call["examples"]] == [row["problem_id"] for row in sets[model]] and all(example["theorem"] and example["completion"] for example in call["examples"])
        assert all(example["completion"] == own_examples[example["problem_id"]]["completion"] for example in call["examples"] if example["problem_id"] in own_examples)
    # FOURTEEN ENGINES, each started for rank 64: the search's; the start model's on the held-aside set; then for each model the rungs, G twice, and the held-aside set.
    assert [kit["max_lora_rank"] for kit in kits] == [64] * 14 and [kit["block_problems"] for kit in kits] == [1, 2, *[5, 5, 5, 2] * 3]       # the search's blocks and the held-aside set's are their own
    for model in MODELS:
        train, measured = summaries[f"ladder_l5_train_{model}"], summaries[f"ladder_l5_measure_{model}"]
        assert (train["model"], train["rows"], train["steps"], train["adapter"]) == (model, 2, 1, str(store.root / "adapters" / model)) and train["adapter_saved"]["rank"] == 64
        assert list(measured["goal"]) == [REACH, MORE] and measured["rungs"]["set"] == f"l4_rungs_{model}" and store.is_done(f"ladder_l5_measure_{model}")

    # ---- THE HELD-ASIDE SET, measured by the FOUR models: 16 attempts at its problem each, ONE sampling seed (1051), and no other problem
    held_side = f"negation_of_{HELD}" if HELD == "fixture_p3" else HELD
    assert [entry_ for entry_ in solver.asked if entry_[2] == 1051] == [(held_side, 16, 1051)] * 4 and not [entry_ for entry_ in solver.asked if entry_[2] == 1050 and entry_[1] != 56]
    on_the_set = {}
    for who, name in zip(("start", *MODELS), FOUR):
        own = summaries[f"ladder_l5_held_aside_{who}"]
        assert (own["model"], own["set"], own["problems"], own["episodes_each"], own["sampling_seed"], own["block_problems"]) == (name, f"l5_held_aside_{name}", 1, 16, 1051, 2)
        assert own["adapter"] == (str(start) if who == "start" else str(store.root / "adapters" / name)) and own["attempts"] == 16 and store.is_done(f"ladder_l5_held_aside_{who}")
        on_the_set[name] = store.read_rows(f"episodes_l5_held_aside_{name}_problems.jsonl")
        assert [(row["problem_id"], row["episodes"], row["resolved"]) for row in on_the_set[name]] == [(HELD, 16, ON_THE_SET[name])] and own["successes"] == ON_THE_SET[name]
        assert {row["attempt_id"].split("#")[1] for row in _file_rows(store.path(f"episodes_l5_held_aside_{name}_attempts_0000.jsonl"))} == {f"l5_held_aside_{name}"}
    assert list(summaries).index("ladder_l5_held_aside_start") == list(summaries).index("ladder_l5_prepare") + 1                      # the start model's, before anything is trained
    assert all(list(summaries).index(f"ladder_l5_held_aside_{model}") == list(summaries).index(f"ladder_l5_measure_{model}") + 1 for model in MODELS)

    # ---- the report
    report = summaries["ladder_l5_report"]
    assert list(report) == REPORT_KEYS and json.loads(store.path("report_ladder_l5.json").read_text()) == json.loads(json.dumps(report, default=str))
    assert (report["label"], report["check"], report["start"], report["rank"], report["measured"], report["rows_of_each_kind"]) == ("pretrained on published proofs", "L5", "pre_r64", 64, list(MODELS), 1)
    assert all(line.startswith(f"{SAY}: ") for line in report["lines"]) and report["ok"] is True
    branches = report["branches"]
    assert list(branches) == ["the_primary", "the_mechanism", "the_replication", "what_follows"]
    assert branches["the_primary"] in PRIMARY and branches["the_mechanism"] in MECHANISM and branches["the_replication"] in REPLICATION and branches["what_follows"] in FOLLOWS
    assert list(report["can_this_run_be_read"]) == ["what", "the_deep_search", *MODELS, "the_held_aside_sets"] and report["the_deep_search"]["the_finding"] is False
    assert report["can_this_run_be_read"]["the_held_aside_sets"]["passes"] is True and list(report["can_this_run_be_read"]["the_held_aside_sets"]["share_without_an_answer"]) == [f"held_aside_{name}" for name in FOUR]
    assert report["can_this_run_be_read"][DEEP]["no_barred_row"]["passes"] is True and report["can_this_run_be_read"][SHALLOW]["no_barred_row"]["rows_of_the_models_own"] == 1
    # THE PRIMARY IS READ ON THE HELD-ASIDE ROWS THE FOUR STEPS STORED: `deep`'s against `published`'s (a model of two rows is INCONCLUSIVE here: the figure is kept apart).
    of_the_primary = report["primary"]["measured_and_not_read"] if report["inconclusive"] else report["primary"]
    assert of_the_primary["difference"] == per_attempt(on_the_set[DEEP], on_the_set[PUBLISHED], [HELD], config["evaluation"]["bootstrap_resamples"], config["evaluation"]["bootstrap_seed"])
    assert of_the_primary["difference"]["attempts_each"] == 16 and report["the_held_aside_set"]["problems_measured"] == 1 and {f"held_aside_{name}" for name in FOUR} <= set(report["attempts"])
    assert (of_the_primary["difference"]["successes"], of_the_primary["difference"]["successes_of_the_base"]) == (ON_THE_SET[DEEP], ON_THE_SET[PUBLISHED])       # `deep`'s 5 against `published`'s 3
    for key, (after, before) in (("the_mechanism", (DEEP, SHALLOW)), ("the_replication", (SHALLOW, PUBLISHED))):
        own = (report[key]["measured_and_not_read"] if report["inconclusive"] else report[key])["difference"]
        assert (own["successes"], own["successes_of_the_base"]) == (ON_THE_SET[after], ON_THE_SET[before])
    against = report["the_held_aside_set"]["against_the_start"]
    against = against["measured_and_not_read"] if report["inconclusive"] else against
    assert {name: (own["per_attempt"]["successes"], own["per_attempt"]["successes_of_the_base"]) for name, own in against.items()} == {name: (ON_THE_SET[name], ON_THE_SET["pre_r64"]) for name in MODELS}
    assert report["adapters"]["there"] == sorted(MODELS) and report["adapters"]["kept"] is True and "`pre_r64` is its own run's and was only read" in report["adapters"]["what"]
    assert any("THE DEEP SEARCH: 1 problems attempted (56 more attempts each, after the map's 8), 1 solved" in line for line in report["lines"])
    assert store.done_summary("ladder_l5_report") == {"stage": "l4", "label": "pretrained on published proofs", "check": "L5", "headline": report["headline"], "branches": branches,
                                                      "measured": list(MODELS), "ok": True}
    # ---- queued again: every step returns what is stored; nothing is sampled or trained
    trained, sampled, asked = len(calls), solver.calls, len(solver.asked)
    again = _stage5(config, monkeypatch)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in summaries.items() if "report" not in name}
    assert (len(calls), solver.calls, len(solver.asked)) == (trained, sampled, asked) and again["ladder_l5_report"]["branches"] == branches


def test_as_on_the_gpu_the_search_and_every_measurement_serve_the_adapter_of_their_own_model(deep, monkeypatch):
    config, kits = deep.config, deep.kits
    # The stand-in engine takes no adapter, so in the scripted world none is asked for. As on the GPU every sampling step asks the model server for ITS model's adapter.
    with monkeypatch.context() as patched:
        for module in (ladder_l5, ladder_ceiling):
            patched.setattr(module, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        summaries = _stage5(config, patched)
        store = deep.store()
        start, kept = deep.rank / "adapters" / "pre_r64", store.root / "adapters"
        served = lambda name, directory: (f"ladder_l5_deep_{name}", 1, str(directory))      # noqa: E731
        # The search and the start model's held-aside attempts: `pre_r64`'s adapter, in its own run. Then for each model its rungs, G twice and the held-aside set: the adapter THIS run trained.
        assert [kit["adapter"] for kit in kits] == [served("pre_r64", start)] * 2 + [served(model, kept / model) for model in MODELS for _ in range(4)]
        assert [kit["max_lora_rank"] for kit in kits] == [64] * 14 and summaries["ladder_l5_search"]["stand_in_engine"] is False and len(summaries) == len(STEPS)
        # An adapter that is no longer there: the measurement is refused, not made from another model's.
        shutil.rmtree(kept / DEEP)
        for step in ("ladder_l5_held_aside_deep", "ladder_l5_measure_deep"):
            store.path(f"{step}.done.json").unlink()
            with pytest.raises(RuntimeError, match="adapters/deep is not there: the adapter `deep` is kept by this stage"):
                ladder_l5.STEPS[step](config)
            assert not store.is_done(step)


def test_more_deep_solves_than_the_cap_are_a_seeded_draw_and_the_rest_are_said_to_be_left_out(deep, monkeypatch):
    config = deep.config
    # A map with THREE candidates and one ordinary solve: the search draws two, the third is held aside; the cap keeps one of the two deep solves.
    _write_the_map(config, {"fixture_p1": (), "fixture_p2": (), "fixture_p3": (), "fixture_p4": (5,)})
    config["ladder_loop"]["l5"]["deep"].update({"problems": 2, "cap": 1})
    summaries = _stage5(config, monkeypatch, until="ladder_l5_prepare")
    store, prepared = deep.store(), summaries["ladder_l5_prepare"]
    candidates = ("fixture_p1", "fixture_p2", "fixture_p3")
    searched = sorted(candidates, key=lambda key: rank(0, SEARCH_LABEL, key))[:2]                                                    # the two of the smallest hash, in the draw's order
    assert [row["problem_id"] for row in store.read_rows("l5_search_problems.jsonl")] == searched
    assert [row["problem_id"] for row in store.read_rows("l5_held_aside_problems.jsonl")] == [key for key in candidates if key not in searched]
    kept = min(searched, key=lambda key: rank(0, CAP_LABEL, key))
    assert (prepared["deep_solves"], prepared["rows_of_each_kind"]) == (2, 1) and prepared["the_search_found"]["left_out_by_the_cap"] == 1 and prepared["the_search_found"]["cap"] == 1
    assert [row["problem_id"] for row in store.read_rows("l5_training_deep.jsonl") if row["origin"] == DEEP] == [kept]
    assert {row["problem_id"]: row["in_the_set"] for row in store.read_rows("l5_deep_solves.jsonl")} == {key: key == kept for key in searched}
    assert {row["problem_id"]: row["first_success"] for row in store.read_rows("l5_deep_solves.jsonl")} == {key: FIRST_SUCCESS[key] for key in searched}
    assert "a seeded draw of 1 of the 2 deep solves, the cap" in prepared["training_sets"][DEEP]["what"] and [len(store.read_rows(f"l5_training_{model}.jsonl")) for model in MODELS] == [2, 2, 2]
    assert [row["problem_id"] for row in store.read_rows("l5_training_shallow.jsonl") if row["origin"] == SHALLOW] == ["fixture_p4"]
    # WITHOUT THE CAP there are two deep rows and ONE ordinary solve: a real run is refused, and takes no problem the rule does not give.
    store.path("ladder_l5_prepare.done.json").unlink()
    config["ladder_loop"]["l5"]["deep"]["cap"] = 400
    with pytest.raises(RuntimeError, match="the map holds 1 problems its model solved in 1 to 3 of its attempts and `shallow` takes 2 of them .as many as `deep` has deep rows.: refused"):
        ladder_l5.ladder_l5_prepare(config)
    assert not store.is_done("ladder_l5_prepare")


def test_a_real_run_takes_no_ordinary_solve_the_rule_does_not_give(deep, monkeypatch):
    config = deep.config
    # Two candidates, and two problems the map's model solved EVERY time: no ordinary solve (1 to 3 of 8). A smoke run would take one of the two loosely; a real run is refused.
    _write_the_map(config, {"fixture_p1": (), "fixture_p2": tuple(range(8)), "fixture_p3": (), "fixture_p4": tuple(range(8))})
    _stage5(config, monkeypatch, until="ladder_l5_search")
    store = deep.store()
    assert store.done_summary("ladder_l5_search_prepare")["smoke"] is None and store.done_summary("ladder_l5_search_prepare")["the_map"]["ordinary_solves"]["problems"] == 0
    with pytest.raises(RuntimeError, match="the map holds 0 problems its model solved in 1 to 3 of its attempts and `shallow` takes 1 of them .as many as `deep` has deep rows.: refused"):
        ladder_l5.ladder_l5_prepare(config)
    assert not store.is_done("ladder_l5_prepare") and not [path.name for path in store.root.glob("l5_training_*")] and not store.path("l5_ordinary_solves.jsonl").exists()


def test_with_too_few_deep_solves_no_set_is_built_nothing_is_trained_and_the_report_says_the_search_is_the_finding(deep, monkeypatch):
    config, calls, solver = deep.config, deep.calls, deep.solver
    solver.script.clear()                                                       # the search solves NO problem, and one deep solve is asked
    summaries = _stage5(config, monkeypatch)
    store = deep.store()
    prepared = summaries["ladder_l5_prepare"]
    assert list(summaries) == STEPS and prepared["enough_deep_solves"]["passes"] is False and (prepared["deep_solves"], prepared["rows_of_each_kind"], prepared["training_sets"]) == (0, 0, {})
    assert prepared["trainings"]["run"] is False and "the deep search found 0 deep solves and 1 are asked" in prepared["trainings"]["why"] and "the search is the finding" in prepared["trainings"]["why"]
    # NO SET, NO TRAINING, NO MEASUREMENT (the held-aside set's among them): the ten steps say so, mark nothing and sample nothing.
    not_run = {"stage": "l4", "label": "pretrained on published proofs", "check": "L5", "run": False, "why": prepared["trainings"]["why"]}
    for step in STEPS[3:-1]:
        assert summaries[step] == not_run and not store.is_done(step)
    assert not [path.name for path in store.root.glob("l5_training_*")] and not [path.name for path in store.root.glob("episodes_l5_held_aside_*")]
    assert calls == [] and not (store.root / "adapters").exists() and len(deep.kits) == 1 and not store.path("l5_ordinary_solves.jsonl").exists()
    assert store.read_rows("l5_deep_solves.jsonl") == [] and store.is_done("ladder_l5_prepare") and not [entry_ for entry_ in solver.asked if entry_[1] != ATTEMPTS]
    report = summaries["ladder_l5_report"]
    assert report["inconclusive"] is True and report["failed_checks"] == ["the_deep_search_found_enough"] and report["measured"] == [] and report["ok"] is True
    assert report["branches"] == {"the_primary": INCONCLUSIVE, "the_mechanism": INCONCLUSIVE, "the_replication": INCONCLUSIVE, "what_follows": NOT_READ}
    assert report["the_deep_search"]["the_finding"] is True and (report["the_deep_search"]["solved"], report["the_deep_search"]["problems"]) == (0, 1)
    assert any(line.startswith(f"{SAY}: THE DEEP SEARCH, WHICH IS THE FINDING: 1 problems attempted") for line in report["lines"]) and report["adapters"]["there"] == []
    assert any(line == f"{SAY}: CHECK, the deep search found at least 1 deep solves: 0: FAIL" for line in report["lines"]) and "no model was trained" in report["adapters"]["what"]
    assert store.done_summary("ladder_l5_report")["measured"] == [] and json.loads(store.path("report_ladder_l5.json").read_text())["headline"] == report["headline"]


def test_the_search_prepare_step_refuses_before_anything_is_written(deep, monkeypatch):
    config = deep.config
    store_root = ladder_l5._runs(config) / "ladder_l5_deep_pre_r64_seed0"

    def refused(match):
        with pytest.raises(RuntimeError, match=match):
            ladder_l5.ladder_l5_search_prepare(config)
        assert not store_root.exists() or not any(store_root.iterdir())        # NOTHING WAS WRITTEN

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
    marker = deep.map / "ladder_l4b_map.done.json"
    # ---- THE MAP: there, of this model, of the base map's problems at the stored map's attempts and seed, agreeing with itself and with the package's data
    for name in ("ladder_l4b_map.done.json", "l4_map_pre_r64.jsonl", "problems.jsonl", f"episodes_{MAP_SET}_problems.jsonl", f"episodes_{MAP_SET}.done.json", f"episodes_{MAP_SET}_attempts_0000.jsonl"):
        without(deep.map / name, "ladder_l4_map_pre_r64_seed0 does not hold .*the map of `pre_r64`, its problems or its stored attempts.*Run the step `ladder_l4b_map` to its end first.*nothing was written")
    with_changed(marker, as_json(lambda own: {**own, "model": "pre"}), "recorded the model 'pre'; `pre_r64` is asked")
    with_changed(marker, as_json(lambda own: {**own, "sampling_seed": 102}), "is not the base map of `pre_r64`: it was made with sampling seed 102 and 8 attempts a problem; the stored map's are 101 and 8")
    with_changed(marker, as_json(lambda own: {**own, "episodes_each": 4}), "it was made with sampling seed 101 and 4 attempts a problem")
    with_changed(deep.map / f"episodes_{MAP_SET}_problems.jsonl", as_rows(lambda own: own[1:]), "it is of 3 problems and the base map holds 4")
    with_changed(deep.map / "l4_map_pre_r64.jsonl", as_rows(lambda own: [{**own[0], "resolved": 5}, *own[1:]]), "its per-problem rows do not agree with its own file l4_map_pre_r64.jsonl")
    with_changed(deep.map / "problems.jsonl", as_rows(lambda own: [{**own[0], "statement": own[0]["statement"] + " "}, *own[1:]]), "its problems are not the package's base map with their statements")
    with_changed(deep.map / "problems.jsonl", as_rows(lambda own: own[:3]), "its problems are not the package's base map with their statements")
    # ---- L4t2's run, as L4p reads it: finished, of this seed and start, with a report to be read
    for name in ("ladder_l4_rows2_prepare.done.json", "ladder_l4_rows2_report.done.json", "report_ladder_l4_rows2.json", "problems.jsonl", "l4_heldout_groups.jsonl", "l4_goal_set_again.jsonl"):
        without(deep.l4t2 / name, "ladder_l4_rows2_pre_r64_seed0 does not hold .*L5 makes no round and no pretraining of its own: it reads what the task of stage `ladder_l4_rows2_r64` for seed 0")
    of_l4t2 = deep.l4t2 / "ladder_l4_rows2_prepare.done.json"
    with_changed(of_l4t2, as_json(lambda own: {**own, "seed": 1}), "L4t2's run ladder_l4_rows2_pre_r64_seed0 recorded seed 1 and the start 'pre_r64'; this task's seed is 0")
    with_changed(of_l4t2, as_json(lambda own: {**own, "start_recipe": {**own["start_recipe"], "rank": 32}}), "L4t2's run ladder_l4_rows2_pre_r64_seed0 trained its models at rank 32 and alpha 128")
    with_changed(of_l4t2, as_json(lambda own: {**own, "rung_episodes": 4}), "L4t2's run ladder_l4_rows2_pre_r64_seed0 measured its models otherwise")
    with_changed(of_l4t2, as_json(lambda own: {**own, "pretraining_file_sha256": "0" * 64}), "L4t2's run ladder_l4_rows2_pre_r64_seed0 was prepared on a pretraining file of SHA-256 0000")
    with_changed(deep.l4t2 / "report_ladder_l4_rows2.json", as_json(lambda own: {**own, "ok": False}), "the report of L4t2's run ladder_l4_rows2_pre_r64_seed0 is not to be read")
    with_changed(deep.l4t2 / "l4_goal_set_again.jsonl", as_rows(lambda own: [*own, {"problem_id": "another"}]), "is not the G' L4t2's run ladder_l4_rows2_pre_r64_seed0 stored")
    # ---- the start model's run
    without(deep.rank / "report_ladder_l4_rank.json", "ladder_l4_pretrain_r64_seed0 does not hold .'report_ladder_l4_rank.json'.")
    without(deep.rank / "l4_pretraining_rows.jsonl", "does not hold .'l4_pretraining_rows.jsonl'.: the rows `pre_r64` was pretrained on")
    # ---- the two draws: fewer candidates than the search draws; fewer LEFT than the held-aside set draws (a candidate is searched or held aside, never both)
    deep_settings = config["ladder_loop"]["l5"]["deep"]
    monkeypatch.setitem(deep_settings, "problems", 3)
    refused("the map holds 2 problems its model failed in every attempt and the deep search draws 3 of them: refused .the map ladder_l4_map_pre_r64_seed0.. Nothing was written")
    monkeypatch.setitem(deep_settings, "problems", 2)
    refused("the deep search's draw left 0 of the problems the map's model failed in every attempt and the held-aside set draws 1 of them: refused")
    monkeypatch.setitem(deep_settings, "problems", 1)
    monkeypatch.setitem(deep_settings, "held_aside", 2)
    refused("the deep search's draw left 1 of the problems the map's model failed in every attempt and the held-aside set draws 2 of them: refused")
    monkeypatch.setitem(deep_settings, "held_aside", 1)
    with monkeypatch.context() as patched:                  # a draw gone wrong, which names a SEARCHED problem: held where it is made, before anything is written
        patched.setattr(ladder_l5, "held_aside", lambda rows_, searched, count, seed, at_least=0: [{"problem_id": SEARCHED, "map_k": 0, "map_n": 8, "taken_by": THE_RULE}])
        refused(f"1 problems of the held-aside set are problems the deep search attempted .first: {SEARCHED}.: no model here trains on a held-aside problem, and none of them was searched")
    # ---- the two sampling seeds: each its own, and no stored sampling's
    for place, said in ((2, "`pre_r64`'s rungs"), (1, "`pre_r64`'s reach"), (20, "`pre_r64`'s more"), (40, "L4t's hot sampling 1")):
        monkeypatch.setitem(deep_settings, "sampling_place", place)
        refused(f"the deep search's sampling seed is {1000 + place} .ladder_loop.l5.deep.sampling_place {place}. and .*{said}.* already sampled with it: it has a sampling seed of its own")
    monkeypatch.setitem(deep_settings, "sampling_place", 50)
    for place, said in ((2, "`pre_r64`'s rungs"), (41, "L4t's hot sampling 2")):
        monkeypatch.setitem(deep_settings, "held_aside_sampling_place", place)
        refused(f"the held-aside set's sampling seed is {1000 + place} .ladder_loop.l5.deep.held_aside_sampling_place {place}. and .*{said}.* already sampled with it")
    monkeypatch.setitem(deep_settings, "held_aside_sampling_place", 50)
    with pytest.raises(ValueError, match="deep.sampling_place is 50 and deep.held_aside_sampling_place 50.*the held-aside set's sampling seed is not the search's"):
        ladder_l5.ladder_l5_search_prepare(config)
    monkeypatch.setitem(deep_settings, "held_aside_sampling_place", 51)
    # ---- settings that are not the stage's
    for key, value, said in (("attempts", 0, "ladder_loop.l5.deep.attempts is 0"), ("cap", True, "ladder_loop.l5.deep.cap is True"), ("minimum", 401, "ladder_loop.l5.deep.minimum is 401 and deep.cap 400"),
                             ("problems", 2.5, "ladder_loop.l5.deep.problems is 2.5"), ("held_aside", 0, "ladder_loop.l5.deep.held_aside is 0"),
                             ("held_aside_attempts", -1, "ladder_loop.l5.deep.held_aside_attempts is -1")):
        monkeypatch.setitem(deep_settings, key, value)
        with pytest.raises(ValueError, match=said):
            ladder_l5.ladder_l5_search_prepare(config)
        monkeypatch.setitem(deep_settings, key, {"attempts": 56, "cap": 400, "minimum": 1, "problems": 1, "held_aside": 1, "held_aside_attempts": 16}[key])
    monkeypatch.setitem(config["ladder_loop"]["l5"], "models", ["deep", "shallow", "published"])
    with pytest.raises(ValueError, match="ladder_loop.l5.models is .'deep', 'shallow', 'published'. and the stage trains and measures .'published', 'shallow', 'deep'.: change both together"):
        ladder_l5.the_settings(config)
    monkeypatch.setitem(config["ladder_loop"]["l5"], "models", list(MODELS))
    monkeypatch.setitem(config["ladder_loop"]["l5"], "start", "pre")
    with pytest.raises(ValueError, match="ladder_loop.l5.start is `pre` and ladder_loop.l4.rows2.start is `pre_r64`"):
        ladder_l5.the_settings(config)
    monkeypatch.setitem(config["ladder_loop"]["l5"], "start", "pre_r64")
    monkeypatch.setitem(config["ladder_loop"]["l5"]["shallow"], "solved_at_least", 4)
    with pytest.raises(ValueError, match="shallow.solved_at_least is 4 and solved_at_most 3"):
        ladder_l5.the_settings(config)
    monkeypatch.setitem(config["ladder_loop"]["l5"]["shallow"], "solved_at_least", 1)
    # ---- a task of another seed is A FURTHER SEED (`test_ladder_l5_seeds.py`): it reads the run of the search's seed, which is not there, and no run of L4t2 of ITS seed; the
    # two sampling seeds are the search's seed's at every seed. And after every refusal the step runs to its end
    monkeypatch.setenv(SEEDS, "1")
    with pytest.raises(RuntimeError, match="ladder_l5_deep_pre_r64_seed0 does not hold .*the FINISHED run of the deep search.*Run the task of stage `ladder_l5_deep` for seed 0"):
        ladder_l5.ladder_l5_search_prepare(config)
    assert not any((ladder_l5._runs(config) / "ladder_l5_deep_pre_r64_seed1").iterdir()) and not (ladder_l5._runs(config) / "ladder_l4_rows2_pre_r64_seed1").exists()
    assert (ladder_l5.search_seed(config, ladder_l5.the_settings(config)), ladder_l5.held_aside_seed(config, ladder_l5.the_settings(config))) == (1050, 1051)
    monkeypatch.delenv(SEEDS)
    assert (ladder_l5.search_seed(config, ladder_l5.the_settings(config)), ladder_l5.held_aside_seed(config, ladder_l5.the_settings(config))) == (1050, 1051)
    assert ladder_l5.ladder_l5_search_prepare(config)["the_search"]["problems"] == 1 and (store_root / "ladder_l5_search_prepare.done.json").exists()
    # A later step before its own: refused.
    with pytest.raises(RuntimeError, match="L5's three training sets needs the step ladder_l5_search of this run, which is not done: the stage `ladder_l5_deep` runs the steps in order"):
        ladder_l5.ladder_l5_prepare(config)
    with pytest.raises(RuntimeError, match="the training of `deep` needs the step ladder_l5_prepare of this run"):
        ladder_l5.STEPS["ladder_l5_train_deep"](config)
    with pytest.raises(RuntimeError, match="the measurement of `pre_r64` on the held-aside set needs the step ladder_l5_prepare of this run"):
        ladder_l5.STEPS["ladder_l5_held_aside_start"](config)
    with pytest.raises(RuntimeError, match="L5's report needs the step ladder_l5_prepare of this run"):
        ladder_l5.ladder_l5_report(config)


def test_no_set_holds_a_held_aside_problem_and_each_model_is_measured_there_once_it_can_be(deep, monkeypatch):
    config = deep.config
    _stage5(config, monkeypatch, until="ladder_l5_search")
    store = deep.store()
    kept = store.path("l5_held_aside_problems.jsonl").read_text()
    as_held = lambda problem: store.path("l5_held_aside_problems.jsonl").write_text(json.dumps({"problem_id": problem, "map_k": 0, "map_n": 8, "taken_by": THE_RULE}) + "\n")      # noqa: E731
    # A held-aside file that names the SEARCHED problem (a store gone wrong): the three sets are refused, and nothing of them is written.
    as_held(SEARCHED)
    with pytest.raises(RuntimeError, match="1 problems of the held-aside set are problems the deep search attempted.*no model here trains on a held-aside problem, and none of them was searched. Refused"):
        ladder_l5.ladder_l5_prepare(config)
    assert not store.is_done("ladder_l5_prepare") and not [path.name for path in store.root.glob("l5_training_*")]
    # One that names an ordinary solve: the ordinary draw passes it over, and `shallow` takes the other.
    first = min(ORDINARY, key=lambda key: rank(0, SHALLOW_LABEL, key))
    as_held(first)
    ladder_l5.ladder_l5_prepare(config)
    assert [row["problem_id"] for row in store.read_rows("l5_training_shallow.jsonl") if row["origin"] == SHALLOW] == [key for key in ORDINARY if key != first]
    store.path("ladder_l5_prepare.done.json").unlink()
    store.path("l5_held_aside_problems.jsonl").write_text(kept)
    ladder_l5.ladder_l5_prepare(config)
    assert [row["problem_id"] for row in store.read_rows("l5_training_shallow.jsonl") if row["origin"] == SHALLOW] == [first]
    # A trained model's held-aside step before its training: refused; the start model's needs none.
    with pytest.raises(RuntimeError, match="the measurement of `deep` on the held-aside set needs the step ladder_l5_train_deep of this run"):
        ladder_l5.STEPS["ladder_l5_held_aside_deep"](config)
    measured = ladder_l5.STEPS["ladder_l5_held_aside_start"](config)
    assert (measured["model"], measured["problems"], measured["sampling_seed"]) == ("pre_r64", 1, 1051) and store.is_done("ladder_l5_held_aside_start") and "measured_again" not in measured
    # A HELD-ASIDE SET LEAN DID NOT ANSWER is measured again, and no other, when the task is queued again.
    name = "episodes_l5_held_aside_pre_r64_problems.jsonl"
    own = store.read_rows(name)
    own[0]["attempts_without_an_answer"] = 5
    store.write_rows(name, own)
    asked = len(deep.solver.asked)
    again = ladder_l5.STEPS["ladder_l5_held_aside_start"](config)
    assert again["measured_again"] == ["l5_held_aside_pre_r64"] and len(deep.solver.asked) == asked + 1 and store.read_rows(name)[0]["attempts_without_an_answer"] == 0
    assert (measured["successes"], again["successes"]) == (2, 3) and store.read_rows(name)[0]["resolved"] == 3                        # sampled AGAIN, not read again
    assert ladder_l5.STEPS["ladder_l5_held_aside_start"](config) == again and len(deep.solver.asked) == asked + 1                     # answered now: returned as stored
    # Sizes that are no longer the ones the run was prepared for: nothing is sampled.
    store.path("ladder_l5_held_aside_start.done.json").unlink()
    monkeypatch.setitem(config["ladder_loop"]["l5"]["deep"], "held_aside_attempts", 8)
    with pytest.raises(RuntimeError, match="this run was prepared for 16 attempts a held-aside problem with sampling seed 1051 in blocks of 2 problems, and the config now gives 8, 1051 and 5"):
        ladder_l5.STEPS["ladder_l5_held_aside_start"](config)
    assert len(deep.solver.asked) == asked + 1


def test_the_smoke_only_path_is_refused_in_a_run_of_the_seeds_own_and_on_a_map_that_is_no_fixtures(deep, monkeypatch):
    config = deep.config
    store_root = ladder_l5._runs(config) / "ladder_l5_deep_pre_r64_seed0"
    said = "RLVR_LEAN_LADDER_L5_SMOKE_PROBLEMS is set .1.: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_RUN. and on the fixtures' base map"
    # THE VARIABLE WITHOUT A RUN DIRECTORY OF THE TASK'S NAMING: the seed's own directory is never filled by the looser rule.
    monkeypatch.setenv(SMOKE, "1")
    with pytest.raises(RuntimeError, match=said + ".*takes its problems by the rule and by nothing else. Nothing was written"):
        ladder_l5.ladder_l5_search_prepare(config)
    assert not store_root.exists() or not any(store_root.iterdir())
    # ... and WITH one, on a map that is not a fixture's (the pool's base map): refused all the same.
    monkeypatch.setenv(RUN, "ladder_l5_deep_elsewhere")
    marker = deep.map / "ladder_l4b_map.done.json"
    recorded = json.loads(marker.read_text())
    marker.write_text(json.dumps({**recorded, "fixture": False}))
    with pytest.raises(RuntimeError, match=said):
        ladder_l5.ladder_l5_search_prepare(config)
    assert not (ladder_l5._runs(config) / "ladder_l5_deep_elsewhere").exists() or not any((ladder_l5._runs(config) / "ladder_l5_deep_elsewhere").iterdir())
    marker.write_text(json.dumps(recorded))
    # A smoke run (its own directory, the fixtures' map): the rule's problems first, and nothing joins when the rule gives enough.
    summary = ladder_l5.ladder_l5_search_prepare(config)
    assert summary["smoke"]["problems_at_least"] == 1 and summary["the_search"]["minimum_deep_solves"] == 1
    assert summary["the_search"]["taken_by_the_runs_minimum"] == summary["the_held_aside"]["taken_by_the_runs_minimum"] == 0
    # WITHOUT THE VARIABLE the minimum is the setting's and a map with no candidate is refused: there is no looser rule to fall back on.
    monkeypatch.delenv(SMOKE)
    monkeypatch.delenv(RUN)
    assert ladder_l5.smoke_problems() == 0
    _write_the_map(config, {"fixture_p1": (0, 1, 2), "fixture_p2": tuple(range(8)), "fixture_p3": tuple(range(8)), "fixture_p4": tuple(range(8))})
    with pytest.raises(RuntimeError, match="the map holds 0 problems its model failed in every attempt and the deep search draws 1 of them: refused"):
        ladder_l5.ladder_l5_search_prepare(config)
    monkeypatch.setitem(config["ladder_loop"]["l5"]["deep"], "minimum", 150)
    _write_the_map(config)
    assert ladder_l5.ladder_l5_search_prepare(config)["the_search"]["minimum_deep_solves"] == 150


def test_a_search_lean_did_not_answer_is_not_kept_and_is_sampled_again(deep, monkeypatch):
    config, solver = deep.config, deep.solver
    ladder_l5.ladder_l5_search_prepare(config)
    store = deep.store()
    # ---- Lean gave no verdict on too many attempts: the step fails, unmarked, and the set is forgotten; the task queued again samples it again
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l5, "lean_did_not_answer", lambda rows_: True)
        with pytest.raises(RuntimeError, match="Lean gave no verdict on too many attempts of l5_deep_search: the search is not kept. Queue the task again: this step samples it again"):
            ladder_l5.ladder_l5_search(config)
    assert not store.is_done("ladder_l5_search") and not store.is_done("episodes_l5_deep_search") and not list(store.root.glob("episodes_l5_deep_search_block_*.done.json"))
    sampled = len(solver.searched)
    assert ladder_l5.ladder_l5_search(config)["resolved_at_least_once"] == 1 and len(solver.searched) == 2 * sampled and store.is_done("ladder_l5_search")
    # ---- a search prepared with other sizes than the settings now give is not made
    monkeypatch.setitem(config["ladder_loop"]["l5"]["deep"], "attempts", 40)
    store.path("ladder_l5_search.done.json").unlink()
    with pytest.raises(RuntimeError, match="this run was prepared for a search of 56 attempts a problem with sampling seed 1050 in blocks of 1 problems, and the config now gives 40"):
        ladder_l5.ladder_l5_search(config)


def test_a_proof_of_the_ruled_out_side_in_the_deep_search_raises_the_soundness_alarm(deep, monkeypatch):
    config, solver = deep.config, deep.solver
    # Two known-true candidates; every problem's ruled-out side is audited, and the verifier goes wrong on the searched one: a proof of the negation of a known-true problem.
    _write_the_map(config, {"fixture_p1": (), "fixture_p2": (), "fixture_p3": (1,), "fixture_p4": (5,)})
    searched = min(("fixture_p1", "fixture_p2"), key=lambda key: rank(0, SEARCH_LABEL, key))
    config["ladder_loop"]["episode"]["contradicted_side_audit_share"] = 1.0
    monkeypatch.setattr(deep.lean, "alarm", searched)
    ladder_l5.ladder_l5_search_prepare(config)
    with pytest.raises(SoundnessAlarm, match=searched):
        ladder_l5.ladder_l5_search(config)
    store = deep.store()
    assert not store.is_done("ladder_l5_search") and not store.is_done("episodes_l5_deep_search")
    assert {name for name, *_ in solver.searched} == {searched, f"negation_of_{searched}"}                                          # both sides were sampled: the audit rule


def test_the_steps_run_through_the_gpu_entry_point(deep, monkeypatch, tmp_path):
    monkeypatch.setattr(milestone2, "load_config", lambda path: deep.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert deep.store().is_done(ladder_l5.REPORT)


# ------------------------------------------------------------------- the stages that existed before this one
def test_l4p_l4t2_and_l4t_from_pre_r64_write_what_they_wrote_and_are_left_as_they_were_after_this_stage_ran_beside_them(deep, monkeypatch):
    config, dose_, l4t2, l4t = deep.config, deep.dose, deep.l4t2, deep.l4t
    # L4p's own stage, run to its end in this world BEFORE this stage: what it leaves is what its own tests froze.
    with monkeypatch.context() as patched:
        patched.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (dose_.solver, stand_in_parameters)))
        l4p_summaries = _l4p(config, patched)
    l4p = dose_.store().root
    as_it_wrote = {path: _all(path) for path in (l4p, l4t2, l4t)}
    assert l4p.name == "ladder_l4_dose_pre_r64_seed0" and sorted(name for name in as_it_wrote[l4p] if not name.startswith("episodes_")) == L4P_FILES and list(l4p_summaries) == L4P_STEPS
    assert sorted(name for name in as_it_wrote[l4t2] if not name.startswith("episodes_")) == L4T2_FILES and len(as_it_wrote[l4t2]) == 61
    assert sorted(name for name in as_it_wrote[l4t] if not name.startswith("episodes_")) == sorted(L4T_FILES_FROM_PRE_R64) and len(as_it_wrote[l4t]) == 100
    assert list(dose_.l4t2_summaries) == L4T2_STEPS and list(dose_.rows2.l4t_summaries) == L4T_STEPS_FROM_PRE_R64
    del deep.calls[:]
    _stage5(config, monkeypatch)
    assert {path: _all(path) for path in as_it_wrote} == as_it_wrote and not [name for files in as_it_wrote.values() for name in files if "l5" in name or "deep" in name or "shallow" in name]
    # Each one's own task, queued again AFTER this stage ran in the same store: every step returns what is stored, nothing is trained or sampled, no file moves.
    trained, sampled = len(deep.calls), deep.solver.calls
    comparable = lambda summaries: {name: ({key: value for key, value in summary.items() if key != "adapters"} if "report" in name else summary) for name, summary in summaries.items()}      # noqa: E731
    assert comparable(_l4p(config, monkeypatch)) == comparable(l4p_summaries)
    assert comparable(_l4t2(config, monkeypatch)) == comparable(dose_.l4t2_summaries)
    monkeypatch.setenv(L4T_START, "pre_r64")
    monkeypatch.setenv(L4T_MINIMUM, "2")
    assert comparable(_l4t(config, monkeypatch)) == comparable(dose_.rows2.l4t_summaries)
    assert (len(deep.calls), deep.solver.calls) == (trained, sampled) and {path: _all(path) for path in as_it_wrote} == as_it_wrote
    # Their names are their own as they were; this stage's are beside them and never the same.
    own = ladder_l5.THE_STAGE
    assert (own.name, own.steps, own.files, own.reader, own.prepare, own.start_said_by) == ("ladder_l5_deep", "ladder_l5", "l5", "L5", "ladder_l5_prepare", "ladder_loop.l5.start")
    assert (ladder_l4_dose.THE_STAGE.name, ladder_l4_rows2.THE_STAGE.name, ladder_l4_rows.THE_STAGE.name) == ("ladder_l4_dose_r64", "ladder_l4_rows2_r64", "ladder_l4_rows")
    assert not (set(ladder_l4_rows.STEPS) | set(ladder_l4_rows2.STEPS) | set(ladder_l4_dose.STEPS) | set(ladder_l4b.STEPS)) & set(ladder_l5.STEPS)
    assert (own.training_file(DEEP), own.loss_file(SHALLOW), own.train_marker(PUBLISHED)) == ("l5_training_deep.jsonl", "l5_loss_shallow.json", "ladder_l5_train_published")
    assert config["ladder_loop"]["l4"]["dose"] == {"start": "pre_r64", "models": ["matched_1262", "second_half", "second_full"]} and config["ladder_loop"]["l4"]["rows"]["start"] == "pre"


def test_l4t_from_pre_writes_what_it_wrote(rows, monkeypatch):  # noqa: F811
    config = rows.config
    summaries = _l4t_from_pre(config, monkeypatch)
    written = _all(rows.store().root)
    assert rows.store().root.name == "ladder_l4_rows_seed0" and sorted(name for name in written if not name.startswith("episodes_")) == L4T_FILES_FROM_PRE and len(written) == 79
    assert list(summaries) == L4T_STEPS_FROM_PRE == ladder_l4_rows.steps_of("pre") and not [name for name in written if "l5" in name or "deep" in name or "shallow" in name]
    assert ladder_l5.the_settings(config).start == "pre_r64" == ladder_l5.the_settings(config).l4t2.start       # this stage does not run from `pre`


# ------------------------------------------------------------------------------------------- the smoke stage
def test_the_smoke_stage_takes_its_problems_loosely_marks_every_one_and_reads_the_three_smoke_runs_and_nothing_else(arm, monkeypatch):  # noqa: F811
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config, calls, kits = arm.config, [], []
    monkeypatch.delenv(ARM)
    for name in (RUN, L4T2_RUN, SMOKE, MAP_RUN, ladder_l4_rows2.RUN_VARIABLE, ladder_l4_rows2.L4T_RUN_VARIABLE, ladder_l4_rows.RUN_VARIABLE, ladder_l4_rows.ARM_RUN_VARIABLE, L4T_MINIMUM, L4T_START,
                 ladder_l4_rows.START_CHECKS_VARIABLE, ladder_l4_rank.CHECK_VARIABLE, RANK_RUN):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    _trainings(monkeypatch, calls)
    solver = DeepSolver(arm.solver, {f"fixture_p{index}": (4,) for index in range(1, 5)})       # in the search every problem is solved at its fifth attempt
    engines = lambda self, adapter=None: (lambda given: (kits.append((given["sampling"]["temperature"], given["vllm"]["max_lora_rank"])), (solver, stand_in_parameters))[1])      # noqa: E731
    # ---- the five smoke tasks before it, each in ITS OWN environment: the pretraining's, the arm's, the check's, L4t's from `pre_r64`, then L4t2's
    steps = {**ladder_l2.STEPS, **ladder_l3d2.STEPS, **ladder_l4.STEPS, **ladder_l4_rank.STEPS, **ladder_l4_rows.STEPS, **ladder_l4_rows2.STEPS}
    for stage_name in ("ladder_l4_pretrain_smoke", "ladder_l4_smoke", "ladder_l4_rank_smoke", "ladder_l4_rows_r64_smoke", "ladder_l4_rows2_r64_smoke"):
        options = entry.step_fields(entry.STAGES[stage_name][2])[2]
        with monkeypatch.context() as patched:                                 # one task's environment: gone when the task is
            patched.delenv(ladder_loop.DATA_VARIABLE)
            for name, value in options["environment"].items():
                patched.setenv(name, value)
            if stage_name == "ladder_l4_smoke":
                arm.with_lean()
                patched.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
            if stage_name in ("ladder_l4_rows_r64_smoke", "ladder_l4_rows2_r64_smoke"):
                patched.setattr(ladder_round.Engines, "kit", engines)
            for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
                if environment == "gpu" and step in steps:
                    _on_the_gpu(patched, steps[step], config) if "_train" in step else steps[step](config)
    runs = pipeline.STORE / "runs-v4.27"
    # ---- this stage's smoke task: the three smoke runs it reads, the fixtures, a run directory of its own, and the smoke-only variable, which this stage ALONE sets
    options = entry.step_fields(entry.STAGES["ladder_l5_deep_smoke"][2])[2]
    assert all(entry.step_fields(step)[2] == options for step in entry.STAGES["ladder_l5_deep_smoke"])
    of_the_pretraining = entry.step_fields(entry.STAGES["ladder_l4_pretrain_smoke"][2])[2]["environment"]
    assert options["environment"] == {ladder_loop.DATA_VARIABLE: of_the_pretraining[ladder_loop.DATA_VARIABLE], ladder_l4.FILE_VARIABLE: str(FIXTURE), RANK_RUN: "ladder_l4_rank_smoke",
                                      MAP_RUN: "ladder_l4b_map_smoke", L4T2_RUN: "ladder_l4_rows2_r64_smoke", RUN: "ladder_l5_deep_smoke", SMOKE: "1"}
    of_l4t2s_smoke = entry.step_fields(entry.STAGES["ladder_l4_rows2_r64_smoke"][2])[2]["environment"]
    of_the_arms_smoke = entry.step_fields(entry.STAGES["ladder_l4b_reward_rehearse_smoke"][2])[2]["environment"]
    assert (of_l4t2s_smoke[ladder_l4_rows2.RUN_VARIABLE], of_l4t2s_smoke[RANK_RUN], of_the_arms_smoke[MAP_RUN]) == (
        options["environment"][L4T2_RUN], options["environment"][RANK_RUN], options["environment"][MAP_RUN])                      # the three runs it reads are the ones those tasks write
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)
    for name, value in options["environment"].items():
        monkeypatch.setenv(name, value)
    arm.with_lean()
    monkeypatch.setattr(ladder_round.Engines, "kit", engines)
    # THE MAP of the smoke `pre_r64`, as the box's has it: every problem solved (k = 3, 8, 8, 8): no problem failed every time, one ordinary solve.
    the_map = _write_the_map(config, {"fixture_p1": (0, 1, 2), "fixture_p2": tuple(range(8)), "fixture_p3": tuple(range(8)), "fixture_p4": tuple(range(8))}, directory=runs / "ladder_l4b_map_smoke")
    l4t2, rank_run = runs / "ladder_l4_rows2_r64_smoke", runs / "ladder_l4_rank_smoke"
    as_stored = {path: _all(path) for path in runs.iterdir() if path.is_dir()}
    del calls[:], kits[:]
    # It reads NOTHING else: with the four other smoke runs gone from the box it runs to its end.
    others = ("ladder_l1_smoke", "ladder_l4_pretrain_smoke", "ladder_l4_smoke", "ladder_l4_rows_r64_smoke")
    for name in others:
        (runs / name).rename(runs / f"{name}_elsewhere")
    summaries = _stage5(config, monkeypatch, stage_name="ladder_l5_deep_smoke")
    for name in others:
        (runs / f"{name}_elsewhere").rename(runs / name)
    store = ladder_l5._store(config, ladder_l5.the_settings(config))
    assert store.root.name == "ladder_l5_deep_smoke" and list(summaries) == STEPS and {path: _all(path) for path in as_stored} == as_stored
    assert sorted(path.name for path in runs.iterdir() if path.name.startswith("ladder_l5")) == ["ladder_l5_deep_smoke"]
    read = summaries["ladder_l5_search_prepare"]
    assert (read["start_run"], read["l4t2_run"], read["map_run"]) == (str(rank_run), str(l4t2), str(the_map)) and read["start_recipe"]["rank"] == 64
    assert read["start_run_read"]["checks_waived_as_l4t2s_run_recorded"] is json.loads((l4t2 / "ladder_l4_rows2_prepare.done.json").read_text())["start_run_read"]["checks_waived_as_l4ts_run_recorded"]
    # THE SMOKE-ONLY PATH: the rule gives the search NO problem and the held-aside set none; ONE joins each by its own hash order, each marked; one deep solve is enough; the
    # real numbers (56 attempts, 16, the two seeds, the cap) are untouched.
    of_the_search, of_the_set = read["the_search"], read["the_held_aside"]
    assert (read["the_map"]["failed_in_every_attempt"], read["the_map"]["problems_by_k"], read["the_map"]["ordinary_solves"]["problems"]) == (0, {"3": 1, "8": 3}, 1)
    assert (of_the_search["candidates"], of_the_search["problems"], of_the_search["taken_by_the_rule"], of_the_search["taken_by_the_runs_minimum"], of_the_search["minimum_deep_solves"]) == (0, 1, 0, 1, 1)
    assert (of_the_set["problems"], of_the_set["taken_by_the_rule"], of_the_set["taken_by_the_runs_minimum"], of_the_set["episodes_each"], of_the_set["sampling_seed"]) == (1, 0, 1, 16, 1051)
    assert read["smoke"]["problems_at_least"] == 1 and (of_the_search["episodes_each"], of_the_search["sampling_seed"], of_the_search["cap"]) == (56, 1050, 400)
    ids = [f"fixture_p{index}" for index in range(1, 5)]
    searched = min(ids, key=lambda key: rank(0, SEARCH_LABEL, key))
    aside = min((key for key in ids if key != searched), key=lambda key: rank(0, HELD_ASIDE_LABEL, key))
    assert store.read_rows("l5_search_problems.jsonl") == [{"problem_id": searched, "map_k": 3 if searched == "fixture_p1" else 8, "map_n": 8, "taken_by": THE_MINIMUM}]
    assert store.read_rows("l5_held_aside_problems.jsonl") == [{"problem_id": aside, "map_k": 3 if aside == "fixture_p1" else 8, "map_n": 8, "taken_by": THE_MINIMUM}]
    prepared = summaries["ladder_l5_prepare"]
    assert (prepared["deep_solves"], prepared["rows_of_each_kind"]) == (1, 1) and prepared["rows_not_taken_by_the_rule"][DEEP] == 1
    sets = {model: store.read_rows(f"l5_training_{model}.jsonl") for model in MODELS}
    assert [len(sets[model]) for model in MODELS] == [2, 2, 2] and [(row["problem_id"], row["taken_by"], row["first_success"]) for row in sets[DEEP] if row["origin"] == DEEP] == [(searched, THE_MINIMUM, 13)]
    (ordinary,) = store.read_rows("l5_ordinary_solves.jsonl")                                                                        # ... the fifth attempt of the search, after the map's 8
    assert ordinary["problem_id"] not in (searched, aside) and (ordinary["taken_by"] == THE_RULE) is (1 <= ordinary["map_k"] <= 3)   # a row is the rule's exactly when the rule gives it
    assert prepared["rows_not_taken_by_the_rule"][SHALLOW] == int(ordinary["taken_by"] == THE_MINIMUM) and aside not in {row["problem_id"] for model in MODELS for row in sets[model]}
    assert list(map(_placed, _draw_order(sets[DEEP]))) == list(map(_placed, _draw_order(sets[SHALLOW]))) == list(map(_placed, _draw_order(sets[PUBLISHED])[:1]))
    # Each model trained from the smoke `pre_r64` AT RANK 64 and measured under a model server started for rank 64; the held-aside problem attempted 16 times by each of the four.
    assert [(call["lora"]["rank"], call["lora"]["alpha"], call["more"]) for call in calls] == [(64, 128, {"start": rank_run / "adapters" / "pre_r64"})] * 3
    assert sorted(path.name for path in (store.root / "adapters").iterdir()) == sorted(MODELS) and kits == [(1.0, 64)] * 11         # the search; the start model on the held-aside set; then the rungs, G and that set for each model
    assert len([entry_ for entry_ in solver.asked if entry_[1:] == (16, 1051)]) == 4 and all(summaries[f"ladder_l5_held_aside_{who}"]["problems"] == 1 for who in ("start", *MODELS))
    report = summaries["ladder_l5_report"]
    assert report["measured"] == list(MODELS) and all(line.startswith(f"{SAY}: ") for line in report["lines"]) and report["rows_not_taken_by_the_rule"][DEEP] == 1
    assert any("A SMOKE RUN: rows of problems the rule did not give were taken" in line and "1 of `deep`'s own rows" in line and "1 of the search's problems; 1 of the held-aside problems" in line
               for line in report["lines"])
    # The primary is on the held-aside set, which the smoke measures: it is read (or a model of two rows is INCONCLUSIVE). G has ONE sampling: nothing stands beside on G'.
    assert report["branches"]["the_primary"] in (*PRIMARY[:3], INCONCLUSIVE) and report["the_held_aside_set"]["problems_measured"] == 1
    assert any("the models have ONE sampling of G here (a smoke run)" in line for line in report["lines"])


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_the_smoke_variable_is_the_smoke_stages_alone():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_deep"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l5.STEPS) == STEPS and ladder_l5.ENTRY_STAGE == "ladder_l5_deep" and len(STEPS) == 14
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_deep_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    # THE ORDER IS THE SETTING'S: the search first; the start model on the held-aside set; then each model trained, measured and measured on that set before the next.
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    settings = whole["ladder_loop"]["l5"]
    assert settings["models"] == ["published", "shallow", "deep"] and [step for _, step, _ in steps if step and "_train_" in step] == [f"ladder_l5_train_{model}" for model in settings["models"]] == list(TRAININGS)
    assert [step for _, step, _ in steps if step and "_held_aside_" in step] == [f"ladder_l5_held_aside_{who}" for who in ("start", *settings["models"])]
    own = ladder_l5.the_settings(whole)
    assert (own.start, own.models, own.problems, own.attempts, own.sampling_place, own.cap, own.minimum, own.least, own.most) == ("pre_r64", MODELS, 1000, 56, 50, 400, 150, 1, 3)
    assert (own.held_aside, own.held_aside_attempts, own.held_aside_place) == (600, 16, 51) and ladder_l5.held_aside_name(own, "start") == "pre_r64" and ladder_l5.held_aside_name(own, "deep") == "deep"
    assert ladder_l5.run_name(own, 0) == "ladder_l5_deep_pre_r64_seed0" and ladder_l4_rows2.run_name(own.l4t2, 0) == "ladder_l4_rows2_pre_r64_seed0"
    assert (ladder_l5.search_seed(whole, own), ladder_l5.held_aside_seed(whole, own)) == (1050, 1051) and ladder_l5.held_aside_set("pre_r64") == "l5_held_aside_pre_r64"
    assert ladder_l4_start.the_start(whole, own.start).map_directory.name == "ladder_l4_map_pre_r64_seed0" and ladder_l4_rank.check_of(whole, own.start).rank == 64
    # A BLOCK IS ABOUT AS MANY ATTEMPTS AS A ROUND'S (128 problems x 8): 18 problems of the search's 56 attempts (56 blocks of its 1,000), 64 of the held-aside set's 16.
    assert (ladder_l5.block_of(whole, 56), ladder_l5.block_of(whole, 16), ladder_l5.block_of(whole, 8), ladder_l5.block_of(whole, 5000)) == (18, 64, 128, 1) and -(-1000 // 18) == 56
    of_the_search = ladder_l5.episodes_config(whole, 56)
    assert of_the_search["ladder_loop"]["episode"] == {**whole["ladder_loop"]["episode"], "block_problems": 18} and whole["ladder_loop"]["episode"]["block_problems"] == 128
    assert {key: value for key, value in of_the_search.items() if key != "ladder_loop"} == {key: value for key, value in whole.items() if key != "ladder_loop"}
    assert {key: value for key, value in of_the_search["ladder_loop"].items() if key != "episode"} == {key: value for key, value in whole["ladder_loop"].items() if key != "episode"}
    assert ladder_l5.episodes_config(whole, 16)["ladder_loop"]["episode"]["block_problems"] == 64
    # ONE reader of its settings; its steps are its own, and only its two stages run them.
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["ladder_loop"]["l5"]' in path.read_text()] == ["gpu/ladder_l5.py"]
    others = set()
    for module in (ladder_round, ladder_l2, ladder_loop, ladder_ceiling, ladder_l3d2, ladder_l4, ladder_l4_rank, ladder_l4_rows, ladder_l4_rows2, ladder_l4_dose, ladder_l4b):
        others |= set(module.STEPS)
    assert not set(ladder_l5.STEPS) & others and all(name.startswith("ladder_l5_") for name in ladder_l5.STEPS)
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l5.STEPS for step in entry.STAGES[name])) == ["ladder_l5_deep", "ladder_l5_deep_smoke"]
    # ITS VARIABLES are named by its smoke stage, and the two that say WHERE A RUN IS by L6's smoke stage and L5n's too, which read this stage's smoke run by this stage's own
    # functions. THE SMOKE-ONLY ONE is its smoke stage's ALONE: no other stage of the package sets it and no other module reads it.
    names = lambda variable: sorted(name for name, own_ in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own_))      # noqa: E731
    assert names(RUN) == names(L4T2_RUN) == ["ladder_l5_deep_smoke", "ladder_l5_noise_smoke", "ladder_l6_cycle_smoke"] and names(SMOKE) == ["ladder_l5_deep_smoke"] and SMOKE == "RLVR_LEAN_LADDER_L5_SMOKE_PROBLEMS"
    assert sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if SMOKE in path.read_text()) == ["gpu/ladder_l5.py", "runner/entry.py"]
    assert "ladder_l5_deep_smoke" in names(MAP_RUN) and "ladder_l5_deep" not in names(MAP_RUN)
    # ONE BODY EACH: this stage's train and measure steps are L4t's own two bodies, and its report reads its models by L4t's own reader.
    assert ladder_l5.train_a_model is ladder_l4_rows.train_a_model and ladder_l5.measure_a_model is ladder_l4_rows.measure_a_model and ladder_l5.read_the_trained is ladder_l4_rows.read_the_trained
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l5.STEPS" in main and "ladder_l5," in main
