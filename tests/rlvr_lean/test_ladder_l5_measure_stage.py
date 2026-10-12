"""L5m, the stage end to end on what L5's FINISHED run, L4p's run and the FINISHED RUN OF THE STAGE THAT MADE THE
CANDIDATE stored: the refusals of its first step (the seed, the candidate the settings name, L5's run, `second_full` and
its stored rungs and samplings, the candidate's run, its record of the mean, the rows it was trained on); the candidate
MEASURED IN FULL from the adapter another run keeps (the rungs, G's two samplings, the held-aside set), its first
sampling measured again; the bar on the held-aside set, measured here or READ FROM A STORED RUN with the same setting;
the report's checks failing when they should; the two reads at the branches a one-problem world reaches; a candidate of
L5c; the smoke stage; the registration; and every run it reads left file for file as it was. No step trains.
Spec: docs/spec/ladder-loop.spec.md, "L5m: the best model, measured in full". The engine and Lean are scripted;
nothing touches a GPU or the network. The rules on hand-made rows are `test_ladder_l5_measure.py`."""

import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l2_stage import ARM  # noqa: E402
from test_ladder_l4_dose_stage import _stage_p  # noqa: E402 - L4p's own stage, as its tests run it
from test_ladder_l4_rows_r64 import _trainings as _trainings_of_the_chain  # noqa: E402 - the smoke runs before L5d's: a training records no learning rate
from test_ladder_l4_rows_stage import _all  # noqa: E402
from test_ladder_l4_stage import FIXTURE, _on_the_gpu  # noqa: E402
from test_ladder_l5_cont_stage import OF_L5D, OF_L5F, SUCCESSES, _rewritten, _stage5c  # noqa: E402 - L5c's own stage, whose models are candidates too
from test_ladder_l5_cont_stage import OWN_FILES as L5CS_FILES  # noqa: E402 - what a seed of L5c leaves, frozen there
from test_ladder_l5_cont_stage import cont  # noqa: E402, F401 - L5's, L4p's, L5d's and L5f's runs of the seed, each run to its end by its own stage
from test_ladder_l5_draws_stage import _l5ns_report, _stage5d  # noqa: E402
from test_ladder_l5_full_stage import OWN_FILES as L5FS_FILES  # noqa: E402 - what a seed of L5f leaves, frozen there
from test_ladder_l5_full_stage import _as_text, _changed, _file_rows, _stage5f  # noqa: E402
from test_ladder_l5_noise_stage import GOAL, NoiseSolver, _adapter_of, _trainings  # noqa: E402
from test_ladder_l5_stage import (  # noqa: E402, F401
    CONFIG,
    PACKAGE,
    SEEDS,
    DeepSolver,
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
from test_ladder_l6_stage import _smoke_world  # noqa: E402 - the smoke runs' world
from test_ladder_round import _run_stage  # noqa: E402 - L1's own stage, as its tests run it

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_cont import OF_THEIRS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_measure import (  # noqa: E402
    DOES_NOT_HOLD,
    DOES_NOT_HOLD_THERE,
    HOLDS_ON_FRESH,
    HOLDS_ON_THE_SET,
    NOT_NAMED,
    NOT_SHOWN_ON_THE_SET,
    SELECTION_ON_G,
    THE_BEST_MEASURED,
)
from rlvr_lean.domain.ladder_round.l5_noise import AVERAGE_OF_FOUR  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import (  # noqa: E402
    ladder_ceiling,
    ladder_l2,
    ladder_l3d2,
    ladder_l4,
    ladder_l4_dose,
    ladder_l4_rank,
    ladder_l4_rows,
    ladder_l4_rows2,
    ladder_l4_start,
    ladder_l5,
    ladder_l5_cont,
    ladder_l5_draws,
    ladder_l5_full,
    ladder_l5_measure,
    ladder_l5_noise,
    ladder_l5_stock,
    ladder_l6,
    ladder_loop,
    ladder_round,
    milestone2,
    pipeline,
)
from rlvr_lean.gpu.stand_in_engine import StandInOutput, StandInSample, stand_in_parameters  # noqa: E402
from rlvr_lean.infrastructure.adapter_files import compared  # noqa: E402

RUN, SMOKE = ladder_l5_measure.RUN_VARIABLE, ladder_l5_measure.SMOKE_VARIABLE
L4P_RUN, L5F_RUN, L5C_RUN, L6_RUN = ladder_l6.L4P_RUN_VARIABLE, ladder_l5_full.RUN_VARIABLE, ladder_l5_cont.RUN_VARIABLE, ladder_l6.RUN_VARIABLE
STEPS = ["ladder_l5m_prepare", "ladder_l5m_measure_candidate", "ladder_l5m_held_aside_candidate", "ladder_l5m_held_aside_second_full", "ladder_l5m_report"]
PREPARE, MEASURE, ASIDE_CANDIDATE, ASIDE_BAR, REPORT = STEPS
STORED = ["l4_stored_candidate_reach.jsonl", *(f"l4_stored_{model}_{part}.jsonl" for model in ("pre_r64", "second_full") for part in ("rungs", "reach", "more"))]
OWN_FILES = sorted([*(f"{step}.done.json" for step in STEPS), *STORED, "problems.jsonl", "report_ladder_l5_measure.json", "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl",
                    "l5_held_aside_problems.jsonl", "l5m_trained_rows.jsonl"])          # NO adapter: nothing is trained here
REPORT_KEYS = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "candidate", "bar", "first_start", "rank", "branches", "inconclusive",
               "failed_checks", "can_this_run_be_read", "l4ps_own_checks_of_the_bar", "of_this_seed", "the_held_aside_set", "fresh_attempts", "the_measurement_again", "beside",
               "timed_out_in_lean", "what_follows", "the_candidate", "the_bar", "the_search", "the_measurement", "the_held_aside", "the_bars_held_aside", "smoke",
               "what_it_cannot_say", "attempts", "not_to_be_read", "not_built", "not_measured", "adapters"]
THE_ADAPTERS, THE_ONE, THE_SETS, NO_BARRED, ANSWERED = ("the_two_adapters_are_there_at_the_rank_and_are_not_each_others", "the_candidate_is_the_adapter_its_stage_measured",
                                                         "every_set_is_the_settings", "no_barred_row_in_what_the_candidate_was_trained_on", "lean_answered")
HELD_ATTEMPTS, FRESH_ATTEMPTS = 16, 48          # this world's attempts at a held-aside problem, and at the goal problem in its SECOND sampling (the real run's: 16 and 61)
ON_THE_SET, ON_FRESH = (9, 5), 24               # of 16, the candidate's and then the bar's successes on the held-aside problem; of 48, the candidate's fresh ones (the bar's stored: 18)
OF_THE_BAR = {"reach": 12, "more": 18}          # `second_full`'s stored successes at the goal problem, as L4p's run in this world left them (of 32 and of 48)
NOTHING_READ = {"the_held_aside_set": INCONCLUSIVE, "fresh_attempts": INCONCLUSIVE, "what_follows": NOT_READ}


class MeasureSolver:
    """`other` (the world's solvers) for every sampling but the two this stage's reads are made on: the k-th measurement
    of the held-aside set (16 samples a prompt) closes the goal in its first `held[k]` attempts, and a second sampling of
    the goal set (48 samples a prompt) in its first `fresh` (left to `other` when None). `asked`: every prompt of every
    call, with its samples and its seed."""

    def __init__(self, other, held=ON_THE_SET, fresh=ON_FRESH):
        self.other, self.held, self.fresh, self.on_the_set, self.asked = other, list(held), fresh, 0, []

    def generate(self, prompts, parameters, lora_request=None):
        self.asked += [(re.findall(r"theorem (\S+)", prompt)[-1], parameters.n, parameters.seed) for prompt in prompts]
        closes = None
        if parameters.n == HELD_ATTEMPTS:
            closes, self.on_the_set = self.held[self.on_the_set % len(self.held)], self.on_the_set + 1
        elif parameters.n == FRESH_ATTEMPTS:
            closes = self.fresh
        if closes is None:
            return self.other.generate(prompts, parameters, lora_request)
        return [StandInOutput([StandInSample("  done\n" if index < closes else "  bad_step\n  done\n", [0] * 4, "stop") for index in range(parameters.n)]) for _ in prompts]


def _stage5m(config, monkeypatch, until=None, stage_name="ladder_l5_measure", on_the_gpu=()):
    """The stage's steps in its order. No step trains; `on_the_gpu`: the steps run by their GPU path (the first step
    then reads the adapters' files)."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l5_measure.STEPS:
            continue
        run = ladder_l5_measure.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if step in on_the_gpu else run(config)
        if step == until:
            break
    return summaries


def _nothing_written(runs):
    return not [path for path in runs.iterdir() if path.name.startswith("ladder_l5_measure") and any(path.iterdir())]


def _engines(monkeypatch, solver, kits):
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"]}),
        (solver, stand_in_parameters))[1]))


@pytest.fixture
def best(cont, monkeypatch):  # noqa: F811
    """L5c's world (L5's finished search, L4p's run with `second_full`, L5d's and L5f's OWN STAGES run to their end at
    seed 3: L5f's `published` closes the goal in 8 of 32) and this stage's task of SEED 3, which carries NO variable.
    The candidate is the settings': L5f's `published`. Here G is ONE problem and the held-aside set ONE. `solver.asked`:
    every prompt this stage asks for; the candidate closes the held-aside problem in 9 of 16 and the bar in 5, and the
    goal problem in 24 of its 48 fresh attempts (the bar's stored: 18); its first sampling, measured again, in 8 of 32."""
    for name in (RUN, SMOKE, L6_RUN):
        monkeypatch.delenv(name, raising=False)
    kits, solver = [], MeasureSolver(cont.solver)
    _engines(monkeypatch, solver, kits)
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l5_measure"])        # THE STAGE names no variable
    config = cont.config
    assert config["ladder_loop"]["l5_measure"] == {"candidate": {"of": "l5_full", "seed": 3, "model": "published"}, "bar": "second_full"}
    aside = [row["problem_id"] for row in _file_rows(cont.l5 / "l5_held_aside_problems.jsonl")]
    return SimpleNamespace(config=config, cont=cont, deep=cont.deep, kits=kits, solver=solver, runs=cont.runs, l5=cont.l5, l5f=cont.l5f, l5d=cont.l5d, l4p=cont.l4p, aside=aside,
                           candidate=cont.l5f / "adapters" / "published", bar=cont.l4p / "adapters" / "second_full",
                           store=lambda: ladder_l5_measure._store(config, ladder_l5_measure.the_settings(config)))


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_measures_the_candidate_in_full_from_the_adapter_l5fs_run_keeps_the_bar_on_the_held_aside_set_and_makes_the_two_reads(best, monkeypatch):
    config, kits, solver, deep_ = best.config, best.kits, best.solver, best.deep
    read_only = {path: _all(path) for path in (best.l5, best.l5f, best.l5d, best.l4p, deep_.map, deep_.l4t2, deep_.rank)}
    summaries = _stage5m(config, monkeypatch)
    store = best.store()
    # THE RUN DIRECTORY IS ITS OWN, named by its candidate, beside the runs it reads, which were ONLY READ: file for file. NO ADAPTER IS MADE.
    assert store.root.name == "ladder_l5_measure_l5_full_published_seed3" and store.root.parent == best.l5.parent and list(summaries) == STEPS == list(ladder_l5_measure.STEPS)
    assert {path: _all(path) for path in read_only} == read_only
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES and not (store.root / "adapters").exists()
    assert sorted(name for name in _all(best.l5f) if not name.startswith("episodes_")) == L5FS_FILES

    # ---- the first step: the candidate and what was held of it, the bar, the search, how each set is measured
    read = summaries[PREPARE]
    pre_r64, aside = deep_.rank / "adapters" / "pre_r64", best.aside
    assert (read["stage"], read["label"], read["check"], read["seed"], read["bar"], read["first_start"], read["smoke"]) == (
        "l4", "pretrained on published proofs", "L5m", 3, "second_full", "pre_r64", None)
    assert (read["bar_adapter"], read["bar_run"], read["first_start_adapter"], read["first_start_run"]) == (str(best.bar), str(best.l4p), str(pre_r64), str(deep_.rank))
    assert read["start_recipe"]["rank"] == 64 and read["start_recipe"]["alpha"] == 128 and read["trainings"]["run"] is True
    of_it, averaged = read["the_candidate"], best.cont.theirs["ladder_l5_full_average_published"]
    assert (of_it["of"], of_it["made_by"], of_it["seed"], of_it["model"], of_it["run"], of_it["adapter"]) == ("l5_full", "L5f", 3, "published", str(best.l5f), str(best.candidate))
    assert of_it["the_mean_of"] == list(OF_THEIRS) and of_it["kept_by_another_run"] == {} and "`ladder_l5_full` for seed 3" in of_it["task"]
    assert of_it["held"] == {"adapter": str(best.candidate), "read_now": None,                                   # no GPU box here: what L5f's own steps recorded
                             "recorded": {"adapter_saved": averaged["adapter_saved"], "numbers": averaged["against_the_start_adapter"]["elements"],
                                          "numbers_not_its_start_adapters": averaged["against_the_start_adapter"]["elements_changed"]}}
    record = of_it["the_record_of_the_mean"]
    assert (record["passes"], record["is_the_mean"], record["at_its_rank"], record["one_adapter"], record["one_of_its_models_in_a_smoke_run"]) == (True, True, True, True, False)
    assert (record["averaged_as"], record["measured_there_from"], record["of"], record["elements_not_the_mean"]) == (str(best.candidate), str(best.candidate), list(OF_THEIRS), 0)
    assert of_it["at_the_first_step"] == {"seed": 3, "start": "second_full", "recipe": "mean_constant", "ok": True, "inconclusive": False, "failed_checks": [],
                                          "the_measurement": {"sampling": "reach", "episodes": 32, "sampling_seed": 1001, "problems": 1}}
    assert read["the_bar"]["summary"] == best.cont.theirs["ladder_l5_full_prepare"]["the_start_model"]                # `second_full`, read as L5f read it
    assert (read["the_search"]["run"], read["the_search"]["seed"], read["the_search"]["inconclusive"], read["the_search"]["held_aside_problems"]) == (str(best.l5), 0, False, 1)
    of_the_measurement = read["the_measurement"]
    assert of_the_measurement["first_sampling"] == {"sampling": "reach", "episodes": 32, "sampling_seed": 1001, "problems": 1}
    assert of_the_measurement["fresh_sampling"] == {"sampling": "more", "episodes": FRESH_ATTEMPTS, "sampling_seed": 1020}
    assert of_the_measurement["sets"] == {"rungs": "l4_rungs_candidate", "reach": "l4_reach_candidate", "more": "l4_more_candidate", "held_aside": "l5m_held_aside_candidate"}
    assert of_the_measurement["stored"] == {**{part: f"l4_stored_second_full_{part}.jsonl" for part in ("rungs", "reach", "more")},
                                            "first_start": {part: f"l4_stored_pre_r64_{part}.jsonl" for part in ("rungs", "reach", "more")},
                                            "the_candidates_first_sampling": "l4_stored_candidate_reach.jsonl"}
    of_the_set = read["the_held_aside"]
    assert (of_the_set["problems"], of_the_set["episodes_each"], of_the_set["sampling_seed"], of_the_set["sampled_here"]) == (1, HELD_ATTEMPTS, 1051, ["candidate", "second_full"])
    assert of_the_set["sets"] == {"candidate": "l5m_held_aside_candidate", "second_full": "l5m_held_aside_second_full"}
    # THE BAR ON THE HELD-ASIDE SET: no stored measurement of it with the same setting is on the box (L6's run is where one would be), so it is MEASURED HERE.
    of_the_bars = read["the_bars_held_aside"]
    assert (of_the_bars["read_from_a_stored_run"], of_the_bars["looked_in"], of_the_bars["step_there"], of_the_bars["set_there"], of_the_bars["file"], of_the_bars["set"]) == (
        False, str(best.runs / "ladder_l6_cycle_pre_r64_seed0"), "ladder_l6_held_aside_second_full", "l6_held_aside_second_full", None, "l5m_held_aside_second_full")
    assert "MEASURED HERE from the adapter L4p's run keeps" in of_the_bars["what"] and "ladder_l6_cycle_pre_r64_seed0 does not hold" in of_the_bars["why_not"]
    # THE STORED SETS, each with the run that holds it and what its own step recorded: the bar's three in L4p's run, the candidate's first sampling in L5f's.
    of_stored = read["the_stored_sets"]
    assert of_stored["the_bar"] == {
        "rungs": {"file": "l4_stored_second_full_rungs.jsonl", "run": str(best.l4p), "set": "l4_rungs_second_full", "sampling_seed": 1002, "episodes_each": 8},
        "reach": {"file": "l4_stored_second_full_reach.jsonl", "run": str(best.l4p), "set": "l4_reach_second_full", "sampling_seed": 1001, "episodes_each": 32},
        "more": {"file": "l4_stored_second_full_more.jsonl", "run": str(best.l4p), "set": "l4_more_second_full", "sampling_seed": 1020, "episodes_each": FRESH_ATTEMPTS}}
    assert of_stored["the_candidates_first_sampling"] == {"file": "l4_stored_candidate_reach.jsonl", "run": str(best.l5f), "set": "l4_reach_published", "sampling_seed": 1001,
                                                           "episodes_each": 32} and of_stored["the_bars_held_aside"] is None
    for part in ("rungs", "reach", "more"):
        assert store.read_rows(f"l4_stored_second_full_{part}.jsonl") == _file_rows(best.l4p / f"episodes_l4_{part}_second_full_problems.jsonl")
    stored_first = store.read_rows("l4_stored_candidate_reach.jsonl")
    assert stored_first == _file_rows(best.l5f / "episodes_l4_reach_published_problems.jsonl") and [(row["problem_id"], row["resolved"]) for row in stored_first] == [(GOAL, OF_L5F[0])]
    assert {part: sum(row["resolved"] for row in store.read_rows(f"l4_stored_second_full_{part}.jsonl")) for part in OF_THE_BAR} == OF_THE_BAR
    # THE PROBLEMS OF EVERY SET THIS RUN SAMPLES, and of no other: the candidate's four and the bar's held-aside set.
    by_set = {}
    for row in store.read_rows("problems.jsonl"):
        by_set.setdefault(row["set"], set()).add(row["problem_id"])
    assert by_set["l4_reach_candidate"] == by_set["l4_more_candidate"] == {GOAL} and by_set["l5m_held_aside_candidate"] == by_set["l5m_held_aside_second_full"] == set(aside)
    assert sorted(by_set) == ["l4_more_candidate", "l4_reach_candidate", "l4_rungs_candidate", "l5m_held_aside_candidate", "l5m_held_aside_second_full"] and len(by_set["l4_rungs_candidate"]) == 5
    # THE ROWS THE CANDIDATE WAS TRAINED ON, from L5f's own record of its four trainings: by id and place, NEVER with text.
    trained = store.read_rows("l5m_trained_rows.jsonl")
    assert of_it["trained_rows"]["rows"] == len(trained) == 16 and of_it["trained_rows"]["trainings"] == 4 and of_it["trained_rows"]["file"] == "l5m_trained_rows.jsonl"
    assert all(list(row) == ["training", "id", "problem_id", "origin", "row_of_the_file"] and row["origin"] == "pretraining" for row in trained)
    assert not any(field in row for row in trained for field in TEXT_FIELDS)
    for name in OF_THEIRS:
        loss = json.loads((best.l5f / f"l5_full_loss_{name}.json").read_text())
        assert [row["id"] for row in trained if row["training"] == name] == loss["rows_trained"] == [row["id"] for row in _file_rows(best.l5f / f"l5_full_training_{name}.jsonl")]
    assert of_it["trained_rows"]["problems_of_g_or_of_the_held_aside_set"] == {"problems_of_the_goal_set": 0, "problems_of_the_held_aside_set": 0}

    # ---- THE CANDIDATE IN FULL: five rung problems (8 episodes), the goal problem twice (ITS FIRST SAMPLING AGAIN, then the fresh attempts), the held-aside problem; then the bar there
    assert [asked[1:] for asked in solver.asked[:5]] == [(8, 1002)] * 5
    assert solver.asked[5:] == [(GOAL, 32, 1001), (GOAL, FRESH_ATTEMPTS, 1020), (aside[0], HELD_ATTEMPTS, 1051), (aside[0], HELD_ATTEMPTS, 1051)]
    assert [(kit["max_lora_rank"], kit["rank"], kit["temperature"]) for kit in kits] == [(64, 64, 1.0)] * 5                 # every engine at the start adapter's rank
    measured = summaries[MEASURE]
    assert (measured["model"], measured["adapter"], measured["candidate"], measured["check"]) == ("candidate", str(best.candidate), {"of": "l5_full", "seed": 3, "model": "published"}, "L5m")
    assert (measured["rungs"]["set"], measured["rungs"]["episodes_each"], measured["rungs"]["sampling_seed"], measured["rungs"]["problems"]) == ("l4_rungs_candidate", 8, 1002, 5)
    assert [(own["set"], own["episodes_each"], own["sampling_seed"]) for own in measured["goal"].values()] == [("l4_reach_candidate", 32, 1001), ("l4_more_candidate", FRESH_ATTEMPTS, 1020)]
    on_the_set = {who: summaries[step] for who, step in (("candidate", ASIDE_CANDIDATE), ("second_full", ASIDE_BAR))}
    assert [(own["model"], own["adapter"], own["set"], own["episodes_each"], own["sampling_seed"], own["successes"]) for own in on_the_set.values()] == [
        ("candidate", str(best.candidate), "l5m_held_aside_candidate", HELD_ATTEMPTS, 1051, ON_THE_SET[0]),
        ("second_full", str(best.bar), "l5m_held_aside_second_full", HELD_ATTEMPTS, 1051, ON_THE_SET[1])]

    # ---- THE REPORT: every check passes; the two reads, each the candidate minus the bar over problems; what stands beside them
    report = summaries[REPORT]
    assert list(report) == REPORT_KEYS and report == json.loads(store.path("report_ladder_l5_measure.json").read_text())
    assert (report["ok"], report["inconclusive"], report["failed_checks"], report["not_to_be_read"], report["seed"], report["rank"]) == (True, False, [], [], 3, 64)
    assert (report["candidate"], report["bar"], report["first_start"]) == ({"of": "l5_full", "seed": 3, "model": "published"}, "second_full", "pre_r64")
    checks = report["can_this_run_be_read"]
    assert list(checks) == ["what", THE_ADAPTERS, THE_ONE, THE_SETS, NO_BARRED, ANSWERED] and all(checks[name]["passes"] is True for name in list(checks)[1:])
    assert (checks[THE_ADAPTERS]["the_bars_held_aside_set_was_sampled_from"], checks[THE_ADAPTERS]["the_bar_was_served"]) == (str(best.bar), True)
    assert (checks[THE_ONE]["served_here_from"], checks[THE_ONE]["its_report_now"], checks[THE_ONE]["why_not"]) == ([str(best.candidate)] * 2, True, None)
    assert list(checks[THE_SETS]["sets"]) == ["the candidate (rungs)", "the candidate (reach)", "the candidate (more)", "the candidate (held-aside)", "`second_full` (rungs, stored)",
                                              "`second_full` (reach, stored)", "`second_full` (more, stored)", "`second_full` (held-aside)", "the candidate's stored first sampling"]
    assert (checks[NO_BARRED]["rows"], checks[NO_BARRED]["trainings"]) == (16, list(OF_THEIRS)) and list(checks[ANSWERED]["share_without_an_answer"]) == [
        "the candidate (rungs)", "the candidate (reach)", "the candidate (more)", "the candidate (held-aside)", "`second_full` (held-aside)"]
    primary, second, again = report["the_held_aside_set"], report["fresh_attempts"], report["the_measurement_again"]
    assert (primary["name"], primary["difference"]["mean"], primary["problems"], primary["attempts_a_problem"]) == (HOLDS_ON_THE_SET, (ON_THE_SET[0] - ON_THE_SET[1]) / 16, 1, 16)
    assert (second["name"], second["difference"]["mean"], second["problems"], second["attempts_a_problem"]) == (HOLDS_ON_FRESH, (ON_FRESH - OF_THE_BAR["more"]) / 48, 1, 48)
    assert (again["mean"], again["per_1000"], again["per_1000_of_the_base"]) == (0.0, 250.0, 250.0)                # the same adapter, attempts and seed: here, the same 8 of 32
    assert report["branches"] == {"the_held_aside_set": HOLDS_ON_THE_SET, "fresh_attempts": HOLDS_ON_FRESH, "what_follows": THE_BEST_MEASURED} and report["what_follows"]["name"] == THE_BEST_MEASURED
    assert report["of_this_seed"] == {"seed": 3, "candidate": report["candidate"], "bar": "second_full", "ok": True, "inconclusive": False, "failed_checks": [],
                                      "the_held_aside_set": primary["difference"], "fresh_attempts": second["difference"]}
    beside = report["beside"]
    assert list(beside) == ["what", "all_of_g", "first_sampling_measured_here", "g_again", "solved", "solved_reliably", "never_solved_by_the_first_start", "long", "rungs",
                            "held_aside_solved", "share_of_distinct_attempts"]
    assert (beside["all_of_g"]["attempts_a_problem"], beside["all_of_g"]["mean"]) == (80, round((SUCCESSES[0] + ON_FRESH - sum(OF_THE_BAR.values())) / 80, 5))
    assert beside["first_sampling_measured_here"]["mean"] == (SUCCESSES[0] - OF_THE_BAR["reach"]) / 32 and list(beside["rungs"]) == ["below", "in", "above"]
    assert (beside["solved"]["resolved_after"], beside["solved"]["resolved_before"], beside["held_aside_solved"]["resolved_after"]) == (1, 1, 1)
    assert beside["solved_reliably"] == {"what": "solved in half its attempts or more", "attempts_a_problem": 80, "candidate": 0, "bar": 0, "gained": 0, "lost": 0}       # 32 and 30 of 80
    assert report["timed_out_in_lean"]["apart"] == [] and report["timed_out_in_lean"]["checked_again"] is False and len(report["what_it_cannot_say"]) == 5
    assert report["adapters"]["kept"] is True and report["adapters"]["made_here"] == [] and report["smoke"] is None and report["the_bars_held_aside"] == of_the_bars
    lines = report["lines"]
    assert all(line.startswith("l4 (pretrained on published proofs): ") for line in lines) and "NO TRAINING" in lines[0] and "MEASURED HERE from its stored adapter" in lines[0]
    assert sum("CHECK, " in line and line.endswith(": PASS") for line in lines) == 5 and not any(line.endswith(": FAIL") for line in lines)
    assert any("its files were not read by the first step (no GPU box)" in line for line in lines)
    assert any("L4p's OWN CHECKS OF `second_full`, PRINTED AND DECIDING NOTHING" in line for line in lines)
    assert any(f"THE HELD-ASIDE SET (the primary): {HOLDS_ON_THE_SET}." in line and "562.5 against 312.5 per 1,000 over 1 problems; the interval is above zero" in line for line in lines)
    assert any(f"FRESH ATTEMPTS: {HOLDS_ON_FRESH}." in line and "on all of G over the 48 fresh attempts of the second sampling" in line for line in lines)
    assert any("THE MEASUREMENT AGAIN, naming nothing" in line for line in lines) and any("BESIDE, each rung (the easy rung is where L4r read a cost)" in line for line in lines)
    assert lines[-1].startswith(f"l4 (pretrained on published proofs): WHAT FOLLOWS: {THE_BEST_MEASURED}.")
    assert report["headline"].startswith("L5m (the best model, measured in full; pretrained on published proofs) seed 3, the candidate `published` of L5f against `second_full`: "
                                         f"THE HELD-ASIDE SET: {HOLDS_ON_THE_SET} (+0.25000 [+0.25000, +0.25000]")
    marked = store.done_summary(REPORT)
    assert (marked["headline"], marked["branches"], marked["measured"], marked["ok"], marked["candidate"]) == (report["headline"], report["branches"], ["candidate", "second_full"], True,
                                                                                                           report["candidate"])
    # EVERY STEP IS RESUMABLE: queued again, nothing is sampled again and the report is built again from what is stored.
    asked = len(solver.asked)
    again_ = _stage5m(config, monkeypatch)
    assert len(solver.asked) == asked and {step: again_[step] for step in STEPS[:-1]} == {step: summaries[step] for step in STEPS[:-1]} and again_[REPORT]["branches"] == report["branches"]
    assert {path: _all(path) for path in read_only} == read_only


@pytest.mark.parametrize("held, fresh, on_the_set, on_fresh, follows", [
    ((5, 9), 24, DOES_NOT_HOLD_THERE, HOLDS_ON_FRESH, SELECTION_ON_G),          # THE PRIMARY DECIDES: it does not hold on the held-aside set, whatever the fresh attempts say
    ((7, 7), 12, NOT_SHOWN_ON_THE_SET, DOES_NOT_HOLD, SELECTION_ON_G),          # an interval that ends at zero spans it
    ((9, 5), None, HOLDS_ON_THE_SET, "NOT SHOWN", NOT_NAMED),                   # it holds on the set and not on fresh attempts: THE SECTION NAMES NOTHING, and that is said
])
def test_the_two_reads_at_the_branches_a_one_problem_world_reaches(best, monkeypatch, held, fresh, on_the_set, on_fresh, follows):
    best.solver.held, best.solver.fresh = list(held), fresh
    report = _stage5m(best.config, monkeypatch)[REPORT]
    assert report["inconclusive"] is False and report["branches"] == {"the_held_aside_set": on_the_set, "fresh_attempts": on_fresh, "what_follows": follows}
    assert report["the_held_aside_set"]["difference"]["mean"] == (held[0] - held[1]) / 16
    assert report["fresh_attempts"]["difference"]["mean"] == ((OF_THE_BAR["more"] if fresh is None else fresh) - OF_THE_BAR["more"]) / 48
    assert report["lines"][-1].startswith(f"l4 (pretrained on published proofs): WHAT FOLLOWS: {follows}.") and f". What follows: {follows}" in report["headline"]
    if follows == NOT_NAMED:
        assert "nothing is claimed of it, and it needs a decision" in report["what_follows"]["reason"] and report["fresh_attempts"]["resolves"] == 0.0
    if follows == SELECTION_ON_G:
        assert "selection on G until a further seed shows otherwise" in report["what_follows"]["reason"]


def test_a_failed_check_makes_the_run_inconclusive_and_nothing_else_is_said(best, monkeypatch):
    config = best.config
    report = _stage5m(config, monkeypatch)[REPORT]
    store = best.store()
    assert report["inconclusive"] is False

    def failing(path, changed):
        kept = _changed(path, changed)
        failed = ladder_l5_measure.ladder_l5m_report(config)
        Path(path).write_text(kept)
        return failed

    marked = lambda **changed: (lambda text: json.dumps({**json.loads(text), **changed}))      # noqa: E731
    prepared = store.path(f"{PREPARE}.done.json")

    def of_the_candidate(**changed):
        return lambda text: json.dumps({**json.loads(text), "the_candidate": {**json.loads(text)["the_candidate"], **changed}})

    # THE CANDIDATE's ADAPTER IS, BY ITS OWN STAGE's RECORD, ITS START ADAPTER NUMBER FOR NUMBER: the first check fails. THE RUN IS INCONCLUSIVE, no read is made.
    held = json.loads(prepared.read_text())["the_candidate"]["held"]
    failed = failing(prepared, of_the_candidate(held={**held, "recorded": {**held["recorded"], "numbers_not_its_start_adapters": 0}}))
    assert failed["failed_checks"] == [THE_ADAPTERS] and failed["inconclusive"] is True and failed["ok"] is True and failed["branches"] == NOTHING_READ
    assert failed["can_this_run_be_read"][THE_ADAPTERS]["the_candidate"]["passes"] is False and failed["can_this_run_be_read"][THE_ADAPTERS]["the_bar"]["passes"] is True
    assert "difference" not in failed["the_held_aside_set"] and "difference" not in failed["fresh_attempts"] and failed["the_held_aside_set"]["name"] == INCONCLUSIVE
    assert list(failed["beside"]) == list(failed["the_measurement_again"]) == ["measured_and_not_read"]
    assert failed["of_this_seed"]["the_held_aside_set"] is None and failed["of_this_seed"]["fresh_attempts"] is None and failed["of_this_seed"]["inconclusive"] is True
    assert failed["lines"][-1].endswith(f"THIS RUN IS INCONCLUSIVE: {THE_ADAPTERS}. Nothing else is said of it") and f"THIS RUN IS INCONCLUSIVE ({THE_ADAPTERS})" in failed["headline"]
    assert not any("THE HELD-ASIDE SET (the primary)" in line or "WHAT FOLLOWS" in line or "FRESH ATTEMPTS:" in line or "BESIDE" in line for line in failed["lines"])
    assert any("CHECK, the candidate's adapter and `second_full`'s are there" in line and line.endswith(": FAIL") for line in failed["lines"])
    assert store.done_summary(REPORT)["branches"] == NOTHING_READ
    # THE BAR's HELD-ASIDE SET WAS SAMPLED FROM ANOTHER ADAPTER than the one L4p's run keeps.
    failed = failing(store.path(f"{ASIDE_BAR}.done.json"), marked(adapter=str(best.candidate)))
    assert failed["failed_checks"] == [THE_ADAPTERS] and failed["can_this_run_be_read"][THE_ADAPTERS]["the_bar_was_served"] is False
    # THE CANDIDATE's RUN, AS IT IS NOW: its report gone; INCONCLUSIVE; its record of the mean no longer the mean. What is measured here is then not shown to be the model it measured.
    for name in ("ladder_l5_full_report.done.json", "report_ladder_l5_full.json"):
        (best.l5f / name).rename(best.l5f / f"kept_{name}")
    gone = ladder_l5_measure.ladder_l5m_report(config)
    for name in ("ladder_l5_full_report.done.json", "report_ladder_l5_full.json"):
        (best.l5f / f"kept_{name}").rename(best.l5f / name)
    check = gone["can_this_run_be_read"][THE_ONE]
    assert gone["failed_checks"] == [THE_ONE] and check["its_report_now"] is False and "L5f's run of this seed is not on the box, or not finished" in check["why_not"]
    assert any("CHECK, the candidate is the adapter L5f measured" in line and "its report now: NOT (" in line and line.endswith(": FAIL") for line in gone["lines"])
    of_the_seed = lambda **changed: (lambda text: json.dumps({**json.loads(text), "of_this_seed": {**json.loads(text)["of_this_seed"], **changed}}))      # noqa: E731
    failed = failing(best.l5f / "report_ladder_l5_full.json", of_the_seed(inconclusive=True, failed_checks=["`deep`: lean_answered"]))
    assert failed["failed_checks"] == [THE_ONE] and failed["can_this_run_be_read"][THE_ONE]["why_not"] == "it reads INCONCLUSIVE (`deep`: lean_answered)"
    averaging = best.l5f / "ladder_l5_full_average_published.done.json"
    failed = failing(averaging, lambda text: json.dumps({**json.loads(text), "read_back": {**json.loads(text)["read_back"], "is_the_mean": False, "passes": False}}))
    assert failed["failed_checks"] == [THE_ONE] and (failed["can_this_run_be_read"][THE_ONE]["is_the_mean"], failed["can_this_run_be_read"][THE_ONE]["its_report_now"]) == (False, True)
    # AN AVERAGED ADAPTER THAT IS ONE OF ITS MEMBERS fails in a real run, whatever that stage recorded of a smoke run.
    failed = failing(averaging, lambda text: json.dumps({**json.loads(text), "one_of_its_models_in_a_smoke_run": True,
                                                         "read_back": {**json.loads(text)["read_back"], "is_none_of_them": False, "passes": False}}))
    assert failed["failed_checks"] == [THE_ONE] and failed["can_this_run_be_read"][THE_ONE]["one_of_its_models_in_a_smoke_run"] is False
    # A MEASUREMENT OF THIS RUN SAMPLED FROM ANOTHER ADAPTER than the candidate's: the rungs and G; the held-aside set.
    for step in (MEASURE, ASIDE_CANDIDATE):
        failed = failing(store.path(f"{step}.done.json"), marked(adapter=str(best.l5f / "adapters" / "published_2")))
        assert failed["failed_checks"] == [THE_ONE] and failed["can_this_run_be_read"][THE_ONE]["served_here"] is False
    # A SET THAT IS NOT THE SETTING'S. A stored one, as this run copied it (half the attempts a problem): the read on it does not pair, and nothing crashes.
    halved = lambda text: _as_text([{**row, "episodes": row["episodes"] // 2} for row in map(json.loads, text.splitlines())])      # noqa: E731
    failed = failing(store.path("l4_stored_second_full_more.jsonl"), halved)
    assert failed["failed_checks"] == [THE_SETS] and failed["can_this_run_be_read"][THE_SETS]["not_the_settings"] == ["`second_full` (more, stored)"] and failed["branches"] == NOTHING_READ
    failed = failing(store.path("l4_stored_candidate_reach.jsonl"), halved)
    assert failed["failed_checks"] == [THE_SETS] and failed["can_this_run_be_read"][THE_SETS]["not_the_settings"] == ["the candidate's stored first sampling"]
    # ... one this run sampled, whose own step recorded another sampling seed; and the held-aside set with another number of attempts.
    failed = failing(store.path("episodes_l4_more_candidate.done.json"), marked(sampling_seed=1021))
    assert failed["failed_checks"] == [THE_SETS] and failed["can_this_run_be_read"][THE_SETS]["not_the_settings"] == ["the candidate (more)"]
    assert failed["can_this_run_be_read"][THE_SETS]["sets"]["the candidate (more)"]["recorded_sampling_seed"] == 1021
    failed = failing(store.path("episodes_l5m_held_aside_second_full.done.json"), marked(episodes_each=8))
    assert failed["failed_checks"] == [THE_SETS] and failed["can_this_run_be_read"][THE_SETS]["not_the_settings"] == ["`second_full` (held-aside)"]
    # ... and what the first step recorded of a stored set (the bar's rungs, sampled with another seed).
    stored_sets = json.loads(prepared.read_text())["the_stored_sets"]
    failed = failing(prepared, marked(the_stored_sets={**stored_sets, "the_bar": {**stored_sets["the_bar"], "rungs": {**stored_sets["the_bar"]["rungs"], "sampling_seed": 7}}}))
    assert failed["failed_checks"] == [THE_SETS] and failed["can_this_run_be_read"][THE_SETS]["not_the_settings"] == ["`second_full` (rungs, stored)"]
    # A ROW THE CANDIDATE WAS TRAINED ON THAT IS A GOAL PROBLEM (held out), and one that is a held-aside problem.
    trained = store.path("l5m_trained_rows.jsonl")
    for barred, key in ((GOAL, "problems_of_the_goal_set"), (best.aside[0], "problems_of_the_held_aside_set")):
        failed = failing(trained, lambda text: _as_text([{**row, "problem_id": barred if index == 0 else row["problem_id"]} for index, row in enumerate(map(json.loads, text.splitlines()))]))
        assert failed["failed_checks"] == [NO_BARRED] and failed["can_this_run_be_read"][NO_BARRED]["of_g_or_of_the_held_aside_set"][key] == 1
        assert any("CHECK, no row the candidate was trained on is barred" in line and line.endswith(": FAIL") for line in failed["lines"])
    # LEAN DID NOT ANSWER a set this run sampled: INCONCLUSIVE, and the report is NOT TO BE READ (its step fails; the task queued again samples that set again).
    rows_of = store.path("episodes_l5m_held_aside_candidate_problems.jsonl")
    kept = _rewritten(rows_of, attempts_without_an_answer=HELD_ATTEMPTS)
    failed = ladder_l5_measure.ladder_l5m_report(config)
    rows_of.write_text(kept)
    assert failed["failed_checks"] == [ANSWERED] and failed["ok"] is False and failed["not_to_be_read"] == ["the candidate (held-aside)"] and failed["branches"] == NOTHING_READ
    assert failed["lines"][-1].startswith("l4 (pretrained on published proofs): NOT TO BE READ: too many attempts without a verdict from Lean in the candidate (held-aside)")
    assert "NOT TO BE READ" in failed["headline"] and store.done_summary(REPORT)["ok"] is False
    # ... and with everything as it was the run reads again.
    assert ladder_l5_measure.ladder_l5m_report(config)["branches"] == report["branches"]
    # A RUN KEEPS ITS CANDIDATE: the setting names another one now, in the run directory this run was prepared in.
    monkeypatch.setenv(RUN, store.root.name)
    config["ladder_loop"]["l5_measure"]["candidate"]["model"] = "shallow"
    for step in (ladder_l5_measure.ladder_l5m_measure, ladder_l5_measure.STEPS[ASIDE_CANDIDATE], ladder_l5_measure.STEPS[ASIDE_BAR], ladder_l5_measure.ladder_l5m_report):
        with pytest.raises(RuntimeError, match=r"was prepared for the candidate \('l5_full', 3, 'published'\) and ladder_loop.l5_measure.candidate now names \('l5_full', 3, "
                                               r"'shallow'\): a run keeps its candidate"):
            step(config)
    monkeypatch.delenv(RUN)
    config["ladder_loop"]["l5_measure"]["candidate"]["model"] = "published"
    # A STEP NEEDS THE ONES BEFORE IT: the report with the candidate's held-aside set not measured; every step with the first one not done.
    store.path(f"{ASIDE_CANDIDATE}.done.json").unlink()
    with pytest.raises(RuntimeError, match="L5m's report needs the step ladder_l5m_held_aside_candidate of this run, which is not done"):
        ladder_l5_measure.ladder_l5m_report(config)
    monkeypatch.setenv(RUN, "ladder_l5_measure_not_prepared")
    with pytest.raises(RuntimeError, match="the measurement of the candidate needs the step ladder_l5m_prepare of this run, which is not done"):
        ladder_l5_measure.ladder_l5m_measure(config)
    with pytest.raises(RuntimeError, match="the measurement of `second_full` on the held-aside set needs the step ladder_l5m_prepare of this run, which is not done"):
        ladder_l5_measure.STEPS[ASIDE_BAR](config)


def test_as_on_the_gpu_the_first_step_reads_the_two_adapters_and_every_measurement_serves_its_models_adapter(best, monkeypatch):
    config, kits, runs = best.config, best.kits, best.runs
    prepare, theirs = ladder_l5_measure.ladder_l5m_prepare, best.l5f / "adapters"
    _adapter_of(best.bar, 64, 128, 9.0)             # the scripted trainings of this world leave adapters alike: as on the box, `second_full`'s numbers are its own
    # THE CANDIDATE's ADAPTER IS NOT THERE (its run recorded it, the files are gone): refused by the first step, naming the task that makes it, before anything is written.
    best.candidate.rename(theirs / "kept")
    with pytest.raises(RuntimeError, match="adapters/published is not there: the adapter `published` of L5f's run of seed 3 is the candidate, and that run keeps it. Queue the task of "
                                           "stage `ladder_l5_full` for seed 3 .* again .*nothing was written"):
        _on_the_gpu(monkeypatch, prepare, config)
    # IT IS `second_full`'s ADAPTER, number for number (whatever its run recorded of it): nothing would be read of it against the bar. Refused.
    shutil.copytree(best.bar, best.candidate)
    with pytest.raises(RuntimeError, match=r"the candidate's adapter at .*adapters/published, read back, has rank 64 and alpha 128 .* and 0 of its [\d,]+ numbers are not "
                                           r"`second_full`'s: .*Nothing was written"):
        _on_the_gpu(monkeypatch, prepare, config)
    # AT ANOTHER RANK than `pre_r64`'s: refused too.
    _adapter_of(best.candidate, 32, 64, 2.0)
    with pytest.raises(RuntimeError, match="read back, has rank 32 and alpha 64 with matrices of rank .32. .`pre_r64`: 64 and 128."):
        _on_the_gpu(monkeypatch, prepare, config)
    shutil.rmtree(best.candidate)
    (theirs / "kept").rename(best.candidate)
    assert _nothing_written(runs)
    # WITH BOTH AS THEIR RUNS LEFT THEM the first step reads the files: there, read back at the rank, numbers that are not the bar's. The report's line says so.
    with monkeypatch.context() as patched:
        for module in (ladder_ceiling, ladder_l5, ladder_l5_measure):          # each names the test of a GPU box as its own: the measuring body, L5's body, this stage's record
            patched.setattr(module, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        summaries = _stage5m(config, patched)
    now = summaries[PREPARE]["the_candidate"]["held"]["read_now"]
    assert (now["there"], now["passes"], now["read_back"]["rank"], now["read_back"]["alpha"]) == (True, True, 64, 128)
    assert now["against_the_bar"] == compared(best.candidate / "adapter_model.safetensors", best.bar / "adapter_model.safetensors") and now["against_the_bar"]["elements_changed"] > 0
    report = summaries[REPORT]
    assert report["inconclusive"] is False and report["stand_in_engine"] is False
    assert any("CHECK, the candidate's adapter and `second_full`'s are there" in line and "; read now at rank 64 and alpha 128, " in line and line.endswith(": PASS") for line in report["lines"])
    # AS ON THE GPU every measurement asks the model server for ITS model's adapter: the candidate's four sets for the one L5F's RUN keeps, the bar's for the one L4p's keeps.
    assert [kit["adapter"][2] for kit in kits] == [*[str(best.candidate)] * 4, str(best.bar)] and [kit["max_lora_rank"] for kit in kits] == [64] * 5
    assert [kit["adapter"][0] for kit in kits][-2:] == ["ladder_l5_measure_candidate", "ladder_l5_measure_second_full"]
    assert summaries[MEASURE]["stand_in_engine"] is False and not (best.store().root / "adapters").exists()


# ------------------------------------------------------------------------------------- the bar's held-aside set
def test_the_bar_on_the_held_aside_set_is_read_from_a_stored_run_with_the_same_setting_and_measured_here_otherwise(best, monkeypatch):
    config, runs, solver = best.config, best.runs, best.solver
    first = _stage5m(config, monkeypatch)
    own, of_l6 = best.store().root, runs / "ladder_l6_cycle_pre_r64_seed0"
    assert first[PREPARE]["the_bars_held_aside"]["looked_in"] == str(of_l6) and not of_l6.exists() and first[ASIDE_BAR]["successes"] == ON_THE_SET[1]
    # L6's RUN, AS ITS STEP WOULD LEAVE IT (hand-made: this run's own measurement of the bar under L6's names): the same adapter, problems, attempts and sampling seed.
    of_l6.mkdir()
    for path in own.glob("episodes_l5m_held_aside_second_full*"):
        shutil.copy(path, of_l6 / path.name.replace("l5m_held_aside_second_full", "l6_held_aside_second_full"))
    marker = of_l6 / "ladder_l6_held_aside_second_full.done.json"
    marker.write_text(json.dumps({"model": "second_full", "adapter": str(best.bar), "set": "l6_held_aside_second_full"}))
    stored, asked = _all(of_l6), len(solver.asked)
    monkeypatch.setenv(RUN, "ladder_l5_measure_beside")           # a run directory the task names: the first run is left as it is
    summaries = _stage5m(config, monkeypatch)
    store = best.store()
    assert store.root.name == "ladder_l5_measure_beside" and _all(of_l6) == stored and _all(own) == _all(runs / "ladder_l5_measure_l5_full_published_seed3")
    of_the_bars = summaries[PREPARE]["the_bars_held_aside"]
    assert (of_the_bars["read_from_a_stored_run"], of_the_bars["looked_in"], of_the_bars["file"], of_the_bars["set"], of_the_bars["why_not"]) == (
        True, str(of_l6), "l5m_stored_second_full_held_aside.jsonl", None, None) and "READ FROM A STORED RUN" in of_the_bars["what"]
    assert summaries[PREPARE]["the_held_aside"]["sampled_here"] == ["candidate"] and summaries[PREPARE]["the_held_aside"]["sets"] == {"candidate": "l5m_held_aside_candidate"}
    assert summaries[PREPARE]["the_stored_sets"]["the_bars_held_aside"] == {"file": "l5m_stored_second_full_held_aside.jsonl", "run": str(of_l6), "set": "l6_held_aside_second_full",
                                                                           "sampling_seed": 1051, "episodes_each": HELD_ATTEMPTS}
    assert store.read_rows("l5m_stored_second_full_held_aside.jsonl") == _file_rows(of_l6 / "episodes_l6_held_aside_second_full_problems.jsonl")
    # THE BAR's STEP MAKES NOTHING AND MARKS NOTHING, and says where its rows are read from: the held-aside problem is asked for ONCE in this run, by the candidate.
    made = summaries[ASIDE_BAR]
    assert (made["run"], made["check"], made["label"]) == (False, "L5m", "pretrained on published proofs") and f"READ from {of_l6}" in made["why"]
    assert not store.is_done(ASIDE_BAR) and not list(store.root.glob("episodes_l5m_held_aside_second_full*"))
    assert "l5m_held_aside_second_full" not in {row["set"] for row in store.read_rows("problems.jsonl")}
    assert [asked_[1] for asked_ in solver.asked[asked:]].count(HELD_ATTEMPTS) == 1 and len(best.kits) == 5 + 4
    # THE REPORT reads the stored rows: the same read as the run that measured the bar itself, and it says where they are from.
    report = summaries[REPORT]
    assert report["inconclusive"] is False and report["branches"] == first[REPORT]["branches"] and report["the_held_aside_set"]["difference"] == first[REPORT]["the_held_aside_set"]["difference"]
    checks = report["can_this_run_be_read"]
    assert (checks[THE_ADAPTERS]["the_bar_was_served"], checks[THE_ADAPTERS]["the_bars_held_aside_set_was_sampled_from"]) == (True, None)
    assert "`second_full` (held-aside, stored)" in checks[THE_SETS]["sets"] and "`second_full` (held-aside)" not in checks[ANSWERED]["share_without_an_answer"]
    assert f"on the held-aside set it is READ FROM A STORED RUN ({of_l6})" in report["lines"][0] and store.done_summary(REPORT)["measured"] == ["candidate"]
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == sorted(
        [*(name for name in OWN_FILES if name != f"{ASIDE_BAR}.done.json"), "l5m_stored_second_full_held_aside.jsonl"])
    # THE STORED ROWS, as this run copied them, NO LONGER THE SETTING'S: the check of the sets fails.
    kept = _changed(store.path("l5m_stored_second_full_held_aside.jsonl"), lambda text: _as_text([{**row, "episodes": 8} for row in map(json.loads, text.splitlines())]))
    failed = ladder_l5_measure.ladder_l5m_report(config)
    store.path("l5m_stored_second_full_held_aside.jsonl").write_text(kept)
    assert failed["failed_checks"] == [THE_SETS] and failed["can_this_run_be_read"][THE_SETS]["not_the_settings"] == ["`second_full` (held-aside, stored)"]

    # NOT WITH THE SAME SETTING: NOTHING IS REFUSED, the bar is measured here, and the first step says why. Sampled from another adapter; with another sampling seed; on other problems.
    def looked(name, why):
        monkeypatch.setenv(RUN, name)
        found = _stage5m(config, monkeypatch, until=PREPARE)[PREPARE]["the_bars_held_aside"]
        assert (found["read_from_a_stored_run"], found["set"], found["file"]) == (False, "l5m_held_aside_second_full", None) and re.search(why, found["why_not"]), found["why_not"]

    kept = _changed(marker, lambda text: json.dumps({**json.loads(text), "adapter": str(best.candidate)}))
    looked("ladder_l5_measure_other_adapter", "sampled that set from .*adapters/published, and the bar's adapter is .*adapters/second_full")
    marker.write_text(kept)
    recorded = of_l6 / "episodes_l6_held_aside_second_full.done.json"
    kept = _changed(recorded, lambda text: json.dumps({**json.loads(text), "sampling_seed": 1052}))
    looked("ladder_l5_measure_other_seed", "measured `second_full` on the held-aside set with sampling seed 1052 and 16 episodes; this config gives 1051 and 16")
    recorded.write_text(kept)
    kept = _rewritten(of_l6 / "episodes_l6_held_aside_second_full_problems.jsonl", problem_id=GOAL)
    looked("ladder_l5_measure_other_problems", "measured `second_full` on the held-aside set on other problems")
    (of_l6 / "episodes_l6_held_aside_second_full_problems.jsonl").write_text(kept)
    marker.unlink()
    looked("ladder_l5_measure_no_marker", "ladder_l6_cycle_pre_r64_seed0 does not hold .'ladder_l6_held_aside_second_full.done.json'.")
    # THE RUN L6's OWN VARIABLE NAMES is where a stored measurement is looked for (a smoke run of L6 names one; this stage's smoke stage does not).
    monkeypatch.setenv(L6_RUN, "ladder_l6_cycle_smoke")
    monkeypatch.setenv(RUN, "ladder_l5_measure_named")
    assert _stage5m(config, monkeypatch, until=PREPARE)[PREPARE]["the_bars_held_aside"]["looked_in"] == str(runs / "ladder_l6_cycle_smoke")


# ----------------------------------------------------------------------------------------- a candidate of L5c
def test_a_model_of_l5c_is_a_candidate_by_two_lines_of_the_settings_and_its_single_training_is_refused(best, monkeypatch):
    config, solver = best.config, best.solver
    of_l5c = _stage5c(config, monkeypatch)
    l5c = best.cont.store().root
    assert of_l5c["ladder_l5_cont_report"]["inconclusive"] is False and l5c.name == "ladder_l5_cont_second_full_seed3"
    read_only = {path: _all(path) for path in (l5c, best.l5f, best.l4p, best.l5)}
    # THE TWO LINES: the stage that made it, and its name there. The seed stays the task's.
    config["ladder_loop"]["l5_measure"]["candidate"].update({"of": "l5_cont", "model": "wide"})
    asked = len(solver.asked)
    summaries = _stage5m(config, monkeypatch)
    store, wide = best.store(), l5c / "adapters" / "wide"
    assert store.root.name == "ladder_l5_measure_l5_cont_wide_seed3" and {path: _all(path) for path in read_only} == read_only
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES and sorted(name for name in _all(l5c) if not name.startswith("episodes_")) == L5CS_FILES
    of_it = summaries[PREPARE]["the_candidate"]
    assert (of_it["of"], of_it["made_by"], of_it["seed"], of_it["model"], of_it["run"], of_it["adapter"]) == ("l5_cont", "L5c", 3, "wide", str(l5c), str(wide))
    # `wide` IS THE MEAN OF EIGHT: L5f's four trainings, kept by L5F's run, and L5c's own four. What it was trained on is read from each training's own record, where it lives.
    own_four = [f"wide_{order}" for order in (5, 6, 7, 8)]
    assert of_it["the_mean_of"] == [*OF_THEIRS, *own_four] and of_it["kept_by_another_run"] == {name: str(best.l5f) for name in OF_THEIRS} and "`ladder_l5_cont` for seed 3" in of_it["task"]
    assert of_it["the_record_of_the_mean"]["passes"] is True and of_it["the_record_of_the_mean"]["of"] == [*OF_THEIRS, *own_four]
    trained = store.read_rows("l5m_trained_rows.jsonl")
    assert of_it["trained_rows"]["rows"] == len(trained) == 32 and of_it["trained_rows"]["trainings"] == 8 and not any(field in row for row in trained for field in TEXT_FIELDS)
    for name, directory, files in (*((name, best.l5f, "l5_full") for name in OF_THEIRS), *((name, l5c, "l5_cont") for name in own_four)):
        assert [row["id"] for row in trained if row["training"] == name] == json.loads((directory / f"{files}_loss_{name}.json").read_text())["rows_trained"]
    # ITS STORED FIRST SAMPLING IS L5c's (9 of 32); measured again here it reads otherwise in this scripted world (4 of 32): THE MEASUREMENT AGAIN says by how much, and names nothing.
    stored_first = store.read_rows("l4_stored_candidate_reach.jsonl")
    assert stored_first == _file_rows(l5c / "episodes_l4_reach_wide_problems.jsonl") and stored_first[0]["resolved"] == SUCCESSES[1]
    assert summaries[PREPARE]["the_stored_sets"]["the_candidates_first_sampling"]["set"] == "l4_reach_wide" and solver.asked[asked + 5] == (GOAL, 32, 1001)
    assert [summaries[step]["adapter"] for step in (MEASURE, ASIDE_CANDIDATE, ASIDE_BAR)] == [str(wide), str(wide), str(best.bar)]
    report = summaries[REPORT]
    assert report["inconclusive"] is False and report["candidate"] == {"of": "l5_cont", "seed": 3, "model": "wide"} and "the candidate `wide` of L5c against `second_full`" in report["headline"]
    assert report["the_measurement_again"]["mean"] == (OF_L5D[0] - SUCCESSES[1]) / 32 and report["can_this_run_be_read"][NO_BARRED]["trainings"] == sorted([*OF_THEIRS, *own_four])
    assert any("CHECK, the candidate is the adapter L5c measured" in line and line.endswith(": PASS") for line in report["lines"])
    assert report["branches"]["the_held_aside_set"] == HOLDS_ON_THE_SET
    # L5c's REPORT, AS IT IS NOW, INCONCLUSIVE: this run's check fails, as for a candidate of L5f.
    kept = _changed(l5c / "report_ladder_l5_cont.json", lambda text: json.dumps({**json.loads(text), "of_this_seed": {**json.loads(text)["of_this_seed"], "inconclusive": True,
                                                                                                                  "failed_checks": ["`more`: lean_answered"]}}))
    failed = ladder_l5_measure.ladder_l5m_report(config)
    (l5c / "report_ladder_l5_cont.json").write_text(kept)
    assert failed["failed_checks"] == [THE_ONE] and failed["can_this_run_be_read"][THE_ONE]["why_not"] == "it reads INCONCLUSIVE (`more`: lean_answered)"
    # `again` AND `more` ARE CANDIDATES TOO, each the mean of its own four, each in a run directory of its own.
    for model in ("again", "more"):
        config["ladder_loop"]["l5_measure"]["candidate"]["model"] = model
        read = _stage5m(config, monkeypatch, until=PREPARE)[PREPARE]
        assert best.store().root.name == f"ladder_l5_measure_l5_cont_{model}_seed3" and read["the_candidate"]["adapter"] == str(l5c / "adapters" / model)
        assert read["the_candidate"]["the_mean_of"] == [f"{model}_{order}" for order in (1, 2, 3, 4)] and read["the_candidate"]["kept_by_another_run"] == {}
        assert read["the_candidate"]["trained_rows"]["trainings"] == 4 and read["the_candidate"]["the_record_of_the_mean"]["passes"] is True
    # `single` IS ONE TRAINING, not an averaged adapter: refused by the settings' reader. So is a model of another stage, and a bar that is not `second_full`.
    for change, said in (({"model": "single"}, "ladder_loop.l5_measure.candidate: the candidate is named 'single' of `l5_cont`"),
                         ({"model": "published"}, "the candidate is named 'published' of `l5_cont`"),
                         ({"of": "l5_draws", "model": "published"}, "the candidate is named of the stage 'l5_draws'"),
                         ({"of": "l5_stock", "model": "stock"}, "the candidate is named of the stage 'l5_stock'"),
                         ({"of": "l5_full", "model": "wide"}, "the candidate is named 'wide' of `l5_full`"),
                         ({"of": "l5_full", "model": "published", "seed": 6}, "the candidate's seed is 6: `l5_full` lists the seeds .3, 4, 5.")):
        config["ladder_loop"]["l5_measure"]["candidate"].update({"of": "l5_cont", "seed": 3, **change})
        with pytest.raises(ValueError, match=said):
            ladder_l5_measure.ladder_l5m_prepare(config)
    config["ladder_loop"]["l5_measure"].update({"candidate": {"of": "l5_full", "seed": 3, "model": "published"}, "bar": "pre_r64"})
    with pytest.raises(ValueError, match="ladder_loop.l5_measure.bar is 'pre_r64': this stage is specified against `second_full` alone"):
        ladder_l5_measure.ladder_l5m_prepare(config)


# ----------------------------------------------------------------------------------------------- the refusals
def test_the_first_step_refuses_before_anything_is_written(best, monkeypatch):
    config, runs, l5f, l4p = best.config, best.runs, best.l5f, best.l4p
    prepare = ladder_l5_measure.ladder_l5m_prepare
    task = "Run the task of stage `ladder_l5_full` for seed 3 .* to its end first; nothing was written"

    def refused(path, changed, said):
        kept = _changed(path, changed)
        with pytest.raises(RuntimeError, match=said):
            prepare(config)
        Path(path).write_text(kept)

    def without(path, said):
        path.rename(path.with_name(f"kept_{path.name}"))
        with pytest.raises(RuntimeError, match=said):
            prepare(config)
        path.with_name(f"kept_{path.name}").rename(path)

    # A TASK OF ANOTHER SEED THAN THE CANDIDATE'S; and the smoke-only variable in the candidate's own run directory.
    monkeypatch.setenv(SEEDS, "4")
    with pytest.raises(RuntimeError, match=r"this task's seed is 4 and the candidate is of seed 3: .* \(ladder_loop.l5_measure.candidate\). Nothing was written"):
        prepare(config)
    monkeypatch.setenv(SEEDS, "3")
    monkeypatch.setenv(SMOKE, "1")
    with pytest.raises(RuntimeError, match="RLVR_LEAN_LADDER_L5_MEASURE_SMOKE is set: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_MEASURE_RUN."):
        prepare(config)
    monkeypatch.delenv(SMOKE)
    # A CANDIDATE OF A SEED WHOSE RUN IS NOT ON THE BOX (a further seed of L5f's, which the settings may name): refused, naming L5f's task of that seed.
    config["ladder_loop"]["l5_measure"]["candidate"]["seed"] = 4
    monkeypatch.setenv(SEEDS, "4")
    with pytest.raises(RuntimeError, match="L5f's run of seed 4 cannot carry L5m: ladder_l5_draws_second_full_seed4 does not hold .* Run the task of stage `ladder_l5_full` for seed 4 "):
        prepare(config)
    config["ladder_loop"]["l5_measure"]["candidate"]["seed"] = 3
    monkeypatch.setenv(SEEDS, "3")
    # L5's FINISHED SEARCH IS NOT THERE: refused by L5's own first-step reading.
    best.l5.rename(runs / "kept_l5")
    with pytest.raises(RuntimeError, match="ladder_l5_deep_pre_r64_seed0"):
        prepare(config)
    (runs / "kept_l5").rename(best.l5)
    # THE BAR. `second_full` not there by L5f's own reading (L4p's run gone); its stored rungs not there; a stored sampling of it made with another seed.
    l4p.rename(runs / "kept_l4p")
    with pytest.raises(RuntimeError, match="second_full"):
        prepare(config)
    (runs / "kept_l4p").rename(l4p)
    without(l4p / "episodes_l4_rungs_second_full.done.json", r"ladder_l4_dose_pre_r64_seed0 does not hold \['episodes_l4_rungs_second_full.done.json'\]: the attempts of `second_full` "
                                                             "on the rungs and on G's samplings, .* Run the task of stage `ladder_l4_dose_r64` for seed 0 .* nothing was written")
    refused(l4p / "episodes_l4_rungs_second_full.done.json", lambda text: json.dumps({**json.loads(text), "sampling_seed": 7}),
            r"ladder_l4_dose_pre_r64_seed0 measured `second_full` \(rungs\) with sampling seed 7 and 8 episodes; this config gives 1002 and 8: L5m's two models would not pair with the "
            r"stored results \(L4p's run: the bar's stored attempts are the other side of every read on the rungs and on G\). Nothing was written")
    # THE CANDIDATE's RUN. Its report gone; of another start or recipe; INCONCLUSIVE; not to be read; measured otherwise.
    for name in ("ladder_l5_full_report.done.json", "report_ladder_l5_full.json"):
        without(l5f / name, f"L5f's run of seed 3 cannot carry L5m: ladder_l5_draws_second_full_seed3 does not hold .*{name}.*: L5f's run of this seed is not on the box, or not "
                            f"finished. Its `published` is the candidate: .* {task}")
    of_the_seed = lambda **changed: (lambda text: json.dumps({**json.loads(text), "of_this_seed": {**json.loads(text)["of_this_seed"], **changed}}))      # noqa: E731
    report = l5f / "report_ladder_l5_full.json"
    refused(report, of_the_seed(inconclusive=True, failed_checks=["`deep`: lean_answered"]), f"cannot carry L5m: it reads INCONCLUSIVE .`deep`: lean_answered.. .* {task}")
    refused(report, of_the_seed(ok=False), f"cannot carry L5m: its report is not to be read. .* {task}")
    refused(report, of_the_seed(seed=5), "cannot carry L5m: it is a report of seed 5")
    refused(report, of_the_seed(start="pre_r64"), "cannot carry L5m: it is a run from `pre_r64` and this stage's bar is `second_full`")
    refused(report, of_the_seed(recipe="constant"), "cannot carry L5m: its recipe is `constant` and the candidate is an averaged adapter .`mean_constant`.")
    refused(report, of_the_seed(the_measurement={"sampling": "reach", "episodes": 16, "sampling_seed": 1001, "problems": 1}), "cannot carry L5m: it was measured otherwise")
    # ... without the record of the mean, of its measurement, or the candidate's rows on the first sampling.
    for name in ("ladder_l5_full_average_published.done.json", "ladder_l5_full_measure_published.done.json", "episodes_l4_reach_published_problems.jsonl"):
        without(l5f / name, rf"ladder_l5_draws_second_full_seed3 does not hold \['{name}'\]: the record of the averaged adapter `published` L5f's run of seed 3 made, of its "
                            f"measurement, or its rows on `reach`. {task}")
    # ... without a training's marker, stored set or loss record: what the candidate was trained on is read from each training's own record.
    for name in ("ladder_l5_full_train_published_2.done.json", "l5_full_training_published_3.jsonl", "l5_full_loss_published_4.json"):
        without(l5f / name, rf"L5f's run of seed 3 recorded `published` as the mean of \['published_1', 'published_2', 'published_3', 'published_4'\], and .*{name}.* is not there: "
                            f"the rows the candidate was trained on are read from each training's own record. {task}")
    # ... a training whose loop recorded a row its stored set does not hold, or no row at all.
    refused(l5f / "l5_full_loss_published_1.json", lambda text: json.dumps({**json.loads(text), "rows_trained": [*json.loads(text)["rows_trained"], "a_row_of_no_set"]}),
            r"the trainings \['published_1'\] of L5f's run of seed 3 recorded rows their stored sets do not hold: what `published` was trained on cannot be read from that run's own record")
    # ... A RECORD OF THE MEAN THAT DOES NOT HOLD: not the mean; one of its members (which only a smoke run passes); another rank; another adapter than the one that run measured.
    averaging, held = l5f / "ladder_l5_full_average_published.done.json", "the candidate `published` of L5f's run ladder_l5_draws_second_full_seed3 is not held by that run's own record"
    read_back = lambda **changed: (lambda text: json.dumps({**json.loads(text), "read_back": {**json.loads(text)["read_back"], **changed}}))      # noqa: E731
    refused(averaging, read_back(is_the_mean=False), f"{held}: .*the mean of its members: False; at rank 64 and alpha 128: True.*Queue the task of stage `ladder_l5_full` for seed 3 .* again")
    refused(averaging, lambda text: json.dumps({**json.loads(read_back(is_none_of_them=False)(text)), "one_of_its_models_in_a_smoke_run": True}), f"{held}: .*the mean of its members: False")
    refused(averaging, lambda text: json.dumps({**json.loads(text), "adapter_saved": {**json.loads(text)["adapter_saved"], "rank": 16, "alpha": 32}}), f"{held}: .*at rank 64 and alpha 128: False")
    refused(l5f / "ladder_l5_full_measure_published.done.json", lambda text: json.dumps({**json.loads(text), "adapter": str(l5f / "adapters" / "published_1")}),
            f"{held}: .*its measurement was sampled from .*adapters/published_1, and this stage would serve .*adapters/published: they are one adapter")
    # ... ITS STORED FIRST SAMPLING made with another sampling seed than its report says: it would not pair with the measurement made again here.
    refused(l5f / "episodes_l4_reach_published.done.json", lambda text: json.dumps({**json.loads(text), "sampling_seed": 1003}),
            r"ladder_l5_draws_second_full_seed3 measured `published` \(reach\) with sampling seed 1003 and 32 episodes; this config gives 1001 and 32: .* \(L5f's run of seed 3: the "
            r"candidate's stored first sampling is what its measurement here is read beside\). Nothing was written")
    # ... A ROW IT WAS TRAINED ON THAT IS A GOAL PROBLEM (held out), or a held-aside problem: that run's own record of its trainings, read before anything is sampled.
    first_is = lambda problem: (lambda text: _as_text([{**row, "problem_id": problem if index == 0 else row["problem_id"]}      # noqa: E731
                                                       for index, row in enumerate(map(json.loads, text.splitlines()))]))
    refused(l5f / "l5_full_training_published_2.jsonl", first_is(GOAL),
            rf"1 problems of the rows `published` was trained on are held-out problems \(first: {GOAL}\): refused, nothing was written \(L5f's run ladder_l5_draws_second_full_seed3\)")
    refused(l5f / "l5_full_training_published_2.jsonl", first_is(best.aside[0]), r"the rows `published` was trained on .*refused, nothing was written \(L5f's run ladder_l5_draws_second_full_seed3\)")
    assert _nothing_written(runs)
    # ... and with everything as it was the first step goes on.
    assert prepare(config)["the_candidate"]["the_record_of_the_mean"]["passes"] is True


def test_the_steps_run_through_the_gpu_entry_point(best, monkeypatch, tmp_path):
    monkeypatch.setattr(milestone2, "load_config", lambda path: best.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    # Nothing is trained here, so a stand-in's world reads too (the candidate's adapter is L5f's run's): the report is built, and says the engine was a stand-in.
    report = json.loads((tmp_path / f"{REPORT}.json").read_text())
    assert best.store().is_done(REPORT) and report["inconclusive"] is False and report["stand_in_engine"] is True and report["branches"]["what_follows"] == THE_BEST_MEASURED
    # A REPORT THAT IS NOT TO BE READ FAILS ITS STEP (Lean did not answer a set this run sampled).
    _rewritten(best.store().path("episodes_l4_more_candidate_problems.jsonl"), attempts_without_an_answer=FRESH_ATTEMPTS)
    monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", REPORT, "--out", str(tmp_path / "again.json")])
    assert gpu_main.main() != 0 and json.loads((tmp_path / "again.json").read_text())["ok"] is False


# ---------------------------------------------------------------------------------------- the smoke-only path
def test_the_smoke_stage_goes_on_from_the_smoke_runs_of_l5_l4p_and_l5f_measures_a_smoke_runs_candidate_and_says_so(deep, monkeypatch):  # noqa: F811
    for name in (RUN, SMOKE, L4P_RUN, L5F_RUN, L5C_RUN, L6_RUN, ladder_l5_full.SMOKE_VARIABLE, ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    first = _smoke_world(deep, monkeypatch)
    config, runs = deep.config, ladder_l5._runs(deep.config)
    of_l5, failed = runs / "ladder_l5_deep_smoke", first["ladder_l5_report"]["failed_checks"]
    calls, kits = [], []
    _trainings(monkeypatch, calls, value=lambda number: 1.0)          # AS ON THE BOX: one optimizer step on the same rows, whatever their order, is ONE training: adapters alike
    solver = MeasureSolver(NoiseSolver(deep.solver, (*OF_L5D, *OF_L5F, SUCCESSES[0])), fresh=None)      # L5d's smoke run's three models, L5f's three, then the candidate's first sampling again
    _engines(monkeypatch, solver, kits)
    options = entry.step_fields(entry.STAGES["ladder_l5_measure_smoke"][2])[2]["environment"]
    of_l5fs = entry.step_fields(entry.STAGES["ladder_l5_full_smoke"][2])[2]["environment"]
    # ---- THE SMOKE RUNS THIS ONE READS, each by its own smoke stage: L4p's (this world's run of L4p, under the name L6's variable gives it), L5d's, then L5f's.
    of_l4p = runs / options[L4P_RUN]
    deep.dose.store().root.rename(of_l4p)
    monkeypatch.setenv(L4P_RUN, options[L4P_RUN])
    monkeypatch.setenv(SEEDS, "3")
    monkeypatch.setenv(L5F_RUN, options[L5F_RUN])
    monkeypatch.setenv(RUN, options[RUN])
    monkeypatch.setenv(SMOKE, options[SMOKE])
    # WITHOUT THE SMOKE RUN OF THE STAGE THAT MADE THE CANDIDATE the first step REFUSES, naming its job.
    with pytest.raises(RuntimeError, match="L5f's run of seed 3 cannot carry L5m: ladder_l5_full_smoke does not hold .*Run the task of stage `ladder_l5_full` for seed 3 "
                                           ".*stage `ladder_l5_full_smoke`. to its end first; nothing was written"):
        ladder_l5_measure.ladder_l5m_prepare(config)
    assert _nothing_written(runs)
    monkeypatch.delenv(RUN)
    monkeypatch.delenv(SMOKE)
    _l5ns_report(runs, AVERAGE_OF_FOUR)
    monkeypatch.setenv(ladder_l5_draws.RUN_VARIABLE, of_l5fs[ladder_l5_draws.RUN_VARIABLE])
    monkeypatch.setenv(ladder_l5_draws.SMOKE_VARIABLE, "1")
    assert _stage5d(config, monkeypatch, stage_name="ladder_l5_draws_smoke")["ladder_l5_draws_report"]["inconclusive"] is False
    monkeypatch.delenv(ladder_l5_draws.SMOKE_VARIABLE)
    monkeypatch.setenv(ladder_l5_full.SMOKE_VARIABLE, of_l5fs[ladder_l5_full.SMOKE_VARIABLE])
    of_l5f = _stage5f(config, monkeypatch, stage_name="ladder_l5_full_smoke")
    theirs = runs / "ladder_l5_full_smoke"
    assert of_l5f["ladder_l5_full_report"]["inconclusive"] is False and sorted(name for name in _all(theirs) if not name.startswith("episodes_")) == L5FS_FILES
    # THIS stage's smoke task names L5's, L4p's, L5f's and L5c's RUNS, and no other stage's smoke-only variable, nor L5d's run variable, nor L6's.
    monkeypatch.delenv(ladder_l5_full.SMOKE_VARIABLE)
    monkeypatch.delenv(ladder_l5_draws.RUN_VARIABLE)
    assert not {ladder_l5_full.SMOKE_VARIABLE, ladder_l5_cont.SMOKE_VARIABLE, ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE, ladder_l5.SMOKE_VARIABLE,
                ladder_l6.SMOKE_VARIABLE, L6_RUN} & set(options)
    # ---- A REAL RUN IS REFUSED FOR SUCH A RUN OF L5 (its report is INCONCLUSIVE), in the candidate's own directory and in one the task names.
    assert failed != []
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE .*: no model here is measured on the held-aside set of such a run"):
        ladder_l5_measure.ladder_l5m_prepare(config)
    monkeypatch.setenv(RUN, options[RUN])
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE"):
        ladder_l5_measure.ladder_l5m_prepare(config)
    # THE SMOKE-ONLY VARIABLE on a map that is no fixture's: refused, and nothing is written.
    monkeypatch.setenv(SMOKE, options[SMOKE])
    marker = deep.map / "ladder_l4b_map.done.json"
    recorded = json.loads(marker.read_text())
    marker.write_text(json.dumps({**recorded, "fixture": False}))
    with pytest.raises(RuntimeError, match="RLVR_LEAN_LADDER_L5_MEASURE_SMOKE is set: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_MEASURE_RUN. and "
                                           "on the fixtures' base map. A run on the pool's problems reads a run of L5 whose checks passed. Nothing was written"):
        ladder_l5_measure.ladder_l5m_prepare(config)
    marker.write_text(json.dumps(recorded))
    assert _nothing_written(runs)
    # ---- THE SMOKE RUN: its own directory, the fixtures' map, the variable. L5's run is read all the same, and that is RECORDED. NOTHING IS TRAINED.
    read_only = {path: _all(path) for path in (of_l5, theirs, of_l4p, deep.map, deep.l4t2, deep.rank)}
    before = len(calls)
    summaries = _stage5m(config, monkeypatch, stage_name="ladder_l5_measure_smoke")
    store = ladder_l5_measure._store(config, ladder_l5_measure.the_settings(config))
    assert store.root.name == "ladder_l5_measure_smoke" and list(summaries) == STEPS and {path: _all(path) for path in read_only} == read_only and len(calls) == before
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    read = summaries[PREPARE]
    smoke, of_it = read["smoke"], read["the_candidate"]
    assert (smoke["problems_at_least"], smoke["l5_read_though_inconclusive"], smoke["l5s_failed_checks"]) == (1, True, failed) and read["seed"] == 3
    assert (read["the_search"]["run"], read["the_search"]["inconclusive"], read["bar_adapter"], read["bar_run"]) == (str(of_l5), True, str(of_l4p / "adapters" / "second_full"), str(of_l4p))
    # THE CANDIDATE IS L5f's SMOKE RUN's `published`: the mean of four trainings that are one adapter number for number, which that run SAID and only a smoke run measures.
    assert (of_it["run"], of_it["adapter"], of_it["the_mean_of"]) == (str(theirs), str(theirs / "adapters" / "published"), list(OF_THEIRS))
    record = of_it["the_record_of_the_mean"]
    assert (record["passes"], record["is_the_mean"], record["one_of_its_models_in_a_smoke_run"]) == (True, True, True) and of_it["trained_rows"] == {
        **of_it["trained_rows"], "rows": 8, "trainings": 4}                                                    # L5f's smoke `published` is two rows, in four orders
    # THE BAR ON THE HELD-ASIDE SET IS MEASURED HERE: the smoke stage does not name L6's run, and the run it would be in is not there.
    assert read["the_bars_held_aside"]["read_from_a_stored_run"] is False and read["the_held_aside"]["sampled_here"] == ["candidate", "second_full"]
    assert [summaries[step]["adapter"] for step in (MEASURE, ASIDE_CANDIDATE, ASIDE_BAR)] == [str(theirs / "adapters" / "published")] * 2 + [str(of_l4p / "adapters" / "second_full")]
    assert [kit["max_lora_rank"] for kit in kits[-5:]] == [64] * 5 and [asked[1] for asked in solver.asked].count(HELD_ATTEMPTS) == 2
    # THE REPORT says A SMOKE RUN; the checks pass; its lines are no reads of a model.
    report = summaries[REPORT]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["smoke"] == smoke
    assert any(f"A SMOKE RUN: L5's smoke run reads INCONCLUSIVE ({', '.join(failed)}) and was read all the same; the candidate is a smoke run's adapter" in line and
               line.endswith("Nothing here is a read of a model") for line in report["lines"])
    assert report["can_this_run_be_read"][THE_ONE]["one_of_its_models_in_a_smoke_run"] is True and report["branches"]["the_held_aside_set"] == HOLDS_ON_THE_SET
    assert report["the_measurement_again"]["mean"] == (SUCCESSES[0] - OF_L5F[0]) / 32
    # THE VARIABLE ALONE makes a run a smoke run: without it the same directory is a real run's (which is refused for this run of L5, as above).
    monkeypatch.delenv(SMOKE)
    assert ladder_l5_measure.smoke_run() is False


def _a_smoke_task(monkeypatch, stage_name, run, seed=None):
    """ONE SMOKE TASK AS THE BOX RUNS IT: the environment its stage names (the fixtures' data, the runs it reads, its own
    run, its smoke-only variable) and nothing of another task's, gone when the task is; `seed`: the job's `--seeds`."""
    options = entry.step_fields(entry.STAGES[stage_name][2])[2]["environment"]
    with monkeypatch.context() as patched:
        patched.delenv(ladder_loop.DATA_VARIABLE, raising=False)
        for name, value in options.items():
            patched.setenv(name, value)
        if seed is not None:
            patched.setenv(SEEDS, str(seed))
        return run(patched)


def test_in_the_boxs_own_smoke_world_g_has_one_sampling_and_the_smoke_stage_runs_every_step_and_makes_no_read_on_fresh_attempts(arm, monkeypatch):  # noqa: F811
    """THE SMOKE WORLD AS THE BOX HAS IT, every run made by its own smoke stage in its own environment and in the box's
    order: L1's; the pretraining's, the arm's, the rank check's (which reads L1's smoke run ALONE, no stored run of L2
    beside it: it records ONE sampling of G), L4t's and L4t2's; L4p's; L5's; L5d's; L5f's. Then this stage's smoke task,
    which the first run on the box (task 333ceb41) refused at its first step for that one sampling."""
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config, calls, kits = arm.config, [], []
    monkeypatch.delenv(ARM)
    for name in (RUN, SMOKE, L4P_RUN, L5F_RUN, L5C_RUN, L6_RUN, SEEDS, ladder_l5.RUN_VARIABLE, ladder_l5.L4T2_RUN_VARIABLE, ladder_l5.SMOKE_VARIABLE, ladder_l4_start.MAP_RUN_VARIABLE,
                 ladder_l4_rank.RUN_VARIABLE, ladder_l4_rank.CHECK_VARIABLE, ladder_l4_rows.RUN_VARIABLE, ladder_l4_rows.ARM_RUN_VARIABLE, ladder_l4_rows.START_VARIABLE,
                 ladder_l4_rows.MINIMUM_VARIABLE, ladder_l4_rows.START_CHECKS_VARIABLE, ladder_l4_rows2.RUN_VARIABLE, ladder_l4_rows2.L4T_RUN_VARIABLE, ladder_l4_dose.RUN_VARIABLE,
                 ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE, ladder_l5_full.SMOKE_VARIABLE, ladder_l5_cont.SMOKE_VARIABLE, ladder_l6.SMOKE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    _trainings_of_the_chain(monkeypatch, calls)
    searching = DeepSolver(arm.solver, {f"fixture_p{index}": (4,) for index in range(1, 5)})       # in L5's search every problem is solved at its fifth attempt
    engines = lambda solver: lambda self, adapter=None: (lambda given: (kits.append(given["vllm"]["max_lora_rank"]), (solver, stand_in_parameters))[1])      # noqa: E731
    # ---- the five smoke tasks before L4p's (as L5's own smoke test runs them): the pretraining's, the arm's, the check's, L4t's from `pre_r64`, then L4t2's
    steps = {**ladder_l2.STEPS, **ladder_l3d2.STEPS, **ladder_l4.STEPS, **ladder_l4_rank.STEPS, **ladder_l4_rows.STEPS, **ladder_l4_rows2.STEPS}

    def of_the_chain(stage_name):
        def run(patched):
            if stage_name == "ladder_l4_smoke":
                arm.with_lean()
                patched.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
            if stage_name in ("ladder_l4_rows_r64_smoke", "ladder_l4_rows2_r64_smoke"):
                patched.setattr(ladder_round.Engines, "kit", engines(arm.solver))
            for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
                if environment == "gpu" and step in steps:
                    _on_the_gpu(patched, steps[step], config) if "_train" in step else steps[step](config)
        return run

    for stage_name in ("ladder_l4_pretrain_smoke", "ladder_l4_smoke", "ladder_l4_rank_smoke", "ladder_l4_rows_r64_smoke", "ladder_l4_rows2_r64_smoke"):
        _a_smoke_task(monkeypatch, stage_name, of_the_chain(stage_name))
    runs = pipeline.STORE / "runs-v4.27"
    # THE WORLD'S ONE SAMPLING: the smoke `pre_r64`'s run recorded G's first sampling and no second, and L4t2's smoke run measured its models so.
    of_the_start = json.loads((runs / "ladder_l4_rank_smoke" / f"{ladder_l4_rank.PREPARE}.done.json").read_text())
    assert [own["name"] for own in of_the_start["goal_samplings"]] == ["reach"]
    arm.with_lean()
    monkeypatch.setattr(ladder_round.Engines, "kit", engines(arm.solver))
    # ---- L4p's smoke task (`second_full` of the smoke world, THE BAR), then the smoke `pre_r64`'s map as the box has it (every problem solved), then L5's
    of_l4p = _a_smoke_task(monkeypatch, "ladder_l4_dose_r64_smoke", lambda patched: _stage_p(config, patched, stage_name="ladder_l4_dose_r64_smoke"))
    assert (runs / "ladder_l4_dose_r64_smoke" / "adapters" / "second_full").exists() and list(of_l4p)[-1] == "ladder_l4_dose_report"
    _write_the_map(config, {"fixture_p1": (0, 1, 2), "fixture_p2": tuple(range(8)), "fixture_p3": tuple(range(8)), "fixture_p4": tuple(range(8))}, directory=runs / "ladder_l4b_map_smoke")
    monkeypatch.setattr(ladder_round.Engines, "kit", engines(searching))
    of_l5 = _a_smoke_task(monkeypatch, "ladder_l5_deep_smoke", lambda patched: _stage5(config, patched, stage_name="ladder_l5_deep_smoke"))
    assert any("the models have ONE sampling of G here (a smoke run)" in line for line in of_l5["ladder_l5_report"]["lines"])
    # ---- L5d's and L5f's smoke tasks, seed 3 (their jobs' `--seeds`). AS ON THE BOX one optimizer step on the same rows is ONE training: the adapters of a set are alike.
    del calls[:]
    _trainings(monkeypatch, calls, value=lambda number: 1.0)
    _l5ns_report(runs, AVERAGE_OF_FOUR)                                        # L5n's REAL run, which L5d's smoke stage reads for the recipe (it is on the box)
    solver = MeasureSolver(searching, fresh=None)
    _engines(monkeypatch, solver, kits)
    of_l5d = _a_smoke_task(monkeypatch, "ladder_l5_draws_smoke", lambda patched: _stage5d(config, patched, stage_name="ladder_l5_draws_smoke"), seed=3)
    assert of_l5d["ladder_l5_draws_report"]["inconclusive"] is False and (runs / "ladder_l5_draws_smoke_mean_constant").exists()
    of_l5f = _a_smoke_task(monkeypatch, "ladder_l5_full_smoke", lambda patched: _stage5f(config, patched, stage_name="ladder_l5_full_smoke"), seed=3)
    theirs, l4p, l5 = runs / "ladder_l5_full_smoke", runs / "ladder_l4_dose_r64_smoke", runs / "ladder_l5_deep_smoke"
    assert of_l5f["ladder_l5_full_report"]["inconclusive"] is False and sorted(name for name in _all(theirs) if not name.startswith("episodes_")) == L5FS_FILES
    # ---- THIS STAGE's SMOKE TASK, as its job queues it: EVERY STEP RUNS; every run it reads is left file for file as it was; nothing is trained.
    read_only = {path: _all(path) for path in runs.iterdir() if path.is_dir()}
    before, asked = len(calls), len(solver.asked)
    summaries = _a_smoke_task(monkeypatch, "ladder_l5_measure_smoke", lambda patched: _stage5m(config, patched, stage_name="ladder_l5_measure_smoke"), seed=3)
    own = runs / "ladder_l5_measure_smoke"
    assert list(summaries) == STEPS and {path: _all(path) for path in read_only} == read_only and len(calls) == before
    assert sorted(path.name for path in runs.iterdir() if path.name.startswith("ladder_l5_measure")) == ["ladder_l5_measure_smoke"]
    # No file of a second sampling: the stored sets are the rungs' and `reach`'s, and the candidate was sampled on the rungs, on `reach` and on the held-aside set.
    one_sampling = sorted(name for name in OWN_FILES if not name.endswith("_more.jsonl"))
    assert sorted(name for name in _all(own) if not name.startswith("episodes_")) == one_sampling and len(one_sampling) == len(OWN_FILES) - 2
    read = summaries[PREPARE]
    smoke, of_it = read["smoke"], read["the_candidate"]
    assert (smoke["one_sampling_of_g"], smoke["problems_at_least"], smoke["l5_read_though_inconclusive"]) == (True, 1, of_l5["ladder_l5_report"]["inconclusive"])
    assert [own_["name"] for own_ in read["goal_samplings"]] == ["reach"] and read["the_measurement"]["fresh_sampling"] is None and sorted(read["the_measurement"]["sets"]) == [
        "held_aside", "reach", "rungs"] and read["goal_samplings"] == of_the_start["goal_samplings"]
    assert (of_it["run"], of_it["adapter"], read["bar_adapter"], read["the_search"]["run"]) == (str(theirs), str(theirs / "adapters" / "published"), str(l4p / "adapters" / "second_full"), str(l5))
    assert read["the_bars_held_aside"]["read_from_a_stored_run"] is False and read["the_held_aside"]["sampled_here"] == ["candidate", "second_full"]
    assert [summaries[step]["adapter"] for step in (MEASURE, ASIDE_CANDIDATE, ASIDE_BAR)] == [str(theirs / "adapters" / "published")] * 2 + [str(l4p / "adapters" / "second_full")]
    held = [entry_ for entry_ in solver.asked[asked:] if entry_[1] == HELD_ATTEMPTS]
    assert len(held) == 2 * read["the_held_aside"]["problems"] and {entry_[2] for entry_ in held} == {read["the_held_aside"]["sampling_seed"]} and FRESH_ATTEMPTS not in {
        entry_[1] for entry_ in solver.asked[asked:]}
    # THE REPORT: the checks pass on the sets that are there (seven, not nine); the primary is read on the held-aside set; FRESH ATTEMPTS and what follows are NOT READ, and said.
    report = summaries[REPORT]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["failed_checks"] == [] and report["smoke"] == smoke
    assert len(report["can_this_run_be_read"][THE_SETS]["sets"]) == 7 and not [name for name in report["can_this_run_be_read"][THE_SETS]["sets"] if "more" in name]
    assert report["branches"] == {"the_held_aside_set": report["branches"]["the_held_aside_set"], "fresh_attempts": NOT_READ, "what_follows": NOT_READ}
    assert report["branches"]["the_held_aside_set"] in (HOLDS_ON_THE_SET, NOT_SHOWN_ON_THE_SET, DOES_NOT_HOLD_THERE) and report["the_held_aside_set"]["problems"] == read["the_held_aside"]["problems"]
    assert report["fresh_attempts"]["name"] == NOT_READ and "ONE sampling of G" in report["fresh_attempts"]["reason"] and report["of_this_seed"]["fresh_attempts"] is None
    assert report["beside"]["g_again"]["mean"] is None and report["beside"]["all_of_g"]["attempts_a_problem"] == read["goal_samplings"][0]["episodes"]
    assert any("A SMOKE RUN" in line and "The models have ONE sampling of G here (the smoke runs recorded its first sampling alone)" in line and
               line.endswith("Nothing here is a read of a model") for line in report["lines"])
    assert any(line.startswith("l4 (pretrained on published proofs): FRESH ATTEMPTS: NOT READ") for line in report["lines"]) and "ONE sampling of G here, a smoke run" in report["headline"]
    # ---- ONE SAMPLING IS A SMOKE RUN's ALONE. The report refuses such a run that did not record it went on from one ...
    marker = own / f"{PREPARE}.done.json"
    recorded = json.loads(marker.read_text())
    marker.write_text(json.dumps({**recorded, "smoke": {**recorded["smoke"], "one_sampling_of_g": False}}))
    with monkeypatch.context() as patched:
        patched.setenv(RUN, "ladder_l5_measure_smoke")
        patched.setenv(SEEDS, "3")
        with pytest.raises(ValueError, match="the report is given a run against `second_full` with 1 samplings of the goal set"):
            ladder_l5_measure.ladder_l5m_report(config)
    marker.write_text(json.dumps(recorded))
    assert {path: _all(path) for path in read_only} == read_only


def test_a_real_run_is_refused_for_a_start_run_that_recorded_one_sampling_of_g(best, monkeypatch):
    """THE REAL STAGE's refusal stands: where the start model's run and L4t2's recorded ONE sampling of G and the task is
    no smoke run, the first step refuses before anything is written (the box's first smoke run met it, and no real run may)."""
    config, deep_ = best.config, best.deep
    changed = {}
    for directory, marker in ((deep_.rank, f"{ladder_l4_rank.PREPARE}.done.json"), (deep_.l4t2, f"{ladder_l4_rows2.PREPARE}.done.json")):
        path = directory / marker
        changed[path] = path.read_text()
        recorded = json.loads(changed[path])
        path.write_text(json.dumps({**recorded, "goal_samplings": recorded["goal_samplings"][:1]}))
    try:
        ladder_l5_measure.ladder_l5m_prepare(config)
    except RuntimeError as error:
        refusal = str(error)
    else:
        refusal = ""
    for path, as_it_was in changed.items():
        path.write_text(as_it_was)
    assert _nothing_written(best.runs)
    # L5's own first-step reading refuses it FIRST here (L5's real run was measured on two samplings, and this task would measure on one); where L5's run was itself made
    # on one sampling, this stage's own first step refuses, in the words the box's first smoke run met. Either way: a real run, refused, nothing written.
    assert re.search(r"the deep search's run \S+ was made with another start model, rank, search or held-aside set than this task would make: how a model is measured .*"
                     r"'goal_samplings': \[\{'name': 'reach', 'episodes': 32, 'sampling_seed': 1001\}\]\}\)\. .*Nothing was written|"
                     r"recorded 1 samplings of the goal set \(\['reach'\]\): this stage reads its first .* and its second \(fresh attempts\)\. Nothing was written", refusal), refusal


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_the_smoke_variable_is_the_smoke_stages_alone(monkeypatch):
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_measure"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l5_measure.STEPS) == STEPS and ladder_l5_measure.ENTRY_STAGE == "ladder_l5_measure" and len(STEPS) == 5
    # THE SMOKE STAGE: the same steps, under the runs L5c's smoke stage names WITHOUT its smoke-only variable, a run of its own, and ITS variable. L6's run is NOT named.
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_measure_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    of_l5c = entry.step_fields(entry.STAGES["ladder_l5_cont_smoke"][2])[2]["environment"]
    own_options = smoke[0][2]["environment"]
    assert own_options == {**{name: value for name, value in of_l5c.items() if name != ladder_l5_cont.SMOKE_VARIABLE}, RUN: "ladder_l5_measure_smoke", SMOKE: "1"}
    assert (own_options[ladder_l5.RUN_VARIABLE], own_options[L4P_RUN], own_options[L5F_RUN], own_options[L5C_RUN]) == (
        "ladder_l5_deep_smoke", "ladder_l4_dose_r64_smoke", "ladder_l5_full_smoke", "ladder_l5_cont_smoke")
    assert not {ladder_l5.SMOKE_VARIABLE, ladder_l6.SMOKE_VARIABLE, ladder_l5_stock.SMOKE_VARIABLE, ladder_l5_full.SMOKE_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE,
                ladder_l5_cont.SMOKE_VARIABLE, ladder_l5_draws.RUN_VARIABLE, L6_RUN} & set(own_options)
    # THE SETTING, as the package's own config has it: the candidate (the stage that made it, its seed, its name there) and the bar.
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    assert whole["ladder_loop"]["l5_measure"] == {"candidate": {"of": "l5_full", "seed": 3, "model": "published"}, "bar": "second_full"} == CONFIG["ladder_loop"]["l5_measure"]
    own = ladder_l5_measure.the_settings(whole)
    of_l5f = ladder_l5_full.the_settings(whole)
    assert (own.of, own.seed, own.model, own.bar, own.full, own.cont, own.draws, own.deep) == (
        "l5_full", 3, "published", "second_full", of_l5f, ladder_l5_cont.the_settings(whole), of_l5f.draws, of_l5f.deep)
    # THE RUN DIRECTORY IS ITS OWN, WITH THE CANDIDATE IN ITS NAME; the candidate's run and `second_full` live where their own stages' functions find them: no path is stated twice.
    for name in (RUN, L5F_RUN, L5C_RUN):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(SEEDS, "3")
    assert ladder_l5_measure.run_name(own, 3) == "ladder_l5_measure_l5_full_published_seed3" == ladder_l5_measure.own_run(whole, own)
    maker = ladder_l5_measure.the_maker(whole, own)
    assert (maker.said, maker.directory.name, maker.report, maker.report_file, maker.average("published"), maker.stage) == (
        "L5f", "ladder_l5_draws_second_full_seed3", "ladder_l5_full_report", "report_ladder_l5_full.json", "ladder_l5_full_average_published", ladder_l5_full.THE_STAGE)
    whole["ladder_loop"]["l5_measure"]["candidate"].update({"of": "l5_cont", "model": "more"})
    of_l5c_ = ladder_l5_measure.the_settings(whole)
    maker = ladder_l5_measure.the_maker(whole, of_l5c_)
    assert (maker.said, maker.directory.name, maker.report, maker.report_file, maker.average("more"), maker.stage) == (
        "L5c", "ladder_l5_cont_second_full_seed3", "ladder_l5_cont_report", "report_ladder_l5_cont.json", "ladder_l5_cont_average_more", ladder_l5_cont.THE_STAGE)
    assert ladder_l5_measure.run_name(of_l5c_, 3) == "ladder_l5_measure_l5_cont_more_seed3" and ladder_l5_measure.its_adapter(maker, of_l5c_) == maker.directory / "adapters" / "more"
    assert (ladder_l5_measure.held_aside_marker("candidate"), ladder_l5_measure.held_aside_marker("second_full"), ladder_l5_measure.REPORT_FILE, ladder_l5_measure.TRAINED_FILE) == (
        ASIDE_CANDIDATE, ASIDE_BAR, "report_ladder_l5_measure.json", "l5m_trained_rows.jsonl")
    # ONE reader of its settings; its steps are its own (`ladder_l5_measure_<model>` is a step of L5's: this stage's begin `ladder_l5m_`), and only its two stages run them.
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["ladder_loop"]["l5_measure"]' in path.read_text()] == ["gpu/ladder_l5_measure.py"]
    others = (set(ladder_l5.STEPS) | set(ladder_l6.STEPS) | set(ladder_l4_rows.STEPS) | set(ladder_l5_noise.STEPS) | set(ladder_l5_draws.STEPS) | set(ladder_l5_stock.STEPS)
              | set(ladder_l5_full.STEPS) | set(ladder_l5_cont.STEPS))
    assert not set(ladder_l5_measure.STEPS) & others and all(name.startswith("ladder_l5m_") for name in ladder_l5_measure.STEPS) and "ladder_l5_measure_published" in ladder_l5.STEPS
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l5_measure.STEPS for step in entry.STAGES[name])) == ["ladder_l5_measure", "ladder_l5_measure_smoke"]
    # ITS VARIABLES are named by its smoke stage alone, and no other module reads them.
    names = lambda variable: sorted(name for name, own_ in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own_))      # noqa: E731
    assert names(RUN) == names(SMOKE) == ["ladder_l5_measure_smoke"] and SMOKE == "RLVR_LEAN_LADDER_L5_MEASURE_SMOKE" and RUN == "RLVR_LEAN_LADDER_L5_MEASURE_RUN"
    assert sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if SMOKE in path.read_text()) == ["gpu/ladder_l5_measure.py", "runner/entry.py"]
    # ONE BODY EACH, CALLED AND NOT COPIED: the ceiling's measuring body (with the rungs), L5's own body for the held-aside set and L5's own first-step reading, L5f's reading of
    # `second_full`, the ceiling's reading of a stored set. NOTHING IS TRAINED OR AVERAGED: no training body and no averaging body is named here.
    assert ladder_l5_measure.measure_model is ladder_ceiling.measure_model
    own_stage = ladder_l5_measure.THE_STAGE
    assert (own_stage.name, own_stage.steps, own_stage.files, own_stage.reader, own_stage.prepare, own_stage.start_said_by) == (
        "ladder_l5_measure", "ladder_l5m", "l5m", "L5m", "ladder_l5m_prepare", "ladder_loop.l5_measure.bar")
    source = (PACKAGE / "gpu" / "ladder_l5_measure.py").read_text()
    for called in ("ladder_l5.read_the_search(", "ladder_l5.read_for_the_search(", "ladder_l5.on_the_held_aside_set(", "ladder_l5_full.read_second_full(", "ladder_l5_full.the_start(",
                   "ladder_l5_full.own_run(", "ladder_l5_cont.own_run(", "ladder_l5_cont.l5f_directory(", "ladder_ceiling._stored(", "ladder_l6.run_name(", "measure_model("):
        assert called in source, called
    assert "train_a_model" not in source and "the_mean_adapter" not in source and "mean_of(" not in source and "LoRARequest" not in source and "_episodes(" not in source
    assert "one_pass(" not in source and "rungs=False" not in source and '"second_full"' not in source and "ladder_l4_dose_pre_r64" not in source
    report_source = (PACKAGE / "reporting" / "ladder_l5_measure.py").read_text()
    for called in ("by_the_interval(", "sets_are_the_settings(", "adapters_are_held(", "is_the_one_measured(", "of_its_run(", "per_attempt(", "the_start_adapter(", "no_barred_row(",
                   "no_problem_of(", "timed_out("):
        assert called in report_source, called
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l5_measure.STEPS" in main and "ladder_l5_measure," in main
