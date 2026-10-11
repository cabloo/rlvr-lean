"""L5, further seeds: a task of another seed than the search's reads the ONE search, the held-aside set and the start
model's measurement on it from the finished run of the seed that made them, draws its rows by ITS seed, trains and
measures its three models as that seed's were, and reports them; its refusals; the search's own seed left as it was; the
rules of the read over the seeds, the read itself on hand-made rows, and the command on written task directories.
Spec: docs/spec/ladder-loop.spec.md, "L5, further seeds" and its amendment ("where the lead sits, and a read
of reach"). The engine and Lean are scripted; nothing touches a GPU or the network."""

import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GOAL_IDS, GROUPS, LENGTHS  # noqa: E402 - 12 goal problems, 3 a length group: the last three are the long ones
from test_ladder_l4_dose import MORE, START  # noqa: E402
from test_ladder_l4_rows_stage import _all  # noqa: E402
from test_ladder_l5 import AGAIN, ASIDE, HA_BASE, HA_LESS, HA_MORE, LESS, PREPARE, _aside, _model, _report  # noqa: E402 - the report's hand-made world
from test_ladder_l5_stage import (  # noqa: E402, F401
    CANDIDATES,
    CONFIG,
    DEEP_SCRIPT,
    FOUR,
    HELD,
    HELD_ASIDE_LABEL,
    MAP_SET,
    MAP_VERIFIED,
    ON_THE_SET,
    ORDINARY,
    OWN_FILES,
    PACKAGE,
    REPORT_KEYS,
    SEARCHED,
    SEEDS,
    STEPS,
    _draw_order,
    _file_rows,
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
from rlvr_lean.domain.ladder_round.l4_dose import THEY_COST  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5 import (  # noqa: E402
    A_LOOP_OF_DEEP_SEARCHES,
    A_POINT,
    AFTER_THE_SEEDS,
    CAP_LABEL,
    DEEP,
    DEEP_ADDS,
    LEAD,
    LEAD_CONFIRMED,
    MODELS,
    NEW_SEEDS,
    NOT_CONFIRMED,
    NOT_SHOWN_AT_THREE,
    OTHERWISE,
    OVER_THE_SEEDS,
    PUBLISHED,
    REACH,
    REACH_CONFIRMED,
    REACH_TOO,
    RELIABILITY,
    RELIABILITY_ALONE,
    SEARCH_LABEL,
    SHALLOW,
    SHALLOW_LABEL,
    STOP,
    after_the_seeds,
    primary_over_the_seeds,
    reach_on_the_long,
    the_lead,
    the_split,
    timed_out,
)
from rlvr_lean.domain.problem_pool.selection import rank  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_dose, ladder_l4_start, ladder_l5  # noqa: E402
from rlvr_lean.reporting.ladder_ceiling import attempts_on_g, per_attempt, solved  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import NOT_TO_BE_READ, SAY  # noqa: E402
from rlvr_lean.reporting.ladder_l5 import LONG, SPREAD_OF, over_seeds, pair_name, sets_by_kind  # noqa: E402
from rlvr_lean.tools import ladder_l5_seeds  # noqa: E402

RESAMPLES, BOOTSTRAP_SEED = EVALUATION["bootstrap_resamples"], EVALUATION["bootstrap_seed"]
FURTHER_FILE = "l5_stored_pre_r64_held_aside.jsonl"         # the start model's rows on the held-aside set, as a further seed's first step copies them from the search's run
NOT_MADE_AGAIN = ("ladder_l5_search", "ladder_l5_held_aside_start")      # the two steps that make nothing at a further seed
ABOVE, HOLDS, BELOW = ({"problems": 9, "mean": mean, "low": low, "high": high} for mean, low, high in ((0.02, 0.01, 0.03), (0.01, -0.01, 0.03), (-0.02, -0.03, -0.01)))
ENDS_AT_ZERO, NOT_MEASURED = {"problems": 9, "mean": 0.01, "low": 0.0, "high": 0.02}, {"problems": 0, "mean": None, "low": None, "high": None}


# --------------------------------------------------------------------------- the stage at a further seed
ORDINARY_VERIFIED = {**MAP_VERIFIED, "fixture_p4": (2, 5)}     # this file's map: BOTH ordinary solves have two verified map attempts, so that which one is a row's is a draw


def _map_attempts(key):
    """The ids of an ordinary solve's verified map attempts (both are known true: the side is the statement)."""
    return [f"{key}#{MAP_SET}#statement#{episode}" for episode in ORDINARY_VERIFIED[key]]


def _draws_otherwise(seed, verified, published):
    """Whether a seed draws another verified attempt of the deep solve, another ordinary solve, another of ITS verified map attempts than seed 0's hash would, and other
    published rows than seed 0 does; and WOULD search another problem than seed 0 did, were the search drawn by it."""
    first = lambda own_seed: (min(verified, key=lambda key: rank(own_seed, "training_target", key)), min(ORDINARY, key=lambda key: rank(own_seed, SHALLOW_LABEL, key)),      # noqa: E731
                              [row["id"] for row in _draw_order(published, own_seed)][:1])
    ordinary = _map_attempts(first(seed)[1])
    return all(own != of_seed_0 for own, of_seed_0 in zip(first(seed), first(0))) and {row["id"] for row in _draw_order(published, seed)[:2]} != {row["id"] for row in _draw_order(published, 0)[:2]} \
        and min(ordinary, key=lambda key: rank(seed, "training_target", key)) != min(ordinary, key=lambda key: rank(0, "training_target", key)) \
        and min(CANDIDATES, key=lambda key: rank(seed, SEARCH_LABEL, key)) != SEARCHED


@pytest.fixture
def further(deep, monkeypatch):  # noqa: F811
    """`deep`'s world with seed 0's task RUN TO ITS END: the finished run a further seed reads. The stand-in training's
    losses rise with a row's place, and two rows have no loss to compare (the scripted report would read INCONCLUSIVE,
    which a further seed is refused for), so here every row's loss is level. `seed`: a further seed that draws another
    verified attempt, another ordinary solve and other published rows than seed 0 does."""
    made_up = ladder_dose._stand_in_calls
    monkeypatch.setattr(ladder_dose, "_stand_in_calls", lambda examples, sample, pairs: ((lambda batch: [[2.0, 1.0, 1.0, 1.0, 0.25, 0.125] for _ in batch]), *made_up(examples, sample, pairs)[1:]))
    config = deep.config
    _write_the_map(config, ORDINARY_VERIFIED)
    first = _stage5(config, monkeypatch)
    root = deep.store().root
    assert root.name == "ladder_l5_deep_pre_r64_seed0" and first["ladder_l5_report"]["inconclusive"] is False and first["ladder_l5_report"]["ok"] is True
    verified = [row["attempt_id"] for path in sorted(root.glob("episodes_l5_deep_search_attempts_*.jsonl")) for row in _file_rows(path) if row["status"] == "verified"]
    pretraining = ladder_l4_start.pretrained_on(deep.rank, first["ladder_l5_prepare"]["pretraining_file_sha256"], "pre_r64")[0]
    published = [{**row, "id": f"{row['problem_id']}#pretraining", "origin": "pretraining", "row_of_the_file": place} for place, row in enumerate(pretraining)]
    seed = next(number for number in range(1, 200) if _draws_otherwise(number, verified, published))
    return SimpleNamespace(config=config, deep=deep, first=first, root=root, seed=seed, verified=verified, published=published, asked=list(deep.solver.asked),
                           runs=ladder_l5._runs(config), own=lambda: ladder_l5._store(config, ladder_l5.the_settings(config)))


def test_a_further_seed_reads_the_one_search_draws_its_rows_by_its_own_seed_and_measures_its_three_models_as_seed_0s_were(further, monkeypatch):
    config, deep_, seed, first = further.config, further.deep, further.seed, further.first
    calls, kits, solver = deep_.calls, deep_.kits, deep_.solver
    read_only = {path: _all(path) for path in (further.root, deep_.map, deep_.l4t2, deep_.rank, deep_.l4t, deep_.arm, deep_.pre)}
    monkeypatch.setenv(SEEDS, str(seed))
    del calls[:], kits[:]
    summaries = _stage5(config, monkeypatch)
    store = further.own()
    assert store.root.name == f"ladder_l5_deep_pre_r64_seed{seed}" and store.root.parent == further.root.parent and list(summaries) == STEPS
    assert {path: _all(path) for path in read_only} == read_only               # SEED 0's RUN AND EVERY RUN IT READ WERE ONLY READ: file for file
    assert sorted(path.name for path in further.runs.iterdir() if path.name.startswith("ladder_l5")) == sorted([further.root.name, store.root.name])
    assert not (further.runs / f"ladder_l4_rows2_pre_r64_seed{seed}").exists()                                                       # no L4t2 run of ITS seed is looked for, or made
    # ITS FILES: seed 0's, less the markers of the two steps that make nothing here, with the start model's rows on the held-aside set COPIED; no search and no start model sampled.
    written = _all(store.root)
    assert sorted(name for name in written if not name.startswith("episodes_")) == sorted({*OWN_FILES, FURTHER_FILE} - {f"{step}.done.json" for step in NOT_MADE_AGAIN})
    assert not [name for name in written if "l5_deep_search" in name or "held_aside_pre_r64" in name and name != FURTHER_FILE]

    # ---- the first step: what it read of seed 0's run and recorded; the search and the held-aside set are seed 0's, by seed 0's draws and sampling seeds
    read, of_seed_0 = summaries["ladder_l5_search_prepare"], first["ladder_l5_search_prepare"]
    of_it = read["a_further_seed"]
    assert list(read) == [*of_seed_0, "a_further_seed"] and "a_further_seed" not in of_seed_0 and (read["seed"], read["recipe"]["adapter_seed"]) == (seed, seed)
    assert (of_it["seed"], of_it["search_seed"], of_it["search_run"]) == (seed, 0, str(further.root)) and list(of_it["not_made_again"]) == list(NOT_MADE_AGAIN)
    assert all(read[key] == of_seed_0[key] for key in of_seed_0 if key not in ("seed", "recipe")) and read["recipe"] == {**of_seed_0["recipe"], "adapter_seed": seed}
    assert (read["the_search"]["sampling_seed"], read["the_held_aside"]["sampling_seed"], read["l4t2_run"]) == (1050, 1051, str(deep_.l4t2))
    assert store.read_rows("l5_search_problems.jsonl") == _file_rows(further.root / "l5_search_problems.jsonl") == [{"problem_id": SEARCHED, "map_k": 0, "map_n": 8, "taken_by": "the rule"}]
    assert min(CANDIDATES, key=lambda key: rank(seed, SEARCH_LABEL, key)) == HELD != SEARCHED          # a search drawn by THIS seed would have taken the other candidate
    assert store.read_rows("l5_held_aside_problems.jsonl") == _file_rows(further.root / "l5_held_aside_problems.jsonl") and store.read_rows("l5_held_aside_problems.jsonl")[0]["problem_id"] == HELD
    assert store.read_rows(FURTHER_FILE) == _file_rows(further.root / "episodes_l5_held_aside_pre_r64_problems.jsonl")                 # `pre_r64` on the held-aside set: seed 0's rows
    assert of_it["the_start_on_the_held_aside_set"] == {"file": FURTHER_FILE, "set": "l5_held_aside_pre_r64", "problems": 1, "episodes_each": 16, "sampling_seed": 1051,
                                                        "successes": ON_THE_SET["pre_r64"]}
    theirs = of_it["of_the_search_run"]
    assert (theirs["deep_solves"], theirs["rows_of_each_kind"], theirs["headline"], theirs["branches"]) == (1, 1, first["ladder_l5_report"]["headline"], first["ladder_l5_report"]["branches"])
    assert theirs["the_search_found"] == {key: first["ladder_l5_prepare"]["the_search_found"][key] for key in ladder_l5.SEARCH_TABLE}
    sets_of = lambda name: sorted({row["set"] for row in _file_rows(name / "problems.jsonl")})      # noqa: E731
    assert sets_of(store.root) == sorted(set(sets_of(further.root)) - {"l5_deep_search", "l5_held_aside_pre_r64"})                    # every set of its problems.jsonl is one IT samples

    # ---- NO SEARCH, and the start model is NOT measured on the held-aside set again: each of the two steps says so, marks nothing, samples nothing
    for step in NOT_MADE_AGAIN:
        own = summaries[step]
        assert (own["run"], own["search_seed"], own["search_run"]) == (False, 0, str(further.root)) and "only read, and is not made a second time" in own["why"] and not store.is_done(step)
    new = solver.asked[len(further.asked):]
    held_side = f"negation_of_{HELD}" if HELD == "fixture_p3" else HELD
    assert not [asked for asked in new if asked[1] == 56] and [asked for asked in new if asked[1] == 16] == [(held_side, 16, 1051)] * 3      # THREE models there, with SEED 0's sampling seed
    assert not [asked for asked in new if asked[2] in (1050 + 100 * seed, 1051 + 100 * seed)]
    # The rungs and G: with the sampling seeds seed 0's three models were measured with, on the same problems, the same number of attempts.
    of_a_model = lambda asked: sorted(entry_ for entry_ in asked if entry_[1] not in (16, 56))      # noqa: E731
    assert of_a_model(new) == of_a_model(further.asked) and len(of_a_model(new)) > 0
    assert [kit["block_problems"] for kit in kits] == [5, 5, 5, 2] * 3 and [kit["max_lora_rank"] for kit in kits] == [64] * 12       # twelve engines: no search's, no start model's

    # ---- the three sets, BY THIS TASK'S SEED: which verified attempt of the deep solve, which ordinary solve, the first D and the first 2D of ITS hash order, its training order
    prepared = summaries["ladder_l5_prepare"]
    assert prepared["a_further_seed"] == of_it and "a_further_seed" not in first["ladder_l5_prepare"] and (prepared["seed"], prepared["deep_solves"], prepared["rows_of_each_kind"]) == (seed, 1, 1)
    sets = {model: store.read_rows(f"l5_training_{model}.jsonl") for model in MODELS}
    of_0 = {model: _file_rows(further.root / f"l5_training_{model}.jsonl") for model in MODELS}
    (deep_row,), (deep_row_0,) = ([row for row in own if row["origin"] == DEEP] for own in (sets[DEEP], of_0[DEEP]))
    assert deep_row["id"] == min(further.verified, key=lambda key: rank(seed, "training_target", key)) != deep_row_0["id"] and deep_row["problem_id"] == deep_row_0["problem_id"] == SEARCHED
    assert (deep_row["first_success"], deep_row["k"], deep_row["n"]) == (deep_row_0["first_success"], len(DEEP_SCRIPT[SEARCHED]), 56) and deep_row["completion"] != deep_row_0["completion"]
    (shallow_row,), (shallow_row_0,) = ([row for row in own if row["origin"] == SHALLOW] for own in (sets[SHALLOW], of_0[SHALLOW]))
    assert shallow_row["problem_id"] == min(ORDINARY, key=lambda key: rank(seed, SHALLOW_LABEL, key)) != shallow_row_0["problem_id"]
    assert store.read_rows("l5_ordinary_solves.jsonl")[0]["problem_id"] == shallow_row["problem_id"]
    of_the_map = _map_attempts(shallow_row["problem_id"])                       # ... and WHICH of its two verified map attempts: by this task's seed, not by seed 0's
    assert shallow_row["id"] == min(of_the_map, key=lambda key: rank(seed, "training_target", key)) != min(of_the_map, key=lambda key: rank(0, "training_target", key)) and len(of_the_map) == 2
    order = _draw_order(further.published, seed)
    assert list(map(_placed, _draw_order(sets[DEEP], seed))) == list(map(_placed, _draw_order(sets[SHALLOW], seed))) == list(map(_placed, order[:1])) != list(map(_placed, _draw_order(of_0[DEEP])))
    assert list(map(_placed, _draw_order(sets[PUBLISHED], seed))) == list(map(_placed, order[:2])) and {row["id"] for row in sets[PUBLISHED]} != {row["id"] for row in of_0[PUBLISHED]}
    for model in MODELS:
        assert [row["id"] for row in sets[model]] == [row["id"] for row in training_order(sets[model], seed)] and len(sets[model]) == 2
        assert not any(field in row for row in sets[model] if row["origin"] == "pretraining" for field in TEXT_FIELDS)              # no published row with its text
        assert HELD not in {row["problem_id"] for row in sets[model]}
    assert store.read_rows("l5_deep_solves.jsonl") == [{**{key: value for key, value in deep_row.items() if key != "row"}, "in_the_set": True}]

    # ---- each model trained from `pre_r64` with THIS seed, measured, and measured on the held-aside set; the report reads them against seed 0's rows of `pre_r64`
    start = deep_.rank / "adapters" / "pre_r64"
    assert [(call["seed"], call["lora"]["rank"], call["more"]) for call in calls] == [(seed, 64, {"start": start})] * 3
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l5_deep_pre_r64_seed{seed}_{model}" for model in MODELS]
    assert [[example["problem_id"] for example in call["examples"]] for call in calls] == [[row["problem_id"] for row in sets[model]] for model in MODELS]
    on_the_set = {name: store.read_rows(f"episodes_l5_held_aside_{name}_problems.jsonl") for name in MODELS}
    successes = {name: 6 + place for place, name in enumerate(MODELS)}          # the scripted solver's fifth, sixth and seventh call for the held-aside set: not seed 0's 3, 4, 5
    assert {name: own[0]["resolved"] for name, own in on_the_set.items()} == successes and all(summaries[f"ladder_l5_held_aside_{name}"]["sampling_seed"] == 1051 for name in MODELS)
    report = summaries["ladder_l5_report"]
    assert list(report) == [*REPORT_KEYS[:-1], "a_further_seed", "timed_out_in_lean", "adapters"] and (report["seed"], report["measured"], report["ok"]) == (seed, list(MODELS), True)
    assert json.loads(store.path("report_ladder_l5.json").read_text()) == json.loads(json.dumps(report, default=str)) and report["a_further_seed"] == of_it
    assert report["inconclusive"] is False and report["primary"]["difference"] == per_attempt(on_the_set[DEEP], on_the_set[PUBLISHED], [HELD], RESAMPLES_OF_THE_RUN, 0)
    against = report["the_held_aside_set"]["against_the_start"]
    assert {name: (own["per_attempt"]["successes"], own["per_attempt"]["successes_of_the_base"]) for name, own in against.items()} == {name: (successes[name], ON_THE_SET["pre_r64"]) for name in MODELS}
    assert "those of the run of seed 0, which made the search: only read here" in report["the_stored_model"]["pre_r64"]
    lines = report["lines"]
    assert all(line.startswith(f"{SAY}: ") for line in lines) and f"Seed {seed}, A FURTHER SEED of the scout of seed 0, which made the search: no round. THE DEEP SEARCH" in lines[0]
    assert lines[1].startswith(f"{SAY}: A FURTHER SEED: seed {seed} reads the deep search, the held-aside set and the measurement of `pre_r64` on it from the run of seed 0 ({further.root})")
    assert "which 1 of the search's 1 deep solves and which verified attempt of each" in lines[1] and "python -m rlvr_lean.tools.ladder_l5_seeds" in lines[1]
    (timeouts,) = [line for line in lines if line.startswith(f"{SAY}: TIMEOUTS, the share of each measured set's attempts that timed out in Lean: ")]
    assert list(report["timed_out_in_lean"]["share"]) == list(report["attempts"]) and all(f"{key} " in timeouts for key in report["attempts"]) and report["timed_out_in_lean"]["apart"] == []
    assert timeouts.endswith("WHERE TWO MODELS' SHARES DIFFER BY MORE THAN A POINT: nowhere") and report["adapters"]["there"] == sorted(MODELS)
    assert store.done_summary("ladder_l5_report")["measured"] == list(MODELS)

    # ---- queued again: every step returns what is stored (the two that make nothing say it again); nothing is sampled or trained; seed 0's run is as it was
    trained, asked = len(calls), len(solver.asked)
    again = _stage5(config, monkeypatch)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in summaries.items() if "report" not in name}
    assert (len(calls), len(solver.asked)) == (trained, asked) and again["ladder_l5_report"]["branches"] == report["branches"] and {path: _all(path) for path in read_only} == read_only


RESAMPLES_OF_THE_RUN = CONFIG["evaluation"]["bootstrap_resamples"]


def test_the_cap_keeps_the_deep_solves_of_the_tasks_own_draw_and_the_same_number_as_seed_0(further, deep, monkeypatch):  # noqa: F811
    config = deep.config
    # Another world for seed 0: THREE candidates, two searched and both solved, a cap of one. Seed 0's run is made again from its first step.
    for path in further.root.iterdir():
        path.unlink() if path.is_file() else None
    _write_the_map(config, {"fixture_p1": (), "fixture_p2": (), "fixture_p3": (), "fixture_p4": (5,)})
    config["ladder_loop"]["l5"]["deep"].update({"problems": 2, "cap": 1})
    of_seed_0 = _stage5(config, monkeypatch)
    searched = [row["problem_id"] for row in _file_rows(further.root / "l5_search_problems.jsonl")]
    assert of_seed_0["ladder_l5_prepare"]["deep_solves"] == 2 and of_seed_0["ladder_l5_report"]["inconclusive"] is False and len(searched) == 2
    kept_at = lambda seed: min(searched, key=lambda key: rank(seed, CAP_LABEL, key))      # noqa: E731
    seed = next(number for number in range(1, 200) if kept_at(number) != kept_at(0))
    monkeypatch.setenv(SEEDS, str(seed))
    summaries = _stage5(config, monkeypatch, until="ladder_l5_prepare")
    store = further.own()
    # The search's problems are SEED 0's draw (not this seed's), and of its two deep solves the cap keeps the one of THIS seed's draw: another than seed 0's.
    assert [row["problem_id"] for row in store.read_rows("l5_search_problems.jsonl")] == searched == sorted(("fixture_p1", "fixture_p2", "fixture_p3"), key=lambda key: rank(0, SEARCH_LABEL, key))[:2]
    kept = [row["problem_id"] for row in store.read_rows("l5_training_deep.jsonl") if row["origin"] == DEEP]
    assert kept == [kept_at(seed)] != [row["problem_id"] for row in _file_rows(further.root / "l5_training_deep.jsonl") if row["origin"] == DEEP]
    assert {row["problem_id"]: row["in_the_set"] for row in store.read_rows("l5_deep_solves.jsonl")} == {key: key == kept_at(seed) for key in searched}
    assert (summaries["ladder_l5_prepare"]["deep_solves"], summaries["ladder_l5_prepare"]["rows_of_each_kind"]) == (2, 1)
    # ANOTHER CAP than the one seed 0's sets were built under is refused by the first step: the models of the two seeds would not be trained on as many rows.
    for path in store.root.iterdir():
        path.unlink()
    config["ladder_loop"]["l5"]["deep"]["cap"] = 2
    with pytest.raises(RuntimeError, match="was made with another start model, rank, search or held-aside set than this task would make: the cap and the minimum of deep solves .that run: .1, 1.; this task: .2, 1.."):
        ladder_l5.ladder_l5_search_prepare(config)
    assert not any(store.root.iterdir())


def test_the_held_aside_set_is_seed_0s_draw_where_another_seed_would_draw_another(further, deep, monkeypatch):  # noqa: F811
    config = deep.config
    # Another world for seed 0: THREE candidates, one searched, and ONE of the two it leaves held aside: a draw with a choice. Seed 0's run is made again from its first step.
    for path in further.root.iterdir():
        path.unlink() if path.is_file() else None
    _write_the_map(config, {"fixture_p1": (), "fixture_p2": (), "fixture_p3": (), "fixture_p4": (2, 5)})
    of_seed_0 = _stage5(config, monkeypatch)
    (searched,), (held,) = ([row["problem_id"] for row in _file_rows(further.root / name)] for name in ("l5_search_problems.jsonl", "l5_held_aside_problems.jsonl"))
    left = [key for key in ("fixture_p1", "fixture_p2", "fixture_p3") if key != searched]
    drawn_at = lambda seed: min(left, key=lambda key: rank(seed, HELD_ASIDE_LABEL, key))      # noqa: E731
    assert held == drawn_at(0) and of_seed_0["ladder_l5_report"]["inconclusive"] is False and of_seed_0["ladder_l5_search_prepare"]["the_held_aside"]["candidates_not_searched"] == 2
    seed = next(number for number in range(1, 200) if drawn_at(number) != held and min(("fixture_p1", "fixture_p2", "fixture_p3"), key=lambda key: rank(number, SEARCH_LABEL, key)) != searched)
    monkeypatch.setenv(SEEDS, str(seed))
    read = ladder_l5.ladder_l5_search_prepare(config)                           # NOT refused: both draws are made again with SEED 0, and are the stored ones
    store = further.own()
    assert [row["problem_id"] for row in store.read_rows("l5_held_aside_problems.jsonl")] == [held] != [drawn_at(seed)] and read["a_further_seed"]["search_seed"] == 0
    assert [row["problem_id"] for row in store.read_rows("l5_search_problems.jsonl")] == [searched]
    assert {row["problem_id"] for row in store.read_rows("problems.jsonl") if row["set"].startswith("l5_held_aside_")} == {held}


def test_at_the_searchs_own_seed_nothing_is_a_further_seed_and_a_task_of_the_seed_the_setting_names_makes_its_own_search(further, monkeypatch):
    config, first, settings = further.config, further.first, further.config["ladder_loop"]["l5"]
    # SEED 0's OWN PATH: no summary and no report says anything of a further seed; its files, keys and lines are what `test_ladder_l5_stage.py` holds them to.
    assert not [step for step, summary in first.items() if "a_further_seed" in summary or "timed_out_in_lean" in summary] and list(first["ladder_l5_report"]) == REPORT_KEYS
    assert not [line for line in first["ladder_l5_report"]["lines"] if "FURTHER SEED" in line or "TIMEOUTS" in line] and "Seed 0, a scout: one seed, no round. THE DEEP SEARCH" in first["ladder_l5_report"]["lines"][0]
    assert sorted(name for name in _all(further.root) if not name.startswith("episodes_")) == OWN_FILES and not (further.root / FURTHER_FILE).exists()
    assert "Its attempts at the held-aside set are THIS run's (its 8 stored map attempts chose the set)" in first["ladder_l5_report"]["the_stored_model"]["pre_r64"]
    own = ladder_l5.the_settings(config)
    assert own.seed_of_the_search == 0 == settings["search_seed"] and not ladder_l5.a_further_seed(config, own) and ladder_l5.own_run(config, own) == "ladder_l5_deep_pre_r64_seed0"
    assert ladder_l5.search_directory(config, own) == further.root == further.own().root
    # A TASK OF ANOTHER SEED is a further seed: a run of its own, the search's run and L4t2's of the SEARCH's seed, the search's seed's two sampling seeds.
    monkeypatch.setenv(SEEDS, "2")
    assert ladder_l5.a_further_seed(config, own) and ladder_l5.own_run(config, own) == "ladder_l5_deep_pre_r64_seed2" and ladder_l5.search_directory(config, own) == further.root
    assert ladder_l5.l4t2_directory(config, own).name == "ladder_l4_rows2_pre_r64_seed0" and (ladder_l5.search_seed(config, own), ladder_l5.held_aside_seed(config, own)) == (1050, 1051)
    # ... UNLESS the setting names ITS seed: then it is the search's own task, simply: its own search (sampling seeds of its own), L4t2's run of its seed, no run of seed 0 read.
    monkeypatch.setitem(settings, "search_seed", 2)
    own = ladder_l5.the_settings(config)
    assert not ladder_l5.a_further_seed(config, own) and ladder_l5.search_directory(config, own).name == ladder_l5.own_run(config, own) == "ladder_l5_deep_pre_r64_seed2"
    assert (ladder_l5.search_seed(config, own), ladder_l5.held_aside_seed(config, own)) == (1250, 1251) and ladder_l5.l4t2_directory(config, own).name == "ladder_l4_rows2_pre_r64_seed2"
    with pytest.raises(RuntimeError, match="ladder_l4_rows2_pre_r64_seed2 does not hold .*it reads what the task of stage `ladder_l4_rows2_r64` for seed 2"):
        ladder_l5.ladder_l5_search_prepare(config)                              # the search's own path: it looks for what IT reads, and for no finished run of this stage
    monkeypatch.delenv(SEEDS)
    assert ladder_l5.a_further_seed(config, own) and ladder_l5.own_run(config, own) == "ladder_l5_deep_pre_r64_seed0"                 # ... and seed 0 would now be the further one
    # A RUN THE TASK NAMES (a smoke run) at a further seed writes BESIDE it and reads it: the named run is the search's own, and is never written over.
    monkeypatch.setitem(settings, "search_seed", 0)
    monkeypatch.setenv(ladder_l5.RUN_VARIABLE, "ladder_l5_deep_smoke")
    own = ladder_l5.the_settings(config)
    assert ladder_l5.own_run(config, own) == "ladder_l5_deep_smoke" == ladder_l5.search_directory(config, own).name
    monkeypatch.setenv(SEEDS, "1")
    assert ladder_l5.own_run(config, own) == "ladder_l5_deep_smoke_seed1" and ladder_l5.search_directory(config, own).name == "ladder_l5_deep_smoke"
    with pytest.raises(RuntimeError, match="ladder_l5_deep_smoke does not hold .*the FINISHED run of the deep search.*for the smoke run, stage `ladder_l5_deep_smoke`"):
        ladder_l5.ladder_l5_search_prepare(config)
    # The setting is a seed: a whole number of at least 0.
    for wrong in (-1, True, "0", 1.0, None):
        monkeypatch.setitem(settings, "search_seed", wrong)
        with pytest.raises(ValueError, match="ladder_loop.l5.search_seed is .*: it is the seed of the ONE task that makes the deep search"):
            ladder_l5.the_settings(config)


def test_a_further_seed_is_refused_before_anything_is_written_when_seed_0s_run_is_not_one_it_can_stand_beside(further, monkeypatch):
    config, root, settings = further.config, further.root, further.config["ladder_loop"]["l5"]["deep"]
    monkeypatch.setenv(SEEDS, str(further.seed))
    own_root = further.runs / f"ladder_l5_deep_pre_r64_seed{further.seed}"
    task = "Run the task of stage `ladder_l5_deep` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l5_deep --seeds 0`; for the smoke run, stage `ladder_l5_deep_smoke`. to its end first; nothing was written"

    def refused(match):
        with pytest.raises(RuntimeError, match=match):
            ladder_l5.ladder_l5_search_prepare(config)
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
    # ---- MISSING: no run of seed 0 at all
    root.rename(root.with_name("elsewhere"))
    refused(f"ladder_l5_deep_pre_r64_seed0 does not hold .*the FINISHED run of the deep search. This task's seed is {further.seed} and ladder_loop.l5.search_seed is 0: a further seed makes no "
            f"search of its own.*{task}")
    root.with_name("elsewhere").rename(root)
    # ---- UNFINISHED: a step's marker, its report, a draw, its problems, or a stored sampling is not there
    for name in ("ladder_l5_search_prepare.done.json", "ladder_l5_search.done.json", "ladder_l5_prepare.done.json", "ladder_l5_held_aside_start.done.json", "ladder_l5_report.done.json",
                 "report_ladder_l5.json", "problems.jsonl", "l5_search_problems.jsonl", "l5_held_aside_problems.jsonl", "episodes_l5_deep_search_problems.jsonl",
                 "episodes_l5_deep_search.done.json", "episodes_l5_deep_search_attempts_0000.jsonl", "episodes_l5_held_aside_pre_r64_problems.jsonl",
                 "episodes_l5_held_aside_pre_r64.done.json", "episodes_l5_held_aside_pre_r64_attempts_0000.jsonl"):
        without(root / name, f"ladder_l5_deep_pre_r64_seed0 does not hold .*{name.replace('_0000', '_.')}.*the FINISHED run of the deep search.*{task}")
    # ---- A REPORT NOT TO BE READ, or INCONCLUSIVE (a search that found too few deep solves is one)
    report = root / "report_ladder_l5.json"
    with_changed(report, as_json(lambda own: {**own, "ok": False}), "the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is not to be read .Lean gave no verdict on too much of a set.: "
                 "no further seed stands beside such a run.*Queue the task of stage `ladder_l5_deep` for seed 0 .* again, or repair that run; nothing was written")
    with_changed(report, as_json(lambda own: {**own, "inconclusive": True, "failed_checks": ["`shallow`: its_training_ran"]}),
                 "the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is INCONCLUSIVE .`shallow`: its_training_ran.: no further seed stands beside such a run")
    # ---- ANOTHER START MODEL, SEED OR MODELS than the search's own
    first = root / "ladder_l5_search_prepare.done.json"
    with_changed(first, as_json(lambda own: {**own, "start": "pre"}), "the run ladder_l5_deep_pre_r64_seed0 recorded seed 0, the start 'pre' and the models .*ladder_loop.l5 names the search's seed 0, "
                 "the start 'pre_r64'")
    with_changed(first, as_json(lambda own: {**own, "seed": 3}), "the run ladder_l5_deep_pre_r64_seed0 recorded seed 3, the start 'pre_r64'")
    with_changed(first, as_json(lambda own: {**own, "models": ["deep", "shallow", "published"]}), "the run ladder_l5_deep_pre_r64_seed0 recorded seed 0, the start 'pre_r64' and the models .'deep', 'shallow', 'published'.")
    # ---- ANOTHER RANK, SEARCH OR HELD-ASIDE SET than this task would make: what the run recorded, held to what this task reads now
    other = "the deep search's run ladder_l5_deep_pre_r64_seed0 was made with another start model, rank, search or held-aside set than this task would make: "
    inside = lambda key, **changed: as_json(lambda own: {**own, key: {**own[key], **changed}})      # noqa: E731
    with_changed(first, inside("start_recipe", rank=32), other + "the start adapter's rank and alpha .that run: .32, 128.; this task: .64, 128..*Nothing was written")
    with_changed(first, inside("start_recipe", alpha=64), other + "the start adapter's rank and alpha .that run: .64, 64.")
    with_changed(first, inside("the_search", episodes_each=40), other + "the search's attempts a problem and its sampling seed .that run: .40, 1050.; this task: .56, 1050..")
    with_changed(first, inside("the_search", sampling_seed=1150), other + "the search's attempts a problem and its sampling seed .that run: .56, 1150.")
    with_changed(first, inside("the_held_aside", episodes_each=8), other + "the held-aside set's attempts a problem and its sampling seed .that run: .8, 1051.; this task: .16, 1051..")
    with_changed(first, inside("the_held_aside", sampling_seed=1151), other + "the held-aside set's attempts a problem and its sampling seed .that run: .16, 1151.")
    with_changed(first, inside("the_search", cap=300), other + "the cap and the minimum of deep solves .that run: .300, 1.; this task: .400, 1..")
    with_changed(first, inside("the_search", minimum_deep_solves=150), other + "the cap and the minimum of deep solves .that run: .400, 150.")
    with_changed(first, inside("the_map", problems_by_k={"0": 4}), other + "the map's problems by k")
    with_changed(first, as_json(lambda own: {**own, "rung_episodes": 4}), other + "how a model is measured")
    with_changed(first, as_json(lambda own: {**own, "pretraining_file_sha256": "0" * 64}), other + "the pretraining file's SHA-256")
    with_changed(first, as_json(lambda own: {**own, "goal_set_again": 7}), other + "the size of G'")
    # The two draws, made again from the map with the SEARCH's seed and held to the stored ones: a search of another problem, a held-aside set of another.
    with_changed(root / "l5_search_problems.jsonl", as_rows(lambda own: [{**own[0], "problem_id": HELD}]), other + "the problems searched .that run: 1 problems; this task: 1 problems.")
    with_changed(root / "l5_held_aside_problems.jsonl", as_rows(lambda own: [{**own[0], "problem_id": SEARCHED}]), other + "the held-aside problems .that run: 1 problems")
    for key, value, said in (("attempts", 40, "the search's attempts a problem and its sampling seed .that run: .56, 1050.; this task: .40, 1050.."),
                             ("held_aside_attempts", 8, "the held-aside set's attempts a problem and its sampling seed .that run: .16, 1051.; this task: .8, 1051.."),
                             ("sampling_place", 52, "the search's attempts a problem and its sampling seed .that run: .56, 1050.; this task: .56, 1052.."),
                             ("held_aside_sampling_place", 53, "the held-aside set's attempts a problem and its sampling seed .that run: .16, 1051.; this task: .16, 1053.."),
                             ("cap", 5, "the cap and the minimum of deep solves .that run: .400, 1.; this task: .5, 1..")):
        kept = settings[key]
        monkeypatch.setitem(settings, key, value)
        refused(other + said)
        monkeypatch.setitem(settings, key, kept)
    # ---- the two stored samplings themselves: made with another seed, or on other problems, than the search's own step recorded
    of = lambda name: root / f"episodes_{name}.done.json"      # noqa: E731
    with_changed(of("l5_deep_search"), as_json(lambda own: {**own, "sampling_seed": 7}), "ladder_l5_deep_pre_r64_seed0 measured the deep search of `pre_r64` with sampling seed 7 and 56 episodes; this config gives 1050 and 56")
    with_changed(of("l5_held_aside_pre_r64"), as_json(lambda own: {**own, "episodes_each": 8}), "ladder_l5_deep_pre_r64_seed0 measured `pre_r64` on the held-aside set with sampling seed 1051 and 8 episodes; this config gives 1051 and 16")
    with_changed(root / "episodes_l5_held_aside_pre_r64_problems.jsonl", as_rows(lambda own: [{**own[0], "problem_id": SEARCHED}]), "ladder_l5_deep_pre_r64_seed0 measured `pre_r64` on the held-aside set on other problems")
    with_changed(root / "episodes_l5_held_aside_pre_r64_problems.jsonl", as_rows(lambda own: [{**own[0], "attempts_without_an_answer": 9}]), "with too many attempts without a verdict from Lean")
    # ---- L4t2's run is the SEARCH's seed's: a run that recorded this task's seed is not it
    of_l4t2 = further.deep.l4t2 / "ladder_l4_rows2_prepare.done.json"
    with_changed(of_l4t2, as_json(lambda own: {**own, "seed": further.seed}), f"L4t2's run ladder_l4_rows2_pre_r64_seed0 recorded seed {further.seed} and the start 'pre_r64'; this task's seed is "
                 f"{further.seed} .a further seed: it reads the runs the search's task of seed 0 read.")
    # ---- and after every refusal the step runs to its end; the second step holds the search it finds in the stored rows to the one seed 0's task counted
    read = ladder_l5.ladder_l5_search_prepare(config)
    assert read["a_further_seed"]["search_seed"] == 0 and (own_root / "ladder_l5_search_prepare.done.json").exists()
    results = root / "episodes_l5_deep_search_problems.jsonl"
    kept = results.read_text()
    results.write_text(as_rows(lambda own: [{**own[0], "resolved": own[0]["resolved"] + 1}])(kept))
    with pytest.raises(RuntimeError, match="the deep search read from ladder_l5_deep_pre_r64_seed0 is not the one its own task built its sets on.*Nothing was written"):
        ladder_l5.ladder_l5_prepare(config)
    assert not (own_root / "l5_deep_solves.jsonl").exists() and not (own_root / "ladder_l5_prepare.done.json").exists()
    results.write_text(kept)
    assert ladder_l5.ladder_l5_prepare(config)["rows_of_each_kind"] == 1
    # A later step before its own, at a further seed: the report needs its three models, and no step of the start model's.
    with pytest.raises(RuntimeError, match="L5's report needs the step ladder_l5_train_published of this run"):
        ladder_l5.ladder_l5_report(config)


# -------------------------------------------------------------------------------------- the rules of the read
def test_the_names_of_the_read_over_the_seeds_are_the_sections():
    assert (DEEP_ADDS, NOT_SHOWN_AT_THREE, THEY_COST) == ("DEEP SOLVES ADD WHAT THE SAME NUMBER OF PUBLISHED PROOFS DO NOT", "NOT SHOWN AT THREE SEEDS", "THEY COST")
    assert OVER_THE_SEEDS == (DEEP_ADDS, NOT_SHOWN_AT_THREE, THEY_COST, NOT_READ) and LEAD == (LEAD_CONFIRMED, NOT_CONFIRMED, NOT_READ) == ("THE LEAD IS CONFIRMED", "NOT CONFIRMED", NOT_READ)
    assert (RELIABILITY, REACH_TOO) == ("RELIABILITY", "REACH TOO") and REACH == (REACH_CONFIRMED, NOT_CONFIRMED, NOT_READ) and REACH_CONFIRMED == "REACH IS CONFIRMED ON THE LONG PROBLEMS"
    assert AFTER_THE_SEEDS == (A_LOOP_OF_DEEP_SEARCHES, RELIABILITY_ALONE, STOP, OTHERWISE, NOT_READ) and (NEW_SEEDS, A_POINT, LONG) == (2, 0.01, "8+")
    assert list(SPREAD_OF) == ["all_of_g_per_1000", "goal_problems_solved", "held_aside_per_1000", "goal_set_again_per_1000"]


def test_the_primary_over_the_seeds_has_three_names_and_is_read_over_three_seeds_and_no_other_number():
    three = {0: 0.001, 1: -0.002, 2: 0.004}
    for difference, name in ((ABOVE, DEEP_ADDS), (HOLDS, NOT_SHOWN_AT_THREE), (BELOW, THEY_COST), (ENDS_AT_ZERO, NOT_SHOWN_AT_THREE), ({**BELOW, "high": 0.0}, NOT_SHOWN_AT_THREE)):
        read = primary_over_the_seeds(difference, three)
        assert read["name"] == name and read["seeds"] == [0, 1, 2] and read["signs"] == {"positive": [0, 2], "negative": [1]} and "seed 1 -0.00200" in read["reason"]
    assert primary_over_the_seeds(HOLDS, three)["resolves"] == 0.02 and "resolves" not in primary_over_the_seeds(ABOVE, three)
    # Over one seed, two, or four: NOT READ, however clear the interval; and when nothing was measured.
    for at_each in ({0: 0.02}, {0: 0.02, 1: 0.02}, {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02}, {}):
        read = primary_over_the_seeds(ABOVE, at_each)
        assert read["name"] == NOT_READ and f"the primary is fixed over 3 seeds (the search's own and 2 further) and {len(at_each)} are read here" in read["reason"]
    assert primary_over_the_seeds(NOT_MEASURED, three)["name"] == NOT_READ and all(primary_over_the_seeds(ABOVE, three)["name"] in OVER_THE_SEEDS for _ in (0,))


def test_the_lead_is_confirmed_by_an_interval_above_zero_and_a_positive_difference_at_both_new_seeds_and_by_nothing_less():
    both = {1: 0.01, 2: 0.02}
    confirmed = the_lead(ABOVE, both)
    assert confirmed["name"] == LEAD_CONFIRMED and (confirmed["interval_above_zero"], confirmed["positive_at_both_new_seeds"], confirmed["seeds"]) == (True, True, [1, 2])
    # EACH condition by itself is not enough: the interval above zero with one seed not positive (negative, or exactly zero); both positive with an interval that holds or ends at zero.
    for difference, at_each, above, at_both in ((ABOVE, {1: 0.04, 2: -0.001}, True, False), (ABOVE, {1: 0.04, 2: 0.0}, True, False), (HOLDS, both, False, True), (ENDS_AT_ZERO, both, False, True),
                                                (BELOW, {1: -0.01, 2: -0.03}, False, False)):
        read = the_lead(difference, at_each)
        assert (read["name"], read["interval_above_zero"], read["positive_at_both_new_seeds"]) == (NOT_CONFIRMED, above, at_both)
    assert "the interval is above zero and the difference is not positive at both new seeds" in the_lead(ABOVE, {1: 0.04, 2: -0.001})["reason"]
    # ON THE TWO NEW SEEDS ALONE: one new seed, none, or three, is NOT READ (seed 0 found the lead, so a read with it among the seeds confirms nothing).
    for at_each in ({1: 0.02}, {}, {0: 0.02, 1: 0.02, 2: 0.02}):
        read = the_lead(ABOVE, at_each)
        assert read["name"] == NOT_READ and f"on the 2 new seeds alone, and {len(at_each)} of them are read here" in read["reason"]
    assert the_lead(NOT_MEASURED, both)["name"] == NOT_READ


def test_a_confirmed_lead_is_named_by_its_split_and_one_that_is_not_confirmed_has_no_name():
    confirmed, unconfirmed, unread = the_lead(ABOVE, {1: 0.01, 2: 0.02}), the_lead(HOLDS, {1: 0.01, 2: 0.02}), the_lead(ABOVE, {1: 0.01})
    assert the_split(confirmed, HOLDS)["name"] == RELIABILITY and the_split(confirmed, ENDS_AT_ZERO)["name"] == RELIABILITY and the_split(confirmed, ABOVE)["name"] == REACH_TOO
    below = the_split(confirmed, BELOW)                                         # the section names two cases; below zero is reliability too, and says it is below
    assert below["name"] == RELIABILITY and below["below_zero_on_the_goal_set_again"] is True and "BELOW zero" in below["reason"] and the_split(confirmed, HOLDS)["below_zero_on_the_goal_set_again"] is False
    for lead in (unconfirmed, unread):
        assert the_split(lead, ABOVE) == {"name": None, "reason": f"the lead reads {lead['name']}: its split names nothing"} and the_split(lead, HOLDS)["name"] is None
    assert the_split(confirmed, NOT_MEASURED)["name"] is None


def test_reach_on_the_long_problems_is_confirmed_by_an_interval_above_zero_and_more_solved_at_both_new_seeds():
    confirmed = reach_on_the_long(ABOVE, {1: (5, 3), 2: (4, 3)})
    assert confirmed["name"] == REACH_CONFIRMED and (confirmed["interval_above_zero"], confirmed["more_at_both_new_seeds"], confirmed["seeds"]) == (True, True, [1, 2])
    assert "long problems solved, `deep` against `published`: seed 1 5 against 3, seed 2 4 against 3" in confirmed["reason"]
    # EACH condition by itself is not enough; AS MANY at a seed is not more.
    for difference, counts, above, at_both in ((ABOVE, {1: (5, 3), 2: (3, 3)}, True, False), (ABOVE, {1: (5, 3), 2: (2, 3)}, True, False), (HOLDS, {1: (5, 3), 2: (4, 3)}, False, True),
                                               (ENDS_AT_ZERO, {1: (5, 3), 2: (4, 3)}, False, True)):
        read = reach_on_the_long(difference, counts)
        assert (read["name"], read["interval_above_zero"], read["more_at_both_new_seeds"]) == (NOT_CONFIRMED, above, at_both)
    for counts in ({1: (5, 3)}, {}, {0: (5, 3), 1: (5, 3), 2: (5, 3)}):
        assert reach_on_the_long(ABOVE, counts)["name"] == NOT_READ
    assert reach_on_the_long(NOT_MEASURED, {1: (5, 3), 2: (4, 3)})["name"] == NOT_READ
    assert "`deep` minus `shallow`" in reach_on_the_long(ABOVE, {1: (5, 3), 2: (4, 3)}, after=DEEP, before=SHALLOW)["reason"]


def test_what_follows_the_seeds_is_the_amended_branch_in_the_sections_order():
    three, both, counts = {0: 0.01, 1: 0.01, 2: 0.01}, {1: 0.01, 2: 0.02}, {1: (5, 3), 2: (4, 3)}
    adds, held, cost = (primary_over_the_seeds(difference, three) for difference in (ABOVE, HOLDS, BELOW))
    lead, no_lead = the_lead(ABOVE, both), the_lead(HOLDS, both)
    reliability, reach_too, unnamed = the_split(lead, HOLDS), the_split(lead, ABOVE), the_split(no_lead, HOLDS)
    reach, no_reach = reach_on_the_long(ABOVE, counts), reach_on_the_long(HOLDS, counts)
    # REACH CONFIRMED ON THE LONG PROBLEMS, or ADD on the held-aside set: the loop of deep searches, whatever else is read (the section's order: even under THEY COST).
    for given in ((held, lead, reliability, reach), (held, no_lead, unnamed, reach), (cost, no_lead, unnamed, reach), (adds, no_lead, unnamed, no_reach), (adds, lead, reliability, no_reach)):
        follows = after_the_seeds(*given)
        assert follows["name"] == A_LOOP_OF_DEEP_SEARCHES and "with ordinary solves matched in length as a control" in follows["reason"]
    # The lead confirmed as RELIABILITY and reach not: before THEY COST, in the section's order.
    for primary in (held, cost):
        follows = after_the_seeds(primary, lead, reliability, no_reach)
        assert follows["name"] == RELIABILITY_ALONE and "the lead reads THE LEAD IS CONFIRMED (RELIABILITY)" in follows["reason"] and "the depth of the search is the thing to raise" in follows["reason"]
    # THEY COST on the held-aside set: stop.
    for lead_, split in ((no_lead, unnamed), (lead, reach_too)):
        assert after_the_seeds(cost, lead_, split, no_reach)["name"] == STOP and "the three seeds need a decision" in after_the_seeds(cost, lead_, split, no_reach)["reason"]
    # OTHERWISE: the lead not confirmed; or (which the section names no branch for) confirmed as reach too with reach on the long problems not confirmed.
    otherwise = after_the_seeds(held, no_lead, unnamed, no_reach)
    assert otherwise["name"] == OTHERWISE and "the lead is not confirmed, deep solves are no worse than published rows at equal steps" in otherwise["reason"]
    corner = after_the_seeds(held, lead, reach_too, no_reach)
    assert corner["name"] == OTHERWISE and "the section names no branch of its own for a lead confirmed as reach too" in corner["reason"] and "the lead is not confirmed" not in corner["reason"]
    # Nothing is named while a read it stands on is not read.
    unread = primary_over_the_seeds(ABOVE, {0: 0.01})
    for given in ((unread, lead, reliability, reach), (adds, the_lead(ABOVE, {1: 0.01}), unnamed, reach), (adds, lead, reliability, reach_on_the_long(ABOVE, {1: (5, 3)}))):
        assert after_the_seeds(*given)["name"] == NOT_READ
    assert all(after_the_seeds(primary, lead_, split, reach_)["name"] in AFTER_THE_SEEDS for primary in (adds, held, cost) for lead_, split in ((lead, reliability), (lead, reach_too), (no_lead, unnamed))
               for reach_ in (reach, no_reach))


def test_the_share_of_attempts_that_timed_out_is_given_for_every_set_and_the_pairs_more_than_a_point_apart_are_named():
    health = {"rungs_pre": {"attempts": 1000, "timed_out_in_lean": 10}, "rungs_deep": {"attempts": 1000, "timed_out_in_lean": 20}, "rungs_shallow": {"attempts": 1000, "timed_out_in_lean": 21},
              "goal_pre_1": {"attempts": 2000, "timed_out_in_lean": 100}, "goal_deep_1": {"attempts": 2000, "timed_out_in_lean": 40}, "goal_shallow_1": {"attempts": 0, "timed_out_in_lean": 0}}
    kinds = {"rungs": {"pre": "rungs_pre", "deep": "rungs_deep", "shallow": "rungs_shallow"}, "goal_1": {"pre": "goal_pre_1", "deep": "goal_deep_1", "shallow": "goal_shallow_1"}}
    found = timed_out(health, kinds)
    assert found["share"] == {"rungs_pre": 0.01, "rungs_deep": 0.02, "rungs_shallow": 0.021, "goal_pre_1": 0.05, "goal_deep_1": 0.02, "goal_shallow_1": None} and found["a_point"] == 0.01
    # EXACTLY a point apart is not more than a point (0.01 against 0.02); 1.1 points is; a set with no attempt is in no pair; two sets of different kinds are never compared.
    assert found["apart"] == [{"kind": "rungs", "models": ["pre", "shallow"], "shares": [0.01, 0.021]}, {"kind": "goal_1", "models": ["pre", "deep"], "shares": [0.05, 0.02]}]
    assert sets_by_kind("pre_r64", MODELS, 2) == {"rungs": {name: f"rungs_{name}" for name in FOUR}, "goal_1": {name: f"goal_{name}_1" for name in FOUR},
                                                  "goal_2": {name: f"goal_{name}_2" for name in FOUR}, "held_aside": {name: f"held_aside_{name}" for name in FOUR}}


# ------------------------------------------------------------------------- the read over the seeds, on hand-made rows
# On G (12 goal problems; G' is the last four, the long problems the last three): successes in (the 32 attempts, the 61 more). ONE more than the start model on every
# problem; one more on the first eight alone (the problems NOT in G'); and the start model with the first long problem unsolved, or the last two.
ONE_MORE = ([own + 1 for own in START[0]], [own + 1 for own in START[1]])
ON_THE_SOLVED = ([own + (index < 8) for index, own in enumerate(START[0])], [own + (index < 8) for index, own in enumerate(START[1])])
WITHOUT = lambda *lost: ([0 if index in lost else own for index, own in enumerate(START[0])], [0 if index in lost else own for index, own in enumerate(START[1])])      # noqa: E731
SEARCHED_ROWS = [{"problem_id": f"m{index}", "map_k": 0, "map_n": 8, "taken_by": "the rule"} for index in range(20)]
SOLVES = {f"m{index}": [1 + index % 3, 56, 9 + 5 * index] for index in range(20)}
ASIDE_ROWS = [{"problem_id": key, "map_k": 0, "map_n": 8, "taken_by": "the rule"} for key in ASIDE]
A_FURTHER_SEED = lambda seed: {"seed": seed, "search_seed": 0, "search_run": "runs/ladder_l5_deep_pre_r64_seed0"}      # noqa: E731


def _seed(seed, on_g=None, on_ha=None, **changed):
    """One seed of L5 as `over_seeds` reads it: its report (built on its own three models), the one search, the one
    held-aside set, G and G', and each model's rows. `on_g`, `on_ha`: by trained model, its successes on G (two
    samplings) and on the twelve held-aside problems; the start model's own stand in for a model that is not given."""
    models = {name: _model(on_g=(on_g or {}).get(name, START)) for name in MODELS}
    aside = {name: (on_ha or {}).get(name, HA_BASE) for name in MODELS}
    report = _report(models=models, aside=aside, prepare={**PREPARE, "seed": seed, **({"a_further_seed": A_FURTHER_SEED(seed)} if seed else {})})
    return {"report": report, "searched": SEARCHED_ROWS, "solves": SOLVES, "aside": ASIDE_ROWS, "groups": GROUPS, "again": AGAIN,
            "held_aside": {"pre_r64": _aside(HA_BASE), **{name: _aside(aside[name]) for name in MODELS}},
            "goal": {"pre_r64": _model()["goal"], **{name: models[name]["goal"] for name in MODELS}}, **changed}


def _three(**of_the_new):
    """Seed 0 found a lead (`deep` one more on every goal problem); seeds 1 and 2 as given, the start model's own rows otherwise."""
    return {0: _seed(0, on_g={DEEP: ONE_MORE}), 1: _seed(1, **{key: value[0] for key, value in of_the_new.items()}), 2: _seed(2, **{key: value[1] for key, value in of_the_new.items()})}


def _read(by_seed):
    return over_seeds(by_seed, LENGTHS, RESAMPLES, BOOTSTRAP_SEED)


def test_the_hand_made_world_is_what_the_arithmetic_below_assumes():
    assert AGAIN == GOAL_IDS[8:] and [key for key in GOAL_IDS if LENGTHS[key]["length_group"] == LONG] == GOAL_IDS[9:] and len(ASIDE) == 12 and HA_BASE == [2] * 12
    assert all(first == 0 for first in START[0][8:]) and all(more > 0 for more in START[1]) and (HA_MORE, HA_LESS) == ([3] * 12, [1] * 12)
    assert _seed(0)["report"]["inconclusive"] is False and "a_further_seed" not in _seed(0)["report"] and _seed(1)["report"]["a_further_seed"]["search_seed"] == 0


def test_each_seed_is_read_from_its_rows_and_the_primary_is_each_problems_mean_over_the_three_seeds():
    by_seed = {0: _seed(0, on_ha={DEEP: HA_MORE}), 1: _seed(1), 2: _seed(2, on_ha={DEEP: HA_LESS})}
    read = _read(by_seed)
    assert (read["seeds"], read["seeds_read"], read["made_the_search"], read["new_seeds"], read["start"]) == ([0, 1, 2], [0, 1, 2], 0, [1, 2], "pre_r64")
    assert read["sizes"] == {"aside": 12, "goal": 12, "again": 4, "not_again": 8, "long": 3} and read["not_pooled"] is None and read["could_not_read"] == []
    # EACH SEED: its report's primary, computed again from its rows; every model's figures; every pair's.
    for seed, entry in by_seed.items():
        own = read["by_seed"][seed]
        assert (own["read"], own["made_the_search"], own["branches"]) == (True, seed == 0, entry["report"]["branches"])
        assert own["pairs"][pair_name(DEEP, PUBLISHED)]["on_the_held_aside_set"] == entry["report"]["primary"]["difference"]
        assert list(own["models"]) == list(FOUR) and list(own["pairs"]) == [pair_name(DEEP, PUBLISHED), pair_name(DEEP, SHALLOW), pair_name(SHALLOW, PUBLISHED)]
        assert own["models"]["pre_r64"] == {"held_aside_per_1000": 125.0, "all_of_g_per_1000": round(1000 * sum(map(sum, START)) / (12 * 93), 2), "goal_problems_solved": 12,
                                            "goal_set_again_per_1000": round(1000 * sum(START[1][8:]) / (4 * 61), 2), "long_problems_solved": 3}
    assert [read["by_seed"][seed]["models"][DEEP]["held_aside_per_1000"] for seed in (0, 1, 2)] == [187.5, 125.0, 62.5]
    # THE PRIMARY: `deep` minus `published` on the held-aside set, each problem's attempts ADDED over the three seeds: +1, 0 and -1 of 16 at every problem is 0 of 48.
    primary = read["pooled"]["the_primary"]
    added = lambda name: attempts_on_g([by_seed[seed]["held_aside"][name] for seed in (0, 1, 2)])      # noqa: E731
    assert primary["difference"] == per_attempt(added(DEEP), added(PUBLISHED), sorted(ASIDE), RESAMPLES, BOOTSTRAP_SEED) and primary["difference"]["attempts_each"] == 12 * 48
    assert (primary["difference"]["mean"], primary["difference"]["successes"], primary["difference"]["successes_of_the_base"]) == (0.0, 72, 72)
    assert primary["at_each_seed"] == {0: 0.0625, 1: 0.0, 2: -0.0625} and primary["name"] == NOT_SHOWN_AT_THREE and primary["signs"] == {"positive": [0], "negative": [2]}
    # One more at EVERY seed: above zero, and named so; one fewer: THEY COST.
    assert _read({seed: _seed(seed, on_ha={DEEP: HA_MORE}) for seed in (0, 1, 2)})["pooled"]["the_primary"]["name"] == DEEP_ADDS
    assert _read({seed: _seed(seed, on_ha={DEEP: HA_LESS}) for seed in (0, 1, 2)})["pooled"]["the_primary"]["name"] == THEY_COST
    # It is `deep` MINUS `published`: `published` above `deep` at every seed costs, and `shallow` moves nothing.
    assert _read({seed: _seed(seed, on_ha={PUBLISHED: HA_MORE}) for seed in (0, 1, 2)})["pooled"]["the_primary"]["name"] == THEY_COST
    assert _read({seed: _seed(seed, on_ha={SHALLOW: HA_MORE}) for seed in (0, 1, 2)})["pooled"]["the_primary"]["difference"]["mean"] == 0.0
    # The order the seeds are given in changes nothing.
    assert _read({seed: by_seed[seed] for seed in (2, 0, 1)}) == read
    # THE HELD-ASIDE PROBLEMS ARE READ IN THE ORDER A SEED'S REPORT READS THEM (sorted), whatever order the draw stored them in: a bootstrap over problems is a draw of places.
    uneven = {seed: _seed(seed, on_ha={DEEP: [9, 0, 2, 2, 5, 2, 2, 1, 2, 2, 3, 2]}, aside=ASIDE_ROWS[::-1]) for seed in (0, 1, 2)}
    in_the_draws_order = _read(uneven)
    assert in_the_draws_order["by_seed"][0]["pairs"][pair_name(DEEP, PUBLISHED)]["on_the_held_aside_set"] == uneven[0]["report"]["primary"]["difference"]
    assert uneven[0]["report"]["primary"]["difference"]["low"] < uneven[0]["report"]["primary"]["difference"]["mean"] < uneven[0]["report"]["primary"]["difference"]["high"]
    assert in_the_draws_order["pooled"]["the_primary"]["difference"] == _read({seed: {**entry, "aside": ASIDE_ROWS} for seed, entry in uneven.items()})["pooled"]["the_primary"]["difference"]
    assert all(line.startswith(f"{SAY}: ") for line in read["lines"]) and read["lines"][0].startswith(f"{SAY}: L5, THE READ OVER 3 SEEDS (0, 1, 2), each from `pre_r64`. Seed 0 made the ONE deep search")
    assert any("seed 0 (IT MADE THE SEARCH AND FOUND THE LEAD)" in line for line in read["lines"]) and sum(" (a new seed)" in line for line in read["lines"]) == 2
    assert any(f"THE PRIMARY, on the held-aside set, `deep` minus `published`, each problem's mean over the seeds 0, 1, 2 of its difference" in line and NOT_SHOWN_AT_THREE in line for line in read["lines"])


def test_the_lead_is_read_on_the_two_new_seeds_alone_and_seed_0_which_found_it_confirms_nothing():
    # Seed 0 has the lead and the two new seeds have NOTHING: not confirmed, though every seed pooled is above zero only by seed 0.
    nothing = _read(_three())
    lead = nothing["pooled"]["the_lead"]
    assert lead["name"] == NOT_CONFIRMED and lead["difference"]["mean"] == 0.0 and lead["at_each_new_seed"] == {1: 0.0, 2: 0.0} and lead["seeds"] == [1, 2]
    assert lead["every_seed_pooled"]["mean"] > 0 and lead["every_seed_pooled"]["low"] > 0 and nothing["by_seed"][0]["pairs"][pair_name(DEEP, PUBLISHED)]["on_all_of_g"]["low"] > 0
    assert nothing["pooled"]["the_split"]["name"] is None and nothing["pooled"]["what_follows"]["name"] == OTHERWISE
    # BOTH new seeds have it: confirmed, on each problem's attempts added over the TWO new seeds (186 of them), whatever seed 0 read.
    by_seed = _three(on_g=({DEEP: ONE_MORE}, {DEEP: ONE_MORE}))
    both = _read(by_seed)
    lead = both["pooled"]["the_lead"]
    added = lambda name, seeds=(1, 2): attempts_on_g([rows_ for seed in seeds for rows_ in by_seed[seed]["goal"][name]])      # noqa: E731
    assert lead["difference"] == per_attempt(added(DEEP), added(PUBLISHED), GOAL_IDS, RESAMPLES, BOOTSTRAP_SEED) and lead["difference"]["attempts_each"] == 12 * 186
    assert lead["name"] == LEAD_CONFIRMED and lead["at_each_new_seed"] == {1: lead["difference"]["mean"], 2: lead["difference"]["mean"]} and lead["difference"]["low"] > 0
    assert lead["every_seed_pooled"] == per_attempt(added(DEEP, (0, 1, 2)), added(PUBLISHED, (0, 1, 2)), GOAL_IDS, RESAMPLES, BOOTSTRAP_SEED)
    assert list(lead["goal_problems_solved_at_each_seed"]) == [0, 1, 2] and lead["goal_problems_solved_at_each_seed"][1] == solved(added(DEEP, (1,)), added(PUBLISHED, (1,)), GOAL_IDS)
    with_seed_0_changed = _read({**by_seed, 0: _seed(0, on_g={DEEP: LESS})})    # seed 0 BELOW `published`: the lead on the new seeds is what it was
    assert with_seed_0_changed["pooled"]["the_lead"]["difference"] == lead["difference"] and with_seed_0_changed["pooled"]["the_lead"]["name"] == LEAD_CONFIRMED
    # ONE new seed has it and the other is below: the difference is not positive at both, and it is not confirmed.
    mixed = _read(_three(on_g=({DEEP: MORE}, {DEEP: LESS})))["pooled"]["the_lead"]
    assert mixed["name"] == NOT_CONFIRMED and mixed["at_each_new_seed"][1] > 0 > mixed["at_each_new_seed"][2] and mixed["positive_at_both_new_seeds"] is False
    # BESIDE IT, naming nothing: the same two reads of `deep` minus `shallow` and of `shallow` minus `published`.
    beside = both["pooled"]["beside_the_lead"]
    assert list(beside) == [pair_name(DEEP, SHALLOW), pair_name(SHALLOW, PUBLISHED)] and all("name" not in own for own in beside.values())
    assert beside[pair_name(DEEP, SHALLOW)]["difference"] == lead["difference"] and beside[pair_name(SHALLOW, PUBLISHED)]["difference"]["mean"] == 0.0
    assert beside[pair_name(SHALLOW, PUBLISHED)]["at_each_new_seed"] == {1: 0.0, 2: 0.0} and list(beside[pair_name(DEEP, SHALLOW)]["goal_problems_solved_at_each_seed"]) == [0, 1, 2]
    assert any("THE GOAL SET'S LEAD, on the new seeds ALONE (1, 2)" in line and "THE LEAD IS CONFIRMED, REACH TOO" in line for line in both["lines"])
    assert any("seed 0 FOUND the lead and confirms nothing" in line for line in both["lines"])


def test_the_split_reads_the_lead_on_the_goal_problems_not_in_the_goal_set_again_and_on_it_and_names_a_confirmed_lead():
    # One more on EVERY goal problem at both new seeds: above zero on G' too: REACH TOO. One more on the first eight alone (not in G'): RELIABILITY.
    for own, name, on_again in ((ONE_MORE, REACH_TOO, 2 / 93), (ON_THE_SOLVED, RELIABILITY, 0.0)):      # one more in each of the two samplings: two of 93
        by_seed = _three(on_g=({DEEP: own}, {DEEP: own}))
        split = _read(by_seed)["pooled"]["the_split"]
        added = lambda name_: attempts_on_g([rows_ for seed in (1, 2) for rows_ in by_seed[seed]["goal"][name_]])      # noqa: E731
        assert split["on_the_goal_problems_not_in_the_goal_set_again"] == per_attempt(added(DEEP), added(PUBLISHED), GOAL_IDS[:8], RESAMPLES, BOOTSTRAP_SEED)
        assert split["on_the_goal_set_again"] == per_attempt(added(DEEP), added(PUBLISHED), AGAIN, RESAMPLES, BOOTSTRAP_SEED)       # over ALL the attempts, 93 a seed
        assert split["name"] == name and split["on_the_goal_set_again"]["mean"] == round(on_again, 5) and split["on_the_goal_set_again"]["attempts_each"] == 4 * 186
        assert split["on_the_goal_problems_not_in_the_goal_set_again"]["mean"] == round(2 / 93, 5) and split["on_the_goal_problems_not_in_the_goal_set_again"]["problems"] == 8
    reliability = _read(_three(on_g=({DEEP: ON_THE_SOLVED}, {DEEP: ON_THE_SOLVED})))
    assert reliability["pooled"]["the_lead"]["name"] == LEAD_CONFIRMED and reliability["pooled"]["what_follows"]["name"] == RELIABILITY_ALONE
    assert any("THE LEAD IS CONFIRMED, RELIABILITY" in line for line in reliability["lines"]) and any("(a) on the 8 goal problems NOT in G'" in line and ": RELIABILITY." in line for line in reliability["lines"])
    # A lead that is not confirmed has NO NAME, whatever its split reads.
    unnamed = _read(_three(on_g=({DEEP: ONE_MORE}, {})))
    assert unnamed["pooled"]["the_lead"]["name"] == NOT_CONFIRMED and unnamed["pooled"]["the_split"]["name"] is None and unnamed["pooled"]["the_split"]["on_the_goal_set_again"]["mean"] > 0
    assert any("THE SPLIT of the lead" in line and ": NO NAME." in line for line in unnamed["lines"])


def test_reach_on_the_long_problems_is_the_share_of_the_new_seeds_in_which_a_problem_is_solved_paired_over_the_long_problems():
    # `published` solves NO long problem at either new seed and `deep` all three: a share of 1 against 0 on each; more at both seeds: CONFIRMED.
    lost = WITHOUT(9, 10, 11)
    confirmed = _read(_three(on_g=({PUBLISHED: lost}, {PUBLISHED: lost})))
    reach = confirmed["pooled"]["reach_on_the_long_problems"]
    assert (reach["name"], reach["difference"]["mean"], reach["difference"]["problems"], reach["counts_at_each_new_seed"]) == (REACH_CONFIRMED, 1.0, 3, {1: (3, 0), 2: (3, 0)})
    assert confirmed["pooled"]["what_follows"]["name"] == A_LOOP_OF_DEEP_SEARCHES and confirmed["by_seed"][1]["models"][PUBLISHED]["long_problems_solved"] == 0
    assert confirmed["by_seed"][1]["pairs"][pair_name(DEEP, PUBLISHED)]["long_problems_solved"]["gained"] == 3
    # `published` loses ONE long problem at seed 1 alone: a share of 1 against 1/2 on one problem of three (+1/6); `deep` solves more at seed 1 and AS MANY at seed 2: NOT CONFIRMED.
    one_seed = _read(_three(on_g=({PUBLISHED: WITHOUT(9)}, {})))["pooled"]["reach_on_the_long_problems"]
    assert (one_seed["name"], one_seed["difference"]["mean"], one_seed["counts_at_each_new_seed"], one_seed["more_at_both_new_seeds"]) == (NOT_CONFIRMED, round(1 / 6, 5), {1: (3, 2), 2: (3, 3)}, False)
    # It is read on the NEW seeds: seed 0 alone losing every long problem moves nothing, and is not confirmed.
    seed_0_alone = _read({**_three(), 0: _seed(0, on_g={PUBLISHED: lost})})["pooled"]["reach_on_the_long_problems"]
    assert (seed_0_alone["name"], seed_0_alone["difference"]["mean"], seed_0_alone["counts_at_each_new_seed"]) == (NOT_CONFIRMED, 0.0, {1: (3, 3), 2: (3, 3)})
    # A problem NOT of 8 lines or more is not counted: `published` losing the first eight goal problems at both new seeds reads no reach.
    short = _read(_three(on_g=({PUBLISHED: WITHOUT(*range(8))}, {PUBLISHED: WITHOUT(*range(8))})))["pooled"]["reach_on_the_long_problems"]
    assert (short["name"], short["difference"]["mean"]) == (NOT_CONFIRMED, 0.0)
    # BESIDE IT, naming nothing: the same read of `deep` minus `shallow`.
    beside = _read(_three(on_g=({SHALLOW: lost}, {SHALLOW: lost})))["pooled"]
    assert beside["reach_on_the_long_problems"]["name"] == NOT_CONFIRMED and list(beside["beside_the_reach"]) == [pair_name(DEEP, SHALLOW)]
    own = beside["beside_the_reach"][pair_name(DEEP, SHALLOW)]
    assert "name" not in own and (own["difference"]["mean"], own["counts_at_each_new_seed"], own["interval_above_zero"], own["more_at_both_new_seeds"]) == (1.0, {1: (3, 0), 2: (3, 0)}, True, True)
    assert any("REACH ON THE LONG PROBLEMS (3 goal problems whose shortest published proof has 8 lines or more), on the new seeds alone: REACH IS CONFIRMED ON THE LONG PROBLEMS" in line
               for line in confirmed["lines"]) and any("REACH, BESIDE IT: long problems solved at each seed: seed 0: `published` 3, `shallow` 3, `deep` 3" in line for line in confirmed["lines"])


def test_the_spread_is_each_kind_of_models_figures_over_its_seeds_and_their_standard_deviation():
    by_seed = {0: _seed(0, on_ha={DEEP: HA_MORE}), 1: _seed(1), 2: _seed(2, on_ha={DEEP: HA_LESS}, on_g={DEEP: WITHOUT(0)})}
    spread = _read(by_seed)["pooled"]["the_spread"]
    assert list(spread) == list(MODELS) and all(list(own) == list(SPREAD_OF) for own in spread.values())
    assert spread[DEEP]["held_aside_per_1000"] == {"by_seed": {0: 187.5, 1: 125.0, 2: 62.5}, "mean": 125.0, "standard_deviation": 62.5}       # the SAMPLE's, n - 1 (a population's is 51.031)
    assert spread[DEEP]["goal_problems_solved"] == {"by_seed": {0: 12, 1: 12, 2: 11}, "mean": 11.667, "standard_deviation": 0.577}
    assert spread[PUBLISHED]["held_aside_per_1000"]["standard_deviation"] == 0.0 and spread[SHALLOW]["all_of_g_per_1000"]["by_seed"] == {seed: round(1000 * sum(map(sum, START)) / (12 * 93), 2) for seed in (0, 1, 2)}
    assert spread[DEEP]["goal_set_again_per_1000"]["by_seed"][1] == round(1000 * sum(START[1][8:]) / (4 * 61), 2)


def test_with_fewer_seeds_the_reads_that_can_be_made_are_made_and_the_others_are_said_not_to_be():
    # SEED 0 ALONE: its own figures; nothing over the seeds is named.
    alone = _read({0: _seed(0, on_g={DEEP: ONE_MORE})})
    pooled = alone["pooled"]
    assert (alone["seeds"], alone["new_seeds"]) == ([0], []) and [pooled[key]["name"] for key in ("the_primary", "the_lead", "reach_on_the_long_problems", "what_follows")] == [NOT_READ] * 4
    assert pooled["the_split"]["name"] is None and pooled["the_lead"]["difference"]["mean"] is None and pooled["the_lead"]["every_seed_pooled"]["mean"] > 0
    assert pooled["the_primary"]["difference"] == alone["by_seed"][0]["pairs"][pair_name(DEEP, PUBLISHED)]["on_the_held_aside_set"]
    assert pooled["the_spread"][DEEP]["all_of_g_per_1000"]["standard_deviation"] is None and len(alone["could_not_read"]) == 6
    assert [said.split(" (")[0] for said in alone["could_not_read"]] == ["the primary over the three seeds", "the goal set's lead, confirmed or not", "the split of the lead",
                                                                        "reach on the long problems", "the spread", "the branch for what follows"]
    assert alone["lines"][-1].startswith(f"{SAY}: NOT READ, for the seeds that are not here: the primary over the three seeds (it takes the seed that made the search, 0, and 2 further; here: 0)")
    # SEED 0 AND ONE NEW SEED: still nothing is named (the primary takes three, the lead two NEW ones); the spread is given.
    two = _read({0: _seed(0), 1: _seed(1, on_g={DEEP: ONE_MORE})})
    assert [two["pooled"][key]["name"] for key in ("the_primary", "the_lead", "reach_on_the_long_problems", "what_follows")] == [NOT_READ] * 4 and two["new_seeds"] == [1]
    assert two["pooled"]["the_spread"][DEEP]["goal_problems_solved"]["standard_deviation"] == 0.0 and "the spread (a standard deviation takes two seeds)" not in two["could_not_read"]
    # THE TWO NEW SEEDS without seed 0: the lead, its split and the reach ARE read (they are the new seeds' alone); the primary is not, nor the branch.
    new = _read({1: _seed(1, on_g={DEEP: ONE_MORE}), 2: _seed(2, on_g={DEEP: ONE_MORE})})
    assert (new["made_the_search"], new["new_seeds"]) == (0, [1, 2]) and new["pooled"]["the_lead"]["name"] == LEAD_CONFIRMED and new["pooled"]["the_split"]["name"] == REACH_TOO
    assert new["pooled"]["reach_on_the_long_problems"]["name"] == NOT_CONFIRMED and new["pooled"]["the_primary"]["name"] == NOT_READ == new["pooled"]["what_follows"]["name"]
    assert [said.split(" (")[0] for said in new["could_not_read"]] == ["the primary over the three seeds", "the branch for what follows"]
    # THREE further seeds without the one that made the search: three seeds, and still no primary.
    assert _read({seed: _seed(seed, on_ha={DEEP: HA_MORE}) for seed in (1, 2, 3)})["pooled"]["the_primary"]["name"] == NOT_READ
    with pytest.raises(ValueError, match="the read over the seeds of L5 was given no seed"):
        _read({})


def test_the_read_is_refused_for_seeds_that_do_not_share_one_search_one_held_aside_set_one_start_model_or_the_rows_their_reports_read():
    by_seed = _three()
    for key, other, what in (("searched", SEARCHED_ROWS[:-1], "search"), ("solves", {**SOLVES, "m0": [9, 56, 9]}, "search"), ("aside", ASIDE_ROWS[:-1], "held-aside set"),
                             ("again", AGAIN[:-1], "G and G'"), ("groups", GROUPS[:-1], "G and G'")):
        with pytest.raises(ValueError, match=f"seed 2 does not hold the ONE {what} seed 0 holds .{key} differ.: every seed of L5 reads one search, one held-aside set and one held-out goal set"):
            _read({**by_seed, 2: {**by_seed[2], key: other}})
    # ONE measurement of the start model: on the held-aside set (the search's own run's) and on G.
    for changed in ({"held_aside": {**by_seed[1]["held_aside"], "pre_r64": _aside(HA_MORE)}}, {"goal": {**by_seed[1]["goal"], "pre_r64": _model(on_g=MORE)["goal"]}}):
        with pytest.raises(ValueError, match="seed 1 holds other rows of `pre_r64` than seed 0: every seed of L5 is read against the ONE measurement of `pre_r64`"):
            _read({**by_seed, 1: {**by_seed[1], **changed}})
    # TWO seeds that each made a search (neither report says it is a further seed), or a further seed of another search.
    with pytest.raises(ValueError, match="the seeds do not share ONE search: by seed, the seed whose task made the search it read is .0: 0, 1: 1, 2: 0."):
        _read({**by_seed, 1: {**by_seed[1], "report": {key: value for key, value in by_seed[1]["report"].items() if key != "a_further_seed"}}})
    with pytest.raises(ValueError, match="the seeds do not share ONE search"):
        _read({**by_seed, 2: {**by_seed[2], "report": {**by_seed[2]["report"], "a_further_seed": {**A_FURTHER_SEED(2), "search_seed": 1}}}})
    with pytest.raises(ValueError, match="the seeds are of other start models"):
        _read({**by_seed, 2: {**by_seed[2], "report": {**by_seed[2]["report"], "start": "pre"}}})
    # Rows that are not the ones a seed's report read; and other bootstrap settings than the runs'.
    with pytest.raises(ValueError, match="seed 1: the primary computed from its stored rows is not its report's .*these are not the rows that report read"):
        _read({**by_seed, 1: {**by_seed[1], "held_aside": {**by_seed[1]["held_aside"], DEEP: _aside(HA_MORE)}}})
    with pytest.raises(ValueError, match="seed 0: the primary computed from its stored rows is not its report's"):
        over_seeds({0: _seed(0, on_ha={DEEP: [9, 0, 2, 2, 5, 2, 2, 1, 2, 2, 3, 2]})}, LENGTHS, RESAMPLES, BOOTSTRAP_SEED + 1)
    # Another number of attempts at a seed: a problem's mean over the seeds is not its attempts added.
    fewer = [{**row, "episodes": 8} for row in _aside(HA_BASE)]
    with pytest.raises(ValueError, match="seed 2 made other numbers of attempts for `published` than seed 0: a problem's mean over the seeds is read on its attempts added over them"):
        _read({**by_seed, 2: {**by_seed[2], "held_aside": {**by_seed[2]["held_aside"], PUBLISHED: fewer}}})
    with pytest.raises(ValueError, match="seed 1 made other numbers of attempts for `deep` than seed 0"):
        _read({**by_seed, 1: {**by_seed[1], "goal": {**by_seed[1]["goal"], DEEP: [by_seed[1]["goal"][DEEP][0], [{**row, "episodes": 60} for row in by_seed[1]["goal"][DEEP][1]]]}}})


def test_a_seed_that_is_inconclusive_or_not_to_be_read_is_named_and_nothing_is_pooled():
    by_seed = _three(on_g=({DEEP: ONE_MORE}, {DEEP: ONE_MORE}))
    for changed, why in (({"inconclusive": True}, INCONCLUSIVE), ({"ok": False}, NOT_TO_BE_READ), ({"primary": {"difference": {"mean": None}}}, NOT_READ)):
        read = _read({**by_seed, 1: {"report": {**by_seed[1]["report"], **changed}}})           # of such a run the report alone is handed on
        assert read["pooled"] is None and read["not_pooled"] == {"1": why} and (read["seeds"], read["seeds_read"], read["new_seeds"]) == ([0, 1, 2], [0, 2], [2])
        assert read["by_seed"][1] == {"read": False, "why_not": why, "made_the_search": False, "branches": by_seed[1]["report"]["branches"]} and read["by_seed"][2]["read"] is True
        assert f"{SAY}: seed 1: {why}: nothing is said of what its models did, and nothing is pooled over it" in read["lines"]
        assert read["lines"][-1] == f"{SAY}: NOT POOLED: seed 1 ({why}): nothing is said of such a run, and nothing is pooled over it"
        assert not [line for line in read["lines"] if "THE PRIMARY" in line or "THE GOAL SET'S LEAD" in line or "WHAT FOLLOWS" in line or "THE SPREAD" in line]


# ------------------------------------------------------------------------------------------------ the command
def _task_directory(root, seed, entry, name=None):
    """What is pulled for one task of the stage `ladder_l5_deep`: a directory with its step files under `steps`."""
    steps = root / (name or f"ladder_l5_deep_seed{seed}_r1") / "steps"
    steps.mkdir(parents=True)
    write = lambda file, rows_: (steps / file).write_text("".join(json.dumps(row) + "\n" for row in rows_))      # noqa: E731
    (steps / "report_ladder_l5.json").write_text(json.dumps(entry["report"], default=str))
    if "searched" not in entry:
        return steps.parent
    write("l5_search_problems.jsonl", entry["searched"])
    write("l5_deep_solves.jsonl", [{"problem_id": key, "k": k, "n": n, "first_success": first, "completion": "  a proof of the model's own\n", "in_the_set": seed == 0}
                                   for key, (k, n, first) in entry["solves"].items()])
    write("l5_held_aside_problems.jsonl", entry["aside"])
    write("l4_heldout_groups.jsonl", entry["groups"])
    write("l4_goal_set_again.jsonl", [{"problem_id": key} for key in entry["again"]])
    for name_ in MODELS:
        write(f"episodes_l5_held_aside_{name_}_problems.jsonl", entry["held_aside"][name_])
        for sampling, rows_ in zip(("reach", "more"), entry["goal"][name_]):
            write(f"episodes_l4_{sampling}_{name_}_problems.jsonl", rows_)
    # The start model on the held-aside set: measured by the seed that made the search; COPIED by a further seed's first step.
    write(FURTHER_FILE if entry["report"].get("a_further_seed") else "episodes_l5_held_aside_pre_r64_problems.jsonl", entry["held_aside"]["pre_r64"])
    for sampling, rows_ in zip(("reach", "more"), entry["goal"]["pre_r64"]):
        write(f"l4_stored_pre_r64_{sampling}.jsonl", rows_)
    return steps.parent


@pytest.fixture
def pulled(tmp_path):
    config, lengths = tmp_path / "experiment.yaml", tmp_path / "lengths.jsonl"
    config.write_text(json.dumps({"evaluation": EVALUATION}))                                      # the runs' own bootstrap settings (YAML reads JSON)
    lengths.write_text("".join(json.dumps({"problem_id": key, **value}) + "\n" for key, value in LENGTHS.items()))
    by_seed = _three(on_g=({DEEP: ONE_MORE}, {DEEP: ON_THE_SOLVED}))
    return tmp_path, ["--config", str(config), "--lengths", str(lengths)], by_seed, {seed: _task_directory(tmp_path, seed, entry) for seed, entry in by_seed.items()}


def test_the_command_reads_the_pulled_task_directories_that_are_there_and_prints_the_read(pulled, capsys):
    root, settings, by_seed, directories = pulled
    assert ladder_l5_seeds.FILES == ("l5_search_problems.jsonl", "l5_deep_solves.jsonl", "l5_held_aside_problems.jsonl", "l4_heldout_groups.jsonl", "l4_goal_set_again.jsonl")
    before = _all(root)
    assert ladder_l5_seeds.main([*(str(directories[seed]) for seed in (0, 1, 2)), *settings]) == 0
    printed = capsys.readouterr()
    expected = _read(by_seed)
    assert printed.out.splitlines() == expected["lines"] and printed.err == "" and _all(root) == before          # the directories are only read; nothing is written
    assert expected["pooled"]["the_lead"]["name"] == LEAD_CONFIRMED and expected["pooled"]["the_split"]["name"] == REACH_TOO
    # A `steps` directory named in a task directory's place, the seeds in another order, and `--out`: the same read, written where asked and nowhere else.
    out = root / "elsewhere" / "read.json"
    assert ladder_l5_seeds.main([str(directories[2] / "steps"), str(directories[0]), str(directories[1] / "steps"), *settings, "--out", str(out)]) == 0
    assert capsys.readouterr().out.splitlines() == expected["lines"]
    written = json.loads(out.read_text())
    assert written == json.loads(json.dumps({**expected, "directories": {str(seed): str(directories[seed] / "steps") for seed in (0, 1, 2)}}, default=str))
    assert written["seeds"] == [0, 1, 2] and list(written["by_seed"]) == ["0", "1", "2"] and written["pooled"]["the_lead"]["name"] == LEAD_CONFIRMED
    assert {name: content for name, content in _all(root).items() if not name.startswith("elsewhere")} == before
    assert "a proof of the model's own" not in out.read_text()                                     # of a deep solve the read takes its k, its n and its first success, never a text
    # FEWER SEEDS: seed 0 alone, and the two new seeds without it: each is the read of the seeds that are there.
    assert ladder_l5_seeds.main([str(directories[0]), *settings]) == 0
    assert capsys.readouterr().out.splitlines() == _read({0: by_seed[0]})["lines"]
    assert ladder_l5_seeds.main([str(directories[2]), str(directories[1]), *settings]) == 0
    assert capsys.readouterr().out.splitlines() == _read({1: by_seed[1], 2: by_seed[2]})["lines"]
    # The start model on the held-aside set is read where each kind of seed has it; the seed is the report's own, whatever the directory is called.
    seed, entry = ladder_l5_seeds.read_seed(directories[1])
    assert seed == 1 and entry["held_aside"]["pre_r64"] == by_seed[1]["held_aside"]["pre_r64"] and not (directories[1] / "steps" / "episodes_l5_held_aside_pre_r64_problems.jsonl").exists()
    assert entry["solves"] == SOLVES and entry["again"] == AGAIN and list(entry["goal"]) == list(FOUR) and ladder_l5_seeds.steps_directory(directories[1]) == directories[1] / "steps"
    # The package's own config and its own lengths are the defaults.
    assert ladder_l5_seeds.CONFIG.name == "experiment.yaml" and ladder_l5_seeds.CONFIG.is_file()


def test_the_command_refuses_a_directory_that_is_not_a_finished_task_of_the_stage_and_what_the_read_refuses(pulled, capsys):
    root, settings, by_seed, directories = pulled
    paths = [str(directories[seed]) for seed in (0, 1, 2)]

    def refused(arguments, said):
        assert ladder_l5_seeds.main([*arguments, *settings]) == 2
        printed = capsys.readouterr()
        assert printed.out == "" and printed.err.startswith("refused: ") and said in printed.err

    # A task that did not reach its report delivered no read: each missing file is named.
    for name in ("report_ladder_l5.json", *ladder_l5_seeds.FILES, FURTHER_FILE, "episodes_l5_held_aside_deep_problems.jsonl", "episodes_l4_more_published_problems.jsonl", "l4_stored_pre_r64_reach.jsonl"):
        lost = directories[1] / "steps" / name
        lost.rename(lost.with_name("elsewhere"))
        refused(paths, f"does not hold {name}" if name == "report_ladder_l5.json" else f"does not hold ['{name}']: it is not the pulled directory of a task of the stage `ladder_l5_deep` that wrote its report")
        lost.with_name("elsewhere").rename(lost)
    # The seed that MADE the search holds the start model's own measurement, not a copy.
    lost = directories[0] / "steps" / "episodes_l5_held_aside_pre_r64_problems.jsonl"
    lost.rename(lost.with_name(FURTHER_FILE))
    refused(paths, "does not hold ['episodes_l5_held_aside_pre_r64_problems.jsonl']")
    lost.with_name(FURTHER_FILE).rename(lost)
    # A report of another stage; two directories of one seed; what the read itself refuses.
    report = directories[2] / "steps" / "report_ladder_l5.json"
    kept = report.read_text()
    report.write_text(json.dumps({**json.loads(kept), "check": "L4p"}))
    refused(paths, "is not a report of L5 with its seed and its start model")
    report.write_text(kept)
    again = _task_directory(root, 1, by_seed[1], name="ladder_l5_deep_seed1_r2")
    refused([*paths, str(again)], "are both of seed 1: one task directory a seed")
    other = _task_directory(root, 2, {**by_seed[2], "aside": ASIDE_ROWS[:-1]}, name="another_held_aside_set")
    refused([paths[0], paths[1], str(other)], "seed 2 does not hold the ONE held-aside set seed 0 holds")
    # A seed whose report is INCONCLUSIVE: its report alone is read (its other files may not be there), it is named, and nothing is pooled.
    inconclusive = _task_directory(root, 2, {"report": {**by_seed[2]["report"], "inconclusive": True}}, name="an_inconclusive_seed")
    assert ladder_l5_seeds.main([paths[0], paths[1], str(inconclusive), *settings]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == f"{SAY}: NOT POOLED: seed 2 ({INCONCLUSIVE}): nothing is said of such a run, and nothing is pooled over it"
