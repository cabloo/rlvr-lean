"""L4, the read over the seeds of the arm on hand-made rows with a known answer, and its command on hand-made task
directories: each seed's primary held to its report's, the pooled read (each problem's mean over the seeds of its
difference, bootstrap over problems), each seed's sign and own interval, what is refused, and a seed that is not read.
Spec: docs/spec/ladder-loop.spec.md, "L4: the loop from a model pretrained on published proofs", "Further seeds of
the arm, made exact before they run" ("The read over the three seeds, fixed now"). Pure: no model, no Lean. A further
seed's own report is `test_ladder_l4_report.py`; the arm at a further seed, end to end, is `test_ladder_l4_stage.py`."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GOAL_IDS  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l4 import AGAIN, ON_G, OF_THE_OTHER_HALF, TRAININGS, _made  # noqa: E402 - the rules' hand-made world
from test_ladder_l4_report import BASE_ARM, FURTHER, OWN, OWN_1, PREPARE, PREPARE_1, _report  # noqa: E402 - the arm's report on it, at seed 0 and at a further seed

from rlvr_lean.domain.evaluation.bootstrap import paired_bootstrap  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.reporting.ladder_ceiling import per_attempt  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import HELD_TO, NOT_TO_BE_READ, SAY, not_read, over_seeds  # noqa: E402
from rlvr_lean.tools import ladder_l4_seeds  # noqa: E402

RESAMPLES, BOOTSTRAP_SEED = EVALUATION["bootstrap_resamples"], EVALUATION["bootstrap_seed"]
# `with`'s successes over the 61 attempts of the second sampling on G' (g4c, g8a, g8b, g8c), by seed of the arm; `pre`'s there are 1, 0, 0, 0 at every seed.
ON_AGAIN = {0: [3, 2, 1, 0], 1: [2, 2, 0, 1], 2: [1, 0, 3, 0]}
PRE_ON_AGAIN = ON_G["pre"][1][8:]


def _with(on_again):
    return _made("with", on_g=(ON_G["with"][0], [*ON_G["with"][1][:8], *on_again]))


def _seed(seed, on_again=None, **changed):
    """One seed of the arm as `over_seeds` reads it: its report (built on its own `with`), G', and `with`'s and `pre`'s rows on the second sampling of G."""
    model = _with(on_again or ON_AGAIN[seed])
    at = {"prepare": PREPARE, "own": OWN, "base_arm": BASE_ARM} if seed == 0 else {
        "prepare": {**PREPARE_1, "seed": seed}, "own": {**OWN_1, "seed": seed, "a_further_seed": {**FURTHER, "seed": seed}}, "base_arm": None}
    return {"report": _report(**{**at, "with": model, **changed}), "again": list(AGAIN), "with": model["goal"][1], "pre": _made("pre")["goal"][1]}


def _three():
    return {seed: _seed(seed) for seed in (0, 1, 2)}


def test_the_hand_made_world_is_what_the_arithmetic_below_assumes():
    assert AGAIN == GOAL_IDS[8:] == ["g4c", "g8a", "g8b", "g8c"] and PRE_ON_AGAIN == [1, 0, 0, 0] and ON_AGAIN[0] == ON_G["with"][1][8:]
    assert HELD_TO == ("problems", "mean", "low", "high", "successes", "successes_of_the_base", "attempts_each")


def test_each_seeds_primary_is_its_reports_and_the_pooled_read_is_each_problems_mean_over_the_seeds():
    by_seed = _three()
    read = over_seeds(by_seed, RESAMPLES, BOOTSTRAP_SEED)
    assert (read["seeds"], read["goal_set_again"], read["label"], read["stage"]) == ([0, 1, 2], 4, "pretrained on published proofs", "l4")
    # ---- per seed: the primary as that seed's report reads it (6, 5 and 4 successes of `with` against `pre`'s 1, in 4 x 61 attempts each)
    for seed, successes in ((0, 6), (1, 5), (2, 4)):
        entry = read["by_seed"][seed]
        assert entry["read"] is True and entry["why_not"] is None and entry["branch"] == by_seed[seed]["report"]["branch"]["name"]
        primary = entry["primary"]
        assert {key: primary[key] for key in HELD_TO} == {key: by_seed[seed]["report"]["primary"][key] for key in HELD_TO}
        assert (primary["successes"], primary["successes_of_the_base"], primary["attempts_each"], primary["problems"]) == (successes, 1, 244, 4)
        assert primary["mean"] == pytest.approx((successes - 1) / 244, abs=1e-5) and entry["sign"] == "+"
        assert entry["own_interval"] == ("above zero" if primary["low"] > 0 else "below zero" if primary["high"] < 0 else "holds zero")
    # ---- pooled: for each problem of G' the MEAN OVER THE SEEDS of its difference, bootstrap over the four problems
    differences = [sum(ON_AGAIN[seed][index] - PRE_ON_AGAIN[index] for seed in (0, 1, 2)) / 3 / 61 for index in range(4)]
    assert differences == pytest.approx([3 / 183, 4 / 183, 4 / 183, 1 / 183])
    by_hand = paired_bootstrap(differences, resamples=RESAMPLES, seed=BOOTSTRAP_SEED)
    pooled = read["pooled"]
    assert (pooled["mean"], pooled["low"], pooled["high"]) == (round(by_hand.mean, 5), round(by_hand.low, 5), round(by_hand.high, 5)) and pooled["problems"] == 4
    assert pooled["mean"] == pytest.approx(12 / 732, abs=1e-5) == pytest.approx(sum(read["by_seed"][seed]["primary"]["mean"] for seed in (0, 1, 2)) / 3, abs=1e-5)
    # Its totals: `with`'s attempts of the three seeds added, and `pre`'s ONE measurement counted once for each seed.
    assert (pooled["successes"], pooled["successes_of_the_base"], pooled["attempts_each"]) == (15, 3, 732) and (pooled["per_1000"], pooled["per_1000_of_the_base"]) == (20.49, 4.1)
    assert read["not_pooled"] is None
    # ---- beside it: each seed's sign, and the seeds whose own interval is clear of zero
    assert read["signs"] == {"positive": [0, 1, 2], "negative": [], "zero": []}
    clear = read["own_interval_clear_of_zero"]
    assert sorted(clear["above"] + clear["below"] + clear["holds_zero"]) == [0, 1, 2] and clear["below"] == []
    assert clear["above"] == [seed for seed in (0, 1, 2) if read["by_seed"][seed]["primary"]["low"] > 0]
    # ---- the lines: every one labelled; the seeds, then the pooled read, then the counts; NO VERDICT is named from them
    lines = read["lines"]
    assert all(line.startswith(f"{SAY}: ") for line in lines) and len(lines) == 6
    assert lines[0].startswith(f"{SAY}: L4, THE READ OVER 3 SEEDS OF THE ARM (0, 1, 2), each from the ONE pretraining: on the one G' (4 goal problems")
    assert [line.split(": ", 1)[1][:7] for line in lines[1:4]] == ["seed 0:", "seed 1:", "seed 2:"] and "(5 successes against 1 in 244 attempts each)" in lines[2]
    assert lines[4].startswith(f"{SAY}: POOLED over seeds 0, 1, 2, each problem's mean over the seeds of its difference: ") and (
        "(15 successes of `with` against 3 of `pre`, counted once for each seed, in 732 attempts each)") in lines[4]
    assert "of positive sign 3 of 3 (seeds 0, 1, 2), of negative sign 0 (seeds none)" in lines[5] and lines[5].endswith("No verdict is named here: the spec's branches stand")
    assert "All seeds share one measurement of `pre`: they are not independent reads of its side" in lines[5]
    assert not [key for key in read if key in ("branch", "verdict", "headline", "conclusion")]
    # The order the seeds are given in changes nothing; two seeds are read as three are.
    assert over_seeds({seed: by_seed[seed] for seed in (2, 0, 1)}, RESAMPLES, BOOTSTRAP_SEED) == read
    two = over_seeds({seed: by_seed[seed] for seed in (0, 2)}, RESAMPLES, BOOTSTRAP_SEED)
    assert two["seeds"] == [0, 2] and (two["pooled"]["successes"], two["pooled"]["successes_of_the_base"], two["pooled"]["attempts_each"]) == (10, 2, 488)
    assert two["pooled"]["mean"] == pytest.approx(8 / 488, abs=1e-5)


def test_a_seed_below_pre_has_its_own_sign_and_interval_and_pulls_the_pooled_read_down():
    # Seed 2's `with` solves nothing on G' in its second sampling, where `pre` has one success: its difference is negative, on one problem of four.
    by_seed = {**_three(), 2: _seed(2, on_again=[0, 0, 0, 0])}
    read = over_seeds(by_seed, RESAMPLES, BOOTSTRAP_SEED)
    assert read["by_seed"][2]["sign"] == "-" and read["by_seed"][2]["primary"]["mean"] == pytest.approx(-1 / 244, abs=1e-5)
    # ... and its interval ENDS at zero (three of its four problems are level), which holds zero: a negative sign is not an interval below zero.
    assert read["by_seed"][2]["primary"]["low"] < 0 and read["by_seed"][2]["primary"]["high"] == 0 and read["by_seed"][2]["own_interval"] == "holds zero"
    assert read["signs"] == {"positive": [0, 1], "negative": [2], "zero": []} and 2 in read["own_interval_clear_of_zero"]["holds_zero"]
    assert read["own_interval_clear_of_zero"]["below"] == []
    assert read["pooled"]["mean"] == pytest.approx((5 + 4 - 1) / 732, abs=1e-5) and "of negative sign 1 (seeds 2)" in read["lines"][5]
    # One problem down and one up by more at a seed: a positive sign, and an interval with zero inside it.
    mixed = over_seeds({**_three(), 1: _seed(1, on_again=[0, 2, 0, 0])}, RESAMPLES, BOOTSTRAP_SEED)
    primary = mixed["by_seed"][1]["primary"]
    assert primary["low"] < 0 < primary["high"] and (mixed["by_seed"][1]["sign"], mixed["by_seed"][1]["own_interval"]) == ("+", "holds zero")
    assert 1 in mixed["own_interval_clear_of_zero"]["holds_zero"] and 1 not in mixed["own_interval_clear_of_zero"]["above"] + mixed["own_interval_clear_of_zero"]["below"]
    # Every problem of G' lower than `pre` at a seed: its own interval lies below zero, and is counted there.
    stronger_pre = _made("pre", on_g=(ON_G["pre"][0], [*ON_G["pre"][1][:8], 5, 4, 4, 3]))
    below = {seed: {**_seed(seed, pre=stronger_pre), "pre": stronger_pre["goal"][1]} for seed in (0, 1, 2)}
    read = over_seeds(below, RESAMPLES, BOOTSTRAP_SEED)
    assert read["own_interval_clear_of_zero"] == {"above": [], "below": [0, 1, 2], "holds_zero": []} and read["pooled"]["high"] < 0
    # A seed whose `with` equals `pre` on every problem of G': sign zero, an interval that holds zero.
    level = over_seeds({**_three(), 1: _seed(1, on_again=PRE_ON_AGAIN)}, RESAMPLES, BOOTSTRAP_SEED)
    assert level["by_seed"][1]["sign"] == "0" and level["signs"]["zero"] == [1] and level["own_interval_clear_of_zero"]["holds_zero"].count(1) == 1


def test_the_read_over_the_seeds_is_refused_without_one_goal_set_again_one_pre_or_the_rows_its_reports_read():
    by_seed = _three()
    with pytest.raises(ValueError, match=r"needs at least two seeds, and was given \[0\]: one seed's read is its own report's"):
        over_seeds({0: by_seed[0]}, RESAMPLES, BOOTSTRAP_SEED)
    # ONE G': another list at a seed (a problem less, or the same problems in another order) is refused.
    for other in (AGAIN[:-1], list(reversed(AGAIN))):
        with pytest.raises(ValueError, match="seed 1 holds another G' than seed 0 .* every seed of the arm is read on the ONE G' of the pretraining"):
            over_seeds({**by_seed, 1: {**by_seed[1], "again": other}}, RESAMPLES, BOOTSTRAP_SEED)
    # ONE `pre`: other rows of it at a seed (one success more on one problem, anywhere in G; or another number of attempts) are refused.
    for change in ({"resolved": 4}, {"episodes": 60}):
        other = [{**row, **change} if row["problem_id"] == "g1a" else row for row in by_seed[2]["pre"]]
        with pytest.raises(ValueError, match="seed 2 holds other rows of `pre` than seed 0: every seed of the arm is read against the ONE measurement of `pre`"):
            over_seeds({**by_seed, 2: {**by_seed[2], "pre": other}}, RESAMPLES, BOOTSTRAP_SEED)
    # Rows that do not give the report's primary are not the rows that report read: refused, with what differs.
    other = [{**row, "resolved": row["resolved"] + 1} if row["problem_id"] == "g8c" else row for row in by_seed[1]["with"]]
    with pytest.raises(ValueError, match=r"seed 1: the primary computed from its stored rows is not its report's .mean, .*successes: .* these are not the rows that report read"):
        over_seeds({**by_seed, 1: {**by_seed[1], "with": other}}, RESAMPLES, BOOTSTRAP_SEED)
    # ... but a row outside G' is not in the primary and moves nothing (the first eight goal problems here).
    outside = [{**row, "resolved": 0} if row["problem_id"] == "g1a" else row for row in by_seed[1]["with"]]
    assert over_seeds({**by_seed, 1: {**by_seed[1], "with": outside}}, RESAMPLES, BOOTSTRAP_SEED)["by_seed"] == over_seeds(by_seed, RESAMPLES, BOOTSTRAP_SEED)["by_seed"]
    # `with` on other problems than `pre` at a seed cannot be added over the seeds: refused by the function that adds samplings.
    with pytest.raises(ValueError):
        over_seeds({**by_seed, 2: {**by_seed[2], "with": by_seed[2]["with"][:-1]}}, RESAMPLES, BOOTSTRAP_SEED)


def test_a_seed_that_is_inconclusive_or_not_to_be_read_is_named_and_nothing_is_pooled():
    by_seed = _three()
    barred = {**TRAININGS, "M(2)": [*TRAININGS["M(2)"][:-1], OF_THE_OTHER_HALF]}                   # a training row of the `pretrain` half: the fifth check fails
    inconclusive = _seed(1, trained_on=barred)
    assert inconclusive["report"]["inconclusive"] is True and not_read(inconclusive["report"]) == INCONCLUSIVE and not_read(by_seed[0]["report"]) is None
    read = over_seeds({**by_seed, 1: inconclusive}, RESAMPLES, BOOTSTRAP_SEED)
    assert read["by_seed"][1] == {"read": False, "why_not": INCONCLUSIVE, "branch": INCONCLUSIVE, "primary": None}        # nothing is said of such a run
    assert read["pooled"] is None and read["not_pooled"] == {"1": INCONCLUSIVE}
    assert read["by_seed"][0]["read"] is True and read["by_seed"][2]["primary"]["successes"] == 4                          # the seeds that can be read still are
    assert read["signs"] == {"positive": [0, 2], "negative": [], "zero": []} and 1 not in sum(read["own_interval_clear_of_zero"].values(), [])
    lines = read["lines"]
    assert lines[2] == f"{SAY}: seed 1: INCONCLUSIVE: its primary is not given, and nothing is said of what the loop added at this seed"
    assert lines[4] == f"{SAY}: NOT POOLED: seed 1 (INCONCLUSIVE): nothing is said of such a run, and nothing is pooled over it" and "of positive sign 2 of 3 (seeds 0, 2)" in lines[5]
    assert not any("POOLED over" in line for line in lines)
    # A report that is not to be read (Lean gave no verdict on too much of a set) and one with nothing to read its primary on: the same.
    unanswered = _with(ON_AGAIN[2])
    unanswered["goal"][1][0]["attempts_without_an_answer"] = 20
    unread = {**_seed(2), "report": _report(prepare={**PREPARE_1, "seed": 2}, own=OWN_1, base_arm=None, **{"with": unanswered})}
    assert unread["report"]["ok"] is False and not_read(unread["report"]) == NOT_TO_BE_READ == "NOT TO BE READ"
    assert not_read({"ok": True, "inconclusive": False, "primary": {"mean": None}}) == NOT_READ and not_read({"ok": True, "inconclusive": False, "primary": None}) == NOT_READ
    both = over_seeds({0: by_seed[0], 1: inconclusive, 2: unread}, RESAMPLES, BOOTSTRAP_SEED)
    assert both["pooled"] is None and both["not_pooled"] == {"1": INCONCLUSIVE, "2": NOT_TO_BE_READ}
    assert both["lines"][4] == f"{SAY}: NOT POOLED: seeds 1, 2 (INCONCLUSIVE, NOT TO BE READ): nothing is said of such a run, and nothing is pooled over it"
    # The rows of a seed that is not read are not held to anything (its report gives no primary to hold them to); one G' and one `pre` still are.
    with pytest.raises(ValueError, match="seed 1 holds another G' than seed 0"):
        over_seeds({**by_seed, 1: {**inconclusive, "again": AGAIN[:-1]}}, RESAMPLES, BOOTSTRAP_SEED)


def test_the_pooled_read_is_the_primarys_own_function_on_each_problems_attempts_added_over_the_seeds():
    by_seed = _three()
    read = over_seeds(by_seed, RESAMPLES, BOOTSTRAP_SEED)
    added = lambda name: [{"problem_id": problem_id, "resolved": sum(row["resolved"] for seed in by_seed for row in by_seed[seed][name] if row["problem_id"] == problem_id),      # noqa: E731
                           "episodes": sum(row["episodes"] for seed in by_seed for row in by_seed[seed][name] if row["problem_id"] == problem_id)} for problem_id in GOAL_IDS]
    assert all(row["episodes"] == 183 for row in added("with")) and all(row["episodes"] == 183 for row in added("pre"))       # three times 61; `pre`'s 61 once for each seed
    assert read["pooled"] == per_attempt(added("with"), added("pre"), AGAIN, RESAMPLES, BOOTSTRAP_SEED)
    # Other bootstrap settings than the runs' give other intervals than the reports hold: refused, not read against another estimator's figure.
    with pytest.raises(ValueError, match="or the bootstrap's settings are not the run's"):
        over_seeds(by_seed, RESAMPLES, BOOTSTRAP_SEED + 1)


# --------------------------------------------------------------------------------------------- the command
def _task_directory(root, seed, entry, name=None):
    """What is pulled for one task of the stage `ladder_l4`: a directory with its step files under `steps`."""
    steps = root / (name or f"ladder_l4_seed{seed}_r1") / "steps"
    steps.mkdir(parents=True)
    (steps / "report_ladder_l4.json").write_text(json.dumps(entry["report"], default=str))
    (steps / "l4_goal_set_again.jsonl").write_text("".join(json.dumps({"problem_id": problem_id}) + "\n" for problem_id in entry["again"]))
    (steps / "episodes_l3d2_more_with_problems.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entry["with"]))
    (steps / "l4_stored_pre_more.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entry["pre"]))
    return steps.parent


def _files(directory):
    return {str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}


@pytest.fixture
def pulled(tmp_path):
    config = tmp_path / "experiment.yaml"
    config.write_text(json.dumps({"evaluation": EVALUATION}))                                      # the runs' own bootstrap settings (YAML reads JSON)
    return tmp_path, config, {seed: _task_directory(tmp_path, seed, entry) for seed, entry in _three().items()}


def test_the_command_reads_three_pulled_task_directories_and_prints_the_read(pulled, capsys):
    root, config, directories = pulled
    assert ladder_l4_seeds.FILES == ("report_ladder_l4.json", "l4_goal_set_again.jsonl", "episodes_l3d2_more_with_problems.jsonl", "l4_stored_pre_more.jsonl")
    before = _files(root)
    assert ladder_l4_seeds.main([*(str(directories[seed]) for seed in (0, 1, 2)), "--config", str(config)]) == 0
    printed = capsys.readouterr()
    expected = over_seeds(_three(), RESAMPLES, BOOTSTRAP_SEED)
    assert printed.out.splitlines() == expected["lines"] and printed.err == "" and _files(root) == before          # the directories are only read; nothing is written
    # A `steps` directory named in a task directory's place, the seeds in another order, and `--out`: the same read, written where asked and nowhere else.
    out = root / "elsewhere" / "read.json"
    assert ladder_l4_seeds.main([str(directories[2] / "steps"), str(directories[0]), str(directories[1] / "steps"), "--config", str(config), "--out", str(out)]) == 0
    assert capsys.readouterr().out.splitlines() == expected["lines"]
    written = json.loads(out.read_text())
    assert written["pooled"] == expected["pooled"] and written["seeds"] == [0, 1, 2] and list(written["by_seed"]) == ["0", "1", "2"]
    assert written["directories"] == {str(seed): str(directories[seed] / "steps") for seed in (0, 1, 2)}
    assert {name: content for name, content in _files(root).items() if not name.startswith("elsewhere")} == before
    # The seed is the report's own, whatever the directory is called.
    assert ladder_l4_seeds.read_seed(directories[1])[0] == 1 and ladder_l4_seeds.steps_directory(directories[1]) == directories[1] / "steps"
    # The package's own config is the default, and holds the runs' settings.
    assert ladder_l4_seeds.CONFIG.name == "experiment.yaml" and ladder_l4_seeds.CONFIG.is_file()


def test_the_command_refuses_a_directory_that_is_not_a_finished_task_of_the_arm_and_what_the_read_refuses(pulled, capsys):
    root, config, directories = pulled
    paths = [str(directories[seed]) for seed in (0, 1, 2)]

    def refused(arguments, said):
        assert ladder_l4_seeds.main([*arguments, "--config", str(config)]) == 2
        printed = capsys.readouterr()
        assert printed.out == "" and printed.err.startswith("refused: ") and said in printed.err

    # A task that did not reach its report delivered no read: each missing file is named.
    for name in ladder_l4_seeds.FILES:
        lost = directories[1] / "steps" / name
        lost.rename(lost.with_name("elsewhere"))
        refused(paths, f"does not hold ['{name}']: it is not the pulled directory of a task of the stage `ladder_l4` that wrote its report")
        lost.with_name("elsewhere").rename(lost)
    refused([paths[0], str(root / "nowhere"), paths[2]], "nowhere does not hold ['report_ladder_l4.json', 'l4_goal_set_again.jsonl', 'episodes_l3d2_more_with_problems.jsonl', "
                                                     "'l4_stored_pre_more.jsonl']")
    # Another stage's report under that name, or one with no seed.
    report_file = directories[2] / "steps" / "report_ladder_l4.json"
    as_written = report_file.read_text()
    for change in ({"stage": "l3d2"}, {"seed": None}):
        report_file.write_text(json.dumps({**json.loads(as_written), **change}))
        refused(paths, "report_ladder_l4.json is not a report of L4's arm with its seed")
    report_file.write_text(as_written)
    # Two directories of one seed; one seed alone.
    again = _task_directory(root, 1, _seed(1), name="ladder_l4_seed1_r2")
    refused([*paths, str(again)], "are both of seed 1: one task directory a seed")
    refused(paths[:1], "needs at least two seeds, and was given [0]")
    # What the read refuses is said as it says it: another G' at a seed.
    other = directories[2] / "steps" / "l4_goal_set_again.jsonl"
    other.write_text("".join(json.dumps({"problem_id": problem_id}) + "\n" for problem_id in AGAIN[:-1]))
    refused(paths, "seed 2 holds another G' than seed 0")
