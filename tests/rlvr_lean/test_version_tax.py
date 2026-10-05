"""The version-tax report (the OEIS Open spec, O2 item 9, O2a item 9e, fixture 3): identical
proof texts under two Lean pins. What is compared, what is left out, how a failure is classed, and the
pre-registered read."""

import math

import pytest

from rlvr_lean.domain.evaluation.version_tax import (
    OTHER_ERROR,
    REJECTED_LEXICAL,
    TIMEOUT,
    UNKNOWN_IDENTIFIER,
    Outcome,
    band_names,
    band_of,
    build_version_tax,
    error_kind,
    failure_class,
    is_echo_of_a_failed_declaration,
    pre_registered_read,
    wilson_interval,
)

V, E, T, N = "verified", "lean_error", "timeout", "server_error"


def outcomes(statuses, errors=()):
    return {attempt_id: Outcome(status, tuple(errors)) for attempt_id, status in statuses.items()}


def attempts_of(*statements):
    """((statement, samples), ...) -> [(attempt id, statement id)] with ids `s#0`, `s#1`, ..."""
    return [(f"{statement}#{index}", statement) for statement, samples in statements for index in range(samples)]


# ------------------------------------------------------------------------------- fixture 3: identical texts
def test_the_report_fails_if_the_two_pins_judged_different_attempts():
    attempts = attempts_of(("a", 2))
    both = outcomes({"a#0": V, "a#1": E})
    with pytest.raises(ValueError, match="same attempts.*later pin lacks 1"):
        build_version_tax(attempts, both, outcomes({"a#0": V}), {"a"})
    with pytest.raises(ValueError, match="earlier pin lacks 0 of the 2 and has 1 others"):
        build_version_tax(attempts, {**both, **outcomes({"b#0": V})}, both, {"a"})
    with pytest.raises(ValueError, match="twice"):
        build_version_tax(attempts + attempts[:1], both, both, {"a"})


def test_only_statements_that_compile_at_both_pins_are_compared():
    attempts = attempts_of(("kept", 2), ("gone", 2))
    earlier = outcomes({"kept#0": V, "kept#1": E, "gone#0": V, "gone#1": V})
    later = outcomes({"kept#0": V, "kept#1": E, "gone#0": E, "gone#1": E})
    report = build_version_tax(attempts, earlier, later, {"kept"}, resamples=200)
    assert report["attempts"] == {"given": 4, "on_statements_compiling_at_both_pins": 2, "no_answer_earlier": 0,
                                  "no_answer_later": 0, "compared": 2}
    overall = report["overall"]
    assert (overall["statements"], overall["verified_earlier"], overall["verified_later"]) == (1, 1, 1)
    assert overall["per_attempt"]["later_as_share_of_earlier"]["value"] == 1.0      # the lost statement is not a tax


def test_no_answer_at_either_pin_leaves_both_pins_counts_and_is_never_a_failure():
    attempts = attempts_of(("a", 4))
    earlier = outcomes({"a#0": V, "a#1": V, "a#2": N, "a#3": V})
    later = outcomes({"a#0": V, "a#1": N, "a#2": V, "a#3": E})
    report = build_version_tax(attempts, earlier, later, {"a"}, resamples=200)
    assert report["attempts"]["no_answer_earlier"] == 1 and report["attempts"]["no_answer_later"] == 1
    assert report["attempts"]["compared"] == 2                                       # a#0 and a#3
    overall = report["overall"]
    assert (overall["verified_earlier"], overall["verified_later"]) == (2, 1)
    assert sum(overall["later_failures"].values()) == 1 and sum(overall["lost"].values()) == 1
    assert failure_class(N, ()) is None


# --------------------------------------------------------------------------------------- rates and intervals
def test_rates_per_attempt_and_per_statement_proved_at_least_once():
    attempts = attempts_of(("a", 4), ("b", 4), ("c", 4))
    earlier = outcomes({"a#0": V, "a#1": V, "a#2": V, "a#3": V, "b#0": V, "b#1": E, "b#2": E, "b#3": E,
                        "c#0": E, "c#1": E, "c#2": E, "c#3": E})
    later = outcomes({"a#0": V, "a#1": V, "a#2": E, "a#3": E, "b#0": E, "b#1": E, "b#2": E, "b#3": E,
                      "c#0": V, "c#1": E, "c#2": E, "c#3": E})
    overall = build_version_tax(attempts, earlier, later, {"a", "b", "c"}, resamples=2000)["overall"]
    assert overall["per_attempt"]["earlier"]["value"] == pytest.approx(5 / 12)
    assert overall["per_attempt"]["later"]["value"] == pytest.approx(3 / 12)
    assert overall["per_attempt"]["later_as_share_of_earlier"]["value"] == pytest.approx(3 / 5)
    assert overall["per_attempt"]["difference"]["value"] == pytest.approx(-2 / 12)
    assert (overall["solved_earlier"], overall["solved_later"], overall["solved_at_both"]) == (2, 2, 1)
    assert overall["per_statement"]["later_as_share_of_earlier"]["value"] == 1.0
    assert overall["gained"] == 1 and sum(overall["lost"].values()) == 3
    for figure in overall["per_attempt"].values():
        assert figure["low"] <= figure["value"] <= figure["high"]


def test_identical_verdicts_give_a_share_of_exactly_one_with_no_spread():
    attempts = attempts_of(*((f"s{index}", 8) for index in range(20)))
    verdicts = outcomes({attempt_id: V if index % 3 == 0 else E for index, (attempt_id, _) in enumerate(attempts)})
    report = build_version_tax(attempts, verdicts, verdicts, {statement for _, statement in attempts}, resamples=500)
    share = report["overall"]["per_attempt"]["later_as_share_of_earlier"]
    assert (share["value"], share["low"], share["high"]) == (1.0, 1.0, 1.0)
    assert report["overall"]["per_attempt"]["difference"] == {"value": 0.0, "low": 0.0, "high": 0.0}
    assert report["read"]["read"] == "small" and report["read"]["interval_on_one_side_of_the_threshold"]


def test_the_interval_is_over_statements_so_one_statement_has_no_spread_and_many_do():
    one = build_version_tax(attempts_of(("a", 8)), outcomes({f"a#{i}": V if i < 4 else E for i in range(8)}),
                            outcomes({f"a#{i}": V if i < 2 else E for i in range(8)}), {"a"}, resamples=300)["overall"]
    assert one["per_attempt"]["later"] == {"value": 0.25, "low": 0.25, "high": 0.25}
    attempts = attempts_of(*((f"s{index}", 2) for index in range(40)))
    earlier = outcomes({attempt_id: V if int(statement[1:]) % 2 == 0 else E for attempt_id, statement in attempts})
    later = outcomes({attempt_id: V if int(statement[1:]) % 4 == 0 else E for attempt_id, statement in attempts})
    many = build_version_tax(attempts, earlier, later, {statement for _, statement in attempts}, resamples=2000, seed=1)["overall"]
    assert many["per_attempt"]["later"]["low"] < 0.25 < many["per_attempt"]["later"]["high"]
    again = build_version_tax(attempts, earlier, later, {statement for _, statement in attempts}, resamples=2000, seed=1)["overall"]
    assert again == many                                                              # a fixed seed reproduces it


def test_nothing_verified_earlier_makes_the_share_undefined_not_an_error():
    attempts = attempts_of(("a", 2))
    report = build_version_tax(attempts, outcomes({"a#0": E, "a#1": E}), outcomes({"a#0": V, "a#1": E}), {"a"}, resamples=100)
    assert report["overall"]["per_attempt"]["later_as_share_of_earlier"]["value"] is None
    assert report["read"]["read"] == "undefined" and report["overall"]["gained"] == 1
    empty = build_version_tax([], {}, {}, set(), resamples=100)
    assert empty["overall"] == {"statements": 0, "attempts": 0, "later_failures": dict.fromkeys(
        (UNKNOWN_IDENTIFIER, OTHER_ERROR, TIMEOUT, REJECTED_LEXICAL), 0), "lost": dict.fromkeys(
        (UNKNOWN_IDENTIFIER, OTHER_ERROR, TIMEOUT, REJECTED_LEXICAL), 0), "gained": 0}


# ----------------------------------------------------------------------------------------- failure classes
@pytest.mark.parametrize("message", [
    "unknown identifier 'Nat.pos_of_ne_zero'",
    "Unknown identifier `sq_sqrt`",
    "unknown constant 'Finset.sum_const_nat'",
    "Unknown constant `Real.rpow_natCast`",
    "Invalid field `le_div_iff`: The environment does not contain `Nat.le_div_iff`\n  h\nhas type\n  a ≤ b",
    "unknown namespace 'BigOperators'",
])
def test_a_missing_name_is_an_unknown_identifier_however_lean_words_it(message):
    assert failure_class(E, (message,)) == UNKNOWN_IDENTIFIER
    assert failure_class(E, ("unsolved goals\n⊢ False", message)) == UNKNOWN_IDENTIFIER       # any message, not only the first


@pytest.mark.parametrize("message", ["linarith failed to find a contradiction", "unsolved goals\nx : ℝ\n⊢ 0 ≤ x ^ 2",
                                     "The rfl tactic failed", "type mismatch\n  h\nhas type", "unexpected token 'at'; expected term",
                                     "unknown tactic", "unexpected token 'in'; expected ','", "`simp` made no progress"])
def test_every_other_lean_error_is_an_other_error(message):
    assert failure_class(E, (message,)) == OTHER_ERROR


def test_print_axioms_not_finding_the_attempts_own_theorem_is_an_echo_not_a_missing_name():
    """A declaration that fails leaves no theorem, so the trailing `#print axioms` adds "Unknown constant
    `<its name>`". The tool drops that message before classing; a library name stays."""
    assert is_echo_of_a_failed_declaration("Unknown constant `lean_workbook_49459`", "lean_workbook_49459")
    assert is_echo_of_a_failed_declaration("unknown constant 'conjecture_0592f50eac31703c'", "conjecture_0592f50eac31703c")
    assert not is_echo_of_a_failed_declaration("Unknown constant `lean_workbook_494590`", "lean_workbook_49459")
    assert not is_echo_of_a_failed_declaration("Unknown constant `Real.sqrt_eq_iff_sq_eq`", "lean_workbook_49459")
    assert not is_echo_of_a_failed_declaration("unexpected token 'in'; expected ','", "lean_workbook_49459")


def test_error_kinds_drop_the_names_a_message_mentions():
    assert error_kind("Unknown constant `Real.sqrt_eq_iff_sq_eq`") == error_kind("Unknown constant `Int.zero_eq`") == "Unknown constant _"
    assert error_kind("unexpected token 'in'; expected ','") == "unexpected token _; expected _"
    assert error_kind("unsolved goals\nx : ℝ\n⊢ 0 ≤ x") == "unsolved goals"
    assert error_kind("Invalid field `foo`: The environment does not contain `Nat.foo`, so it is not possible") \
        == "Invalid field _: The environment does not contain _, so it is not possible"


def test_classes_by_status():
    assert failure_class(V, ()) is None
    assert failure_class(T, ()) == TIMEOUT
    assert failure_class("rejected_lexical", ()) == REJECTED_LEXICAL
    assert failure_class("uses_sorry", ()) == OTHER_ERROR and failure_class("forbidden_axiom", ()) == OTHER_ERROR


def test_failures_are_split_for_all_later_failures_and_for_the_lost_ones():
    attempts = attempts_of(("a", 5))
    earlier = outcomes({"a#0": V, "a#1": V, "a#2": V, "a#3": E, "a#4": "rejected_lexical"})
    later = {"a#0": Outcome(E, ("unknown identifier 'foo'",)), "a#1": Outcome(T), "a#2": Outcome(V),
             "a#3": Outcome(E, ("linarith failed",)), "a#4": Outcome("rejected_lexical")}
    overall = build_version_tax(attempts, earlier, later, {"a"}, resamples=100)["overall"]
    assert overall["later_failures"] == {UNKNOWN_IDENTIFIER: 1, OTHER_ERROR: 1, TIMEOUT: 1, REJECTED_LEXICAL: 1}
    assert overall["lost"] == {UNKNOWN_IDENTIFIER: 1, OTHER_ERROR: 0, TIMEOUT: 1, REJECTED_LEXICAL: 0}


# ---------------------------------------------------------------------------------------- bands and groups
def test_bands_are_the_earlier_pins_pass_rate_and_cover_every_statement_once():
    assert band_names((0.25, 0.75)) == ["0", "(0, 0.25]", "(0.25, 0.75]", "(0.75, 1]"]
    assert [band_of(rate, (0.25, 0.75)) for rate in (0, 1 / 12, 0.25, 0.26, 0.75, 0.76, 1)] == [
        "0", "(0, 0.25]", "(0, 0.25]", "(0.25, 0.75]", "(0.25, 0.75]", "(0.75, 1]", "(0.75, 1]"]
    attempts = attempts_of(("none", 4), ("low", 4), ("half", 4), ("all", 4))
    earlier = outcomes({f"{name}#{i}": V if i < count else E for name, count in (("none", 0), ("low", 1), ("half", 2), ("all", 4))
                        for i in range(4)})
    later = outcomes({attempt_id: V if attempt_id in ("all#0", "none#0") else E for attempt_id, _ in attempts})
    report = build_version_tax(attempts, earlier, later, {"none", "low", "half", "all"}, resamples=200,
                               group_of={"none": "reward", "low": "reward", "half": "conjecture", "all": "conjecture"})
    bands = report["by_band"]
    assert [bands[name]["statements"] for name in band_names((0.25, 0.75))] == [1, 1, 1, 1]
    assert bands["0"]["gained"] == 1 and bands["0"]["per_attempt"]["later_as_share_of_earlier"]["value"] is None
    assert bands["(0.75, 1]"]["per_attempt"]["later_as_share_of_earlier"]["value"] == 0.25
    assert sum(section["attempts"] for section in bands.values()) == report["overall"]["attempts"]
    assert {group: section["attempts"] for group, section in report["by_group"].items()} == {"conjecture": 8, "reward": 8}
    # The earlier run's own pass rates decide the band when they are given (a sample compares only some attempts).
    given = build_version_tax(attempts, earlier, later, {"none", "low", "half", "all"}, resamples=200,
                              base_pass_rate={"none": 0.5, "low": 0.5, "half": 0.5, "all": 0.5})["by_band"]
    assert [given[name]["statements"] for name in band_names((0.25, 0.75))] == [0, 0, 4, 0]


# ------------------------------------------------------------------------------------- the pre-registered read
def figure(value, low, high):
    return {"per_attempt": {"later_as_share_of_earlier": {"value": value, "low": low, "high": high}}}


def test_the_tax_is_small_within_twenty_percent_and_the_interval_says_whether_the_data_decide():
    assert pre_registered_read(figure(0.9, 0.85, 0.95))["read"] == "small"
    assert pre_registered_read(figure(0.9, 0.85, 0.95))["interval_on_one_side_of_the_threshold"]
    assert pre_registered_read(figure(0.8, 0.7, 0.9))["read"] == "small"                 # exactly 20% is within 20%
    assert not pre_registered_read(figure(0.8, 0.7, 0.9))["interval_on_one_side_of_the_threshold"]
    lost = pre_registered_read(figure(0.5, 0.4, 0.6))
    assert lost["read"] == "not small" and lost["interval_on_one_side_of_the_threshold"]
    assert lost["relative_loss"] == pytest.approx(0.5)


def test_wilson_interval():
    low, high = wilson_interval(1900, 2000)
    assert low < 0.95 < high and (round(low, 4), round(high, 4)) == (0.9396, 0.9587)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert 0.0 < wilson_interval(0, 10)[1] < 0.31 and 0.69 < wilson_interval(10, 10)[0] < 1.0
    assert math.isclose(sum(wilson_interval(5, 10)), 1.0)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
