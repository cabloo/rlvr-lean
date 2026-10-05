"""What L0 decides, from the base model's episodes. Spec: docs/spec/ladder-loop.spec.md, milestone L0 and
"Held-out sets". Pure: rows in, a report out.

  the goal set G    the Lean Workbook problems of H the base resolved in none of its placing episodes
  the rungs         the problems of H with at least one success, placed against the band computed from t
  the base map      how the base's pass rates on a random sample of the pool sit against the band
  the certificates  how many published answers survived the pin (from the data half's own summary)
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping

from rlvr_lean.domain.problem_pool.certificates import FALSE_SIDE, TRUE_SIDE
from rlvr_lean.domain.problem_pool.episodes import ABOVE_BAND, BELOW_BAND, IN_BAND, RUNGS, band, reward, rung
from rlvr_lean.domain.problem_pool.selection import LEAN_WORKBOOK, STP_CONJECTURE

MAXIMUM_SHARE_WITHOUT_AN_ANSWER = 0.02      # above this share of attempts with no verdict the counts are not to be read


def _histogram(rows: list[Mapping]) -> dict[str, int]:
    """Problems by k (the episodes that resolved them), every k from 0 to n listed."""
    if not rows:
        return {}
    counts = Counter(row["resolved"] for row in rows)
    return {str(k): counts.get(k, 0) for k in range(max(row["episodes"] for row in rows) + 1)}


def _by(rows: list[Mapping], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _attempt_health(rows: list[Mapping]) -> dict:
    attempts = sum(row["episodes"] * row["sides"] for row in rows)
    without = sum(row["attempts_without_an_answer"] for row in rows)
    return {"attempts": attempts, "capped_at_the_token_limit": sum(row["attempts_capped"] for row in rows),
            "timed_out_in_lean": sum(row["attempts_timed_out"] for row in rows), "without_an_answer": without,
            "share_without_an_answer": round(without / attempts, 5) if attempts else None,
            # The side a certificate rules out, when it is not checked for every problem (`episode.contradicted_side`).
            "not_checked_on_the_contradicted_side": sum(row.get("attempts_not_checked", 0) for row in rows),
            "contradicted_side": dict(Counter(str(row.get("contradicted_side")) for row in rows)),
            "problems_with_one_side_only": sum(row["sides"] == 1 for row in rows),
            "one_side_by_reason": dict(Counter(row["one_side_reason"] for row in rows if row["one_side_reason"]))}


def _resolutions(rows: list[Mapping]) -> dict:
    by_statement = sum(row["resolved_by_statement"] for row in rows)
    by_negation = sum(row["resolved_by_negation"] for row in rows)
    return {"episodes": sum(row["episodes"] for row in rows), "resolved": sum(row["resolved"] for row in rows),
            "attempts_verified_at_the_statement": by_statement, "attempts_verified_at_the_negation": by_negation,
            "share_of_verified_attempts_that_are_negations": round(by_negation / (by_statement + by_negation), 4) if by_statement + by_negation else None}


def build_l0_report(heldout: list[Mapping], base_map: list[Mapping], data_summary: Mapping, settings: Mapping, steps: Mapping) -> dict:
    target, floor = settings["challenger"]["target_rate"], settings["challenger"]["band_reward"]
    minimum = settings["goal"]["rung_minimum"]
    low, high = band(target, floor)

    def placed(row: Mapping) -> str | None:
        return rung(row["resolved"], row["episodes"], target, floor)

    # ---- H: the goal set and the rungs
    workbook = [row for row in heldout if row["heldout_part"] == LEAN_WORKBOOK]
    conjectures = [row for row in heldout if row["heldout_part"] == STP_CONJECTURE]
    goal = [row for row in workbook if row["resolved"] == 0]
    rungs = {}
    for name in RUNGS:
        members = [row for row in heldout if placed(row) == name]
        rungs[name] = {"problems": len(members), "lean_workbook": sum(row["heldout_part"] == LEAN_WORKBOOK for row in members),
                       "stp_conjecture": sum(row["heldout_part"] == STP_CONJECTURE for row in members),
                       "known_false": sum(row["side"] == FALSE_SIDE for row in members),
                       "at_least_the_minimum": len(members) >= minimum}
    episodes = max((row["episodes"] for row in heldout), default=0)
    edges = {name: sorted(k for k in range(1, episodes + 1) if rung(k, episodes, target, floor) == name) for name in RUNGS}

    # ---- the base map: the pool against the band
    in_band = [row for row in base_map if placed(row) == IN_BAND]
    rewards = [reward(row["resolved"], row["episodes"], target) for row in base_map]
    map_report = {
        "problems": len(base_map), "by_kind": _by(base_map, "kind"), "by_side": _by(base_map, "side"), "k_histogram": _histogram(base_map),
        "in_the_band": len(in_band), "share_in_the_band": round(len(in_band) / len(base_map), 4) if base_map else None,
        "below_the_band": sum(placed(row) == BELOW_BAND for row in base_map), "above_the_band": sum(placed(row) == ABOVE_BAND for row in base_map),
        "never_resolved": sum(row["resolved"] == 0 for row in base_map), "always_resolved": sum(row["resolved"] == row["episodes"] for row in base_map),
        "mean_reward_of_a_random_draw": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "share_in_the_band_by_kind": {kind: round(sum(row["kind"] == kind for row in in_band) / max(1, sum(row["kind"] == kind for row in base_map)), 4)
                                      for kind in sorted({row["kind"] for row in base_map})},
        "resolutions": _resolutions(base_map), "attempts": _attempt_health(base_map)}

    health = {"heldout": _attempt_health(heldout), "base_map": map_report["attempts"]}
    not_to_be_read = [name for name, entry in health.items()
                      if entry["share_without_an_answer"] is not None and entry["share_without_an_answer"] > MAXIMUM_SHARE_WITHOUT_AN_ANSWER]
    short_rungs = [name for name in RUNGS if not rungs[name]["at_least_the_minimum"]]
    pool_summary = data_summary.get("pool") or {}
    decides = {
        "certificates_surviving_the_pin": {key: pool_summary.get(key) for key in (
            "candidates", "verified", "not_verified_by_reason", "verified_by_kind_side_source", "verified_by_a_renamed_certificate",
            "verified_with_a_rewritten_statement")},
        "goal_set": {"problems": len(goal), "known_true": sum(row["side"] == TRUE_SIDE for row in goal),
                     "known_false": sum(row["side"] == FALSE_SIDE for row in goal),
                     "of_lean_workbook_problems_held_out": len(workbook)},
        "rungs": rungs, "rungs_below_the_minimum": short_rungs, "rung_minimum": minimum,
        "share_of_the_pool_in_the_band_for_the_base": map_report["share_in_the_band"]}
    headline = (f"G = {len(goal)} of {len(workbook)} held-out Lean Workbook problems unresolved by the base in {episodes} episodes; rungs below/in/above "
                f"the band: {rungs[BELOW_BAND]['problems']}/{rungs[IN_BAND]['problems']}/{rungs[ABOVE_BAND]['problems']}"
                f"{' (BELOW THE MINIMUM: ' + ', '.join(short_rungs) + ')' if short_rungs else ''}; "
                f"{map_report['in_the_band']} of {len(base_map)} random pool problems in the band"
                f"{'; NOT TO BE READ: too many attempts without an answer in ' + ', '.join(not_to_be_read) if not_to_be_read else ''}")
    return {
        "spec": "docs/spec/ladder-loop.spec.md, milestone L0", "headline": headline, "ok": not not_to_be_read,
        "fixture": bool(data_summary.get("fixture")), "stand_in_engine": any(steps[name].get("stand_in_engine") for name in ("heldout", "base_map")),
        "target_rate": target, "band_reward_floor": floor, "band": {"low": round(low, 4), "high": round(high, 4)},
        "rung_edges_in_successes": {name: [edges[name][0], edges[name][-1]] if edges[name] else [] for name in RUNGS},
        "decides": decides,
        "heldout": {"problems": len(heldout), "lean_workbook": len(workbook), "stp_conjecture": len(conjectures),
                    "episodes_each": episodes, "k_histogram": _histogram(heldout), "k_histogram_lean_workbook": _histogram(workbook),
                    "stp_conjectures_with_no_success": sum(row["resolved"] == 0 for row in conjectures),
                    "resolved_at_least_once": sum(row["resolved"] > 0 for row in heldout),
                    "known_false": {"problems": sum(row["side"] == FALSE_SIDE for row in heldout),
                                    "resolved_at_least_once": sum(row["side"] == FALSE_SIDE and row["resolved"] > 0 for row in heldout)},
                    "resolutions": _resolutions(heldout), "attempts": health["heldout"]},
        "base_map": map_report, "not_to_be_read": not_to_be_read, "data": {key: value for key, value in data_summary.items() if key != "pool"},
        "steps": steps,
    }
